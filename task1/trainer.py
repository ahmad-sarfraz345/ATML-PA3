from __future__ import annotations
import os
import sys
import math
import time
import copy
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
from common.utils import (
    set_seed,
    get_device,
    GPUMemoryTracker,
    save_json,
    load_json
)
from task1.models import apply_lora_to_model, apply_oplora_to_model

def get_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (trainable_params, all_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def build_optimizer_with_weight_decay(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8
) -> torch.optim.AdamW:
    """
    Builds AdamW optimizer excluding biases and parameters with fewer than 2 dimensions
    (e.g., LayerNorm/RMSNorm weights).
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Exclude biases and 1D parameters from weight decay
        if param.ndim < 2 or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

    try:
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            fused=torch.cuda.is_available()
        )
    except Exception:
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            betas=betas,
            eps=eps
        )

class Task1Trainer:
    """
    Dedicated Trainer for Task 1 supporting:
    - full_ft
    - full_ft_l2sp (with starting-point penalty lambda=1e-3)
    - lora_4
    - lora_16
    - oplora_16
    """
    def __init__(
        self,
        variant_name: str,
        model_name_or_path: str = "Qwen/Qwen3-0.6B",
        output_dir: str = "outputs/task1",
        seed: int = 6304,
        max_seq_length: int = 2048,
        gradient_accumulation_steps: int = 16,
        micro_batch_size: int = 2,
        total_steps: int = 1250,
        warmup_steps: int = 38,
        max_grad_norm: float = 1.0,
        l2sp_lambda: float = 1e-3,
        lora_r: int = 16,
        lora_alpha: float = 32.0,
        oplora_k: int = 128
    ):
        self.variant_name = variant_name
        self.model_name_or_path = model_name_or_path
        self.output_dir = Path(output_dir) / variant_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.seed = seed
        self.max_seq_length = max_seq_length
        self.grad_accum_steps = gradient_accumulation_steps
        self.micro_batch_size = micro_batch_size
        self.effective_batch_size = micro_batch_size * gradient_accumulation_steps  # 32
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.l2sp_lambda = l2sp_lambda if variant_name == "full_ft_l2sp" else 0.0
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.oplora_k = oplora_k

        # Checkpoint target steps: [312, 625, 937, 1250]
        self.checkpoint_steps = [312, 625, 937, 1250]
        self.device = get_device()
        self.mem_tracker = GPUMemoryTracker(self.device)

        # LR config
        if variant_name in ["full_ft", "full_ft_l2sp"]:
            self.peak_lr = 2e-5
        else:
            self.peak_lr = 2e-4

        set_seed(self.seed)

    def setup_model_and_tokenizer(self):
        print(f"Loading tokenizer and base model from {self.model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model:
        # Full FT uses FP32 master parameters; Adapter variants use frozen FP16 backbone.
        if self.variant_name in ["full_ft", "full_ft_l2sp"]:
            model_dtype = torch.float32
        else:
            model_dtype = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=model_dtype,
            trust_remote_code=True
        )
        self.model.gradient_checkpointing_enable()

        # Keep initial parameters reference for L2-SP or displacement measurement
        self.initial_params_cpu = {}
        if self.variant_name in ["full_ft", "full_ft_l2sp"]:
            for name, p in self.model.named_parameters():
                self.initial_params_cpu[name] = p.detach().cpu().clone()

        # Apply adapters if required
        self.overhead_stats = {}
        if self.variant_name == "lora_4":
            apply_lora_to_model(self.model, r=4, alpha=8.0, dropout=0.05)
        elif self.variant_name == "lora_16":
            apply_lora_to_model(self.model, r=16, alpha=32.0, dropout=0.05)
        elif self.variant_name == "oplora_16":
            _, self.overhead_stats = apply_oplora_to_model(
                self.model, r=16, alpha=32.0, k=self.oplora_k, dropout=0.05, device=self.device
            )

        self.model.to(self.device)

        trainable_p, total_p = get_trainable_parameters(self.model)
        print(f"Model [{self.variant_name}] initialized: {trainable_p:,} trainable / {total_p:,} total parameters.")

        self.optimizer = build_optimizer_with_weight_decay(self.model, lr=self.peak_lr)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    def compute_l2sp_penalty(self) -> float:
        """
        Computes L2-SP starting-point penalty value:
        Omega(theta, theta_0) = 0.5 * sum_j (theta_j - theta_{0,j})^2
        Sum over all unique parameters. theta_0 is kept fixed. Not averaged over parameters.
        """
        omega_val = 0.0
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self.initial_params_cpu:
                    p_init = self.initial_params_cpu[name].to(p.device, non_blocking=True)
                    omega_val += 0.5 * torch.sum((p.to(torch.float32) - p_init.to(torch.float32)) ** 2).item()
        return omega_val

    def compute_full_model_displacement(self) -> float:
        """Computes ||theta - theta_0||_2 over all parameters."""
        sq_dist = 0.0
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self.initial_params_cpu:
                    p_init = self.initial_params_cpu[name].to(p.device, non_blocking=True)
                    sq_dist += torch.sum((p.to(torch.float32) - p_init.to(torch.float32)) ** 2).item()
        return math.sqrt(sq_dist)

    def train(self, dataloader: DataLoader, resume_from: Optional[str] = None):
        self.mem_tracker.reset()
        start_time = time.perf_counter()

        # Check if run is already completed
        final_ckpt = self.output_dir / "checkpoint_step_1250.pt"
        summary_file = self.output_dir / "train_summary.json"
        if final_ckpt.exists() and summary_file.exists():
            print(f"Variant [{self.variant_name}] is already completed (found {final_ckpt}). Skipping training.", flush=True)
            return load_json(summary_file)

        # Training state
        current_step = 0
        examples_seen = 0
        skipped_updates = 0
        training_logs = []

        if resume_from is None:
            # Check for latest available partial checkpoint
            existing_ckpts = sorted(
                self.output_dir.glob("checkpoint_step_*.pt"),
                key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0
            )
            if existing_ckpts:
                resume_from = str(existing_ckpts[-1])
                print(f"Auto-resuming [{self.variant_name}] from latest checkpoint: {resume_from}", flush=True)

        if resume_from and Path(resume_from).exists():
            print(f"Resuming checkpoint from {resume_from}...", flush=True)
            ckpt = torch.load(resume_from, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            current_step = ckpt["current_step"]
            examples_seen = ckpt["examples_seen"]
            training_logs = ckpt.get("training_logs", [])
            print(f"Resumed at step {current_step}, examples seen: {examples_seen}", flush=True)

        self.model.train()
        loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)

        data_iter = iter(dataloader)
        num_batches = len(dataloader)
        batch_idx = examples_seen // self.micro_batch_size

        # Advance dataloader to resumed position if needed
        for _ in range(batch_idx):
            try:
                next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                next(data_iter)

        print(f"Starting training run [{self.variant_name}]: {self.total_steps} optimizer steps...")

        while current_step < self.total_steps:
            self.optimizer.zero_grad()

            accum_sft_loss = 0.0
            accum_tokens = 0
            micro_batches = []

            # 1. Fetch gradient accumulation slice
            for _ in range(self.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                micro_batches.append(batch)

            # Count total valid response tokens across the effective batch (32 examples)
            total_valid_tokens = sum(
                (b["labels"] != -100).sum().item() for b in micro_batches
            )
            if total_valid_tokens == 0:
                total_valid_tokens = 1  # Guard against division by zero

            # 2. Forward & backward for each microbatch
            for batch in micro_batches:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                examples_seen += input_ids.shape[0]

                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.float16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False
                    )

                    # Shift for causal prediction: token t predicts token t+1
                    shift_labels = labels[:, 1:]
                    valid_mask = (shift_labels != -100)
                    active_count = valid_mask.sum().item()

                    if active_count > 0:
                        # Slice logits only to active response tokens, avoiding allocating 1.24 GB for prompt/padding
                        active_logits = outputs.logits[:, :-1, :][valid_mask]
                        active_labels = shift_labels[valid_mask]
                        micro_loss_sum = nn.functional.cross_entropy(active_logits, active_labels, reduction="sum")
                        micro_loss_normalized = micro_loss_sum / total_valid_tokens
                        self.scaler.scale(micro_loss_normalized).backward()
                        accum_sft_loss += micro_loss_sum.item()
                        accum_tokens += active_count
                        del active_logits, active_labels, micro_loss_sum, micro_loss_normalized

                    del outputs, shift_labels, valid_mask, input_ids, labels, attention_mask

            effective_sft_loss = accum_sft_loss / max(1, accum_tokens)

            # 3. Unscale gradients before clipping
            self.scaler.unscale_(self.optimizer)

            # 4. Handle L2-SP proximity penalty if applicable
            total_loss = effective_sft_loss
            omega_val = 0.0
            lambda_omega_val = 0.0
            if self.variant_name == "full_ft_l2sp":
                # Compute Omega in no_grad to avoid constructing an autograd graph over 600M params
                with torch.no_grad():
                    for name, p in self.model.named_parameters():
                        if p.requires_grad and name in self.initial_params_cpu:
                            p_init = self.initial_params_cpu[name].to(p.device, non_blocking=True)
                            diff = p.data.to(torch.float32) - p_init.data.to(torch.float32)
                            omega_val += 0.5 * torch.sum(diff ** 2).item()
                            # Analytical gradient: grad += lambda * (theta - theta_0)
                            if p.grad is not None:
                                p.grad.data.add_(diff.to(p.grad.dtype), alpha=self.l2sp_lambda)
                    lambda_omega_val = self.l2sp_lambda * omega_val
                    total_loss = effective_sft_loss + lambda_omega_val

            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # 5. Optimizer step
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = self.scaler.get_scale()

            if scale_after < scale_before:
                skipped_updates += 1

            self.scheduler.step()
            current_step += 1

            # Log step metrics
            current_lr = self.scheduler.get_last_lr()[0]
            log_entry = {
                "step": current_step,
                "examples_seen": examples_seen,
                "sft_loss": effective_sft_loss,
                "total_loss": total_loss,
                "learning_rate": current_lr,
                "grad_norm": float(grad_norm) if hasattr(grad_norm, "item") else float(grad_norm),
                "peak_gpu_mem_mb": self.mem_tracker.get_peak_memory_mb(),
                "skipped_updates": skipped_updates
            }
            if self.variant_name == "full_ft_l2sp":
                log_entry["omega"] = omega_val
                log_entry["lambda_omega"] = lambda_omega_val

            training_logs.append(log_entry)

            if current_step % 25 == 0 or current_step in self.checkpoint_steps:
                print(f"[{self.variant_name}] Step {current_step}/{self.total_steps} | "
                      f"Examples: {examples_seen:,} | SFT Loss: {effective_sft_loss:.4f} | "
                      f"LR: {current_lr:.2e} | Peak VRAM: {self.mem_tracker.get_peak_memory_gb():.2f} GB")

            # 6. Checkpointing schedule: 312, 625, 937, 1250
            if current_step in self.checkpoint_steps:
                ckpt_path = self.output_dir / f"checkpoint_step_{current_step}.pt"
                ckpt_dict = {
                    "current_step": current_step,
                    "examples_seen": examples_seen,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "scaler_state_dict": self.scaler.state_dict(),
                    "training_logs": training_logs,
                    "variant": self.variant_name
                }
                if self.variant_name in ["full_ft", "full_ft_l2sp"]:
                    displacement = self.compute_full_model_displacement()
                    ckpt_dict["model_displacement"] = displacement
                    print(f"[{self.variant_name}] Model displacement at step {current_step}: {displacement:.4f}")

                torch.save(ckpt_dict, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

        total_train_time = time.perf_counter() - start_time
        final_displacement = None
        if self.variant_name in ["full_ft", "full_ft_l2sp"]:
            final_displacement = self.compute_full_model_displacement()

        trainable_p, total_p = get_trainable_parameters(self.model)

        summary = {
            "variant": self.variant_name,
            "total_steps": self.total_steps,
            "examples_seen": examples_seen,
            "total_train_time_sec": total_train_time,
            "peak_gpu_mem_mb": self.mem_tracker.get_peak_memory_mb(),
            "peak_gpu_mem_gb": self.mem_tracker.get_peak_memory_gb(),
            "trainable_parameters": trainable_p,
            "total_parameters": total_p,
            "skipped_updates": skipped_updates,
            "final_displacement": final_displacement,
            "overhead_stats": self.overhead_stats
        }
        save_json(summary, self.output_dir / "train_summary.json")
        save_json(training_logs, self.output_dir / "training_logs.json")
        print(f"Finished training [{self.variant_name}] in {total_train_time/60:.2f} mins. Peak VRAM: {summary['peak_gpu_mem_gb']:.2f} GB.")
        return summary

