from __future__ import annotations
import os

# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import time
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup

from common.utils import set_seed, get_device, GPUMemoryTracker, save_json, load_json
from task2.models import QwenVLM
from task2.data import collate_task2_batch, CurriculumBatchSampler
from task2.eval import compute_teacher_forced_nll

class Task2Trainer:
    """
    Curriculum Trainer for Task 2:
    - Stage 1 (1 - 2250 updates): Caption alignment, connector only (LR 1e-3)
    - Stage 2 (2251 - 3750 updates): Visual instruction tuning, connector (1e-4) + LoRA (2e-5)
    - Stage 3 (3751 - 4500 updates): Natural VQA specialization, connector (5e-5) + LoRA (1e-5)
    """
    def __init__(
        self,
        base_model_path: str = "Qwen/Qwen3-0.6B",
        output_dir: str = "outputs/task2",
        seed: int = 6304,
        micro_batch_size: int = 8,
        grad_accum_steps: int = 2,
        connector_spatial_pooling: bool = False
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.micro_batch_size = micro_batch_size
        self.grad_accum_steps = grad_accum_steps
        self.effective_batch_size = micro_batch_size * grad_accum_steps  # 16
        self.device = get_device()
        self.mem_tracker = GPUMemoryTracker(self.device)

        set_seed(self.seed)

        # Initialize QwenVLM
        self.vlm = QwenVLM(
            base_model_path=base_model_path,
            connector_spatial_pooling=connector_spatial_pooling,
            device=self.device
        )
        self.tokenizer = self.vlm.tokenizer
        self.scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

        # Checkpointing schedule: every 250 steps and stage endpoints
        self.checkpoint_steps = set(list(range(250, 4501, 250)) + [2250, 3750, 4500])
        self.dev_eval_steps = set([0] + list(range(500, 4501, 500)) + [2250, 3750, 4500])

    def calibrate_connector(self, sample_train_captions: List[Dict[str, Any]], cached_features: Dict[int, torch.Tensor]):
        """Calibrates connector output magnitude against text embeddings."""
        print("\n--- Calibrating Connector Output Scale ---")
        qwen = self.vlm.qwen
        embed_tokens = qwen.get_input_embeddings()

        with torch.no_grad():
            sample_caps = [c["answer"] for c in sample_train_captions[:64]]
            text_ids = [self.tokenizer.encode(c, return_tensors="pt") for c in sample_caps]
            text_embeds = [embed_tokens(t.to(self.device)) for t in text_ids]
            all_text = torch.cat(text_embeds, dim=1)
            mean_text_norm = torch.norm(all_text.to(torch.float32), p=2, dim=-1).mean().item()

            vis_samples = []
            for c in sample_train_captions[:64]:
                feat = cached_features.get(c["image_id"], None)
                if feat is not None:
                    vis_samples.append(feat)
            if vis_samples:
                sample_vis = torch.stack(vis_samples).to(self.device, dtype=torch.float32)
                self.vlm.connector.calibrate(mean_text_norm, sample_vis)

    def build_optimizer_and_scheduler(self, stage: int) -> Tuple[torch.optim.Optimizer, Any]:
        """
        Builds per-stage optimizer and learning rate schedule:
        Stage 1: updates 1 - 2250, connector peak LR 1e-3, cosine decay to 10%
        Stage 2: updates 2251 - 3750 (1500 updates), connector peak LR 1e-4, LoRA peak LR 2e-5
        Stage 3: updates 3751 - 4500 (750 updates), connector peak LR 5e-5, LoRA peak LR 1e-5
        """
        if stage == 1:
            total_stage_steps = 2250
            param_groups = [
                {"params": [p for p in self.vlm.connector.parameters() if p.requires_grad], "lr": 1e-3, "weight_decay": 0.01}
            ]
        elif stage == 2:
            total_stage_steps = 1500
            param_groups = [
                {"params": [p for p in self.vlm.connector.parameters() if p.requires_grad], "lr": 1e-4, "weight_decay": 0.01},
                {"params": [p for p in self.vlm.lora_model.parameters() if p.requires_grad], "lr": 2e-5, "weight_decay": 0.01}
            ]
        else:
            total_stage_steps = 750
            param_groups = [
                {"params": [p for p in self.vlm.connector.parameters() if p.requires_grad], "lr": 5e-5, "weight_decay": 0.01},
                {"params": [p for p in self.vlm.lora_model.parameters() if p.requires_grad], "lr": 1e-5, "weight_decay": 0.01}
            ]

        optimizer = torch.optim.AdamW(param_groups, eps=1e-8)
        warmup_steps = int(0.05 * total_stage_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_stage_steps
        )
        return optimizer, scheduler

    def evaluate_development_suite(
        self,
        step: int,
        dev_captions: List[Dict[str, Any]],
        grounding_dev: List[Dict[str, Any]],
        vqa_dev: List[Dict[str, Any]],
        cached_features: Dict[int, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Evaluates development suites at step:
        1. Caption loss (96 examples)
        2. Grounding dev NLL: correct, partner, no-image -> delta_nll
        3. Natural VQAv2 dev NLL: correct, partner, no-image -> delta_nll
        """
        print(f"\n--- Running Development Evaluation at Step {step} ---")
        self.vlm.eval()
        pad_id = self.tokenizer.pad_token_id

        # 1. Dev caption loss
        cap_items = []
        for c in dev_captions:
            ex = {
                "image_id": c["image_id"],
                "prefix_ids": self.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False),
                "suffix_ids": self.tokenizer.encode(f"{c['question']}<|im_end|>\n<|im_start|>assistant\n{c['answer']}<|im_end|>\n", add_special_tokens=False),
                "labels": [-100] * len(self.tokenizer.encode(f"{c['question']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)) + self.tokenizer.encode(f"{c['answer']}<|im_end|>\n", add_special_tokens=False),
                "visual_feature": cached_features.get(c["image_id"], None)
            }
            cap_items.append(ex)

        batch_cap = collate_task2_batch(cap_items, pad_id)
        with torch.no_grad():
            outputs = self.vlm(
                batch_cap["prefix_ids"].to(self.device),
                batch_cap["suffix_ids"].to(self.device),
                batch_cap["attention_mask"].to(self.device),
                batch_cap["visual_features"].to(self.device) if batch_cap["visual_features"] is not None else None
            )
            # Example-weighted NLL
            labels = batch_cap["labels"].to(self.device)[:, 1:].contiguous()
            logits = outputs.logits[:, -labels.shape[1]:, :].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            token_losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)).view(labels.size())
            val_mask = (labels != -100)
            ex_losses = []
            for i in range(labels.size(0)):
                if val_mask[i].sum() > 0:
                    ex_losses.append((token_losses[i] * val_mask[i]).sum().item() / val_mask[i].sum().item())
            dev_cap_loss = float(sum(ex_losses) / max(1, len(ex_losses)))

        # 2. Helper to evaluate pair suite under (correct, partner, no-image)
        def eval_pairs(pair_list, task_name):
            nll_correct_list = []
            nll_partner_list = []
            nll_noimg_list = []

            for p in pair_list:
                for img_key, ans_key, partner_img_key in [("image_1", "canonical_answer_1" if "canonical_answer_1" in p else "answer_1", "image_2"),
                                                          ("image_2", "canonical_answer_2" if "canonical_answer_2" in p else "answer_2", "image_1")]:
                    q_text = p["question"]
                    ans_text = p[ans_key]
                    img_corr = p[img_key]
                    img_part = p[partner_img_key]

                    # Format base example
                    def make_ex(img_id):
                        return {
                            "image_id": img_id,
                            "prefix_ids": self.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False),
                            "suffix_ids": self.tokenizer.encode(f"{q_text}<|im_end|>\n<|im_start|>assistant\n{ans_text}<|im_end|>\n", add_special_tokens=False),
                            "labels": [-100] * len(self.tokenizer.encode(f"{q_text}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)) + self.tokenizer.encode(f"{ans_text}<|im_end|>\n", add_special_tokens=False),
                            "visual_feature": cached_features.get(img_id, None) if img_id is not None else None
                        }

                    # Correct image
                    b_corr = collate_task2_batch([make_ex(img_corr)], pad_id)
                    with torch.no_grad():
                        out = self.vlm(b_corr["prefix_ids"].to(self.device), b_corr["suffix_ids"].to(self.device), b_corr["attention_mask"].to(self.device), b_corr["visual_features"].to(self.device) if b_corr["visual_features"] is not None else None)
                        lab = b_corr["labels"].to(self.device)[:, 1:].contiguous()
                        log = out.logits[:, -lab.shape[1]:, :].contiguous()
                        l = nn.functional.cross_entropy(log.view(-1, log.size(-1)), lab.view(-1), ignore_index=-100).item()
                        nll_correct_list.append(l)

                    # Partner image (complementary swap)
                    b_part = collate_task2_batch([make_ex(img_part)], pad_id)
                    with torch.no_grad():
                        out = self.vlm(b_part["prefix_ids"].to(self.device), b_part["suffix_ids"].to(self.device), b_part["attention_mask"].to(self.device), b_part["visual_features"].to(self.device) if b_part["visual_features"] is not None else None)
                        lab = b_part["labels"].to(self.device)[:, 1:].contiguous()
                        log = out.logits[:, -lab.shape[1]:, :].contiguous()
                        l = nn.functional.cross_entropy(log.view(-1, log.size(-1)), lab.view(-1), ignore_index=-100).item()
                        nll_partner_list.append(l)

                    # No image
                    b_noimg = collate_task2_batch([make_ex(None)], pad_id)
                    with torch.no_grad():
                        out = self.vlm(b_noimg["prefix_ids"].to(self.device), b_noimg["suffix_ids"].to(self.device), b_noimg["attention_mask"].to(self.device), None)
                        lab = b_noimg["labels"].to(self.device)[:, 1:].contiguous()
                        log = out.logits[:, -lab.shape[1]:, :].contiguous()
                        l = nn.functional.cross_entropy(log.view(-1, log.size(-1)), lab.view(-1), ignore_index=-100).item()
                        nll_noimg_list.append(l)

            mean_corr = float(sum(nll_correct_list) / max(1, len(nll_correct_list)))
            mean_part = float(sum(nll_partner_list) / max(1, len(nll_partner_list)))
            mean_noimg = float(sum(nll_noimg_list) / max(1, len(nll_noimg_list)))
            delta_nll = mean_part - mean_corr  # Image-swap penalty
            return {
                "nll_correct": mean_corr,
                "nll_partner": mean_part,
                "nll_no_image": mean_noimg,
                "delta_nll_swap_penalty": delta_nll
            }

        grounding_dev_metrics = eval_pairs(grounding_dev, "grounding")
        vqa_dev_metrics = eval_pairs(vqa_dev, "vqa")

        record = {
            "step": step,
            "dev_caption_loss": dev_cap_loss,
            "grounding_dev": grounding_dev_metrics,
            "vqa_dev": vqa_dev_metrics
        }
        print(f"Step {step} | Dev Cap Loss: {dev_cap_loss:.4f} | "
              f"Grounding Delta NLL: {grounding_dev_metrics['delta_nll_swap_penalty']:.4f} | "
              f"VQA Delta NLL: {vqa_dev_metrics['delta_nll_swap_penalty']:.4f}")
        return record

    def train_curriculum(
        self,
        sampler: CurriculumBatchSampler,
        dev_captions: List[Dict[str, Any]],
        grounding_dev: List[Dict[str, Any]],
        vqa_dev: List[Dict[str, Any]],
        cached_features: Dict[int, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Executes the full 4,500 curriculum updates across 3 stages.
        """
        self.mem_tracker.reset()
        start_time = time.perf_counter()

        training_logs = []
        dev_eval_logs = []

        # Check if already completed
        final_ckpt = self.output_dir / "checkpoint_step_4500.pt"
        summary_file = self.output_dir / "train_summary.json"
        if final_ckpt.exists() and summary_file.exists():
            print(f"Task 2 is already completed. Skipping training.", flush=True)
            return load_json(summary_file)

        # Evaluate step 0 before training
        if 0 in self.dev_eval_steps:
            d0 = self.evaluate_development_suite(0, dev_captions, grounding_dev, vqa_dev, cached_features)
            dev_eval_logs.append(d0)

        current_step = 0
        examples_seen = 0
        pad_id = self.tokenizer.pad_token_id

        # Stages: 1, 2, 3
        for stage in [1, 2, 3]:
            print(f"\n=======================================================")
            print(f">>> STARTING CURRICULUM STAGE {stage} <<<")
            if stage == 1:
                print("Stage 1: Caption Alignment (Updates 1 - 2250, Connector only)")
                stage_end_step = 2250
            elif stage == 2:
                print("Stage 2: Visual Instruction Tuning (Updates 2251 - 3750, Connector + LoRA)")
                self.vlm.enable_lora(r=16, alpha=32.0, dropout=0.05)
                stage_end_step = 3750
            else:
                print("Stage 3: Natural VQA Specialization (Updates 3751 - 4500, Connector + LoRA)")
                stage_end_step = 4500
            print(f"=======================================================\n")

            optimizer, scheduler = self.build_optimizer_and_scheduler(stage)

            while current_step < stage_end_step:
                current_step += 1
                self.vlm.train()
                optimizer.zero_grad()

                # Sample 16 examples for this update step
                step_examples = sampler.sample_update_batch(current_step)
                examples_seen += len(step_examples)

                # Split into microbatches of 8
                accum_loss = 0.0
                for mb_idx in range(self.grad_accum_steps):
                    mb_items = step_examples[mb_idx * self.micro_batch_size:(mb_idx + 1) * self.micro_batch_size]
                    mb_batch = collate_task2_batch(mb_items, pad_id)

                    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.float16):
                        outputs = self.vlm(
                            mb_batch["prefix_ids"].to(self.device),
                            mb_batch["suffix_ids"].to(self.device),
                            mb_batch["attention_mask"].to(self.device),
                            mb_batch["visual_features"].to(self.device) if mb_batch["visual_features"] is not None else None
                        )

                        labels = mb_batch["labels"].to(self.device)[:, 1:].contiguous()
                        logits = outputs.logits[:, -labels.shape[1]:, :].contiguous()

                        # Example-weighted loss: average token loss within each example, then across batch
                        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
                        token_losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)).view(labels.size())
                        val_mask = (labels != -100)
                        ex_losses = []
                        for i in range(labels.size(0)):
                            v_count = val_mask[i].sum().item()
                            if v_count > 0:
                                ex_losses.append((token_losses[i] * val_mask[i]).sum() / v_count)

                        if ex_losses:
                            mb_loss = torch.stack(ex_losses).mean() / self.grad_accum_steps
                        else:
                            mb_loss = logits.sum() * 0.0

                        self.scaler.scale(mb_loss).backward()
                        accum_loss += mb_loss.item() * self.grad_accum_steps

                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.vlm.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()

                if current_step % 25 == 0 or current_step in self.checkpoint_steps:
                    lr_val = scheduler.get_last_lr()[0]
                    print(f"[Stage {stage}] Step {current_step}/{stage_end_step} (Total: {current_step}/4500) | "
                          f"Loss: {accum_loss:.4f} | LR: {lr_val:.2e} | Peak VRAM: {self.mem_tracker.get_peak_memory_gb():.2f} GB")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                training_logs.append({
                    "step": current_step,
                    "stage": stage,
                    "loss": accum_loss,
                    "examples_seen": examples_seen,
                    "peak_vram_gb": self.mem_tracker.get_peak_memory_gb()
                })

                # Development evaluation check
                if current_step in self.dev_eval_steps:
                    dev_record = self.evaluate_development_suite(current_step, dev_captions, grounding_dev, vqa_dev, cached_features)
                    dev_eval_logs.append(dev_record)

                # Checkpoint check
                if current_step in self.checkpoint_steps:
                    ckpt_p = self.output_dir / f"checkpoint_step_{current_step}.pt"
                    torch.save({
                        "step": current_step,
                        "stage": stage,
                        "examples_seen": examples_seen,
                        "connector_state": self.vlm.connector.state_dict(),
                        "lora_state": self.vlm.lora_model.state_dict() if self.vlm.lora_enabled else None
                    }, ckpt_p)
                    print(f"Saved checkpoint to {ckpt_p}")

            # Save stage endpoints connector & adapter weights
            torch.save(self.vlm.connector.state_dict(), self.output_dir / f"connector_stage_{stage}.pt")
            if self.vlm.lora_enabled:
                torch.save(self.vlm.lora_model.state_dict(), self.output_dir / f"lora_stage_{stage}.pt")

        total_time = time.perf_counter() - start_time
        summary = {
            "total_steps": current_step,
            "total_train_time_mins": total_time / 60.0,
            "peak_gpu_mem_gb": self.mem_tracker.get_peak_memory_gb(),
            "training_logs": training_logs,
            "dev_eval_logs": dev_eval_logs
        }
        save_json(summary, self.output_dir / "train_summary.json")
        save_json(dev_eval_logs, self.output_dir / "dev_eval_metrics.json")
        print(f"\nTask 2 Training Complete in {total_time/60.0:.2f} mins. Peak VRAM: {summary['peak_gpu_mem_gb']:.2f} GB.")
        return summary
