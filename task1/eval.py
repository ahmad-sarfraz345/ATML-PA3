from __future__ import annotations
import os
import sys
import copy
import json
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM
from common.utils import set_seed, get_device, save_json, load_json
from task1.models import apply_lora_to_model, apply_oplora_to_model

def verify_and_merge_adapter_model(
    model_name_or_path: str,
    ckpt_path: str,
    variant_name: str,
    lora_r: int = 16,
    lora_alpha: float = 32.0,
    oplora_k: int = 128,
    device: Optional[torch.device] = None
) -> AutoModelForCausalLM:
    """
    Loads checkpoint, instantiates unmerged model, checks logits against merged model
    on a small batch, and returns the merged model in FP16 for clean zero-shot evaluation.
    """
    device = device or get_device()
    print(f"\n--- Loading and Verifying Adapter Merge for [{variant_name}] ---")

    # 1. Load unmerged adapter model
    unmerged_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    if variant_name == "lora_4":
        apply_lora_to_model(unmerged_model, r=4, alpha=8.0, dropout=0.0)
    elif variant_name == "lora_16":
        apply_lora_to_model(unmerged_model, r=16, alpha=32.0, dropout=0.0)
    elif variant_name == "oplora_16":
        apply_oplora_to_model(unmerged_model, r=16, alpha=32.0, k=oplora_k, dropout=0.0, device=device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    unmerged_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    unmerged_model.to(device)
    unmerged_model.eval()

    # Small synthetic verification batch for unmerged logits
    test_input_ids = torch.randint(0, 1000, (2, 32), device=device)
    with torch.no_grad():
        out_unmerged = unmerged_model(test_input_ids).logits

    # 2. In-place merge adapters into base layers: W0 <- W0 + delta_W
    for name, module in unmerged_model.named_modules():
        if hasattr(module, "merge_into_base"):
            module.merge_into_base()

    # 3. Clean state dict to map back to standard CausalLM architecture
    # e.g., 'model.layers.0.mlp.down_proj.base_layer.weight' -> 'model.layers.0.mlp.down_proj.weight'
    cleaned_state_dict = {}
    for k, v in unmerged_model.state_dict().items():
        if any(k.endswith(suffix) for suffix in [".lora_A", ".lora_B", ".U_k", ".V_k"]):
            continue
        new_key = k.replace(".base_layer", "")
        cleaned_state_dict[new_key] = v

    # 4. Load fresh backbone for merged model and load merged weights
    merged_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    merged_model.load_state_dict(cleaned_state_dict, strict=True)
    merged_model.to(device)
    merged_model.eval()

    # 5. Check merged vs unmerged logits on verification batch
    with torch.no_grad():
        out_merged = merged_model(test_input_ids).logits
        max_abs_diff = torch.max(torch.abs(out_unmerged - out_merged)).item()

    print(f"Logits check for [{variant_name}]: max |logits_unmerged - logits_merged| = {max_abs_diff:.6e}")
    if max_abs_diff > 1e-2:
        print(f"WARNING: Logits discrepancy ({max_abs_diff}) higher than expected!")
    else:
        print("Merged vs unmerged logits verification PASSED.")

    # Free unmerged model from GPU
    del unmerged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return merged_model

    print(f"Logits check for [{variant_name}]: max |logits_unmerged - logits_merged| = {max_abs_diff:.6e}")
    if max_abs_diff > 1e-2:
        print(f"WARNING: Logits discrepancy ({max_abs_diff}) higher than expected!")
    else:
        print("Merged vs unmerged logits verification PASSED.")

    # Free unmerged model from GPU
    del unmerged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return merged_model

def load_evaluation_model(
    condition: str,
    base_model_path: str = "Qwen/Qwen3-0.6B",
    ckpt_dir: str = "outputs/task1",
    device: Optional[torch.device] = None
) -> AutoModelForCausalLM:
    """
    Loads model corresponding to one of the 7 evaluation conditions:
    1. baseline
    2. full_ft_midpoint
    3. full_ft_final
    4. full_ft_l2sp_final
    5. lora_4_final
    6. lora_16_final
    7. oplora_16_final
    """
    device = device or get_device()
    ckpt_base = Path(ckpt_dir)

    if condition == "baseline":
        print(f"Loading unmodified baseline from {base_model_path} in FP16...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model.to(device)
        model.eval()
        return model

    elif condition == "full_ft_midpoint":
        ckpt_path = ckpt_base / "full_ft" / "checkpoint_step_625.pt"
        if not ckpt_path.exists():
            print(f"Notice: Checkpoint for [{condition}] not found at {ckpt_path}. Skipping.")
            return None
        print(f"Loading Full FT Midpoint from {ckpt_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(dtype=torch.float16, device=device)
        model.eval()
        return model

    elif condition == "full_ft_final":
        ckpt_path = ckpt_base / "full_ft" / "checkpoint_step_1250.pt"
        if not ckpt_path.exists():
            print(f"Notice: Checkpoint for [{condition}] not found at {ckpt_path}. Skipping.")
            return None
        print(f"Loading Full FT Final from {ckpt_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(dtype=torch.float16, device=device)
        model.eval()
        return model

    elif condition == "full_ft_l2sp_final":
        ckpt_path = ckpt_base / "full_ft_l2sp" / "checkpoint_step_1250.pt"
        if not ckpt_path.exists():
            print(f"Notice: Checkpoint for [{condition}] not found at {ckpt_path}. Skipping.")
            return None
        print(f"Loading Full FT + L2-SP Final from {ckpt_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(dtype=torch.float16, device=device)
        model.eval()
        return model

    elif condition == "lora_4_final":
        ckpt_path = ckpt_base / "lora_4" / "checkpoint_step_1250.pt"
        if not ckpt_path.exists():
            print(f"Notice: Checkpoint for [{condition}] not found at {ckpt_path}. Skipping.")
            return None
        return verify_and_merge_adapter_model(
            base_model_path, str(ckpt_path), "lora_4", lora_r=4, lora_alpha=8.0, device=device
        )

    elif condition == "lora_16_final":
        ckpt_path = ckpt_base / "lora_16" / "checkpoint_step_1250.pt"
        if not ckpt_path.exists():
            print(f"Notice: Checkpoint for [{condition}] not found at {ckpt_path}. Skipping.")
            return None
        return verify_and_merge_adapter_model(
            base_model_path, str(ckpt_path), "lora_16", lora_r=16, lora_alpha=32.0, device=device
        )

    elif condition == "oplora_16_final":
        ckpt_path = ckpt_base / "oplora_16" / "checkpoint_step_1250.pt"
        if not ckpt_path.exists():
            print(f"Notice: Checkpoint for [{condition}] not found at {ckpt_path}. Skipping.")
            return None
        return verify_and_merge_adapter_model(
            base_model_path, str(ckpt_path), "oplora_16", lora_r=16, lora_alpha=32.0, oplora_k=128, device=device
        )

    else:
        raise ValueError(f"Unknown evaluation condition: {condition}")

def evaluate_condition_benchmarks(
    condition: str,
    base_model_path: str = "Qwen/Qwen3-0.6B",
    ckpt_dir: str = "outputs/task1",
    output_dir: str = "outputs/task1/benchmarks",
    tasks: Optional[List[str]] = None,
    batch_size: int = 8
) -> Dict[str, Any]:
    """
    Evaluates a specific model condition across the 4 assignment benchmarks using lm-eval:
    - gsm8k (gsm8k_cot with greedy generation, 1024 new tokens)
    - hellaswag (likelihood, acc & acc_norm)
    - boolq (likelihood, acc)
    - arc_easy (likelihood, acc & acc_norm)
    """
    output_path = Path(output_dir) / f"{condition}_results.json"
    if output_path.exists():
        print(f"Results for [{condition}] already exist at {output_path}. Loading...")
        return load_json(output_path)

    if tasks is None:
        tasks = ["gsm8k_cot", "hellaswag", "boolq", "arc_easy"]

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load condition model in FP16
    model = load_evaluation_model(condition, base_model_path, ckpt_dir, device)
    if model is None:
        print(f"Skipping evaluation for [{condition}] because checkpoint is missing.", flush=True)
        return {}

    # Wrap model with lm-eval HFLM
    hflm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=2048,
        device=str(device)
    )

    print(f"\n========================================================")
    print(f"Running zero-shot benchmark evaluations for [{condition}]...")
    print(f"Tasks: {tasks}")
    print(f"========================================================")

    eval_results = lm_eval.simple_evaluate(
        model=hflm,
        tasks=tasks,
        num_fewshot=0,
        batch_size=batch_size,
        log_samples=True
    )

    # Format summary dictionary
    summary_metrics = {
        "condition": condition,
        "results": eval_results["results"],
        "versions": eval_results.get("versions", {}),
        "n-shot": eval_results.get("n-shot", {})
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary_metrics, output_path)

    # Save samples separately if available
    if "samples" in eval_results:
        samples_path = Path(output_dir) / f"{condition}_samples.json"
        save_json(eval_results["samples"], samples_path)

    print(f"Saved evaluation results for [{condition}] to {output_path}")

    del hflm
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary_metrics

