from __future__ import annotations
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from common.utils import get_device, save_json, load_json
from task1.models import apply_lora_to_model, apply_oplora_to_model

TARGET_LAYERS = [0, 13, 27]

def compute_weight_space_diagnostics(
    base_model_path: str = "Qwen/Qwen3-0.6B",
    ckpt_dir: str = "outputs/task1",
    k: int = 128
) -> Dict[str, Any]:
    """
    Computes weight-space diagnostics for down_proj in layers 0, 13, and 27:
    - rho_k = ||Uk Uk^T delta_W||_F^2 / ||delta_W||_F^2
    - eta = ||delta_W||_F / ||W0||_F
    For all 5 final variants (step 1250) and Full FT midpoint (step 625).
    """
    print("\n--- Running Weight-Space Diagnostics on Layers [0, 13, 27] ---")
    device = torch.device("cpu")  # FP32 on CPU or GPU

    # 1. Load base model and compute SVD bases Uk for down_proj in layers 0, 13, 27
    print(f"Loading starting model W0 from {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )

    w0_dict = {}
    uk_dict = {}
    for layer_idx in TARGET_LAYERS:
        down_proj = base_model.model.layers[layer_idx].mlp.down_proj
        w0 = down_proj.weight.data.clone().to(torch.float32)
        w0_dict[layer_idx] = w0

        # SVD of W0
        U, S, Vh = torch.linalg.svd(w0, full_matrices=False)
        uk = U[:, :k].contiguous()  # (dout, k)
        uk_dict[layer_idx] = uk

    # 2. Checkpoints to evaluate
    checkpoints = {
        "full_ft_midpoint": ("full_ft", 625),
        "full_ft_final": ("full_ft", 1250),
        "full_ft_l2sp_final": ("full_ft_l2sp", 1250),
        "lora_4_final": ("lora_4", 1250),
        "lora_16_final": ("lora_16", 1250),
        "oplora_16_final": ("oplora_16", 1250)
    }

    results = {}
    ckpt_base = Path(ckpt_dir)

    for cond_name, (variant, step) in checkpoints.items():
        ckpt_path = ckpt_base / variant / f"checkpoint_step_{step}.pt"
        if not ckpt_path.exists():
            print(f"Skipping {cond_name}: {ckpt_path} not found.")
            continue

        print(f"Analyzing {cond_name}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"]

        results[cond_name] = {}

        for layer_idx in TARGET_LAYERS:
            w0 = w0_dict[layer_idx]
            uk = uk_dict[layer_idx]

            # Extract delta_W
            if variant in ["full_ft", "full_ft_l2sp"]:
                weight_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
                w_trained = state_dict[weight_key].to(torch.float32)
                delta_w = w_trained - w0
            elif variant in ["lora_4", "lora_16"]:
                r = 4 if variant == "lora_4" else 16
                alpha = 8.0 if variant == "lora_4" else 32.0
                scaling = alpha / r
                a_key = f"model.layers.{layer_idx}.mlp.down_proj.lora_A"
                b_key = f"model.layers.{layer_idx}.mlp.down_proj.lora_B"
                lora_a = state_dict[a_key].to(torch.float32)
                lora_b = state_dict[b_key].to(torch.float32)
                delta_w = scaling * (lora_b @ lora_a)
            elif variant == "oplora_16":
                scaling = 32.0 / 16.0
                a_key = f"model.layers.{layer_idx}.mlp.down_proj.lora_A"
                b_key = f"model.layers.{layer_idx}.mlp.down_proj.lora_B"
                uk_key = f"model.layers.{layer_idx}.mlp.down_proj.U_k"
                vk_key = f"model.layers.{layer_idx}.mlp.down_proj.V_k"

                lora_a = state_dict[a_key].to(torch.float32)
                lora_b = state_dict[b_key].to(torch.float32)
                layer_uk = state_dict[uk_key].to(torch.float32)
                layer_vk = state_dict[vk_key].to(torch.float32)

                b_tilde = lora_b - layer_uk @ (layer_uk.T @ lora_b)
                a_tilde = lora_a - (lora_a @ layer_vk) @ layer_vk.T
                delta_w = scaling * (b_tilde @ a_tilde)

            # Compute metrics
            # 1. Subspace interference ratio rho_k
            # ||Uk Uk^T delta_W||_F^2 = ||Uk^T delta_W||_F^2 because Uk is orthonormal
            proj_delta_w = uk.T @ delta_w  # (k, din)
            norm_proj_sq = torch.sum(proj_delta_w ** 2).item()
            norm_delta_sq = torch.sum(delta_w ** 2).item()
            rho_k = (norm_proj_sq / max(norm_delta_sq, 1e-12)) if norm_delta_sq > 0 else 0.0

            # 2. Relative update magnitude eta
            norm_w0 = torch.norm(w0, p="fro").item()
            norm_delta = torch.norm(delta_w, p="fro").item()
            eta = norm_delta / max(norm_w0, 1e-12)

            results[cond_name][f"layer_{layer_idx}"] = {
                "subspace_interference_rho_k": rho_k,
                "relative_update_magnitude_eta": eta,
                "delta_w_frobenius_norm": norm_delta,
                "w0_frobenius_norm": norm_w0
            }

    out_file = Path(ckpt_dir) / "weight_space_diagnostics.json"
    save_json(results, out_file)
    print(f"Saved weight-space diagnostics to {out_file}")
    return results

def compute_activation_diagnostics(
    base_model_path: str = "Qwen/Qwen3-0.6B",
    probe_set_path: str = "outputs/task1/probe_set.json",
    ckpt_dir: str = "outputs/task1"
) -> Dict[str, Any]:
    """
    Computes activation diagnostics R(z) = ||delta_W z||_2 / (||W0 z||_2 + 10^-8)
    for math vs non-math prompts on down_proj inputs z at the last non-padding prompt token.
    Evaluated in FP32 across layers [0, 13, 27].
    """
    print("\n--- Running Activation Diagnostics on Probe Set ---")
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    probe_data = load_json(probe_set_path)
    math_probes = probe_data["gsm8k_math_probes"]
    nonmath_probes = probe_data["hellaswag_nonmath_probes"]

    # 1. Forward hook to cache z (down_proj input at last prompt token)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to(device)
    base_model.eval()

    cached_activations = {"math": {}, "nonmath": {}}

    def make_hook(layer_idx, storage_list):
        def hook_fn(module, input, output):
            # input[0] shape: (batch_size, seq_len, din)
            # We want last non-padding prompt token
            z_last = input[0][0, -1, :].detach().cpu()
            storage_list.append(z_last)
        return hook_fn

    # Register hooks
    hook_handles = []
    layer_act_math = {l: [] for l in TARGET_LAYERS}
    layer_act_nonmath = {l: [] for l in TARGET_LAYERS}

    for l in TARGET_LAYERS:
        handle = base_model.model.layers[l].mlp.down_proj.register_forward_hook(
            make_hook(l, layer_act_math[l])
        )
        hook_handles.append(handle)

    print("Caching down_proj inputs z for 32 GSM8K math prompts...")
    with torch.no_grad():
        for item in math_probes:
            q = item["question"]
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            ) if hasattr(tokenizer, "apply_chat_template") else f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            base_model(**enc)

    for h in hook_handles:
        h.remove()

    hook_handles = []
    for l in TARGET_LAYERS:
        handle = base_model.model.layers[l].mlp.down_proj.register_forward_hook(
            make_hook(l, layer_act_nonmath[l])
        )
        hook_handles.append(handle)

    print("Caching down_proj inputs z for 32 HellaSwag non-math prompts...")
    with torch.no_grad():
        for item in nonmath_probes:
            ctx = item["ctx"]
            # HellaSwag uses stock completion prompt without chat template
            enc = tokenizer(ctx, return_tensors="pt").to(device)
            base_model(**enc)

    for h in hook_handles:
        h.remove()

    # Store base W0
    w0_dict = {l: base_model.model.layers[l].mlp.down_proj.weight.data.clone().cpu().to(torch.float32) for l in TARGET_LAYERS}

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Evaluate R(z) for each checkpoint
    checkpoints = {
        "full_ft_midpoint": ("full_ft", 625),
        "full_ft_final": ("full_ft", 1250),
        "full_ft_l2sp_final": ("full_ft_l2sp", 1250),
        "lora_4_final": ("lora_4", 1250),
        "lora_16_final": ("lora_16", 1250),
        "oplora_16_final": ("oplora_16", 1250)
    }

    ckpt_base = Path(ckpt_dir)
    act_results = {}

    for cond_name, (variant, step) in checkpoints.items():
        ckpt_path = ckpt_base / variant / f"checkpoint_step_{step}.pt"
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"]
        act_results[cond_name] = {}

        for l in TARGET_LAYERS:
            w0 = w0_dict[l]
            if variant in ["full_ft", "full_ft_l2sp"]:
                w_trained = state_dict[f"model.layers.{l}.mlp.down_proj.weight"].to(torch.float32)
                delta_w = w_trained - w0
            elif variant in ["lora_4", "lora_16"]:
                r = 4 if variant == "lora_4" else 16
                alpha = 8.0 if variant == "lora_4" else 32.0
                lora_a = state_dict[f"model.layers.{l}.mlp.down_proj.lora_A"].to(torch.float32)
                lora_b = state_dict[f"model.layers.{l}.mlp.down_proj.lora_B"].to(torch.float32)
                delta_w = (alpha / r) * (lora_b @ lora_a)
            elif variant == "oplora_16":
                lora_a = state_dict[f"model.layers.{l}.mlp.down_proj.lora_A"].to(torch.float32)
                lora_b = state_dict[f"model.layers.{l}.mlp.down_proj.lora_B"].to(torch.float32)
                uk = state_dict[f"model.layers.{l}.mlp.down_proj.U_k"].to(torch.float32)
                vk = state_dict[f"model.layers.{l}.mlp.down_proj.V_k"].to(torch.float32)
                b_tilde = lora_b - uk @ (uk.T @ lora_b)
                a_tilde = lora_a - (lora_a @ vk) @ vk.T
                delta_w = 2.0 * (b_tilde @ a_tilde)

            # Calculate R(z) = ||delta_W z||_2 / (||W0 z||_2 + 1e-8)
            def compute_rz_list(z_list):
                rz_vals = []
                for z in z_list:
                    # z shape: (din,)
                    dw_z = delta_w @ z
                    w0_z = w0 @ z
                    num = torch.norm(dw_z, p=2).item()
                    denom = torch.norm(w0_z, p=2).item() + 1e-8
                    rz_vals.append(num / denom)
                return float(np.mean(rz_vals)), float(np.std(rz_vals))

            math_mean, math_std = compute_rz_list(layer_act_math[l])
            nonmath_mean, nonmath_std = compute_rz_list(layer_act_nonmath[l])

            act_results[cond_name][f"layer_{l}"] = {
                "math_rz_mean": math_mean,
                "math_rz_std": math_std,
                "nonmath_rz_mean": nonmath_mean,
                "nonmath_rz_std": nonmath_std,
                "ratio_math_to_nonmath": math_mean / max(nonmath_mean, 1e-8)
            }

    out_file = Path(ckpt_dir) / "activation_diagnostics.json"
    save_json(act_results, out_file)
    print(f"Saved activation diagnostics to {out_file}")
    return act_results

