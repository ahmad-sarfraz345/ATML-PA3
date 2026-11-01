import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from common.utils import load_json, save_json

CONDITIONS = [
    "baseline",
    "full_ft_midpoint",
    "full_ft_final",
    "full_ft_l2sp_final",
    "lora_4_final",
    "lora_16_final",
    "oplora_16_final"
]

def generate_benchmark_summary_table(
    bench_dir: str = "outputs/task1/benchmarks",
    output_dir: str = "outputs/task1"
) -> Dict[str, Any]:
    """
    Compiles master benchmark table across all 7 evaluation conditions and 4 benchmarks:
    - GSM8K (flexible-extraction exact match)
    - HellaSwag (normalized accuracy & raw accuracy)
    - BoolQ (accuracy)
    - ARC-Easy (normalized accuracy & raw accuracy)
    Computes absolute percentage point differences relative to baseline.
    """
    bench_dir = Path(bench_dir)
    results_by_cond = {}

    for cond in CONDITIONS:
        res_file = bench_dir / f"{cond}_results.json"
        if res_file.exists():
            data = load_json(res_file)["results"]
            results_by_cond[cond] = data
        else:
            print(f"Warning: Benchmark results missing for {cond}")

    if "baseline" not in results_by_cond:
        print("Baseline results not found; cannot compute percentage point deltas.")
        return {}

    baseline_data = results_by_cond["baseline"]

    def extract_scores(data_dict):
        gsm8k = data_dict.get("gsm8k_cot", {}).get("exact_match,flexible-extract", 0.0) * 100
        hella_norm = data_dict.get("hellaswag", {}).get("acc_norm,none", 0.0) * 100
        hella_raw = data_dict.get("hellaswag", {}).get("acc,none", 0.0) * 100
        boolq_acc = data_dict.get("boolq", {}).get("acc,none", 0.0) * 100
        arc_norm = data_dict.get("arc_easy", {}).get("acc_norm,none", 0.0) * 100
        arc_raw = data_dict.get("arc_easy", {}).get("acc,none", 0.0) * 100
        return {
            "gsm8k_em": gsm8k,
            "hellaswag_acc_norm": hella_norm,
            "hellaswag_acc": hella_raw,
            "boolq_acc": boolq_acc,
            "arc_easy_acc_norm": arc_norm,
            "arc_easy_acc": arc_raw
        }

    base_scores = extract_scores(baseline_data)
    summary_rows = []

    for cond in CONDITIONS:
        if cond not in results_by_cond:
            continue
        scores = extract_scores(results_by_cond[cond])
        row = {
            "Condition": cond,
            "GSM8K (EM %)": round(scores["gsm8k_em"], 2),
            "GSM8K (Delta pp)": round(scores["gsm8k_em"] - base_scores["gsm8k_em"], 2),
            "HellaSwag (Norm Acc %)": round(scores["hellaswag_acc_norm"], 2),
            "HellaSwag (Delta pp)": round(scores["hellaswag_acc_norm"] - base_scores["hellaswag_acc_norm"], 2),
            "BoolQ (Acc %)": round(scores["boolq_acc"], 2),
            "BoolQ (Delta pp)": round(scores["boolq_acc"] - base_scores["boolq_acc"], 2),
            "ARC-Easy (Norm Acc %)": round(scores["arc_easy_acc_norm"], 2),
            "ARC-Easy (Delta pp)": round(scores["arc_easy_acc_norm"] - base_scores["arc_easy_acc_norm"], 2)
        }
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    out_json = Path(output_dir) / "benchmark_summary.json"
    out_csv = Path(output_dir) / "benchmark_summary.csv"
    out_md = Path(output_dir) / "benchmark_summary.md"

    save_json(summary_rows, out_json)
    df.to_csv(out_csv, index=False)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))

    print(f"Generated benchmark summary table at {out_json} and {out_md}")
    return summary_rows

def plot_combined_learning_curves(
    task1_dir: str = "outputs/task1",
    output_path: str = "outputs/task1/learning_curves.png"
):
    """
    Plots combined SFT loss against optimizer updates for the five fine-tuning variants.
    Identifies checkpoint steps [312, 625, 937, 1250].
    """
    task1_dir = Path(task1_dir)
    variants = ["full_ft", "full_ft_l2sp", "lora_4", "lora_16", "oplora_16"]
    colors = {
        "full_ft": "#1f77b4",
        "full_ft_l2sp": "#ff7f0e",
        "lora_4": "#2ca02c",
        "lora_16": "#d62728",
        "oplora_16": "#9467bd"
    }

    plt.figure(figsize=(10, 6), dpi=300)
    found_any = False

    for var in variants:
        log_file = task1_dir / var / "training_logs.json"
        if log_file.exists():
            found_any = True
            logs = load_json(log_file)
            steps = [x["step"] for x in logs]
            sft_losses = [x["sft_loss"] for x in logs]
            plt.plot(steps, sft_losses, label=var, color=colors.get(var, "black"), linewidth=1.8)

    if not found_any:
        print("No training logs found to plot.")
        plt.close()
        return

    # Highlight checkpoint steps
    for s in [312, 625, 937, 1250]:
        plt.axvline(x=s, color="gray", linestyle="--", alpha=0.5, label="Checkpoint" if s == 312 else None)

    plt.title("MetaMathQA SFT Training Loss vs Optimizer Updates (1,250 steps)", fontsize=14, fontweight="bold")
    plt.xlabel("Optimizer Update Step", fontsize=12)
    plt.ylabel("Supervised Cross-Entropy Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved learning curves plot to {output_path}")

def generate_resource_comparison_table(
    task1_dir: str = "outputs/task1",
    output_dir: str = "outputs/task1"
) -> List[Dict[str, Any]]:
    """
    Summarizes trainable parameters, training time, peak GPU memory,
    and checkpoint sizes for the five training variants.
    """
    task1_dir = Path(task1_dir)
    variants = ["full_ft", "full_ft_l2sp", "lora_4", "lora_16", "oplora_16"]
    rows = []

    for var in variants:
        summary_file = task1_dir / var / "train_summary.json"
        if summary_file.exists():
            data = load_json(summary_file)
            # Checkpoint size
            ckpt_file = task1_dir / var / "checkpoint_step_1250.pt"
            ckpt_size_mb = (ckpt_file.stat().st_size / (1024 * 1024)) if ckpt_file.exists() else 0.0

            rows.append({
                "Variant": var,
                "Trainable Parameters": f"{data.get('trainable_parameters', 0):,}",
                "Trainable %": round(data.get("trainable_parameters", 0) / max(1, data.get("total_parameters", 1)) * 100, 2),
                "Training Time (min)": round(data.get("total_train_time_sec", 0.0) / 60.0, 2),
                "Peak GPU VRAM (GB)": round(data.get("peak_gpu_mem_gb", 0.0), 2),
                "Final Checkpoint (MB)": round(ckpt_size_mb, 2),
                "Full-Model Displacement": round(data.get("final_displacement", 0.0), 4) if data.get("final_displacement") is not None else "N/A"
            })

    out_json = Path(output_dir) / "resource_usage_summary.json"
    save_json(rows, out_json)
    if rows:
        df = pd.DataFrame(rows)
        out_md = Path(output_dir) / "resource_usage_summary.md"
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(df.to_markdown(index=False))
        print(f"Generated resource usage summary at {out_json} and {out_md}")
    return rows
