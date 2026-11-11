from __future__ import annotations
import os
import json
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from common.utils import load_json, save_json
from task2.eval import (
    vqa_normalize_answer,
    compute_vqa_consensus_score,
    compute_grounding_exact_match,
    bootstrap_pair_metric_confidence_intervals
)
from task2.data import collate_task2_batch

def run_final_test_evaluations(
    vlm,
    grounding_test: List[Dict[str, Any]],
    vqa_test: List[Dict[str, Any]],
    cached_features: Dict[int, Any],
    output_dir: str = "outputs/task2",
    seed: int = 6304
) -> Dict[str, Any]:
    """
    Evaluates both final-test suites across 4 image conditions:
    1. Correct image
    2. Complementary swap image
    3. Mismatched image (unrelated image from another pair, fixed mapping)
    4. No image (visual tokens omitted)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = vlm.device
    pad_id = vlm.tokenizer.pad_token_id

    rng = np.random.default_rng(seed)

    # 1. Fixed mismatched mapping for Grounding test
    n_g = len(grounding_test)
    perm_g = (np.arange(n_g) + 1) % n_g  # Shift by 1 ensures different pair
    # Fixed mismatched mapping for VQA test
    n_v = len(vqa_test)
    perm_v = (np.arange(n_v) + 1) % n_v

    conditions = ["correct", "complementary_swap", "mismatched", "no_image"]

    def eval_suite_condition(suite_name: str, pair_list: List[Dict[str, Any]], condition: str, perm_indices: np.ndarray):
        pair_records = []
        gen_times = []

        for p_idx, p in enumerate(pair_list):
            q_text = p["question"]
            task_type = p.get("task", p.get("answer_type", "unknown"))

            # Determine images for pair
            if condition == "correct":
                img1, img2 = p["image_1"], p["image_2"]
            elif condition == "complementary_swap":
                img1, img2 = p["image_2"], p["image_1"]  # Swapped partner
            elif condition == "mismatched":
                partner_pair = pair_list[perm_indices[p_idx]]
                img1, img2 = partner_pair["image_1"], partner_pair["image_2"]
            else:  # no_image
                img1, img2 = None, None

            # Reference canonical targets (always original canonical answers)
            ans1_canon = p.get("canonical_answer_1", p.get("answer_1"))
            ans2_canon = p.get("canonical_answer_2", p.get("answer_2"))
            all_ans1 = p.get("all_answers_1", [ans1_canon])
            all_ans2 = p.get("all_answers_2", [ans2_canon])

            def make_eval_item(img_id):
                prefix_ids = vlm.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
                suffix_ids = vlm.tokenizer.encode(f"{q_text}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
                feat = cached_features.get(img_id, None) if img_id is not None else None
                return {
                    "prefix_ids": prefix_ids,
                    "suffix_ids": suffix_ids,
                    "labels": [-100] * len(suffix_ids),
                    "visual_feature": feat
                }

            # Generate prediction for example 1
            t0 = time.perf_counter()
            b1 = collate_task2_batch([make_eval_item(img1)], pad_id)
            pred1 = vlm.generate_answer(b1["prefix_ids"].to(device), b1["suffix_ids"].to(device), b1["visual_features"], max_new_tokens=16)[0]
            gen_times.append(time.perf_counter() - t0)

            # Generate prediction for example 2
            t0 = time.perf_counter()
            b2 = collate_task2_batch([make_eval_item(img2)], pad_id)
            pred2 = vlm.generate_answer(b2["prefix_ids"].to(device), b2["suffix_ids"].to(device), b2["visual_features"], max_new_tokens=16)[0]
            gen_times.append(time.perf_counter() - t0)

            # Scoring
            if suite_name == "grounding":
                score1 = compute_grounding_exact_match(pred1, ans1_canon)
                score2 = compute_grounding_exact_match(pred2, ans2_canon)
                is_pair_correct = (score1 == 1.0 and score2 == 1.0)
            else:
                score1 = compute_vqa_consensus_score(pred1, all_ans1)
                score2 = compute_vqa_consensus_score(pred2, all_ans2)
                # Pair correct if both canonical match
                is_pair_correct = (vqa_normalize_answer(pred1) == vqa_normalize_answer(ans1_canon) and
                                   vqa_normalize_answer(pred2) == vqa_normalize_answer(ans2_canon))

            pair_identical = (vqa_normalize_answer(pred1) == vqa_normalize_answer(pred2) and len(vqa_normalize_answer(pred1)) > 0)
            pair_empty = (len(pred1.strip()) == 0 or len(pred2.strip()) == 0)

            pair_records.append({
                "pair_idx": p_idx,
                "task_type": task_type,
                "question": q_text,
                "pred_1": pred1,
                "pred_2": pred2,
                "ans_canon_1": ans1_canon,
                "ans_canon_2": ans2_canon,
                "score_1": score1,
                "score_2": score2,
                "mean_acc": 0.5 * (score1 + score2),
                "pair_correct": float(is_pair_correct),
                "identical_answer": float(pair_identical),
                "empty_answer": float(pair_empty)
            })

        # Calculate summary metrics & bootstrap 95% CIs
        mean_acc, acc_ci_low, acc_ci_high = bootstrap_pair_metric_confidence_intervals(
            pair_records, lambda plist: float(np.mean([p["mean_acc"] for p in plist])), seed=seed
        )
        mean_pair_acc, pair_ci_low, pair_ci_high = bootstrap_pair_metric_confidence_intervals(
            pair_records, lambda plist: float(np.mean([p["pair_correct"] for p in plist])), seed=seed
        )
        mean_ident = float(np.mean([p["identical_answer"] for p in pair_records])) * 100.0
        mean_empty = float(np.mean([p["empty_answer"] for p in pair_records])) * 100.0
        avg_gen_time_ms = float(np.mean(gen_times)) * 1000.0

        # Sub-task accuracies
        by_task = {}
        for p in pair_records:
            by_task.setdefault(p["task_type"], []).append(p["mean_acc"])
        task_accs = {k: float(np.mean(v)) * 100.0 for k, v in by_task.items()}

        return {
            "condition": condition,
            "accuracy_percent": round(mean_acc * 100.0, 2),
            "accuracy_95_ci": [round(acc_ci_low * 100.0, 2), round(acc_ci_high * 100.0, 2)],
            "pair_accuracy_percent": round(mean_pair_acc * 100.0, 2),
            "pair_accuracy_95_ci": [round(pair_ci_low * 100.0, 2), round(pair_ci_high * 100.0, 2)],
            "identical_answer_rate_percent": round(mean_ident, 2),
            "empty_answer_rate_percent": round(mean_empty, 2),
            "avg_generation_time_ms": round(avg_gen_time_ms, 2),
            "task_type_accuracies": task_accs,
            "pair_records": pair_records
        }

    all_results = {}
    print("\n--- Evaluating Grounding Final Test (64 Pairs / 128 Questions) ---")
    all_results["grounding"] = {}
    for cond in conditions:
        print(f"Evaluating Grounding Test under [{cond}] condition...")
        all_results["grounding"][cond] = eval_suite_condition("grounding", grounding_test, cond, perm_g)

    print("\n--- Evaluating Natural VQAv2 Final Test (96 Pairs / 192 Questions) ---")
    all_results["vqa"] = {}
    for cond in conditions:
        print(f"Evaluating Natural VQAv2 Test under [{cond}] condition...")
        all_results["vqa"][cond] = eval_suite_condition("vqa", vqa_test, cond, perm_v)

    # Save complete evaluation output
    save_json(all_results, output_dir / "final_test_results.json")
    print(f"Saved final evaluation results to {output_dir / 'final_test_results.json'}")
    return all_results

def generate_task2_plots_and_tables(
    train_summary_path: str = "outputs/task2/train_summary.json",
    final_results_path: str = "outputs/task2/final_test_results.json",
    output_dir: str = "outputs/task2"
):
    """
    Synthesizes Task 2 evidence:
    1. learning_curves.png (training loss, dev caption loss, dev answer NLL, and delta NLL across stages)
    2. final_results_summary.md (markdown tables for Grounding and VQA with bootstrap CIs)
    3. qualitative_examples.json (illustrative complementary pairs)
    4. resource_usage_summary.md
    """
    output_dir = Path(output_dir)
    train_data = load_json(train_summary_path) if Path(train_summary_path).exists() else {}
    final_data = load_json(final_results_path) if Path(final_results_path).exists() else {}

    # 1. Learning Curves Plot
    if "training_logs" in train_data and "dev_eval_logs" in train_data:
        t_logs = train_data["training_logs"]
        d_logs = train_data["dev_eval_logs"]

        steps_train = [entry["step"] for entry in t_logs]
        loss_train = [entry["loss"] for entry in t_logs]

        steps_dev = [entry["step"] for entry in d_logs]
        cap_loss = [entry["dev_caption_loss"] for entry in d_logs]
        delta_nll_ground = [entry["grounding_dev"]["delta_nll_swap_penalty"] for entry in d_logs]
        delta_nll_vqa = [entry["vqa_dev"]["delta_nll_swap_penalty"] for entry in d_logs]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True)

        # Subplot 1: Training & Dev Losses
        ax1.plot(steps_train, loss_train, label="SFT Loss (Example-Weighted)", color="#1f77b4", alpha=0.6, linewidth=1.2)
        ax1.plot(steps_dev, cap_loss, label="Dev Caption Loss", color="#d62728", marker="o", linewidth=2.0)
        ax1.axvline(x=2250, color="gray", linestyle="--", label="Stage 1 -> 2 (LoRA Added)")
        ax1.axvline(x=3750, color="black", linestyle="--", label="Stage 2 -> 3 (VQA Focus)")
        ax1.set_ylabel("Loss / NLL", fontsize=11)
        ax1.set_title("Task 2: VLM Curriculum Learning Curves", fontsize=13, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right")

        # Subplot 2: Image-Swap Penalty Delta NLL
        ax2.plot(steps_dev, delta_nll_ground, label="Grounding Swap Penalty (Delta NLL)", color="#2ca02c", marker="s", linewidth=2.0)
        ax2.plot(steps_dev, delta_nll_vqa, label="Natural VQA Swap Penalty (Delta NLL)", color="#ff7f0e", marker="^", linewidth=2.0)
        ax2.axvline(x=2250, color="gray", linestyle="--")
        ax2.axvline(x=3750, color="black", linestyle="--")
        ax2.axhline(y=0.0, color="black", linestyle=":", alpha=0.5)
        ax2.set_xlabel("Optimizer Updates", fontsize=11)
        ax2.set_ylabel("Image Swap Penalty Delta NLL", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper left")

        plt.tight_layout()
        plot_path = output_dir / "learning_curves.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved learning curves plot to {plot_path}")

    # 2. Final Results Markdown Table
    md_lines = ["# Task 2: Visual Grounding Evaluation Summary\n\n"]
    for suite in ["grounding", "vqa"]:
        if suite in final_data:
            s_title = "Annotation-Derived Grounding Test" if suite == "grounding" else "Natural VQAv2 Test"
            md_lines.append(f"## {s_title}\n\n")
            rows = []
            for cond in ["correct", "complementary_swap", "mismatched", "no_image"]:
                cd = final_data[suite].get(cond, {})
                row = {
                    "Condition": cond.replace("_", " ").title(),
                    "Accuracy (%)": f"{cd.get('accuracy_percent', 0.0):.2f}%",
                    "95% CI": f"[{cd.get('accuracy_95_ci', [0,0])[0]:.2f}, {cd.get('accuracy_95_ci', [0,0])[1]:.2f}]",
                    "Pair Acc (%)": f"{cd.get('pair_accuracy_percent', 0.0):.2f}%",
                    "Identical (%)": f"{cd.get('identical_answer_rate_percent', 0.0):.2f}%",
                    "Empty (%)": f"{cd.get('empty_answer_rate_percent', 0.0):.2f}%",
                    "Latency (ms)": f"{cd.get('avg_generation_time_ms', 0.0):.1f}"
                }
                rows.append(row)
            df = pd.DataFrame(rows)
            md_lines.append(df.to_markdown(index=False) + "\n\n")

    summary_md_path = output_dir / "final_results_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("".join(md_lines))
    print(f"Generated results summary table at {summary_md_path}")
