from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.utils import load_json, save_json

def generate_task3_visualizations_and_report(
    output_dir: str = "outputs/task3"
):
    """
    Synthesizes Task 3 evidence into plots and markdown tables:
    1. scaling_curves.png (Accuracy vs. N and Accuracy vs. Token Cost for MATH and MathVista)
    2. failure_decomposition.png (Sampling vs. Selection failure rates across budgets)
    3. task3_summary.md (Master tables, method comparisons, adaptive evaluation, live inference)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    math_file = out_dir / "offline_math_results.json"
    vista_file = out_dir / "offline_mathvista_results.json"
    live_file = out_dir / "live_refinement_results.json"
    adapt_math_file = out_dir / "adaptive_eval_math.json"
    adapt_vista_file = out_dir / "adaptive_eval_mathvista.json"
    step3_math_file = out_dir / "signal_analysis_math.json"
    step3_vista_file = out_dir / "signal_analysis_mathvista.json"

    math_data = load_json(math_file) if math_file.exists() else {}
    vista_data = load_json(vista_file) if vista_file.exists() else {}
    live_data = load_json(live_file) if live_file.exists() else {}
    adapt_math = load_json(adapt_math_file) if adapt_math_file.exists() else {}
    adapt_vista = load_json(adapt_vista_file) if adapt_vista_file.exists() else {}
    step3_math = load_json(step3_math_file) if step3_math_file.exists() else {}
    step3_vista = load_json(step3_vista_file) if step3_vista_file.exists() else {}

    # 1. Scaling Curves Plot (Accuracy vs. N and Accuracy vs. Token Cost)
    if "budgets" in math_data and "budgets" in vista_data:
        budgets = [int(b) for b in math_data["budgets"]]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

        # Subplot (0, 0): MATH Accuracy vs. N
        ax = axes[0, 0]
        # Oracle coverage
        o_acc = [math_data["oracle_coverage"][str(N)]["mean_percent"] for N in budgets]
        ax.plot(budgets, o_acc, label="Oracle Coverage", color="black", linestyle="--", marker="x", linewidth=1.5)

        m_colors = {
            "FirstAns": "gray",
            "Self-Consistency": "#1f77b4",
            "Highest-LogProb": "#2ca02c",
            "Highest-SelfCertainty": "#ff7f0e",
            "SelfCertainty-RWC": "#d62728"
        }
        for m, color in m_colors.items():
            if m in math_data["method_results"]:
                accs = [math_data["method_results"][m][str(N)]["accuracy_percent"] for N in budgets]
                ax.plot(budgets, accs, label=m, color=color, marker="o", linewidth=1.8)

        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.set_xlabel("Candidate Budget (N)", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title("MATH: Accuracy vs. Candidate Budget (N)", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=9)

        # Subplot (0, 1): MATH Accuracy vs. Generated Token Cost
        ax = axes[0, 1]
        for m, color in m_colors.items():
            if m in math_data["method_results"]:
                costs = [math_data["method_results"][m][str(N)]["mean_token_cost"] for N in budgets]
                accs = [math_data["method_results"][m][str(N)]["accuracy_percent"] for N in budgets]
                ax.plot(costs, accs, label=m, color=color, marker="o", linewidth=1.8)

        ax.set_xlabel("Mean Generated Token Cost", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title("MATH: Accuracy vs. Generated Token Cost", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=9)

        # Subplot (1, 0): MathVista Accuracy vs. N
        ax = axes[1, 0]
        o_acc_v = [vista_data["oracle_coverage"][str(N)]["mean_percent"] for N in budgets]
        ax.plot(budgets, o_acc_v, label="Oracle Coverage", color="black", linestyle="--", marker="x", linewidth=1.5)

        v_colors = {
            "FirstAns": "gray",
            "Self-Consistency": "#1f77b4",
            "VisConf-Highest": "#9467bd",
            "VisConf-RWC": "#d62728",
            "VisionOnly-RWC": "#2ca02c"
        }
        for m, color in v_colors.items():
            if m in vista_data["method_results"]:
                accs = [vista_data["method_results"][m][str(N)]["accuracy_percent"] for N in budgets]
                ax.plot(budgets, accs, label=m, color=color, marker="o", linewidth=1.8)

        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.set_xlabel("Candidate Budget (N)", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title("MathVista: Accuracy vs. Candidate Budget (N)", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=9)

        # Subplot (1, 1): MathVista Accuracy vs. Generated Token Cost
        ax = axes[1, 1]
        for m, color in v_colors.items():
            if m in vista_data["method_results"]:
                costs = [vista_data["method_results"][m][str(N)]["mean_token_cost"] for N in budgets]
                accs = [vista_data["method_results"][m][str(N)]["accuracy_percent"] for N in budgets]
                ax.plot(costs, accs, label=m, color=color, marker="o", linewidth=1.8)

        ax.set_xlabel("Mean Generated Token Cost", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title("MathVista: Accuracy vs. Generated Token Cost", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=9)

        plt.tight_layout()
        scaling_path = out_dir / "scaling_curves.png"
        plt.savefig(scaling_path)
        plt.close()
        print(f"Saved scaling curves to {scaling_path}")

    # 2. Failure Decomposition Plot
    if "budgets" in math_data and "Self-Consistency" in math_data["method_results"]:
        budgets = [int(b) for b in math_data["budgets"]]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

        # MATH Decomposition for Self-Consistency
        m_sc = math_data["method_results"]["Self-Consistency"]
        samp_fail = [m_sc[str(N)]["sampling_failure_percent"] for N in budgets]
        sel_fail = [m_sc[str(N)]["selection_failure_percent"] for N in budgets]
        correct = [m_sc[str(N)]["accuracy_percent"] for N in budgets]

        x = np.arange(len(budgets))
        w = 0.55
        ax1.bar(x, correct, w, label="Correct", color="#2ca02c")
        ax1.bar(x, sel_fail, w, bottom=correct, label="Selection Failure (Correct missed)", color="#ff7f0e")
        ax1.bar(x, samp_fail, w, bottom=np.array(correct) + np.array(sel_fail), label="Sampling Failure (No correct)", color="#d62728")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"N={b}" for b in budgets])
        ax1.set_ylabel("Share of Questions (%)", fontsize=11)
        ax1.set_title("MATH: Sampling vs. Selection Failures (Self-Consistency)", fontsize=11, fontweight="bold")
        ax1.set_ylim(0, 100)
        ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax1.legend(loc="upper right", fontsize=9)

        # MathVista Decomposition for Self-Consistency
        if "Self-Consistency" in vista_data["method_results"]:
            v_sc = vista_data["method_results"]["Self-Consistency"]
            v_samp = [v_sc[str(N)]["sampling_failure_percent"] for N in budgets]
            v_sel = [v_sc[str(N)]["selection_failure_percent"] for N in budgets]
            v_corr = [v_sc[str(N)]["accuracy_percent"] for N in budgets]

            ax2.bar(x, v_corr, w, label="Correct", color="#2ca02c")
            ax2.bar(x, v_sel, w, bottom=v_corr, label="Selection Failure (Correct missed)", color="#ff7f0e")
            ax2.bar(x, v_samp, w, bottom=np.array(v_corr) + np.array(v_sel), label="Sampling Failure (No correct)", color="#d62728")
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"N={b}" for b in budgets])
            ax2.set_ylabel("Share of Questions (%)", fontsize=11)
            ax2.set_title("MathVista: Sampling vs. Selection Failures (Self-Consistency)", fontsize=11, fontweight="bold")
            ax2.set_ylim(0, 100)
            ax2.grid(True, linestyle=":", alpha=0.6, axis="y")
            ax2.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        fail_path = out_dir / "failure_decomposition.png"
        plt.savefig(fail_path)
        plt.close()
        print(f"Saved failure decomposition plot to {fail_path}")

    # 3. Master Markdown Synthesis
    md_lines = ["# Task 3: Test-Time Compute & Adaptive Inference Summary\n\n"]

    # Table 1: MATH Method Comparisons across Budgets
    if "budgets" in math_data:
        md_lines.append("## 1. MATH: Offline Test-Time Compute Scaling\n\n")
        rows = []
        for m in math_data["methods"]:
            row = {"Method": m}
            for N in math_data["budgets"]:
                res = math_data["method_results"][m][str(N)]
                row[f"N={N} Acc (%)"] = f"{res['accuracy_percent']:.2f}%"
            rows.append(row)
        # Add Oracle
        o_row = {"Method": "**Oracle Coverage**"}
        for N in math_data["budgets"]:
            o_res = math_data["oracle_coverage"][str(N)]
            o_row[f"N={N} Acc (%)"] = f"**{o_res['mean_percent']:.2f}%**"
        rows.append(o_row)
        df_math = pd.DataFrame(rows)
        md_lines.append(df_math.to_markdown(index=False) + "\n\n")

    # Table 2: MathVista Method Comparisons across Budgets
    if "budgets" in vista_data:
        md_lines.append("## 2. MathVista: Quality-Aware Selection & Consensus\n\n")
        rows = []
        for m in vista_data["methods"]:
            row = {"Method": m}
            for N in vista_data["budgets"]:
                res = vista_data["method_results"][m][str(N)]
                row[f"N={N} Acc (%)"] = f"{res['accuracy_percent']:.2f}%"
            rows.append(row)
        # Add Oracle
        o_row = {"Method": "**Oracle Coverage**"}
        for N in vista_data["budgets"]:
            o_res = vista_data["oracle_coverage"][str(N)]
            o_row[f"N={N} Acc (%)"] = f"**{o_res['mean_percent']:.2f}%**"
        rows.append(o_row)
        df_vista = pd.DataFrame(rows)
        md_lines.append(df_vista.to_markdown(index=False) + "\n\n")

    # Section 3: Signal Analysis (Step 3)
    md_lines.append("## 3. Signal Analysis & Trajectory vs. Tokenwise Fusion\n\n")
    if step3_math:
        md_lines.append(f"- **MATH Shortest-Rollout Accuracy**: {step3_math.get('shortest_rollout_accuracy_percent', 0.0):.2f}%\n")
        md_lines.append(f"- **MATH Top vs. Bottom Quartile Accuracy Diff**:\n")
        for k, v in step3_math.get("top_minus_bottom_quartile_acc_diff", {}).items():
            md_lines.append(f"  - `{k}`: {v:+.2f} pp\n")
    if step3_vista:
        md_lines.append(f"- **MathVista Trajectory VisConf RWC Accuracy**: {step3_vista.get('trajectory_visconf_rwc_acc', 0.0):.2f}%\n")
        md_lines.append(f"- **MathVista Tokenwise Fusion RWC Accuracy**: {step3_vista.get('tokenwise_fusion_rwc_acc', 0.0):.2f}%\n")
        md_lines.append(f"- **Confidence-Vision Token Covariance $\\text{{Cov}}_t(c_{{i,t}}, v_{{i,t}})$**: {step3_vista.get('mean_token_covariance_cov_c_v', 0.0):.6f}\n\n")

    # Section 4: Live Generation & Sequential Refinement
    if "strategy_comparisons" in live_data:
        md_lines.append("## 4. Live Inference: Parallel Sampling vs. Sequential Refinement (50 Questions)\n\n")
        strat_rows = []
        for strat, data in live_data["strategy_comparisons"].items():
            strat_rows.append({
                "Strategy": strat.replace("_", " ").title(),
                "Accuracy (%)": f"{data['accuracy_percent']:.2f}%",
                "Input Tokens": f"{data['mean_input_tokens']:.1f}",
                "Output Tokens": f"{data['mean_output_tokens']:.1f}",
                "Latency (s)": f"{data['avg_runtime_sec']:.2f}s",
                "Truncation (%)": f"{data['truncation_rate_percent']:.1f}%",
                "Extraction Fail (%)": f"{data['extraction_failure_percent']:.1f}%"
            })
        df_live = pd.DataFrame(strat_rows)
        md_lines.append(df_live.to_markdown(index=False) + "\n\n")

        if "refinement_transitions" in live_data:
            trans = live_data["refinement_transitions"]
            md_lines.append(
                f"**Sequential Refinement Answer Transitions:**\n"
                f"- Wrong $\\to$ Correct: **+{trans['wrong_to_correct']}**\n"
                f"- Correct $\\to$ Wrong: **-{trans['correct_to_wrong']}**\n"
                f"- Net Gain: **{trans['net_gain']:+d}**\n\n"
            )

    # Section 5: Adaptive Sampling Evaluation
    md_lines.append("## 5. Adaptive Inference Stopping Evaluation (800 Held-Out Questions)\n\n")
    if adapt_math:
        md_lines.append("### MATH Adaptive Controller Results\n\n")
        rows = []
        for pol, data in adapt_math.items():
            rows.append({
                "Policy": pol,
                "Accuracy (%)": f"{data['accuracy_percent']:.2f}%",
                "Mean Token Cost": f"{data['mean_token_cost']:.1f}",
                "Mean Rollouts": f"{data['mean_rollouts']:.2f}",
                "Stop @ 4 (%)": f"{data['stopping_distribution_percent'].get('4', data['stopping_distribution_percent'].get(4, 0.0)):.1f}%",
                "Stop @ 8 (%)": f"{data['stopping_distribution_percent'].get('8', data['stopping_distribution_percent'].get(8, 0.0)):.1f}%",
                "Stop @ 16 (%)": f"{data['stopping_distribution_percent'].get('16', data['stopping_distribution_percent'].get(16, 0.0)):.1f}%",
                "Stop @ 32 (%)": f"{data['stopping_distribution_percent'].get('32', data['stopping_distribution_percent'].get(32, 0.0)):.1f}%",
                "Premature Stop (%)": f"{data['premature_stopping_rate_percent']:.2f}%"
            })
        df_ad_m = pd.DataFrame(rows)
        md_lines.append(df_ad_m.to_markdown(index=False) + "\n\n")

    if adapt_vista:
        md_lines.append("### MathVista Adaptive Controller Results\n\n")
        rows = []
        for pol, data in adapt_vista.items():
            rows.append({
                "Policy": pol,
                "Accuracy (%)": f"{data['accuracy_percent']:.2f}%",
                "Mean Token Cost": f"{data['mean_token_cost']:.1f}",
                "Mean Rollouts": f"{data['mean_rollouts']:.2f}",
                "Stop @ 4 (%)": f"{data['stopping_distribution_percent'].get('4', data['stopping_distribution_percent'].get(4, 0.0)):.1f}%",
                "Stop @ 8 (%)": f"{data['stopping_distribution_percent'].get('8', data['stopping_distribution_percent'].get(8, 0.0)):.1f}%",
                "Stop @ 16 (%)": f"{data['stopping_distribution_percent'].get('16', data['stopping_distribution_percent'].get(16, 0.0)):.1f}%",
                "Stop @ 32 (%)": f"{data['stopping_distribution_percent'].get('32', data['stopping_distribution_percent'].get(32, 0.0)):.1f}%",
                "Premature Stop (%)": f"{data['premature_stopping_rate_percent']:.2f}%"
            })
        df_ad_v = pd.DataFrame(rows)
        md_lines.append(df_ad_v.to_markdown(index=False) + "\n\n")

    summary_path = out_dir / "task3_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("".join(md_lines))
    print(f"Generated master Task 3 report at {summary_path}")
