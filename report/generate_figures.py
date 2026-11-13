import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Publication styling calibrated for NeurIPS 10pt layout
plt.rcParams.update({
    "font.size": 9.5,
    "axes.labelsize": 10.0,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 8.5,
    "figure.titlesize": 11.0,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "figure.autolayout": True
})

OUTPUT_DIR = Path("report/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ==============================================================================
# Figure 1: Task 1 SFT Training Loss Curves
# ==============================================================================
def plot_figure_1():
    print("Generating Figure 1: Task 1 Learning Curves...")
    variants = {
        "full_ft": ("Full FT (lambda=0)", "#1f77b4", "-"),
        "full_ft_l2sp": ("Full FT + L2-SP", "#ff7f0e", "-"),
        "lora_4": ("LoRA-4 (r=4)", "#2ca02c", "--"),
        "lora_16": ("LoRA-16 (r=16)", "#d62728", "-"),
        "oplora_16": ("OPLoRA-16 (k=128)", "#9467bd", "-.")
    }
    
    fig, ax = plt.subplots(figsize=(6.5, 2.4), dpi=300)
    
    for var, (label, color, ls) in variants.items():
        log_file = Path(f"outputs/task1/{var}/training_logs.json")
        if log_file.exists():
            data = load_json(log_file)
            steps = [d["step"] for d in data]
            losses = [d["sft_loss"] for d in data]
            window = 15
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            smooth_steps = steps[window-1:]
            ax.plot(smooth_steps, smoothed, label=label, color=color, linestyle=ls, linewidth=1.6)
            ax.plot(steps, losses, color=color, alpha=0.08, linewidth=0.5)

    for ckpt in [312, 625, 937, 1250]:
        ax.axvline(x=ckpt, color="gray", linestyle=":", alpha=0.55, linewidth=1.0)

    ax.text(312 + 10, 0.95, "Checkpoint 1 (312)", color="dimgray", fontsize=7.5)
    ax.text(625 + 10, 0.95, "Checkpoint 2 (625)", color="dimgray", fontsize=7.5)

    ax.set_xlabel("Optimizer Updates (Batch Size 32, 40,000 Total Examples)", fontsize=9.0)
    ax.set_ylabel("SFT Cross-Entropy Loss", fontsize=9.0)
    ax.set_title("MetaMathQA SFT Training Trajectories Across Adaptation Regimes", fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8.0)
    ax.set_ylim(0.12, 1.08)
    ax.set_xlim(0, 1270)
    
    fig.savefig(OUTPUT_DIR / "fig1_task1_learning_curves.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig1_task1_learning_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 1.")

# ==============================================================================
# Figure 2: Task 1 Weight-Space and Activation Diagnostics
# ==============================================================================
def plot_figure_2():
    print("Generating Figure 2: Task 1 Diagnostics...")
    weight_file = Path("outputs/task1/weight_space_diagnostics.json")
    act_file = Path("outputs/task1/activation_diagnostics.json")
    
    if not weight_file.exists() or not act_file.exists():
        print("Diagnostic files missing; skipping Figure 2.")
        return
        
    w_data = load_json(weight_file)
    a_data = load_json(act_file)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 2.6), dpi=300)
    
    # Subplot (a): Subspace Interference rho_k across layers
    models = ["full_ft_final", "full_ft_l2sp_final", "lora_4_final", "lora_16_final", "oplora_16_final"]
    model_labels = ["Full FT", "L2-SP", "LoRA-4", "LoRA-16", "OPLoRA-16"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    layers = [0, 13, 27]
    x = np.arange(len(layers))
    width = 0.15
    
    for idx, (m, lbl, col) in enumerate(zip(models, model_labels, colors)):
        rho_vals = []
        for l in layers:
            val = w_data[m][f"layer_{l}"]["subspace_interference_rho_k"]
            rho_vals.append(val)
        ax1.bar(x + idx * width - width * 2, rho_vals, width, label=lbl, color=col, edgecolor="black", linewidth=0.5)
        
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Layer {l}" for l in layers])
    ax1.set_ylabel(r"Subspace Interference $\rho_k$")
    ax1.set_title(r"(a) Output Subspace Overlap $\rho_k$ on $W_{\mathrm{down}}$")
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax1.legend(loc="upper left", fontsize=8.0, framealpha=0.95)
    ax1.set_ylim(0.0, 0.42)
    
    # Subplot (b): Activation Correction Ratio R(z) on Math vs Non-Math Prompts (Layer 13)
    r_math = []
    r_nonmath = []
    for m in models:
        r_math.append(a_data[m]["layer_13"]["math_rz_mean"])
        r_nonmath.append(a_data[m]["layer_13"]["nonmath_rz_mean"])
        
    x2 = np.arange(len(models))
    w2 = 0.35
    ax2.bar(x2 - w2/2, r_math, w2, label="Math Prompts", color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax2.bar(x2 + w2/2, r_nonmath, w2, label="Non-Math Prompts", color="#aec7e8", edgecolor="black", linewidth=0.5)
    
    ax2.set_xticks(x2)
    ax2.set_xticklabels(model_labels, rotation=12)
    ax2.set_ylabel(r"Activation Correction $R(z)$")
    ax2.set_title(r"(b) Layer 13 Activation Impact $R(z)$")
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax2.legend(loc="upper right", fontsize=8.2, framealpha=0.95)
    
    fig.savefig(OUTPUT_DIR / "fig2_task1_diagnostics.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig2_task1_diagnostics.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 2.")

# ==============================================================================
# Figure 3: Task 2 VLM Curriculum Learning Curves & Image-Swap Penalty
# ==============================================================================
def plot_figure_3():
    print("Generating Figure 3: Task 2 Curriculum Learning Curves...")
    train_file = Path("outputs/task2/train_summary.json")
    if not train_file.exists():
        print("Task 2 train summary missing; skipping Figure 3.")
        return
        
    data = load_json(train_file)
    t_logs = data["training_logs"]
    d_logs = data["dev_eval_logs"]
    
    steps_t = [x["step"] for x in t_logs]
    loss_t = [x["loss"] for x in t_logs]
    
    window = 25
    smoothed_loss = np.convolve(loss_t, np.ones(window)/window, mode='valid')
    smooth_steps = steps_t[window-1:]
    
    steps_d = [x["step"] for x in d_logs]
    dev_caption_loss = [x["dev_caption_loss"] for x in d_logs]
    ground_delta_nll = [x["grounding_dev"]["delta_nll_swap_penalty"] for x in d_logs]
    vqa_delta_nll = [x["vqa_dev"]["delta_nll_swap_penalty"] for x in d_logs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 2.6), dpi=300)
    
    # Subplot 1: SFT Loss & Dev Caption Loss
    ax1.plot(smooth_steps, smoothed_loss, label="Training SFT Loss", color="#1f77b4", linewidth=1.8)
    ax1.plot(steps_t, loss_t, color="#1f77b4", alpha=0.15, linewidth=0.7)
    ax1.plot(steps_d, dev_caption_loss, label="Dev Caption NLL", color="#d62728", marker="o", markersize=4, linewidth=1.8)
    
    ax1.axvline(x=2250, color="gray", linestyle="--", linewidth=1.2)
    ax1.axvline(x=3750, color="black", linestyle="--", linewidth=1.2)
    ax1.text(600, 7.8, "Stage 1", fontsize=8.5, fontweight="bold", color="dimgray")
    ax1.text(2500, 7.8, "Stage 2", fontsize=8.5, fontweight="bold", color="dimgray")
    ax1.text(3850, 7.8, "Stage 3", fontsize=8.5, fontweight="bold", color="dimgray")
    
    ax1.set_xlabel("Optimizer Updates")
    ax1.set_ylabel("Loss / NLL")
    ax1.set_title("(a) Curriculum SFT Loss & Dev NLL")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", framealpha=0.95, fontsize=8.0, facecolor="white", edgecolor="lightgray")
    ax1.set_ylim(0.0, 9.5)
    
    # Subplot 2: Development Image-Swap Penalty Delta NLL
    ax2.plot(steps_d, ground_delta_nll, label=r"Grounding Swap $\Delta\mathrm{NLL}$", color="#2ca02c", marker="s", markersize=4, linewidth=1.8)
    ax2.plot(steps_d, vqa_delta_nll, label=r"Natural VQAv2 Swap $\Delta\mathrm{NLL}$", color="#ff7f0e", marker="^", markersize=4, linewidth=1.8)
    
    ax2.axvline(x=2250, color="gray", linestyle="--", linewidth=1.2)
    ax2.axvline(x=3750, color="black", linestyle="--", linewidth=1.2)
    ax2.axhline(y=0.0, color="black", linestyle=":", alpha=0.7)
    
    ax2.set_xlabel("Optimizer Updates")
    ax2.set_ylabel(r"Swap Penalty $\Delta\mathrm{NLL}$")
    ax2.set_title(r"(b) Dev Image-Swap Penalty $\Delta\mathrm{NLL}$")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", framealpha=0.95, fontsize=8.0, facecolor="white", edgecolor="lightgray")
    ax2.set_ylim(-0.2, 0.55)
    
    fig.savefig(OUTPUT_DIR / "fig3_task2_curriculum_curves.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig3_task2_curriculum_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 3.")

# ==============================================================================
# Figure 4: Task 2 Visual Grounding Final Test Evaluation
# ==============================================================================
def plot_figure_4():
    print("Generating Figure 4: Task 2 Final Grounding Evaluations...")
    eval_file = Path("outputs/task2/final_test_results.json")
    if not eval_file.exists():
        print("Final test results missing; skipping Figure 4.")
        return
        
    data = load_json(eval_file)
    conditions = ["correct", "complementary_swap", "mismatched", "no_image"]
    cond_labels = ["Correct\nImage", "Compl.\nSwap", "Mismatch\nPair", "No Image\n(Blind)"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 2.6), dpi=300)
    
    # Subplot 1: Accuracies with 95% Bootstrap CIs
    g_acc = [data["grounding"][c]["accuracy_percent"] for c in conditions]
    g_ci_low = [data["grounding"][c]["accuracy_percent"] - data["grounding"][c]["accuracy_95_ci"][0] for c in conditions]
    g_ci_high = [data["grounding"][c]["accuracy_95_ci"][1] - data["grounding"][c]["accuracy_percent"] for c in conditions]
    
    v_acc = [data["vqa"][c]["accuracy_percent"] for c in conditions]
    v_ci_low = [data["vqa"][c]["accuracy_percent"] - data["vqa"][c]["accuracy_95_ci"][0] for c in conditions]
    v_ci_high = [data["vqa"][c]["accuracy_95_ci"][1] - data["vqa"][c]["accuracy_percent"] for c in conditions]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    ax1.bar(x - width/2, g_acc, width, yerr=[g_ci_low, g_ci_high], capsize=3, label="Grounding Test", color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax1.bar(x + width/2, v_acc, width, yerr=[v_ci_low, v_ci_high], capsize=3, label="Natural VQAv2", color="#ff7f0e", edgecolor="black", linewidth=0.5)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(cond_labels)
    ax1.set_ylabel("Evaluation Accuracy (%)")
    ax1.set_title("(a) Accuracy Under Visual Controls")
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax1.legend(loc="lower left", framealpha=0.95, fontsize=8.0, facecolor="white", edgecolor="lightgray")
    ax1.set_ylim(0, 80)
    
    # Subplot 2: Reference Canonical NLL with 95% Bootstrap CIs
    g_nll = [data["grounding"][c]["mean_reference_nll"] for c in conditions]
    g_nll_low = [data["grounding"][c]["mean_reference_nll"] - data["grounding"][c]["reference_nll_95_ci"][0] for c in conditions]
    g_nll_high = [data["grounding"][c]["reference_nll_95_ci"][1] - data["grounding"][c]["mean_reference_nll"] for c in conditions]
    
    v_nll = [data["vqa"][c]["mean_reference_nll"] for c in conditions]
    v_nll_low = [data["vqa"][c]["mean_reference_nll"] - data["vqa"][c]["reference_nll_95_ci"][0] for c in conditions]
    v_nll_high = [data["vqa"][c]["reference_nll_95_ci"][1] - data["vqa"][c]["mean_reference_nll"] for c in conditions]
    
    ax2.bar(x - width/2, g_nll, width, yerr=[g_nll_low, g_nll_high], capsize=3, label="Grounding Test", color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax2.bar(x + width/2, v_nll, width, yerr=[v_nll_low, v_nll_high], capsize=3, label="Natural VQAv2", color="#ff7f0e", edgecolor="black", linewidth=0.5)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(cond_labels)
    ax2.set_ylabel("Target Reference NLL")
    ax2.set_title("(b) Target NLL Across Visual Controls")
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax2.legend(loc="lower right", framealpha=0.95, fontsize=8.0, facecolor="white", edgecolor="lightgray")
    ax2.set_ylim(0, 5.5)
    
    fig.savefig(OUTPUT_DIR / "fig4_task2_grounding_eval.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig4_task2_grounding_eval.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 4.")

# ==============================================================================
# Figure 5: Task 3 Test-Time Scaling Curves (Accuracy vs N and Token Cost)
# ==============================================================================
def plot_figure_5():
    print("Generating Figure 5: Task 3 Test-Time Scaling Curves...")
    prev_autolayout = plt.rcParams.get("figure.autolayout", False)
    plt.rcParams["figure.autolayout"] = False
    math_file = Path("outputs/task3/offline_math_results.json")
    vista_file = Path("outputs/task3/offline_mathvista_results.json")
    
    if not math_file.exists() or not vista_file.exists():
        print("Task 3 offline files missing; skipping Figure 5.")
        return
        
    m_data = load_json(math_file)
    v_data = load_json(vista_file)
    budgets = [1, 2, 4, 8, 16, 32]
    
    fig = plt.figure(figsize=(7.6, 3.6), dpi=300)
    gs = gridspec.GridSpec(3, 2, height_ratios=[0.45, 1.0, 1.0], hspace=0.48, wspace=0.26)
    
    # Dedicated axis for the unified top legend
    ax_leg = fig.add_subplot(gs[0, :])
    ax_leg.axis('off')
    
    # Subplot (0, 0): MATH Accuracy vs N
    ax_a = fig.add_subplot(gs[1, 0])
    l_oracle, = ax_a.plot(budgets, [m_data["oracle_coverage"][str(N)]["mean_percent"] for N in budgets], color="black", linestyle="--", marker="x", markersize=3.5, linewidth=1.3, label="Oracle Coverage")
    l_rwc, = ax_a.plot(budgets, [m_data["method_results"]["SelfCertainty-RWC"][str(N)]["accuracy_percent"] for N in budgets], color="#d62728", marker="s", markersize=3.5, linewidth=1.4, label="Quality-RWC (Ours)")
    l_sc, = ax_a.plot(budgets, [m_data["method_results"]["Self-Consistency"][str(N)]["accuracy_percent"] for N in budgets], color="#1f77b4", marker="o", markersize=3.5, linewidth=1.3, label="Self-Consistency")
    l_high, = ax_a.plot(budgets, [m_data["method_results"]["Highest-SelfCertainty"][str(N)]["accuracy_percent"] for N in budgets], color="#ff7f0e", marker="^", markersize=3.5, linewidth=1.2, label="Highest-Quality")
    l_lp, = ax_a.plot(budgets, [m_data["method_results"]["Highest-LogProb"][str(N)]["accuracy_percent"] for N in budgets], color="#2ca02c", marker="v", markersize=3.5, linewidth=1.2, label="Highest-LogProb")
    l_base, = ax_a.plot(budgets, [m_data["method_results"]["FirstAns"][str(N)]["accuracy_percent"] for N in budgets], color="gray", linestyle=":", linewidth=1.2, label="FirstAns Baseline")
    
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(budgets)
    ax_a.set_xticklabels([str(b) for b in budgets], fontsize=8.0)
    ax_a.set_ylabel("MATH Acc (%)", fontsize=8.5)
    ax_a.set_title(r"(a) MATH: Accuracy vs. Budget ($N$)", fontsize=9.0)
    ax_a.grid(True, linestyle=":", alpha=0.5)
    ax_a.set_ylim(58, 92)
    
    # Subplot (0, 1): MATH Accuracy vs Token Cost
    ax_b = fig.add_subplot(gs[1, 1])
    for m, col, mk in [("SelfCertainty-RWC", "#d62728", "s"), ("Self-Consistency", "#1f77b4", "o"), ("Highest-LogProb", "#2ca02c", "v")]:
        costs = [m_data["method_results"][m][str(N)]["mean_token_cost"] for N in budgets]
        accs = [m_data["method_results"][m][str(N)]["accuracy_percent"] for N in budgets]
        ax_b.plot(costs, accs, color=col, marker=mk, markersize=3.5, linewidth=1.3)
    ax_b.set_ylabel("MATH Acc (%)", fontsize=8.5)
    ax_b.set_xlabel("Mean Generated Output Tokens", fontsize=8.5)
    ax_b.set_title("(b) MATH: Accuracy vs. Token Cost", fontsize=9.0)
    ax_b.grid(True, linestyle=":", alpha=0.5)
    ax_b.set_ylim(58, 75)
    
    # Subplot (1, 0): MathVista Accuracy vs N
    ax_c = fig.add_subplot(gs[2, 0])
    ax_c.plot(budgets, [v_data["oracle_coverage"][str(N)]["mean_percent"] for N in budgets], color="black", linestyle="--", marker="x", markersize=3.5, linewidth=1.3)
    ax_c.plot(budgets, [v_data["method_results"]["VisConf-RWC"][str(N)]["accuracy_percent"] for N in budgets], color="#d62728", marker="s", markersize=3.5, linewidth=1.4)
    ax_c.plot(budgets, [v_data["method_results"]["Self-Consistency"][str(N)]["accuracy_percent"] for N in budgets], color="#1f77b4", marker="o", markersize=3.5, linewidth=1.3)
    ax_c.plot(budgets, [v_data["method_results"]["VisConf-Highest"][str(N)]["accuracy_percent"] for N in budgets], color="#ff7f0e", marker="^", markersize=3.5, linewidth=1.2)
    ax_c.plot(budgets, [v_data["method_results"]["Highest-LogProb"][str(N)]["accuracy_percent"] for N in budgets], color="#2ca02c", marker="v", markersize=3.5, linewidth=1.2)
    ax_c.plot(budgets, [v_data["method_results"]["FirstAns"][str(N)]["accuracy_percent"] for N in budgets], color="gray", linestyle=":", linewidth=1.2)
    
    ax_c.set_xscale("log", base=2)
    ax_c.set_xticks(budgets)
    ax_c.set_xticklabels([str(b) for b in budgets], fontsize=8.0)
    ax_c.set_xlabel(r"Candidate Budget ($N$)", fontsize=8.5)
    ax_c.set_ylabel("MathVista Acc (%)", fontsize=8.5)
    ax_c.set_title(r"(c) MathVista: Accuracy vs. Budget ($N$)", fontsize=9.0)
    ax_c.grid(True, linestyle=":", alpha=0.5)
    ax_c.set_ylim(46, 92)
    
    # Subplot (1, 1): MathVista Accuracy vs Token Cost
    ax_d = fig.add_subplot(gs[2, 1])
    for m, col, mk in [("VisConf-RWC", "#d62728", "s"), ("Self-Consistency", "#1f77b4", "o"), ("Highest-LogProb", "#2ca02c", "v")]:
        costs = [v_data["method_results"][m][str(N)]["mean_token_cost"] for N in budgets]
        accs = [v_data["method_results"][m][str(N)]["accuracy_percent"] for N in budgets]
        ax_d.plot(costs, accs, color=col, marker=mk, markersize=3.5, linewidth=1.3)
    ax_d.set_ylabel("MathVista Acc (%)", fontsize=8.5)
    ax_d.set_xlabel("Mean Generated Output Tokens", fontsize=8.5)
    ax_d.set_title("(d) MathVista: Accuracy vs. Token Cost", fontsize=9.0)
    ax_d.grid(True, linestyle=":", alpha=0.5)
    ax_d.set_ylim(46, 67)
    
    # Unified Legend in ax_leg
    handles = [l_oracle, l_rwc, l_sc, l_high, l_lp, l_base]
    labels = ["Oracle Coverage", "Quality-RWC (Ours)", "Self-Consistency", "Highest-Quality", "Highest-LogProb", "FirstAns Baseline"]
    ax_leg.legend(handles, labels, loc="center", ncol=3, frameon=True, facecolor="white", edgecolor="lightgray", framealpha=0.95, fontsize=7.8)
    
    fig.savefig(OUTPUT_DIR / "fig5_task3_scaling_curves.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig5_task3_scaling_curves.png", bbox_inches="tight")
    plt.close(fig)
    plt.rcParams["figure.autolayout"] = prev_autolayout
    print("Saved Figure 5.")

# ==============================================================================
# Figure 6: Task 3 Failure Decomposition (Sampling vs Selection)
# ==============================================================================
def plot_figure_6():
    print("Generating Figure 6: Task 3 Failure Decomposition...")
    math_file = Path("outputs/task3/offline_math_results.json")
    vista_file = Path("outputs/task3/offline_mathvista_results.json")
    
    if not math_file.exists() or not vista_file.exists():
        print("Task 3 offline files missing; skipping Figure 6.")
        return
        
    m_data = load_json(math_file)
    v_data = load_json(vista_file)
    budgets = [1, 2, 4, 8, 16, 32]
    x = np.arange(len(budgets))
    w = 0.55
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 2.5), dpi=300)
    
    # MATH
    m_sc = m_data["method_results"]["Self-Consistency"]
    corr_m = [m_sc[str(N)]["accuracy_percent"] for N in budgets]
    sel_fail_m = [m_sc[str(N)]["selection_failure_percent"] for N in budgets]
    samp_fail_m = [m_sc[str(N)]["sampling_failure_percent"] for N in budgets]
    
    ax1.bar(x, corr_m, w, label="Correct Selection", color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax1.bar(x, sel_fail_m, w, bottom=corr_m, label="Selection Failure", color="#ff7f0e", edgecolor="black", linewidth=0.5)
    ax1.bar(x, samp_fail_m, w, bottom=np.array(corr_m) + np.array(sel_fail_m), label="Sampling Failure", color="#d62728", edgecolor="black", linewidth=0.5)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"N={b}" for b in budgets])
    ax1.set_ylabel("Proportion of Questions (%)")
    ax1.set_title("(a) MATH: Failure Decomposition (SC)")
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax1.legend(loc="lower right", fontsize=7.5, framealpha=0.95, facecolor="white", edgecolor="lightgray")
    ax1.set_ylim(0, 100)
    
    # MathVista
    v_sc = v_data["method_results"]["Self-Consistency"]
    corr_v = [v_sc[str(N)]["accuracy_percent"] for N in budgets]
    sel_fail_v = [v_sc[str(N)]["selection_failure_percent"] for N in budgets]
    samp_fail_v = [v_sc[str(N)]["sampling_failure_percent"] for N in budgets]
    
    ax2.bar(x, corr_v, w, label="Correct Selection", color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax2.bar(x, sel_fail_v, w, bottom=corr_v, label="Selection Failure", color="#ff7f0e", edgecolor="black", linewidth=0.5)
    ax2.bar(x, samp_fail_v, w, bottom=np.array(corr_v) + np.array(sel_fail_v), label="Sampling Failure", color="#d62728", edgecolor="black", linewidth=0.5)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"N={b}" for b in budgets])
    ax2.set_ylabel("Proportion of Questions (%)")
    ax2.set_title("(b) MathVista: Failure Decomposition (SC)")
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax2.legend(loc="lower right", fontsize=7.5, framealpha=0.95, facecolor="white", edgecolor="lightgray")
    ax2.set_ylim(0, 100)
    
    fig.savefig(OUTPUT_DIR / "fig6_task3_failure_decomposition.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig6_task3_failure_decomposition.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 6.")

if __name__ == "__main__":
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    plot_figure_6()
    print("All figures successfully regenerated with publication dimensions!")
