# CS-6304 / EE-5102: Advanced Topics in Machine Learning – Fall 2026
## Programming Assignment 3: LLM Fine-Tuning, VLMs & Visual Grounding, and Test-Time Compute

**Student:** Muhammad Ahmad Sarfraz  
**Instructor:** Dr. Muhammad Tahir  
**Institution:** Lahore University of Management Sciences (LUMS)  
**Repository:** https://github.com/ahmad-sarfraz345/ATML-PA3.git  

---

## Overview

This repository contains the complete implementation, configuration, and evaluation pipelines for Programming Assignment 3:
- **Task 1: Large Language Model Fine-Tuning**  
  Investigates adaptation vs. retention (catastrophic forgetting) across 5 fine-tuning variants on `Qwen/Qwen3-0.6B`:
  1. Full Fine-Tuning (Full FT, $\lambda=0$)
  2. Full FT + L2-SP ($\lambda=10^{-3}$, un-averaged Euclidean proximity penalty to starting checkpoint $\theta_0$)
  3. LoRA Rank 4 ($r=4, \alpha=8$)
  4. LoRA Rank 16 ($r=16, \alpha=32$)
  5. Orthogonal Projection LoRA (OPLoRA, $r=16, \alpha=32, k=128$, with frozen singular bases $U_k, V_k$)
  Zero-shot evaluation across 7 conditions on GSM8K (`gsm8k_cot`), HellaSwag, BoolQ, ARC-Easy, weight-space diagnostics ($\rho_k, \eta$), activation diagnostics ($R(z)$), and qualitative evaluation (24 prompts from `configs/first_plot_questions.yaml`).
- **Task 2: Vision-Language Models and Visual Grounding**  
  Connects frozen `google/siglip-base-patch16-224` to `Qwen/Qwen3-0.6B` using a 2-layer MLP connector + LoRA across 3 curriculum stages: Caption alignment $\to$ Visual instruction tuning $\to$ Natural VQA specialization.
- **Task 3: Test-Time Compute: Sampling, Selection, and Adaptive Inference**  
  Offline rollout quality scoring (self-certainty, visual engagement, VisConf, RWC), live sequential refinement vs. parallel sampling, and agreement/quality-aware adaptive inference controllers.

---

## Directory Structure

```
ATML-PA3/
|-- common/
|   |-- __init__.py
|   |-- data_utils.py         # Text normalization, contamination filtering, prompt formats
|   `-- utils.py              # Seeding (6304), device management, memory tracker, exporters
|-- configs/
|   `-- first_plot_questions.yaml # 24 prompts across 8 questions in plain, JSON, template formats
|-- task1/
|   |-- __init__.py
|   |-- data.py               # MetaMathQA 40k pipeline + 64-item activation probe sampler
|   |-- models.py             # Custom LoRALinear and OPLoRALinear modules with SVD basis caching
|   |-- trainer.py            # Unified trainer with AdamW, FP16 AMP, L2-SP penalty, and checkpoints
|   |-- eval.py               # Adapter merging and lm-eval Python API zero-shot benchmark runner
|   |-- diagnostics.py        # Weight-space (rho_k, eta) and activation diagnostics (R(z))
|   |-- qualitative.py        # 168 qualitative generations and 42-response annotation subset
|   `-- analysis.py           # Master tables, learning curve plots, and resource benchmarks
|-- scripts/
|   `-- run_task1.py          # Modular CLI execution runner for Task 1
|-- run_task1_kaggle.ipynb    # End-to-end Kaggle execution notebook for Task 1
|-- requirements.txt          # Python dependencies
|-- .gitignore
`-- README.md
```

---

## Task 1: Reproduction Instructions

### 1. Local / Cloud Environment Setup
```bash
git clone https://github.com/ahmad-sarfraz345/ATML-PA3.git
cd ATML-PA3
pip install -r requirements.txt
```

### 2. Kaggle Execution
Run the dedicated Kaggle notebook `run_task1_kaggle.ipynb`:
1. Upload and launch `run_task1_kaggle.ipynb` on Kaggle with a GPU accelerator (T4x2, P100, or L4) and **Internet: On**.
2. Select your desired stages using the boolean flags:
   ```python
   STAGE_PREPARE_DATA       = True
   STAGE_EVAL_BASELINE      = True
   STAGE_TRAIN_FULL_FT      = True
   STAGE_TRAIN_FULL_FT_L2SP = True
   STAGE_TRAIN_LORA_4       = True
   STAGE_TRAIN_LORA_16      = True
   STAGE_TRAIN_OPLORA_16    = True
   STAGE_EVAL_BENCHMARKS    = True
   STAGE_DIAGNOSTICS        = True
   STAGE_QUALITATIVE        = True
   STAGE_ANALYSIS           = True
   ```
3. Run the notebook to completion. Checkpoints and outputs are generated in `outputs/task1/`.
4. The final cell bundles all generated artifacts into `/kaggle/working/task1_outputs.zip`.

### 3. Modular CLI Execution
Each stage can also be run independently via the command line:
```bash
# Prepare dataset and activation probes
python -m scripts.run_task1 --stage prepare_data

# Evaluate shared baseline
python -m scripts.run_task1 --stage eval_baseline

# Train individual variants
python -m scripts.run_task1 --stage train_full_ft
python -m scripts.run_task1 --stage train_full_ft_l2sp
python -m scripts.run_task1 --stage train_lora_4
python -m scripts.run_task1 --stage train_lora_16
python -m scripts.run_task1 --stage train_oplora_16

# Benchmark evaluations
python -m scripts.run_task1 --stage eval_benchmarks

# Diagnostics & qualitative responses
python -m scripts.run_task1 --stage diagnostics
python -m scripts.run_task1 --stage qualitative

# Aggregate tables and plots
python -m scripts.run_task1 --stage analysis
```

