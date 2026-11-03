from __future__ import annotations
import argparse
import sys
import os
import torch
from pathlib import Path

from common.utils import set_seed, get_device

def parse_args():
    parser = argparse.ArgumentParser(description="ATML PA3 Task 1 Runner")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=[
            "all",
            "prepare_data",
            "eval_baseline",
            "train_full_ft",
            "train_full_ft_l2sp",
            "train_lora_4",
            "train_lora_16",
            "train_oplora_16",
            "eval_benchmarks",
            "diagnostics",
            "qualitative",
            "analysis"
        ],
        help="Stage to execute."
    )
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, default="outputs/task1")
    parser.add_argument("--cache_data_path", type=str, default="data/metamath_40k_tokenized.pt")
    parser.add_argument("--probe_path", type=str, default="outputs/task1/probe_set.json")
    parser.add_argument("--seed", type=int, default=6304)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"=== ATML PA3 Task 1 Pipeline ===", flush=True)
    print(f"Stage: {args.stage} | Model: {args.model_name_or_path} | Device: {device} | Seed: {args.seed}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Prepare data & probe set
    if args.stage in ["all", "prepare_data"]:
        print("\n>>> Stage: Preparing Data and Activation Probe Set <<<", flush=True)
        from task1.data import prepare_metamath_40k, sample_activation_probe_set
        sample_activation_probe_set(seed=args.seed, save_path=args.probe_path)
        prepare_metamath_40k(
            tokenizer=tokenizer,
            cache_path=args.cache_data_path,
            target_count=40000,
            max_length=2048
        )

    # 2. Evaluate Baseline
    if args.stage in ["all", "eval_baseline"]:
        print("\n>>> Stage: Evaluating Shared Unchanged Baseline <<<", flush=True)
        from task1.eval import evaluate_condition_benchmarks
        evaluate_condition_benchmarks(
            condition="baseline",
            base_model_path=args.model_name_or_path,
            ckpt_dir=args.output_dir,
            output_dir=f"{args.output_dir}/benchmarks",
            batch_size=args.eval_batch_size
        )

    # Helper to train a variant
    def train_variant(variant_name):
        print(f"\n>>> Stage: Training Variant [{variant_name}] <<<", flush=True)
        from task1.data import prepare_metamath_40k, get_dataloader
        from task1.trainer import Task1Trainer

        examples = prepare_metamath_40k(tokenizer=tokenizer, cache_path=args.cache_data_path)
        dataloader = get_dataloader(examples, pad_token_id=tokenizer.pad_token_id, micro_batch_size=2)
        trainer = Task1Trainer(
            variant_name=variant_name,
            model_name_or_path=args.model_name_or_path,
            output_dir=args.output_dir,
            seed=args.seed
        )
        trainer.setup_model_and_tokenizer()
        trainer.train(dataloader)

    if args.stage in ["all", "train_full_ft"]:
        train_variant("full_ft")

    if args.stage in ["all", "train_full_ft_l2sp"]:
        train_variant("full_ft_l2sp")

    if args.stage in ["all", "train_lora_4"]:
        train_variant("lora_4")

    if args.stage in ["all", "train_lora_16"]:
        train_variant("lora_16")

    if args.stage in ["all", "train_oplora_16"]:
        train_variant("oplora_16")

    # 3. Evaluate Checkpoints on Benchmarks
    if args.stage in ["all", "eval_benchmarks"]:
        print("\n>>> Stage: Evaluating Required Checkpoints on Benchmarks <<<", flush=True)
        from task1.eval import evaluate_condition_benchmarks
        eval_conditions = [
            "full_ft_midpoint",
            "full_ft_final",
            "full_ft_l2sp_final",
            "lora_4_final",
            "lora_16_final",
            "oplora_16_final"
        ]
        for cond in eval_conditions:
            evaluate_condition_benchmarks(
                condition=cond,
                base_model_path=args.model_name_or_path,
                ckpt_dir=args.output_dir,
                output_dir=f"{args.output_dir}/benchmarks",
                batch_size=args.eval_batch_size
            )

    # 4. Weight-space and Activation Diagnostics
    if args.stage in ["all", "diagnostics"]:
        print("\n>>> Stage: Running Diagnostics <<<", flush=True)
        from task1.diagnostics import compute_weight_space_diagnostics, compute_activation_diagnostics
        compute_weight_space_diagnostics(
            base_model_path=args.model_name_or_path,
            ckpt_dir=args.output_dir,
            k=128
        )
        compute_activation_diagnostics(
            base_model_path=args.model_name_or_path,
            probe_set_path=args.probe_path,
            ckpt_dir=args.output_dir
        )

    # 5. Qualitative Generations & Annotations
    if args.stage in ["all", "qualitative"]:
        print("\n>>> Stage: Running Qualitative Generations and Annotations <<<", flush=True)
        from task1.qualitative import run_qualitative_generation
        run_qualitative_generation(
            base_model_path=args.model_name_or_path,
            ckpt_dir=args.output_dir,
            output_dir=args.output_dir,
            max_new_tokens=512
        )

    # 6. Analysis, Plots & Summary Tables
    if args.stage in ["all", "analysis"]:
        print("\n>>> Stage: Generating Analysis, Tables and Plots <<<", flush=True)
        from task1.analysis import (
            generate_benchmark_summary_table,
            plot_combined_learning_curves,
            generate_resource_comparison_table
        )
        generate_benchmark_summary_table(
            bench_dir=f"{args.output_dir}/benchmarks",
            output_dir=args.output_dir
        )
        plot_combined_learning_curves(
            task1_dir=args.output_dir,
            output_path=f"{args.output_dir}/learning_curves.png"
        )
        generate_resource_comparison_table(
            task1_dir=args.output_dir,
            output_dir=args.output_dir
        )

    print("\n=== ATML PA3 Task 1 Run Complete ===", flush=True)

if __name__ == "__main__":
    main()
