from __future__ import annotations
import os

# Prevent CUDA memory fragmentation on limited VRAM GPUs (Kaggle T4)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import argparse
import torch
from pathlib import Path

from common.utils import set_seed, get_device, save_json, load_json

def parse_args():
    parser = argparse.ArgumentParser(description="ATML PA3 Task 2 Runner (SigLIP-Qwen VLM & Grounding)")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=[
            "all",
            "prepare_data",
            "extract_features",
            "train_curriculum",
            "eval_final",
            "analysis"
        ],
        help="Stage to execute."
    )
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--siglip_model_id", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--data_dir", type=str, default="data/task2")
    parser.add_argument("--output_dir", type=str, default="outputs/task2")
    parser.add_argument("--seed", type=int, default=6304)
    parser.add_argument("--spatial_pooling", action="store_true", help="Optional 49 visual tokens experiment")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print("=== ATML PA3 Task 2 Pipeline (SigLIP-Qwen VLM) ===")
    print(f"Stage: {args.stage} | Model: {args.model_name_or_path} | Device: {device} | Seed: {args.seed}")
    print(f"Spatial Pooling (49 tokens): {args.spatial_pooling}\n")

    from task2.data import Task2DataPreparer, CurriculumBatchSampler
    preparer = Task2DataPreparer(data_dir=args.data_dir, seed=args.seed)

    splits_file = Path(args.data_dir) / "curriculum_splits.pt"
    features_cache_file = Path(args.data_dir) / "cached_patch_features.pt"

    # 1. Prepare Data
    if args.stage in ["all", "prepare_data"]:
        print(">>> Stage: Preparing Annotations & Data Splits <<<", flush=True)
        preparer.fetch_all_annotations()

        # Build datasets
        train_caps, dev_caps, res_train_imgs, res_val_imgs = preparer.build_coco_captions_datasets()
        ground_train_pool, ground_dev, ground_test, u_train1, u_val1 = preparer.build_grounding_datasets(res_train_imgs, res_val_imgs)
        res_train_imgs.update(u_train1)
        res_val_imgs.update(u_val1)

        vqa_train, vqa_dev, vqa_test, u_train2, u_val2 = preparer.build_natural_vqav2_datasets(res_train_imgs, res_val_imgs)

        # Collect all required images
        all_specs = []
        for c in train_caps:
            all_specs.append((c["image_id"], "train2014"))
        for c in dev_caps:
            all_specs.append((c["image_id"], "val2014"))
        for p in ground_train_pool:
            all_specs.append((p["image_1"], "train2014"))
            all_specs.append((p["image_2"], "train2014"))
        for p in ground_dev + ground_test:
            all_specs.append((p["image_1"], "val2014"))
            all_specs.append((p["image_2"], "val2014"))
        for p in vqa_train:
            all_specs.append((p["image_1"], "train2014"))
            all_specs.append((p["image_2"], "train2014"))
        for p in vqa_dev + vqa_test:
            all_specs.append((p["image_1"], "val2014"))
            all_specs.append((p["image_2"], "val2014"))

        # Deduplicate
        all_specs = sorted(list(set(all_specs)))
        print(f"Total unique images required: {len(all_specs)}")

        # Download required images
        preparer.download_required_images(all_specs)

        # Save splits
        torch.save({
            "train_caps": train_caps,
            "dev_caps": dev_caps,
            "ground_train_pool": ground_train_pool,
            "ground_dev": ground_dev,
            "ground_test": ground_test,
            "vqa_train": vqa_train,
            "vqa_dev": vqa_dev,
            "vqa_test": vqa_test,
            "all_specs": all_specs
        }, splits_file)
        print(f"Saved splits to {splits_file}")

    # 2. Extract SigLIP Features
    if args.stage in ["all", "extract_features"]:
        print("\n>>> Stage: Extracting SigLIP Features <<<", flush=True)
        if not splits_file.exists():
            raise FileNotFoundError(f"Splits file {splits_file} not found. Run --stage prepare_data first.")
        splits_data = torch.load(splits_file, weights_only=False)
        all_specs = splits_data["all_specs"]
        preparer.precompute_siglip_features(all_specs)

    # 3. Train Curriculum
    if args.stage in ["all", "train_curriculum"]:
        print("\n>>> Stage: Training 3-Stage Curriculum <<<", flush=True)
        from task2.trainer import Task2Trainer

        splits_data = torch.load(splits_file, weights_only=False)
        cached_features = torch.load(features_cache_file, weights_only=False)

        trainer = Task2Trainer(
            base_model_path=args.model_name_or_path,
            output_dir=args.output_dir,
            seed=args.seed,
            connector_spatial_pooling=args.spatial_pooling
        )

        # Calibrate connector scale
        trainer.calibrate_connector(splits_data["train_caps"], cached_features)

        # Build curriculum sampler
        sampler = CurriculumBatchSampler(
            captions_train=splits_data["train_caps"],
            grounding_train_pool=splits_data["ground_train_pool"],
            vqa_train_pairs=splits_data["vqa_train"],
            tokenizer=trainer.tokenizer,
            cached_features=cached_features,
            seed=args.seed
        )

        trainer.train_curriculum(
            sampler=sampler,
            dev_captions=splits_data["dev_caps"],
            grounding_dev=splits_data["ground_dev"],
            vqa_dev=splits_data["vqa_dev"],
            cached_features=cached_features
        )

    # 4. Final Test Evaluations
    if args.stage in ["all", "eval_final"]:
        print("\n>>> Stage: Running Final Test Evaluations <<<", flush=True)
        from task2.models import QwenVLM
        from task2.analysis import run_final_test_evaluations

        splits_data = torch.load(splits_file, weights_only=False)
        cached_features = torch.load(features_cache_file, weights_only=False)

        # Load final trained model
        vlm = QwenVLM(
            base_model_path=args.model_name_or_path,
            connector_spatial_pooling=args.spatial_pooling,
            device=device
        )
        # Load final connector & LoRA
        conn_path = Path(args.output_dir) / "connector_stage_3.pt"
        lora_path = Path(args.output_dir) / "lora_stage_3.pt"
        if conn_path.exists():
            vlm.connector.load_state_dict(torch.load(conn_path, map_location=device))
            print(f"Loaded trained connector from {conn_path}")
        if lora_path.exists():
            vlm.enable_lora()
            vlm.lora_model.load_state_dict(torch.load(lora_path, map_location=device))
            print(f"Loaded trained LoRA from {lora_path}")

        run_final_test_evaluations(
            vlm=vlm,
            grounding_test=splits_data["ground_test"],
            vqa_test=splits_data["vqa_test"],
            cached_features=cached_features,
            output_dir=args.output_dir,
            seed=args.seed
        )

    # 5. Analysis, Plots & Markdown Evidence
    if args.stage in ["all", "analysis"]:
        print("\n>>> Stage: Synthesizing Evidence, Tables and Plots <<<", flush=True)
        from task2.analysis import generate_task2_plots_and_tables
        generate_task2_plots_and_tables(
            train_summary_path=f"{args.output_dir}/train_summary.json",
            final_results_path=f"{args.output_dir}/final_test_results.json",
            output_dir=args.output_dir
        )

    print("\n=== ATML PA3 Task 2 Complete ===")

if __name__ == "__main__":
    main()
