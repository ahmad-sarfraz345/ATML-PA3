from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path

from common.utils import set_seed, get_device, save_json, load_json

def parse_args():
    parser = argparse.ArgumentParser(description="ATML PA3 Task 3 Runner (Test-Time Compute & Adaptive Inference)")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=[
            "all",
            "download_data",
            "offline_math",
            "offline_mathvista",
            "adaptive_math",
            "adaptive_mathvista",
            "live_refinement",
            "analysis"
        ],
        help="Stage to execute."
    )
    parser.add_argument("--data_dir", type=str, default="data/task3")
    parser.add_argument("--output_dir", type=str, default="outputs/task3")
    parser.add_argument("--seed", type=int, default=6304)
    parser.add_argument("--gdrive_folder_id", type=str, default="1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23")
    parser.add_argument("--live_model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    return parser.parse_args()

def ensure_data_downloaded(data_dir: Path, gdrive_folder_id: str):
    """Downloads ATML-PA3-Data from Google Drive or links from Kaggle input if parquet files are missing."""
    data_dir.mkdir(parents=True, exist_ok=True)
    math_gen = data_dir / "qwen25-math" / "generations.parquet"
    vista_gen = data_dir / "qwen25vl-mathvista" / "generations.parquet"

    if math_gen.exists() and vista_gen.exists():
        print(f"ATML-PA3-Data already available at {data_dir}.")
        return

    # Check potential Kaggle input paths
    kaggle_input_base = Path("/kaggle/input")
    if kaggle_input_base.exists():
        print("Checking /kaggle/input for ATML-PA3-Data...")
        for candidate in kaggle_input_base.rglob("generations.parquet"):
            # candidate could be .../qwen25-math/generations.parquet
            if candidate.parent.name == "qwen25-math":
                dataset_root = candidate.parent.parent
                print(f"Found ATML-PA3-Data root at {dataset_root}. Copying to {data_dir}...")
                import shutil
                for item in dataset_root.iterdir():
                    dst = data_dir / item.name
                    if not dst.exists():
                        if item.is_dir():
                            shutil.copytree(item, dst)
                        else:
                            shutil.copy2(item, dst)
                break

    if math_gen.exists() and vista_gen.exists():
        print(f"ATML-PA3-Data successfully sourced from /kaggle/input to {data_dir}.")
        return

    # If still not found, download automatically from Google Drive via gdown
    print(f"Downloading ATML-PA3-Data from Google Drive folder {gdrive_folder_id}...")
    import gdown
    url = f"https://drive.google.com/drive/folders/{gdrive_folder_id}"
    gdown.download_folder(url=url, output=str(data_dir), quiet=False, use_cookies=False)

    # In case gdown created a nested subfolder like data/task3/ATML-PA3-Data
    nested = data_dir / "ATML-PA3-Data"
    if nested.exists() and (nested / "qwen25-math").exists():
        import shutil
        print(f"Flattening nested directory {nested} into {data_dir}...")
        for sub in nested.iterdir():
            dst = data_dir / sub.name
            if not dst.exists():
                shutil.move(str(sub), str(dst))

    if math_gen.exists() and vista_gen.exists():
        print("Download completed successfully.")
    else:
        print("WARNING: Could not verify generations.parquet in both qwen25-math and qwen25vl-mathvista.")

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== ATML PA3 Task 3 Pipeline (Test-Time Compute & Adaptive Inference) ===")
    print(f"Stage: {args.stage} | Data: {data_dir} | Output: {out_dir} | Device: {device} | Seed: {args.seed}\n")

    # 1. Ensure Data Downloaded
    if args.stage in ["all", "download_data"]:
        ensure_data_downloaded(data_dir, args.gdrive_folder_id)

    from task3.data import Task3Dataset
    from task3.offline_sampling import (
        generate_rollout_orders,
        OfflineSamplingEvaluator
    )
    from task3.adaptive import AdaptiveSamplingSimulator

    dataset_loader = Task3Dataset(data_dir=str(data_dir), seed=args.seed)

    # 2. Offline MATH Evaluation (Steps 1, 2, 3)
    if args.stage in ["all", "offline_math"]:
        print("\n>>> Stage: Offline MATH Test-Time Compute Scaling <<<", flush=True)
        df_math = dataset_loader.load_math_data()
        sample_ids = df_math["sample_id"].unique().tolist()
        dev_ids, eval_ids, _ = dataset_loader.compute_dataset_splits(sample_ids)

        evaluator = OfflineSamplingEvaluator(df_math, is_math=True, seed=args.seed)
        evaluator.precompute_question_features()

        rollout_orders = generate_rollout_orders(sample_ids, n_orders=10, seed=args.seed)

        # Evaluate on Held-out Evaluation Set (800 questions)
        math_summary = evaluator.evaluate_budgets(eval_ids, rollout_orders)
        save_json(math_summary, out_dir / "offline_math_results.json")
        print(f"Saved MATH offline results to {out_dir / 'offline_math_results.json'}")

        # Step 3 signal analysis at N=32
        step3_math = evaluator.analyze_signal_properties_step3(eval_ids)
        save_json(step3_math, out_dir / "signal_analysis_math.json")
        print(f"Saved MATH signal analysis to {out_dir / 'signal_analysis_math.json'}")

    # 3. Offline MathVista Evaluation (Steps 1, 2, 3)
    if args.stage in ["all", "offline_mathvista"]:
        print("\n>>> Stage: Offline MathVista Quality-Aware & VisConf Scaling <<<", flush=True)
        df_vista = dataset_loader.load_mathvista_data()
        sample_ids_v = df_vista["sample_id"].unique().tolist()
        dev_ids_v, eval_ids_v, _ = dataset_loader.compute_dataset_splits(sample_ids_v)

        evaluator_v = OfflineSamplingEvaluator(df_vista, is_math=False, seed=args.seed)
        evaluator_v.precompute_question_features()

        rollout_orders_v = generate_rollout_orders(sample_ids_v, n_orders=10, seed=args.seed)

        # Evaluate on Held-out Evaluation Set (800 questions)
        vista_summary = evaluator_v.evaluate_budgets(eval_ids_v, rollout_orders_v)
        save_json(vista_summary, out_dir / "offline_mathvista_results.json")
        print(f"Saved MathVista offline results to {out_dir / 'offline_mathvista_results.json'}")

        # Step 3 signal analysis at N=32
        step3_vista = evaluator_v.analyze_signal_properties_step3(eval_ids_v)
        save_json(step3_vista, out_dir / "signal_analysis_mathvista.json")
        print(f"Saved MathVista signal analysis to {out_dir / 'signal_analysis_mathvista.json'}")

    # 4. Adaptive Inference Simulation on MATH (Step 5)
    if args.stage in ["all", "adaptive_math"]:
        print("\n>>> Stage: Adaptive Sampling Simulation on MATH <<<", flush=True)
        df_math = dataset_loader.load_math_data()
        sample_ids = df_math["sample_id"].unique().tolist()
        dev_ids, eval_ids, _ = dataset_loader.compute_dataset_splits(sample_ids)

        evaluator = OfflineSamplingEvaluator(df_math, is_math=True, seed=args.seed)
        evaluator.precompute_question_features()
        rollout_orders = generate_rollout_orders(sample_ids, n_orders=10, seed=args.seed)

        adapt_sim = AdaptiveSamplingSimulator(
            questions_data=evaluator.questions_data,
            is_math=True,
            check_eq_fn=evaluator.check_eq_fn
        )

        sweep_res = adapt_sim.sweep_hyperparameters_dev(dev_ids, rollout_orders)
        save_json(sweep_res, out_dir / "adaptive_sweep_math.json")

        eval_res = adapt_sim.evaluate_held_out(
            eval_ids,
            rollout_orders,
            sweep_res["best_agreement"],
            sweep_res["best_quality_aware"]
        )
        save_json(eval_res, out_dir / "adaptive_eval_math.json")
        print(f"Saved MATH adaptive results to {out_dir / 'adaptive_eval_math.json'}")

    # 5. Adaptive Inference Simulation on MathVista (Step 5)
    if args.stage in ["all", "adaptive_mathvista"]:
        print("\n>>> Stage: Adaptive Sampling Simulation on MathVista <<<", flush=True)
        df_vista = dataset_loader.load_mathvista_data()
        sample_ids_v = df_vista["sample_id"].unique().tolist()
        dev_ids_v, eval_ids_v, _ = dataset_loader.compute_dataset_splits(sample_ids_v)

        evaluator_v = OfflineSamplingEvaluator(df_vista, is_math=False, seed=args.seed)
        evaluator_v.precompute_question_features()
        rollout_orders_v = generate_rollout_orders(sample_ids_v, n_orders=10, seed=args.seed)

        adapt_sim_v = AdaptiveSamplingSimulator(
            questions_data=evaluator_v.questions_data,
            is_math=False,
            check_eq_fn=evaluator_v.check_eq_fn
        )

        sweep_res_v = adapt_sim_v.sweep_hyperparameters_dev(dev_ids_v, rollout_orders_v)
        save_json(sweep_res_v, out_dir / "adaptive_sweep_mathvista.json")

        eval_res_v = adapt_sim_v.evaluate_held_out(
            eval_ids_v,
            rollout_orders_v,
            sweep_res_v["best_agreement"],
            sweep_res_v["best_quality_aware"]
        )
        save_json(eval_res_v, out_dir / "adaptive_eval_mathvista.json")
        print(f"Saved MathVista adaptive results to {out_dir / 'adaptive_eval_mathvista.json'}")

    # 6. Live Inference & Refinement on 50 MATH Questions (Step 4)
    if args.stage in ["all", "live_refinement"]:
        print("\n>>> Stage: Live Generation & Sequential Refinement (50 Questions) <<<", flush=True)
        df_math = dataset_loader.load_math_data()
        sample_ids = df_math["sample_id"].unique().tolist()
        _, eval_ids, live_ids = dataset_loader.compute_dataset_splits(sample_ids)

        # Get first rollout record of each of the 50 live questions for question_text and gold_answer
        live_questions = []
        for sid in live_ids:
            subset = df_math[df_math["sample_id"] == sid]
            if len(subset) > 0:
                live_questions.append(subset.iloc[0].to_dict())

        from task3.live_refinement import LiveRefinementRunner
        live_runner = LiveRefinementRunner(
            model_name=args.live_model_name,
            device=device,
            output_dir=args.output_dir
        )
        live_runner.run_live_comparison(live_questions, seed=args.seed)

    # 7. Analysis & Visualizations
    if args.stage in ["all", "analysis"]:
        print("\n>>> Stage: Synthesizing Task 3 Master Figures and Summary <<<", flush=True)
        from task3.analysis import generate_task3_visualizations_and_report
        generate_task3_visualizations_and_report(output_dir=args.output_dir)

    print("\n=== ATML PA3 Task 3 Complete ===")

if __name__ == "__main__":
    main()
