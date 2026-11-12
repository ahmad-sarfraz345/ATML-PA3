from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch

from task3.data import extract_boxed_answer, check_math_equivalence, NO_ANSWER
from common.utils import save_json, load_json

SYSTEM_PROMPT = "You are a helpful assistant."
PROMPT_SUFFIX = "\n\nThink step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
CRITIQUE_PROMPT = "Review the solution above. Identify and correct any errors; if it is correct, retain its answer. Give your final answer inside \\boxed{}."

def format_chat_prompt(tokenizer, question_text: str) -> str:
    """Formats the initial reasoning prompt using Qwen chat template."""
    content = f"{question_text.strip()}{PROMPT_SUFFIX}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def format_critique_prompt(tokenizer, question_text: str, initial_solution: str) -> str:
    """Formats the sequential critique and revision prompt."""
    content = f"{question_text.strip()}{PROMPT_SUFFIX}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
        {"role": "assistant", "content": initial_solution.strip()},
        {"role": "user", "content": CRITIQUE_PROMPT}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

class LiveRefinementRunner:
    """
    Executes live generation on the fixed 50 MATH evaluation questions
    using Qwen2.5-3B-Instruct in FP16 comparing:
    1. Single longer response (cap 3,072)
    2. Parallel sampling (3 responses x cap 1,024, SC)
    3. Sequential refinement (initial 1,024 + revision 2,048)
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: torch.device = None,
        output_dir: str = "outputs/task3"
    ):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Loads Qwen2.5-3B-Instruct in FP16 with native tokenizer."""
        if self.model is not None:
            return
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"Loading {self.model_name} in FP16 on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        print("Model successfully loaded.")

    @torch.no_grad()
    def generate_single_response(
        self,
        prompt_text: str,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Runs single rollout with generation hyperparameters."""
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        n_input_tokens = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        gen_time = time.perf_counter() - t0

        gen_tokens = outputs[0][n_input_tokens:]
        n_output_tokens = len(gen_tokens)
        hit_max = (n_output_tokens >= max_new_tokens)

        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        extracted = extract_boxed_answer(text)

        return {
            "text": text,
            "extracted": extracted,
            "n_input_tokens": n_input_tokens,
            "n_output_tokens": n_output_tokens,
            "hit_max_new_tokens": hit_max,
            "runtime_sec": gen_time
        }

    def run_live_comparison(
        self,
        questions_records: List[Dict[str, Any]],
        seed: int = 6304
    ) -> Dict[str, Any]:
        """
        Runs the 3 strategies on the 50 MATH evaluation questions.
        """
        self.load_model()
        results_file = self.output_dir / "live_refinement_results.json"

        # Check if already cached
        if results_file.exists():
            print(f"Found existing live refinement results at {results_file}.")
            return load_json(results_file)

        print(f"\n--- Running Live Inference Comparison on {len(questions_records)} MATH Questions ---")
        rng = np.random.default_rng(seed)

        strategy_records = {
            "single_longer": [],
            "parallel_sampling": [],
            "sequential_refinement": []
        }

        # Tracking transitions for refinement relative to initial parallel response
        w2c_transitions = 0
        c2w_transitions = 0
        w2w_transitions = 0
        c2c_transitions = 0

        for q_idx, q in enumerate(questions_records):
            sid = q["sample_id"]
            q_text = q["question_text"]
            gold = q.get("gold_answer", "")
            base_prompt = format_chat_prompt(self.tokenizer, q_text)

            print(f"Processing question {q_idx + 1}/{len(questions_records)} (ID: {sid})...", flush=True)

            # Strategy 1: Single longer response (cap 3,072)
            s1_res = self.generate_single_response(
                prompt_text=base_prompt,
                max_new_tokens=3072,
                seed=int(rng.integers(0, 1000000))
            )
            s1_corr = float(check_math_equivalence(s1_res["extracted"], gold))
            strategy_records["single_longer"].append({
                "sample_id": sid,
                "extracted": s1_res["extracted"],
                "is_correct": s1_corr,
                "input_tokens": s1_res["n_input_tokens"],
                "output_tokens": s1_res["n_output_tokens"],
                "runtime_sec": s1_res["runtime_sec"],
                "hit_max": s1_res["hit_max_new_tokens"]
            })

            # Strategy 2: Parallel sampling (3 independent responses, cap 1,024 each)
            parallel_cands = []
            for p_i in range(3):
                p_res = self.generate_single_response(
                    prompt_text=base_prompt,
                    max_new_tokens=1024,
                    seed=int(rng.integers(0, 1000000))
                )
                parallel_cands.append(p_res)

            # Extract answers & group for Self-Consistency
            p_extracted = [p["extracted"] for p in parallel_cands]
            p_corrects = [float(check_math_equivalence(p["extracted"], gold)) for p in parallel_cands]

            # Form answer groups without gold answer
            group_ids = []
            canon_reps = []
            for ans in p_extracted:
                if ans is None or not str(ans).strip() or ans == NO_ANSWER:
                    group_ids.append(-1)
                else:
                    matched = None
                    for gid, rep in enumerate(canon_reps):
                        if check_math_equivalence(ans, rep):
                            matched = gid
                            break
                    if matched is not None:
                        group_ids.append(matched)
                    else:
                        new_id = len(canon_reps)
                        canon_reps.append(ans)
                        group_ids.append(new_id)

            # SC majority vote
            from task3.scoring import aggregate_self_consistency
            sc_win_gid, sc_win_idx, _ = aggregate_self_consistency(group_ids)
            s2_corr = p_corrects[sc_win_idx]
            s2_input_tokens = sum(p["n_input_tokens"] for p in parallel_cands)
            s2_output_tokens = sum(p["n_output_tokens"] for p in parallel_cands)
            s2_runtime = sum(p["runtime_sec"] for p in parallel_cands)
            s2_hit_max = any(p["hit_max_new_tokens"] for p in parallel_cands)

            strategy_records["parallel_sampling"].append({
                "sample_id": sid,
                "selected_extracted": p_extracted[sc_win_idx],
                "is_correct": s2_corr,
                "input_tokens": s2_input_tokens,
                "output_tokens": s2_output_tokens,
                "runtime_sec": s2_runtime,
                "hit_max": s2_hit_max
            })

            # Strategy 3: Sequential Refinement
            # Reuse first parallel response (candidate 0) as initial response
            init_res = parallel_cands[0]
            init_corr = p_corrects[0]

            critique_prompt = format_critique_prompt(self.tokenizer, q_text, init_res["text"])
            rev_res = self.generate_single_response(
                prompt_text=critique_prompt,
                max_new_tokens=2048,
                seed=int(rng.integers(0, 1000000))
            )
            rev_corr = float(check_math_equivalence(rev_res["extracted"], gold))

            # Record transition relative to initial response
            if init_corr == 0.0 and rev_corr == 1.0:
                w2c_transitions += 1
            elif init_corr == 1.0 and rev_corr == 0.0:
                c2w_transitions += 1
            elif init_corr == 0.0 and rev_corr == 0.0:
                w2w_transitions += 1
            else:
                c2c_transitions += 1

            s3_input_tokens = init_res["n_input_tokens"] + rev_res["n_input_tokens"]
            s3_output_tokens = init_res["n_output_tokens"] + rev_res["n_output_tokens"]
            s3_runtime = init_res["runtime_sec"] + rev_res["runtime_sec"]
            s3_hit_max = init_res["hit_max_new_tokens"] or rev_res["hit_max_new_tokens"]

            strategy_records["sequential_refinement"].append({
                "sample_id": sid,
                "initial_extracted": init_res["extracted"],
                "revised_extracted": rev_res["extracted"],
                "initial_correct": init_corr,
                "is_correct": rev_corr,
                "input_tokens": s3_input_tokens,
                "output_tokens": s3_output_tokens,
                "runtime_sec": s3_runtime,
                "hit_max": s3_hit_max
            })

        # Summary Metrics
        n_q = len(questions_records)
        summary = {
            "num_questions": n_q,
            "strategy_comparisons": {}
        }
        for strat in ["single_longer", "parallel_sampling", "sequential_refinement"]:
            recs = strategy_records[strat]
            acc = float(np.mean([r["is_correct"] for r in recs])) * 100.0
            mean_in = float(np.mean([r["input_tokens"] for r in recs]))
            mean_out = float(np.mean([r["output_tokens"] for r in recs]))
            mean_time = float(np.mean([r["runtime_sec"] for r in recs]))
            hit_rate = float(np.mean([r["hit_max"] for r in recs])) * 100.0
            def is_fail(r):
                ans = r.get("extracted", r.get("selected_extracted", r.get("revised_extracted")))
                return ans is None or ans == NO_ANSWER or not str(ans).strip()
            fail_ext = float(np.mean([is_fail(r) for r in recs])) * 100.0

            summary["strategy_comparisons"][strat] = {
                "accuracy_percent": round(acc, 2),
                "mean_input_tokens": round(mean_in, 1),
                "mean_output_tokens": round(mean_out, 1),
                "avg_runtime_sec": round(mean_time, 2),
                "truncation_rate_percent": round(hit_rate, 2),
                "extraction_failure_percent": round(fail_ext, 2)
            }

        summary["refinement_transitions"] = {
            "wrong_to_correct": w2c_transitions,
            "correct_to_wrong": c2w_transitions,
            "wrong_to_wrong": w2w_transitions,
            "correct_to_correct": c2c_transitions,
            "net_gain": w2c_transitions - c2w_transitions
        }
        summary["detailed_records"] = strategy_records

        save_json(summary, results_file)
        print(f"Saved live refinement evaluation results to {results_file}")
        return summary
