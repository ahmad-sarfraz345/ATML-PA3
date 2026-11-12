from __future__ import annotations
import math
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from task3.scoring import (
    compute_rollout_metrics,
    compute_visual_engagement,
    calibrate_visconf_pool,
    aggregate_self_consistency,
    aggregate_highest_score,
    aggregate_rank_weighted_consensus
)
from task3.data import (
    NO_ANSWER,
    check_math_equivalence,
    check_mathvista_equivalence
)

def generate_rollout_orders(
    sample_ids: List[str],
    n_orders: int = 10,
    seed: int = 6304
) -> Dict[str, List[List[int]]]:
    """
    Generates 10 seeded random permutations of candidate IDs {0..31} per question.
    Reused across all candidate budgets, scoring methods, and stopping policies.
    """
    rng = np.random.default_rng(seed)
    orders_by_q = {}
    for sid in sorted(sample_ids):
        orders = []
        for _ in range(n_orders):
            orders.append(rng.permutation(32).tolist())
        orders_by_q[sid] = orders
    return orders_by_q

def bootstrap_question_metric_ci(
    per_question_values: List[float],
    num_bootstrap: int = 1000,
    seed: int = 6304,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Paired question-level bootstrap 95% confidence intervals over 1,000 resamples.
    Returns (mean, lower_ci, upper_ci).
    """
    arr = np.asarray(per_question_values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)

    boot_means = np.empty(num_bootstrap, dtype=np.float64)
    for b in range(num_bootstrap):
        sampled_indices = rng.choice(n, size=n, replace=True)
        boot_means[b] = np.mean(arr[sampled_indices])

    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    return float(np.mean(boot_means)), float(np.percentile(boot_means, lower_pct)), float(np.percentile(boot_means, upper_pct))

class OfflineSamplingEvaluator:
    """
    Evaluates test-time compute scaling across budgets N in {1, 2, 4, 8, 16, 32}.
    Averages over 10 permutations per question, then across questions.
    """
    def __init__(
        self,
        df_dataset: pd.DataFrame,
        is_math: bool = True,
        seed: int = 6304
    ):
        self.df = df_dataset
        self.is_math = is_math
        self.seed = seed
        self.check_eq_fn = check_math_equivalence if is_math else check_mathvista_equivalence

        # Index data by sample_id -> list of 32 rollout records sorted by rollout_id
        self.questions_data: Dict[str, List[Dict[str, Any]]] = {}
        grouped = self.df.groupby("sample_id")
        for sid, group in grouped:
            records = group.sort_values("rollout_id").to_dict("records")
            self.questions_data[str(sid)] = records

        self.sample_ids = sorted(list(self.questions_data.keys()))

    def precompute_question_features(self):
        """
        Precomputes correctness labels, tokens, logp, kl, and visual engagement per rollout.
        """
        print("Precomputing rollout metrics and correctness labels...")
        for sid, rollouts in self.questions_data.items():
            for r in rollouts:
                gold = r.get("gold_answer", "")
                pred = r.get("extracted_answer", None)

                # Correctness against reference answer
                r["is_correct"] = float(self.check_eq_fn(pred, gold))

                # Quality scores
                logp = r.get("logp", [])
                kl = r.get("kl_u_p", [])
                L_i, K_i = compute_rollout_metrics(logp, kl)
                r["L_i"] = L_i
                r["K_i"] = K_i

                # VLM visual engagement
                if not self.is_math and "img_attn_total" in r:
                    v_it, V_i = compute_visual_engagement(
                        r["img_attn_total"],
                        r["prompt_text_attn_total"],
                        r["generated_text_attn_total"]
                    )
                    r["v_it"] = v_it
                    r["V_i"] = V_i
                else:
                    r["v_it"] = np.array([])
                    r["V_i"] = 0.0

            # Precompute candidate equivalence answer groups within the 32 rollouts of this question
            canon_reps: List[str] = []
            for r in rollouts:
                ans = r.get("extracted_answer", None)
                if ans is None or not str(ans).strip() or ans == NO_ANSWER:
                    r["group_id"] = -1
                else:
                    ans_str = str(ans).strip()
                    matched = None
                    for gid, rep in enumerate(canon_reps):
                        if self.check_eq_fn(ans_str, rep):
                            matched = gid
                            break
                    if matched is not None:
                        r["group_id"] = matched
                    else:
                        new_gid = len(canon_reps)
                        canon_reps.append(ans_str)
                        r["group_id"] = new_gid

    def evaluate_budgets(
        self,
        target_sample_ids: List[str],
        rollout_orders: Dict[str, List[List[int]]],
        budgets: List[int] = [1, 2, 4, 8, 16, 32]
    ) -> Dict[str, Any]:
        """
        Evaluates all candidate budgets N in {1, 2, 4, 8, 16, 32} across all selection methods.
        """
        methods = [
            "FirstAns",
            "Self-Consistency",
            "Highest-LogProb",
            "Highest-SelfCertainty",
            "SelfCertainty-RWC"
        ]
        if not self.is_math:
            methods.extend([
                "VisionOnly-Highest",
                "VisionOnly-RWC",
                "VisConf-Highest",
                "VisConf-RWC"
            ])

        # Results structures: method -> N -> list of per-question averages
        budget_accs: Dict[str, Dict[int, List[float]]] = {m: {N: [] for N in budgets} for m in methods}
        budget_costs: Dict[str, Dict[int, List[float]]] = {m: {N: [] for N in budgets} for m in methods}
        oracle_coverages: Dict[int, List[float]] = {N: [] for N in budgets}
        num_answer_groups: Dict[int, List[float]] = {N: [] for N in budgets}
        sampling_failures: Dict[str, Dict[int, List[float]]] = {m: {N: [] for N in budgets} for m in methods}
        selection_failures: Dict[str, Dict[int, List[float]]] = {m: {N: [] for N in budgets} for m in methods}

        for sid in target_sample_ids:
            rollouts = self.questions_data[sid]
            orders = rollout_orders[sid]

            # Collect metrics across 10 orders for this question
            for N in budgets:
                order_oracle = []
                order_groups = []
                order_method_acc = {m: [] for m in methods}
                order_method_cost = {m: [] for m in methods}
                order_samp_fail = {m: [] for m in methods}
                order_sel_fail = {m: [] for m in methods}

                for perm in orders:
                    cand_indices = perm[:N]
                    cands = [rollouts[i] for i in cand_indices]

                    # Token costs
                    cand_tokens = [c.get("n_generated_text_tokens", 0) for c in cands]
                    total_tokens_N = sum(cand_tokens)

                    # Group candidate answers from precomputed groups
                    group_ids = [c["group_id"] for c in cands]
                    distinct_groups = len(set(g for g in group_ids if g != -1))
                    order_groups.append(distinct_groups)

                    # Oracle coverage
                    cand_corrects = [c["is_correct"] for c in cands]
                    has_correct = float(any(c > 0.5 for c in cand_corrects))
                    order_oracle.append(has_correct)

                    # Scores
                    logp_scores = [c["L_i"] for c in cands]
                    sc_scores = [c["K_i"] for c in cands]

                    # VisConf calibration if MathVista
                    if not self.is_math:
                        kl_lists = [c.get("kl_u_p", []) for c in cands]
                        v_lists = [c.get("v_it", np.array([])) for c in cands]
                        v_scores = [c["V_i"] for c in cands]
                        visconf_scores, _, _ = calibrate_visconf_pool(kl_lists, v_lists)

                    # Selections
                    selections = {}

                    # 1. FirstAns
                    selections["FirstAns"] = (group_ids[0], 0, cand_tokens[0])

                    # 2. Self-Consistency
                    sc_gid, sc_idx, _ = aggregate_self_consistency(group_ids)
                    selections["Self-Consistency"] = (sc_gid, sc_idx, total_tokens_N)

                    # 3. Highest Log-Probability
                    logp_gid, logp_idx = aggregate_highest_score(group_ids, logp_scores)
                    selections["Highest-LogProb"] = (logp_gid, logp_idx, total_tokens_N)

                    # 4. Highest Self-Certainty
                    cert_gid, cert_idx = aggregate_highest_score(group_ids, sc_scores)
                    selections["Highest-SelfCertainty"] = (cert_gid, cert_idx, total_tokens_N)

                    # 5. Self-Certainty + RWC
                    rwc_gid, rwc_idx, _, _ = aggregate_rank_weighted_consensus(group_ids, sc_scores)
                    selections["SelfCertainty-RWC"] = (rwc_gid, rwc_idx, total_tokens_N)

                    if not self.is_math:
                        # 6. Vision-Only Highest
                        vo_h_gid, vo_h_idx = aggregate_highest_score(group_ids, v_scores)
                        selections["VisionOnly-Highest"] = (vo_h_gid, vo_h_idx, total_tokens_N)

                        # 7. Vision-Only RWC
                        vo_rwc_gid, vo_rwc_idx, _, _ = aggregate_rank_weighted_consensus(group_ids, v_scores)
                        selections["VisionOnly-RWC"] = (vo_rwc_gid, vo_rwc_idx, total_tokens_N)

                        # 8. VisConf Highest
                        vc_h_gid, vc_h_idx = aggregate_highest_score(group_ids, visconf_scores)
                        selections["VisConf-Highest"] = (vc_h_gid, vc_h_idx, total_tokens_N)

                        # 9. VisConf RWC
                        vc_rwc_gid, vc_rwc_idx, _, _ = aggregate_rank_weighted_consensus(group_ids, visconf_scores)
                        selections["VisConf-RWC"] = (vc_rwc_gid, vc_rwc_idx, total_tokens_N)

                    # Record correctness & failure decomposition for each method
                    for m in methods:
                        win_gid, win_idx, cost = selections[m]
                        is_corr = cand_corrects[win_idx]
                        order_method_acc[m].append(is_corr)
                        order_method_cost[m].append(cost)

                        # Decomposition:
                        # Sampling failure: no correct candidate generated in the pool of N
                        # Selection failure: correct candidate existed, but method missed it
                        samp_fail = float(has_correct == 0.0)
                        sel_fail = float(has_correct == 1.0 and is_corr == 0.0)
                        order_samp_fail[m].append(samp_fail)
                        order_sel_fail[m].append(sel_fail)

                # Average across 10 orders for this question
                oracle_coverages[N].append(float(np.mean(order_oracle)))
                num_answer_groups[N].append(float(np.mean(order_groups)))
                for m in methods:
                    budget_accs[m][N].append(float(np.mean(order_method_acc[m])))
                    budget_costs[m][N].append(float(np.mean(order_method_cost[m])))
                    sampling_failures[m][N].append(float(np.mean(order_samp_fail[m])))
                    selection_failures[m][N].append(float(np.mean(order_sel_fail[m])))

        # Compute summary metrics & 95% bootstrap CIs
        summary: Dict[str, Any] = {
            "budgets": budgets,
            "methods": methods,
            "oracle_coverage": {},
            "distinct_groups": {},
            "method_results": {}
        }

        for N in budgets:
            o_mean, o_low, o_high = bootstrap_question_metric_ci(oracle_coverages[N], seed=self.seed)
            g_mean, _, _ = bootstrap_question_metric_ci(num_answer_groups[N], seed=self.seed)
            summary["oracle_coverage"][N] = {
                "mean_percent": round(o_mean * 100.0, 2),
                "ci_95": [round(o_low * 100.0, 2), round(o_high * 100.0, 2)]
            }
            summary["distinct_groups"][N] = round(g_mean, 2)

        for m in methods:
            summary["method_results"][m] = {}
            for N in budgets:
                acc_mean, acc_low, acc_high = bootstrap_question_metric_ci(budget_accs[m][N], seed=self.seed)
                cost_mean = float(np.mean(budget_costs[m][N]))
                samp_fail_mean = float(np.mean(sampling_failures[m][N])) * 100.0
                sel_fail_mean = float(np.mean(selection_failures[m][N])) * 100.0

                summary["method_results"][m][N] = {
                    "accuracy_percent": round(acc_mean * 100.0, 2),
                    "accuracy_95_ci": [round(acc_low * 100.0, 2), round(acc_high * 100.0, 2)],
                    "mean_token_cost": round(cost_mean, 1),
                    "sampling_failure_percent": round(samp_fail_mean, 2),
                    "selection_failure_percent": round(sel_fail_mean, 2)
                }

        return summary

    def analyze_signal_properties_step3(
        self,
        target_sample_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Step 3 Analysis at N = 32:
        - Associations between rollout length, quality scores, and correctness
        - Top vs bottom quartile correctness differences within questions
        - Shortest-rollout selection as length-only control
        - MathVista confidence-vision correlation Cov_t(c_{i,t}, v_{i,t})
        - Tokenwise fusion vs trajectory fusion
        """
        print("Analyzing signal properties (Step 3 at N=32)...")
        shortest_corrects = []
        q_quartile_diffs = {
            "logp": [],
            "self_certainty": []
        }
        if not self.is_math:
            q_quartile_diffs["visconf"] = []
            q_quartile_diffs["vision_only"] = []
            cov_c_v_list = []
            tokenwise_rwc_accs = []
            trajectory_rwc_accs = []

        for sid in target_sample_ids:
            rollouts = self.questions_data[sid]
            n_cands = len(rollouts)
            corrects = np.array([r["is_correct"] for r in rollouts])
            lengths = np.array([r.get("n_generated_text_tokens", 0) for r in rollouts])

            # Shortest-rollout selection
            shortest_idx = int(np.argmin(lengths))
            shortest_corrects.append(corrects[shortest_idx])

            # Quartile correctness difference helper
            def calc_q_diff(scores_arr):
                q75 = np.percentile(scores_arr, 75)
                q25 = np.percentile(scores_arr, 25)
                top_mask = (scores_arr >= q75)
                bot_mask = (scores_arr <= q25)
                top_acc = float(np.mean(corrects[top_mask])) if np.any(top_mask) else 0.0
                bot_acc = float(np.mean(corrects[bot_mask])) if np.any(bot_mask) else 0.0
                return top_acc - bot_acc

            logp_scores = np.array([r["L_i"] for r in rollouts])
            sc_scores = np.array([r["K_i"] for r in rollouts])
            q_quartile_diffs["logp"].append(calc_q_diff(logp_scores))
            q_quartile_diffs["self_certainty"].append(calc_q_diff(sc_scores))

            if not self.is_math:
                v_scores = np.array([r["V_i"] for r in rollouts])
                kl_lists = [r.get("kl_u_p", []) for r in rollouts]
                v_lists = [r.get("v_it", np.array([])) for r in rollouts]
                visconf_scores, tokenwise_scores, _ = calibrate_visconf_pool(kl_lists, v_lists)

                q_quartile_diffs["vision_only"].append(calc_q_diff(v_scores))
                q_quartile_diffs["visconf"].append(calc_q_diff(np.array(visconf_scores)))

                # Tokenwise covariance Cov_t(c_{i,t}, v_{i,t}) across rollouts
                for i in range(n_cands):
                    cov_diff = tokenwise_scores[i] - visconf_scores[i]
                    cov_c_v_list.append(cov_diff)

                # Compare Tokenwise RWC vs Trajectory VisConf RWC
                group_ids = [r["group_id"] for r in rollouts]

                _, traj_win_idx, _, _ = aggregate_rank_weighted_consensus(group_ids, visconf_scores)
                _, tok_win_idx, _, _ = aggregate_rank_weighted_consensus(group_ids, tokenwise_scores)

                trajectory_rwc_accs.append(corrects[traj_win_idx])
                tokenwise_rwc_accs.append(corrects[tok_win_idx])

        results = {
            "shortest_rollout_accuracy_percent": round(float(np.mean(shortest_corrects)) * 100.0, 2),
            "top_minus_bottom_quartile_acc_diff": {
                k: round(float(np.mean(v)) * 100.0, 2) for k, v in q_quartile_diffs.items()
            }
        }
        if not self.is_math:
            results["mean_token_covariance_cov_c_v"] = round(float(np.mean(cov_c_v_list)), 6)
            results["trajectory_visconf_rwc_acc"] = round(float(np.mean(trajectory_rwc_accs)) * 100.0, 2)
            results["tokenwise_fusion_rwc_acc"] = round(float(np.mean(tokenwise_rwc_accs)) * 100.0, 2)

        return results
