from __future__ import annotations
import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from task3.scoring import (
    calibrate_visconf_pool,
    aggregate_self_consistency,
    aggregate_rank_weighted_consensus
)
from task3.data import NO_ANSWER

class AdaptiveSamplingSimulator:
    """
    Simulates adaptive inference stopping policies between completed rollouts.
    Checks stopping after n in {4, 8, 16} candidates; otherwise stops at 32.
    Both controllers return the current self-consistency winner.
    """
    def __init__(
        self,
        questions_data: Dict[str, List[Dict[str, Any]]],
        is_math: bool = True,
        check_eq_fn = None
    ):
        self.questions_data = questions_data
        self.is_math = is_math
        self.check_eq_fn = check_eq_fn

    def simulate_question_order(
        self,
        sample_id: str,
        order: List[int],
        policy_type: str,
        tau: float,
        gamma: float = 0.0
    ) -> Dict[str, Any]:
        """
        Reveals rollouts sequentially along order.
        Checks stopping at n in {4, 8, 16}; otherwise stops at 32.
        Returns:
            stop_n: int (4, 8, 16, or 32)
            is_correct: float (1.0 or 0.0)
            token_cost: int
            fixed_32_correct: float
        """
        rollouts = self.questions_data[sample_id]
        check_points = [4, 8, 16]

        # Determine fixed-32 result first for comparison
        cands_32 = [rollouts[i] for i in order]
        gid_32 = self._get_group_ids(cands_32)
        win_32_gid, win_32_idx, _ = aggregate_self_consistency(gid_32)
        fixed_32_correct = cands_32[win_32_idx]["is_correct"]

        stopped_n = 32
        stopped_is_corr = fixed_32_correct
        stopped_cost = sum(c.get("n_generated_text_tokens", 0) for c in cands_32)

        for n in check_points:
            cands_n = [rollouts[i] for i in order[:n]]
            gid_n = self._get_group_ids(cands_n)

            # SC majority vote
            sc_win_gid, sc_win_idx, counts = aggregate_self_consistency(gid_n)

            # Condition 1: Unique leading answer supported by at least fraction tau, not NO_ANSWER
            if sc_win_gid == -1:
                continue

            # Check if leading answer has >= tau fraction of n
            leading_count = counts[sc_win_gid]
            if leading_count / float(n) < tau:
                continue

            # Check uniqueness (strictly greater than runner-up)
            sorted_counts = sorted(counts.values(), reverse=True)
            if len(sorted_counts) > 1 and sorted_counts[0] == sorted_counts[1]:
                continue

            should_stop = False

            if policy_type == "agreement":
                should_stop = True
            elif policy_type == "quality_aware":
                # Condition 2: Quality-aware requires agreement with RWC winner and RWC margin >= gamma
                if self.is_math:
                    scores_n = [c["K_i"] for c in cands_n]
                else:
                    kl_lists = [c.get("kl_u_p", []) for c in cands_n]
                    v_lists = [c.get("v_it", np.array([])) for c in cands_n]
                    scores_n, _, _ = calibrate_visconf_pool(kl_lists, v_lists)

                rwc_win_gid, _, _, rwc_margin = aggregate_rank_weighted_consensus(gid_n, scores_n)

                # Both win same group and margin >= gamma
                if sc_win_gid == rwc_win_gid and rwc_margin >= gamma:
                    should_stop = True

            if should_stop:
                stopped_n = n
                stopped_is_corr = cands_n[sc_win_idx]["is_correct"]
                stopped_cost = sum(c.get("n_generated_text_tokens", 0) for c in cands_n)
                break

        return {
            "stop_n": stopped_n,
            "is_correct": stopped_is_corr,
            "token_cost": stopped_cost,
            "fixed_32_correct": fixed_32_correct
        }

    def _get_group_ids(self, cands: List[Dict[str, Any]]) -> List[int]:
        """Returns precomputed candidate answer group IDs."""
        return [c["group_id"] for c in cands]

    def sweep_hyperparameters_dev(
        self,
        dev_sample_ids: List[str],
        rollout_orders: Dict[str, List[List[int]]]
    ) -> Dict[str, Any]:
        """
        Sweeps tau in {0.5, 0.625, 0.75, 0.875, 1.0} for Agreement controller,
        and (tau, gamma) in {0, 0.1, 0.25, 0.5} for Quality-aware controller.
        Selects setting with lowest mean token cost whose dev accuracy is within 1.0 pp of fixed-32 SC.
        """
        tau_candidates = [0.5, 0.625, 0.75, 0.875, 1.0]
        gamma_candidates = [0.0, 0.1, 0.25, 0.5]

        # 1. Compute fixed-32 SC baseline on Dev set
        fixed_32_accs = []
        fixed_32_costs = []
        for sid in dev_sample_ids:
            for order in rollout_orders[sid]:
                res = self.simulate_question_order(sid, order, "agreement", tau=1.01)  # Never stops early
                fixed_32_accs.append(res["is_correct"])
                fixed_32_costs.append(res["token_cost"])

        dev_fixed_32_acc = float(np.mean(fixed_32_accs)) * 100.0
        dev_fixed_32_cost = float(np.mean(fixed_32_costs))
        threshold_acc = dev_fixed_32_acc - 1.0

        print(f"[Adaptive Dev Grid Search] Fixed-32 SC Dev Acc: {dev_fixed_32_acc:.2f}%, Cost: {dev_fixed_32_cost:.1f} tokens.")
        print(f"[Adaptive Dev Grid Search] Target Min Acc: {threshold_acc:.2f}%\n")

        # 2. Sweep Agreement Controller
        agreement_records = []
        for tau in tau_candidates:
            acc_list = []
            cost_list = []
            stops_list = []
            for sid in dev_sample_ids:
                for order in rollout_orders[sid]:
                    res = self.simulate_question_order(sid, order, "agreement", tau=tau)
                    acc_list.append(res["is_correct"])
                    cost_list.append(res["token_cost"])
                    stops_list.append(res["stop_n"])

            mean_acc = float(np.mean(acc_list)) * 100.0
            mean_cost = float(np.mean(cost_list))
            mean_stops = float(np.mean(stops_list))
            agreement_records.append({
                "tau": tau,
                "accuracy_percent": round(mean_acc, 2),
                "mean_token_cost": round(mean_cost, 1),
                "mean_rollouts": round(mean_stops, 2),
                "eligible": mean_acc >= threshold_acc
            })

        # Select best agreement setting: lowest token cost among eligible
        eligible_agree = [r for r in agreement_records if r["eligible"]]
        if eligible_agree:
            best_agree = min(eligible_agree, key=lambda r: (r["mean_token_cost"], -r["accuracy_percent"], r["tau"]))
        else:
            best_agree = max(agreement_records, key=lambda r: r["accuracy_percent"])

        # 3. Sweep Quality-Aware Controller
        qa_records = []
        for tau in tau_candidates:
            for gamma in gamma_candidates:
                acc_list = []
                cost_list = []
                stops_list = []
                for sid in dev_sample_ids:
                    for order in rollout_orders[sid]:
                        res = self.simulate_question_order(sid, order, "quality_aware", tau=tau, gamma=gamma)
                        acc_list.append(res["is_correct"])
                        cost_list.append(res["token_cost"])
                        stops_list.append(res["stop_n"])

                mean_acc = float(np.mean(acc_list)) * 100.0
                mean_cost = float(np.mean(cost_list))
                mean_stops = float(np.mean(stops_list))
                qa_records.append({
                    "tau": tau,
                    "gamma": gamma,
                    "accuracy_percent": round(mean_acc, 2),
                    "mean_token_cost": round(mean_cost, 1),
                    "mean_rollouts": round(mean_stops, 2),
                    "eligible": mean_acc >= threshold_acc
                })

        eligible_qa = [r for r in qa_records if r["eligible"]]
        if eligible_qa:
            best_qa = min(eligible_qa, key=lambda r: (r["mean_token_cost"], -r["accuracy_percent"], r["tau"], r["gamma"]))
        else:
            best_qa = max(qa_records, key=lambda r: r["accuracy_percent"])

        print(f"Selected Agreement Setting: tau={best_agree['tau']} (Acc: {best_agree['accuracy_percent']}%, Cost: {best_agree['mean_token_cost']})")
        print(f"Selected Quality-Aware Setting: tau={best_qa['tau']}, gamma={best_qa['gamma']} (Acc: {best_qa['accuracy_percent']}%, Cost: {best_qa['mean_token_cost']})\n")

        return {
            "dev_fixed_32_sc": {
                "accuracy_percent": round(dev_fixed_32_acc, 2),
                "mean_token_cost": round(dev_fixed_32_cost, 1)
            },
            "best_agreement": best_agree,
            "best_quality_aware": best_qa,
            "all_agreement_records": agreement_records,
            "all_quality_aware_records": qa_records
        }

    def evaluate_held_out(
        self,
        eval_sample_ids: List[str],
        rollout_orders: Dict[str, List[List[int]]],
        best_agreement_setting: Dict[str, Any],
        best_quality_aware_setting: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluates frozen selected controllers on the held-out evaluation set (800 questions).
        """
        tau_agree = best_agreement_setting["tau"]
        tau_qa = best_quality_aware_setting["tau"]
        gamma_qa = best_quality_aware_setting["gamma"]

        policies = {
            "Fixed-32 SC": ("agreement", 1.01, 0.0),
            "Agreement Controller": ("agreement", tau_agree, 0.0),
            "Quality-Aware Controller": ("quality_aware", tau_qa, gamma_qa)
        }

        eval_summary = {}

        for pol_name, (pol_type, tau_val, gamma_val) in policies.items():
            acc_by_q = []
            cost_by_q = []
            rollouts_by_q = []
            stopping_counts = {4: 0, 8: 0, 16: 0, 32: 0}
            premature_stops = 0  # stopped early incorrectly, but fixed-32 was correct
            total_evals = 0

            for sid in eval_sample_ids:
                q_accs = []
                q_costs = []
                q_rolls = []
                for order in rollout_orders[sid]:
                    res = self.simulate_question_order(sid, order, pol_type, tau=tau_val, gamma=gamma_val)
                    q_accs.append(res["is_correct"])
                    q_costs.append(res["token_cost"])
                    q_rolls.append(res["stop_n"])

                    stopping_counts[res["stop_n"]] += 1
                    total_evals += 1
                    if res["stop_n"] < 32 and res["is_correct"] == 0.0 and res["fixed_32_correct"] == 1.0:
                        premature_stops += 1

                acc_by_q.append(float(np.mean(q_accs)))
                cost_by_q.append(float(np.mean(q_costs)))
                rollouts_by_q.append(float(np.mean(q_rolls)))

            acc_mean = float(np.mean(acc_by_q)) * 100.0
            cost_mean = float(np.mean(cost_by_q))
            rolls_mean = float(np.mean(rollouts_by_q))
            stop_dist = {n: round((cnt / float(total_evals)) * 100.0, 2) for n, cnt in stopping_counts.items()}

            eval_summary[pol_name] = {
                "accuracy_percent": round(acc_mean, 2),
                "mean_token_cost": round(cost_mean, 1),
                "mean_rollouts": round(rolls_mean, 2),
                "stopping_distribution_percent": stop_dist,
                "premature_stopping_rate_percent": round((premature_stops / float(max(1, total_evals))) * 100.0, 2)
            }

        return eval_summary
