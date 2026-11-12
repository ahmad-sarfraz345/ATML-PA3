from __future__ import annotations
import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

def compute_rollout_metrics(
    logp_list: Any,
    kl_u_p_list: Any
) -> Tuple[float, float]:
    """
    Computes scalar rollout quality metrics:
    L_i = 1/T_i * sum_t logp_{i,t} (mean log-probability)
    K_i = 1/T_i * sum_t kl_u_p_{i,t} (mean self-certainty)
    """
    if logp_list is None or len(logp_list) == 0:
        return -1e9, 0.0
    T_i = len(logp_list)
    L_i = float(np.mean(logp_list))
    K_i = float(np.mean(kl_u_p_list)) if kl_u_p_list is not None and len(kl_u_p_list) > 0 else 0.0
    return L_i, K_i

def compute_visual_engagement(
    img_attn: Any,
    prompt_attn: Any,
    gen_attn: Any
) -> Tuple[np.ndarray, float]:
    """
    Computes per-token and mean visual engagement:
    v_{i,t} = A^img / (A^img + A^prompt + A^gen)
    V_i = 1/T_i * sum_t v_{i,t}
    """
    if img_attn is None or len(img_attn) == 0:
        return np.array([]), 0.0

    T_i = len(img_attn)
    a_img = np.asarray(img_attn, dtype=np.float64)
    a_prompt = np.asarray(prompt_attn, dtype=np.float64)
    a_gen = np.asarray(gen_attn, dtype=np.float64)

    denom = a_img + a_prompt + a_gen
    # Avoid zero division
    denom = np.where(denom <= 1e-12, 1e-12, denom)
    v_it = a_img / denom
    V_i = float(np.mean(v_it))
    return v_it, V_i

def calibrate_visconf_pool(
    kl_lists: List[List[float]],
    v_token_lists: List[np.ndarray]
) -> Tuple[List[float], List[float], List[float]]:
    """
    Calibrates token KL values across the current candidate set G (size N):
    c_{i,t} = clip_{[0, 1]}( (k_{i,t} - Q_{0.05}(G)) / max(Q_{0.95}(G) - Q_{0.05}(G), 10^-8) )
    C_i = 1/T_i * sum_t c_{i,t}
    Z_i = V_i * C_i  (Trajectory-level VisConf)
    Z^{token}_i = 1/T_i * sum_t c_{i,t} * v_{i,t} (Tokenwise fusion ablation)
    """
    # Pool all token KL values across candidate set G
    all_kl = []
    for kl in kl_lists:
        all_kl.extend(kl)

    if len(all_kl) == 0:
        n_cands = len(kl_lists)
        return [0.0] * n_cands, [0.0] * n_cands, [0.0] * n_cands

    all_kl_arr = np.asarray(all_kl, dtype=np.float64)
    q05 = float(np.percentile(all_kl_arr, 5))
    q95 = float(np.percentile(all_kl_arr, 95))
    scale = max(q95 - q05, 1e-8)

    visconf_scores = []
    tokenwise_scores = []
    calibrated_conf_means = []

    for i, (kl, v_it) in enumerate(zip(kl_lists, v_token_lists)):
        if len(kl) == 0:
            visconf_scores.append(0.0)
            tokenwise_scores.append(0.0)
            calibrated_conf_means.append(0.0)
            continue

        k_arr = np.asarray(kl, dtype=np.float64)
        c_it = np.clip((k_arr - q05) / scale, 0.0, 1.0)
        C_i = float(np.mean(c_it))
        V_i = float(np.mean(v_it)) if len(v_it) > 0 else 0.0

        # Trajectory-level VisConf
        Z_i = V_i * C_i
        visconf_scores.append(Z_i)

        # Tokenwise fusion ablation
        min_len = min(len(c_it), len(v_it))
        if min_len > 0:
            Z_token_i = float(np.mean(c_it[:min_len] * v_it[:min_len]))
        else:
            Z_token_i = 0.0
        tokenwise_scores.append(Z_token_i)
        calibrated_conf_means.append(C_i)

    return visconf_scores, tokenwise_scores, calibrated_conf_means

def compute_average_ranks(scores: List[float]) -> np.ndarray:
    """
    Computes descending average ranks (rank 1 is highest score).
    Handles ties by assigning average rank (e.g. tie for 1st and 2nd gets rank 1.5).
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    n = len(scores_arr)
    if n == 1:
        return np.array([1.0])

    # Sort descending
    order = np.argsort(-scores_arr)
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        while j < n and scores_arr[order[j]] == scores_arr[order[i]]:
            j += 1
        # Indices i to j-1 have equal score, ranks run from (i+1) to j
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    return ranks

def aggregate_rank_weighted_consensus(
    group_ids: List[int],
    scores: List[float]
) -> Tuple[int, int, Dict[int, float], float]:
    """
    Rank-Weighted Consensus (RWC):
    w_i = (N - r_i) / (N - 1) for N > 1, else 1.0
    W(a) = sum_{i: a_i = a} w_i
    Returns (winning_group_id, winning_candidate_idx, vote_weights_by_group, rwc_margin).
    Ties broken by earliest supporting rollout in the candidate order.
    """
    N = len(group_ids)
    if N == 1:
        return group_ids[0], 0, {group_ids[0]: 1.0}, 1.0

    ranks = compute_average_ranks(scores)
    weights = (N - ranks) / (N - 1.0)

    # Accumulate votes by answer group
    group_votes: Dict[int, float] = {}
    earliest_idx_by_group: Dict[int, int] = {}

    for idx, (gid, w) in enumerate(zip(group_ids, weights)):
        group_votes[gid] = group_votes.get(gid, 0.0) + float(w)
        if gid not in earliest_idx_by_group:
            earliest_idx_by_group[gid] = idx

    # Find winning group (break ties by earliest appearance)
    sorted_groups = sorted(
        group_votes.keys(),
        key=lambda g: (-group_votes[g], earliest_idx_by_group[g])
    )

    winning_gid = sorted_groups[0]
    winning_cand_idx = earliest_idx_by_group[winning_gid]

    # Calculate RWC margin = (W_1 - W_2) / total_weight
    total_w = sum(group_votes.values())
    if total_w > 0:
        w1 = group_votes[winning_gid]
        w2 = group_votes[sorted_groups[1]] if len(sorted_groups) > 1 else 0.0
        rwc_margin = (w1 - w2) / total_w
    else:
        rwc_margin = 0.0

    return winning_gid, winning_cand_idx, group_votes, rwc_margin

def aggregate_self_consistency(
    group_ids: List[int]
) -> Tuple[int, int, Dict[int, int]]:
    """
    Majority vote (Self-Consistency):
    Each rollout casts 1 vote.
    Ties broken by earliest candidate in candidate order.
    Returns (winning_group_id, winning_candidate_idx, vote_counts).
    """
    counts: Dict[int, int] = {}
    earliest: Dict[int, int] = {}
    for idx, gid in enumerate(group_ids):
        counts[gid] = counts.get(gid, 0) + 1
        if gid not in earliest:
            earliest[gid] = idx

    sorted_groups = sorted(
        counts.keys(),
        key=lambda g: (-counts[g], earliest[g])
    )
    win_gid = sorted_groups[0]
    return win_gid, earliest[win_gid], counts

def aggregate_highest_score(
    group_ids: List[int],
    scores: List[float]
) -> Tuple[int, int]:
    """
    Highest-score selection:
    Picks candidate with highest quality score. Ties broken by earliest candidate.
    Returns (winning_group_id, winning_candidate_idx).
    """
    best_idx = 0
    best_score = scores[0]
    for idx in range(1, len(scores)):
        if scores[idx] > best_score:
            best_score = scores[idx]
            best_idx = idx
    return group_ids[best_idx], best_idx
