from __future__ import annotations
import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
import numpy as np
import pandas as pd

try:
    from math_verify import parse as mv_parse, verify as mv_verify
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False

NO_ANSWER = "NO_ANSWER"

def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Finds the last literal \\boxed{ and scans balanced braces,
    preserving nested LaTeX and stripping surrounding whitespace.
    Returns None if absent, empty, or unclosed.
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                extracted = text[start:i].strip()
                return extracted if extracted else None
    return None

def normalize_math_answer(s: Optional[str]) -> str:
    """Normalizes MATH answers for string fallback comparison."""
    if s is None or s == NO_ANSWER:
        return NO_ANSWER
    s = s.strip()
    if not s:
        return NO_ANSWER
    # Strip enclosing dollars
    if s.startswith("$") and s.endswith("$") and len(s) > 1:
        s = s[1:-1].strip()
    # Strip dollar signs inside
    s = s.replace("\\$", "").replace("$", "").strip()
    # Normalize fractions \\frac{a}{b} -> a/b if simple
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = s.replace("\\,", "").replace(" ", "")
    return s.lower()

def check_math_equivalence(pred: Optional[str], gold: Optional[str]) -> bool:
    """
    Checks mathematical equivalence using math-verify with timeout disabled (avoids IPC crashes),
    falling back to normalized string equivalence.
    """
    if pred is None or pred == NO_ANSWER or not str(pred).strip():
        return False
    if gold is None or not str(gold).strip():
        return False

    pred_str = str(pred).strip()
    gold_str = str(gold).strip()

    # Fast path: normalized string match
    if normalize_math_answer(pred_str) == normalize_math_answer(gold_str):
        return True

    # Use math_verify if available
    if HAS_MATH_VERIFY:
        try:
            # math-verify parse accepts strings with or without LaTeX
            p_cand = mv_parse(pred_str, parsing_timeout=None)
            p_gold = mv_parse(gold_str, parsing_timeout=None)
            if p_cand and p_gold:
                return bool(mv_verify(p_cand, p_gold, timeout_seconds=None))
        except Exception:
            pass

    return False

def normalize_mathvista_answer(s: Optional[str]) -> str:
    """
    Normalizes MathVista answers:
    - Lowercase, strip punctuation and whitespace
    - Represent multiple-choice as option letters (e.g. 'A', 'B')
    """
    if s is None or s == NO_ANSWER or not str(s).strip():
        return NO_ANSWER
    s = str(s).strip().lower()
    # Strip brackets, parentheses
    s = re.sub(r"^[(\[]([a-z0-9])[)\]]$", r"\1", s)
    s = re.sub(r"\\boxed\{([^}]+)\}", r"\1", s)
    s = s.replace("$", "").replace("\\", "").strip()
    return s

def check_mathvista_equivalence(pred: Optional[str], gold: Optional[str]) -> bool:
    """Checks MathVista answer equivalence."""
    if pred is None or pred == NO_ANSWER or not str(pred).strip():
        return False
    if gold is None or not str(gold).strip():
        return False
    p = normalize_mathvista_answer(pred)
    g = normalize_mathvista_answer(gold)
    if p == NO_ANSWER:
        return False
    if p == g:
        return True
    try:
        # Numeric equality fallback
        if abs(float(p) - float(g)) < 1e-4:
            return True
    except (ValueError, TypeError):
        pass
    return False

class Task3Dataset:
    """
    Loads, joins, and prepares rollouts for MATH and MathVista datasets.
    Creates 200 Dev / 800 Held-Out Evaluation splits permuted with default_rng(6304).
    """
    def __init__(self, data_dir: str = "data/task3", seed: int = 6304):
        self.data_dir = Path(data_dir)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def load_math_data(self) -> pd.DataFrame:
        """Loads and joins MATH generations and probability metrics (32,000 rows)."""
        gen_path = self.data_dir / "qwen25-math" / "generations.parquet"
        prob_path = self.data_dir / "qwen25-math" / "probability_metrics.parquet"

        if not gen_path.exists() or not prob_path.exists():
            raise FileNotFoundError(f"Missing MATH parquet files in {self.data_dir / 'qwen25-math'}")

        df_gen = pd.read_parquet(gen_path)
        df_prob = pd.read_parquet(prob_path)

        # Join on (dataset, sample_id, rollout_id)
        df_merged = pd.merge(df_gen, df_prob, on=["dataset", "sample_id", "rollout_id"], how="inner")
        print(f"Loaded MATH dataset: {len(df_merged)} rollouts across {df_merged['sample_id'].nunique()} questions.")
        return df_merged

    def load_mathvista_data(self) -> pd.DataFrame:
        """Loads and joins MathVista generations, probability, and attention metrics (32,000 rows)."""
        gen_path = self.data_dir / "qwen25vl-mathvista" / "generations.parquet"
        prob_path = self.data_dir / "qwen25vl-mathvista" / "probability_metrics.parquet"
        attn_path = self.data_dir / "qwen25vl-mathvista" / "attention_metrics.parquet"

        if not gen_path.exists() or not prob_path.exists() or not attn_path.exists():
            raise FileNotFoundError(f"Missing MathVista parquet files in {self.data_dir / 'qwen25vl-mathvista'}")

        df_gen = pd.read_parquet(gen_path)
        df_prob = pd.read_parquet(prob_path)
        df_attn = pd.read_parquet(attn_path)

        df_merged = pd.merge(df_gen, df_prob, on=["dataset", "sample_id", "rollout_id"], how="inner")
        # attention_metrics has n_generated_text_tokens as per-token list; rename to avoid collision with scalar
        if "n_generated_text_tokens" in df_attn.columns:
            df_attn = df_attn.rename(columns={"n_generated_text_tokens": "attn_token_indices"})

        df_merged = pd.merge(df_merged, df_attn, on=["dataset", "sample_id", "rollout_id"], how="inner")
        print(f"Loaded MathVista dataset: {len(df_merged)} rollouts across {df_merged['sample_id'].nunique()} questions.")
        return df_merged

    def compute_dataset_splits(self, sample_ids: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Sorts 1,000 question IDs lexicographically and permutes with default_rng(6304):
        - Dev: first 200 questions
        - Held-out Evaluation: remaining 800 questions
        - Live subtask: first 50 questions of the held-out evaluation set
        """
        unique_ids = sorted(list(set(sample_ids)))
        if len(unique_ids) != 1000:
            print(f"Warning: expected 1,000 unique questions, found {len(unique_ids)}")

        permuted = self.rng.permutation(unique_ids).tolist()
        dev_ids = permuted[:200]
        eval_ids = permuted[200:]
        live_ids = eval_ids[:50]

        return dev_ids, eval_ids, live_ids

    def group_equivalent_answers(
        self,
        extracted_answers: List[Optional[str]],
        is_math: bool = True
    ) -> List[int]:
        """
        Groups candidate answers by equivalence without consulting gold answers.
        Returns list of group integer IDs (one per candidate).
        All NO_ANSWER or failed extractions map to a special NO_ANSWER group (id = -1).
        """
        check_eq_fn = check_math_equivalence if is_math else check_mathvista_equivalence
        group_ids = []
        canonical_groups: List[str] = []

        for ans in extracted_answers:
            if ans is None or not str(ans).strip() or ans == NO_ANSWER:
                group_ids.append(-1)
                continue

            ans_str = str(ans).strip()
            # Check existing groups
            matched_gid = None
            for gid, rep in enumerate(canonical_groups):
                if check_eq_fn(ans_str, rep):
                    matched_gid = gid
                    break

            if matched_gid is not None:
                group_ids.append(matched_gid)
            else:
                new_gid = len(canonical_groups)
                canonical_groups.append(ans_str)
                group_ids.append(new_gid)

        return group_ids
