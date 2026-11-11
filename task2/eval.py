from __future__ import annotations
import re
import math
import string
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn

# Standard VQA answer normalization
ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
PUNCT_TABLE = str.maketrans("", "", string.punctuation)
CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "didnt": "didn't",
    "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", "hadntve": "hadn't've",
    "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hedve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Id've": "I'd've", "I'll": "I'll", "Im": "I'm", "Ive": "I've",
    "isnt": "isn't", "itd": "it'd", "itdve": "it'd've", "itll": "it'll",
    "let's": "let's", "maam": "ma'am", "mightnt": "mightn't", "mightntve": "mightn't've",
    "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
    "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'s": "'s", "shant": "shan't", "shed": "she'd",
    "shedve": "she'd've", "shell": "she'll", "shes": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldntve": "shouldn't've", "somebodys": "somebody's",
    "someones": "someone's", "somethings": "something's", "thatll": "that'll",
    "thats": "that's", "thatd": "that'd", "thered": "there'd", "therere": "there're",
    "theres": "there's", "theyd": "they'd", "theydve": "they'd've", "theyll": "they'll",
    "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
    "wed": "we'd", "wedve": "we'd've", "we'll": "we'll", "were": "we're",
    "weve": "we've", "werent": "weren't", "whatd": "what'd", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd",
    "whodve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've",
    "whyd": "why'd", "whyre": "why're", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't", "wouldntve": "wouldn't've",
    "yall": "y'all", "yallll": "y'all'll", "yall'd've": "y'all'd've",
    "youd": "you'd", "youdve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've"
}
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "none": "0"
}

def vqa_normalize_answer(s: str) -> str:
    """
    Standard VQA evaluation answer normalization:
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Convert number words to digits
    - Collapse extra whitespace
    """
    if not isinstance(s, str):
        s = str(s)
    s = s.lower().strip()
    # Replace contractions
    for k, v in CONTRACTIONS.items():
        if v in s:
            s = s.replace(v, k)
    # Remove punctuation
    s = s.translate(PUNCT_TABLE)
    # Remove articles
    s = ARTICLES_REGEX.sub("", s)
    # Convert number words
    tokens = s.split()
    tokens = [NUMBER_WORDS.get(t, t) for t in tokens]
    return " ".join(tokens).strip()

def compute_vqa_consensus_score(prediction: str, ground_truth_answers: List[str]) -> float:
    """
    Official VQA soft consensus accuracy against 10 reference answers:
    s(a_hat) = 1/10 sum_{i=1}^{10} min(1.0, 1/3 sum_{j != i} 1[a_hat == a_j])
    """
    norm_pred = vqa_normalize_answer(prediction)
    norm_gt = [vqa_normalize_answer(a) for a in ground_truth_answers]
    n = len(norm_gt)
    if n == 0:
        return 0.0

    total_score = 0.0
    for i in range(n):
        match_count = sum(1 for j in range(n) if j != i and norm_gt[j] == norm_pred)
        score_i = min(1.0, match_count / 3.0)
        total_score += score_i
    return total_score / float(n)

def compute_grounding_exact_match(prediction: str, canonical_answer: str) -> float:
    """
    Normalized exact match for grounding presence/counting questions.
    Returns 1.0 if match else 0.0.
    """
    return 1.0 if vqa_normalize_answer(prediction) == vqa_normalize_answer(canonical_answer) else 0.0

def bootstrap_pair_metric_confidence_intervals(
    pairs_data: List[Dict[str, Any]],
    metric_fn,
    num_bootstrap: int = 4000,
    seed: int = 6304,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Computes 95% bootstrap confidence intervals over COMPLETE pairs:
    - Resamples complete pairs with replacement 4,000 times
    - Computes metric on each resample
    - Returns (mean, lower_ci, upper_ci)
    """
    n_pairs = len(pairs_data)
    if n_pairs == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    indices = np.arange(n_pairs)

    bootstrap_scores = np.empty(num_bootstrap, dtype=np.float64)
    for b in range(num_bootstrap):
        sampled_indices = rng.choice(indices, size=n_pairs, replace=True)
        sampled_pairs = [pairs_data[idx] for idx in sampled_indices]
        bootstrap_scores[b] = metric_fn(sampled_pairs)

    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    point_estimate = float(np.mean(bootstrap_scores))
    lower_ci = float(np.percentile(bootstrap_scores, lower_pct))
    upper_ci = float(np.percentile(bootstrap_scores, upper_pct))

    return point_estimate, lower_ci, upper_ci

def compute_teacher_forced_nll(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    visual_embeds: Optional[torch.Tensor] = None
) -> float:
    """
    Computes average per-example negative log likelihood (NLL) on target tokens.
    Excludes termination / padding tokens (-100).
    Averages token NLL within each example first, then averages across examples.
    """
    model.eval()
    with torch.no_grad():
        if visual_embeds is not None:
            # Inputs with visual embeddings prepended
            inputs_embeds = model.get_input_embeddings()(input_ids)
            # Insert visual embeds right after user header
            # (handled by caller or custom forward)
            logits = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_losses = token_losses.view(shift_labels.size())

        valid_mask = (shift_labels != -100)
        example_losses = []
        for i in range(shift_labels.size(0)):
            val_tokens = valid_mask[i].sum().item()
            if val_tokens > 0:
                ex_loss = (token_losses[i] * valid_mask[i]).sum().item() / val_tokens
                example_losses.append(ex_loss)

        if not example_losses:
            return 0.0
        return float(np.mean(example_losses))
