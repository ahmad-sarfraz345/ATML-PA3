from __future__ import annotations
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from common.utils import set_seed, save_json
from common.data_utils import normalize_text_for_contamination

class MetaMathDataset(Dataset):
    """
    Supervised Fine-Tuning dataset for MetaMathQA filtered according to assignment rules:
    - 40,000 examples in exact source order (no shuffle).
    - Excludes empty questions/responses.
    - Excludes formatted length > 2,048 tokens.
    - Excludes query or original_question matching GSM8K test questions (text normalized).
    - Supervised labels strictly for assistant response + end-of-turn token.
    """
    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def prepare_metamath_40k(
    tokenizer: PreTrainedTokenizerBase,
    cache_path: str = "data/metamath_40k_tokenized.pt",
    target_count: int = 40000,
    max_length: int = 2048
) -> List[Dict[str, Any]]:
    """
    Loads MetaMathQA, applies GSM8K contamination filtering and length constraints,
    and returns 40,000 processed examples. Caches to disk.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"Loading cached MetaMathQA 40k dataset from {cache_path}...")
        return torch.load(cache_path, weights_only=False)

    print("Fetching GSM8K test split for contamination filtering...")
    gsm8k_test = load_dataset("gsm8k", "main", split="test")
    gsm8k_norm_questions = set()
    for item in gsm8k_test:
        norm_q = normalize_text_for_contamination(item["question"])
        if norm_q:
            gsm8k_norm_questions.add(norm_q)
    print(f"Loaded {len(gsm8k_norm_questions)} unique normalized GSM8K test questions.")

    print("Streaming / loading MetaMathQA...")
    metamath = load_dataset("meta-math/MetaMathQA", split="train")

    processed_examples = []
    skipped_empty = 0
    skipped_contamination = 0
    skipped_length = 0

    print(f"Filtering MetaMathQA in source order to collect {target_count} examples...")
    for idx, item in enumerate(metamath):
        query = item.get("query", "").strip()
        response = item.get("response", "").strip()
        orig_q = item.get("original_question", "").strip() if "original_question" in item else ""

        # 1. Empty check
        if not query or not response:
            skipped_empty += 1
            continue

        # 2. Contamination check against GSM8K test
        norm_query = normalize_text_for_contamination(query)
        norm_orig = normalize_text_for_contamination(orig_q) if orig_q else ""
        if norm_query in gsm8k_norm_questions or (norm_orig and norm_orig in gsm8k_norm_questions):
            skipped_contamination += 1
            continue

        # 3. Format with chat template (non-thinking)
        messages_prompt = [{"role": "user", "content": query}]
        messages_full = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ]

        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                full_text = tokenizer.apply_chat_template(
                    messages_full, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
            except TypeError:
                # In case enable_thinking is not an argument in some versions
                prompt_text = tokenizer.apply_chat_template(
                    messages_prompt, tokenize=False, add_generation_prompt=True
                )
                full_text = tokenizer.apply_chat_template(
                    messages_full, tokenize=False, add_generation_prompt=False
                )
        else:
            prompt_text = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
            full_text = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        # 4. Length check (do not truncate, exclude if > max_length)
        if len(full_ids) > max_length:
            skipped_length += 1
            continue

        prompt_len = len(prompt_ids)
        # Verify prompt is prefix
        labels = list(full_ids)
        # Mask prompt tokens with -100
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        processed_examples.append({
            "input_ids": full_ids,
            "labels": labels,
            "query": query,
            "response": response
        })

        if len(processed_examples) >= target_count:
            break

        if len(processed_examples) % 5000 == 0 and len(processed_examples) > 0:
            print(f"Collected {len(processed_examples)} / {target_count} examples...")

    print(f"Filtering complete: Collected {len(processed_examples)} examples.")
    print(f"Skipped stats: Empty: {skipped_empty}, Contaminated: {skipped_contamination}, OverLength: {skipped_length}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(processed_examples, cache_path)
    print(f"Saved dataset cache to {cache_path}")
    return processed_examples

def collate_fn(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """
    Pads microbatch to longest sequence in the batch.
    Labels are padded with -100.
    """
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids_batch = []
    labels_batch = []
    attention_mask_batch = []

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len

        # Right padding
        inp = item["input_ids"] + [pad_token_id] * pad_len
        lab = item["labels"] + [-100] * pad_len
        att = [1] * seq_len + [0] * pad_len

        input_ids_batch.append(inp)
        labels_batch.append(lab)
        attention_mask_batch.append(att)

    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "labels": torch.tensor(labels_batch, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long)
    }

def get_dataloader(examples: List[Dict[str, Any]], pad_token_id: int, micro_batch_size: int = 2) -> DataLoader:
    """Returns a deterministic PyTorch DataLoader without shuffling."""
    dataset = MetaMathDataset(examples)
    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=False,  # Strict source order preserved
        collate_fn=lambda b: collate_fn(b, pad_token_id)
    )

def sample_activation_probe_set(
    seed: int = 6304,
    save_path: str = "outputs/task1/probe_set.json"
) -> Dict[str, Any]:
    """
    Samples 32 GSM8K test questions and 32 HellaSwag validation contexts
    using seed 6304 before inspecting model results.
    """
    save_path = Path(save_path)
    if save_path.exists():
        with open(save_path, "r", encoding="utf-8") as f:
            return json.load(f)

    set_seed(seed)
    print("Sampling 32 GSM8K test questions for activation probe...")
    gsm8k = load_dataset("gsm8k", "main", split="test")
    gsm8k_indices = sorted(list(range(len(gsm8k))))
    import random
    rng = random.Random(seed)
    selected_gsm8k_idx = rng.sample(gsm8k_indices, 32)
    gsm8k_samples = [
        {"id": idx, "question": gsm8k[idx]["question"], "answer": gsm8k[idx]["answer"]}
        for idx in selected_gsm8k_idx
    ]

    print("Sampling 32 HellaSwag validation contexts for activation probe...")
    hellaswag = load_dataset("hellaswag", split="validation")
    hellaswag_indices = sorted(list(range(len(hellaswag))))
    selected_hella_idx = rng.sample(hellaswag_indices, 32)
    hella_samples = [
        {
            "id": idx,
            "ctx": hellaswag[idx]["ctx"],
            "activity_label": hellaswag[idx].get("activity_label", ""),
            "endings": hellaswag[idx]["endings"],
            "label": hellaswag[idx]["label"]
        }
        for idx in selected_hella_idx
    ]

    probe_set = {
        "seed": seed,
        "gsm8k_math_probes": gsm8k_samples,
        "hellaswag_nonmath_probes": hella_samples
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(probe_set, save_path)
    print(f"Saved activation probe set to {save_path}")
    return probe_set

