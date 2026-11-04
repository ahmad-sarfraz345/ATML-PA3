from __future__ import annotations
import os
import json
import yaml
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
from transformers import AutoTokenizer
from common.utils import get_device, save_json, load_json
from task1.eval import load_evaluation_model

QUALITATIVE_YAML = "configs/first_plot_questions.yaml"

def parse_qualitative_prompts(yaml_path: str = QUALITATIVE_YAML) -> List[Dict[str, Any]]:
    """
    Parses first_plot_questions.yaml into 24 distinct prompt specifications:
    8 questions across 3 formats: plain, json, template.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    prompts = []
    for item in data:
        qid = item["id"]
        # Determine format
        if qid.endswith("_json"):
            base_id = qid[:-5]
            fmt = "json"
        elif qid.endswith("_template"):
            base_id = qid[:-9]
            fmt = "template"
        else:
            base_id = qid
            fmt = "plain"

        system_prompt = item.get("system", None)
        user_prompt = item["paraphrases"][0].strip()

        prompts.append({
            "id": qid,
            "base_id": base_id,
            "format": fmt,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        })

    return prompts

def check_format_compliance(response_text: str, fmt: str) -> Dict[str, Any]:
    """
    Checks format compliance:
    - plain: always true (text)
    - json: valid JSON object
    - template: presence of expected template keys (e.g. key = ...)
    """
    res = response_text.strip()
    if fmt == "plain":
        return {"compliant": True, "details": "plain text"}
    elif fmt == "json":
        # Check if contains valid JSON object
        try:
            # Strip potential markdown fences
            clean = re.sub(r"^```(?:json)?\s*", "", res)
            clean = re.sub(r"\s*```$", "", clean)
            json_obj = json.loads(clean.strip())
            return {"compliant": isinstance(json_obj, dict), "details": "valid JSON"}
        except Exception:
            return {"compliant": False, "details": "invalid JSON"}
    elif fmt == "template":
        # Checks if key = value pattern exists
        has_assignments = bool(re.search(r"^\s*\w+\s*=", res, re.MULTILINE))
        return {"compliant": has_assignments, "details": "has template assignments" if has_assignments else "missing assignments"}
    return {"compliant": True, "details": "unknown format"}

def check_math_intrusion(response_text: str) -> Dict[str, Any]:
    """
    Checks if mathematical framing intrudes into general responses
    (e.g., equations, LaTeX, theorem-style statements, algebraic symbols).
    """
    patterns = [
        r"\\[a-zA-Z]+",            # LaTeX commands like \frac, \text, \boxed
        r"\$\$.*?\$\$|\$.*?\$",     # Inline or display math
        r"\b(?:let|assume|given)\s+[xyzabc]\b",
        r"\b(?:equation|proof|theorem|lemma|Q\.E\.D\.)\b",
        r"\b(?:calculate|step 1:|step 2:)\b"
    ]
    matches = []
    for pat in patterns:
        found = re.findall(pat, response_text, re.IGNORECASE)
        if found:
            matches.extend(found[:3])
    return {
        "has_math_intrusion": len(matches) > 0,
        "math_features": matches
    }

def run_qualitative_generation(
    base_model_path: str = "Qwen/Qwen3-0.6B",
    ckpt_dir: str = "outputs/task1",
    output_dir: str = "outputs/task1",
    max_new_tokens: int = 512
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generates greedy responses (512 new tokens cap, thinking disabled) for all 24 prompts
    across all 7 evaluation conditions (168 responses total).
    Annotates the first and last questions (42 responses total).
    """
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = parse_qualitative_prompts()
    conditions = [
        "baseline",
        "full_ft_midpoint",
        "full_ft_final",
        "full_ft_l2sp_final",
        "lora_4_final",
        "lora_16_final",
        "oplora_16_final"
    ]

    out_168_path = Path(output_dir) / "qualitative_168.json"
    records_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if out_168_path.exists():
        try:
            for r in load_json(out_168_path):
                records_by_key[(r["condition"], r["id"])] = r
        except Exception:
            records_by_key = {}

    for cond in conditions:
        print(f"\n--- Generating Qualitative Responses for [{cond}] ---")
        model = load_evaluation_model(cond, base_model_path, ckpt_dir, device)
        if model is None:
            print(f"Skipping qualitative generation for [{cond}]: checkpoint not found.", flush=True)
            continue

        for p in prompts:
            # Check if already generated
            if (cond, p["id"]) in records_by_key:
                continue

            # Build messages
            messages = []
            if p["system_prompt"]:
                messages.append({"role": "system", "content": p["system_prompt"]})
            messages.append({"role": "user", "content": p["user_prompt"]})

            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    formatted_input = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                except TypeError:
                    formatted_input = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
            else:
                formatted_input = f"<|im_start|>user\n{p['user_prompt']}<|im_end|>\n<|im_start|>assistant\n"

            inputs = tokenizer(formatted_input, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Strict greedy decoding
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            gen_tokens = output_ids[0, input_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            truncated = len(gen_tokens) >= max_new_tokens

            compliance = check_format_compliance(gen_text, p["format"])
            math_check = check_math_intrusion(gen_text)

            record = {
                "condition": cond,
                "id": p["id"],
                "base_id": p["base_id"],
                "format": p["format"],
                "user_prompt": p["user_prompt"],
                "system_prompt": p["system_prompt"],
                "response": gen_text,
                "generated_tokens": len(gen_tokens),
                "truncated": truncated,
                "format_compliance": compliance,
                "math_intrusion": math_check
            }
            records_by_key[(cond, p["id"])] = record

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save all responses
    all_responses = list(records_by_key.values())
    save_json(all_responses, out_168_path)
    print(f"Saved {len(all_responses)} qualitative responses to {out_168_path}")

    # Extract 42-response annotation subset (first and last questions in plain order)
    # First: what_is_your_wish, Last: enough_of_my_husband
    annotation_base_ids = ["what_is_your_wish", "enough_of_my_husband"]
    annotation_42 = [
        r for r in all_responses if r["base_id"] in annotation_base_ids
    ]

    out_42_path = Path(output_dir) / "qualitative_42_annotation.json"
    save_json(annotation_42, out_42_path)
    print(f"Saved {len(annotation_42)} annotated responses subset to {out_42_path}")

    return all_responses, annotation_42

