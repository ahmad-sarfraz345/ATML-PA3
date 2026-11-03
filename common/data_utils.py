from __future__ import annotations
import re
import string

def normalize_text_for_contamination(text: str) -> str:
    """
    Normalize text for contamination matching between MetaMathQA questions and GSM8K test questions.
    Lowercases, removes punctuation, normalizes numbers and whitespace.
    """
    if not text:
        return ""
    text = text.lower()
    # Replace newlines and tabs with spaces
    text = re.sub(r"\s+", " ", text)
    # Strip punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

def format_chat_prompt(tokenizer, question: str, response: str = None, enable_thinking: bool = False) -> str:
    """
    Format user question and optional assistant response using tokenizer's chat template.
    Forces non-thinking mode by disabling thinking where applicable.
    """
    messages = [
        {"role": "user", "content": question}
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking
        ) if hasattr(tokenizer, "apply_chat_template") else f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"
    else:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        ) if hasattr(tokenizer, "apply_chat_template") else f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

