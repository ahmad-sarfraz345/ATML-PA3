from __future__ import annotations
import os

# Mitigate CUDA memory fragmentation on limited VRAM GPUs (e.g. Kaggle T4)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import time
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

def set_seed(seed: int = 6304):
    """
    Fix all random seeds for deterministic reproduction.
    Required seed for ATML PA3 is 6304.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_device() -> torch.device:
    """Returns CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GPUMemoryTracker:
    """Tracks peak GPU memory usage in MB / GB."""
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.reset()

    def reset(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_peak_memory_mb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        return 0.0

    def get_peak_memory_gb(self) -> float:
        return self.get_peak_memory_mb() / 1024.0

class Timer:
    """Simple context manager and timer."""
    def __init__(self):
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start_time
        return self.elapsed

def save_json(data: Any, path: str | Path, indent: int = 2):
    """Save serializable data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)

def load_json(path: str | Path) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_jsonl(records: list[Dict[str, Any]], path: str | Path):
    """Save list of dicts to JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")

def load_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records

