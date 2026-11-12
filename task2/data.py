from __future__ import annotations
import os
import sys
import json
import math
import random
import zipfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from task2.eval import vqa_normalize_answer

COCO_TRAIN_IMG_URL = "http://images.cocodataset.org/train2014"
COCO_VAL_IMG_URL = "http://images.cocodataset.org/val2014"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

VQA_TRAIN_QUESTIONS_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip"
VQA_VAL_QUESTIONS_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
VQA_TRAIN_ANNOTATIONS_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip"
VQA_VAL_ANNOTATIONS_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
VQA_COMPLEMENTARY_PAIRS_TRAIN_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip"
VQA_COMPLEMENTARY_PAIRS_VAL_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip"

def download_file(url: str, dest_path: Path):
    """Downloads a file if it does not already exist or is empty."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return
    print(f"Downloading {url} to {dest_path}...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Downloaded {dest_path.name}")

def extract_zip(zip_path: Path, extract_dir: Path, marker_file: Optional[Path] = None):
    """Extracts a zip file if destination directory or marker file does not exist."""
    if marker_file is not None and marker_file.exists():
        print(f"Found existing {marker_file.name}, skipping extraction.")
        return
    print(f"Extracting {zip_path.name} to {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"Extracted {zip_path.name}")

def format_coco_image_filename(image_id: int, split: str = "train2014") -> str:
    """Formats 12-digit COCO image filename: e.g. COCO_train2014_000000012345.jpg"""
    return f"COCO_{split}_{image_id:012d}.jpg"

class Task2DataPreparer:
    """
    Handles downloading annotations, forming pairs, downloading images,
    and building curriculum datasets for Task 2.
    """
    def __init__(self, data_dir: str = "data/task2", seed: int = 6304):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.rng = random.Random(seed)

    def fetch_all_annotations(self):
        """Downloads and unpacks COCO 2014 and VQAv2 annotations."""
        coco_zip = self.data_dir / "annotations_trainval2014.zip"
        download_file(COCO_ANNOTATIONS_URL, coco_zip)
        extract_zip(coco_zip, self.data_dir, marker_file=self.data_dir / "annotations" / "captions_train2014.json")

        # VQAv2 train & val
        vqa_dir = self.data_dir / "vqa"
        for name, url, marker in [
            ("v2_Questions_Train_mscoco.zip", VQA_TRAIN_QUESTIONS_URL, "v2_OpenEnded_mscoco_train2014_questions.json"),
            ("v2_Questions_Val_mscoco.zip", VQA_VAL_QUESTIONS_URL, "v2_OpenEnded_mscoco_val2014_questions.json"),
            ("v2_Annotations_Train_mscoco.zip", VQA_TRAIN_ANNOTATIONS_URL, "v2_mscoco_train2014_annotations.json"),
            ("v2_Annotations_Val_mscoco.zip", VQA_VAL_ANNOTATIONS_URL, "v2_mscoco_val2014_annotations.json"),
            ("v2_Complementary_Pairs_Train_mscoco.zip", VQA_COMPLEMENTARY_PAIRS_TRAIN_URL, "v2_mscoco_train2014_complementary_pairs.json"),
            ("v2_Complementary_Pairs_Val_mscoco.zip", VQA_COMPLEMENTARY_PAIRS_VAL_URL, "v2_mscoco_val2014_complementary_pairs.json"),
        ]:
            zpath = self.data_dir / name
            download_file(url, zpath)
            extract_zip(zpath, vqa_dir, marker_file=vqa_dir / marker)

    def build_coco_captions_datasets(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Set[int], Set[int]]:
        """
        Component 1: COCO Captions
        - Training: 6,000 images x 3 captions = 18,000 examples (train2014)
        - Development: 96 images x 1 caption = 96 examples (val2014)
        """
        train_caps_file = self.data_dir / "annotations" / "captions_train2014.json"
        val_caps_file = self.data_dir / "annotations" / "captions_val2014.json"

        with open(train_caps_file, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open(val_caps_file, "r", encoding="utf-8") as f:
            val_data = json.load(f)

        # Group captions by image_id
        train_by_img: Dict[int, List[str]] = {}
        for ann in train_data["annotations"]:
            train_by_img.setdefault(ann["image_id"], []).append(ann["caption"].strip())

        val_by_img: Dict[int, List[str]] = {}
        for ann in val_data["annotations"]:
            val_by_img.setdefault(ann["image_id"], []).append(ann["caption"].strip())

        # Select 6,000 train images having at least 3 captions
        eligible_train_imgs = sorted([img_id for img_id, caps in train_by_img.items() if len(caps) >= 3])
        self.rng.shuffle(eligible_train_imgs)
        selected_train_imgs = set(eligible_train_imgs[:6000])

        train_captions_examples = []
        for img_id in sorted(selected_train_imgs):
            caps = train_by_img[img_id][:3]
            for c in caps:
                train_captions_examples.append({
                    "type": "caption",
                    "image_id": img_id,
                    "split": "train2014",
                    "question": "Provide a descriptive caption for this image.",
                    "answer": c
                })

        # Select 96 dev images having at least 1 caption
        eligible_val_imgs = sorted([img_id for img_id, caps in val_by_img.items() if len(caps) >= 1])
        self.rng.shuffle(eligible_val_imgs)
        selected_dev_imgs = set(eligible_val_imgs[:96])

        dev_captions_examples = []
        for img_id in sorted(selected_dev_imgs):
            dev_captions_examples.append({
                "type": "caption",
                "image_id": img_id,
                "split": "val2014",
                "question": "Provide a descriptive caption for this image.",
                "answer": val_by_img[img_id][0]
            })

        print(f"COCO Captions ready: {len(train_captions_examples)} train examples, {len(dev_captions_examples)} dev examples.")
        return train_captions_examples, dev_captions_examples, selected_train_imgs, selected_dev_imgs

    def build_grounding_datasets(
        self,
        reserved_train_imgs: Set[int],
        reserved_val_imgs: Set[int]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Set[int], Set[int]]:
        """
        Component 2: Annotation-Derived Grounding (Presence & Counting)
        - Programmatic template questions from COCO object annotations
        - Presence: 'Does this image contain [object]?' (yes / no)
        - Counting: 'How many [objects] can you see?' (1 to 4)
        - Dev: 24 pairs / 48 questions (val2014)
        - Final test: 64 pairs / 128 questions (val2014)
        - Train: reusable pools (train2014)
        """
        train_inst_file = self.data_dir / "annotations" / "instances_train2014.json"
        val_inst_file = self.data_dir / "annotations" / "instances_val2014.json"

        with open(train_inst_file, "r", encoding="utf-8") as f:
            train_inst = json.load(f)
        with open(val_inst_file, "r", encoding="utf-8") as f:
            val_inst = json.load(f)

        cat_id_to_name = {c["id"]: c["name"] for c in train_inst["categories"]}

        def index_instances(inst_data, excluded_imgs):
            # img_id -> cat_id -> count
            counts_by_img: Dict[int, Dict[int, int]] = {}
            for ann in inst_data["annotations"]:
                img_id = ann["image_id"]
                if img_id in excluded_imgs:
                    continue
                cat_id = ann["category_id"]
                counts_by_img.setdefault(img_id, {})
                counts_by_img[img_id][cat_id] = counts_by_img[img_id].get(cat_id, 0) + 1
            return counts_by_img

        val_counts = index_instances(val_inst, reserved_val_imgs)
        val_imgs = sorted(list(val_counts.keys()))
        self.rng.shuffle(val_imgs)

        # Form Grounding Pairs from Val split (24 dev pairs, 64 test pairs = 88 total pairs)
        grounding_pairs_val = []
        used_val_imgs = set()

        all_cat_ids = sorted(list(cat_id_to_name.keys()))
        self.rng.shuffle(all_cat_ids)

        for cat_id in all_cat_ids:
            cat_name = cat_id_to_name[cat_id]
            # Presence pairs
            pos_imgs = [img for img in val_imgs if img not in used_val_imgs and val_counts[img].get(cat_id, 0) >= 1]
            neg_imgs = [img for img in val_imgs if img not in used_val_imgs and val_counts[img].get(cat_id, 0) == 0]

            for i in range(min(len(pos_imgs), len(neg_imgs))):
                img_pos, img_neg = pos_imgs[i], neg_imgs[i]
                used_val_imgs.add(img_pos)
                used_val_imgs.add(img_neg)
                grounding_pairs_val.append({
                    "task": "presence",
                    "category": cat_name,
                    "question": f"Does this image contain a {cat_name}?",
                    "image_1": img_pos, "answer_1": "yes",
                    "image_2": img_neg, "answer_2": "no",
                    "split": "val2014"
                })
                if len(grounding_pairs_val) >= 44:
                    break
            if len(grounding_pairs_val) >= 44:
                break

        # Counting pairs
        for cat_id in all_cat_ids:
            cat_name = cat_id_to_name[cat_id]
            by_count = {c: [] for c in range(1, 5)}
            for img in val_imgs:
                if img not in used_val_imgs:
                    cnt = val_counts[img].get(cat_id, 0)
                    if 1 <= cnt <= 4:
                        by_count[cnt].append(img)

            # Pair images with different counts
            for c1 in range(1, 4):
                c2 = c1 + 1
                for i in range(min(len(by_count[c1]), len(by_count[c2]))):
                    i1, i2 = by_count[c1][i], by_count[c2][i]
                    used_val_imgs.add(i1)
                    used_val_imgs.add(i2)
                    grounding_pairs_val.append({
                        "task": "counting",
                        "category": cat_name,
                        "question": f"How many {cat_name}s can you see?",
                        "image_1": i1, "answer_1": str(c1),
                        "image_2": i2, "answer_2": str(c2),
                        "split": "val2014"
                    })
                    if len(grounding_pairs_val) >= 88:
                        break
                if len(grounding_pairs_val) >= 88:
                    break
            if len(grounding_pairs_val) >= 88:
                break

        self.rng.shuffle(grounding_pairs_val)
        grounding_dev_pairs = grounding_pairs_val[:24]
        grounding_test_pairs = grounding_pairs_val[24:88]

        # Form reusable training pool from train split
        train_counts = index_instances(train_inst, reserved_train_imgs)
        train_imgs = sorted(list(train_counts.keys()))
        self.rng.shuffle(train_imgs)

        grounding_train_pool = []
        used_train_imgs = set()

        for cat_id in all_cat_ids:
            cat_name = cat_id_to_name[cat_id]
            pos_imgs = [img for img in train_imgs if train_counts.get(img, {}).get(cat_id, 0) >= 1]
            neg_imgs = [img for img in train_imgs if train_counts.get(img, {}).get(cat_id, 0) == 0]
            for i in range(min(50, len(pos_imgs), len(neg_imgs))):
                grounding_train_pool.append({
                    "task": "presence",
                    "category": cat_name,
                    "question": f"Does this image contain a {cat_name}?",
                    "image_1": pos_imgs[i], "answer_1": "yes",
                    "image_2": neg_imgs[i], "answer_2": "no",
                    "split": "train2014"
                })
                used_train_imgs.add(pos_imgs[i])
                used_train_imgs.add(neg_imgs[i])

            # Counting
            c1_imgs = [img for img in train_imgs if train_counts.get(img, {}).get(cat_id, 0) == 1]
            c2_imgs = [img for img in train_imgs if train_counts.get(img, {}).get(cat_id, 0) == 2]
            for i in range(min(30, len(c1_imgs), len(c2_imgs))):
                grounding_train_pool.append({
                    "task": "counting",
                    "category": cat_name,
                    "question": f"How many {cat_name}s can you see?",
                    "image_1": c1_imgs[i], "answer_1": "1",
                    "image_2": c2_imgs[i], "answer_2": "2",
                    "split": "train2014"
                })
                used_train_imgs.add(c1_imgs[i])
                used_train_imgs.add(c2_imgs[i])

        print(f"Grounding sets ready: {len(grounding_train_pool)} train pool pairs, {len(grounding_dev_pairs)} dev pairs, {len(grounding_test_pairs)} test pairs.")
        return grounding_train_pool, grounding_dev_pairs, grounding_test_pairs, used_train_imgs, used_val_imgs

    def build_natural_vqav2_datasets(
        self,
        reserved_train_imgs: Set[int],
        reserved_val_imgs: Set[int]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Set[int], Set[int]]:
        """
        Component 3: Natural VQAv2 Complementary Pairs
        - Keep pairs whose normalized consensus answers differ and have agreement >= 8/10 annotators
        - Training: 1,536 pairs (640 yes/no, 256 number, 640 other)
        - Development: 32 pairs (12 yes/no, 8 number, 12 other)
        - Final test: 96 pairs (36 yes/no, 24 number, 36 other)
        """
        vqa_dir = self.data_dir / "vqa"
        train_pairs_path = vqa_dir / "v2_mscoco_train2014_complementary_pairs.json"
        val_pairs_path = vqa_dir / "v2_mscoco_val2014_complementary_pairs.json"

        # Load question and annotation mappings
        def load_q_and_a(split):
            q_file = vqa_dir / f"v2_OpenEnded_mscoco_{split}_questions.json"
            a_file = vqa_dir / f"v2_mscoco_{split}_annotations.json"
            with open(q_file, "r", encoding="utf-8") as f:
                q_list = json.load(f)["questions"]
            with open(a_file, "r", encoding="utf-8") as f:
                a_list = json.load(f)["annotations"]

            q_map = {item["question_id"]: item for item in q_list}
            a_map = {item["question_id"]: item for item in a_list}
            return q_map, a_map

        train_q, train_a = load_q_and_a("train2014")
        val_q, val_a = load_q_and_a("val2014")

        with open(train_pairs_path, "r", encoding="utf-8") as f:
            raw_train_pairs = json.load(f)
        with open(val_pairs_path, "r", encoding="utf-8") as f:
            raw_val_pairs = json.load(f)

        def filter_consensus_pairs(raw_pairs, q_map, a_map, excluded_imgs, split_name):
            valid_by_type = {"yes/no": [], "number": [], "other": []}
            for q1_id, q2_id in raw_pairs:
                if q1_id not in a_map or q2_id not in a_map:
                    continue
                ann1, ann2 = a_map[q1_id], a_map[q2_id]
                img1, img2 = ann1["image_id"], ann2["image_id"]
                if img1 in excluded_imgs or img2 in excluded_imgs or img1 == img2:
                    continue

                # Check consensus agreement (>= 8 of 10)
                raw_ans1 = [ans["answer"] for ans in ann1["answers"]]
                raw_ans2 = [ans["answer"] for ans in ann2["answers"]]
                norm_ans1 = [vqa_normalize_answer(a) for a in raw_ans1]
                norm_ans2 = [vqa_normalize_answer(a) for a in raw_ans2]

                ans1_canonical = ann1["multiple_choice_answer"]
                ans2_canonical = ann2["multiple_choice_answer"]
                norm_canon1 = vqa_normalize_answer(ans1_canonical)
                norm_canon2 = vqa_normalize_answer(ans2_canonical)

                if norm_canon1 == norm_canon2:
                    continue  # Must differ

                agree1 = sum(1 for a in norm_ans1 if a == norm_canon1)
                agree2 = sum(1 for a in norm_ans2 if a == norm_canon2)
                if agree1 < 8 or agree2 < 8:
                    continue

                atype = ann1.get("answer_type", "other")
                if atype not in valid_by_type:
                    atype = "other"

                q_text = q_map[q1_id]["question"]
                pair_record = {
                    "question_id_1": q1_id,
                    "question_id_2": q2_id,
                    "image_1": img1,
                    "image_2": img2,
                    "question": q_text,
                    "answer_type": atype,
                    "canonical_answer_1": ans1_canonical,
                    "canonical_answer_2": ans2_canonical,
                    "all_answers_1": raw_ans1,
                    "all_answers_2": raw_ans2,
                    "split": split_name
                }
                valid_by_type[atype].append(pair_record)
            return valid_by_type

        # Filter val pairs for dev and test
        val_valid = filter_consensus_pairs(raw_val_pairs, val_q, val_a, reserved_val_imgs, "val2014")

        # Quotas: Dev: 12 yes/no, 8 number, 12 other (32 pairs)
        # Test: 36 yes/no, 24 number, 36 other (96 pairs)
        vqa_dev_pairs = []
        vqa_test_pairs = []
        used_val_imgs = set()

        for atype, need_dev, need_test in [("yes/no", 12, 36), ("number", 8, 24), ("other", 12, 36)]:
            pairs = val_valid[atype]
            self.rng.shuffle(pairs)
            dev_slice = pairs[:need_dev]
            test_slice = pairs[need_dev:need_dev + need_test]
            vqa_dev_pairs.extend(dev_slice)
            vqa_test_pairs.extend(test_slice)
            for p in dev_slice + test_slice:
                used_val_imgs.add(p["image_1"])
                used_val_imgs.add(p["image_2"])

        # Filter train pairs (640 yes/no, 256 number, 640 other = 1,536 pairs)
        train_valid = filter_consensus_pairs(raw_train_pairs, train_q, train_a, reserved_train_imgs, "train2014")
        vqa_train_pairs = []
        used_train_imgs = set()

        for atype, need_train in [("yes/no", 640), ("number", 256), ("other", 640)]:
            pairs = train_valid[atype]
            self.rng.shuffle(pairs)
            train_slice = pairs[:need_train]
            vqa_train_pairs.extend(train_slice)
            for p in train_slice:
                used_train_imgs.add(p["image_1"])
                used_train_imgs.add(p["image_2"])

        self.rng.shuffle(vqa_dev_pairs)
        self.rng.shuffle(vqa_test_pairs)
        self.rng.shuffle(vqa_train_pairs)

        print(f"Natural VQAv2 ready: {len(vqa_train_pairs)} train pairs, {len(vqa_dev_pairs)} dev pairs, {len(vqa_test_pairs)} test pairs.")
        return vqa_train_pairs, vqa_dev_pairs, vqa_test_pairs, used_train_imgs, used_val_imgs

    def download_required_images(self, all_image_specs: List[Tuple[int, str]], max_workers: int = 16):
        """
        Downloads only the selected images concurrently from the official COCO CDN.
        all_image_specs: list of (image_id, split) e.g. (12345, "train2014")
        """
        img_cache_dir = self.data_dir / "images"
        img_cache_dir.mkdir(parents=True, exist_ok=True)

        to_download = []
        for img_id, split in all_image_specs:
            fname = format_coco_image_filename(img_id, split)
            dest = img_cache_dir / fname
            if not dest.exists() or dest.stat().st_size == 0:
                url = f"{COCO_TRAIN_IMG_URL if split == 'train2014' else COCO_VAL_IMG_URL}/{fname}"
                to_download.append((url, dest))

        print(f"Checking images: {len(to_download)} need downloading out of {len(all_image_specs)} requested.")
        if not to_download:
            return

        def fetch(item):
            u, d = item
            try:
                urllib.request.urlretrieve(u, d)
            except Exception as e:
                print(f"Warning: Failed downloading {u}: {e}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(fetch, to_download)
        print("Image download complete.")

    def precompute_siglip_features(self, all_image_specs: List[Tuple[int, str]], batch_size: int = 64) -> Path:
        """
        Offline Feature Caching:
        Extracts 196 patch tokens (14x14) in FP16 with SigLIP vision encoder.
        Saves cached_features.pt and unloads SigLIP from GPU.
        """
        cache_file = self.data_dir / "cached_patch_features.pt"
        if cache_file.exists():
            print(f"Found cached patch features at {cache_file}.")
            return cache_file

        from task2.models import SigLIPVisionEncoder

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = SigLIPVisionEncoder(device=device)

        transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        img_cache_dir = self.data_dir / "images"
        features_dict: Dict[int, torch.Tensor] = {}

        unique_specs = sorted(list(set(all_image_specs)))
        print(f"Precomputing SigLIP patch features for {len(unique_specs)} unique images...")

        for i in range(0, len(unique_specs), batch_size):
            batch_specs = unique_specs[i:i + batch_size]
            tensors = []
            valid_ids = []
            for img_id, split in batch_specs:
                fpath = img_cache_dir / format_coco_image_filename(img_id, split)
                if fpath.exists():
                    try:
                        with Image.open(fpath) as img:
                            img_rgb = img.convert("RGB")
                            tensors.append(transform(img_rgb))
                            valid_ids.append(img_id)
                    except Exception:
                        tensors.append(torch.zeros(3, 224, 224))
                        valid_ids.append(img_id)
                else:
                    tensors.append(torch.zeros(3, 224, 224))
                    valid_ids.append(img_id)

            batch_tensor = torch.stack(tensors).to(device)
            feats = encoder.extract_patch_features(batch_tensor).cpu()  # (B, 196, 768) in FP16
            for v_id, feat in zip(valid_ids, feats):
                features_dict[v_id] = feat.half()

        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        torch.save(features_dict, cache_file)
        print(f"Saved {len(features_dict)} image features to {cache_file}.")
        return cache_file

class CurriculumBatchSampler:
    """
    Curriculum Sampler producing 16 examples (8 draws x 2 examples) per update step:
    - Stage 1 (1 - 2250 updates): 2 passes over 18k captions
    - Stage 2 (2251 - 3750 updates): 50% presence pair, 25% counting pair, 25% caption replay
    - Stage 3 (3751 - 4500 updates): 70% natural VQAv2 pair, 10% presence, 10% counting, 10% caption replay
    """
    def __init__(
        self,
        captions_train: List[Dict[str, Any]],
        grounding_train_pool: List[Dict[str, Any]],
        vqa_train_pairs: List[Dict[str, Any]],
        tokenizer,
        cached_features: Dict[int, torch.Tensor],
        seed: int = 6304
    ):
        self.captions_train = captions_train
        self.presence_pool = [p for p in grounding_train_pool if p["task"] == "presence"]
        self.counting_pool = [p for p in grounding_train_pool if p["task"] == "counting"]
        self.vqa_train_pairs = vqa_train_pairs
        self.tokenizer = tokenizer
        self.cached_features = cached_features
        self.rng = random.Random(seed)

        # Stage 1 caption order (2 complete passes)
        self.stage1_indices = list(range(len(captions_train))) * 2
        self.rng.shuffle(self.stage1_indices)
        self.stage1_ptr = 0

    def format_example(self, image_id: Optional[int], question: str, answer: str, max_q_len: int = 80, max_a_len: int = 64) -> Dict[str, Any]:
        """
        Formats a single multimodal example:
        Prefix: <|im_start|>user\n
        Visual tokens inserted in connector
        Suffix: {question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n
        """
        prefix_str = "<|im_start|>user\n"
        prefix_ids = self.tokenizer.encode(prefix_str, add_special_tokens=False)

        q_clean = question.strip()
        a_clean = answer.strip()

        prompt_str = f"{q_clean}<|im_end|>\n<|im_start|>assistant\n"
        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)[:max_q_len]

        target_str = f"{a_clean}<|im_end|>\n"
        target_ids = self.tokenizer.encode(target_str, add_special_tokens=False)[:max_a_len]

        suffix_ids = prompt_ids + target_ids
        # Labels: -100 on prompt, actual token IDs on target
        labels = [-100] * len(prompt_ids) + target_ids

        vis_feat = self.cached_features.get(image_id, None) if image_id is not None else None

        return {
            "image_id": image_id,
            "prefix_ids": prefix_ids,
            "suffix_ids": suffix_ids,
            "labels": labels,
            "visual_feature": vis_feat
        }

    def sample_update_batch(self, update_step: int) -> List[Dict[str, Any]]:
        """Samples 16 examples (effective batch size 16) for a given update step."""
        batch_examples = []

        if update_step <= 2250:
            # Stage 1: Caption alignment (16 caption examples per update)
            for _ in range(16):
                idx = self.stage1_indices[self.stage1_ptr % len(self.stage1_indices)]
                self.stage1_ptr += 1
                c_item = self.captions_train[idx]
                ex = self.format_example(c_item["image_id"], c_item["question"], c_item["answer"], max_q_len=80, max_a_len=64)
                batch_examples.append(ex)

        elif update_step <= 3750:
            # Stage 2: Visual instruction tuning (8 draws x 2 examples = 16 examples)
            # 50% presence pair, 25% counting pair, 25% caption replay pair
            for _ in range(8):
                r = self.rng.random()
                if r < 0.50 and self.presence_pool:
                    pair = self.rng.choice(self.presence_pool)
                    ex1 = self.format_example(pair["image_1"], pair["question"], pair["answer_1"], max_q_len=80, max_a_len=16)
                    ex2 = self.format_example(pair["image_2"], pair["question"], pair["answer_2"], max_q_len=80, max_a_len=16)
                    batch_examples.extend([ex1, ex2])
                elif r < 0.75 and self.counting_pool:
                    pair = self.rng.choice(self.counting_pool)
                    ex1 = self.format_example(pair["image_1"], pair["question"], pair["answer_1"], max_q_len=80, max_a_len=16)
                    ex2 = self.format_example(pair["image_2"], pair["question"], pair["answer_2"], max_q_len=80, max_a_len=16)
                    batch_examples.extend([ex1, ex2])
                else:
                    c1 = self.rng.choice(self.captions_train)
                    c2 = self.rng.choice(self.captions_train)
                    ex1 = self.format_example(c1["image_id"], c1["question"], c1["answer"], max_q_len=80, max_a_len=64)
                    ex2 = self.format_example(c2["image_id"], c2["question"], c2["answer"], max_q_len=80, max_a_len=64)
                    batch_examples.extend([ex1, ex2])

        else:
            # Stage 3: Natural VQA specialization (8 draws x 2 examples = 16 examples)
            # 70% natural VQAv2 pair, 10% presence, 10% counting, 10% caption replay
            for _ in range(8):
                r = self.rng.random()
                if r < 0.70 and self.vqa_train_pairs:
                    pair = self.rng.choice(self.vqa_train_pairs)
                    ex1 = self.format_example(pair["image_1"], pair["question"], pair["canonical_answer_1"], max_q_len=80, max_a_len=16)
                    ex2 = self.format_example(pair["image_2"], pair["question"], pair["canonical_answer_2"], max_q_len=80, max_a_len=16)
                    batch_examples.extend([ex1, ex2])
                elif r < 0.80 and self.presence_pool:
                    pair = self.rng.choice(self.presence_pool)
                    ex1 = self.format_example(pair["image_1"], pair["question"], pair["answer_1"], max_q_len=80, max_a_len=16)
                    ex2 = self.format_example(pair["image_2"], pair["question"], pair["answer_2"], max_q_len=80, max_a_len=16)
                    batch_examples.extend([ex1, ex2])
                elif r < 0.90 and self.counting_pool:
                    pair = self.rng.choice(self.counting_pool)
                    ex1 = self.format_example(pair["image_1"], pair["question"], pair["answer_1"], max_q_len=80, max_a_len=16)
                    ex2 = self.format_example(pair["image_2"], pair["question"], pair["answer_2"], max_q_len=80, max_a_len=16)
                    batch_examples.extend([ex1, ex2])
                else:
                    c1 = self.rng.choice(self.captions_train)
                    c2 = self.rng.choice(self.captions_train)
                    ex1 = self.format_example(c1["image_id"], c1["question"], c1["answer"], max_q_len=80, max_a_len=64)
                    ex2 = self.format_example(c2["image_id"], c2["question"], c2["answer"], max_q_len=80, max_a_len=64)
                    batch_examples.extend([ex1, ex2])

        return batch_examples

def collate_task2_batch(items: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """
    Pads prefix_ids and suffix_ids to max length in microbatch.
    Labels are padded with -100.
    """
    max_pre = max(len(it["prefix_ids"]) for it in items)
    max_suf = max(len(it["suffix_ids"]) for it in items)

    batch_pre = []
    batch_suf = []
    batch_lab = []
    batch_mask = []
    has_visual = any(it["visual_feature"] is not None for it in items)
    vis_list = []

    for it in items:
        # Pad prefix
        pre_pad = max_pre - len(it["prefix_ids"])
        pre = it["prefix_ids"] + [pad_token_id] * pre_pad

        # Pad suffix
        suf_pad = max_suf - len(it["suffix_ids"])
        suf = it["suffix_ids"] + [pad_token_id] * suf_pad
        lab = it["labels"] + [-100] * suf_pad

        # Attention mask
        vis_count = 196 if has_visual else 0
        if has_visual:
            vis_mask = [1] * 196 if it["visual_feature"] is not None else [0] * 196
        else:
            vis_mask = []
        mask = [1] * len(it["prefix_ids"]) + vis_mask + [1] * len(it["suffix_ids"]) + [0] * suf_pad

        batch_pre.append(pre)
        batch_suf.append(suf)
        batch_lab.append(lab)
        batch_mask.append(mask)

        if has_visual:
            if it["visual_feature"] is not None:
                vis_list.append(it["visual_feature"])
            else:
                vis_list.append(torch.zeros(196, 768, dtype=torch.float16))

    res = {
        "prefix_ids": torch.tensor(batch_pre, dtype=torch.long),
        "suffix_ids": torch.tensor(batch_suf, dtype=torch.long),
        "labels": torch.tensor(batch_lab, dtype=torch.long),
        "attention_mask": torch.tensor(batch_mask, dtype=torch.long)
    }
    if has_visual:
        res["visual_features"] = torch.stack(vis_list)
    else:
        res["visual_features"] = None
    return res
