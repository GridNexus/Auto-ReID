"""
hpt_dataset.py - Hierarchical Progressive Tuning dataset construction.

Implements training data for both stages of HPT:

Stage 1: Fine-Grained Attribute Alignment
    - Single-image instruction-following: generate structured description
    - Fixed prompt P_struct elicits comprehensive attribute-oriented analysis
    - Dataset: Market-1501, MSMT17 training images

Stage 2: Identity Verification and Feedback Generation
    Seven auxiliary tasks:
    T1. Attribute Annotations Matching
    T2. Attribute Difference Mining (two images)
    T3. Image-to-Image Matching (same person? Yes/No)
    T4. Image-to-Images Retrieval (find target in gallery)
    T5. Image-to-Texts Retrieval (select best caption)
    T6. Text-to-Image Matching (does image match caption? Yes/No)
    T7. Text-to-Images Retrieval (pick image matching caption)
"""

import os
import random
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

from ..models.reasoner import STRUCT_PROMPT

# ---------------------------------------------------------------------------
# Instruction templates for each task (T1–T7)
# ---------------------------------------------------------------------------

_T1_INSTRUCTION = (
    "Extract and list the visible attributes (e.g., gender, clothing, "
    "accessories) for the pedestrian in image."
)
_T1_RESPONSE_PREFIX = "The attributes are:"

_T2_INSTRUCTION = (
    "Compare these two pedestrians. List their shared features and identify "
    "what makes each appearance distinct."
)
_T2_RESPONSE_PREFIX_SHARED = "Shared:"
_T2_RESPONSE_PREFIX_UNIQUE_A = "Unique to A:"
_T2_RESPONSE_PREFIX_UNIQUE_B = "Unique to B:"

_T3_INSTRUCTION = (
    "Verify if the person depicted in these two images is the same individual. "
    "Answer Yes or No and provide a concise reasoning based on attribute "
    "consistency or discrepancy."
)

_T4_INSTRUCTION = (
    "Find all occurrences of this specific pedestrian within the provided gallery."
)
_T4_RESPONSE_PREFIX = "Images"
_T4_RESPONSE_SUFFIX = "contain the target pedestrian."

_T5_INSTRUCTION = (
    "Identify the most accurate description for this person from the options below:"
)

_T6_INSTRUCTION_TEMPLATE = (
    "Assess whether this image corresponds to the description: <Caption>."
)

_T7_INSTRUCTION = (
    "From the candidate images, pick the one that aligns with this description:"
)


class HPTStage1Dataset(Dataset):
    """
    Stage 1: Fine-Grained Attribute Alignment dataset.

    Each sample is a single image with the structured P_struct prompt.
    The model should produce a comprehensive, attribute-oriented description.
    Target descriptions are either:
        (a) Cached VLM-generated descriptions (preferred for training)
        (b) Placeholder that forces structured output format

    Args:
        image_list: List of (img_path, pid, camid) tuples from any ReID dataset.
        descriptions: Optional dict mapping img_path → target description string.
                      If None, the model generates its own training targets
                      (self-supervised attribute alignment).
        transform:   Optional image transform (for preprocessing).
    """

    def __init__(
        self,
        image_list: List[Tuple[str, int, int]],
        descriptions: Optional[Dict[str, str]] = None,
        transform=None,
    ):
        self.image_list = image_list
        self.descriptions = descriptions or {}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, pid, camid = self.image_list[idx][:3]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        target_desc = self.descriptions.get(img_path, "")

        return {
            "task": "stage1_attribute_alignment",
            "image": image,
            "image_path": img_path,
            "pid": pid,
            "prompt": STRUCT_PROMPT,
            "target": target_desc,
        }


class HPTStage2Dataset(Dataset):
    """
    Stage 2: Identity Verification and Feedback Generation dataset.

    Constructs 7 types of training samples from ReID datasets.
    Positive pairs: same PID, different camera.
    Negative pairs: different PID.

    Args:
        image_list:    List of (img_path, pid, camid, ...) tuples.
        descriptions:  Dict mapping img_path → pre-generated text descriptions.
        task_weights:  Probabilities for each task type [T1..T7].
        num_gallery:   Number of gallery images for T4/T7 tasks.
        num_negatives: Number of negative samples for hard tasks.
        transform:     Optional image transform.
        seed:          Random seed for reproducibility.
    """

    _DEFAULT_WEIGHTS = [0.15, 0.15, 0.20, 0.15, 0.10, 0.15, 0.10]

    def __init__(
        self,
        image_list: List[Tuple],
        descriptions: Optional[Dict[str, str]] = None,
        task_weights: Optional[List[float]] = None,
        num_gallery: int = 4,
        num_negatives: int = 3,
        transform=None,
        seed: int = 42,
    ):
        self.image_list = image_list
        self.descriptions = descriptions or {}
        self.task_weights = task_weights or self._DEFAULT_WEIGHTS
        self.num_gallery = num_gallery
        self.num_negatives = num_negatives
        self.transform = transform

        # Build PID → image path index
        from collections import defaultdict
        self.pid2paths: Dict[int, List[str]] = defaultdict(list)
        self.all_paths: List[str] = []
        for item in image_list:
            img_path, pid = item[0], item[1]
            self.pid2paths[pid].append(img_path)
            self.all_paths.append(img_path)
        self.all_pids = list(self.pid2paths.keys())

        random.seed(seed)

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Sample one training example by randomly selecting a task type."""
        task_id = random.choices(
            range(1, 8), weights=self.task_weights, k=1
        )[0]

        item = self.image_list[idx]
        anchor_path, anchor_pid = item[0], item[1]

        if task_id == 1:
            return self._build_t1(anchor_path, anchor_pid)
        elif task_id == 2:
            return self._build_t2(anchor_path, anchor_pid)
        elif task_id == 3:
            return self._build_t3(anchor_path, anchor_pid)
        elif task_id == 4:
            return self._build_t4(anchor_path, anchor_pid)
        elif task_id == 5:
            return self._build_t5(anchor_path, anchor_pid)
        elif task_id == 6:
            return self._build_t6(anchor_path, anchor_pid)
        else:
            return self._build_t7(anchor_path, anchor_pid)

    # ------------------------------------------------------------------
    # Task builders
    # ------------------------------------------------------------------

    def _build_t1(self, img_path: str, pid: int) -> Dict[str, Any]:
        """
        T1: Attribute Annotations Matching
        Input:  <Image> + "Extract and list the visible attributes..."
        Output: attribute list
        """
        image = self._load(img_path)
        desc = self.descriptions.get(img_path, "")
        return {
            "task": "T1_attribute_matching",
            "images": [image],
            "prompt": _T1_INSTRUCTION,
            "target": f"{_T1_RESPONSE_PREFIX} {desc}",
        }

    def _build_t2(self, img_path: str, pid: int) -> Dict[str, Any]:
        """
        T2: Attribute Difference Mining
        Input:  <Image1> <Image2> + compare instruction
        Output: Shared: ... Unique to A: ... Unique to B: ...
        """
        image_a = self._load(img_path)
        # Sample a negative (different identity) for contrast
        neg_pid = random.choice([p for p in self.all_pids if p != pid])
        neg_path = random.choice(self.pid2paths[neg_pid])
        image_b = self._load(neg_path)
        desc_a = self.descriptions.get(img_path, "")
        desc_b = self.descriptions.get(neg_path, "")
        return {
            "task": "T2_attribute_diff",
            "images": [image_a, image_b],
            "prompt": _T2_INSTRUCTION,
            "target": (
                f"{_T2_RESPONSE_PREFIX_SHARED} [common attributes] "
                f"{_T2_RESPONSE_PREFIX_UNIQUE_A} [unique to A] "
                f"{_T2_RESPONSE_PREFIX_UNIQUE_B} [unique to B]"
            ),
            "desc_a": desc_a,
            "desc_b": desc_b,
        }

    def _build_t3(self, img_path: str, pid: int) -> Dict[str, Any]:
        """
        T3: Image-to-Image Matching (same person? Yes/No + reasoning)
        50% positive pairs, 50% negative pairs.
        """
        image_a = self._load(img_path)
        is_positive = random.random() > 0.5

        if is_positive and len(self.pid2paths[pid]) > 1:
            other_paths = [p for p in self.pid2paths[pid] if p != img_path]
            img_path_b = random.choice(other_paths)
            label = "Yes"
        else:
            neg_pid = random.choice([p for p in self.all_pids if p != pid])
            img_path_b = random.choice(self.pid2paths[neg_pid])
            label = "No"

        image_b = self._load(img_path_b)
        return {
            "task": "T3_image_match",
            "images": [image_a, image_b],
            "prompt": _T3_INSTRUCTION,
            "target": label,
            "is_same": label == "Yes",
        }

    def _build_t4(self, img_path: str, pid: int) -> Dict[str, Any]:
        """
        T4: Image-to-Images Retrieval
        Input:  <QueryImage> <Gallery images (positives + negatives)>
        Output: indices of images containing the target pedestrian
        """
        image_q = self._load(img_path)

        # Positive images: same PID, different images
        pos_paths = [p for p in self.pid2paths[pid] if p != img_path]
        n_pos = min(len(pos_paths), max(1, self.num_gallery // 2))
        pos_paths_sel = random.sample(pos_paths, n_pos) if pos_paths else []

        # Negative images: different PIDs
        n_neg = self.num_gallery - n_pos
        neg_paths_sel = []
        for _ in range(n_neg):
            neg_pid = random.choice([p for p in self.all_pids if p != pid])
            neg_paths_sel.append(random.choice(self.pid2paths[neg_pid]))

        gallery = pos_paths_sel + neg_paths_sel
        random.shuffle(gallery)
        gallery_images = [self._load(p) for p in gallery]

        # Build target: which gallery indices match (1-indexed)
        match_indices = [
            str(i + 1) for i, p in enumerate(gallery) if p in set(pos_paths_sel)
        ]
        target = (
            f"{_T4_RESPONSE_PREFIX} <{'> and <'.join(match_indices)}> "
            f"{_T4_RESPONSE_SUFFIX}"
        )
        return {
            "task": "T4_image_retrieval",
            "images": [image_q] + gallery_images,
            "prompt": _T4_INSTRUCTION,
            "target": target,
        }

    def _build_t5(self, img_path: str, pid: int) -> Dict[str, Any]:
        """
        T5: Image-to-Texts Retrieval
        Input:  <Image> + "Identify the most accurate description from options:"
                <Caption 1> <Caption 2> <Caption 3>
        Output: "Option N describes the image best."
        """
        image = self._load(img_path)
        desc_correct = self.descriptions.get(img_path, "a person in the image")

        # Generate distractor captions (from other identities)
        distractors = []
        for _ in range(self.num_negatives):
            neg_pid = random.choice([p for p in self.all_pids if p != pid])
            neg_path = random.choice(self.pid2paths[neg_pid])
            distractors.append(
                self.descriptions.get(neg_path, "a different person")
            )

        captions = [desc_correct] + distractors
        random.shuffle(captions)
        correct_idx = captions.index(desc_correct) + 1   # 1-indexed

        options_str = " ".join(
            [f"<Caption {i + 1}> {cap}" for i, cap in enumerate(captions)]
        )
        return {
            "task": "T5_image_to_text",
            "images": [image],
            "prompt": f"{_T5_INSTRUCTION} {options_str}",
            "target": f"Option {correct_idx} describes the image best.",
        }

    def _build_t6(self, img_path: str, pid: int) -> Dict[str, Any]:
        """
        T6: Text-to-Image Matching
        Input:  <Image> <Caption>
        Output: Yes or No
        """
        image = self._load(img_path)
        is_match = random.random() > 0.5

        if is_match:
            caption = self.descriptions.get(img_path, "a person")
            label = "Yes"
        else:
            neg_pid = random.choice([p for p in self.all_pids if p != pid])
            neg_path = random.choice(self.pid2paths[neg_pid])
            caption = self.descriptions.get(neg_path, "a different person")
            label = "No"

        instruction = _T6_INSTRUCTION_TEMPLATE.replace("<Caption>", f'"{caption}"')
        return {
            "task": "T6_text_to_image",
            "images": [image],
            "prompt": instruction,
            "target": label,
            "is_match": is_match,
        }

    def _build_t7(self, img_path: str, pid: int) -> Dict[str, Any]:
        """
        T7: Text-to-Images Retrieval
        Input:  <multiple candidate images> + caption
        Output: "Image <X> matches the description."
        """
        desc_query = self.descriptions.get(img_path, "a person")

        # Positive: another image of same person
        pos_paths = [p for p in self.pid2paths[pid] if p != img_path]
        pos_path = random.choice(pos_paths) if pos_paths else img_path
        image_pos = self._load(pos_path)

        # Negatives
        neg_images = []
        for _ in range(self.num_negatives):
            neg_pid = random.choice([p for p in self.all_pids if p != pid])
            neg_images.append(self._load(random.choice(self.pid2paths[neg_pid])))

        all_images = [image_pos] + neg_images
        random.shuffle(all_images)
        correct_idx = all_images.index(image_pos) + 1   # 1-indexed

        return {
            "task": "T7_text_retrieval",
            "images": all_images,
            "prompt": f"{_T7_INSTRUCTION} {desc_query}",
            "target": f"Image <{correct_idx}> matches the description.",
        }

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _load(self, img_path: str) -> Image.Image:
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


# Alias for unified import
HPTDataset = HPTStage2Dataset
