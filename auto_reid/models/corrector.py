import logging
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from ..utils.attribute_parser import (
    parse_attributes_from_vlm_response,
    parse_attributes_rule_based,
    build_attribute_question,
    build_negative_constraint,
    build_emphasis_constraint,
    ATTR_KEYS,
)

logger = logging.getLogger(__name__)

# ACS probability threshold below which a conflict is declared
# (set via cfg.AUTO_REID.TAU_LOW, default 0.4)
DEFAULT_TAU_LOW = 0.4


class Corrector:

    def __init__(
        self,
        vlm_model,        # Reasoner instance or equivalent VLM wrapper
        vlm_tokenizer,
        tau_low: float = DEFAULT_TAU_LOW,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = vlm_model
        self.tokenizer = vlm_tokenizer
        self.tau_low = tau_low
        self.device = device
        self.dtype = torch_dtype

    # ------------------------------------------------------------------
    # Step 1: Semantic Deconstruction
    # ------------------------------------------------------------------

    def parse_to_attributes(
        self,
        text_desc: str,
        use_vlm: bool = True,
    ) -> Dict[str, str]:
        
        if use_vlm and self.model is not None:
            try:
                return self._vlm_parse(text_desc)
            except Exception as e:
                logger.warning("VLM parsing failed (%s), using rule-based.", e)

        return parse_attributes_rule_based(text_desc)

    def _vlm_parse(self, text_desc: str) -> Dict[str, str]:
        parse_prompt = (
            "Given the following person description, extract the structured "
            "attributes in the format 'Key: Value', one per line. "
            "Use these keys: Gender, Age, Hair, Upper, Lower, Footwear, Bag.\n\n"
            f"Description: {text_desc}\n\n"
            "Attributes:"
        )
        generation_config = dict(
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=None,   # text-only for parsing
            question=parse_prompt,
            generation_config=generation_config,
            history=None,
            return_history=False,
        )
        return parse_attributes_from_vlm_response(response)

    # ------------------------------------------------------------------
    # Step 2: Attribute Consistency Verification (ACS)
    # ------------------------------------------------------------------

    def compute_acs(
        self,
        candidates: List[Image.Image],
        attributes: Dict[str, str],
    ) -> Dict[str, float]:
        
        K = len(candidates)
        if K == 0:
            return {}

        acs_scores: Dict[str, float] = {}

        for key, value in attributes.items():
            if value in ("unknown", ""):
                continue   # skip unparsed attributes

            question = build_attribute_question(key, value)
            scores_per_candidate = []

            for img in candidates:
                prob = self._query_attribute(img, question)
                scores_per_candidate.append(prob)

            acs = sum(scores_per_candidate) / K
            acs_scores[key] = acs

        return acs_scores

    def _query_attribute(self, image: Image.Image, question: str) -> float:
        if self.model is None:
            return 0.5

        pixel_values = self._preprocess_image(image)

        try:
            with torch.no_grad():
                full_question = f"<image>\n{question} Answer with Yes or No."
                output = self.model.chat(
                    tokenizer=self.tokenizer,
                    pixel_values=pixel_values,
                    question=full_question,
                    generation_config=dict(
                        max_new_tokens=5,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    ),
                    history=None,
                    return_history=False,
                )
                answer = output.strip().lower()
                if answer.startswith("yes"):
                    return 1.0
                elif answer.startswith("no"):
                    return 0.0
                else:
                    return 0.5

        except Exception as e:
            logger.warning("Attribute query failed: %s", e)
            return 0.5

    # ------------------------------------------------------------------
    # Step 3: Feedback Generation
    # ------------------------------------------------------------------

    def generate_feedback(
        self,
        attributes: Dict[str, str],
        acs_scores: Dict[str, float],
    ) -> Optional[str]:
        conflicts = {
            key: val for key, val in attributes.items()
            if key in acs_scores and acs_scores[key] < self.tau_low
        }

        if not conflicts:
            return None   # No conflicts → trigger early stopping check

        feedback_lines = []
        for key, value in conflicts.items():
            # Skip truly unparsed attributes
            if value.lower() in ("unknown", ""):
                continue

            if value.lower() == "none":
                # Query says person does NOT have this feature,
                # but ACS is low → candidates incorrectly DO have it.
                # Example: Bag=None but most candidates carry bags →
                #   "Exclude candidates with backpacks."  (Negative Constraint)
                feedback_lines.append(build_negative_constraint(key, value))
            else:
                # Query says person HAS this attribute,
                # but ACS is low → candidates are missing it.
                # Example: Upper="white dress" but candidates lack it →
                #   "Prioritize candidates wearing white dresses." (Attribute Emphasis)
                feedback_lines.append(build_emphasis_constraint(key, value))

        if not feedback_lines:
            return None

        return " ".join(feedback_lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        from torchvision import transforms

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
            transforms.Resize((448, 448),
                               interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        image = image.convert("RGB")
        pixel_values = transform(image).unsqueeze(0)
        return pixel_values.to(dtype=self.dtype, device=self.device)
