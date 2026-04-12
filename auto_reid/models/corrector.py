"""
corrector.py - Corrector module for Auto-ReID.

Implements the Corrector ("Deconstruction, Verification and Feedback"):

    "The Corrector mimics the 'System 2' slow-thinking process. It takes the
     candidate set C^(t) and scrutinizes it to generate a refined query T_q^(t+1).
     This module executes three logical steps:
       1. Semantic Deconstruction
       2. Attribute Consistency Verification (ACS)
       3. Feedback Generation"

ACS formula (Equation 4):
    ACS(k, v) = (1/K) Σ_{j=1}^K P(v | I_{cj}, k)

where P(v | I, k) is the VLM's probability that candidate I matches
attribute (k, v).
"""

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
    """
    Auto-ReID Corrector module.

    Performs three steps:
      (1) Semantic Deconstruction: parse T_q^(t) → atomic attribute pairs A^(t)
      (2) ACS Verification: score candidate images against each attribute
      (3) Feedback Generation: produce negative constraints / attribute emphasis

    The VLM (same InternVL used as Reasoner, via its attribute-Q&A capability
    trained in HPT Stage 2) answers attribute-specific Yes/No questions about
    each candidate image to estimate P(v | I_{cj}, k).
    """

    def __init__(
        self,
        vlm_model,        # Reasoner instance or equivalent VLM wrapper
        vlm_tokenizer,
        tau_low: float = DEFAULT_TAU_LOW,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            vlm_model: The fine-tuned VLM model object (InternVL).
                       Re-uses the same model as the Reasoner to save GPU memory.
            vlm_tokenizer: Tokenizer for the VLM.
            tau_low: ACS threshold below which a conflict is detected.
            device: Torch device.
            torch_dtype: Computation dtype.
        """
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
        """
        Parse the monolithic textual query T_q^(t) into atomic attribute-value
        pairs A^(t) = {(k_j, v_j)}_{j=1}^M.

        Example:
            Input:  "A woman in her late 20s has long dark hair and wears a
                     sleeveless white dress."
            Output: {Gender: Female, Age: late 20s, Hair: long dark,
                     Upper: sleeveless white dress, Bag: None}

        Args:
            text_desc: Current text description T_q^(t).
            use_vlm: If True, prompt the VLM for structured parsing.
                     If False (or VLM unavailable), use rule-based parser.

        Returns:
            Dict mapping attribute key → value string.
        """
        if use_vlm and self.model is not None:
            try:
                return self._vlm_parse(text_desc)
            except Exception as e:
                logger.warning("VLM parsing failed (%s), using rule-based.", e)

        return parse_attributes_rule_based(text_desc)

    def _vlm_parse(self, text_desc: str) -> Dict[str, str]:
        """
        Ask the VLM to output a structured attribute list from the description.
        The VLM was trained in HPT Stage 1 to output structured attributes.
        """
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
        """
        Compute Attribute Consistency Score for each attribute over the
        candidate set C^(t).

        Equation (4):
            ACS(k, v) = (1/K) Σ_{j=1}^K P(v | I_{cj}, k)

        For each attribute (k, v) in A^(t):
          - Ask the VLM: "Is the person [k = v]?"
          - Interpret Yes → 1.0, No → 0.0 (or use logit probabilities)
          - Average over all K candidates

        Args:
            candidates: List of K candidate PIL images from Top-K retrieval.
            attributes: Attribute dict from parse_to_attributes().

        Returns:
            Dict mapping attribute key → ACS score ∈ [0, 1].
            Only attributes with ACS < tau_low represent conflicts.
        """
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
        """
        Ask the VLM a yes/no question about an image.
        Returns probability ∈ [0, 1] that the answer is "Yes".

        The HPT Stage 2 Task 2 (Attribute-Specific Q&A) trains the VLM
        to answer these questions accurately.
        """
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
        """
        Generate feedback instruction I_feedback based on ACS conflicts.

        Two types of feedback:
            "Negative Constraints: If distractors share a common feature
             (e.g., backpacks), the instruction is:
             'Exclude candidates with backpacks.'"

            "Attribute Emphasis: If a key visual trait (e.g., white dress)
             is missed, the instruction is:
             'Prioritize candidates wearing white dresses.'"

        Args:
            attributes: Attribute dict from parse_to_attributes().
            acs_scores: ACS score dict from compute_acs().

        Returns:
            Feedback string I_feedback, or None if no conflicts (→ early stop).
        """
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
        """Preprocess PIL image for InternVL (same as Reasoner)."""
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
