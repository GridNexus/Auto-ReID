import os
import logging
from typing import Optional

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, LoraConfig, get_peft_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structured prompt P_struct (Stage 1 attribute alignment)
# ---------------------------------------------------------------------------
STRUCT_PROMPT = (
    "Describe the person in the image. Include: gender, estimated age group, "
    "hairstyle (length/color), upper clothing (type/color/pattern), "
    "lower clothing (type/color), footwear (type/color), "
    "and any carried items or notable accessories."
)

# ---------------------------------------------------------------------------
# Refining prompt template P_re
# Used when incorporating Corrector feedback for T_q^(t+1)
# ---------------------------------------------------------------------------
REFINE_PROMPT_TEMPLATE = (
    "You are re-identifying a person. Here is the current description:\n"
    "{current_desc}\n\n"
    "Feedback from the search system:\n"
    "{feedback}\n\n"
    "Based on the feedback and the original image, please provide a refined "
    "description that addresses the issues mentioned. Maintain accurate "
    "details and focus on identity-discriminative cues.\n"
    "Refined description:"
)

# System prompt for HPT-tuned VLM
SYSTEM_PROMPT = (
    "You are a specialized person re-identification assistant. Your task is "
    "to provide accurate, detailed descriptions of pedestrian appearance for "
    "the purpose of person retrieval across camera views."
)


class Reasoner:

    def __init__(
        self,
        vlm_model_path: str = "OpenGVLab/InternVL2_5-8B",
        lora_checkpoint: Optional[str] = None,
        device: str = "cuda",
        load_in_8bit: bool = False,
        max_new_tokens: int = 256,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.torch_dtype = torch_dtype

        logger.info("Loading tokenizer from %s", vlm_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            vlm_model_path, trust_remote_code=True, use_fast=False
        )

        logger.info("Loading VLM backbone from %s", vlm_model_path)
        load_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        else:
            load_kwargs["device_map"] = device

        self.model = AutoModel.from_pretrained(vlm_model_path, **load_kwargs)

        # Load HPT LoRA adapters if provided
        if lora_checkpoint is not None:
            logger.info("Loading HPT LoRA adapters from %s", lora_checkpoint)
            self.model = PeftModel.from_pretrained(
                self.model, lora_checkpoint
            )
            self.model = self.model.merge_and_unload()
            logger.info("LoRA adapters merged into model weights")

        self.model.eval()
        if not load_in_8bit:
            self.model = self.model.to(device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initial_description(self, image: Image.Image) -> str:
        return self._generate(image=image, text_prompt=STRUCT_PROMPT)

    def refine_description(
        self,
        image: Image.Image,
        current_desc: str,
        feedback: str,
    ) -> str:
        prompt = REFINE_PROMPT_TEMPLATE.format(
            current_desc=current_desc,
            feedback=feedback,
        )
        return self._generate(image=image, text_prompt=prompt)

    # ------------------------------------------------------------------
    # Internal generation
    # ------------------------------------------------------------------

    def _generate(self, image: Image.Image, text_prompt: str) -> str:
        # InternVL uses a specific chat template; construct the conversation
        # with <image> token in the user message
        generation_config = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,       # greedy decoding for reproducibility
            temperature=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Preprocess image to pixel_values tensor
        pixel_values = self._preprocess_image(image)

        # Build query with <image> placeholder
        question = f"<image>\n{text_prompt}"

        try:
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
                history=None,
                return_history=False,
                system_message=SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.warning("VLM generation failed: %s. Returning empty string.", e)
            response = ""

        return response.strip()

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        from torchvision import transforms

        # InternVL standard normalization
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
            transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        image = image.convert("RGB")
        pixel_values = transform(image).unsqueeze(0)  # [1, 3, 448, 448]
        return pixel_values.to(dtype=self.torch_dtype, device=self.device)
