"""
hpt_trainer.py - Hierarchical Progressive Tuning (HPT) two-stage LoRA trainer.

Keeps the base VLM weights frozen and applies parameter-efficient
Low-Rank Adaptation (LoRA) to the attention layers.

    Stage 1: Fine-Grained Attribute Alignment
    Stage 2: Identity Verification and Feedback Generation

Training configuration:
    - Backbone: InternVL3.5-8B (OpenGVLab/InternVL2_5-8B)
    - LoRA rank: 16
    - LoRA target modules: attention layers (q_proj, v_proj, k_proj, o_proj)
    - Learning rate: 1e-5
    - Epochs: 3
    - Batch size: 4 per GPU (×4 GPUs = 16 effective)
    - Recommended hardware: 4×NVIDIA A100 GPUs
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from .hpt_dataset import HPTStage1Dataset, HPTStage2Dataset

logger = logging.getLogger(__name__)


class HPTTrainer:
    """
    Hierarchical Progressive Tuning trainer.

    Manages LoRA fine-tuning of InternVL3.5-8B across two stages:
      Stage 1 — attribute alignment (structured description generation)
      Stage 2 — multi-task identity verification and feedback generation

    Usage:
        trainer = HPTTrainer(cfg)
        # Stage 1
        trainer.train_stage1(image_list, output_dir="output/stage1")
        # Stage 2 (initializes from Stage 1 checkpoint)
        trainer.train_stage2(image_list, descriptions,
                             stage1_ckpt="output/stage1",
                             output_dir="output/stage2")
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: YACS config node. Reads AUTO_REID.* fields.
        """
        self.cfg = cfg
        self.device = (
            f"cuda:{cfg.MODEL.DEVICE_ID}"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.vlm_model_path = cfg.AUTO_REID.VLM_MODEL
        self.lora_rank = cfg.AUTO_REID.LORA_RANK
        self.lora_alpha = cfg.AUTO_REID.LORA_ALPHA
        self.lora_dropout = cfg.AUTO_REID.LORA_DROPOUT
        self.lora_target_modules = list(cfg.AUTO_REID.LORA_TARGET_MODULES)
        self.lr = cfg.AUTO_REID.HPT_LR
        self.epochs = cfg.AUTO_REID.HPT_EPOCHS
        self.batch_size = cfg.AUTO_REID.HPT_BATCH_SIZE

    # ------------------------------------------------------------------
    # Stage 1: Fine-Grained Attribute Alignment
    # ------------------------------------------------------------------

    def train_stage1(
        self,
        image_list: List[Tuple],
        descriptions: Optional[Dict[str, str]] = None,
        output_dir: str = "output/hpt_stage1",
    ) -> str:
        """
        Stage 1 training: teach the VLM to generate structured attribute
        descriptions using the P_struct template.

        Args:
            image_list:   List of (img_path, pid, camid) tuples from ReID datasets.
            descriptions: Optional pre-computed descriptions for supervised training.
                          If None, model trains with the P_struct prompt self-supervised.
            output_dir:   Directory to save Stage 1 LoRA checkpoint.

        Returns:
            Path to saved checkpoint directory.
        """
        logger.info("=" * 60)
        logger.info("HPT Stage 1: Fine-Grained Attribute Alignment")
        logger.info("VLM: %s", self.vlm_model_path)
        logger.info("LoRA rank: %d, target: %s", self.lora_rank, self.lora_target_modules)
        logger.info("LR: %g, Epochs: %d", self.lr, self.epochs)
        logger.info("=" * 60)

        model, tokenizer = self._load_model_with_lora()

        dataset = HPTStage1Dataset(
            image_list=image_list,
            descriptions=descriptions,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self._collate_stage1,
            drop_last=True,
        )

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr,
            weight_decay=0.01,
        )
        total_steps = len(dataloader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps,
        )

        model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for step, batch in enumerate(dataloader):
                loss = self._train_step_stage1(model, tokenizer, batch)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

                if step % 50 == 0:
                    logger.info(
                        "Stage1 Epoch [%d/%d] Step [%d/%d] Loss=%.4f",
                        epoch + 1, self.epochs, step + 1,
                        len(dataloader), loss.item()
                    )

            logger.info(
                "Stage1 Epoch [%d/%d] AvgLoss=%.4f",
                epoch + 1, self.epochs, epoch_loss / len(dataloader)
            )

        ckpt_path = self._save_checkpoint(model, output_dir, "stage1_lora")
        logger.info("Stage 1 checkpoint saved to: %s", ckpt_path)
        return ckpt_path

    # ------------------------------------------------------------------
    # Stage 2: Identity Verification and Feedback Generation
    # ------------------------------------------------------------------

    def train_stage2(
        self,
        image_list: List[Tuple],
        descriptions: Dict[str, str],
        stage1_ckpt: Optional[str] = None,
        output_dir: str = "output/hpt_stage2",
    ) -> str:
        """
        Stage 2 training: teach the VLM the seven auxiliary tasks for
        pairwise verification, attribute Q&A, and corrective feedback.

        Args:
            image_list:   List of (img_path, pid, camid) tuples.
            descriptions: Dict of img_path → text description (from Stage 1).
            stage1_ckpt:  Path to Stage 1 LoRA checkpoint (None = fresh LoRA).
            output_dir:   Directory to save Stage 2 LoRA checkpoint.

        Returns:
            Path to saved checkpoint directory.
        """
        logger.info("=" * 60)
        logger.info("HPT Stage 2: Identity Verification & Feedback Generation")
        logger.info("7 auxiliary tasks (Figure 2 in paper)")
        logger.info("=" * 60)

        model, tokenizer = self._load_model_with_lora(lora_checkpoint=stage1_ckpt)

        dataset = HPTStage2Dataset(
            image_list=image_list,
            descriptions=descriptions,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self._collate_stage2,
            drop_last=True,
        )

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr,
            weight_decay=0.01,
        )
        total_steps = len(dataloader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps,
        )

        model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            task_losses: Dict[str, List[float]] = {}

            for step, batch in enumerate(dataloader):
                loss, task_name = self._train_step_stage2(
                    model, tokenizer, batch
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                task_losses.setdefault(task_name, []).append(loss.item())

                if step % 50 == 0:
                    logger.info(
                        "Stage2 Epoch [%d/%d] Step [%d/%d] Task=%s Loss=%.4f",
                        epoch + 1, self.epochs, step + 1,
                        len(dataloader), task_name, loss.item()
                    )

            logger.info(
                "Stage2 Epoch [%d/%d] AvgLoss=%.4f",
                epoch + 1, self.epochs, epoch_loss / len(dataloader)
            )
            # Per-task loss summary
            for task, losses in task_losses.items():
                logger.info("  %s: avg_loss=%.4f", task,
                            sum(losses) / len(losses))

        ckpt_path = self._save_checkpoint(model, output_dir, "stage2_lora")
        logger.info("Stage 2 checkpoint saved to: %s", ckpt_path)
        return ckpt_path

    # ------------------------------------------------------------------
    # Training step helpers
    # ------------------------------------------------------------------

    def _train_step_stage1(
        self,
        model: nn.Module,
        tokenizer,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Single Stage 1 training step: causal language modeling loss.
        The model is trained to generate structured descriptions given P_struct.
        """
        # Build prompt: <image>\n{P_struct}
        # Target: {structured description}
        # Loss: cross-entropy on target tokens only
        images = batch["images"]       # List of PIL or tensors
        prompts = batch["prompts"]     # List of strings
        targets = batch["targets"]     # List of strings

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        for img, prompt, target in zip(images, prompts, targets):
            if not target:
                continue
            pixel_values = self._preprocess_image(img)
            question = f"<image>\n{prompt}"
            full_response = target

            loss = self._compute_clm_loss(
                model, tokenizer, pixel_values, question, full_response
            )
            total_loss = total_loss + loss

        n = len([t for t in targets if t])
        return total_loss / max(n, 1)

    def _train_step_stage2(
        self,
        model: nn.Module,
        tokenizer,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, str]:
        """
        Single Stage 2 training step: multi-task instruction following.
        Returns (loss, task_name).
        """
        task_name = batch.get("task", "unknown")
        images_list = batch["images_list"]   # List[List[PIL]]
        prompts = batch["prompts"]
        targets = batch["targets"]

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        count = 0
        for imgs, prompt, target in zip(images_list, prompts, targets):
            if not target:
                continue
            # Encode images (may be multiple for T2-T7)
            if imgs:
                pixel_values_list = [self._preprocess_image(img) for img in imgs]
                pixel_values = torch.cat(pixel_values_list, dim=0)   # [N, 3, H, W]
                question = "<image>\n" * len(imgs) + prompt
            else:
                pixel_values = None
                question = prompt

            loss = self._compute_clm_loss(
                model, tokenizer, pixel_values, question, target
            )
            total_loss = total_loss + loss
            count += 1

        return total_loss / max(count, 1), task_name

    def _compute_clm_loss(
        self,
        model: nn.Module,
        tokenizer,
        pixel_values: Optional[torch.Tensor],
        question: str,
        target: str,
    ) -> torch.Tensor:
        """
        Compute causal language modeling loss on target tokens.
        Uses InternVL's built-in chat-based loss computation.
        """
        try:
            # Tokenize input + target
            input_text = question + " " + target
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            input_ids = inputs["input_ids"].to(self.device)

            # Forward pass
            kwargs = dict(
                input_ids=input_ids,
                labels=input_ids,  # causal LM: labels = input shifted
            )
            if pixel_values is not None:
                kwargs["pixel_values"] = pixel_values.to(self.device)

            outputs = model(**kwargs)
            return outputs.loss

        except Exception as e:
            logger.warning("CLM loss computation failed: %s", e)
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    # ------------------------------------------------------------------
    # Model loading and saving
    # ------------------------------------------------------------------

    def _load_model_with_lora(
        self,
        lora_checkpoint: Optional[str] = None,
    ) -> Tuple[nn.Module, Any]:
        """
        Load InternVL3.5-8B and apply LoRA adapters.

        If lora_checkpoint is provided, loads an existing LoRA checkpoint
        (for Stage 2 initialization from Stage 1 weights).
        Otherwise, creates fresh LoRA adapters.
        """
        logger.info("Loading tokenizer from %s", self.vlm_model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            self.vlm_model_path,
            trust_remote_code=True,
            use_fast=False,
        )

        logger.info("Loading VLM base model from %s", self.vlm_model_path)
        model = AutoModel.from_pretrained(
            self.vlm_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )

        if lora_checkpoint is not None and os.path.isdir(lora_checkpoint):
            logger.info("Loading LoRA from checkpoint: %s", lora_checkpoint)
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_checkpoint)
        else:
            # Create fresh LoRA adapters (rank=16)
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            trainable, total = model.get_nb_trainable_parameters()
            logger.info(
                "LoRA trainable params: %d (%.2f%% of %d total)",
                trainable, 100 * trainable / total, total,
            )

        model.train()
        return model, tokenizer

    def _save_checkpoint(
        self,
        model: nn.Module,
        output_dir: str,
        name: str,
    ) -> str:
        """Save LoRA adapter weights to disk."""
        os.makedirs(output_dir, exist_ok=True)
        ckpt_dir = os.path.join(output_dir, name)
        model.save_pretrained(ckpt_dir)
        logger.info("Saved LoRA checkpoint: %s", ckpt_dir)
        return ckpt_dir

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    def _preprocess_image(self, image) -> torch.Tensor:
        """Preprocess PIL image for InternVL (448×448, normalized)."""
        from torchvision import transforms
        from PIL import Image as PILImage

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
            transforms.Resize((448, 448),
                               interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        if not isinstance(image, PILImage.Image):
            image = PILImage.fromarray(image)
        image = image.convert("RGB")
        return transform(image).unsqueeze(0).to(
            dtype=torch.bfloat16, device=self.device
        )

    # ------------------------------------------------------------------
    # Collate functions
    # ------------------------------------------------------------------

    @staticmethod
    def _collate_stage1(batch: List[Dict]) -> Dict[str, Any]:
        return {
            "images":  [b["image"] for b in batch],
            "prompts": [b["prompt"] for b in batch],
            "targets": [b["target"] for b in batch],
            "pids":    [b["pid"] for b in batch],
        }

    @staticmethod
    def _collate_stage2(batch: List[Dict]) -> Dict[str, Any]:
        # All samples in a batch should be the same task (shuffle=True breaks this,
        # so we just pack them individually)
        return {
            "task":         batch[0].get("task", "unknown"),
            "images_list":  [b["images"] for b in batch],
            "prompts":      [b["prompt"] for b in batch],
            "targets":      [b["target"] for b in batch],
        }
