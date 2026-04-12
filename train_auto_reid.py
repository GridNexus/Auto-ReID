"""
train_auto_reid.py - HPT training entry point for Auto-ReID.

Hierarchical Progressive Tuning in two stages:
    Stage 1: Fine-Grained Attribute Alignment
    Stage 2: Identity Verification and Feedback Generation

Usage:
    # Stage 1 training (attribute alignment)
    python train_auto_reid.py \\
        --config configs/AutoReID/msmt17.yml \\
        AUTO_REID.HPT_STAGE 1 \\
        OUTPUT_DIR output/autored_msmt17

    # Stage 2 training (multi-task, from Stage 1 checkpoint)
    python train_auto_reid.py \\
        --config configs/AutoReID/msmt17.yml \\
        AUTO_REID.HPT_STAGE 2 \\
        AUTO_REID.STAGE1_CKPT output/autored_msmt17/hpt_stage1/stage1_lora \\
        OUTPUT_DIR output/autored_msmt17

Training configuration:
    - VLM: InternVL3.5-8B
    - LoRA rank=16, target=attention layers
    - lr=1e-5, epochs=3, batch_size=4 per GPU
    - Recommended hardware: 4×NVIDIA A100 GPUs
"""

import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from auto_reid.training.hpt_trainer import HPTTrainer
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-ReID HPT Training"
    )
    parser.add_argument(
        "--config", default="configs/AutoReID/msmt17.yml",
        help="Path to YACS config file"
    )
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER,
        help="Modify config options using the command-line (key value pairs)"
    )
    return parser.parse_args()


def build_image_list_from_dataloader(cfg):
    """
    Extract (img_path, pid, camid) tuples from the ReID dataset.
    Uses existing dataset infrastructure.
    """
    from datasets.make_dataloader import __factory
    dataset_name = cfg.DATASETS.NAMES
    dataset = __factory[dataset_name](root=cfg.DATASETS.ROOT_DIR)
    # Use training split
    image_list = [(img_path, pid, camid)
                  for img_path, pid, camid, _ in dataset.train]
    logger.info("Loaded %d training images from %s", len(image_list), dataset_name)
    return image_list


def main():
    args = parse_args()

    # Merge config
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Setup output dir and logger
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    setup_logger(cfg.OUTPUT_DIR, dist_rank=0, name="auto_reid_train")

    logger.info("=" * 70)
    logger.info("Auto-ReID HPT Training — Stage %d", cfg.AUTO_REID.HPT_STAGE)
    logger.info("Dataset: %s", cfg.DATASETS.NAMES)
    logger.info("VLM: %s", cfg.AUTO_REID.VLM_MODEL)
    logger.info("LoRA rank: %d", cfg.AUTO_REID.LORA_RANK)
    logger.info("LR: %g, Epochs: %d", cfg.AUTO_REID.HPT_LR, cfg.AUTO_REID.HPT_EPOCHS)
    logger.info("=" * 70)

    # Build image list from dataset
    image_list = build_image_list_from_dataloader(cfg)

    # Create trainer
    trainer = HPTTrainer(cfg)

    stage = cfg.AUTO_REID.HPT_STAGE

    if stage == 1:
        # Stage 1: Fine-Grained Attribute Alignment
        logger.info("Starting Stage 1: Fine-Grained Attribute Alignment")
        stage1_ckpt = trainer.train_stage1(
            image_list=image_list,
            descriptions=None,   # self-supervised
            output_dir=os.path.join(cfg.OUTPUT_DIR, "hpt_stage1"),
        )
        logger.info("Stage 1 complete. Checkpoint: %s", stage1_ckpt)

    elif stage == 2:
        # Stage 2: Identity Verification and Feedback Generation
        stage1_ckpt = cfg.AUTO_REID.STAGE1_CKPT
        if not stage1_ckpt or not os.path.isdir(stage1_ckpt):
            logger.warning(
                "Stage 1 checkpoint not found at '%s'. "
                "Stage 2 will use fresh LoRA adapters.", stage1_ckpt
            )
            stage1_ckpt = None

        # Load Stage 1 descriptions for Stage 2 training
        # (In practice, these are cached after Stage 1 inference)
        # Here we pass an empty dict and fall back to filename-based captions
        descriptions = {}
        desc_cache_path = os.path.join(
            cfg.OUTPUT_DIR, "hpt_stage1", "descriptions.pth"
        )
        if os.path.exists(desc_cache_path):
            descriptions = torch.load(desc_cache_path)
            logger.info("Loaded %d descriptions from cache", len(descriptions))
        else:
            logger.warning(
                "No description cache found at %s. "
                "Stage 2 tasks will use empty descriptions.", desc_cache_path
            )

        logger.info("Starting Stage 2: Identity Verification & Feedback Generation")
        stage2_ckpt = trainer.train_stage2(
            image_list=image_list,
            descriptions=descriptions,
            stage1_ckpt=stage1_ckpt,
            output_dir=os.path.join(cfg.OUTPUT_DIR, "hpt_stage2"),
        )
        logger.info("Stage 2 complete. Checkpoint: %s", stage2_ckpt)

    else:
        raise ValueError(f"AUTO_REID.HPT_STAGE must be 1 or 2, got {stage}")


if __name__ == "__main__":
    main()
