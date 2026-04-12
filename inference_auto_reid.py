import argparse
import logging
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from datasets.make_dataloader import __factory
from auto_reid.models.reasoner import Reasoner
from auto_reid.models.hybrid_retriever import HybridRetriever
from auto_reid.models.corrector import Corrector
from auto_reid.inference.pipeline import AutoReIDPipeline
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-ReID Iterative Inference"
    )
    parser.add_argument(
        "--config", default="configs/AutoReID/msmt17.yml",
        help="Path to YACS config file"
    )
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER,
        help="Overwrite config options as key=value pairs"
    )
    return parser.parse_args()


def load_dataset_splits(cfg):
    dataset_name = cfg.DATASETS.NAMES
    dataset = __factory[dataset_name](root=cfg.DATASETS.ROOT_DIR)

    query_list = [
        (img_path, pid, camid)
        for img_path, pid, camid, _ in dataset.query
    ]
    gallery_list = [
        (img_path, pid, camid)
        for img_path, pid, camid, _ in dataset.gallery
    ]
    logger.info(
        "Dataset: %s | Query: %d | Gallery: %d",
        dataset_name, len(query_list), len(gallery_list)
    )
    return query_list, gallery_list


def load_images(image_list: List[Tuple], target_size: Tuple[int, int] = (256, 128)) -> List[Image.Image]:
    images = []
    for img_path, _, _ in image_list:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((target_size[1], target_size[0]), Image.BICUBIC)
        images.append(img)
    return images


def build_gallery_captions(gallery_list: List[Tuple]) -> List[str]:
    captions = []
    for img_path, pid, camid in gallery_list:
        fname = os.path.basename(img_path)
        captions.append(fname)
    return captions


def main():
    args = parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    setup_logger(cfg.OUTPUT_DIR, dist_rank=0, name="auto_reid_infer")

    device = f"cuda:{cfg.MODEL.DEVICE_ID}" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 70)
    logger.info("Auto-ReID Inference Pipeline")
    logger.info("Dataset:    %s", cfg.DATASETS.NAMES)
    logger.info("VLM:        %s", cfg.AUTO_REID.VLM_MODEL)
    logger.info("VLM LoRA:   %s", cfg.AUTO_REID.VLM_CKPT)
    logger.info("Encoder:    %s", cfg.AUTO_REID.VIS_ENCODER)
    logger.info("α=%.2f, T_max=%d, K=%d",
                cfg.AUTO_REID.ALPHA, cfg.AUTO_REID.T_MAX, cfg.AUTO_REID.K)
    logger.info("=" * 70)

    # ── 1. Load dataset ──────────────────────────────────────────────
    query_list, gallery_list = load_dataset_splits(cfg)

    img_size = (cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1])
    logger.info("Loading query images (size=%dx%d)...", *img_size)
    query_images = load_images(query_list, target_size=img_size)
    logger.info("Loading gallery images...")
    gallery_images = load_images(gallery_list, target_size=img_size)

    query_pids   = [x[1] for x in query_list]
    query_camids = [x[2] for x in query_list]
    gallery_pids   = [x[1] for x in gallery_list]
    gallery_camids = [x[2] for x in gallery_list]

    # ── 2. Initialize Hybrid Retriever (SigLIP2) ─────────────────────
    logger.info("Initializing HybridRetriever (α=%.2f)...", cfg.AUTO_REID.ALPHA)
    retriever = HybridRetriever(
        encoder_name=cfg.AUTO_REID.VIS_ENCODER,
        alpha=cfg.AUTO_REID.ALPHA,
        device=device,
    )

    # Pre-compute gallery features (done once)
    gallery_captions = build_gallery_captions(gallery_list)
    logger.info("Indexing gallery (%d images)...", len(gallery_images))
    retriever.index_gallery(
        images=gallery_images,
        captions=gallery_captions,
        batch_size=cfg.TEST.IMS_PER_BATCH,
    )

    # ── 3. Initialize Reasoner (InternVL + HPT LoRA) ─────────────────
    vlm_ckpt = cfg.AUTO_REID.VLM_CKPT if cfg.AUTO_REID.VLM_CKPT else None
    logger.info("Initializing Reasoner (VLM=%s)...", cfg.AUTO_REID.VLM_MODEL)
    reasoner = Reasoner(
        vlm_model_path=cfg.AUTO_REID.VLM_MODEL,
        lora_checkpoint=vlm_ckpt,
        device=device,
        torch_dtype=torch.bfloat16,
    )

    # ── 4. Initialize Corrector (reuses VLM from Reasoner) ───────────
    logger.info("Initializing Corrector (τ_low=%.2f)...", cfg.AUTO_REID.TAU_LOW)
    corrector = Corrector(
        vlm_model=reasoner.model,
        vlm_tokenizer=reasoner.tokenizer,
        tau_low=cfg.AUTO_REID.TAU_LOW,
        device=device,
        torch_dtype=torch.bfloat16,
    )

    # ── 5. Build pipeline (Algorithm 1) ──────────────────────────────
    pipeline = AutoReIDPipeline(
        reasoner=reasoner,
        retriever=retriever,
        corrector=corrector,
        t_max=cfg.AUTO_REID.T_MAX,
        k=cfg.AUTO_REID.K,
        iou_threshold=cfg.AUTO_REID.IOU_THRESHOLD,
        top_n_prefilt=cfg.AUTO_REID.TOP_N_PREFILT,
    )

    # ── 6. Run evaluation ─────────────────────────────────────────────
    logger.info("Running iterative inference (T_max=%d)...", cfg.AUTO_REID.T_MAX)
    results = pipeline.evaluate(
        query_images=query_images,
        query_pids=query_pids,
        query_camids=query_camids,
        gallery_images=gallery_images,
        gallery_pids=gallery_pids,
        gallery_camids=gallery_camids,
        max_rank=50,
    )

    # ── 7. Report results ─────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Results on %s:", cfg.DATASETS.NAMES)
    logger.info("  mAP:    %.1f%%", results['mAP'])
    logger.info("  Rank-1: %.1f%%", results['rank1'])
    logger.info("  Rank-5: %.1f%%", results['rank5'])
    logger.info("  Rank-10:%.1f%%", results['rank10'])
    logger.info("=" * 50)

    return results


if __name__ == "__main__":
    main()
