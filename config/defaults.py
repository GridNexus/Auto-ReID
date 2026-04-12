from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# Device
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
# Device to use: "cuda" or "cpu"
_C.MODEL.DEVICE = "cuda"
# GPU device ID (single GPU; for multi-GPU set via CUDA_VISIBLE_DEVICES)
_C.MODEL.DEVICE_ID = '0'

# ---------------------------------------------------------------------------- #
# Input
# ---------------------------------------------------------------------------- #
_C.INPUT = CN()
# Image size for training [H, W] — 256×128
_C.INPUT.SIZE_TRAIN = [256, 128]
# Image size for inference [H, W]
_C.INPUT.SIZE_TEST = [256, 128]
# Random horizontal flip probability (data augmentation)
_C.INPUT.PROB = 0.5
# Random erasing probability (data augmentation)
_C.INPUT.RE_PROB = 0.5
# ImageNet normalization mean
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# ImageNet normalization std
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Padding size for random crop
_C.INPUT.PADDING = 10

# ---------------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------------- #
_C.DATASETS = CN()
# Dataset name; one of: market1501, msmt17, cuhk03, occ_duke, prcc, ltcc
_C.DATASETS.NAMES = 'market1501'
# Root directory containing all dataset folders
_C.DATASETS.ROOT_DIR = '../data'

# ---------------------------------------------------------------------------- #
# DataLoader
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Number of images per training batch
_C.DATALOADER.IMS_PER_BATCH = 64

# ---------------------------------------------------------------------------- #
# Test / Inference
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Number of images per batch during gallery feature extraction
_C.TEST.IMS_PER_BATCH = 128

# ---------------------------------------------------------------------------- #
# Output
# ---------------------------------------------------------------------------- #
# Directory for saving logs and checkpoints
_C.OUTPUT_DIR = ""

# ---------------------------------------------------------------------------- #
# AUTO_REID — Auto-ReID
# "Iterative Self-Correction for Text-Driven Person Re-Identification
#  with Large Vision-Language Models"
# ---------------------------------------------------------------------------- #
_C.AUTO_REID = CN()

# VLM backbone: InternVL3.5-8B
_C.AUTO_REID.VLM_MODEL = "OpenGVLab/InternVL2_5-8B"

# Visual & text encoder: SigLIP2-base-patch16-224
_C.AUTO_REID.VIS_ENCODER = "google/siglip2-base-patch16-224"

# LoRA hyperparameters (rank=16, attention layers only)
_C.AUTO_REID.LORA_RANK = 16
_C.AUTO_REID.LORA_ALPHA = 32
_C.AUTO_REID.LORA_DROPOUT = 0.05
# Target modules for LoRA adaptation (attention projection layers)
_C.AUTO_REID.LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

# Hybrid Retriever mixing coefficient (α=0.65, best performance)
# S^(t)(I_q, I_g) = α·cos(v_q, v_g) + (1-α)·cos(h_q^(t), h_g)
_C.AUTO_REID.ALPHA = 0.65

# Iterative inference loop parameters (Algorithm 1)
_C.AUTO_REID.T_MAX = 3           # maximum correction iterations
_C.AUTO_REID.K = 20              # candidate set size for ACS verification
_C.AUTO_REID.IOU_THRESHOLD = 0.9 # early-stop: IoU(C, C_prev) > 0.9 (Algorithm 1 line 16)
_C.AUTO_REID.TAU_LOW = 0.4       # ACS conflict threshold: ACS(k,v) < τ_low → conflict

# Two-stage evaluation: visual pre-filtering shortlist size
_C.AUTO_REID.TOP_N_PREFILT = 200

# HPT training
_C.AUTO_REID.HPT_STAGE = 1       # 1: fine-grained attribute alignment; 2: multi-task
_C.AUTO_REID.HPT_LR = 1e-5       # LoRA fine-tuning learning rate
_C.AUTO_REID.HPT_EPOCHS = 3      # training epochs per stage
_C.AUTO_REID.HPT_BATCH_SIZE = 4  # per-GPU batch size (4×A100 recommended)

# Checkpoint paths
_C.AUTO_REID.STAGE1_CKPT = ""   # Stage 1 LoRA checkpoint dir (required for Stage 2)
_C.AUTO_REID.VLM_CKPT = ""      # Fine-tuned VLM LoRA checkpoint dir (for inference)
