
# Auto-ReID: Iterative Self-Correction for Text-Driven Person Re-Identification with Large Vision-Language Models

Implementation of Auto-ReID, a closed-loop iterative inference framework for text-driven person re-identification using large vision-language models.

## Overview

Auto-ReID addresses a fundamental limitation of existing ReID methods: the inability to self-correct retrieval errors during inference. Our method introduces an iterative closed-loop pipeline with three core modules:

- **Reasoner**: Generates and refines structured textual descriptions of the query person using a fine-tuned VLM (InternVL3.5-8B).
- **Hybrid Retriever**: Combines visual and textual similarity via `S = α·cos(v_q, v_g) + (1-α)·cos(h_q, h_g)` with α=0.65 (SigLIP2 encoder).
- **Corrector**: Performs Semantic Deconstruction → Attribute Consistency Verification (ACS) → Feedback Generation to refine the next query description.

The iterative loop (Algorithm 1) runs for up to T_max=3 iterations with early termination when IoU(C, C_prev) > 0.9 and no attribute conflicts are detected.



## Requirements

### Installation

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch >= 1.10.0
- transformers >= 4.37.0 (InternVL3.5-8B)
- peft >= 0.8.0 (LoRA fine-tuning)
- accelerate >= 0.27.0

Recommended hardware: 4x NVIDIA A100 (80GB) GPUs.

### Prepare Datasets

```bash
mkdir data
```

Download and organize datasets under `data/`:

```
data/
├── market1501/
│   └── Market-1501-v15.09.15/
│       ├── bounding_box_train/
│       ├── bounding_box_test/
│       └── query/
├── MSMT17/
│   └── MSMT17_V1/
│       ├── train/
│       └── test/
├── cuhk03/
│   ├── detected/
│   └── labeled/
├── Occluded_Duke/
│   ├── bounding_box_train/
│   ├── bounding_box_test/
│   └── query/
├── PRCC/
│   ├── rgb/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── ...
└── LTCC/
    ├── train/
    ├── test/
    └── query/
```

- [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)
- [MSMT17](https://arxiv.org/abs/1711.08565)
- [CUHK03](https://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html) — new protocol split (767/700)
- [Occluded-Duke](https://github.com/lightas/Occluded-DukeMTMC-Dataset)
- [PRCC](https://drive.google.com/file/d/1yTYawRm4ap3M-j0PjLQJ--xmZHseFDLz/view)
- [LTCC](https://naiq.github.io/LTCC_Perosn_ReID.html)

### Prepare VLM Backbone

The Reasoner and Corrector use **InternVL3.5-8B** (`OpenGVLab/InternVL2_5-8B`).
The HybridRetriever uses **SigLIP2** (`google/siglip2-base-patch16-224`).

Both models are automatically downloaded from Hugging Face on first use.

## Training (HPT: Hierarchical Progressive Tuning)

Auto-ReID fine-tunes InternVL3.5-8B with LoRA (rank=16, lr=1e-5, 3 epochs) in two stages.

### HPT Stage 1: Fine-grained Attribute Alignment

Trains the VLM to generate structured person descriptions from the structured prompt P_struct.

```bash
# MSMT17
python train_auto_reid.py \
    --config configs/AutoReID/msmt17.yml \
    AUTO_REID.HPT_STAGE 1 \
    OUTPUT_DIR output/autored_msmt17/hpt_stage1

# Market-1501
python train_auto_reid.py \
    --config configs/AutoReID/market1501.yml \
    AUTO_REID.HPT_STAGE 1 \
    OUTPUT_DIR output/autored_market/hpt_stage1
```

### HPT Stage 2: Multi-task Identity Verification

Trains 7 tasks (Figure 2 in paper): attribute matching, difference mining, image-to-image matching, image-to-images retrieval, image-to-texts retrieval, text-to-image matching, text-to-images retrieval.

```bash
# MSMT17
python train_auto_reid.py \
    --config configs/AutoReID/msmt17.yml \
    AUTO_REID.HPT_STAGE 2 \
    AUTO_REID.STAGE1_CKPT output/autored_msmt17/hpt_stage1/stage1_lora \
    OUTPUT_DIR output/autored_msmt17/hpt_stage2

# Market-1501
python train_auto_reid.py \
    --config configs/AutoReID/market1501.yml \
    AUTO_REID.HPT_STAGE 2 \
    AUTO_REID.STAGE1_CKPT output/autored_market/hpt_stage1/stage1_lora \
    OUTPUT_DIR output/autored_market/hpt_stage2
```

## Evaluation

Auto-ReID uses a two-stage evaluation strategy:
1. **Visual pre-filtering**: Use SigLIP2 visual features to shortlist top-200 gallery candidates (fast, high recall).
2. **Closed-loop refinement**: Run Algorithm 1 on the shortlist (T_max=3 iterations, K=20 candidates for ACS).

```bash
# MSMT17  (target: mAP=89.2%, Rank-1=91.8%)
python inference_auto_reid.py \
    --config configs/AutoReID/msmt17.yml \
    AUTO_REID.VLM_CKPT output/autored_msmt17/hpt_stage2/stage2_lora \
    TEST.WEIGHT ""

# Market-1501  (target: mAP=97.1%, Rank-1=97.8%)
python inference_auto_reid.py \
    --config configs/AutoReID/market1501.yml \
    AUTO_REID.VLM_CKPT output/autored_market/hpt_stage2/stage2_lora \
    TEST.WEIGHT ""

# CUHK03  (target: mAP=92.4%, Rank-1=93.1%)
python inference_auto_reid.py \
    --config configs/AutoReID/cuhk03.yml \
    AUTO_REID.VLM_CKPT output/autored_cuhk03/hpt_stage2/stage2_lora \
    TEST.WEIGHT ""

# Occluded-Duke  (target: mAP=72.4%, Rank-1=79.5%)
python inference_auto_reid.py \
    --config configs/AutoReID/occ_duke.yml \
    AUTO_REID.VLM_CKPT output/autored_occ_duke/hpt_stage2/stage2_lora \
    TEST.WEIGHT ""

# LTCC  (target: mAP=66.3%, Rank-1=79.7%)
python inference_auto_reid.py \
    --config configs/AutoReID/ltcc.yml \
    AUTO_REID.VLM_CKPT output/autored_ltcc/hpt_stage2/stage2_lora \
    TEST.WEIGHT ""

# PRCC  (target: mAP=65.6%, Rank-1=73.2%)
python inference_auto_reid.py \
    --config configs/AutoReID/prcc.yml \
    AUTO_REID.VLM_CKPT output/autored_prcc/hpt_stage2/stage2_lora \
    TEST.WEIGHT ""
```

