"""
Auto-ReID: Iterative Self-Correction for Text-Driven Person Re-Identification
with Large Vision-Language Models.

Package structure:
    auto_reid.models.reasoner          - VLM-based structured description generator
    auto_reid.models.hybrid_retriever  - Visual anchor + dynamic text retriever
    auto_reid.models.corrector         - ACS verification and feedback generator
    auto_reid.inference.pipeline       - Algorithm 1: full iterative inference loop
    auto_reid.training.hpt_dataset     - HPT training data (Stage 1 & 2)
    auto_reid.training.hpt_trainer     - HPT two-stage LoRA fine-tuning
    auto_reid.utils.attribute_parser   - Text → attribute key-value parsing
"""
