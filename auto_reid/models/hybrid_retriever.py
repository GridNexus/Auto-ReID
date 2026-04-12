"""
hybrid_retriever.py - Hybrid Retriever module for Auto-ReID.

Implements the Hybrid Retriever ("Anchored Semantic Search"):

    "We employ a dual-encoder scheme that balances semantic guidance with
     visual fidelity. A text encoder f_txt (from CLIP) embeds the textual
     query T_q^(t) into h_q^(t). The similarity between the query and a
     gallery image I_g is a convex combination of visual and textual
     similarities:"

    S^(t)(I_q, I_g) = α · (v_q · v_g)/(||v_q|| ||v_g||)
                    + (1-α) · (h_q^(t) · h_g)/(||h_q^(t)|| ||h_g||)   ... (3)

    where:
        v_g = f_vis(I_g)
        h_g = f_txt(φ(I_g))      φ(I_g) = generic caption or filename
        α = 0.65  (default best value; equivalent to λ1=0.65, λ2=0.35)

Visual encoder: SigLIP2-base-patch16-224 (f_vis) — fixed visual anchor
Text encoder:   SigLIP2-base-patch16-224 (f_txt) — dynamic text queries
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModel

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Anchored Semantic Search retriever.

    The visual anchor v_q = f_vis(I_q) is computed once and remains FIXED
    throughout all correction iterations — this prevents catastrophic drift
    due to erroneous text descriptions.

    The text term h_q^(t) = f_txt(T_q^(t)) is updated each iteration,
    allowing the search focus to be iteratively refined.

    Usage:
        retriever = HybridRetriever(alpha=0.65)
        # Pre-index gallery (done once before the loop)
        retriever.index_gallery(gallery_images, gallery_captions)
        # Compute fixed visual anchor for query
        v_q = retriever.extract_visual_anchor(query_image)
        # Encode dynamic text query (updated each iteration)
        h_q = retriever.encode_text(text_desc)
        # Retrieve top-K candidates
        topk_indices = retriever.retrieve_topk(v_q, h_q, k=20)
        # Full gallery ranking (for final output)
        scores = retriever.compute_scores(v_q, h_q)
    """

    # SigLIP2-base-patch16-224 processes 224×224 images
    _IMG_SIZE = 224

    def __init__(
        self,
        encoder_name: str = "google/siglip2-base-patch16-224",
        alpha: float = 0.65,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            encoder_name: HuggingFace model ID for SigLIP2 visual+text encoder.
            alpha: Modality mixing coefficient α (visual weight). Default 0.65.
                   Text weight = (1 - α) = 0.35.
            device: Device string for torch tensors.
            torch_dtype: Computation dtype.
        """
        self.alpha = alpha
        self.device = device
        self.dtype = torch_dtype

        logger.info("Loading SigLIP2 encoder from %s", encoder_name)
        self.processor = AutoProcessor.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(
            encoder_name,
            torch_dtype=torch_dtype,
        ).to(device)
        self.encoder.eval()

        # Gallery feature banks (populated by index_gallery)
        self._gallery_vis: Optional[torch.Tensor] = None   # [N, D]
        self._gallery_txt: Optional[torch.Tensor] = None   # [N, D]
        self._gallery_size: int = 0

    # ------------------------------------------------------------------
    # Gallery indexing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def index_gallery(
        self,
        images: List[Image.Image],
        captions: Optional[List[str]] = None,
        batch_size: int = 64,
    ) -> None:
        """
        Pre-compute and cache gallery visual features v_g and text features h_g.

        Args:
            images: List of gallery PIL images.
            captions: Optional list of generic captions φ(I_g) or filenames.
                      If None, empty strings are used (text branch uses zeros).
            batch_size: Batch size for encoding (trade-off memory vs speed).
        """
        if captions is None:
            captions = ["" for _ in images]
        assert len(images) == len(captions), \
            "images and captions must have the same length"

        N = len(images)
        self._gallery_size = N
        vis_feats = []
        txt_feats = []

        logger.info("Indexing %d gallery images...", N)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_imgs = images[start:end]
            batch_caps = captions[start:end]

            # Visual features
            vis = self._encode_images(batch_imgs)
            vis_feats.append(vis)

            # Text features from gallery captions/filenames
            txt = self._encode_texts(batch_caps)
            txt_feats.append(txt)

        self._gallery_vis = torch.cat(vis_feats, dim=0)   # [N, D]
        self._gallery_txt = torch.cat(txt_feats, dim=0)   # [N, D]
        logger.info("Gallery indexed: %d images, feature dim=%d",
                    N, self._gallery_vis.shape[-1])

    # ------------------------------------------------------------------
    # Query encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_visual_anchor(self, image: Image.Image) -> torch.Tensor:
        """
        Compute the fixed visual anchor v_q = f_vis(I_q).
        This tensor is computed ONCE and reused across all iterations.

        Returns:
            Normalized visual feature vector of shape [D].
        """
        feat = self._encode_images([image])   # [1, D]
        return feat.squeeze(0)                # [D]

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode a text description into h_q^(t) = f_txt(T_q^(t)).
        Called at each iteration with the updated description.

        Returns:
            Normalized text feature vector of shape [D].
        """
        feat = self._encode_texts([text])    # [1, D]
        return feat.squeeze(0)               # [D]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def compute_scores(
        self,
        v_q: torch.Tensor,
        h_q: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute hybrid similarity scores for all gallery images.

        Implements Equation (3):
            S^(t) = α · cos(v_q, v_g) + (1-α) · cos(h_q^(t), h_g)

        Args:
            v_q: Visual anchor [D], normalized.
            h_q: Text query [D], normalized.

        Returns:
            Scores tensor of shape [N] (one score per gallery image).
        """
        assert self._gallery_vis is not None, \
            "Call index_gallery() before retrieve_topk()"

        # Visual similarity: cos(v_q, v_g) for all gallery images
        # v_q: [D], _gallery_vis: [N, D] → vis_sim: [N]
        vis_sim = torch.mv(self._gallery_vis, v_q)   # [N]  (already normalized)

        # Text similarity: cos(h_q, h_g) for all gallery images
        # h_q: [D], _gallery_txt: [N, D] → txt_sim: [N]
        txt_sim = torch.mv(self._gallery_txt, h_q)   # [N]

        # Convex combination (Equation 3)
        scores = self.alpha * vis_sim + (1.0 - self.alpha) * txt_sim   # [N]
        return scores

    def retrieve_topk(
        self,
        v_q: torch.Tensor,
        h_q: torch.Tensor,
        k: int = 20,
    ) -> List[int]:
        """
        Retrieve indices of the top-K gallery candidates.

        Args:
            v_q: Visual anchor [D].
            h_q: Text query [D].
            k: Number of candidates to return.

        Returns:
            List of gallery indices (0-indexed), sorted by descending score.
        """
        scores = self.compute_scores(v_q, h_q)  # [N]
        k = min(k, self._gallery_size)
        topk_vals, topk_idx = torch.topk(scores, k)
        return topk_idx.cpu().tolist()

    def rank_gallery(
        self,
        v_q: torch.Tensor,
        h_q: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rank the full gallery by hybrid similarity (descending).

        Returns:
            (sorted_indices, scores): numpy arrays of shape [N].
        """
        scores = self.compute_scores(v_q, h_q).cpu().numpy()
        sorted_idx = np.argsort(-scores)    # descending
        return sorted_idx, scores[sorted_idx]

    # ------------------------------------------------------------------
    # Two-stage evaluation support
    # ------------------------------------------------------------------

    @torch.no_grad()
    def visual_prefilt(
        self,
        v_q: torch.Tensor,
        top_n: int = 200,
    ) -> List[int]:
        """
        Stage 1 of two-stage evaluation: fast visual pre-filtering.

        "We compute the similarity between the query and all gallery images
         using E_vis and keep a shortlist (top-N or those above a threshold).
         This step is fast and ensures that the candidate set retains high recall."

        Args:
            v_q: Visual anchor [D].
            top_n: Shortlist size.

        Returns:
            List of gallery indices in the shortlist.
        """
        assert self._gallery_vis is not None
        vis_sim = torch.mv(self._gallery_vis, v_q)   # [N]
        top_n = min(top_n, self._gallery_size)
        _, idx = torch.topk(vis_sim, top_n)
        return idx.cpu().tolist()

    # ------------------------------------------------------------------
    # Low-level encoding helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Encode a batch of PIL images with the SigLIP2 vision encoder.
        Returns L2-normalized features of shape [B, D].
        """
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}

        outputs = self.encoder.get_image_features(**inputs)
        feats = F.normalize(outputs, p=2, dim=-1)
        return feats.to(dtype=self.dtype)

    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a batch of text strings with the SigLIP2 text encoder.
        Returns L2-normalized features of shape [B, D].
        """
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}

        outputs = self.encoder.get_text_features(**inputs)
        feats = F.normalize(outputs, p=2, dim=-1)
        return feats.to(dtype=self.dtype)
