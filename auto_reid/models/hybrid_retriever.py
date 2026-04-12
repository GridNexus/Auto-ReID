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

    # SigLIP2-base-patch16-224 processes 224×224 images
    _IMG_SIZE = 224

    def __init__(
        self,
        encoder_name: str = "google/siglip2-base-patch16-224",
        alpha: float = 0.65,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float32,
    ):
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
        scores = self.compute_scores(v_q, h_q)  # [N]
        k = min(k, self._gallery_size)
        topk_vals, topk_idx = torch.topk(scores, k)
        return topk_idx.cpu().tolist()

    def rank_gallery(
        self,
        v_q: torch.Tensor,
        h_q: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
