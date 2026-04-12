import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ..models.reasoner import Reasoner
from ..models.hybrid_retriever import HybridRetriever
from ..models.corrector import Corrector

logger = logging.getLogger(__name__)


class AutoReIDPipeline:

    def __init__(
        self,
        reasoner: Reasoner,
        retriever: HybridRetriever,
        corrector: Corrector,
        t_max: int = 3,
        k: int = 20,
        iou_threshold: float = 0.9,
        top_n_prefilt: int = 200,
    ):
        
        self.reasoner = reasoner
        self.retriever = retriever
        self.corrector = corrector
        self.t_max = t_max
        self.k = k
        self.iou_threshold = iou_threshold
        self.top_n_prefilt = top_n_prefilt

    # ------------------------------------------------------------------
    # Main entry point: process one query image against an indexed gallery
    # ------------------------------------------------------------------

    def run_query(
        self,
        query_image: Image.Image,
        gallery_images: List[Image.Image],
        gallery_pids: List[int],
        gallery_camids: List[int],
        query_pid: int,
        query_camid: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        N = len(gallery_images)

        # ── Algorithm 1, Line 1: Visual anchor (fixed throughout loop) ──
        logger.debug("Step 1: Extracting visual anchor v_q")
        v_q = self.retriever.extract_visual_anchor(query_image)

        # ── Algorithm 1, Line 2: Initial description T_q^(0) ─────────
        logger.debug("Step 2: Generating initial description T_q^(0)")
        T_q = self.reasoner.initial_description(query_image)
        h_q = self.retriever.encode_text(T_q)
        logger.debug("T_q^(0): %s", T_q[:120])

        # ── Two-Stage Evaluation Stage 1: Visual pre-filtering ────────
        # Compute visual similarity to shortlist top-N candidates (fast, high recall)
        if self.top_n_prefilt < N:
            logger.debug("Visual pre-filtering: shortlisting top-%d", self.top_n_prefilt)
            shortlist_idx = self.retriever.visual_prefilt(v_q, self.top_n_prefilt)
            shortlist_images = [gallery_images[i] for i in shortlist_idx]
        else:
            shortlist_idx = list(range(N))
            shortlist_images = gallery_images

        # ── Algorithm 1, Line 3: C_prev = ∅ ──────────────────────────
        C_prev: set = set()

        # ── Algorithm 1, Lines 4-22: Main correction loop ─────────────
        # Convert shortlist_idx to tensor once for efficient GPU indexing
        sl_idx_t = torch.tensor(
            shortlist_idx,
            dtype=torch.long,
            device=self.retriever._gallery_vis.device,
        )

        for t in range(self.t_max):
            logger.debug("Iteration t=%d", t)

            # Lines 5-9: Hybrid retrieval (within shortlist)
            # Equation (3): S = α·vis_sim + (1-α)·txt_sim
            sl_vis = self.retriever._gallery_vis[sl_idx_t]   # [M, D]
            sl_txt = self.retriever._gallery_txt[sl_idx_t]   # [M, D]

            vis_sim = torch.mv(sl_vis, v_q)                   # [M]
            txt_sim = torch.mv(sl_txt, h_q)                   # [M]
            scores = self.retriever.alpha * vis_sim + \
                     (1.0 - self.retriever.alpha) * txt_sim   # [M]

            k_eff = min(self.k, len(shortlist_idx))
            _, topk_local_idx = torch.topk(scores, k_eff)
            C_local = topk_local_idx.cpu().tolist()           # local indices within shortlist
            C = [shortlist_idx[i] for i in C_local]           # global gallery indices

            # Lines 10-12: Last iteration → return full ranking
            if t == self.t_max - 1:
                logger.debug("Final iteration: returning full gallery ranking")
                return self._full_rank(v_q, h_q, sl_idx_t, N)

            # Line 14: Semantic Deconstruction A ← Parse(T_q)
            A = self.corrector.parse_to_attributes(T_q)
            logger.debug("Parsed attributes: %s", A)

            # Line 15: Verify F ← Verify(C, A, τ_low)
            candidate_images = [gallery_images[i] for i in C]
            acs_scores = self.corrector.compute_acs(candidate_images, A)
            logger.debug("ACS scores: %s", acs_scores)

            # Line 16-18: Early termination
            F = self.corrector.generate_feedback(A, acs_scores)
            C_set = set(C)
            if F is None and self._iou(C_set, C_prev) > self.iou_threshold:
                logger.debug("Early termination at t=%d (no conflict, IoU=%.3f)",
                             t, self._iou(C_set, C_prev))
                return self._full_rank(v_q, h_q, shortlist_idx, N)

            # Line 20: T_q ← M_ReID(I_q, T_q, F)
            if F is not None:
                logger.debug("Feedback: %s", F)
                T_q = self.reasoner.refine_description(query_image, T_q, F)
                h_q = self.retriever.encode_text(T_q)
                logger.debug("T_q^(%d): %s", t + 1, T_q[:120])

            # Line 21: C_prev ← C
            C_prev = C_set

        # Line 23: return C (full ranking based on final T_q)
        return self._full_rank(v_q, h_q, shortlist_idx, N)

    # ------------------------------------------------------------------
    # Batch evaluation (run all queries in a dataset)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        query_images: List[Image.Image],
        query_pids: List[int],
        query_camids: List[int],
        gallery_images: List[Image.Image],
        gallery_pids: List[int],
        gallery_camids: List[int],
        max_rank: int = 50,
    ) -> Dict[str, float]:
        
        from utils.metrics import eval_func

        num_q = len(query_images)
        num_g = len(gallery_images)

        all_sorted_idx = []
        all_scores = []

        logger.info("Running Auto-ReID on %d queries vs %d gallery images",
                    num_q, num_g)

        for q_idx, (qimg, qpid, qcam) in enumerate(
                zip(query_images, query_pids, query_camids)):
            logger.info("Query %d/%d (pid=%d)", q_idx + 1, num_q, qpid)
            sorted_idx, scores = self.run_query(
                query_image=qimg,
                gallery_images=gallery_images,
                gallery_pids=gallery_pids,
                gallery_camids=gallery_camids,
                query_pid=qpid,
                query_camid=qcam,
            )
            all_sorted_idx.append(sorted_idx)
            all_scores.append(scores)

        # Build distance matrix (negate scores → lower = better match)
        # Shape: [num_q, num_g]
        dist_mat = np.zeros((num_q, num_g), dtype=np.float32)
        for q_idx, (sorted_idx, scores) in enumerate(
                zip(all_sorted_idx, all_scores)):
            for rank_pos, g_idx in enumerate(sorted_idx):
                dist_mat[q_idx, g_idx] = -scores[rank_pos]

        q_pids = np.array(query_pids)
        g_pids = np.array(gallery_pids)
        q_camids = np.array(query_camids)
        g_camids = np.array(gallery_camids)

        cmc, mAP = eval_func(dist_mat, q_pids, g_pids, q_camids, g_camids,
                              max_rank=max_rank)

        results = {
            'mAP': float(mAP * 100),
            'rank1': float(cmc[0] * 100),
            'rank5': float(cmc[4] * 100) if max_rank >= 5 else -1.0,
            'rank10': float(cmc[9] * 100) if max_rank >= 10 else -1.0,
        }
        logger.info("Results: mAP=%.1f%%, Rank-1=%.1f%%",
                    results['mAP'], results['rank1'])
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iou(self, c1: set, c2: set) -> float:
        
        if not c1 and not c2:
            return 1.0
        if not c1 or not c2:
            return 0.0
        return len(c1 & c2) / len(c1 | c2)

    def _full_rank(
        self,
        v_q: torch.Tensor,
        h_q: torch.Tensor,
        shortlist_idx: List[int],
        total_gallery_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        sl_vis = self.retriever._gallery_vis[shortlist_idx]
        sl_txt = self.retriever._gallery_txt[shortlist_idx]

        vis_sim = torch.mv(sl_vis, v_q).cpu().numpy()
        txt_sim = torch.mv(sl_txt, h_q).cpu().numpy()
        sl_scores = self.retriever.alpha * vis_sim + \
                    (1.0 - self.retriever.alpha) * txt_sim

        # Build full score array (non-shortlist gets minimum score)
        full_scores = np.full(total_gallery_size, -1e9, dtype=np.float32)
        for local_i, global_i in enumerate(shortlist_idx):
            full_scores[global_i] = sl_scores[local_i]

        sorted_idx = np.argsort(-full_scores)   # descending
        return sorted_idx, full_scores[sorted_idx]
