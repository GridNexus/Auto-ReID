#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-reciprocal re-ranking algorithm for person re-identification.

This module implements a re-ranking approach that enhances distance metrics
by leveraging k-reciprocal neighbor relationships.
"""

import numpy as np
import torch


def rerank_distance(query_features, gallery_features, k1=20, k2=6, lambda_val=0.3):
    """
    Re-rank distances using k-reciprocal encoding.

    Args:
        query_features: Feature vectors of query samples (torch tensor)
        gallery_features: Feature vectors of gallery samples (torch tensor)
        k1: Size of initial k-reciprocal neighborhood
        k2: Size of expanded neighborhood for query expansion
        lambda_val: Balance weight between Jaccard and original distance
    """
    query_count = query_features.size(0)
    total_count = query_count + gallery_features.size(0)

    # Compute pairwise Euclidean distance
    all_features = torch.cat([query_features, gallery_features])
    sq_dists = torch.pow(all_features, 2).sum(dim=1, keepdim=True).expand(total_count, total_count)
    dist_matrix = sq_dists + sq_dists.t()
    dist_matrix.addmm_(1, -2, all_features, all_features.t())
    dist_matrix = dist_matrix.cpu().numpy()

    # Normalize by column max
    dist_matrix = dist_matrix / np.max(dist_matrix, axis=0)
    dist_matrix = dist_matrix.T

    # Initialize encoding matrix and ranking
    encoding_matrix = np.zeros_like(dist_matrix, dtype=np.float16)
    rank_matrix = np.argsort(dist_matrix).astype(np.int32)

    # Build k-reciprocal encoding
    for i in range(total_count):
        # Forward: get k1 nearest neighbors
        forward_neighbors = rank_matrix[i, :k1 + 1]

        # Backward: check who also has i in their k1 neighbors
        backward_neighbors = rank_matrix[forward_neighbors, :k1 + 1]
        matches = np.where(backward_neighbors == i)[0]

        # k-reciprocal set
        reciprocal_set = forward_neighbors[matches]
        expanded_set = reciprocal_set.copy()

        # Expand reciprocal set
        half_k = int(np.around(k1 / 2)) + 1
        for idx in range(len(reciprocal_set)):
            candidate = reciprocal_set[idx]
            cand_neighbors = rank_matrix[candidate, :half_k]
            cand_backward = rank_matrix[cand_neighbors, :half_k]
            cand_matches = np.where(cand_backward == candidate)[0]
            cand_reciprocal = cand_neighbors[cand_matches]

            # Merge if overlap exceeds threshold
            overlap_ratio = len(np.intersect1d(cand_reciprocal, reciprocal_set)) / len(cand_reciprocal)
            if overlap_ratio > 2 / 3:
                expanded_set = np.append(expanded_set, cand_reciprocal)

        # Deduplicate and compute weights
        expanded_set = np.unique(expanded_set)
        weights = np.exp(-dist_matrix[i, expanded_set])
        encoding_matrix[i, expanded_set] = weights / weights.sum()

    # Query expansion
    if k2 != 1:
        expanded_enc = np.zeros_like(encoding_matrix, dtype=np.float16)
        for i in range(total_count):
            neighbors = rank_matrix[i, :k2]
            expanded_enc[i, :] = np.mean(encoding_matrix[neighbors, :], axis=0)
        encoding_matrix = expanded_enc

    # Build inverted index
    inv_index = []
    for i in range(dist_matrix.shape[0]):
        inv_index.append(np.where(encoding_matrix[:, i] != 0)[0])

    # Compute Jaccard distance
    jaccard_dist = np.zeros((query_count, dist_matrix.shape[0]), dtype=np.float16)

    for i in range(query_count):
        min_dist = np.zeros(shape=[1, dist_matrix.shape[0]], dtype=np.float16)
        nonzero_indices = np.where(encoding_matrix[i, :] != 0)[0]

        for j in range(len(nonzero_indices)):
            gallery_idx = nonzero_indices[j]
            min_dist[0, inv_index[gallery_idx]] += np.minimum(
                encoding_matrix[i, gallery_idx],
                encoding_matrix[inv_index[gallery_idx], gallery_idx]
            )

        jaccard_dist[i] = 1 - min_dist[0, :] / (2 - min_dist[0, :])

    # Combine distances
    original_dist = dist_matrix[:query_count, :]
    final_dist = jaccard_dist * (1 - lambda_val) + original_dist * lambda_val

    return final_dist[:query_count, query_count:]