"""Shared strict-distance geometry helpers."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree


def strict_neighbor_pairs(
    coordinates_a: np.ndarray,
    coordinates_b: np.ndarray,
    cutoff: float,
) -> tuple[tuple[int, int], ...]:
    """Return atom-index pairs whose Euclidean distance is strictly below cutoff."""
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Distance cutoff must be finite and positive")
    if coordinates_a.ndim != 2 or coordinates_a.shape[1:] != (3,):
        raise ValueError("coordinates_a must have shape (N, 3)")
    if coordinates_b.ndim != 2 or coordinates_b.shape[1:] != (3,):
        raise ValueError("coordinates_b must have shape (N, 3)")
    if not np.isfinite(coordinates_a).all() or not np.isfinite(coordinates_b).all():
        raise ValueError("Coordinates must be finite")
    if len(coordinates_a) == 0 or len(coordinates_b) == 0:
        return tuple()

    cutoff_squared = cutoff * cutoff
    tree_b = cKDTree(coordinates_b)
    pairs: list[tuple[int, int]] = []
    for index_a, neighbors in enumerate(
        tree_b.query_ball_point(coordinates_a, r=cutoff)
    ):
        for index_b in neighbors:
            delta = coordinates_a[index_a] - coordinates_b[index_b]
            if float(np.dot(delta, delta)) < cutoff_squared:
                pairs.append((index_a, int(index_b)))
    return tuple(pairs)
