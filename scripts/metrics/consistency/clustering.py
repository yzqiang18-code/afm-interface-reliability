"""Average-linkage clustering of ensemble contact maps."""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from .contacts import Contact, contact_jaccard


@dataclass(frozen=True)
class ClusterResult:
    cluster_ids: tuple[int, ...]
    max_cluster_fraction: float


def cluster_contact_sets(
    contact_sets: Sequence[frozenset[Contact]],
    distance_threshold: float,
) -> ClusterResult:
    """Cluster non-empty maps; every empty map remains a noise singleton."""
    if not math.isfinite(distance_threshold) or not 0 <= distance_threshold <= 1:
        raise ValueError("Cluster distance threshold must be between 0 and 1")

    total_count = len(contact_sets)
    if total_count == 0:
        return ClusterResult(cluster_ids=tuple(), max_cluster_fraction=0.0)

    nonempty_indices = [
        index for index, contacts in enumerate(contact_sets) if contacts
    ]
    labels = [-1] * total_count
    if len(nonempty_indices) == 1:
        labels[nonempty_indices[0]] = 1
    elif len(nonempty_indices) >= 2:
        condensed_distances: list[float] = []
        for first, second in itertools.combinations(nonempty_indices, 2):
            similarity = contact_jaccard(
                contact_sets[first],
                contact_sets[second],
            )
            if similarity is None:
                raise ValueError("Non-empty contact maps produced an invalid Jaccard")
            condensed_distances.append(1.0 - similarity)

        raw_labels = fcluster(
            linkage(np.asarray(condensed_distances, dtype=float), method="average"),
            t=distance_threshold,
            criterion="distance",
        )

        # SciPy cluster labels are arbitrary. Canonicalize by each cluster's
        # first ensemble position so model-level audit CSVs are deterministic.
        members_by_raw_label: dict[int, list[int]] = defaultdict(list)
        for index, raw_label in zip(nonempty_indices, raw_labels):
            members_by_raw_label[int(raw_label)].append(index)
        ordered_raw_labels = sorted(
            members_by_raw_label,
            key=lambda label: min(members_by_raw_label[label]),
        )
        canonical_labels = {
            raw_label: canonical
            for canonical, raw_label in enumerate(ordered_raw_labels, start=1)
        }
        for index, raw_label in zip(nonempty_indices, raw_labels):
            labels[index] = canonical_labels[int(raw_label)]

    nonempty_cluster_sizes = Counter(label for label in labels if label != -1)
    largest_nonempty_cluster = max(nonempty_cluster_sizes.values(), default=0)
    largest_noise_singleton = 1 if len(nonempty_indices) < total_count else 0
    largest_cluster = max(largest_nonempty_cluster, largest_noise_singleton)
    return ClusterResult(
        cluster_ids=tuple(labels),
        max_cluster_fraction=largest_cluster / total_count,
    )
