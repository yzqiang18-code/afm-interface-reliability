"""Aggregate model-pair metrics into one ensemble-level consistency result."""

from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .clustering import cluster_contact_sets
from .confidence import IptmSummary, summarize_iptm
from .contacts import Contact, ContactComparison, compare_contact_sets
from .pose import receptor_aligned_ligand_rmsd
from .structure import StructuralModel


@dataclass(frozen=True)
class EnsembleMember:
    seed: int
    model_weight: int
    structure: StructuralModel
    iptm: float


@dataclass(frozen=True)
class PairwiseMetrics:
    first_index: int
    second_index: int
    heavy: ContactComparison
    cb: ContactComparison
    receptor_aligned_ligand_rmsd: float


@dataclass(frozen=True)
class ContactEnsembleMetrics:
    nonempty_count: int
    valid_pair_count: int
    mean_contact_jaccard: float | None
    mean_interface_residue_jaccard_a: float | None
    mean_interface_residue_jaccard_b: float | None
    mean_interface_residue_jaccard: float | None
    max_cluster_fraction: float
    across_seeds_pair_count: int
    mean_across_seeds: float | None
    across_model_weights_pair_count: int
    mean_across_model_weights: float | None
    cluster_ids: tuple[int, ...]
    status: str


@dataclass(frozen=True)
class EnsembleMetrics:
    pairs: tuple[PairwiseMetrics, ...]
    heavy: ContactEnsembleMetrics
    cb: ContactEnsembleMetrics
    pose_pair_count: int
    median_receptor_aligned_ligand_rmsd: float | None
    iptm: IptmSummary
    pose_status: str
    iptm_status: str


def _mean_or_none(values: Sequence[float]) -> float | None:
    # Preserve the previous batch script's NumPy mean semantics for existing
    # contact-Jaccard output columns.
    return float(np.mean(values)) if values else None


def _validate_members(members: Sequence[EnsembleMember]) -> None:
    if not members:
        raise ValueError("An ensemble must contain at least one model")

    reference = members[0].structure
    seen_keys: set[tuple[int, int]] = set()
    for member in members:
        key = (member.seed, member.model_weight)
        if key in seen_keys:
            raise ValueError(
                f"Duplicate ensemble member key: seed={key[0]}, model={key[1]}"
            )
        seen_keys.add(key)
        structure = member.structure
        if (
            structure.sequence_a != reference.sequence_a
            or structure.sequence_b != reference.sequence_b
        ):
            raise ValueError(
                "Chain sequence mismatch across ensemble models: "
                f"{structure.path}"
            )


def _contact_summary(
    members: Sequence[EnsembleMember],
    pairs: Sequence[PairwiseMetrics],
    *,
    contact_sets: Sequence[frozenset[Contact]],
    comparison_name: str,
    distance_threshold: float,
) -> ContactEnsembleMetrics:
    comparisons = [
        getattr(pair, comparison_name)
        for pair in pairs
        if getattr(pair, comparison_name).valid
    ]
    contact_values = [
        comparison.contact_jaccard
        for comparison in comparisons
        if comparison.contact_jaccard is not None
    ]
    residue_values_a = [
        comparison.interface_residue_jaccard_a
        for comparison in comparisons
        if comparison.interface_residue_jaccard_a is not None
    ]
    residue_values_b = [
        comparison.interface_residue_jaccard_b
        for comparison in comparisons
        if comparison.interface_residue_jaccard_b is not None
    ]
    residue_values = [
        comparison.interface_residue_jaccard
        for comparison in comparisons
        if comparison.interface_residue_jaccard is not None
    ]
    if not (
        len(comparisons)
        == len(contact_values)
        == len(residue_values_a)
        == len(residue_values_b)
        == len(residue_values)
    ):
        raise ValueError("A valid contact comparison is missing a similarity value")

    across_seeds: list[float] = []
    across_model_weights: list[float] = []
    for pair in pairs:
        comparison = getattr(pair, comparison_name)
        if not comparison.valid or comparison.contact_jaccard is None:
            continue
        first = members[pair.first_index]
        second = members[pair.second_index]
        if first.model_weight == second.model_weight:
            across_seeds.append(comparison.contact_jaccard)
        if first.seed == second.seed:
            across_model_weights.append(comparison.contact_jaccard)

    cluster = cluster_contact_sets(contact_sets, distance_threshold)
    nonempty_count = sum(bool(contacts) for contacts in contact_sets)
    return ContactEnsembleMetrics(
        nonempty_count=nonempty_count,
        valid_pair_count=len(comparisons),
        mean_contact_jaccard=_mean_or_none(contact_values),
        mean_interface_residue_jaccard_a=_mean_or_none(residue_values_a),
        mean_interface_residue_jaccard_b=_mean_or_none(residue_values_b),
        mean_interface_residue_jaccard=_mean_or_none(residue_values),
        max_cluster_fraction=cluster.max_cluster_fraction,
        across_seeds_pair_count=len(across_seeds),
        mean_across_seeds=_mean_or_none(across_seeds),
        across_model_weights_pair_count=len(across_model_weights),
        mean_across_model_weights=_mean_or_none(across_model_weights),
        cluster_ids=cluster.cluster_ids,
        status="ok" if nonempty_count >= 2 else "insufficient_nonempty_models",
    )


def analyze_ensemble(
    members: Sequence[EnsembleMember],
    *,
    cluster_distance_threshold: float,
) -> EnsembleMetrics:
    """Calculate all pairwise and ensemble-level consistency metrics."""
    _validate_members(members)

    pairs: list[PairwiseMetrics] = []
    for first_index, second_index in itertools.combinations(range(len(members)), 2):
        first = members[first_index]
        second = members[second_index]
        pairs.append(
            PairwiseMetrics(
                first_index=first_index,
                second_index=second_index,
                heavy=compare_contact_sets(
                    first.structure.contact_maps.heavy,
                    second.structure.contact_maps.heavy,
                ),
                cb=compare_contact_sets(
                    first.structure.contact_maps.cb,
                    second.structure.contact_maps.cb,
                ),
                receptor_aligned_ligand_rmsd=receptor_aligned_ligand_rmsd(
                    first.structure.receptor_ca_coordinates,
                    first.structure.ligand_ca_coordinates,
                    second.structure.receptor_ca_coordinates,
                    second.structure.ligand_ca_coordinates,
                ),
            )
        )

    heavy_sets = [member.structure.contact_maps.heavy for member in members]
    cb_sets = [member.structure.contact_maps.cb for member in members]
    pose_values = [pair.receptor_aligned_ligand_rmsd for pair in pairs]
    return EnsembleMetrics(
        pairs=tuple(pairs),
        heavy=_contact_summary(
            members,
            pairs,
            contact_sets=heavy_sets,
            comparison_name="heavy",
            distance_threshold=cluster_distance_threshold,
        ),
        cb=_contact_summary(
            members,
            pairs,
            contact_sets=cb_sets,
            comparison_name="cb",
            distance_threshold=cluster_distance_threshold,
        ),
        pose_pair_count=len(pose_values),
        median_receptor_aligned_ligand_rmsd=(
            float(statistics.median(pose_values)) if pose_values else None
        ),
        iptm=summarize_iptm([member.iptm for member in members]),
        pose_status="ok" if pose_values else "insufficient_models",
        iptm_status="ok",
    )
