#!/usr/bin/env python3
"""Calculate modular AF-M ensemble-consistency metrics for dimer predictions."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from tqdm import tqdm

# Keep numerical libraries conservative by default on the shared server. An
# explicitly supplied process environment still takes precedence.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

from common import (  # noqa: E402
    PredictionFiles,
    discover_predictions,
    prediction_metadata,
    read_ids,
)
from consistency import (  # noqa: E402
    EnsembleMember,
    PairwiseMetrics,
    analyze_ensemble,
    extract_structural_model,
    interface_residue_sets,
    read_iptm,
)


SUMMARY_FIELDS = [
    "complex_id",
    "n_models",
    "expected_model_count",
    "ensemble_complete",
    "missing_model_keys",
    "unexpected_model_keys",
    "duplicate_model_keys",
    "nonempty_model_count",
    "nonempty_model_fraction",
    "valid_pair_count",
    "mean_contact_jaccard",
    "mean_interface_residue_jaccard",
    "mean_interface_residue_jaccard_chain_a",
    "mean_interface_residue_jaccard_chain_b",
    "max_interface_cluster_fraction",
    "across_seeds_pair_count",
    "mean_contact_jaccard_across_seeds",
    "across_model_weights_pair_count",
    "mean_contact_jaccard_across_model_weights",
    "nonempty_model_count_cb8",
    "nonempty_model_fraction_cb8",
    "valid_pair_count_cb8",
    "mean_contact_jaccard_cb8",
    "mean_interface_residue_jaccard_cb8",
    "mean_interface_residue_jaccard_chain_a_cb8",
    "mean_interface_residue_jaccard_chain_b_cb8",
    "max_interface_cluster_fraction_cb8",
    "across_seeds_pair_count_cb8",
    "mean_contact_jaccard_across_seeds_cb8",
    "across_model_weights_pair_count_cb8",
    "mean_contact_jaccard_across_model_weights_cb8",
    "pose_pair_count",
    "median_receptor_aligned_ligand_rmsd",
    "iptm_model_count",
    "iptm_mean_across_models",
    "iptm_std_across_models",
    "heavy5_status",
    "cb8_status",
    "pose_status",
    "iptm_status",
    "heavy_atom_cutoff_angstrom",
    "cb_cutoff_angstrom",
    "cluster_distance_threshold",
    "model_chains",
    "pose_alignment_chain",
    "pose_rmsd_chain",
    "pose_atom_selection",
    "iptm_std_ddof",
    "expected_seeds",
    "expected_model_weights",
    "status",
    "error",
]

MODEL_FIELDS = [
    "complex_id",
    "rank",
    "model_family",
    "model_weight",
    "seed",
    "iptm",
    "chain_a_length",
    "chain_b_length",
    "contact_count",
    "interface_residue_count_a",
    "interface_residue_count_b",
    "cluster_id",
    "contact_count_cb8",
    "interface_residue_count_a_cb8",
    "interface_residue_count_b_cb8",
    "cluster_id_cb8",
    "prediction_path",
    "scores_path",
    "status",
    "error",
]

PAIR_FIELDS = [
    "complex_id",
    "seed_1",
    "model_weight_1",
    "rank_1",
    "prediction_path_1",
    "seed_2",
    "model_weight_2",
    "rank_2",
    "prediction_path_2",
    "same_seed",
    "same_model_weight",
    "jaccard",
    "jaccard_valid",
    "jaccard_reason",
    "interface_residue_jaccard_a",
    "interface_residue_jaccard_b",
    "interface_residue_jaccard",
    "jaccard_cb8",
    "jaccard_cb8_valid",
    "jaccard_cb8_reason",
    "interface_residue_jaccard_a_cb8",
    "interface_residue_jaccard_b_cb8",
    "interface_residue_jaccard_cb8",
    "receptor_aligned_ligand_rmsd",
]


@dataclass(frozen=True)
class AnalysisConfig:
    model_chains: tuple[str, str]
    heavy_atom_cutoff: float
    cb_cutoff: float
    cluster_distance_threshold: float
    expected_seeds: tuple[int, ...]
    expected_model_weights: tuple[int, ...]


@dataclass(frozen=True)
class SystemResult:
    summary: dict[str, object]
    model_rows: list[dict[str, object]]
    pair_rows: list[dict[str, object]]


def format_model_keys(keys: Iterable[tuple[int, int]]) -> str:
    return ";".join(f"seed_{seed}:model_{weight}" for seed, weight in sorted(keys))


def _summary_base(
    complex_id: str,
    predictions: Sequence[PredictionFiles],
    config: AnalysisConfig,
) -> dict[str, object]:
    observed_keys = [(item.seed, item.model_weight) for item in predictions]
    observed_counter = Counter(observed_keys)
    expected_keys = {
        (seed, weight)
        for seed in config.expected_seeds
        for weight in config.expected_model_weights
    }
    observed_key_set = set(observed_keys)
    missing = expected_keys - observed_key_set
    unexpected = observed_key_set - expected_keys
    duplicates = {key for key, count in observed_counter.items() if count > 1}
    complete = not missing and not unexpected and not duplicates
    return {
        "complex_id": complex_id,
        "n_models": len(predictions),
        "expected_model_count": len(expected_keys),
        "ensemble_complete": complete,
        "missing_model_keys": format_model_keys(missing),
        "unexpected_model_keys": format_model_keys(unexpected),
        "duplicate_model_keys": format_model_keys(duplicates),
        "heavy_atom_cutoff_angstrom": config.heavy_atom_cutoff,
        "cb_cutoff_angstrom": config.cb_cutoff,
        "cluster_distance_threshold": config.cluster_distance_threshold,
        "model_chains": "".join(config.model_chains),
        "pose_alignment_chain": config.model_chains[0],
        "pose_rmsd_chain": config.model_chains[1],
        "pose_atom_selection": "CA",
        "iptm_std_ddof": 0,
        "expected_seeds": ";".join(map(str, config.expected_seeds)),
        "expected_model_weights": ";".join(map(str, config.expected_model_weights)),
        "status": "failed",
        "error": "",
    }


def _pair_row(
    complex_id: str,
    predictions: Sequence[PredictionFiles],
    pair: PairwiseMetrics,
) -> dict[str, object]:
    first = predictions[pair.first_index]
    second = predictions[pair.second_index]
    return {
        "complex_id": complex_id,
        "seed_1": first.seed,
        "model_weight_1": first.model_weight,
        "rank_1": first.rank,
        "prediction_path_1": str(first.pdb_path),
        "seed_2": second.seed,
        "model_weight_2": second.model_weight,
        "rank_2": second.rank,
        "prediction_path_2": str(second.pdb_path),
        "same_seed": first.seed == second.seed,
        "same_model_weight": first.model_weight == second.model_weight,
        "jaccard": pair.heavy.contact_jaccard,
        "jaccard_valid": pair.heavy.valid,
        "jaccard_reason": pair.heavy.reason,
        "interface_residue_jaccard_a": (
            pair.heavy.interface_residue_jaccard_a
        ),
        "interface_residue_jaccard_b": (
            pair.heavy.interface_residue_jaccard_b
        ),
        "interface_residue_jaccard": (
            pair.heavy.interface_residue_jaccard
        ),
        "jaccard_cb8": pair.cb.contact_jaccard,
        "jaccard_cb8_valid": pair.cb.valid,
        "jaccard_cb8_reason": pair.cb.reason,
        "interface_residue_jaccard_a_cb8": (
            pair.cb.interface_residue_jaccard_a
        ),
        "interface_residue_jaccard_b_cb8": (
            pair.cb.interface_residue_jaccard_b
        ),
        "interface_residue_jaccard_cb8": (
            pair.cb.interface_residue_jaccard
        ),
        "receptor_aligned_ligand_rmsd": (
            pair.receptor_aligned_ligand_rmsd
        ),
    }


def analyze_system(
    complex_id: str,
    predictions: Sequence[PredictionFiles],
    config: AnalysisConfig,
) -> SystemResult:
    summary = _summary_base(complex_id, predictions, config)
    if not summary["ensemble_complete"]:
        summary["error"] = "Incomplete or non-unique ensemble model keys"
        return SystemResult(summary=summary, model_rows=[], pair_rows=[])

    ordered_predictions = sorted(
        predictions,
        key=lambda item: (
            item.seed,
            item.model_weight,
            item.rank,
            str(item.pdb_path),
        ),
    )
    try:
        members: list[EnsembleMember] = []
        for prediction in ordered_predictions:
            members.append(
                EnsembleMember(
                    seed=prediction.seed,
                    model_weight=prediction.model_weight,
                    structure=extract_structural_model(
                        prediction.pdb_path,
                        model_chains=config.model_chains,
                        heavy_atom_cutoff=config.heavy_atom_cutoff,
                        cb_cutoff=config.cb_cutoff,
                    ),
                    iptm=read_iptm(prediction.scores_path),
                )
            )

        metrics = analyze_ensemble(
            members,
            cluster_distance_threshold=config.cluster_distance_threshold,
        )
        pair_rows = [
            _pair_row(complex_id, ordered_predictions, pair)
            for pair in metrics.pairs
        ]

        model_rows: list[dict[str, object]] = []
        for index, (prediction, member) in enumerate(
            zip(ordered_predictions, members)
        ):
            heavy_a, heavy_b = interface_residue_sets(
                member.structure.contact_maps.heavy
            )
            cb_a, cb_b = interface_residue_sets(
                member.structure.contact_maps.cb
            )
            model_rows.append(
                {
                    **prediction_metadata(prediction),
                    "iptm": member.iptm,
                    "chain_a_length": member.structure.chain_a_length,
                    "chain_b_length": member.structure.chain_b_length,
                    "contact_count": len(member.structure.contact_maps.heavy),
                    "interface_residue_count_a": len(heavy_a),
                    "interface_residue_count_b": len(heavy_b),
                    "cluster_id": metrics.heavy.cluster_ids[index],
                    "contact_count_cb8": len(member.structure.contact_maps.cb),
                    "interface_residue_count_a_cb8": len(cb_a),
                    "interface_residue_count_b_cb8": len(cb_b),
                    "cluster_id_cb8": metrics.cb.cluster_ids[index],
                    "status": "ok",
                    "error": "",
                }
            )

        model_count = len(members)
        summary.update(
            {
                "nonempty_model_count": metrics.heavy.nonempty_count,
                "nonempty_model_fraction": (
                    metrics.heavy.nonempty_count / model_count
                ),
                "valid_pair_count": metrics.heavy.valid_pair_count,
                "mean_contact_jaccard": metrics.heavy.mean_contact_jaccard,
                "mean_interface_residue_jaccard": (
                    metrics.heavy.mean_interface_residue_jaccard
                ),
                "mean_interface_residue_jaccard_chain_a": (
                    metrics.heavy.mean_interface_residue_jaccard_a
                ),
                "mean_interface_residue_jaccard_chain_b": (
                    metrics.heavy.mean_interface_residue_jaccard_b
                ),
                "max_interface_cluster_fraction": (
                    metrics.heavy.max_cluster_fraction
                ),
                "across_seeds_pair_count": (
                    metrics.heavy.across_seeds_pair_count
                ),
                "mean_contact_jaccard_across_seeds": (
                    metrics.heavy.mean_across_seeds
                ),
                "across_model_weights_pair_count": (
                    metrics.heavy.across_model_weights_pair_count
                ),
                "mean_contact_jaccard_across_model_weights": (
                    metrics.heavy.mean_across_model_weights
                ),
                "nonempty_model_count_cb8": metrics.cb.nonempty_count,
                "nonempty_model_fraction_cb8": (
                    metrics.cb.nonempty_count / model_count
                ),
                "valid_pair_count_cb8": metrics.cb.valid_pair_count,
                "mean_contact_jaccard_cb8": metrics.cb.mean_contact_jaccard,
                "mean_interface_residue_jaccard_cb8": (
                    metrics.cb.mean_interface_residue_jaccard
                ),
                "mean_interface_residue_jaccard_chain_a_cb8": (
                    metrics.cb.mean_interface_residue_jaccard_a
                ),
                "mean_interface_residue_jaccard_chain_b_cb8": (
                    metrics.cb.mean_interface_residue_jaccard_b
                ),
                "max_interface_cluster_fraction_cb8": (
                    metrics.cb.max_cluster_fraction
                ),
                "across_seeds_pair_count_cb8": (
                    metrics.cb.across_seeds_pair_count
                ),
                "mean_contact_jaccard_across_seeds_cb8": (
                    metrics.cb.mean_across_seeds
                ),
                "across_model_weights_pair_count_cb8": (
                    metrics.cb.across_model_weights_pair_count
                ),
                "mean_contact_jaccard_across_model_weights_cb8": (
                    metrics.cb.mean_across_model_weights
                ),
                "pose_pair_count": metrics.pose_pair_count,
                "median_receptor_aligned_ligand_rmsd": (
                    metrics.median_receptor_aligned_ligand_rmsd
                ),
                "iptm_model_count": metrics.iptm.model_count,
                "iptm_mean_across_models": metrics.iptm.mean,
                "iptm_std_across_models": metrics.iptm.population_std,
                "heavy5_status": metrics.heavy.status,
                "cb8_status": metrics.cb.status,
                "pose_status": metrics.pose_status,
                "iptm_status": metrics.iptm_status,
                "status": "ok",
                "error": "",
            }
        )
        return SystemResult(
            summary=summary,
            model_rows=model_rows,
            pair_rows=pair_rows,
        )
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        return SystemResult(summary=summary, model_rows=[], pair_rows=[])


def run_system_task(
    task: tuple[str, list[PredictionFiles], AnalysisConfig],
) -> SystemResult:
    return analyze_system(*task)


def discover_structures(
    prediction_dirs: Sequence[Path],
    allowed_ids: set[str] | None,
) -> list[PredictionFiles]:
    predictions: list[PredictionFiles] = []
    for prediction_dir in prediction_dirs:
        predictions.extend(
            discover_predictions(
                prediction_dir,
                allowed_ids=allowed_ids,
                # Missing scores are reported as a failed system row instead
                # of aborting discovery for the entire batch.
                require_scores=False,
            )
        )
    resolved_paths = [prediction.pdb_path.resolve() for prediction in predictions]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("The same prediction PDB was discovered more than once")
    return predictions


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        action="append",
        required=True,
        help="Repeat once for every seed directory.",
    )
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-model-csv", type=Path, required=True)
    parser.add_argument("--output-pairwise-csv", type=Path, required=True)
    parser.add_argument("--model-chains", default="AB")
    parser.add_argument("--heavy-atom-cutoff", type=float, default=5.0)
    parser.add_argument("--cb-cutoff", type=float, default=8.0)
    parser.add_argument(
        "--cluster-distance-threshold",
        type=float,
        required=True,
        help="Average-linkage cutoff for distance = 1 - contact Jaccard.",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--expected-model-weights",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-systems", type=int)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> AnalysisConfig:
    if len(args.model_chains) != 2 or len(set(args.model_chains)) != 2:
        raise ValueError("--model-chains must contain exactly two different chain IDs")
    for name, value in (
        ("--heavy-atom-cutoff", args.heavy_atom_cutoff),
        ("--cb-cutoff", args.cb_cutoff),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (
        not math.isfinite(args.cluster_distance_threshold)
        or not 0 <= args.cluster_distance_threshold <= 1
    ):
        raise ValueError("--cluster-distance-threshold must be between 0 and 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.max_systems is not None and args.max_systems < 1:
        raise ValueError("--max-systems must be at least 1")
    if len(set(args.expected_seeds)) != len(args.expected_seeds):
        raise ValueError("--expected-seeds contains duplicates")
    if len(set(args.expected_model_weights)) != len(args.expected_model_weights):
        raise ValueError("--expected-model-weights contains duplicates")
    if not args.expected_seeds or not args.expected_model_weights:
        raise ValueError("Expected seeds and model weights cannot be empty")
    output_paths = {
        args.output_summary_csv.resolve(),
        args.output_model_csv.resolve(),
        args.output_pairwise_csv.resolve(),
    }
    if len(output_paths) != 3:
        raise ValueError("The three output CSV paths must be different")
    return AnalysisConfig(
        model_chains=(args.model_chains[0], args.model_chains[1]),
        heavy_atom_cutoff=args.heavy_atom_cutoff,
        cb_cutoff=args.cb_cutoff,
        cluster_distance_threshold=args.cluster_distance_threshold,
        expected_seeds=tuple(sorted(args.expected_seeds)),
        expected_model_weights=tuple(sorted(args.expected_model_weights)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = validate_args(args)
    allowed_ids = read_ids(args.ids_file)
    predictions = discover_structures(args.prediction_dir, allowed_ids)

    grouped: dict[str, list[PredictionFiles]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.complex_id].append(prediction)

    target_ids = sorted(allowed_ids if allowed_ids is not None else grouped)
    if args.max_systems is not None:
        target_ids = target_ids[: args.max_systems]
    tasks = [
        (complex_id, grouped.get(complex_id, []), config)
        for complex_id in target_ids
    ]

    if args.workers == 1:
        results = [
            run_system_task(task)
            for task in tqdm(
                tasks,
                total=len(tasks),
                desc="Consistency",
                unit="system",
            )
        ]
    else:
        results_by_index: list[SystemResult | None] = [None] * len(tasks)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            future_to_index = {
                pool.submit(run_system_task, task): index
                for index, task in enumerate(tasks)
            }
            for future in tqdm(
                as_completed(future_to_index),
                total=len(future_to_index),
                desc="Consistency",
                unit="system",
            ):
                results_by_index[future_to_index[future]] = future.result()
        if any(result is None for result in results_by_index):
            raise RuntimeError("Not all consistency tasks produced a result")
        results = [result for result in results_by_index if result is not None]

    summary_rows = [result.summary for result in results]
    model_rows = [row for result in results for row in result.model_rows]
    pair_rows = [row for result in results for row in result.pair_rows]
    write_csv(args.output_summary_csv, SUMMARY_FIELDS, summary_rows)
    write_csv(args.output_model_csv, MODEL_FIELDS, model_rows)
    write_csv(args.output_pairwise_csv, PAIR_FIELDS, pair_rows)

    failures = sum(row["status"] != "ok" for row in summary_rows)
    print(
        f"Consistency: {len(summary_rows) - failures} systems succeeded, "
        f"{failures} failed; models={len(model_rows)}, pairs={len(pair_rows)}; "
        f"summary={args.output_summary_csv}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
