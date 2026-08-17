#!/usr/bin/env python3
"""Create public candidate tables and audits from the private analysis bundle.

The script never modifies its inputs.  It corrects the historical
``UNDEFINED == UNDEFINED`` chain-swap decision by restoring the fixed-mapping
DockQ already present in ``direct_DockQ`` and writes compact, path-free public
artifacts.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPECTED_TRAINING_SYSTEMS = 500
EXPECTED_HOLDOUT_SYSTEMS = 180
EXPECTED_CANDIDATES = 20
UNDEFINED_ACCESSIONS = frozenset({"", "UNDEFINED", "NONE", "NAN", "NA"})
MANIFEST_COLUMNS = [
    "id",
    "pdb_id",
    "cluster_id",
    "cluster_id_R",
    "cluster_id_L",
    "uniprot_R",
    "uniprot_L",
]
AUDIT_LEVELS = (
    "id",
    "pdb_id",
    "cluster_id",
    "chain_cluster_pair",
    "uniprot_pair",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_csv_gz(path: Path, frame: pd.DataFrame) -> None:
    """Write deterministic gzip so hashes do not depend on wall-clock time."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False)
    temporary.replace(path)


def read_ids(path: Path) -> list[str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate IDs in {path}")
    return values


def accession_columns(ids: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = ids.astype(str).str.split("--", n=1, expand=True)
    if parts.shape[1] != 2 or parts.isna().any(axis=None):
        raise ValueError("Could not parse every PINDER dimer ID")
    left = parts[0].str.rsplit("_", n=1).str[-1].str.strip()
    right = parts[1].str.rsplit("_", n=1).str[-1].str.strip()
    return left, right


def undefined(values: pd.Series) -> pd.Series:
    return values.str.upper().isin(UNDEFINED_ACCESSIONS)


def correct_chain_exchange(
    frame: pd.DataFrame,
    *,
    cohort: str,
    expected_systems: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    required = {
        "complex_id",
        "rank",
        "model_weight",
        "seed",
        "DockQ",
        "direct_DockQ",
        "mapping_mode",
        "selected_model_chains",
        "swapped_DockQ",
        "symmetry_gain",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{cohort} table lacks columns: {missing}")
    if frame["complex_id"].nunique() != expected_systems:
        raise ValueError(f"{cohort} must contain {expected_systems} systems")
    sizes = frame.groupby("complex_id").size()
    if not sizes.eq(EXPECTED_CANDIDATES).all():
        raise ValueError(f"{cohort} does not have 20 candidates per system")

    corrected = frame.copy()
    left, right = accession_columns(corrected["complex_id"])
    left_undefined = undefined(left)
    right_undefined = undefined(right)
    both_undefined = left_undefined & right_undefined
    one_undefined = left_undefined ^ right_undefined
    same_known = left.str.upper().eq(right.str.upper()) & ~(
        left_undefined | right_undefined
    )

    direct = pd.to_numeric(corrected.loc[both_undefined, "direct_DockQ"], errors="coerce")
    if not np.isfinite(direct).all():
        raise ValueError(f"{cohort} has missing direct_DockQ for undefined accessions")
    previous = pd.to_numeric(corrected.loc[both_undefined, "DockQ"], errors="raise")
    changes = corrected.loc[
        both_undefined,
        ["complex_id", "rank", "model_weight", "seed", "mapping_mode"],
    ].copy()
    changes.insert(0, "cohort", cohort)
    changes = changes.rename(columns={"mapping_mode": "previous_mapping_mode"})
    changes["corrected_mapping_mode"] = "fixed"
    changes["previous_DockQ"] = previous.to_numpy(dtype=float)
    changes["corrected_DockQ"] = direct.to_numpy(dtype=float)
    changes["dockq_delta"] = changes["corrected_DockQ"] - changes["previous_DockQ"]
    changes["continuous_value_changed"] = changes["dockq_delta"].abs().gt(1e-12)
    changes["primary_class_changed"] = changes["previous_DockQ"].ge(0.23).ne(
        changes["corrected_DockQ"].ge(0.23)
    )

    corrected.loc[both_undefined, "DockQ"] = direct.to_numpy(dtype=float)
    corrected.loc[both_undefined, "mapping_mode"] = "fixed"
    corrected.loc[both_undefined, "selected_model_chains"] = "AB"
    corrected.loc[both_undefined, "swapped_DockQ"] = np.nan
    corrected.loc[both_undefined, "symmetry_gain"] = 0.0
    corrected["same_known_uniprot"] = same_known.to_numpy(dtype=bool)
    corrected["chain_exchange_eligible"] = same_known.to_numpy(dtype=bool)
    corrected["accession_status"] = np.select(
        [both_undefined, one_undefined, same_known],
        ["both_undefined", "one_undefined", "same_known"],
        default="different_known",
    )

    expected_mode = np.where(same_known, "symmetry_aware", "fixed")
    if not corrected["mapping_mode"].astype(str).eq(expected_mode).all():
        bad = corrected.loc[
            corrected["mapping_mode"].astype(str).ne(expected_mode),
            ["complex_id", "mapping_mode"],
        ].drop_duplicates()
        raise ValueError(f"{cohort} still has invalid mapping modes: {bad.head().to_dict('records')}")

    summary = {
        "cohort": cohort,
        "systems": int(corrected["complex_id"].nunique()),
        "rows": int(len(corrected)),
        "both_undefined_systems": int(corrected.loc[both_undefined, "complex_id"].nunique()),
        "both_undefined_rows": int(both_undefined.sum()),
        "one_undefined_systems": int(corrected.loc[one_undefined, "complex_id"].nunique()),
        "one_undefined_rows": int(one_undefined.sum()),
        "continuous_dockq_changed_rows": int(changes["continuous_value_changed"].sum()),
        "continuous_dockq_changed_systems": int(
            changes.loc[changes["continuous_value_changed"], "complex_id"].nunique()
        ),
        "primary_class_changed_rows": int(changes["primary_class_changed"].sum()),
        "primary_class_changed_systems": int(
            changes.loc[changes["primary_class_changed"], "complex_id"].nunique()
        ),
        "maximum_removed_symmetry_gain": float((-changes["dockq_delta"]).max()),
    }
    return corrected, changes, summary


def unordered_pair(left: object, right: object) -> str:
    return "|".join(sorted((str(left), str(right))))


def level_map(frame: pd.DataFrame, level: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in frame.itertuples(index=False):
        if level in {"id", "pdb_id", "cluster_id"}:
            value = str(getattr(row, level))
        elif level == "chain_cluster_pair":
            value = unordered_pair(row.cluster_id_R, row.cluster_id_L)
        elif level == "uniprot_pair":
            if (
                str(row.uniprot_R).upper() in UNDEFINED_ACCESSIONS
                or str(row.uniprot_L).upper() in UNDEFINED_ACCESSIONS
            ):
                continue
            value = unordered_pair(row.uniprot_R, row.uniprot_L)
        else:
            raise ValueError(level)
        result.setdefault(value, []).append(str(row.id))
    return result


def leakage_audit(
    training: pd.DataFrame,
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    detail_rows: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for level in AUDIT_LEVELS:
        training_map = level_map(training, level)
        holdout_map = level_map(holdout, level)
        overlap = sorted(set(training_map).intersection(holdout_map))
        counts[level] = len(overlap)
        for value in overlap:
            detail_rows.append(
                {
                    "level": level,
                    "value": value,
                    "training_system_ids": ";".join(sorted(training_map[value])),
                    "holdout_system_ids": ";".join(sorted(holdout_map[value])),
                }
            )
    details = pd.DataFrame(
        detail_rows,
        columns=["level", "value", "training_system_ids", "holdout_system_ids"],
    )
    summary = {
        "pinder_release": "2024-02",
        "training_systems": int(len(training)),
        "holdout_systems": int(len(holdout)),
        "intersection_counts": counts,
        "release_gate_passed": all(
            counts[level] == 0 for level in ("id", "pdb_id", "cluster_id")
        ),
        "undefined_uniprot_pairs_excluded": True,
        "interpretation": (
            "Exact system, PDB, and interface-cluster overlap are release gates. "
            "Chain-cluster and UniProt-pair overlaps are disclosed and require "
            "sensitivity analysis rather than being silently removed."
        ),
    }
    return details, summary


def select_manifest(index: pd.DataFrame, ids: Iterable[str], *, cohort: str) -> pd.DataFrame:
    ordered = list(ids)
    selected = index[index["id"].astype(str).isin(ordered)].copy()
    if len(selected) != len(ordered) or selected["id"].nunique() != len(ordered):
        raise ValueError(f"Official index does not uniquely resolve every {cohort} ID")
    selected = selected.set_index("id", drop=False).loc[ordered].reset_index(drop=True)
    selected.insert(0, "pinder_release", "2024-02")
    selected.insert(1, "cohort", cohort)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-csv", type=Path, required=True)
    parser.add_argument("--holdout-csv", type=Path, required=True)
    parser.add_argument("--index-parquet", type=Path, required=True)
    parser.add_argument("--training-assignment", type=Path, required=True)
    parser.add_argument("--holdout-ids", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = [
        args.training_csv,
        args.holdout_csv,
        args.index_parquet,
        args.training_assignment,
        args.holdout_ids,
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_root = args.output_root.resolve()
    data_dir = output_root / "data"
    chain_dir = output_root / "audits" / "chain_exchange"
    leakage_dir = output_root / "audits" / "leakage"
    outputs = [
        data_dir / "training500_candidates.csv.gz",
        data_dir / "pinder_af2_180_labels.csv.gz",
        data_dir / "training500_manifest.csv",
        data_dir / "pinder_af2_180_manifest.csv",
        data_dir / "data_manifest.json",
        chain_dir / "candidate_changes.csv.gz",
        chain_dir / "chain_exchange_summary.json",
        leakage_dir / "leakage_intersections.csv",
        leakage_dir / "leakage_summary.json",
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to overwrite outputs: " + ", ".join(existing))

    training_raw = pd.read_csv(args.training_csv, low_memory=False)
    holdout_raw = pd.read_csv(args.holdout_csv, low_memory=False)
    training, training_changes, training_summary = correct_chain_exchange(
        training_raw,
        cohort="Training500",
        expected_systems=EXPECTED_TRAINING_SYSTEMS,
    )
    holdout, holdout_changes, holdout_summary = correct_chain_exchange(
        holdout_raw,
        cohort="PINDER-AF2",
        expected_systems=EXPECTED_HOLDOUT_SYSTEMS,
    )

    assignment = pd.read_csv(args.training_assignment)
    training_ids = assignment["id"].astype(str).tolist()
    holdout_ids = read_ids(args.holdout_ids)
    index = pd.read_parquet(args.index_parquet, columns=MANIFEST_COLUMNS)
    training_manifest = select_manifest(index, training_ids, cohort="Training500")
    holdout_manifest = select_manifest(index, holdout_ids, cohort="PINDER-AF2")
    training_manifest = training_manifest.merge(
        assignment[["id", "source_group", "class_id", "class_key", "cv_fold"]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    leakage_details, leakage_summary = leakage_audit(
        training_manifest, holdout_manifest
    )

    atomic_csv_gz(data_dir / "training500_candidates.csv.gz", training)
    atomic_csv_gz(data_dir / "pinder_af2_180_labels.csv.gz", holdout)
    atomic_csv(data_dir / "training500_manifest.csv", training_manifest)
    atomic_csv(data_dir / "pinder_af2_180_manifest.csv", holdout_manifest)
    atomic_csv_gz(
        chain_dir / "candidate_changes.csv.gz",
        pd.concat([training_changes, holdout_changes], ignore_index=True),
    )
    atomic_json(
        chain_dir / "chain_exchange_summary.json",
        {
            "rule": (
                "Missing/UNDEFINED accession tokens never establish chain-exchange "
                "eligibility; affected rows use the stored fixed-mapping direct_DockQ."
            ),
            "training": training_summary,
            "holdout": holdout_summary,
            "primary_threshold": 0.23,
        },
    )
    atomic_csv(leakage_dir / "leakage_intersections.csv", leakage_details)
    leakage_summary["official_index_sha256"] = sha256_file(args.index_parquet)
    atomic_json(leakage_dir / "leakage_summary.json", leakage_summary)

    artifact_records = []
    for path, rows in [
        (data_dir / "training500_candidates.csv.gz", len(training)),
        (data_dir / "pinder_af2_180_labels.csv.gz", len(holdout)),
        (data_dir / "training500_manifest.csv", len(training_manifest)),
        (data_dir / "pinder_af2_180_manifest.csv", len(holdout_manifest)),
    ]:
        artifact_records.append(
            {
                "path": str(path.relative_to(output_root.parent)),
                "rows": int(rows),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    atomic_json(
        data_dir / "data_manifest.json",
        {
            "schema_version": 1,
            "source_files": [
                {"name": args.training_csv.name, "sha256": sha256_file(args.training_csv)},
                {"name": args.holdout_csv.name, "sha256": sha256_file(args.holdout_csv)},
                {"name": args.index_parquet.name, "sha256": sha256_file(args.index_parquet)},
            ],
            "artifacts": artifact_records,
            "contains_raw_coordinates": False,
            "contains_local_paths": False,
        },
    )
    print(json.dumps({"chain_exchange": [training_summary, holdout_summary], "leakage": leakage_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
