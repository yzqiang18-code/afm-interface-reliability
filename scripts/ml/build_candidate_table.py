#!/usr/bin/env python3
"""Merge AF-M confidence and interface metrics into one candidate-level table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baseline_core import (
    atomic_write_csv,
    atomic_write_json,
    check_output_targets,
    require_columns,
    sha256_file,
    utc_now,
)


KEY_COLUMNS = ["complex_id", "rank", "model_weight", "seed"]

CONFIDENCE_COLUMNS = [
    "ranking_confidence",
    "iptm_full_precision",
    "ptm_full_precision",
]
ILIS_COLUMNS = [
    "iLIS",
    "iLIA",
    "iLISA",
    "ipSAE",
    "actifpTM",
    "LIS",
    "cLIS",
    "ipTM_ilis",
    "pTM_ilis",
    "pLDDT",
    "LIpLDDT",
    "cLIpLDDT",
]
PDOCKQ2_COLUMNS = [
    "pDockQ2_chain1_to_chain2",
    "pDockQ2_chain2_to_chain1",
    "pDockQ2_min",
    "pDockQ2_mean",
    "pDockQ2_max",
]
DOCKQ_COLUMNS = [
    "DockQ",
    "mapping_mode",
    "selected_model_chains",
    "direct_DockQ",
    "swapped_DockQ",
    "symmetry_gain",
]
PHYSICS_COLUMNS = [
    "contact_pair_count",
    "interface_residue_count_total",
    "contact_density",
    "bsa_per_interface_residue",
    "contact_component_count",
    "largest_contact_component_fraction",
    "interface_contact_asymmetry",
    "clash_count",
    "backbone_backbone_clash_count",
    "interface_heavy_atom_count_total",
    "clash_density",
    "bsa_a2",
    "log1p_bsa_a2",
    "hydrophobic_contact_count",
    "hydrophobic_contact_fraction",
    "salt_bridge_count",
    "salt_bridge_density",
    "same_charge_contact_count",
    "same_charge_contact_density",
    "interface_status",
]
CONSISTENCY_MODEL_COLUMNS = [
    "contact_count",
    "interface_residue_count_a",
    "interface_residue_count_b",
    "cluster_id",
    "contact_count_cb8",
    "cluster_id_cb8",
]
CONSISTENCY_SUMMARY_COLUMNS = [
    "n_models",
    "ensemble_complete",
    "nonempty_model_fraction",
    "mean_contact_jaccard",
    "mean_interface_residue_jaccard",
    "max_interface_cluster_fraction",
    "mean_contact_jaccard_across_seeds",
    "mean_contact_jaccard_across_model_weights",
    "nonempty_model_fraction_cb8",
    "mean_contact_jaccard_cb8",
    "max_interface_cluster_fraction_cb8",
    "median_receptor_aligned_ligand_rmsd",
    "iptm_mean_across_models",
    "iptm_std_across_models",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confidence-csv", required=True)
    parser.add_argument("--ilis-csv", required=True)
    parser.add_argument("--pdockq2-csv", required=True)
    parser.add_argument("--dockq-csv")
    parser.add_argument("--physics-csv")
    parser.add_argument("--consistency-models-csv")
    parser.add_argument("--consistency-summary-csv")
    parser.add_argument(
        "--metadata-csv",
        help="Optional system metadata containing id/complex_id and cv_fold.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--audit-json")
    parser.add_argument("--expected-systems", type=int, required=True)
    parser.add_argument(
        "--expected-candidates-per-system", type=int, default=20
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_csv(path: Path, *, source: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{source} does not exist: {path}")
    return pd.read_csv(path, low_memory=False)


def normalize_key_types(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    require_columns(frame, KEY_COLUMNS, source=source)
    normalized = frame.copy()
    normalized["complex_id"] = normalized["complex_id"].astype(str)
    for column in ["rank", "model_weight", "seed"]:
        numeric = pd.to_numeric(normalized[column], errors="coerce")
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{source} contains invalid {column} values")
        normalized[column] = numeric.astype(int)
    duplicates = normalized.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        examples = normalized.loc[duplicates, KEY_COLUMNS].head(5).to_dict("records")
        raise ValueError(f"{source} contains duplicate candidate keys: {examples}")
    return normalized


def normalize_confidence(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.rename(
        columns={
            "system_id": "complex_id",
            "iptm": "iptm_full_precision",
            "ptm": "ptm_full_precision",
        }
    )
    normalized = normalize_key_types(normalized, source="confidence CSV")
    require_columns(
        normalized, CONFIDENCE_COLUMNS, source="confidence CSV"
    )
    return normalized[KEY_COLUMNS + CONFIDENCE_COLUMNS]


def normalize_ilis(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.rename(
        columns={
            "name": "complex_id",
            "model": "model_weight",
            "ipTM": "ipTM_ilis",
            "pTM": "pTM_ilis",
        }
    )
    if "seed" not in normalized.columns:
        require_columns(normalized, ["structure_file"], source="iLIS CSV")
        seed = normalized["structure_file"].astype(str).str.extract(
            r"_seed_(\d+)", expand=False
        )
        if seed.isna().any():
            raise ValueError("Could not derive seed from every iLIS structure_file")
        normalized["seed"] = seed.astype(int)
    normalized = normalize_key_types(normalized, source="iLIS CSV")
    require_columns(normalized, ILIS_COLUMNS, source="iLIS CSV")
    return normalized[KEY_COLUMNS + ILIS_COLUMNS]


def normalize_source(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    source: str,
    status_name: str | None,
) -> pd.DataFrame:
    normalized = normalize_key_types(frame, source=source)
    require_columns(normalized, columns, source=source)
    output_columns = KEY_COLUMNS + columns
    if status_name is not None:
        require_columns(normalized, ["status"], source=source)
        bad = normalized.loc[normalized["status"].ne("ok"), KEY_COLUMNS + ["status"]]
        if not bad.empty:
            raise ValueError(
                f"{source} contains non-ok status rows: {bad.head(5).to_dict('records')}"
            )
        normalized = normalized.rename(columns={"status": status_name})
        output_columns.append(status_name)
    return normalized[output_columns]


def exact_key_merge(
    base: pd.DataFrame,
    source: pd.DataFrame,
    *,
    source_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = base.merge(
        source,
        on=KEY_COLUMNS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    counts = merged["_merge"].value_counts().to_dict()
    if not merged["_merge"].eq("both").all():
        examples = merged.loc[
            merged["_merge"].ne("both"), KEY_COLUMNS + ["_merge"]
        ].head(5)
        raise ValueError(
            f"Candidate keys differ while merging {source_name}: "
            f"{examples.to_dict('records')}"
        )
    return merged.drop(columns="_merge"), {
        "source": source_name,
        "rows": int(len(source)),
        "both": int(counts.get("both", 0)),
        "left_only": int(counts.get("left_only", 0)),
        "right_only": int(counts.get("right_only", 0)),
    }


def add_cluster_support(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for cluster_column, result_column in [
        ("cluster_id", "cluster_support_fraction"),
        ("cluster_id_cb8", "cluster_support_fraction_cb8"),
    ]:
        if cluster_column not in output.columns:
            continue
        valid = output[cluster_column].notna()
        sizes = output.groupby("complex_id")["rank"].transform("size")
        support = pd.Series(np.nan, index=output.index, dtype=float)
        support.loc[valid] = (
            output.loc[valid]
            .groupby(["complex_id", cluster_column])[cluster_column]
            .transform("size")
            .astype(float)
            / sizes.loc[valid].astype(float)
        )
        output[result_column] = support
    return output


def merge_candidate_table(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    input_paths: dict[str, Path] = {
        "confidence": Path(args.confidence_csv).expanduser().resolve(),
        "ilis": Path(args.ilis_csv).expanduser().resolve(),
        "pdockq2": Path(args.pdockq2_csv).expanduser().resolve(),
    }
    optional_values = {
        "dockq": args.dockq_csv,
        "physics": args.physics_csv,
        "consistency_models": args.consistency_models_csv,
        "consistency_summary": args.consistency_summary_csv,
        "metadata": args.metadata_csv,
    }
    input_paths.update(
        {
            name: Path(value).expanduser().resolve()
            for name, value in optional_values.items()
            if value is not None
        }
    )

    confidence = normalize_confidence(
        read_csv(input_paths["confidence"], source="confidence CSV")
    )
    ilis = normalize_ilis(read_csv(input_paths["ilis"], source="iLIS CSV"))
    pdockq2 = normalize_source(
        read_csv(input_paths["pdockq2"], source="pDockQ2 CSV"),
        columns=PDOCKQ2_COLUMNS,
        source="pDockQ2 CSV",
        status_name="pdockq2_status",
    )
    table = confidence.copy()
    merge_audits: list[dict[str, Any]] = []
    for name, source in [("ilis", ilis), ("pdockq2", pdockq2)]:
        table, audit = exact_key_merge(table, source, source_name=name)
        merge_audits.append(audit)

    if "dockq" in input_paths:
        dockq = normalize_source(
            read_csv(input_paths["dockq"], source="DockQ CSV"),
            columns=DOCKQ_COLUMNS,
            source="DockQ CSV",
            status_name="dockq_status",
        )
        table, audit = exact_key_merge(table, dockq, source_name="dockq")
        merge_audits.append(audit)

    if "physics" in input_paths:
        physics = normalize_source(
            read_csv(input_paths["physics"], source="physics CSV"),
            columns=PHYSICS_COLUMNS,
            source="physics CSV",
            status_name="physics_status",
        )
        table, audit = exact_key_merge(table, physics, source_name="physics")
        merge_audits.append(audit)

    if "consistency_models" in input_paths:
        consistency_models = normalize_source(
            read_csv(
                input_paths["consistency_models"],
                source="consistency-model CSV",
            ),
            columns=CONSISTENCY_MODEL_COLUMNS,
            source="consistency-model CSV",
            status_name="consistency_model_status",
        )
        table, audit = exact_key_merge(
            table, consistency_models, source_name="consistency_models"
        )
        merge_audits.append(audit)
        table = add_cluster_support(table)

    if "consistency_summary" in input_paths:
        summary = read_csv(
            input_paths["consistency_summary"], source="consistency-summary CSV"
        )
        require_columns(
            summary,
            ["complex_id", "status"] + CONSISTENCY_SUMMARY_COLUMNS,
            source="consistency-summary CSV",
        )
        if summary["complex_id"].duplicated().any():
            raise ValueError("consistency-summary CSV contains duplicate systems")
        bad = summary.loc[summary["status"].ne("ok"), ["complex_id", "status"]]
        if not bad.empty:
            raise ValueError(
                "consistency-summary CSV contains non-ok systems: "
                f"{bad.head(5).to_dict('records')}"
            )
        summary = summary[
            ["complex_id"] + CONSISTENCY_SUMMARY_COLUMNS + ["status"]
        ].rename(columns={"status": "consistency_summary_status"})
        before = len(table)
        table = table.merge(
            summary,
            on="complex_id",
            how="outer",
            validate="many_to_one",
            indicator=True,
        )
        if not table["_merge"].eq("both").all():
            examples = table.loc[
                table["_merge"].ne("both"), ["complex_id", "_merge"]
            ].head(5)
            raise ValueError(
                "System sets differ while merging consistency summary: "
                f"{examples.to_dict('records')}"
            )
        table = table.drop(columns="_merge")
        merge_audits.append(
            {
                "source": "consistency_summary",
                "rows": int(len(summary)),
                "candidate_rows_before": int(before),
                "candidate_rows_after": int(len(table)),
            }
        )

    if "metadata" in input_paths:
        metadata = read_csv(input_paths["metadata"], source="metadata CSV")
        identifier = "complex_id" if "complex_id" in metadata.columns else "id"
        require_columns(metadata, [identifier, "cv_fold"], source="metadata CSV")
        metadata = metadata[[identifier, "cv_fold"]].rename(
            columns={identifier: "complex_id"}
        )
        if metadata["complex_id"].duplicated().any():
            raise ValueError("metadata CSV contains duplicate systems")
        table = table.merge(
            metadata,
            on="complex_id",
            how="left",
            validate="many_to_one",
            indicator=True,
        )
        if not table["_merge"].eq("both").all():
            missing = table.loc[
                table["_merge"].ne("both"), "complex_id"
            ].drop_duplicates().head(5)
            raise ValueError(
                f"metadata CSV lacks systems from the candidate table: {missing.tolist()}"
            )
        table = table.drop(columns="_merge")

    table["negative_clash_density"] = -table["clash_density"] if "clash_density" in table else np.nan
    table["negative_backbone_clashes"] = (
        -table["backbone_backbone_clash_count"]
        if "backbone_backbone_clash_count" in table
        else np.nan
    )
    table["negative_same_charge_density"] = (
        -table["same_charge_contact_density"]
        if "same_charge_contact_density" in table
        else np.nan
    )

    table = table.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
    observed_systems = int(table["complex_id"].nunique())
    if observed_systems != int(args.expected_systems):
        raise ValueError(
            f"Expected {args.expected_systems} systems, found {observed_systems}"
        )
    counts = table.groupby("complex_id").size()
    expected_candidates = int(args.expected_candidates_per_system)
    if not counts.eq(expected_candidates).all():
        raise ValueError(
            f"Expected {expected_candidates} candidates per system; bad examples: "
            f"{counts[counts.ne(expected_candidates)].head(5).to_dict()}"
        )
    expected_rows = int(args.expected_systems) * expected_candidates
    if len(table) != expected_rows:
        raise ValueError(f"Expected {expected_rows} candidate rows, found {len(table)}")

    primary = [
        "ranking_confidence",
        "iptm_full_precision",
        "ptm_full_precision",
        "pDockQ2_min",
        "iLIS",
        "ipSAE",
    ]
    primary_missing = {
        column: int(
            pd.to_numeric(table[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .isna()
            .sum()
        )
        for column in primary
    }
    if any(primary_missing.values()):
        raise ValueError(f"Primary model columns contain missing values: {primary_missing}")

    audit = {
        "created_at_utc": utc_now(),
        "rows": int(len(table)),
        "systems": observed_systems,
        "candidates_per_system": expected_candidates,
        "columns": list(table.columns),
        "primary_feature_missing_counts": primary_missing,
        "has_DockQ": "DockQ" in table.columns,
        "has_cv_fold": "cv_fold" in table.columns,
        "merge_audits": merge_audits,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in input_paths.items()
        },
    }
    return table, audit


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output_csv = Path(args.output_csv).expanduser().resolve()
        audit_json = (
            Path(args.audit_json).expanduser().resolve()
            if args.audit_json
            else output_csv.with_suffix(".audit.json")
        )
        check_output_targets(
            [output_csv, audit_json], overwrite=bool(args.overwrite)
        )
        table, audit = merge_candidate_table(args)
        atomic_write_csv(output_csv, table)
        audit["output_csv"] = str(output_csv)
        audit["output_sha256"] = sha256_file(output_csv)
        atomic_write_json(audit_json, audit)
        print(
            f"Candidate table complete: rows={len(table)} "
            f"systems={table['complex_id'].nunique()} output={output_csv}"
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
