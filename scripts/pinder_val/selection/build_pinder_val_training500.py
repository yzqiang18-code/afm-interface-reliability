#!/usr/bin/env python3
"""Build a reproducible 500-system PINDER-Val training proposal.

The input is the symmetry-aware seed-0 six-class master table.  Classes 2, 3,
and 5 are retained in full; classes 1 and 4 are diversity-sampled under fixed
same-/different-UniProt quotas.  All 50 feasibility systems are retained so
their existing multi-seed results remain useful as an audit anchor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_QUOTAS = {1: 200, 2: 94, 3: 65, 4: 139, 5: 2, 6: 0}
TOPOLOGY_QUOTAS = {
    1: {False: 135, True: 65},
    4: {False: 22, True: 117},
}
SELECTION_SEED = "pinder-val-training500-v1"

DIVERSITY_FEATURES = [
    "rank1_dockq",
    "oracle5_dockq",
    "n_correct_5",
    "oracle_gain",
    "rank1_iptm",
    "iptm_range_5",
    "rank1_ilis",
    "rank1_pdockq2_min",
    "seed0_mean_contact_jaccard",
    "total_length",
    "length_ratio_max_to_min",
    "min_chain_neff",
    "contains_enzyme",
]


def stable_hash(value: str) -> int:
    payload = f"{SELECTION_SEED}|{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def scaled_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns: list[np.ndarray] = []
    for column in DIVERSITY_FEATURES:
        values = pd.to_numeric(frame[column], errors="coerce").astype(float)
        if column == "min_chain_neff":
            values = np.log1p(values.clip(lower=0))
        median = float(values.median())
        values = values.fillna(median)
        lower = float(values.quantile(0.05))
        upper = float(values.quantile(0.95))
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            scaled = np.zeros(len(values), dtype=float)
        else:
            scaled = ((values.to_numpy() - lower) / (upper - lower)).clip(0, 1)
        columns.append(scaled)
    return np.column_stack(columns)


def maximin_select(
    frame: pd.DataFrame,
    count: int,
    required_ids: set[str] | None = None,
) -> list[str]:
    """Select broad feature coverage while retaining required systems."""

    if count > len(frame):
        raise ValueError(f"Cannot select {count} rows from a {len(frame)}-row stratum")
    required_ids = required_ids or set()
    frame = frame.sort_values("id").reset_index(drop=True)
    if not required_ids.issubset(set(frame["id"])):
        raise ValueError("Required IDs are not contained in the selection stratum")
    if len(required_ids) > count:
        raise ValueError("Required IDs exceed the stratum quota")

    matrix = scaled_feature_matrix(frame)
    id_to_index = {system_id: index for index, system_id in enumerate(frame["id"])}
    selected = [id_to_index[system_id] for system_id in sorted(required_ids)]
    available = np.ones(len(frame), dtype=bool)
    available[selected] = False

    if selected:
        chosen = matrix[selected]
        min_distance = ((matrix[:, None, :] - chosen[None, :, :]) ** 2).sum(axis=2).min(axis=1)
    else:
        center = np.nanmedian(matrix, axis=0)
        distance_to_center = ((matrix - center) ** 2).sum(axis=1)
        candidates = np.where(available)[0]
        start = min(
            candidates,
            key=lambda index: (
                distance_to_center[index],
                stable_hash(str(frame.at[index, "id"])),
            ),
        )
        selected.append(start)
        available[start] = False
        min_distance = ((matrix - matrix[start]) ** 2).sum(axis=1)

    while len(selected) < count:
        candidates = np.where(available)[0]
        best = max(
            candidates,
            key=lambda index: (
                min_distance[index],
                -stable_hash(str(frame.at[index, "id"])),
            ),
        )
        selected.append(best)
        available[best] = False
        distance = ((matrix - matrix[best]) ** 2).sum(axis=1)
        min_distance = np.minimum(min_distance, distance)

    return frame.loc[selected, "id"].tolist()


def selected_reason(row: pd.Series) -> str:
    if row["source_group"] == "feasibility50":
        return "existing_multiseed_anchor"
    if row["class_id"] in (2, 3, 5):
        return "retain_complete_high_information_class"
    if row["class_id"] == 4 and not row["same_uniprot"]:
        return "retain_all_different_uniprot_failures"
    if row["class_id"] == 4 and row["oracle5_dockq"] >= 0.18:
        return "near_threshold_sampling_failure"
    if row["class_id"] == 4 and (
        row["high_confidence_flag"] or row["high_consistency_flag"]
    ):
        return "confident_or_consistent_wrong"
    return "maximin_feature_diversity"


def assign_cv_folds(frame: pd.DataFrame, n_folds: int = 5) -> pd.Series:
    """Balance class/topology strata while keeping every system intact."""

    assignments: dict[str, int] = {}
    fold_totals = [0] * n_folds
    frame = frame.assign(
        cv_stratum=frame["class_id"].astype(str)
        + "|"
        + frame["same_uniprot"].map({True: "same", False: "different"})
    )
    strata = sorted(
        frame.groupby("cv_stratum"),
        key=lambda item: (-len(item[1]), item[0]),
    )
    for stratum, group in strata:
        stratum_counts = [0] * n_folds
        ordered = sorted(group["id"], key=lambda value: (stable_hash(value), value))
        for system_id in ordered:
            fold = min(
                range(n_folds),
                key=lambda index: (
                    stratum_counts[index],
                    fold_totals[index],
                    stable_hash(f"{stratum}|fold{index}"),
                ),
            )
            assignments[system_id] = fold
            stratum_counts[fold] += 1
            fold_totals[fold] += 1
    result = frame["id"].map(assignments)
    if sorted(result.value_counts().tolist()) != [100] * n_folds:
        raise ValueError(f"Expected five 100-system folds, got {result.value_counts().to_dict()}")
    return result.astype(int)


def build_selection(master: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "id",
        "class_id",
        "same_uniprot",
        "source_group",
        "data_status",
        *DIVERSITY_FEATURES,
    }
    missing = required_columns.difference(master.columns)
    if missing:
        raise ValueError(f"Master table is missing columns: {sorted(missing)}")
    if len(master) != 1958 or master["id"].nunique() != 1958:
        raise ValueError("Expected a one-to-one 1,958-system PINDER-Val master table")
    if int(master["data_status"].eq("ok").sum()) != 1927:
        raise ValueError("Expected 1,927 systems with complete seed-0 screening results")

    selected_ids: set[str] = set()
    for class_id in (2, 3, 5):
        selected_ids.update(master.loc[master["class_id"].eq(class_id), "id"])

    for class_id in (1, 4):
        class_frame = master[master["class_id"].eq(class_id)]
        for same_uniprot, quota in TOPOLOGY_QUOTAS[class_id].items():
            stratum = class_frame[class_frame["same_uniprot"].eq(same_uniprot)]
            required = set(stratum.loc[stratum["source_group"].eq("feasibility50"), "id"])
            if class_id == 4:
                required.update(stratum.loc[stratum["oracle5_dockq"].ge(0.18), "id"])
                required.update(
                    stratum.loc[
                        stratum["high_confidence_flag"]
                        | stratum["high_consistency_flag"],
                        "id",
                    ]
                )
                if not same_uniprot:
                    required.update(stratum["id"])
            selected_ids.update(maximin_select(stratum, quota, required))

    selected = master[master["id"].isin(selected_ids)].copy()
    if len(selected) != 500 or selected["id"].nunique() != 500:
        raise ValueError(f"Expected 500 selected systems, found {len(selected)}")
    observed_class = selected["class_id"].value_counts().to_dict()
    if observed_class != {1: 200, 4: 139, 2: 94, 3: 65, 5: 2}:
        raise ValueError(f"Unexpected class quotas: {observed_class}")
    if int((~selected["same_uniprot"]).sum()) != 175:
        raise ValueError("Expected 175 different-UniProt systems")
    if int(selected["source_group"].eq("feasibility50").sum()) != 50:
        raise ValueError("All 50 feasibility systems must be retained")
    if selected["cluster_id"].nunique() != 500:
        raise ValueError("Selected systems are not unique by PINDER cluster_id")

    selected["selection_reason"] = selected.apply(selected_reason, axis=1)
    selected["cv_fold"] = assign_cv_folds(selected)
    selected["planned_seeds"] = "0,1,2,3"
    selected["planned_seed_count"] = 4
    selected["models_per_seed"] = 5
    selected["planned_model_count"] = 20
    selected["selection_version"] = "training500_v1_symmetry_aware"
    selected["selection_hash"] = selected["id"].map(stable_hash)
    selected = selected.sort_values(
        ["class_id", "same_uniprot", "selection_hash", "id"]
    ).reset_index(drop=True)
    selected.insert(0, "selection_order", range(1, len(selected) + 1))
    return selected


def write_outputs(selected: pd.DataFrame, output_dir: Path, input_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(
        output_dir / "recommended_training_500.csv",
        index=False,
        float_format="%.10g",
    )
    (output_dir / "recommended_training_500.txt").write_text(
        "".join(f"{system_id}\n" for system_id in selected["id"]),
        encoding="utf-8",
    )

    composition = (
        selected.groupby(["class_id", "class_key", "class_name_cn", "same_uniprot"])
        .size()
        .rename("count")
        .reset_index()
    )
    composition["topology"] = composition["same_uniprot"].map(
        {True: "same_uniprot", False: "different_uniprot"}
    )
    composition["fraction_of_500"] = composition["count"] / 500
    composition.to_csv(output_dir / "training_500_composition.csv", index=False)

    fold_summary = (
        selected.groupby(["cv_fold", "class_id", "same_uniprot"])
        .size()
        .rename("count")
        .reset_index()
    )
    fold_summary.to_csv(output_dir / "training_500_cv_fold_summary.csv", index=False)

    positive_model_count_seed0 = int(selected["n_correct_5"].sum())
    metadata = {
        "generated_on": date.today().isoformat(),
        "input": str(input_path),
        "selection_version": "training500_v1_symmetry_aware",
        "selection_seed": SELECTION_SEED,
        "system_count": 500,
        "class_quotas": {str(key): value for key, value in CLASS_QUOTAS.items()},
        "same_uniprot_count": int(selected["same_uniprot"].sum()),
        "different_uniprot_count": int((~selected["same_uniprot"]).sum()),
        "feasibility_anchor_count": int(
            selected["source_group"].eq("feasibility50").sum()
        ),
        "seed0_model_rows": 2500,
        "seed0_positive_model_rows": positive_model_count_seed0,
        "seed0_positive_model_fraction": positive_model_count_seed0 / 2500,
        "planned_seed_count": 4,
        "models_per_seed": 5,
        "planned_model_rows": 10000,
        "cv_design": "five system-level folds of 100, stratified by class and same-UniProt status",
        "selection_rules": [
            "Retain all class-2, class-3, and class-5 systems.",
            "Retain all 50 feasibility systems as existing multi-seed anchors.",
            "Retain all 22 different-UniProt class-4 failures.",
            "Within class 4, retain every oracle5 DockQ >=0.18 boundary failure and every high-confidence or high-consistency wrong case.",
            "Fill class-1 and remaining class-4 quotas by deterministic maximin coverage of DockQ, confidence, consistency, length, Neff, and enzyme status.",
            "Exclude class 6 because it has no seed-0 labels and violates the current <1500-aa compute scope.",
        ],
    }
    (output_dir / "selection_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    selected = build_selection(pd.read_csv(input_path))
    write_outputs(selected, output_dir, input_path)
    print(f"Wrote {len(selected)} systems to {output_dir}")
    print(pd.crosstab(selected["class_id"], selected["same_uniprot"], margins=True))
    print("CV folds:", selected["cv_fold"].value_counts().sort_index().to_dict())
    print(
        "Seed0 positive model fraction:",
        f"{selected['n_correct_5'].sum() / (len(selected) * 5):.3f}",
    )


if __name__ == "__main__":
    main()
