#!/usr/bin/env python3
"""Classify the complete PINDER-Val release into six seed-0 screening classes.

The classification uses the five AF-M v2.3 model weights from seed 0.  It
combines the 1,877 length-filtered group results, the 50 feasibility systems,
and the 31 systems that were not run because their resolved total length is at
least 1,500 residues.  The latter remain visible in class 6 rather than being
silently dropped.

Class 5 uses empirical thresholds calculated over all systems with complete
seed-0 results: the 80th percentile of rank-1 ipTM and the 80th percentile of
the mean heavy-atom 5 A contact-map Jaccard across the five model weights.

DockQ labels can be read either from the original fixed-chain mapping or from
the symmetry-aware homodimer rerun.  The mode must be selected explicitly on
the command line so that the two partitions can be kept as separate artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd


DOCKQ_THRESHOLD = 0.23
HIGH_QUANTILE = 0.80
NEAR_DOCKQ_LOWER = 0.18
NEAR_DOCKQ_UPPER = 0.28

CLASS_INFO = {
    1: (
        "robust_correct",
        "稳健正确",
        "rank1正确，且五个weight中至少四个正确",
    ),
    2: (
        "fragile_correct",
        "脆弱正确",
        "rank1正确，但五个weight中最多三个正确",
    ),
    3: (
        "rerank_rescuable",
        "可重排序挽救",
        "rank1错误，但oracle-best-of-5正确",
    ),
    4: (
        "ordinary_sampling_failure",
        "普通采样失败",
        "oracle-best-of-5错误，且不满足高置信度稳定错误定义",
    ),
    5: (
        "high_confidence_stable_wrong",
        "高置信度稳定错误",
        "oracle-best-of-5错误，同时rank1 ipTM和五weight界面一致性均位于前20%",
    ),
    6: (
        "technical_evaluation_or_uncomputed",
        "技术、评估异常或未计算",
        "缺少可靠的完整seed0五weight标签；当前包括因总长度>=1500 aa而未运行的system",
    ),
}

CLASS_FILENAMES = {
    class_id: f"class{class_id}_{info[0]}"
    for class_id, info in CLASS_INFO.items()
}


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def require_unique(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    duplicates = frame.duplicated(columns, keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, columns].head(5).to_dict("records")
        raise ValueError(f"Duplicate keys in {source}: {examples}")


def require_ok_status(frame: pd.DataFrame, source: Path) -> None:
    if "status" in frame.columns:
        bad = frame[frame["status"].astype(str).str.lower() != "ok"]
        if not bad.empty:
            raise ValueError(f"Non-ok rows in {source}: {len(bad)}")
    if "error" in frame.columns and frame["error"].notna().any():
        raise ValueError(f"Non-empty error values in {source}")


def validate_five_models(
    frame: pd.DataFrame,
    source: Path,
    id_column: str,
    weight_column: str,
) -> None:
    counts = frame.groupby(id_column).size()
    if not counts.eq(5).all():
        raise ValueError(f"Expected five model rows per system in {source}")
    weights = frame.groupby(id_column)[weight_column].apply(set)
    if not weights.apply(lambda value: value == {1, 2, 3, 4, 5}).all():
        raise ValueError(f"Expected model weights 1-5 for every system in {source}")
    ranks = frame.groupby(id_column)["rank"].apply(set)
    if not ranks.apply(lambda value: value == {1, 2, 3, 4, 5}).all():
        raise ValueError(f"Expected ranks 1-5 for every system in {source}")


def build_model_metrics(
    dockq_path: Path,
    ilis_path: Path,
    pdockq2_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dockq = pd.read_csv(dockq_path)
    ilis = pd.read_csv(ilis_path)
    pdockq2 = pd.read_csv(pdockq2_path)

    require_columns(
        dockq,
        {"complex_id", "rank", "model_weight", "seed", "DockQ", "status", "error"},
        dockq_path,
    )
    require_columns(
        ilis,
        {"name", "rank", "model", "ipTM", "iLIS", "len_i", "len_j"},
        ilis_path,
    )
    require_columns(
        pdockq2,
        {
            "complex_id",
            "rank",
            "model_weight",
            "seed",
            "pDockQ2_min",
            "status",
            "error",
        },
        pdockq2_path,
    )
    require_ok_status(dockq, dockq_path)
    require_ok_status(pdockq2, pdockq2_path)
    validate_five_models(dockq, dockq_path, "complex_id", "model_weight")
    validate_five_models(ilis, ilis_path, "name", "model")
    validate_five_models(pdockq2, pdockq2_path, "complex_id", "model_weight")
    require_unique(dockq, ["complex_id", "seed", "model_weight"], dockq_path)
    require_unique(ilis, ["name", "model"], ilis_path)
    require_unique(
        pdockq2,
        ["complex_id", "seed", "model_weight"],
        pdockq2_path,
    )
    if set(dockq["seed"].unique()) != {0} or set(pdockq2["seed"].unique()) != {0}:
        raise ValueError("The screening inputs must contain seed 0 only")
    if dockq["DockQ"].isna().any() or not dockq["DockQ"].between(0, 1).all():
        raise ValueError(f"Invalid DockQ values in {dockq_path}")
    if ilis[["ipTM", "iLIS"]].isna().any().any():
        raise ValueError(f"Missing ipTM/iLIS values in {ilis_path}")
    if not ilis["ipTM"].between(0, 1).all() or not ilis["iLIS"].between(0, 1).all():
        raise ValueError(f"Out-of-range ipTM/iLIS values in {ilis_path}")
    if pdockq2["pDockQ2_min"].isna().any() or not pdockq2["pDockQ2_min"].between(0, 1).all():
        raise ValueError(f"Invalid pDockQ2_min values in {pdockq2_path}")

    dockq_keys = set(zip(dockq["complex_id"], dockq["model_weight"], strict=True))
    ilis_keys = set(zip(ilis["name"], ilis["model"], strict=True))
    pdockq2_keys = set(
        zip(pdockq2["complex_id"], pdockq2["model_weight"], strict=True)
    )
    if dockq_keys != ilis_keys or dockq_keys != pdockq2_keys:
        raise ValueError("DockQ, iLIS and pDockQ2 model keys do not match")

    dockq_rank = dockq[["complex_id", "model_weight", "rank"]]
    ilis_rank = ilis[["name", "model", "rank"]].rename(
        columns={"name": "complex_id", "model": "model_weight"}
    )
    pdockq2_rank = pdockq2[["complex_id", "model_weight", "rank"]]
    for other, label in ((ilis_rank, "iLIS"), (pdockq2_rank, "pDockQ2")):
        merged = dockq_rank.merge(
            other,
            on=["complex_id", "model_weight"],
            validate="one_to_one",
            suffixes=("_dockq", f"_{label}"),
        )
        if not merged["rank_dockq"].eq(merged[f"rank_{label}"]).all():
            raise ValueError(f"DockQ and {label} ranks do not match")

    dockq_model_columns = ["complex_id", "rank", "model_weight", "DockQ"]
    optional_symmetry_columns = [
        "mapping_mode",
        "selected_model_chains",
        "direct_DockQ",
        "swapped_DockQ",
        "symmetry_gain",
    ]
    dockq_model_columns.extend(
        column for column in optional_symmetry_columns if column in dockq.columns
    )
    models = (
        dockq[dockq_model_columns]
        .merge(
            ilis[["name", "model", "ipTM", "iLIS", "len_i", "len_j"]],
            left_on=["complex_id", "model_weight"],
            right_on=["name", "model"],
            validate="one_to_one",
        )
        .merge(
            pdockq2[["complex_id", "model_weight", "pDockQ2_min"]],
            on=["complex_id", "model_weight"],
            validate="one_to_one",
        )
    )
    models["is_correct"] = models["DockQ"] >= DOCKQ_THRESHOLD
    models["near_dockq_threshold"] = models["DockQ"].between(
        NEAR_DOCKQ_LOWER, NEAR_DOCKQ_UPPER, inclusive="both"
    )

    rank1 = models[models["rank"] == 1].set_index("complex_id")
    grouped = models.groupby("complex_id", sort=False)
    system = pd.DataFrame(index=rank1.index)
    system.index.name = "id"
    system["rank1_dockq"] = rank1["DockQ"]
    system["oracle5_dockq"] = grouped["DockQ"].max()
    system["n_correct_5"] = grouped["is_correct"].sum().astype(int)
    system["oracle_gain"] = system["oracle5_dockq"] - system["rank1_dockq"]
    system["rank1_iptm"] = rank1["ipTM"]
    system["iptm_min_5"] = grouped["ipTM"].min()
    system["iptm_max_5"] = grouped["ipTM"].max()
    system["iptm_range_5"] = system["iptm_max_5"] - system["iptm_min_5"]
    system["rank1_ilis"] = rank1["iLIS"]
    system["max_ilis_5"] = grouped["iLIS"].max()
    system["rank1_pdockq2_min"] = rank1["pDockQ2_min"]
    system["max_pdockq2_min_5"] = grouped["pDockQ2_min"].max()
    system["any_model_near_dockq_threshold"] = grouped[
        "near_dockq_threshold"
    ].any()
    system["length_R"] = rank1["len_i"].astype(int)
    system["length_L"] = rank1["len_j"].astype(int)
    system["total_length"] = system["length_R"] + system["length_L"]
    if "mapping_mode" in models.columns:
        system["rank1_mapping_mode"] = rank1["mapping_mode"]
        system["rank1_selected_model_chains"] = rank1["selected_model_chains"]
        system["rank1_symmetry_gain"] = rank1["symmetry_gain"]
        system["max_symmetry_gain_5"] = grouped["symmetry_gain"].max()
        system["n_swapped_selected_5"] = grouped["selected_model_chains"].apply(
            lambda values: int((values == "BA").sum())
        )
    return system.reset_index(), models


def dockq_input_path(directory: Path, dockq_mode: str) -> Path:
    if dockq_mode == "fixed":
        return directory / "dockq" / "afm23_5models_seed0_dockq.csv"
    return (
        directory
        / "dockq_symmetry"
        / "afm23_5models_seed0_dockq_symmetry.csv"
    )


def load_group(root: Path, group_number: int, dockq_mode: str) -> pd.DataFrame:
    group_dir = root / f"group{group_number}"
    system, _ = build_model_metrics(
        dockq_input_path(group_dir, dockq_mode),
        group_dir / "ilis" / "afm23_5models_seed0_ilis.csv",
        group_dir / "pdockq2" / "afm23_5models_seed0_pdockq2.csv",
    )
    manifest_path = group_dir / "msa_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    require_columns(
        manifest,
        {"pinder_id", "status", "length_R", "length_L"},
        manifest_path,
    )
    if not manifest["status"].eq("complete").all():
        raise ValueError(f"Non-complete MSA rows in {manifest_path}")
    require_unique(manifest, ["pinder_id"], manifest_path)
    manifest["total_length_manifest"] = manifest["length_R"] + manifest["length_L"]
    if not manifest["total_length_manifest"].lt(1500).all():
        raise ValueError(f"Current group manifest contains total length >=1500: {manifest_path}")
    if set(system["id"]) != set(manifest["pinder_id"]):
        raise ValueError(f"Metric and MSA IDs do not match in group {group_number}")
    system = system.merge(
        manifest[["pinder_id", "length_R", "length_L", "total_length_manifest"]],
        left_on="id",
        right_on="pinder_id",
        validate="one_to_one",
        suffixes=("", "_manifest"),
    )
    if not system["total_length"].eq(system["total_length_manifest"]).all():
        raise ValueError(f"iLIS and manifest lengths disagree in group {group_number}")
    system = system.drop(
        columns=[
            "pinder_id",
            "length_R_manifest",
            "length_L_manifest",
            "total_length_manifest",
        ]
    )

    ranking_path = group_dir / "full_precision_ranking.csv"
    ranking = pd.read_csv(ranking_path)
    require_columns(
        ranking,
        {"system_id", "rank", "model_weight", "seed", "ranking_confidence", "iptm"},
        ranking_path,
    )
    validate_five_models(ranking, ranking_path, "system_id", "model_weight")
    require_unique(ranking, ["system_id", "seed", "model_weight"], ranking_path)
    ranking_rank1 = ranking[ranking["rank"] == 1][
        ["system_id", "ranking_confidence", "iptm"]
    ].rename(
        columns={
            "system_id": "id",
            "ranking_confidence": "rank1_ranking_confidence",
            "iptm": "rank1_iptm_full_precision",
        }
    )
    system = system.merge(ranking_rank1, on="id", validate="one_to_one")

    consistency_path = (
        group_dir
        / "consistency"
        / "afm23_seed0_weights1-5_consistency_summary.csv"
    )
    consistency = pd.read_csv(consistency_path)
    require_columns(
        consistency,
        {
            "complex_id",
            "n_models",
            "ensemble_complete",
            "nonempty_model_count",
            "mean_contact_jaccard",
            "max_interface_cluster_fraction",
            "pose_pair_count",
            "median_receptor_aligned_ligand_rmsd",
            "status",
            "error",
        },
        consistency_path,
    )
    require_ok_status(consistency, consistency_path)
    require_unique(consistency, ["complex_id"], consistency_path)
    if not consistency["n_models"].eq(5).all() or not consistency[
        "ensemble_complete"
    ].astype(bool).all():
        raise ValueError(f"Incomplete five-weight ensemble in {consistency_path}")
    if consistency["mean_contact_jaccard"].isna().any():
        raise ValueError(f"Missing contact Jaccard in {consistency_path}")
    consistency = consistency.rename(
        columns={
            "complex_id": "id",
            "mean_contact_jaccard": "seed0_mean_contact_jaccard",
            "max_interface_cluster_fraction": "seed0_max_interface_cluster_fraction",
            "median_receptor_aligned_ligand_rmsd": "seed0_median_pose_rmsd",
        }
    )
    system = system.merge(
        consistency[
            [
                "id",
                "nonempty_model_count",
                "seed0_mean_contact_jaccard",
                "seed0_max_interface_cluster_fraction",
                "seed0_median_pose_rmsd",
            ]
        ],
        on="id",
        validate="one_to_one",
    )
    system["source_group"] = f"group{group_number}"
    system["ranking_source"] = "full_precision_ranking_csv"
    return system


def load_feasibility(root: Path, dockq_mode: str) -> pd.DataFrame:
    feasibility_dir = root / "feasibility_50"
    system, _ = build_model_metrics(
        dockq_input_path(feasibility_dir, dockq_mode),
        feasibility_dir / "ilis" / "afm23_5models_seed0_ilis.csv",
        feasibility_dir / "pdockq2" / "afm23_5models_seed0_pdockq2.csv",
    )
    expected_path = root / "pinder_val_feasibility_50.csv"
    expected = pd.read_csv(expected_path)
    require_columns(expected, {"id"}, expected_path)
    require_unique(expected, ["id"], expected_path)
    if len(expected) != 50 or set(system["id"]) != set(expected["id"]):
        raise ValueError("Feasibility metric IDs do not match the fixed 50-system list")

    consistency_dir = feasibility_dir / "consistency" / "full"
    pairs_path = consistency_dir / "afm23_5models_seed0-4_consistency_pairs.csv"
    models_path = consistency_dir / "afm23_5models_seed0-4_consistency_models.csv"
    pairs = pd.read_csv(pairs_path)
    models = pd.read_csv(models_path)
    require_columns(
        pairs,
        {
            "complex_id",
            "seed_1",
            "seed_2",
            "jaccard",
            "jaccard_valid",
        },
        pairs_path,
    )
    require_columns(
        models,
        {"complex_id", "seed", "model_weight", "contact_count", "status", "error"},
        models_path,
    )
    require_ok_status(models, models_path)
    seed0_pairs = pairs[(pairs["seed_1"] == 0) & (pairs["seed_2"] == 0)].copy()
    pair_counts = seed0_pairs.groupby("complex_id").size()
    if len(pair_counts) != 50 or not pair_counts.eq(10).all():
        raise ValueError("Expected ten seed-0 model pairs per feasibility system")
    if not seed0_pairs["jaccard_valid"].astype(bool).all():
        raise ValueError("Invalid seed-0 contact Jaccard pair in feasibility data")
    seed0_consistency = seed0_pairs.groupby("complex_id").agg(
        seed0_mean_contact_jaccard=("jaccard", "mean")
    )
    seed0_consistency["seed0_median_pose_rmsd"] = float("nan")
    seed0_models = models[models["seed"] == 0].copy()
    model_counts = seed0_models.groupby("complex_id").size()
    if len(model_counts) != 50 or not model_counts.eq(5).all():
        raise ValueError("Expected five seed-0 consistency models per feasibility system")
    nonempty = seed0_models.groupby("complex_id")["contact_count"].apply(
        lambda values: int((values > 0).sum())
    )
    seed0_consistency["nonempty_model_count"] = nonempty
    seed0_consistency["seed0_max_interface_cluster_fraction"] = float("nan")
    system = system.merge(
        seed0_consistency.reset_index().rename(columns={"complex_id": "id"}),
        on="id",
        validate="one_to_one",
    )
    system["rank1_ranking_confidence"] = float("nan")
    system["rank1_iptm_full_precision"] = float("nan")
    system["source_group"] = "feasibility50"
    system["ranking_source"] = "colabfold_seed0_rank_filename"
    return system


def recover_pre_filter_lengths(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for group_number in (1, 2, 3):
        group_dir = root / f"group{group_number}"
        for path in sorted(group_dir.glob("msa_manifest_before_total_length_lt1500*.csv")):
            frame = pd.read_csv(path)
            require_columns(frame, {"pinder_id", "length_R", "length_L"}, path)
            frames.append(frame[["pinder_id", "length_R", "length_L"]])
    combined = pd.concat(frames, ignore_index=True)
    conflicts = combined.groupby("pinder_id")[["length_R", "length_L"]].nunique()
    if (conflicts > 1).any().any():
        raise ValueError("Conflicting chain lengths across pre-filter manifests")
    combined = combined.drop_duplicates("pinder_id").rename(columns={"pinder_id": "id"})
    combined["total_length"] = combined["length_R"] + combined["length_L"]
    return combined


def classify(master: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    computed = master["data_status"].eq("ok")
    if not computed.any():
        raise ValueError("No complete systems available for threshold calculation")
    high_iptm_threshold = float(master.loc[computed, "rank1_iptm"].quantile(HIGH_QUANTILE))
    high_consistency_threshold = float(
        master.loc[computed, "seed0_mean_contact_jaccard"].quantile(HIGH_QUANTILE)
    )
    master["high_confidence_flag"] = computed & master["rank1_iptm"].ge(
        high_iptm_threshold
    )
    master["high_consistency_flag"] = computed & master[
        "seed0_mean_contact_jaccard"
    ].ge(high_consistency_threshold)
    master["rank1_correct"] = computed & master["rank1_dockq"].ge(DOCKQ_THRESHOLD)
    master["oracle5_correct"] = computed & master["oracle5_dockq"].ge(DOCKQ_THRESHOLD)
    master["has_empty_interface_model"] = computed & master[
        "nonempty_model_count"
    ].lt(5)

    master["class_id"] = 6
    class1 = computed & master["rank1_correct"] & master["n_correct_5"].ge(4)
    class2 = computed & master["rank1_correct"] & master["n_correct_5"].le(3)
    class3 = computed & ~master["rank1_correct"] & master["oracle5_correct"]
    class5 = (
        computed
        & ~master["oracle5_correct"]
        & master["high_confidence_flag"]
        & master["high_consistency_flag"]
        & master["nonempty_model_count"].eq(5)
    )
    class4 = computed & ~master["oracle5_correct"] & ~class5
    master.loc[class1, "class_id"] = 1
    master.loc[class2, "class_id"] = 2
    master.loc[class3, "class_id"] = 3
    master.loc[class4, "class_id"] = 4
    master.loc[class5, "class_id"] = 5

    class_masks = {1: class1, 2: class2, 3: class3, 4: class4, 5: class5}
    assigned_computed = pd.concat(
        [mask.rename(str(class_id)) for class_id, mask in class_masks.items()], axis=1
    ).sum(axis=1)
    if not assigned_computed.loc[computed].eq(1).all():
        raise ValueError("Computed systems were not assigned to exactly one class")

    master["class_key"] = master["class_id"].map(
        {class_id: info[0] for class_id, info in CLASS_INFO.items()}
    )
    master["class_name_cn"] = master["class_id"].map(
        {class_id: info[1] for class_id, info in CLASS_INFO.items()}
    )
    master["classification_reason"] = master["class_id"].map(
        {class_id: info[2] for class_id, info in CLASS_INFO.items()}
    )
    master.loc[master["class_id"].eq(5), "classification_reason"] = (
        "oracle5_DockQ<0.23; rank1_ipTM>="
        f"{high_iptm_threshold:.10g}; seed0_mean_contact_jaccard>="
        f"{high_consistency_threshold:.10g}; five models have nonempty interfaces"
    )
    master.loc[master["class_id"].eq(6), "classification_reason"] = master.loc[
        master["class_id"].eq(6), "technical_reason"
    ]
    thresholds = {
        "dockq_correct_threshold": DOCKQ_THRESHOLD,
        "near_dockq_lower": NEAR_DOCKQ_LOWER,
        "near_dockq_upper": NEAR_DOCKQ_UPPER,
        "high_quantile": HIGH_QUANTILE,
        "rank1_iptm_high_threshold": high_iptm_threshold,
        "seed0_mean_contact_jaccard_high_threshold": high_consistency_threshold,
    }
    return master, thresholds


def make_summary(master: pd.DataFrame) -> pd.DataFrame:
    counts = (
        master.groupby(["class_id", "class_key", "class_name_cn"], sort=True)
        .size()
        .rename("count")
        .reset_index()
    )
    counts["fraction_of_1958"] = counts["count"] / len(master)
    sources = pd.crosstab(master["class_id"], master["source_group"])
    sources.columns = [f"count_{column}" for column in sources.columns]
    summary = counts.merge(sources.reset_index(), on="class_id", how="left")
    for source in ("group1", "group2", "group3", "feasibility50", "not_run_ge1500"):
        column = f"count_{source}"
        if column not in summary:
            summary[column] = 0
    return summary.sort_values("class_id")


def write_outputs(
    master: pd.DataFrame,
    summary: pd.DataFrame,
    thresholds: dict[str, float],
    output_dir: Path,
    dockq_mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / "pinder_val_seed0_six_class_master.csv"
    summary_path = output_dir / "pinder_val_seed0_six_class_summary.csv"
    master.to_csv(master_path, index=False, float_format="%.10g")
    summary.to_csv(summary_path, index=False, float_format="%.10g")

    for class_id, filename in CLASS_FILENAMES.items():
        subset = master[master["class_id"] == class_id].sort_values("id")
        subset.to_csv(output_dir / f"{filename}.csv", index=False, float_format="%.10g")
        (output_dir / f"{filename}.txt").write_text(
            "".join(f"{system_id}\n" for system_id in subset["id"]),
            encoding="utf-8",
        )

    metadata = {
        "generated_on": date.today().isoformat(),
        "pinder_release": "2024-02",
        "classification_scope": (
            "seed0, AF-M v2.3, five model weights, "
            f"DockQ mapping={dockq_mode}"
        ),
        "total_pinder_val_systems": int(len(master)),
        "computed_seed0_systems": int(master["data_status"].eq("ok").sum()),
        "class_counts": {
            str(int(row.class_id)): int(row.count)
            for row in summary.itertuples(index=False)
        },
        "thresholds": thresholds,
        "class_definitions": {
            str(class_id): {
                "key": info[0],
                "name_cn": info[1],
                "definition_cn": info[2],
            }
            for class_id, info in CLASS_INFO.items()
        },
        "notes": [
            "Class 5 thresholds are calculated over all systems with complete seed0 results.",
            "Feasibility50 seed0 consistency is recomputed from its 25-model pair table using only seed0 pairs.",
            "Class 6 includes 31 systems not run because resolved total length is at least 1500 aa.",
            (
                "Same-UniProt systems use direct/swapped best-chain DockQ."
                if dockq_mode == "symmetry-aware"
                else "Same-UniProt negative labels carry a symmetry-review flag because DockQ used a fixed native-to-model chain map."
            ),
        ],
    }
    (output_dir / "classification_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--watch-root",
        type=Path,
        required=True,
        help="Directory containing the derived PINDER-Val screening tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the master table and per-class lists.",
    )
    parser.add_argument(
        "--dockq-mode",
        choices=("fixed", "symmetry-aware"),
        default="fixed",
        help="DockQ chain-mapping input to use for class labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.watch_root.resolve()
    output_dir = args.output_dir.resolve()

    metadata_path = root / "pinder_val_manifest.csv"
    metadata = pd.read_csv(metadata_path)
    require_columns(
        metadata,
        {
            "id",
            "pdb_id",
            "cluster_id",
            "cluster_id_R",
            "cluster_id_L",
            "uniprot_R",
            "uniprot_L",
            "chain1_neff",
            "chain2_neff",
            "contains_antibody",
            "contains_antigen",
            "contains_enzyme",
        },
        metadata_path,
    )
    require_unique(metadata, ["id"], metadata_path)
    if len(metadata) != 1958:
        raise ValueError(f"Expected 1,958 PINDER-Val rows, found {len(metadata)}")
    metadata["manifest_order"] = range(1, len(metadata) + 1)

    computed = pd.concat(
        [
            load_group(root, group_number, args.dockq_mode)
            for group_number in (1, 2, 3)
        ]
        + [load_feasibility(root, args.dockq_mode)],
        ignore_index=True,
        sort=False,
    )
    require_unique(computed, ["id"], root)
    if len(computed) != 1927:
        raise ValueError(f"Expected 1,927 computed systems, found {len(computed)}")
    if not set(computed["id"]).issubset(set(metadata["id"])):
        raise ValueError("Computed system IDs are not a subset of PINDER-Val")
    computed["data_status"] = "ok"
    computed["technical_reason"] = ""

    master = metadata.merge(computed, on="id", how="left", validate="one_to_one")
    missing = master["data_status"].isna()
    if int(missing.sum()) != 31:
        raise ValueError(f"Expected 31 uncomputed systems, found {int(missing.sum())}")
    pre_filter_lengths = recover_pre_filter_lengths(root)
    missing_lengths = pre_filter_lengths[pre_filter_lengths["id"].isin(master.loc[missing, "id"])]
    if len(missing_lengths) != 31 or not missing_lengths["total_length"].ge(1500).all():
        raise ValueError("The 31 uncomputed systems are not exactly the >=1500 aa exclusions")
    missing_length_map = missing_lengths.set_index("id")
    for column in ("length_R", "length_L", "total_length"):
        master.loc[missing, column] = master.loc[missing, "id"].map(
            missing_length_map[column]
        )
    master.loc[missing, "source_group"] = "not_run_ge1500"
    master.loc[missing, "ranking_source"] = "not_available"
    master.loc[missing, "data_status"] = "not_run"
    master.loc[missing, "technical_reason"] = "not_run_total_length_ge_1500"

    master["min_chain_neff"] = master[["chain1_neff", "chain2_neff"]].min(axis=1)
    min_length = master[["length_R", "length_L"]].min(axis=1)
    max_length = master[["length_R", "length_L"]].max(axis=1)
    master["length_ratio_max_to_min"] = max_length / min_length
    master["same_uniprot"] = master["uniprot_R"].eq(master["uniprot_L"])
    master["same_chain_cluster"] = master["cluster_id_R"].eq(master["cluster_id_L"])
    master["homomer_like"] = master["same_uniprot"] | master["same_chain_cluster"]
    master["antibody_or_antigen"] = master["contains_antibody"] | master[
        "contains_antigen"
    ]
    master, thresholds = classify(master)
    master["dockq_symmetry_review_recommended"] = (
        args.dockq_mode == "fixed"
    ) & master["same_uniprot"] & master["class_id"].isin([3, 4, 5])
    master["classification_scope"] = (
        "seed0_5weights_symmetry_aware"
        if args.dockq_mode == "symmetry-aware"
        else "seed0_5weights_fixed_mapping"
    )

    preferred_columns = [
        "manifest_order",
        "id",
        "source_group",
        "classification_scope",
        "class_id",
        "class_key",
        "class_name_cn",
        "classification_reason",
        "data_status",
        "technical_reason",
        "rank1_dockq",
        "oracle5_dockq",
        "n_correct_5",
        "oracle_gain",
        "rank1_mapping_mode",
        "rank1_selected_model_chains",
        "rank1_symmetry_gain",
        "max_symmetry_gain_5",
        "n_swapped_selected_5",
        "rank1_correct",
        "oracle5_correct",
        "rank1_iptm",
        "rank1_iptm_full_precision",
        "iptm_min_5",
        "iptm_max_5",
        "iptm_range_5",
        "rank1_ranking_confidence",
        "rank1_ilis",
        "max_ilis_5",
        "rank1_pdockq2_min",
        "max_pdockq2_min_5",
        "seed0_mean_contact_jaccard",
        "seed0_max_interface_cluster_fraction",
        "seed0_median_pose_rmsd",
        "nonempty_model_count",
        "high_confidence_flag",
        "high_consistency_flag",
        "has_empty_interface_model",
        "any_model_near_dockq_threshold",
        "same_uniprot",
        "same_chain_cluster",
        "homomer_like",
        "dockq_symmetry_review_recommended",
        "antibody_or_antigen",
        "length_R",
        "length_L",
        "total_length",
        "length_ratio_max_to_min",
        "min_chain_neff",
        "ranking_source",
        "pdb_id",
        "cluster_id",
        "cluster_id_R",
        "cluster_id_L",
        "uniprot_R",
        "uniprot_L",
        "contains_antibody",
        "contains_antigen",
        "contains_enzyme",
    ]
    preferred_columns = [
        column for column in preferred_columns if column in master.columns
    ]
    remaining_columns = [
        column for column in master.columns if column not in preferred_columns
    ]
    master = master[preferred_columns + remaining_columns].sort_values("manifest_order")
    summary = make_summary(master)
    if len(master) != 1958 or master["id"].nunique() != 1958:
        raise ValueError("Final master table is not a one-to-one partition of PINDER-Val")
    if set(master["class_id"]) != {1, 2, 3, 4, 5, 6}:
        raise ValueError("Not all six classes are represented")
    if int(summary["count"].sum()) != 1958:
        raise ValueError("Class counts do not sum to 1,958")

    write_outputs(master, summary, thresholds, output_dir, args.dockq_mode)
    print(f"Wrote complete six-class partition to {output_dir}")
    print(summary.to_string(index=False))
    print("Thresholds:", json.dumps(thresholds, sort_keys=True))


if __name__ == "__main__":
    main()
