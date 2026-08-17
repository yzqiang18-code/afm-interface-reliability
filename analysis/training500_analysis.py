#!/usr/bin/env python3
"""Audit and analyze the Training500 multi-seed metric outputs."""

from __future__ import annotations

import argparse
import json
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPOSITORY_ROOT / "data" / "derived" / "training_500_metrics"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "generated" / "training500"
DOCKQ_ACCEPTABLE = 0.23
BOOTSTRAP_REPS = 1_000
RANDOM_SEED = 20_260_812


def spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(
        frame["x"].rank(method="average").corr(
            frame["y"].rank(method="average"), method="pearson"
        )
    )


def average_precision(y_true: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    valid = np.isfinite(s)
    y, s = y[valid], s[valid]
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    group_ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    cumulative_tp = np.cumsum(y)
    cumulative_fp = np.cumsum(1 - y)
    result = 0.0
    previous_recall = 0.0
    for end in group_ends:
        recall = cumulative_tp[end] / positives
        precision = cumulative_tp[end] / (cumulative_tp[end] + cumulative_fp[end])
        result += (recall - previous_recall) * precision
        previous_recall = recall
    return float(result)


def roc_auc(y_true: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = pd.Series(np.asarray(scores, dtype=float))
    valid = s.notna().to_numpy()
    y, s = y[valid], s[valid]
    positives = int(y.sum())
    negatives = int((1 - y).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = s.rank(method="average").to_numpy()
    rank_sum_positive = float(ranks[y == 1].sum())
    return (
        rank_sum_positive - positives * (positives + 1) / 2
    ) / (positives * negatives)


def metric_performance(frame: pd.DataFrame, metric: str) -> dict[str, float | int | str]:
    subset = frame[[metric, "DockQ"]].dropna()
    y = subset["DockQ"].ge(DOCKQ_ACCEPTABLE).astype(int)
    return {
        "metric": metric,
        "n": int(len(subset)),
        "positive_count": int(y.sum()),
        "positive_rate": float(y.mean()),
        "spearman_rho": spearman(subset[metric], subset["DockQ"]),
        "roc_auc": roc_auc(y, subset[metric]),
        "average_precision": average_precision(y, subset[metric]),
    }


def bootstrap_grouped_performance(
    frame: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """Bootstrap complete systems so correlated predictions stay together."""

    systems = frame["complex_id"].drop_duplicates().to_numpy()
    by_system = {system: group for system, group in frame.groupby("complex_id")}
    rng = np.random.default_rng(RANDOM_SEED)
    distributions: dict[tuple[str, str], list[float]] = {
        (metric, statistic): []
        for metric in metrics
        for statistic in ("spearman_rho", "roc_auc", "average_precision")
    }
    for _ in range(BOOTSTRAP_REPS):
        sampled_ids = rng.choice(systems, size=len(systems), replace=True)
        sampled = pd.concat(
            [by_system[system].assign(_bootstrap_group=index) for index, system in enumerate(sampled_ids)],
            ignore_index=True,
        )
        for metric in metrics:
            result = metric_performance(sampled, metric)
            for statistic in ("spearman_rho", "roc_auc", "average_precision"):
                value = float(result[statistic])
                if np.isfinite(value):
                    distributions[(metric, statistic)].append(value)

    rows: list[dict[str, float | int | str]] = []
    for metric in metrics:
        point = metric_performance(frame, metric)
        row = dict(point)
        for statistic in ("spearman_rho", "roc_auc", "average_precision"):
            values = distributions[(metric, statistic)]
            row[f"{statistic}_ci_low"] = float(np.quantile(values, 0.025))
            row[f"{statistic}_ci_high"] = float(np.quantile(values, 0.975))
            row[f"{statistic}_bootstrap_valid"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows)


def precision_at_fraction(frame: pd.DataFrame, score: str, fraction: float) -> dict[str, float | int | str]:
    count = max(1, math.ceil(len(frame) * fraction))
    selected = frame.sort_values([score, "complex_id"], ascending=[False, True]).head(count)
    return {
        "score": score,
        "fraction": fraction,
        "selected_count": int(len(selected)),
        "acceptable_rate": float(selected["DockQ"].ge(DOCKQ_ACCEPTABLE).mean()),
        "median_DockQ": float(selected["DockQ"].median()),
        "cutoff": float(selected[score].min()),
    }


def selector_summary(model: pd.DataFrame, score: str, label: str) -> tuple[pd.DataFrame, dict]:
    chosen = (
        model.sort_values(["complex_id", score, "rank"], ascending=[True, False, True])
        .groupby("complex_id", as_index=False)
        .first()
    )
    chosen["selector"] = label
    summary = {
        "selector": label,
        "score": score,
        "acceptable_rate": float(chosen["DockQ"].ge(DOCKQ_ACCEPTABLE).mean()),
        "medium_high_rate": float(chosen["DockQ"].ge(0.49).mean()),
        "high_rate": float(chosen["DockQ"].ge(0.80).mean()),
        "mean_DockQ": float(chosen["DockQ"].mean()),
        "median_DockQ": float(chosen["DockQ"].median()),
    }
    return chosen, summary


def fit_ridge_logistic(
    x: np.ndarray,
    y: np.ndarray,
    penalty: float = 1.0,
    max_iter: int = 100,
) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(design.shape[1], dtype=float)
    penalty_matrix = np.eye(design.shape[1], dtype=float) * penalty
    penalty_matrix[0, 0] = 0.0
    for _ in range(max_iter):
        eta = np.clip(design @ beta, -30, 30)
        probability = 1.0 / (1.0 + np.exp(-eta))
        weights = np.clip(probability * (1.0 - probability), 1e-7, None)
        gradient = design.T @ (y - probability) - penalty_matrix @ beta
        information = design.T @ (design * weights[:, None]) + penalty_matrix
        step = np.linalg.solve(information, gradient)
        beta_next = beta + step
        if np.max(np.abs(beta_next - beta)) < 1e-8:
            beta = beta_next
            break
        beta = beta_next
    return beta


def group_fold_predictions(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> pd.DataFrame:
    result = frame[
        ["complex_id", "rank", "model_weight", "seed", "cv_fold", "DockQ", "acceptable"]
    ].copy()
    for model_name, features in feature_sets.items():
        predictions = np.full(len(frame), np.nan, dtype=float)
        for fold in sorted(frame["cv_fold"].unique()):
            train_mask = frame["cv_fold"].ne(fold).to_numpy()
            test_mask = ~train_mask
            x_train = frame.loc[train_mask, features].to_numpy(dtype=float)
            x_test = frame.loc[test_mask, features].to_numpy(dtype=float)
            y_train = frame.loc[train_mask, "acceptable"].astype(float).to_numpy()
            medians = np.nanmedian(x_train, axis=0)
            medians[~np.isfinite(medians)] = 0.0
            x_train = np.where(np.isfinite(x_train), x_train, medians)
            x_test = np.where(np.isfinite(x_test), x_test, medians)
            means = x_train.mean(axis=0)
            scales = x_train.std(axis=0, ddof=0)
            scales[~np.isfinite(scales) | (scales == 0)] = 1.0
            x_train = (x_train - means) / scales
            x_test = (x_test - means) / scales
            beta = fit_ridge_logistic(x_train, y_train, penalty=1.0)
            eta = np.clip(np.column_stack([np.ones(len(x_test)), x_test]) @ beta, -30, 30)
            predictions[test_mask] = 1.0 / (1.0 + np.exp(-eta))
        if not np.isfinite(predictions).all():
            raise ValueError(f"OOF predictions are incomplete for {model_name}")
        result[model_name] = predictions
    return result


def system_fold_predictions(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    outcome: str,
) -> pd.DataFrame:
    result = frame[["complex_id", "cv_fold", outcome]].copy()
    y_all = frame[outcome].astype(float).to_numpy()
    for model_name, features in feature_sets.items():
        predictions = np.full(len(frame), np.nan, dtype=float)
        for fold in sorted(frame["cv_fold"].unique()):
            train_mask = frame["cv_fold"].ne(fold).to_numpy()
            test_mask = ~train_mask
            x_train = frame.loc[train_mask, features].to_numpy(dtype=float)
            x_test = frame.loc[test_mask, features].to_numpy(dtype=float)
            medians = np.nanmedian(x_train, axis=0)
            medians[~np.isfinite(medians)] = 0.0
            x_train = np.where(np.isfinite(x_train), x_train, medians)
            x_test = np.where(np.isfinite(x_test), x_test, medians)
            means = x_train.mean(axis=0)
            scales = x_train.std(axis=0, ddof=0)
            scales[~np.isfinite(scales) | (scales == 0)] = 1.0
            x_train = (x_train - means) / scales
            x_test = (x_test - means) / scales
            beta = fit_ridge_logistic(x_train, y_all[train_mask], penalty=1.0)
            eta = np.clip(np.column_stack([np.ones(len(x_test)), x_test]) @ beta, -30, 30)
            predictions[test_mask] = 1.0 / (1.0 + np.exp(-eta))
        if not np.isfinite(predictions).all():
            raise ValueError(f"System OOF predictions are incomplete for {model_name}")
        result[model_name] = predictions
    return result


def brier_score(y_true: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    valid = np.isfinite(y) & np.isfinite(s)
    return float(np.mean((y[valid] - s[valid]) ** 2))


def paired_selector_bootstrap(
    choices: pd.DataFrame,
    reference: str,
    comparisons: list[str],
) -> pd.DataFrame:
    wide_dockq = choices.pivot(index="complex_id", columns="selector", values="DockQ")
    systems = wide_dockq.index.to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict] = []
    for comparison in comparisons:
        observed_delta_rate = float(
            wide_dockq[comparison].ge(DOCKQ_ACCEPTABLE).mean()
            - wide_dockq[reference].ge(DOCKQ_ACCEPTABLE).mean()
        )
        observed_delta_dockq = float((wide_dockq[comparison] - wide_dockq[reference]).mean())
        rate_deltas: list[float] = []
        dockq_deltas: list[float] = []
        for _ in range(BOOTSTRAP_REPS):
            sampled = rng.choice(systems, size=len(systems), replace=True)
            sample = wide_dockq.loc[sampled]
            rate_deltas.append(
                float(
                    sample[comparison].ge(DOCKQ_ACCEPTABLE).mean()
                    - sample[reference].ge(DOCKQ_ACCEPTABLE).mean()
                )
            )
            dockq_deltas.append(float((sample[comparison] - sample[reference]).mean()))
        rows.append(
            {
                "comparison": comparison,
                "reference": reference,
                "delta_acceptable_rate": observed_delta_rate,
                "delta_acceptable_rate_ci_low": float(np.quantile(rate_deltas, 0.025)),
                "delta_acceptable_rate_ci_high": float(np.quantile(rate_deltas, 0.975)),
                "delta_mean_DockQ": observed_delta_dockq,
                "delta_mean_DockQ_ci_low": float(np.quantile(dockq_deltas, 0.025)),
                "delta_mean_DockQ_ci_high": float(np.quantile(dockq_deltas, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def paired_prediction_bootstrap(
    frame: pd.DataFrame,
    outcome: str,
    reference: str,
    comparisons: list[str],
) -> pd.DataFrame:
    systems = frame["complex_id"].drop_duplicates().to_numpy()
    by_system = {system: group for system, group in frame.groupby("complex_id")}
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict] = []

    def statistics(sample: pd.DataFrame, score: str) -> dict[str, float]:
        y = sample[outcome].astype(int)
        return {
            "roc_auc": roc_auc(y, sample[score]),
            "average_precision": average_precision(y, sample[score]),
            "brier": brier_score(y, sample[score]),
        }

    reference_point = statistics(frame, reference)
    for comparison in comparisons:
        comparison_point = statistics(frame, comparison)
        distributions = {key: [] for key in reference_point}
        for _ in range(BOOTSTRAP_REPS):
            sampled_ids = rng.choice(systems, size=len(systems), replace=True)
            sample = pd.concat(
                [by_system[system].assign(_bootstrap_group=index) for index, system in enumerate(sampled_ids)],
                ignore_index=True,
            )
            ref = statistics(sample, reference)
            comp = statistics(sample, comparison)
            for statistic in distributions:
                delta = comp[statistic] - ref[statistic]
                if np.isfinite(delta):
                    distributions[statistic].append(delta)
        row: dict[str, float | str] = {"comparison": comparison, "reference": reference}
        for statistic, values in distributions.items():
            row[f"delta_{statistic}"] = comparison_point[statistic] - reference_point[statistic]
            row[f"delta_{statistic}_ci_low"] = float(np.quantile(values, 0.025))
            row[f"delta_{statistic}_ci_high"] = float(np.quantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def clean_records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.replace([np.inf, -np.inf], np.nan).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and analyze the Training500 multi-seed metric bundle."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Derived Training500 metric-bundle root (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for regenerated tables and summaries (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(
            f"Training500 input root does not exist: {input_root}. "
            "See analysis/README.md for the expected layout."
        )

    confidence = pd.read_csv(input_root / "afm_confidence" / "full_precision_ranking.csv")
    dockq = pd.read_csv(input_root / "dockq" / "afm23_5models_seeds0-3_dockq_symmetry.csv")
    ilis = pd.read_csv(input_root / "ilis" / "afm23_5models_seeds0-3_ilis.csv")
    pdockq2 = pd.read_csv(input_root / "pdockq2" / "afm23_5models_seeds0-3_pdockq2.csv")
    physics = pd.read_csv(input_root / "physics" / "afm23_5models_seeds0-3_physics.csv")
    consistency_models = pd.read_csv(
        input_root / "consistency" / "afm23_5models_seeds0-3_consistency_models.csv"
    )
    consistency_summary = pd.read_csv(
        input_root / "consistency" / "afm23_5models_seeds0-3_consistency_summary.csv"
    )
    consistency_pairs = pd.read_csv(
        input_root / "consistency" / "afm23_5models_seeds0-3_consistency_pairs.csv",
        low_memory=False,
    )
    metadata = pd.read_csv(input_root / "metadata" / "recommended_training_500.csv")
    assignment = pd.read_csv(input_root / "metadata" / "assignment.csv")

    ilis = ilis.rename(columns={"name": "complex_id", "model": "model_weight"})
    if "seed" not in ilis:
        ilis["seed"] = ilis["structure_file"].str.extract(r"_seed_(\d+)", expand=False).astype(int)
    confidence = confidence.rename(columns={"system_id": "complex_id"})

    key = ["complex_id", "rank", "model_weight", "seed"]
    model = dockq.copy()
    merge_audits: list[dict[str, int | str]] = []
    sources = [
        (
            "confidence",
            confidence[key + ["ranking_confidence", "iptm", "ptm"]],
            {"iptm": "iptm_full_precision", "ptm": "ptm_full_precision"},
        ),
        (
            "ilis",
            ilis[
                key
                + [
                    "iLIS",
                    "iLIA",
                    "iLISA",
                    "ipSAE",
                    "actifpTM",
                    "LIS",
                    "cLIS",
                    "ipTM",
                    "pTM",
                    "pLDDT",
                    "LIpLDDT",
                    "cLIpLDDT",
                ]
            ],
            {"ipTM": "ipTM_ilis", "pTM": "pTM_ilis"},
        ),
        (
            "pdockq2",
            pdockq2[
                key
                + [
                    "pDockQ2_chain1_to_chain2",
                    "pDockQ2_chain2_to_chain1",
                    "pDockQ2_min",
                    "pDockQ2_mean",
                    "pDockQ2_max",
                ]
            ],
            {},
        ),
        (
            "physics",
            physics[
                key
                + [
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
            ],
            {},
        ),
        (
            "consistency_models",
            consistency_models[
                key
                + [
                    "contact_count",
                    "interface_residue_count_a",
                    "interface_residue_count_b",
                    "cluster_id",
                    "contact_count_cb8",
                    "cluster_id_cb8",
                ]
            ],
            {},
        ),
    ]

    for name, source, rename in sources:
        source = source.rename(columns=rename)
        before = len(model)
        model = model.merge(source, on=key, how="outer", validate="one_to_one", indicator=f"_{name}_merge")
        unmatched = int(model[f"_{name}_merge"].ne("both").sum())
        merge_audits.append({"source": name, "before_rows": before, "after_rows": len(model), "unmatched_rows": unmatched})
        model = model.drop(columns=[f"_{name}_merge"])

    model = model.merge(
        metadata[
            [
                "id",
                "class_id",
                "class_key",
                "class_name_cn",
                "same_uniprot",
                "source_group",
                "cv_fold",
                "total_length",
                "length_ratio_max_to_min",
                "min_chain_neff",
                "contains_antibody",
                "contains_antigen",
                "contains_enzyme",
                "selection_reason",
            ]
        ].rename(columns={"id": "complex_id"}),
        on="complex_id",
        how="left",
        validate="many_to_one",
    )
    model["acceptable"] = model["DockQ"].ge(DOCKQ_ACCEPTABLE)
    model["DockQ_category"] = pd.cut(
        model["DockQ"],
        bins=[-np.inf, 0.23, 0.49, 0.80, np.inf],
        labels=["incorrect", "acceptable", "medium", "high"],
        right=False,
    )
    model["negative_clash_density"] = -model["clash_density"]
    model["negative_backbone_clashes"] = -model["backbone_backbone_clash_count"]
    model["negative_same_charge_density"] = -model["same_charge_contact_density"]
    model["negative_component_count"] = -model["contact_component_count"]

    required_metric_columns = [
        "DockQ",
        "ranking_confidence",
        "iptm_full_precision",
        "ptm_full_precision",
        "iLIS",
        "ipSAE",
        "pDockQ2_min",
        "contact_pair_count",
        "bsa_a2",
        "clash_count",
    ]
    key_counts = model.groupby("complex_id").agg(
        row_count=("rank", "size"),
        unique_rank_count=("rank", "nunique"),
        unique_seed_count=("seed", "nunique"),
        unique_weight_count=("model_weight", "nunique"),
    )
    expected_pairs = len(consistency_summary) * math.comb(20, 2)
    pair_reason_counts = {
        "heavy_atom_invalid": int((~consistency_pairs["jaccard_valid"]).sum()),
        "cb8_invalid": int((~consistency_pairs["jaccard_cb8_valid"]).sum()),
    }
    data_quality = {
        "expected_systems": 500,
        "expected_predictions": 10_000,
        "expected_pairs": expected_pairs,
        "observed_rows": {
            "confidence": int(len(confidence)),
            "dockq": int(len(dockq)),
            "ilis": int(len(ilis)),
            "pdockq2": int(len(pdockq2)),
            "physics": int(len(physics)),
            "consistency_models": int(len(consistency_models)),
            "consistency_summary": int(len(consistency_summary)),
            "consistency_pairs": int(len(consistency_pairs)),
            "metadata": int(len(metadata)),
            "assignment": int(len(assignment)),
        },
        "observed_systems": int(model["complex_id"].nunique()),
        "duplicate_model_keys": int(model.duplicated(key).sum()),
        "systems_not_exactly_20_predictions": int(key_counts["row_count"].ne(20).sum()),
        "systems_without_ranks_1_to_20": int(key_counts["unique_rank_count"].ne(20).sum()),
        "systems_without_four_seeds": int(key_counts["unique_seed_count"].ne(4).sum()),
        "systems_without_five_weights": int(key_counts["unique_weight_count"].ne(5).sum()),
        "merge_audits": merge_audits,
        "required_metric_missingness": {
            column: int(model[column].isna().sum()) for column in required_metric_columns
        },
        "dockq_status_counts": dockq["status"].value_counts(dropna=False).to_dict(),
        "pdockq2_status_counts": pdockq2["status"].value_counts(dropna=False).to_dict(),
        "physics_status_counts": physics["status"].value_counts(dropna=False).to_dict(),
        "physics_interface_status_counts": physics["interface_status"].value_counts(dropna=False).to_dict(),
        "consistency_status_counts": consistency_summary["status"].value_counts(dropna=False).to_dict(),
        "ensemble_incomplete_count": int((~consistency_summary["ensemble_complete"]).sum()),
        "invalid_pair_counts": pair_reason_counts,
        "invalid_pair_rates": {
            key: value / expected_pairs for key, value in pair_reason_counts.items()
        },
        "symmetry_mapping_rows": int(dockq["mapping_mode"].eq("symmetry_aware").sum()),
        "fixed_mapping_rows": int(dockq["mapping_mode"].eq("fixed").sum()),
        "swapped_mapping_rows": int(dockq["selected_model_chains"].eq("BA").sum()),
        "swapped_mapping_systems": int(
            dockq.loc[dockq["selected_model_chains"].eq("BA"), "complex_id"].nunique()
        ),
        "positive_symmetry_gain_rows": int(dockq["symmetry_gain"].gt(1e-12).sum()),
    }

    selector_specs = [
        ("ranking_confidence", "AF-M full-precision rank-1"),
        ("iptm_full_precision", "Maximum ipTM"),
        ("pDockQ2_min", "Maximum pDockQ2-min"),
        ("iLIS", "Maximum iLIS"),
        ("ipSAE", "Maximum ipSAE"),
        ("bsa_a2", "Maximum BSA"),
        ("largest_contact_component_fraction", "Maximum connected-interface fraction"),
        ("negative_clash_density", "Minimum clash density"),
    ]
    selector_frames: list[pd.DataFrame] = []
    selector_rows: list[dict] = []
    for score, label in selector_specs:
        chosen, row = selector_summary(model, score, label)
        selector_frames.append(chosen)
        selector_rows.append(row)
    oracle = (
        model.sort_values(["complex_id", "DockQ", "rank"], ascending=[True, False, True])
        .groupby("complex_id", as_index=False)
        .first()
    )
    oracle["selector"] = "Oracle best-of-20"
    selector_frames.append(oracle)
    selector_rows.append(
        {
            "selector": "Oracle best-of-20",
            "score": "DockQ",
            "acceptable_rate": float(oracle["acceptable"].mean()),
            "medium_high_rate": float(oracle["DockQ"].ge(0.49).mean()),
            "high_rate": float(oracle["DockQ"].ge(0.80).mean()),
            "mean_DockQ": float(oracle["DockQ"].mean()),
            "median_DockQ": float(oracle["DockQ"].median()),
        }
    )
    selector_summary_frame = pd.DataFrame(selector_rows)
    selector_choices = pd.concat(selector_frames, ignore_index=True)
    top1 = selector_frames[0].copy()
    top1["top1_DockQ"] = top1["DockQ"]
    oracle_system = oracle[["complex_id", "DockQ", "acceptable"]].rename(
        columns={"DockQ": "oracle20_DockQ", "acceptable": "oracle20_acceptable"}
    )

    system_aggregates = (
        model.groupby("complex_id", as_index=False)
        .agg(
            acceptable_model_fraction=("acceptable", "mean"),
            mean_model_DockQ=("DockQ", "mean"),
            min_model_DockQ=("DockQ", "min"),
            std_model_DockQ=("DockQ", "std"),
            dockq_range=("DockQ", lambda values: float(values.max() - values.min())),
            rank1_DockQ=("DockQ", lambda values: float(values.loc[model.loc[values.index, "rank"].idxmin()])),
        )
    )
    system = (
        top1[
            [
                "complex_id",
                "DockQ",
                "acceptable",
                "DockQ_category",
                "ranking_confidence",
                "iptm_full_precision",
                "pDockQ2_min",
                "iLIS",
                "ipSAE",
                "bsa_a2",
                "clash_density",
                "backbone_backbone_clash_count",
                "largest_contact_component_fraction",
                "class_id",
                "class_key",
                "class_name_cn",
                "same_uniprot",
                "source_group",
                "cv_fold",
                "total_length",
                "length_ratio_max_to_min",
                "min_chain_neff",
                "contains_enzyme",
            ]
        ]
        .rename(columns={"DockQ": "top1_DockQ", "acceptable": "top1_acceptable", "DockQ_category": "top1_DockQ_category"})
        .merge(oracle_system, on="complex_id", validate="one_to_one")
        .merge(system_aggregates.drop(columns="rank1_DockQ"), on="complex_id", validate="one_to_one")
        .merge(consistency_summary, on="complex_id", validate="one_to_one", suffixes=("", "_consistency"))
    )
    system["oracle_rescue"] = (~system["top1_acceptable"]) & system["oracle20_acceptable"]
    system["never_acceptable"] = ~system["oracle20_acceptable"]
    system["top1_oracle_gap"] = system["oracle20_DockQ"] - system["top1_DockQ"]
    system["all_models_acceptable"] = system["acceptable_model_fraction"].eq(1.0)
    system["stable_wrong"] = (
        system["never_acceptable"]
        & system["mean_contact_jaccard"].ge(0.80)
        & system["max_interface_cluster_fraction"].ge(0.80)
    )

    feature_sets = {
        "M1_ipTM": ["iptm_full_precision"],
        "M2_AF_confidence": [
            "iptm_full_precision",
            "ptm_full_precision",
            "pDockQ2_min",
            "iLIS",
            "ipSAE",
        ],
        "M3_AF_plus_physics": [
            "iptm_full_precision",
            "ptm_full_precision",
            "pDockQ2_min",
            "iLIS",
            "ipSAE",
            "log1p_bsa_a2",
            "bsa_per_interface_residue",
            "negative_clash_density",
            "negative_backbone_clashes",
            "largest_contact_component_fraction",
            "salt_bridge_density",
            "negative_same_charge_density",
        ],
    }
    oof_predictions = group_fold_predictions(model, feature_sets)
    oof_rows: list[dict] = []
    oof_selector_frames: list[pd.DataFrame] = []
    for model_name in feature_sets:
        overall = metric_performance(
            oof_predictions.rename(columns={model_name: "score"}), "score"
        )
        chosen = (
            oof_predictions.sort_values(
                ["complex_id", model_name, "rank"], ascending=[True, False, True]
            )
            .groupby("complex_id", as_index=False)
            .first()
        )
        chosen["selector"] = model_name
        oof_selector_frames.append(chosen)
        oof_rows.append(
            {
                "model": model_name,
                "features": ";".join(feature_sets[model_name]),
                "prediction_roc_auc": overall["roc_auc"],
                "prediction_average_precision": overall["average_precision"],
                "prediction_brier": brier_score(
                    oof_predictions["acceptable"], oof_predictions[model_name]
                ),
                "selector_acceptable_rate": float(chosen["acceptable"].mean()),
                "selector_medium_high_rate": float(chosen["DockQ"].ge(0.49).mean()),
                "selector_mean_DockQ": float(chosen["DockQ"].mean()),
                "selector_median_DockQ": float(chosen["DockQ"].median()),
            }
        )
    oof_performance = pd.DataFrame(oof_rows)
    oof_choices = pd.concat(oof_selector_frames, ignore_index=True)
    reference_choices = top1[
        ["complex_id", "rank", "model_weight", "seed", "DockQ", "acceptable"]
    ].copy()
    reference_choices["selector"] = "AF-M full-precision rank-1"
    oof_bootstrap = paired_selector_bootstrap(
        pd.concat([reference_choices, oof_choices], ignore_index=True),
        reference="AF-M full-precision rank-1",
        comparisons=list(feature_sets),
    )

    system_feature_sets = {
        "S1_top1_AF": [
            "ranking_confidence",
            "iptm_full_precision",
            "pDockQ2_min",
            "iLIS",
            "ipSAE",
        ],
        "S2_top1_AF_plus_physics": [
            "ranking_confidence",
            "iptm_full_precision",
            "pDockQ2_min",
            "iLIS",
            "ipSAE",
            "bsa_a2",
            "clash_density",
            "backbone_backbone_clash_count",
            "largest_contact_component_fraction",
        ],
        "S3_top1_AF_plus_ensemble": [
            "ranking_confidence",
            "iptm_full_precision",
            "pDockQ2_min",
            "iLIS",
            "ipSAE",
            "mean_contact_jaccard",
            "max_interface_cluster_fraction",
            "median_receptor_aligned_ligand_rmsd",
            "iptm_std_across_models",
        ],
        "S4_all_features": [
            "ranking_confidence",
            "iptm_full_precision",
            "pDockQ2_min",
            "iLIS",
            "ipSAE",
            "bsa_a2",
            "clash_density",
            "backbone_backbone_clash_count",
            "largest_contact_component_fraction",
            "mean_contact_jaccard",
            "max_interface_cluster_fraction",
            "median_receptor_aligned_ligand_rmsd",
            "iptm_std_across_models",
        ],
    }
    system_oof = system_fold_predictions(
        system,
        system_feature_sets,
        outcome="top1_acceptable",
    )
    system_oof_rows: list[dict] = []
    for model_name, features in system_feature_sets.items():
        y = system_oof["top1_acceptable"].astype(int)
        system_oof_rows.append(
            {
                "model": model_name,
                "features": ";".join(features),
                "roc_auc": roc_auc(y, system_oof[model_name]),
                "average_precision": average_precision(y, system_oof[model_name]),
                "brier": brier_score(y, system_oof[model_name]),
            }
        )
    system_oof_performance = pd.DataFrame(system_oof_rows)
    system_oof_deltas = paired_prediction_bootstrap(
        system_oof,
        outcome="top1_acceptable",
        reference="S1_top1_AF",
        comparisons=["S2_top1_AF_plus_physics", "S3_top1_AF_plus_ensemble", "S4_all_features"],
    )

    raw_metrics = [
        "ranking_confidence",
        "iptm_full_precision",
        "pDockQ2_min",
        "pDockQ2_mean",
        "iLIS",
        "ipSAE",
        "actifpTM",
        "bsa_a2",
        "bsa_per_interface_residue",
        "contact_density",
        "largest_contact_component_fraction",
        "hydrophobic_contact_fraction",
        "salt_bridge_density",
        "negative_clash_density",
        "negative_backbone_clashes",
        "negative_same_charge_density",
        "negative_component_count",
    ]
    performance = bootstrap_grouped_performance(model, raw_metrics)
    metric_group = {
        "ranking_confidence": "AF confidence",
        "iptm_full_precision": "AF confidence",
        "pDockQ2_min": "AF-derived confidence",
        "pDockQ2_mean": "AF-derived confidence",
        "iLIS": "AF-derived confidence",
        "ipSAE": "AF-derived confidence",
        "actifpTM": "AF-derived confidence",
        "bsa_a2": "Physics",
        "bsa_per_interface_residue": "Physics",
        "contact_density": "Physics",
        "largest_contact_component_fraction": "Physics",
        "hydrophobic_contact_fraction": "Physics",
        "salt_bridge_density": "Physics",
        "negative_clash_density": "Physics",
        "negative_backbone_clashes": "Physics",
        "negative_same_charge_density": "Physics",
        "negative_component_count": "Physics",
    }
    performance["metric_group"] = performance["metric"].map(metric_group)

    within_rows: list[dict] = []
    for metric in raw_metrics:
        per_system = model.groupby("complex_id").apply(
            lambda frame: spearman(frame[metric], frame["DockQ"]),
            include_groups=False,
        ).dropna()
        selector = selector_summary_frame.loc[selector_summary_frame["score"].eq(metric)]
        within_rows.append(
            {
                "metric": metric,
                "metric_group": metric_group[metric],
                "valid_systems": int(len(per_system)),
                "median_within_system_spearman": float(per_system.median()) if len(per_system) else float("nan"),
                "q25_within_system_spearman": float(per_system.quantile(0.25)) if len(per_system) else float("nan"),
                "q75_within_system_spearman": float(per_system.quantile(0.75)) if len(per_system) else float("nan"),
                "positive_within_system_fraction": float(per_system.gt(0).mean()) if len(per_system) else float("nan"),
                "selector_acceptable_rate": float(selector["acceptable_rate"].iloc[0]) if not selector.empty else float("nan"),
                "selector_mean_DockQ": float(selector["mean_DockQ"].iloc[0]) if not selector.empty else float("nan"),
            }
        )
    within_system = pd.DataFrame(within_rows)

    top1_metric_frame = system.rename(columns={"top1_DockQ": "DockQ"})
    system_metrics = [
        "ranking_confidence",
        "iptm_full_precision",
        "pDockQ2_min",
        "iLIS",
        "ipSAE",
        "mean_contact_jaccard",
        "max_interface_cluster_fraction",
        "median_receptor_aligned_ligand_rmsd",
        "iptm_std_across_models",
        "acceptable_model_fraction",
    ]
    system_performance_rows = []
    for metric in system_metrics:
        oriented_metric = metric
        if metric in {"median_receptor_aligned_ligand_rmsd", "iptm_std_across_models"}:
            oriented_metric = f"negative_{metric}"
            top1_metric_frame[oriented_metric] = -top1_metric_frame[metric]
        row = metric_performance(top1_metric_frame, oriented_metric)
        row["metric"] = metric
        system_performance_rows.append(row)
    system_performance = pd.DataFrame(system_performance_rows)

    precision_rows: list[dict] = []
    for score in ["ranking_confidence", "iptm_full_precision", "pDockQ2_min", "iLIS", "ipSAE"]:
        for fraction in [0.10, 0.20, 0.50, 1.00]:
            precision_rows.append(precision_at_fraction(top1_metric_frame, score, fraction))
    precision_at_coverage = pd.DataFrame(precision_rows)

    class_summary = (
        system.groupby(["class_id", "class_key", "class_name_cn"], as_index=False)
        .agg(
            systems=("complex_id", "size"),
            top1_acceptable_rate=("top1_acceptable", "mean"),
            oracle20_acceptable_rate=("oracle20_acceptable", "mean"),
            oracle_rescue_count=("oracle_rescue", "sum"),
            all_models_acceptable_rate=("all_models_acceptable", "mean"),
            median_acceptable_model_fraction=("acceptable_model_fraction", "median"),
            median_top1_DockQ=("top1_DockQ", "median"),
            median_oracle20_DockQ=("oracle20_DockQ", "median"),
        )
        .sort_values("class_id")
    )
    topology_summary = (
        system.assign(topology=system["same_uniprot"].map({True: "Same UniProt", False: "Different UniProt"}))
        .groupby("topology", as_index=False)
        .agg(
            systems=("complex_id", "size"),
            top1_acceptable_rate=("top1_acceptable", "mean"),
            oracle20_acceptable_rate=("oracle20_acceptable", "mean"),
            median_top1_DockQ=("top1_DockQ", "median"),
            median_contact_jaccard=("mean_contact_jaccard", "median"),
        )
    )
    fold_summary = (
        system.groupby("cv_fold", as_index=False)
        .agg(
            systems=("complex_id", "size"),
            top1_acceptable_rate=("top1_acceptable", "mean"),
            oracle20_acceptable_rate=("oracle20_acceptable", "mean"),
            median_top1_DockQ=("top1_DockQ", "median"),
            median_acceptable_model_fraction=("acceptable_model_fraction", "median"),
        )
    )

    category_counts = (
        model["DockQ_category"].value_counts(sort=False).rename_axis("quality_band").reset_index(name="count")
    )
    category_counts["fraction"] = category_counts["count"] / len(model)

    failures = system.loc[~system["top1_acceptable"]].copy()
    failures["failure_type"] = np.select(
        [failures["stable_wrong"], failures["never_acceptable"], failures["oracle_rescue"]],
        ["Stable wrong", "Sampling failure", "Rerank-rescuable"],
        default="Top-1 failure",
    )
    failures = failures.sort_values(
        ["stable_wrong", "never_acceptable", "top1_DockQ", "mean_contact_jaccard"],
        ascending=[False, False, True, False],
    )

    top1_row = selector_summary_frame.loc[
        selector_summary_frame["selector"].eq("AF-M full-precision rank-1")
    ].iloc[0]
    oracle_row = selector_summary_frame.loc[
        selector_summary_frame["selector"].eq("Oracle best-of-20")
    ].iloc[0]
    summary = {
        "cohort": {
            "systems": 500,
            "predictions": 10_000,
            "predictions_per_system": 20,
            "seeds": [0, 1, 2, 3],
            "model_weights": [1, 2, 3, 4, 5],
            "same_uniprot_systems": int(system["same_uniprot"].sum()),
            "different_uniprot_systems": int((~system["same_uniprot"]).sum()),
        },
        "quality": {
            "all_prediction_acceptable_rate": float(model["acceptable"].mean()),
            "top1_acceptable_rate": float(top1_row["acceptable_rate"]),
            "top1_medium_high_rate": float(top1_row["medium_high_rate"]),
            "top1_high_rate": float(top1_row["high_rate"]),
            "oracle20_acceptable_rate": float(oracle_row["acceptable_rate"]),
            "oracle20_medium_high_rate": float(oracle_row["medium_high_rate"]),
            "oracle_rescue_count": int(system["oracle_rescue"].sum()),
            "never_acceptable_count": int(system["never_acceptable"].sum()),
            "all_models_acceptable_count": int(system["all_models_acceptable"].sum()),
            "top1_error_count": int((~system["top1_acceptable"]).sum()),
            "mean_top1_oracle_gap": float(system["top1_oracle_gap"].mean()),
            "median_top1_oracle_gap": float(system["top1_oracle_gap"].median()),
        },
        "selection": {
            "rank1_mean_DockQ": float(top1_row["mean_DockQ"]),
            "rank1_median_DockQ": float(top1_row["median_DockQ"]),
            "oracle_mean_DockQ": float(oracle_row["mean_DockQ"]),
            "oracle_median_DockQ": float(oracle_row["median_DockQ"]),
            "oof_selector_acceptable_rates": {
                row.model: float(row.selector_acceptable_rate)
                for row in oof_performance.itertuples(index=False)
            },
            "system_oof_roc_auc": {
                row.model: float(row.roc_auc)
                for row in system_oof_performance.itertuples(index=False)
            },
        },
        "consistency": {
            "median_mean_contact_jaccard": float(system["mean_contact_jaccard"].median()),
            "median_max_cluster_fraction": float(system["max_interface_cluster_fraction"].median()),
            "median_pose_rmsd": float(system["median_receptor_aligned_ligand_rmsd"].median()),
            "stable_wrong_count": int(system["stable_wrong"].sum()),
            "spearman_jaccard_vs_acceptable_fraction": spearman(system["mean_contact_jaccard"], system["acceptable_model_fraction"]),
            "spearman_jaccard_vs_dockq_range": spearman(system["mean_contact_jaccard"], system["dockq_range"]),
            "spearman_pose_rmsd_vs_dockq_range": spearman(system["median_receptor_aligned_ligand_rmsd"], system["dockq_range"]),
        },
        "data_quality": data_quality,
        "selection_bias": (
            "Training500 is a deliberately stratified training set selected from seed-0 screening classes, "
            "with difficult and rerank-informative cases oversampled. Aggregate rates are descriptive of "
            "this training cohort and are not unbiased PINDER-Val benchmark estimates."
        ),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    model.to_csv(output_root / "model_level_merged.csv", index=False)
    system.to_csv(output_root / "system_level_summary.csv", index=False)
    selector_summary_frame.to_csv(output_root / "selector_summary.csv", index=False)
    selector_choices.to_csv(output_root / "selector_choices.csv", index=False)
    oof_predictions.to_csv(output_root / "oof_model_predictions.csv", index=False)
    oof_performance.to_csv(output_root / "oof_model_performance.csv", index=False)
    oof_choices.to_csv(output_root / "oof_selector_choices.csv", index=False)
    oof_bootstrap.to_csv(output_root / "oof_selector_bootstrap.csv", index=False)
    system_oof.to_csv(output_root / "system_oof_predictions.csv", index=False)
    system_oof_performance.to_csv(output_root / "system_oof_performance.csv", index=False)
    system_oof_deltas.to_csv(output_root / "system_oof_deltas.csv", index=False)
    performance.to_csv(output_root / "model_level_metric_performance.csv", index=False)
    within_system.to_csv(output_root / "within_system_metric_performance.csv", index=False)
    system_performance.to_csv(output_root / "system_level_metric_performance.csv", index=False)
    precision_at_coverage.to_csv(output_root / "top1_precision_at_coverage.csv", index=False)
    class_summary.to_csv(output_root / "class_summary.csv", index=False)
    topology_summary.to_csv(output_root / "topology_summary.csv", index=False)
    fold_summary.to_csv(output_root / "cv_fold_summary.csv", index=False)
    category_counts.to_csv(output_root / "dockq_category_counts.csv", index=False)
    failures.to_csv(output_root / "failure_cases.csv", index=False)
    (output_root / "data_quality.json").write_text(
        json.dumps(data_quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
