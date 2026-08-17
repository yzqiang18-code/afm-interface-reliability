#!/usr/bin/env python3
"""Shared utilities for the candidate-level ridge-logistic baseline."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


MODEL_SCHEMA_VERSION = 2
PREDICTION_SCORE_COLUMN = "acceptable_score"
MODEL_SELECTED_COLUMN = "model_selected"
REFERENCE_SELECTED_COLUMN = "reference_selected"
UNDEFINED_ACCESSIONS = frozenset({"", "UNDEFINED", "NONE", "NAN", "NA"})
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Record repository-relative paths and avoid publishing local home paths."""

    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return resolved.name


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    frame.to_csv(temporary, index=False)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def check_output_targets(paths: Iterable[Path], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        joined = "\n  ".join(existing)
        raise FileExistsError(
            "Refusing to replace existing output files without --overwrite:\n  "
            + joined
        )


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    source: str,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "model_name",
        "task",
        "key_columns",
        "label_mapping_policy",
        "group_column",
        "fold_column",
        "target_column",
        "target_threshold",
        "reference_score_column",
        "feature_columns",
        "ridge_penalty",
        "expected_candidates_per_system",
        "expected_fold_values",
        "max_missing_fraction_per_feature",
        "bootstrap_replicates",
        "random_seed",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Model config is missing fields: {missing}")
    if not config["feature_columns"]:
        raise ValueError("feature_columns must not be empty")
    if len(set(config["feature_columns"])) != len(config["feature_columns"]):
        raise ValueError("feature_columns contains duplicates")
    if float(config["ridge_penalty"]) <= 0:
        raise ValueError("ridge_penalty must be positive")
    if int(config["expected_candidates_per_system"]) <= 0:
        raise ValueError("expected_candidates_per_system must be positive")
    missing_limit = float(config["max_missing_fraction_per_feature"])
    if not 0 <= missing_limit <= 1:
        raise ValueError("max_missing_fraction_per_feature must be between 0 and 1")
    mapping_policy = config["label_mapping_policy"]
    if not isinstance(mapping_policy, dict):
        raise ValueError("label_mapping_policy must be a JSON object")
    required_mapping_fields = {
        "mode_column",
        "same_accession_mode",
        "different_accession_mode",
    }
    mapping_missing = sorted(required_mapping_fields.difference(mapping_policy))
    if mapping_missing:
        raise ValueError(
            f"label_mapping_policy is missing fields: {mapping_missing}"
        )
    return config


def numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def validate_candidate_frame(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    require_target: bool,
    require_folds: bool,
    source: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate grain, feature availability, folds, and optional labels."""

    config = validate_config(config)
    key_columns = list(config["key_columns"])
    group_column = str(config["group_column"])
    feature_columns = list(config["feature_columns"])
    required = key_columns + feature_columns + [str(config["reference_score_column"])]
    if require_target:
        required.append(str(config["target_column"]))
        required.append(str(config["label_mapping_policy"]["mode_column"]))
    if require_folds:
        required.append(str(config["fold_column"]))
    require_columns(frame, required, source=source)

    checked = frame.copy().reset_index(drop=True)
    if checked[key_columns].isna().any(axis=None):
        raise ValueError(f"{source} contains missing candidate-key values")
    duplicate_mask = checked.duplicated(key_columns, keep=False)
    if duplicate_mask.any():
        examples = checked.loc[duplicate_mask, key_columns].head(5).to_dict("records")
        raise ValueError(f"{source} contains duplicate candidate keys: {examples}")

    expected_candidates = int(config["expected_candidates_per_system"])
    group_sizes = checked.groupby(group_column, sort=False).size()
    bad_sizes = group_sizes[group_sizes.ne(expected_candidates)]
    if not bad_sizes.empty:
        raise ValueError(
            f"{source} does not have exactly {expected_candidates} candidates per "
            f"system; examples: {bad_sizes.head(5).to_dict()}"
        )

    for column in feature_columns + [str(config["reference_score_column"])]:
        checked[column] = pd.to_numeric(checked[column], errors="coerce")
        checked.loc[~np.isfinite(checked[column]), column] = np.nan

    feature_values = numeric_frame(checked, feature_columns)
    missing_fractions = feature_values.isna().mean()
    limit = float(config["max_missing_fraction_per_feature"])
    excessive = missing_fractions[missing_fractions.gt(limit)]
    if not excessive.empty:
        raise ValueError(
            f"{source} exceeds the feature missingness limit {limit}: "
            f"{excessive.to_dict()}"
        )
    checked.loc[:, feature_columns] = feature_values

    reference = pd.to_numeric(
        checked[str(config["reference_score_column"])], errors="coerce"
    )
    if not np.isfinite(reference).all():
        raise ValueError(f"{source} contains missing/non-finite reference scores")
    checked[str(config["reference_score_column"])] = reference

    target_summary: dict[str, Any] = {}
    if require_target:
        target_column = str(config["target_column"])
        target = pd.to_numeric(checked[target_column], errors="coerce")
        if not np.isfinite(target).all():
            raise ValueError(f"{source} contains missing/non-finite target values")
        checked[target_column] = target
        acceptable = target.ge(float(config["target_threshold"]))
        if acceptable.nunique() != 2:
            raise ValueError(f"{source} target contains only one class")
        target_summary = {
            "positive_count": int(acceptable.sum()),
            "positive_rate": float(acceptable.mean()),
        }
        policy = config["label_mapping_policy"]
        mode_column = str(policy["mode_column"])
        modes = checked[mode_column].astype(str)
        accession_pairs = checked[group_column].astype(str).map(
            accession_pair_from_system
        )
        same_accession = accession_pairs.map(same_known_accession)
        expected_modes = np.where(
            same_accession,
            str(policy["same_accession_mode"]),
            str(policy["different_accession_mode"]),
        )
        policy_failures = modes.ne(expected_modes)
        if policy_failures.any():
            examples = checked.loc[
                policy_failures, [group_column, mode_column]
            ].drop_duplicates().head(5)
            raise ValueError(
                "Target mapping policy mismatch: expected "
                f"{policy['same_accession_mode']} for equal known accession tokens and "
                f"{policy['different_accession_mode']} otherwise; examples: "
                f"{examples.to_dict('records')}"
            )
        target_summary["label_mapping_mode_counts"] = {
            str(key): int(value) for key, value in modes.value_counts().items()
        }
        target_summary["same_known_accession_systems"] = int(
            checked.loc[same_accession, group_column].nunique()
        )
        undefined_accession = accession_pairs.map(
            lambda pair: is_undefined_accession(pair[0])
            or is_undefined_accession(pair[1])
        )
        target_summary["undefined_accession_systems"] = int(
            checked.loc[undefined_accession, group_column].nunique()
        )

    fold_summary: dict[str, Any] = {}
    if require_folds:
        fold_column = str(config["fold_column"])
        folds = pd.to_numeric(checked[fold_column], errors="coerce")
        if not np.isfinite(folds).all() or not np.equal(folds, np.floor(folds)).all():
            raise ValueError(f"{source} contains invalid fold values")
        checked[fold_column] = folds.astype(int)
        folds_per_system = checked.groupby(group_column)[fold_column].nunique()
        if not folds_per_system.eq(1).all():
            raise ValueError(f"{source} assigns a system to more than one fold")
        observed_folds = sorted(checked[fold_column].unique().tolist())
        expected_folds = sorted(int(value) for value in config["expected_fold_values"])
        if observed_folds != expected_folds:
            raise ValueError(
                f"{source} fold values differ: expected {expected_folds}, "
                f"observed {observed_folds}"
            )
        fold_system_counts = (
            checked[[group_column, fold_column]]
            .drop_duplicates()
            .groupby(fold_column)
            .size()
        )
        fold_summary = {
            str(int(key)): int(value) for key, value in fold_system_counts.items()
        }

    audit = {
        "rows": int(len(checked)),
        "systems": int(checked[group_column].nunique()),
        "candidates_per_system": expected_candidates,
        "duplicate_candidate_keys": 0,
        "feature_missing_counts": {
            column: int(feature_values[column].isna().sum())
            for column in feature_columns
        },
        "feature_missing_fractions": {
            column: float(missing_fractions[column]) for column in feature_columns
        },
        "fold_system_counts": fold_summary,
        **target_summary,
    }
    return checked, audit


def fit_preprocessor(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, dict[str, list[float]]]:
    values = numeric_frame(frame, feature_columns).to_numpy(dtype=float)
    medians = np.nanmedian(values, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    imputed = np.where(np.isfinite(values), values, medians)
    means = imputed.mean(axis=0)
    scales = imputed.std(axis=0, ddof=0)
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    transformed = (imputed - means) / scales
    preprocessing = {
        "medians": medians.tolist(),
        "means": means.tolist(),
        "scales": scales.tolist(),
    }
    return transformed, preprocessing


def apply_preprocessor(
    frame: pd.DataFrame,
    feature_columns: list[str],
    preprocessing: dict[str, Any],
) -> np.ndarray:
    values = numeric_frame(frame, feature_columns).to_numpy(dtype=float)
    medians = np.asarray(preprocessing["medians"], dtype=float)
    means = np.asarray(preprocessing["means"], dtype=float)
    scales = np.asarray(preprocessing["scales"], dtype=float)
    expected_shape = (len(feature_columns),)
    if medians.shape != expected_shape or means.shape != expected_shape or scales.shape != expected_shape:
        raise ValueError("Model preprocessing arrays do not match feature_columns")
    imputed = np.where(np.isfinite(values), values, medians)
    return (imputed - means) / scales


def fit_ridge_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    penalty: float,
    max_iter: int = 200,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit L2-regularized logistic regression with Newton/IRLS updates."""

    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Invalid x/y shapes for ridge logistic regression")
    if set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("Ridge logistic regression requires both binary classes")
    design = np.column_stack([np.ones(len(x)), np.asarray(x, dtype=float)])
    coefficients = np.zeros(design.shape[1], dtype=float)
    penalty_matrix = np.eye(design.shape[1], dtype=float) * float(penalty)
    penalty_matrix[0, 0] = 0.0
    converged = False
    last_step = float("nan")
    used_lstsq = False

    for iteration in range(1, max_iter + 1):
        linear = np.clip(design @ coefficients, -30, 30)
        probabilities = 1.0 / (1.0 + np.exp(-linear))
        weights = np.clip(probabilities * (1.0 - probabilities), 1e-7, None)
        gradient = design.T @ (y - probabilities) - penalty_matrix @ coefficients
        information = design.T @ (design * weights[:, None]) + penalty_matrix
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(information, gradient, rcond=None)[0]
            used_lstsq = True
        coefficients_next = coefficients + step
        last_step = float(np.max(np.abs(step)))
        coefficients = coefficients_next
        if last_step < tolerance:
            converged = True
            break

    if not np.isfinite(coefficients).all():
        raise ValueError("Ridge logistic solver produced non-finite coefficients")
    if not converged:
        raise RuntimeError(
            f"Ridge logistic solver did not converge after {max_iter} iterations"
        )
    return coefficients, {
        "converged": converged,
        "iterations": iteration,
        "max_abs_final_step": last_step,
        "used_least_squares_fallback": used_lstsq,
        "max_iter": max_iter,
        "tolerance": tolerance,
    }


def predict_probabilities(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), np.asarray(x, dtype=float)])
    coefficients = np.asarray(coefficients, dtype=float)
    if design.shape[1] != len(coefficients):
        raise ValueError("Model coefficients do not match transformed input")
    linear = np.clip(design @ coefficients, -30, 30)
    return 1.0 / (1.0 + np.exp(-linear))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(scores, dtype=float)
    valid = np.isfinite(score)
    y, score = y[valid], score[valid]
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y, score = y[order], score[order]
    group_ends = np.r_[np.flatnonzero(score[1:] != score[:-1]), len(score) - 1]
    cumulative_true = np.cumsum(y)
    cumulative_false = np.cumsum(1 - y)
    result = 0.0
    previous_recall = 0.0
    for end in group_ends:
        recall = cumulative_true[end] / positives
        precision = cumulative_true[end] / (
            cumulative_true[end] + cumulative_false[end]
        )
        result += (recall - previous_recall) * precision
        previous_recall = recall
    return float(result)


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    score = pd.Series(np.asarray(scores, dtype=float))
    valid = score.notna().to_numpy()
    y, score = y[valid], score[valid]
    positives = int(y.sum())
    negatives = int((1 - y).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = score.rank(method="average").to_numpy()
    rank_sum_positive = float(ranks[y == 1].sum())
    return float(
        (rank_sum_positive - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def candidate_metrics(
    target: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    target = np.asarray(target, dtype=float)
    labels = target >= float(threshold)
    scores = np.asarray(scores, dtype=float)
    return {
        "rows": int(len(target)),
        "positive_count": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "roc_auc": roc_auc(labels.astype(int), scores),
        "average_precision": average_precision(labels.astype(int), scores),
        "brier": float(np.mean((labels.astype(float) - scores) ** 2)),
    }


def select_candidates(
    frame: pd.DataFrame,
    *,
    score_column: str,
    config: dict[str, Any],
    selector: str,
) -> pd.DataFrame:
    group = str(config["group_column"])
    key_columns = list(config["key_columns"])
    tie_columns = [column for column in key_columns if column != group]
    ascending = [True, False] + [True] * len(tie_columns)
    ordered = frame.sort_values(
        [group, score_column] + tie_columns,
        ascending=ascending,
        kind="mergesort",
    )
    selected = ordered.groupby(group, sort=False, as_index=False).head(1).copy()
    selected["selector"] = selector
    return selected.reset_index(drop=True)


def selector_metrics(
    selected: pd.DataFrame,
    *,
    target_column: str,
    threshold: float,
) -> dict[str, Any]:
    target = pd.to_numeric(selected[target_column], errors="raise")
    return {
        "selector": str(selected["selector"].iloc[0]),
        "systems": int(len(selected)),
        "acceptable_rate": float(target.ge(threshold).mean()),
        "medium_high_rate": float(target.ge(0.49).mean()),
        "high_rate": float(target.ge(0.80).mean()),
        "mean_DockQ": float(target.mean()),
        "median_DockQ": float(target.median()),
    }


def paired_selector_bootstrap(
    model_selected: pd.DataFrame,
    reference_selected: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    group = str(config["group_column"])
    target_column = str(config["target_column"])
    threshold = float(config["target_threshold"])
    model = model_selected[[group, target_column]].rename(
        columns={target_column: "model_DockQ"}
    )
    reference = reference_selected[[group, target_column]].rename(
        columns={target_column: "reference_DockQ"}
    )
    aligned = model.merge(reference, on=group, how="outer", validate="one_to_one", indicator=True)
    if not aligned["_merge"].eq("both").all():
        raise ValueError("Model and reference selector system sets differ")
    aligned = aligned.drop(columns="_merge").sort_values(group).reset_index(drop=True)

    model_values = aligned["model_DockQ"].to_numpy(dtype=float)
    reference_values = aligned["reference_DockQ"].to_numpy(dtype=float)
    observed_rate = float(
        (model_values >= threshold).mean() - (reference_values >= threshold).mean()
    )
    observed_mean = float(model_values.mean() - reference_values.mean())
    replicates = int(config["bootstrap_replicates"])
    rng = np.random.default_rng(int(config["random_seed"]))
    rate_deltas = np.empty(replicates, dtype=float)
    mean_deltas = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = rng.integers(0, len(aligned), size=len(aligned))
        sampled_model = model_values[sampled]
        sampled_reference = reference_values[sampled]
        rate_deltas[index] = (
            (sampled_model >= threshold).mean()
            - (sampled_reference >= threshold).mean()
        )
        mean_deltas[index] = sampled_model.mean() - sampled_reference.mean()
    return {
        "systems": int(len(aligned)),
        "bootstrap_replicates": replicates,
        "random_seed": int(config["random_seed"]),
        "delta_acceptable_rate": observed_rate,
        "delta_acceptable_rate_ci_low": float(np.quantile(rate_deltas, 0.025)),
        "delta_acceptable_rate_ci_high": float(np.quantile(rate_deltas, 0.975)),
        "delta_mean_DockQ": observed_mean,
        "delta_mean_DockQ_ci_low": float(np.quantile(mean_deltas, 0.025)),
        "delta_mean_DockQ_ci_high": float(np.quantile(mean_deltas, 0.975)),
    }


def load_model(path: Path) -> dict[str, Any]:
    model = load_json(path)
    if model.get("model_schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported model schema version: {model.get('model_schema_version')}"
        )
    required = {
        "model_name",
        "config",
        "feature_columns",
        "preprocessing",
        "intercept",
        "coefficients",
        "training_system_ids",
    }
    missing = sorted(required.difference(model))
    if missing:
        raise ValueError(f"Model artifact is missing fields: {missing}")
    validate_config(model["config"])
    if list(model["feature_columns"]) != list(model["config"]["feature_columns"]):
        raise ValueError("Model feature_columns differ from embedded config")
    return model


def prediction_coefficients(model: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [float(model["intercept"])]
        + [float(value) for value in model["coefficients"]],
        dtype=float,
    )


def pdb_id_from_system(system_id: str) -> str:
    return str(system_id).split("__", 1)[0].lower()


def accession_pair_from_system(system_id: str) -> tuple[str, str]:
    parts = str(system_id).split("--", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse accession pair from system ID: {system_id}")
    return parts[0].rsplit("_", 1)[-1], parts[1].rsplit("_", 1)[-1]


def is_undefined_accession(value: object) -> bool:
    return str(value).strip().upper() in UNDEFINED_ACCESSIONS


def same_known_accession(pair: tuple[str, str]) -> bool:
    left, right = pair
    if is_undefined_accession(left) or is_undefined_accession(right):
        return False
    return left.strip().upper() == right.strip().upper()
