#!/usr/bin/env python3
"""Train and freeze a candidate-level ridge-logistic reranking baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baseline_core import (
    MODEL_SCHEMA_VERSION,
    PREDICTION_SCORE_COLUMN,
    atomic_write_csv,
    atomic_write_json,
    candidate_metrics,
    check_output_targets,
    fit_preprocessor,
    fit_ridge_logistic,
    load_json,
    paired_selector_bootstrap,
    portable_path,
    predict_probabilities,
    select_candidates,
    selector_metrics,
    sha256_file,
    utc_now,
    validate_candidate_frame,
    validate_config,
)


OUTPUT_NAMES = {
    "model": "model.json",
    "summary": "training_summary.json",
    "oof_predictions": "oof_predictions.csv",
    "candidate_metrics": "oof_candidate_metrics.csv",
    "selector_choices": "oof_selector_choices.csv",
    "selector_summary": "oof_selector_summary.csv",
    "bootstrap": "oof_paired_bootstrap.csv",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def fit_model(
    frame: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    feature_columns = list(config["feature_columns"])
    target_column = str(config["target_column"])
    threshold = float(config["target_threshold"])
    target = frame[target_column].ge(threshold).astype(float).to_numpy()
    transformed, preprocessing = fit_preprocessor(frame, feature_columns)
    coefficients, solver = fit_ridge_logistic(
        transformed,
        target,
        penalty=float(config["ridge_penalty"]),
    )
    return {
        "preprocessing": preprocessing,
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:].tolist(),
    }, solver


def oof_predictions(
    frame: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    predictions = np.full(len(frame), np.nan, dtype=float)
    fold_column = str(config["fold_column"])
    target_column = str(config["target_column"])
    threshold = float(config["target_threshold"])
    target = frame[target_column].ge(threshold).astype(float).to_numpy()
    solver_audits: list[dict[str, Any]] = []

    for fold in sorted(int(value) for value in config["expected_fold_values"]):
        train_mask = frame[fold_column].ne(fold).to_numpy()
        validation_mask = ~train_mask
        if not train_mask.any() or not validation_mask.any():
            raise ValueError(f"Fold {fold} has an empty train or validation partition")
        if len(np.unique(target[train_mask])) != 2:
            raise ValueError(f"Fold {fold} training partition contains one target class")
        training = frame.loc[train_mask]
        validation = frame.loc[validation_mask]
        transformed_train, preprocessing = fit_preprocessor(
            training, list(config["feature_columns"])
        )
        transformed_validation = (
            validation[list(config["feature_columns"])]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy(dtype=float)
        )
        medians = np.asarray(preprocessing["medians"], dtype=float)
        means = np.asarray(preprocessing["means"], dtype=float)
        scales = np.asarray(preprocessing["scales"], dtype=float)
        transformed_validation = (
            np.where(np.isfinite(transformed_validation), transformed_validation, medians)
            - means
        ) / scales
        coefficients, solver = fit_ridge_logistic(
            transformed_train,
            target[train_mask],
            penalty=float(config["ridge_penalty"]),
        )
        predictions[validation_mask] = predict_probabilities(
            transformed_validation, coefficients
        )
        solver_audits.append(
            {
                "fold": fold,
                "train_rows": int(train_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "train_systems": int(
                    frame.loc[train_mask, str(config["group_column"])].nunique()
                ),
                "validation_systems": int(
                    frame.loc[validation_mask, str(config["group_column"])].nunique()
                ),
                **solver,
            }
        )
    if not np.isfinite(predictions).all():
        raise ValueError("OOF predictions are incomplete")

    columns = list(config["key_columns"]) + [
        fold_column,
        target_column,
        str(config["reference_score_column"]),
    ]
    output = frame[columns].copy()
    output["acceptable"] = target.astype(bool)
    output[PREDICTION_SCORE_COLUMN] = predictions
    return output, solver_audits


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        train_csv = Path(args.train_csv).expanduser().resolve()
        config_path = Path(args.config).expanduser().resolve()
        output_dir = Path(args.output_dir).expanduser().resolve()
        if not train_csv.is_file():
            raise FileNotFoundError(train_csv)
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config = validate_config(load_json(config_path))
        output_paths = {
            key: output_dir / name for key, name in OUTPUT_NAMES.items()
        }
        check_output_targets(output_paths.values(), overwrite=bool(args.overwrite))

        raw = pd.read_csv(train_csv, low_memory=False)
        frame, input_audit = validate_candidate_frame(
            raw,
            config,
            require_target=True,
            require_folds=True,
            source="training CSV",
        )
        oof, fold_solver_audits = oof_predictions(frame, config=config)
        target_column = str(config["target_column"])
        reference_score = str(config["reference_score_column"])
        threshold = float(config["target_threshold"])

        oof_candidate = candidate_metrics(
            oof[target_column].to_numpy(dtype=float),
            oof[PREDICTION_SCORE_COLUMN].to_numpy(dtype=float),
            threshold=threshold,
        )
        model_selected = select_candidates(
            oof,
            score_column=PREDICTION_SCORE_COLUMN,
            config=config,
            selector=str(config["model_name"]),
        )
        reference_selected = select_candidates(
            oof,
            score_column=reference_score,
            config=config,
            selector="AF-M full-precision rank-1",
        )
        model_selector_metrics = selector_metrics(
            model_selected,
            target_column=target_column,
            threshold=threshold,
        )
        reference_selector_metrics = selector_metrics(
            reference_selected,
            target_column=target_column,
            threshold=threshold,
        )
        bootstrap = paired_selector_bootstrap(
            model_selected, reference_selected, config=config
        )

        fitted, final_solver = fit_model(frame, config=config)
        training_system_ids = sorted(
            frame[str(config["group_column"])].astype(str).unique().tolist()
        )
        model_artifact = {
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "model_name": str(config["model_name"]),
            "task": str(config["task"]),
            "created_at_utc": utc_now(),
            "config": config,
            "feature_columns": list(config["feature_columns"]),
            "target_definition": {
                "column": target_column,
                "positive_when": f"{target_column} >= {threshold}",
                "threshold": threshold,
            },
            "training_input": {
                "path": portable_path(train_csv),
                "sha256": sha256_file(train_csv),
                **input_audit,
            },
            "config_input": {
                "path": portable_path(config_path),
                "sha256": sha256_file(config_path),
            },
            "training_system_ids": training_system_ids,
            "preprocessing": fitted["preprocessing"],
            "intercept": fitted["intercept"],
            "coefficients": fitted["coefficients"],
            "solver": final_solver,
            "oof_candidate_metrics": oof_candidate,
            "oof_selector_metrics": {
                "model": model_selector_metrics,
                "reference": reference_selector_metrics,
                "paired_bootstrap": bootstrap,
            },
        }
        summary = {
            "created_at_utc": utc_now(),
            "model_name": str(config["model_name"]),
            "training_input_audit": input_audit,
            "fold_solver_audits": fold_solver_audits,
            "final_solver": final_solver,
            "oof_candidate_metrics": oof_candidate,
            "oof_selector_metrics": {
                "model": model_selector_metrics,
                "reference": reference_selector_metrics,
            },
            "oof_paired_bootstrap": bootstrap,
            "output_files": {key: portable_path(path) for key, path in output_paths.items()},
        }

        selector_columns = list(config["key_columns"]) + [
            target_column,
            "acceptable",
            PREDICTION_SCORE_COLUMN,
            reference_score,
            "selector",
        ]
        selector_choices = pd.concat(
            [
                model_selected[selector_columns],
                reference_selected[selector_columns],
            ],
            ignore_index=True,
        )
        atomic_write_json(output_paths["model"], model_artifact)
        atomic_write_json(output_paths["summary"], summary)
        atomic_write_csv(output_paths["oof_predictions"], oof)
        atomic_write_csv(
            output_paths["candidate_metrics"], pd.DataFrame([oof_candidate])
        )
        atomic_write_csv(output_paths["selector_choices"], selector_choices)
        atomic_write_csv(
            output_paths["selector_summary"],
            pd.DataFrame([reference_selector_metrics, model_selector_metrics]),
        )
        atomic_write_csv(output_paths["bootstrap"], pd.DataFrame([bootstrap]))
        print(
            f"Training complete: model={config['model_name']} "
            f"systems={input_audit['systems']} rows={input_audit['rows']} "
            f"output={output_dir}"
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
