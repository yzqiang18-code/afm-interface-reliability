#!/usr/bin/env python3
"""Apply a frozen candidate baseline without reading or emitting target labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from baseline_core import (
    MODEL_SELECTED_COLUMN,
    PREDICTION_SCORE_COLUMN,
    REFERENCE_SELECTED_COLUMN,
    apply_preprocessor_dispatch,
    atomic_write_csv,
    atomic_write_json,
    check_output_targets,
    load_model,
    portable_path,
    predict_probabilities,
    prediction_coefficients,
    select_candidates,
    sha256_file,
    utc_now,
    validate_candidate_frame,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        input_csv = Path(args.input_csv).expanduser().resolve()
        model_path = Path(args.model).expanduser().resolve()
        output_csv = Path(args.output_csv).expanduser().resolve()
        summary_json = (
            Path(args.summary_json).expanduser().resolve()
            if args.summary_json
            else output_csv.with_suffix(".summary.json")
        )
        if not input_csv.is_file():
            raise FileNotFoundError(input_csv)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        check_output_targets(
            [output_csv, summary_json], overwrite=bool(args.overwrite)
        )

        model = load_model(model_path)
        config = model["config"]
        raw = pd.read_csv(input_csv, low_memory=False)
        frame, input_audit = validate_candidate_frame(
            raw,
            config,
            require_target=False,
            require_folds=False,
            source="prediction input CSV",
        )
        transformed = apply_preprocessor_dispatch(
            frame,
            list(model["feature_columns"]),
            model["preprocessing"],
            group_column=str(config["group_column"]),
        )
        probabilities = predict_probabilities(
            transformed, prediction_coefficients(model)
        )

        key_columns = list(config["key_columns"])
        reference_score = str(config["reference_score_column"])
        predictions = frame[key_columns + [reference_score]].copy()
        predictions[PREDICTION_SCORE_COLUMN] = probabilities
        predictions[MODEL_SELECTED_COLUMN] = False
        predictions[REFERENCE_SELECTED_COLUMN] = False

        model_selected = select_candidates(
            predictions,
            score_column=PREDICTION_SCORE_COLUMN,
            config=config,
            selector=str(model["model_name"]),
        )
        reference_selected = select_candidates(
            predictions,
            score_column=reference_score,
            config=config,
            selector="AF-M full-precision rank-1",
        )
        model_keys = pd.MultiIndex.from_frame(model_selected[key_columns])
        reference_keys = pd.MultiIndex.from_frame(reference_selected[key_columns])
        all_keys = pd.MultiIndex.from_frame(predictions[key_columns])
        predictions[MODEL_SELECTED_COLUMN] = all_keys.isin(model_keys)
        predictions[REFERENCE_SELECTED_COLUMN] = all_keys.isin(reference_keys)
        predictions = predictions.sort_values(key_columns, kind="mergesort").reset_index(
            drop=True
        )

        group_column = str(config["group_column"])
        model_selected_count = predictions.groupby(group_column)[
            MODEL_SELECTED_COLUMN
        ].sum()
        reference_selected_count = predictions.groupby(group_column)[
            REFERENCE_SELECTED_COLUMN
        ].sum()
        if not model_selected_count.eq(1).all():
            raise ValueError("Prediction output does not select exactly one model per system")
        if not reference_selected_count.eq(1).all():
            raise ValueError("Reference output does not select exactly one model per system")

        target_column = str(config["target_column"])
        summary = {
            "created_at_utc": utc_now(),
            "model_name": str(model["model_name"]),
            "model_path": portable_path(model_path),
            "model_sha256": sha256_file(model_path),
            "input_path": portable_path(input_csv),
            "input_sha256": sha256_file(input_csv),
            "input_audit": input_audit,
            "target_column_present_but_ignored": target_column in raw.columns,
            "output_path": portable_path(output_csv),
            "output_columns": list(predictions.columns),
            "selected_systems": int(model_selected_count.sum()),
        }
        if target_column in predictions.columns:
            raise ValueError("Prediction output unexpectedly contains the target column")
        atomic_write_csv(output_csv, predictions)
        summary["output_sha256"] = sha256_file(output_csv)
        atomic_write_json(summary_json, summary)
        print(
            f"Prediction complete: systems={input_audit['systems']} "
            f"rows={input_audit['rows']} output={output_csv}"
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
