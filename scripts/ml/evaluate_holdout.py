#!/usr/bin/env python3
"""Evaluate frozen predictions against holdout DockQ labels exactly once."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from baseline_core import (
    MODEL_SELECTED_COLUMN,
    PREDICTION_SCORE_COLUMN,
    REFERENCE_SELECTED_COLUMN,
    atomic_write_csv,
    atomic_write_json,
    candidate_metrics,
    check_output_targets,
    load_model,
    paired_selector_bootstrap,
    pdb_id_from_system,
    portable_path,
    require_columns,
    selector_metrics,
    sha256_file,
    utc_now,
    validate_candidate_frame,
)


OUTPUT_NAMES = {
    "summary": "evaluation_summary.json",
    "candidate_metrics": "candidate_metrics.csv",
    "selector_summary": "selector_summary.csv",
    "selector_choices": "selector_choices.csv",
    "bootstrap": "paired_bootstrap.csv",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        predictions_csv = Path(args.predictions_csv).expanduser().resolve()
        labels_csv = Path(args.labels_csv).expanduser().resolve()
        model_path = Path(args.model).expanduser().resolve()
        output_dir = Path(args.output_dir).expanduser().resolve()
        for path in [predictions_csv, labels_csv, model_path]:
            if not path.is_file():
                raise FileNotFoundError(path)
        output_paths = {
            key: output_dir / name for key, name in OUTPUT_NAMES.items()
        }
        check_output_targets(output_paths.values(), overwrite=bool(args.overwrite))

        model = load_model(model_path)
        config = model["config"]
        labels_raw = pd.read_csv(labels_csv, low_memory=False)
        labels, labels_audit = validate_candidate_frame(
            labels_raw,
            config,
            require_target=True,
            require_folds=False,
            source="holdout labels CSV",
        )
        predictions = pd.read_csv(predictions_csv, low_memory=False)
        key_columns = list(config["key_columns"])
        require_columns(
            predictions,
            key_columns
            + [
                PREDICTION_SCORE_COLUMN,
                MODEL_SELECTED_COLUMN,
                REFERENCE_SELECTED_COLUMN,
            ],
            source="predictions CSV",
        )
        if predictions.duplicated(key_columns).any():
            raise ValueError("predictions CSV contains duplicate candidate keys")
        target_column = str(config["target_column"])
        if target_column in predictions.columns:
            raise ValueError(
                "predictions CSV contains the target column; use the label-free output "
                "from predict.py"
            )
        for column in [MODEL_SELECTED_COLUMN, REFERENCE_SELECTED_COLUMN]:
            if predictions[column].dtype != bool:
                normalized = predictions[column].astype(str).str.lower()
                if not normalized.isin(["true", "false"]).all():
                    raise ValueError(f"predictions CSV contains invalid {column} values")
                predictions[column] = normalized.eq("true")

        group_column = str(config["group_column"])
        for column in [MODEL_SELECTED_COLUMN, REFERENCE_SELECTED_COLUMN]:
            selected_counts = predictions.groupby(group_column)[column].sum()
            if not selected_counts.eq(1).all():
                raise ValueError(
                    f"predictions CSV must have exactly one {column}=True per system"
                )

        label_columns = key_columns + [target_column]
        evaluation = predictions.merge(
            labels[label_columns],
            on=key_columns,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if not evaluation["_merge"].eq("both").all():
            examples = evaluation.loc[
                evaluation["_merge"].ne("both"), key_columns + ["_merge"]
            ].head(5)
            raise ValueError(
                "Prediction and label candidate keys differ: "
                f"{examples.to_dict('records')}"
            )
        evaluation = evaluation.drop(columns="_merge")

        test_systems = set(evaluation[group_column].astype(str).unique())
        training_systems = set(str(value) for value in model["training_system_ids"])
        exact_overlap = sorted(test_systems.intersection(training_systems))
        training_pdb_ids = {pdb_id_from_system(value) for value in training_systems}
        test_pdb_ids = {pdb_id_from_system(value) for value in test_systems}
        pdb_overlap = sorted(test_pdb_ids.intersection(training_pdb_ids))
        if exact_overlap or pdb_overlap:
            raise ValueError(
                "Holdout overlaps training data: "
                f"exact_systems={exact_overlap[:5]} pdb_ids={pdb_overlap[:5]}"
            )

        threshold = float(config["target_threshold"])
        candidate = candidate_metrics(
            evaluation[target_column].to_numpy(dtype=float),
            evaluation[PREDICTION_SCORE_COLUMN].to_numpy(dtype=float),
            threshold=threshold,
        )
        model_selected = evaluation.loc[evaluation[MODEL_SELECTED_COLUMN]].copy()
        model_selected["selector"] = str(model["model_name"])
        reference_selected = evaluation.loc[
            evaluation[REFERENCE_SELECTED_COLUMN]
        ].copy()
        reference_selected["selector"] = "AF-M full-precision rank-1"
        model_selector = selector_metrics(
            model_selected,
            target_column=target_column,
            threshold=threshold,
        )
        reference_selector = selector_metrics(
            reference_selected,
            target_column=target_column,
            threshold=threshold,
        )
        bootstrap = paired_selector_bootstrap(
            model_selected, reference_selected, config=config
        )

        selector_columns = key_columns + [
            target_column,
            PREDICTION_SCORE_COLUMN,
            "selector",
        ]
        selector_choices = pd.concat(
            [
                reference_selected[selector_columns],
                model_selected[selector_columns],
            ],
            ignore_index=True,
        )
        summary = {
            "created_at_utc": utc_now(),
            "model_name": str(model["model_name"]),
            "model_path": portable_path(model_path),
            "model_sha256": sha256_file(model_path),
            "predictions_path": portable_path(predictions_csv),
            "predictions_sha256": sha256_file(predictions_csv),
            "labels_path": portable_path(labels_csv),
            "labels_sha256": sha256_file(labels_csv),
            "labels_audit": labels_audit,
            "leakage_gate": {
                "exact_system_overlap_count": len(exact_overlap),
                "pdb_id_overlap_count": len(pdb_overlap),
                "passed": True,
                "note": (
                    "This script checks exact system and PDB-ID overlap. The separate "
                    "The separate PINDER manifest audit records cluster and "
                    "UniProt-pair overlap under results/audits/leakage/."
                ),
            },
            "candidate_metrics": candidate,
            "selector_metrics": {
                "reference": reference_selector,
                "model": model_selector,
            },
            "paired_bootstrap": bootstrap,
            "output_files": {key: portable_path(path) for key, path in output_paths.items()},
        }
        atomic_write_json(output_paths["summary"], summary)
        atomic_write_csv(
            output_paths["candidate_metrics"], pd.DataFrame([candidate])
        )
        atomic_write_csv(
            output_paths["selector_summary"],
            pd.DataFrame([reference_selector, model_selector]),
        )
        atomic_write_csv(output_paths["selector_choices"], selector_choices)
        atomic_write_csv(output_paths["bootstrap"], pd.DataFrame([bootstrap]))
        print(
            f"Holdout evaluation complete: systems={labels_audit['systems']} "
            f"model_acceptable_rate={model_selector['acceptable_rate']:.4f} "
            f"reference_acceptable_rate={reference_selector['acceptable_rate']:.4f} "
            f"output={output_dir}"
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
