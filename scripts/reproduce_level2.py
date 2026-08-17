#!/usr/bin/env python3
"""Reproduce the public Candidate Ridge analysis and data figures on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def selector_metrics(frame: pd.DataFrame) -> dict[str, object]:
    dockq = pd.to_numeric(frame["DockQ"], errors="raise")
    return {
        "systems": int(len(frame)),
        "acceptable_rate": float(dockq.ge(0.23).mean()),
        "mean_DockQ": float(dockq.mean()),
        "median_DockQ": float(dockq.median()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "model"
    prediction_dir = output / "prediction"
    evaluation_dir = output / "evaluation"
    figure_dir = output / "figures"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    training = ROOT / "results/data/training500_candidates.csv.gz"
    holdout = ROOT / "results/data/pinder_af2_180_labels.csv.gz"
    config = ROOT / "configs/ml/candidate_ridge_v1.json"
    predictions = prediction_dir / "pinder_af2_180_predictions.csv"
    model = model_dir / "model.json"
    overwrite = ["--overwrite"] if args.overwrite else []

    run(
        [
            args.python,
            "scripts/ml/train_baseline.py",
            "--train-csv",
            str(training),
            "--config",
            str(config),
            "--output-dir",
            str(model_dir),
            *overwrite,
        ]
    )
    run(
        [
            args.python,
            "scripts/ml/predict.py",
            "--input-csv",
            str(holdout),
            "--model",
            str(model),
            "--output-csv",
            str(predictions),
            *overwrite,
        ]
    )
    run(
        [
            args.python,
            "scripts/ml/evaluate_holdout.py",
            "--predictions-csv",
            str(predictions),
            "--labels-csv",
            str(holdout),
            "--model",
            str(model),
            "--output-dir",
            str(evaluation_dir),
            *overwrite,
        ]
    )
    run(
        [
            args.python,
            "scripts/figures/make_figures.py",
            "--training-csv",
            str(training),
            "--selector-choices",
            str(evaluation_dir / "selector_choices.csv"),
            "--metric-table",
            "results/tables/training500_metric_performance.csv",
            "--within-system-table",
            "results/tables/training500_within_system_ranking.csv",
            "--ablation-table",
            "results/tables/training500_system_risk_models.csv",
            "--output-dir",
            str(figure_dir),
        ]
    )

    training_frame = pd.read_csv(training, low_memory=False)
    top1 = (
        training_frame.sort_values(
            ["complex_id", "ranking_confidence", "rank"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("complex_id", as_index=False)
        .first()
    )
    oracle = (
        training_frame.sort_values(
            ["complex_id", "DockQ", "rank"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("complex_id", as_index=False)
        .first()
    )
    choices = pd.read_csv(evaluation_dir / "selector_choices.csv")
    holdout_metrics = {
        selector: selector_metrics(group)
        for selector, group in choices.groupby("selector", sort=True)
    }

    intersections = pd.read_csv(ROOT / "results/audits/leakage/leakage_intersections.csv")
    overlapping_holdout = set()
    if not intersections.empty:
        for value in intersections.loc[
            intersections["level"].eq("uniprot_pair"), "holdout_system_ids"
        ].dropna():
            overlapping_holdout.update(str(value).split(";"))
    sensitivity_choices = choices.loc[~choices["complex_id"].isin(overlapping_holdout)]
    sensitivity = {
        "excluded_holdout_system_ids": sorted(overlapping_holdout),
        "selector_metrics": {
            selector: selector_metrics(group)
            for selector, group in sensitivity_choices.groupby("selector", sort=True)
        },
    }
    reference = sensitivity["selector_metrics"]["AF-M full-precision rank-1"]
    model_metrics = sensitivity["selector_metrics"]["candidate_ridge_v1"]
    sensitivity["delta_acceptable_rate"] = (
        model_metrics["acceptable_rate"] - reference["acceptable_rate"]
    )
    sensitivity["delta_mean_DockQ"] = model_metrics["mean_DockQ"] - reference["mean_DockQ"]

    model_payload = json.loads(model.read_text(encoding="utf-8"))
    model_payload.pop("created_at_utc", None)
    canonical_model = json.dumps(
        model_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    summary = {
        "schema_version": 2,
        "hash_note": (
            "model_sha256 hashes model.json without its volatile created_at_utc "
            "field; byte-identical output additionally requires the pinned "
            "environment (environment.yml)"
        ),
        "inputs": {
            str(training.relative_to(ROOT)): sha256(training),
            str(holdout.relative_to(ROOT)): sha256(holdout),
            str(config.relative_to(ROOT)): sha256(config),
        },
        "training500": {
            "systems": int(top1["complex_id"].nunique()),
            "candidates": int(len(training_frame)),
            "afm_top1": selector_metrics(top1),
            "oracle_best_of_20": selector_metrics(oracle),
        },
        "pinder_af2": holdout_metrics,
        "uniprot_overlap_sensitivity": sensitivity,
        "model_sha256": hashlib.sha256(canonical_model).hexdigest(),
        "predictions_sha256": sha256(predictions),
    }
    (output / "reproduction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
