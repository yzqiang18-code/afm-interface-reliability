from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


try:
    import numpy as np
    import pandas as pd

    HAS_ML_DEPS = True
except ModuleNotFoundError:
    HAS_ML_DEPS = False


REPO_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = REPO_ROOT / "scripts" / "ml"

if HAS_ML_DEPS:
    sys.path.insert(0, str(ML_DIR))
    import baseline_core as core
    import build_candidate_table as builder
    import evaluate_holdout as evaluate
    import predict
    import train_baseline as train


@unittest.skipUnless(HAS_ML_DEPS, "NumPy and pandas are required")
class MLBaselineTests(unittest.TestCase):
    def config(self, path: Path) -> dict[str, object]:
        payload: dict[str, object] = {
            "bootstrap_replicates": 50,
            "expected_candidates_per_system": 3,
            "expected_fold_values": [0, 1, 2, 3, 4],
            "feature_columns": ["feature_a", "feature_b"],
            "fold_column": "cv_fold",
            "group_column": "complex_id",
            "key_columns": ["complex_id", "rank", "model_weight", "seed"],
            "label_mapping_policy": {
                "different_accession_mode": "fixed",
                "mode_column": "mapping_mode",
                "same_accession_mode": "symmetry_aware",
            },
            "max_missing_fraction_per_feature": 0.1,
            "model_name": "synthetic_ridge",
            "random_seed": 7,
            "reference_score_column": "ranking_confidence",
            "ridge_penalty": 1.0,
            "target_column": "DockQ",
            "target_threshold": 0.23,
            "task": "synthetic candidate reranking",
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def candidate_frame(
        self,
        *,
        prefix: str,
        systems: int,
        include_folds: bool,
    ) -> "pd.DataFrame":
        rows: list[dict[str, object]] = []
        for system in range(systems):
            complex_id = (
                f"{prefix}{system}__A1_P{system:05d}--"
                f"{prefix}{system}__B1_Q{system:05d}"
            )
            for candidate, dockq in enumerate([0.10, 0.35, 0.70], start=1):
                row: dict[str, object] = {
                    "complex_id": complex_id,
                    "rank": candidate,
                    "model_weight": candidate,
                    "seed": 0,
                    "feature_a": dockq + system * 0.001,
                    "feature_b": 1.0 - dockq,
                    "ranking_confidence": 100.0 - candidate,
                    "DockQ": dockq,
                    "mapping_mode": "fixed",
                }
                if include_folds:
                    row["cv_fold"] = system % 5
                rows.append(row)
        return pd.DataFrame(rows)

    def test_end_to_end_training_prediction_and_holdout_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            self.config(config_path)
            train_csv = root / "train.csv"
            test_csv = root / "test.csv"
            self.candidate_frame(
                prefix="train", systems=10, include_folds=True
            ).to_csv(train_csv, index=False)
            self.candidate_frame(
                prefix="test", systems=4, include_folds=False
            ).to_csv(test_csv, index=False)

            model_dir = root / "model"
            self.assertEqual(
                train.main(
                    [
                        "--train-csv",
                        str(train_csv),
                        "--config",
                        str(config_path),
                        "--output-dir",
                        str(model_dir),
                    ]
                ),
                0,
            )
            prediction_csv = root / "predictions.csv"
            self.assertEqual(
                predict.main(
                    [
                        "--input-csv",
                        str(test_csv),
                        "--model",
                        str(model_dir / "model.json"),
                        "--output-csv",
                        str(prediction_csv),
                    ]
                ),
                0,
            )
            predictions = pd.read_csv(prediction_csv)
            self.assertNotIn("DockQ", predictions.columns)
            self.assertIn("acceptable_score", predictions.columns)
            self.assertNotIn("acceptable_probability", predictions.columns)
            self.assertTrue(
                predictions.groupby("complex_id")["model_selected"].sum().eq(1).all()
            )

            evaluation_dir = root / "evaluation"
            self.assertEqual(
                evaluate.main(
                    [
                        "--predictions-csv",
                        str(prediction_csv),
                        "--labels-csv",
                        str(test_csv),
                        "--model",
                        str(model_dir / "model.json"),
                        "--output-dir",
                        str(evaluation_dir),
                    ]
                ),
                0,
            )
            summary = json.loads(
                (evaluation_dir / "evaluation_summary.json").read_text()
            )
            self.assertTrue(summary["leakage_gate"]["passed"])
            self.assertEqual(summary["labels_audit"]["systems"], 4)

    def test_label_mapping_policy_rejects_fixed_same_accession(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "config.json"
            config = self.config(config_path)
            frame = self.candidate_frame(
                prefix="same", systems=1, include_folds=False
            )
            frame["complex_id"] = "same__A1_P12345--same__B1_P12345"
            with self.assertRaisesRegex(ValueError, "mapping policy mismatch"):
                core.validate_candidate_frame(
                    frame,
                    config,
                    require_target=True,
                    require_folds=False,
                    source="synthetic labels",
                )

    def test_undefined_accessions_require_fixed_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "config.json"
            config = self.config(config_path)
            frame = self.candidate_frame(
                prefix="undefined", systems=1, include_folds=False
            )
            frame["complex_id"] = (
                "undefined__A1_UNDEFINED--undefined__B1_UNDEFINED"
            )
            checked, audit = core.validate_candidate_frame(
                frame,
                config,
                require_target=True,
                require_folds=False,
                source="synthetic labels",
            )
            self.assertTrue(checked["mapping_mode"].eq("fixed").all())
            self.assertEqual(audit["same_known_accession_systems"], 0)
            self.assertEqual(audit["undefined_accession_systems"], 1)

    def test_duplicate_candidate_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "config.json"
            config = self.config(config_path)
            frame = self.candidate_frame(
                prefix="duplicate", systems=1, include_folds=False
            )
            frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
            with self.assertRaisesRegex(ValueError, "duplicate candidate keys"):
                core.validate_candidate_frame(
                    frame,
                    config,
                    require_target=True,
                    require_folds=False,
                    source="synthetic labels",
                )

    def test_ilis_seed_is_derived_from_structure_filename(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "name": "example__A1_P1--example__B1_Q1",
                    "rank": 1,
                    "model": 2,
                    "iLIS": 0.5,
                    "iLIA": 1.0,
                    "iLISA": 1.0,
                    "ipSAE": 0.6,
                    "actifpTM": 0.7,
                    "LIS": 0.4,
                    "cLIS": 0.3,
                    "ipTM": 0.8,
                    "pTM": 0.7,
                    "pLDDT": 80.0,
                    "LIpLDDT": 82.0,
                    "cLIpLDDT": 84.0,
                    "structure_file": "example_model_2_seed_003.pdb",
                }
            ]
        )
        normalized = builder.normalize_ilis(frame)
        self.assertEqual(int(normalized.loc[0, "seed"]), 3)
        self.assertEqual(int(normalized.loc[0, "model_weight"]), 2)


if __name__ == "__main__":
    unittest.main()
