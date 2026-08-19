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
    def config(self, path: Path, *, mode: str = "zscore") -> dict[str, object]:
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
        if mode == "within_system_rank":
            payload["model_name"] = "synthetic_ridge_rank"
            payload["preprocessing"] = {
                "mode": "within_system_rank",
                "tie_method": "average",
                "centering": "minus_0.5",
                "missing_policy": "system_median_then_fold_median",
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


    def test_within_rank_monotone_exact(self) -> None:
        frame = pd.DataFrame(
            {
                "complex_id": ["s1"] * 4,
                "rank": [1, 2, 3, 4],
                "model_weight": [1, 2, 3, 4],
                "seed": [0] * 4,
                "f": [1.0, 2.0, 3.0, 4.0],
            }
        )
        out = core.within_system_rank_transform(frame, ["f"])
        expected = np.array([1.0, 2.0, 3.0, 4.0]) / 4.0 - 0.5
        np.testing.assert_allclose(out[:, 0], expected)
        self.assertTrue(np.all(np.diff(out[:, 0]) > 0))

    def test_within_rank_constant_is_neutral(self) -> None:
        frame = pd.DataFrame(
            {
                "complex_id": ["s1"] * 4,
                "rank": [1, 2, 3, 4],
                "model_weight": [1, 2, 3, 4],
                "seed": [0] * 4,
                "f": [2.0] * 4,
            }
        )
        out = core.within_system_rank_transform(frame, ["f"])
        np.testing.assert_allclose(out[:, 0], [1.0 / 8.0] * 4)

    def test_within_rank_ties_average(self) -> None:
        frame = pd.DataFrame(
            {
                "complex_id": ["s1"] * 4,
                "rank": [1, 2, 3, 4],
                "model_weight": [1, 2, 3, 4],
                "seed": [0] * 4,
                "f": [1.0, 2.0, 2.0, 4.0],
            }
        )
        out = core.within_system_rank_transform(frame, ["f"])
        ranks = np.array([1.0, 2.5, 2.5, 4.0])
        np.testing.assert_allclose(out[:, 0], ranks / 4.0 - 0.5)

    def test_within_rank_shuffle_invariant(self) -> None:
        rows: list[dict[str, object]] = []
        for system in range(2):
            for candidate, value in enumerate([0.1, 0.4, 0.7], start=1):
                rows.append(
                    {
                        "complex_id": f"s{system}__A_P{system}--s{system}__B_Q{system}",
                        "rank": candidate,
                        "model_weight": candidate,
                        "seed": 0,
                        "f": value + system * 0.01,
                    }
                )
        frame = pd.DataFrame(rows)
        out = core.within_system_rank_transform(frame, ["f"])
        shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
        out_shuffled = core.within_system_rank_transform(shuffled, ["f"])
        keys = ["complex_id", "rank", "model_weight", "seed"]
        original = frame[keys].copy()
        original["orig"] = out[:, 0]
        permuted = shuffled[keys].copy()
        permuted["shuf"] = out_shuffled[:, 0]
        compared = original.merge(permuted, on=keys, validate="one_to_one")
        np.testing.assert_allclose(compared["orig"], compared["shuf"])

    def test_within_rank_missing_two_level(self) -> None:
        # 体系 A: 单个缺失 -> 体系内中位数命中; 体系 B: 全缺失 -> fold 中位数命中
        frame = pd.DataFrame(
            {
                "complex_id": ["A__P--A__Q"] * 3 + ["B__P--B__Q"] * 3,
                "rank": [1, 2, 3, 1, 2, 3],
                "model_weight": [1, 2, 3, 1, 2, 3],
                "seed": [0] * 6,
                "f": [1.0, 3.0, np.nan, np.nan, np.nan, np.nan],
            }
        )
        out = core.within_system_rank_transform(
            frame, ["f"], medians=np.array([2.0])
        )
        # 体系 A: 填补 2.0 -> [1, 3, 2] -> 秩 [1, 3, 2]
        expected_a = np.array([1.0, 3.0, 2.0]) / 3.0 - 0.5
        np.testing.assert_allclose(out[:3, 0], expected_a)
        # 体系 B: 全部用 fold 中位数 2.0 -> 三值同秩 -> 1/(2*3) = 1/6
        np.testing.assert_allclose(out[3:, 0], [1.0 / 6.0] * 3)

    def test_within_rank_end_to_end_schema_v3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            self.config(config_path, mode="within_system_rank")
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
            model = core.load_model(model_dir / "model.json")
            self.assertEqual(model["model_schema_version"], 3)
            self.assertEqual(
                model["preprocessing"]["mode"], "within_system_rank"
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
            self.assertTrue(
                predictions.groupby("complex_id")["model_selected"]
                .sum()
                .eq(1)
                .all()
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

    def test_unsupported_preprocessing_mode_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "config.json"
            payload = self.config(config_path)
            payload["preprocessing"] = {"mode": "quantum"}
            with self.assertRaisesRegex(ValueError, "unsupported preprocessing mode"):
                core.validate_config(payload)


if __name__ == "__main__":
    unittest.main()
