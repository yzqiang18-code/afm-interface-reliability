from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = REPO_ROOT / "scripts" / "ml"

sys.path.insert(0, str(ML_DIR))
import train_jaccard_nn as train  # noqa: E402

try:
    import torch

    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False


SLOTS = [(1, 0), (1, 1), (2, 0), (2, 1)]
FEATURES = ["j_0", "j_1", "j_2"]


def make_nn_frame(n_systems: int = 6, seed: int = 11) -> pd.DataFrame:
    """Candidate frame with 1x19 jaccard features, folds, labels, and a signal.

    Labels depend on rank parity (even ranks acceptable), and ``j_0`` encodes
    that parity, so every fold contains both classes and a model can learn a
    within-fold signal. System IDs parse as ``<pdb>__<chain>_<acc>`` pairs so
    the repository's label-mapping audit accepts them.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_systems):
        cid = f"x{s:02d}__A_P{s:03d}1--x{s:02d}__B_P{s:03d}2"
        fold = s % 3
        for rank, (mw, seed_) in enumerate(SLOTS, start=1):
            acc = (rank % 2) == 0
            j0 = 0.75 if acc else 0.25
            j0 += rng.uniform(-0.05, 0.05)
            j1 = float(rng.uniform(0.2, 0.9))
            j2 = float(rng.uniform(0.2, 0.9))
            row = {
                "complex_id": cid,
                "rank": rank,
                "model_weight": mw,
                "seed": seed_,
                "cv_fold": fold,
                "DockQ": 0.6 if acc else 0.1,
                "ranking_confidence": float(100 - rank * 2),
                "mapping_mode": "fixed",
                "j_0": j0,
                "j_1": j1,
                "j_2": j2,
                "mean_j": float(np.mean([j0, j1, j2])),
                "iptm_full_precision": float(rng.uniform(0.5, 0.9)),
                "ptm_full_precision": float(rng.uniform(0.5, 0.9)),
                "pDockQ2_min": float(rng.uniform(0.2, 0.8)),
                "iLIS": float(rng.uniform(0.2, 0.8)),
                "ipSAE": float(rng.uniform(0.2, 0.8)),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def make_nn_config(nn_overrides: dict | None = None) -> dict:
    nn_cfg = {
        "activation": "relu",
        "batch_size": 16,
        "epochs": 3,
        "hidden_sizes": [4, 2],
        "learning_rate": 0.05,
        "weight_decay": 0.0,
    }
    if nn_overrides:
        nn_cfg.update(nn_overrides)
    return {
        "bootstrap_replicates": 50,
        "expected_candidates_per_system": 4,
        "expected_fold_values": [0, 1, 2],
        "feature_columns": FEATURES,
        "fold_column": "cv_fold",
        "group_column": "complex_id",
        "key_columns": ["complex_id", "rank", "model_weight", "seed"],
        "label_mapping_policy": {
            "different_accession_mode": "fixed",
            "mode_column": "mapping_mode",
            "same_accession_mode": "symmetry_aware",
        },
        "max_missing_fraction_per_feature": 0.1,
        "model_name": "synthetic_jaccard_nn",
        "nn": nn_cfg,
        "random_seed": 2026,
        "reference_score_column": "ranking_confidence",
        "ridge_penalty": 1.0,
        "target_column": "DockQ",
        "target_threshold": 0.23,
        "task": "synthetic test",
    }


def make_ridge_config() -> dict:
    return {
        "feature_columns": ["j_0", "j_1", "j_2"],
        "ridge_penalty": 1.0,
    }


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for the NN tests")
class JaccardNNUnitTests(unittest.TestCase):
    def test_small_mlp_output_shape(self) -> None:
        model = train.SmallMLP(3, [4, 2])
        x = torch.randn(7, 3)
        out = model(x)
        self.assertEqual(out.shape, (7, 1))

    def test_predict_fold_permutation_equivariance(self) -> None:
        """A row-wise shared MLP must give permuted scores under row permutation."""
        frame = make_nn_frame(n_systems=3)
        config = make_nn_config()
        result = train.fit_fold(
            frame, frame, FEATURES, config, fold=0, device="cpu"
        )
        scores = train.predict_fold(frame, FEATURES, result, device="cpu")
        perm = [2, 0, 3, 1, 5, 4, 6, 7, 9, 8, 10, 11]
        perm_scores = train.predict_fold(
            frame.iloc[perm].reset_index(drop=True), FEATURES, result, device="cpu"
        )
        self.assertTrue(np.allclose(scores[perm], perm_scores))

    def test_oof_cv_no_system_leakage(self) -> None:
        """Each fold's model must never train on the systems it validates."""
        frame = make_nn_frame()
        config = make_nn_config()
        seen: list[set] = []
        original = train.fit_fold

        def recording_fit(training, validation, feature_columns, config, *, fold, device):
            seen.append(set(training["complex_id"].astype(str)))
            return original(training, validation, feature_columns, config, fold=fold, device=device)

        train.fit_fold = recording_fit
        try:
            oof, folds = train.oof_cv(frame, config=config, device="cpu")
        finally:
            train.fit_fold = original
        self.assertTrue(np.isfinite(oof["acceptable_score"]).all())
        for fold, systems in enumerate(seen):
            val_systems = set(
                frame.loc[frame["cv_fold"] == fold, "complex_id"].astype(str)
            )
            self.assertTrue(systems.isdisjoint(val_systems))

    def test_grouped_ridge_oof_learns_signal(self) -> None:
        frame = make_nn_frame(n_systems=6, seed=5)
        oof = train.grouped_ridge_oof(frame, ["j_0"], penalty=1.0)
        self.assertTrue(np.isfinite(oof["acceptable_score"]).all())
        rho = np.corrcoef(oof["acceptable_score"], oof["DockQ"])[0, 1]
        self.assertGreater(rho, 0.3)

    def test_rerank_metrics(self) -> None:
        frame = make_nn_frame()
        frame["acceptable"] = frame["DockQ"].ge(0.23)
        groups = {cid: g for cid, g in frame.groupby("complex_id")}
        # Perfect ordering by DockQ gives recall@1 = fraction with any acceptable.
        frame["perfect"] = frame["DockQ"]
        metrics = train.rerank_metrics(groups, "DockQ")
        self.assertIn("within_system_spearman_median", metrics)
        self.assertIn("recall_at_3", metrics)
        self.assertGreaterEqual(metrics["recall_at_1"], 0.0)


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for the NN tests")
class JaccardNNE2ETests(unittest.TestCase):
    def test_end_to_end_training(self) -> None:
        frame = make_nn_frame(n_systems=6, seed=9)
        config = make_nn_config({"epochs": 2})
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            train_csv = tmp / "frame.csv"
            frame.to_csv(train_csv, index=False)
            config_path = tmp / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            ridge_path = tmp / "ridge.json"
            ridge_path.write_text(json.dumps(make_ridge_config()), encoding="utf-8")
            out = tmp / "out"
            rc = train.main(
                [
                    "--train-csv",
                    str(train_csv),
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(out),
                    "--ridge-config",
                    str(ridge_path),
                    "--device",
                    "cpu",
                ]
            )
            self.assertEqual(rc, 0)
            expected = [
                "model.json",
                "training_summary.json",
                "oof_predictions.csv",
                "oof_candidate_metrics.csv",
                "oof_selector_choices.csv",
                "oof_selector_summary.csv",
                "oof_paired_bootstrap.csv",
                "comparison_metrics.json",
                "rerank_metrics.csv",
            ]
            for name in expected:
                self.assertTrue((out / name).is_file(), f"missing {name}")
            for fold in range(3):
                self.assertTrue((out / "checkpoints" / f"fold_{fold}.pt").is_file())
            oof = pd.read_csv(out / "oof_predictions.csv")
            self.assertEqual(len(oof), len(frame))
            self.assertTrue(np.isfinite(oof["acceptable_score"]).all())


if __name__ == "__main__":
    unittest.main()
