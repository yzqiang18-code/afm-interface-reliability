from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = REPO_ROOT / "scripts" / "ml"

sys.path.insert(0, str(ML_DIR))
import build_jaccard_matrix as builder  # noqa: E402

SLOTS = [(1, 0), (1, 1), (2, 0), (2, 1)]
N_SYSTEMS = 3


def make_synthetic(n_systems: int = N_SYSTEMS, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small pair + candidate tables with complete slot coverage.

    System ``sys0`` has exactly one invalid pair (slots (1,0)-(1,1)) whose
    jaccard is NaN; every other pair is valid with a uniform value.
    """
    rng = np.random.default_rng(seed)
    pairs_rows = []
    cand_rows = []
    for s in range(n_systems):
        cid = f"sys{s}"
        for rank, (mw, seed_) in enumerate(SLOTS, start=1):
            cand_rows.append(
                {
                    "complex_id": cid,
                    "rank": rank,
                    "model_weight": mw,
                    "seed": seed_,
                }
            )
        for (a, b) in itertools.combinations(range(len(SLOTS)), 2):
            mw1, seed1 = SLOTS[a]
            mw2, seed2 = SLOTS[b]
            invalid = s == 0 and (a, b) == (0, 1)
            pairs_rows.append(
                {
                    "complex_id": cid,
                    "model_weight_1": mw1,
                    "seed_1": seed1,
                    "model_weight_2": mw2,
                    "seed_2": seed2,
                    "jaccard": np.nan if invalid else float(rng.uniform(0.1, 0.9)),
                    "jaccard_valid": not invalid,
                }
            )
    return pd.DataFrame(pairs_rows), pd.DataFrame(cand_rows)


class BuildJaccardMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pairs, self.candidates = make_synthetic()

    def test_canonical_slots_sorted(self) -> None:
        slots = builder.canonical_slots(self.candidates)
        self.assertEqual(slots, SLOTS)

    def test_matrices_symmetric_diagonal_finite(self) -> None:
        matrices, extras = builder.build_matrices(
            self.pairs, self.candidates, impute_method="mean"
        )
        self.assertEqual(sorted(matrices), ["sys0", "sys1", "sys2"])
        for cid, M in matrices.items():
            self.assertEqual(M.shape, (4, 4))
            self.assertTrue(np.allclose(M, M.T))
            self.assertTrue(np.allclose(np.diag(M), 1.0))
            self.assertTrue(np.isfinite(M).all())

    def test_imputation_fills_invalid_cell_with_system_mean(self) -> None:
        matrices, extras = builder.build_matrices(
            self.pairs, self.candidates, impute_method="mean"
        )
        M = matrices["sys0"]
        # The invalid pair is slots 0 <-> 1 (row 0, col 1).
        valid_vals = self.pairs.loc[
            (self.pairs["complex_id"] == "sys0")
            & self.pairs["jaccard_valid"].astype(str).str.lower().eq("true"),
            "jaccard",
        ].astype(float)
        expected = float(valid_vals.mean())
        # After imputation the cell is exactly the mean of the valid off-diag.
        self.assertAlmostEqual(M[0, 1], expected)
        self.assertAlmostEqual(M[1, 0], expected)

    def test_missing_slot_raises(self) -> None:
        pairs, candidates = make_synthetic()
        missing = candidates[
            ~((candidates["complex_id"] == "sys0") & (candidates["model_weight"] == 1))
        ]
        with self.assertRaises(ValueError):
            builder.build_matrices(pairs, missing, impute_method="mean")

    def test_row_table_columns_and_self_semantics(self) -> None:
        matrices, extras = builder.build_matrices(
            self.pairs, self.candidates, impute_method="mean"
        )
        frame = builder.build_row_table(
            self.candidates,
            matrices,
            extras["slots"],
            extras["mean_j"],
            expected_candidates_per_system=4,
        )
        self.assertEqual(len(frame), N_SYSTEMS * 4)
        self.assertEqual(frame.groupby("complex_id").size().unique().tolist(), [4])
        # 1x19 has n-1 columns, 1x20 has n columns.
        self.assertEqual([c for c in frame.columns if c.startswith("j_")], ["j_0", "j_1", "j_2"])
        self.assertEqual(
            [c for c in frame.columns if c.startswith("j20_")],
            ["j20_0", "j20_1", "j20_2", "j20_3"],
        )
        feat_cols = ["j_0", "j_1", "j_2"] + ["j20_0", "j20_1", "j20_2", "j20_3"]
        self.assertFalse(frame[feat_cols].isna().any(axis=None))

        slot_index = {slot: i for i, slot in enumerate(SLOTS)}
        for _, row in frame.iterrows():
            i = slot_index[(row["model_weight"], row["seed"])]
            M = matrices[row["complex_id"]]
            # 1x20 row equals the matrix row; the self entry is 1.0.
            self.assertTrue(np.allclose(
                [row[f"j20_{k}"] for k in range(4)], M[i, :]
            ))
            self.assertEqual(row[f"j20_{i}"], 1.0)
            # 1x19 row equals the matrix row with the self slot removed.
            others = [k for k in range(4) if k != i]
            self.assertTrue(np.allclose(
                [row[f"j_{k}"] for k in range(3)], M[i, others]
            ))

    def test_row_table_order_independent(self) -> None:
        matrices, extras = builder.build_matrices(
            self.pairs, self.candidates, impute_method="mean"
        )
        base = builder.build_row_table(
            self.candidates,
            matrices,
            extras["slots"],
            extras["mean_j"],
            expected_candidates_per_system=4,
        )
        shuffled = self.candidates.sample(frac=1.0, random_state=3).reset_index(drop=True)
        again = builder.build_row_table(
            shuffled,
            matrices,
            extras["slots"],
            extras["mean_j"],
            expected_candidates_per_system=4,
        )
        key = ["complex_id", "model_weight", "seed"]
        merged = base[key + ["j_0", "j_1", "j_2", "j20_0", "j20_3", "mean_j"]].merge(
            again[key + ["j_0", "j_1", "j_2", "j20_0", "j20_3", "mean_j"]],
            on=key,
            suffixes=("_a", "_b"),
        )
        for col in ["j_0", "j_1", "j_2", "j20_0", "j20_3"]:
            pd.testing.assert_series_equal(
                merged[f"{col}_a"], merged[f"{col}_b"], check_names=False
            )

    def test_row_table_rejects_duplicate_keys(self) -> None:
        matrices, extras = builder.build_matrices(
            self.pairs, self.candidates, impute_method="mean"
        )
        dup = pd.concat([self.candidates, self.candidates.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            builder.build_row_table(
                dup,
                matrices,
                extras["slots"],
                extras["mean_j"],
                expected_candidates_per_system=4,
            )

    def test_zero_imputation_is_exactly_zero(self) -> None:
        matrices, _ = builder.build_matrices(
            self.pairs, self.candidates, impute_method="zero"
        )
        self.assertEqual(matrices["sys0"][0, 1], 0.0)
        self.assertEqual(matrices["sys0"][1, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
