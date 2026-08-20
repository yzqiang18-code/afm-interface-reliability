#!/usr/bin/env python3
"""Build per-system 20x20 contact-Jaccard matrices and candidate row vectors.

Inputs
------
- ``results/data/training500_consistency_pairs.csv.gz``: 190 unordered
  candidate pairs per Training500 system (contact Jaccard, validity flags).
- ``results/data/training500_candidates.csv.gz``: the 10,000 development
  candidates with ``(complex_id, model_weight, seed)`` keys, folds, and labels.

Outputs
-------
- ``matrices.npz``: one symmetric 20x20 contact-Jaccard matrix per system,
  in a fixed canonical slot order (``(model_weight, seed)`` ascending).
  Diagonal entries are 1.0 (self-similarity); invalid pairs are NaN before
  imputation.
- ``training500_jaccard_rows.csv.gz``: one row per candidate containing the
  candidate table plus two native-independent feature sets built from the
  matrix:
    * ``j_0..j_18``   — the 1x19 row excluding the candidate's own slot;
    * ``j20_0..j20_19`` — the 1x20 full row including self (diagonal = 1.0).
  plus ``mean_j`` (mean of the candidate's *valid* pairwise contact Jaccards,
  matching ``analysis/pairwise_consistency_diagnostic.py``).
- ``matrix_manifest.json``: hashes, row/slot coverage, invalid-pair and
  imputation statistics, and the column schema.

Discipline: only candidate-vs-candidate similarities enter the features.
``DockQ`` and related native-referenced columns are labels/evaluation only.
This is a Training500 exploratory artifact; the PINDER-AF2 holdout has no
published pair table, so this pipeline cannot be extended to the holdout
from public data alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_core import (  # noqa: E402
    atomic_write_json,
    portable_path,
    sha256_file,
    utc_now,
)


def as_bool(series: pd.Series) -> pd.Series:
    """Robustly parse a column that may be bool or 'True'/'False' strings."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().eq("true")


def canonical_slots(candidates: pd.DataFrame) -> list[tuple[int, int]]:
    """Fixed slot order: (model_weight, seed) ascending."""
    slots = sorted(set(zip(candidates["model_weight"], candidates["seed"])))
    if not slots:
        raise ValueError("No (model_weight, seed) slots found")
    return slots


def build_matrices(
    pairs: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    impute_method: str,
) -> tuple[dict[str, np.ndarray], dict]:
    slots = canonical_slots(candidates)
    slot_index = {slot: index for index, slot in enumerate(slots)}
    n = len(slots)

    matrices: dict[str, np.ndarray] = {}
    stats: dict = {
        "slot_order": [list(map(int, slot)) for slot in slots],
        "n_slots": n,
        "invalid_pair_count": 0,
        "invalid_pair_fraction": 0.0,
        "imputed_cell_count": 0,
        "impute_method": impute_method,
    }

    valid = as_bool(pairs["jaccard_valid"]).to_numpy()
    values = pairs["jaccard"].to_numpy(dtype=float)
    if (values[~valid] == values[~valid]).any() or not np.isfinite(
        values[valid]
    ).all():
        raise ValueError("Pair table contains non-finite jaccard on valid rows")
    side1 = pairs[["model_weight_1", "seed_1"]]
    side2 = pairs[["model_weight_2", "seed_2"]]
    i_idx = np.array([slot_index[(mw, s)] for mw, s in zip(side1["model_weight_1"], side1["seed_1"])])
    j_idx = np.array([slot_index[(mw, s)] for mw, s in zip(side2["model_weight_2"], side2["seed_2"])])

    stats["invalid_pair_count"] = int((~valid).sum())
    stats["invalid_pair_fraction"] = float((~valid).mean())

    off_diag = ~np.eye(n, dtype=bool)
    for cid, group in pairs.groupby("complex_id", sort=True):
        g_valid = valid[group.index]
        g_values = values[group.index]
        gi = i_idx[group.index]
        gj = j_idx[group.index]
        M = np.full((n, n), np.nan, dtype=float)
        np.fill_diagonal(M, 1.0)
        ok = g_valid & np.isfinite(g_values)
        M[gi[ok], gj[ok]] = g_values[ok]
        M[gj[ok], gi[ok]] = g_values[ok]
        if not np.allclose(M, M.T, equal_nan=True):
            raise ValueError(f"Asymmetric matrix for {cid}")
        if impute_method == "mean":
            fill = float(np.nanmean(M[off_diag])) if np.isfinite(np.nanmean(M[off_diag])) else 0.0
        elif impute_method == "zero":
            fill = 0.0
        else:
            raise ValueError(f"Unknown impute method: {impute_method}")
        n_missing = int(np.isnan(M).sum())
        stats["imputed_cell_count"] += n_missing
        M = np.where(np.isnan(M), fill, M)
        if not np.isfinite(M).all():
            raise ValueError(f"Non-finite entries remain for {cid}")
        if not np.allclose(np.diag(M), 1.0):
            raise ValueError(f"Diagonal is not 1.0 for {cid}")
        matrices[cid] = M

    # Coverage: every system must occupy exactly the canonical slot set.
    present = candidates.groupby("complex_id").apply(
        lambda g: set(zip(g["model_weight"], g["seed"])), include_groups=False
    )
    expected = set(slots)
    bad = present[present.ne(expected)]
    if not bad.empty:
        raise ValueError(f"Systems missing canonical slots: {bad.head().to_dict()}")

    # Candidate-level mean over valid pairs (matches the diagnostic's mean_j).
    long = pd.concat(
        [
            pd.DataFrame(
                {
                    "complex_id": pairs["complex_id"],
                    "model_weight": pairs["model_weight_1"],
                    "seed": pairs["seed_1"],
                    "jaccard": values,
                    "valid": valid,
                }
            ),
            pd.DataFrame(
                {
                    "complex_id": pairs["complex_id"],
                    "model_weight": pairs["model_weight_2"],
                    "seed": pairs["seed_2"],
                    "jaccard": values,
                    "valid": valid,
                }
            ),
        ],
        ignore_index=True,
    )
    mean_j = (
        long.loc[long["valid"], ["complex_id", "model_weight", "seed", "jaccard"]]
        .groupby(["complex_id", "model_weight", "seed"])["jaccard"]
        .mean()
        .rename("mean_j")
    )
    stats["candidates_with_mean_j"] = int(len(mean_j))
    stats["mean_j_summary"] = {
        "mean": float(mean_j.mean()),
        "median": float(mean_j.median()),
    }
    return matrices, {"stats": stats, "mean_j": mean_j, "slots": slots}


def build_row_table(
    candidates: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    slots: list[tuple[int, int]],
    mean_j: pd.Series,
    *,
    expected_candidates_per_system: int,
) -> pd.DataFrame:
    """Assemble the per-candidate row-vector table from per-system matrices.

    Each candidate row carries the candidate table plus:
    - ``j_0..j_{n-2}``: the 1x19 row vector (all *other* slots, ascending);
    - ``j20_0..j20_{n-1}``: the 1x20 full row including self (diagonal = 1.0);
    - ``mean_j``: mean of the candidate's valid pairwise contact Jaccards.
    """
    slot_index = {slot: index for index, slot in enumerate(slots)}
    n = len(slots)
    rows: list[dict] = []
    for _, candidate in candidates.iterrows():
        cid = candidate["complex_id"]
        M = matrices[cid]
        i = slot_index[(candidate["model_weight"], candidate["seed"])]
        row = dict(candidate)
        key = (cid, candidate["model_weight"], candidate["seed"])
        row["mean_j"] = float(mean_j.get(key, np.nan))
        other = [j for j in range(n) if j != i]
        for k, j in enumerate(other):
            row[f"j_{k}"] = float(M[i, j])
        for j in range(n):
            row[f"j20_{j}"] = float(M[i, j])
        rows.append(row)

    frame = pd.DataFrame(rows)

    key_cols = ["complex_id", "model_weight", "seed"]
    if len(frame) != len(candidates):
        raise ValueError("Row table row count differs from candidate table")
    if frame.duplicated(key_cols).any():
        raise ValueError("Row table contains duplicate candidate keys")
    sizes = frame.groupby("complex_id").size()
    if not sizes.eq(expected_candidates_per_system).all():
        raise ValueError(
            "Row table does not have exactly "
            f"{expected_candidates_per_system} rows per system: "
            f"{sizes.value_counts().to_dict()}"
        )
    feature_cols = [f"j_{k}" for k in range(n - 1)] + [f"j20_{k}" for k in range(n)]
    if frame[feature_cols].isna().any(axis=None):
        raise ValueError("Row table contains missing jaccard features")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        default="results/data/training500_consistency_pairs.csv.gz",
    )
    parser.add_argument("--candidates", default="results/data/training500_candidates.csv.gz")
    parser.add_argument("--rows-out", default="results/data/training500_jaccard_rows.csv.gz")
    parser.add_argument("--matrix-out", default="results/ml/pairwise_jaccard_nn/matrices.npz")
    parser.add_argument("--manifest-out", default="results/ml/pairwise_jaccard_nn/matrix_manifest.json")
    parser.add_argument("--impute-method", choices=["mean", "zero"], default="mean")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pairs_path = Path(args.pairs).expanduser().resolve()
    candidates_path = Path(args.candidates).expanduser().resolve()
    rows_out = Path(args.rows_out).expanduser().resolve()
    matrix_out = Path(args.matrix_out).expanduser().resolve()
    manifest_out = Path(args.manifest_out).expanduser().resolve()

    for path, name in [(pairs_path, "pairs"), (candidates_path, "candidates")]:
        if not path.is_file():
            print(f"ERROR: {name} input not found: {path}", file=sys.stderr)
            return 1
    from baseline_core import check_output_targets

    check_output_targets([rows_out, matrix_out, manifest_out], overwrite=args.overwrite)

    pairs = pd.read_csv(pairs_path, low_memory=False)
    candidates = pd.read_csv(candidates_path, low_memory=False)

    matrices, extras = build_matrices(
        pairs, candidates, impute_method=args.impute_method
    )
    slots = extras["slots"]
    mean_j = extras["mean_j"]
    stats = extras["stats"]
    n = len(slots)

    frame = build_row_table(
        candidates,
        matrices,
        slots,
        mean_j,
        expected_candidates_per_system=20,
    )
    key_cols = ["complex_id", "model_weight", "seed"]

    matrix_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        matrix_out,
        **{str(cid): matrices[cid].astype(np.float32) for cid in sorted(matrices)},
    )

    # Append the new columns to the manifest-compatible audit.
    manifest = {
        "created_at_utc": utc_now(),
        "inputs": {
            "pairs": {
                "path": portable_path(pairs_path),
                "sha256": sha256_file(pairs_path),
                "rows": int(len(pairs)),
            },
            "candidates": {
                "path": portable_path(candidates_path),
                "sha256": sha256_file(candidates_path),
                "rows": int(len(candidates)),
            },
        },
        "outputs": {
            "row_table": {
                "path": portable_path(rows_out),
                "rows": int(len(frame)),
                "systems": int(frame["complex_id"].nunique()),
            },
            "matrices": {"path": portable_path(matrix_out), "systems": len(matrices)},
        },
        "schema": {
            "key_columns": key_cols,
            "row_19_columns": [f"j_{k}" for k in range(n - 1)],
            "row_20_columns": [f"j20_{k}" for k in range(n)],
            "slot_order": [list(map(int, slot)) for slot in slots],
        },
        "stats": stats,
    }

    rows_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = rows_out.with_name(rows_out.name + ".part")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(rows_out)
    atomic_write_json(manifest_out, manifest)

    print(
        f"Built {len(matrices)} matrices and {len(frame)} candidate rows "
        f"({frame['complex_id'].nunique()} systems); "
        f"invalid pairs {stats['invalid_pair_fraction']:.4%}, "
        f"imputed cells {stats['imputed_cell_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
