#!/usr/bin/env python3
"""Run the vendored IPSAE implementation and collect pDockQ2 for AF-M dimers."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from common import (
    PredictionFiles,
    discover_predictions,
    prediction_metadata,
    read_ids,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IPSAE = REPO_ROOT / "third_party" / "ipsae" / "ipsae.py"
FIELDNAMES = [
    "complex_id",
    "rank",
    "model_family",
    "model_weight",
    "seed",
    "chain_1",
    "chain_2",
    "pDockQ2_chain1_to_chain2",
    "pDockQ2_chain2_to_chain1",
    "pDockQ2_min",
    "pDockQ2_mean",
    "pDockQ2_max",
    "prediction_path",
    "scores_path",
    "status",
    "error",
]


def parse_ipsae_output(path: Path) -> dict[str, object]:
    asym_rows: list[tuple[str, str, float]] = []
    max_value: float | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 13 or fields[4] not in {"asym", "max"}:
            continue
        value = float(fields[11])
        if fields[4] == "asym":
            asym_rows.append((fields[0], fields[1], value))
        else:
            max_value = value

    if len(asym_rows) != 2:
        raise ValueError(
            f"Expected exactly two directional dimer rows in {path}, got {len(asym_rows)}"
        )
    first, second = asym_rows
    if first[0] != second[1] or first[1] != second[0]:
        raise ValueError(f"Directional chain rows are not reciprocal in {path}")

    values = [first[2], second[2]]
    calculated_max = max(values)
    if max_value is not None and abs(max_value - calculated_max) > 5e-4:
        raise ValueError(
            f"Upstream max pDockQ2 {max_value} disagrees with directions {values}"
        )
    return {
        "chain_1": first[0],
        "chain_2": first[1],
        "pDockQ2_chain1_to_chain2": first[2],
        "pDockQ2_chain2_to_chain1": second[2],
        "pDockQ2_min": min(values),
        "pDockQ2_mean": statistics.fmean(values),
        "pDockQ2_max": calculated_max,
    }


def run_one(
    prediction: PredictionFiles,
    *,
    ipsae_script: Path,
    pae_cutoff: int,
    distance_cutoff: int,
) -> dict[str, object]:
    row = {
        **prediction_metadata(prediction),
        "chain_1": "",
        "chain_2": "",
        "pDockQ2_chain1_to_chain2": "",
        "pDockQ2_chain2_to_chain1": "",
        "pDockQ2_min": "",
        "pDockQ2_mean": "",
        "pDockQ2_max": "",
        "status": "failed",
        "error": "",
    }
    try:
        with tempfile.TemporaryDirectory(prefix="afm_pdockq2_") as tmp:
            tmpdir = Path(tmp)
            pdb_link = tmpdir / prediction.pdb_path.name
            scores_link = tmpdir / prediction.scores_path.name
            os.symlink(prediction.pdb_path.resolve(), pdb_link)
            os.symlink(prediction.scores_path.resolve(), scores_link)

            command = [
                sys.executable,
                str(ipsae_script),
                str(scores_link),
                str(pdb_link),
                str(pae_cutoff),
                str(distance_cutoff),
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                details = (completed.stderr or completed.stdout).strip()
                raise RuntimeError(
                    f"IPSAE exited {completed.returncode}: {details[-1000:]}"
                )

            output = pdb_link.with_suffix("").with_name(
                f"{pdb_link.stem}_{pae_cutoff:02d}_{distance_cutoff:02d}.txt"
            )
            if not output.is_file():
                raise FileNotFoundError(f"Expected IPSAE output not found: {output}")
            row.update(parse_ipsae_output(output))
            row["status"] = "ok"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--ipsae-script", type=Path, default=DEFAULT_IPSAE)
    parser.add_argument("--pae-cutoff", type=int, default=15)
    parser.add_argument("--distance-cutoff", type=int, default=15)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-models", type=int)
    return parser.parse_args()


def run_batch(
    predictions: list[PredictionFiles],
    *,
    worker: Callable[[PredictionFiles], dict[str, object]],
    workers: int,
) -> list[dict[str, object]]:
    if workers == 1:
        return [
            worker(item)
            for item in tqdm(
                predictions,
                total=len(predictions),
                desc="pDockQ2",
                unit="model",
            )
        ]

    rows_by_index: list[dict[str, object] | None] = [None] * len(predictions)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_index = {
            pool.submit(worker, item): index
            for index, item in enumerate(predictions)
        }
        for future in tqdm(
            as_completed(future_to_index),
            total=len(future_to_index),
            desc="pDockQ2",
            unit="model",
        ):
            rows_by_index[future_to_index[future]] = future.result()

    if any(row is None for row in rows_by_index):
        raise RuntimeError("Not all pDockQ2 tasks produced a result")
    return [row for row in rows_by_index if row is not None]


def main() -> int:
    args = parse_args()
    if not args.ipsae_script.is_file():
        raise FileNotFoundError(args.ipsae_script)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    predictions = discover_predictions(
        args.prediction_dir,
        allowed_ids=read_ids(args.ids_file),
        max_models=args.max_models,
    )
    worker = lambda item: run_one(
        item,
        ipsae_script=args.ipsae_script,
        pae_cutoff=args.pae_cutoff,
        distance_cutoff=args.distance_cutoff,
    )
    rows = run_batch(predictions, worker=worker, workers=args.workers)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    failures = sum(row["status"] != "ok" for row in rows)
    print(
        f"pDockQ2: {len(rows) - failures} succeeded, {failures} failed; "
        f"output={args.output_csv}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
