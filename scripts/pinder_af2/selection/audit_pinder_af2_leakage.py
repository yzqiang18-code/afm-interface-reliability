#!/usr/bin/env python3
"""Audit PINDER-AF2 overlap with frozen training/development manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd


REQUIRED = {
    "id",
    "pdb_id",
    "cluster_id",
    "cluster_id_R",
    "cluster_id_L",
    "uniprot_R",
    "uniprot_L",
}
LEVELS = (
    "id",
    "pdb_id",
    "cluster_id",
    "chain_cluster_pair",
    "uniprot_pair",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest {path} lacks columns: {sorted(missing)}")
    if frame["id"].astype(str).duplicated().any():
        raise ValueError(f"Manifest contains duplicate IDs: {path}")
    return frame


def unordered_pair(left: object, right: object) -> str:
    return "|".join(sorted((str(left), str(right))))


def keys(frame: pd.DataFrame) -> dict[str, set[str]]:
    uni = {
        unordered_pair(row.uniprot_R, row.uniprot_L)
        for row in frame.itertuples(index=False)
        if str(row.uniprot_R) != "UNDEFINED" and str(row.uniprot_L) != "UNDEFINED"
    }
    return {
        "id": set(frame["id"].astype(str)),
        "pdb_id": set(frame["pdb_id"].astype(str)),
        "cluster_id": set(frame["cluster_id"].astype(str)),
        "chain_cluster_pair": {
            unordered_pair(row.cluster_id_R, row.cluster_id_L)
            for row in frame.itertuples(index=False)
        },
        "uniprot_pair": uni,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--reference-manifest", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        test_path = Path(args.test_manifest).expanduser().resolve()
        reference_paths = [
            Path(value).expanduser().resolve() for value in args.reference_manifest
        ]
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        test_keys = keys(load_manifest(test_path))
        rows: list[dict[str, object]] = []
        counts: list[dict[str, object]] = []
        for reference_path in reference_paths:
            reference_keys = keys(load_manifest(reference_path))
            for level in LEVELS:
                intersections = sorted(
                    test_keys[level].intersection(reference_keys[level])
                )
                counts.append(
                    {
                        "reference_manifest": str(reference_path),
                        "level": level,
                        "count": len(intersections),
                    }
                )
                for value in intersections:
                    rows.append(
                        {
                            "reference_manifest": str(reference_path),
                            "level": level,
                            "value": value,
                        }
                    )
        detail_path = output_dir / "leakage_intersections.csv"
        detail_temp = detail_path.with_name(detail_path.name + ".part")
        with detail_temp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["reference_manifest", "level", "value"]
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        detail_temp.replace(detail_path)
        summary = {
            "test_manifest": str(test_path),
            "test_manifest_sha256": sha256_file(test_path),
            "reference_manifests": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in reference_paths
            ],
            "intersection_counts": counts,
            "undefined_uniprot_pairs_excluded": True,
            "detail_csv": str(detail_path),
        }
        summary_path = output_dir / "leakage_summary.json"
        summary_temp = summary_path.with_name(summary_path.name + ".part")
        with summary_temp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        summary_temp.replace(summary_path)
        print(json.dumps(summary, indent=2))
        return 1 if rows else 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
