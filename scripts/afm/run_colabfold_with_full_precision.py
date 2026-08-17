#!/usr/bin/env python3
"""Run ColabFold 1.5.5 while preserving full-precision AF-M rank metrics.

The script accepts the same command-line arguments as ``colabfold_batch``.  It
wraps ColabFold's existing prediction callback so that ``ranking_confidence``,
``iptm`` and ``ptm`` are captured before ColabFold rounds ipTM/pTM in its
default scores JSON files.

For every successfully predicted system it atomically writes
``<system>_full_precision_ranking.json``.  On exit it aggregates every such
manifest in the result directory into ``full_precision_ranking.csv``.

This wrapper is intentionally pinned to ColabFold 1.5.5 because it relies on
that version's internal ``predict_structure`` callback contract.  It does not
enable ``--save-all`` and does not modify the installed ColabFold package.
"""

from __future__ import annotations

import csv
import inspect
import json
import math
import os
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SUPPORTED_COLABFOLD_VERSION = "1.5.5"
MANIFEST_SUFFIX = "_full_precision_ranking.json"
AGGREGATE_CSV_NAME = "full_precision_ranking.csv"
FULL_PRECISION_FIELDS = ("ranking_confidence", "iptm", "ptm")
CSV_FIELDS = (
    "system_id",
    "rank",
    "model_type",
    "model_weight",
    "seed",
    "prediction_tag",
    "rank_tag",
    "ranking_confidence",
    "iptm",
    "ptm",
    "pdb_file",
    "scores_file",
    "manifest_file",
)

_PREDICTION_TAG_RE = re.compile(
    r"^(?P<model_type>.+)_model_(?P<model_weight>\d+)_seed_(?P<seed>\d+)$"
)
_RANK_TAG_RE = re.compile(r"^rank_(?P<rank>\d+)_(?P<prediction_tag>.+)$")


def _finite_scalar(value: Any, *, field: str, prediction_tag: str) -> float:
    """Convert a model scalar to a finite Python float without rounding it."""

    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} is not a scalar for {prediction_tag}: {value!r}"
        ) from exc
    if not math.isfinite(scalar):
        raise ValueError(
            f"{field} is not finite for {prediction_tag}: {scalar!r}"
        )
    return scalar


def parse_prediction_tag(prediction_tag: str) -> tuple[str, int, int]:
    """Return model type, model weight and seed from a ColabFold tag."""

    match = _PREDICTION_TAG_RE.fullmatch(prediction_tag)
    if match is None:
        raise ValueError(f"Unrecognized ColabFold prediction tag: {prediction_tag}")
    return (
        match.group("model_type"),
        int(match.group("model_weight")),
        int(match.group("seed")),
    )


def parse_rank_tag(rank_tag: str) -> tuple[int, str]:
    """Return the one-based global rank and prediction tag."""

    match = _RANK_TAG_RE.fullmatch(rank_tag)
    if match is None:
        raise ValueError(f"Unrecognized ColabFold rank tag: {rank_tag}")
    return int(match.group("rank")), match.group("prediction_tag")


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text through a same-directory temporary file and atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _tag_from_callback(tag_info: Any) -> str:
    if isinstance(tag_info, (tuple, list)) and tag_info:
        return str(tag_info[0])
    return str(tag_info)


def _build_ranked_rows(
    *,
    system_id: str,
    result_dir: Path,
    rank_tags: Sequence[str],
    captured: Mapping[str, Mapping[str, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ranked_prediction_tags: set[str] = set()

    for rank_tag in rank_tags:
        rank, prediction_tag = parse_rank_tag(rank_tag)
        if prediction_tag not in captured:
            raise RuntimeError(
                f"No full-precision callback metrics for {system_id} "
                f"prediction {prediction_tag}"
            )
        model_type, model_weight, seed = parse_prediction_tag(prediction_tag)
        metrics = captured[prediction_tag]
        rows.append(
            {
                "system_id": system_id,
                "rank": rank,
                "model_type": model_type,
                "model_weight": model_weight,
                "seed": seed,
                "prediction_tag": prediction_tag,
                "rank_tag": rank_tag,
                "ranking_confidence": metrics["ranking_confidence"],
                "iptm": metrics["iptm"],
                "ptm": metrics["ptm"],
                "pdb_file": str(
                    result_dir / f"{system_id}_unrelaxed_{rank_tag}.pdb"
                ),
                "scores_file": str(
                    result_dir / f"{system_id}_scores_{rank_tag}.json"
                ),
            }
        )
        ranked_prediction_tags.add(prediction_tag)

    extra = sorted(set(captured) - ranked_prediction_tags)
    if extra:
        raise RuntimeError(
            f"Captured predictions missing from ColabFold ranks for {system_id}: "
            + ", ".join(extra)
        )
    return rows


def write_system_manifest(
    *,
    system_id: str,
    result_dir: Path,
    rank_by: str,
    colabfold_version: str,
    rows: Sequence[Mapping[str, object]],
) -> Path:
    """Atomically write one system's full-precision ranking manifest."""

    manifest_path = result_dir / f"{system_id}{MANIFEST_SUFFIX}"
    payload = {
        "schema_version": 1,
        "source": "colabfold_prediction_callback_before_default_rounding",
        "colabfold_version": colabfold_version,
        "system_id": system_id,
        "rank_by": rank_by,
        "num_predictions": len(rows),
        "ranking_confidence_formula_for_afm_multimer": (
            "0.8 * iptm + 0.2 * ptm"
        ),
        "predictions": list(rows),
    }
    _atomic_write_text(
        manifest_path,
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )
    return manifest_path


def make_predict_structure_wrapper(
    original_predict_structure: Callable[..., Mapping[str, Any]],
    *,
    colabfold_version: str,
) -> Callable[..., Mapping[str, Any]]:
    """Wrap ColabFold's predictor and persist metrics after its global rerank."""

    signature = inspect.signature(original_predict_structure)

    def wrapped_predict_structure(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        bound = signature.bind_partial(*args, **kwargs)
        system_id = str(bound.arguments["prefix"])
        result_dir = Path(bound.arguments["result_dir"]).resolve()
        rank_by = str(bound.arguments.get("rank_by", "auto"))
        downstream_callback = bound.arguments.get("prediction_callback")
        captured: dict[str, dict[str, float]] = {}

        def capture_callback(
            unrelaxed_protein: Any,
            sequence_lengths: Any,
            result: Mapping[str, Any],
            input_features: Any,
            tag_info: Any,
        ) -> None:
            prediction_tag = _tag_from_callback(tag_info)
            if prediction_tag in captured:
                raise RuntimeError(
                    f"Duplicate prediction callback tag for {system_id}: "
                    f"{prediction_tag}"
                )
            missing = [field for field in FULL_PRECISION_FIELDS if field not in result]
            if missing:
                raise RuntimeError(
                    f"Missing AF-M confidence fields for {system_id} "
                    f"{prediction_tag}: {', '.join(missing)}"
                )
            captured[prediction_tag] = {
                field: _finite_scalar(
                    result[field], field=field, prediction_tag=prediction_tag
                )
                for field in FULL_PRECISION_FIELDS
            }
            if downstream_callback is not None:
                downstream_callback(
                    unrelaxed_protein,
                    sequence_lengths,
                    result,
                    input_features,
                    tag_info,
                )

        bound.arguments["prediction_callback"] = capture_callback
        results = original_predict_structure(*bound.args, **bound.kwargs)
        rank_tags = results.get("rank")
        if not isinstance(rank_tags, (list, tuple)):
            raise RuntimeError(
                f"ColabFold did not return a rank list for {system_id}: {rank_tags!r}"
            )
        rows = _build_ranked_rows(
            system_id=system_id,
            result_dir=result_dir,
            rank_tags=[str(tag) for tag in rank_tags],
            captured=captured,
        )
        write_system_manifest(
            system_id=system_id,
            result_dir=result_dir,
            rank_by=rank_by,
            colabfold_version=colabfold_version,
            rows=rows,
        )
        return results

    return wrapped_predict_structure


def _csv_value(field: str, value: object) -> object:
    if field in FULL_PRECISION_FIELDS:
        return format(float(value), ".17g")
    return value


def aggregate_manifests(result_dir: Path) -> tuple[Path | None, int, int]:
    """Aggregate all per-system manifests into a deterministic CSV."""

    manifests = sorted(result_dir.glob(f"*{MANIFEST_SUFFIX}"))
    if not manifests:
        return None, 0, 0

    rows: list[dict[str, object]] = []
    systems: set[str] = set()
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        system_id = str(payload["system_id"])
        predictions = payload.get("predictions")
        if not isinstance(predictions, list):
            raise ValueError(f"Manifest has no prediction list: {manifest_path}")
        systems.add(system_id)
        for prediction in predictions:
            if not isinstance(prediction, dict):
                raise ValueError(f"Invalid prediction row in {manifest_path}")
            row = dict(prediction)
            row["manifest_file"] = str(manifest_path)
            rows.append(row)

    rows.sort(
        key=lambda row: (
            str(row["system_id"]),
            int(row["rank"]),
        )
    )
    output_path = result_dir / AGGREGATE_CSV_NAME
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {field: _csv_value(field, row[field]) for field in CSV_FIELDS}
                )
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_path, len(systems), len(rows)


def _result_dir_from_argv(argv: Sequence[str]) -> Path | None:
    # colabfold_batch has two required leading positionals: input and results.
    if len(argv) < 3 or argv[1] in {"-h", "--help"}:
        return None
    return Path(argv[2]).resolve()


def main() -> int:
    try:
        colabfold_version = metadata.version("colabfold")
    except metadata.PackageNotFoundError as exc:
        raise SystemExit(
            "colabfold is not installed in this Python environment; run this "
            "script with the LocalColabFold AF-M Python interpreter"
        ) from exc
    if colabfold_version != SUPPORTED_COLABFOLD_VERSION:
        raise SystemExit(
            "Unsupported ColabFold version: "
            f"{colabfold_version}; expected {SUPPORTED_COLABFOLD_VERSION}"
        )

    from colabfold import batch as colabfold_batch

    result_dir = _result_dir_from_argv(sys.argv)
    original_predict_structure = colabfold_batch.predict_structure
    colabfold_batch.predict_structure = make_predict_structure_wrapper(
        original_predict_structure,
        colabfold_version=colabfold_version,
    )
    try:
        colabfold_batch.main()
    finally:
        colabfold_batch.predict_structure = original_predict_structure
        if result_dir is not None and result_dir.is_dir():
            output_path, system_count, row_count = aggregate_manifests(result_dir)
            if output_path is not None:
                print(
                    "Full-precision AF-M metrics: "
                    f"{system_count} systems, {row_count} predictions -> "
                    f"{output_path}",
                    file=sys.stderr,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
