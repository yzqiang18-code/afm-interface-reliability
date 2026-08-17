"""Shared filename and input helpers for metric batch adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


PREDICTION_RE = re.compile(
    r"^(?P<complex_id>.+)_unrelaxed_rank_(?P<rank>\d+)_"
    r"(?P<model_family>.+)_model_(?P<model_weight>\d+)_"
    r"seed_(?P<seed>\d+)\.pdb$"
)


@dataclass(frozen=True)
class PredictionFiles:
    complex_id: str
    rank: int
    model_family: str
    model_weight: int
    seed: int
    pdb_path: Path
    scores_path: Path


def read_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    ids = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not ids:
        raise ValueError(f"No IDs found in {path}")
    return ids


def parse_prediction_path(pdb_path: Path) -> PredictionFiles:
    match = PREDICTION_RE.match(pdb_path.name)
    if match is None:
        raise ValueError(f"Unsupported ColabFold prediction filename: {pdb_path.name}")

    scores_name = pdb_path.name.replace("_unrelaxed_", "_scores_", 1)
    scores_path = pdb_path.with_name(Path(scores_name).with_suffix(".json").name)
    fields = match.groupdict()
    return PredictionFiles(
        complex_id=fields["complex_id"],
        rank=int(fields["rank"]),
        model_family=fields["model_family"],
        model_weight=int(fields["model_weight"]),
        seed=int(fields["seed"]),
        pdb_path=pdb_path,
        scores_path=scores_path,
    )


def discover_predictions(
    prediction_dir: Path,
    *,
    allowed_ids: set[str] | None = None,
    max_models: int | None = None,
    require_scores: bool = True,
) -> list[PredictionFiles]:
    if not prediction_dir.is_dir():
        raise NotADirectoryError(prediction_dir)

    predictions: list[PredictionFiles] = []
    for pdb_path in sorted(prediction_dir.rglob("*_unrelaxed_rank_*.pdb")):
        prediction = parse_prediction_path(pdb_path)
        if allowed_ids is not None and prediction.complex_id not in allowed_ids:
            continue
        if require_scores and not prediction.scores_path.is_file():
            raise FileNotFoundError(
                f"Missing scores JSON for {pdb_path}: {prediction.scores_path}"
            )
        predictions.append(prediction)
        if max_models is not None and len(predictions) >= max_models:
            break

    if not predictions:
        raise FileNotFoundError(
            f"No matching unrelaxed ColabFold predictions found in {prediction_dir}"
        )
    return predictions


def prediction_metadata(prediction: PredictionFiles) -> dict[str, object]:
    return {
        "complex_id": prediction.complex_id,
        "rank": prediction.rank,
        "model_family": prediction.model_family,
        "model_weight": prediction.model_weight,
        "seed": prediction.seed,
        "prediction_path": str(prediction.pdb_path),
        "scores_path": str(prediction.scores_path),
    }
