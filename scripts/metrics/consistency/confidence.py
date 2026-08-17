"""Read model ipTM values and summarize their ensemble variability."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class IptmSummary:
    model_count: int
    mean: float
    population_std: float


def _numeric_iptm(value: object, *, field: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a numeric scalar in {path}")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field} must be finite and between 0 and 1 in {path}")
    return numeric


def read_iptm(path: Path) -> float:
    """Read the scalar ColabFold ipTM field from one scores JSON."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Scores JSON root must be an object: {path}")

    present = [field for field in ("iptm", "ipTM") if field in payload]
    if not present:
        raise KeyError(f"Scores JSON has no iptm field: {path}")
    values = [
        _numeric_iptm(payload[field], field=field, path=path) for field in present
    ]
    if len(values) == 2 and not math.isclose(values[0], values[1], abs_tol=1e-12):
        raise ValueError(f"iptm and ipTM disagree in {path}")
    return values[0]


def summarize_iptm(values: Sequence[float]) -> IptmSummary:
    """Return ensemble mean and population standard deviation (ddof=0)."""
    if not values:
        raise ValueError("At least one ipTM value is required")
    checked = [
        _numeric_iptm(value, field="iptm", path=Path("<in-memory>"))
        for value in values
    ]
    return IptmSummary(
        model_count=len(checked),
        mean=float(statistics.fmean(checked)),
        population_std=float(statistics.pstdev(checked)),
    )
