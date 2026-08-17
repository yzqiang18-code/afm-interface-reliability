#!/usr/bin/env python3
"""Calculate full-precision DockQ labels for PINDER AF-M dimer predictions."""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Sequence

from tqdm import tqdm

from common import (
    PredictionFiles,
    discover_predictions,
    prediction_metadata,
    read_ids,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCKQ_DIR = REPO_ROOT / "third_party" / "dockq" / "src" / "DockQ"
FIELDNAMES = [
    "complex_id",
    "rank",
    "model_family",
    "model_weight",
    "seed",
    "DockQ",
    "DockQ_F1",
    "iRMSD",
    "LRMSD",
    "fnat",
    "fnonnat",
    "F1",
    "clashes",
    "model_chains",
    "native_chains",
    "mapping_mode",
    "same_known_uniprot",
    "chain_exchange_eligible",
    "selected_model_chains",
    "direct_DockQ",
    "swapped_DockQ",
    "symmetry_gain",
    "prediction_path",
    "scores_path",
    "native_path",
    "status",
    "error",
]

_WORKER_DOCKQ: ModuleType | None = None


def load_dockq(dockq_dir: Path) -> ModuleType:
    if not (dockq_dir / "DockQ.py").is_file():
        raise FileNotFoundError(dockq_dir / "DockQ.py")
    sys.path.insert(0, str(dockq_dir))
    return importlib.import_module("DockQ")


def clear_dockq_caches(dockq: ModuleType) -> None:
    for name in (
        "get_aligned_residues",
        "get_residue_distances",
        "align_chains",
        "list_atoms_per_residue",
        "subset_atoms",
        "run_on_chains",
    ):
        function = getattr(dockq, name, None)
        if function is not None and hasattr(function, "cache_clear"):
            function.cache_clear()


def scalar(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    return value


UNDEFINED_ACCESSIONS = frozenset({"", "UNDEFINED", "NONE", "NAN", "NA"})


def pinder_accessions(complex_id: str) -> tuple[str, str]:
    """Parse receptor and ligand accession tokens from a PINDER dimer ID."""

    try:
        receptor, ligand = complex_id.split("--", 1)
        receptor_uniprot = receptor.rsplit("_", 1)[1]
        ligand_uniprot = ligand.rsplit("_", 1)[1]
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unsupported PINDER complex ID: {complex_id}") from exc
    return receptor_uniprot, ligand_uniprot


def pinder_same_known_uniprot(complex_id: str) -> bool:
    """Return True only when both accession tokens are known and identical."""

    receptor_uniprot, ligand_uniprot = pinder_accessions(complex_id)
    receptor_normalized = receptor_uniprot.strip().upper()
    ligand_normalized = ligand_uniprot.strip().upper()
    if (
        receptor_normalized in UNDEFINED_ACCESSIONS
        or ligand_normalized in UNDEFINED_ACCESSIONS
    ):
        return False
    return receptor_normalized == ligand_normalized


def evaluate_mapping(
    dockq: ModuleType,
    model_structure: object,
    native_structure: object,
    *,
    model_chains: str,
    native_chains: str,
) -> dict[str, object]:
    chain_map = dict(zip(native_chains, model_chains))
    result, _total = dockq.run_on_all_native_interfaces(
        model_structure,
        native_structure,
        chain_map=chain_map,
        low_memory=False,
    )
    if native_chains not in result:
        raise ValueError(
            f"DockQ did not return native interface {native_chains}: "
            f"{sorted(result)}"
        )
    return result[native_chains]


def initialize_dockq_worker(dockq_source_dir: Path) -> None:
    """Load one independent DockQ module instance in each worker process."""

    global _WORKER_DOCKQ
    _WORKER_DOCKQ = load_dockq(dockq_source_dir)


def run_one(
    prediction: PredictionFiles,
    *,
    native_dir: Path,
    model_chains: str,
    native_chains: str,
    symmetry_aware_homodimers: bool,
    dockq: ModuleType,
) -> dict[str, object]:
    native_path = native_dir / f"{prediction.complex_id}.pdb"
    same_known_uniprot = (
        pinder_same_known_uniprot(prediction.complex_id)
        if symmetry_aware_homodimers
        else False
    )
    row = {
        **prediction_metadata(prediction),
        "DockQ": "",
        "DockQ_F1": "",
        "iRMSD": "",
        "LRMSD": "",
        "fnat": "",
        "fnonnat": "",
        "F1": "",
        "clashes": "",
        "model_chains": model_chains,
        "native_chains": native_chains,
        "mapping_mode": "fixed",
        "same_known_uniprot": same_known_uniprot,
        "chain_exchange_eligible": same_known_uniprot,
        "selected_model_chains": model_chains,
        "direct_DockQ": "",
        "swapped_DockQ": "",
        "symmetry_gain": "",
        "native_path": str(native_path),
        "status": "failed",
        "error": "",
    }
    try:
        if not native_path.is_file():
            raise FileNotFoundError(native_path)
        model_structure = dockq.load_PDB(
            str(prediction.pdb_path), chains=list(model_chains)
        )
        native_structure = dockq.load_PDB(
            str(native_path), chains=list(native_chains)
        )
        direct_metrics = evaluate_mapping(
            dockq,
            model_structure,
            native_structure,
            model_chains=model_chains,
            native_chains=native_chains,
        )
        metrics = direct_metrics
        direct_dockq = float(direct_metrics["DockQ"])
        row["direct_DockQ"] = direct_dockq
        row["symmetry_gain"] = 0.0

        if symmetry_aware_homodimers and same_known_uniprot:
            swapped_model_chains = model_chains[::-1]
            clear_dockq_caches(dockq)
            swapped_metrics = evaluate_mapping(
                dockq,
                model_structure,
                native_structure,
                model_chains=swapped_model_chains,
                native_chains=native_chains,
            )
            swapped_dockq = float(swapped_metrics["DockQ"])
            row["mapping_mode"] = "symmetry_aware"
            row["swapped_DockQ"] = swapped_dockq
            if swapped_dockq > direct_dockq:
                metrics = swapped_metrics
                row["selected_model_chains"] = swapped_model_chains
            row["symmetry_gain"] = max(0.0, swapped_dockq - direct_dockq)

        for field in (
            "DockQ",
            "DockQ_F1",
            "iRMSD",
            "LRMSD",
            "fnat",
            "fnonnat",
            "F1",
            "clashes",
        ):
            row[field] = scalar(metrics.get(field, ""))
        row["status"] = "ok"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        clear_dockq_caches(dockq)
    return row


def run_one_in_worker(
    task: tuple[PredictionFiles, Path, str, str, bool],
) -> dict[str, object]:
    if _WORKER_DOCKQ is None:
        raise RuntimeError("DockQ worker was not initialized")
    (
        prediction,
        native_dir,
        model_chains,
        native_chains,
        symmetry_aware_homodimers,
    ) = task
    return run_one(
        prediction,
        native_dir=native_dir,
        model_chains=model_chains,
        native_chains=native_chains,
        symmetry_aware_homodimers=symmetry_aware_homodimers,
        dockq=_WORKER_DOCKQ,
    )


def run_batch(
    predictions: list[PredictionFiles],
    *,
    native_dir: Path,
    model_chains: str,
    native_chains: str,
    dockq_source_dir: Path,
    workers: int,
    symmetry_aware_homodimers: bool = False,
) -> list[dict[str, object]]:
    if workers == 1:
        dockq = load_dockq(dockq_source_dir)
        return [
            run_one(
                prediction,
                native_dir=native_dir,
                model_chains=model_chains,
                native_chains=native_chains,
                symmetry_aware_homodimers=symmetry_aware_homodimers,
                dockq=dockq,
            )
            for prediction in tqdm(
                predictions,
                total=len(predictions),
                desc="DockQ",
                unit="model",
            )
        ]

    tasks = [
        (
            prediction,
            native_dir,
            model_chains,
            native_chains,
            symmetry_aware_homodimers,
        )
        for prediction in predictions
    ]
    rows_by_index: list[dict[str, object] | None] = [None] * len(tasks)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_dockq_worker,
        initargs=(dockq_source_dir,),
    ) as pool:
        future_to_index = {
            pool.submit(run_one_in_worker, task): index
            for index, task in enumerate(tasks)
        }
        for future in tqdm(
            as_completed(future_to_index),
            total=len(future_to_index),
            desc="DockQ",
            unit="model",
        ):
            rows_by_index[future_to_index[future]] = future.result()

    if any(row is None for row in rows_by_index):
        raise RuntimeError("Not all DockQ tasks produced a result")
    return [row for row in rows_by_index if row is not None]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--dockq-source-dir", type=Path, default=DEFAULT_DOCKQ_DIR)
    parser.add_argument("--model-chains", default="AB")
    parser.add_argument("--native-chains", default="RL")
    parser.add_argument(
        "--symmetry-aware-homodimers",
        action="store_true",
        help=(
            "For PINDER IDs with identical known receptor/ligand UniProt accessions, "
            "evaluate both direct and chain-swapped dimer mappings and keep the "
            "mapping with higher DockQ. Missing/UNDEFINED accessions always use "
            "the fixed mapping."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent DockQ worker processes (default: 1).",
    )
    parser.add_argument("--max-models", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if len(args.model_chains) != len(args.native_chains):
        raise ValueError("--model-chains and --native-chains must have equal length")
    if len(args.model_chains) != 2:
        raise ValueError("This project adapter currently supports dimers only")
    if not args.native_dir.is_dir():
        raise NotADirectoryError(args.native_dir)
    if not (args.dockq_source_dir / "DockQ.py").is_file():
        raise FileNotFoundError(args.dockq_source_dir / "DockQ.py")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    predictions = discover_predictions(
        args.prediction_dir,
        allowed_ids=read_ids(args.ids_file),
        max_models=args.max_models,
    )
    rows = run_batch(
        predictions,
        native_dir=args.native_dir,
        model_chains=args.model_chains,
        native_chains=args.native_chains,
        dockq_source_dir=args.dockq_source_dir,
        workers=args.workers,
        symmetry_aware_homodimers=args.symmetry_aware_homodimers,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    failures = sum(row["status"] != "ok" for row in rows)
    symmetry_aware = sum(
        row["mapping_mode"] == "symmetry_aware" for row in rows
    )
    print(
        f"DockQ: {len(rows) - failures} succeeded, {failures} failed; "
        f"symmetry-aware homodimer models={symmetry_aware}; "
        f"output={args.output_csv}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
