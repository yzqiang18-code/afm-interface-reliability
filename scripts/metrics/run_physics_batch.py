#!/usr/bin/env python3
"""Calculate modular physical-plausibility metrics for AF-M dimer models."""

from __future__ import annotations

import argparse
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

# Keep numerical libraries conservative by default on the shared server. An
# explicitly supplied process environment still takes precedence.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

from common import (  # noqa: E402
    PredictionFiles,
    discover_predictions,
    prediction_metadata,
    read_ids,
)
from physics import (  # noqa: E402
    calculate_bsa,
    calculate_chemistry,
    calculate_clashes,
    calculate_contacts,
    calculate_interface_metrics,
    parse_dimer,
)
from physics.chemistry import HYDROPHOBIC_RESIDUES  # noqa: E402
from physics.sasa import ensure_freesasa_available  # noqa: E402


FIELDNAMES = [
    "complex_id",
    "rank",
    "model_family",
    "model_weight",
    "seed",
    "chain_a_length",
    "chain_b_length",
    "contact_pair_count",
    "interface_residue_count_a",
    "interface_residue_count_b",
    "interface_residue_count_total",
    "contact_density",
    "bsa_per_interface_residue",
    "contact_component_count",
    "largest_contact_component_fraction",
    "interface_contact_asymmetry",
    "clash_count",
    "backbone_backbone_clash_count",
    "interface_heavy_atom_count_a",
    "interface_heavy_atom_count_b",
    "interface_heavy_atom_count_total",
    "clash_density",
    "sasa_a_a2",
    "sasa_b_a2",
    "sasa_complex_a2",
    "delta_sasa_a2",
    "bsa_a2",
    "log1p_bsa_a2",
    "hydrophobic_contact_count",
    "hydrophobic_contact_fraction",
    "salt_bridge_count",
    "salt_bridge_density",
    "same_charge_contact_count",
    "same_charge_contact_density",
    "interface_status",
    "contact_cutoff_angstrom",
    "clash_cutoff_angstrom",
    "salt_bridge_cutoff_angstrom",
    "same_charge_cutoff_angstrom",
    "hydrophobic_residue_set",
    "sasa_probe_radius_angstrom",
    "sasa_algorithm",
    "sasa_n_slices",
    "bsa_definition",
    "model_chains",
    "prediction_path",
    "scores_path",
    "status",
    "error",
]


@dataclass(frozen=True)
class PhysicsConfig:
    model_chains: tuple[str, str]
    contact_cutoff: float
    clash_cutoff: float
    salt_bridge_cutoff: float
    same_charge_cutoff: float
    sasa_probe_radius: float
    sasa_n_slices: int


def _empty_row(prediction: PredictionFiles, config: PhysicsConfig) -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in FIELDNAMES}
    row.update(prediction_metadata(prediction))
    row.update(
        {
            "contact_cutoff_angstrom": config.contact_cutoff,
            "clash_cutoff_angstrom": config.clash_cutoff,
            "salt_bridge_cutoff_angstrom": config.salt_bridge_cutoff,
            "same_charge_cutoff_angstrom": config.same_charge_cutoff,
            "hydrophobic_residue_set": ";".join(sorted(HYDROPHOBIC_RESIDUES)),
            "sasa_probe_radius_angstrom": config.sasa_probe_radius,
            "sasa_algorithm": "LeeRichards",
            "sasa_n_slices": config.sasa_n_slices,
            "bsa_definition": "(SASA_A+SASA_B-SASA_AB)/2",
            "model_chains": "".join(config.model_chains),
            "status": "failed",
            "error": "",
        }
    )
    return row


def run_one(
    prediction: PredictionFiles,
    config: PhysicsConfig,
) -> dict[str, object]:
    row = _empty_row(prediction, config)
    try:
        dimer = parse_dimer(prediction.pdb_path, config.model_chains)
        contacts = calculate_contacts(dimer, cutoff=config.contact_cutoff)
        clashes = calculate_clashes(dimer, contacts, cutoff=config.clash_cutoff)
        sasa = calculate_bsa(
            dimer,
            probe_radius=config.sasa_probe_radius,
            n_slices=config.sasa_n_slices,
        )
        interface = calculate_interface_metrics(
            contacts,
            bsa_a2=sasa.bsa_a2,
        )
        chemistry = calculate_chemistry(
            dimer,
            contacts,
            bsa_a2=sasa.bsa_a2,
            salt_bridge_cutoff=config.salt_bridge_cutoff,
            same_charge_cutoff=config.same_charge_cutoff,
        )
        interface_count_a = len(contacts.interface_residues_a)
        interface_count_b = len(contacts.interface_residues_b)
        row.update(
            {
                "chain_a_length": len(dimer.chain_a.residues),
                "chain_b_length": len(dimer.chain_b.residues),
                "contact_pair_count": contacts.contact_pair_count,
                "interface_residue_count_a": interface_count_a,
                "interface_residue_count_b": interface_count_b,
                "interface_residue_count_total": (
                    interface_count_a + interface_count_b
                ),
                "contact_density": contacts.contact_density,
                "bsa_per_interface_residue": (
                    interface.bsa_per_interface_residue
                ),
                "contact_component_count": interface.contact_component_count,
                "largest_contact_component_fraction": (
                    interface.largest_contact_component_fraction
                ),
                "interface_contact_asymmetry": (
                    interface.interface_contact_asymmetry
                ),
                "clash_count": clashes.clash_count,
                "backbone_backbone_clash_count": (
                    clashes.backbone_backbone_clash_count
                ),
                "interface_heavy_atom_count_a": (
                    clashes.interface_heavy_atom_count_a
                ),
                "interface_heavy_atom_count_b": (
                    clashes.interface_heavy_atom_count_b
                ),
                "interface_heavy_atom_count_total": (
                    clashes.interface_heavy_atom_count
                ),
                "clash_density": clashes.clash_density,
                "sasa_a_a2": sasa.sasa_a_a2,
                "sasa_b_a2": sasa.sasa_b_a2,
                "sasa_complex_a2": sasa.sasa_complex_a2,
                "delta_sasa_a2": sasa.delta_sasa_a2,
                "bsa_a2": sasa.bsa_a2,
                "log1p_bsa_a2": sasa.log1p_bsa_a2,
                "hydrophobic_contact_count": (
                    chemistry.hydrophobic_contact_count
                ),
                "hydrophobic_contact_fraction": (
                    chemistry.hydrophobic_contact_fraction
                ),
                "salt_bridge_count": chemistry.salt_bridge_count,
                "salt_bridge_density": chemistry.salt_bridge_density,
                "same_charge_contact_count": (
                    chemistry.same_charge_contact_count
                ),
                "same_charge_contact_density": (
                    chemistry.same_charge_contact_density
                ),
                "interface_status": (
                    "ok" if contacts.contact_pair_count else "empty_interface"
                ),
                "status": "ok",
                "error": "",
            }
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def run_task(task: tuple[PredictionFiles, PhysicsConfig]) -> dict[str, object]:
    return run_one(*task)


def initialize_physics_worker() -> None:
    """Validate FreeSASA and suppress its warning stream in each worker."""
    ensure_freesasa_available()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        action="append",
        required=True,
        help="May be repeated to process several seed directories into one CSV.",
    )
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--model-chains", default="AB")
    parser.add_argument("--contact-cutoff", type=float, default=5.0)
    parser.add_argument("--clash-cutoff", type=float, default=2.0)
    parser.add_argument("--salt-bridge-cutoff", type=float, default=4.0)
    parser.add_argument("--same-charge-cutoff", type=float, default=5.0)
    parser.add_argument("--sasa-probe-radius", type=float, default=1.4)
    parser.add_argument("--sasa-n-slices", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-models", type=int)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> PhysicsConfig:
    if len(args.model_chains) != 2 or len(set(args.model_chains)) != 2:
        raise ValueError("--model-chains must contain exactly two different chain IDs")
    for name, value in (
        ("--contact-cutoff", args.contact_cutoff),
        ("--clash-cutoff", args.clash_cutoff),
        ("--salt-bridge-cutoff", args.salt_bridge_cutoff),
        ("--same-charge-cutoff", args.same_charge_cutoff),
        ("--sasa-probe-radius", args.sasa_probe_radius),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if args.clash_cutoff > args.contact_cutoff:
        raise ValueError("--clash-cutoff cannot exceed --contact-cutoff")
    if args.salt_bridge_cutoff > args.contact_cutoff:
        raise ValueError("--salt-bridge-cutoff cannot exceed --contact-cutoff")
    if args.same_charge_cutoff > args.contact_cutoff:
        raise ValueError("--same-charge-cutoff cannot exceed --contact-cutoff")
    if args.sasa_n_slices < 1:
        raise ValueError("--sasa-n-slices must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.max_models is not None and args.max_models < 1:
        raise ValueError("--max-models must be at least 1")
    return PhysicsConfig(
        model_chains=(args.model_chains[0], args.model_chains[1]),
        contact_cutoff=args.contact_cutoff,
        clash_cutoff=args.clash_cutoff,
        salt_bridge_cutoff=args.salt_bridge_cutoff,
        same_charge_cutoff=args.same_charge_cutoff,
        sasa_probe_radius=args.sasa_probe_radius,
        sasa_n_slices=args.sasa_n_slices,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = validate_args(args)
    ensure_freesasa_available()
    allowed_ids = read_ids(args.ids_file)

    predictions: list[PredictionFiles] = []
    for prediction_dir in args.prediction_dir:
        predictions.extend(
            discover_predictions(
                prediction_dir,
                allowed_ids=allowed_ids,
                require_scores=False,
            )
        )
    predictions.sort(
        key=lambda item: (
            item.complex_id,
            item.seed,
            item.model_weight,
            item.rank,
            str(item.pdb_path),
        )
    )
    if len({item.pdb_path.resolve() for item in predictions}) != len(predictions):
        raise ValueError("The same prediction PDB was discovered more than once")
    if args.max_models is not None:
        predictions = predictions[: args.max_models]

    tasks = [(prediction, config) for prediction in predictions]
    if args.workers == 1:
        rows = [
            run_task(task)
            for task in tqdm(
                tasks,
                total=len(tasks),
                desc="Physics",
                unit="model",
            )
        ]
    else:
        rows_by_index: list[dict[str, object] | None] = [None] * len(tasks)
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_physics_worker,
        ) as pool:
            future_to_index = {
                pool.submit(run_task, task): index
                for index, task in enumerate(tasks)
            }
            for future in tqdm(
                as_completed(future_to_index),
                total=len(future_to_index),
                desc="Physics",
                unit="model",
            ):
                rows_by_index[future_to_index[future]] = future.result()
        if any(row is None for row in rows_by_index):
            raise RuntimeError("Not all physics tasks produced a result")
        rows = [row for row in rows_by_index if row is not None]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    failures = sum(row["status"] != "ok" for row in rows)
    empty_interfaces = sum(row["interface_status"] == "empty_interface" for row in rows)
    print(
        f"Physics metrics: {len(rows) - failures} succeeded, {failures} failed, "
        f"{empty_interfaces} empty interfaces; output={args.output_csv}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
