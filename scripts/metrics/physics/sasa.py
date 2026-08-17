"""FreeSASA-backed buried surface area using the fixed BSA = delta-SASA / 2 rule."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from types import ModuleType

from .structure import ChainRecord, DimerStructure, iter_residue_atoms


@dataclass(frozen=True)
class SasaMetrics:
    sasa_a_a2: float
    sasa_b_a2: float
    sasa_complex_a2: float
    delta_sasa_a2: float
    bsa_a2: float
    log1p_bsa_a2: float


def _load_freesasa() -> ModuleType:
    try:
        return importlib.import_module("freesasa")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FreeSASA Python bindings are required; install the dependencies "
            "declared in environment.yml"
        ) from exc


def ensure_freesasa_available() -> None:
    module = _load_freesasa()
    required = (
        "Structure",
        "Parameters",
        "LeeRichards",
        "calc",
        "setVerbosity",
        "nowarnings",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"FreeSASA module is missing API members: {missing}")
    # Structure.addAtom() may emit one element-guess warning per atom even
    # when the guess succeeds. Large model batches multiply those warnings
    # across monomer/complex structures and worker processes. Keep genuine
    # FreeSASA errors visible while suppressing warnings only.
    module.setVerbosity(module.nowarnings)


def summarize_sasa(
    sasa_a_a2: float,
    sasa_b_a2: float,
    sasa_complex_a2: float,
) -> SasaMetrics:
    values = (sasa_a_a2, sasa_b_a2, sasa_complex_a2)
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError(f"SASA values must be finite and non-negative: {values}")

    raw_delta = sasa_a_a2 + sasa_b_a2 - sasa_complex_a2
    numerical_tolerance = max(1e-6, 1e-7 * (sasa_a_a2 + sasa_b_a2))
    if raw_delta < -numerical_tolerance:
        raise ValueError(
            "Complex SASA exceeds the sum of monomer SASAs beyond numerical "
            f"tolerance: delta={raw_delta}"
        )
    delta_sasa = max(0.0, raw_delta)
    bsa = delta_sasa / 2.0
    return SasaMetrics(
        sasa_a_a2=sasa_a_a2,
        sasa_b_a2=sasa_b_a2,
        sasa_complex_a2=sasa_complex_a2,
        delta_sasa_a2=delta_sasa,
        bsa_a2=bsa,
        log1p_bsa_a2=math.log1p(bsa),
    )


def _build_freesasa_structure(
    chains: tuple[ChainRecord, ...],
    freesasa_module: ModuleType,
) -> object:
    structure = freesasa_module.Structure()
    expected_atoms = 0
    for chain in chains:
        for residue, atom in iter_residue_atoms(chain):
            x, y, z = atom.coordinate
            structure.addAtom(
                atom.name,
                residue.name,
                str(residue.sequence_index),
                chain.chain_id,
                x,
                y,
                z,
            )
            expected_atoms += 1
    if structure.nAtoms() != expected_atoms:
        raise ValueError(
            "FreeSASA skipped atoms while constructing the structure: "
            f"expected={expected_atoms}, accepted={structure.nAtoms()}"
        )
    return structure


def calculate_bsa(
    dimer: DimerStructure,
    *,
    probe_radius: float = 1.4,
    n_slices: int = 20,
    freesasa_module: ModuleType | None = None,
) -> SasaMetrics:
    """Calculate monomer/complex SASA and fixed single-side-average BSA."""
    if not math.isfinite(probe_radius) or probe_radius <= 0:
        raise ValueError("FreeSASA probe radius must be finite and positive")
    if n_slices < 1:
        raise ValueError("FreeSASA Lee-Richards slice count must be positive")

    module = freesasa_module or _load_freesasa()
    parameters = module.Parameters(
        {
            "algorithm": module.LeeRichards,
            "probe-radius": probe_radius,
            "n-slices": n_slices,
            "n-threads": 1,
        }
    )
    structure_a = _build_freesasa_structure((dimer.chain_a,), module)
    structure_b = _build_freesasa_structure((dimer.chain_b,), module)
    structure_complex = _build_freesasa_structure(
        (dimer.chain_a, dimer.chain_b), module
    )
    return summarize_sasa(
        float(module.calc(structure_a, parameters).totalArea()),
        float(module.calc(structure_b, parameters).totalArea()),
        float(module.calc(structure_complex, parameters).totalArea()),
    )
