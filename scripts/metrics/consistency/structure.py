"""Extract metric-ready structural data from one predicted dimer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from physics.structure import parse_dimer

from .contacts import ContactMaps, calculate_contact_maps
from .pose import chain_ca_coordinates


@dataclass(frozen=True)
class StructuralModel:
    path: Path
    sequence_a: str
    sequence_b: str
    contact_maps: ContactMaps
    receptor_ca_coordinates: np.ndarray
    ligand_ca_coordinates: np.ndarray

    @property
    def chain_a_length(self) -> int:
        return len(self.sequence_a)

    @property
    def chain_b_length(self) -> int:
        return len(self.sequence_b)


def extract_structural_model(
    path: Path,
    *,
    model_chains: tuple[str, str],
    heavy_atom_cutoff: float,
    cb_cutoff: float,
) -> StructuralModel:
    """Parse one model and extract contacts plus chain-A/chain-B C-alpha arrays."""
    dimer = parse_dimer(path, model_chains)
    return StructuralModel(
        path=path,
        sequence_a=dimer.chain_a.sequence,
        sequence_b=dimer.chain_b.sequence,
        contact_maps=calculate_contact_maps(
            dimer,
            heavy_atom_cutoff=heavy_atom_cutoff,
            cb_cutoff=cb_cutoff,
        ),
        receptor_ca_coordinates=chain_ca_coordinates(dimer.chain_a),
        ligand_ca_coordinates=chain_ca_coordinates(dimer.chain_b),
    )
