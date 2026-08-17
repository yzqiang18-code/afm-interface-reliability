"""Auditable residue-level hydrophobic and charged interface contacts."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .contacts import ContactMetrics
from .geometry import strict_neighbor_pairs
from .structure import ChainRecord, DimerStructure, ResidueRecord


# A deliberately conservative fixed set. Cys and Pro are excluded because
# their classification is context-sensitive; His is handled as uncharged.
HYDROPHOBIC_RESIDUES = frozenset(
    {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "TYR"}
)

CHARGE_SIGN = {
    "ARG": 1,
    "LYS": 1,
    "ASP": -1,
    "GLU": -1,
}

CHARGED_ATOM_NAMES = {
    "ARG": frozenset({"NE", "NH1", "NH2"}),
    "LYS": frozenset({"NZ"}),
    "ASP": frozenset({"OD1", "OD2"}),
    "GLU": frozenset({"OE1", "OE2"}),
}


@dataclass(frozen=True)
class ChemistryMetrics:
    hydrophobic_contact_count: int
    hydrophobic_contact_fraction: float | None
    salt_bridge_count: int
    salt_bridge_density: float | None
    same_charge_contact_count: int
    same_charge_contact_density: float | None


def fraction_or_none(numerator: int, denominator: int) -> float | None:
    if numerator < 0 or denominator < 0:
        raise ValueError("Counts cannot be negative")
    if numerator > denominator:
        raise ValueError("Fraction numerator cannot exceed denominator")
    return numerator / denominator if denominator else None


def density_per_1000_a2(count: int, bsa_a2: float) -> float | None:
    if count < 0:
        raise ValueError("Count cannot be negative")
    if not math.isfinite(bsa_a2) or bsa_a2 < 0:
        raise ValueError("BSA must be finite and non-negative")
    return 1000.0 * count / bsa_a2 if bsa_a2 else None


def _residue_lookup(chain: ChainRecord) -> dict[int, ResidueRecord]:
    return {residue.sequence_index: residue for residue in chain.residues}


def _charged_atom_table(
    chain: ChainRecord,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates: list[tuple[float, float, float]] = []
    residue_indices: list[int] = []
    charge_signs: list[int] = []
    for residue in chain.residues:
        selected_names = CHARGED_ATOM_NAMES.get(residue.name)
        if selected_names is None:
            continue
        for atom in residue.atoms:
            if atom.name in selected_names:
                coordinates.append(atom.coordinate)
                residue_indices.append(residue.sequence_index)
                charge_signs.append(CHARGE_SIGN[residue.name])
    return (
        np.asarray(coordinates, dtype=float).reshape((-1, 3)),
        np.asarray(residue_indices, dtype=int),
        np.asarray(charge_signs, dtype=int),
    )


def _charged_residue_pairs(
    dimer: DimerStructure,
    contacts: ContactMetrics,
    *,
    cutoff: float,
    same_sign: bool,
) -> frozenset[tuple[int, int]]:
    coordinates_a, residue_indices_a, signs_a = _charged_atom_table(dimer.chain_a)
    coordinates_b, residue_indices_b, signs_b = _charged_atom_table(dimer.chain_b)
    atom_pairs = strict_neighbor_pairs(coordinates_a, coordinates_b, cutoff)
    return frozenset(
        (
            int(residue_indices_a[index_a]),
            int(residue_indices_b[index_b]),
        )
        for index_a, index_b in atom_pairs
        if bool(signs_a[index_a] == signs_b[index_b]) == same_sign
    ).intersection(contacts.contact_pairs)


def calculate_chemistry(
    dimer: DimerStructure,
    contacts: ContactMetrics,
    *,
    bsa_a2: float,
    salt_bridge_cutoff: float = 4.0,
    same_charge_cutoff: float = 5.0,
) -> ChemistryMetrics:
    """Calculate hydrophobic fraction and BSA-normalized charged contacts."""
    residues_a = _residue_lookup(dimer.chain_a)
    residues_b = _residue_lookup(dimer.chain_b)
    try:
        hydrophobic_count = sum(
            residues_a[residue_a].name in HYDROPHOBIC_RESIDUES
            and residues_b[residue_b].name in HYDROPHOBIC_RESIDUES
            for residue_a, residue_b in contacts.contact_pairs
        )
    except KeyError as exc:
        raise ValueError(f"Contact references an unknown residue index: {exc}") from exc

    salt_bridges = _charged_residue_pairs(
        dimer,
        contacts,
        cutoff=salt_bridge_cutoff,
        same_sign=False,
    )
    same_charge_contacts = _charged_residue_pairs(
        dimer,
        contacts,
        cutoff=same_charge_cutoff,
        same_sign=True,
    )
    return ChemistryMetrics(
        hydrophobic_contact_count=hydrophobic_count,
        hydrophobic_contact_fraction=fraction_or_none(
            hydrophobic_count,
            contacts.contact_pair_count,
        ),
        salt_bridge_count=len(salt_bridges),
        salt_bridge_density=density_per_1000_a2(len(salt_bridges), bsa_a2),
        same_charge_contact_count=len(same_charge_contacts),
        same_charge_contact_density=density_per_1000_a2(
            len(same_charge_contacts),
            bsa_a2,
        ),
    )
