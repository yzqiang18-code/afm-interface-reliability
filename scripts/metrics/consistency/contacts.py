"""Contact maps and pairwise contact/interface-residue similarities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics.contacts import calculate_contacts
from physics.geometry import strict_neighbor_pairs
from physics.structure import ChainRecord, DimerStructure


Contact = tuple[int, int]


@dataclass(frozen=True)
class ContactMaps:
    """The two contact-map definitions used by the consistency analysis."""

    heavy: frozenset[Contact]
    cb: frozenset[Contact]


@dataclass(frozen=True)
class ContactComparison:
    """Pairwise similarities for one contact-map definition."""

    contact_jaccard: float | None
    interface_residue_jaccard_a: float | None
    interface_residue_jaccard_b: float | None
    interface_residue_jaccard: float | None
    valid: bool
    reason: str


def interface_residue_sets(
    contacts: frozenset[Contact],
) -> tuple[frozenset[int], frozenset[int]]:
    """Return the chain-A and chain-B contact endpoints."""
    return (
        frozenset(residue_a for residue_a, _residue_b in contacts),
        frozenset(residue_b for _residue_a, residue_b in contacts),
    )


def contact_jaccard(
    contacts_a: frozenset[Contact],
    contacts_b: frozenset[Contact],
) -> float | None:
    """Return Jaccard for two non-empty interfaces; empty interfaces are invalid."""
    if not contacts_a or not contacts_b:
        return None
    return len(contacts_a & contacts_b) / len(contacts_a | contacts_b)


def _set_jaccard(values_a: frozenset[int], values_b: frozenset[int]) -> float:
    if not values_a or not values_b:
        raise ValueError("Interface-residue Jaccard requires two non-empty sets")
    return len(values_a & values_b) / len(values_a | values_b)


def _empty_pair_reason(
    contacts_a: frozenset[Contact],
    contacts_b: frozenset[Contact],
) -> str:
    if not contacts_a and not contacts_b:
        return "both_empty"
    if not contacts_a:
        return "model_1_empty"
    if not contacts_b:
        return "model_2_empty"
    return ""


def compare_contact_sets(
    contacts_a: frozenset[Contact],
    contacts_b: frozenset[Contact],
) -> ContactComparison:
    """Compare contact pairs and the two chains' interface-residue sets."""
    contact_value = contact_jaccard(contacts_a, contacts_b)
    if contact_value is None:
        return ContactComparison(
            contact_jaccard=None,
            interface_residue_jaccard_a=None,
            interface_residue_jaccard_b=None,
            interface_residue_jaccard=None,
            valid=False,
            reason=_empty_pair_reason(contacts_a, contacts_b),
        )

    residues_a_1, residues_b_1 = interface_residue_sets(contacts_a)
    residues_a_2, residues_b_2 = interface_residue_sets(contacts_b)
    value_a = _set_jaccard(residues_a_1, residues_a_2)
    value_b = _set_jaccard(residues_b_1, residues_b_2)
    return ContactComparison(
        contact_jaccard=contact_value,
        interface_residue_jaccard_a=value_a,
        interface_residue_jaccard_b=value_b,
        interface_residue_jaccard=(value_a + value_b) / 2.0,
        valid=True,
        reason="",
    )


def _representative_points(chain: ChainRecord) -> tuple[np.ndarray, np.ndarray]:
    coordinates: list[tuple[float, float, float]] = []
    residue_indices: list[int] = []
    for residue in chain.residues:
        atom_name = "CA" if residue.name == "GLY" else "CB"
        selected = [atom for atom in residue.atoms if atom.name == atom_name]
        if not selected:
            raise ValueError(
                f"Missing {atom_name} for chain {chain.chain_id} residue "
                f"{residue.pdb_residue_id} ({residue.name})"
            )
        if len(selected) > 1:
            raise ValueError(
                f"Expected one {atom_name} in chain {chain.chain_id} residue "
                f"{residue.pdb_residue_id} ({residue.name}), got {len(selected)}"
            )
        coordinates.append(selected[0].coordinate)
        residue_indices.append(residue.sequence_index)
    return (
        np.asarray(coordinates, dtype=float).reshape((-1, 3)),
        np.asarray(residue_indices, dtype=int),
    )


def _calculate_cb_contacts(
    dimer: DimerStructure,
    *,
    cutoff: float,
) -> frozenset[Contact]:
    coordinates_a, residue_indices_a = _representative_points(dimer.chain_a)
    coordinates_b, residue_indices_b = _representative_points(dimer.chain_b)
    atom_pairs = strict_neighbor_pairs(coordinates_a, coordinates_b, cutoff)
    return frozenset(
        (
            int(residue_indices_a[index_a]),
            int(residue_indices_b[index_b]),
        )
        for index_a, index_b in atom_pairs
    )


def calculate_contact_maps(
    dimer: DimerStructure,
    *,
    heavy_atom_cutoff: float = 5.0,
    cb_cutoff: float = 8.0,
) -> ContactMaps:
    """Calculate heavy-atom and C-beta (Gly C-alpha) residue contact maps."""
    heavy = calculate_contacts(dimer, cutoff=heavy_atom_cutoff).contact_pairs
    cb = _calculate_cb_contacts(dimer, cutoff=cb_cutoff)
    return ContactMaps(heavy=heavy, cb=cb)
