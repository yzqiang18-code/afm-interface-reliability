"""Inter-chain heavy-atom clashes normalized by interface atom count."""

from __future__ import annotations

from dataclasses import dataclass

from .contacts import ContactMetrics
from .geometry import strict_neighbor_pairs
from .structure import DimerStructure, atom_table, iter_residue_atoms


BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})


@dataclass(frozen=True)
class ClashMetrics:
    clash_count: int
    backbone_backbone_clash_count: int
    interface_heavy_atom_count_a: int
    interface_heavy_atom_count_b: int
    clash_density: float | None

    @property
    def interface_heavy_atom_count(self) -> int:
        return self.interface_heavy_atom_count_a + self.interface_heavy_atom_count_b


def compute_clash_density(
    clash_count: int,
    interface_heavy_atom_count: int,
) -> float | None:
    """Return clashes per 100 interface heavy atoms, or None for no interface."""
    if clash_count < 0 or interface_heavy_atom_count < 0:
        raise ValueError("Clash and atom counts cannot be negative")
    if interface_heavy_atom_count == 0:
        if clash_count:
            raise ValueError("Clashes cannot be normalized without interface atoms")
        return None
    return 100.0 * clash_count / interface_heavy_atom_count


def calculate_clashes(
    dimer: DimerStructure,
    contacts: ContactMetrics,
    *,
    cutoff: float = 2.0,
) -> ClashMetrics:
    """Count unique inter-chain heavy-atom pairs at a strict distance cutoff."""
    atoms_a, coordinates_a, _residue_indices_a = atom_table(dimer.chain_a)
    atoms_b, coordinates_b, _residue_indices_b = atom_table(dimer.chain_b)
    atom_pairs = strict_neighbor_pairs(coordinates_a, coordinates_b, cutoff)
    backbone_count = sum(
        atoms_a[index_a].name in BACKBONE_ATOM_NAMES
        and atoms_b[index_b].name in BACKBONE_ATOM_NAMES
        for index_a, index_b in atom_pairs
    )
    interface_atom_count_a = sum(
        1
        for _residue, _atom in iter_residue_atoms(
            dimer.chain_a, contacts.interface_residues_a
        )
    )
    interface_atom_count_b = sum(
        1
        for _residue, _atom in iter_residue_atoms(
            dimer.chain_b, contacts.interface_residues_b
        )
    )
    interface_atom_count = interface_atom_count_a + interface_atom_count_b
    return ClashMetrics(
        clash_count=len(atom_pairs),
        backbone_backbone_clash_count=backbone_count,
        interface_heavy_atom_count_a=interface_atom_count_a,
        interface_heavy_atom_count_b=interface_atom_count_b,
        clash_density=compute_clash_density(len(atom_pairs), interface_atom_count),
    )
