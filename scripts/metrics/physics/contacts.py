"""Inter-chain heavy-atom residue contacts and contact density."""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import strict_neighbor_pairs
from .structure import DimerStructure, atom_table


ContactPair = tuple[int, int]


@dataclass(frozen=True)
class ContactMetrics:
    contact_pairs: frozenset[ContactPair]
    interface_residues_a: frozenset[int]
    interface_residues_b: frozenset[int]
    contact_density: float

    @property
    def contact_pair_count(self) -> int:
        return len(self.contact_pairs)


def compute_contact_density(
    contact_pair_count: int,
    interface_residue_count_a: int,
    interface_residue_count_b: int,
) -> float:
    """Calculate 2*C/(N_A+N_B), with an empty interface defined as zero."""
    if min(
        contact_pair_count,
        interface_residue_count_a,
        interface_residue_count_b,
    ) < 0:
        raise ValueError("Contact counts cannot be negative")
    denominator = interface_residue_count_a + interface_residue_count_b
    if denominator == 0:
        if contact_pair_count:
            raise ValueError("Contacts cannot exist without interface residues")
        return 0.0
    return 2.0 * contact_pair_count / denominator


def calculate_contacts(
    dimer: DimerStructure,
    *,
    cutoff: float = 5.0,
) -> ContactMetrics:
    """Calculate residue contacts from any inter-chain heavy-atom pair at < cutoff."""
    _atoms_a, coordinates_a, residue_indices_a = atom_table(dimer.chain_a)
    _atoms_b, coordinates_b, residue_indices_b = atom_table(dimer.chain_b)
    atom_pairs = strict_neighbor_pairs(coordinates_a, coordinates_b, cutoff)
    contacts = frozenset(
        (
            int(residue_indices_a[index_a]),
            int(residue_indices_b[index_b]),
        )
        for index_a, index_b in atom_pairs
    )
    interface_a = frozenset(first for first, _second in contacts)
    interface_b = frozenset(second for _first, second in contacts)
    return ContactMetrics(
        contact_pairs=contacts,
        interface_residues_a=interface_a,
        interface_residues_b=interface_b,
        contact_density=compute_contact_density(
            len(contacts),
            len(interface_a),
            len(interface_b),
        ),
    )
