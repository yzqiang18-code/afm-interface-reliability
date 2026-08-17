"""Interface-size normalization, contact-graph topology, and side asymmetry."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .contacts import ContactMetrics


GraphNode = tuple[str, int]


@dataclass(frozen=True)
class InterfaceMetrics:
    bsa_per_interface_residue: float | None
    contact_component_count: int
    largest_contact_component_fraction: float | None
    interface_contact_asymmetry: float | None


def _connected_component_sizes(
    contacts: frozenset[tuple[int, int]],
) -> tuple[int, ...]:
    adjacency: dict[GraphNode, set[GraphNode]] = {}
    for residue_a, residue_b in contacts:
        node_a = ("A", residue_a)
        node_b = ("B", residue_b)
        adjacency.setdefault(node_a, set()).add(node_b)
        adjacency.setdefault(node_b, set()).add(node_a)

    component_sizes: list[int] = []
    unvisited = set(adjacency)
    while unvisited:
        seed = unvisited.pop()
        stack = [seed]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in adjacency[node]:
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    stack.append(neighbor)
        component_sizes.append(size)
    return tuple(sorted(component_sizes, reverse=True))


def calculate_interface_metrics(
    contacts: ContactMetrics,
    *,
    bsa_a2: float,
) -> InterfaceMetrics:
    """Calculate the three interface-derived metrics requested by the project."""
    if not math.isfinite(bsa_a2) or bsa_a2 < 0:
        raise ValueError("BSA must be finite and non-negative")

    derived_a = frozenset(first for first, _second in contacts.contact_pairs)
    derived_b = frozenset(second for _first, second in contacts.contact_pairs)
    if (
        derived_a != contacts.interface_residues_a
        or derived_b != contacts.interface_residues_b
    ):
        raise ValueError("Interface residue sets disagree with contact-pair endpoints")

    residue_count_a = len(contacts.interface_residues_a)
    residue_count_b = len(contacts.interface_residues_b)
    total_residue_count = residue_count_a + residue_count_b
    if total_residue_count == 0:
        return InterfaceMetrics(
            bsa_per_interface_residue=None,
            contact_component_count=0,
            largest_contact_component_fraction=None,
            interface_contact_asymmetry=None,
        )

    component_sizes = _connected_component_sizes(contacts.contact_pairs)
    if not component_sizes or sum(component_sizes) != total_residue_count:
        raise ValueError("Contact graph does not cover all interface residues")
    return InterfaceMetrics(
        bsa_per_interface_residue=bsa_a2 / total_residue_count,
        contact_component_count=len(component_sizes),
        largest_contact_component_fraction=(
            component_sizes[0] / total_residue_count
        ),
        interface_contact_asymmetry=(
            abs(residue_count_a - residue_count_b) / total_residue_count
        ),
    )
