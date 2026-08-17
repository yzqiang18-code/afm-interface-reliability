"""Read a two-chain protein PDB into a small, metric-independent data model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from Bio.PDB import PDBParser


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


@dataclass(frozen=True)
class AtomRecord:
    """One selected heavy atom from a standard protein residue."""

    name: str
    element: str
    coordinate: tuple[float, float, float]


@dataclass(frozen=True)
class ResidueRecord:
    """A standard protein residue indexed by resolved sequence position."""

    sequence_index: int
    pdb_residue_id: str
    name: str
    atoms: tuple[AtomRecord, ...]


@dataclass(frozen=True)
class ChainRecord:
    chain_id: str
    residues: tuple[ResidueRecord, ...]

    @property
    def sequence(self) -> str:
        return "".join(AA3_TO_1[residue.name] for residue in self.residues)

    @property
    def heavy_atom_count(self) -> int:
        return sum(len(residue.atoms) for residue in self.residues)


@dataclass(frozen=True)
class DimerStructure:
    path: Path
    chain_a: ChainRecord
    chain_b: ChainRecord


def _atom_element(atom: object) -> str:
    element = str(getattr(atom, "element", "") or "").strip().upper()
    if element:
        return element
    letters = [character for character in atom.get_name().strip() if character.isalpha()]
    return letters[0].upper() if letters else ""


def _pdb_residue_id(residue: object) -> str:
    _hetero_flag, residue_number, insertion_code = residue.id
    return f"{residue_number}{str(insertion_code).strip()}"


def _read_chain(chain: object, chain_id: str, path: Path) -> ChainRecord:
    residues: list[ResidueRecord] = []
    for residue in chain.get_residues():
        residue_name = residue.get_resname().strip().upper()
        if residue.id[0] != " " or residue_name not in AA3_TO_1:
            continue

        atoms: list[AtomRecord] = []
        for atom in residue.get_atoms():
            element = _atom_element(atom)
            if element in {"H", "D"}:
                continue
            coordinate = tuple(float(value) for value in atom.coord)
            if len(coordinate) != 3 or not all(math.isfinite(value) for value in coordinate):
                raise ValueError(
                    f"Non-finite coordinate in chain {chain_id} residue "
                    f"{_pdb_residue_id(residue)}: {path}"
                )
            atoms.append(
                AtomRecord(
                    name=atom.get_name().strip().upper(),
                    element=element,
                    coordinate=coordinate,
                )
            )

        if not atoms:
            raise ValueError(
                f"No heavy atoms in chain {chain_id} residue "
                f"{_pdb_residue_id(residue)} ({residue_name}): {path}"
            )
        residues.append(
            ResidueRecord(
                sequence_index=len(residues) + 1,
                pdb_residue_id=_pdb_residue_id(residue),
                name=residue_name,
                atoms=tuple(atoms),
            )
        )

    if not residues:
        raise ValueError(f"No standard protein residues found in chain {chain_id}: {path}")
    return ChainRecord(chain_id=chain_id, residues=tuple(residues))


def parse_dimer(path: Path, model_chains: tuple[str, str]) -> DimerStructure:
    """Parse the first and only PDB model and select two named protein chains."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(f"Expected exactly one PDB MODEL in {path}, got {len(models)}")

    chain_lookup = {chain.id: chain for chain in models[0].get_chains()}
    missing = [chain_id for chain_id in model_chains if chain_id not in chain_lookup]
    if missing:
        raise ValueError(
            f"Missing model chains {missing} in {path}; found {sorted(chain_lookup)}"
        )

    chain_a_id, chain_b_id = model_chains
    return DimerStructure(
        path=path,
        chain_a=_read_chain(chain_lookup[chain_a_id], chain_a_id, path),
        chain_b=_read_chain(chain_lookup[chain_b_id], chain_b_id, path),
    )


def iter_residue_atoms(
    chain: ChainRecord,
    residue_indices: frozenset[int] | None = None,
) -> Iterable[tuple[ResidueRecord, AtomRecord]]:
    for residue in chain.residues:
        if residue_indices is not None and residue.sequence_index not in residue_indices:
            continue
        for atom in residue.atoms:
            yield residue, atom


def atom_table(
    chain: ChainRecord,
) -> tuple[tuple[AtomRecord, ...], np.ndarray, np.ndarray]:
    """Return atoms, Nx3 coordinates, and their 1-based sequence positions."""
    atoms: list[AtomRecord] = []
    coordinates: list[tuple[float, float, float]] = []
    residue_indices: list[int] = []
    for residue, atom in iter_residue_atoms(chain):
        atoms.append(atom)
        coordinates.append(atom.coordinate)
        residue_indices.append(residue.sequence_index)
    return (
        tuple(atoms),
        np.asarray(coordinates, dtype=float).reshape((-1, 3)),
        np.asarray(residue_indices, dtype=int),
    )
