"""Receptor-aligned ligand pose consistency using C-alpha atoms."""

from __future__ import annotations

import math

import numpy as np

from physics.structure import ChainRecord


def chain_ca_coordinates(chain: ChainRecord) -> np.ndarray:
    """Return one C-alpha coordinate per resolved standard residue."""
    coordinates: list[tuple[float, float, float]] = []
    for residue in chain.residues:
        selected = [atom for atom in residue.atoms if atom.name == "CA"]
        if len(selected) != 1:
            raise ValueError(
                f"Expected one CA in chain {chain.chain_id} residue "
                f"{residue.pdb_residue_id} ({residue.name}), got {len(selected)}"
            )
        coordinates.append(selected[0].coordinate)
    return np.asarray(coordinates, dtype=float).reshape((-1, 3))


def _validate_coordinates(name: str, coordinates: np.ndarray) -> None:
    if coordinates.ndim != 2 or coordinates.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (N, 3)")
    if len(coordinates) == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{name} must contain only finite coordinates")


def _fit_transform(
    moving: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the proper rotation and translation mapping moving to reference."""
    _validate_coordinates("moving receptor coordinates", moving)
    _validate_coordinates("reference receptor coordinates", reference)
    if moving.shape != reference.shape:
        raise ValueError(
            "Receptor coordinate arrays must have identical shapes: "
            f"{moving.shape} != {reference.shape}"
        )

    moving_centroid = np.mean(moving, axis=0)
    reference_centroid = np.mean(reference, axis=0)
    moving_centered = moving - moving_centroid
    reference_centered = reference - reference_centroid
    covariance = moving_centered.T @ reference_centered
    left, _singular_values, right_transpose = np.linalg.svd(covariance)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_transpose
    translation = reference_centroid - moving_centroid @ rotation
    return rotation, translation


def receptor_aligned_ligand_rmsd(
    receptor_reference: np.ndarray,
    ligand_reference: np.ndarray,
    receptor_moving: np.ndarray,
    ligand_moving: np.ndarray,
) -> float:
    """Align moving receptor to reference, then calculate ligand C-alpha RMSD."""
    _validate_coordinates("ligand reference coordinates", ligand_reference)
    _validate_coordinates("ligand moving coordinates", ligand_moving)
    if ligand_reference.shape != ligand_moving.shape:
        raise ValueError(
            "Ligand coordinate arrays must have identical shapes: "
            f"{ligand_reference.shape} != {ligand_moving.shape}"
        )

    rotation, translation = _fit_transform(receptor_moving, receptor_reference)
    aligned_ligand = ligand_moving @ rotation + translation
    squared_distances = np.sum(
        np.square(aligned_ligand - ligand_reference),
        axis=1,
    )
    value = float(np.sqrt(np.mean(squared_distances)))
    if not math.isfinite(value):
        raise ValueError("Receptor-aligned ligand RMSD is not finite")
    return value
