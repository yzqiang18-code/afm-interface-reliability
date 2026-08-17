from __future__ import annotations

import csv
import importlib.util
import math
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO_ROOT / "scripts" / "metrics"
sys.path.insert(0, str(METRICS_DIR))

import run_physics_batch as PHYSICS
from common import parse_prediction_path
from physics.chemistry import calculate_chemistry
from physics.clashes import calculate_clashes, compute_clash_density
from physics.contacts import ContactMetrics, calculate_contacts, compute_contact_density
from physics.interface import calculate_interface_metrics
from physics.sasa import (
    SasaMetrics,
    calculate_bsa,
    ensure_freesasa_available,
    summarize_sasa,
)
from physics.structure import (
    AtomRecord,
    ChainRecord,
    DimerStructure,
    ResidueRecord,
    parse_dimer,
)


def atom_line(
    serial: int,
    chain_id: str,
    coordinate: tuple[float, float, float],
) -> str:
    x, y, z = coordinate
    return (
        f"ATOM  {serial:5d} {'CA':>4s} {'ALA':>3s} "
        f"{chain_id}{1:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{90.00:6.2f}"
        f"          {'C':>2s}\n"
    )


def write_two_atom_dimer(path: Path, distance: float) -> None:
    path.write_text(
        atom_line(1, "A", (0.0, 0.0, 0.0))
        + atom_line(2, "B", (distance, 0.0, 0.0))
        + "TER\nEND\n",
        encoding="utf-8",
    )


def prediction_path(directory: Path, complex_id: str = "synthetic") -> Path:
    return directory / (
        f"{complex_id}_unrelaxed_rank_001_"
        "alphafold2_multimer_v3_model_1_seed_000.pdb"
    )


def residue(
    sequence_index: int,
    residue_name: str,
    atoms: tuple[tuple[str, tuple[float, float, float]], ...],
) -> ResidueRecord:
    return ResidueRecord(
        sequence_index=sequence_index,
        pdb_residue_id=str(sequence_index),
        name=residue_name,
        atoms=tuple(
            AtomRecord(
                name=atom_name,
                element=atom_name[0],
                coordinate=coordinate,
            )
            for atom_name, coordinate in atoms
        ),
    )


def chemical_test_dimer() -> DimerStructure:
    return DimerStructure(
        path=Path("chemical_test.pdb"),
        chain_a=ChainRecord(
            chain_id="A",
            residues=(
                residue(1, "LEU", (("CD1", (0.0, 0.0, 0.0)),)),
                residue(2, "LYS", (("NZ", (0.0, 10.0, 0.0)),)),
                residue(
                    3,
                    "ASP",
                    (
                        ("OD1", (0.0, 20.0, 0.0)),
                        ("OD2", (0.0, 20.5, 0.0)),
                    ),
                ),
            ),
        ),
        chain_b=ChainRecord(
            chain_id="B",
            residues=(
                residue(1, "VAL", (("CG1", (3.0, 0.0, 0.0)),)),
                residue(
                    2,
                    "GLU",
                    (
                        ("OE1", (3.5, 10.0, 0.0)),
                        ("OE2", (3.5, 10.5, 0.0)),
                    ),
                ),
                residue(
                    3,
                    "GLU",
                    (
                        ("OE1", (3.5, 20.0, 0.0)),
                        ("OE2", (3.5, 20.5, 0.0)),
                    ),
                ),
            ),
        ),
    )


class FakeResult:
    def __init__(self, total_area: float) -> None:
        self.total_area = total_area

    def totalArea(self) -> float:
        return self.total_area


class FakeStructure:
    def __init__(self) -> None:
        self.atom_count = 0

    def addAtom(self, *_args: object) -> None:
        self.atom_count += 1

    def nAtoms(self) -> int:
        return self.atom_count


class FakeFreeSASA:
    LeeRichards = "LeeRichards"
    Structure = FakeStructure
    nowarnings = "nowarnings"
    verbosity = None

    class Parameters:
        def __init__(self, values: dict[str, object]) -> None:
            self.values = values

    @staticmethod
    def calc(structure: FakeStructure, _parameters: object) -> FakeResult:
        # Each one-atom monomer has area 10, while the two-atom complex has
        # area 16. Thus delta-SASA=4 and fixed BSA=2.
        return FakeResult(10.0 if structure.nAtoms() == 1 else 16.0)

    @classmethod
    def setVerbosity(cls, verbosity: object) -> None:
        cls.verbosity = verbosity


class PhysicsMetricsTest(unittest.TestCase):
    def test_freesasa_availability_suppresses_warnings_not_errors(self) -> None:
        FakeFreeSASA.verbosity = None
        with patch("physics.sasa._load_freesasa", return_value=FakeFreeSASA):
            ensure_freesasa_available()
        self.assertEqual(FakeFreeSASA.verbosity, FakeFreeSASA.nowarnings)

    def test_contact_and_clash_density_formulas(self) -> None:
        self.assertAlmostEqual(compute_contact_density(3, 2, 2), 1.5)
        self.assertEqual(compute_contact_density(0, 0, 0), 0.0)
        self.assertAlmostEqual(compute_clash_density(2, 10), 20.0)
        self.assertIsNone(compute_clash_density(0, 0))

    def test_strict_cutoffs_and_interface_atom_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            close_path = root / "close.pdb"
            write_two_atom_dimer(close_path, 1.5)
            close_dimer = parse_dimer(close_path, ("A", "B"))
            contacts = calculate_contacts(close_dimer, cutoff=5.0)
            clashes = calculate_clashes(close_dimer, contacts, cutoff=2.0)

            boundary_path = root / "boundary.pdb"
            write_two_atom_dimer(boundary_path, 5.0)
            boundary_dimer = parse_dimer(boundary_path, ("A", "B"))
            boundary_contacts = calculate_contacts(
                boundary_dimer,
                cutoff=5.0,
            )
            boundary_clashes = calculate_clashes(
                boundary_dimer,
                boundary_contacts,
                cutoff=2.0,
            )

        self.assertEqual(contacts.contact_pairs, frozenset({(1, 1)}))
        self.assertAlmostEqual(contacts.contact_density, 1.0)
        self.assertEqual(clashes.clash_count, 1)
        self.assertEqual(clashes.interface_heavy_atom_count, 2)
        self.assertAlmostEqual(float(clashes.clash_density), 50.0)
        self.assertEqual(boundary_contacts.contact_pairs, frozenset())
        self.assertIsNone(boundary_clashes.clash_density)

    def test_bsa_definition_and_log_transform(self) -> None:
        metrics = summarize_sasa(100.0, 80.0, 140.0)
        self.assertAlmostEqual(metrics.delta_sasa_a2, 40.0)
        self.assertAlmostEqual(metrics.bsa_a2, 20.0)
        self.assertAlmostEqual(metrics.log1p_bsa_a2, math.log1p(20.0))

        with self.assertRaisesRegex(ValueError, "exceeds"):
            summarize_sasa(10.0, 10.0, 30.0)

    def test_interface_size_topology_and_asymmetry_metrics(self) -> None:
        contacts = ContactMetrics(
            contact_pairs=frozenset({(1, 1), (2, 1), (3, 3)}),
            interface_residues_a=frozenset({1, 2, 3}),
            interface_residues_b=frozenset({1, 3}),
            contact_density=6 / 5,
        )
        metrics = calculate_interface_metrics(contacts, bsa_a2=100.0)

        self.assertAlmostEqual(float(metrics.bsa_per_interface_residue), 20.0)
        self.assertEqual(metrics.contact_component_count, 2)
        self.assertAlmostEqual(
            float(metrics.largest_contact_component_fraction),
            3 / 5,
        )
        self.assertAlmostEqual(float(metrics.interface_contact_asymmetry), 1 / 5)

        empty = calculate_interface_metrics(
            ContactMetrics(
                contact_pairs=frozenset(),
                interface_residues_a=frozenset(),
                interface_residues_b=frozenset(),
                contact_density=0.0,
            ),
            bsa_a2=0.0,
        )
        self.assertEqual(empty.contact_component_count, 0)
        self.assertIsNone(empty.bsa_per_interface_residue)
        self.assertIsNone(empty.largest_contact_component_fraction)
        self.assertIsNone(empty.interface_contact_asymmetry)

    def test_hydrophobic_salt_bridge_and_same_charge_metrics(self) -> None:
        dimer = chemical_test_dimer()
        contacts = ContactMetrics(
            contact_pairs=frozenset({(1, 1), (2, 2), (3, 3)}),
            interface_residues_a=frozenset({1, 2, 3}),
            interface_residues_b=frozenset({1, 2, 3}),
            contact_density=1.0,
        )
        metrics = calculate_chemistry(
            dimer,
            contacts,
            bsa_a2=100.0,
            salt_bridge_cutoff=4.0,
            same_charge_cutoff=5.0,
        )

        self.assertEqual(metrics.hydrophobic_contact_count, 1)
        self.assertAlmostEqual(float(metrics.hydrophobic_contact_fraction), 1 / 3)
        self.assertEqual(metrics.salt_bridge_count, 1)
        self.assertAlmostEqual(float(metrics.salt_bridge_density), 10.0)
        self.assertEqual(metrics.same_charge_contact_count, 1)
        self.assertAlmostEqual(float(metrics.same_charge_contact_density), 10.0)

        strict_boundary = calculate_chemistry(
            dimer,
            contacts,
            bsa_a2=100.0,
            salt_bridge_cutoff=3.5,
            same_charge_cutoff=3.5,
        )
        self.assertEqual(strict_boundary.salt_bridge_count, 0)
        self.assertEqual(strict_boundary.same_charge_contact_count, 0)

        zero_bsa = calculate_chemistry(
            dimer,
            contacts,
            bsa_a2=0.0,
        )
        self.assertIsNone(zero_bsa.salt_bridge_density)
        self.assertIsNone(zero_bsa.same_charge_contact_density)

    def test_freesasa_adapter_builds_two_monomers_and_one_complex(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dimer.pdb"
            write_two_atom_dimer(path, 3.0)
            dimer = parse_dimer(path, ("A", "B"))
            metrics = calculate_bsa(
                dimer,
                freesasa_module=FakeFreeSASA,
            )

        self.assertAlmostEqual(metrics.sasa_a_a2, 10.0)
        self.assertAlmostEqual(metrics.sasa_b_a2, 10.0)
        self.assertAlmostEqual(metrics.sasa_complex_a2, 16.0)
        self.assertAlmostEqual(metrics.bsa_a2, 2.0)
        self.assertAlmostEqual(metrics.log1p_bsa_a2, math.log(3.0))

    @unittest.skipUnless(
        importlib.util.find_spec("freesasa") is not None,
        "FreeSASA is not installed in this local test environment",
    )
    def test_installed_freesasa_api_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dimer.pdb"
            write_two_atom_dimer(path, 3.0)
            ensure_freesasa_available()
            metrics = calculate_bsa(parse_dimer(path, ("A", "B")))

        self.assertGreaterEqual(metrics.delta_sasa_a2, 0.0)
        self.assertAlmostEqual(metrics.bsa_a2, metrics.delta_sasa_a2 / 2.0)
        self.assertAlmostEqual(metrics.log1p_bsa_a2, math.log1p(metrics.bsa_a2))

    @unittest.skipUnless(
        importlib.util.find_spec("freesasa") is not None,
        "FreeSASA is not installed in this local test environment",
    )
    def test_installed_freesasa_batch_suppresses_element_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = prediction_path(root)
            write_two_atom_dimer(prediction, 3.0)
            output = root / "physics.csv"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(METRICS_DIR / "run_physics_batch.py"),
                    "--prediction-dir",
                    str(root),
                    "--output-csv",
                    str(output),
                    "--workers",
                    "1",
                    "--max-models",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertNotIn("guessing that atom", completed.stderr)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "ok")

    def test_batch_csv_does_not_require_scores_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = prediction_path(root)
            write_two_atom_dimer(prediction, 1.5)
            output = root / "physics.csv"
            fake_sasa = SasaMetrics(
                sasa_a_a2=10.0,
                sasa_b_a2=10.0,
                sasa_complex_a2=16.0,
                delta_sasa_a2=4.0,
                bsa_a2=2.0,
                log1p_bsa_a2=math.log(3.0),
            )
            with (
                patch.object(
                    PHYSICS,
                    "ensure_freesasa_available",
                ) as ensure_freesasa,
                patch.object(PHYSICS, "calculate_bsa", return_value=fake_sasa),
                patch.object(
                    PHYSICS,
                    "ProcessPoolExecutor",
                    ThreadPoolExecutor,
                ),
                patch.object(
                    PHYSICS,
                    "tqdm",
                    side_effect=lambda items, **_: items,
                ) as progress,
            ):
                exit_code = PHYSICS.main(
                    [
                        "--prediction-dir",
                        str(root),
                        "--output-csv",
                        str(output),
                        "--workers",
                        "2",
                    ]
                )
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(exit_code, 0)
        # Once in the parent before discovery and once in the worker
        # initializer. This keeps warning suppression reliable for both
        # fork- and spawn-based process pools.
        self.assertGreaterEqual(ensure_freesasa.call_count, 2)
        self.assertEqual(progress.call_args.kwargs["total"], 1)
        self.assertEqual(progress.call_args.kwargs["unit"], "model")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["contact_pair_count"], "1")
        self.assertAlmostEqual(float(rows[0]["contact_density"]), 1.0)
        self.assertAlmostEqual(float(rows[0]["bsa_per_interface_residue"]), 1.0)
        self.assertEqual(rows[0]["contact_component_count"], "1")
        self.assertAlmostEqual(
            float(rows[0]["largest_contact_component_fraction"]),
            1.0,
        )
        self.assertAlmostEqual(float(rows[0]["interface_contact_asymmetry"]), 0.0)
        self.assertAlmostEqual(float(rows[0]["clash_density"]), 50.0)
        self.assertAlmostEqual(float(rows[0]["log1p_bsa_a2"]), math.log(3.0))
        self.assertAlmostEqual(float(rows[0]["hydrophobic_contact_fraction"]), 1.0)
        self.assertAlmostEqual(float(rows[0]["salt_bridge_density"]), 0.0)
        self.assertAlmostEqual(float(rows[0]["same_charge_contact_density"]), 0.0)
        self.assertTrue(rows[0]["scores_path"].endswith(".json"))


if __name__ == "__main__":
    unittest.main()
