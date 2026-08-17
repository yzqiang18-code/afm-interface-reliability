from __future__ import annotations

import csv
import json
import statistics
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO_ROOT / "scripts" / "metrics"
sys.path.insert(0, str(METRICS_DIR))

import run_consistency_batch as CONSISTENCY
from common import PredictionFiles, parse_prediction_path
from consistency import (
    ContactMaps,
    EnsembleMember,
    StructuralModel,
    analyze_ensemble,
    cluster_contact_sets,
    compare_contact_sets,
    contact_jaccard,
    extract_structural_model,
    receptor_aligned_ligand_rmsd,
    summarize_iptm,
)


def atom_line(
    serial: int,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    residue_number: int,
    coordinate: tuple[float, float, float],
    element: str,
) -> str:
    x, y, z = coordinate
    return (
        f"ATOM  {serial:5d} {atom_name:>4s} {residue_name:>3s} "
        f"{chain_id}{residue_number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{90.00:6.2f}"
        f"          {element:>2s}\n"
    )


def write_dimer(
    path: Path,
    *,
    residue_a: str = "ALA",
    residue_b: str = "ALA",
    chain_b_offset: float = 3.0,
    include_chain_b: bool = True,
    omit_cb_a: bool = False,
) -> None:
    lines: list[str] = []
    serial = 1

    def add_residue(
        residue_name: str,
        chain_id: str,
        offset: float,
        omit_cb: bool = False,
    ) -> None:
        nonlocal serial
        atoms = [
            ("N", (offset - 1.0, 0.0, 0.0), "N"),
            ("CA", (offset, 0.0, 0.0), "C"),
            ("C", (offset + 1.0, 0.0, 0.0), "C"),
            ("O", (offset + 1.5, 0.0, 0.0), "O"),
        ]
        if residue_name != "GLY" and not omit_cb:
            atoms.append(("CB", (offset, 1.0, 0.0), "C"))
        for atom_name, coordinate, element in atoms:
            lines.append(
                atom_line(
                    serial,
                    atom_name,
                    residue_name,
                    chain_id,
                    1,
                    coordinate,
                    element,
                )
            )
            serial += 1

    add_residue(residue_a, "A", 0.0, omit_cb_a)
    if include_chain_b:
        add_residue(residue_b, "B", chain_b_offset)
    lines.extend(["TER\n", "END\n"])
    path.write_text("".join(lines), encoding="utf-8")


def prediction_path(
    directory: Path,
    *,
    complex_id: str,
    seed: int,
    model_weight: int,
    rank: int | None = None,
) -> Path:
    actual_rank = model_weight if rank is None else rank
    return directory / (
        f"{complex_id}_unrelaxed_rank_{actual_rank:03d}_"
        f"alphafold2_multimer_v3_model_{model_weight}_seed_{seed:03d}.pdb"
    )


def write_scores(pdb_path: Path, iptm: float) -> None:
    prediction = parse_prediction_path(pdb_path)
    prediction.scores_path.write_text(
        json.dumps({"iptm": iptm}),
        encoding="utf-8",
    )


def fake_prediction(
    seed: int,
    model_weight: int,
    name: str = "synthetic",
) -> PredictionFiles:
    return parse_prediction_path(
        Path(
            f"{name}_unrelaxed_rank_{model_weight:03d}_"
            f"alphafold2_multimer_v3_model_{model_weight}_seed_{seed:03d}.pdb"
        )
    )


def ensemble_member(
    seed: int,
    model_weight: int,
    contacts: frozenset[tuple[int, int]],
    *,
    iptm: float = 0.5,
) -> EnsembleMember:
    receptor = np.asarray(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        dtype=float,
    )
    ligand = receptor + np.asarray((3.0, 0.0, 0.0))
    return EnsembleMember(
        seed=seed,
        model_weight=model_weight,
        iptm=iptm,
        structure=StructuralModel(
            path=Path(f"seed_{seed}_model_{model_weight}.pdb"),
            sequence_a="AAA",
            sequence_b="AAA",
            contact_maps=ContactMaps(heavy=contacts, cb=contacts),
            receptor_ca_coordinates=receptor,
            ligand_ca_coordinates=ligand,
        ),
    )


class ConsistencyMetricTest(unittest.TestCase):
    def test_heavy_atom_and_glycine_cb_contact_definitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = prediction_path(
                Path(tmp),
                complex_id="gly_test",
                seed=0,
                model_weight=1,
            )
            # Closest heavy atoms are exactly 5 A apart, so the strict <5 A
            # definition is empty. Gly CA to chain-B CB is <8 A.
            write_dimer(
                path,
                residue_a="GLY",
                residue_b="ALA",
                chain_b_offset=7.5,
            )
            parsed = extract_structural_model(
                path,
                model_chains=("A", "B"),
                heavy_atom_cutoff=5.0,
                cb_cutoff=8.0,
            )

        self.assertEqual(parsed.sequence_a, "G")
        self.assertEqual(parsed.contact_maps.heavy, frozenset())
        self.assertEqual(parsed.contact_maps.cb, frozenset({(1, 1)}))

    def test_contact_and_interface_residue_jaccards(self) -> None:
        first = frozenset({(1, 1), (2, 2)})
        second = frozenset({(1, 1), (3, 2)})
        comparison = compare_contact_sets(first, second)

        self.assertTrue(comparison.valid)
        self.assertAlmostEqual(float(comparison.contact_jaccard), 1 / 3)
        self.assertAlmostEqual(
            float(comparison.interface_residue_jaccard_a),
            1 / 3,
        )
        self.assertAlmostEqual(
            float(comparison.interface_residue_jaccard_b),
            1.0,
        )
        self.assertAlmostEqual(
            float(comparison.interface_residue_jaccard),
            2 / 3,
        )

    def test_empty_interfaces_are_invalid_pairs_and_noise_singletons(self) -> None:
        shared = frozenset({(1, 1), (2, 2)})
        other = frozenset({(9, 9)})
        empty: frozenset[tuple[int, int]] = frozenset()
        contact_sets = [shared, shared, other, empty]

        self.assertIsNone(contact_jaccard(empty, empty))
        cluster = cluster_contact_sets(contact_sets, 0.5)
        self.assertEqual(cluster.cluster_ids, (1, 1, 2, -1))
        self.assertAlmostEqual(cluster.max_cluster_fraction, 0.5)

        all_empty = cluster_contact_sets([empty, empty, empty], 0.5)
        self.assertEqual(all_empty.cluster_ids, (-1, -1, -1))
        self.assertAlmostEqual(all_empty.max_cluster_fraction, 1 / 3)

    def test_seed_and_model_weight_decomposition(self) -> None:
        interface_a = frozenset({(1, 1)})
        interface_b = frozenset({(2, 2)})
        models = [
            ensemble_member(0, 1, interface_a),
            ensemble_member(0, 2, interface_b),
            ensemble_member(1, 1, interface_a),
            ensemble_member(1, 2, interface_b),
        ]
        metrics = analyze_ensemble(models, cluster_distance_threshold=0.5).heavy

        self.assertEqual(metrics.valid_pair_count, 6)
        self.assertAlmostEqual(float(metrics.mean_contact_jaccard), 1 / 3)
        self.assertAlmostEqual(
            float(metrics.mean_interface_residue_jaccard),
            1 / 3,
        )
        self.assertEqual(metrics.across_seeds_pair_count, 2)
        self.assertAlmostEqual(float(metrics.mean_across_seeds), 1.0)
        self.assertEqual(metrics.across_model_weights_pair_count, 2)
        self.assertAlmostEqual(float(metrics.mean_across_model_weights), 0.0)
        self.assertAlmostEqual(metrics.max_cluster_fraction, 0.5)

    def test_receptor_aligned_ligand_rmsd(self) -> None:
        receptor = np.asarray(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            dtype=float,
        )
        ligand = np.asarray([(3.0, 0.0, 0.0), (3.0, 1.0, 0.0)])
        rotation = np.asarray(
            [(0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
        )
        translation = np.asarray((5.0, 2.0, 1.0))

        rigid_value = receptor_aligned_ligand_rmsd(
            receptor,
            ligand,
            receptor @ rotation + translation,
            ligand @ rotation + translation,
        )
        shifted_value = receptor_aligned_ligand_rmsd(
            receptor,
            ligand,
            receptor,
            ligand + np.asarray((0.0, 0.0, 2.0)),
        )

        self.assertAlmostEqual(rigid_value, 0.0)
        self.assertAlmostEqual(shifted_value, 2.0)

    def test_iptm_population_standard_deviation(self) -> None:
        summary = summarize_iptm([0.2, 0.4, 0.6])
        self.assertEqual(summary.model_count, 3)
        self.assertAlmostEqual(summary.mean, 0.4)
        self.assertAlmostEqual(
            summary.population_std,
            statistics.pstdev([0.2, 0.4, 0.6]),
        )

    def test_incomplete_and_duplicate_ensembles_are_rejected(self) -> None:
        config = CONSISTENCY.AnalysisConfig(
            model_chains=("A", "B"),
            heavy_atom_cutoff=5.0,
            cb_cutoff=8.0,
            cluster_distance_threshold=0.5,
            expected_seeds=(0, 1),
            expected_model_weights=(1, 2),
        )
        incomplete = [
            fake_prediction(0, 1),
            fake_prediction(0, 2),
            fake_prediction(1, 1),
        ]
        result = CONSISTENCY.analyze_system("synthetic", incomplete, config)
        self.assertEqual(result.summary["status"], "failed")
        self.assertEqual(result.summary["missing_model_keys"], "seed_1:model_2")

        complete = incomplete + [fake_prediction(1, 2)]
        duplicate = complete + [fake_prediction(1, 2)]
        result = CONSISTENCY.analyze_system("synthetic", duplicate, config)
        self.assertEqual(result.summary["status"], "failed")
        self.assertEqual(result.summary["duplicate_model_keys"], "seed_1:model_2")

    def test_missing_chain_and_sequence_mismatch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_chain_path = prediction_path(
                root,
                complex_id="missing",
                seed=0,
                model_weight=1,
            )
            write_dimer(missing_chain_path, include_chain_b=False)
            with self.assertRaisesRegex(ValueError, "Missing model chains"):
                extract_structural_model(
                    missing_chain_path,
                    model_chains=("A", "B"),
                    heavy_atom_cutoff=5.0,
                    cb_cutoff=8.0,
                )

            path_1 = prediction_path(
                root,
                complex_id="mismatch",
                seed=0,
                model_weight=1,
            )
            path_2 = prediction_path(
                root,
                complex_id="mismatch",
                seed=0,
                model_weight=2,
            )
            write_dimer(path_1, residue_a="ALA")
            write_dimer(path_2, residue_a="GLY")
            write_scores(path_1, 0.5)
            write_scores(path_2, 0.6)
            config = CONSISTENCY.AnalysisConfig(
                model_chains=("A", "B"),
                heavy_atom_cutoff=5.0,
                cb_cutoff=8.0,
                cluster_distance_threshold=0.5,
                expected_seeds=(0,),
                expected_model_weights=(1, 2),
            )
            result = CONSISTENCY.analyze_system(
                "mismatch",
                [parse_prediction_path(path_1), parse_prediction_path(path_2)],
                config,
            )

        self.assertEqual(result.summary["status"], "failed")
        self.assertIn("sequence mismatch", str(result.summary["error"]).lower())

    def test_missing_non_glycine_cb_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = prediction_path(
                Path(tmp),
                complex_id="missing_cb",
                seed=0,
                model_weight=1,
            )
            write_dimer(path, omit_cb_a=True)
            with self.assertRaisesRegex(ValueError, "Missing CB"):
                extract_structural_model(
                    path,
                    model_chains=("A", "B"),
                    heavy_atom_cutoff=5.0,
                    cb_cutoff=8.0,
                )

    def test_end_to_end_complete_25_model_ensemble(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction_dirs: list[Path] = []
            iptm_values: list[float] = []
            for seed in range(5):
                seed_dir = root / f"seed_{seed}"
                seed_dir.mkdir()
                prediction_dirs.append(seed_dir)
                for model_weight in range(1, 6):
                    path = prediction_path(
                        seed_dir,
                        complex_id="complete_system",
                        seed=seed,
                        model_weight=model_weight,
                    )
                    iptm = 0.50 + 0.01 * (seed * 5 + model_weight)
                    iptm_values.append(iptm)
                    write_dimer(path)
                    write_scores(path, iptm)

            summary_path = root / "summary.csv"
            model_path = root / "models.csv"
            pair_path = root / "pairs.csv"
            arguments: list[str] = []
            for directory in prediction_dirs:
                arguments.extend(["--prediction-dir", str(directory)])
            arguments.extend(
                [
                    "--output-summary-csv",
                    str(summary_path),
                    "--output-model-csv",
                    str(model_path),
                    "--output-pairwise-csv",
                    str(pair_path),
                    "--cluster-distance-threshold",
                    "0.5",
                    "--workers",
                    "2",
                ]
            )
            with (
                patch.object(
                    CONSISTENCY,
                    "ProcessPoolExecutor",
                    ThreadPoolExecutor,
                ),
                patch.object(
                    CONSISTENCY,
                    "tqdm",
                    side_effect=lambda items, **_: items,
                ) as progress,
            ):
                exit_code = CONSISTENCY.main(arguments)

            with summary_path.open(encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            with model_path.open(encoding="utf-8", newline="") as handle:
                model_rows = list(csv.DictReader(handle))
            with pair_path.open(encoding="utf-8", newline="") as handle:
                pair_rows = list(csv.DictReader(handle))

        self.assertEqual(exit_code, 0)
        self.assertEqual(progress.call_args.kwargs["total"], 1)
        self.assertEqual(progress.call_args.kwargs["unit"], "system")
        self.assertEqual(len(summary_rows), 1)
        self.assertEqual(len(model_rows), 25)
        self.assertEqual(len(pair_rows), 300)
        summary = summary_rows[0]
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["n_models"], "25")
        self.assertEqual(summary["valid_pair_count"], "300")
        self.assertEqual(summary["pose_pair_count"], "300")
        self.assertEqual(summary["iptm_model_count"], "25")
        self.assertEqual(summary["across_seeds_pair_count"], "50")
        self.assertEqual(summary["across_model_weights_pair_count"], "50")
        self.assertAlmostEqual(float(summary["mean_contact_jaccard"]), 1.0)
        self.assertAlmostEqual(
            float(summary["mean_interface_residue_jaccard"]),
            1.0,
        )
        self.assertAlmostEqual(
            float(summary["max_interface_cluster_fraction"]),
            1.0,
        )
        self.assertAlmostEqual(
            float(summary["median_receptor_aligned_ligand_rmsd"]),
            0.0,
        )
        self.assertAlmostEqual(
            float(summary["iptm_std_across_models"]),
            statistics.pstdev(iptm_values),
        )
        self.assertIn("iptm", model_rows[0])
        self.assertIn("interface_residue_jaccard", pair_rows[0])
        self.assertIn("receptor_aligned_ligand_rmsd", pair_rows[0])


if __name__ == "__main__":
    unittest.main()
