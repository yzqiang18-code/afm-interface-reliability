from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO_ROOT / "scripts" / "metrics"
sys.path.insert(0, str(METRICS_DIR))

import common as COMMON
import run_dockq_batch as DOCKQ
import run_pdockq2_batch as PDOCKQ2


class MetricBatchHelpersTest(unittest.TestCase):
    def test_parse_colabfold_prediction_filename(self) -> None:
        path = Path(
            "6fc2__A1_P07260--6fc2__B1_P36041_"
            "unrelaxed_rank_001_alphafold2_multimer_v3_model_4_seed_002.pdb"
        )
        parsed = COMMON.parse_prediction_path(path)
        self.assertEqual(parsed.complex_id, "6fc2__A1_P07260--6fc2__B1_P36041")
        self.assertEqual(parsed.rank, 1)
        self.assertEqual(parsed.model_weight, 4)
        self.assertEqual(parsed.seed, 2)
        self.assertIn("_scores_rank_001_", parsed.scores_path.name)

    def test_parse_ipsae_dimer_output(self) -> None:
        content = """
Chn1 Chn2 PAE Dist Type ipSAE ipSAE_d0chn ipSAE_d0dom ipTM_af ipTM_d0chn pDockQ pDockQ2 LIS
A B 15 15 asym 0.5 0.5 0.5 0.7 0.5 0.2 0.0408 0.3
B A 15 15 asym 0.6 0.6 0.6 0.7 0.6 0.2 0.0419 0.4
A B 15 15 max 0.6 0.6 0.6 0.7 0.6 0.2 0.0419 0.35
"""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "ipsae.txt"
            output.write_text(content, encoding="utf-8")
            parsed = PDOCKQ2.parse_ipsae_output(output)
        self.assertEqual(parsed["chain_1"], "A")
        self.assertEqual(parsed["chain_2"], "B")
        self.assertAlmostEqual(parsed["pDockQ2_min"], 0.0408)
        self.assertAlmostEqual(parsed["pDockQ2_max"], 0.0419)
        self.assertAlmostEqual(parsed["pDockQ2_mean"], 0.04135)

    def test_pdockq2_parallel_progress_preserves_prediction_order(self) -> None:
        predictions = [
            COMMON.PredictionFiles(
                complex_id=f"complex-{index}",
                rank=index,
                model_family="multimer",
                model_weight=index,
                seed=0,
                pdb_path=Path(f"model-{index}.pdb"),
                scores_path=Path(f"model-{index}.json"),
            )
            for index in range(3)
        ]

        with patch.object(PDOCKQ2, "tqdm", side_effect=lambda items, **_: items) as progress:
            rows = PDOCKQ2.run_batch(
                predictions,
                worker=lambda item: {"complex_id": item.complex_id},
                workers=2,
            )

        self.assertEqual(
            [row["complex_id"] for row in rows],
            ["complex-0", "complex-1", "complex-2"],
        )
        self.assertEqual(progress.call_args.kwargs["total"], 3)
        self.assertEqual(progress.call_args.kwargs["unit"], "model")

    def test_dockq_batch_reports_model_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction_dir = root / "predictions"
            native_dir = root / "native"
            prediction_dir.mkdir()
            native_dir.mkdir()
            complex_id = "synthetic"
            pdb_path = prediction_dir / (
                f"{complex_id}_unrelaxed_rank_001_"
                "alphafold2_multimer_v3_model_1_seed_000.pdb"
            )
            pdb_path.write_text("END\n", encoding="utf-8")
            COMMON.parse_prediction_path(pdb_path).scores_path.write_text(
                "{}\n",
                encoding="utf-8",
            )
            (native_dir / f"{complex_id}.pdb").write_text(
                "END\n",
                encoding="utf-8",
            )
            output = root / "dockq.csv"
            fake_dockq = SimpleNamespace(
                load_PDB=lambda *_args, **_kwargs: object(),
                run_on_all_native_interfaces=lambda *_args, **_kwargs: (
                    {"RL": {"DockQ": 0.5}},
                    None,
                ),
            )

            with (
                patch.object(DOCKQ, "load_dockq", return_value=fake_dockq),
                patch.object(
                    DOCKQ,
                    "tqdm",
                    side_effect=lambda items, **_: items,
                ) as progress,
            ):
                exit_code = DOCKQ.main(
                    [
                        "--prediction-dir",
                        str(prediction_dir),
                        "--native-dir",
                        str(native_dir),
                        "--output-csv",
                        str(output),
                    ]
                )
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(exit_code, 0)
        self.assertEqual(rows[0]["DockQ"], "0.5")
        self.assertEqual(progress.call_args.kwargs["total"], 1)
        self.assertEqual(progress.call_args.kwargs["unit"], "model")

    def test_dockq_parallel_progress_preserves_prediction_order(self) -> None:
        predictions = [
            COMMON.PredictionFiles(
                complex_id=f"complex-{index}",
                rank=index,
                model_family="multimer",
                model_weight=index,
                seed=0,
                pdb_path=Path(f"model-{index}.pdb"),
                scores_path=Path(f"model-{index}.json"),
            )
            for index in range(3)
        ]

        class FakeFuture:
            def __init__(self, row: dict[str, object]) -> None:
                self.row = row

            def result(self) -> dict[str, object]:
                return self.row

        pool = MagicMock()
        pool.submit.side_effect = lambda _worker, task: FakeFuture(
            {"complex_id": task[0].complex_id}
        )
        executor = MagicMock()
        executor.__enter__.return_value = pool
        dockq_source_dir = Path("dockq-source")

        with (
            patch.object(DOCKQ, "ProcessPoolExecutor", return_value=executor) as factory,
            patch.object(
                DOCKQ,
                "as_completed",
                side_effect=lambda futures: reversed(list(futures)),
            ),
            patch.object(DOCKQ, "tqdm", side_effect=lambda items, **_: items) as progress,
        ):
            rows = DOCKQ.run_batch(
                predictions,
                native_dir=Path("native"),
                model_chains="AB",
                native_chains="RL",
                dockq_source_dir=dockq_source_dir,
                workers=2,
            )

        self.assertEqual(
            [row["complex_id"] for row in rows],
            ["complex-0", "complex-1", "complex-2"],
        )
        factory.assert_called_once_with(
            max_workers=2,
            initializer=DOCKQ.initialize_dockq_worker,
            initargs=(dockq_source_dir,),
        )
        self.assertEqual(progress.call_args.kwargs["total"], 3)
        self.assertEqual(progress.call_args.kwargs["unit"], "model")

    def test_dockq_symmetry_aware_homodimer_selects_swapped_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction_dir = root / "predictions"
            native_dir = root / "native"
            prediction_dir.mkdir()
            native_dir.mkdir()
            complex_id = "1abc__A1_P12345--1abc__B1_P12345"
            pdb_path = prediction_dir / (
                f"{complex_id}_unrelaxed_rank_001_"
                "alphafold2_multimer_v3_model_1_seed_000.pdb"
            )
            pdb_path.write_text("END\n", encoding="utf-8")
            COMMON.parse_prediction_path(pdb_path).scores_path.write_text(
                "{}\n",
                encoding="utf-8",
            )
            (native_dir / f"{complex_id}.pdb").write_text(
                "END\n",
                encoding="utf-8",
            )

            def fake_run(*_args, chain_map, **_kwargs):
                score = 0.8 if chain_map == {"R": "B", "L": "A"} else 0.2
                return {"RL": {"DockQ": score}}, score

            output = root / "dockq.csv"
            fake_dockq = SimpleNamespace(
                load_PDB=lambda *_args, **_kwargs: object(),
                run_on_all_native_interfaces=fake_run,
            )
            with (
                patch.object(DOCKQ, "load_dockq", return_value=fake_dockq),
                patch.object(DOCKQ, "tqdm", side_effect=lambda items, **_: items),
            ):
                exit_code = DOCKQ.main(
                    [
                        "--prediction-dir",
                        str(prediction_dir),
                        "--native-dir",
                        str(native_dir),
                        "--output-csv",
                        str(output),
                        "--symmetry-aware-homodimers",
                    ]
                )
            with output.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(exit_code, 0)
        self.assertEqual(row["mapping_mode"], "symmetry_aware")
        self.assertEqual(row["selected_model_chains"], "BA")
        self.assertEqual(row["direct_DockQ"], "0.2")
        self.assertEqual(row["swapped_DockQ"], "0.8")
        self.assertAlmostEqual(float(row["symmetry_gain"]), 0.6)
        self.assertEqual(row["DockQ"], "0.8")

    def test_dockq_undefined_accessions_never_swap(self) -> None:
        self.assertFalse(
            DOCKQ.pinder_same_known_uniprot(
                "1abc__A1_UNDEFINED--1abc__B1_UNDEFINED"
            )
        )
        self.assertFalse(
            DOCKQ.pinder_same_known_uniprot(
                "1abc__A1_P12345--1abc__B1_UNDEFINED"
            )
        )
        self.assertTrue(
            DOCKQ.pinder_same_known_uniprot(
                "1abc__A1_P12345--1abc__B1_P12345"
            )
        )

    def test_dockq_parallel_process_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction_dir = root / "predictions"
            native_dir = root / "native"
            dockq_source_dir = root / "dockq"
            prediction_dir.mkdir()
            native_dir.mkdir()
            dockq_source_dir.mkdir()
            (dockq_source_dir / "DockQ.py").write_text(
                """
def load_PDB(path, chains):
    return path

def run_on_all_native_interfaces(model, native, chain_map, low_memory):
    return {"RL": {"DockQ": 0.75, "F1": 0.5}}, None
""".lstrip(),
                encoding="utf-8",
            )

            for index in range(2):
                complex_id = f"synthetic-{index}"
                pdb_path = prediction_dir / (
                    f"{complex_id}_unrelaxed_rank_00{index + 1}_"
                    f"alphafold2_multimer_v3_model_{index + 1}_seed_000.pdb"
                )
                pdb_path.write_text("END\n", encoding="utf-8")
                COMMON.parse_prediction_path(pdb_path).scores_path.write_text(
                    "{}\n",
                    encoding="utf-8",
                )
                (native_dir / f"{complex_id}.pdb").write_text(
                    "END\n",
                    encoding="utf-8",
                )

            output = root / "dockq.csv"
            exit_code = DOCKQ.main(
                [
                    "--prediction-dir",
                    str(prediction_dir),
                    "--native-dir",
                    str(native_dir),
                    "--output-csv",
                    str(output),
                    "--dockq-source-dir",
                    str(dockq_source_dir),
                    "--workers",
                    "2",
                ]
            )
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [row["complex_id"] for row in rows],
            ["synthetic-0", "synthetic-1"],
        )
        self.assertTrue(all(row["status"] == "ok" for row in rows))
        self.assertTrue(all(row["DockQ"] == "0.75" for row in rows))


if __name__ == "__main__":
    unittest.main()
