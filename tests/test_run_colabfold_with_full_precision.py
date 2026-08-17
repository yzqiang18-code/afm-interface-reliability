from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "afm"
    / "run_colabfold_with_full_precision.py"
)
SPEC = importlib.util.spec_from_file_location("full_precision_wrapper", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FullPrecisionWrapperTests(unittest.TestCase):
    def test_parse_prediction_and_rank_tags(self) -> None:
        self.assertEqual(
            MODULE.parse_prediction_tag(
                "alphafold2_multimer_v3_model_4_seed_003"
            ),
            ("alphafold2_multimer_v3", 4, 3),
        )
        self.assertEqual(
            MODULE.parse_rank_tag(
                "rank_012_alphafold2_multimer_v3_model_4_seed_003"
            ),
            (12, "alphafold2_multimer_v3_model_4_seed_003"),
        )

    def test_wrapper_captures_metrics_before_rounding_and_global_rank(self) -> None:
        tags = [
            "alphafold2_multimer_v3_model_1_seed_000",
            "alphafold2_multimer_v3_model_2_seed_001",
        ]
        values = {
            tags[0]: {
                "ranking_confidence": 0.8123456835746765,
                "iptm": 0.8234567642211914,
                "ptm": 0.7679012417793274,
            },
            tags[1]: {
                "ranking_confidence": 0.8456789255142212,
                "iptm": 0.8567890524864197,
                "ptm": 0.8012344837188721,
            },
        }

        def fake_predict_structure(
            *,
            prefix: str,
            result_dir: Path,
            prediction_callback=None,
            rank_by: str = "multimer",
        ):
            for tag in tags:
                prediction_callback(None, [100, 90], values[tag], {}, (tag, False))
            return {
                "rank": [f"rank_001_{tags[1]}", f"rank_002_{tags[0]}"],
                "metric": [],
                "result_files": [],
            }

        wrapped = MODULE.make_predict_structure_wrapper(
            fake_predict_structure,
            colabfold_version="1.5.5",
        )
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary)
            wrapped(prefix="example", result_dir=result_dir, rank_by="multimer")
            manifest_path = result_dir / "example_full_precision_ranking.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["num_predictions"], 2)
        first = payload["predictions"][0]
        self.assertEqual(first["rank"], 1)
        self.assertEqual(first["prediction_tag"], tags[1])
        self.assertEqual(first["iptm"], values[tags[1]]["iptm"])
        self.assertEqual(first["ptm"], values[tags[1]]["ptm"])
        self.assertEqual(
            first["ranking_confidence"],
            values[tags[1]]["ranking_confidence"],
        )

    def test_aggregate_writes_deterministic_full_precision_csv(self) -> None:
        rows = [
            {
                "system_id": "system_b",
                "rank": 1,
                "model_type": "alphafold2_multimer_v3",
                "model_weight": 3,
                "seed": 4,
                "prediction_tag": "alphafold2_multimer_v3_model_3_seed_004",
                "rank_tag": "rank_001_alphafold2_multimer_v3_model_3_seed_004",
                "ranking_confidence": 0.8123456835746765,
                "iptm": 0.8234567642211914,
                "ptm": 0.7679012417793274,
                "pdb_file": "/data/system_b.pdb",
                "scores_file": "/data/system_b.json",
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary)
            MODULE.write_system_manifest(
                system_id="system_b",
                result_dir=result_dir,
                rank_by="multimer",
                colabfold_version="1.5.5",
                rows=rows,
            )
            output_path, system_count, row_count = MODULE.aggregate_manifests(
                result_dir
            )
            self.assertEqual(
                output_path,
                result_dir / "full_precision_ranking.csv",
            )
            self.assertEqual(system_count, 1)
            self.assertEqual(row_count, 1)
            with output_path.open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(csv_rows[0]["iptm"], format(rows[0]["iptm"], ".17g"))
        self.assertEqual(csv_rows[0]["ptm"], format(rows[0]["ptm"], ".17g"))
        self.assertEqual(
            csv_rows[0]["ranking_confidence"],
            format(rows[0]["ranking_confidence"], ".17g"),
        )


if __name__ == "__main__":
    unittest.main()
