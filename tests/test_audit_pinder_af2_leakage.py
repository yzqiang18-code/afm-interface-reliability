from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


try:
    import pandas  # noqa: F401
except ModuleNotFoundError:
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = object
    sys.modules["pandas"] = pandas_stub


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "pinder_af2"
    / "selection"
    / "audit_pinder_af2_leakage.py"
)
SPEC = importlib.util.spec_from_file_location("audit_pinder_af2_leakage", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class FakeFrame:
    def __init__(self, rows: list[types.SimpleNamespace]) -> None:
        self.rows = rows

    def itertuples(self, index: bool = False):
        del index
        return iter(self.rows)

    def __getitem__(self, name: str):
        class Values:
            def __init__(self, values: list[object]) -> None:
                self.values = values

            def astype(self, value: object):
                del value
                return set(map(str, self.values))

        return Values([getattr(row, name) for row in self.rows])


class LeakageAuditTests(unittest.TestCase):
    def test_pairs_are_direction_independent(self) -> None:
        self.assertEqual(audit.unordered_pair("R", "L"), audit.unordered_pair("L", "R"))

    def test_undefined_uniprot_pair_is_excluded(self) -> None:
        row = types.SimpleNamespace(
            id="id1",
            pdb_id="pdb1",
            cluster_id="c1",
            cluster_id_R="r1",
            cluster_id_L="l1",
            uniprot_R="UNDEFINED",
            uniprot_L="UNDEFINED",
        )
        result = audit.keys(FakeFrame([row]))
        self.assertEqual(result["uniprot_pair"], set())
        self.assertEqual(result["chain_cluster_pair"], {"l1|r1"})


if __name__ == "__main__":
    unittest.main()
