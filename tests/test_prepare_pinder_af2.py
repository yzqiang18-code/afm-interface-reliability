from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


try:
    import pandas  # noqa: F401
    HAS_PANDAS = True
except ModuleNotFoundError:
    HAS_PANDAS = False
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = object
    sys.modules["pandas"] = pandas_stub

try:
    import requests  # noqa: F401
except ModuleNotFoundError:
    requests_stub = types.ModuleType("requests")

    class RequestException(Exception):
        pass

    class HTTPError(RequestException):
        def __init__(self, *args: object, response: object | None = None) -> None:
            super().__init__(*args)
            self.response = response

    class Session:
        pass

    requests_stub.RequestException = RequestException
    requests_stub.HTTPError = HTTPError
    requests_stub.Session = Session
    sys.modules["requests"] = requests_stub

try:
    import Bio  # noqa: F401
    HAS_BIOPYTHON = True
except ModuleNotFoundError:
    HAS_BIOPYTHON = False
    bio_stub = types.ModuleType("Bio")
    pdb_stub = types.ModuleType("Bio.PDB")
    sequtils_stub = types.ModuleType("Bio.SeqUtils")

    class PDBParser:
        pass

    pdb_stub.PDBParser = PDBParser
    sequtils_stub.seq1 = lambda value, undef_code="X": value
    sys.modules["Bio"] = bio_stub
    sys.modules["Bio.PDB"] = pdb_stub
    sys.modules["Bio.SeqUtils"] = sequtils_stub

try:
    import tqdm  # noqa: F401
except ModuleNotFoundError:
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable=None, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_stub


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "pinder_af2"
    / "data_prep"
    / "prepare_pinder_af2.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_pinder_af2", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
af2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = af2
SPEC.loader.exec_module(af2)


class FakeManifest:
    def itertuples(self, index: bool = False):
        del index
        return iter(
            [
                types.SimpleNamespace(
                    id="example",
                    native_pdb="example.pdb",
                    holo_R_pdb="example-R.pdb",
                    holo_L_pdb="example-L.pdb",
                    mapping_R="example-R.parquet",
                    mapping_L="example-L.parquet",
                )
            ]
        )


class PinderAF2PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not HAS_PANDAS or not HAS_BIOPYTHON:
            sys.stderr.write(
                "NOTE: local unit tests use import stubs; run the documented "
                "manifest/verify/fastas integration checks in the server data "
                "environment before submitting MSA.\n"
            )

    def test_frozen_selection_contains_180_unique_ids(self) -> None:
        identifiers = af2.load_selection_ids(af2.DEFAULT_SELECTION_FILE)
        self.assertEqual(len(identifiers), 180)
        self.assertEqual(len(set(identifiers)), 180)
        self.assertEqual(
            identifiers[0],
            "7rzb__A1_A0A229LVN5--7rzb__A2_A0A229LVN5",
        )
        self.assertEqual(
            identifiers[-1],
            "8hci__B1_A0A8T8BZJ9--8hci__A1_A0A8T8BZN3",
        )

    def test_selection_rejects_wrong_size_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ids.txt"
            path.write_text("same\nsame\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                af2.load_selection_ids(path)
            path.write_text("one\ntwo\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Expected 180"):
                af2.load_selection_ids(path)

    def test_inventory_uses_test_set_holo_directory(self) -> None:
        items = af2.build_inventory(FakeManifest(), include_mappings=True)
        by_kind = {item.kind: item.relative_path for item in items}
        self.assertEqual(by_kind["native"], "pdbs/example.pdb")
        self.assertEqual(
            by_kind["holo_R"], "test_set_pdbs/example-R.pdb"
        )
        self.assertEqual(
            by_kind["holo_L"], "test_set_pdbs/example-L.pdb"
        )
        self.assertEqual(by_kind["mapping_R"], "mappings/example-R.parquet")

    def test_manifest_directory_is_isolated_from_val(self) -> None:
        self.assertEqual(
            af2.manifest_dir(Path("/data/2024-02")),
            Path("/data/2024-02/manifests/pinder_af2"),
        )

    def test_existing_fasta_must_match_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "example.fasta"
            path.write_text(">example\nAAA:BBB\n", encoding="ascii")
            af2.expected_text(path, ">example\nAAA:BBB\n", overwrite=False)
            with self.assertRaisesRegex(ValueError, "differs"):
                af2.expected_text(path, ">example\nCCC:DDD\n", overwrite=False)

    def test_length_policy_is_strictly_less_than_1500(self) -> None:
        self.assertEqual(af2.LENGTH_CUTOFF, 1500)
        self.assertTrue(1499 < af2.LENGTH_CUTOFF)
        self.assertFalse(1500 < af2.LENGTH_CUTOFF)

    def test_split_id_lists_are_plain_and_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ids.txt"
            af2.write_id_list(path, ["first", "second"])
            self.assertEqual(path.read_text(), "first\nsecond\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                af2.write_id_list(path, ["same", "same"])

    def test_download_rejects_nonempty_file_with_wrong_remote_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            path = data_root / "pdbs" / "example.pdb"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"truncated")
            original = af2.remote_content_length
            af2.remote_content_length = lambda *args, **kwargs: 100
            try:
                result = af2.download_one(
                    af2.DownloadItem("native", "pdbs/example.pdb", "example"),
                    data_root=data_root,
                    base_url=af2.DEFAULT_BASE_URL,
                    timeout=1,
                    retries=0,
                )
            finally:
                af2.remote_content_length = original
            self.assertEqual(result.status, "failed")
            self.assertIn("differs from remote", result.error)
            self.assertEqual(path.read_bytes(), b"truncated")

    @unittest.skipUnless(HAS_BIOPYTHON, "Biopython is required for real PDB parsing")
    def test_real_pdb_fixtures_enforce_side_and_extract_sequence(self) -> None:
        fixture_dir = REPO_ROOT / "tests" / "fixtures" / "pinder_af2"
        self.assertEqual(
            af2.sequence_from_holo(fixture_dir / "holo_R.pdb", "R"),
            ("AG", "R"),
        )
        self.assertEqual(
            af2.sequence_from_holo(fixture_dir / "holo_L.pdb", "L"),
            ("S", "L"),
        )
        with self.assertRaisesRegex(ValueError, "expected PINDER side"):
            af2.sequence_from_holo(fixture_dir / "holo_R.pdb", "L")


if __name__ == "__main__":
    unittest.main()
