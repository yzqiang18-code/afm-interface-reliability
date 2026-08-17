from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path


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
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

    requests_stub.RequestException = RequestException
    requests_stub.HTTPError = HTTPError
    requests_stub.Session = Session
    requests_stub.exceptions = types.SimpleNamespace(
        RequestException=RequestException,
    )
    sys.modules["requests"] = requests_stub

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
    / "pinder_val"
    / "msa"
    / "build_colabfold_msas.py"
)
SPEC = importlib.util.spec_from_file_location("build_colabfold_msas", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
msa = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = msa
SPEC.loader.exec_module(msa)


def make_tar(members: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, text in members.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def unpaired_archive() -> bytes:
    return make_tar(
        {
            "uniref.a3m": (
                ">101\nAAA\n>uniref_R\nA-A\n"
                "\x00>102\nBBBB\n>uniref_L\nBB-B\n"
            ),
            "bfd.mgnify30.metaeuk30.smag30.a3m": (
                ">101\nAAA\n>env_R\nAAA\n"
                "\x00>102\nBBBB\n>env_L\nB-BB\n"
            ),
        }
    )


def paired_archive() -> bytes:
    return make_tar(
        {
            "pair.a3m": (
                ">101\nAAA\n>pair_R\nA-A\n"
                "\x00>102\nBBBB\n>pair_L\nBB-B\n"
            )
        }
    )


class FakeResponse:
    def __init__(
        self,
        *,
        json_value: dict[str, object] | None = None,
        content: bytes = b"",
        status_code: int = 200,
    ) -> None:
        self._json_value = json_value
        self.content = content
        self.status_code = status_code
        self.text = "" if json_value is None else str(json_value)
        self.closed = False

    def json(self) -> dict[str, object]:
        if self._json_value is None:
            raise ValueError("not JSON")
        return self._json_value

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise msa.requests.HTTPError(f"HTTP {self.status_code}")

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def request(
        self, method: str, url: str, **kwargs: object
    ) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        if not self.responses:
            raise AssertionError("Unexpected HTTP request")
        return self.responses.pop(0)


class FakeAPI:
    host_url = msa.DEFAULT_API_SERVER

    def __init__(self) -> None:
        self.submissions: list[tuple[str, str]] = []
        self.query_counts: list[int] = []

    def submit(
        self,
        sequences: tuple[str, ...],
        *,
        endpoint: str,
        mode: str,
    ) -> str:
        self.submissions.append((endpoint, mode))
        self.query_counts.append(len(sequences))
        return f"{endpoint}-ticket"

    def wait(self, ticket_id: str) -> None:
        return None

    def download(self, ticket_id: str) -> bytes:
        return paired_archive() if ticket_id.startswith("pair-") else unpaired_archive()


class NoNetworkAPI(FakeAPI):
    def submit(
        self,
        sequences: tuple[str, ...],
        *,
        endpoint: str,
        mode: str,
    ) -> str:
        raise AssertionError("Cached output should not submit an API request")

    def wait(self, ticket_id: str) -> None:
        raise AssertionError("Cached output should not poll an API request")

    def download(self, ticket_id: str) -> bytes:
        raise AssertionError("Cached output should not download an API result")


class ColabFoldMSATests(unittest.TestCase):
    def test_run_uses_non_contact_user_agent_by_default(self) -> None:
        args = msa.build_parser().parse_args(
            ["run", "--data-root", "/tmp/example"]
        )
        self.assertEqual(args.user_agent, "afm-interface-reliability/1.0")

    def test_fixed_selection_contains_50_unique_ids(self) -> None:
        identifiers = msa.load_selection_ids(msa.DEFAULT_SELECTION_FILE)
        self.assertEqual(len(identifiers), 50)
        self.assertEqual(len(set(identifiers)), 50)

    def test_selection_size_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ids.txt"
            path.write_text("# test\nfirst\nsecond\n", encoding="utf-8")
            self.assertEqual(
                msa.load_selection_ids(path, expected_systems=2),
                ["first", "second"],
            )
            with self.assertRaisesRegex(ValueError, "Expected 3"):
                msa.load_selection_ids(path, expected_systems=3)

    def test_parse_complex_fasta(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "example.fasta"
            path.write_text(">example\nACDX:EFGH\n", encoding="ascii")
            self.assertEqual(
                msa.parse_complex_fasta(path, "example"),
                ("ACDX", "EFGH"),
            )

    def test_parse_exact_homodimer_fasta(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "homodimer.fasta"
            path.write_text(">homodimer\nACDX:ACDX\n", encoding="ascii")
            self.assertEqual(
                msa.parse_complex_fasta(path, "homodimer"),
                ("ACDX", "ACDX"),
            )

    def test_build_combined_a3m_contains_paired_and_unpaired_rows(self) -> None:
        unpaired = msa.stage_msas(unpaired_archive(), "unpaired")
        paired = msa.stage_msas(paired_archive(), "paired")
        combined = msa.build_combined_a3m(("AAA", "BBBB"), unpaired, paired)
        self.assertTrue(combined.startswith("#3,4\t1,1\n"))
        self.assertIn("AAABBBB", combined)
        self.assertIn("AAA----", combined)
        self.assertIn("---BBBB", combined)
        msa.validate_combined_a3m(combined, ("AAA", "BBBB"))

    def test_build_combined_a3m_uses_cardinality_for_exact_homodimer(self) -> None:
        unpaired = msa.stage_msas(unpaired_archive(), "unpaired")[0]
        combined = msa.build_combined_a3m(
            ("AAA", "AAA"),
            [unpaired],
            None,
        )
        self.assertTrue(combined.startswith("#3\t2\n"))
        self.assertIn(">101\nAAA\n", combined)
        msa.validate_combined_a3m(combined, ("AAA", "AAA"))

    def test_api_retries_rate_limit_then_polls_to_completion(self) -> None:
        session = FakeSession(
            [
                FakeResponse(json_value={"status": "RATELIMIT"}),
                FakeResponse(json_value={"status": "PENDING", "id": "ticket-1"}),
                FakeResponse(json_value={"status": "RUNNING"}),
                FakeResponse(json_value={"status": "COMPLETE"}),
                FakeResponse(content=b"archive"),
            ]
        )
        sleeps: list[float] = []
        api = msa.ColabFoldAPI(
            host_url=msa.DEFAULT_API_SERVER,
            user_agent="afm-interface-reliability-test/1.0",
            request_timeout=10,
            retries=0,
            poll_interval=1,
            max_job_wait=100,
            session=session,
            sleep=sleeps.append,
            randint=lambda start, end: 0,
        )
        ticket = api.submit(("AAA", "BBBB"), endpoint="msa", mode="env")
        api.wait(ticket)
        self.assertEqual(api.download(ticket), b"archive")
        self.assertEqual(ticket, "ticket-1")
        self.assertEqual(len(sleeps), 2)
        self.assertIn("/ticket/msa", session.calls[0][1])
        self.assertIn("/result/download/ticket-1", session.calls[-1][1])

    def test_api_maintenance_is_terminal(self) -> None:
        api = msa.ColabFoldAPI(
            host_url=msa.DEFAULT_API_SERVER,
            user_agent="afm-interface-reliability-test/1.0",
            request_timeout=10,
            retries=0,
            poll_interval=1,
            max_job_wait=100,
            session=FakeSession(
                [FakeResponse(json_value={"status": "MAINTENANCE"})]
            ),
            sleep=lambda seconds: None,
        )

        with self.assertRaises(msa.APIMaintenanceError):
            api.submit(("AAA", "CCCC"), endpoint="msa", mode="env")

    def test_completed_job_is_reused_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            paths = msa.resolve_output_paths(data_root, None)
            job = msa.MSAJob(
                pinder_id="example",
                sequence_r="AAA",
                sequence_l="BBBB",
                sha256_r=msa.sha256_text("AAA"),
                sha256_l=msa.sha256_text("BBBB"),
                fasta_path=data_root / "example.fasta",
            )
            first_api = FakeAPI()
            self.assertEqual(
                msa.process_job(
                    job=job,
                    paths=paths,
                    api=first_api,
                    overwrite=False,
                ),
                "downloaded",
            )
            self.assertEqual(
                first_api.submissions,
                [("msa", "env"), ("pair", "pairgreedy")],
            )
            self.assertEqual(first_api.query_counts, [2, 2])
            interrupted_state = msa.load_json(paths.state_path(job.pinder_id))
            interrupted_state["output"] = {}
            msa.write_json_atomic(
                paths.state_path(job.pinder_id),
                interrupted_state,
            )
            self.assertEqual(
                msa.process_job(
                    job=job,
                    paths=paths,
                    api=NoNetworkAPI(),
                    overwrite=False,
                ),
                "recovered",
            )
            self.assertEqual(
                msa.process_job(
                    job=job,
                    paths=paths,
                    api=NoNetworkAPI(),
                    overwrite=False,
                ),
                "skipped",
            )

    def test_exact_homodimer_submits_only_one_unpaired_query(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            paths = msa.resolve_output_paths(data_root, None)
            job = msa.MSAJob(
                pinder_id="homodimer",
                sequence_r="AAA",
                sequence_l="AAA",
                sha256_r=msa.sha256_text("AAA"),
                sha256_l=msa.sha256_text("AAA"),
                fasta_path=data_root / "homodimer.fasta",
            )
            api = FakeAPI()

            self.assertEqual(
                msa.process_job(
                    job=job,
                    paths=paths,
                    api=api,
                    overwrite=False,
                ),
                "downloaded",
            )
            self.assertEqual(api.submissions, [("msa", "env")])
            self.assertEqual(api.query_counts, [1])
            self.assertFalse(paths.raw_tar_path("homodimer", "paired").exists())
            output = paths.a3m_path("homodimer").read_text(encoding="utf-8")
            self.assertTrue(output.startswith("#3\t2\n"))
            msa.validate_combined_a3m(output, ("AAA", "AAA"))

    def test_verify_rejects_stale_sequence_state_without_a3m(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            paths = msa.resolve_output_paths(data_root, None)
            job = msa.MSAJob(
                pinder_id="example",
                sequence_r="AAA",
                sequence_l="CCCC",
                sha256_r=msa.sha256_text("AAA"),
                sha256_l=msa.sha256_text("CCCC"),
                fasta_path=data_root / "example.fasta",
            )
            stale_state = msa.new_state(job, msa.DEFAULT_API_SERVER)
            stale_state["sha256_R"] = msa.sha256_text("DDD")
            msa.write_json_atomic(paths.state_path(job.pinder_id), stale_state)

            row = msa.inspect_job_output(
                job,
                paths,
                msa.DEFAULT_API_SERVER,
            )

            self.assertEqual(row["status"], "failed")
            self.assertIn("state signature", str(row["error"]))

    def test_verify_requires_authoritative_raw_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            paths = msa.resolve_output_paths(data_root, None)
            job = msa.MSAJob(
                pinder_id="example",
                sequence_r="AAA",
                sequence_l="BBBB",
                sha256_r=msa.sha256_text("AAA"),
                sha256_l=msa.sha256_text("BBBB"),
                fasta_path=data_root / "example.fasta",
            )
            msa.process_job(
                job=job,
                paths=paths,
                api=FakeAPI(),
                overwrite=False,
            )
            paths.raw_tar_path(job.pinder_id, "paired").unlink()

            row = msa.inspect_job_output(
                job,
                paths,
                msa.DEFAULT_API_SERVER,
                require_raw_archives=True,
            )

            self.assertEqual(row["status"], "failed")
            self.assertIn("missing authoritative raw paired", str(row["error"]))

    def test_plan_validates_all_fastas_without_creating_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary).resolve()
            fasta_dir = data_root / "fastas" / "colabfold"
            fasta_dir.mkdir(parents=True)
            for pinder_id in msa.load_selection_ids(msa.DEFAULT_SELECTION_FILE):
                (fasta_dir / f"{pinder_id}.fasta").write_text(
                    f">{pinder_id}\nAAA:CCCC\n",
                    encoding="ascii",
                )

            args = types.SimpleNamespace(
                data_root=data_root,
                selection_file=msa.DEFAULT_SELECTION_FILE,
                expected_systems=50,
                output_dir=None,
                host_url=msa.DEFAULT_API_SERVER,
                max_systems=None,
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = msa.command_plan(args)

            self.assertEqual(exit_code, 0)
            self.assertIn("Systems valid: 50", stdout.getvalue())
            self.assertFalse((data_root / "msas").exists())

    def test_verify_exact_set_rejects_extra_a3m(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary).resolve()
            fasta_dir = data_root / "fastas" / "colabfold"
            fasta_dir.mkdir(parents=True)
            selection = data_root / "ids.txt"
            selection.write_text("example\n", encoding="utf-8")
            (fasta_dir / "example.fasta").write_text(
                ">example\nAAA:CCCC\n", encoding="ascii"
            )
            paths = msa.resolve_output_paths(data_root, str(data_root / "msas" / "test"))
            job = msa.load_jobs(data_root, selection, expected_systems=1)[0]
            paths.a3m_dir.mkdir(parents=True)
            paths.state_dir.mkdir(parents=True)
            combined = msa.build_combined_a3m(
                job.sequences,
                [">101\nAAA\n", ">102\nCCCC\n"],
                [">101\nAAA\n", ">102\nCCCC\n"],
            )
            msa.write_text_atomic(paths.a3m_path(job.pinder_id), combined)
            state = msa.new_state(job, msa.DEFAULT_API_SERVER)
            state["output"] = {
                "status": "complete",
                "a3m_sha256": msa.sha256_text(combined),
            }
            msa.write_json_atomic(paths.state_path(job.pinder_id), state)
            (paths.a3m_dir / "extra.a3m").write_text(">extra\nAAA\n", encoding="ascii")
            args = types.SimpleNamespace(
                data_root=data_root,
                selection_file=selection,
                expected_systems=1,
                output_dir=str(paths.output_dir),
                host_url=msa.DEFAULT_API_SERVER,
                max_systems=None,
                require_raw_archives=False,
                require_exact_a3m_set=True,
            )

            self.assertEqual(msa.command_verify(args), 1)


if __name__ == "__main__":
    unittest.main()
