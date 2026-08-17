#!/usr/bin/env python3
"""Build ColabFold-compatible complex MSAs with the public MMseqs2 API.

For each selected PINDER dimer this script obtains:

* unpaired per-chain MSAs from UniRef30 plus environmental databases (``env``);
* for distinct chain sequences, a greedy species-paired MSA from UniRef30
  (``pairgreedy``).

Exact-sequence homodimers are searched once and serialized with ColabFold
cardinality 2; they do not require a paired API search.

The two components are serialized into one ColabFold complex A3M.  That A3M
can be passed directly to ``colabfold_batch`` without another MSA search.

The public API is a limited shared service.  Requests are intentionally serial,
and this client sends a stable, non-contact User-Agent by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import random
import re
import sys
import tarfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import requests
from tqdm import tqdm


DEFAULT_API_SERVER = "https://api.colabfold.com"
DEFAULT_USER_AGENT = "afm-interface-reliability/1.0"
DEFAULT_OUTPUT_RELATIVE = Path("msas") / "feasibility_50"
DEFAULT_SELECTION_FILE = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "cohorts"
    / "feasibility50_ids.txt"
)
DEFAULT_EXPECTED_SYSTEMS = 50
QUERY_START = 101
UNPAIRED_MODE = "env"
PAIRED_MODE = "pairgreedy"
ALLOWED_PROTEIN_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYX")
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
TEMPORARY_HTTP_STATUS = {429, 500, 502, 503, 504}
WAITING_API_STATUS = {"UNKNOWN", "PENDING", "RUNNING", "RATELIMIT"}
FAILED_API_STATUS = {"ERROR", "MAINTENANCE"}
MANIFEST_FIELDS = [
    "pinder_id",
    "status",
    "length_R",
    "length_L",
    "sha256_R",
    "sha256_L",
    "unpaired_mode",
    "paired_mode",
    "a3m_path",
    "a3m_sha256",
    "error",
]

logger = logging.getLogger("pinder_colabfold_msa")


class APIMaintenanceError(RuntimeError):
    """The public MSA service is undergoing maintenance."""


@dataclass(frozen=True)
class MSAJob:
    pinder_id: str
    sequence_r: str
    sequence_l: str
    sha256_r: str
    sha256_l: str
    fasta_path: Path

    @property
    def sequences(self) -> tuple[str, str]:
        return self.sequence_r, self.sequence_l

    @property
    def is_exact_homodimer(self) -> bool:
        return self.sequence_r == self.sequence_l

    @property
    def query_sequences(self) -> tuple[str, ...]:
        if self.is_exact_homodimer:
            return (self.sequence_r,)
        return self.sequences

    @property
    def query_cardinalities(self) -> tuple[int, ...]:
        if self.is_exact_homodimer:
            return (2,)
        return (1, 1)

    @property
    def paired_mode(self) -> str:
        return "none" if self.is_exact_homodimer else PAIRED_MODE


@dataclass(frozen=True)
class OutputPaths:
    data_root: Path
    output_dir: Path
    a3m_dir: Path
    raw_dir: Path
    state_dir: Path
    manifest_path: Path
    failures_path: Path
    log_path: Path

    def a3m_path(self, pinder_id: str) -> Path:
        return self.a3m_dir / f"{pinder_id}.a3m"

    def state_path(self, pinder_id: str) -> Path:
        return self.state_dir / f"{pinder_id}.json"

    def raw_tar_path(self, pinder_id: str, stage: str) -> Path:
        return self.raw_dir / pinder_id / stage / "out.tar.gz"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("ascii"))


def is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def resolve_data_root(raw_path: str) -> Path:
    data_root = Path(raw_path).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"PINDER release data root does not exist: {data_root}")
    return data_root


def resolve_output_paths(data_root: Path, raw_output_dir: str | None) -> OutputPaths:
    data_root = data_root.resolve()
    output_dir = (
        Path(raw_output_dir).expanduser().resolve()
        if raw_output_dir
        else (data_root / DEFAULT_OUTPUT_RELATIVE).resolve()
    )
    if not is_within(output_dir, data_root):
        raise ValueError(
            f"Resolved MSA output must remain inside {data_root}: {output_dir}"
        )
    return OutputPaths(
        data_root=data_root,
        output_dir=output_dir,
        a3m_dir=output_dir / "a3m",
        raw_dir=output_dir / "raw",
        state_dir=output_dir / "state",
        manifest_path=output_dir / "msa_manifest.csv",
        failures_path=output_dir / "msa_failures.csv",
        log_path=output_dir / "run.log",
    )


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_text_atomic(path: Path, text: str, encoding: str = "utf-8") -> None:
    write_bytes_atomic(path, text.encode(encoding))


def write_json_atomic(path: Path, value: dict[str, object]) -> None:
    write_text_atomic(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read state file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"State file must contain a JSON object: {path}")
    return value


def load_selection_ids(
    path: Path,
    expected_systems: int = DEFAULT_EXPECTED_SYSTEMS,
) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Selection file does not exist: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "id" not in (reader.fieldnames or []):
                raise ValueError(f"Selection CSV must contain an 'id' column: {path}")
            identifiers = [str(row["id"]).strip() for row in reader]
    else:
        identifiers = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    invalid = [
        identifier
        for identifier in identifiers
        if not identifier or not SAFE_ID.fullmatch(identifier)
    ]
    if invalid:
        raise ValueError(f"Unsafe or invalid PINDER IDs: {invalid[:5]}")
    duplicates = sorted(
        identifier
        for identifier in set(identifiers)
        if identifiers.count(identifier) > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate PINDER IDs in selection: {duplicates[:5]}")
    if len(identifiers) != expected_systems:
        raise ValueError(
            f"Expected {expected_systems} fixed PINDER IDs, "
            f"found {len(identifiers)} in {path}"
        )
    return identifiers


def parse_complex_fasta(path: Path, expected_id: str) -> tuple[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing complex FASTA: {path}")
    headers: list[str] = []
    sequence_parts: list[str] = []
    for raw_line in path.read_text(encoding="ascii").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            headers.append(line[1:].strip())
        else:
            sequence_parts.append(line)
    if len(headers) != 1:
        raise ValueError(f"Expected one FASTA record in {path}, found {len(headers)}")
    if headers[0].split()[0] != expected_id:
        raise ValueError(
            f"FASTA header does not match selection ID in {path}: {headers[0]!r}"
        )
    sequence = "".join(sequence_parts).upper()
    chains = sequence.split(":")
    if len(chains) != 2 or not all(chains):
        raise ValueError(f"Expected exactly two non-empty chains separated by ':' in {path}")
    for label, chain in zip(("R", "L"), chains):
        unsupported = sorted(set(chain).difference(ALLOWED_PROTEIN_ALPHABET))
        if unsupported:
            raise ValueError(
                f"Unsupported residues in {path} chain {label}: {unsupported}"
            )
    return chains[0], chains[1]


def load_jobs(
    data_root: Path,
    selection_file: Path,
    expected_systems: int = DEFAULT_EXPECTED_SYSTEMS,
) -> list[MSAJob]:
    jobs: list[MSAJob] = []
    for pinder_id in load_selection_ids(selection_file, expected_systems):
        fasta_path = data_root / "fastas" / "colabfold" / f"{pinder_id}.fasta"
        sequence_r, sequence_l = parse_complex_fasta(fasta_path, pinder_id)
        jobs.append(
            MSAJob(
                pinder_id=pinder_id,
                sequence_r=sequence_r,
                sequence_l=sequence_l,
                sha256_r=sha256_text(sequence_r),
                sha256_l=sha256_text(sequence_l),
                fasta_path=fasta_path,
            )
        )
    return jobs


class ColabFoldAPI:
    """Minimal serial client matching the current ColabFold MMseqs2 protocol."""

    def __init__(
        self,
        *,
        host_url: str,
        user_agent: str,
        request_timeout: float,
        retries: int,
        poll_interval: float,
        max_job_wait: float,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        randint: Callable[[int, int], int] = random.randint,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("--user-agent must not be empty")
        self.host_url = host_url.rstrip("/")
        self.request_timeout = request_timeout
        self.retries = retries
        self.poll_interval = poll_interval
        self.max_job_wait = max_job_wait
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent.strip()})
        self.sleep = sleep
        self.clock = clock
        self.randint = randint

    def _request(self, method: str, path: str, **kwargs: object) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.request(
                    method,
                    f"{self.host_url}/{path.lstrip('/')}",
                    timeout=(6.02, self.request_timeout),
                    **kwargs,
                )
                if response.status_code in TEMPORARY_HTTP_STATUS:
                    response.close()
                    raise requests.HTTPError(
                        f"temporary HTTP {response.status_code}", response=response
                    )
                response.raise_for_status()
                return response
            except (requests.RequestException, OSError) as exc:
                last_error = exc
                if attempt == self.retries:
                    break
                delay = min(2**attempt, 30) + self.randint(0, 3)
                logger.warning(
                    "Network/API error (%s/%s): %s; retrying in %ss",
                    attempt + 1,
                    self.retries + 1,
                    exc,
                    delay,
                )
                self.sleep(delay)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _response_json(response: requests.Response) -> dict[str, object]:
        try:
            value = response.json()
        except ValueError as exc:
            preview = response.text[:500]
            raise RuntimeError(f"MSA server returned non-JSON data: {preview!r}") from exc
        finally:
            response.close()
        if not isinstance(value, dict):
            raise RuntimeError(f"MSA server returned unexpected JSON: {value!r}")
        return value

    def submit(
        self,
        sequences: Sequence[str],
        *,
        endpoint: str,
        mode: str,
    ) -> str:
        query = "".join(
            f">{QUERY_START + index}\n{sequence}\n"
            for index, sequence in enumerate(sequences)
        )
        deadline = self.clock() + self.max_job_wait
        while True:
            response = self._request(
                "POST",
                f"ticket/{endpoint}",
                data={"q": query, "mode": mode},
            )
            result = self._response_json(response)
            status = str(result.get("status", "UNKNOWN")).upper()
            if status in {"UNKNOWN", "RATELIMIT"}:
                if self.clock() >= deadline:
                    raise TimeoutError(f"Timed out submitting MSA job ({status})")
                delay = self.poll_interval + self.randint(0, 5)
                logger.warning("MSA submission status %s; retrying in %ss", status, delay)
                self.sleep(delay)
                continue
            if status == "MAINTENANCE":
                raise APIMaintenanceError(
                    "MSA API is undergoing maintenance; stop and retry later"
                )
            if status in FAILED_API_STATUS:
                raise RuntimeError(f"MSA submission failed with status {status}")
            ticket_id = str(result.get("id", "")).strip()
            if not ticket_id:
                raise RuntimeError(f"MSA server response has no ticket ID: {result!r}")
            return ticket_id

    def wait(self, ticket_id: str) -> None:
        deadline = self.clock() + self.max_job_wait
        while True:
            response = self._request("GET", f"ticket/{ticket_id}")
            result = self._response_json(response)
            status = str(result.get("status", "UNKNOWN")).upper()
            if status == "COMPLETE":
                return
            if status == "MAINTENANCE":
                raise APIMaintenanceError(
                    f"MSA ticket {ticket_id} stopped because the API is "
                    "undergoing maintenance"
                )
            if status in FAILED_API_STATUS:
                raise RuntimeError(
                    f"MSA ticket {ticket_id} failed with status {status}"
                )
            if status not in WAITING_API_STATUS:
                raise RuntimeError(
                    f"MSA ticket {ticket_id} returned unknown status {status!r}"
                )
            if self.clock() >= deadline:
                raise TimeoutError(
                    f"MSA ticket {ticket_id} did not complete within "
                    f"{self.max_job_wait:g}s"
                )
            delay = self.poll_interval + self.randint(0, 5)
            logger.info("Ticket %s: %s; polling again in %ss", ticket_id, status, delay)
            self.sleep(delay)

    def download(self, ticket_id: str) -> bytes:
        response = self._request("GET", f"result/download/{ticket_id}")
        try:
            content = bytes(response.content)
        finally:
            response.close()
        if not content:
            raise RuntimeError(f"MSA ticket {ticket_id} returned an empty archive")
        return content


def read_tar_member(archive: bytes, expected_name: str) -> str:
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            matches = [
                member
                for member in tar.getmembers()
                if member.name.lstrip("./") == expected_name
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one {expected_name!r} in archive, found {len(matches)}"
                )
            member = matches[0]
            if not member.isfile() or member.size <= 0:
                raise ValueError(f"Invalid archive member: {member.name}")
            if member.size > 2 * 1024**3:
                raise ValueError(f"Archive member is unexpectedly large: {member.name}")
            handle = tar.extractfile(member)
            if handle is None:
                raise ValueError(f"Could not read archive member: {member.name}")
            return handle.read().decode("utf-8")
    except (tarfile.TarError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid ColabFold result archive: {exc}") from exc


def split_api_a3m(texts: Iterable[str]) -> dict[int, str]:
    """Split API multi-query A3M files on their NUL query separators."""

    grouped: dict[int, list[str]] = {}
    for text in texts:
        for chunk in text.split("\x00"):
            normalized = chunk.lstrip("\r\n")
            if not normalized.strip():
                continue
            lines = normalized.splitlines()
            if not lines or not lines[0].startswith(">"):
                raise ValueError("API A3M query block does not start with a header")
            token = lines[0][1:].strip().split()[0]
            try:
                query_id = int(token)
            except ValueError as exc:
                raise ValueError(
                    f"API A3M query block has non-numeric first header: {lines[0]!r}"
                ) from exc
            grouped.setdefault(query_id, []).append("\n".join(lines) + "\n")
    return {query_id: "".join(parts) for query_id, parts in grouped.items()}


def stage_msas(archive: bytes, stage: str, query_count: int = 2) -> list[str]:
    if stage == "unpaired":
        texts = [
            read_tar_member(archive, "uniref.a3m"),
            read_tar_member(archive, "bfd.mgnify30.metaeuk30.smag30.a3m"),
        ]
    elif stage == "paired":
        texts = [read_tar_member(archive, "pair.a3m")]
    else:
        raise ValueError(f"Unknown MSA stage: {stage}")

    grouped = split_api_a3m(texts)
    expected_ids = [QUERY_START + index for index in range(query_count)]
    missing = [query_id for query_id in expected_ids if query_id not in grouped]
    if missing:
        raise ValueError(f"{stage} MSA archive is missing query blocks: {missing}")
    return [grouped[query_id] for query_id in expected_ids]


def pair_sequences(a3m_lines: Sequence[str]) -> str:
    split_lines = [
        [line for line in a3m.splitlines() if line]
        for a3m in a3m_lines
    ]
    if not split_lines or not split_lines[0]:
        raise ValueError("Paired MSA is empty")
    expected_line_count = len(split_lines[0])
    if any(len(lines) != expected_line_count for lines in split_lines):
        raise ValueError("Paired chain MSAs have different line counts")

    combined = [""] * expected_line_count
    for chain_index, lines in enumerate(split_lines):
        for line_index, line in enumerate(lines):
            if line.startswith(">"):
                if chain_index:
                    line = line.replace(">", "\t", 1)
                combined[line_index] += line
            else:
                combined[line_index] += line
    return "\n".join(combined)


def pad_sequences(a3m_lines: Sequence[str], sequences: Sequence[str]) -> str:
    blanks = ["-" * len(sequence) for sequence in sequences]
    combined: list[str] = []
    for chain_index, a3m in enumerate(a3m_lines):
        for line in a3m.splitlines():
            if not line:
                continue
            if line.startswith(">"):
                combined.append(line)
            else:
                row = list(blanks)
                row[chain_index] = line
                combined.append("".join(row))
    return "\n".join(combined)


def build_combined_a3m(
    sequences: Sequence[str],
    unpaired_msas: Sequence[str],
    paired_msas: Sequence[str] | None,
) -> str:
    if len(sequences) != 2:
        raise ValueError("This workflow requires exactly two protein chains")
    if sequences[0] == sequences[1]:
        unique_sequences = (sequences[0],)
        cardinalities = (2,)
        if len(unpaired_msas) != 1 or paired_msas is not None:
            raise ValueError(
                "Exact-sequence homodimers require one unpaired MSA and no "
                "paired API MSA"
            )
        body = pad_sequences(unpaired_msas, unique_sequences)
    else:
        unique_sequences = tuple(sequences)
        cardinalities = (1, 1)
        if len(unpaired_msas) != 2 or paired_msas is None or len(paired_msas) != 2:
            raise ValueError(
                "Distinct-chain dimers require two unpaired and two paired MSAs"
            )
        body = (
            pair_sequences(paired_msas)
            + "\n"
            + pad_sequences(unpaired_msas, unique_sequences)
        )
    header = (
        "#"
        + ",".join(str(len(sequence)) for sequence in unique_sequences)
        + "\t"
        + ",".join(str(cardinality) for cardinality in cardinalities)
    )
    result = f"{header}\n{body}\n"
    validate_combined_a3m(result, sequences)
    return result


def validate_combined_a3m(text: str, sequences: Sequence[str]) -> None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(sequences) != 2:
        raise ValueError("This workflow requires exactly two protein chains")
    if sequences[0] == sequences[1]:
        unique_sequences = (sequences[0],)
        cardinalities = (2,)
    else:
        unique_sequences = tuple(sequences)
        cardinalities = (1, 1)
    expected_header = (
        "#"
        + ",".join(str(len(sequence)) for sequence in unique_sequences)
        + "\t"
        + ",".join(str(cardinality) for cardinality in cardinalities)
    )
    if not lines or lines[0] != expected_header:
        raise ValueError(
            f"Combined A3M header mismatch; expected {expected_header!r}"
        )
    body = lines[1:]
    sequence_rows = [line for line in body if not line.startswith(">")]
    if not sequence_rows:
        raise ValueError("Combined A3M contains no aligned sequences")
    expected_query = "".join(unique_sequences)
    if sequence_rows[0] != expected_query:
        raise ValueError("Combined A3M query row does not match the input chains")
    expected_length = len(expected_query)
    for index, row in enumerate(sequence_rows, start=1):
        aligned = "".join(
            character
            for character in row
            if not character.islower() and character != "."
        )
        if len(aligned) != expected_length:
            raise ValueError(
                f"Combined A3M row {index} has aligned length {len(aligned)}, "
                f"expected {expected_length}"
            )


def job_signature(job: MSAJob, host_url: str) -> dict[str, object]:
    signature: dict[str, object] = {
        "pinder_id": job.pinder_id,
        "sha256_R": job.sha256_r,
        "sha256_L": job.sha256_l,
        "length_R": len(job.sequence_r),
        "length_L": len(job.sequence_l),
        "api_server": host_url.rstrip("/"),
        "unpaired_mode": UNPAIRED_MODE,
        "paired_mode": job.paired_mode,
    }
    if job.is_exact_homodimer:
        signature["query_cardinalities"] = list(job.query_cardinalities)
    return signature


def state_matches(
    state: dict[str, object], job: MSAJob, host_url: str
) -> bool:
    signature = job_signature(job, host_url)
    return all(state.get(key) == value for key, value in signature.items())


def new_state(job: MSAJob, host_url: str) -> dict[str, object]:
    return {
        **job_signature(job, host_url),
        "unpaired": {},
        "paired": (
            {"status": "not_required", "mode": "none"}
            if job.is_exact_homodimer
            else {}
        ),
        "output": {},
        "updated_at": utc_now(),
    }


def update_state(
    state_path: Path,
    state: dict[str, object],
    *,
    last_error: str | None = None,
) -> None:
    state["updated_at"] = utc_now()
    if last_error is None:
        state.pop("last_error", None)
    else:
        state["last_error"] = last_error
    write_json_atomic(state_path, state)


def obtain_stage_archive(
    *,
    api: ColabFoldAPI,
    job: MSAJob,
    stage: str,
    endpoint: str,
    mode: str,
    raw_path: Path,
    state_path: Path,
    state: dict[str, object],
    overwrite: bool,
    query_count: int,
) -> bytes:
    stage_state = state.setdefault(stage, {})
    if not isinstance(stage_state, dict):
        raise ValueError(f"Invalid {stage} state in {state_path}")

    if not overwrite and raw_path.is_file():
        archive = raw_path.read_bytes()
        expected_hash = str(stage_state.get("archive_sha256", ""))
        current_hash = sha256_bytes(archive)
        stage_msas(archive, stage, query_count)
        if expected_hash and current_hash != expected_hash:
            raise ValueError(
                f"Existing {stage} archive does not match its state hash: "
                f"{raw_path}. Use --overwrite to replace it."
            )
        if not expected_hash:
            stage_state.update(
                {
                    "status": "complete",
                    "archive_sha256": current_hash,
                    "recovered_at": utc_now(),
                }
            )
            update_state(state_path, state)
        return archive

    ticket_id = "" if overwrite else str(stage_state.get("ticket_id", "")).strip()
    if ticket_id:
        logger.info("Resuming %s ticket %s for %s", stage, ticket_id, job.pinder_id)
        try:
            api.wait(ticket_id)
        except APIMaintenanceError:
            raise
        except (requests.RequestException, RuntimeError, TimeoutError) as exc:
            logger.warning(
                "Could not resume %s ticket %s (%s); submitting a new ticket",
                stage,
                ticket_id,
                exc,
            )
            ticket_id = ""

    if not ticket_id:
        ticket_id = api.submit(job.query_sequences, endpoint=endpoint, mode=mode)
        stage_state.update(
            {
                "ticket_id": ticket_id,
                "status": "submitted",
                "submitted_at": utc_now(),
            }
        )
        update_state(state_path, state)
        api.wait(ticket_id)

    archive = api.download(ticket_id)
    stage_msas(archive, stage, query_count)
    write_bytes_atomic(raw_path, archive)
    stage_state.update(
        {
            "ticket_id": ticket_id,
            "status": "complete",
            "archive_sha256": sha256_bytes(archive),
            "completed_at": utc_now(),
        }
    )
    update_state(state_path, state)
    return archive


def process_job(
    *,
    job: MSAJob,
    paths: OutputPaths,
    api: ColabFoldAPI,
    overwrite: bool,
) -> str:
    state_path = paths.state_path(job.pinder_id)
    a3m_path = paths.a3m_path(job.pinder_id)
    existing_state = load_json(state_path)
    if existing_state and not state_matches(existing_state, job, api.host_url):
        if not overwrite:
            raise ValueError(
                f"Existing state does not match current sequence/API settings: "
                f"{state_path}. Use --overwrite to replace this system."
            )
        state = new_state(job, api.host_url)
    else:
        state = existing_state or new_state(job, api.host_url)

    output_state = state.setdefault("output", {})
    if not isinstance(output_state, dict):
        raise ValueError(f"Invalid output state in {state_path}")
    if not overwrite and a3m_path.is_file():
        current_text = a3m_path.read_text(encoding="utf-8")
        current_hash = sha256_text(current_text)
        validate_combined_a3m(current_text, job.sequences)
        if (
            output_state.get("status") == "complete"
            and output_state.get("a3m_sha256") == current_hash
        ):
            return "skipped"
        required_stages = (
            ("unpaired",) if job.is_exact_homodimer else ("unpaired", "paired")
        )
        stages_complete = all(
            isinstance(state.get(stage), dict)
            and state[stage].get("status") == "complete"
            and state[stage].get("archive_sha256")
            for stage in required_stages
        )
        if not output_state.get("a3m_sha256") and stages_complete:
            output_state.update(
                {
                    "status": "complete",
                    "a3m_sha256": current_hash,
                    "recovered_at": utc_now(),
                }
            )
            update_state(state_path, state)
            return "recovered"
        raise ValueError(
            f"Existing A3M is not backed by matching state: {a3m_path}. "
            "Use --overwrite to replace it."
        )

    update_state(state_path, state)
    unpaired_archive = obtain_stage_archive(
        api=api,
        job=job,
        stage="unpaired",
        endpoint="msa",
        mode=UNPAIRED_MODE,
        raw_path=paths.raw_tar_path(job.pinder_id, "unpaired"),
        state_path=state_path,
        state=state,
        overwrite=overwrite,
        query_count=len(job.query_sequences),
    )
    paired_archive: bytes | None = None
    if not job.is_exact_homodimer:
        paired_archive = obtain_stage_archive(
            api=api,
            job=job,
            stage="paired",
            endpoint="pair",
            mode=PAIRED_MODE,
            raw_path=paths.raw_tar_path(job.pinder_id, "paired"),
            state_path=state_path,
            state=state,
            overwrite=overwrite,
            query_count=len(job.query_sequences),
        )
    combined = build_combined_a3m(
        job.sequences,
        stage_msas(
            unpaired_archive,
            "unpaired",
            query_count=len(job.query_sequences),
        ),
        (
            stage_msas(
                paired_archive,
                "paired",
                query_count=len(job.query_sequences),
            )
            if paired_archive is not None
            else None
        ),
    )
    write_text_atomic(a3m_path, combined)
    output_state.update(
        {
            "status": "complete",
            "a3m_sha256": sha256_text(combined),
            "completed_at": utc_now(),
        }
    )
    update_state(state_path, state)
    return "downloaded"


def relative_to_data_root(path: Path, data_root: Path) -> str:
    return str(path.relative_to(data_root))


def inspect_job_output(
    job: MSAJob,
    paths: OutputPaths,
    host_url: str,
    *,
    require_raw_archives: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "pinder_id": job.pinder_id,
        "status": "pending",
        "length_R": len(job.sequence_r),
        "length_L": len(job.sequence_l),
        "sha256_R": job.sha256_r,
        "sha256_L": job.sha256_l,
        "unpaired_mode": UNPAIRED_MODE,
        "paired_mode": job.paired_mode,
        "a3m_path": relative_to_data_root(
            paths.a3m_path(job.pinder_id), paths.data_root
        ),
        "a3m_sha256": "",
        "error": "",
    }
    state_path = paths.state_path(job.pinder_id)
    a3m_path = paths.a3m_path(job.pinder_id)
    try:
        state = load_json(state_path)
        if state and not state_matches(state, job, host_url):
            raise ValueError("state signature does not match current input")
        if not a3m_path.is_file():
            if state.get("last_error"):
                row["status"] = "failed"
                row["error"] = str(state["last_error"])
            return row
        if not state:
            raise ValueError("A3M exists without a state file")
        text = a3m_path.read_text(encoding="utf-8")
        validate_combined_a3m(text, job.sequences)
        a3m_hash = sha256_text(text)
        output_state = state.get("output", {})
        if not isinstance(output_state, dict):
            raise ValueError("invalid output state")
        if output_state.get("a3m_sha256") != a3m_hash:
            raise ValueError("A3M hash does not match state")
        if require_raw_archives:
            required_stages = (
                ("unpaired",) if job.is_exact_homodimer else ("unpaired", "paired")
            )
            for stage_name in required_stages:
                stage_state = state.get(stage_name, {})
                if not isinstance(stage_state, dict):
                    raise ValueError(f"invalid {stage_name} stage state")
                archive_path = paths.raw_tar_path(job.pinder_id, stage_name)
                if not archive_path.is_file():
                    raise FileNotFoundError(
                        f"missing authoritative raw {stage_name} archive: {archive_path}"
                    )
                archive = archive_path.read_bytes()
                archive_sha256 = sha256_bytes(archive)
                if stage_state.get("archive_sha256") != archive_sha256:
                    raise ValueError(
                        f"raw {stage_name} archive hash does not match state"
                    )
                stage_msas(archive, stage_name, len(job.query_sequences))
        row["status"] = "complete"
        row["a3m_sha256"] = a3m_hash
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = str(exc)
    return row


def write_csv_atomic(
    path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]
) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    write_text_atomic(path, buffer.getvalue())


def write_manifests(jobs: Sequence[MSAJob], paths: OutputPaths, host_url: str) -> None:
    rows = [inspect_job_output(job, paths, host_url) for job in jobs]
    write_manifest_rows(rows, paths)


def write_manifest_rows(
    rows: Sequence[dict[str, object]],
    paths: OutputPaths,
) -> None:
    write_csv_atomic(paths.manifest_path, rows, MANIFEST_FIELDS)
    failures = [row for row in rows if row["status"] == "failed"]
    write_csv_atomic(paths.failures_path, failures, MANIFEST_FIELDS)


def setup_run_logging(paths: OutputPaths) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(paths.log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


def load_context(args: argparse.Namespace) -> tuple[list[MSAJob], OutputPaths]:
    data_root = resolve_data_root(args.data_root)
    selection_file = Path(args.selection_file).expanduser().resolve()
    jobs = load_jobs(data_root, selection_file, args.expected_systems)
    paths = resolve_output_paths(data_root, args.output_dir)
    return jobs, paths


def selected_targets(
    jobs: Sequence[MSAJob], max_systems: int | None
) -> list[MSAJob]:
    if max_systems is None:
        return list(jobs)
    if max_systems < 1:
        raise ValueError("--max-systems must be at least 1")
    return list(jobs[:max_systems])


def command_plan(args: argparse.Namespace) -> int:
    jobs, paths = load_context(args)
    targets = selected_targets(jobs, args.max_systems)
    print(f"Selection file: {Path(args.selection_file).expanduser().resolve()}")
    print(f"PINDER data:   {paths.data_root}")
    print(f"MSA output:    {paths.output_dir}")
    print(f"Systems valid: {len(jobs)}")
    print(f"Systems target:{len(targets):>3}")
    print(f"Protein chains:{2 * len(jobs):>3}")
    print(f"Total residues:{sum(len(s) for job in jobs for s in job.sequences):>3}")
    exact_homodimers = sum(job.is_exact_homodimer for job in jobs)
    print(f"Exact homodimers:{exact_homodimers:>3}")
    print(f"Paired searches:{len(jobs) - exact_homodimers:>3}")
    print(f"Unpaired mode: {UNPAIRED_MODE}")
    print(f"Paired mode:   {PAIRED_MODE} for distinct sequences; none for exact homodimers")
    print("Network access: none (plan only)")
    return 0


def command_run(args: argparse.Namespace) -> int:
    jobs, paths = load_context(args)
    targets = selected_targets(jobs, args.max_systems)
    for directory in (paths.a3m_dir, paths.raw_dir, paths.state_dir):
        directory.mkdir(parents=True, exist_ok=True)
    setup_run_logging(paths)
    api = ColabFoldAPI(
        host_url=args.host_url,
        user_agent=args.user_agent,
        request_timeout=args.request_timeout,
        retries=args.retries,
        poll_interval=args.poll_interval,
        max_job_wait=args.max_job_wait,
    )
    rows_by_id = {
        job.pinder_id: inspect_job_output(job, paths, api.host_url)
        for job in jobs
    }
    write_manifest_rows(
        [rows_by_id[job.pinder_id] for job in jobs],
        paths,
    )
    failures = 0
    interrupted = False
    try:
        for job in tqdm(targets, unit="system", desc="ColabFold MSA"):
            try:
                status = process_job(
                    job=job,
                    paths=paths,
                    api=api,
                    overwrite=args.overwrite,
                )
                logger.info("%s: %s", job.pinder_id, status)
            except KeyboardInterrupt:
                interrupted = True
                raise
            except Exception as exc:
                failures += 1
                logger.exception("%s: failed: %s", job.pinder_id, exc)
                state_path = paths.state_path(job.pinder_id)
                try:
                    state = load_json(state_path) or new_state(job, api.host_url)
                    update_state(state_path, state, last_error=str(exc))
                except Exception:
                    logger.exception("Could not update failure state for %s", job.pinder_id)
                if isinstance(exc, APIMaintenanceError):
                    logger.error(
                        "Stopping the batch because the MSA API is in maintenance"
                    )
                    break
            finally:
                rows_by_id[job.pinder_id] = inspect_job_output(
                    job,
                    paths,
                    api.host_url,
                )
                write_manifest_rows(
                    [rows_by_id[item.pinder_id] for item in jobs],
                    paths,
                )
    finally:
        if interrupted:
            logger.warning("Interrupted; completed archives and state files were retained")

    rows = [rows_by_id[job.pinder_id] for job in jobs]
    complete = sum(row["status"] == "complete" for row in rows)
    pending = sum(row["status"] == "pending" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    print(f"Complete: {complete}; pending: {pending}; failed: {failed}")
    print(f"Manifest: {paths.manifest_path}")
    return 1 if failures else 0


def command_verify(args: argparse.Namespace) -> int:
    jobs, paths = load_context(args)
    targets = selected_targets(jobs, args.max_systems)
    rows = [
        inspect_job_output(
            job,
            paths,
            args.host_url,
            require_raw_archives=args.require_raw_archives,
        )
        for job in targets
    ]
    complete = [row for row in rows if row["status"] == "complete"]
    failed = [row for row in rows if row["status"] == "failed"]
    pending = [row for row in rows if row["status"] == "pending"]
    extra_a3ms: list[str] = []
    extra_state_files: list[str] = []
    if args.require_exact_a3m_set and args.max_systems is None:
        expected_a3ms = {f"{job.pinder_id}.a3m" for job in targets}
        actual_a3ms = (
            {path.name for path in paths.a3m_dir.glob("*.a3m")}
            if paths.a3m_dir.is_dir()
            else set()
        )
        expected_states = {f"{job.pinder_id}.json" for job in targets}
        actual_states = (
            {path.name for path in paths.state_dir.glob("*.json")}
            if paths.state_dir.is_dir()
            else set()
        )
        extra_a3ms = sorted(actual_a3ms - expected_a3ms)
        extra_state_files = sorted(actual_states - expected_states)
    print(
        json.dumps(
            {
                "systems_expected": len(targets),
                "systems_complete": len(complete),
                "systems_pending": len(pending),
                "systems_failed": len(failed),
                "extra_a3m_files": len(extra_a3ms),
                "extra_state_files": len(extra_state_files),
                "a3m_directory": str(paths.a3m_dir),
            },
            indent=2,
        )
    )
    if failed:
        print("First failures:", file=sys.stderr)
        for row in failed[:5]:
            print(f"  {row['pinder_id']}: {row['error']}", file=sys.stderr)
    if extra_a3ms:
        print(f"Unexpected A3M files: {extra_a3ms[:5]}", file=sys.stderr)
    if extra_state_files:
        print(f"Unexpected state files: {extra_state_files[:5]}", file=sys.stderr)
    return 1 if failed or pending or extra_a3ms or extra_state_files else 0


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        required=True,
        help="PINDER 2024-02 release directory.",
    )
    parser.add_argument(
        "--selection-file",
        default=str(DEFAULT_SELECTION_FILE),
        help="Fixed ID text file or CSV containing an id column.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Optional output directory inside --data-root. Defaults to "
            "<data-root>/msas/feasibility_50."
        ),
    )
    parser.add_argument("--host-url", default=DEFAULT_API_SERVER)
    parser.add_argument(
        "--expected-systems",
        type=int,
        default=DEFAULT_EXPECTED_SYSTEMS,
        help=(
            "Require exactly this many unique IDs in --selection-file. "
            f"Defaults to {DEFAULT_EXPECTED_SYSTEMS} for Feasibility50; "
            "override for another frozen cohort."
        ),
    )
    parser.add_argument("--max-systems", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Validate all selected IDs and FASTAs without network access.",
    )
    add_common_arguments(plan_parser)
    plan_parser.set_defaults(func=command_plan)

    run_parser = subparsers.add_parser(
        "run",
        help="Serially obtain and cache unpaired plus paired MSAs.",
    )
    add_common_arguments(run_parser)
    run_parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help=(
            "Optional client identifier. Defaults to 'afm-interface-reliability/1.0'; "
            "no email or account is required."
        ),
    )
    run_parser.add_argument("--request-timeout", type=float, default=120.0)
    run_parser.add_argument("--retries", type=int, default=4)
    run_parser.add_argument("--poll-interval", type=float, default=5.0)
    run_parser.add_argument("--max-job-wait", type=float, default=7200.0)
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace cached artifacts for selected systems.",
    )
    run_parser.set_defaults(func=command_run)

    verify_parser = subparsers.add_parser(
        "verify",
        help="Validate cached state and combined A3M files without network access.",
    )
    add_common_arguments(verify_parser)
    verify_parser.add_argument(
        "--require-raw-archives",
        action="store_true",
        help=(
            "Also require every authoritative API tar archive to exist, match "
            "its state SHA-256, and contain the expected A3M members."
        ),
    )
    verify_parser.add_argument(
        "--require-exact-a3m-set",
        action="store_true",
        help=(
            "On a full verify (without --max-systems), reject extra A3M or "
            "state files outside the selected target set."
        ),
    )
    verify_parser.set_defaults(func=command_verify)
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.expected_systems < 1:
        raise ValueError("--expected-systems must be at least 1")
    if args.max_systems is not None and args.max_systems < 1:
        raise ValueError("--max-systems must be at least 1")
    for name in ("request_timeout", "poll_interval", "max_job_wait"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "retries") and args.retries < 0:
        raise ValueError("--retries must be non-negative")


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate_arguments(args)
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted; cached archives and state files are retained.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
