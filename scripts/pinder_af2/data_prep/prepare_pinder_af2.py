#!/usr/bin/env python3
"""Prepare and validate the frozen PINDER-AF2 2024-02 test set.

The official subset is selected with both ``split == "test"`` and
``pinder_af2 == True``.  Its native dimers live in ``pdbs/``, whereas test-set
holo monomers live in ``test_set_pdbs/``.  All audit products use the isolated
``manifests/pinder_af2/`` directory so PINDER-Val records are never replaced.

The FASTA rule deliberately matches the PINDER-Val workflow: residues that
have coordinates in the first model of each PINDER holo PDB, in PDB order.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import pandas as pd
import requests
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from tqdm import tqdm


EXPECTED_SYSTEMS = 180
EXPECTED_SPLIT = "test"
EXPECTED_RELEASE = "2024-02"
DEFAULT_BASE_URL = "https://storage.googleapis.com/pinder"
DEFAULT_SELECTION_FILE = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "cohorts"
    / "pinder_af2_holdout_180_ids.txt"
)
SEQUENCE_RULE = "pinder_holo_resolved_residues"
LENGTH_CUTOFF = 1500
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
ALLOWED_PROTEIN_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYX")
MANIFEST_COLUMNS = [
    "pinder_release",
    "split",
    "pinder_af2",
    "pinder_xl",
    "id",
    "pdb_id",
    "cluster_id",
    "cluster_id_R",
    "cluster_id_L",
    "uniprot_R",
    "uniprot_L",
    "chain_R",
    "chain_L",
    "holo_R_pdb",
    "holo_L_pdb",
    "native_pdb",
    "mapping_R",
    "mapping_L",
    "chain1_neff",
    "chain2_neff",
    "contains_antibody",
    "contains_antigen",
    "contains_enzyme",
]
FASTA_FIELDS = [
    "pinder_id",
    "sequence_rule",
    "uniprot_R",
    "uniprot_L",
    "pdb_chain_R",
    "pdb_chain_L",
    "length_R",
    "length_L",
    "total_length",
    "exact_sequence_homodimer",
    "same_uniprot_different_sequence",
    "sha256_R",
    "sha256_L",
    "colabfold_fasta",
    "chain_R_fasta",
    "chain_L_fasta",
]

_thread_local = threading.local()


@dataclass(frozen=True)
class DownloadItem:
    kind: str
    relative_path: str
    system_id: str = ""


@dataclass
class DownloadResult:
    relative_path: str
    status: str
    bytes_on_disk: int = 0
    error: str = ""


def resolve_data_root(raw_path: str) -> Path:
    data_root = Path(raw_path).expanduser().resolve()
    if data_root.name != EXPECTED_RELEASE:
        raise ValueError(
            f"This frozen workflow requires release directory {EXPECTED_RELEASE!r}; "
            f"resolved {data_root}"
        )
    return data_root


def manifest_dir(data_root: Path) -> Path:
    return data_root / "manifests" / "pinder_af2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def dataframe_to_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def write_text_atomic(path: Path, value: str, encoding: str = "utf-8") -> None:
    write_bytes_atomic(path, value.encode(encoding))


def write_json_atomic(path: Path, value: object) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv_atomic(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_id_list(path: Path, identifiers: list[str]) -> None:
    if any(not SAFE_ID.fullmatch(identifier) for identifier in identifiers):
        raise ValueError(f"Refusing to write invalid ID list: {path}")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Refusing to write duplicate IDs: {path}")
    write_text_atomic(
        path,
        "\n".join(identifiers) + ("\n" if identifiers else ""),
    )


def load_selection_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Selection file does not exist: {path}")
    identifiers = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    invalid = [value for value in identifiers if not SAFE_ID.fullmatch(value)]
    if invalid:
        raise ValueError(f"Unsafe or invalid PINDER IDs: {invalid[:5]}")
    duplicates = sorted(
        value for value in set(identifiers) if identifiers.count(value) > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate PINDER IDs in selection: {duplicates[:5]}")
    if len(identifiers) != EXPECTED_SYSTEMS:
        raise ValueError(
            f"Expected {EXPECTED_SYSTEMS} fixed PINDER-AF2 IDs, found "
            f"{len(identifiers)} in {path}"
        )
    return identifiers


def build_manifest(
    data_root: Path,
    selection_file: Path,
    *,
    write_outputs: bool = True,
) -> pd.DataFrame:
    index_path = data_root / "index.parquet"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Missing official release index: {index_path}. "
            "Download the PINDER 2024-02 catalog first."
        )
    selection_ids = load_selection_ids(selection_file)
    index_columns = [
        column
        for column in MANIFEST_COLUMNS
        if column not in {"pinder_release", "native_pdb", "mapping_R", "mapping_L"}
    ]
    try:
        af2 = pd.read_parquet(
            index_path,
            columns=index_columns,
            filters=[("split", "==", EXPECTED_SPLIT), ("pinder_af2", "==", True)],
        ).reset_index(drop=True)
    except Exception as exc:
        raise ValueError(f"Could not read the expected PINDER index schema: {exc}") from exc

    if len(af2) != EXPECTED_SYSTEMS:
        raise ValueError(
            f"Expected {EXPECTED_SYSTEMS} rows for split={EXPECTED_SPLIT!r} and "
            f"pinder_af2=True, found {len(af2)}"
        )
    if af2["id"].nunique() != EXPECTED_SYSTEMS:
        raise ValueError("Official PINDER-AF2 IDs are not unique")
    if af2["pdb_id"].nunique() != EXPECTED_SYSTEMS:
        raise ValueError("Expected 180 unique PINDER-AF2 PDB IDs")
    if af2["cluster_id"].nunique() != EXPECTED_SYSTEMS:
        raise ValueError("Expected 180 unique PINDER-AF2 interface clusters")
    if not af2["split"].eq(EXPECTED_SPLIT).all() or not af2["pinder_af2"].eq(True).all():
        raise ValueError("Filtered index contains a row outside the official PINDER-AF2 subset")
    if "pinder_xl" in af2 and not af2["pinder_xl"].eq(True).all():
        raise ValueError("PINDER-AF2 is expected to be a subset of PINDER-XL")

    official_ids = set(af2["id"].astype(str))
    frozen_ids = set(selection_ids)
    if official_ids != frozen_ids:
        missing = sorted(official_ids - frozen_ids)
        extra = sorted(frozen_ids - official_ids)
        raise ValueError(
            "Frozen PINDER-AF2 list does not exactly match the local official index; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    af2 = af2.set_index("id", drop=False).loc[selection_ids].reset_index(drop=True)
    af2.insert(0, "pinder_release", EXPECTED_RELEASE)
    af2["native_pdb"] = af2["id"].astype(str) + ".pdb"
    af2["mapping_R"] = af2["holo_R_pdb"].map(
        lambda value: f"{Path(str(value)).stem}.parquet"
    )
    af2["mapping_L"] = af2["holo_L_pdb"].map(
        lambda value: f"{Path(str(value)).stem}.parquet"
    )
    selected = af2.loc[:, MANIFEST_COLUMNS]

    if write_outputs:
        output_dir = manifest_dir(data_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataframe_to_parquet_atomic(
            selected, output_dir / "pinder_af2_manifest.parquet"
        )
        csv_rows = selected.to_dict(orient="records")
        write_csv_atomic(
            output_dir / "pinder_af2_manifest.csv",
            csv_rows,
            MANIFEST_COLUMNS,
        )
        metadata = {
            "release": EXPECTED_RELEASE,
            "predicate": 'split == "test" and pinder_af2 == True',
            "systems": len(selected),
            "unique_pdb_ids": int(selected["pdb_id"].nunique()),
            "unique_interface_clusters": int(selected["cluster_id"].nunique()),
            "index_path": str(index_path),
            "index_sha256": sha256_file(index_path),
            "selection_file": str(selection_file.resolve()),
            "selection_file_sha256": sha256_file(selection_file),
            "manifest_csv_sha256": sha256_file(
                output_dir / "pinder_af2_manifest.csv"
            ),
        }
        write_json_atomic(output_dir / "selection_metadata.json", metadata)
    return selected


def build_inventory(
    manifest: pd.DataFrame,
    *,
    include_mappings: bool,
) -> list[DownloadItem]:
    items: list[DownloadItem] = []
    for row in manifest.itertuples(index=False):
        items.extend(
            [
                DownloadItem("native", f"pdbs/{row.native_pdb}", row.id),
                DownloadItem("holo_R", f"test_set_pdbs/{row.holo_R_pdb}", row.id),
                DownloadItem("holo_L", f"test_set_pdbs/{row.holo_L_pdb}", row.id),
            ]
        )
        if include_mappings:
            items.extend(
                [
                    DownloadItem("mapping_R", f"mappings/{row.mapping_R}", row.id),
                    DownloadItem("mapping_L", f"mappings/{row.mapping_L}", row.id),
                ]
            )
    unique: dict[str, DownloadItem] = {}
    for item in items:
        unique.setdefault(item.relative_path, item)
    return sorted(unique.values(), key=lambda item: item.relative_path)


def write_inventory(items: list[DownloadItem], data_root: Path) -> None:
    write_csv_atomic(
        manifest_dir(data_root) / "download_inventory.csv",
        (asdict(item) for item in items),
        ["kind", "relative_path", "system_id"],
    )


def write_catalog_provenance(data_root: Path) -> None:
    records: list[dict[str, object]] = []
    for name in ("index.parquet", "metadata.parquet"):
        path = data_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing official release catalog: {path}")
        records.append(
            {
                "relative_path": name,
                "bytes_on_disk": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_csv_atomic(
        manifest_dir(data_root) / "release_catalog.csv",
        records,
        ["relative_path", "bytes_on_disk", "sha256"],
    )


def get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": "afm-interface-reliability/1.0"})
        _thread_local.session = session
    return session


def remote_content_length(
    base_url: str,
    relative_path: str,
    *,
    timeout: float,
    retries: int,
) -> int:
    response = request_with_retries(
        "HEAD",
        remote_url(base_url, relative_path),
        timeout=timeout,
        retries=retries,
    )
    try:
        return int(response.headers["Content-Length"])
    finally:
        response.close()


def remote_url(base_url: str, relative_path: str) -> str:
    encoded = quote(relative_path, safe="/-_.")
    return f"{base_url.rstrip('/')}/{EXPECTED_RELEASE}/{encoded}"


def request_with_retries(
    method: str,
    url: str,
    *,
    timeout: float,
    retries: int,
    **kwargs: object,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = get_session().request(
                method,
                url,
                timeout=(min(timeout, 20.0), timeout),
                **kwargs,
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                status_code = response.status_code
                response.close()
                raise requests.HTTPError(
                    f"temporary HTTP {status_code}", response=response
                )
            response.raise_for_status()
            return response
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2**attempt, 15))
    assert last_error is not None
    raise last_error


def download_one(
    item: DownloadItem,
    *,
    data_root: Path,
    base_url: str,
    timeout: float,
    retries: int,
) -> DownloadResult:
    destination = data_root / item.relative_path
    part_path = destination.with_name(destination.name + ".part")
    try:
        remote_size: int | None = None
        if destination.is_file() and destination.stat().st_size > 0:
            local_size = destination.stat().st_size
            remote_size = remote_content_length(
                base_url,
                item.relative_path,
                timeout=timeout,
                retries=retries,
            )
            if local_size == remote_size:
                return DownloadResult(item.relative_path, "skipped", local_size)
            raise ValueError(
                f"existing file size {local_size} differs from remote "
                f"Content-Length {remote_size}; move the invalid file aside, then "
                "repeat the same download command"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        resume_at = part_path.stat().st_size if part_path.exists() else 0
        if resume_at:
            remote_size = remote_content_length(
                base_url,
                item.relative_path,
                timeout=timeout,
                retries=retries,
            )
            if resume_at == remote_size:
                part_path.replace(destination)
                return DownloadResult(item.relative_path, "downloaded", remote_size)
            if resume_at > remote_size:
                raise ValueError(
                    f"partial file size {resume_at} exceeds remote Content-Length "
                    f"{remote_size}; move the invalid .part file aside, then repeat "
                    "the same download command"
                )
        elif remote_size is None:
            remote_size = remote_content_length(
                base_url,
                item.relative_path,
                timeout=timeout,
                retries=retries,
            )

        headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}
        response = request_with_retries(
            "GET",
            remote_url(base_url, item.relative_path),
            timeout=timeout,
            retries=retries,
            headers=headers,
            stream=True,
        )
        if resume_at and response.status_code != 206:
            resume_at = 0
            part_path.unlink(missing_ok=True)
        with part_path.open("ab" if resume_at else "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        response.close()
        if part_path.stat().st_size != remote_size:
            raise ValueError(
                f"downloaded .part size {part_path.stat().st_size} differs from remote "
                f"Content-Length {remote_size}; retained for diagnosis"
            )
        part_path.replace(destination)
        return DownloadResult(
            item.relative_path, "downloaded", destination.stat().st_size
        )
    except Exception as exc:
        return DownloadResult(item.relative_path, "failed", error=str(exc))


def download_many(
    items: list[DownloadItem],
    *,
    data_root: Path,
    base_url: str,
    workers: int,
    timeout: float,
    retries: int,
) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                download_one,
                item,
                data_root=data_root,
                base_url=base_url,
                timeout=timeout,
                retries=retries,
            ): item
            for item in items
        }
        for future in tqdm(
            as_completed(future_map), total=len(future_map), unit="file", desc="PINDER-AF2"
        ):
            results.append(future.result())
    return sorted(results, key=lambda row: row.relative_path)


def protein_chain_sequences(path: Path) -> dict[str, str]:
    structure = PDBParser(QUIET=True).get_structure(path.stem, path)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(f"Expected one PDB model in {path}, found {len(models)}")
    model = models[0]
    chain_sequences: dict[str, str] = {}
    for chain in model:
        residues = [
            seq1(residue.resname, undef_code="X")
            for residue in chain
            if residue.id[0] == " "
        ]
        if residues:
            chain_sequences[str(chain.id)] = "".join(residues).upper()
    if not chain_sequences:
        raise ValueError(f"No protein residues found in {path}")
    return chain_sequences


def sequence_from_holo(path: Path, expected_side: str) -> tuple[str, str]:
    chains = protein_chain_sequences(path)
    if len(chains) != 1:
        raise ValueError(
            f"Expected exactly one protein chain in holo PDB {path}, found "
            f"{sorted(chains)}"
        )
    chain_id, sequence = next(iter(chains.items()))
    if chain_id != expected_side:
        raise ValueError(
            f"Holo PDB {path} protein chain is {chain_id!r}, expected "
            f"PINDER side {expected_side!r}"
        )
    unsupported = sorted(set(sequence).difference(ALLOWED_PROTEIN_ALPHABET))
    if unsupported:
        raise ValueError(f"Unsupported residues in {path}: {unsupported}")
    return sequence, chain_id


def validate_mapping(path: Path) -> None:
    frame = pd.read_parquet(path)
    required_any = ({"resi", "resi_pdb", "resi_auth"}, {"chain", "asym_id", "pdb_strand_id"})
    missing_groups = [sorted(group) for group in required_any if not group.intersection(frame.columns)]
    if missing_groups:
        raise ValueError(
            f"Mapping parquet {path} lacks a residue or chain identity column: "
            f"{missing_groups}"
        )


def verify_items(
    items: list[DownloadItem], data_root: Path
) -> tuple[list[str], list[str], int]:
    missing: list[str] = []
    invalid: list[str] = []
    total_bytes = 0
    for item in tqdm(items, unit="file", desc="Verifying PINDER-AF2"):
        path = data_root / item.relative_path
        if not path.is_file():
            missing.append(item.relative_path)
            continue
        size = path.stat().st_size
        total_bytes += size
        if size == 0:
            invalid.append(f"{item.relative_path}\tempty file")
            continue
        try:
            if path.suffix == ".pdb":
                protein_chain_sequences(path)
            elif path.suffix == ".parquet":
                validate_mapping(path)
        except Exception as exc:
            invalid.append(f"{item.relative_path}\t{exc}")
    return missing, invalid, total_bytes


def verify_system_structures(manifest: pd.DataFrame, data_root: Path) -> list[str]:
    invalid: list[str] = []
    for row in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        unit="system",
        desc="Verifying PINDER-AF2 systems",
    ):
        try:
            native_chains = protein_chain_sequences(
                data_root / "pdbs" / row.native_pdb
            )
            if "R" not in native_chains or "L" not in native_chains:
                raise ValueError(
                    f"native dimer requires R/L protein chains, found {sorted(native_chains)}"
                )
            sequence_from_holo(
                data_root / "test_set_pdbs" / row.holo_R_pdb, "R"
            )
            sequence_from_holo(
                data_root / "test_set_pdbs" / row.holo_L_pdb, "L"
            )
        except Exception as exc:
            invalid.append(f"{row.id}\t{exc}")
    return invalid


def expected_text(path: Path, value: str, overwrite: bool) -> None:
    if path.is_file() and not overwrite:
        existing = path.read_text(encoding="ascii")
        if existing != value:
            raise ValueError(
                f"Existing FASTA differs from the deterministic PINDER-AF2 input: "
                f"{path}. Inspect it, then use --overwrite if replacement is intended."
            )
        return
    write_text_atomic(path, value, encoding="ascii")


def validate_written_fastas(
    colabfold_path: Path,
    chain_r_path: Path,
    chain_l_path: Path,
    pinder_id: str,
    sequence_r: str,
    sequence_l: str,
) -> None:
    expected = {
        colabfold_path: f">{pinder_id}\n{sequence_r}:{sequence_l}\n",
        chain_r_path: None,
        chain_l_path: None,
    }
    for path, exact in expected.items():
        if not path.is_file():
            raise FileNotFoundError(f"Expected FASTA was not written: {path}")
        text = path.read_text(encoding="ascii")
        if exact is not None and text != exact:
            raise ValueError(f"Combined FASTA content changed after writing: {path}")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) != 2 or not lines[0].startswith(">"):
            raise ValueError(f"Expected a single-record FASTA: {path}")
        sequence = lines[1].replace(":", "")
        if path == chain_r_path and sequence != sequence_r:
            raise ValueError(f"R-chain FASTA sequence mismatch: {path}")
        if path == chain_l_path and sequence != sequence_l:
            raise ValueError(f"L-chain FASTA sequence mismatch: {path}")


def build_fastas(data_root: Path, manifest: pd.DataFrame, overwrite: bool) -> int:
    records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    colabfold_dir = data_root / "fastas" / "colabfold"
    chain_dir = data_root / "fastas" / "chains"

    for row in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        unit="system",
        desc="PINDER-AF2 FASTA",
    ):
        try:
            sequence_r, pdb_chain_r = sequence_from_holo(
                data_root / "test_set_pdbs" / row.holo_R_pdb,
                "R",
            )
            sequence_l, pdb_chain_l = sequence_from_holo(
                data_root / "test_set_pdbs" / row.holo_L_pdb,
                "L",
            )
            colabfold_path = colabfold_dir / f"{row.id}.fasta"
            chain_r_path = chain_dir / f"{row.id}__R.fasta"
            chain_l_path = chain_dir / f"{row.id}__L.fasta"
            expected_text(
                colabfold_path,
                f">{row.id}\n{sequence_r}:{sequence_l}\n",
                overwrite,
            )
            expected_text(
                chain_r_path,
                f">{row.id}|R|{row.uniprot_R}\n{sequence_r}\n",
                overwrite,
            )
            expected_text(
                chain_l_path,
                f">{row.id}|L|{row.uniprot_L}\n{sequence_l}\n",
                overwrite,
            )
            validate_written_fastas(
                colabfold_path,
                chain_r_path,
                chain_l_path,
                str(row.id),
                sequence_r,
                sequence_l,
            )
            records.append(
                {
                    "pinder_id": row.id,
                    "sequence_rule": SEQUENCE_RULE,
                    "uniprot_R": row.uniprot_R,
                    "uniprot_L": row.uniprot_L,
                    "pdb_chain_R": pdb_chain_r,
                    "pdb_chain_L": pdb_chain_l,
                    "length_R": len(sequence_r),
                    "length_L": len(sequence_l),
                    "total_length": len(sequence_r) + len(sequence_l),
                    "exact_sequence_homodimer": sequence_r == sequence_l,
                    "same_uniprot_different_sequence": (
                        row.uniprot_R == row.uniprot_L and sequence_r != sequence_l
                    ),
                    "sha256_R": sha256_text(sequence_r),
                    "sha256_L": sha256_text(sequence_l),
                    "colabfold_fasta": str(colabfold_path.relative_to(data_root)),
                    "chain_R_fasta": str(chain_r_path.relative_to(data_root)),
                    "chain_L_fasta": str(chain_l_path.relative_to(data_root)),
                }
            )
        except Exception as exc:
            failures.append({"pinder_id": row.id, "error": str(exc)})

    output_dir = manifest_dir(data_root)
    write_csv_atomic(output_dir / "fasta_manifest.csv", records, FASTA_FIELDS)
    write_csv_atomic(
        output_dir / "fasta_failures.csv",
        failures,
        ["pinder_id", "error"],
    )

    all_ids = [str(row["pinder_id"]) for row in records]
    eligible = [
        str(row["pinder_id"]) for row in records if int(row["total_length"]) < LENGTH_CUTOFF
    ]
    excluded = [
        row for row in records if int(row["total_length"]) >= LENGTH_CUTOFF
    ]
    write_id_list(output_dir / "pinder_af2_all_ids.txt", all_ids)
    write_id_list(
        output_dir / "pinder_af2_total_length_lt1500_ids.txt",
        eligible,
    )
    write_csv_atomic(
        output_dir / "pinder_af2_total_length_ge1500.csv",
        excluded,
        ["pinder_id", "length_R", "length_L", "total_length"],
    )
    write_json_atomic(
        output_dir / "length_filter_summary.json",
        {
            "rule": "length_R + length_L < 1500",
            "sequence_rule": SEQUENCE_RULE,
            "official_systems": EXPECTED_SYSTEMS,
            "fastas_complete": len(records),
            "fastas_failed": len(failures),
            "eligible_lt1500": len(eligible),
            "excluded_ge1500": len(excluded),
        },
    )
    print(f"FASTA systems created or validated: {len(records)}")
    print(f"Failures: {len(failures)}")
    print(f"AF-M length scope (<1500): {len(eligible)}")
    print(f"Excluded from AF-M inference (>=1500): {len(excluded)}")
    return 1 if failures or len(records) != EXPECTED_SYSTEMS else 0


def command_manifest(args: argparse.Namespace) -> int:
    manifest = build_manifest(args.data_root, args.selection_file)
    write_catalog_provenance(args.data_root)
    print(f"PINDER-AF2 systems: {len(manifest)}")
    print(f"Manifest: {manifest_dir(args.data_root) / 'pinder_af2_manifest.csv'}")
    return 0


def command_download(args: argparse.Namespace) -> int:
    manifest = build_manifest(args.data_root, args.selection_file)
    if args.max_systems is not None:
        manifest = manifest.iloc[: args.max_systems].copy()
    inventory = build_inventory(manifest, include_mappings=args.include_mappings)
    write_inventory(inventory, args.data_root)
    print(f"Data root: {args.data_root}")
    print(f"Systems:  {len(manifest)}")
    print(f"Objects:  {len(inventory)}")
    print(f"Free disk: {shutil.disk_usage(args.data_root).free} bytes")
    if not args.yes:
        print("No objects downloaded. Re-run with --yes after reviewing the plan.")
        return 2
    results = download_many(
        inventory,
        data_root=args.data_root,
        base_url=args.base_url,
        workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
    )
    result_rows = [asdict(result) for result in results]
    failures = [row for row in result_rows if row["status"] == "failed"]
    output_dir = manifest_dir(args.data_root)
    write_csv_atomic(
        output_dir / "download_results.csv",
        result_rows,
        ["relative_path", "status", "bytes_on_disk", "error"],
    )
    write_csv_atomic(
        output_dir / "download_failures.csv",
        failures,
        ["relative_path", "status", "bytes_on_disk", "error"],
    )
    print(
        f"Downloaded: {sum(row['status'] == 'downloaded' for row in result_rows)}; "
        f"already present: {sum(row['status'] == 'skipped' for row in result_rows)}; "
        f"failed: {len(failures)}"
    )
    return 1 if failures else 0


def command_verify(args: argparse.Namespace) -> int:
    manifest = build_manifest(args.data_root, args.selection_file)
    if args.max_systems is not None:
        manifest = manifest.iloc[: args.max_systems].copy()
    inventory = build_inventory(manifest, include_mappings=args.include_mappings)
    write_inventory(inventory, args.data_root)
    missing, invalid, total_bytes = verify_items(inventory, args.data_root)
    invalid.extend(verify_system_structures(manifest, args.data_root))
    output_dir = manifest_dir(args.data_root)
    write_text_atomic(output_dir / "missing_files.txt", "\n".join(missing) + ("\n" if missing else ""))
    write_text_atomic(output_dir / "invalid_files.txt", "\n".join(invalid) + ("\n" if invalid else ""))
    report = {
        "release": EXPECTED_RELEASE,
        "subset": "pinder_af2",
        "systems_checked": len(manifest),
        "objects_expected": len(inventory),
        "objects_missing": len(missing),
        "objects_invalid": len(invalid),
        "data_bytes_present": total_bytes,
    }
    write_json_atomic(output_dir / "verification_report.json", report)
    print(json.dumps(report, indent=2))
    return 1 if missing or invalid else 0


def command_fastas(args: argparse.Namespace) -> int:
    manifest = build_manifest(args.data_root, args.selection_file)
    return build_fastas(args.data_root, manifest, args.overwrite)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        required=True,
        help="PINDER 2024-02 release directory.",
    )
    parser.add_argument(
        "--selection-file",
        default=str(DEFAULT_SELECTION_FILE),
        help="Frozen 180-ID PINDER-AF2 text file.",
    )


def add_transfer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--include-mappings", action="store_true")
    parser.add_argument("--max-systems", type=int)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser(
        "manifest", help="Freeze and validate the official 180-system manifest."
    )
    add_common_arguments(manifest_parser)
    manifest_parser.set_defaults(func=command_manifest)

    download_parser = subparsers.add_parser(
        "download", help="Download native, test-set holo, and optional mapping files."
    )
    add_common_arguments(download_parser)
    add_transfer_arguments(download_parser)
    download_parser.add_argument("--yes", action="store_true")
    download_parser.set_defaults(func=command_download)

    verify_parser = subparsers.add_parser(
        "verify", help="Parse and validate every selected structure and mapping."
    )
    add_common_arguments(verify_parser)
    add_transfer_arguments(verify_parser)
    verify_parser.set_defaults(func=command_verify)

    fasta_parser = subparsers.add_parser(
        "fastas", help="Build deterministic two-chain FASTAs and length scopes."
    )
    add_common_arguments(fasta_parser)
    fasta_parser.add_argument("--overwrite", action="store_true")
    fasta_parser.set_defaults(func=command_fastas)
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    args.data_root = resolve_data_root(args.data_root)
    args.selection_file = Path(args.selection_file).expanduser().resolve()
    if hasattr(args, "workers") and args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if hasattr(args, "max_systems") and args.max_systems is not None:
        if not 1 <= args.max_systems <= EXPECTED_SYSTEMS:
            raise ValueError(f"--max-systems must be between 1 and {EXPECTED_SYSTEMS}")
    if hasattr(args, "timeout") and args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if hasattr(args, "retries") and args.retries < 0:
        raise ValueError("--retries must be non-negative")


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate_arguments(args)
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted; partial downloads are retained for resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
