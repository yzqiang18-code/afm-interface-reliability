#!/usr/bin/env python3
"""Build AF-M FASTA inputs from downloaded PINDER-Val holo monomers.

The sequence rule is deliberately explicit: use residues with coordinates in
the PINDER holo PDB files, in PDB residue order.  This gives a compact sequence
that maps directly to the native reference.  It may omit experimentally
unresolved residues, so full-length UniProt experiments should be generated as
a separate condition rather than mixed with these files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from tqdm import tqdm


SEQUENCE_RULE = "pinder_holo_resolved_residues"


def resolve_data_root(raw_path: str) -> Path:
    data_root = Path(raw_path).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"PINDER release directory does not exist: {data_root}")
    return data_root


def sequence_from_pdb(path: Path) -> str:
    structure = PDBParser(QUIET=True).get_structure(path.stem, path)
    model = next(structure.get_models())
    residues: list[str] = []
    for chain in model:
        for residue in chain:
            hetflag = residue.id[0]
            if hetflag not in {" ", "W"}:
                continue
            if hetflag == "W":
                continue
            residues.append(seq1(residue.resname, undef_code="X"))
    sequence = "".join(residues)
    if not sequence:
        raise ValueError(f"No protein residues found in {path}")
    return sequence


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(text, encoding="ascii")
    temporary.replace(path)


def build_fastas(data_root: Path, overwrite: bool) -> int:
    manifest_path = data_root / "manifests" / "pinder_val_manifest.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}; prepare the PINDER-Val source manifest first."
        )
    manifest = pd.read_parquet(manifest_path)

    colabfold_dir = data_root / "fastas" / "colabfold"
    chain_dir = data_root / "fastas" / "chains"
    records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for row in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        unit="system",
        desc="FASTA",
    ):
        try:
            holo_r_path = data_root / "pdbs" / row.holo_R_pdb
            holo_l_path = data_root / "pdbs" / row.holo_L_pdb
            if not holo_r_path.is_file() or not holo_l_path.is_file():
                raise FileNotFoundError("holo R/L PDB is missing")

            sequence_r = sequence_from_pdb(holo_r_path)
            sequence_l = sequence_from_pdb(holo_l_path)
            colabfold_path = colabfold_dir / f"{row.id}.fasta"
            chain_r_path = chain_dir / f"{row.id}__R.fasta"
            chain_l_path = chain_dir / f"{row.id}__L.fasta"

            outputs = [colabfold_path, chain_r_path, chain_l_path]
            if overwrite or not all(path.is_file() for path in outputs):
                write_text_atomic(
                    colabfold_path,
                    f">{row.id}\n{sequence_r}:{sequence_l}\n",
                )
                write_text_atomic(
                    chain_r_path,
                    f">{row.id}|R|{row.uniprot_R}\n{sequence_r}\n",
                )
                write_text_atomic(
                    chain_l_path,
                    f">{row.id}|L|{row.uniprot_L}\n{sequence_l}\n",
                )

            records.append(
                {
                    "pinder_id": row.id,
                    "sequence_rule": SEQUENCE_RULE,
                    "uniprot_R": row.uniprot_R,
                    "uniprot_L": row.uniprot_L,
                    "length_R": len(sequence_r),
                    "length_L": len(sequence_l),
                    "total_length": len(sequence_r) + len(sequence_l),
                    "sha256_R": sha256_text(sequence_r),
                    "sha256_L": sha256_text(sequence_l),
                    "colabfold_fasta": str(colabfold_path.relative_to(data_root)),
                    "chain_R_fasta": str(chain_r_path.relative_to(data_root)),
                    "chain_L_fasta": str(chain_l_path.relative_to(data_root)),
                }
            )
        except Exception as exc:
            failures.append({"pinder_id": row.id, "error": str(exc)})

    manifest_dir = data_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pinder_id",
        "sequence_rule",
        "uniprot_R",
        "uniprot_L",
        "length_R",
        "length_L",
        "total_length",
        "sha256_R",
        "sha256_L",
        "colabfold_fasta",
        "chain_R_fasta",
        "chain_L_fasta",
    ]
    with (manifest_dir / "fasta_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    with (manifest_dir / "fasta_failures.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["pinder_id", "error"])
        writer.writeheader()
        writer.writerows(failures)

    print(f"FASTA systems created: {len(records)}")
    print(f"Failures:              {len(failures)}")
    print(f"Manifest: {manifest_dir / 'fasta_manifest.csv'}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        required=True,
        help="PINDER 2024-02 release directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        data_root = resolve_data_root(args.data_root)
        return build_fastas(data_root, args.overwrite)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
