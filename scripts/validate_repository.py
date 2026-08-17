#!/usr/bin/env python3
"""Validate the public repository and its scientific release gates."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def non_comment_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if path.suffix == ".gz":
        handle = gzip.open(path, mode="rt", encoding="utf-8", newline="")
    else:
        handle = path.open(mode="r", encoding="utf-8", newline="")
    with handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(value: object, expected: float, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def main() -> int:
    errors: list[str] = []
    required = [
        ".github/workflows/ci.yml",
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "environment.yml",
        "analysis/README.md",
        "analysis/training500_analysis.py",
        "configs/README.md",
        "configs/cohorts/feasibility50_ids.txt",
        "configs/cohorts/training500_assignment.csv",
        "configs/cohorts/pinder_af2_holdout_180_ids.txt",
        "configs/ml/candidate_ridge_v1.json",
        "docs/METHODS.md",
        "docs/DATA.md",
        "docs/LIMITATIONS.md",
        "docs/RESULTS.md",
        "docs/REFERENCES.md",
        "docs/PROJECT_SUMMARY_CN.md",
        "results/README.md",
        "results/data/README.md",
        "results/data/data_manifest.json",
        "results/data/training500_candidates.csv.gz",
        "results/data/pinder_af2_180_labels.csv.gz",
        "results/data/training500_manifest.csv",
        "results/data/pinder_af2_180_manifest.csv",
        "results/audits/README.md",
        "results/audits/chain_exchange/chain_exchange_summary.json",
        "results/audits/chain_exchange/candidate_changes.csv.gz",
        "results/audits/leakage/leakage_summary.json",
        "results/audits/leakage/leakage_intersections.csv",
        "results/audits/leakage/uniprot_overlap_sensitivity.json",
        "results/summaries/training500_summary.json",
        "results/summaries/training500_data_quality.json",
        "results/summaries/reproduction_summary.json",
        "results/ml/candidate_ridge_v1/model.json",
        "results/ml/candidate_ridge_v1/training_summary.json",
        "results/ml/candidate_ridge_v1/evaluation_summary.json",
        "results/ml/candidate_ridge_v1/holdout/pinder_af2_180_predictions.csv",
        "results/ml/candidate_ridge_v1/holdout/selector_choices.csv",
        "scripts/data/prepare_public_data.py",
        "scripts/figures/make_figures.py",
        "scripts/reproduce_level2.py",
        "scripts/validate_repository.py",
        "tests/README.md",
        "tests/test_audit_pinder_af2_leakage.py",
        "tests/test_colabfold_msa_api.py",
        "tests/test_consistency_metrics.py",
        "tests/test_metric_batch_helpers.py",
        "tests/test_ml_baseline.py",
        "tests/test_physics_metrics.py",
        "tests/test_prepare_pinder_af2.py",
        "tests/test_run_colabfold_with_full_precision.py",
        "third_party/checksums.sha256",
        "figures/workflow.svg",
        "figures/top1_vs_oracle.svg",
        "figures/metric_performance.svg",
        "figures/ensemble_ablation.svg",
    ]
    for relative in required:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty required file: {relative}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    for line in non_comment_lines(ROOT / "third_party/checksums.sha256"):
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"vendored checksum mismatch: {relative}")

    cohort_counts = {
        "configs/cohorts/feasibility50_ids.txt": 50,
        "configs/cohorts/pinder_af2_holdout_180_ids.txt": 180,
    }
    for relative, expected in cohort_counts.items():
        values = non_comment_lines(ROOT / relative)
        if len(values) != expected or len(set(values)) != expected:
            errors.append(f"{relative}: expected {expected} unique IDs")

    _, assignment = csv_rows(ROOT / "configs/cohorts/training500_assignment.csv")
    assignment_ids = [row["id"] for row in assignment]
    if len(assignment) != 500 or len(set(assignment_ids)) != 500:
        errors.append("Training500 assignment must contain 500 unique systems")
    if len({row["cluster_id"] for row in assignment}) != 500:
        errors.append("Training500 assignment must contain 500 unique clusters")
    fold_counts = Counter(int(row["cv_fold"]) for row in assignment)
    if fold_counts != Counter({fold: 100 for fold in range(5)}):
        errors.append(f"Training500 fold counts are invalid: {dict(fold_counts)}")
    for row in assignment:
        parts = row["id"].split("--", 1)
        accessions = [part.rsplit("_", 1)[-1].upper() for part in parts]
        if "UNDEFINED" in accessions and row["same_uniprot"] == "True":
            errors.append(f"missing accession marked same_uniprot: {row['id']}")

    manifest = read_json("results/data/data_manifest.json")
    if manifest.get("schema_version") != 1:
        errors.append("public data manifest schema version must be 1")
    manifest_records = {row["path"]: row for row in manifest.get("artifacts", [])}
    expected_public = {
        "results/data/training500_candidates.csv.gz": (10000, 500),
        "results/data/pinder_af2_180_labels.csv.gz": (3600, 180),
        "results/data/training500_manifest.csv": (500, 500),
        "results/data/pinder_af2_180_manifest.csv": (180, 180),
    }
    public_rows: dict[str, list[dict[str, str]]] = {}
    for relative, (expected_rows, expected_systems) in expected_public.items():
        fields, rows = csv_rows(ROOT / relative)
        public_rows[relative] = rows
        record = manifest_records.get(relative)
        if record is None:
            errors.append(f"data manifest is missing {relative}")
        else:
            if int(record["rows"]) != expected_rows or record["sha256"] != sha256(ROOT / relative):
                errors.append(f"data manifest count/hash mismatch: {relative}")
        if len(rows) != expected_rows:
            errors.append(f"{relative}: expected {expected_rows} rows, found {len(rows)}")
        id_column = "complex_id" if "complex_id" in fields else "id"
        if len({row[id_column] for row in rows}) != expected_systems:
            errors.append(f"{relative}: expected {expected_systems} systems")

    for relative in [
        "results/data/training500_candidates.csv.gz",
        "results/data/pinder_af2_180_labels.csv.gz",
    ]:
        fields = set(public_rows[relative][0])
        required_fields = {
            "complex_id", "rank", "model_weight", "seed", "DockQ",
            "same_known_uniprot", "chain_exchange_eligible", "accession_status",
        }
        if not required_fields.issubset(fields):
            errors.append(f"{relative}: missing audit/model fields")
        counts = Counter(row["complex_id"] for row in public_rows[relative])
        if set(counts.values()) != {20}:
            errors.append(f"{relative}: every system must have exactly 20 candidates")

    model = read_json("results/ml/candidate_ridge_v1/model.json")
    if model.get("model_schema_version") != 2:
        errors.append("Candidate Ridge model schema must be 2")
    if set(assignment_ids) != set(model.get("training_system_ids", [])):
        errors.append("Training500 IDs differ from frozen model IDs")
    config_path = ROOT / "configs/ml/candidate_ridge_v1.json"
    if model["config_input"]["sha256"] != sha256(config_path):
        errors.append("frozen model config hash mismatch")
    training_path = ROOT / "results/data/training500_candidates.csv.gz"
    if model["training_input"]["sha256"] != sha256(training_path):
        errors.append("frozen model training-data hash mismatch")

    prediction_fields, predictions = csv_rows(
        ROOT / "results/ml/candidate_ridge_v1/holdout/pinder_af2_180_predictions.csv"
    )
    if len(predictions) != 3600:
        errors.append("holdout prediction artifact must have 3,600 rows")
    if "acceptable_score" not in prediction_fields or "acceptable_probability" in prediction_fields:
        errors.append("prediction artifact must use acceptable_score semantics")
    if "DockQ" in prediction_fields:
        errors.append("label-free prediction artifact unexpectedly contains DockQ")
    holdout_ids = set(non_comment_lines(ROOT / "configs/cohorts/pinder_af2_holdout_180_ids.txt"))
    if {row["complex_id"] for row in predictions} != holdout_ids:
        errors.append("holdout prediction IDs differ from frozen cohort")
    if set(assignment_ids) & holdout_ids:
        errors.append("exact Training500/PINDER-AF2 system overlap")

    training_summary = read_json("results/summaries/training500_summary.json")
    if training_summary["cohort"]["systems"] != 500 or training_summary["cohort"]["predictions"] != 10000:
        errors.append("Training500 summary counts do not match 500 / 10,000")
    if training_summary["cohort"]["same_uniprot_systems"] != 313:
        errors.append("Training500 known same-accession count must be 313")
    if not close(training_summary["selection"]["rank1_mean_DockQ"], 0.42434334416814606):
        errors.append("corrected Training500 mean DockQ is inconsistent")

    evaluation = read_json("results/ml/candidate_ridge_v1/evaluation_summary.json")
    selectors = evaluation.get("selector_metrics", {})
    if set(selectors) != {"model", "reference"}:
        errors.append("evaluation must contain model and reference selectors")
    else:
        for selector in selectors.values():
            if selector["systems"] != 180 or not close(selector["acceptable_rate"], 105 / 180):
                errors.append("holdout selector counts/rates are inconsistent")
        if not close(selectors["reference"]["mean_DockQ"], 0.4523995232915598):
            errors.append("corrected AF-M holdout mean DockQ is inconsistent")
        if not close(selectors["model"]["mean_DockQ"], 0.453850207225718):
            errors.append("corrected Candidate Ridge holdout mean DockQ is inconsistent")
    paired = evaluation["paired_bootstrap"]
    if not close(paired["delta_acceptable_rate"], 0.0):
        errors.append("primary holdout effect must remain zero")
    if not (paired["delta_mean_DockQ_ci_low"] <= 0 <= paired["delta_mean_DockQ_ci_high"]):
        errors.append("mean-DockQ confidence interval must contain zero")

    chain = read_json("results/audits/chain_exchange/chain_exchange_summary.json")
    expected_chain = {
        "training": (130, 11, 0),
        "holdout": (299, 25, 0),
    }
    for cohort, (rows, systems, transitions) in expected_chain.items():
        audit = chain[cohort]
        if (
            audit["continuous_dockq_changed_rows"] != rows
            or audit["continuous_dockq_changed_systems"] != systems
            or audit["primary_class_changed_rows"] != transitions
            or audit["primary_class_changed_systems"] != transitions
        ):
            errors.append(f"chain-exchange audit changed unexpectedly: {cohort}")

    leakage = read_json("results/audits/leakage/leakage_summary.json")
    expected_intersections = {
        "id": 0, "pdb_id": 0, "cluster_id": 0,
        "chain_cluster_pair": 0, "uniprot_pair": 1,
    }
    if leakage.get("intersection_counts") != expected_intersections:
        errors.append("split-overlap counts changed unexpectedly")
    if not leakage.get("release_gate_passed"):
        errors.append("split-overlap release gate failed")
    sensitivity = read_json("results/audits/leakage/uniprot_overlap_sensitivity.json")
    if sensitivity.get("excluded_holdout_system_ids") != [
        "8cnx__A1_Q68T42--8cnx__B1_Q68T42"
    ]:
        errors.append("UniProt-overlap sensitivity excludes the wrong system")
    for selector in sensitivity.get("selector_metrics", {}).values():
        if selector["systems"] != 179 or not close(selector["acceptable_rate"], 105 / 179):
            errors.append("UniProt sensitivity counts/rates are inconsistent")

    reproduction = read_json("results/summaries/reproduction_summary.json")
    input_hashes = reproduction.get("inputs", {})
    for relative in [
        "configs/ml/candidate_ridge_v1.json",
        "results/data/training500_candidates.csv.gz",
        "results/data/pinder_af2_180_labels.csv.gz",
    ]:
        if input_hashes.get(relative) != sha256(ROOT / relative):
            errors.append(f"reproduction input hash mismatch: {relative}")

    model_payload = json.loads(
        (ROOT / "results/ml/candidate_ridge_v1/model.json").read_text(encoding="utf-8")
    )
    canonical = dict(model_payload)
    canonical.pop("created_at_utc", None)
    canonical_bytes = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if reproduction.get("model_sha256") != hashlib.sha256(canonical_bytes).hexdigest():
        errors.append(
            "reproduction model hash does not match the canonical model content"
        )

    private_markers = (b"/Users/", b"/mnt/", b"indicator_testing", b"indicator-testing", b"indicator_")
    text_extensions = {".md", ".py", ".json", ".csv", ".yml", ".yaml", ".cff", ".txt", ".svg"}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in {".git", "__pycache__"} for part in relative.parts):
            continue
        if relative == Path("scripts/validate_repository.py"):
            continue
        if path.suffix.lower() in {".a3m", ".pkl", ".pickle"}:
            errors.append(f"unexpected raw asset: {relative}")
        if path.suffix.lower() in {".pdb", ".cif", ".mmcif"} and not str(relative).startswith("tests/fixtures/"):
            errors.append(f"unexpected structure asset: {relative}")
        if path.stat().st_size > 25 * 1024 * 1024:
            errors.append(f"unexpected file larger than 25 MiB: {relative}")
        if path.suffix.lower() in text_extensions:
            content = path.read_bytes()
            if any(marker in content for marker in private_markers):
                errors.append(f"private or legacy path/name in {relative}")

    for relative in [
        "results/data/training500_candidates.csv.gz",
        "results/data/pinder_af2_180_labels.csv.gz",
        "results/audits/chain_exchange/candidate_changes.csv.gz",
    ]:
        with gzip.open(ROOT / relative, "rb") as handle:
            content = handle.read()
        if any(marker in content for marker in private_markers):
            errors.append(f"private path in compressed artifact: {relative}")

    markdown_link = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for markdown in ROOT.rglob("*.md"):
        for target in markdown_link.findall(markdown.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            clean = target.split("#", 1)[0]
            if clean and not (markdown.parent / clean).resolve().exists():
                errors.append(f"broken local link in {markdown.relative_to(ROOT)}: {target}")

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    if "REPLACE_WITH_PUBLIC_REPOSITORY_URL" in citation:
        errors.append("CITATION.cff contains an unresolved URL placeholder")

    test_count = sum(
        len(re.findall(r"^\s+def test_", path.read_text(encoding="utf-8"), re.MULTILINE))
        for path in (ROOT / "tests").glob("test_*.py")
    )
    print(f"Repository root: {ROOT}")
    print(f"Test methods: {test_count}")
    print(f"Files: {sum(1 for path in ROOT.rglob('*') if path.is_file())}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        print(f"FAILED with {len(errors)} error(s)", file=sys.stderr)
        return 1
    print("PASS: repository and scientific release gates succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
