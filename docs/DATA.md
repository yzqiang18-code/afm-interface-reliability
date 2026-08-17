# Data and artifacts

## Cohorts

The study uses the public PINDER 2024-02 release.

| Cohort | Definition | Frozen record |
| --- | --- | --- |
| Feasibility50 | Fixed pipeline-validation subset | `configs/cohorts/feasibility50_ids.txt` |
| Training500 | Stratified development subset | `configs/cohorts/training500_assignment.csv` |
| PINDER-AF2 | Frozen held-out subset | `configs/cohorts/pinder_af2_holdout_180_ids.txt` |

Training500 deliberately enriches difficult and rerank-informative systems. Its
event rates are therefore descriptive of this cohort, not PINDER-Val prevalence
estimates.

## Public derived tables

Two compressed CSVs make the Level-2 analysis reproducible without distributing
large structural files:

| File | Rows | Purpose |
| --- | ---: | --- |
| `results/data/training500_candidates.csv.gz` | 10,000 | Development features, labels, folds, and audit fields |
| `results/data/pinder_af2_180_labels.csv.gz` | 3,600 | Frozen held-out features and labels for evaluation |

Each row represents one AF-M candidate. Candidate identity is the tuple
`(complex_id, model_weight, seed)`. `DockQ` is native-referenced and must never
be used as a prediction-time input. `acceptable_score` in generated prediction
files is a logistic decision score in `[0, 1]`; it is used for ranking and is
not claimed to be a calibrated probability.

The corresponding manifest CSVs contain release, cohort, PDB, cluster, chain
cluster, and accession provenance. `data_manifest.json` records row counts,
schemas, SHA-256 hashes, and the preparation rule. The compressed files use a
fixed gzip timestamp so repeated preparation is byte-reproducible.

The public tables were prepared by `scripts/data/prepare_public_data.py` from
the private analysis outputs and the official PINDER 2024-02 index. No machine
paths or credentials are stored in the output. The script is provided for
provenance; most readers should use `scripts/reproduce_level2.py` directly.

## Included and excluded material

Included:

- cohort records and grouped folds;
- the two derived candidate tables and provenance manifests;
- model configuration, frozen model, label-free predictions, summaries, and
  audit tables;
- code, tests, deterministic data-figure scripts, and vendored metric sources.

Excluded:

- native/holo structures and release parquet files;
- predicted PDB files, PAE/score JSON, FASTA, A3M/MSA archives, and AlphaFold
  parameters;
- credentials, scheduler state, logs, and transfer archives.

## Licensing and attribution

PINDER's official project is distributed under Apache-2.0 and documents the
public 2024-02 release. This repository's MIT license applies only to
project-authored code. PINDER-derived identifiers and annotations, structural
sources, and third-party metrics retain their upstream attribution and terms.
Users redistributing or extending the data should review the current PINDER and
RCSB terms themselves. Upstream links and citations are in
[REFERENCES.md](REFERENCES.md).
