# Public reproduction data

- `training500_candidates.csv.gz`: 10,000 development candidates from 500
  systems.
- `pinder_af2_180_labels.csv.gz`: 3,600 held-out candidates from 180 systems.
- `training500_consistency_pairs.csv.gz`: 95,000 path-free pairwise agreement
  rows (190 unordered candidate pairs for each of 500 Training500 systems), used
  only by the exploratory consistency and cluster diagnostic.
- `*_manifest.csv`: system provenance used by overlap audits.
- `data_manifest.json`: schemas, row counts, hashes, and preparation metadata.

These are derived analysis tables, not raw PINDER structures. `DockQ` and
related native-referenced fields are labels; they must not be used as
prediction-time features. Run `python scripts/reproduce_level2.py` from the
repository root to retrain and evaluate the public baseline.

The two candidate-level tables are sufficient for baseline reproduction. The
pair table is an auxiliary diagnostic artifact; its candidate identities are
`(complex_id, model_weight, seed)`, and no machine-specific prediction paths are
published.
