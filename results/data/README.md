# Public reproduction data

- `training500_candidates.csv.gz`: 10,000 development candidates from 500
  systems.
- `pinder_af2_180_labels.csv.gz`: 3,600 held-out candidates from 180 systems.
- `*_manifest.csv`: system provenance used by overlap audits.
- `data_manifest.json`: schemas, row counts, hashes, and preparation metadata.

These are derived analysis tables, not raw PINDER structures. `DockQ` and
related native-referenced fields are labels; they must not be used as
prediction-time features. Run `python scripts/reproduce_level2.py` from the
repository root to retrain and evaluate the public baseline.
