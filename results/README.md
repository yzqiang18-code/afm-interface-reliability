# Curated results

This directory contains the compact, reviewable evidence for the project.

- `data/`: compressed candidate-level inputs and provenance manifests for the
  public reproduction;
- `audits/`: chain-assignment correction and split-overlap evidence;
- `summaries/`: Training500 and end-to-end reproduction summaries;
- `tables/`: selected exact-value Training500 tables;
- `ml/`: the frozen model, label-free holdout predictions, and evaluation.

Training500 is a deliberately stratified development cohort. The
[`ml/candidate_ridge_v1/`](ml/candidate_ridge_v1/README.md) artifact records the
frozen PINDER-AF2 evaluation. Model-ready derived tables are included; native
structures, predicted PDBs, PAE JSON, and MSA assets are excluded.
