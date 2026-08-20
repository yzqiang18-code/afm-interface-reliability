# Curated results

This directory contains the compact, reviewable evidence for the project.

- `data/`: compressed candidate-level inputs, a path-free candidate-pair
  diagnostic table, and provenance manifests for the public reproduction;
- `audits/`: chain-assignment correction and split-overlap evidence;
- `summaries/`: Training500 and end-to-end reproduction summaries;
- `tables/`: selected exact-value Training500 tables;
- `ml/`: the frozen model, label-free holdout predictions, and evaluation.

Training500 is a deliberately stratified development cohort. The
[`ml/candidate_ridge_v1/`](ml/candidate_ridge_v1/README.md) artifact records the
original frozen PINDER-AF2 evaluation; adjacent `candidate_ridge_v1_within_rank`
and `candidate_ridge_v2_group_softmax` directories record later
ranking-focused follow-ups. None demonstrates a statistically significant
improvement in the primary top-1 endpoint.

The selected tables include post-hoc diagnostics for the outcome-defined
rerank-rescuable subset. They show modest, metric-dependent front-of-list
enrichment and are descriptive rather than confirmatory. Ensemble consistency
is reported separately as a reproducible system-level trust signal, including
stable-wrong systems that are highly consistent while every candidate is
incorrect.

Model-ready derived tables are included; native structures, predicted PDBs, PAE
JSON, and MSA assets are excluded.
