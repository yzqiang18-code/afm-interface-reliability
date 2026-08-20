# Pairwise-Jaccard matrix artifacts

Inputs for the exploratory Jaccard-matrix neural network study (see
[`docs/NN_JACCARD.md`](../../../docs/NN_JACCARD.md)).

- `matrices.npz` — one symmetric 20×20 contact-Jaccard matrix per Training500
  system, in canonical `(model_weight, seed)` ascending slot order; diagonal
  entries are 1.0 and invalid pairs (0.11% of rows) are imputed with the
  per-system mean of valid off-diagonal values.
- `matrix_manifest.json` — input hashes, slot schema, invalid-pair and
  imputation statistics.

Built by `scripts/ml/build_jaccard_matrix.py` from
`results/data/training500_consistency_pairs.csv.gz`. The candidate-level row
table derived from these matrices is
`results/data/training500_jaccard_rows.csv.gz`.

Training artifacts live next to this directory:

- `../pairwise_jaccard_nn_1x19/` — 1×19 row-vector MLP (self-excluded).
- `../pairwise_jaccard_nn_1x20/` — 1×20 row-vector MLP (self diagonal included).

Both models are significantly worse than AF-M rank-1 on the Training500 top-1
endpoint; this is an exploratory negative result, not a frozen-holdout
evaluation (the public data has no holdout pair table).
