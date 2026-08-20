# Pairwise-Jaccard matrix neural network

This note documents an exploratory Training500 study: build a 20×20
within-system contact-Jaccard matrix from the public pair table, train a small
shared-weight PyTorch MLP on the matrix rows (1×19 or 1×20 per candidate), and
rerank the 20 candidates of each system. It is a direct test of whether the
*matrix form* of ensemble consistency adds within-system selection signal over
the scalar summaries already studied.

## Data and formulation

- Input: [`training500_consistency_pairs.csv.gz`](../results/data/training500_consistency_pairs.csv.gz)
  (95,000 unordered pairs; 190 per system). `jaccard_valid` is `True` for
  99.89% of pairs; the 108 invalid pairs (0.11%) are NaN and are imputed with
  the per-system mean of valid off-diagonal entries.
- Matrix: one symmetric 20×20 contact-Jaccard matrix per system, in a fixed
  canonical slot order `(model_weight, seed)` ascending (`model_weight ∈
  {1..5}`, `seed ∈ {0..3}`; every system has all 20 slots). Diagonal entries
  are 1.0 (self-similarity).
- Candidate rows: each candidate is represented by its row of the matrix —
  `j_0..j_18` (1×19, all *other* slots) or `j20_0..j20_19` (1×20, including the
  self diagonal). Because the same network scores every row, the scoring is
  permutation-equivariant in candidate order.
- Derived artifact: [`training500_jaccard_rows.csv.gz`](../results/data/training500_jaccard_rows.csv.gz)
  (one row per candidate: keys, folds, labels, the two row-feature sets, and
  `mean_j`); matrices and schema statistics are under
  [`results/ml/pairwise_jaccard_nn/`](../results/ml/pairwise_jaccard_nn/).

Build with:

```bash
python scripts/ml/build_jaccard_matrix.py
```

## Model and evaluation

- Architecture: input → `Linear(19 → 32) → ReLU → Linear(32 → 16) → ReLU →
  Linear(16 → 1)`, `BCEWithLogits` on `DockQ >= 0.23`, Adam (`lr = 0.01`,
  weight decay `1e-4`), 80 epochs, batch 256. Same config for the 1×20 variant.
- Grouped 5-fold CV on Training500 using the frozen `cv_fold` column:
  candidates of one system are never split across folds. OOF scores select one
  candidate per system by argmax.
- Training runs on CPU with PyTorch 2.5.1 (`conda env physics_ai`), e.g.:

```bash
python scripts/ml/train_jaccard_nn.py \
  --train-csv results/data/training500_jaccard_rows.csv.gz \
  --config configs/ml/pairwise_jaccard_nn_1x19.json \
  --output-dir results/ml/pairwise_jaccard_nn_1x19 --device cpu
```

- References: AF-M rank-1 (`ranking_confidence`) and regenerated grouped-OOF
  ridge baselines (Candidate Ridge v1's five native features; `mean_j` alone;
  the five native features + `mean_j`), all under the same grouped fold
  discipline. The regenerated v1 OOF reproduces the committed v1 numbers
  exactly (acceptable rate 0.604, ROC-AUC 0.854), so the comparisons are on the
  same footing.

## Results (Training500 grouped OOF)

### Selector endpoints

| Endpoint | AF-M rank-1 | 1×19 NN | 1×20 NN |
| --- | ---: | ---: | ---: |
| Acceptable rate (`DockQ >= 0.23`) | 0.600 | **0.540** | **0.532** |
| Mean DockQ | 0.42434 | 0.37265 | 0.36893 |
| Paired acceptable-rate difference | — | −0.060 (95% CI −0.092 to −0.032) | −0.068 (95% CI −0.104 to −0.034) |

Both neural models are **significantly worse** than AF-M rank-1 on the primary
top-1 endpoint, and worse than every scalar ridge reference:

| Pairing | Δ acceptable rate (95% CI) |
| --- | ---: |
| 1×19 NN vs ridge `mean_j` | −0.020 (−0.044 to +0.006) |
| 1×19 NN vs ridge five-native + `mean_j` | −0.058 (−0.086 to −0.030) |
| 1×19 NN vs Candidate Ridge v1 | −0.064 (−0.094 to −0.034) |
| 1×20 NN vs Candidate Ridge v1 | −0.072 |

### Candidate-level discrimination

| Model | ROC-AUC | Average precision | Brier |
| --- | ---: | ---: | ---: |
| 1×19 NN | 0.716 | 0.657 | 0.231 |
| 1×20 NN | 0.705 | 0.645 | 0.241 |
| Candidate Ridge v1 (regenerated) | 0.854 | 0.818 | 0.151 |

The Jaccard-row features alone are much weaker discriminators than AF-M's own
five native-confidence features, consistent with consistency being a
*trust* signal rather than a *selection* signal.

### Within-system ranking

| Selector | Median within-Spearman | Recall@3 (all 500) | Recall@1 | Recall@3 | Median first-acceptable rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| AF-M rank-1 confidence | 0.227 | 0.827 | 0 / 88 | 23.9% | 6 |
| Candidate Ridge v1 | 0.206 | 0.843 | 5 / 88 | 30.7% | 6 |
| 1×19 NN | 0.044 | 0.763 | 16 / 88 | 31.8% | 11 |
| 1×20 NN | 0.065 | 0.760 | 23 / 88 | 42.0% | 11 |

The right three columns are the 88 rerank-rescuable systems. The NN genuinely
promotes acceptable candidates *within* rescuable systems — 1×20 reaches
recall@3 42.0% vs 23.9% for AF-M — but its within-system Spearman is near zero
overall and its rescues come at the cost of breaking far more
rank-1-acceptable systems, so the net top-1 effect is negative and significant.

## Interpretation

- The matrix/row representation of within-system Jaccard does **not** beat AF-M
  rank-1 or any scalar pairwise/native ridge on the top-1 endpoint; on this
  cohort it is significantly worse.
- The rescuable-subset enrichment (recall@1/3 on the 88 rescuable systems) is a
  real but insufficient signal: it cannot be harvested without a reliable
  system-level gate that decides *when* to trust the consistency-based rerank,
  which is exactly the separate system-level trust direction the project
  already identifies.
- Consistent with the project's central finding: ensemble consistency explains
  where acceptable candidates cluster but not which candidate is best.

## Limitations

- **Exploratory, Training500 only.** The public data has no PINDER-AF2 pair
  table (only candidate labels; predicted structures are not published), so no
  frozen-holdout evaluation was run. Any confirmatory claim would require
  regenerating holdout pair rows from the full pipeline.
- The architecture and hyper-parameters were chosen on Training500; the negative
  result therefore does not generalize by construction, but the failure mode is
  informative and matches the established negative finding.
- Inputs use contact Jaccard only (single channel); interface-residue Jaccard,
  CB8 Jaccard, and RMSD remain untested as additional channels.

## Artifacts

- [`results/data/training500_jaccard_rows.csv.gz`](../results/data/training500_jaccard_rows.csv.gz)
  — candidate row-vector table (10,000 rows, 500 systems).
- [`results/ml/pairwise_jaccard_nn/matrices.npz`](../results/ml/pairwise_jaccard_nn/matrices.npz)
  — 20×20 matrices; `matrix_manifest.json` — schema and imputation statistics.
- [`results/ml/pairwise_jaccard_nn_1x19/`](../results/ml/pairwise_jaccard_nn_1x19/)
  and `.../pairwise_jaccard_nn_1x20/` — model artifacts, per-fold checkpoints,
  OOF predictions, selector and paired-bootstrap tables, and rerank metrics.
- Configs: `configs/ml/pairwise_jaccard_nn_1x19.json` and `..._1x20.json`.
- Code: `scripts/ml/build_jaccard_matrix.py`, `scripts/ml/train_jaccard_nn.py`;
  tests: `tests/test_jaccard_matrix.py`, `tests/test_jaccard_nn.py`.
