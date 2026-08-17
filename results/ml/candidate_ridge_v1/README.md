# Candidate Ridge v1 artifacts and results

This directory contains the compact public artifact for the frozen five-feature ridge-logistic candidate reranker re-evaluated on 2026-08-16 after the chain-assignment correction.

## Frozen model

[`model.json`](model.json) contains the ordered features, missing-value medians,
standardization parameters, intercept, coefficients, solver audit, public input
hashes, and frozen Training500 system IDs. The label-free prediction table uses
`acceptable_score`, a logistic ranking score rather than a claimed calibrated
probability.

## Development result: Training500 grouped OOF

| Endpoint | AF-M rank-1 | Candidate Ridge v1 | Paired difference |
|---|---:|---:|---:|
| Acceptable rate (`DockQ >= 0.23`) | 0.600 | 0.604 | +0.004 (95% CI −0.006 to +0.014) |
| Mean DockQ | 0.42434 | 0.42680 | +0.00245 (95% CI −0.00377 to +0.00837) |

Candidate-level grouped-OOF metrics were ROC-AUC `0.854245`, average precision `0.817633`, and Brier score `0.150955`. The selector changed 111 of 500 choices, improving DockQ in 62 systems and worsening it in 49; it rescued five unacceptable AF-M choices and harmed three acceptable choices.

[`training_summary.json`](training_summary.json) records the exact aggregate metrics, paired bootstrap, fold audit, and solver audits.

## Frozen holdout result: PINDER-AF2 180

| Endpoint | AF-M rank-1 | Candidate Ridge v1 | Paired difference |
|---|---:|---:|---:|
| Acceptable rate (`DockQ >= 0.23`) | 0.58333 | 0.58333 | 0.00000 |
| Mean DockQ | 0.45240 | 0.45385 | +0.00145 (95% CI −0.00129 to +0.00472) |
| Median DockQ | 0.59310 | 0.59421 | — |

Candidate-level holdout metrics were ROC-AUC `0.933849`, average precision `0.936705`, and Brier score `0.100976`. The model changed 41 of 180 selected candidates (18 higher-DockQ, 23 lower-DockQ), but every changed pair stayed on the same side of the acceptable threshold. Thus the learned reranker did not improve the primary selected-structure success rate.

The [`holdout/`](holdout/) directory contains the label-free 3,600-row frozen
prediction table and selected-candidate evidence.
[`evaluation_summary.json`](evaluation_summary.json) records the exact endpoint
summaries, paired bootstrap, 180 × 20 data audit, and exact system/PDB-ID gate.
The broader audit under [`results/audits/leakage/`](../../audits/leakage/)
also checks interface clusters, unordered chain-cluster pairs, and known
UniProt pairs.

## Public inputs and excluded raw assets

The 10,000-row Training500 candidate table and 3,600-row labeled holdout table
are distributed in [`results/data/`](../../data/README.md), so the model can be
retrained and evaluated on CPU. Native structures, MSAs, PAE JSON files, and
raw AF-M outputs are not distributed.

## Scientific interpretation

The negative top-1 result is retained intentionally. Candidate-level discrimination can be high while within-system ordering remains difficult because candidates from the same protein pair are strongly correlated and often receive similar confidence signals. This artifact therefore supports a narrower conclusion: the fixed five-feature linear baseline is reproducible, but it does not beat AF-M's own full-precision ranking on the frozen PINDER-AF2 selected-structure endpoint.

The model definition and evaluation design are summarized in
[Methods](../../../docs/METHODS.md); headline outcomes are collected in
[Results](../../../docs/RESULTS.md).
