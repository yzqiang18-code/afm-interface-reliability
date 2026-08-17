# Results

`DockQ >= 0.23` defines an acceptable interface. Training500 is a stratified
development cohort, so its rates are not population estimates for PINDER-Val.

## Training500

All 500 systems had 20 candidates, four seeds, five model weights, ranks 1–20,
unique candidate keys, complete metric joins, and the expected 95,000
within-system pairs.

| Outcome | Systems | Rate |
| --- | ---: | ---: |
| AF-M full-precision top-1 acceptable | 300 | 60.0% |
| Oracle best-of-20 acceptable | 388 | 77.6% |
| Acceptable candidate sampled but missed by AF-M top-1 | 88 | 17.6% |
| No acceptable candidate among 20 | 112 | 22.4% |

AF-M rank-1 mean DockQ was `0.42434`; oracle mean DockQ was `0.52327`.
Candidate-level ranking confidence correlated with DockQ (`Spearman rho =
0.661`), but its median within-system correlation was only `0.227`. Candidate
Ridge's grouped out-of-fold acceptable rate was 60.4% versus 60.0% for AF-M;
the paired difference was `+0.4` percentage points (95% CI `-0.6` to `+1.4`).

## Frozen PINDER-AF2 evaluation

| Endpoint | AF-M rank-1 | Candidate Ridge v1 |
| --- | ---: | ---: |
| Acceptable systems | 105 / 180 | 105 / 180 |
| Acceptable rate | 58.33% | 58.33% |
| Mean DockQ | 0.45240 | 0.45385 |
| Median DockQ | 0.59310 | 0.59421 |

The paired mean-DockQ difference was `+0.00145` (95% bootstrap CI `-0.00129`
to `+0.00472`). Candidate-level ROC-AUC was `0.93385`, average precision was
`0.93670`, and Brier score was `0.10098`. The model therefore discriminated
acceptable candidates across the cohort but did not improve the primary
selected-structure success rate within systems.

## Sensitivity and audits

Correcting chain-exchange eligibility changed some continuous DockQ values but
caused zero primary-label transitions in either cohort. The split audit found
zero exact-system, PDB, interface-cluster, or chain-cluster-pair intersections.
It found one known UniProt-pair intersection. After excluding the affected
PINDER-AF2 system, both selectors remained at 58.66% acceptable; the mean-DockQ
difference was `+0.00146`. The negative conclusion is unchanged.

## Ensemble finding

In exploratory grouped development analysis, adding ensemble features changed
system-risk ROC-AUC from `0.796` to `0.812`. Consistency is a system-level trust
signal, not a selection signal: across Training500 and the frozen PINDER-AF2
holdout, the within-system acceptable-candidate fraction correlates with
contact-map agreement (`Spearman ρ = 0.655` and `0.800`; see
[Consistency](CONSISTENCY.md)). Seventeen Training500 systems were highly
consistent while all 20 candidates were incorrect. Ensemble convergence
therefore cannot be treated as correctness.

## Evidence

- [Reproduction summary](../results/summaries/reproduction_summary.json)
- [Training500 summary](../results/summaries/training500_summary.json)
- [Chain-assignment audit](../results/audits/chain_exchange/chain_exchange_summary.json)
- [Split-overlap audit](../results/audits/leakage/leakage_summary.json)
- [Candidate Ridge evaluation](../results/ml/candidate_ridge_v1/README.md)
- [Selected result tables](../results/tables/README.md)
