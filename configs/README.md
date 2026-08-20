# Frozen cohorts and model configuration

This directory contains only the scientific cohort records and frozen model
configuration needed to interpret the public results. Machine-specific GPU
assignments and temporary workload-rebalancing files are excluded.

## Reader-facing cohort files

| File | Role | Expected systems |
|---|---|---:|
| `cohorts/feasibility50_ids.txt` | Pipeline/metric feasibility cohort | 50 |
| `cohorts/training500_assignment.csv` | Stratified Training500 cohort, provenance, and five-fold assignment | 500 |
| `cohorts/pinder_af2_holdout_180_ids.txt` | Frozen held-out PINDER-AF2 cohort | 180 |

## Model configuration

`ml/candidate_ridge_v1.json` records the frozen feature set, preprocessing
policy, grouped-fold column, label definition, ridge penalty, and bootstrap
settings used for Candidate Ridge v1. The `candidate_ridge_v1_within_rank` and
`candidate_ridge_v2_group_softmax*` configurations record exploratory
within-system preprocessing and conditional-logit follow-ups. Their presence
does not imply a significant improvement over AF-M; exact development and
holdout results are summarized in [`docs/RESULTS.md`](../docs/RESULTS.md).

Training500 is a deliberately stratified development cohort, not a random or
prevalence-representative PINDER-Val sample. The PINDER-AF2 list records the
completed frozen held-out evaluation cohort and must remain unchanged. The
`same_uniprot` field means equality of two known accessions; missing accession
tokens do not count as equal. Correcting that audit field did not change cohort
membership, ordering, folds, or selection hashes.

PINDER-AF2 has now been inspected for multiple model variants. It remains a
fixed held-out cohort for reporting these results, but future feature or model
selection should not treat it as a new untouched confirmatory test set.
