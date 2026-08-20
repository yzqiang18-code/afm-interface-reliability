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

## Within-system ranking follow-ups

The pooled candidate-level ROC-AUC evaluates discrimination across all
candidates and systems; it is not evidence of effective ranking among the 20
candidates of one system. Two follow-ups therefore changed the preprocessing or
loss to target within-system comparisons more directly.

The within-system quantile-rank model increased the Training500 out-of-fold
median within-system Spearman from `0.227` for AF-M ranking and `0.206` for
Candidate Ridge v1 to `0.243`. The median-Spearman difference was not
statistically significant: the system-level paired 95% CI was `-0.026` to
`+0.047` versus AF-M and `-0.018` to `+0.061` versus v1. Its out-of-fold
acceptable rate was 60.4% versus 60.0% for AF-M (95% CI for the difference
`-1.0` to `+1.8` percentage points).

The conditional-logit group-softmax model was fitted only on development
systems containing both acceptable and unacceptable candidates. Its
Training500 out-of-fold acceptable rate was 60.2%, a `+0.2` percentage-point
difference from AF-M (95% CI `-0.8` to `+1.2`). On PINDER-AF2, within-rank
selected 106/180 acceptable structures and group-softmax selected 105/180,
compared with 105/180 for AF-M. None of these results establishes a significant
improvement in the primary top-1 endpoint.

## Exploratory rerank-rescuable subset

The 88 Training500 systems in which AF-M rank-1 was unacceptable but at least
one acceptable candidate had been sampled were examined post hoc. The most
direct descriptive question is how far down each selector's list one must go to
find the first acceptable candidate:

| Selector | Mean first-acceptable rank | Median rank | Recall@3 | Recall@5 | Acceptable at rank 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AF-M ranking | 7.51 | 6 | 23.9% | 47.7% | 0 / 88 |
| Within-system ranks | 7.08 | 6 | 34.1% | 46.6% | 7 / 88 |
| Group-softmax | 7.32 | 6 | 30.7% | 45.5% | 4 / 88 |

Within-system ranks and group-softmax show modest front-of-list enrichment,
most clearly at recall@3, but the median rank was unchanged and recall@5 did not
improve. The 0/88 AF-M rank-1 value follows from the definition of this subset
and is not an inferential comparison. On the 12-system holdout subset, the mean
first-acceptable ranks were 7.58 for AF-M, 6.75 for within-rank, and 6.92 for
group-softmax; this small, repeatedly inspected subset is descriptive only.
These metric-dependent observations motivate further ranking work but do not
support a claim of general or statistically significant ranking improvement.

## Exploratory Jaccard-matrix neural network

A further exploratory Training500 study built a 20×20 within-system
contact-Jaccard matrix per system and trained a small shared-weight PyTorch MLP
on its rows (1×19 and 1×20 per candidate), reranking by argmax under the same
grouped five-fold discipline. The result is negative and significant on the
primary endpoint: the 1×19 and 1×20 models selected an acceptable structure for
54.0% and 53.2% of systems versus 60.0% for AF-M rank-1 (paired differences
`-6.0` and `-6.8` percentage points; 95% CIs `-9.2 to -3.2` and `-10.4 to -3.4`).
The models were also worse than a scalar `mean_j` ridge and than Candidate Ridge
v1 (regenerated grouped OOF, which reproduced the committed v1 numbers exactly).
Candidate-level ROC-AUC was 0.716/0.705 versus 0.854 for v1. The one positive
sub-signal was inside the 88 rerank-rescuable systems, where the 1×20 model
reached recall@3 of 42.0% (vs 23.9% for AF-M) and rescued 23/88 rank-1
failures — but it broke many more rank-1-acceptable systems, so net top-1
performance was significantly worse. The matrix form of consistency therefore
does not beat AF-M's own confidence or scalar summaries on selection; it again
reinforces that consistency locates where acceptable candidates cluster, not
which candidate is best. This study is Training500-only because the public data
has no holdout pair table. Full details are in [NN_JACCARD.md](NN_JACCARD.md).

## Exploratory cluster structure

A separate Training500 diagnostic asked why the acceptable minority remains
hard to promote. In the 88 rerank-rescuable systems, the 20 candidates formed a
median of 7.5 contact-map clusters. The cluster containing the largest number of
acceptable candidates had median acceptable-candidate purity 1.00, showing that
acceptability was strongly cluster-structured. It was not usually the
highest-confidence cluster: its median rank by cluster-mean ipTM was 2.5, only
14.8% ranked first, and the top-ipTM cluster was entirely unacceptable in 84.1%
of rescuable systems.

This structure did not yield a successful hand-built selector. Across the
tested consistency interactions, low-consistency gates, second-ipTM/minority-
cluster rules, and hydrophobic-cluster rules, overall top-1 performance was
negative or tied relative to AF-M rank-1; cluster-gated save-minus-break counts
ranged from `-15` to `-45`. The analysis is post hoc and uses DockQ labels to
identify the acceptable cluster, so it is mechanistic evidence and motivation,
not a deployable selection rule. It suggests that future models need signals
beyond the current scalar AF-M/PAE summaries, motivating paired-MSA
coevolution and richer residue-pair PAE representations. The full diagnostic is
in [`pairwise_consistency_diagnostic.py`](../analysis/pairwise_consistency_diagnostic.py)
and uses the public path-free
[`training500_consistency_pairs.csv.gz`](../results/data/training500_consistency_pairs.csv.gz).

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
consistent while all 20 candidates were incorrect; the same stable-wrong pattern
occurred in four holdout systems. On the holdout, only 8% of the
low-consistency tercile had an acceptable rank-1, compared with 95% of the
high-consistency tercile. Ensemble convergence is therefore a reproducible
trust signal, but it cannot be treated as a correctness guarantee or used by
itself to reorder candidates within one system.

## Evidence

- [Reproduction summary](../results/summaries/reproduction_summary.json)
- [Training500 summary](../results/summaries/training500_summary.json)
- [Chain-assignment audit](../results/audits/chain_exchange/chain_exchange_summary.json)
- [Split-overlap audit](../results/audits/leakage/leakage_summary.json)
- [Candidate Ridge evaluation](../results/ml/candidate_ridge_v1/README.md)
- [Within-rank training summary](../results/ml/candidate_ridge_v1_within_rank/training_summary.json)
- [Within-rank holdout summary](../results/ml/candidate_ridge_v1_within_rank/holdout/evaluation_summary.json)
- [Group-softmax training summary](../results/ml/candidate_ridge_v2_group_softmax/training_summary.json)
- [Group-softmax holdout summary](../results/ml/candidate_ridge_v2_group_softmax/holdout/evaluation_summary.json)
- [Rerank-rescuable ranking diagnostics](../results/tables/rerank_rescuable_ranking.csv)
- [Training500 candidate-pair diagnostic data](../results/data/training500_consistency_pairs.csv.gz)
- [Selected result tables](../results/tables/README.md)
- [Jaccard-matrix neural network](NN_JACCARD.md)
- [Jaccard-matrix rows data](../results/data/training500_jaccard_rows.csv.gz)
- [Jaccard-matrix NN training summary](../results/ml/pairwise_jaccard_nn_1x19/training_summary.json)
