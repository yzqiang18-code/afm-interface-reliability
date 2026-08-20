# Consistency as a system-level trust signal

This note records a core secondary finding: **ensemble consistency tracks whether
AF-M is likely to be right (a system-level trust signal), not which of the 20
candidates is best (a selection signal).** It complements the primary negative
result of the baseline reranker and explains where selection failures occur.

## Definitions

- **Consistency features** measure agreement within the 20-candidate ensemble:
  `mean_contact_jaccard` (mean contact-map agreement),
  `max_interface_cluster_fraction`, `median_receptor_aligned_ligand_rmsd`, and
  `iptm_std_across_models`. These are **system-level**: every candidate in a
  system shares the same value.
- **Acceptable** means `DockQ >= 0.23`.
- **Failure modes** of the AF-M rank-1 selection:
  - *Sampling failure* (`never_acceptable`): no acceptable candidate among the 20.
  - *Rerank-rescuable* (`oracle_rescue`): rank-1 is not acceptable but at least
    one acceptable candidate exists among the 20.
  - *Stable wrong*: `never_acceptable` and `mean_contact_jaccard >= 0.80` and
    `max_interface_cluster_fraction >= 0.80` (a highly consistent ensemble that
    converged on an incorrect structure).

## Evidence on Training500

System-level Spearman correlations with `mean_contact_jaccard`, across the 500
systems:

| Quantity | Spearman ρ |
| --- | ---: |
| within-system acceptable fraction | **0.655** |
| rank-1 acceptability (0/1) | **0.437** |
| within-system DockQ range | **−0.414** |
| within-system confidence–DockQ Spearman | **−0.129** |

The first three values are stored in
[`training500_summary.json`](../results/summaries/training500_summary.json)
(`consistency` block). A system-risk model predicting whether the AF-M rank-1 is
acceptable improves when ensemble features are added: ROC-AUC `0.796` → `0.812`
([`training500_system_risk_models.csv`](../results/tables/training500_system_risk_models.csv)).

## Holdout confirmation

The same relationships reproduce, and are stronger, on the frozen PINDER-AF2
holdout (180 systems):

| Quantity | Training500 | PINDER-AF2 |
| --- | ---: | ---: |
| jaccard ↔ acceptable fraction | 0.655 | **0.800** |
| jaccard ↔ rank-1 acceptability | 0.437 | **0.737** |
| jaccard ↔ DockQ range | −0.414 | −0.112 |
| jaccard ↔ within-system Spearman | −0.129 | 0.091 |

The consistency–trust relationship therefore generalizes out of sample and
sharpens: low-consistency systems are even stronger "do not trust rank-1" flags
on the holdout (only 8% of low-consistency-tercile systems had an acceptable
rank-1, versus 95% of the high-consistency tercile).

## Failure modes

Across Training500, 200 of 500 systems had an unacceptable rank-1: 88 were
rerank-rescuable, 112 sampling failures, and 17 stable-wrong. Consistency
separates "rank-1 acceptable" systems sharply from both failure groups:

| Group | Systems | jaccard median | jaccard < 0.3 | jaccard >= 0.8 |
| --- | ---: | ---: | ---: | ---: |
| rank-1 acceptable | 300 | 0.692 | 23% | 37% |
| rerank-rescuable | 88 | 0.234 | 59% | 2% |
| sampling failure | 112 | 0.230 | 54% | 15% |

On the holdout the pattern is sharper (12 rescuable, 63 sampling failures, 105
acceptable): rescuable jaccard median `0.152` with 83% below 0.3 and none above
0.8.

Rescuable systems are the ones a better selector could save, and they are
overwhelmingly low-consistency. However, consistency does not separate rescuable
from sampling failures — both are low-consistency — because whether sampling
happened to include a good candidate is a separate question from how convergent
the ensemble is.

## Why the rescue is hard: where the correct candidate sits

In rescuable systems, the best acceptable candidate is buried by AF-M's
confidence ranking: median rank **6** on Training500 (mean 7.5, range 2–20; 26%
worse than rank 10) and median rank **5** on the holdout (range 3–20). Saving
these systems requires lifting a candidate from around rank 6 to rank 1, which
is exactly what the weak within-system signal (median within-system `Spearman ≈
0.227`; see [Results](RESULTS.md)) cannot reliably do.

Exploratory within-rank and group-softmax models modestly enriched acceptable
candidates near the top of the Training500 rescuable subset: recall@3 changed
from 23.9% under AF-M ranking to 34.1% and 30.7%, respectively. Median
first-acceptable rank remained 6, and recall@5 did not improve. This is a
metric-dependent ranking signal rather than evidence of a general selector
improvement; exact descriptive values are in
[`rerank_rescuable_ranking.csv`](../results/tables/rerank_rescuable_ranking.csv).

The acceptable minority is also strongly cluster-structured. The 20 candidates
in a rescuable system formed a median of 7.5 contact clusters, and the dominant
acceptable cluster had median purity 1.00. However, its median rank by
cluster-mean ipTM was 2.5, and the top-ipTM cluster was entirely wrong in 84.1%
of rescuable systems. Hand-built consistency interactions and cluster gates did
not turn this diagnostic structure into a net top-1 gain. These post-hoc results
motivate new candidate-specific signals rather than a rule that simply selects
the largest, most consistent, or highest-ipTM cluster; see the
[`pairwise consistency diagnostic`](../analysis/pairwise_consistency_diagnostic.py).

## Stable wrong: the hard core

17 Training500 and 4 holdout systems are stable-wrong: all 20 candidates
incorrect (`DockQ ≈ 0.01` median), yet the ensemble is highly consistent
(jaccard 0.80–0.99, cluster fraction 1.0, small model-to-model ipTM variation)
and AF-M is confident (rank-1 confidence mostly 80–95, range 59–96). These are
ensembles that genuinely converged — on the wrong structure. Consistency cannot
flag them, which is why convergence must not be equated with correctness.

## Why consistency cannot change selection

`mean_contact_jaccard` is constant within a system, so it adds the same value to
all 20 candidates' scores and cannot reorder them. It is therefore never a
within-system selection signal; its value is as a system-level trust indicator.

## Caveats

- The system-risk improvement has no separately frozen holdout test
  ([Limitations](LIMITATIONS.md)); all holdout and failure-mode values in this
  note are computed directly from the public candidate tables under
  `results/data/`, and a reproducible script for them is a planned follow-up.
- Consistency is a trust indicator, not a correctness guarantee.
