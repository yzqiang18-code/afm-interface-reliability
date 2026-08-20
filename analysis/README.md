# Statistical analysis

`training500_analysis.py` contains the audit and statistical analysis used to
produce the committed Training500 summaries and selected tables. It covers
candidate-level metrics, within-system ranking, grouped bootstrap intervals,
system-level cross-validation, selector comparisons, and ensemble-feature
ablations.

The original analysis expects a metric bundle containing AF-M confidence,
DockQ, iLIS, pDockQ2, interface-physics, ensemble-consistency, and cohort
metadata tables. The complete structural pipeline is intentionally separate
from the compact public reproduction. The two candidate-level inputs needed to
retrain and evaluate Candidate Ridge are distributed under
[`results/data/`](../results/data/README.md).

The curated outputs are available in [`results/summaries/`](../results/summaries/)
and [`results/tables/`](../results/tables/).
For the shortest end-to-end reproduction, run
`python scripts/reproduce_level2.py` from the repository root.

`pairwise_consistency_diagnostic.py` is a separate post-hoc Training500
diagnostic. It consumes the public, path-free
`results/data/training500_consistency_pairs.csv.gz` table and examines whether
candidate-pair agreement, contact clusters, interactions, or gated selectors
add within-system signal. Sections 6–9 use labels for exploratory subgroup or
cluster diagnostics and are not confirmatory model evaluations.
