# Limitations

## Scope of the conclusion

Training500 intentionally enriches difficult and rerank-informative systems;
its event rates are not unbiased PINDER-Val estimates. PINDER-AF2 supports one
narrow negative conclusion: the frozen five-feature linear reranker did not
improve AF-M top-1 acceptable rate. It does not show that reranking is
impossible, and `acceptable_score` is not established as a calibrated
probability.

The pooled candidate-level ROC-AUC is dominated by discrimination across
systems and is not a measure of whether the best candidate is ordered correctly
within each system. The within-rank and group-softmax follow-ups show small,
metric-dependent point-estimate changes, but no statistically significant
improvement in the primary top-1 endpoint.

## Exploratory ranking subset

The rerank-rescuable subset is defined using the observed DockQ outcomes: AF-M
rank-1 must be wrong and another sampled candidate must be acceptable. AF-M
therefore has zero top-1 successes in this subset by construction. First-rank,
MRR, and recall@k comparisons in this group are useful diagnostics but are
post-hoc, subject to conditional-selection bias, and were examined alongside
multiple models and endpoints without a multiplicity correction. They should be
described as hypothesis-generating rather than confirmatory.

The cluster diagnostic is likewise post hoc. DockQ labels are used to identify
and measure the dominant acceptable cluster, and multiple interaction, gating,
and cluster-selection rules were examined on Training500. The observed cluster
purity and confidence-rank mismatch are useful mechanistic clues, but they do
not establish that cluster membership is a deployable label or that the same
pattern will reproduce on a new cohort. No corresponding cluster diagnostic has
been confirmed on an untouched holdout.

## Remaining dependence and leakage risk

The audit found no exact system, PDB, interface-cluster, or unordered
chain-cluster-pair overlap, but one known UniProt-pair overlap remained. Removing
the affected holdout system did not change the conclusion. This audit does not
exclude every possible evolutionary, domain-level, or remote structural
relationship, so the cohort should be described as held out rather than fully
independent in every biological sense.

## Labels and structures

The chain-swap correction prevents missing accession tokens from being mistaken
for homodimer evidence. Equality of known accessions is still only a proxy for
chain exchangeability; exact sequence and biological-assembly checks would be
stricter. Alternative interfaces, multiple valid binding modes, flexibility,
oligomeric state, and unresolved residues can also make one native-referenced
DockQ label incomplete.

Sequences were limited to residues resolved in holo structures and may omit
unresolved UniProt segments. Physics features are geometric or chemical
approximations, not binding free-energy calculations.

## Protocol dependence

Results apply to the specified AF-M v2.3 / LocalColabFold 1.5.5 setup, MSA
strategy, four seeds, five model weights, three recycles, no templates, and no
relaxation. The small exploratory system-risk improvement from ensemble features
has no separately frozen holdout test.

PINDER-AF2 was frozen at the row and label level, but it has now been inspected
for Candidate Ridge v1, within-rank, and group-softmax. Reusing it to choose
future features or objectives would weaken a one-shot confirmatory
interpretation. A new untouched cohort is preferable for future claims about
coevolution or richer PAE models.

Consistency is a system-level trust signal, not a candidate-level selector.
High consistency is associated with higher rank-1 reliability, but 17
Training500 and 4 holdout systems were highly consistent while all candidates
were wrong. Conversely, low consistency does not distinguish a rerank-rescuable
system from a sampling failure. It can guide whether to trust or resample a
system, but cannot guarantee correctness or identify the best existing
candidate by itself.

## Supported statement

The repository supports a reproducible evaluation workflow, a clear separation
between sampling and selection failures, and a negative frozen-holdout result
for Candidate Ridge v1. It also supports a reproducible association between
ensemble consistency and system-level reliability, together with documented
stable-wrong exceptions. It does not support claims of significantly improved
AF-M accuracy, an unbiased PINDER benchmark, or a final general-purpose
confidence model.
