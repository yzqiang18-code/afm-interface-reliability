# Limitations

## Scope of the conclusion

Training500 intentionally enriches difficult and rerank-informative systems;
its event rates are not unbiased PINDER-Val estimates. PINDER-AF2 supports one
narrow negative conclusion: the frozen five-feature linear reranker did not
improve AF-M top-1 acceptable rate. It does not show that reranking is
impossible, and `acceptable_score` is not established as a calibrated
probability.

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

## Supported statement

The repository supports a reproducible evaluation workflow, a clear separation
between sampling and selection failures, and a negative frozen-holdout result
for Candidate Ridge v1. It does not support claims of improved AF-M accuracy,
an unbiased PINDER benchmark, or a final general-purpose confidence model.
