# Scientific audits

## Chain assignment

`chain_exchange/chain_exchange_summary.json` records the effect of treating
missing accession tokens as unknown rather than as evidence of exchangeable
chains. `candidate_changes.csv.gz` contains affected candidate rows. Continuous
DockQ changed, but no primary acceptable/unacceptable label changed.

## Development–holdout overlap

`leakage/leakage_summary.json` and `leakage_intersections.csv` compare frozen
cohorts at five identity levels using the official PINDER 2024-02 index. Four
levels have zero overlap; one known UniProt-pair overlap is disclosed.
`uniprot_overlap_sensitivity.json` shows that excluding the affected holdout
system leaves the main conclusion unchanged.
