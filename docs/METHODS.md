# Methods

## Question and endpoints

The project tests whether a simple reference-free model can select a better
AlphaFold-Multimer (AF-M) candidate than AF-M's full-precision ranking. One
candidate structure is a prediction-level unit; all candidates for one protein
pair form a system-level unit. `DockQ >= 0.23` is the prespecified primary
acceptable-interface endpoint. DockQ v2 defines quality bands at 0.23
(acceptable), 0.49 (medium), and 0.80 (high); selection here uses only the
0.23 endpoint, while the 0.49 and 0.80 bands appear as secondary rate
summaries in the results tables. DockQ and all other native-derived quantities
are labels or evaluation outputs, never deployable model inputs.

## Cohorts and inference

| Cohort | Role | Systems | Candidates per system |
| --- | --- | ---: | ---: |
| Feasibility50 | Pipeline and metric checks | 50 | 25 |
| Training500 | Stratified model development | 500 | 20 |
| PINDER-AF2 | Frozen held-out evaluation | 180 | 20 |

The study uses PINDER 2024-02 and LocalColabFold 1.5.5 with AF-M v2.3
(`alphafold2_multimer_v3`), five model weights, three recycles, no templates,
and no Amber relaxation. Training500 and PINDER-AF2 used four seeds, yielding
20 candidates per system. Full-precision ranking confidence, ipTM, and pTM were
captured before JSON rounding.

Training500 was selected from seed-0 screening strata to enrich difficult and
rerank-informative cases. It is not prevalence-representative. Its five fixed
folds keep all candidates from a system together. PINDER-AF2 was held out from
model fitting and used only after the pipeline was frozen.

## Signals

- **AF-M confidence:** full-precision ranking confidence, ipTM, and pTM.
- **Interface confidence:** pDockQ2, iLIS, and ipSAE.
- **Physical plausibility:** buried surface area, contacts, clashes, chemical
  complementarity, and interface size.
- **Ensemble consistency:** contact-map agreement, interface clustering, and
  receptor-aligned ligand-pose variation.
- **Native-referenced evaluation:** DockQ, Fnat, interface RMSD, and ligand RMSD.

Candidate Ridge v1 is an L2-regularized logistic model using five inputs:
full-precision ipTM, full-precision pTM, pDockQ2-min, iLIS, and ipSAE. Missing
values are median-imputed and features standardized using development data only.
The output `acceptable_score` is used to order candidates within a system; it is
not presented as a calibrated probability.

Two ranking-focused variants use the same five inputs. The within-rank variant
replaces each raw feature by its centered quantile rank among the 20 candidates
of that system before fitting the same L2-regularized binary model. The
group-softmax variant uses those within-system ranks with a conditional-logit
loss. It is trained only on *mixed systems*, which contain at least one
acceptable and one unacceptable candidate, because single-class systems do not
identify a within-system binary ordering objective. All reported Training500
predictions for these models are grouped out of fold.

A *rerank-rescuable system* is defined retrospectively as a system whose AF-M
rank-1 is unacceptable but whose 20 candidates include at least one acceptable
structure. Ranking diagnostics for this outcome-defined subset report the rank
of the first acceptable candidate, mean reciprocal rank (MRR), and recall@k,
where recall@k is the fraction of systems with an acceptable candidate among the
top k positions. These are exploratory diagnostics, not prespecified primary
endpoints.

The pairwise consistency diagnostic uses all 190 unordered candidate pairs per
Training500 system. Contact-map Jaccard, interface-residue Jaccard, and
receptor-aligned ligand RMSD are aggregated into candidate- and cluster-level
descriptors. Contact clusters come from the existing consistency pipeline.
DockQ is used only after clustering to describe cluster purity and to identify
the cluster containing the largest number of acceptable candidates. Interaction,
gating, and cluster-selection probes are post-hoc Training500 analyses and are
not treated as frozen model evaluations.

## Chain assignment rule

Direct and receptor/ligand-swapped DockQ mappings are compared only when both
chains have the same non-missing accession. Empty, `UNDEFINED`, `NONE`, `NAN`,
and `NA` tokens never establish chain-exchange eligibility. All other systems
use the stored fixed direct mapping.

An audit reprocessed both cohorts after correcting the earlier
`UNDEFINED == UNDEFINED` logic. Continuous DockQ changed for 130 Training500
rows in 11 systems and 299 PINDER-AF2 rows in 25 systems. No row or system
crossed the primary `DockQ >= 0.23` boundary. This accession-based rule is
auditable, but exact sequence identity would be a stricter future eligibility
test.

## Split-overlap audit

The frozen manifests were compared using the official PINDER 2024-02 index at
five levels: exact system ID, PDB ID, interface cluster, unordered chain-cluster
pair, and unordered known UniProt pair. The first four intersections were zero.
One UniProt-pair overlap (`Q68T42|Q68T42`) connected one Training500 system and
one PINDER-AF2 system. A sensitivity calculation excludes the affected holdout
system and reports the selectors on the remaining 179.

## Statistical analysis

Prediction-level analyses report Spearman correlation, ROC-AUC, average
precision, and Brier score. These candidate-level discrimination metrics pool
candidates across systems and must not be interpreted as within-system ranking
performance. Within-system analyses report per-system Spearman summaries and
ranking endpoints such as first-acceptable rank, MRR, and recall@k. Candidate
selection is evaluated once per system. All confidence intervals comparing
selectors use paired bootstrap resampling of complete systems. Oracle
best-of-20 uses native DockQ retrospectively and is reported only as a sampling
ceiling.

## Quality assurance and reproduction

Training500 contains 500 systems, 10,000 unique candidate rows, and 95,000
within-system candidate pairs; PINDER-AF2 contains 180 systems and 3,600 rows.
The repository validator checks schemas, hashes, cohort membership, leakage
gates, chain-audit invariants, private-path absence, and headline counts.
`scripts/reproduce_level2.py` retrains, predicts, evaluates, and regenerates the
four data figures using only committed public inputs.
