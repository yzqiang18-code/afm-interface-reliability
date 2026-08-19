# AFM Interface Reliability

An research project on the reliability of AlphaFold-Multimer
(AF-M) protein–protein interface predictions. The central question is simple:
when AF-M generates 20 candidates for one protein pair, can a small,
interpretable model choose a better structure than AF-M's own ranking?

## Current result

This repository documents the current stage of an ongoing project. The answer
for the tested baseline is **no**: on the held-out 180-system PINDER-AF2
cohort, AF-M rank-1 and the five-feature Candidate Ridge model both selected an
acceptable structure (`DockQ >= 0.23`) for **58.33%** of systems. Mean DockQ
changed from **0.45240** to **0.45385**; the paired difference was `+0.00145`
(95% bootstrap CI `-0.00129` to `+0.00472`). Candidate-level ROC-AUC was high
(`0.934`), but that did not translate into better within-system selection. This
negative result is retained because it is the currently supported conclusion
for this baseline.

A within-system quantile-rank follow-up (`candidate_ridge_v1_within_rank`)
replaced global standardization with per-system rank features. Its
within-system ordering point estimate improved: on Training500 out-of-fold
data the median within-system Spearman rose from `0.206` (v1 baseline) and
`0.227` (AF-M's own ranking) to `0.243`. This difference is **not
statistically significant** (system-level paired 95% CI `-0.018` to
`+0.061` vs the v1 baseline). On the frozen PINDER-AF2 holdout it selected
an acceptable structure for `58.89%` (106/180) versus `58.33%` (105/180)
for both AF-M rank-1 and the v1 baseline — one additional system, with a
paired 95% CI lower bound of `0.0000`. The selection conclusion is
therefore unchanged.

![Current selection result](figures/top1_vs_oracle.svg)

Training500 shows why the problem is worth studying: AF-M top-1 was acceptable
for **60.0%** of systems, while a retrospective best-of-20 oracle reached
**77.6%**. The gap is sampling headroom, not evidence that the current model can
recover it.

A related secondary finding: ensemble consistency is a system-level trust
signal, not a selection signal — more consistent systems are more likely to
contain acceptable candidates and to have a correct AF-M rank-1 (see
[Consistency](docs/CONSISTENCY.md)).

## Planned follow-ups

The current negative result leaves the within-system ordering bottleneck
unresolved. Two follow-up designs are planned, developed on Training500 grouped
out-of-fold validation and applied to the frozen PINDER-AF2 holdout only after
they are frozen, under the same discipline as the baseline:

- **Within-system feature standardization.** Features are currently standardized
  with development-wide statistics, so the baseline mostly sees between-system
  differences. Standardizing each feature within its own 20-candidate system
  would let the model rank candidates relative to that system, directly
  targeting the weak within-system signal (median Spearman ≈ 0.227).
- **Continuous regression target.** The baseline predicts the binary
  `DockQ >= 0.23` label, while within-system selection needs to order candidates
  by degree of quality. Predicting continuous DockQ (still label-only, never an
  input) should fit that ordering objective more closely.

## Study design

| Cohort | Purpose | Systems | Candidates per system |
| --- | --- | ---: | ---: |
| Feasibility50 | Pipeline checks | 50 | 25 |
| Training500 | Stratified development and grouped validation | 500 | 20 |
| PINDER-AF2 | Held-out evaluation | 180 | 20 |

Candidate Ridge uses five native-independent inputs: full-precision ipTM, pTM,
pDockQ2-min, iLIS, and ipSAE. All 20 candidates from a protein pair remain in
the same development fold. DockQ is used only as a label and evaluation metric.

The workflow diagram below is a fixed schematic overview; the three
quantitative SVGs in `figures/` are regenerated from committed tables by
`scripts/figures/make_figures.py`.

![Project workflow](figures/workflow.svg)

## Reproduce the public analysis

The repository includes the two derived candidate tables needed to retrain and
evaluate the baseline. It does not include native structures, predicted PDBs,
MSAs, AlphaFold parameters, or PAE JSON files.

```bash
conda env create -f environment.yml
conda run -n afm-interface-reliability python -m unittest discover -s tests -v
conda run -n afm-interface-reliability python scripts/validate_repository.py
conda run -n afm-interface-reliability python scripts/reproduce_level2.py
```

The last command retrains the model, predicts the held-out candidates, evaluates
both selectors, and regenerates all three data figures in `reproduced/`. It runs
on CPU and refuses to overwrite a non-empty output directory unless
`--overwrite` is supplied.

## Scientific safeguards

- **Held-out evaluation:** features, preprocessing, coefficients, threshold, and
  selection rule were fixed before evaluation on PINDER-AF2.
- **No native leakage into inputs:** DockQ, Fnat, interface RMSD, ligand RMSD,
  and other native-derived values are labels or evaluation outputs only.
- **Grouped validation:** candidates from one system are never split across
  development folds.
- **Chain assignment audit:** chain swapping is allowed only when both chains
  have the same known accession. Missing tokens such as `UNDEFINED` do not
  establish symmetry. Correcting this changed continuous DockQ for 130
  Training500 and 299 holdout rows, but changed no `DockQ >= 0.23` class.
- **Split audit:** exact system, PDB, interface-cluster, and unordered
  chain-cluster-pair overlap counts are all zero. One known UniProt-pair overlap
  was found. Excluding that holdout system leaves the conclusion unchanged
  (58.66% vs 58.66%; mean-DockQ difference `+0.00146`).
- **Paired inference:** selector comparisons resample complete protein-pair
  systems, not individual candidate rows.

## Repository map

```text
analysis/          Statistical analysis used for Training500
configs/           Frozen cohort records and model configuration
docs/              Methods, results, limitations, data notes, references
figures/           One workflow diagram and three generated SVG figures
results/data/      Public derived candidate tables and manifests
results/audits/    Chain-assignment and split-overlap audits
results/ml/        Frozen model, predictions, and evaluation evidence
scripts/           Data preparation, metrics, modeling, figures, validation
tests/             Unit and integration-style tests
third_party/       Pinned metric implementations and upstream licenses
```

Start with [Methods](docs/METHODS.md), [Results](docs/RESULTS.md),
[Limitations](docs/LIMITATIONS.md), and [Data](docs/DATA.md). Exact values are
stored in machine-readable JSON and CSV files under `results/`.

## About the author

This repository is my undergraduate research project in computational
structural biology. I built the pipeline end to end: cohort selection, the
full-precision ColabFold wrapper, the metric adapters, the Training500
analysis, and the Candidate Ridge baseline, together with the tests and this
documentation.

Two decisions here came directly from problems I hit while running the
experiments. ColabFold rounds ipTM and pTM in its default JSON output, which
silently breaks joint ranking of candidates, so the wrapper captures the
full-precision values before that rounding. And the original DockQ chain-swap
rule compared accession strings literally, which treated `UNDEFINED ==
UNDEFINED` as homodimer evidence; the correction is documented under
`results/audits/chain_exchange/`.

## Data, citation, and license

The study uses the public PINDER 2024-02 release. Project-authored code is MIT
licensed; PINDER-derived records and all upstream assets remain subject to their
original attribution and terms. See [Data](docs/DATA.md),
[References](docs/REFERENCES.md), [CITATION.cff](CITATION.cff), and the vendored
licenses under `third_party/`.
