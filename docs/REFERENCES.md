# References and upstream resources

## Dataset and benchmark

- Kovtun D, Akdel M, Goncearenco A, et al. **PINDER: The protein interaction dataset and evaluation resource.** bioRxiv (2024). DOI: [10.1101/2024.07.17.603980](https://doi.org/10.1101/2024.07.17.603980). Project: [pinder-org/pinder](https://github.com/pinder-org/pinder).

## Structure prediction and MSA workflow

- Evans R, O'Neill M, Pritzel A, et al. **Protein complex prediction with AlphaFold-Multimer.** bioRxiv (2021/2022). DOI: [10.1101/2021.10.04.463034](https://doi.org/10.1101/2021.10.04.463034).
- Mirdita M, Schütze K, Moriwaki Y, Heo L, Ovchinnikov S, Steinegger M. **ColabFold: making protein folding accessible to all.** *Nature Methods* 19, 679–682 (2022). DOI: [10.1038/s41592-022-01488-1](https://doi.org/10.1038/s41592-022-01488-1).

## Evaluation and confidence metrics

- Basu S, Wallner B. **DockQ: A Quality Measure for Protein-Protein Docking
  Models.** *PLOS ONE* 11(8), e0161879 (2016). DOI:
  [10.1371/journal.pone.0161879](https://doi.org/10.1371/journal.pone.0161879).
  This work defines the continuous DockQ score and the `0.23` quality boundary
  used here (called "medium" quality in that paper).
- Mirabello C, Wallner B. **DockQ v2: improved automatic quality measure for protein multimers, nucleic acids, and small molecules.** *Bioinformatics* 40(10), btae586 (2024). DOI: [10.1093/bioinformatics/btae586](https://doi.org/10.1093/bioinformatics/btae586). This version defines the quality bands used here: `DockQ >= 0.23` acceptable, `>= 0.49` medium, and `>= 0.80` high.
- Zhu W, Shenoy A, Kundrotas P, Elofsson A. **Evaluation of AlphaFold-Multimer prediction on multi-chain protein complexes.** *Bioinformatics* 39(7), btad424 (2023). DOI: [10.1093/bioinformatics/btad424](https://doi.org/10.1093/bioinformatics/btad424). This work introduced pDockQ2.
- Kim AR, Hu Y, Comjean A, et al. **Enhanced Protein-Protein Interaction
  Discovery via AlphaFold-Multimer.** bioRxiv (2024). DOI:
  [10.1101/2024.02.19.580970](https://doi.org/10.1101/2024.02.19.580970).
  This work introduced the LIS/LIA framework.
- Kim AR, et al. **FlyPredictome: a structural atlas of predicted
  protein-protein interactions in Drosophila.** bioRxiv (2026). DOI:
  [10.64898/2026.04.14.718529](https://doi.org/10.64898/2026.04.14.718529).
  This work describes integrated LIS (iLIS). Upstream implementation:
  [flyark/AFM-LIS](https://github.com/flyark/AFM-LIS).
- Dunbrack Lab. **IPSAE: scoring interprotein interactions in AlphaFold2 and AlphaFold3.** Upstream project: [DunbrackLab/IPSAE](https://github.com/DunbrackLab/IPSAE).

## Vendored sources

Exact upstream commits and entry points are recorded in `third_party/README.md`; file integrity is recorded in `third_party/checksums.sha256`. The upstream MIT licenses remain beside each vendored source tree.
