# Pairwise-Jaccard NN (1×19)

Exploratory Training500 grouped-CV study. A small shared-weight PyTorch MLP
(`Linear(19→32)→ReLU→Linear(32→16)→ReLU→Linear(16→1)`, BCE on `DockQ >= 0.23`)
scores each candidate from its 1×19 within-system contact-Jaccard row vector
(20×20 matrix, self slot excluded) and reranks by argmax per system.

**Result: significantly worse than AF-M rank-1** — acceptable rate 0.540 vs
0.600 (paired difference −0.060, 95% CI −0.092 to −0.032). See
[`docs/NN_JACCARD.md`](../../../docs/NN_JACCARD.md).

Artifacts: `model.json` (config, fold audits, all selector/comparison/rerank
metrics), `training_summary.json`, per-fold `checkpoints/fold_*.pt`,
`oof_predictions.csv`, selector and paired-bootstrap tables,
`comparison_metrics.json` (vs AF-M rank-1 and regenerated ridge references),
`rerank_metrics.csv` (within-system Spearman, rescuable recall).

Reproduce:

```bash
python scripts/ml/train_jaccard_nn.py \
  --train-csv results/data/training500_jaccard_rows.csv.gz \
  --config configs/ml/pairwise_jaccard_nn_1x19.json \
  --output-dir results/ml/pairwise_jaccard_nn_1x19 --device cpu
```
