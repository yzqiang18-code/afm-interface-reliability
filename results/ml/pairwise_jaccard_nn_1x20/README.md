# Pairwise-Jaccard NN (1×20)

Exploratory Training500 grouped-CV study, identical to the 1×19 variant except
each candidate's input row is the full 1×20 matrix row including the self
diagonal (1.0). See `../pairwise_jaccard_nn_1x19/README.md` and
[`docs/NN_JACCARD.md`](../../../docs/NN_JACCARD.md).

**Result: significantly worse than AF-M rank-1** — acceptable rate 0.532 vs
0.600 (paired difference −0.068, 95% CI −0.104 to −0.034). Its one positive
sub-signal is on the 88 rerank-rescuable systems (recall@3 42.0% vs 23.9% for
AF-M), which does not offset the top-1 losses.

Reproduce:

```bash
python scripts/ml/train_jaccard_nn.py \
  --train-csv results/data/training500_jaccard_rows.csv.gz \
  --config configs/ml/pairwise_jaccard_nn_1x20.json \
  --output-dir results/ml/pairwise_jaccard_nn_1x20 --device cpu
```
