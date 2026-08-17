# Selected result tables

| File | Purpose |
|---|---|
| `training500_metric_performance.csv` | Prediction-level metric correlation and discrimination with grouped bootstrap intervals |
| `training500_within_system_ranking.csv` | Within-system rank correlation across 20 candidates |
| `training500_selector_performance.csv` | Five-fold OOF candidate-selector baselines |
| `training500_system_risk_models.csv` | Five-fold OOF top-1 system-risk feature ablations |
| `training500_system_risk_deltas.csv` | System-grouped paired-bootstrap differences versus AF-only risk features |
| `training500_selector_deltas.csv` | System-grouped paired-bootstrap selector differences versus AF-M rank-1 |

`DockQ >= 0.23` is the acceptable-quality threshold throughout these files. Oracle columns use native quality after prediction and are analysis-only upper bounds.
