#!/usr/bin/env python3
"""Train a small shared-weight MLP on per-candidate Jaccard row vectors.

The 20x20 within-system contact-Jaccard matrix (built by
``build_jaccard_matrix.py``) is used in row form: each candidate is scored
from its 1x19 (or 1x20) row vector by one small fully-connected network that
is shared across all candidates and systems. Because the same network is
applied to every row, the scoring is permutation-equivariant in the candidate
order — reordering candidates just reorders rows and scores.

Training and evaluation
-----------------------
Grouped 5-fold cross-validation on Training500 using the frozen ``cv_fold``
column: candidates of one system are never split across folds. OOF scores are
then used for selector comparisons (argmax per system) with system-level
paired bootstrap against AF-M rank-1 and a regenerated Candidate Ridge v1
grouped-OOF reference, plus scalar-summary ridge references (``mean_j`` and
the five native features + ``mean_j``) to test whether the matrix/row form
adds signal over scalar pairwise summaries.

Discipline
----------
Only candidate-vs-candidate Jaccard similarities and AF-M's own scores are
inputs. ``DockQ`` is a label/evaluation target only. This is an exploratory
Training500 study: the public data contains no PINDER-AF2 pair table, so the
frozen-holdout step would require regenerating pair rows from the full
pipeline and is out of scope here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_core import (  # noqa: E402
    PREDICTION_SCORE_COLUMN,
    atomic_write_csv,
    atomic_write_json,
    candidate_metrics,
    check_output_targets,
    fit_preprocessor,
    fit_ridge_logistic,
    load_json,
    paired_selector_bootstrap,
    portable_path,
    predict_probabilities,
    select_candidates,
    selector_metrics,
    sha256_file,
    utc_now,
    validate_candidate_frame,
    validate_config,
)

try:
    import torch
    from torch import nn

    HAS_TORCH = True
except ModuleNotFoundError:  # pragma: no cover - exercised by CLI only
    HAS_TORCH = False


OUTPUT_NAMES = {
    "model": "model.json",
    "summary": "training_summary.json",
    "oof_predictions": "oof_predictions.csv",
    "candidate_metrics": "oof_candidate_metrics.csv",
    "selector_choices": "oof_selector_choices.csv",
    "selector_summary": "oof_selector_summary.csv",
    "bootstrap": "oof_paired_bootstrap.csv",
    "comparison": "comparison_metrics.json",
    "rerank": "rerank_metrics.csv",
}

V1_FEATURES = [
    "iptm_full_precision",
    "ptm_full_precision",
    "pDockQ2_min",
    "iLIS",
    "ipSAE",
]

ACCEPT_DOCKQ = 0.23
EXPECTED_FOLDS = [0, 1, 2, 3, 4]


class SmallMLP(nn.Module):
    """Tiny fully-connected scorer: input -> ReLU hidden layers -> logit."""

    def __init__(self, input_size: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        sizes = [input_size] + list(hidden_sizes)
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def fit_fold(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
    *,
    fold: int,
    device: str,
) -> dict[str, Any]:
    nn_cfg = config["nn"]
    target_column = str(config["target_column"])
    threshold = float(config["target_threshold"])
    train_target = training[target_column].ge(threshold).to_numpy(dtype=float)
    if set(np.unique(train_target)) != {0.0, 1.0}:
        raise ValueError("Fold training partition contains one target class")

    transformed_train, preprocessing = fit_preprocessor(training, feature_columns)
    transformed_val = _preprocess(validation, feature_columns, preprocessing)
    x_train = torch.as_tensor(transformed_train, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(train_target, dtype=torch.float32, device=device)
    x_val = torch.as_tensor(transformed_val, dtype=torch.float32, device=device)
    y_val = validation[target_column].ge(threshold).to_numpy(dtype=float)

    _torch_seed(int(config["random_seed"]) + int(fold))
    model = SmallMLP(len(feature_columns), [int(s) for s in nn_cfg["hidden_sizes"]])
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(nn_cfg["learning_rate"]),
        weight_decay=float(nn_cfg["weight_decay"]),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    batch_size = int(nn_cfg["batch_size"])
    epochs = int(nn_cfg["epochs"])
    n = len(x_train)

    train_loss_history: list[float] = []
    rng = np.random.default_rng(int(config["random_seed"]) + int(fold))
    model.train()
    for _ in range(epochs):
        order = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(x_train[idx]).squeeze(1)
            loss = loss_fn(logits, y_train[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        train_loss_history.append(epoch_loss / n_batches)

    model.eval()
    with torch.no_grad():
        val_logits = model(x_val).squeeze(1).cpu().numpy()
    val_loss = float(
        loss_fn(
            torch.as_tensor(val_logits),
            torch.as_tensor(y_val),
        ).item()
    )
    val_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -30, 30)))
    val_auc = candidate_metrics(
        validation[target_column].to_numpy(dtype=float),
        val_probs,
        threshold=threshold,
    )["roc_auc"]

    state_dict = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
    return {
        "preprocessing": preprocessing,
        "state_dict": state_dict,
        "hidden_sizes": [int(s) for s in nn_cfg["hidden_sizes"]],
        "train_loss_history": train_loss_history,
        "val_loss": val_loss,
        "val_roc_auc": val_auc,
        "epochs": epochs,
        "batch_size": batch_size,
    }


def _preprocess(
    frame: pd.DataFrame,
    feature_columns: list[str],
    preprocessing: dict[str, Any],
) -> np.ndarray:
    values = frame[feature_columns].to_numpy(dtype=float)
    medians = np.asarray(preprocessing["medians"], dtype=float)
    means = np.asarray(preprocessing["means"], dtype=float)
    scales = np.asarray(preprocessing["scales"], dtype=float)
    imputed = np.where(np.isfinite(values), values, medians)
    return (imputed - means) / scales


def predict_fold(
    frame: pd.DataFrame,
    feature_columns: list[str],
    fold_result: dict[str, Any],
    *,
    device: str,
) -> np.ndarray:
    transformed = _preprocess(frame, feature_columns, fold_result["preprocessing"])
    x = torch.as_tensor(transformed, dtype=torch.float32, device=device)
    model = SmallMLP(
        len(feature_columns),
        [int(s) for s in fold_result["hidden_sizes"]],
    )
    model.load_state_dict(fold_result["state_dict"])
    model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x).squeeze(1).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


def oof_cv(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    device: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    predictions = np.full(len(frame), np.nan, dtype=float)
    feature_columns = list(config["feature_columns"])
    fold_column = str(config["fold_column"])
    fold_results: list[dict[str, Any]] = []
    nn_cfg = config["nn"]

    for fold in sorted(int(v) for v in config["expected_fold_values"]):
        train_mask = frame[fold_column].ne(fold).to_numpy()
        val_mask = ~train_mask
        training = frame.loc[train_mask]
        validation = frame.loc[val_mask]
        result = fit_fold(
            training, validation, feature_columns, config, fold=fold, device=device
        )
        predictions[val_mask] = predict_fold(
            validation, feature_columns, result, device=device
        )
        fold_results.append(
            {
                "fold": fold,
                "train_rows": int(train_mask.sum()),
                "validation_rows": int(val_mask.sum()),
                "train_systems": int(training[config["group_column"]].nunique()),
                "validation_systems": int(validation[config["group_column"]].nunique()),
                "epochs": result["epochs"],
                "batch_size": result["batch_size"],
                "hidden_sizes": [int(s) for s in nn_cfg["hidden_sizes"]],
                "final_train_loss": float(result["train_loss_history"][-1]),
                "train_loss_first": float(result["train_loss_history"][0]),
                "val_loss": result["val_loss"],
                "val_roc_auc": result["val_roc_auc"],
                "preprocessing": result["preprocessing"],
                "state_dict": result["state_dict"],
            }
        )
    if not np.isfinite(predictions).all():
        raise ValueError("OOF predictions are incomplete")

    columns = list(config["key_columns"]) + [
        fold_column,
        str(config["target_column"]),
        str(config["reference_score_column"]),
    ]
    output = frame[columns].copy()
    output["acceptable"] = frame[str(config["target_column"])].ge(
        float(config["target_threshold"])
    )
    output[PREDICTION_SCORE_COLUMN] = predictions
    return output, fold_results


def grouped_ridge_oof(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    penalty: float,
) -> pd.DataFrame:
    """Regenerate grouped-OOF ridge-logistic predictions (v1-style zscore)."""
    predictions = np.full(len(frame), np.nan, dtype=float)
    for fold in EXPECTED_FOLDS:
        train_mask = frame["cv_fold"].ne(fold).to_numpy()
        val_mask = ~train_mask
        training = frame.loc[train_mask]
        validation = frame.loc[val_mask]
        transformed_train, preprocessing = fit_preprocessor(training, feature_columns)
        transformed_val = _preprocess(validation, feature_columns, preprocessing)
        target = training["DockQ"].ge(ACCEPT_DOCKQ).astype(float).to_numpy()
        coefficients, _solver = fit_ridge_logistic(
            transformed_train, target, penalty=penalty
        )
        predictions[val_mask] = predict_probabilities(transformed_val, coefficients)

    output = frame[
        ["complex_id", "rank", "model_weight", "seed", "cv_fold", "DockQ"]
    ].copy()
    output["acceptable"] = frame["DockQ"].ge(ACCEPT_DOCKQ)
    output[PREDICTION_SCORE_COLUMN] = predictions
    return output


def within_spearman(
    groups: dict[str, pd.DataFrame], col: str, systems: set[str] | None = None
) -> np.ndarray:
    vals = []
    for cid, g in groups.items():
        if systems is not None and cid not in systems:
            continue
        gg = g.dropna(subset=[col])
        if (
            len(gg) >= 3
            and gg["DockQ"].nunique() > 1
            and gg[col].nunique() > 1
        ):
            rho = spearmanr(gg[col], gg["DockQ"]).statistic
            if np.isfinite(rho):
                vals.append(float(rho))
    return np.array(vals)


def first_acceptable_rank(groups: dict[str, pd.DataFrame], col: str) -> np.ndarray:
    ranks = []
    for cid, g in groups.items():
        ordered = g.sort_values(col, ascending=False)
        acc = ordered.loc[ordered["acceptable"], "rank"]
        if len(acc):
            ranks.append(int(acc.iloc[0]))
    return np.array(ranks)


def recall_at_k(groups: dict[str, pd.DataFrame], col: str, k: int) -> float:
    hits = 0
    n = 0
    for cid, g in groups.items():
        ordered = g.sort_values(col, ascending=False)
        if ordered["acceptable"].any():
            n += 1
            if ordered["acceptable"].head(k).any():
                hits += 1
    return hits / n if n else float("nan")


def rerank_metrics(
    groups: dict[str, pd.DataFrame],
    score_column: str,
    systems: set[str] | None = None,
) -> dict[str, Any]:
    rho = within_spearman(groups, score_column, systems)
    first = first_acceptable_rank(groups, score_column)
    return {
        "within_system_spearman_median": float(np.median(rho)) if len(rho) else float("nan"),
        "within_system_spearman_n": int(len(rho)),
        "first_acceptable_rank_median": float(np.median(first)) if len(first) else float("nan"),
        "first_acceptable_rank_mean": float(np.mean(first)) if len(first) else float("nan"),
        "recall_at_1": recall_at_k(groups, score_column, 1),
        "recall_at_3": recall_at_k(groups, score_column, 3),
        "recall_at_5": recall_at_k(groups, score_column, 5),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ridge-config", default="configs/ml/candidate_ridge_v1.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if not HAS_TORCH:
        print(
            "ERROR: PyTorch is required (conda env physics_ai has torch 2.5.1).",
            file=sys.stderr,
        )
        return 1

    try:
        train_csv = Path(args.train_csv).expanduser().resolve()
        config_path = Path(args.config).expanduser().resolve()
        ridge_config_path = Path(args.ridge_config).expanduser().resolve()
        output_dir = Path(args.output_dir).expanduser().resolve()
        for path, name in [
            (train_csv, "training CSV"),
            (config_path, "config"),
            (ridge_config_path, "ridge config"),
        ]:
            if not path.is_file():
                raise FileNotFoundError(f"{name} not found: {path}")
        config = validate_config(load_json(config_path))
        ridge_config = load_json(ridge_config_path)
        if not isinstance(config.get("nn"), dict):
            raise ValueError("config requires an 'nn' object")
        output_paths = {key: output_dir / name for key, name in OUTPUT_NAMES.items()}
        check_output_targets(list(output_paths.values()), overwrite=bool(args.overwrite))

        raw = pd.read_csv(train_csv, low_memory=False)
        frame, input_audit = validate_candidate_frame(
            raw,
            config,
            require_target=True,
            require_folds=True,
            source="training CSV",
        )

        oof, fold_results = oof_cv(frame, config=config, device=args.device)
        target_column = str(config["target_column"])
        threshold = float(config["target_threshold"])

        oof_candidate = candidate_metrics(
            oof[target_column].to_numpy(dtype=float),
            oof[PREDICTION_SCORE_COLUMN].to_numpy(dtype=float),
            threshold=threshold,
        )
        model_selected = select_candidates(
            oof,
            score_column=PREDICTION_SCORE_COLUMN,
            config=config,
            selector=str(config["model_name"]),
        )
        reference_selected = select_candidates(
            oof,
            score_column=str(config["reference_score_column"]),
            config=config,
            selector="AF-M full-precision rank-1",
        )
        model_selector_metrics = selector_metrics(
            model_selected, target_column=target_column, threshold=threshold
        )
        reference_selector_metrics = selector_metrics(
            reference_selected, target_column=target_column, threshold=threshold
        )
        bootstrap = paired_selector_bootstrap(
            model_selected, reference_selected, config=config
        )

        # Reference ridge selectors (regenerated grouped OOF, same discipline).
        groups = {cid: g for cid, g in oof.groupby("complex_id")}
        rank1_ok = set(
            reference_selected.loc[reference_selected["acceptable"], "complex_id"]
        )
        any_ok = {cid for cid, g in groups.items() if g["acceptable"].any()}
        rescuable = any_ok - rank1_ok

        ridge_v1_oof = grouped_ridge_oof(
            frame,
            list(ridge_config["feature_columns"]),
            penalty=float(ridge_config["ridge_penalty"]),
        )
        ridge_v1_selected = select_candidates(
            ridge_v1_oof,
            score_column=PREDICTION_SCORE_COLUMN,
            config=config,
            selector="candidate_ridge_v1 (regenerated grouped OOF)",
        )
        ridge_meanj_oof = grouped_ridge_oof(frame, ["mean_j"], penalty=1.0)
        ridge_meanj_selected = select_candidates(
            ridge_meanj_oof,
            score_column=PREDICTION_SCORE_COLUMN,
            config=config,
            selector="ridge mean_j",
        )
        ridge_native_meanj_oof = grouped_ridge_oof(
            frame, V1_FEATURES + ["mean_j"], penalty=1.0
        )
        ridge_native_meanj_selected = select_candidates(
            ridge_native_meanj_oof,
            score_column=PREDICTION_SCORE_COLUMN,
            config=config,
            selector="ridge 5native+mean_j",
        )

        comparison = {
            "model_vs_rank1": bootstrap,
            "model_vs_ridge_v1": paired_selector_bootstrap(
                model_selected, ridge_v1_selected, config=config
            ),
            "model_vs_ridge_meanj": paired_selector_bootstrap(
                model_selected, ridge_meanj_selected, config=config
            ),
            "model_vs_ridge_native_meanj": paired_selector_bootstrap(
                model_selected, ridge_native_meanj_selected, config=config
            ),
            "ridge_v1_vs_rank1": paired_selector_bootstrap(
                ridge_v1_selected, reference_selected, config=config
            ),
            "ridge_v1_candidate_metrics": candidate_metrics(
                ridge_v1_oof[target_column].to_numpy(dtype=float),
                ridge_v1_oof[PREDICTION_SCORE_COLUMN].to_numpy(dtype=float),
                threshold=threshold,
            ),
            "ridge_meanj_selector": selector_metrics(
                ridge_meanj_selected, target_column=target_column, threshold=threshold
            ),
            "ridge_native_meanj_selector": selector_metrics(
                ridge_native_meanj_selected,
                target_column=target_column,
                threshold=threshold,
            ),
        }

        # Rerank (within-system) metrics for NN and references.
        rerank_rows = []
        for label, src, score_col in [
            ("pairwise_jaccard_nn", oof, PREDICTION_SCORE_COLUMN),
            ("AF-M rank-1 confidence", oof, str(config["reference_score_column"])),
            ("candidate_ridge_v1", ridge_v1_oof, PREDICTION_SCORE_COLUMN),
        ]:
            gg = {cid: g for cid, g in src.groupby("complex_id")}
            all_metrics = rerank_metrics(gg, score_col)
            resc_gg = {cid: g for cid, g in gg.items() if cid in rescuable}
            resc_metrics = rerank_metrics(resc_gg, score_col)
            row = {
                "selector": label,
                "systems": len(gg),
                **{k: v for k, v in all_metrics.items()},
                "rescuable_systems": len(resc_gg),
                **{f"rescuable_{k}": v for k, v in resc_metrics.items()},
            }
            rerank_rows.append(row)
        rerank_frame = pd.DataFrame(rerank_rows)

        model_artifact = {
            "model_schema_version": 3,
            "model_name": str(config["model_name"]),
            "task": str(config["task"]),
            "created_at_utc": utc_now(),
            "framework": f"pytorch {torch.__version__}",
            "device": args.device,
            "config": config,
            "feature_columns": list(config["feature_columns"]),
            "target_definition": {
                "column": target_column,
                "positive_when": f"{target_column} >= {threshold}",
                "threshold": threshold,
            },
            "training_input": {
                "path": portable_path(train_csv),
                "sha256": sha256_file(train_csv),
                **input_audit,
            },
            "config_input": {
                "path": portable_path(config_path),
                "sha256": sha256_file(config_path),
            },
            "training_system_ids": sorted(
                frame[str(config["group_column"])].astype(str).unique().tolist()
            ),
            "fold_audits": [
                {k: v for k, v in result.items() if k != "state_dict"}
                for result in fold_results
            ],
            "oof_candidate_metrics": oof_candidate,
            "oof_selector_metrics": {
                "model": model_selector_metrics,
                "reference": reference_selector_metrics,
                "paired_bootstrap": bootstrap,
            },
            "comparison_metrics": comparison,
            "rerank_metrics": rerank_frame.to_dict("records"),
            "limitation": (
                "Exploratory Training500 grouped-CV study. The public data has no "
                "PINDER-AF2 pair table, so no frozen-holdout evaluation was run."
            ),
        }
        summary = {
            "created_at_utc": utc_now(),
            "model_name": str(config["model_name"]),
            "framework": f"pytorch {torch.__version__}",
            "training_input_audit": input_audit,
            "fold_audits": [
                {k: v for k, v in result.items() if k != "state_dict"}
                for result in fold_results
            ],
            "oof_candidate_metrics": oof_candidate,
            "oof_selector_metrics": {
                "model": model_selector_metrics,
                "reference": reference_selector_metrics,
            },
            "oof_paired_bootstrap": bootstrap,
            "output_files": {
                key: portable_path(path) for key, path in output_paths.items()
            },
        }

        selector_columns = list(config["key_columns"]) + [
            target_column,
            "acceptable",
            PREDICTION_SCORE_COLUMN,
            str(config["reference_score_column"]),
            "selector",
        ]
        selector_choices = pd.concat(
            [model_selected[selector_columns], reference_selected[selector_columns]],
            ignore_index=True,
        )

        # Persist per-fold weights for reproducibility.
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for result in fold_results:
            torch.save(
                {
                    "fold": result["fold"],
                    "hidden_sizes": result["hidden_sizes"],
                    "feature_columns": list(config["feature_columns"]),
                    "preprocessing": result["preprocessing"],
                    "state_dict": result["state_dict"],
                },
                checkpoint_dir / f"fold_{result['fold']}.pt",
            )

        atomic_write_json(output_paths["model"], model_artifact)
        atomic_write_json(output_paths["summary"], summary)
        atomic_write_csv(output_paths["oof_predictions"], oof)
        atomic_write_csv(
            output_paths["candidate_metrics"], pd.DataFrame([oof_candidate])
        )
        atomic_write_csv(output_paths["selector_choices"], selector_choices)
        atomic_write_csv(
            output_paths["selector_summary"],
            pd.DataFrame([reference_selector_metrics, model_selector_metrics]),
        )
        atomic_write_csv(output_paths["bootstrap"], pd.DataFrame([bootstrap]))
        atomic_write_json(output_paths["comparison"], comparison)
        atomic_write_csv(output_paths["rerank"], rerank_frame)

        print(
            f"Training complete: model={config['model_name']} "
            f"systems={input_audit['systems']} rows={input_audit['rows']} "
            f"acc={model_selector_metrics['acceptable_rate']:.4f} "
            f"vs rank1={reference_selector_metrics['acceptable_rate']:.4f} "
            f"diff={bootstrap['delta_acceptable_rate']:+.4f} "
            f"CI [{bootstrap['delta_acceptable_rate_ci_low']:+.4f}, "
            f"{bootstrap['delta_acceptable_rate_ci_high']:+.4f}]"
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
