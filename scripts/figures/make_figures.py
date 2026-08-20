#!/usr/bin/env python3
"""Generate the four data figures from committed CSV artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "blue": "#31688E",
    "green": "#35B779",
    "orange": "#E38C2D",
    "purple": "#7B5AA6",
    "gray": "#6B7280",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "svg.fonttype": "none",
            "svg.hashsalt": "afm-interface-reliability",
        }
    )


def save_svg(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "afm-interface-reliability"},
    )
    plt.close(fig)
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2)) / denominator
    return float(center - half), float(center + half)


def top1_figure(
    training_csv: Path,
    selector_choices_csv: Path,
    within_rank_selector_choices_csv: Path,
    output: Path,
) -> None:
    training = pd.read_csv(training_csv, low_memory=False)
    top1 = (
        training.sort_values(
            ["complex_id", "ranking_confidence", "rank"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("complex_id", as_index=False)
        .first()
    )
    oracle = (
        training.sort_values(
            ["complex_id", "DockQ", "rank"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("complex_id", as_index=False)
        .first()
    )
    choices = pd.read_csv(selector_choices_csv)
    within_rank_choices = pd.read_csv(within_rank_selector_choices_csv)
    groups = [
        ("Training500\nAF-M top-1", top1),
        ("Training500\noracle best-of-20", oracle),
    ]
    for selector in ["AF-M full-precision rank-1", "candidate_ridge_v1"]:
        groups.append(
            (
                "PINDER-AF2\n" + ("AF-M top-1" if selector.startswith("AF-M") else "Candidate Ridge"),
                choices.loc[choices["selector"].eq(selector)],
            )
        )
    groups.append(
        (
            "PINDER-AF2\nWithin-rank",
            within_rank_choices.loc[
                within_rank_choices["selector"].eq("candidate_ridge_v1_within_rank")
            ],
        )
    )

    labels: list[str] = []
    rates: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for label, frame in groups:
        successes = int(pd.to_numeric(frame["DockQ"]).ge(0.23).sum())
        total = int(len(frame))
        low, high = wilson(successes, total)
        labels.append(label)
        rates.append(successes / total)
        lows.append(low)
        highs.append(high)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    rate_array = np.asarray(rates)
    low_array = np.asarray(lows)
    high_array = np.asarray(highs)
    ax.errorbar(
        x[:4],
        rate_array[:4],
        yerr=[rate_array[:4] - low_array[:4], high_array[:4] - rate_array[:4]],
        fmt="o",
        markersize=7,
        capsize=4,
        color=COLORS["blue"],
        ecolor=COLORS["gray"],
        linewidth=1.4,
    )
    ax.errorbar(
        x[4:],
        rate_array[4:],
        yerr=[rate_array[4:] - low_array[4:], high_array[4:] - rate_array[4:]],
        fmt="o",
        markersize=7,
        capsize=4,
        color=COLORS["orange"],
        ecolor=COLORS["orange"],
        linewidth=1.4,
    )
    for index, rate in enumerate(rates):
        ax.text(index, rate + 0.045, f"{rate:.1%}", ha="center", va="bottom")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Acceptable selected structure (DockQ ≥ 0.23)")
    ax.set_ylim(0.45, 0.87)
    ax.set_xlim(-0.3, 4.5)
    ax.set_title("Sampling headroom did not translate into holdout reranking gain")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.text(
        4,
        rates[4] - 0.055,
        "+1/180 vs AF-M\npaired 95% CI includes 0",
        ha="center",
        va="top",
        fontsize=8,
        color=COLORS["orange"],
    )
    save_svg(fig, output)


def metric_figure(metric_csv: Path, within_csv: Path, output: Path) -> None:
    metrics = pd.read_csv(metric_csv).set_index("metric")
    within = pd.read_csv(within_csv).set_index("metric")
    selected = ["ranking_confidence", "iptm_full_precision", "pDockQ2_min", "iLIS", "ipSAE"]
    display = ["AF-M ranking", "ipTM", "pDockQ2-min", "iLIS", "ipSAE"]
    auc = metrics.loc[selected, "roc_auc"].to_numpy(dtype=float)
    rho = within.loc[selected, "median_within_system_spearman"].to_numpy(dtype=float)
    x = np.arange(len(selected))

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    axes[0].bar(x, auc, color=COLORS["blue"])
    axes[0].set_ylim(0.5, 0.9)
    axes[0].set_ylabel("Candidate-level ROC-AUC")
    axes[0].set_title("Across all candidates")
    axes[1].bar(x, rho, color=COLORS["orange"])
    axes[1].axhline(0, color="#111827", linewidth=0.8)
    axes[1].set_ylim(-0.05, 0.35)
    axes[1].set_ylabel("Median within-system Spearman ρ")
    axes[1].set_title("Within each 20-candidate system")
    for ax in axes:
        ax.set_xticks(x, display, rotation=28, ha="right")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    fig.suptitle("Strong candidate discrimination does not imply strong within-system ranking", fontweight="bold")
    fig.tight_layout()
    save_svg(fig, output)


def within_system_reranking_figure(summary_csv: Path, output: Path) -> None:
    summary = pd.read_csv(summary_csv).set_index("metric")["value"]
    labels = ["Candidate Ridge v1", "AF-M ranking", "Within-rank"]
    medians = np.asarray(
        [
            summary["median_within_system_spearman_candidate_ridge_v1"],
            summary["median_within_system_spearman_AF_M_ranking"],
            summary["median_within_system_spearman_candidate_ridge_v1_within_rank"],
        ],
        dtype=float,
    )
    delta_afm = float(summary["delta_median_vs_AF_M_ranking"])
    low_afm = float(summary["delta_median_vs_AF_M_ranking_ci_low"])
    high_afm = float(summary["delta_median_vs_AF_M_ranking_ci_high"])
    delta_ridge = float(summary["delta_median_vs_candidate_ridge_v1"])
    low_ridge = float(summary["delta_median_vs_candidate_ridge_v1_ci_low"])
    high_ridge = float(summary["delta_median_vs_candidate_ridge_v1_ci_high"])

    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    x = np.arange(len(labels))
    ax.bar(
        x,
        medians,
        color=[COLORS["gray"], COLORS["blue"], COLORS["orange"]],
        width=0.68,
    )
    for index, value in enumerate(medians):
        ax.text(index, value + 0.007, f"{value:.3f}", ha="center", va="bottom")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.28)
    ax.set_ylabel("Median within-system Spearman ρ")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)

    fig.suptitle(
        "Within-rank has the highest point estimate, but no significant difference",
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.89,
        "Median per-system Spearman ρ; grouped out-of-fold Training500 (n = 500)",
        ha="center",
        color=COLORS["gray"],
    )
    fig.text(
        0.5,
        0.075,
        (
            f"Within-rank − AF-M: {delta_afm:+.3f} "
            f"(95% CI {low_afm:+.3f} to {high_afm:+.3f});  "
            f"within-rank − Ridge v1: {delta_ridge:+.3f} "
            f"({low_ridge:+.3f} to {high_ridge:+.3f})"
        ),
        ha="center",
        fontsize=8.5,
        color=COLORS["gray"],
    )
    fig.text(
        0.5,
        0.025,
        "Paired bootstrap resampled the same systems for both methods; both intervals include zero.",
        ha="center",
        fontsize=8.5,
        color=COLORS["gray"],
    )
    fig.subplots_adjust(top=0.79, bottom=0.24, left=0.12, right=0.98)
    save_svg(fig, output)


def ablation_figure(ablation_csv: Path, output: Path) -> None:
    frame = pd.read_csv(ablation_csv)
    labels = ["AF only", "+ physics", "+ ensemble", "all features"]
    x = np.arange(len(frame))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - width / 2, frame["roc_auc"], width, label="ROC-AUC", color=COLORS["blue"])
    ax.bar(x + width / 2, frame["average_precision"], width, label="Average precision", color=COLORS["green"])
    ax.set_xticks(x, labels)
    ax.set_ylim(0.72, 0.86)
    ax.set_ylabel("Grouped out-of-fold score")
    ax.set_title("System-risk feature ablation on Training500 (exploratory)")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    save_svg(fig, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-csv", type=Path, required=True)
    parser.add_argument("--selector-choices", type=Path, required=True)
    parser.add_argument("--within-rank-selector-choices", type=Path, required=True)
    parser.add_argument("--metric-table", type=Path, required=True)
    parser.add_argument("--within-system-table", type=Path, required=True)
    parser.add_argument("--within-rank-summary", type=Path, required=True)
    parser.add_argument("--ablation-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_style()
    top1_figure(
        args.training_csv,
        args.selector_choices,
        args.within_rank_selector_choices,
        args.output_dir / "top1_vs_oracle.svg",
    )
    metric_figure(args.metric_table, args.within_system_table, args.output_dir / "metric_performance.svg")
    within_system_reranking_figure(
        args.within_rank_summary,
        args.output_dir / "within_system_reranking.svg",
    )
    ablation_figure(args.ablation_table, args.output_dir / "ensemble_ablation.svg")
    print(f"Generated four data figures in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
