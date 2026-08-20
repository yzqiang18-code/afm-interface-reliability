#!/usr/bin/env python3
"""Exploratory diagnostic: candidate-level ensemble-consistency signal.

Question
--------
Does the pairwise similarity structure inside each 20-candidate system
(contact Jaccard, interface-residue Jaccard, receptor-aligned ligand RMSD)
carry *within-system selection* signal: information that ranks candidates by
DockQ inside one system, which the system-level consistency features
(mean_contact_jaccard etc.) provably cannot provide because they are constant
within a system?

Sections
--------
1. Data quality of the pair table and its join with the candidate table.
2. Within-system signal of pairwise-derived per-candidate features, compared
   with AF-M's own confidence and the existing cluster features.
3. Consistency x confidence combos (within-system z-scored sums).
4. System-level paired bootstrap vs AF-M rank-1 for the best combos.
5. Where the signal lives: sanity probes.
6. Interaction and conditional probes focused on the rescuable subset.
7. Rescuable-system within-AUC screen over all native-independent features.
8. Grouped-CV ridge on continuous DockQ with and without the new features
   (the "cheap ML ceiling" test before any neural model).
9. Rescuable-system deep dive: cluster structure of the acceptable minority,
   cluster-level ranks of the correct cluster, and gated cluster selectors.

Discipline: no native-derived quantity is used as a feature. DockQ and the
acceptable label are evaluation targets only. All inputs are
candidate-vs-candidate similarities and AF-M's own per-candidate scores.
Sections 6-9 are exploratory probes evaluated on the full Training500 cohort;
they involve multiple comparisons and must be re-validated under grouped
out-of-fold discipline before any frozen-holdout use.
"""

from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", message="An input array is constant")

ACCEPT_DOCKQ = 0.23
BOOTSTRAP_N = 2000
RNG = np.random.default_rng(20240820)


# ---------------------------------------------------------------------------
# Loading and quality control
# ---------------------------------------------------------------------------

def as_bool(series: pd.Series) -> pd.Series:
    """Robustly parse a column that may be bool or 'True'/'False' strings."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().eq("true")


def load_data(pairs_path: str, cand_path: str):
    pairs = pd.read_csv(pairs_path)
    cand = pd.read_csv(cand_path)
    cand["acceptable"] = cand["DockQ"] >= ACCEPT_DOCKQ
    return pairs, cand


def quality_report(pairs: pd.DataFrame, cand: pd.DataFrame) -> dict:
    report: dict = {}
    report["pair_rows"] = len(pairs)
    report["pair_systems"] = int(pairs["complex_id"].nunique())
    per_sys = pairs.groupby("complex_id").size()
    report["pairs_per_system"] = (
        int(per_sys.min()),
        int(per_sys.max()),
    )

    valid = as_bool(pairs["jaccard_valid"])
    valid_cb8 = as_bool(pairs["jaccard_cb8_valid"])
    report["jaccard_valid_fraction"] = float(valid.mean())
    report["jaccard_cb8_valid_fraction"] = float(valid_cb8.mean())
    report["rmsd_nonnull_fraction"] = float(
        pairs["receptor_aligned_ligand_rmsd"].notna().mean()
    )
    reasons = pairs.loc[~valid, "jaccard_reason"].fillna("(empty)").value_counts()
    report["invalid_reasons"] = reasons.to_dict()

    # Identity coverage: every candidate should appear in exactly 19 pairs.
    def identities(side: int) -> pd.DataFrame:
        return pairs[["complex_id", f"model_weight_{side}", f"seed_{side}"]].rename(
            columns={f"model_weight_{side}": "model_weight", f"seed_{side}": "seed"}
        )

    both = pd.concat([identities(1), identities(2)], ignore_index=True)
    counts = both.groupby(["complex_id", "model_weight", "seed"]).size()
    report["candidate_coverage_min_max"] = (int(counts.min()), int(counts.max()))
    report["candidates_in_pairs"] = int(len(counts))
    report["candidates_in_table"] = int(
        cand.groupby(["complex_id", "model_weight", "seed"]).ngroups
    )
    report["candidates_per_system_table"] = (
        int(cand.groupby("complex_id").size().min()),
        int(cand.groupby("complex_id").size().max()),
    )
    return report


# ---------------------------------------------------------------------------
# Per-candidate features from the pair table
# ---------------------------------------------------------------------------

def _directed(pairs: pd.DataFrame, self_side: int, other_side: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "complex_id": pairs["complex_id"],
            "mw_self": pairs[f"model_weight_{self_side}"],
            "seed_self": pairs[f"seed_{self_side}"],
            "mw_other": pairs[f"model_weight_{other_side}"],
            "seed_other": pairs[f"seed_{other_side}"],
            "j": pairs["jaccard"],
            "j_cb8": pairs["jaccard_cb8"],
            "irj": pairs["interface_residue_jaccard"],
            "rmsd": pairs["receptor_aligned_ligand_rmsd"],
        }
    )


def build_candidate_features(pairs: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    valid = as_bool(pairs["jaccard_valid"])
    valid_cb8 = as_bool(pairs["jaccard_cb8_valid"])
    pairs = pairs.copy()
    pairs.loc[~valid, "jaccard"] = np.nan
    pairs.loc[~valid, ["interface_residue_jaccard", "interface_residue_jaccard_a",
                       "interface_residue_jaccard_b"]] = np.nan
    pairs.loc[~valid_cb8, "jaccard_cb8"] = np.nan

    long = pd.concat(
        [_directed(pairs, 1, 2), _directed(pairs, 2, 1)], ignore_index=True
    )

    other_info = cand[
        ["complex_id", "model_weight", "seed", "iptm_full_precision", "pDockQ2_min"]
    ].rename(
        columns={
            "model_weight": "mw_other",
            "seed": "seed_other",
            "iptm_full_precision": "iptm_other",
            "pDockQ2_min": "pdockq_other",
        }
    )
    long = long.merge(
        other_info, on=["complex_id", "mw_other", "seed_other"], how="left"
    )
    if long["iptm_other"].isna().any():
        raise ValueError("Pair rows reference candidates missing from the table")

    key = ["complex_id", "mw_self", "seed_self"]
    long["j_x_iptm"] = long["j"] * long["iptm_other"]
    long["j_x_pdockq"] = long["j"] * long["pdockq_other"]
    long["same_mw"] = long["mw_self"].eq(long["mw_other"])

    def top3_mean(s: pd.Series) -> float:
        v = s.dropna()
        if len(v) == 0:
            return np.nan
        return float(v.nlargest(min(3, len(v))).mean())

    agg = long.groupby(key).agg(
        mean_j=("j", "mean"),
        median_j=("j", "median"),
        max_j=("j", "max"),
        top3_j=("j", top3_mean),
        mean_irj=("irj", "mean"),
        mean_j_cb8=("j_cb8", "mean"),
        mean_rmsd=("rmsd", "mean"),
        n_valid_j=("j", "count"),
        sum_j=("j", "sum"),
        sum_j_iptm=("j_x_iptm", "sum"),
        sum_j_pdockq=("j_x_pdockq", "sum"),
    )
    # Within-model-weight (across-seed) and across-model-weight variants.
    same_mw = long[long["same_mw"]].groupby(key)["j"].mean().rename("mean_j_same_mw")
    diff_mw = long[~long["same_mw"]].groupby(key)["j"].mean().rename("mean_j_diff_mw")

    feats = agg.join(same_mw).join(diff_mw).reset_index()
    feats["sw_iptm"] = feats["sum_j_iptm"] / feats["sum_j"]
    feats["sw_pdockq"] = feats["sum_j_pdockq"] / feats["sum_j"]
    feats = feats.drop(columns=["sum_j", "sum_j_iptm", "sum_j_pdockq"])

    feats = feats.rename(
        columns={"mw_self": "model_weight", "seed_self": "seed"}
    )
    merged = cand.merge(
        feats, on=["complex_id", "model_weight", "seed"], how="left", validate="1:1"
    )
    return merged


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def system_groups(cand: pd.DataFrame) -> dict:
    return {cid: g for cid, g in cand.groupby("complex_id")}


def within_spearman(groups: dict, col: str, systems=None) -> np.ndarray:
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


def selector_picks(groups: dict, col: str, higher_better: bool = True) -> pd.DataFrame:
    """Per-system candidate chosen by argmax/argmin of col (rank-1 fallback)."""
    picks = []
    for cid, g in groups.items():
        gg = g.dropna(subset=[col])
        if gg.empty:
            pick = g.loc[g["rank"] == 1].iloc[0]
        else:
            idx = gg[col].idxmax() if higher_better else gg[col].idxmin()
            pick = g.loc[idx]
        picks.append(pick)
    return pd.DataFrame(picks)


def oracle_picks(groups: dict) -> pd.DataFrame:
    picks = []
    for cid, g in groups.items():
        if g["acceptable"].any():
            pick = g.loc[g["DockQ"].idxmax()]
        else:
            pick = g.loc[g["rank"] == 1].iloc[0]
        picks.append(pick)
    return pd.DataFrame(picks)


def rank1_picks(groups: dict) -> pd.DataFrame:
    picks = []
    for cid, g in groups.items():
        picks.append(g.loc[g["rank"] == 1].iloc[0])
    return pd.DataFrame(picks)


def within_system_auc(groups: dict, col: str, systems) -> tuple[float, int]:
    """Mean across systems of within-system AUC (acceptable vs not)."""
    aucs = []
    for cid, g in groups.items():
        if cid not in systems:
            continue
        gg = g.dropna(subset=[col])
        pos = gg.loc[gg["acceptable"], col].to_numpy(dtype=float)
        neg = gg.loc[~gg["acceptable"], col].to_numpy(dtype=float)
        if len(pos) == 0 or len(neg) == 0:
            continue
        diff = pos[:, None] - neg[None, :]
        auc = (float((diff > 0).sum()) + 0.5 * float((diff == 0).sum())) / (
            len(pos) * len(neg)
        )
        aucs.append(auc)
    if not aucs:
        return float("nan"), 0
    return float(np.mean(aucs)), len(aucs)


def paired_bootstrap(sel_a: pd.DataFrame, sel_b: pd.DataFrame) -> tuple[float, float]:
    """System-level paired bootstrap CI for acceptable-rate difference (a - b)."""
    a = sel_a.set_index("complex_id")["acceptable"].astype(float)
    b = sel_b.set_index("complex_id")["acceptable"].astype(float)
    joined = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    va = joined["a"].to_numpy()
    vb = joined["b"].to_numpy()
    m = len(va)
    idx = RNG.integers(0, m, size=(BOOTSTRAP_N, m))
    diffs = va[idx].mean(axis=1) - vb[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def z_within(cand: pd.DataFrame, col: str) -> pd.Series:
    def _z(s: pd.Series) -> pd.Series:
        sd = s.std()
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / sd

    return cand.groupby("complex_id")[col].transform(_z)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pairs",
        default="results/data/training500_consistency_pairs.csv.gz",
    )
    ap.add_argument("--candidates", default="results/data/training500_candidates.csv.gz")
    ap.add_argument("--features-out", default=None)
    args = ap.parse_args()

    pairs, cand = load_data(args.pairs, args.candidates)

    print("=" * 78)
    print("1. DATA QUALITY")
    print("=" * 78)
    qc = quality_report(pairs, cand)
    for k, v in qc.items():
        print(f"  {k}: {v}")

    cand = build_candidate_features(pairs, cand)
    if args.features_out:
        cand.to_csv(args.features_out, index=False)
        print(f"\n  per-candidate features written to {args.features_out}")

    groups = system_groups(cand)
    all_systems = set(groups)

    # System classes (should reproduce docs/CONSISTENCY.md numbers).
    rank1 = rank1_picks(groups)
    oracle = oracle_picks(groups)
    rank1_ok = set(rank1.loc[rank1["acceptable"], "complex_id"])
    any_ok = {
        cid for cid, g in groups.items() if g["acceptable"].any()
    }
    rescuable = any_ok - rank1_ok
    sampling_failure = all_systems - any_ok
    print("\n  system classes:")
    print(f"    total systems          : {len(all_systems)}")
    print(f"    rank-1 acceptable       : {len(rank1_ok)}")
    print(f"    rescuable               : {len(rescuable)}")
    print(f"    sampling failure        : {len(sampling_failure)}")
    print(
        f"    AF-M rank-1 acceptable rate : {rank1['acceptable'].mean():.4f} | "
        f"oracle : {oracle['acceptable'].mean():.4f}"
    )

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("2. WITHIN-SYSTEM SIGNAL OF CANDIDATE-LEVEL CONSISTENCY FEATURES")
    print("=" * 78)

    print("\n  reference selectors:")
    print(
        f"    AF-M rank-1            : acc {rank1['acceptable'].mean():.4f} "
        f"mean DockQ {rank1['DockQ'].mean():.4f}"
    )
    for col, hb in [
        ("ranking_confidence", True),
        ("iptm_full_precision", True),
        ("pDockQ2_min", True),
        ("cluster_support_fraction", True),
    ]:
        picks = selector_picks(groups, col, hb)
        rho_all = within_spearman(groups, col)
        rho_res = within_spearman(groups, col, rescuable)
        print(
            f"    argmax {col:<28s}: acc {picks['acceptable'].mean():.4f} "
            f"mean DockQ {picks['DockQ'].mean():.4f} | "
            f"within-Spearman median {np.median(rho_all):+.3f} (n={len(rho_all)}, "
            f"rescuable {np.median(rho_res):+.3f}, n={len(rho_res)})"
        )

    features = [
        ("mean_j", True),
        ("median_j", True),
        ("max_j", True),
        ("top3_j", True),
        ("mean_irj", True),
        ("mean_j_cb8", True),
        ("mean_rmsd", False),
        ("mean_j_same_mw", True),
        ("mean_j_diff_mw", True),
        ("sw_iptm", True),
        ("sw_pdockq", True),
    ]

    print("\n  pairwise-derived features:")
    print(
        f"    {'feature':<16s} {'withinSp(all)':>14s} {'withinSp(resc)':>14s} "
        f"{'AUC(resc)':>10s} {'top1 acc':>9s} {'rescue':>7s}"
    )
    for col, hb in features:
        picks = selector_picks(groups, col, hb)
        rho_all = within_spearman(groups, col)
        rho_res = within_spearman(groups, col, rescuable)
        auc_res, n_auc = within_system_auc(groups, col, rescuable)
        rescue = picks.loc[picks["complex_id"].isin(rescuable), "acceptable"].mean()
        print(
            f"    {col:<16s} {np.median(rho_all):>+14.3f} "
            f"{np.median(rho_res):>+14.3f} {auc_res:>10.3f} "
            f"{picks['acceptable'].mean():>9.4f} {rescue:>7.3f}"
        )
    print(f"    (rescuable AUC averaged over {n_auc} systems with both classes)")

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("3. COMBOS: consistency x confidence (within-system z-scored sums)")
    print("=" * 78)

    cand["z_iptm"] = z_within(cand, "iptm_full_precision")
    cand["z_rankconf"] = z_within(cand, "ranking_confidence")
    cand["z_mean_j"] = z_within(cand, "mean_j")
    cand["z_sw_iptm"] = z_within(cand, "sw_iptm")
    cand["z_pdockq"] = z_within(cand, "pDockQ2_min")
    groups = system_groups(cand)

    cand["z_iptm__plus__z_mean_j"] = cand["z_iptm"] + cand["z_mean_j"]
    cand["z_rankconf__plus__z_mean_j"] = cand["z_rankconf"] + cand["z_mean_j"]
    cand["z_iptm__plus__z_sw_iptm"] = cand["z_iptm"] + cand["z_sw_iptm"]
    cand["z_iptm__plus__z_mean_j__plus__z_pdockq"] = (
        cand["z_iptm"] + cand["z_mean_j"] + cand["z_pdockq"]
    )
    groups = system_groups(cand)

    combos = [
        ("z_iptm+z_mean_j", "z_iptm__plus__z_mean_j"),
        ("z_rankconf+z_mean_j", "z_rankconf__plus__z_mean_j"),
        ("z_iptm+z_sw_iptm", "z_iptm__plus__z_sw_iptm"),
        ("z_sw_iptm alone", "z_sw_iptm"),
        ("z_iptm+z_mean_j+z_pdockq", "z_iptm__plus__z_mean_j__plus__z_pdockq"),
    ]
    print(
        f"    {'combo':<26s} {'withinSp(all)':>14s} {'withinSp(resc)':>14s} "
        f"{'AUC(resc)':>10s} {'top1 acc':>9s} {'rescue':>7s} {'meanDockQ':>9s}"
    )
    combo_results = {}
    for label, col in combos:
        picks = selector_picks(groups, col, True)
        rho_all = within_spearman(groups, col)
        rho_res = within_spearman(groups, col, rescuable)
        auc_res, _ = within_system_auc(groups, col, rescuable)
        rescue = picks.loc[picks["complex_id"].isin(rescuable), "acceptable"].mean()
        combo_results[label] = picks
        print(
            f"    {label:<26s} {np.median(rho_all):>+14.3f} "
            f"{np.median(rho_res):>+14.3f} {auc_res:>10.3f} "
            f"{picks['acceptable'].mean():>9.4f} {rescue:>7.3f} "
            f"{picks['DockQ'].mean():>9.4f}"
        )

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("4. PAIRED BOOTSTRAP vs AF-M RANK-1 (system level, 2000 resamples)")
    print("=" * 78)

    best = ["z_iptm+z_mean_j", "z_rankconf+z_mean_j", "z_iptm+z_sw_iptm",
            "z_iptm+z_mean_j+z_pdockq"]
    iptm_picks = selector_picks(groups, "iptm_full_precision", True)
    for label in best:
        picks = combo_results[label]
        rate = picks["acceptable"].mean()
        diff = rate - rank1["acceptable"].mean()
        lo, hi = paired_bootstrap(picks, rank1)
        lo2, hi2 = paired_bootstrap(picks, iptm_picks)
        print(
            f"    {label:<26s} acc {rate:.4f} vs rank-1 "
            f"{rank1['acceptable'].mean():.4f} (diff {diff:+.4f}, "
            f"95% CI [{lo:+.4f}, {hi:+.4f}]) | vs argmax-ipTM "
            f"diff {rate - iptm_picks['acceptable'].mean():+.4f} "
            f"(CI [{lo2:+.4f}, {hi2:+.4f}])"
        )

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("5. WHERE THE SIGNAL LIVES (sanity probes)")
    print("=" * 78)

    resc = cand[cand["complex_id"].isin(rescuable)]
    for col in ["mean_j", "top3_j", "sw_iptm"]:
        pos = resc.loc[resc["acceptable"], col].dropna()
        neg = resc.loc[~resc["acceptable"], col].dropna()
        print(
            f"    rescuable systems, {col:<10s}: acceptable median "
            f"{pos.median():.3f} (n={len(pos)}) vs not {neg.median():.3f} "
            f"(n={len(neg)})"
        )

    picks = combo_results["z_rankconf+z_mean_j"]
    res_picks = picks[picks["complex_id"].isin(rescuable)]
    print(
        f"\n    AF-M rank of consistency-picked candidates (rescuable systems): "
        f"median {res_picks['rank'].median():.0f}"
    )
    print(
        f"    its mean_j median {res_picks['mean_j'].median():.3f} vs "
        f"system mean_j median {cand['mean_j'].median():.3f}"
    )

    print(
        f"\n    mean jaccard same model weight (across seeds): "
        f"{cand['mean_j_same_mw'].mean():.3f} | "
        f"different model weight: {cand['mean_j_diff_mw'].mean():.3f}"
    )

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("6. INTERACTION AND CONDITIONAL PROBES (rescuable-focused)")
    print("=" * 78)

    cand["z_csf"] = z_within(cand, "cluster_support_fraction")
    cand["z_rmsd"] = z_within(cand, "mean_rmsd")
    cand["outlier_conf"] = cand["z_iptm"] - cand["z_mean_j"]
    cand["outlier_conf_rmsd"] = cand["z_iptm"] + cand["z_rmsd"]
    cand["iptm_csf"] = cand["z_iptm"] + cand["z_csf"]
    groups = system_groups(cand)

    print("\n  signal decomposition by system class (where does it live?):")
    failures = all_systems - any_ok
    print(
        f"    {'system class':<28s} {'Sp(mean_j)':>10s} {'AUC(mean_j)':>11s} "
        f"{'Sp(iptm)':>9s} {'AUC(iptm)':>10s}"
    )
    for name, sysset in [("rank-1 acceptable (300)", rank1_ok),
                         ("rescuable (88)", rescuable),
                         ("sampling failure (112)", failures)]:
        rho_j = within_spearman(groups, "mean_j", sysset)
        auc_j, _ = within_system_auc(groups, "mean_j", sysset)
        rho_i = within_spearman(groups, "iptm_full_precision", sysset)
        auc_i, _ = within_system_auc(groups, "iptm_full_precision", sysset)
        print(
            f"    {name:<28s} {np.median(rho_j):>+10.3f} {auc_j:>11.3f} "
            f"{np.median(rho_i):>+9.3f} {auc_i:>10.3f}"
        )

    print("\n  interaction combos:")
    for col in ["outlier_conf", "outlier_conf_rmsd", "iptm_csf"]:
        picks = selector_picks(groups, col, True)
        rho_all = within_spearman(groups, col)
        auc_res, _ = within_system_auc(groups, col, rescuable)
        rescue = picks.loc[picks["complex_id"].isin(rescuable), "acceptable"].mean()
        lo, hi = paired_bootstrap(picks, rank1)
        print(
            f"    {col:<20s} Sp(all) {np.median(rho_all):+.3f} "
            f"AUC(resc) {auc_res:.3f} top1 {picks['acceptable'].mean():.4f} "
            f"rescue {rescue:.3f} boot_vs_rank1 [{lo:+.3f},{hi:+.3f}]"
        )

    print("\n  conditional selector: rank-1 unless system mean_contact_jaccard")
    print("    < gate, then switch to argmax(outlier_conf):")
    for thr in [0.2, 0.3, 0.4, 0.5]:
        picks = []
        for cid, g in groups.items():
            if g["mean_contact_jaccard"].iloc[0] < thr:
                gg = g.dropna(subset=["outlier_conf"])
                picks.append(gg.loc[gg["outlier_conf"].idxmax()])
            else:
                picks.append(g.loc[g["rank"] == 1].iloc[0])
        picks = pd.DataFrame(picks)
        rescue = picks.loc[picks["complex_id"].isin(rescuable), "acceptable"].mean()
        lo, hi = paired_bootstrap(picks, rank1)
        n_switch = int((picks["rank"] != 1).sum())
        print(
            f"    gate<{thr:.1f}: acc {picks['acceptable'].mean():.4f} "
            f"rescue {rescue:.3f} switched {n_switch} "
            f"mdq {picks['DockQ'].mean():.4f} boot [{lo:+.3f},{hi:+.3f}]"
        )

    print("\n  break/save accounting for outlier_conf = z(iptm) - z(mean_j):")
    oc_picks = selector_picks(groups, "outlier_conf", True)
    p = oc_picks.set_index("complex_id")["acceptable"]
    r = rank1.set_index("complex_id")["acceptable"]
    saved = int((p & ~r).sum())
    broken = int((~p & r).sum())
    print(f"    rescued systems : {saved}  (all inside the 88 rescuable)")
    print(f"    broken systems  : {broken}  (all inside the 300 rank-1-correct)")
    print(
        f"    net             : {saved - broken:+d} "
        f"(acc {p.mean():.4f} vs rank-1 {r.mean():.4f})"
    )

    print("\n  what the best candidate looks like inside rescuable systems:")
    best_df = pd.DataFrame(
        [g.loc[g["DockQ"].idxmax()] for cid, g in groups.items() if cid in rescuable]
    )
    for label, frame in [("best candidate", best_df), ("all candidates", resc)]:
        print(
            f"    {label:<15s} median z_iptm {frame['z_iptm'].median():+.3f} | "
            f"z_mean_j {frame['z_mean_j'].median():+.3f} | "
            f"cluster_support {frame['cluster_support_fraction'].median():.3f} | "
            f"AF-M rank {frame['rank'].median():.0f}"
        )
    med = {cid: g["mean_j"].median() for cid, g in groups.items() if cid in rescuable}
    frac = np.mean([r["mean_j"] >= med[r["complex_id"]] for _, r in best_df.iterrows()])
    print(f"    best candidate at-or-above system median mean_j: {frac:.3f}")

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("7. RESCUABLE-SYSTEM WITHIN-AUC SCREEN OVER ALL FEATURES")
    print("=" * 78)

    exclude = {"DockQ", "direct_DockQ", "swapped_DockQ", "symmetry_gain",
               "acceptable", "rank", "model_weight", "seed", "cv_fold",
               "n_models", "ensemble_complete", "contact_pair_count"}
    num = [c for c in cand.columns
           if pd.api.types.is_numeric_dtype(cand[c]) and c not in exclude]
    rows = []
    for col in num:
        vals = cand[col].dropna()
        if len(vals) < 5000 or vals.nunique() < 5:
            continue
        auc, n = within_system_auc(groups, col, rescuable)
        if n == 0:
            continue
        rows.append((col, auc))
    rows.sort(key=lambda r: -abs(r[1] - 0.5))
    print(f"  {len(rows)} features screened; sorted by |AUC - 0.5|")
    print("  (multiple testing on 85 systems - exploratory only)")
    for col, auc in rows[:15]:
        side = "outlier-side" if auc < 0.5 else "consensus-side"
        print(f"    {col:<38s} AUC={auc:.3f}  ({side})")

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("8. GROUPED-CV RIDGE ON CONTINUOUS DockQ (cheap-ML ceiling test)")
    print("=" * 78)

    cand["z_ptm"] = z_within(cand, "ptm_full_precision")
    cand["z_ilis"] = z_within(cand, "iLIS")
    cand["z_ipsae"] = z_within(cand, "ipSAE")
    cand["z_iptm_x_mean_j"] = cand["z_iptm"] * cand["z_mean_j"]
    cand["z_iptm_x_csf"] = cand["z_iptm"] * cand["z_csf"]
    folds = sorted(cand["cv_fold"].unique())
    print(f"  folds: {folds}")

    def grouped_ridge_predict(feats, lam=1.0):
        X = cand[feats].to_numpy(dtype=float)
        X = np.nan_to_num(X)
        y = cand["DockQ"].to_numpy(dtype=float)
        pred = np.full(len(cand), np.nan)
        for f in folds:
            te = (cand["cv_fold"] == f).to_numpy()
            tr = ~te
            Xtr, ytr = X[tr], y[tr]
            mu, sd = Xtr.mean(0), Xtr.std(0)
            sd[sd == 0] = 1.0
            Xtr = (Xtr - mu) / sd
            A = Xtr.T @ Xtr + lam * np.eye(len(feats))
            w = np.linalg.solve(A, Xtr.T @ ytr)
            pred[te] = ((X[te] - mu) / sd) @ w
        cand["_pred"] = pred

    def evaluate(name, feats, lam=1.0):
        grouped_ridge_predict(feats, lam)
        picks = []
        for cid, g in system_groups(cand).items():
            picks.append(g.loc[g["_pred"].idxmax()])
        picks = pd.DataFrame(picks)
        rho = within_spearman(system_groups(cand), "_pred")
        rescue = picks.loc[picks["complex_id"].isin(rescuable), "acceptable"].mean()
        lo, hi = paired_bootstrap(picks, rank1)
        print(
            f"    {name:<46s} acc {picks['acceptable'].mean():.4f} "
            f"mdq {picks['DockQ'].mean():.4f} Sp(all) {np.median(rho):+.3f} "
            f"rescue {rescue:.3f} boot_vs_rank1 [{lo:+.3f},{hi:+.3f}]"
        )

    evaluate("baseline: iptm only", ["z_iptm"])
    evaluate(
        "baseline: 5 native features",
        ["z_iptm", "z_ptm", "z_pdockq", "z_ilis", "z_ipsae"],
    )
    evaluate(
        "5 native + mean_j",
        ["z_iptm", "z_ptm", "z_pdockq", "z_ilis", "z_ipsae", "z_mean_j"],
    )
    evaluate(
        "5 native + mean_j + csf + rmsd",
        ["z_iptm", "z_ptm", "z_pdockq", "z_ilis", "z_ipsae", "z_mean_j",
         "z_csf", "z_rmsd"],
    )
    evaluate(
        "5 native + mean_j + csf + rmsd + interactions",
        ["z_iptm", "z_ptm", "z_pdockq", "z_ilis", "z_ipsae", "z_mean_j",
         "z_csf", "z_rmsd", "z_iptm_x_mean_j", "z_iptm_x_csf"],
    )
    evaluate(
        "consistency only: mean_j+csf+rmsd+interaction",
        ["z_mean_j", "z_csf", "z_rmsd", "z_iptm_x_mean_j"],
    )

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("9. RESCUABLE-SYSTEM DEEP DIVE: the acceptable minority as a cluster")
    print("=" * 78)

    n_acc_sys = resc.groupby("complex_id")["acceptable"].sum()
    print("\n  acceptable candidates per rescuable system:")
    print(
        f"    median {n_acc_sys.median():.0f} | quartiles "
        f"{n_acc_sys.quantile(0.25):.0f}/{n_acc_sys.quantile(0.75):.0f} | "
        f"range {n_acc_sys.min():.0f}-{n_acc_sys.max():.0f}"
    )

    # Pairwise agreement by acceptability group (uses labels for grouping only).
    key = ["complex_id", "model_weight", "seed"]
    lk1 = cand[key + ["acceptable", "cluster_id"]].rename(
        columns={"model_weight": "model_weight_1", "seed": "seed_1",
                 "acceptable": "acc1", "cluster_id": "cl1"})
    lk2 = cand[key + ["acceptable", "cluster_id"]].rename(
        columns={"model_weight": "model_weight_2", "seed": "seed_2",
                 "acceptable": "acc2", "cluster_id": "cl2"})
    pp = pairs.merge(lk1, on=["complex_id", "model_weight_1", "seed_1"], how="left")
    pp = pp.merge(lk2, on=["complex_id", "model_weight_2", "seed_2"], how="left")
    pp["grp"] = np.where(pp["acc1"] & pp["acc2"], "acc-acc",
                  np.where(pp["acc1"] | pp["acc2"], "mixed", "unacc-unacc"))
    pp["same_cl"] = pp["cl1"].eq(pp["cl2"]) & pp["cl1"].ne(-1)

    print("\n  pairwise jaccard by acceptability group (labels group rows only):")
    for label, sysset in [("rescuable (88)", rescuable),
                          ("rank-1-correct (300)", rank1_ok)]:
        sub = pp[pp["complex_id"].isin(sysset)]
        parts = []
        for grp in ["acc-acc", "mixed", "unacc-unacc"]:
            g = sub.loc[sub["grp"] == grp, "jaccard"].dropna()
            parts.append(f"{grp} {g.mean():.3f} (n={len(g)})")
        print(f"    {label:<22s}: " + " | ".join(parts))
    sub = pp[pp["complex_id"].isin(rescuable)]
    for grp in ["acc-acc", "mixed", "unacc-unacc"]:
        m = sub["grp"] == grp
        print(
            f"    P(same contact cluster | {grp}) = "
            f"{sub.loc[m, 'same_cl'].mean():.3f}"
        )

    # Cluster-level structure (n_acc / purity are label-derived diagnostics).
    cl = cand.groupby(["complex_id", "cluster_id"]).agg(
        cluster_size=("rank", "count"),
        cluster_iptm=("iptm_full_precision", "mean"),
        cluster_hydro=("hydrophobic_contact_count", "mean"),
        n_acc=("acceptable", "sum"),
    ).reset_index()
    cand = cand.merge(
        cl.drop(columns=["n_acc"]), on=["complex_id", "cluster_id"], how="left"
    )

    print("\n  cluster-level structure in rescuable systems:")
    iptm_r, size_r, hydro_r, top_pure, correct_pure = [], [], [], [], []
    n_clusters = []
    for cid in rescuable:
        gg = cl[(cl["complex_id"] == cid) & (cl["cluster_id"] != -1)]
        if gg.empty:
            continue
        n_clusters.append(len(gg))
        correct_cl = gg.loc[gg["n_acc"].idxmax(), "cluster_id"]
        cc = gg[gg["cluster_id"] == correct_cl].iloc[0]
        correct_pure.append(cc["n_acc"] / cc["cluster_size"])
        r = gg["cluster_iptm"].rank(ascending=False, method="min")
        iptm_r.append(int(r[gg["cluster_id"] == correct_cl].iloc[0]))
        r = gg["cluster_size"].rank(ascending=False, method="min")
        size_r.append(int(r[gg["cluster_id"] == correct_cl].iloc[0]))
        r = gg["cluster_hydro"].rank(ascending=False, method="min")
        hydro_r.append(int(r[gg["cluster_id"] == correct_cl].iloc[0]))
        top = gg.loc[gg["cluster_iptm"].idxmax()]
        top_pure.append(top["n_acc"] / top["cluster_size"])
    print(f"    clusters per system: median {np.median(n_clusters):.1f}")
    print(f"    correct-cluster purity (n_acc/size): median "
          f"{np.median(correct_pure):.2f}")
    print(
        f"    ipTM-#1 cluster is pure-wrong in "
        f"{np.mean(np.array(top_pure) == 0):.0%} of rescuable systems"
    )
    for name, arr in [("cluster-mean ipTM", iptm_r), ("size", size_r),
                      ("cluster-mean hydrophobic", hydro_r)]:
        s = pd.Series(arr).value_counts(normalize=True).sort_index()
        txt = "  ".join(f"#{int(k)}:{v:.0%}" for k, v in s.items() if k <= 4)
        print(
            f"    correct cluster rank by {name:<24s}: "
            f"median {np.median(arr):.1f} | {txt}"
        )

    print("\n  gated cluster selectors (switch only when system "
          "mean_contact_jaccard < gate):")
    groups9 = system_groups(cand)

    def pick_in_system(g, mode):
        gg = g[g["cluster_id"] != -1]
        if gg.empty:
            return g.loc[g["rank"] == 1].iloc[0]
        cls = gg.groupby("cluster_id").agg(
            s=("cluster_size", "first"), m=("cluster_iptm", "first"),
            h=("cluster_hydro", "first"))
        if mode == "second_iptm":
            order = cls.sort_values("m", ascending=False).index
            target = order[1] if len(order) > 1 else order[0]
        elif mode == "best_minority_iptm":
            biggest = cls["s"].idxmax()
            rest = cls.drop(index=biggest)
            target = rest["m"].idxmax() if len(rest) else biggest
        elif mode == "max_hydro":
            target = cls["h"].idxmax()
        else:
            raise ValueError(mode)
        members = gg[gg["cluster_id"] == target]
        return members.loc[members["iptm_full_precision"].idxmax()]

    for gate in [0.3, 0.4, 0.5]:
        for mode in ["second_iptm", "best_minority_iptm", "max_hydro"]:
            picks = []
            for cid, g in groups9.items():
                if g["mean_contact_jaccard"].iloc[0] < gate:
                    picks.append(pick_in_system(g, mode))
                else:
                    picks.append(g.loc[g["rank"] == 1].iloc[0])
            picks = pd.DataFrame(picks)
            pk = picks.set_index("complex_id")["acceptable"]
            r = rank1.set_index("complex_id")["acceptable"]
            saved = int((pk & ~r).sum())
            broken = int((~pk & r).sum())
            lo, hi = paired_bootstrap(picks, rank1)
            rescue = picks.loc[picks["complex_id"].isin(rescuable),
                               "acceptable"].mean()
            print(
                f"    gate<{gate} {mode:<20s} acc {pk.mean():.4f} "
                f"rescue {rescue:.3f} save {saved} break {broken} "
                f"net {saved - broken:+d} boot [{lo:+.3f},{hi:+.3f}]"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
