"""
aggregate_results.py
====================
Reads all per-site/per-seed eval CSVs produced by run_all_sites.sh,
computes summary statistics, and produces two figures:

Fig A — 10-site EENS comparison (grouped bar, RL-Inv vs Track-B, mean ± std across seeds)
Fig B — EENS gap per site (RL-Inv benefit), ordered by magnitude

Also prints a console table and saves:
  results/all_sites/summary.csv        <- mean/std per site per policy
  results/all_sites/fig_eens_bars.png
  results/all_sites/fig_eens_gap.png

Usage:
  python aggregate_results.py
  python aggregate_results.py --results_dir results/all_sites --out_dir results/figures
"""
from __future__ import annotations

import os
import argparse
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────────────

SITES = [f"site{i}" for i in range(1, 11)]
SEEDS = [42, 123, 777]
POLICIES = {"RLInv": "RL-Inv (Track A)", "TrackB": "Track B (s,S+PPO)"}

COLORS = {
    "RLInv":  "#1F4E79",   # dark navy
    "TrackB": "#C55A11",   # burnt orange
}


# ── Load all CSVs ─────────────────────────────────────────────────────────────

def load_all_results(results_dir: str, scenario: str = "normal") -> pd.DataFrame:
    rows = []
    missing = []

    for site in SITES:
        for seed in SEEDS:
            for policy_key in POLICIES:
                path = os.path.join(
                    results_dir, site, f"seed{seed}", policy_key, f"eval_{scenario}.csv"
                )
                if not os.path.exists(path):
                    missing.append(path)
                    continue
                df = pd.read_csv(path)
                df["site"]   = site
                df["seed"]   = seed
                df["policy"] = policy_key
                rows.append(df)

    if missing:
        print(f"[WARN] {len(missing)} CSV(s) not found:")
        for p in missing[:10]:
            print(f"  {p}")
        if len(missing) > 10:
            print(f"  ... and {len(missing)-10} more")

    if not rows:
        raise FileNotFoundError(
            f"No eval CSVs found under {results_dir}. "
            "Run run_all_sites.sh first."
        )

    return pd.concat(rows, ignore_index=True)


# ── Summary statistics ────────────────────────────────────────────────────────

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of EENS and cost across episodes AND seeds, per site × policy."""
    # First: mean per seed (30 episodes each)
    per_seed = (
        df.groupby(["site", "seed", "policy"])
        ["EENS_kWh", "diesel_kWh", "cost_proxy", "stockout_events"]
        .mean()
        .reset_index()
    )
    # Then: mean ± std across seeds
    summary = (
        per_seed.groupby(["site", "policy"])
        ["EENS_kWh", "diesel_kWh", "cost_proxy", "stockout_events"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    return summary


def compute_gap(summary: pd.DataFrame) -> pd.DataFrame:
    """EENS gap = TrackB_EENS - RLInv_EENS  (positive = RL-Inv is better)."""
    rl  = summary[summary["policy"] == "RLInv"][["site", "EENS_kWh_mean", "EENS_kWh_std"]].rename(
        columns={"EENS_kWh_mean": "rl_eens", "EENS_kWh_std": "rl_std"}
    )
    tb  = summary[summary["policy"] == "TrackB"][["site", "EENS_kWh_mean", "EENS_kWh_std"]].rename(
        columns={"EENS_kWh_mean": "tb_eens", "EENS_kWh_std": "tb_std"}
    )
    gap = rl.merge(tb, on="site")
    gap["eens_gap"]      = gap["tb_eens"] - gap["rl_eens"]          # positive = RL-Inv better
    gap["eens_gap_pct"]  = gap["eens_gap"] / gap["tb_eens"] * 100   # % improvement
    gap["gap_se"]        = np.sqrt(gap["rl_std"]**2 + gap["tb_std"]**2)  # propagated uncertainty
    return gap.sort_values("eens_gap", ascending=False).reset_index(drop=True)


# ── Figure A: Grouped bar chart ───────────────────────────────────────────────

def plot_eens_bars(summary: pd.DataFrame, out_path: str):
    sites_ordered = SITES  # site1 … site10

    rl_means, rl_stds = [], []
    tb_means, tb_stds = [], []

    for site in sites_ordered:
        r = summary[(summary["site"] == site) & (summary["policy"] == "RLInv")]
        t = summary[(summary["site"] == site) & (summary["policy"] == "TrackB")]
        rl_means.append(r["EENS_kWh_mean"].values[0] if len(r) else np.nan)
        rl_stds.append( r["EENS_kWh_std"].values[0]  if len(r) else np.nan)
        tb_means.append(t["EENS_kWh_mean"].values[0] if len(t) else np.nan)
        tb_stds.append( t["EENS_kWh_std"].values[0]  if len(t) else np.nan)

    x      = np.arange(len(sites_ordered))
    width  = 0.35
    labels = [s.replace("site", "Site ") for s in sites_ordered]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8F9FA")

    bars_rl = ax.bar(x - width/2, rl_means, width, yerr=rl_stds,
                     color=COLORS["RLInv"],  alpha=0.88,
                     error_kw=dict(elinewidth=1.2, capsize=3),
                     label="RL-Inv (Track A)")
    bars_tb = ax.bar(x + width/2, tb_means, width, yerr=tb_stds,
                     color=COLORS["TrackB"], alpha=0.88,
                     error_kw=dict(elinewidth=1.2, capsize=3),
                     label="Track B (s,S + PPO)")

    ax.set_xlabel("Site", fontsize=12)
    ax.set_ylabel("EENS (kWh / 30 days)", fontsize=12)
    ax.set_title("Fig 6.x — EENS across all 10 ITU sites: RL-Inv vs Track B\n"
                 "(mean ± std across 3 seeds × 30 episodes)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out_path}")


# ── Figure B: Gap plot (ordered by benefit) ───────────────────────────────────

def plot_eens_gap(gap: pd.DataFrame, out_path: str):
    labels = [s.replace("site", "Site ") for s in gap["site"]]
    colors = [COLORS["RLInv"] if g >= 0 else COLORS["TrackB"] for g in gap["eens_gap"]]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8F9FA")

    ax.bar(labels, gap["eens_gap"], yerr=gap["gap_se"],
           color=colors, alpha=0.88,
           error_kw=dict(elinewidth=1.2, capsize=3))

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Site (ordered by RL-Inv benefit)", fontsize=12)
    ax.set_ylabel("EENS gap: Track B − RL-Inv  (kWh / 30 days)", fontsize=12)
    ax.set_title("Fig 6.y — Per-site EENS improvement of RL-Inv over Track B\n"
                 "(positive = RL-Inv better; bars ordered by gap magnitude)", fontsize=12)

    # Annotate % improvement on each bar
    for i, (g, pct) in enumerate(zip(gap["eens_gap"], gap["eens_gap_pct"])):
        if not np.isnan(g):
            ax.text(i, g + (abs(g) * 0.04 + 2),
                    f"{pct:+.0f}%", ha="center", va="bottom", fontsize=9,
                    color=COLORS["RLInv"] if g >= 0 else COLORS["TrackB"])

    pos_patch = mpatches.Patch(color=COLORS["RLInv"],  alpha=0.88, label="RL-Inv better")
    neg_patch = mpatches.Patch(color=COLORS["TrackB"], alpha=0.88, label="Track B better")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out_path}")


# ── Console table ─────────────────────────────────────────────────────────────

def print_table(summary: pd.DataFrame, gap: pd.DataFrame):
    print("\n" + "="*72)
    print(f"{'Site':<10} {'RL-Inv EENS':>14} {'Track-B EENS':>14} {'Gap':>10} {'Gap %':>8}")
    print("-"*72)
    for _, row in gap.iterrows():
        rl_row = summary[(summary["site"] == row["site"]) & (summary["policy"] == "RLInv")]
        tb_row = summary[(summary["site"] == row["site"]) & (summary["policy"] == "TrackB")]
        rl_s = rl_row["EENS_kWh_std"].values[0] if len(rl_row) else np.nan
        tb_s = tb_row["EENS_kWh_std"].values[0] if len(tb_row) else np.nan
        print(
            f"{row['site']:<10} "
            f"{row['rl_eens']:>8.1f}±{rl_s:>4.1f}  "
            f"{row['tb_eens']:>8.1f}±{tb_s:>4.1f}  "
            f"{row['eens_gap']:>+9.1f}  "
            f"{row['eens_gap_pct']:>+7.1f}%"
        )
    print("="*72)

    wins    = (gap["eens_gap"] > 0).sum()
    losses  = (gap["eens_gap"] < 0).sum()
    avg_pct = gap[gap["eens_gap"] > 0]["eens_gap_pct"].mean()
    print(f"\nRL-Inv wins on {wins}/10 sites | loses on {losses}/10 sites")
    if wins > 0:
        print(f"Average improvement on winning sites: {avg_pct:.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/all_sites")
    ap.add_argument("--out_dir",     default="results/all_sites")
    ap.add_argument("--scenario",    default="normal", choices=["normal","delayed"],
                    help="Which eval scenario to aggregate (default: normal)")
    args = ap.parse_args()

    print(f"Loading results from: {args.results_dir}  [scenario={args.scenario}]")
    df = load_all_results(args.results_dir, scenario=args.scenario)
    print(f"Loaded {len(df)} episode rows from "
          f"{df.groupby(['site','seed','policy']).ngroups} run combinations")

    summary = compute_summary(df)
    gap     = compute_gap(summary)

    # Save summary CSV
    summary_path = os.path.join(args.out_dir, f"summary_{args.scenario}.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"[Saved] {summary_path}")

    # Print table
    print_table(summary, gap)

    # Figures
    plot_eens_bars(
        summary,
        out_path=os.path.join(args.out_dir, f"fig_eens_bars_{args.scenario}.png")
    )
    plot_eens_gap(
        gap,
        out_path=os.path.join(args.out_dir, f"fig_eens_gap_{args.scenario}.png")
    )

    print("\nDone. Add fig_eens_bars.png and fig_eens_gap.png to your thesis Chapter 6.")


if __name__ == "__main__":
    main()
