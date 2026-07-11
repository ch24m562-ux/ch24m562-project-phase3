"""
hypothesis_matrix_hard_vs_all.py

CANONICAL H1/H2/H3 statistical comparison script — Phase 3 final grid.

Reads: results/phase3/master_summary.csv (44,000 rows: 11 policies x 10 sites
       x 4 scenarios x 10 seeds x 10 episodes/seed)

Produces ONE table comparing three core hypothesis pairs (RLInv vs A6,
RLInv vs TrackB, A6 vs TrackB) under two site groupings:
  - ALL SITES      : all 10 ITU/Zindi sites
  - HARD SITES     : site2, site5, site7 (the sites where at least one
                      policy's mean EENS exceeds the constrained-site
                      threshold in >=1 scenario -- see final_thesis_numbers.txt
                      Section 3 for the derivation of this site set)

Statistical unit: seed-level mean EENS (n=10 seeds), NOT raw episodes.
Episodes within a seed share an RNG stream (init_inv_frac draws, lead-time
draws) and are not independent -- pooling episodes as i.i.d. samples
understates standard error. This mirrors the discipline already used in
final_thesis_numbers.txt Section 10 and analyze_results.py.

Test: Welch's t-test (unequal variances assumed). Effect size: Cohen's d
using pooled sample std (ddof=1).

Output: prints the table to stdout AND writes it to
        results/tables/hypothesis_matrix_hard_vs_all.csv

Usage:
    python hypothesis_matrix_hard_vs_all.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

IN_CSV = "results/phase3/master_summary.csv"
OUT_CSV = "results/tables/hypothesis_matrix_hard_vs_all.csv"

HARD_SITES = ["site2", "site5", "site7"]
SCENARIOS = ["normal", "delayed", "monsoon", "extreme"]

COMPARISONS = [
    ("rlinv", "a6",     "RLInv vs A6"),
    ("rlinv", "trackb", "RLInv vs TrackB"),
    ("a6",    "trackb", "A6 vs TrackB"),
]


def seed_means(df, policy, site_filter=None):
    sub = df[df["policy"] == policy]
    if site_filter is not None:
        sub = sub[sub["site"].isin(site_filter)]
    return sub.groupby("seed")["EENS_kWh"].mean()


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2)
                      / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else float("nan")


def fmt_cell(pol_a_mean, pol_b_mean, p, d):
    gap_pct = (pol_b_mean - pol_a_mean) / max(pol_b_mean, 1e-9) * 100
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    return f"{pol_a_mean:.2f} vs {pol_b_mean:.2f} ({gap_pct:+.1f}%, p={p:.4f}{sig}, d={d:.2f})"


def build_rows(df, scen):
    sub_all = df[df["lead_scenario"] == scen]
    row = {"scenario": scen}
    for pol_a, pol_b, label in COMPARISONS:
        a_all = seed_means(sub_all, pol_a)
        b_all = seed_means(sub_all, pol_b)
        a_hard = seed_means(sub_all, pol_a, HARD_SITES)
        b_hard = seed_means(sub_all, pol_b, HARD_SITES)

        _, p_all = stats.ttest_ind(a_all, b_all, equal_var=False)
        d_all = cohens_d(a_all, b_all)
        _, p_hard = stats.ttest_ind(a_hard, b_hard, equal_var=False)
        d_hard = cohens_d(a_hard, b_hard)

        row[f"{label} | All Sites"] = fmt_cell(a_all.mean(), b_all.mean(), p_all, d_all)
        row[f"{label} | Hard Sites"] = fmt_cell(a_hard.mean(), b_hard.mean(), p_hard, d_hard)
    return row


def main():
    df = pd.read_csv(IN_CSV)
    assert set(HARD_SITES).issubset(set(df["site"].unique())), "Hard site names not found in data"

    rows = [build_rows(df, scen) for scen in SCENARIOS]
    out = pd.DataFrame(rows).set_index("scenario")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print("\n" + "=" * 100)
    print("HYPOTHESIS MATRIX: ALL SITES vs HARD SITES (site2, site5, site7)")
    print("Statistical unit: seed-level means (n=10). Cells: mean_A vs mean_B (gap%, p, d)")
    print("=" * 100)
    for label in ["RLInv vs A6", "RLInv vs TrackB", "A6 vs TrackB"]:
        print(f"\n--- {label} ---")
        print(out[[f"{label} | All Sites", f"{label} | Hard Sites"]].to_string())

    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
