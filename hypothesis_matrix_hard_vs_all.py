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
understates standard error.

DESIGN: PAIRED, not independent. Every seed is evaluated under both policies
being compared (seed 42 under RLInv corresponds to seed 42 under A6, etc.),
so the correct test is a paired t-test on the within-seed differences, not
an independent-samples (Welch) test. Effect size is Cohen's d_z (mean of the
paired differences / std of the paired differences), NOT the two-sample
pooled-variance d.

  diff       = a - b   (paired, per seed)
  t, p       = scipy.stats.ttest_rel(a, b)
  d_z        = diff.mean() / diff.std(ddof=1)

Sign convention: d_z < 0 means policy A (first-listed) has lower EENS
(better); d_z > 0 means policy B has lower EENS (better). The sign is
preserved and reported explicitly via a "favours" column rather than
silently taking an absolute value.

Degenerate case: when two policies produce numerically identical seed-level
EENS (e.g. A6 vs TrackB under Normal/Delayed, where masking never binds),
the paired difference has zero variance and t/p/d_z are mathematically
undefined (0/0). These are reported as "IDENTICAL", not as p=1.0 or d=0,
which would misrepresent an undefined statistic as a conventional
null result.

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
    ("rlinv", "a6",     "RLInv vs A6"),          # H1 (ordering isolated)
    ("rlinv", "trackb", "RLInv vs TrackB"),      # H1-legacy (confounded)
    ("a6",    "trackb", "A6 vs TrackB"),         # H2 (masking isolated)
    ("rlinv", "a7",     "RLInv vs A7"),          # H2-secondary (masking, joint ordering held constant)
    ("rlinv", "a5",     "RLInv vs A5"),          # H3 (inventory observation)
]


def seed_means(df, policy, site_filter=None):
    sub = df[df["policy"] == policy]
    if site_filter is not None:
        sub = sub[sub["site"].isin(site_filter)]
    return sub.groupby("seed")["EENS_kWh"].mean().sort_index()


def paired_stats(a, b):
    """
    a, b: pandas Series of seed-level means, indexed by seed.
    Returns (mean_a, mean_b, p, d_z, note) using a PAIRED test.
    note is "" normally, or "IDENTICAL" if the paired difference has zero
    variance (t-test undefined).
    """
    a, b = a.align(b, join="inner")
    assert len(a) > 1, "Need >=2 paired seeds for a paired test"
    diff = a - b
    sd = diff.std(ddof=1)
    if sd == 0:
        return a.mean(), b.mean(), np.nan, np.nan, "IDENTICAL"
    _, p = stats.ttest_rel(a, b)
    d_z = diff.mean() / sd
    return a.mean(), b.mean(), p, d_z, ""


def fmt_cell(mean_a, mean_b, p, d_z, note, label_a, label_b):
    gap_pct = (mean_b - mean_a) / max(abs(mean_b), 1e-9) * 100
    if note == "IDENTICAL":
        return f"{mean_a:.2f} vs {mean_b:.2f} (identical, paired test undefined)"
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    favours = label_a if d_z < 0 else label_b
    return (f"{mean_a:.2f} vs {mean_b:.2f} ({gap_pct:+.1f}%, "
            f"p={p:.4f}{sig}, d_z={d_z:+.2f}, favours {favours})")


def build_rows(df, scen):
    sub_all = df[df["lead_scenario"] == scen]
    row = {"scenario": scen}
    for pol_a, pol_b, label in COMPARISONS:
        label_a, label_b = label.split(" vs ")
        a_all = seed_means(sub_all, pol_a)
        b_all = seed_means(sub_all, pol_b)
        a_hard = seed_means(sub_all, pol_a, HARD_SITES)
        b_hard = seed_means(sub_all, pol_b, HARD_SITES)

        ma, mb, p, d_z, note = paired_stats(a_all, b_all)
        mah, mbh, ph, d_zh, noteh = paired_stats(a_hard, b_hard)

        row[f"{label} | All Sites"] = fmt_cell(ma, mb, p, d_z, note, label_a, label_b)
        row[f"{label} | Hard Sites"] = fmt_cell(mah, mbh, ph, d_zh, noteh, label_a, label_b)
    return row


def main():
    df = pd.read_csv(IN_CSV)
    assert set(HARD_SITES).issubset(set(df["site"].unique())), "Hard site names not found in data"

    rows = [build_rows(df, scen) for scen in SCENARIOS]
    out = pd.DataFrame(rows).set_index("scenario")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 80)
    print("\n" + "=" * 100)
    print("HYPOTHESIS MATRIX: ALL SITES vs HARD SITES (site2, site5, site7)")
    print("PAIRED test (ttest_rel) on seed-level means (n=10 seeds, matched).")
    print("Effect size: Cohen's d_z (paired). Sign preserved; 'favours' column states direction.")
    print("=" * 100)
    for label in ["RLInv vs A6", "RLInv vs TrackB", "A6 vs TrackB", "RLInv vs A7", "RLInv vs A5"]:
        print(f"\n--- {label} ---")
        print(out[[f"{label} | All Sites", f"{label} | Hard Sites"]].to_string())

    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
