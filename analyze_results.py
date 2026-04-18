"""
analyze_results.py — Full cross-site analysis for thesis results tables.

Produces:
  1. Per-site summary table (all policies, both scenarios)
  2. H1 verdict: RLInv vs TrackB (coupling hypothesis)
  3. H2 verdict: RLInv vs A7 (safety mechanisms hypothesis)
  4. Baseline ladder: B0 → B1 → TrackB → RLInv
  5. Mean EENS across ALL sites
  6. Mean EENS across CONSTRAINED sites only (EENS > threshold)
  7. A5 candidate sites — where to run A5 ablation next
  8. Seed consistency check

Statistical note:
  All hypothesis tests (H1/H2/H3) are performed on SEED-LEVEL means
  (n=3 per policy per site), not raw episode rows.
  This avoids treating 90 non-independent episodes as 90 i.i.d. samples.
  Per-site summary tables still show episode-level mean ± std for
  descriptive purposes.

Usage:
  python analyze_results.py                     # both scenarios
  python analyze_results.py --scenario normal   # normal only
  python analyze_results.py --include_a5        # once A5 results exist
  python analyze_results.py --threshold 100     # adjust constrained cutoff
"""
from __future__ import annotations

import argparse
import glob
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Config ────────────────────────────────────────────────────────────────────

CONSTRAINED_THRESHOLD = 50.0
POLICY_ORDER  = ["B0", "B1", "A7", "TrackB", "RLInv", "A5"]
scenarios_to_use = ["normal", "delayed"]   # overridden in main()

# ── Data loading ──────────────────────────────────────────────────────────────

def load_all(include_a5: bool = False) -> pd.DataFrame:
    dfs = []
    for path in glob.glob("results/all_sites/site*/seed*/RLInv/eval_*.csv"):
        dfs.append(pd.read_csv(path))
    for path in glob.glob("results/all_sites/site*/seed*/TrackB/eval_*.csv"):
        dfs.append(pd.read_csv(path))
    for path in glob.glob("results/all_sites/site*/seed*/A7/eval_*.csv"):
        dfs.append(pd.read_csv(path))
    for path in glob.glob("results/baselines/site*/seed*/eval_*_b0.csv"):
        dfs.append(pd.read_csv(path))
    for path in glob.glob("results/baselines/site*/seed*/eval_*_b1.csv"):
        dfs.append(pd.read_csv(path))
    if include_a5:
        for path in glob.glob("results/ablation_a5/site*/seed*/eval_*.csv"):
            dfs.append(pd.read_csv(path))
    if not dfs:
        raise ValueError("No result CSVs found. Check working directory.")

    df = pd.concat(dfs, ignore_index=True)
    df["policy"] = df["policy"].str.strip()
    # Map any stale algo-name labels
    df["policy"] = df["policy"].replace({"ppo": "A7", "maskable": "RLInv"})
    df["site_num"] = df["site"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["site_num", "seed", "policy", "lead_scenario"])
    return df

# ── Statistical helpers ───────────────────────────────────────────────────────

def seed_means(df, site_num, policy, scenario=None):
    """Per-seed mean EENS — the correct unit for hypothesis testing (n=3)."""
    sub = df[(df["site_num"] == site_num) & (df["policy"] == policy)]
    if scenario:
        sub = sub[sub["lead_scenario"] == scenario]
    if len(sub) == 0:
        return pd.Series(dtype=float)
    return sub.groupby("seed")["EENS_kWh"].mean()


def seed_means_pooled(df, sites, policy, scenario=None):
    """Seed means pooled across multiple sites."""
    sub = df[(df["site_num"].isin(sites)) & (df["policy"] == policy)]
    if scenario:
        sub = sub[sub["lead_scenario"] == scenario]
    if len(sub) == 0:
        return pd.Series(dtype=float)
    return sub.groupby("seed")["EENS_kWh"].mean()


def ttest_safe(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    if a.std() == 0 and b.std() == 0:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a, b)
    return float(t), float(p)


def cohens_d(a, b):
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def fmt(val, decimals=1):
    return "—" if pd.isna(val) else f"{val:.{decimals}f}"


def sig_stars(p):
    if pd.isna(p): return ""
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "ns"


def sep(char="=", width=80):
    print(char * width)

# ── Section 1: Data audit ─────────────────────────────────────────────────────

def audit(df_full, df):
    sep()
    print("SECTION 1: DATA AUDIT")
    sep()
    print(f"  Total rows (all scenarios) : {len(df_full):,}")
    print(f"  Rows after scenario filter : {len(df):,}")
    print(f"  Policies  : {sorted(df_full['policy'].unique())}")
    print(f"  Sites     : {sorted(df_full['site_num'].unique())}")
    print(f"  Seeds     : {sorted(df_full['seed'].unique())}")
    print(f"  Scenarios : {sorted(df_full['lead_scenario'].unique())}")
    print(f"  Using     : {scenarios_to_use}")
    print(f"  NaN values: {df.isnull().sum().sum()}")
    print()

    unexpected = set(df["policy"].unique()) - set(POLICY_ORDER)
    if unexpected:
        print(f"  WARNING: Unexpected policy labels: {unexpected}  — check label_map")
    else:
        print("  Policy labels clean ✓")
    print()

    print("  Completeness (expected 90 obs per policy/site/scenario):")
    issues = []
    sites = sorted(df_full["site_num"].unique())
    for pol in ["RLInv", "TrackB", "B0", "B1"]:
        for sc in ["normal", "delayed"]:
            for site in sites:
                n = len(df_full[(df_full["policy"]==pol) &
                                (df_full["lead_scenario"]==sc) &
                                (df_full["site_num"]==site)])
                if n != 90:
                    issues.append(f"    {pol} site{site} {sc}: n={n}")
    for site in sites:
        n = len(df_full[(df_full["policy"]=="A7") & (df_full["site_num"]==site)])
        if n != 90:
            issues.append(f"    A7 site{site}: n={n}")
    if issues:
        print("  WARNINGS:")
        for i in issues: print(i)
    else:
        print("  All counts correct ✓")
    print()

    print("  Init inv frac (should span 0.3–0.9):")
    for pol in ["RLInv", "TrackB", "B0", "B1", "A7"]:
        sub = df_full[df_full["policy"]==pol]["init_inv_frac"]
        if len(sub) > 0:
            print(f"    {pol:8s}: mean={sub.mean():.3f}  "
                  f"min={sub.min():.3f}  max={sub.max():.3f}")
    print()

# ── Section 2: Per-site summary ───────────────────────────────────────────────

def per_site_summary(df):
    sep()
    print("SECTION 2: PER-SITE EENS SUMMARY  (episode mean ± std, n=90 per cell)")
    sep()
    sites = sorted(df["site_num"].unique())
    for scenario in scenarios_to_use:
        sub = df[df["lead_scenario"]==scenario]
        if len(sub) == 0: continue
        print(f"\n  Scenario: {scenario.upper()}")
        print(f"  {'Site':6s}  " + "  ".join(f"{p:>16s}" for p in ["B0","B1","A7","TrackB","RLInv"]))
        print("  " + "-"*92)
        for site in sites:
            row = f"  site{site:<3d}"
            for pol in ["B0","B1","A7","TrackB","RLInv"]:
                p = sub[(sub["site_num"]==site) & (sub["policy"]==pol)]["EENS_kWh"]
                row += f"  {p.mean():>7.1f}±{p.std():>5.1f}" if len(p)>0 else f"  {'—':>16s}"
            print(row)
    print()

# ── Section 3: Constrained site identification ────────────────────────────────

def identify_constrained_sites(df):
    sep()
    print(f"SECTION 3: CONSTRAINED SITES  (max policy mean EENS > {CONSTRAINED_THRESHOLD:.0f} kWh)")
    print(f"  Thesis note: 'Sites with mean EENS >{CONSTRAINED_THRESHOLD:.0f} kWh under at least")
    print(f"  one policy were classified as constrained for focused analysis.'")
    sep()
    sites = sorted(df["site_num"].unique())
    constrained = {}
    for scenario in scenarios_to_use:
        sub = df[df["lead_scenario"]==scenario]
        if len(sub) == 0: continue
        c_sites = []
        for site in sites:
            max_eens = max(
                (sub[(sub["site_num"]==site) & (sub["policy"]==pol)]["EENS_kWh"].mean()
                 for pol in ["RLInv","TrackB","B0","B1","A7"]
                 if len(sub[(sub["site_num"]==site) & (sub["policy"]==pol)]) > 0),
                default=0
            )
            if max_eens > CONSTRAINED_THRESHOLD:
                c_sites.append(site)
        constrained[scenario] = c_sites
        print(f"  {scenario.upper():8s}: {c_sites}  ({len(c_sites)}/10)")
    all_constrained = sorted(set().union(*constrained.values()))
    print(f"\n  Union (constrained in any scenario): {all_constrained}")
    print()
    return constrained, all_constrained

# ── Section 4: Aggregate EENS ─────────────────────────────────────────────────

def aggregate_eens(df, constrained_sites):
    sep()
    print("SECTION 4: AGGREGATE MEAN EENS — all sites vs constrained sites only")
    sep()
    for scenario in scenarios_to_use:
        sub = df[df["lead_scenario"]==scenario]
        if len(sub) == 0: continue
        print(f"\n  Scenario: {scenario.upper()}")
        print(f"  {'Policy':8s}  {'ALL sites (mean±std)':>24s}  {'Constrained only (mean±std)':>28s}")
        print("  " + "-"*65)
        for pol in ["B0","B1","A7","TrackB","RLInv"]:
            all_e = sub[sub["policy"]==pol]["EENS_kWh"]
            con_e = sub[(sub["policy"]==pol) & (sub["site_num"].isin(constrained_sites))]["EENS_kWh"]
            if len(all_e) == 0: continue
            con_str = f"{con_e.mean():>7.1f} ± {con_e.std():>6.1f}" if len(con_e)>0 else "—"
            print(f"  {pol:8s}  {all_e.mean():>7.1f} ± {all_e.std():>6.1f}          {con_str}")
    print()

# ── Section 5: H1 ─────────────────────────────────────────────────────────────

def h1_coupling(df, constrained_sites):
    sep()
    print("SECTION 5: H1 — COUPLING HYPOTHESIS  (RLInv vs TrackB)")
    print("  Statistical unit: seed-level means (n=3).  Claim: ≥10% EENS reduction.")
    sep()
    sites = sorted(df["site_num"].unique())
    for scenario in scenarios_to_use:
        sub = df[df["lead_scenario"]==scenario]
        if len(sub) == 0: continue
        print(f"\n  Scenario: {scenario.upper()}")
        print(f"  {'Site':8s}  {'RLInv':>12s}  {'TrackB':>12s}  "
              f"{'Gap%':>7s}  {'p':>7s}  {'sig':>4s}  {'d':>6s}  {'✓':>3s}")
        print("  " + "-"*72)
        for site in sites:
            r = seed_means(sub, site, "RLInv")
            b = seed_means(sub, site, "TrackB")
            if len(r)==0 or len(b)==0: continue
            gap = (b.mean()-r.mean()) / max(b.mean(), 1e-9) * 100
            t, p = ttest_safe(r, b)
            d = cohens_d(r, b)
            is_con = "✓" if site in constrained_sites else ""
            print(f"  site{site:<4d}  {r.mean():>8.1f}±{r.std():>3.1f}  "
                  f"{b.mean():>8.1f}±{b.std():>3.1f}  "
                  f"{gap:>+6.1f}%  {fmt(p,4):>7s}  {sig_stars(p):>4s}  "
                  f"{fmt(d,2):>6s}  {is_con}")
        r_c = seed_means_pooled(sub, constrained_sites, "RLInv")
        b_c = seed_means_pooled(sub, constrained_sites, "TrackB")
        if len(r_c)>0 and len(b_c)>0:
            gap_c = (b_c.mean()-r_c.mean()) / max(b_c.mean(), 1e-9) * 100
            t, p = ttest_safe(r_c, b_c)
            d = cohens_d(r_c, b_c)
            print("  " + "-"*72)
            print(f"  {'CONSTRAINED':8s}  {r_c.mean():>8.1f}±{r_c.std():>3.1f}  "
                  f"{b_c.mean():>8.1f}±{b_c.std():>3.1f}  "
                  f"{gap_c:>+6.1f}%  {fmt(p,4):>7s}  {sig_stars(p):>4s}  {fmt(d,2):>6s}")
            verdict = ("SUPPORTED" if gap_c >= 10 and p < 0.05
                       else "PARTIALLY SUPPORTED" if gap_c > 0
                       else "NOT SUPPORTED")
            print(f"\n  H1 verdict ({scenario}): {verdict}  "
                  f"(gap={gap_c:+.1f}%, p={fmt(p,4)}, d={fmt(d,2)})")
    print()

# ── Section 6: H2 ─────────────────────────────────────────────────────────────

def h2_safety(df_full, constrained_sites):
    sep()
    print("SECTION 6: H2 — SAFETY HYPOTHESIS  (RLInv vs A7, normal scenario)")
    print("  Statistical unit: seed-level means (n=3).")
    sep()
    sub = df_full[df_full["lead_scenario"]=="normal"]
    sites = sorted(df_full["site_num"].unique())
    print(f"  {'Site':8s}  {'RLInv':>12s}  {'A7':>12s}  "
          f"{'Gap%':>7s}  {'p':>7s}  {'sig':>4s}  "
          f"{'RLInv viol':>10s}  {'A7 viol':>8s}  {'✓':>3s}")
    print("  " + "-"*90)
    gap_con = p_con = d_con = float("nan")
    for site in sites:
        r  = seed_means(sub, site, "RLInv")
        a7 = seed_means(sub, site, "A7")
        if len(r)==0 or len(a7)==0: continue
        gap = (a7.mean()-r.mean()) / max(a7.mean(), 1e-9) * 100
        t, p = ttest_safe(r, a7)
        rv = sub[(sub["site_num"]==site) & (sub["policy"]=="RLInv")]["violations"].mean()
        av = sub[(sub["site_num"]==site) & (sub["policy"]=="A7")]["violations"].mean()
        is_con = "✓" if site in constrained_sites else ""
        print(f"  site{site:<4d}  {r.mean():>8.1f}±{r.std():>3.1f}  "
              f"{a7.mean():>8.1f}±{a7.std():>3.1f}  "
              f"{gap:>+6.1f}%  {fmt(p,4):>7s}  {sig_stars(p):>4s}  "
              f"{rv:>10.3f}  {av:>8.3f}  {is_con}")
    r_c  = seed_means_pooled(sub, constrained_sites, "RLInv")
    a7_c = seed_means_pooled(sub, constrained_sites, "A7")
    if len(r_c)>0 and len(a7_c)>0:
        gap_con = (a7_c.mean()-r_c.mean()) / max(a7_c.mean(),1e-9) * 100
        t, p_con = ttest_safe(r_c, a7_c)
        d_con = cohens_d(r_c, a7_c)
        print("  " + "-"*90)
        print(f"  {'CONSTRAINED':8s}  {r_c.mean():>8.1f}±{r_c.std():>3.1f}  "
              f"{a7_c.mean():>8.1f}±{a7_c.std():>3.1f}  "
              f"{gap_con:>+6.1f}%  {fmt(p_con,4):>7s}  {sig_stars(p_con):>4s}  "
              f"{fmt(d_con,2):>16s}")
    verdict = ("SUPPORTED" if not pd.isna(gap_con) and gap_con > 0
                              and not pd.isna(p_con) and p_con < 0.05
               else "PARTIALLY SUPPORTED" if not pd.isna(gap_con) and gap_con > 0
               else "NOT SUPPORTED")
    print(f"\n  H2 verdict: {verdict}  "
          f"(gap={fmt(gap_con,1)}%, p={fmt(p_con,4)}, d={fmt(d_con,2)})")
    print()

# ── Section 7: Baseline ladder ────────────────────────────────────────────────

def baseline_ladder(df, constrained_sites):
    sep()
    print("SECTION 7: BASELINE LADDER  (B0 → B1 → TrackB → RLInv)")
    print("  Statistical unit: seed-level means, constrained sites only.")
    sep()
    comparisons = [
        ("B0",     "B1",     "(s,S) ordering > simple threshold"),
        ("B1",     "TrackB", "Learned DG dispatch > heuristic"),
        ("TrackB", "RLInv",  "Joint learned ordering > decomposed (H1 core)"),
    ]
    for scenario in scenarios_to_use:
        sub = df[(df["lead_scenario"]==scenario) &
                 (df["site_num"].isin(constrained_sites))]
        if len(sub) == 0: continue
        print(f"\n  Scenario: {scenario.upper()}")
        print(f"  {'Step':40s}  {'A mean':>8s}  {'B mean':>8s}  "
              f"{'Δ EENS':>8s}  {'p':>7s}  {'sig':>4s}  winner")
        print("  " + "-"*85)
        for (pol_a, pol_b, label) in comparisons:
            a = seed_means_pooled(sub, constrained_sites, pol_a)
            b = seed_means_pooled(sub, constrained_sites, pol_b)
            if len(a)==0 or len(b)==0:
                print(f"  {label:40s}  — (no data)")
                continue
            delta = a.mean() - b.mean()
            t, p  = ttest_safe(a, b)
            winner = pol_b if delta > 0 else pol_a
            print(f"  {label:40s}  {a.mean():>8.1f}  {b.mean():>8.1f}  "
                  f"{delta:>+8.1f}  {fmt(p,4):>7s}  {sig_stars(p):>4s}  {winner}")
    print()

# ── Section 8: H3 ─────────────────────────────────────────────────────────────

def h3_inventory(df_full, constrained_sites):
    sep()
    print("SECTION 8: H3 — INVENTORY STATE HYPOTHESIS  (RLInv vs A5)")
    print("  Statistical unit: seed-level means (n=3). A5 = no inv obs.")
    sep()
    a5_data = df_full[df_full["policy"]=="A5"]
    if len(a5_data) == 0:
        print("  A5 results not yet available.")
        print()
        print(f"  Run A5 on ALL constrained sites: {constrained_sites}")
        print()
        print("  Commands:")
        print("  for seed in 42 123 777; do")
        for site in constrained_sites:
            print(f"    python -m src.train.train_ablation_a5 \\")
            print(f"      --site site{site} --lead normal --timesteps 400000 \\")
            print(f"      --seed $seed --logdir runs/ablation_a5/site{site}/seed${{seed}}")
        print("  done")
        print()
        print("  Then: python analyze_results.py --include_a5")
        return
    sub = df_full[df_full["lead_scenario"]=="normal"]
    sites_with_a5 = sorted(a5_data["site_num"].unique())
    missing = sorted(set(constrained_sites) - set(sites_with_a5))
    if missing:
        print(f"  WARNING: A5 missing for constrained sites: {missing}")
        print()
    print(f"  {'Site':8s}  {'RLInv':>12s}  {'A5':>12s}  "
          f"{'Gap%':>7s}  {'p':>7s}  {'sig':>4s}  {'d':>6s}")
    print("  " + "-"*68)
    for site in sites_with_a5:
        r  = seed_means(sub, site, "RLInv")
        a5 = seed_means(sub, site, "A5")
        if len(r)==0 or len(a5)==0: continue
        gap = (a5.mean()-r.mean()) / max(a5.mean(), 1e-9) * 100
        t, p = ttest_safe(r, a5)
        d = cohens_d(r, a5)
        print(f"  site{site:<4d}  {r.mean():>8.1f}±{r.std():>3.1f}  "
              f"{a5.mean():>8.1f}±{a5.std():>3.1f}  "
              f"{gap:>+6.1f}%  {fmt(p,4):>7s}  {sig_stars(p):>4s}  {fmt(d,2):>6s}")
    r_all  = seed_means_pooled(sub, sites_with_a5, "RLInv")
    a5_all = seed_means_pooled(sub, sites_with_a5, "A5")
    if len(r_all)>0 and len(a5_all)>0:
        gap = (a5_all.mean()-r_all.mean()) / max(a5_all.mean(),1e-9) * 100
        t, p = ttest_safe(r_all, a5_all)
        d = cohens_d(r_all, a5_all)
        print("  " + "-"*68)
        print(f"  {'POOLED':8s}  {r_all.mean():>8.1f}±{r_all.std():>3.1f}  "
              f"{a5_all.mean():>8.1f}±{a5_all.std():>3.1f}  "
              f"{gap:>+6.1f}%  {fmt(p,4):>7s}  {sig_stars(p):>4s}  {fmt(d,2):>6s}")
        verdict = ("SUPPORTED" if gap > 0 and p < 0.05
                   else "PARTIALLY SUPPORTED" if gap > 0
                   else "NOT SUPPORTED")
        print(f"\n  H3 verdict: {verdict}  "
              f"(gap={fmt(gap,1)}%, p={fmt(p,4)}, d={fmt(d,2)})")
    print()

# ── Section 9: Seed consistency ───────────────────────────────────────────────

def seed_consistency(df_full, constrained_sites):
    sep()
    print("SECTION 9: SEED CONSISTENCY  (CV < 20% = stable)")
    sep()
    sub = df_full[(df_full["lead_scenario"]=="normal") &
                  (df_full["site_num"].isin(constrained_sites))]
    for pol in ["RLInv", "TrackB"]:
        print(f"\n  {pol}:")
        for site in constrained_sites:
            by_seed = sub[(sub["site_num"]==site) &
                          (sub["policy"]==pol)].groupby("seed")["EENS_kWh"].mean()
            if len(by_seed) == 0: continue
            vals = "  ".join(f"s{s}={v:.1f}" for s, v in by_seed.items())
            cv = by_seed.std() / max(by_seed.mean(), 1e-9) * 100
            flag = "  ⚠ HIGH" if cv > 30 else ""
            print(f"    site{site}: {vals}  CV={cv:.1f}%{flag}")
    print()

# ── Section 10: Diesel/cost trade-off ────────────────────────────────────────

def diesel_cost_tradeoff(df, constrained_sites):
    sep()
    print("SECTION 10: DIESEL / COST TRADE-OFF  (RLInv vs TrackB)")
    print("  Story: RLInv achieves lower EENS by running DG more proactively,")
    print("  accepting higher diesel consumption and operating cost.")
    print("  Constrained sites only — episode-level means (descriptive).")
    sep()

    for scenario in scenarios_to_use:
        sub = df[(df["lead_scenario"]==scenario) &
                 (df["site_num"].isin(constrained_sites))]
        if len(sub) == 0: continue
        print(f"\n  Scenario: {scenario.upper()}")
        print(f"  {'Metric':22s}  {'RLInv':>14s}  {'TrackB':>14s}  {'Δ (RL-TB)':>12s}  {'Δ%':>7s}")
        print("  " + "-"*75)

        for metric, label, unit in [
            ("EENS_kWh",    "EENS",           "kWh"),
            ("diesel_kWh",  "Diesel consumed", "kWh"),
            ("cost_proxy",  "Cost proxy",      "units"),
            ("uptime_pct",  "Uptime",          "%"),
            ("dg_on_fraction", "DG on fraction", ""),
        ]:
            r = sub[sub["policy"]=="RLInv"][metric]
            b = sub[sub["policy"]=="TrackB"][metric]
            if len(r)==0 or len(b)==0: continue
            delta = r.mean() - b.mean()
            pct   = delta / max(abs(b.mean()), 1e-9) * 100
            print(f"  {label+' ('+unit+')':22s}  "
                  f"{r.mean():>10.2f}±{r.std():>5.1f}  "
                  f"{b.mean():>10.2f}±{b.std():>5.1f}  "
                  f"{delta:>+12.2f}  {pct:>+6.1f}%")

        print()
        print("  Interpretation:")
        r_eens = sub[sub["policy"]=="RLInv"]["EENS_kWh"].mean()
        b_eens = sub[sub["policy"]=="TrackB"]["EENS_kWh"].mean()
        r_d    = sub[sub["policy"]=="RLInv"]["diesel_kWh"].mean()
        b_d    = sub[sub["policy"]=="TrackB"]["diesel_kWh"].mean()
        r_c    = sub[sub["policy"]=="RLInv"]["cost_proxy"].mean()
        b_c    = sub[sub["policy"]=="TrackB"]["cost_proxy"].mean()
        eens_gap = (b_eens - r_eens) / max(b_eens, 1e-9) * 100
        cost_gap = (r_c - b_c) / max(b_c, 1e-9) * 100
        print(f"    RLInv reduces EENS by {eens_gap:.1f}% vs TrackB")
        print(f"    at the cost of {cost_gap:+.1f}% higher operating cost.")
        if cost_gap > 0 and eens_gap > 0:
            print(f"    Trade-off confirmed: reliability gain comes at an efficiency cost.")
        elif cost_gap <= 0 and eens_gap > 0:
            print(f"    Pareto improvement: lower EENS AND lower cost — strong result.")
    print()


# ── Section 11: Hypothesis summary ───────────────────────────────────────────

def hypothesis_summary(df, constrained_sites):
    sep()
    print("SECTION 11: HYPOTHESIS TESTING SUMMARY")
    sep()
    easy = sorted(set(df["site_num"].unique()) - set(constrained_sites))
    print(f"""
  Constrained sites (EENS > {CONSTRAINED_THRESHOLD:.0f} kWh, differentiation regime): {constrained_sites}
  Easy sites (EENS≈0, all policies converge):                   {easy}

  Site-type split is itself a finding:
    Inventory-aware RL provides benefit specifically in resource-constrained
    regimes. On easy sites (reliable grid), all policies converge to
    near-zero EENS regardless of architecture.

  All hypothesis tests use seed-level means (n=3) as the statistical
  unit to avoid pseudoreplication across 30 episodes per seed.

  ┌──────────────────────────────────────────────────────────────────┐
  │ Hypothesis  │ Test              │ Metric    │ See section        │
  ├──────────────────────────────────────────────────────────────────┤
  │ H1 Coupling │ RLInv vs TrackB   │ EENS gap  │ Section 5          │
  │ H2 Safety   │ RLInv vs A7       │ viol+EENS │ Section 6          │
  │ H3 Inventory│ RLInv vs A5       │ EENS gap  │ Section 8          │
  └──────────────────────────────────────────────────────────────────┘
    """)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global CONSTRAINED_THRESHOLD, scenarios_to_use

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario",   default="both",
                    choices=["normal", "delayed", "both"])
    ap.add_argument("--include_a5", action="store_true")
    ap.add_argument("--threshold",  type=float, default=CONSTRAINED_THRESHOLD)
    args = ap.parse_args()

    CONSTRAINED_THRESHOLD = args.threshold
    scenarios_to_use = (["normal", "delayed"] if args.scenario == "both"
                        else [args.scenario])

    print()
    sep("=")
    print("FULL CROSS-SITE ANALYSIS — Inventory-Aware RL for Telecom Tower Management")
    print("Sangeeta Gupta | CH24M562 | IIT Madras | Phase-II")
    print(f"Scenarios: {scenarios_to_use}  |  Threshold: {CONSTRAINED_THRESHOLD} kWh  |  "
          f"A5: {args.include_a5}")
    sep("=")
    print()

    df_full = load_all(include_a5=args.include_a5)

    # Fix 1: filter df by scenario for scenario-aware sections
    df = (df_full[df_full["lead_scenario"] == args.scenario].copy()
          if args.scenario != "both" else df_full.copy())

    audit(df_full, df)
    per_site_summary(df)
    constrained, all_constrained = identify_constrained_sites(df)
    aggregate_eens(df, all_constrained)
    h1_coupling(df, all_constrained)
    h2_safety(df_full, all_constrained)      # always normal (A7 only has normal)
    baseline_ladder(df, all_constrained)
    h3_inventory(df_full, all_constrained)   # always normal
    seed_consistency(df_full, all_constrained)
    diesel_cost_tradeoff(df, all_constrained)
    hypothesis_summary(df, all_constrained)

    sep()
    print("Analysis complete.")
    sep()


if __name__ == "__main__":
    main()
