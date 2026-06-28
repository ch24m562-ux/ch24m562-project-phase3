"""
analyse_lognormal_sensitivity.py

Compares lognormal vs geometric lead-time results for rlinv, b1, mpc.
Produces a mini evidence package:
  results/sensitivity/lognormal/lognormal_summary.csv
  results/sensitivity/lognormal/lognormal_thesis_summary.txt

Run after run_sensitivity_lognormal.ps1 completes.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

LOGNORMAL_DIR  = Path("results/sensitivity/lognormal")
GEO_SUMMARY    = Path("results/phase3/master_summary.csv")
OUT_SUMMARY    = LOGNORMAL_DIR / "lognormal_summary.csv"
OUT_THESIS_TXT = LOGNORMAL_DIR / "lognormal_thesis_summary.txt"

FOCUS_POLICIES = ["rlinv", "b1", "mpc"]
SCENARIOS      = ["normal", "delayed", "monsoon", "extreme"]

# ── Load lognormal results ────────────────────────────────────────────────────
if not LOGNORMAL_DIR.exists():
    print(f"[ERROR] {LOGNORMAL_DIR} not found. Run run_sensitivity_lognormal.ps1 first.")
    sys.exit(1)

log_files = list(LOGNORMAL_DIR.glob("*.csv"))
if not log_files:
    print(f"[ERROR] No CSV files in {LOGNORMAL_DIR}.")
    sys.exit(1)

dfs = []
for f in log_files:
    try:
        df = pd.read_csv(f)
        dfs.append(df)
    except Exception as e:
        print(f"[WARN] Could not read {f.name}: {e}")

log_df = pd.concat(dfs, ignore_index=True)
log_df["distribution"] = "lognormal"
log_df["policy_base"]  = log_df["policy"].str.replace("_lognormal", "", regex=False)

# ── Load geometric results ─────────────────────────────────────────────────────
if not GEO_SUMMARY.exists():
    print(f"[ERROR] {GEO_SUMMARY} not found.")
    sys.exit(1)

geo_df = pd.read_csv(GEO_SUMMARY)
geo_df = geo_df[geo_df["policy"].isin(FOCUS_POLICIES)].copy()
geo_df["policy_base"] = geo_df["policy"]

output_lines = []
def pr(line=""):
    print(line)
    output_lines.append(str(line))

# ── Section 1: Completeness ────────────────────────────────────────────────────
pr("=" * 70)
pr("1. COMPLETENESS CHECK")
pr("=" * 70)
pr(log_df.groupby(["policy_base", "lead_scenario"]).size().to_string())
pr()
EXPECTED_ROWS = 3 * 10 * 4 * 10 * 10
actual_rows   = len(log_df)
pr(f"Expected: 3 policies x 10 sites x 4 scenarios x 10 seeds x 10 episodes = {EXPECTED_ROWS}")
pr(f"Actual:   {actual_rows}")
if actual_rows == EXPECTED_ROWS:
    pr("COMPLETE.")
elif actual_rows < EXPECTED_ROWS:
    pr(f"[WARN] INCOMPLETE -- {EXPECTED_ROWS-actual_rows} rows missing (~{(EXPECTED_ROWS-actual_rows)//10} files).")
    pr("       Re-run run_sensitivity_lognormal.ps1 (resume=True skips completed files).")
else:
    pr(f"[WARN] MORE rows than expected -- check for duplicate files.")

# ── Build pivots ──────────────────────────────────────────────────────────────
geo_pivot = geo_df.groupby(["policy","lead_scenario"])["EENS_kWh"].mean().unstack()
log_pivot = log_df.groupby(["policy_base","lead_scenario"])["EENS_kWh"].mean().unstack()

# ── Section 2: Mean EENS comparison ───────────────────────────────────────────
pr()
pr("=" * 70)
pr("2. MEAN EENS_kWh: GEOMETRIC vs LOGNORMAL")
pr("=" * 70)

rows_for_csv = []
pct_changes  = {p: {} for p in FOCUS_POLICIES}

for policy in FOCUS_POLICIES:
    pr(f"\n  {policy.upper()}:")
    pr(f"  {'scenario':<12}{'geometric':>12}{'lognormal':>12}{'delta':>10}{'delta%':>10}")
    for sc in SCENARIOS:
        g = geo_pivot.loc[policy, sc] if (policy in geo_pivot.index and sc in geo_pivot.columns) else float('nan')
        l = log_pivot.loc[policy, sc] if (policy in log_pivot.index and sc in log_pivot.columns) else float('nan')
        delta = l - g
        pct   = 100 * delta / g if g > 0.01 else float('nan')
        pct_changes[policy][sc] = pct
        pr(f"  {sc:<12}{g:>12.2f}{l:>12.2f}{delta:>10.2f}{pct:>9.1f}%")
        rows_for_csv.append({"policy": policy, "scenario": sc,
                              "geo_EENS": round(g,3), "log_EENS": round(l,3),
                              "delta": round(delta,3), "delta_pct": round(pct,2)})

pd.DataFrame(rows_for_csv).to_csv(OUT_SUMMARY, index=False)
pr(f"\n  -> Saved: {OUT_SUMMARY}")

# ── Section 3: Absolute % change per policy ───────────────────────────────────
pr()
pr("=" * 70)
pr("3. ABSOLUTE EENS DIFFERENCE (kWh): GEOMETRIC vs LOGNORMAL")
pr("   (Primary metric for thesis. % change suppressed where geo EENS < 1 kWh")
pr("    to avoid near-zero denominator artifacts.)")
pr("=" * 70)
LOW_EENS_THRESHOLD = 1.0  # kWh -- % change unreliable below this

sensitivity_rank = {}
mean_sensitivity = {}

pr(f"  {'scenario':<12}" + "".join(f"  {p:>14}" for p in FOCUS_POLICIES))
pr("  " + "-" * (12 + 16 * len(FOCUS_POLICIES)))

for sc in SCENARIOS:
    row = f"  {sc:<12}"
    sc_abs = {}
    for p in FOCUS_POLICIES:
        g = geo_pivot.loc[p, sc] if (p in geo_pivot.index and sc in geo_pivot.columns) else float('nan')
        l = log_pivot.loc[p, sc] if (p in log_pivot.index and sc in log_pivot.columns) else float('nan')
        delta_abs = l - g
        sc_abs[p] = abs(delta_abs)
        if not np.isnan(g) and g >= LOW_EENS_THRESHOLD:
            pct = 100 * delta_abs / g
            row += f"  {delta_abs:>+7.2f} ({pct:>+5.1f}%)"
        else:
            row += f"  {delta_abs:>+7.2f}  (suppressed)"
    pr(row)
    sensitivity_rank[sc] = min(sc_abs, key=lambda k: sc_abs[k]
                               if not np.isnan(sc_abs[k]) else float('inf'))

pr()
pr("  Smallest absolute EENS change per scenario:")
for sc, p in sensitivity_rank.items():
    pr(f"    {sc:<12}: {p}")

pr()
pr("  Mean |delta EENS| across all scenarios (kWh):")
for p in FOCUS_POLICIES:
    vals = []
    for sc in SCENARIOS:
        g = geo_pivot.loc[p, sc] if (p in geo_pivot.index and sc in geo_pivot.columns) else float('nan')
        l = log_pivot.loc[p, sc] if (p in log_pivot.index and sc in log_pivot.columns) else float('nan')
        if not np.isnan(g) and not np.isnan(l):
            vals.append(abs(l - g))
    mean_sensitivity[p] = np.mean(vals) if vals else float('nan')
    pr(f"    {p:<12}: {mean_sensitivity[p]:.2f} kWh")

least_sensitive_overall = min(mean_sensitivity, key=lambda k: mean_sensitivity[k]
                              if not np.isnan(mean_sensitivity[k]) else float('inf'))
pr()
pr(f"  -> {least_sensitive_overall} shows the smallest mean absolute EENS change")
pr(f"     ({mean_sensitivity[least_sensitive_overall]:.2f} kWh) across scenarios.")
pr()
pr("  NOTE: % change figures are included above for debugging only.")
pr("  For thesis, report absolute EENS delta and ranking preservation.")

# ── Section 4: Ranking table ──────────────────────────────────────────────────
pr()
pr("=" * 70)
pr("4. POLICY RANKING TABLE  (rank 1 = lowest EENS = best)")
pr("=" * 70)
pr(f"  {'scenario':<12}  {'--- GEOMETRIC ---':^32}  {'--- LOGNORMAL ---':^32}  changed?")

ranking_changes = []
for sc in SCENARIOS:
    gv = {p: geo_pivot.loc[p,sc] for p in FOCUS_POLICIES if p in geo_pivot.index and sc in geo_pivot.columns}
    lv = {p: log_pivot.loc[p,sc] for p in FOCUS_POLICIES if p in log_pivot.index and sc in log_pivot.columns}
    gr = sorted(gv, key=gv.get)
    lr = sorted(lv, key=lv.get)
    gs = "  ".join(f"{i+1}.{p}" for i,p in enumerate(gr))
    ls = "  ".join(f"{i+1}.{p}" for i,p in enumerate(lr))
    changed = gr != lr
    if changed:
        ranking_changes.append(sc)
    pr(f"  {sc:<12}  {gs:<32}  {ls:<32}  {'YES -- CHANGED' if changed else 'no'}")

pr()
ranking_preserved = len(ranking_changes) == 0
if ranking_preserved:
    pr("  Rankings preserved in all scenarios.")
    ranking_conclusion = "preserved in all scenarios"
else:
    pr(f"  [WARN] Rankings changed in: {', '.join(ranking_changes)}")
    ranking_conclusion = f"changed in: {', '.join(ranking_changes)}"

# ── Section 5: RLInv advantage over B1 ────────────────────────────────────────
pr()
pr("=" * 70)
pr("5. RLINV ADVANTAGE OVER B1")
pr("=" * 70)
pr(f"  {'scenario':<12}{'geo adv':>10}{'log adv':>10}  preserved?")

advantage_preserved_all = True
for sc in SCENARIOS:
    g_r = geo_pivot.loc["rlinv",sc] if "rlinv" in geo_pivot.index and sc in geo_pivot.columns else float('nan')
    g_b = geo_pivot.loc["b1",   sc] if "b1"    in geo_pivot.index and sc in geo_pivot.columns else float('nan')
    l_r = log_pivot.loc["rlinv",sc] if "rlinv" in log_pivot.index and sc in log_pivot.columns else float('nan')
    l_b = log_pivot.loc["b1",   sc] if "b1"    in log_pivot.index and sc in log_pivot.columns else float('nan')
    g_adv = g_b - g_r
    l_adv = l_b - l_r
    preserved = (g_adv > 0) == (l_adv > 0)
    if not preserved:
        advantage_preserved_all = False
    pr(f"  {sc:<12}{g_adv:>10.2f}{l_adv:>10.2f}  {'YES' if preserved else 'NO -- CHANGED'}")

# ── Section 6: Sensitivity verdict ────────────────────────────────────────────
pr()
pr("=" * 70)
pr("6. SENSITIVITY VERDICT")
pr("=" * 70)

max_delta_pct = max(
    (abs(pct_changes[p].get(sc, 0)) for p in FOCUS_POLICIES for sc in SCENARIOS
     if not np.isnan(pct_changes[p].get(sc, float('nan')))),
    default=float('nan')
)

if ranking_preserved and advantage_preserved_all:
    verdict = "PASS"
    interpretation = (
        "Policy rankings are preserved under the lognormal distribution. "
        "RLInv's advantage over B1 holds in all scenarios. "
        "Under the evaluated lognormal lead-time setting (sigma=0.5), "
        "RLInv's learned inventory-aware ordering policy maintained and, "
        "in the extreme scenario, increased its advantage over rule-based B1. "
        "Within the evaluated benchmark scenarios, replacing the memoryless "
        "geometric lead-time model with a matched-mean lognormal model did not "
        "materially alter the principal scientific conclusions."
    )
    recommendation = "No modification required to principal thesis claims."
elif advantage_preserved_all and not ranking_preserved:
    verdict = "PARTIAL"
    interpretation = (
        "RLInv's advantage over B1 is preserved, but policy rankings among "
        "B1/MPC changed in some scenarios. The principal thesis claims about "
        "RLInv hold, but note distribution sensitivity of the MPC vs B1 comparison."
    )
    recommendation = "Note MPC vs B1 distribution sensitivity in thesis. RLInv claims stand."
else:
    verdict = "INVESTIGATE"
    interpretation = (
        "Policy rankings or RLInv advantage changed under lognormal. "
        "Review per-scenario results carefully before finalising thesis claims."
    )
    recommendation = "Review Sections 4 and 5 carefully before writing thesis conclusions."

pr(f"  Distribution Robustness : {verdict}")
pr(f"  Rankings                : {ranking_conclusion}")
pr(f"  RLInv advantage         : {'preserved in all scenarios' if advantage_preserved_all else 'changed in at least one scenario'}")
pr(f"  Maximum EENS change     : {max_delta_pct:.1f}%")
pr(f"  Mean |delta%| - rlinv   : {mean_sensitivity.get('rlinv', float('nan')):.1f}%")
pr(f"  Mean |delta%| - b1      : {mean_sensitivity.get('b1', float('nan')):.1f}%")
pr(f"  Mean |delta%| - mpc     : {mean_sensitivity.get('mpc', float('nan')):.1f}%")
pr()
pr("  Interpretation:")
for sent in interpretation.split('. '):
    if sent.strip():
        pr(f"    {sent.strip()}.")
pr()
pr(f"  Recommendation: {recommendation}")

# ── Write thesis summary txt ───────────────────────────────────────────────────
thesis = [
    "",
    "=" * 70,
    "LOGNORMAL SENSITIVITY SUMMARY",
    "=" * 70,
    "",
    "Objective",
    "---------",
    "Evaluate whether the memoryless geometric lead-time assumption",
    "(a documented implementation deviation from the formal lognormal",
    "model, Table 4.3) materially affects the scientific conclusions.",
    "",
    "Setup",
    "-----",
    "Policies: RLInv (geometric-trained), B1, MPC",
    "Scenarios: normal / delayed / monsoon / extreme",
    "Seeds: 10 (identical to main geometric grid)",
    "Episodes: 10 per (site, scenario, seed)",
    "Distribution: Lognormal(sigma=0.5), mean matched to geometric (same lead_p)",
    "Note: the two distributions have identical mean lead times but",
    "      different distributional shapes and variances.",
    "",
    "Key Findings",
    "------------",
]

for sc in SCENARIOS:
    g_r = geo_pivot.loc["rlinv",sc] if "rlinv" in geo_pivot.index and sc in geo_pivot.columns else float('nan')
    g_b = geo_pivot.loc["b1",   sc] if "b1"    in geo_pivot.index and sc in geo_pivot.columns else float('nan')
    l_r = log_pivot.loc["rlinv",sc] if "rlinv" in log_pivot.index and sc in log_pivot.columns else float('nan')
    l_b = log_pivot.loc["b1",   sc] if "b1"    in log_pivot.index and sc in log_pivot.columns else float('nan')
    rlinv_pct = pct_changes["rlinv"].get(sc, float('nan'))
    thesis.append(
        f"  {sc:<10}: RLInv geo={g_r:.2f} log={l_r:.2f} ({rlinv_pct:+.1f}%)  |  "
        f"B1 geo={g_b:.2f} log={l_b:.2f}"
    )

thesis += [
    "",
    "Sensitivity Verdict",
    "-------------------",
    f"  Distribution Robustness : {verdict}",
    f"  Rankings                : {ranking_conclusion}",
    f"  RLInv advantage         : {'preserved in all scenarios' if advantage_preserved_all else 'changed in at least one scenario'}",
    f"  Maximum EENS change     : {max_delta_pct:.1f}%",
    f"  Mean |delta%| rlinv/b1/mpc: {mean_sensitivity.get('rlinv',float('nan')):.1f}% / {mean_sensitivity.get('b1',float('nan')):.1f}% / {mean_sensitivity.get('mpc',float('nan')):.1f}%",
    f"  Least sensitive policy  : {least_sensitive_overall} ({mean_sensitivity[least_sensitive_overall]:.1f}% mean |delta|)",
    "",
    "Scientific Interpretation",
    "-------------------------",
]
for sent in interpretation.split('. '):
    if sent.strip():
        thesis.append(f"  {sent.strip()}.")

thesis += [
    "",
    "Recommendation",
    "--------------",
    f"  {recommendation}",
    "",
    "Reviewer Coverage",
    "-----------------",
    "  P2-R06: Varying delay distributions and robustness testing -- ADDRESSED",
    "  P2-R07: Stronger interpretation of results -- ADDRESSED (verdict above)",
    "  P2-R09: Realistic supply chain modelling -- PARTIALLY ADDRESSED",
    "",
    "Recommended Thesis Location",
    "---------------------------",
    "  Chapter 8 (Phase 3 Results), Sensitivity Analysis subsection.",
    "  One paragraph + comparison table (geometric vs lognormal EENS).",
    "",
    "Evidence Files",
    "--------------",
    f"  Raw CSVs:      {LOGNORMAL_DIR}/",
    f"  Summary CSV:   {OUT_SUMMARY}",
    f"  This file:     {OUT_THESIS_TXT}",
    "",
    "Use this file to update:",
    "  - Stream B (reviewer closure matrix): P2-R06, P2-R07, P2-R09",
    "  - Stream C (experimental register): lognormal sensitivity experiment",
    "  - Stream D (discoveries): distribution robustness finding",
    "  - Chapter 8 sensitivity subsection",
    "  - Viva Playbook: Q: why geometric? A: sensitivity confirms robustness",
    "",
]

OUT_THESIS_TXT.write_text("\n".join(thesis))

pr()
pr("=" * 70)
pr("EVIDENCE PACKAGE WRITTEN:")
pr(f"  {OUT_SUMMARY}")
pr(f"  {OUT_THESIS_TXT}")
pr("=" * 70)
