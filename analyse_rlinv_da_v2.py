"""
analyse_rlinv_da.py
Complete analysis of RLInv-DA exploratory extension (Comments #9/#10).
Produces 3 summary CSVs, one comparison figure, and automatic conclusions.
"""
import pandas as pd
import numpy as np
import glob, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

IN_DIR  = Path("results/rlinv_da")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["normal","delayed","monsoon","extreme"]

# ── Load ──────────────────────────────────────────────────────────────────────
files = (glob.glob(str(IN_DIR/"base_regime"/"*.csv")) +
         glob.glob(str(IN_DIR/"ewma_regime"/"*.csv")))
print(f"Files found: {len(files)}  (expected 36 = 2 variants × 3 sites × 2 scenarios × 3 seeds)")
dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        df["variant"] = "base_regime" if "base_regime" in f else "ewma_regime"
        dfs.append(df)
    except Exception as e:
        print(f"  Error: {f}: {e}")
df = pd.concat(dfs, ignore_index=True)
print(f"Total rows: {len(df)}  (expected 360 = 36 files × 10 episodes)")

# ── Table 1: Overall summary ──────────────────────────────────────────────────
print("\n"+"="*80)
print("TABLE 1: Overall EENS by Variant and Scenario")
print("="*80)
summary_rows = []
for sc in SCENARIOS:
    for var in ["base_regime","ewma_regime"]:
        sub = df[(df.variant==var)&(df.lead_scenario==sc)]
        if sub.empty: continue
        m   = sub["EENS_kWh"].mean()
        sem = sub["EENS_kWh"].sem()
        summary_rows.append({
            "variant": var,
            "scenario": sc,
            "EENS_mean":      round(m,3),
            "EENS_sem":       round(sem,3),
            "EENS_ci95_lo":   round(m-1.96*sem,3),
            "EENS_ci95_hi":   round(m+1.96*sem,3),
            "diesel_mean":    round(sub["diesel_kWh"].mean(),1),
            "cost_mean":      round(sub["cost_proxy"].mean(),1),
            "orders_mean":    round(sub["orders_placed"].mean(),2),
            "stockouts_mean": round(sub["stockout_events"].mean(),3),
        })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(IN_DIR/"rlinv_da_summary.csv", index=False)
print(summary_df[["variant","scenario","EENS_mean","EENS_ci95_lo","EENS_ci95_hi",
                   "diesel_mean","orders_mean"]].to_string(index=False))

# ── Table 2: Per-site breakdown ───────────────────────────────────────────────
print("\n"+"="*80)
print("TABLE 2: Per-Site Breakdown")
print("="*80)
site_agg = df.groupby(["variant","site","lead_scenario"])["EENS_kWh"].mean().round(3).reset_index()
site_pivot = site_agg.pivot_table(
    index=["site","lead_scenario"], columns="variant", values="EENS_kWh").reset_index()
if "ewma_regime" in site_pivot.columns and "base_regime" in site_pivot.columns:
    site_pivot["diff"]      = (site_pivot["ewma_regime"] - site_pivot["base_regime"]).round(3)
    site_pivot["direction"] = site_pivot["diff"].apply(
        lambda x: "EWMA better" if x<-0.1 else ("Base better" if x>0.1 else "Similar"))
print(site_pivot.to_string(index=False))
site_pivot.to_csv(IN_DIR/"rlinv_da_site_breakdown.csv", index=False)

# ── Table 3: Per-site×seed×scenario paired comparison (18 pairs) ─────────────
print("\n"+"="*80)
print("TABLE 3: Paired Comparison (site × seed × scenario = 18 pairs)")
print("="*80)
seed_agg = df.groupby(["variant","site","seed","lead_scenario"])["EENS_kWh"].mean().round(3).reset_index()
seed_pivot = seed_agg.pivot_table(
    index=["site","seed","lead_scenario"],
    columns="variant",
    values="EENS_kWh").reset_index()
if "ewma_regime" in seed_pivot.columns and "base_regime" in seed_pivot.columns:
    seed_pivot["diff"]      = (seed_pivot["ewma_regime"] - seed_pivot["base_regime"]).round(3)
    seed_pivot["direction"] = seed_pivot["diff"].apply(
        lambda x: "EWMA better" if x<-0.1 else ("Base better" if x>0.1 else "Similar"))
print(seed_pivot.to_string(index=False))
seed_pivot.to_csv(IN_DIR/"rlinv_da_seed_breakdown.csv", index=False)

n_total       = len(seed_pivot)
n_ewma_better = int((seed_pivot["diff"] < -0.1).sum()) if "diff" in seed_pivot.columns else 0
n_base_better = int((seed_pivot["diff"] >  0.1).sum()) if "diff" in seed_pivot.columns else 0
n_similar     = n_total - n_ewma_better - n_base_better
print(f"\nPaired ({n_total} pairs): EWMA better={n_ewma_better}  Base better={n_base_better}  Similar={n_similar}")

# ── Table 4: Mechanism evidence (diesel/orders) ───────────────────────────────
print("\n"+"="*80)
print("TABLE 4: Mechanism Evidence (diesel, orders, stockouts)")
print("="*80)
mech_cols = [c for c in ["diesel_kWh","orders_placed","stockout_events"] if c in df.columns]
mech = df.groupby(["variant","lead_scenario"])[mech_cols].mean().round(2)
print(mech)
print("Note: step-level EWMA fields (ewma_lead_h etc.) require --trace_out for mechanism plots.")

# ── Figure ────────────────────────────────────────────────────────────────────
BG, BLUE, GREY = "#FAFAFA", "#0072B2", "#888888"
fig, axes = plt.subplots(2,2,figsize=(11,8))
axes_flat = axes.flatten()
fig.patch.set_facecolor(BG)
fig.suptitle("Exploratory Delay-Aware Inventory Extension\n"
             "Performance under Persistent Supplier Regimes\n"
             r"$\it{Reviewer\text{-}motivated\ exploratory\ extension\ (not\ part\ of\ the\ primary\ RLInv\ architecture)}$",
             fontsize=10.5, fontweight="bold")

for ax_idx, sc in enumerate(SCENARIOS):
    ax = axes_flat[ax_idx]
    ax.set_facecolor(BG)
    ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.8)
    ax.set_axisbelow(True)
    means, errs = [], []
    for var in ["base_regime","ewma_regime"]:
        sub = df[(df.variant==var)&(df.lead_scenario==sc)]
        means.append(sub["EENS_kWh"].mean() if not sub.empty else 0)
        errs.append(sub["EENS_kWh"].sem()*1.96 if not sub.empty else 0)
    x = np.arange(2)
    ax.bar(x, means, width=0.5, color=[GREY,BLUE], edgecolor="white", zorder=3)
    ax.errorbar(x, means, yerr=errs, fmt="none", color="#333333",
                capsize=4, linewidth=1.2, zorder=4)
    for xi,(m,e) in enumerate(zip(means,errs)):
        ax.text(xi, m+e+0.3, f"{m:.2f}", ha="center", fontsize=9,
                fontweight="bold", color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(["Base\n(no EWMA)","RLInv-DA\n(EWMA, α=0.3)"], fontsize=9)
    ax.set_ylabel("Mean EENS (kWh)" if ax_idx % 2 == 0 else "")
    ax.set_title({"normal":"Normal (24h)","delayed":"Delayed (48h)","monsoon":"Monsoon (72h)","extreme":"Extreme (336h)"}.get(sc,sc), fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)



plt.tight_layout()
for ext in ["pdf","png"]:
    out = FIG_DIR/f"fig_rlinv_da.{ext}"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")
plt.close()

# ── Automatic conclusion ──────────────────────────────────────────────────────
print("\n"+"="*80)
print("AUTOMATIC CONCLUSION")
print("="*80)
for sc in SCENARIOS:
    bv = df[(df.variant=="base_regime")&(df.lead_scenario==sc)]["EENS_kWh"].mean()
    ev = df[(df.variant=="ewma_regime")&(df.lead_scenario==sc)]["EENS_kWh"].mean()
    d  = ev - bv
    pct = d/bv*100 if bv > 0.01 else float("nan")
    print(f"  {sc}: base={bv:.3f}  ewma={ev:.3f}  diff={d:+.3f} ({pct:+.1f}%)")

print(f"\n  Paired: {n_ewma_better}/{n_total} EWMA better")
if n_ewma_better > n_total*0.6:
    print("  FINDING A: EWMA HELPS")
    print("  Adaptive lead tracking improves ordering under supplier regime switching.")
elif n_base_better > n_total*0.6:
    print("  FINDING B: BASE BETTER")
    print("  Inventory state already encodes sufficient information.")
else:
    print("  FINDING C: NO CLEAR DIFFERENCE")
    print("  EWMA provides diminishing returns in this setting.")
    print("  (Both outcomes were pre-declared as defensible before the experiment.)")

print("\nSaved: rlinv_da_summary.csv, rlinv_da_site_breakdown.csv, rlinv_da_seed_breakdown.csv")
print("Figure: results/figures/fig_rlinv_da.png")
