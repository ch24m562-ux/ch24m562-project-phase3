"""
plot_phase3_core_figures_v3.py

Cosmetic refinement pass on the 4 core Phase-3 figures per second reviewer
round: consistent policy colours across all figures, shortened axis labels
(parenthetical detail moved to captions), smaller scenario subtitles, H2
collapsed to a single panel, H3 labels simplified to bare policy names.

Reads: results/phase3/master_summary.csv
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

IN_CSV = "results/phase3/master_summary.csv"
OUT_DIR = "results/figures"

df = pd.read_csv(IN_CSV)
SCEN_ORDER = ["normal", "delayed", "monsoon", "extreme"]
SCEN_LABEL = {"normal": "Normal", "delayed": "Delayed", "monsoon": "Monsoon", "extreme": "Extreme"}

# ── Standardised colour palette -- used identically across all 4 figures ────
C = {
    "b0": "#b0b0b0", "b1": "#4dac26", "mpc": "#35978f", "trackb": "#d6604d",
    "oracle_mpc": "#666666", "rlinv": "#2166ac", "a5": "#f1a340", "a6": "#998ec3",
    "a7": "#f4a582",
}

def seed_means(sub):
    return sub.groupby("seed")["EENS_kWh"].mean()

def ci95(vals):
    vals = np.asarray(vals)
    n = len(vals)
    m = vals.mean()
    se = vals.std(ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf(0.975, n - 1)
    return m, m - h, m + h

plt.rcParams.update({
    "font.family": "serif", "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ═══════════════════════════════════════════════════════════════════════
# FIG 1 — Overall Policy Performance Across Delivery Scenarios
# ═══════════════════════════════════════════════════════════════════════
POLICY_ORDER = ["b0", "b1", "mpc", "trackb", "oracle_mpc", "rlinv"]
POLICY_LABEL = {"b0": "B0", "b1": "B1", "mpc": "MPC", "trackb": "TrackB",
                 "oracle_mpc": "Oracle-MPC", "rlinv": "RLInv"}

fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), sharey=False)
for ax, scen in zip(axes, SCEN_ORDER):
    sub_all = df[df.lead_scenario == scen]
    means, los, his = [], [], []
    for pol in POLICY_ORDER:
        sm = seed_means(sub_all[sub_all.policy == pol])
        m, lo, hi = ci95(sm.values)
        means.append(m); los.append(m - lo); his.append(hi - m)
    x = np.arange(len(POLICY_ORDER))
    colors = [C[p] for p in POLICY_ORDER]
    bars = ax.bar(x, means, yerr=[los, his], capsize=3, color=colors, alpha=0.88)
    for b, p in zip(bars, POLICY_ORDER):
        if p == "oracle_mpc":
            b.set_hatch("//"); b.set_edgecolor("black")
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABEL[p] for p in POLICY_ORDER], fontsize=9)
    ax.set_title(SCEN_LABEL[scen], fontsize=10, fontweight="bold")  # reduced ~20%
    if scen == "normal":
        ax.set_ylabel("Mean EENS (kWh / episode)", fontsize=10.5)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{m:.2f}",
                 ha="center", va="bottom", fontsize=7.5)

fig.suptitle("Overall Policy Performance Across Delivery Scenarios\n"
             "10 sites × 10 seeds (n=100 per bar); 95% CI shown",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(f"{OUT_DIR}/fig_eens_comparison_v3.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved fig_eens_comparison_v3.png")

# ═══════════════════════════════════════════════════════════════════════
# FIG 2 — H1: Contribution of Joint Inventory-Dispatch Learning
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(14, 4.0), sharey=False)
pols_h1 = ["rlinv", "trackb", "a6"]
labels_h1 = ["RLInv", "TrackB", "A6"]
for ax, scen in zip(axes, SCEN_ORDER):
    sub = df[df.lead_scenario == scen]
    means, los, his = [], [], []
    for pol in pols_h1:
        sm = seed_means(sub[sub.policy == pol])
        m, lo, hi = ci95(sm.values)
        means.append(m); los.append(m - lo); his.append(hi - m)
    x = np.arange(3)
    ax.bar(x, means, yerr=[los, his], capsize=3,
           color=[C[p] for p in pols_h1], alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(labels_h1, fontsize=9.5)
    ax.set_title(SCEN_LABEL[scen], fontsize=10, fontweight="bold")
    if scen == "normal":
        ax.set_ylabel("Mean EENS (kWh)", fontsize=10)
    for i, m in enumerate(means):
        ax.text(i, m, f"{m:.2f}", ha="center", va="bottom", fontsize=8)

fig.suptitle("H1 --- Contribution of Joint Inventory--Dispatch Learning",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.savefig(f"{OUT_DIR}/fig_ablation_h1_v3.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved fig_ablation_h1_v3.png")

# ═══════════════════════════════════════════════════════════════════════
# FIG 3 — H2: Contribution of Action Masking (single panel: A6 vs TrackB)
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.5, 4.6))
means = {"a6": [], "trackb": []}
for pol in means:
    for scen in SCEN_ORDER:
        sm = seed_means(df[(df.policy == pol) & (df.lead_scenario == scen)])
        means[pol].append(sm.mean())
x = np.arange(len(SCEN_ORDER))
w = 0.38
ax.bar(x - w/2, means["a6"], w, color=C["a6"], label="A6 (masked)")
ax.bar(x + w/2, means["trackb"], w, color=C["trackb"], label="TrackB (unmasked)")
ax.set_xticks(x); ax.set_xticklabels([SCEN_LABEL[s] for s in SCEN_ORDER], fontsize=10)
ax.set_ylabel("Mean EENS (kWh)")
ax.set_title("H2 --- Contribution of Action Masking\n"
             "($(s,S)$ ordering held constant; masking is the only difference)",
             fontsize=11.5, fontweight="bold")
ax.legend(fontsize=9)
for i, (v1, v2) in enumerate(zip(means["a6"], means["trackb"])):
    ax.text(i - w/2, v1, f"{v1:.1f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + w/2, v2, f"{v2:.1f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_ablation_h2_v3.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved fig_ablation_h2_v3.png")

# ═══════════════════════════════════════════════════════════════════════
# FIG 4 — H3: Effect of Explicit Inventory State Observation (RLInv vs A5)
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.6))
means = {"rlinv": [], "a5": []}
cis = {"rlinv": [], "a5": []}
for pol in means:
    for scen in SCEN_ORDER:
        sm = seed_means(df[(df.policy == pol) & (df.lead_scenario == scen)])
        m, lo, hi = ci95(sm.values)
        means[pol].append(m)
        cis[pol].append((m - lo, hi - m))
x = np.arange(len(SCEN_ORDER))
w = 0.38
lo_r = [c[0] for c in cis["rlinv"]]; hi_r = [c[1] for c in cis["rlinv"]]
lo_a = [c[0] for c in cis["a5"]]; hi_a = [c[1] for c in cis["a5"]]
ax.bar(x - w/2, means["rlinv"], w, yerr=[lo_r, hi_r], capsize=3, color=C["rlinv"], label="RLInv")
ax.bar(x + w/2, means["a5"], w, yerr=[lo_a, hi_a], capsize=3, color=C["a5"], label="A5")
ax.set_xticks(x); ax.set_xticklabels([SCEN_LABEL[s] for s in SCEN_ORDER], fontsize=10)
ax.set_ylabel("Mean EENS (kWh)")
ax.set_title("H3 --- Effect of Explicit Inventory State Observation",
              fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9)
for i, (v1, v2) in enumerate(zip(means["rlinv"], means["a5"])):
    ax.text(i - w/2, v1, f"{v1:.2f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + w/2, v2, f"{v2:.2f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_ablation_h3_v3.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved fig_ablation_h3_v3.png")
