"""
plot_lead_sensitivity.py

Generates Figure: EENS vs Mean Delivery Lead Time
For thesis Chapter 8 -- methodology justification subsection.

Three curves only: RLInv, B1, B0.
Scientific question: how does increasing delivery uncertainty affect
EENS, and does RLInv degrade more gracefully than reactive baselines?

Reads: results/lead_sensitivity.csv
Output: results/figures/fig_lead_sensitivity.pdf
        results/figures/fig_lead_sensitivity.png
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
IN_CSV  = Path("results/lead_sensitivity.csv")
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load and aggregate ────────────────────────────────────────────────────────
df = pd.read_csv(IN_CSV)

LEAD_HOURS = {
    "no_delay":    0,
    "fast":       12,
    "normal":     24,
    "delayed":    48,
    "monsoon":    72,
    "very_delayed": 120,
    "extreme":    336,
}
# Named thesis scenarios for reference lines
NAMED = {"Normal": 24, "Delayed": 48, "Monsoon": 72, "Extreme": 336}

if "mean_lead_hours" in df.columns:
    df["lead_h"] = df["mean_lead_hours"]
else:
    df["lead_h"] = df["lead_scenario"].map(LEAD_HOURS)

agg = (df.groupby(["policy", "lead_scenario", "lead_h"])["EENS_kWh"]
         .mean().reset_index().sort_values("lead_h"))

# ── Okabe-Ito palette ─────────────────────────────────────────────────────────
POLICIES = {
    "rlinv": {"label": "RLInv (learned ordering)",  "color": "#0072B2",
              "marker": "s", "lw": 2.8, "zorder": 5},
    "b1":    {"label": "B1 (reactive ordering)",    "color": "#D55E00",
              "marker": "o", "lw": 2.8, "zorder": 4},
    "b0":    {"label": "B0 (no replenishment)",     "color": "#CC79A7",
              "marker": "v", "lw": 1.8, "ls": ":", "zorder": 3},
}

BG   = "#FAFAFA"
GRID = "#E8E8E8"

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "legend.fontsize":    10,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0),
                                gridspec_kw={"width_ratios": [1.6, 1]})
fig.patch.set_facecolor(BG)

# ─── Panel (a): EENS by policy ────────────────────────────────────────────────
ax1.set_facecolor(BG)
ax1.set_axisbelow(True)
ax1.yaxis.grid(True, color=GRID, linewidth=0.8)

for policy, props in POLICIES.items():
    sub = agg[agg["policy"] == policy].sort_values("lead_h")
    if sub.empty:
        continue
    ax1.plot(sub["lead_h"], sub["EENS_kWh"],
             marker=props["marker"],
             color=props["color"],
             linewidth=props.get("lw", 2.0),
             linestyle=props.get("ls", "-"),
             markersize=7,
             label=props["label"],
             zorder=props.get("zorder", 3))

# Shade RLInv vs B1 advantage
rlinv_d = agg[agg["policy"]=="rlinv"].sort_values("lead_h")
b1_d    = agg[agg["policy"]=="b1"].sort_values("lead_h")
common_h = sorted(set(rlinv_d["lead_h"]) & set(b1_d["lead_h"]))
r_vals = rlinv_d.set_index("lead_h").loc[common_h, "EENS_kWh"].values
b_vals = b1_d.set_index("lead_h").loc[common_h, "EENS_kWh"].values
ax1.fill_between(common_h, r_vals, b_vals, alpha=0.10,
                 color="#0072B2", label="_nolegend_")

# Named scenario reference lines -- labels above plot
ymax = agg[agg["policy"].isin(POLICIES)]["EENS_kWh"].max()
for name, h in NAMED.items():
    ax1.axvline(x=h, color="#CCCCCC", linestyle="--", linewidth=1.0, zorder=1)
    ax1.text(h, ymax * 1.10, name, ha="center", fontsize=8.5,
             color="#888888", style="italic")

ax1.set_xlabel("Mean delivery lead time (hours)")
ax1.set_ylabel("Mean EENS (kWh), all sites")
ax1.set_title("(a)  EENS vs Mean Delivery Lead Time\n"
              "Aggregated over 10 sites and all evaluation seeds", fontsize=11)

# Show fewer x-axis ticks to avoid overlap at short lead times
display_hours = [0, 24, 48, 72, 120, 336]
ax1.set_xticks(display_hours)
ax1.set_xticklabels([f"{h}h" for h in display_hours], rotation=30, ha="right")
ax1.set_xlim(-5, 350)
ax1.set_ylim(bottom=-5, top=ymax * 1.22)
leg = ax1.legend(loc="upper left", framealpha=0.9, frameon=True)
leg.get_frame().set_linewidth(0)

# ─── Panel (b): RLInv advantage over B1 ──────────────────────────────────────
ax2.set_facecolor(BG)
ax2.set_axisbelow(True)
ax2.yaxis.grid(True, color=GRID, linewidth=0.8)

adv = b_vals - r_vals
ax2.plot(common_h, adv, "D-", color="#555555", linewidth=2.8,
         markersize=7, zorder=3)
ax2.fill_between(common_h, 0, adv, alpha=0.13, color="#555555")

for h, a in zip(common_h, adv):
    if a > 0.5:
        ax2.text(h, a + 1.5, f"{a:.1f}", ha="center",
                 fontsize=8, color="#444444", fontweight="bold")

# Named scenario reference lines
for name, h in NAMED.items():
    ax2.axvline(x=h, color="#CCCCCC", linestyle="--", linewidth=1.0, zorder=1)

ax2.axhline(y=0, color="#AAAAAA", linewidth=0.8)
ax2.set_xlabel("Mean delivery lead time (hours)")
ax2.set_ylabel("RLInv advantage over B1 (kWh)")
ax2.set_title("(b)  RLInv Advantage over B1\n"
              "RLInv maintains an advantage across increasing delivery delays", fontsize=11)
ax2.set_xticks(display_hours)
ax2.set_xticklabels([f"{h}h" for h in display_hours], rotation=30, ha="right")
ax2.set_xlim(-5, 350)
ax2.set_ylim(bottom=-3)

ax2.text(0.97, 0.04, "n = 10 sites × 10 seeds",
         transform=ax2.transAxes, ha="right", fontsize=8,
         color="#777777", style="italic")

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ["pdf", "png"]:
    out = OUT_DIR / f"fig_lead_sensitivity.{ext}"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close()
print("Done.")
