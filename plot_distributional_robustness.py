"""
plot_distributional_robustness.py

Single-panel distributional robustness figure.
All three policies × three distributions in one panel.
Log y-axis to handle the extreme scenario range.

Reads:
  results/sensitivity/lognormal/lognormal_summary.csv
  results/sensitivity/lognormal_sigma08/lognormal08_summary.csv
Output:
  results/figures/fig_distributional_robustness.pdf
  results/figures/fig_distributional_robustness.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from pathlib import Path

EV15_CSV  = Path("results/sensitivity/lognormal/lognormal_summary.csv")
EV15B_CSV = Path("results/sensitivity/lognormal_sigma08/lognormal08_summary.csv")
OUT_DIR   = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ev15  = pd.read_csv(EV15_CSV)
ev15b = pd.read_csv(EV15B_CSV)

SCENARIOS = ["normal", "delayed", "monsoon", "extreme"]
SC_LABELS = ["Normal\n(24h)", "Delayed\n(48h)", "Monsoon\n(72h)", "Extreme\n(336h)"]
x = np.arange(len(SCENARIOS))

# Okabe-Ito per policy -- consistent with other thesis figures
POL = {
    "rlinv": {"color": "#0072B2", "label": "RLInv"},
    "mpc":   {"color": "#009E73", "label": "MPC"},
    "b1":    {"color": "#D55E00", "label": "B1"},
}
DIST = {
    "geo":   {"ls": "-",  "marker": "o", "ms": 7, "lw": 2.2},
    "log05": {"ls": "--", "marker": "s", "ms": 6, "lw": 1.8},
    "log08": {"ls": ":",  "marker": "^", "ms": 6, "lw": 1.8},
}

BG   = "#FAFAFA"
GRID = "#EEEEEE"

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.labelsize":     11,
    "legend.fontsize":    9,
    "xtick.labelsize":    9.5,
    "ytick.labelsize":    9,
})

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color=GRID, linewidth=0.8)

OFFSET = 0.3  # for symlog

for policy, pp in POL.items():
    # Collect values across distributions
    geo_vals = [
        ev15b[(ev15b.policy==policy) & (ev15b.scenario==sc)]["geo_EENS"].values[0]
        if len(ev15b[(ev15b.policy==policy) & (ev15b.scenario==sc)]) > 0
        else np.nan for sc in SCENARIOS]

    log05_vals = [
        ev15[(ev15.policy==policy) & (ev15.scenario==sc)]["log_EENS"].values[0]
        if len(ev15[(ev15.policy==policy) & (ev15.scenario==sc)]) > 0
        else np.nan for sc in SCENARIOS]

    log08_vals = [
        ev15b[(ev15b.policy==policy) & (ev15b.scenario==sc)]["log08_EENS"].values[0]
        if len(ev15b[(ev15b.policy==policy) & (ev15b.scenario==sc)]) > 0
        else np.nan for sc in SCENARIOS]

    for key, vals in [("geo", geo_vals), ("log05", log05_vals), ("log08", log08_vals)]:
        d = DIST[key]
        ax.semilogy(x, [v + OFFSET for v in vals],
                    color=pp["color"], linestyle=d["ls"],
                    marker=d["marker"], markersize=d["ms"],
                    linewidth=d["lw"], zorder=3,
                    label="_nolegend_")

ax.set_xticks(x)
ax.set_xticklabels(SC_LABELS, fontsize=9.5)
ax.set_xlim(-0.3, 3.3)
ax.set_ylabel("Mean EENS (kWh, logarithmic scale)")
ax.set_title("Distributional Robustness of Policy Performance Under Alternative Delivery-Time Distributions\n"
             "Geometric (training), Lognormal (σ=0.5), and Lognormal (σ=0.8); scenario mean delivery times held constant",
             fontsize=10.5)

# Y-axis ticks in original units
yticks = [0.3, 1, 2, 5, 10, 20, 50, 100]
ax.set_yticks([y + OFFSET for y in yticks])
ax.set_yticklabels(["0", "1", "2", "5", "10", "20", "50", "100"])
ax.set_ylim(0.25, 130)

# ── Legend: two sections ──────────────────────────────────────────────────────
# Policy colours -- order: RLInv, B1, MPC (consistent with thesis)
pol_order = ["rlinv", "b1", "mpc"]
pol_handles = [
    Line2D([0],[0], color=POL[p]["color"], linewidth=2.5,
           label=POL[p]["label"])
    for p in pol_order
]
# Distribution line styles
dist_handles = [
    Line2D([0],[0], color="#555555", linestyle=d["ls"],
           marker=d["marker"], markersize=5, linewidth=1.8,
           label=lbl)
    for (key, d), lbl in zip(
        DIST.items(),
        ["Geometric (training)", "Lognormal σ=0.5 (EV15)", "Lognormal σ=0.8 (EV15b)"])
]

leg1 = ax.legend(handles=pol_handles, loc="upper left",
                 title="Policy", framealpha=0.9,
                 frameon=True, fontsize=9)
leg1.get_frame().set_linewidth(0)
ax.add_artist(leg1)

leg2 = ax.legend(handles=dist_handles, loc="upper center",
                 title="Distribution", framealpha=0.9,
                 frameon=True, fontsize=9)
leg2.get_frame().set_linewidth(0)

# Annotation -- larger font, precise scientific wording
ax.text(3.25, 1.0, "RLInv < MPC < B1\npreserved under\nall evaluated\ndistributions",
        ha="right", va="bottom", fontsize=10, color="#336633",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEFFEE",
                  edgecolor="#AACCAA", linewidth=0.7))

plt.tight_layout()

for ext in ["pdf", "png"]:
    out = OUT_DIR / f"fig_distributional_robustness.{ext}"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close()
print("Done.")
