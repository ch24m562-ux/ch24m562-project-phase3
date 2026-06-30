"""
plot_training_curves.py

Generates Figure: RLInv Training Dynamics (Normal Scenario)
For thesis -- addresses reviewer comment #4 (training curve evidence).

Shows mean episode reward vs training steps for three independent
hard-site training runs (site2/site5/site7, seed=999, normal scenario).
No shaded bands (n=1 seed per site -- no cross-seed variance to show).

Reads: results/training_curves/rollout_reward_hard_sites.csv
Output: results/figures/fig_training_curves.pdf
        results/figures/fig_training_curves.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
IN_CSV  = Path("results/training_curves/rollout_reward_hard_sites.csv")
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(IN_CSV)

SITES = {
    "site2": {"color": "#0072B2", "label": "Site2 (seed=999)"},
    "site5": {"color": "#D55E00", "label": "Site5 (seed=999)"},
    "site7": {"color": "#009E73", "label": "Site7 (seed=999)"},
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
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
})

fig, ax = plt.subplots(figsize=(11, 6.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.xaxis.grid(True, color=GRID, linewidth=0.5, alpha=0.5)

# ── Plot each site -- no shading, clean lines only ────────────────────────────
for site, props in SITES.items():
    sub = df[df.site == site].sort_values("step")
    ax.plot(sub["step"], sub["reward"], color=props["color"],
             linewidth=1.8, label=props["label"], zorder=3)

# symlog y-axis: linear near zero, log scale for large magnitudes
ax.set_yscale("symlog", linthresh=300)

# Converged plateau reference line
ax.axhline(-350, color="#888888", linestyle="--", linewidth=1.2,
           zorder=2, label="Converged plateau (~ -330 to -360)")

ax.set_xlabel("Training steps")
ax.set_ylabel("Mean episode reward (rollout)")
ax.set_title("RLInv Training Dynamics under the Normal Training Scenario\n"
             "Mean episode reward vs training steps across three hard sites, seed = 999",
             fontsize=12)
ax.set_xlim(0, 410000)
ax.legend(loc="upper left", framealpha=0.95, frameon=True,
          bbox_to_anchor=(0.02, 0.98)).get_frame().set_linewidth(0)

# X-axis tick formatting (k notation)
ax.set_xticks([0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000])
ax.set_xticklabels(["0", "50k", "100k", "150k", "200k", "250k", "300k", "350k", "400k"])

# ── Inset: zoom on convergence region (200k-400k) ─────────────────────────────
axins = ax.inset_axes([0.62, 0.10, 0.34, 0.30])
axins.set_facecolor(BG)
for site, props in SITES.items():
    sub = df[(df.site == site) & (df.step >= 200000)].sort_values("step")
    axins.plot(sub["step"], sub["reward"], color=props["color"], linewidth=1.4)
axins.axhline(-350, color="#888888", linestyle="--", linewidth=1.0)
axins.set_xlim(200000, 405000)
axins.set_ylim(-500, -250)
axins.set_xticks([200000, 300000, 400000])
axins.set_xticklabels(["200k", "300k", "400k"], fontsize=7.5)
axins.set_yticks([-250, -350, -450])
axins.tick_params(labelsize=7.5)
axins.set_title("Zoom: convergence region", fontsize=8.5)
axins.grid(True, color=GRID, linewidth=0.5)
ax.indicate_inset_zoom(axins, edgecolor="#888888", linewidth=1.0)

# ── Summary table as text box ─────────────────────────────────────────────────
summary_lines = ["Site    Final reward   Initial→Final"]
for site in ["site2", "site5", "site7"]:
    sub = df[df.site == site].sort_values("step")
    first, last = sub["reward"].iloc[0], sub["reward"].iloc[-1]
    summary_lines.append(f"{site:<8}{last:>10.1f}{'':<6}{first:>9.0f} → {last:>6.0f}")

summary_text = "\n".join(summary_lines)
ax.text(0.015, 0.62, summary_text, transform=ax.transAxes,
        fontsize=8, family="monospace", va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#CCCCCC", linewidth=0.7), zorder=5)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ["pdf", "png"]:
    out = OUT_DIR / f"fig_training_curves.{ext}"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close()
print("Done.")
