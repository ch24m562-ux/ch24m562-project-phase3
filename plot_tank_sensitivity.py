"""
plot_tank_sensitivity.py

Generates Figure: RLInv Advantage over B1 vs Tank Capacity
For thesis Chapter 8 -- methodology justification subsection.

Reads: results/sensitivity/tank/tank_summary.csv
Output: results/figures/fig_tank_sensitivity.pdf
        results/figures/fig_tank_sensitivity.png
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
TANK_SUMMARY = Path("results/sensitivity/tank/tank_summary.csv")
OUT_DIR      = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
if TANK_SUMMARY.exists():
    df = pd.read_csv(TANK_SUMMARY)
else:
    df = pd.DataFrame({
        "tank_hours":    [24,     48,     72,     144,   336],
        "tank_scale":    ["0.33","0.67","1.0",  "2.0", "4.67"],
        "rlinv_EENS":    [215.55, 42.88, 11.95,  0.00,  0.00],
        "b1_EENS":       [283.55, 94.68, 44.63,  4.81,  0.00],
        "advantage_kWh": [68.00,  51.81, 32.68,  4.81,  0.00],
    })
    print("[INFO] tank_summary.csv not found -- using hardcoded values")

df = df.sort_values("tank_hours").reset_index(drop=True)
hours     = df["tank_hours"].values
rlinv     = df["rlinv_EENS"].values
b1        = df["b1_EENS"].values
advantage = df["advantage_kWh"].values

# Compute % advantage for secondary axis -- avoid divide warning
adv_pct = np.zeros(len(b1))
mask = b1 > 0.01
adv_pct[mask] = 100 * advantage[mask] / b1[mask]

# ── Okabe-Ito colorblind-friendly palette ─────────────────────────────────────
BLUE    = "#0072B2"   # RLInv
RED     = "#D55E00"   # B1
GREY    = "#555555"   # advantage
BG      = "#FAFAFA"
GRID    = "#E8E8E8"

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

# ─── Panel (a): EENS by policy, log-scale y ───────────────────────────────────
ax1.set_facecolor(BG)
ax1.set_axisbelow(True)
ax1.yaxis.grid(True, color=GRID, linewidth=0.8)

# Use log scale with small offset so zeros plot cleanly
OFFSET = 0.3
b1_log    = b1    + OFFSET
rlinv_log = rlinv + OFFSET

ax1.semilogy(hours, b1_log,    "o-", color=RED,  linewidth=2.8,
             markersize=8, label="B1 (reactive ordering)", zorder=3)
ax1.semilogy(hours, rlinv_log, "s-", color=BLUE, linewidth=2.8,
             markersize=8, label="RLInv (learned ordering)", zorder=3)

# Shade advantage region
ax1.fill_between(hours, rlinv_log, b1_log, alpha=0.12, color=BLUE, zorder=2)

# Regime bands under x-axis
regime_y = 0.18
ax1.axvspan(0,   72,  ymin=0, ymax=0.04, color="#FFCCCC", alpha=0.6, clip_on=False)
ax1.axvspan(72,  144, ymin=0, ymax=0.04, color="#FFFFCC", alpha=0.6, clip_on=False)
ax1.axvspan(144, 370, ymin=0, ymax=0.04, color="#CCFFCC", alpha=0.6, clip_on=False)

ax1.text(36,  0.13, "Constrained", ha="center", fontsize=7.5,
         color="#AA0000", transform=ax1.get_xaxis_transform(), clip_on=False)
ax1.text(108, 0.13, "Transition", ha="center", fontsize=7.5,
         color="#888800", transform=ax1.get_xaxis_transform(), clip_on=False)
ax1.text(255, 0.13, "Unconstrained", ha="center", fontsize=7.5,
         color="#006600", transform=ax1.get_xaxis_transform(), clip_on=False)

# Baseline line
ax1.axvline(x=72, color="#888888", linestyle="--", linewidth=1.3, zorder=1)
ax1.text(74, 200, "Baseline\n(72h)", fontsize=8.5, color="#666666", va="top")

# Annotate 144h crossover
ax1.annotate("RLInv reaches near-zero EENS\nRLInv ≈ 0 kWh;  B1: 4.8 kWh",
             xy=(144, 4.81 + OFFSET), xytext=(155, 30),
             fontsize=8.5, color="#333333",
             arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0))

# Y-axis ticks in original units
yticks = [0.3, 1, 5, 10, 50, 100, 300]
ax1.set_yticks(yticks)
ax1.set_yticklabels(["0", "1", "5", "10", "50", "100", "300"])
ax1.set_ylim(0.2, 600)

ax1.set_xlabel("Tank capacity (hours of buffer)")
ax1.set_ylabel("Mean EENS (kWh), hard sites  [log scale]")
ax1.set_title("(a)  EENS by Policy vs Tank Capacity\n"
              "Scenario: Delayed (mean lead time = 48h)", fontsize=11)
ax1.set_xticks(hours)
ax1.set_xticklabels([f"{h}h" for h in hours], rotation=30, ha="right")
ax1.set_xlim(10, 370)
leg = ax1.legend(loc="upper right", framealpha=0.9, frameon=True)
leg.get_frame().set_linewidth(0)

# ─── Panel (b): Advantage curve ──────────────────────────────────────────────
ax2.set_facecolor(BG)
ax2.set_axisbelow(True)
ax2.yaxis.grid(True, color=GRID, linewidth=0.8)

ax2.plot(hours, advantage, "D-", color=GREY, linewidth=2.8,
         markersize=8, zorder=3)
ax2.fill_between(hours, 0, advantage, alpha=0.13, color=GREY)

# Secondary y-axis: % advantage
ax2b = ax2.twinx()
ax2b.plot(hours, adv_pct, linestyle="none")  # invisible, just for axis
ax2b.set_ylabel("Relative improvement over B1 (%)", fontsize=9, color="#777777")
ax2b.tick_params(axis="y", labelcolor="#777777", labelsize=8)
ax2b.set_ylim(0, max(adv_pct) * 1.25 if max(adv_pct) > 0 else 100)
ax2b.spines["right"].set_visible(True)
ax2b.spines["right"].set_color("#CCCCCC")
# % labels removed -- secondary axis is sufficient

# Baseline line
ax2.axvline(x=72, color="#888888", linestyle="--", linewidth=1.3, zorder=1)
ax2.text(74, max(advantage) * 0.88, "Baseline", fontsize=8.5,
         color="#666666", va="top")

# Annotate vanishing
ax2.annotate("RLInv advantage becomes\nnegligible beyond 144h",
             xy=(336, 0), xytext=(200, 22),
             fontsize=8.5, color="#333333",
             arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0))

# Value labels
for h, adv in zip(hours, advantage):
    label = f"{adv:.1f}" if adv > 0 else "0"
    ax2.text(h, adv + 1.8, label, ha="center", fontsize=8.5, color=GREY,
             fontweight="bold")

ax2.set_xlabel("Tank capacity (hours of buffer)")
ax2.set_ylabel("RLInv advantage over B1 (kWh)")
ax2.set_title("(b)  RLInv Advantage vs Tank Capacity\n"
              "Monotone decrease confirms inventory mechanism", fontsize=11)
ax2.set_xticks(hours)
ax2.set_xticklabels([f"{h}h" for h in hours], rotation=30, ha="right")
ax2.set_xlim(10, 370)
ax2.set_ylim(-3, max(advantage) * 1.18)

# Sample size note
ax2.text(0.97, 0.04,
         "n = 3 sites × 10 seeds × 10 episodes",
         transform=ax2.transAxes, ha="right", fontsize=8,
         color="#777777", style="italic")

# ── Figure caption ────────────────────────────────────────────────────────────
plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ["pdf", "png"]:
    out = OUT_DIR / f"fig_tank_sensitivity.{ext}"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close()
print("Done.")
