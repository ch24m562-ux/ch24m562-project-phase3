"""
plot_trajectory.py

Behavioural trajectory: RLInv vs B1, site2/monsoon/seed99/ep0
Illustrates the inventory management mechanism -- proactive vs reactive ordering.

Reads:
  results/traces/rlinv_site2_monsoon_s99_ep0.npz
  results/traces/b1_site2_monsoon_s99_ep0.npz

Output:
  results/figures/fig_trajectory.pdf
  results/figures/fig_trajectory.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
RL_NPZ  = Path("results/traces/rlinv_site2_monsoon_s99_ep0.npz")
B1_NPZ  = Path("results/traces/b1_site2_monsoon_s99_ep0.npz")
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load traces ───────────────────────────────────────────────────────────────
rl = np.load(RL_NPZ)
b1 = np.load(B1_NPZ)

hours = np.arange(len(rl["inv_pct"]))

# ── Palette (Okabe-Ito) ───────────────────────────────────────────────────────
BLUE   = "#0072B2"   # RLInv
ORANGE = "#D55E00"   # B1
RED    = "#CC0000"   # unmet load
GREEN  = "#009E73"   # DG on
GREY   = "#BBBBBB"   # grid outage bands
BG     = "#FAFAFA"
GRID   = "#EEEEEE"

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize":  10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# ── Grid outage periods ───────────────────────────────────────────────────────
# grid_avail=0 means outage
outage = (rl["grid_avail"] < 0.5).astype(float)

def shade_outages(ax, alpha=0.12):
    in_outage = False
    start = 0
    for i, v in enumerate(outage):
        if v > 0.5 and not in_outage:
            start = i
            in_outage = True
        elif v < 0.5 and in_outage:
            ax.axvspan(start, i, color=GREY, alpha=alpha, zorder=0)
            in_outage = False
    if in_outage:
        ax.axvspan(start, len(outage), color=GREY, alpha=alpha, zorder=0)

# ── Figure: 3 panels ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(11, 8),
                          gridspec_kw={"height_ratios": [2.5, 1.2, 1.2]},
                          sharex=True)
fig.patch.set_facecolor(BG)
fig.suptitle(
    "Behavioural Trajectory: RLInv vs B1\n"
    "site2  |  Monsoon scenario",
    fontsize=12, fontweight="bold", y=0.98
)

# ─── Panel A: Inventory fraction ─────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor(BG)
ax.yaxis.grid(True, color=GRID, linewidth=0.7)
shade_outages(ax)

ax.plot(hours, rl["inv_pct"], color=BLUE,   linewidth=2.2, label="RLInv", zorder=3)
ax.plot(hours, b1["inv_pct"], color=ORANGE, linewidth=2.2, label="B1",    zorder=3)
ax.axhline(0.20, color=RED, linestyle="--", linewidth=1.2,
           label="Critical threshold (0.20)", zorder=2)

# Shade B1 below-threshold region
below = np.where(b1["inv_pct"] < 0.20, b1["inv_pct"], 0.20)
ax.fill_between(hours, below, 0.20,
                where=(b1["inv_pct"] < 0.20),
                color=ORANGE, alpha=0.20, label="_nolegend_")

# Annotate order events
rl_orders = np.where(rl["order_kwh"] > 0)[0]
b1_orders = np.where(b1["order_kwh"] > 0)[0]
for h in rl_orders:
    ax.annotate("", xy=(h, rl["inv_pct"][h] + 0.06),
                xytext=(h, rl["inv_pct"][h] + 0.02),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.5))
for h in b1_orders:
    ax.annotate("", xy=(h, b1["inv_pct"][h] + 0.06),
                xytext=(h, b1["inv_pct"][h] + 0.02),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5))

ax.set_ylabel("Inventory fraction")
ax.set_ylim(-0.02, 1.05)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Legend
legend_elements = [
    Line2D([0],[0], color=BLUE,   linewidth=2, label=f"RLInv  (4 orders, EENS = 0.00 kWh)"),
    Line2D([0],[0], color=ORANGE, linewidth=2, label=f"B1     (1 order,  EENS = 140.63 kWh)"),
    Line2D([0],[0], color=RED, linestyle="--", linewidth=1.2, label="Critical threshold (0.20)"),
    mpatches.Patch(color=GREY, alpha=0.4, label="Grid outage periods"),
]
ax.legend(handles=legend_elements, loc="upper right",
          framealpha=0.9, frameon=True).get_frame().set_linewidth(0)
ax.set_title("A.  Inventory fraction over episode", fontsize=10, loc="left", pad=4)

# ─── Panel B: Orders placed ───────────────────────────────────────────────────
ax = axes[1]
ax.set_facecolor(BG)
ax.yaxis.grid(True, color=GRID, linewidth=0.7)
shade_outages(ax, alpha=0.08)

# Stem plot for orders
for h in rl_orders:
    ax.annotate("", xy=(h, rl["order_kwh"][h]),
                xytext=(h, 0),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.0))
    ax.plot(h, rl["order_kwh"][h], "^", color=BLUE, markersize=8, zorder=4)

for h in b1_orders:
    ax.annotate("", xy=(h, b1["order_kwh"][h]),
                xytext=(h, 0),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.0))
    ax.plot(h, b1["order_kwh"][h], "v", color=ORANGE, markersize=8, zorder=4)

ax.set_ylabel("Order size (kWh)")
ax.set_ylim(bottom=0)
ax.set_title("B.  Diesel orders placed", fontsize=10, loc="left", pad=4)
order_legend = [
    Line2D([0],[0], marker="^", color=BLUE,   linewidth=0, markersize=8,
           label=f"RLInv orders ({len(rl_orders)} total)"),
    Line2D([0],[0], marker="v", color=ORANGE, linewidth=0, markersize=8,
           label=f"B1 orders ({len(b1_orders)} total)"),
]
ax.legend(handles=order_legend, loc="upper right",
          framealpha=0.9, frameon=True).get_frame().set_linewidth(0)

# ─── Panel C: Unmet load ─────────────────────────────────────────────────────
ax = axes[2]
ax.set_facecolor(BG)
ax.yaxis.grid(True, color=GRID, linewidth=0.7)
shade_outages(ax, alpha=0.08)

ax.fill_between(hours, b1["unmet_kwh"], alpha=0.6, color=ORANGE,
                label=f"B1 unmet load  (total = 140.63 kWh)", step="pre")
ax.fill_between(hours, rl["unmet_kwh"], alpha=0.8, color=BLUE,
                label=f"RLInv unmet load  (total = 0.00 kWh)", step="pre")

ax.set_xlabel("Hour of episode")
ax.set_ylabel("Unmet load (kWh)")
ax.set_title("C.  Unmet energy demand", fontsize=10, loc="left", pad=4)
ax.set_xlim(0, 360)
ax.legend(loc="upper left", framealpha=0.9,
          frameon=True).get_frame().set_linewidth(0)

# ── Key takeaway box ──────────────────────────────────────────────────────────
fig.text(0.5, 0.01,
    "The trajectory plot is presented to explain policy behaviour rather than to establish "
    "statistical significance. Statistical conclusions are drawn from the multi-seed "
    "sensitivity and robustness analyses presented in the preceding sections.",
    ha="center", fontsize=8.5, color="#555555", style="italic")

plt.tight_layout(rect=[0, 0.05, 1, 0.97])

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ["pdf", "png"]:
    out = OUT_DIR / f"fig_trajectory.{ext}"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close()
print("Done.")
