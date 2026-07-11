"""
plot_reward_sensitivity.py

Generates Figure: Reward Weight Sensitivity Analysis
For thesis Chapter -- Reward Function Design and Justification subsection.

Two panels:
  (a) EENS across reward weight variants (monsoon scenario, hard sites)
      Shows stability: all variants << B1 baseline
  (b) Discount factor gamma comparison (0.95 vs 0.995)
      Shows gamma=0.995 is the correct choice (planning horizon argument)

Reads: results/sensitivity/reward_sensitivity_summary.csv
       gamma results hardcoded from MLflow experiment 21
Output: results/figures/fig_reward_sensitivity.pdf
        results/figures/fig_reward_sensitivity.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SENS_CSV = Path("results/sensitivity/reward_sensitivity_summary.csv")
OUT_DIR  = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load reward sensitivity data ──────────────────────────────────────────────
df = pd.read_csv(SENS_CSV)
monsoon = df[df.lead_scenario == "monsoon"].set_index("policy")

# B1 matched baseline (hard sites site2/5/7, seeds 42/123/777, monsoon)
B1_MONSOON = 57.45

# ── Variant definitions ───────────────────────────────────────────────────────
# Each group: (display_label, variant_name, param_value, bar_color_shade)
GROUPS = {
    "β\n(grid cost)": [
        ("β=0.2\n(low)",    "beta_low",   0.20, 0.4),
        ("β=0.8\n(high)",   "beta_high",  0.80, 0.8),
    ],
    "γᵣ\n(DG start)": [
        ("γᵣ=100\n(0.5×)",  "cliff_low",  100, 0.4),
        ("γᵣ=400\n(2×)",    "cliff_high", 400, 0.8),
    ],
    "λ\n(unmet load)": [
        ("λ=25\n(0.25×)",   "lam_vlow",   25,  0.25),
        ("λ=50\n(0.5×)",    "lam_low",    50,  0.45),
        ("λ=200\n(2×)",     "lam_high",   200, 0.70),
        ("λ=500\n(5×)",     "lam_vhigh",  500, 0.90),
    ],
    "μ\n(SOC viol.)": [
        ("μ=5\n(0.25×)",    "mu_low",     5,   0.4),
        ("μ=50\n(2.5×)",    "mu_high",    50,  0.8),
    ],
}

# Base values for each group (from master experiments)
BASE_EENS = 11.29  # RLInv monsoon mean EENS from master_summary (all sites)
# Use hard-site only base from the sensitivity experiment
# cliff_high = cliff_low = mu_high = mu_low = 16.71 (unchanged from base)
BASE_HARD_SITE = 16.71  # base variant EENS at hard sites

# ── Palette ───────────────────────────────────────────────────────────────────
BLUE   = "#0072B2"
ORANGE = "#D55E00"
GREY   = "#555555"
RED    = "#CC0000"
BG     = "#FAFAFA"
GRID   = "#EEEEEE"

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.labelsize":     11.5,
    "legend.fontsize":    10.5,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10.5,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                gridspec_kw={"width_ratios": [3, 1]})
fig.patch.set_facecolor(BG)

# ─── Panel (a): Reward weight sensitivity ─────────────────────────────────────
ax1.set_facecolor(BG)
ax1.set_axisbelow(True)
ax1.yaxis.grid(True, color=GRID, linewidth=0.8)

x_positions = []
x_labels    = []
bar_colors  = []
bar_heights = []
bar_cis     = []
group_centers = []
group_labels  = []

x = 0
gap_between_groups = 0.6

for group_name, variants in GROUPS.items():
    group_start = x
    for label, variant, val, shade in variants:
        eens = monsoon.loc[variant, "mean"] if variant in monsoon.index else BASE_HARD_SITE
        ci   = monsoon.loc[variant, "sem"] * 1.96 if variant in monsoon.index else 0

        x_positions.append(x)
        x_labels.append(label)
        bar_heights.append(eens)
        bar_cis.append(ci)
        # Color by group
        if "beta" in variant:
            bar_colors.append(plt.cm.Blues(0.4 + shade * 0.3))
        elif "cliff" in variant:
            bar_colors.append(plt.cm.Greens(0.4 + shade * 0.3))
        elif "lam" in variant:
            bar_colors.append(plt.cm.Oranges(0.3 + shade * 0.4))
        else:
            bar_colors.append(plt.cm.Purples(0.4 + shade * 0.3))
        x += 1

    group_centers.append((group_start + x - 1) / 2)
    group_labels.append(group_name)
    x += gap_between_groups

# Draw bars
bars = ax1.bar(x_positions, bar_heights, width=0.7, color=bar_colors,
               edgecolor="white", linewidth=0.5, zorder=3)

# Error bars
ax1.errorbar(x_positions, bar_heights, yerr=bar_cis,
             fmt="none", color="#333333", capsize=3, linewidth=1.2, zorder=4)

# Value labels on bars
for xp, h, ci in zip(x_positions, bar_heights, bar_cis):
    ax1.text(xp, h + ci + 0.4, f"{h:.1f}", ha="center",
             fontsize=8.5, color="#333333")

# B1 baseline reference line
ax1.axhline(B1_MONSOON, color=ORANGE, linestyle="--", linewidth=1.8,
            zorder=2, label=f"B1 baseline (monsoon) = {B1_MONSOON:.1f} kWh")

# Base RLInv line
ax1.axhline(BASE_HARD_SITE, color=BLUE, linestyle=":", linewidth=1.5,
            zorder=2, label=f"RLInv base config = {BASE_HARD_SITE:.1f} kWh")

# Shade below B1 region
ax1.fill_between([-0.5, max(x_positions)+0.5], 0, B1_MONSOON,
                 alpha=0.04, color=BLUE, zorder=1)

# Group separator lines and labels -- moved lower with box to avoid overlap
for i, (gc, gl) in enumerate(zip(group_centers, group_labels)):
    ax1.text(gc, -8, gl, ha="center", va="top", fontsize=11,
             color="#333333", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#F0F0F0",
                       edgecolor="#CCCCCC", linewidth=0.5))

# Group separator vertical lines
sep_x = []
cx = 0
for group_name, variants in GROUPS.items():
    cx += len(variants)
    sep_x.append(cx - 1 + gap_between_groups/2)
    cx += gap_between_groups

for sx in sep_x[:-1]:
    ax1.axvline(sx, color="#CCCCCC", linewidth=0.8, linestyle="-", zorder=0)

ax1.set_xticks([])  # no x tick labels -- bars are identified by value labels above
ax1.set_xlim(-0.6, max(x_positions) + 0.6)
ax1.set_ylim(-10, B1_MONSOON * 1.18)

# Add parameter value as small label below each bar
param_labels = {
    "beta_low": "β=0.2", "beta_high": "β=0.8",
    "cliff_low": "γᵣ=100", "cliff_high": "γᵣ=400",
    "lam_vlow": "λ=25", "lam_low": "λ=50",
    "lam_high": "λ=200", "lam_vhigh": "λ=500",
    "mu_low": "μ=5", "mu_high": "μ=50",
}
x_idx = 0
for group_name, variants in GROUPS.items():
    for label, variant, val, shade in variants:
        ax1.text(x_positions[x_idx], -1.5, param_labels.get(variant, ""),
                 ha="center", va="top", fontsize=8.5, color="#555555")
        x_idx += 1
ax1.set_ylabel("Mean EENS (kWh), hard sites\nMonsoon scenario")
ax1.set_title("(a)  Reward Weight Robustness — Monsoon Scenario\n"
              "All variants evaluated on hard sites (site2/5/7), "
              "3 seeds × 400k training steps",
              fontsize=11.5)
ax1.legend(loc="upper right", framealpha=0.9,
           frameon=True).get_frame().set_linewidth(0)

# Variance note
ax1.text(0.01, 0.97,
         "Error bars show 95% CI across sites and seeds.\n"
         "Large CI reflects site heterogeneity, not instability.",
         transform=ax1.transAxes, fontsize=8.5, va="top",
         color="#777777", style="italic")

# Annotation: robustness statement
ax1.annotate(
    f"Performance remains within a narrow band\n"
    f"({min(bar_heights):.1f}–{max(bar_heights):.1f} kWh), substantially\n"
    f"below the matched B1 baseline ({B1_MONSOON:.1f} kWh)",
    xy=(group_centers[2], max(bar_heights) + 1),
    xytext=(group_centers[2], B1_MONSOON * 0.70),
    fontsize=8.5, ha="center", color="#333333",
    arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0)
)

# ─── Panel (b): Gamma comparison ─────────────────────────────────────────────
ax2.set_facecolor(BG)
ax2.set_axisbelow(True)
ax2.yaxis.grid(True, color=GRID, linewidth=0.8)

# Gamma experiment results (from MLflow experiment 21, hard sites, same seeds)
gamma_vals   = [0.95, 0.995]
gamma_eens   = [0.319, 0.000]
gamma_cols   = [ORANGE, BLUE]
gamma_labels = ["γ = 0.95", "γ = 0.995"]

MIN_BAR = 0.025  # minimum visible bar height for zero values
bar_display = [gamma_eens[0], MIN_BAR]  # 0.319 and a thin visible bar
bars2 = ax2.bar([0, 1], bar_display, width=0.5,
                color=gamma_cols, edgecolor="white", linewidth=0.5, zorder=3)

# Value labels above bars
ax2.text(0, gamma_eens[0] + 0.015, "0.319", ha="center",
         fontsize=11, color="#333333", fontweight="bold")
ax2.text(1, MIN_BAR + 0.015, "0.000\n(EENS = 0)", ha="center",
         fontsize=9, color="#333333", fontweight="bold")

# Horizon annotations inside bars
ax2.text(0, 0.10, "Myopic:\n< 1 delivery\ncycle\n(horizon ~20h)",
         ha="center", fontsize=8.5, color="white", style="italic",
         fontweight="bold")
ax2.text(1, MIN_BAR / 2, "EENS=0\n(~200h)",
         ha="center", va="center", fontsize=8.5, color="white",
         fontweight="bold")

ax2.set_xticks([0, 1])
ax2.set_xticklabels(gamma_labels, fontsize=9)
ax2.set_ylim(0, 0.45)
ax2.set_ylabel("Mean EENS (kWh)")
ax2.set_title("(b)  Discount Factor γ\nPlanning horizon validation",
              fontsize=11.5)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ["pdf", "png"]:
    out = OUT_DIR / f"fig_reward_sensitivity.{ext}"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")

plt.close()
print("Done.")
