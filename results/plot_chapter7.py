"""
plot_chapter7.py — Generate all Chapter 7 figures from all_results_final.csv

Run from project root:
    python plot_chapter7.py

Outputs (all 300 dpi, PDF-ready):
    results/figures/fig_eens_comparison.pdf   — §7.2 main bar chart
    results/figures/fig_pareto.pdf            — §7.3 reliability-cost trade-off
    results/figures/fig_ablation_h1.pdf       — §7.4 H1 ordering ablation
    results/figures/fig_ablation_h2.pdf       — §7.4 H2 masking ablation
    results/figures/fig_ablation_h3.pdf       — §7.4 H3 inventory obs ablation
    results/figures/fig_delayed_scenario.pdf  — §7.5 logistics stress
    results/figures/fig_cross_site.pdf        — §7.6 cross-site robustness
    results/figures/fig_hypothesis_summary.pdf— §7.6 hypothesis dot-plot

All statistics computed from raw seed-level data. No hardcoded means.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from decimal import Decimal, ROUND_HALF_UP

os.makedirs("results/figures", exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
})

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "RLInv":  "#2166ac",   # blue   — proposed
    "B1":     "#4dac26",   # green  — best baseline
    "B0":     "#878787",   # grey   — weak baseline
    "A5":     "#f1a340",   # orange — no inv obs
    "A6":     "#998ec3",   # purple — fixed ordering
    "A7":     "#f4a582",   # salmon — no masking
    "TrackB": "#d6604d",   # red    — hierarchical
}
POLICY_ORDER = ["B0", "B1", "A7", "TrackB", "A6", "A5", "RLInv"]
POLICY_LABELS = {
    "RLInv":  "RLInv\n(proposed)",
    "B1":     "B1\n(reorder rule)",
    "B0":     "B0\n(threshold)",
    "A5":     "A5\n(no inv. obs.)",
    "A6":     "A6\n(fixed $(s,S)$)",
    "A7":     "A7\n(no masking)",
    "TrackB": "TrackB\n(hier. RL)",
}
CONSTRAINED = ["site5", "site10"]

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading results/all_results_final.csv ...")
raw_df = pd.read_csv("results/all_results_final.csv")
df = raw_df.drop_duplicates()
n_dropped = len(raw_df) - len(df)
print(f"  Raw rows: {len(raw_df)}  |  Duplicate rows dropped: {n_dropped}  |  Unique records: {len(df)}")
print(f"  Policies: {sorted(df.policy.unique())}")
print(f"  Sites: {sorted(df.site.unique())}")
print(f"  Scenarios: {df.lead_scenario.unique()}")
# Coverage audit
print("\n  Policy coverage:")
for pol in sorted(df.policy.unique()):
    sub = df[df.policy == pol]
    sites = sorted(sub.site.unique())
    scens = sorted(sub.lead_scenario.unique())
    print(f"    {pol:8s}: {len(sites):2d} sites  |  scenarios: {scens}")

def seed_means(data, metric="EENS_kWh"):
    """Aggregate to seed-level means. Returns array of length n_seeds."""
    return data.groupby("seed")[metric].mean().values

def summary_stats(arr):
    """Return mean and sample std (ddof=1)."""
    return arr.mean(), arr.std(ddof=1)

def welch_test(a, b):
    """Welch t-test + Cohen's d, both using sample std (ddof=1)."""
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    t, p = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2 + 1e-9)
    d = abs((b.mean() - a.mean()) / pooled)
    return p, d

def plab(p):
    if np.isnan(p): return "n/a"
    if p < 0.001:  return f"p={p:.4f}***"
    if p < 0.01:   return f"p={p:.3f}**"
    if p < 0.05:   return f"p={p:.3f}*"
    return f"p={p:.3f} (ns)"

# Pre-compute seed-level means for all policies, constrained sites, normal
norm_con = df[(df.lead_scenario == "normal") & (df.site.isin(CONSTRAINED))]
seed_eens = {}
seed_diesel = {}
seed_uptime = {}
for pol in df.policy.unique():
    sub = norm_con[norm_con.policy == pol]
    if len(sub) > 0:
        seed_eens[pol]   = seed_means(sub, "EENS_kWh")
        seed_diesel[pol] = seed_means(sub, "diesel_kWh")
        seed_uptime[pol] = seed_means(sub, "uptime_pct")

print("\nSeed-level EENS (constrained sites, normal):")
for pol in POLICY_ORDER:
    if pol in seed_eens:
        m, s = summary_stats(seed_eens[pol])
        print(f"  {pol:8s}: {m:.1f} ± {s:.1f}")

# Print corrected pairwise stats for chapter text verification
print("\n--- CORRECTED PAIRWISE STATS (ddof=1, use these in chapter text) ---")
REF = "RLInv"
for comp in ["A6", "B0", "B1", "A5", "A7"]:
    if comp in seed_eens and REF in seed_eens:
        p, d = welch_test(seed_eens[REF], seed_eens[comp])
        diff = (seed_eens[comp].mean() - seed_eens[REF].mean()) / seed_eens[comp].mean() * 100
        print(f"  {REF} vs {comp:6s}: diff={diff:+.1f}%  p={p:.3f}  d={d:.2f}")
print("--- END CORRECTED STATS ---\n")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — EENS bar chart (§7.2 main result)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating fig_eens_comparison ...")

policies_plot = [p for p in POLICY_ORDER if p in seed_eens]
means  = [seed_eens[p].mean()         for p in policies_plot]
stds   = [seed_eens[p].std(ddof=1)    for p in policies_plot]
colors = [C[p] for p in policies_plot]
xlabs  = [POLICY_LABELS[p] for p in policies_plot]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(range(len(policies_plot)), means, yerr=stds,
              color=colors, alpha=0.85, width=0.6, capsize=5,
              error_kw=dict(ecolor="#333333", capsize=5, elinewidth=1.5),
              linewidth=0)

# Value labels on bars
for i, (bar, m) in enumerate(zip(bars, means)):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + stds[i] + 15,
            f"{m:.0f}", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color="#222222")

# Significance bracket: RLInv vs A6/TrackB
if "RLInv" in policies_plot and "A6" in policies_plot:
    ri = policies_plot.index("RLInv")
    ai = policies_plot.index("A6")
    p, d = welch_test(seed_eens["RLInv"], seed_eens["A6"])
    ymax = max(means[ri] + stds[ri], means[ai] + stds[ai]) + 80
    ax.annotate("", xy=(ri, ymax), xytext=(ai, ymax),
                arrowprops=dict(arrowstyle="-", color="black", lw=1.5))
    ax.text((ri + ai) / 2, ymax + 20, plab(p),
            ha="center", fontsize=8.5, fontweight="bold")

ax.set_xticks(range(len(policies_plot)))
ax.set_xticklabels(xlabs, fontsize=9)
ax.set_ylabel("Mean EENS (kWh / episode)", fontsize=10)
ax.set_title("Reliability Comparison — All Policies\n"
             "Constrained sites (site 5, site 10), normal logistics, $n=3$ seeds",
             fontsize=11, fontweight="bold")
ax.set_ylim(0, max(means) + max(stds) + 200)

plt.tight_layout()
plt.savefig("results/figures/fig_eens_comparison.pdf", bbox_inches="tight")
plt.savefig("results/figures/fig_eens_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved fig_eens_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Pareto frontier: EENS vs diesel (§7.3)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating fig_pareto ...")

fig, ax = plt.subplots(figsize=(7, 5))

pareto_points = []

for pol in POLICY_ORDER:
    if pol not in seed_eens or pol not in seed_diesel:
        continue

    me, se = summary_stats(seed_eens[pol])
    md, sd = summary_stats(seed_diesel[pol])

    marker_size = 12 if pol == "RLInv" else 9
    alpha_val   = 0.90 if pol == "RLInv" else 0.70
    line_w      = 2.0 if pol == "RLInv" else 1.5
    z_val       = 7 if pol == "RLInv" else 5

    ax.errorbar(
        md, me, xerr=sd, yerr=se,
        fmt="D", color=C[pol], markersize=marker_size,
        ecolor=C[pol], alpha=alpha_val, elinewidth=line_w, capsize=4,
        linewidth=0, zorder=z_val
    )

    # Collect frontier candidates
    if pol in ["B0", "TrackB", "A6", "B1", "RLInv"]:
        pareto_points.append((md, me, pol))

    # Label placement
    if pol == "B0":
        label = "B0"
        offset = (-95, -12)

    elif pol == "TrackB":
        # Show only one combined label for overlapping TrackB/A6 point
        label = "TrackB / A6"
        offset = (-120, 10)

    elif pol == "A6":
        # Skip separate A6 label since TrackB/A6 are identical
        continue

    elif pol == "RLInv":
        label = "RLInv (proposed)"
        offset = (14, 10)

    elif pol == "A7":
        label = "A7"
        offset = (12, 10)

    elif pol == "A5":
        label = "A5"
        offset = (12, 10)

    elif pol == "B1":
        label = "B1"
        offset = (10, 8)

    else:
        label = pol
        offset = (10, 8)

    ax.annotate(
        label,
        xy=(md, me), xytext=offset,
        textcoords="offset points",
        fontsize=9 if pol == "RLInv" else 8,
        color=C[pol],
        fontweight="bold" if pol == "RLInv" else "normal"
    )

# Add faint Pareto frontier guide
pareto_points = sorted(pareto_points, key=lambda x: x[0])  # sort by diesel
frontier_x = [p[0] for p in pareto_points]
frontier_y = [p[1] for p in pareto_points]
ax.plot(
    frontier_x, frontier_y,
    linestyle="--", linewidth=1.4,
    color="#777777", alpha=0.55, zorder=3
)

# Better-region note
ax.text(
    0.97, 0.95,
    "Better region:\nlower EENS, lower diesel",
    transform=ax.transAxes,
    fontsize=7.5,
    color="#888888",
    va="top",
    ha="right",
    style="italic"
)

# Single direction arrow toward better region
ax.annotate(
    "",
    xy=(0.86, 0.83),      # arrow head (better region)
    xytext=(0.93, 0.92),  # arrow tail (near annotation)
    xycoords="axes fraction",
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", color="#999999", lw=1.2)
)

ax.set_xlabel("Mean Diesel Consumption (kWh / episode)", fontsize=10)
ax.set_ylabel("Mean EENS (kWh / episode)\n← lower is better", fontsize=10)
ax.set_title(
    "Reliability–Cost Pareto Frontier\n"
    "Constrained sites, normal logistics, $n=3$ seeds",
    fontsize=11, fontweight="bold"
)

plt.tight_layout()
plt.savefig("results/figures/fig_pareto.pdf", bbox_inches="tight")
plt.savefig("results/figures/fig_pareto.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved fig_pareto")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Ablation H1: joint ordering (§7.4)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating fig_ablation_h1 ...")

fig, ax = plt.subplots(figsize=(6, 4.5))

h1_pols = ["B0", "TrackB", "A6", "B1", "RLInv"]  # B0/B1 kept for context; H1 bracket on A6 vs RLInv
# Reviewer: focus H1 figure on the three policies that directly test H1
h1_pols = ["TrackB", "A6", "B1", "RLInv"]
h1_pols = [p for p in h1_pols if p in seed_eens]
h1_means = [seed_eens[p].mean()      for p in h1_pols]
h1_stds  = [seed_eens[p].std(ddof=1) for p in h1_pols]
h1_colors = [C[p] for p in h1_pols]
h1_alphas = [0.85 if p != "B1" else 0.45 for p in h1_pols]  # B1 = context only

bars = ax.barh(range(len(h1_pols)), h1_means, xerr=h1_stds,
               color=h1_colors, alpha=None, height=0.55,
               capsize=4, error_kw=dict(ecolor="#333333", elinewidth=1.5),
               linewidth=0)
for bar, alpha in zip(bars, h1_alphas):
    bar.set_alpha(alpha)

for i, (bar, m) in enumerate(zip(bars, h1_means)):
    ax.text(bar.get_width() + h1_stds[i] + 10, i,
            f"{m:.0f}", ha="left", va="center",
            fontsize=8.5, fontweight="bold")

ax.set_yticks(range(len(h1_pols)))
h1_ylabels = []
for p in h1_pols:
    lbl = POLICY_LABELS[p].replace("\n", " ")
    if p == "B1":
        lbl += "  [context]"
    h1_ylabels.append(lbl)
ax.set_yticklabels(h1_ylabels, fontsize=9)
ax.set_xlabel("Mean EENS (kWh / episode)", fontsize=10)
ax.set_title("H1 — Contribution of Joint Learned Ordering\n"
             "Constrained sites, normal logistics, $n=3$ seeds",
             fontsize=10.5, fontweight="bold")

# Bracket showing RLInv vs A6
if "RLInv" in h1_pols and "A6" in h1_pols:
    ri = h1_pols.index("RLInv")
    ai = h1_pols.index("A6")
    p, d = welch_test(seed_eens["RLInv"], seed_eens["A6"])
    xmax = max(h1_means[ri] + h1_stds[ri], h1_means[ai] + h1_stds[ai]) + 30
    ax.annotate("", xy=(xmax, ri), xytext=(xmax, ai),
                arrowprops=dict(arrowstyle="-", color="black", lw=1.5))
    ax.text(xmax + 8, (ri + ai) / 2, f"{plab(p)}\nd={d:.2f}",
            ha="left", va="center", fontsize=8, fontweight="bold")

ax.invert_yaxis()
ax.axvline(seed_eens["RLInv"].mean(), color=C["RLInv"],
           linestyle="--", lw=1.2, alpha=0.5)

plt.tight_layout()
plt.savefig("results/figures/fig_ablation_h1.pdf", bbox_inches="tight")
plt.savefig("results/figures/fig_ablation_h1.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved fig_ablation_h1")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Ablation H2: action masking (§7.4)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating fig_ablation_h2 ...")

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

# Left: EENS comparison RLInv vs A7
ax = axes[0]
if "RLInv" in seed_eens and "A7" in seed_eens:
    data = [seed_eens["RLInv"], seed_eens["A7"]]
    bp = ax.boxplot(data, positions=[1, 2], widths=0.45, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2.5))
    bp["boxes"][0].set_facecolor(C["RLInv"]); bp["boxes"][0].set_alpha(0.85)
    bp["boxes"][1].set_facecolor(C["A7"]);    bp["boxes"][1].set_alpha(0.85)

    rng = np.random.default_rng(42)
    for i, (d, c) in enumerate(zip(data, [C["RLInv"], C["A7"]]), 1):
        jitter = rng.uniform(-0.1, 0.1, len(d))
        ax.scatter(i + jitter, d, color=c, alpha=0.8, s=60, zorder=5)

    p, d_val = welch_test(seed_eens["RLInv"], seed_eens["A7"])
    ymax = max(seed_eens["A7"]) + 50
    ax.annotate("", xy=(1, ymax), xytext=(2, ymax),
                arrowprops=dict(arrowstyle="-", color="black", lw=1.5))
    ax.text(1.5, ymax - 30, plab(p), ha="center", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["RLInv\n(with masking)", "A7\n(no masking)"], fontsize=9)
    ax.set_ylabel("Seed-level mean EENS (kWh)", fontsize=9)
    ax.set_title("EENS Distribution\n(3 seeds, constrained sites)", fontsize=9.5)

# Right: variance comparison bar
ax = axes[1]
if "RLInv" in seed_eens and "A7" in seed_eens:
    policies_var = ["RLInv", "B1", "A5", "A6", "TrackB", "A7"]
    policies_var = [p for p in policies_var if p in seed_eens]
    stds_var = [seed_eens[p].std(ddof=1) for p in policies_var]
    colors_var = [C[p] for p in policies_var]
    xlabs_var = [p for p in policies_var]

    bars_var = ax.bar(range(len(policies_var)), stds_var,
                      color=colors_var, alpha=0.85, width=0.6, linewidth=0)
    for i, (bar, s) in enumerate(zip(bars_var, stds_var)):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 5, f"{s:.0f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(range(len(policies_var)))
    ax.set_xticklabels(xlabs_var, fontsize=9)
    ax.set_ylabel("Std of seed-level EENS (kWh)", fontsize=9)
    ax.set_title("Outcome Variance by Policy\n(lower = more stable outcomes across seeds)",
                 fontsize=9.5)
    ax.annotate("A7 shows markedly higher\nvariance across seeds",
                xy=(policies_var.index("A7"), stds_var[policies_var.index("A7")]),
                xytext=(policies_var.index("A7") - 1.2,
                        stds_var[policies_var.index("A7")] * 0.85),
                fontsize=7.5, color=C["A7"],
                arrowprops=dict(arrowstyle="->", color=C["A7"], lw=1.2))

plt.suptitle("H2 — Action Masking and Between-Seed Outcome Stability\n"
             "Constrained sites, normal logistics, $n=3$ seeds",
             fontsize=10.5, fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/fig_ablation_h2.pdf", bbox_inches="tight")
plt.savefig("results/figures/fig_ablation_h2.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved fig_ablation_h2")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Ablation H3: inventory observation (§7.4)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating fig_ablation_h3 ...")

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Left: EENS
ax = axes[0]
if "RLInv" in seed_eens and "A5" in seed_eens:
    data = [seed_eens["RLInv"], seed_eens["A5"]]
    bp = ax.boxplot(data, positions=[1, 2], widths=0.45, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2.5))
    bp["boxes"][0].set_facecolor(C["RLInv"]); bp["boxes"][0].set_alpha(0.85)
    bp["boxes"][1].set_facecolor(C["A5"]);    bp["boxes"][1].set_alpha(0.85)

    rng = np.random.default_rng(42)
    for i, (d, c) in enumerate(zip(data, [C["RLInv"], C["A5"]]), 1):
        jitter = rng.uniform(-0.1, 0.1, len(d))
        ax.scatter(i + jitter, d, color=c, alpha=0.8, s=60, zorder=5)

    p, d_val = welch_test(seed_eens["RLInv"], seed_eens["A5"])
    ymax = max(max(seed_eens["RLInv"]), max(seed_eens["A5"])) + 60
    ax.annotate("", xy=(1, ymax), xytext=(2, ymax),
                arrowprops=dict(arrowstyle="-", color="black", lw=1.5))
    ax.text(1.5, ymax + 25, plab(p), ha="center", fontsize=8.5)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["RLInv\n(obs. inventory)", "A5\n(no inv. obs.)"], fontsize=9)
    ax.set_ylabel("Seed-level mean EENS (kWh)", fontsize=9)
    ax.set_title("EENS: With vs Without\nInventory Observation", fontsize=9.5)

# Right: diesel consumption comparison
ax = axes[1]
if "RLInv" in seed_diesel and "A5" in seed_diesel:
    data_d = [seed_diesel["RLInv"], seed_diesel["A5"]]
    bp2 = ax.boxplot(data_d, positions=[1, 2], widths=0.45, patch_artist=True,
                     medianprops=dict(color="white", linewidth=2.5))
    bp2["boxes"][0].set_facecolor(C["RLInv"]); bp2["boxes"][0].set_alpha(0.85)
    bp2["boxes"][1].set_facecolor(C["A5"]);    bp2["boxes"][1].set_alpha(0.85)

    rng2 = np.random.default_rng(42)
    for i, (d, c) in enumerate(zip(data_d, [C["RLInv"], C["A5"]]), 1):
        jitter = rng2.uniform(-0.1, 0.1, len(d))
        ax.scatter(i + jitter, d, color=c, alpha=0.8, s=60, zorder=5)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["RLInv\n(obs. inventory)", "A5\n(no inv. obs.)"], fontsize=9)
    ax.set_ylabel("Seed-level mean Diesel (kWh)", fontsize=9)
    ax.set_title("Diesel consumption:\nwith vs without inventory observation",
                 fontsize=9.5)

plt.suptitle("H3 — Effect of Inventory State Observation\n"
             "Constrained sites, normal logistics, $n=3$ seeds",
             fontsize=10.5, fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/fig_ablation_h3.pdf", bbox_inches="tight")
plt.savefig("results/figures/fig_ablation_h3.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved fig_ablation_h3")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Delayed logistics scenario (§7.5)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating fig_delayed_scenario ...")

delay_con = df[(df.lead_scenario == "delayed") & (df.site.isin(CONSTRAINED))]
seed_eens_d = {}
for pol in df.policy.unique():
    sub = delay_con[delay_con.policy == pol]
    if len(sub) > 0:
        seed_eens_d[pol] = seed_means(sub, "EENS_kWh")

shared_pols = [p for p in POLICY_ORDER
               if p in seed_eens and p in seed_eens_d]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

# Left: grouped bar normal vs delayed
ax = axes[0]
x = np.arange(len(shared_pols))
w = 0.38
means_n = [seed_eens[p].mean()   for p in shared_pols]
stds_n  = [seed_eens[p].std()    for p in shared_pols]
means_d = [seed_eens_d[p].mean() for p in shared_pols]
stds_d  = [seed_eens_d[p].std()  for p in shared_pols]
colors_s = [C[p] for p in shared_pols]

b1 = ax.bar(x - w/2, means_n, w, yerr=stds_n,
            color=colors_s, alpha=0.85, capsize=4,
            error_kw=dict(ecolor="#333333", elinewidth=1.5), linewidth=0,
            label="Normal logistics")
b2 = ax.bar(x + w/2, means_d, w, yerr=stds_d,
            color=colors_s, alpha=0.45, capsize=4, hatch="//",
            error_kw=dict(ecolor="#333333", elinewidth=1.5), linewidth=0,
            label="Delayed logistics")

ax.set_xticks(x)
ax.set_xticklabels(shared_pols, fontsize=9)
ax.set_ylabel("Mean EENS (kWh / episode)", fontsize=9)
ax.set_title("Normal vs Delayed Logistics\n(constrained sites, $n=3$ seeds)",
             fontsize=10)
# Custom legend
norm_patch  = mpatches.Patch(facecolor="#888888", alpha=0.85, label="Normal logistics")
delay_patch = mpatches.Patch(facecolor="#888888", alpha=0.45,
                              hatch="//", label="Delayed logistics")
ax.legend(handles=[norm_patch, delay_patch], fontsize=8.5)

# Right: absolute kWh increase under delay — sorted most robust → least robust (ascending delta)
ax = axes[1]
deltas_abs_raw = {p: seed_eens_d[p].mean() - seed_eens[p].mean() for p in shared_pols}
sorted_right = sorted(shared_pols, key=lambda p: deltas_abs_raw[p])  # ascending = most robust first
deltas_abs = [deltas_abs_raw[p] for p in sorted_right]
bar_colors_adj = ["#e03030" if d > 200 else "#888888" for d in deltas_abs]

hbars = ax.barh(range(len(sorted_right)), deltas_abs,
                color=bar_colors_adj, alpha=0.8, height=0.55, linewidth=0)
ax.axvline(0, color="black", linewidth=1.0)
ax.set_yticks(range(len(sorted_right)))
ax.set_yticklabels(sorted_right, fontsize=9)
ax.set_xlabel("Absolute EENS increase under delayed logistics (kWh)", fontsize=9)
ax.set_title("Policy Sensitivity to Logistics Delay\n"
             "(absolute EENS increase; lower = more robust)", fontsize=10)

for i, (bar, d) in enumerate(zip(hbars, deltas_abs)):
    ax.text(d + (8 if d >= 0 else -8), i,
            f"+{d:.0f}" if d >= 0 else f"{d:.0f}",
            ha="left" if d >= 0 else "right",
            va="center", fontsize=8.5, fontweight="bold" if d > 200 else "normal")

# No invert_yaxis — ascending = most robust at top naturally

plt.suptitle("Performance Under Logistics Stress (§7.5)\nConstrained sites, $n=3$ seeds",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/fig_delayed_scenario.pdf", bbox_inches="tight")
plt.savefig("results/figures/fig_delayed_scenario.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved fig_delayed_scenario")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Cross-site robustness (§7.6)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating fig_cross_site ...")

site_classes = {
    "site1": "Easy", "site6": "Easy", "site8": "Easy",
    "site3": "Medium", "site4": "Medium", "site7": "Medium",
    "site9": "Medium", "site10": "Medium",
    "site2": "Hard", "site5": "Hard",
}
class_order = ["Easy", "Medium", "Hard"]
class_colors = {"Easy": "#4dac26", "Medium": "#f1a340", "Hard": "#d6604d"}

norm_all = df[df.lead_scenario == "normal"]
all_sites = sorted(df.site.unique())

fig, ax = plt.subplots(figsize=(12, 5))

rlinv_by_site = []
b1_by_site    = []
site_labels   = []

for site in all_sites:
    r_sub = norm_all[(norm_all.site == site) & (norm_all.policy == "RLInv")]
    b_sub = norm_all[(norm_all.site == site) & (norm_all.policy == "B1")]
    if len(r_sub) == 0 or len(b_sub) == 0:
        continue
    rlinv_by_site.append(seed_means(r_sub).mean())
    b1_by_site.append(seed_means(b_sub).mean())
    site_labels.append(site)

x = np.arange(len(site_labels))
w = 0.38

bars_r = ax.bar(x - w/2, rlinv_by_site, w, color=C["RLInv"], alpha=0.85,
                linewidth=0, label="RLInv (proposed)")
bars_b = ax.bar(x + w/2, b1_by_site,    w, color=C["B1"],    alpha=0.85,
                linewidth=0, label="B1 (best baseline)")

# Add value labels for any bar > 5 kWh (site2, site4, site7, site5, site10)
LABEL_THRESH = 5
for bar, val in zip(bars_r, rlinv_by_site):
    if val > LABEL_THRESH:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                f"{val:.0f}", ha="center", va="bottom", fontsize=7.5,
                color=C["RLInv"], fontweight="bold")
for bar, val in zip(bars_b, b1_by_site):
    if val > LABEL_THRESH:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                f"{val:.0f}", ha="center", va="bottom", fontsize=7.5,
                color=C["B1"], fontweight="bold")

# Colour-code x-axis labels by site class
ax.set_xticks(x)
ax.set_xticklabels(site_labels, fontsize=9)
for tick, site in zip(ax.get_xticklabels(), site_labels):
    cls = site_classes.get(site, "Medium")
    tick.set_color(class_colors[cls])

ax.set_ylabel("Mean EENS (kWh / episode)", fontsize=10)
ax.set_title("Cross-Site Robustness: RLInv vs B1 (normal logistics)\n"
             "X-axis colour: green=Easy, orange=Medium, red=Hard",
             fontsize=10.5, fontweight="bold")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("results/figures/fig_cross_site.pdf", bbox_inches="tight")
plt.savefig("results/figures/fig_cross_site.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved fig_cross_site")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Hypothesis summary dot-plot (§7.6 / conclusion)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating fig_hypothesis_summary ...")

hypotheses = [
    "H1: Joint ordering\nvs A6/TrackB",
    "H3: Inventory obs.\nvs A5",
    "H2: Action masking\nvs A7"
]
comparators = ["A6", "A5", "A7"]
short_labels = ["H1", "H3", "H2"]
colors_h = [C["A6"], C["A5"], C["A7"]]

pct_diffs = []
p_vals = []
d_vals = []

for comp in comparators:
    if comp in seed_eens and "RLInv" in seed_eens:
        pct = ((seed_eens[comp].mean() - seed_eens["RLInv"].mean())
               / seed_eens["RLInv"].mean()) * 100
        p, d = welch_test(seed_eens["RLInv"], seed_eens[comp])
        pct_diffs.append(pct)
        p_vals.append(p)
        d_vals.append(d)
    else:
        pct_diffs.append(0.0)
        p_vals.append(1.0)
        d_vals.append(0.0)

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))

# ── Left panel: % reduction ───────────────────────────────────────────────────
ax = axes[0]
ypos = np.arange(len(hypotheses))

ax.barh(
    ypos,
    pct_diffs,
    color=colors_h,
    alpha=0.82,
    height=0.5,
    linewidth=0
)

ax.axvline(0, color="black", lw=1)
ax.set_yticks(ypos)
ax.set_yticklabels(hypotheses, fontsize=9)

ax.set_xlabel("EENS reduction of RLInv relative to ablation (%)", fontsize=9)
ax.set_title(
    "Effect Size: RLInv vs Each Ablation",
    fontsize=10,
    fontweight="bold"
)

for i, (v, p) in enumerate(zip(pct_diffs, p_vals)):
    sig = "*" if p < 0.05 else ""
    ax.text(
        v + 0.9,
        i,
        f"{v:.1f}%{sig}",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold" if p < 0.05 else "normal"
    )

ax.invert_yaxis()

# ── Right panel: Cohen's d + p-value ──────────────────────────────────────────
ax = axes[1]

ax.scatter(
    d_vals,
    ypos,
    color=colors_h,
    s=165,
    zorder=5
)

# Display labels beside dots using rounded display values only
for i, (d, lbl) in enumerate(zip(d_vals, short_labels)):
    d_display = Decimal(str(d)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    ax.text(
        d + 0.08,
        i,
        f"{lbl}   d = {d_display}",
        va="center",
        ha="left",
        fontsize=8.8,
        fontweight="bold",
        zorder=10
    )

x_max_d = max(d_vals) * 1.32 if d_vals else 5
ax.set_xlim(0, x_max_d)

# Conventional effect size guides
ax.axvline(0.5, color="#8f8f8f", lw=1.2, linestyle="--")
ax.axvline(0.8, color="#5f5f5f", lw=1.2, linestyle="--")

# Put guide labels below axis, spaced so they don't overlap
ax.text(
    0.5, -0.18, "medium",
    transform=ax.get_xaxis_transform(),
    ha="center", va="top",
    fontsize=8, color="#8f8f8f"
)
ax.text(
    0.8, -0.18, "large",
    transform=ax.get_xaxis_transform(),
    ha="center", va="top",
    fontsize=8, color="#5f5f5f"
)

ax.set_yticks(ypos)
ax.set_yticklabels(
    [f"{h}\n{plab(p)}" for h, p in zip(hypotheses, p_vals)],
    fontsize=8.5
)

ax.set_xlabel("Cohen's d (effect size)", fontsize=9)
ax.set_title(
    "Effect Size and Statistical Significance",
    fontsize=10,
    fontweight="bold"
)

ax.invert_yaxis()

plt.suptitle(
    "Summary of Hypothesis Tests and Effect Sizes",
    fontsize=11,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig("results/figures/fig_hypothesis_summary.pdf", bbox_inches="tight")
plt.savefig("results/figures/fig_hypothesis_summary.png", dpi=300, bbox_inches="tight")
plt.close()

print("  Saved fig_hypothesis_summary")

print("\n✓ All Chapter 7 figures saved to results/figures/")
print("  PDFs for LaTeX, PNGs for preview")
print("\nFigures generated:")
figs = [
    ("fig_eens_comparison", "§7.2 Main reliability bar chart"),
    ("fig_pareto",          "§7.3 Reliability-cost Pareto"),
    ("fig_ablation_h1",     "§7.4 H1 ordering ablation"),
    ("fig_ablation_h2",     "§7.4 H2 masking ablation"),
    ("fig_ablation_h3",     "§7.4 H3 inventory obs ablation"),
    ("fig_delayed_scenario","§7.5 Delayed logistics"),
    ("fig_cross_site",      "§7.6 Cross-site robustness"),
    ("fig_hypothesis_summary","§7.7 Hypothesis summary"),
]
for name, desc in figs:
    print(f"  {name:30s} — {desc}")
