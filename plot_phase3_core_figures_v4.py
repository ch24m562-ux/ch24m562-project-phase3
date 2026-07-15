"""
plot_phase3_core_figures_v4.py

Layout/legibility pass following PDF proof review: FIG1 (primary comparison)
and FIG H1 were both too flat in aspect ratio to read comfortably at their
LaTeX page width (3.66:1 and 3.51:1 respectively) -- made taller and bumped
font sizes ~15% throughout. H2/H3 unchanged (aspect already fine, not
flagged).

Reads: results/phase3/master_summary.csv
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
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
    "font.family": "serif", "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ═══════════════════════════════════════════════════════════════════════
# FIG 1 — Overall Policy Performance Across Delivery Scenarios
# ═══════════════════════════════════════════════════════════════════════
POLICY_ORDER = ["b0", "b1", "mpc", "trackb", "oracle_mpc", "rlinv"]

POLICY_LABEL = {
    "b0": "B0",
    "b1": "B1",
    "mpc": "MPC",
    "trackb": "TrackB",
    "oracle_mpc": "Oracle-MPC",
    "rlinv": "RLInv",
}

POLICY_DESC = {
    "b0": "B0 (threshold rule)",
    "b1": "B1 ($(s,S)$ + rule dispatch)",
    "mpc": "MPC ($H=24$, persistence)",
    "trackb": "TrackB ($(s,S)$ + learned dispatch)",
    "oracle_mpc": "Oracle-MPC (perfect forecast)",
    "rlinv": "RLInv (proposed, joint learning)",
}

BANNER = {
    "normal": "#cfe0f3",
    "delayed": "#fbe0c4",
    "monsoon": "#d3e8cf",
    "extreme": "#f2cfcf",
}

YLIMS = {
    "normal": (0, 14),
    "delayed": (0, 35),
    "monsoon": (0, 225),
    "extreme": (0, 200),
}

fig, axes = plt.subplots(
    1, 4,
    figsize=(20, 7.3),
    sharey=False
)

for ax, scen in zip(axes, SCEN_ORDER):
    sub_all = df[df["lead_scenario"] == scen]

    means, lower_err, upper_err = [], [], []

    for pol in POLICY_ORDER:
        sm = seed_means(sub_all[sub_all["policy"] == pol])
        mean, lo, hi = ci95(sm.values)

        means.append(mean)
        lower_err.append(mean - lo)
        upper_err.append(hi - mean)

    x = np.arange(len(POLICY_ORDER))
    colors = [C[p] for p in POLICY_ORDER]

    ax.yaxis.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        color="#cccccc",
        alpha=0.75,
        zorder=0,
    )
    ax.set_axisbelow(True)

    bars = ax.bar(
        x,
        means,
        width=0.72,
        yerr=[lower_err, upper_err],
        capsize=4,
        color=colors,
        alpha=0.90,
        zorder=3,
    )

    for bar, policy in zip(bars, POLICY_ORDER):
        if policy == "oracle_mpc":
            bar.set_hatch("//")
            bar.set_edgecolor("black")
            bar.set_linewidth(0.8)

    # Rotated labels prevent overlap
    ax.set_xticks(x)
    ax.set_xticklabels(
        [POLICY_LABEL[p] for p in POLICY_ORDER],
        rotation=38,
        ha="right",
        rotation_mode="anchor",
        fontsize=10.5,
    )
    ax.tick_params(axis="x", pad=3)
    ax.tick_params(axis="y", labelsize=10.5)

    ax.set_ylim(*YLIMS[scen])

    if scen == "normal":
        ax.set_ylabel("Mean EENS (kWh / episode)", fontsize=12)

    # Full-width scenario banner
    ax.add_patch(
        Rectangle(
            (0, 1.025),
            1,
            0.095,
            transform=ax.transAxes,
            facecolor=BANNER[scen],
            edgecolor="#777777",
            linewidth=0.8,
            clip_on=False,
        )
    )
    ax.text(
        0.5,
        1.072,
        SCEN_LABEL[scen],
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    # Mean-value labels
    yrange = YLIMS[scen][1] - YLIMS[scen][0]
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 0.008 * yrange,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            zorder=4,
        )

fig.suptitle(
    "Overall Policy Performance Across Delivery Scenarios\n"
    "10 sites × 10 seeds ($n=100$ per bar); 95% CI shown",
    fontsize=16,
    fontweight="bold",
    y=0.985,
)

legend_handles = []
for policy in POLICY_ORDER:
    legend_handles.append(
        Patch(
            facecolor=C[policy],
            alpha=0.90,
            hatch="//" if policy == "oracle_mpc" else None,
            edgecolor="black" if policy == "oracle_mpc" else "none",
            label=POLICY_DESC[policy],
        )
    )

# Two rows are more readable in an A4 thesis than six tiny entries in one row
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=3,
    fontsize=10,
    frameon=False,
    bbox_to_anchor=(0.5, 0.055),
    columnspacing=1.8,
    handlelength=2.2,
)

fig.text(
    0.5,
    0.018,
    "Lower is better. Error bars show 95% confidence intervals over "
    "100 paired site--seed comparisons ($n=100$ per bar).",
    ha="center",
    fontsize=9.5,
    color="#444444",
)

# Do not use tight_layout here; manual spacing is more predictable
fig.subplots_adjust(
    left=0.055,
    right=0.985,
    top=0.76,
    bottom=0.27,
    wspace=0.22,
)

plt.savefig(
    f"{OUT_DIR}/fig_eens_comparison_v4.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()
print("Saved fig_eens_comparison_v4.png")

# ═══════════════════════════════════════════════════════════════════════
# FIG 2 — H1: Contribution of Joint Inventory-Dispatch Learning
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(14, 5.3), sharey=False)
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
    ax.bar(x, means, yerr=[los, his], capsize=4,
           color=[C[p] for p in pols_h1], alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(labels_h1, fontsize=11)
    ax.set_title(SCEN_LABEL[scen], fontsize=12, fontweight="bold")
    if scen == "normal":
        ax.set_ylabel("Mean EENS (kWh)", fontsize=12)
    for i, m in enumerate(means):
        ax.text(i, m, f"{m:.2f}", ha="center", va="bottom", fontsize=9.5)

fig.suptitle("H1 --- Contribution of Joint Inventory--Dispatch Learning",
             fontsize=14.5, fontweight="bold")
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
