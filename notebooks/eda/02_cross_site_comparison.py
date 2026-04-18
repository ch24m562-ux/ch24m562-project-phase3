# ENHANCED Cross-Site Comparison - Including RL Stress Analysis
# Includes Figure 4: Battery Adequacy, Outage Stress & Surplus Regime
#
# Inputs:
#  • ../../data/processed/site_classification.csv
#  • ../../results/tables/enhanced_site_statistics.csv
#
# Outputs:
#  • ../../results/figures/comparison/fig1_site_overview.(pdf|png)
#  • ../../results/figures/comparison/fig2_training_sites.(pdf|png)
#  • ../../results/figures/comparison/fig3_site8_surplus.(pdf|png)   (if site8 exists)
#  • ../../results/figures/comparison/fig4_rl_stress_analysis.(pdf|png)
#  • ../../results/tables/thesis_rl_stress_table.csv

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# -----------------------------
# Warning control (targeted)
# -----------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=".*Tight layout.*")

plt.style.use("seaborn-v0_8-whitegrid")
ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"
DATA_DIR = ROOT / "data" / "processed"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
# -----------------------------
# Load Data
# -----------------------------
classification = pd.read_csv(DATA_DIR / "site_classification.csv")
stats = pd.read_csv(TAB_DIR / "enhanced_site_statistics.csv")

classification = classification[['site', 'difficulty', 'solar_coverage_pct', 'difficulty_score']]

if "site" not in classification.columns or "site" not in stats.columns:
    raise ValueError("Both site_classification.csv and enhanced_site_statistics.csv must contain 'site' column")

df = pd.merge(classification, stats, on="site", how="inner")
if df.empty:
    raise ValueError("Merge resulted in empty dataframe. Check that 'site' keys match across both CSVs.")

# Keep your training sites selection
training_sites = ["site1", "site5", "site7"]
df["is_training"] = df["site"].astype(str).isin(training_sites)

difficulty_colors = {"Easy": "green", "Medium": "orange", "Hard": "red"}

df["color"] = df["difficulty"].map(difficulty_colors).fillna("gray")

print("Loaded data for", len(df), "sites")


def size_safe(x: pd.Series, scale: float, min_size: float = 40.0) -> np.ndarray:
    """Convert a numeric series to safe scatter sizes with a minimum."""
    arr = pd.to_numeric(x, errors="coerce").fillna(0).to_numpy()
    arr = np.maximum(arr, 0)
    return (arr * scale) + min_size


# =================================
# FIGURE 1: Site Overview Dashboard
# =================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Site Overview Dashboard", fontsize=16, fontweight="bold")

# Load vs Solar
ax = axes[0, 0]
ax.scatter(
    df["mean_load"], df["mean_solar"],
    s=size_safe(df["outage_pct"], scale=10, min_size=60),
    c=df["color"], alpha=0.7,
    edgecolors="black", linewidth=1.2,
)
for _, row in df.iterrows():
    ax.annotate(str(row["site"]), (row["mean_load"], row["mean_solar"]), xytext=(5, 5), textcoords="offset points", fontsize=8)

max_val = float(np.nanmax([df["mean_load"].max(), df["mean_solar"].max()]))
ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, linewidth=2)
ax.set_xlabel("Mean Load (kWh/h)", fontweight="bold")
ax.set_ylabel("Mean Solar (kWh/h)", fontweight="bold")
ax.set_title("Load vs Solar (bubble = outage %)", fontweight="bold")
ax.grid(alpha=0.3)

# Outage Severity
ax = axes[0, 1]
df_sorted = df.sort_values("outage_pct", ascending=False)
ax.barh(df_sorted["site"], df_sorted["outage_pct"], color=df_sorted["color"], alpha=0.7)
ax.set_xlabel("Grid Outage (%)", fontweight="bold")
ax.set_title("Grid Reliability Comparison", fontweight="bold")
ax.grid(axis="x", alpha=0.3)

# Solar Coverage Distribution
ax = axes[1, 0]
for difficulty in ["Easy", "Medium", "Hard"]:
    subset = df[df["difficulty"] == difficulty]
    if subset.empty:
        continue
    ax.hist(subset["solar_coverage_pct"], bins=10, alpha=0.6, color=difficulty_colors[difficulty], label=difficulty, edgecolor="black", linewidth=1)
ax.axvline(100, color="black", linestyle="--", linewidth=2, label="100%")
ax.set_xlabel("Solar Coverage (%)", fontweight="bold")
ax.set_ylabel("Sites", fontweight="bold")
ax.set_title("Solar Coverage Distribution", fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Difficulty Map
ax = axes[1, 1]
ax.scatter(
    df["solar_coverage_pct"], df["outage_pct"],
    s=size_safe(df["mean_load"], scale=30, min_size=60),
    c=df["color"], alpha=0.7,
    edgecolors="black", linewidth=1.2,
)
for _, row in df.iterrows():
    ax.annotate(str(row["site"]), (row["solar_coverage_pct"], row["outage_pct"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
ax.axvline(50, color="gray", linestyle="--", alpha=0.5)
ax.axhline(30, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Solar Coverage (%)", fontweight="bold")
ax.set_ylabel("Grid Outage (%)", fontweight="bold")
ax.set_title("Difficulty Map (bubble = load)", fontweight="bold")
ax.grid(alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG_DIR / "fig1_site_overview.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig1_site_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ Saved: Figure 1 - Site Overview")

# =======================
# FIGURE 2: Training Sites
# =======================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Training Site Selection - Diversity Analysis", fontsize=16, fontweight="bold")

train_df = df[df["is_training"]].sort_values("mean_load")

# Load
ax = axes[0]
ax.barh(range(len(train_df)), train_df["mean_load"], color="steelblue", alpha=0.8)
for i, row in enumerate(train_df.itertuples()):
    ax.text(float(row.mean_load) + 0.2, i, str(row.site), va="center", fontweight="bold")
ax.set_yticks([])
ax.set_xlabel("Mean Load (kWh/h)", fontweight="bold")
ax.set_title("Load Range", fontweight="bold")
ax.grid(axis="x", alpha=0.3)

# Solar Coverage
ax = axes[1]
ax.barh(range(len(train_df)), train_df["solar_coverage_pct"], color="orange", alpha=0.8)
for i, row in enumerate(train_df.itertuples()):
    ax.text(float(row.solar_coverage_pct) + 2, i, str(row.site), va="center", fontweight="bold")
ax.axvline(100, color="black", linestyle="--", linewidth=2)
ax.set_yticks([])
ax.set_xlabel("Solar Coverage (%)", fontweight="bold")
ax.set_title("Solar Coverage", fontweight="bold")
ax.grid(axis="x", alpha=0.3)

# Outage
ax = axes[2]
ax.barh(range(len(train_df)), train_df["outage_pct"], color="red", alpha=0.8)
for i, row in enumerate(train_df.itertuples()):
    ax.text(float(row.outage_pct) + 1, i, str(row.site), va="center", fontweight="bold")
ax.set_yticks([])
ax.set_xlabel("Grid Outage (%)", fontweight="bold")
ax.set_title("Grid Reliability", fontweight="bold")
ax.grid(axis="x", alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG_DIR / "fig2_training_sites.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig2_training_sites.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ Saved: Figure 2 - Training Sites")

# =====================================
# FIGURE 3: Site8 Surplus (if available)
# =====================================
if (DATA_DIR / "site8.csv").exists() and (df["site"] == "site8").any():
    site8 = pd.read_csv(DATA_DIR / "site8.csv")
    site8_info = df[df["site"] == "site8"].iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Site8 - Renewable-Surplus Case Study", fontsize=16, fontweight="bold")

    # Week 1 pattern
    ax = axes[0, 0]
    week1 = site8[site8["day"] <= 7].copy()
    week1["hour_cont"] = (week1["day"] - 1) * 24 + week1["hour"]
    ax.plot(week1["hour_cont"], week1["load_kwh"], label="Load", linewidth=2, color="darkred")
    ax.plot(week1["hour_cont"], week1["solar_kwh"], label="Solar", linewidth=2, color="orange")
    ax.fill_between(week1["hour_cont"], week1["load_kwh"], week1["solar_kwh"],
                    where=(week1["solar_kwh"] >= week1["load_kwh"]),
                    alpha=0.3, label="Surplus")
    ax.set_xlabel("Hour (Week 1)", fontweight="bold")
    ax.set_ylabel("Energy (kWh/h)", fontweight="bold")
    ax.set_title("Load vs Solar Pattern", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Distribution (prefer net_load if present)
    ax = axes[0, 1]
    if "net_load_kwh" in site8.columns:
        bal = pd.to_numeric(site8["net_load_kwh"], errors="coerce")
    else:
        bal = pd.to_numeric(site8["load_kwh"], errors="coerce") - pd.to_numeric(site8["solar_kwh"], errors="coerce")
    surplus = (-bal).clip(lower=0).dropna()
    deficit = (bal).clip(lower=0).dropna()
    ax.hist(surplus, bins=30, alpha=0.6, label=f"Surplus ({len(surplus)}h)")
    ax.hist(-deficit, bins=30, alpha=0.6, label=f"Deficit ({len(deficit)}h)")
    ax.axvline(0, color="black", linestyle="--", linewidth=2)
    ax.set_xlabel("Energy Balance (kWh/h)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Surplus/Deficit Distribution", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Hourly average
    ax = axes[1, 0]
    hourly = site8.groupby("hour")[["load_kwh", "solar_kwh"]].mean(numeric_only=True)
    ax.plot(hourly.index, hourly["load_kwh"], label="Load", linewidth=3, color="darkred", marker="o")
    ax.plot(hourly.index, hourly["solar_kwh"], label="Solar", linewidth=3, color="orange", marker="s")
    ax.fill_between(hourly.index, hourly["load_kwh"], hourly["solar_kwh"], where=(hourly["solar_kwh"] >= hourly["load_kwh"]), alpha=0.3)
    ax.set_xlabel("Hour of Day", fontweight="bold")
    ax.set_ylabel("Energy (kWh/h)", fontweight="bold")
    ax.set_title("Mean Daily Pattern", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 24, 2))

    # Implications panel
    ax = axes[1, 1]
    implications = f"""SITE8 RL IMPLICATIONS:

Load:  {site8_info['mean_load']:.2f} kWh/h
Solar: {site8_info['mean_solar']:.2f} kWh/h
Coverage: {site8_info['solar_coverage_pct']:.1f}%

Battery: {site8_info['battery_capacity_kwh']:.1f} kWh
Autonomy: {site8_info['battery_autonomy_hours']:.1f}h

EXPECTED BEHAVIOR:
• Battery saturates frequently
• Excess solar curtailed
• Diesel near-zero usage
• Challenge: night + outage overlap

STRESS TEST VALUE:
Tests policy robustness under
energy-surplus conditions
"""
    ax.text(0.5, 0.5, implications, transform=ax.transAxes,
            fontsize=11, va="center", ha="center",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
            family="monospace")
    ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "fig3_site8_surplus.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_site8_surplus.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ Saved: Figure 3 - Site8 Surplus")
else:
    print("↷ Skipping Figure 3 (site8 data not found or site8 not in merged table).")

# =================================
# FIGURE 4: RL Stress Analysis
# =================================
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
fig.suptitle("Figure 4: RL Stress Analysis - Battery Adequacy, Outage Severity & Surplus Regime", fontsize=16, fontweight="bold")

# Battery Autonomy vs Outage %
ax = fig.add_subplot(gs[0, 0])
autonomy_capped = pd.to_numeric(df["battery_autonomy_hours"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(50).clip(upper=50)
ax.scatter(df["outage_pct"], autonomy_capped, s=size_safe(df["mean_load"], 30, 60), c=df["color"], alpha=0.7, edgecolors="black", linewidth=1.2)
for _, row in df.iterrows():
    y = float(min(row["battery_autonomy_hours"], 50)) if np.isfinite(row["battery_autonomy_hours"]) else 50.0
    ax.annotate(str(row["site"]), (row["outage_pct"], y), xytext=(5, 5), textcoords="offset points", fontsize=8)
ax.axhline(6, color="orange", linestyle="--", linewidth=2, label="6h threshold")
ax.axhline(12, color="red", linestyle="--", linewidth=2, label="12h threshold")
ax.set_xlabel("Grid Outage (%)", fontweight="bold")
ax.set_ylabel("Battery Autonomy (hours, capped at 50)", fontweight="bold")
ax.set_title("Battery Adequacy vs Grid Reliability", fontweight="bold")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)

# Longest Outage vs Battery Capacity
ax = fig.add_subplot(gs[0, 1])
ax.scatter(df["battery_capacity_kwh"], df["longest_outage_hours"], s=size_safe(df["mean_load"], 30, 60), c=df["color"], alpha=0.7, edgecolors="black", linewidth=1.2)
for _, row in df.iterrows():
    ax.annotate(str(row["site"]), (row["battery_capacity_kwh"], row["longest_outage_hours"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
ax.set_xlabel("Battery Capacity (kWh)", fontweight="bold")
ax.set_ylabel("Longest Outage (hours)", fontweight="bold")
ax.set_title("Battery Sizing vs Worst-Case Outage", fontweight="bold")
ax.grid(alpha=0.3)

# Deficit during outage (mean vs p95)
ax = fig.add_subplot(gs[1, 0])
for difficulty in ["Easy", "Medium", "Hard"]:
    subset = df[df["difficulty"] == difficulty]
    if subset.empty:
        continue
    ax.scatter(subset["mean_deficit_during_outage"], subset["p95_deficit_during_outage"],
               s=150, c=difficulty_colors[difficulty], alpha=0.7,
               edgecolors="black", linewidth=1.2, label=difficulty)
    for _, row in subset.iterrows():
        ax.annotate(str(row["site"]), (row["mean_deficit_during_outage"], row["p95_deficit_during_outage"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
max_val = float(np.nanmax([df["p95_deficit_during_outage"].max(), df["mean_deficit_during_outage"].max()]))
ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, linewidth=2)
ax.set_xlabel("Mean Deficit During Outage (kWh/h)", fontweight="bold")
ax.set_ylabel("95th Percentile Deficit During Outage (kWh/h)", fontweight="bold")
ax.set_title("Outage Stress Severity by Difficulty", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)

# Surplus regime
ax = fig.add_subplot(gs[1, 1])
bubble = size_safe(df["avg_surplus_energy_daily_kwh"], 10, 80)
ax.scatter(df["solar_coverage_pct"], df["surplus_hours_pct"], s=bubble, c=df["color"], alpha=0.7, edgecolors="black", linewidth=1.2)
for _, row in df.iterrows():
    ax.annotate(str(row["site"]), (row["solar_coverage_pct"], row["surplus_hours_pct"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
ax.axhline(30, color="green", linestyle="--", linewidth=2, alpha=0.5, label="30% surplus threshold")
ax.axvline(100, color="blue", linestyle="--", linewidth=2, alpha=0.5, label="100% coverage")
ax.set_xlabel("Solar Coverage (%)", fontweight="bold")
ax.set_ylabel("Surplus Hours (%)", fontweight="bold")
ax.set_title("Surplus Regime Analysis\n(bubble size = surplus energy potential)", fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)

if (df["site"] == "site8").any():
    site8_row = df[df["site"] == "site8"].iloc[0]
    ax.annotate(
        f"OVERFLOW REGIME\n{site8_row['avg_surplus_energy_daily_kwh']:.1f} kWh/day avg surplus",
        xy=(site8_row["solar_coverage_pct"], site8_row["surplus_hours_pct"]),
        xytext=(120, 80),
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
        arrowprops=dict(arrowstyle="->", lw=2, color="darkgreen"),
    )

# Summary text panel
ax = fig.add_subplot(gs[2, :])

summary_text = """RL STRESS ANALYSIS - SCARCITY & SURPLUS REGIMES:
═══════════════════════════════════════════════════════════════════════

SCARCITY REGIME (Battery Inadequacy):          │  SURPLUS REGIME (Overflow Potential):
──────────────────────────────────────────────  │  ──────────────────────────────────────────
"""

inadequate = df[df["autonomy_coverage_pct"] < 50].sort_values("autonomy_coverage_pct")
surplus_sites = df[df["surplus_hours_pct"] > 30].sort_values("surplus_hours_pct", ascending=False)

max_rows = int(max(len(inadequate), len(surplus_sites)))

for i in range(max_rows):
    if i < len(inadequate):
        r = inadequate.iloc[i]
        left = f"{r['site']:6s}: {r['battery_autonomy_hours']:>5.1f}h / {r['longest_outage_hours']:>3.0f}h = {r['autonomy_coverage_pct']:>3.0f}%"
    else:
        left = " " * 45

    if i < len(surplus_sites):
        r = surplus_sites.iloc[i]
        right = f"{r['site']:6s}: {r['surplus_hours_pct']:>5.1f}% hrs surplus, {r['avg_surplus_energy_daily_kwh']:>6.1f} kWh/day avg surplus"
    else:
        right = ""

    summary_text += f"\n{left:45s} │  {right}"

summary_text += "\n\nDEFICIT STRESS (During Outages):               │  INVENTORY ANALOGY MAPPING:"
summary_text += "\n──────────────────────────────────────────────  │  ──────────────────────────────────────────"

high_stress = df[df["mean_deficit_during_outage"] > 5].sort_values("mean_deficit_during_outage", ascending=False)
for i, (_, r) in enumerate(high_stress.iterrows()):
    left = f"{r['site']:6s}: {r['mean_deficit_during_outage']:>6.2f} kWh/h mean (p95: {r['p95_deficit_during_outage']:.2f})"
    if i == 0:
        right = "Scarcity Regime  ↔ Stockout risk (diesel needed)"
    elif i == 1:
        right = "Surplus Regime   ↔ Overflow potential (curtailment depends on SoC)"
    elif i == 2:
        right = "Battery Capacity ↔ Inventory capacity constraint"
    else:
        right = ""
    summary_text += f"\n{left:45s} │  {right}"

summary_text += "\n\nKEY RL INSIGHTS:"
summary_text += "\n  • Battery sizing inadequate for worst-case (diesel CRITICAL for most sites)"
summary_text += "\n  • Site8 tests overflow regime handling (battery saturation + surplus energy management)"
summary_text += "\n  • Policy must handle BOTH scarcity and surplus regimes"
summary_text += "\n  • Inventory analogy: stockout (deficit) + overflow (surplus) + capacity constraints"

ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
        family="monospace")
ax.axis("off")

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG_DIR / "fig4_rl_stress_analysis.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIG_DIR / "fig4_rl_stress_analysis.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ Saved: Figure 4 - RL Stress Analysis with Surplus Regime (ENHANCED)")

# =======================
# Thesis Table
# =======================
thesis_table = df[
    [
        "site", "difficulty", "mean_load", "solar_coverage_pct", "outage_pct",
        "longest_outage_hours", "battery_autonomy_hours", "autonomy_coverage_pct",
        "mean_deficit_during_outage", "surplus_hours_pct", "avg_surplus_energy_daily_kwh",
        "is_training", "is_surplus"
    ]
].copy().round(1)

out_table = TAB_DIR / "thesis_rl_stress_table.csv"
thesis_table.to_csv(out_table, index=False)

print("\n" + "=" * 80)
print("THESIS TABLE - RL STRESS & REGIME METRICS")
print("=" * 80)
print(thesis_table.to_string(index=False))
print(f"\n✓ Saved: {out_table}")

print("\n" + "=" * 80)
print("✅ ENHANCED CROSS-SITE COMPARISON COMPLETE")
print("=" * 80)