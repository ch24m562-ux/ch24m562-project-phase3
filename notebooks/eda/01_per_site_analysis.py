# ENHANCED Per-Site Analysis - ITU/Zindi Dataset
# Includes critical RL environment design metrics
#
# Outputs:
#  • ../../results/figures/per_site/{site}_enhanced_analysis.(pdf|png)
#  • ../../results/tables/enhanced_site_statistics.csv
#
# Notes:
#  - Conservative handling for unknown grid_available values (treated as outage)
#  - Robust to missing net_load_kwh (recomputed as load_kwh - solar_kwh)
#  - Week template integrity check + repair (reindex 0..167)
#  - Site-level try/except so one bad file doesn't kill the run

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Warning control (targeted)
# -----------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=".*Tight layout.*")

# -----------------------------
# Plot styling
# -----------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (20, 14)
plt.rcParams["font.size"] = 10

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "results" / "figures" / "per_site"
TAB_DIR = ROOT / "results" / "tables"
DATA_DIR = ROOT / "data" / "processed"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load classification
# -----------------------------
CLASSIFICATION_PATH = DATA_DIR / "site_classification.csv"
classification = pd.read_csv(CLASSIFICATION_PATH)

REQUIRED_CLASS_COLS = {"site", "difficulty", "solar_coverage_pct", "outage_pct"}
missing = REQUIRED_CLASS_COLS - set(classification.columns)
if missing:
    raise ValueError(f"site_classification.csv missing columns: {sorted(missing)}")


# -----------------------------
# Helpers
# -----------------------------
def to_bool_grid(s: pd.Series) -> pd.Series:
    """
    Normalize grid_available to boolean, handling multiple input types.
    Returns True for grid available, False for outage.
    Unknown/NaN values are treated as False (conservative: assume outage).
    """
    s_str = s.astype(str).str.strip().str.lower()
    true_vals = {"1", "1.0", "true", "t", "yes", "y"}
    false_vals = {"0", "0.0", "false", "f", "no", "n"}

    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[s_str.isin(true_vals)] = 1.0
    out[s_str.isin(false_vals)] = 0.0

    unknown_rate = out.isna().mean()
    if unknown_rate > 0.01:
        print(f"    ⚠ Warning: {unknown_rate:.1%} unknown grid_available values (treated as outage)")

    return out.fillna(0.0).astype(bool)


def compute_outage_streaks(grid_avail: np.ndarray) -> list[int]:
    """Compute lengths of consecutive outage blocks from boolean grid availability array."""
    streaks: list[int] = []
    current = 0
    for available in grid_avail:
        if not bool(available):
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    return streaks if streaks else [0]


def fmt_hours(hours: float) -> str:
    """Format hours for display, showing ∞ for infinity."""
    return "∞" if np.isinf(hours) else f"{hours:.1f}"


def _ensure_required_site_cols(df: pd.DataFrame, site_id: str) -> pd.DataFrame:
    required = {"t", "day", "hour", "load_kwh", "solar_kwh", "grid_available", "battery_capacity_kwh"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"{site_id}: missing columns: {sorted(missing_cols)}")

    # Ensure numeric for time columns
    for col in ["t", "day", "hour"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure numeric for energy columns
    for col in ["load_kwh", "solar_kwh", "battery_capacity_kwh"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # net_load_kwh is optional (compute if missing)
    if "net_load_kwh" in df.columns:
        df["net_load_kwh"] = pd.to_numeric(df["net_load_kwh"], errors="coerce")
    else:
        df["net_load_kwh"] = df["load_kwh"] - df["solar_kwh"]

    # Recompute net_load if it's mostly NaN
    if df["net_load_kwh"].isna().mean() > 0.5:
        df["net_load_kwh"] = df["load_kwh"] - df["solar_kwh"]

    return df


def analyze_site_enhanced(site_id: str) -> dict:
    """Generate per-site 5-panel analysis + return key RL stress metrics."""
    site_path = DATA_DIR / f"{site_id}.csv"
    df = pd.read_csv(site_path)
    df = _ensure_required_site_cols(df, site_id)

    site_info = classification[classification["site"] == site_id].iloc[0]

    # Battery capacity should be constant per site; if not, use median and warn.
    bc_series = df["battery_capacity_kwh"].dropna()
    if bc_series.empty:
        raise ValueError(f"{site_id}: battery_capacity_kwh is all NaN")
    battery_capacity = float(bc_series.median())
    if df["battery_capacity_kwh"].nunique(dropna=True) > 1:
        print(f"    ⚠ Warning: battery_capacity_kwh varies within {site_id}; using median={battery_capacity:.2f} kWh")

    # Normalize grid availability
    df = df.assign(grid_avail=to_bool_grid(df["grid_available"]))

    # Optional sanity check: net_load == load - solar
    net_load_check = (df["net_load_kwh"] - (df["load_kwh"] - df["solar_kwh"])).abs().mean(skipna=True)
    if pd.notna(net_load_check) and net_load_check > 1e-6:
        print(f"    ⚠ Warning: net_load != load - solar (mean abs diff {net_load_check:.3f}) for {site_id}")

    # ----------------------------------------
    # 1) Outage streaks on weekly template
    # ----------------------------------------
    week_df = df[df["t"] < 168].sort_values("t").copy()

    # Repair missing/non-continuous weekly template
    if len(week_df) != 168 or not (week_df["t"].diff().dropna() == 1).all():
        found = len(week_df)
        print(f"    ⚠ Warning: weekly template not clean for {site_id} (expected 168, found {found}); repairing by reindexing 0..167")
        week_df = week_df.set_index("t").reindex(range(168))
        week_df["grid_avail"] = week_df.get("grid_avail", False).fillna(False)  # conservative
    outage_streaks = compute_outage_streaks(week_df["grid_avail"].to_numpy())
    longest_outage = int(np.max(outage_streaks))
    outage_p95 = float(np.percentile(outage_streaks, 95))
    num_gt_6h = int(np.sum(np.array(outage_streaks) > 6))
    num_gt_12h = int(np.sum(np.array(outage_streaks) > 12))

    # ----------------------------------------
    # 2) Outage-conditioned deficit
    # ----------------------------------------
    outage_hours = df.loc[~df["grid_avail"]].copy()
    if len(outage_hours) > 0:
        deficit_outage = outage_hours["net_load_kwh"].clip(lower=0)
        mean_deficit_outage = float(deficit_outage.mean())
        p95_deficit_outage = float(deficit_outage.quantile(0.95))
        worst_deficit_outage = float(deficit_outage.max())
    else:
        mean_deficit_outage = p95_deficit_outage = worst_deficit_outage = 0.0

    # ----------------------------------------
    # 3) Battery autonomy + coverage
    # ----------------------------------------
    battery_autonomy = float("inf") if mean_deficit_outage <= 0 else float(battery_capacity / mean_deficit_outage)
    autonomy_coverage_pct = 100.0 if longest_outage <= 0 else float(100 * min(battery_autonomy / longest_outage, 1.0))

    # ----------------------------------------
    # 4) Seasonality drift (D1-30 vs D31-60)
    # ----------------------------------------
    first_30 = df[df["day"] <= 30]
    last_30 = df[df["day"] > 30]

    base_load = float(first_30["load_kwh"].mean(skipna=True))
    base_solar = float(first_30["solar_kwh"].mean(skipna=True))

    if base_load > 0 and not np.isnan(base_load):
        load_drift = float(100 * (last_30["load_kwh"].mean(skipna=True) / base_load - 1))
    else:
        load_drift = 0.0

    if base_solar > 0 and not np.isnan(base_solar):
        solar_drift = float(100 * (last_30["solar_kwh"].mean(skipna=True) / base_solar - 1))
    else:
        solar_drift = 0.0

    mean_load = float(df["load_kwh"].mean(skipna=True))
    mean_solar = float(df["solar_kwh"].mean(skipna=True))
    is_surplus = bool(mean_load > 0 and (mean_solar / mean_load) > 1.0)

    # ----------------------------------------
    # 5) Curtailment / surplus analysis
    # ----------------------------------------
    surplus_pct = float(100 * (df["net_load_kwh"] < 0).mean())

    surplus_hours = df[df["solar_kwh"] > df["load_kwh"]]
    if len(surplus_hours) > 0:
        surplus_mag = (surplus_hours["solar_kwh"] - surplus_hours["load_kwh"])
        mean_surplus_magnitude = float(surplus_mag.mean())
        max_surplus_hour = float(surplus_mag.max())

        num_days = int(df["day"].nunique())
        total_surplus_daily = float((df["solar_kwh"] - df["load_kwh"]).clip(lower=0).sum() / max(num_days, 1))
    else:
        mean_surplus_magnitude = max_surplus_hour = total_surplus_daily = 0.0

    avg_surplus_energy_daily = total_surplus_daily

    # ----------------------------------------
    # Plot panels
    # ----------------------------------------
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    title_text = f"{site_id.upper()} - Enhanced Energy & RL Stress Analysis\n"
    title_text += (
        f"Difficulty: {site_info['difficulty']} | Solar: {site_info['solar_coverage_pct']:.1f}% | "
        f"Outage: {site_info['outage_pct']:.1f}% | Battery: {battery_capacity:.1f} kWh"
    )
    fig.suptitle(title_text, fontsize=16, fontweight="bold")

    # Plot 1: Load profile
    ax = fig.add_subplot(gs[0, 0])
    hourly_load = df.groupby("hour")["load_kwh"].agg(["mean", "std"])
    ax.plot(hourly_load.index, hourly_load["mean"], linewidth=2, color="darkred", marker="o", markersize=4)
    ax.fill_between(hourly_load.index, hourly_load["mean"] - hourly_load["std"], hourly_load["mean"] + hourly_load["std"], alpha=0.3, color="red")
    ax.set_xlabel("Hour of Day", fontweight="bold")
    ax.set_ylabel("Load (kWh/h)", fontweight="bold")
    ax.set_title("Load Profile (Mean ± Std)", fontweight="bold")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)
    ax.text(
        0.02, 0.98,
        f"Mean: {hourly_load['mean'].mean():.2f}\nPeak: {hourly_load['mean'].max():.2f} @ h{int(hourly_load['mean'].idxmax())}",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        verticalalignment="top",
        fontsize=9,
    )

    # Plot 2: Solar profile
    ax = fig.add_subplot(gs[0, 1])
    hourly_solar = df.groupby("hour")["solar_kwh"].agg(["mean", "std"])
    ax.plot(hourly_solar.index, hourly_solar["mean"], linewidth=2, color="darkorange", marker="o", markersize=4)
    ax.fill_between(hourly_solar.index, hourly_solar["mean"] - hourly_solar["std"], hourly_solar["mean"] + hourly_solar["std"], alpha=0.3, color="orange")
    ax.set_xlabel("Hour of Day", fontweight="bold")
    ax.set_ylabel("Solar (kWh/h)", fontweight="bold")
    ax.set_title("Solar Profile (Mean ± Std)", fontweight="bold")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)
    ax.text(
        0.02, 0.98,
        f"Mean: {hourly_solar['mean'].mean():.2f}\nPeak: {hourly_solar['mean'].max():.2f} @ h{int(hourly_solar['mean'].idxmax())}",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        verticalalignment="top",
        fontsize=9,
    )

    # Plot 3: Outage profile
    ax = fig.add_subplot(gs[1, 0])
    df_plot = df.assign(hour_of_week=df["t"] % 168)
    outage_prob = 1 - df_plot.groupby("hour_of_week")["grid_avail"].mean()
    ax.bar(outage_prob.index, outage_prob.values, color="steelblue", alpha=0.7, width=1.0)
    for day in range(1, 7):
        ax.axvline(day * 24, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Hour of Week", fontweight="bold")
    ax.set_ylabel("Outage Probability", fontweight="bold")
    ax.set_title("Grid Outage Profile (Weekly Pattern)", fontweight="bold")
    ax.set_xlim([0, 168])
    ax.set_ylim([0, 1.1])
    ax.grid(axis="y", alpha=0.3)
    for i, day in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        ax.text(i * 24 + 12, 1.05, day, ha="center", fontsize=8, fontweight="bold")

    # Plot 4: Net load histogram
    ax = fig.add_subplot(gs[1, 1])
    net_load = df["net_load_kwh"].dropna()
    counts, bins, patches = ax.hist(net_load, bins=50, edgecolor="black", alpha=0.7)
    for i, patch in enumerate(patches):
        patch.set_facecolor("green" if bins[i] < 0 else "red")
    ax.axvline(0, color="black", linestyle="--", linewidth=2)
    ax.set_xlabel("Net Load (kWh/h)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Net Load Distribution", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    box_color = "lightgreen" if surplus_pct > 50 else "lightcoral"
    ax.text(
        0.98, 0.98,
        f"Surplus: {surplus_pct:.1f}%\nDeficit: {100 - surplus_pct:.1f}%",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.5),
        va="top", ha="right",
        fontsize=9,
    )

    # Plot 5: Outage duration distribution + autonomy
    ax = fig.add_subplot(gs[2, :])
    if len(outage_streaks) > 1 and longest_outage > 0:
        ax.hist(outage_streaks, bins=range(1, max(outage_streaks) + 2), color="darkblue", alpha=0.7, edgecolor="black")

    if longest_outage > 0:
        ax.axvline(6, color="orange", linestyle="--", linewidth=2, label="6h threshold")
        ax.axvline(12, color="red", linestyle="--", linewidth=2, label="12h threshold")
        ax.axvline(longest_outage, color="darkred", linestyle="-", linewidth=3, label=f"Longest: {longest_outage}h")
        if np.isfinite(battery_autonomy):
            ax.axvline(battery_autonomy, color="green", linestyle=":", linewidth=3, label=f"Battery Autonomy: {fmt_hours(battery_autonomy)}h")
        ax.legend(loc="upper right")
    else:
        ax.text(
            0.5, 0.5,
            "NO OUTAGES IN WEEKLY TEMPLATE\n(100% Grid Availability)",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=14, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
        )

    ax.set_xlabel("Consecutive Outage Duration (hours)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Outage Duration Distribution & Battery Adequacy Analysis", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    stress_stats = f"""CRITICAL RL STRESS METRICS:

Longest Outage: {longest_outage} hours
95th Percentile: {outage_p95:.1f} hours
Outages > 6h: {num_gt_6h}
Outages > 12h: {num_gt_12h}

OUTAGE-CONDITIONED DEFICIT:
Mean Deficit: {mean_deficit_outage:.2f} kWh/h
95th Percentile: {p95_deficit_outage:.2f} kWh/h
Worst Hour: {worst_deficit_outage:.2f} kWh/h

BATTERY AUTONOMY:
Capacity: {battery_capacity:.1f} kWh
Autonomy: {fmt_hours(battery_autonomy)} hours
Coverage: {autonomy_coverage_pct:.0f}% of longest outage

SEASONALITY CHECK:
Load Drift (D1-30 vs D31-60): {load_drift:+.1f}%
Solar Drift (D1-30 vs D31-60): {solar_drift:+.1f}%
"""
    ax.text(
        0.98, 0.97,
        stress_stats,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=9,
        family="monospace",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_pdf = FIG_DIR / f"{site_id}_enhanced_analysis.pdf"
    out_png = FIG_DIR / f"{site_id}_enhanced_analysis.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out_pdf}")

    return {
        "site": site_id,
        "mean_load": float(hourly_load["mean"].mean()),
        "peak_load": float(hourly_load["mean"].max()),
        "mean_solar": float(hourly_solar["mean"].mean()),
        "peak_solar": float(hourly_solar["mean"].max()),
        "outage_pct": float((1 - df["grid_avail"].mean()) * 100),
        "longest_outage_hours": longest_outage,
        "outage_p95_hours": outage_p95,
        "num_outages_gt_6h": num_gt_6h,
        "num_outages_gt_12h": num_gt_12h,
        "mean_deficit_during_outage": mean_deficit_outage,
        "p95_deficit_during_outage": p95_deficit_outage,
        "worst_deficit_during_outage": worst_deficit_outage,
        "battery_capacity_kwh": battery_capacity,
        "battery_autonomy_hours": battery_autonomy,
        "autonomy_coverage_pct": autonomy_coverage_pct,
        "surplus_hours_pct": surplus_pct,
        "mean_surplus_magnitude_kwh": mean_surplus_magnitude,
        "max_surplus_hour_kwh": max_surplus_hour,
        "avg_surplus_energy_daily_kwh": avg_surplus_energy_daily,
        "load_drift_30day_pct": load_drift,
        "solar_drift_30day_pct": solar_drift,
        "is_surplus": is_surplus,
    }


# -----------------------------
# Run analysis (batch)
# -----------------------------
print("\n" + "=" * 80)
print("ENHANCED PER-SITE ANALYSIS - RL STRESS METRICS")
print("=" * 80 + "\n")

all_stats: list[dict] = []
failed: list[tuple[str, str]] = []

for site_id in sorted(classification["site"].astype(str).tolist()):
    print(f"\nAnalyzing {site_id}...")
    try:
        stats = analyze_site_enhanced(site_id)
        all_stats.append(stats)
    except Exception as e:
        failed.append((site_id, str(e)))
        print(f"    ✗ Failed {site_id}: {e}")

stats_df = pd.DataFrame(all_stats).sort_values("site")
out_csv = TAB_DIR / "enhanced_site_statistics.csv"
stats_df.to_csv(out_csv, index=False)

print("\n" + "=" * 80)
print("ENHANCED STATISTICS TABLE (subset)")
print("=" * 80)
if not stats_df.empty:
    print(stats_df[["site", "longest_outage_hours", "battery_autonomy_hours", "autonomy_coverage_pct", "mean_deficit_during_outage"]].to_string(index=False))
print(f"\n✓ Saved: {out_csv}")

print("\n" + "=" * 80)
print("CRITICAL FINDINGS FOR RL ENVIRONMENT")
print("=" * 80)

if not stats_df.empty:
    inadequate = stats_df[stats_df["autonomy_coverage_pct"] < 50]
    print("\n🔴 Sites with INADEQUATE battery autonomy (<50% of longest outage):")
    if len(inadequate) == 0:
        print("  • None")
    for _, row in inadequate.iterrows():
        print(f"  • {row['site']}: {fmt_hours(row['battery_autonomy_hours'])}h autonomy vs {int(row['longest_outage_hours'])}h longest outage")
        print(f"    → DIESEL CRITICAL (coverage: {row['autonomy_coverage_pct']:.0f}%)")

    high_stress = stats_df[stats_df["mean_deficit_during_outage"] > 5]
    print("\n⚠️  High-stress sites (mean deficit during outage > 5 kWh/h):")
    if len(high_stress) == 0:
        print("  • None")
    for _, row in high_stress.iterrows():
        print(f"  • {row['site']}: {row['mean_deficit_during_outage']:.2f} kWh/h (95th: {row['p95_deficit_during_outage']:.2f})")

    surplus_sites = stats_df[stats_df["surplus_hours_pct"] > 30]
    print("\n🌞 Surplus sites (>30% hours with solar > load):")
    if len(surplus_sites) == 0:
        print("  • None")
    for _, row in surplus_sites.iterrows():
        print(f"  • {row['site']}: {row['surplus_hours_pct']:.1f}% surplus hours")
        print(f"    Mean surplus: {row['mean_surplus_magnitude_kwh']:.2f} kWh/h")
        print(f"    Avg surplus energy: {row['avg_surplus_energy_daily_kwh']:.1f} kWh/day")
        if bool(row["is_surplus"]):
            print("    → Tests OVERFLOW REGIME handling")

    drifting = stats_df[(stats_df["load_drift_30day_pct"].abs() > 5) | (stats_df["solar_drift_30day_pct"].abs() > 5)]
    if len(drifting) > 0:
        print("\n📊 Sites with seasonality drift (>5% change):")
        for _, row in drifting.iterrows():
            print(f"  • {row['site']}: Load {row['load_drift_30day_pct']:+.1f}%, Solar {row['solar_drift_30day_pct']:+.1f}%")
    else:
        print("\n✓ All sites stationary (drift <5%) - good for RL generalization")

if failed:
    print("\n" + "-" * 80)
    print("FAILED SITES (fix these and re-run):")
    for site, err in failed:
        print(f"  • {site}: {err}")

print("\n" + "=" * 80)
print("✅ ENHANCED PER-SITE ANALYSIS COMPLETE")
print("=" * 80)
print("\nOutput:")
print("  • Per-site enhanced PDFs/PNGs")
print("  • enhanced_site_statistics.csv with all RL metrics")
print("\nNext: Run 02_cross_site_comparison_enhanced.py")