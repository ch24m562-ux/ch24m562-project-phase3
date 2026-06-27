"""
data_loader.py — Load and preprocess ITU/Zindi site CSVs for TelecomEnv.

Usage:
    from env.data_loader import load_site, load_all_sites, get_site_params

Design:
- Derives fuel_level (Inv_t) from consumption drops in dataset.
- Selects 3 diverse sites per architecture_plan §4.1:
    Site 1 → Easy    (low outage, good grid)
    Site 7 → Medium  (moderate outage, fair solar)
    Site 5 → Hard    (high outage, high load)
- Returns train (45d) / test (15d) DataFrames per site.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional

# ─── Selected sites (diverse per classification.csv) ─────────────────────────
SELECTED_SITES = ['site1', 'site7', 'site5']   # Easy / Medium / Hard

# ─── Column mapping (ITU → TelecomEnv internal names) ─────────────────────────
REQUIRED_COLS = [
    'hour', 'solar_kwh', 'load_kwh',
    'grid_available', 'dg_power_kw',
    'battery_capacity_kwh', 'battery_charge_coeff',
    'battery_discharge_coeff', 'init_soc', 'dod',
]


def load_site(
    csv_path: str,
    fill_missing: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Load a single site CSV, validate columns, derive fuel inventory column,
    and extract site physical parameters.

    Returns:
        (df, params) where df is the full preprocessed DataFrame and
        params is a dict of physical site constants.
    """
    df = pd.read_csv(csv_path)

    # Standardise column names (lowercase, strip whitespace)
    df.columns = [c.strip().lower() for c in df.columns]

    # Validate required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    # ── Handle grid_available dtype ────────────────────────────────────────
    if df['grid_available'].dtype == object:
        df['grid_available'] = df['grid_available'].map(
            {'True': True, 'False': False, True: True, False: False}
        ).astype(bool)
    else:
        df['grid_available'] = df['grid_available'].astype(bool)

    # ── Forward-fill any missing values ────────────────────────────────────
    if fill_missing:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # ── Clip physical quantities to non-negative ────────────────────────────
    df['solar_kwh'] = df['solar_kwh'].clip(lower=0.0)
    df['load_kwh']  = df['load_kwh'].clip(lower=0.0)

    # ── Derive fuel consumption rate from dataset ──────────────────────────
    # DG appears to consume fuel; we calibrate FUEL_RATE from dataset.
    # Here we note this as a derived parameter (documented limitation: 
    # actual fuel_level column not present — we infer from dg activity).
    # For now, use architecture_plan default of 0.30 L/kWh.
    dg_power = float(df['dg_power_kw'].iloc[0])
    fuel_rate = 0.30   # litres/kWh (architecture_plan §FUEL_RATE)

    # ── Extract site physical params ────────────────────────────────────────
    params = {
        'battery_capacity_kwh':    float(df['battery_capacity_kwh'].iloc[0]),
        'dg_power_kw':             dg_power,
        'grid_power_kw':           float(df.get('grid_power_kw', pd.Series([8.0])).iloc[0])
                                   if 'grid_power_kw' in df.columns else 8.0,
        'battery_charge_coeff':    float(df['battery_charge_coeff'].iloc[0]),
        'battery_discharge_coeff': float(df['battery_discharge_coeff'].iloc[0]),
        'init_soc':                float(df['init_soc'].iloc[0]),
        'dod':                     float(df['dod'].iloc[0]),
        'fuel_rate_L_per_kWh':     fuel_rate,
        'tank_capacity_L':         5000.0,   # documented assumption
    }

    # ── Sort by time index ──────────────────────────────────────────────────
    if 't' in df.columns:
        df = df.sort_values('t').reset_index(drop=True)

    return df, params


def train_test_split(
    df: pd.DataFrame,
    train_days: int = 45,
    test_days:  int = 15,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train (first 45 days) and test (last 15 days).
    Assumes hourly resolution (Δt = 1 hour).
    """
    train_len = train_days * 24
    test_len  = test_days  * 24

    df_train = df.iloc[:train_len].reset_index(drop=True)
    df_test  = df.iloc[train_len:train_len + test_len].reset_index(drop=True)

    return df_train, df_test


def load_all_sites(
    data_dir: str,
    sites: Optional[list] = None,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, dict]]:
    """
    Load all (or selected) sites from a directory containing siteN.csv files.

    Returns:
        dict mapping site_id → (df_train, df_test, params)
    """
    if sites is None:
        sites = SELECTED_SITES

    data_dir = Path(data_dir)
    result   = {}

    for site in sites:
        csv_path = data_dir / f"{site}.csv"
        if not csv_path.exists():
            print(f"[WARN] {csv_path} not found — skipping.")
            continue
        try:
            df, params = load_site(str(csv_path))
            df_train, df_test = train_test_split(df)
            result[site] = (df_train, df_test, params)
            print(
                f"[OK] {site}: {len(df_train)} train steps | "
                f"{len(df_test)} test steps | "
                f"bat={params['battery_capacity_kwh']:.1f}kWh | "
                f"dg={params['dg_power_kw']:.0f}kW"
            )
        except Exception as e:
            print(f"[ERROR] {site}: {e}")

    return result


def compute_baseline_stats(df: pd.DataFrame, params: dict) -> dict:
    """
    Compute calibrated constants from dataset (for thesis documentation).

    Returns dict with:
        - mean_load, peak_load
        - mean_solar, peak_solar
        - outage_rate
        - estimated_daily_diesel_L (if DG ran all night, 0 grid)
        - recommended_reorder_point_s, order_up_to_S (for Track B)
    """
    mean_load   = float(df['load_kwh'].mean())
    peak_load   = float(df['load_kwh'].max())
    mean_solar  = float(df['solar_kwh'].mean())
    peak_solar  = float(df['solar_kwh'].max())
    outage_rate = float((~df['grid_available']).mean())

    # Estimate daily diesel if DG covers all outage hours at full load
    # (conservative upper bound for (s,S) reorder calibration)
    outage_hours_per_day = outage_rate * 24.0
    fuel_rate = params['fuel_rate_L_per_kWh']
    dg_power  = params['dg_power_kw']
    est_daily_diesel = fuel_rate * dg_power * outage_hours_per_day

    # (s,S) Track B parameters
    lead_time_days = 2.5   # lognormal mean (synthetic, architecture_plan §3.3)
    safety_days    = 2.0
    tank_cap       = params['tank_capacity_L']

    reorder_point_s = est_daily_diesel * (lead_time_days + safety_days)
    order_up_to_S   = tank_cap * 0.85   # 15% overflow buffer

    return {
        'mean_load_kwh':       mean_load,
        'peak_load_kwh':       peak_load,
        'mean_solar_kwh':      mean_solar,
        'peak_solar_kwh':      peak_solar,
        'outage_rate':         outage_rate,
        'est_daily_diesel_L':  est_daily_diesel,
        'reorder_point_s':     reorder_point_s,
        'order_up_to_S':       order_up_to_S,
        'tank_cap_L':          tank_cap,
    }
