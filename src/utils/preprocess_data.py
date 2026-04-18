"""
FINAL PRODUCTION DATA PREPROCESSING PIPELINE
Thesis-Quality Implementation for ITU/Zindi Telecom Dataset

Outputs:
1. master_timeseries.csv - All 10 sites combined for multi-site RL training
2. site1.csv ... site10.csv - Individual site files for debugging
3. site_classification.csv - Site difficulty ranking
4. validation_report.csv - Data quality checks

Author: Unified from multiple design discussions
Date: Feb 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings


# ============================================================================
# CONFIGURATION
# ============================================================================

RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# Weather features to include in per-site debug CSVs (for EDA and future work)
# These will be excluded from master CSV to keep it lean for training
WEATHER_FEATURES = [
    'ghi', 'dhi', 'dni',
    'clearsky_ghi', 'clearsky_dhi', 'clearsky_dni',
    'solar_zenith_angle', 'relative_humidity'
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names: lowercase, convert special chars to underscores
    
    Handles:
    - BOM characters (\ufeff)
    - Parentheses (both ASCII and full-width Chinese)
    - Spaces, slashes, etc.
    - Collapses multiple underscores
    
    Example: 'Total Energy(kWh)' → 'total_energy_kwh'
    """
    df.columns = (
        df.columns
        .astype(str)
        .str.replace('\ufeff', '', regex=False)           # Remove BOM
        .str.strip()
        .str.lower()
        .str.replace(r'[^a-z0-9]+', '_', regex=True)      # Non-alphanumeric → underscore
        .str.replace(r'_+', '_', regex=True)              # Collapse multiple underscores
        .str.strip('_')                                   # Remove leading/trailing underscores
    )
    return df


def parse_outage_plan(plan_str: str) -> list:
    """
    Parse grid outage plan from string to boolean list.
    
    Input: "[false false true true ...]" (168 values)
    Output: [0, 0, 1, 1, ...] (168 integers)
    """
    plan_str = str(plan_str).replace("[", "").replace("]", "").strip()
    tokens = plan_str.split()
    
    # Convert to 0/1 (not True/False for easier arithmetic)
    bool_list = [1 if token.lower() == "true" else 0 for token in tokens]
    
    # CRITICAL: Must be exactly 168 (one week of hours)
    if len(bool_list) != 168:
        raise ValueError(f"Outage plan length is {len(bool_list)}, expected 168")
    
    return bool_list


# ============================================================================
# VALIDATION CLASSES
# ============================================================================

class SiteDataValidator:
    """Rigorous validation checks for each site"""
    
    @staticmethod
    def validate_site(df: pd.DataFrame, site_id: str) -> dict:
        """
        Run all validation checks on a site's data.
        
        Returns:
            dict: {'site', 'passed', 'errors', 'warnings'}
        """
        results = {
            'site': site_id,
            'passed': True,
            'errors': [],
            'warnings': []
        }
        
        # Check 1: Expected number of rows (60 days × 24 hours)
        expected_rows = 60 * 24
        if len(df) != expected_rows:
            results['passed'] = False
            results['errors'].append(f"Expected {expected_rows} rows, got {len(df)}")
        
        # Check 2: Time index is complete (0 to 1439) and no duplicates
        if 't' in df.columns:
            expected_t = set(range(expected_rows))
            actual_t = set(df['t'].values)
            missing = expected_t - actual_t
            if missing:
                results['passed'] = False
                results['errors'].append(f"Missing time indices: {sorted(list(missing))[:10]}...")
            
            # Check for duplicate timestamps
            if df['t'].duplicated().any():
                results['passed'] = False
                dup_count = df['t'].duplicated().sum()
                results['errors'].append(f"Duplicate time indices found: {dup_count} duplicates")
        
        # Check 3: No negative values
        for col in ['load_kwh', 'solar_kwh']:
            if col in df.columns and (df[col] < 0).any():
                results['passed'] = False
                results['errors'].append(f"Negative values in {col}")
        
        # Check 4: No missing values (NaNs) in critical columns
        critical_cols = ['load_kwh', 'solar_kwh', 'grid_available']
        for col in critical_cols:
            if col in df.columns and df[col].isna().any():
                results['passed'] = False
                nan_count = df[col].isna().sum()
                results['errors'].append(f"Missing values (NaNs) in {col}: {nan_count} rows")
        
        # Check 5: Load never zero (tower always has base load)
        if 'load_kwh' in df.columns and (df['load_kwh'] == 0).any():
            results['warnings'].append("Zero load values found (unusual)")
        
        # Check 6: Solar should be zero at night
        if all(c in df.columns for c in ['solar_kwh', 'hour']):
            night_hours = df[df['hour'].isin([0, 1, 2, 3, 4, 22, 23])]
            # Use 1.0 kWh threshold to account for sensor noise, twilight effects
            if night_hours['solar_kwh'].max() > 1.0:
                results['warnings'].append(f"High solar at night: max={night_hours['solar_kwh'].max():.2f} kWh")
        
        # Check 7: Grid outage percentage sanity
        if 'grid_available' in df.columns:
            outage_pct = 100 * (~df['grid_available']).sum() / len(df)
            if outage_pct > 80:
                results['warnings'].append(f"Very high outage: {outage_pct:.1f}%")
            elif outage_pct < 5:
                results['warnings'].append(f"Very low outage: {outage_pct:.1f}%")
        
        return results


class SiteClassifier:
    """Classify sites by difficulty for training selection"""
    
    @staticmethod
    def classify_site(df: pd.DataFrame, site_id: str) -> dict:
        """
        Calculate difficulty metrics for site.
        
        Difficulty factors:
        1. Solar coverage (lower = harder)
        2. Grid outage severity (higher = harder)
        3. Net load stress (higher = harder)
        
        Returns:
            dict: Classification metrics
        """
        mean_load = df['load_kwh'].mean()
        mean_solar = df['solar_kwh'].mean()
        solar_coverage = mean_solar / mean_load if mean_load > 0 else 0
        
        # Net load (positive = deficit)
        net_load = df['load_kwh'] - df['solar_kwh']
        mean_net_load = net_load.mean()
        
        # Grid outage metrics
        outage_pct = 100 * (~df['grid_available']).sum() / len(df)
        
        # Calculate difficulty score (0-6 range)
        score = 0
        
        # Factor 1: Solar coverage
        if solar_coverage < 0.3:
            score += 2
        elif solar_coverage < 0.6:
            score += 1
        
        # Factor 2: Grid outages
        if outage_pct > 40:
            score += 2
        elif outage_pct > 20:
            score += 1
        
        # Factor 3: Net load stress
        if mean_net_load > mean_load * 0.6:
            score += 1
        
        # Map to category
        if score <= 1:
            difficulty = "Easy"
        elif score <= 3:
            difficulty = "Medium"
        else:
            difficulty = "Hard"
        
        return {
            'site': site_id,
            'mean_load_kwh': mean_load,
            'mean_solar_kwh': mean_solar,
            'solar_coverage_pct': solar_coverage * 100,
            'mean_net_load_kwh': mean_net_load,
            'outage_pct': outage_pct,
            'difficulty_score': score,
            'difficulty': difficulty,
            'is_surplus': solar_coverage > 1.0  # Flag renewable-surplus sites
        }


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def preprocess_data():
    """
    Main preprocessing pipeline.
    
    Steps:
    1. Load 3 raw CSV files
    2. Clean column names
    3. Convert battery Ah → kWh
    4. Parse grid outage patterns
    5. Merge load + solar data
    6. Add absolute time index
    7. Expand grid outage to 1440 hours
    8. Add static site parameters
    9. Validate data quality
    10. Classify site difficulty
    11. Save master CSV + per-site CSVs
    """
    
    print("=" * 80)
    print("TELECOM TOWER RL - DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # ========================================================================
    # STEP 1: Load raw files
    # ========================================================================
    print("\n[1/11] Loading raw CSV files...")
    
    try:
        load_df = pd.read_csv(RAW_PATH / "energy consumption.csv")
        solar_df = pd.read_csv(RAW_PATH / "solar power and weather condition.csv")
        site_df = pd.read_csv(RAW_PATH / "site information and grid outage.csv")
        print(f"  ✓ Loaded: {len(load_df)} load rows, {len(solar_df)} solar rows, {len(site_df)} sites")
    except FileNotFoundError as e:
        print(f"  ✗ Error: Could not find file in {RAW_PATH}")
        print(f"    Expected: energy consumption.csv, solar power and weather condition.csv, site information and grid outage.csv")
        raise
    
    # ========================================================================
    # STEP 2: Clean column names
    # ========================================================================
    print("[2/11] Cleaning column names...")
    
    load_df = clean_columns(load_df)
    solar_df = clean_columns(solar_df)
    site_df = clean_columns(site_df)
    
    # Standardize key column names
    load_df = load_df.rename(columns={
        'site_name': 'site_id',
        'total_energy_kwh': 'load_kwh'
    })
    
    solar_df = solar_df.rename(columns={
        'site_name': 'site_id',
        'energy_output_kwh': 'solar_kwh'
    })
    
    site_df = site_df.rename(columns={
        'site_id': 'site_id',  # Already correct
        'diesel_power_kw': 'dg_power_kw',
        'battery_capacity_ah': 'battery_capacity_ah',
        'rated_voltage_v': 'rated_voltage_v',
        'battery_charge_coefficient': 'battery_charge_coeff',
        'battery_discharge_coefficient': 'battery_discharge_coeff',
        'grid_power_kw': 'grid_power_kw',
        'init_soc': 'init_soc',
        'dod': 'dod',
        'grid_outage_plan': 'grid_outage_plan'
    })
    
    print(f"  ✓ Standardized column names")
    
    # ========================================================================
    # STEP 3: Convert battery Ah → kWh (CRITICAL)
    # ========================================================================
    print("[3/11] Converting battery capacity Ah → kWh...")
    
    site_df['battery_capacity_kwh'] = (
        site_df['battery_capacity_ah'] * site_df['rated_voltage_v'] / 1000
    )
    
    print(f"  ✓ Example: {site_df['battery_capacity_ah'].iloc[0]} Ah × "
          f"{site_df['rated_voltage_v'].iloc[0]} V = "
          f"{site_df['battery_capacity_kwh'].iloc[0]:.2f} kWh")
    
    # ========================================================================
    # STEP 4: Parse grid outage patterns
    # ========================================================================
    print("[4/11] Parsing 168-hour grid outage patterns...")
    
    # IMPORTANT DESIGN DECISION (for thesis documentation):
    # -------------------------------------------------------
    # The dataset provides a 168-hour (1 week) grid outage template per site.
    # We parse this template and will later expand it to 1440 hours (60 days)
    # by repeating the pattern cyclically: outage[t] = pattern[t % 168]
    #
    # POTENTIAL CONCERN: Calendar Overfitting
    # ----------------------------------------
    # A naive cyclic repetition could allow the RL agent to memorize the
    # outage schedule (e.g., "Monday 2pm always has outage") rather than
    # learning reactive energy management based on instantaneous state.
    #
    # MITIGATION STRATEGY (implemented in environment, not preprocessing):
    # --------------------------------------------------------------------
    # 1. RANDOM PHASE SHIFT (Training):
    #    At each episode reset, apply random offset: pattern[(t + φ) % 168]
    #    where φ ~ Uniform(0, 167). This preserves outage statistics while
    #    preventing temporal memorization.
    #
    # 2. STRESS TEST SCENARIOS (Evaluation):
    #    a) Baseline: Same phase-shifted pattern as training
    #    b) Long outage injection: Add 12-24 hour continuous outage once per episode
    #    c) Clustered outages: Same % but in larger blocks (not scattered)
    #
    # JUSTIFICATION (for thesis defense):
    # -----------------------------------
    # - We preserve the empirical outage characteristics from the dataset
    # - Random phase shift is a standard domain randomization technique
    # - Stress tests demonstrate robustness under distribution shift
    # - Stochastic outage modeling deferred to future work (requires
    #   assumptions beyond provided data)
    #
    # THESIS SECTION REFERENCE:
    # -------------------------
    # This decision should be discussed in:
    # - Chapter 5 (Dataset & Simulator): Grid outage modeling approach
    # - Chapter 6 (Results): Evaluation under multiple scenarios
    # - Chapter 7 (Conclusion): Future work - semi-Markov outage model
    
    site_df['outage_pattern'] = site_df['grid_outage_plan'].apply(parse_outage_plan)
    print(f"  ✓ Parsed {len(site_df)} outage patterns (168 hours each)")
    print(f"  ℹ Note: Cyclic pattern will be randomized in environment (see design doc above)")
    
    # ========================================================================
    # STEP 5: Merge load + solar (with weather features)
    # ========================================================================
    print("[5/11] Merging load and solar data...")
    
    # Determine which weather features are actually in solar_df
    available_weather = [c for c in WEATHER_FEATURES if c in solar_df.columns]
    
    if available_weather:
        print(f"  ℹ Including {len(available_weather)} weather features for per-site debug CSVs")
        solar_keep = ['site_id', 'day', 'hour', 'solar_kwh'] + available_weather
    else:
        print(f"  ℹ No weather features found in solar dataset")
        solar_keep = ['site_id', 'day', 'hour', 'solar_kwh']
    
    merged = pd.merge(
        load_df[['site_id', 'day', 'hour', 'load_kwh']],
        solar_df[solar_keep],
        on=['site_id', 'day', 'hour'],
        how='inner'
    )
    
    expected_rows = len(load_df)
    if len(merged) != expected_rows:
        print(f"  ⚠ Warning: Merged {len(merged)} rows, expected {expected_rows}")
    else:
        print(f"  ✓ Merged {len(merged)} rows (all aligned)")
    
    # ========================================================================
    # STEP 6: Add absolute time index
    # ========================================================================
    print("[6/11] Creating absolute time index...")
    
    merged['t'] = (merged['day'] - 1) * 24 + merged['hour']
    
    # Note: Each site has its own t = 0 to 1439
    # Global min/max would be misleading (all sites overlap at 0-1439)
    print(f"  ✓ Created time index (per-site range: 0 to 1439)")
    
    # ========================================================================
    # STEP 7-10: Process each site (in deterministic order)
    # ========================================================================
    print("[7/11] Processing individual sites...")
    
    all_sites = []
    validation_results = []
    classification_results = []
    
    # Process sites in sorted order for stable output and logs
    for site_id in sorted(merged['site_id'].unique()):
        print(f"\n  Processing {site_id}...")
        
        # Get site data
        site_data = merged[merged['site_id'] == site_id].copy()
        site_info = site_df[site_df['site_id'] == site_id]
        
        if len(site_info) == 0:
            print(f"    ✗ No site info found for {site_id}, skipping")
            continue
        
        site_info = site_info.iloc[0]
        
        # Sort by time
        site_data = site_data.sort_values('t').reset_index(drop=True)
        
        # Validate time range for this site
        t_min, t_max = site_data['t'].min(), site_data['t'].max()
        if t_min != 0 or t_max != 1439:
            print(f"    ⚠ Unexpected time range: t = {t_min} to {t_max} (expected 0 to 1439)")
        
        # Expand grid outage pattern (168-hour cycle → 1440 hours)
        # 
        # IMPLEMENTATION NOTE (for thesis Chapter 5):
        # --------------------------------------------
        # Here we expand the 168-hour template to 1440 hours by cyclic repetition.
        # This creates a deterministic baseline pattern for the preprocessed dataset.
        # 
        # At TRAINING TIME (in environment):
        # ----------------------------------
        # The environment will apply random phase shift: pattern[(t + φ) % 168]
        # This prevents the agent from overfitting to a fixed calendar schedule.
        # 
        # At EVALUATION TIME (in evaluation script):
        # -------------------------------------------
        # We test under multiple scenarios:
        #   1. Phase-randomized (same as training)
        #   2. Long outage injection (12-24h continuous blocks)
        #   3. Clustered outages (same % but different temporal structure)
        # 
        # SEMANTIC VERIFICATION:
        # ----------------------
        # Column name is "grid outage plan", so semantics are:
        #   true  = outage is happening (grid unavailable)
        #   false = no outage (grid available)
        # Therefore:
        #   outage_pattern[t] = 1 → outage happening
        #   grid_available = (pattern[t] == 0) → no outage
        
        outage_pattern = site_info['outage_pattern']
        
        # Compute expected outage % analytically (handles partial week correctly)
        # 1440 hours = 8 full weeks (8×168) + 96 extra hours
        full_weeks = 1440 // 168  # = 8
        extra_hours = 1440 % 168   # = 96
        
        expected_outages = (full_weeks * sum(outage_pattern) + 
                           sum(outage_pattern[:extra_hours]))
        expected_outage_pct = 100 * expected_outages / 1440
        
        site_data['grid_available'] = site_data['t'].apply(
            lambda t: outage_pattern[int(t) % 168] == 0  # 0 in pattern = no outage = available
        )
        
        # Verify expanded pattern matches analytical expectation (should be exact)
        actual_outage_pct = 100 * (~site_data['grid_available']).sum() / len(site_data)
        if abs(expected_outage_pct - actual_outage_pct) > 0.01:  # Allow tiny float rounding
            print(f"    ⚠ Outage calculation error: expected={expected_outage_pct:.2f}%, actual={actual_outage_pct:.2f}%")
        
        # Add static site parameters to each row
        site_data['battery_capacity_kwh'] = site_info['battery_capacity_kwh']
        site_data['dg_power_kw'] = site_info['dg_power_kw']
        site_data['grid_power_kw'] = site_info['grid_power_kw']
        site_data['battery_charge_coeff'] = site_info['battery_charge_coeff']
        site_data['battery_discharge_coeff'] = site_info['battery_discharge_coeff']
        site_data['init_soc'] = site_info['init_soc']
        site_data['dod'] = site_info['dod']
        
        # NOTE: soc_min will be computed in environment from DoD
        # DoD semantics unclear from dataset - needs verification:
        #   Option 1: soc_min = 1.0 - dod (if DoD is fraction dischargeable)
        #   Option 2: soc_min = init_soc (if init_soc already at minimum)
        # DO NOT compute here - let environment handle it
        
        # Calculate net load (critical metric)
        site_data['net_load_kwh'] = site_data['load_kwh'] - site_data['solar_kwh']
        
        # Identify which weather columns are actually available in this site's data
        weather_cols = [c for c in WEATHER_FEATURES if c in site_data.columns]
        
        # Validate
        val_result = SiteDataValidator.validate_site(site_data, site_id)
        validation_results.append(val_result)
        
        if not val_result['passed']:
            print(f"    ✗ VALIDATION FAILED:")
            for error in val_result['errors']:
                print(f"       {error}")
        else:
            print(f"    ✓ Validation passed")
        
        if val_result['warnings']:
            for warning in val_result['warnings']:
                print(f"    ⚠ {warning}")
        
        # Classify difficulty
        classification = SiteClassifier.classify_site(site_data, site_id)
        classification_results.append(classification)
        
        print(f"    Load: {classification['mean_load_kwh']:.2f} kWh/h")
        print(f"    Solar: {classification['mean_solar_kwh']:.2f} kWh/h")
        print(f"    Coverage: {classification['solar_coverage_pct']:.1f}%")
        print(f"    Outages: {classification['outage_pct']:.1f}%")
        print(f"    Difficulty: {classification['difficulty']} (score={classification['difficulty_score']})")
        
        # Save per-site debug file (with weather columns for EDA)
        debug_cols = [
            'site_id', 't', 'day', 'hour',
            'load_kwh', 'solar_kwh', 'net_load_kwh',
            'grid_available',
            'battery_capacity_kwh', 'dg_power_kw', 'grid_power_kw',
            'battery_charge_coeff', 'battery_discharge_coeff',
            'init_soc', 'dod'
        ] + weather_cols
        
        # Filter to only columns that exist
        debug_cols = [c for c in debug_cols if c in site_data.columns]
        
        site_data[debug_cols].to_csv(PROCESSED_PATH / f"{site_id}.csv", index=False)
        print(f"    ✓ Saved: {site_id}.csv ({len(debug_cols)} columns)")
        
        all_sites.append(site_data)
    
    # ========================================================================
    # STEP 11: Create master combined CSV
    # ========================================================================
    print("\n[8/11] Creating master combined CSV...")
    
    master_df = pd.concat(all_sites, ignore_index=True)
    master_df = master_df.sort_values(['site_id', 't']).reset_index(drop=True)
    
    # Select final columns (clean schema)
    final_columns = [
        'site_id', 't', 'day', 'hour',
        'load_kwh', 'solar_kwh', 'net_load_kwh',
        'grid_available',
        'battery_capacity_kwh', 'dg_power_kw', 'grid_power_kw',
        'battery_charge_coeff', 'battery_discharge_coeff',
        'init_soc', 'dod'  # Note: soc_min computed in environment, not stored
    ]
    
    master_df = master_df[final_columns]
    
    master_file = PROCESSED_PATH / "master_timeseries.csv"
    master_df.to_csv(master_file, index=False)
    
    print(f"  ✓ Saved: {master_file}")
    print(f"    Shape: {master_df.shape}")
    print(f"    Sites: {master_df['site_id'].nunique()}")
    print(f"    Total hours: {len(master_df)}")
    
    # ========================================================================
    # Save metadata files
    # ========================================================================
    print("\n[9/11] Saving metadata files...")
    
    # Classification
    class_df = pd.DataFrame(classification_results)
    class_df = class_df.sort_values('difficulty_score', ascending=False)
    class_df.to_csv(PROCESSED_PATH / "site_classification.csv", index=False)
    print(f"  ✓ Saved: site_classification.csv")
    
    # Validation
    val_df = pd.DataFrame([
        {
            'site': r['site'],
            'passed': r['passed'],
            'errors': '; '.join(r['errors']) if r['errors'] else '',
            'warnings': '; '.join(r['warnings']) if r['warnings'] else ''
        }
        for r in validation_results
    ])
    val_df.to_csv(PROCESSED_PATH / "validation_report.csv", index=False)
    print(f"  ✓ Saved: validation_report.csv")
    
    # ========================================================================
    # Print summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("[10/11] VALIDATION SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in validation_results if r['passed'])
    print(f"Sites passed validation: {passed} / {len(validation_results)}")
    
    failed = [r for r in validation_results if not r['passed']]
    if failed:
        print("\n✗ FAILED SITES:")
        for r in failed:
            print(f"  {r['site']}: {', '.join(r['errors'])}")
    
    print("\n" + "=" * 80)
    print("[11/11] SITE DIFFICULTY CLASSIFICATION")
    print("=" * 80)
    
    print("\nRanking (hardest to easiest):")
    print("-" * 80)
    for _, row in class_df.iterrows():
        print(f"{row['site']:8s} | {row['difficulty']:8s} | "
              f"Solar: {row['solar_coverage_pct']:5.1f}% | "
              f"Outage: {row['outage_pct']:5.1f}% | "
              f"Score: {row['difficulty_score']}")
    
    # Training recommendation
    print("\n📊 TRAINING SITE RECOMMENDATION:")
    print("-" * 80)
    
    hard = class_df[class_df['difficulty'] == 'Hard']
    medium = class_df[class_df['difficulty'] == 'Medium']
    easy = class_df[class_df['difficulty'] == 'Easy']
    
    recommended = []
    if len(hard) > 0:
        recommended.append(hard.iloc[0]['site'])
        print(f"  Hard:   {hard.iloc[0]['site']}")
    if len(medium) > 0:
        recommended.append(medium.iloc[0]['site'])
        print(f"  Medium: {medium.iloc[0]['site']}")
    if len(easy) > 0:
        recommended.append(easy.iloc[0]['site'])
        print(f"  Easy:   {easy.iloc[0]['site']}")
    
    print(f"\n  → Train on these {len(recommended)} sites for diversity")
    print(f"  → Test on remaining {len(class_df) - len(recommended)} sites")
    
    # Flag surplus sites for special evaluation
    surplus_sites = class_df[class_df['is_surplus'] == True]
    if len(surplus_sites) > 0:
        print(f"\n  ℹ Renewable-surplus sites detected (solar > load):")
        for _, row in surplus_sites.iterrows():
            print(f"    • {row['site']} (coverage={row['solar_coverage_pct']:.1f}%)")
        print(f"    → Use as special test cases (diesel near-zero scenarios)")
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ PREPROCESSING COMPLETE")
    print("=" * 80)
    print("\nOutput files:")
    print(f"  1. master_timeseries.csv - Combined dataset for RL training (15 columns)")
    print(f"  2. site1.csv ... site10.csv - Per-site debug files (15+ columns, includes weather)")
    print(f"  3. site_classification.csv - Difficulty rankings")
    print(f"  4. validation_report.csv - Data quality report")
    print("\nGrid Outage Strategy (IMPORTANT for thesis):")
    print("  • Preprocessed data: 168-hour template repeated cyclically")
    print("  • Training: Random phase shift applied in environment")
    print("  • Evaluation: Multiple scenarios (baseline + stress tests)")
    print("  • See Step 4 comments above for full justification")
    print("\nNext steps:")
    print("  1. Review site_classification.csv to confirm training sites")
    print("  2. Run EDA notebook to visualize site comparisons")
    print("  3. Build environment with phase randomization (see design notes)")
    print("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    preprocess_data()