"""
Pre-analysis file integrity check for lognormal sensitivity eval.
Run BEFORE analyse_lognormal_sensitivity.py
"""
import pandas as pd
from pathlib import Path
import re

LOGNORMAL_DIR = Path("results/sensitivity/lognormal")
FOCUS_POLICIES = ["rlinv", "b1", "mpc"]
SCENARIOS = ["normal", "delayed", "monsoon", "extreme"]
SEEDS = [42, 123, 777, 7, 13, 21, 99, 314, 500, 999]
SITES = [f"site{i}" for i in range(1, 11)]
EXPECTED_FILES = 1200
EXPECTED_ROWS_TOTAL = 12000
EXPECTED_ROWS_PER_FILE = 10

print("=" * 70)
print("PRE-ANALYSIS FILE INTEGRITY CHECK — LOGNORMAL SENSITIVITY")
print("=" * 70)

files = list(LOGNORMAL_DIR.glob("*.csv"))
print(f"\nFiles found: {len(files)}  (expected {EXPECTED_FILES})")

issues = []

# 1. File count
if len(files) != EXPECTED_FILES:
    issues.append(f"File count: got {len(files)}, expected {EXPECTED_FILES}")

# 2. Build expected file set
expected = set()
for p in FOCUS_POLICIES:
    for site in SITES:
        for sc in SCENARIOS:
            for seed in SEEDS:
                expected.add(f"{p}_{site}_{sc}_lognormal_s{seed}.csv")

found_names = {f.name for f in files}
missing = expected - found_names
extra   = found_names - expected

if missing:
    print(f"\n[WARN] {len(missing)} MISSING files:")
    for f in sorted(missing)[:10]:
        print(f"  - {f}")
    if len(missing) > 10:
        print(f"  ... and {len(missing)-10} more")
    issues.append(f"{len(missing)} missing files")
else:
    print("  -> No missing files")

if extra:
    print(f"\n[WARN] {len(extra)} UNEXPECTED files:")
    for f in sorted(extra)[:5]:
        print(f"  + {f}")
    issues.append(f"{len(extra)} unexpected files")
else:
    print("  -> No unexpected files")

# 3. Row count check (sample all files)
print("\nChecking row counts...")
bad_rows = []
total_rows = 0
for fpath in sorted(files):
    try:
        df = pd.read_csv(fpath)
        total_rows += len(df)
        if len(df) != EXPECTED_ROWS_PER_FILE:
            bad_rows.append((fpath.name, len(df)))
    except Exception as e:
        bad_rows.append((fpath.name, f"READ ERROR: {e}"))

print(f"  Total rows: {total_rows}  (expected {EXPECTED_ROWS_TOTAL})")
if total_rows == EXPECTED_ROWS_TOTAL:
    print("  -> Row count correct")
else:
    issues.append(f"Row count: got {total_rows}, expected {EXPECTED_ROWS_TOTAL}")

if bad_rows:
    print(f"\n[WARN] {len(bad_rows)} files with wrong row count:")
    for name, n in bad_rows[:10]:
        print(f"  {name}: {n} rows")
    issues.append(f"{len(bad_rows)} files with wrong row count")
else:
    print("  -> All files have correct row count (10)")

# 4. Check hard sites have non-zero EENS (sanity on distribution working)
print("\nChecking hard sites have non-zero EENS under delayed/extreme...")
hard_site_eens = {}
for site in ["site2", "site5", "site7"]:
    for sc in ["delayed", "extreme"]:
        key = f"rlinv_{site}_{sc}"
        site_files = [f for f in files
                      if f.name.startswith(f"rlinv_{site}_{sc}_lognormal")]
        if site_files:
            dfs = [pd.read_csv(f) for f in site_files]
            mean_eens = pd.concat(dfs)["EENS_kWh"].mean()
            hard_site_eens[key] = mean_eens
            status = "OK" if mean_eens > 0 else "[WARN] ZERO -- check env"
            print(f"  rlinv {site}/{sc}: mean EENS = {mean_eens:.2f}  {status}")
            if mean_eens == 0:
                issues.append(f"rlinv {site}/{sc} has EENS=0 -- unexpected")

# 5. Check lead_distribution column in files
print("\nChecking lead_distribution logged correctly...")
sample_file = sorted(files)[0] if files else None
if sample_file:
    df = pd.read_csv(sample_file)
    if "lead_dist" in df.columns or "lead_distribution" in df.columns:
        col = "lead_dist" if "lead_dist" in df.columns else "lead_distribution"
        vals = df[col].unique()
        print(f"  lead_distribution values: {vals}")
    else:
        print("  [NOTE] No lead_distribution column in CSV (logged at env level, not in output)")

# 6. Final verdict
print()
print("=" * 70)
if not issues:
    print("ALL CHECKS PASSED -- safe to run analyse_lognormal_sensitivity.py")
else:
    print(f"ISSUES FOUND ({len(issues)}):")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
print("=" * 70)
