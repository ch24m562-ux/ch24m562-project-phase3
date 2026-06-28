"""
check_tank_files.py
Pre-analysis integrity check for tank sensitivity eval.
Run BEFORE analyse_tank_sensitivity.py
"""
import pandas as pd
from pathlib import Path

TANK_DIR = Path("results/sensitivity/tank")
POLICIES = ["rlinv", "b1"]
SITES    = ["site2", "site5", "site7"]
SEEDS    = [7,13,21,42,99,123,314,500,777,999]
TANKS    = ["0.33", "0.67", "2.0", "4.67"]
EXPECTED_FILES = 2 * 3 * 4 * 10   # policies x sites x tanks x seeds = 240
EXPECTED_ROWS  = EXPECTED_FILES * 10  # 10 episodes each = 2400

print("=" * 70)
print("PRE-ANALYSIS FILE INTEGRITY CHECK — TANK SENSITIVITY")
print("=" * 70)

files = [f for f in TANK_DIR.glob("*.csv")
         if "summary" not in f.name and "thesis" not in f.name]
print(f"\nFiles found: {len(files)}  (expected {EXPECTED_FILES})")

issues = []

# 1. Build expected file set
expected = set()
for p in POLICIES:
    for site in SITES:
        for tank in TANKS:
            for seed in SEEDS:
                expected.add(f"{p}_{site}_delayed_tank{tank}_s{seed}.csv")

found_names = {f.name for f in files}
missing = expected - found_names
extra   = found_names - expected

if missing:
    print(f"\n[WARN] {len(missing)} MISSING files:")
    for f in sorted(missing)[:15]:
        print(f"  - {f}")
    if len(missing) > 15:
        print(f"  ... and {len(missing)-15} more")
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

# 2. Row counts
print("\nChecking row counts...")
total_rows = 0
bad = []
for fpath in sorted(files):
    df = pd.read_csv(fpath)
    total_rows += len(df)
    if len(df) != 10:
        bad.append((fpath.name, len(df)))

print(f"  Total rows: {total_rows}  (expected {EXPECTED_ROWS})")
if total_rows == EXPECTED_ROWS:
    print("  -> Correct")
else:
    issues.append(f"Row count: got {total_rows}, expected {EXPECTED_ROWS}")

if bad:
    for name, n in bad:
        print(f"  [WARN] {name}: {n} rows")
    issues.append(f"{len(bad)} files with wrong row count")
else:
    print("  -> All files have 10 rows")

# 3. Quick EENS sanity -- hard sites should have non-zero EENS
print("\nEENS sanity at smallest tank (0.33x = 24h):")
for site in SITES:
    site_files = [f for f in files
                  if f"rlinv_{site}_delayed_tank0.33" in f.name]
    if site_files:
        mean_eens = pd.concat([pd.read_csv(f) for f in site_files])["EENS_kWh"].mean()
        status = "OK" if mean_eens > 0 else "[WARN] ZERO"
        print(f"  rlinv {site}/delayed/tank0.33: EENS={mean_eens:.2f}  {status}")

# 4. Verdict
print()
print("=" * 70)
if not issues:
    print("ALL CHECKS PASSED -- safe to run analyse_tank_sensitivity.py")
else:
    print(f"ISSUES ({len(issues)}):")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
print("=" * 70)
