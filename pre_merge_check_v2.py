"""
pre_merge_check_v2.py
Run from repo root BEFORE python src/eval/build_master_summary.py
Now with corrected oracle_mpc seed list based on what actually ran.
"""
import re
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results/phase3")

FILENAME_RE = re.compile(
    r"^(?P<policy>[a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)*)_"
    r"(?P<site>site\d+)_"
    r"(?P<lead>normal|delayed|monsoon|extreme|very_delayed|fast|no_delay|multi)"
    r"(?:_(?P<tag>[^.]+))?"
    r"\.csv$"
)

ALL_SITES     = [f"site{i}" for i in range(1, 11)]
ALL_SCENARIOS = ["normal", "delayed", "monsoon", "extreme"]

# ── Step 0: confirm NO oracle_mpc files exist outside the subfolder ───────────
print("=" * 70)
print("STEP 0: Check for stray oracle_mpc CSVs outside subfolders")
print("=" * 70)
stray = [f for f in RESULTS_DIR.glob("*.csv")
         if "oracle" in f.name.lower() and f.name != "master_summary.csv"]
if stray:
    print(f"[WARN] {len(stray)} oracle_mpc CSVs found flat in {RESULTS_DIR}:")
    for f in sorted(stray):
        print(f"  {f.name}")
    print("Move or delete these before running build_master_summary.py")
else:
    print("OK -- no stray oracle_mpc files in the flat results/phase3/ folder")

# ── Step 1: confirm NO other backup/archive subfolders exist under results/phase3 ──
print()
print("=" * 70)
print("STEP 1: Check for unexpected subfolders under results/phase3/")
print("=" * 70)
known_subfolders = {
    "oracle_mpc", "mpc", "mpc_forecast", "b1", "rlinv",
    "trackb", "b0", "a5", "a6", "a7", "multi"
}
actual_subfolders = {d.name for d in RESULTS_DIR.iterdir() if d.is_dir()}
unexpected_dirs = actual_subfolders - known_subfolders
if unexpected_dirs:
    print(f"[WARN] Unexpected subfolders found: {sorted(unexpected_dirs)}")
    print("These will be recursively globbed by build_master_summary.py -- move them out")
else:
    print(f"OK -- only expected subfolders present: {sorted(actual_subfolders)}")

# ── Step 2: oracle_mpc file count and seed coverage ──────────────────────────
print()
print("=" * 70)
print("STEP 2: oracle_mpc file count and seed coverage")
print("=" * 70)
oracle_dir = RESULTS_DIR / "oracle_mpc"
oracle_files = list(oracle_dir.glob("*.csv"))
print(f"Files in oracle_mpc/: {len(oracle_files)}")

# Infer actual seeds from filenames
seed_re = re.compile(r"_seed(\d+)\.csv$")
found_seeds = set()
for f in oracle_files:
    m = seed_re.search(f.name)
    if m:
        found_seeds.add(int(m.group(1)))

original_seeds = {42, 123, 777}
rlinv_seeds    = {7, 13, 21, 42, 99, 123, 314, 500, 777, 999}
common = found_seeds & rlinv_seeds

print(f"Seeds in oracle_mpc folder: {sorted(found_seeds)}")
print(f"RLInv seeds:                {sorted(rlinv_seeds)}")
print(f"Common (pairable):          {sorted(common)}  -> n_pairs = {len(common)*10}")

# Expected file count
ep_original = 5
ep_new      = 2
expected_files = set()
for site in ALL_SITES:
    for scenario in ALL_SCENARIOS:
        for seed in found_seeds:
            expected_files.add(f"oracle_mpc_{site}_{scenario}_seed{seed}.csv")

found_names = {f.name for f in oracle_files}
missing = expected_files - found_names
extra   = found_names - expected_files

if missing:
    print(f"\n[WARN] {len(missing)} expected files are MISSING:")
    for f in sorted(missing)[:20]:
        print(f"  - {f}")
    if len(missing) > 20:
        print(f"  ... and {len(missing)-20} more")
else:
    print("\nOK -- no missing files")

if extra:
    print(f"\n[WARN] {len(extra)} unexpected extra files:")
    for f in sorted(extra)[:10]:
        print(f"  + {f}")
else:
    print("OK -- no unexpected extra files")

# ── Step 3: expected row counts ───────────────────────────────────────────────
print()
print("=" * 70)
print("STEP 3: Expected row count for oracle_mpc after merge")
print("=" * 70)
orig_rows = len(original_seeds) * len(ALL_SITES) * len(ALL_SCENARIOS) * ep_original
new_seeds = found_seeds - original_seeds
new_rows  = len(new_seeds) * len(ALL_SITES) * len(ALL_SCENARIOS) * ep_new
# Subtract known failures (site6/extreme seeds 21,99,314) if not retried
# We'll just compute the theoretical max and compare to actual
theoretical_total = orig_rows + new_rows
print(f"Original seeds {sorted(original_seeds)}: "
      f"{len(original_seeds)} x {len(ALL_SITES)} sites x {len(ALL_SCENARIOS)} scenarios "
      f"x {ep_original} ep = {orig_rows} rows")
print(f"New seeds {sorted(new_seeds)}: "
      f"{len(new_seeds)} x {len(ALL_SITES)} x {len(ALL_SCENARIOS)} x {ep_new} ep = {new_rows} rows")
print(f"Theoretical total: {theoretical_total}")
print(f"Theoretical per scenario: {theoretical_total // len(ALL_SCENARIOS)}")
print()
print("After build_master_summary.py, check_eens_sanity.py Section 1 should show")
print(f"oracle_mpc with ~{theoretical_total} rows (minus any residual failures).")

# ── Step 4: spot-check other policies ─────────────────────────────────────────
print()
print("=" * 70)
print("STEP 4: Other policy folder row counts")
print("=" * 70)
other_policies = ["mpc", "mpc_forecast", "b1", "rlinv",
                  "trackb", "b0", "a5", "a6", "a7", "multi"]
all_ok = True
for policy in other_policies:
    d = RESULTS_DIR / policy
    if not d.exists():
        continue
    files = list(d.glob("*.csv"))
    total = sum(len(pd.read_csv(f)) for f in files)
    status = "OK" if total == 4000 else f"WARN -- expected 4000, got {total}"
    print(f"  {policy:<15} {total:>6} rows  {status}")
    if total != 4000:
        all_ok = False

# ── Final verdict ─────────────────────────────────────────────────────────────
print()
print("=" * 70)
issues = []
if stray:           issues.append("Stray oracle_mpc files in flat results/phase3/")
if unexpected_dirs: issues.append(f"Unexpected subfolders: {unexpected_dirs}")
if missing:         issues.append(f"{len(missing)} missing oracle_mpc files")
if extra:           issues.append(f"{len(extra)} unexpected extra oracle_mpc files")
if not all_ok:      issues.append("Non-4000 row count in another policy folder")

if issues:
    print("ISSUES FOUND -- resolve before merging:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("ALL CHECKS PASSED -- safe to run build_master_summary.py")
print("=" * 70)
