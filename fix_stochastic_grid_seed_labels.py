"""
fix_stochastic_grid_seed_labels.py

RCA: results/stochastic_grid/*.csv has a `seed` column that reports the
evaluation-episode reset seed (args.seed in evaluate.py, left at its default
42 for all three runs), NOT the training-seed identity of the model
checkpoint under test. The `_s42`/`_s123`/`_s777` in each filename names the
actual trained checkpoint (matches the repo-wide `{site}_s{seed}_final.zip`
convention), and IS the correct per-replicate identifier -- it just was never
also written into the CSV as its own column.

This script does NOT touch the existing `seed` column or any EENS/diesel/etc.
data -- nothing about the underlying numbers is wrong or needs regenerating.
It only ADDS a new `training_seed` column parsed from each filename, and
renames the ambiguous `seed` column to `eval_seed` so the meaning is
unambiguous going forward. Fully non-destructive and reversible (keeps a
`.bak` of each file).

Usage:
    python3 fix_stochastic_grid_seed_labels.py results/stochastic_grid/
"""
import sys
import re
import shutil
from pathlib import Path
import pandas as pd

def main(dir_path: str):
    d = Path(dir_path)
    files = sorted(d.glob("*.csv"))
    files = [f for f in files if f.name != "stochastic_summary.csv"]

    if not files:
        print(f"No per-episode CSVs found in {d}")
        return

    fixed, skipped = 0, 0
    for f in files:
        m = re.search(r"_s(\d+)\.csv$", f.name)
        if not m:
            print(f"[SKIP] Could not parse training seed from filename: {f.name}")
            skipped += 1
            continue
        training_seed = int(m.group(1))

        df = pd.read_csv(f)
        if "training_seed" in df.columns:
            print(f"[SKIP] Already fixed: {f.name}")
            skipped += 1
            continue

        backup = f.with_suffix(".csv.bak")
        shutil.copy2(f, backup)

        if "seed" in df.columns:
            df = df.rename(columns={"seed": "eval_seed"})
        df["training_seed"] = training_seed

        df.to_csv(f, index=False)
        fixed += 1
        print(f"[OK] {f.name}: eval_seed preserved, training_seed={training_seed} added")

    print(f"\nDone. Fixed {fixed} files, skipped {skipped}. Backups written as *.csv.bak")
    print("Verify a couple of files, then `git add`/`git commit` as usual, e.g.:")
    print(f"  git add {d}/*.csv")
    print('  git commit -m "Add training_seed column to stochastic-grid CSVs; '
          'rename ambiguous seed->eval_seed (no data values changed)"')
    print(f"  rm {d}/*.csv.bak   # once you're satisfied, or keep them / gitignore them")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "results/stochastic_grid/"
    main(target)
