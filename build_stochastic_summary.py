"""
build_stochastic_summary.py

Canonical, reproducible builder for results/stochastic_grid/stochastic_summary.csv.

No such builder previously existed in the repo -- the summary file was present
but nothing under version control regenerated it. This script reads the
per-episode CSVs (post training_seed/eval_seed fix) and aggregates to the same
policy x lead_scenario grain as the existing summary, using training_seed as
the unit of replication (3 independently trained models per policy), NOT
eval_seed (which is fixed at 42 across all episodes, by design, for a
controlled comparison).

Usage:
    python3 build_stochastic_summary.py results/stochastic_grid/
"""
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def main(dir_path: str):
    d = Path(dir_path)
    files = [f for f in d.glob("*.csv") if f.name != "stochastic_summary.csv"]
    if not files:
        print(f"No per-episode CSVs found in {d}")
        return

    dfs = [pd.read_csv(f) for f in files]
    all_df = pd.concat(dfs, ignore_index=True)

    if "training_seed" not in all_df.columns:
        raise SystemExit(
            "training_seed column not found -- run fix_stochastic_grid_seed_labels.py first."
        )

    rows = []
    for (policy, scen), g in all_df.groupby(["policy", "lead_scenario"]):
        # Average all sites and episodes within each training seed, then use the
        # three training-seed means as the independent replicates.
        per_seed = g.groupby("training_seed")["EENS_kWh"].mean()
        vals = per_seed.values
        n = len(vals)
        mean = vals.mean()
        std = vals.std(ddof=1) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 1 else 0.0
        if n > 1 and sem > 0:
            h = sem * stats.t.ppf(0.975, n - 1)
            ci_lo, ci_hi = mean - h, mean + h
        else:
            ci_lo, ci_hi = mean, mean
        rows.append({
            "policy": policy, "lead_scenario": scen,
            "mean": round(mean, 2), "std": round(std, 2),
            "n_training_seeds": g["training_seed"].nunique(),
            "n_sites": g["site"].nunique(),
            "n_episodes": len(g),
            "sem": round(sem, 2),
            "ci95_lo": round(ci_lo, 2), "ci95_hi": round(ci_hi, 2),
        })

    out = pd.DataFrame(rows).sort_values(["policy", "lead_scenario"])
    out_path = d / "stochastic_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "results/stochastic_grid/"
    main(target)
