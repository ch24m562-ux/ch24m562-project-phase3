from __future__ import annotations

import re
from pathlib import Path
import pandas as pd


RESULTS_DIR = Path("results/phase3")
OUT_CSV = RESULTS_DIR / "master_summary.csv"


FILENAME_RE = re.compile(
    r"^(?P<policy>[a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)*)_"
    r"(?P<site>site\d+)_"
    r"(?P<lead>normal|delayed|monsoon|extreme|very_delayed|fast|no_delay|multi)"
    r"(?:_(?P<tag>[^.]+))?"
    r"\.csv$"
)


def parse_filename(path: Path) -> dict:
    match = FILENAME_RE.match(path.name)
    if not match:
        return {
            "file_policy": "",
            "file_site": "",
            "file_lead": "",
            "file_tag": "",
            "source_file": path.name,
        }

    d = match.groupdict()
    return {
        "file_policy": d.get("policy") or "",
        "file_site": d.get("site") or "",
        "file_lead": d.get("lead") or "",
        "file_tag": d.get("tag") or "",
        "source_file": path.name,
    }


SKIP_PATTERNS = [
    "audit_",
    "check_",
    "recheck_",
    "gate",
    "debug",
    "reboot",
    "sanity",
]


def main() -> None:
    rows = []
    skipped = []

    for path in sorted(RESULTS_DIR.rglob("*.csv")):
        if path.name == OUT_CSV.name:
            continue
        if path.name.startswith("experiment_registry"):
            continue
        if path.name.startswith("master_summary"):
            continue

        # ── Skip debug/audit/diagnostic files ────────────────────────────
        if any(pat in path.name for pat in SKIP_PATTERNS):
            skipped.append(path.name)
            continue
        # ─────────────────────────────────────────────────────────────────

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}")
            continue

        if df.empty:
            continue

        meta = parse_filename(path)

        for k, v in meta.items():
            if k not in df.columns:
                df[k] = v

        rows.append(df)

    if skipped:
        print(f"[SKIP] {len(skipped)} diagnostic file(s) excluded: {skipped}")

    if not rows:
        print("[WARN] No result CSVs found.")
        return

    master = pd.concat(rows, ignore_index=True, sort=False)

    # ── policy_label normalisation ────────────────────────────────────────
    if "policy_label" not in master.columns and "policy" in master.columns:
        master["policy_label"] = master["policy"]
    elif "policy_label" in master.columns and "policy" in master.columns:
        master["policy_label"] = master["policy_label"].fillna(master["policy"])
    # ─────────────────────────────────────────────────────────────────────

    preferred_cols = [
        "policy",
        "policy_label",
        "file_policy",
        "site",
        "file_site",
        "lead_scenario",
        "file_lead",
        "train_scenario",
        "episode",
        "seed",
        "experiment_tag",
        "train_steps",
        "steps",
        "EENS_kWh",
        "outage_hours",
        "uptime_pct",
        "diesel_kWh",
        "grid_kWh",
        "cost_proxy",
        "orders_placed",
        "stockout_events",
        "violations",
        "mean_soc",
        "mean_inv_pct",
        "min_inv_pct",
        "dg_on_fraction",
        "init_inv_frac",
        "episode_len",
        "file_tag",
        "source_file",
    ]

    existing  = [c for c in preferred_cols if c in master.columns]
    remaining = [c for c in master.columns if c not in existing]
    master    = master[existing + remaining]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUT_CSV, index=False)

    print(f"[OK] Wrote {OUT_CSV}")
    print(f"Rows:  {len(master)}")
    print(f"Files: {master['source_file'].nunique()}")

    if "policy" in master.columns and "lead_scenario" in master.columns:
        print()
        print("Rows by policy/lead:")
        print(master.groupby(["policy", "lead_scenario"]).size())


if __name__ == "__main__":
    main()