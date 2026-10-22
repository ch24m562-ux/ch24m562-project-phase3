#!/usr/bin/env python3
"""G3 — held-out evaluation of the four labelled B1 variants.

Evaluates on the HISTORICAL test grid using the historical protocol, so results
are matched-subset comparable with the submitted Phase 3 campaign:

    days 46-60 (df_test, 360 h) | eval_mode=True | init inv [0.30, 0.90]
    10 sites x 4 scenarios x 10 historical seeds x 10 episodes
    env reseeded per episode with seed + ep, exactly as evaluate.py:272-273

Variants (all four reported regardless of which performs better):
    B1_Original         (0.25, 0.85)  -- what evaluate.py:494/:519 actually ran
    B1_Analytic         (0.3333, 0.85) -- the rule thesis §4.5 DESCRIBES
    B1_star_Deployable  per-site, frozen G3 calibration
    B1_star_HighBuffer  per-site, EXPLORATORY buffer-priority amendment

VALIDATION GATE (runs FIRST, fail-fast): B1_Original is evaluated alone and
compared episode-by-episode against the archived
`results/phase3/master_summary.csv` rows for policy == 'b1'.  Only on PASS are
the remaining three variants evaluated.  A failure therefore costs 4,000
episodes, not 16,000, and no unvalidated comparison is ever written.
A full-grid run additionally asserts exactly 4,000 B1_Original rows.

Writes results/phase3r/ only.  Historical files are never modified.

    python tools/eval_b1_variants.py --repo .
    python tools/eval_b1_variants.py --repo . --sites site5 --scenarios normal   # smoke
"""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

SCENARIOS = ("normal", "delayed", "monsoon", "extreme")
HIST_SEEDS = (42, 123, 777, 7, 13, 21, 99, 314, 500, 999)
EPISODES = 10
EPISODE_LEN = 360
INIT_LOW, INIT_HIGH = 0.30, 0.90          # historical EVALUATION convention
VARIANTS = ("B1_Original", "B1_Analytic", "B1_star_Deployable", "B1_star_HighBuffer")
KEEP = ("EENS_kWh", "diesel_kWh", "grid_kWh", "orders_placed", "stockout_events",
        "violations", "uptime_pct", "mean_inv_pct", "min_inv_pct", "mean_soc",
        "dg_on_fraction", "outage_hours", "emergency_arrivals", "init_inv_frac",
        "steps")


def run_cell(site, scenario, seed, s_norm, S_norm, data_dir,
             load_site, train_test_split, TelecomEnv, SSPolicy, B1Policy):
    """10 episodes for one (site, scenario, seed) under one (s,S)."""
    df, params = load_site(str(data_dir / f"{site}.csv"))
    _train, test = train_test_split(df)
    pol = B1Policy(ss_policy=SSPolicy(s_norm=s_norm, S_norm=S_norm))
    rows = []
    for ep in range(EPISODES):
        env = TelecomEnv(site_data=test, site_params=params,
                         episode_len=EPISODE_LEN, eval_mode=True,
                         lead_scenario=scenario, seed=seed + ep,
                         init_inv_frac_low=INIT_LOW, init_inv_frac_high=INIT_HIGH)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs, _, term, trunc, _ = env.step(int(pol.act(obs, env=env)))
            done = term or trunc
        st = env.get_episode_stats()
        rows.append({"site": site, "lead_scenario": scenario, "seed": seed,
                     "episode": ep, "s_norm": s_norm, "S_norm": S_norm,
                     **{k: st[k] for k in KEEP if k in st}})
        env.close()
    return rows


COMPARE_COLS = ("EENS_kWh", "diesel_kWh", "orders_placed", "init_inv_frac")


def validate_against_history(new_df, master_path, tol=1e-6):
    """Strict replay gate: B1_Original must reproduce the archived 'b1' rows.

    The earlier version used an INNER merge and passed whenever the rows that
    happened to match agreed -- so a half-missing archive reported PASS. This
    version requires key uniqueness on both sides and COMPLETE coverage of every
    newly evaluated row, via a left merge with an indicator.
    """
    key = ["site", "lead_scenario", "seed", "episode"]
    hist = pd.read_csv(master_path)
    hist = hist[hist.policy == "b1"][key + list(COMPARE_COLS)]
    mine = new_df[new_df.variant == "B1_Original"][key + list(COMPARE_COLS)]

    out = {"rows_expected": int(len(mine)),
           "rows_in_history_b1": int(len(hist)),
           "duplicate_keys_new": int(mine.duplicated(key).sum()),
           "duplicate_keys_history": int(hist.duplicated(key).sum())}

    merged = mine.merge(hist, on=key, suffixes=("_new", "_hist"),
                        how="left", indicator=True)
    missing = merged[merged["_merge"] != "both"]
    out["rows_matched"] = int((merged["_merge"] == "both").sum())
    out["rows_missing_from_history"] = int(len(missing))
    out["complete_coverage"] = bool(len(missing) == 0)
    out["keys_unique"] = bool(out["duplicate_keys_new"] == 0
                              and out["duplicate_keys_history"] == 0)
    if len(missing):
        out["missing_examples"] = missing[key].head(10).to_dict("records")

    both = merged[merged["_merge"] == "both"]
    for col in COMPARE_COLS:
        d = (both[f"{col}_new"] - both[f"{col}_hist"]).abs()
        out[col] = {"max_abs_diff": float(d.max()) if len(d) else None,
                    "n_mismatched": int((d > tol).sum()),
                    "match": bool((d <= tol).all()) if len(d) else False}

    out["PASS"] = bool(out["rows_expected"] > 0
                       and out["keys_unique"]
                       and out["complete_coverage"]
                       and out["rows_matched"] == out["rows_expected"]
                       and all(out[c]["match"] for c in COMPARE_COLS))
    return out, merged


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--params", type=Path,
                    default=Path("results/phase3r/b1_variants_params.json"),
                    help="four-variant file from tools/derive_b1_highbuffer.py")
    ap.add_argument("--master", type=Path,
                    default=Path("results/phase3/master_summary.csv"))
    ap.add_argument("--out-csv", type=Path,
                    default=Path("results/phase3r/b1_variants_heldout.csv"))
    ap.add_argument("--out-json", type=Path,
                    default=Path("results/phase3r/b1_variants_heldout_report.json"))
    ap.add_argument("--sites", default="all")
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--seeds", default="all")
    a = ap.parse_args()

    repo = a.repo.expanduser().resolve()
    sys.path.insert(0, str(repo / "src"))
    os.chdir(repo)
    from env.data_loader import load_site, train_test_split
    from env.telecom_env import TelecomEnv
    from baselines.s_S_policy import SSPolicy, B1Policy

    params = json.loads((repo / a.params).read_text(encoding="utf-8"))
    missing = [v for v in VARIANTS if v not in params]
    if missing:
        raise SystemExit(f"params file is missing variants {missing} — "
                         f"run tools/derive_b1_highbuffer.py first")


    sites = ([f"site{i}" for i in range(1, 11)] if a.sites == "all"
             else [x.strip() for x in a.sites.split(",")])
    scens = list(SCENARIOS) if a.scenarios == "all" else [x.strip() for x in a.scenarios.split(",")]
    seeds = list(HIST_SEEDS) if a.seeds == "all" else [int(x) for x in a.seeds.split(",")]

    master = repo / a.master
    if not master.exists():
        raise SystemExit(f"historical archive not found: {master}\n"
                         f"The replay gate cannot be skipped -- without it the "
                         f"evaluator is unvalidated and its numbers unusable.")

    full_grid = (a.sites == "all" and a.scenarios == "all" and a.seeds == "all")
    EXPECTED_FULL_ROWS = 10 * len(SCENARIOS) * len(HIST_SEEDS) * EPISODES  # 4,000
    total = len(VARIANTS) * len(sites) * len(scens) * len(seeds)
    print(f"held-out evaluation | {len(VARIANTS)} variants x {len(sites)} sites x "
          f"{len(scens)} scenarios x {len(seeds)} seeds = {total} cells "
          f"({total*EPISODES:,} episodes)\n")

    def eval_variant(variant):
        vt = time.perf_counter()
        out = []
        for site in sites:
            pv = params[variant][site]
            for scen in scens:
                for seed in seeds:
                    for r in run_cell(site, scen, seed, pv["s_norm"], pv["S_norm"],
                                      repo / "data" / "processed", load_site,
                                      train_test_split, TelecomEnv, SSPolicy, B1Policy):
                        r["variant"] = variant
                        out.append(r)
        print(f"  {variant:20s} {len(out):6d} episodes in "
              f"{time.perf_counter()-vt:6.1f}s")
        return out

    t0 = time.perf_counter()

    # ---- STAGE 1: B1_Original only, then the replay gate -------------------
    # Validate BEFORE spending the other 12,000 episodes.  If the evaluator
    # cannot reproduce the archived baseline, nothing after it is usable.
    print("stage 1/2 -- B1_Original, then the strict replay gate")
    rows = eval_variant("B1_Original")
    df_orig = pd.DataFrame(rows)
    val, _merged = validate_against_history(df_orig, master)

    if full_grid and val["rows_expected"] != EXPECTED_FULL_ROWS:
        raise SystemExit(
            f"full-grid run produced {val['rows_expected']} B1_Original rows, "
            f"expected {EXPECTED_FULL_ROWS}. Refusing to continue.")

    print(f"  rows expected={val['rows_expected']}  matched={val['rows_matched']}  "
          f"missing={val['rows_missing_from_history']}")
    print(f"  keys unique={val['keys_unique']}  complete coverage={val['complete_coverage']}")
    for c in COMPARE_COLS:
        md = val[c]["max_abs_diff"]
        print(f"    {c:16s} max|diff|={md:.3e}  mismatched={val[c]['n_mismatched']}"
              if md is not None else f"    {c:16s} no rows compared")
    print(f"  REPLAY GATE: {'PASS' if val['PASS'] else 'FAIL'}\n")

    if not val["PASS"]:
        fail = repo / a.out_json
        fail.parent.mkdir(parents=True, exist_ok=True)
        fail.write_text(json.dumps(
            {"status": "VALIDATION_FAILED", "stage": "replay gate (stage 1)",
             "date_utc": datetime.now(timezone.utc).isoformat(),
             "historical_validation": val,
        "replay_gate": {"stage": "evaluated and validated BEFORE the remaining "
                                 "three variants", "full_grid_run": full_grid,
                        "expected_rows_if_full_grid": EXPECTED_FULL_ROWS},
             "note": "Remaining variants were NOT evaluated. B1_Original does not "
                     "reproduce the archived Phase 3 'b1' results."},
            indent=2), encoding="utf-8")
        raise SystemExit(
            f"REPLAY GATE FAILED -- stopped after {len(rows):,} episodes instead "
            f"of {total*EPISODES:,}.\nReport: {fail}\n"
            f"Do not interpret any variant comparison until this is resolved.")

    # ---- STAGE 2: the remaining three variants -----------------------------
    print("stage 2/2 -- remaining variants (replay gate passed)")
    for variant in VARIANTS:
        if variant != "B1_Original":
            rows.extend(eval_variant(variant))

    df = pd.DataFrame(rows)
    out_csv = repo / a.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = (df.groupby(["variant", "lead_scenario"])["EENS_kWh"].mean()
                 .unstack().reindex(columns=[s for s in SCENARIOS if s in scens]))

    print("\nmean EENS (kWh/episode), held-out days 46-60:")
    print(summary.round(3).to_string())


    report = {
        "status": "OK" if val["PASS"] else "VALIDATION_FAILED",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "data": "df_test (days 46-60)", "eval_mode": True,
            "episode_len": EPISODE_LEN, "init_inv_frac": [INIT_LOW, INIT_HIGH],
            "seeds": seeds, "episodes_per_seed": EPISODES,
            "reset": "seed + ep, matching evaluate.py:272-273",
            "scenarios": scens, "sites": sites,
        },
        "variants": {v: params[v] for v in VARIANTS},
        "variant_standing": {
            "B1_Original": "historical baseline (as executed in Phase 3)",
            "B1_Analytic": "documentation-correction comparator (thesis §4.5)",
            "B1_star_Deployable": "primary frozen G3 calibrated baseline",
            "B1_star_HighBuffer": "EXPLORATORY tie-break sensitivity (22 Oct amendment)",
        },
        "historical_validation": val,
        "mean_EENS_by_variant_scenario": json.loads(summary.to_json()),
        "episodes": int(len(df)),
        "seconds": round(time.perf_counter() - t0, 1),
        "historical_files_modified": False,
    }
    (repo / a.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  per-episode : {out_csv}")
    print(f"  report      : {repo / a.out_json}")
    print(f"  total       : {report['seconds']:.0f}s")



if __name__ == "__main__":
    main()
