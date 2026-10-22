#!/usr/bin/env python3
"""G3 MPC/Oracle-MPC timing pilot.

Decides G3's propagation scope (protocol §6.3).  Times ONE evaluation cell for
MPC (H=24) and ONE for Oracle-MPC (H=360), then extrapolates to the full
400-cell grid (10 sites x 10 seeds x 4 scenarios).

Run from the repository root:
    python tools/mpc_timing_pilot.py --repo .
Options:
    --site site5 --scenario normal --seed 42 --episodes 10
    --skip-oracle        (time MPC only)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

FULL_GRID_CELLS = 400   # 10 sites x 10 seeds x 4 scenarios


def time_one_cell(kind, site, scenario, seed, episodes, data_dir,
                  load_site, train_test_split, TelecomEnv, SSPolicy,
                  MPCDispatchB1Policy, OracleMPCPolicy):
    ss = SSPolicy()
    policy = (OracleMPCPolicy(ss_policy=ss, verbose=False) if kind == "oracle"
              else MPCDispatchB1Policy(forecast_cache=None, ss_policy=ss,
                                       horizon=24, verbose=False))
    df, params = load_site(str(data_dir / f"{site}.csv"))
    _tr, te = train_test_split(df)

    step_times, ep_times = [], []
    t_cell = time.perf_counter()
    for ep in range(episodes):
        env = TelecomEnv(site_data=te, site_params=params, episode_len=360,
                         eval_mode=True, lead_scenario=scenario, seed=seed + ep,
                         init_inv_frac_low=0.3, init_inv_frac_high=0.9)
        obs, info = env.reset(seed=seed + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        t_ep = time.perf_counter()
        done = False
        while not done:
            t0 = time.perf_counter()
            # Same call signature evaluate.py uses for MPC policies (evaluate.py:175)
            action, _ = policy.predict(obs, env=env, site_id=site, t=env._t_idx)
            step_times.append(time.perf_counter() - t0)
            action = int(action)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        ep_times.append(time.perf_counter() - t_ep)
        env.close()
    cell = time.perf_counter() - t_cell

    s = np.array(step_times)
    return {
        "kind": kind, "site": site, "scenario": scenario, "seed": seed,
        "episodes": episodes, "decisions": int(s.size),
        "cell_seconds": round(cell, 2),
        "episode_mean_s": round(float(np.mean(ep_times)), 3),
        "decision_ms_median": round(float(np.median(s) * 1e3), 3),
        "decision_ms_mean": round(float(s.mean() * 1e3), 3),
        "decision_ms_p95": round(float(np.percentile(s, 95) * 1e3), 3),
        "decision_ms_max": round(float(s.max() * 1e3), 3),
        # A real evaluation cell is 10 episodes.  Normalise so a short pilot
        # (--episodes 2) still projects a FULL cell correctly.
        "cell_seconds_per_10ep": round(cell * 10.0 / episodes, 2),
        "projected_400_cells_hours": round(cell * (10.0 / episodes)
                                           * FULL_GRID_CELLS / 3600.0, 2),
    }


def sample_oracle(site, scenario, seed, n_steps, data_dir, load_site,
                  train_test_split, TelecomEnv, SSPolicy, OracleMPCPolicy):
    """Sample Oracle-MPC decision times.

    A full oracle cell can exceed an hour, so the pilot samples the FIRST
    n_steps decisions.  Those are the SLOWEST of the episode: the planning
    horizon recedes from 360 toward 1 as the episode advances, so the true
    whole-episode mean is lower.  The projection is therefore an single-episode extrapolation,
    which is the right side to err on for a go/no-go decision.
    """
    ss = SSPolicy()
    policy = OracleMPCPolicy(ss_policy=ss, verbose=False)
    df, params = load_site(str(data_dir / f"{site}.csv"))
    _tr, te = train_test_split(df)
    env = TelecomEnv(site_data=te, site_params=params, episode_len=360,
                     eval_mode=True, lead_scenario=scenario, seed=seed,
                     init_inv_frac_low=0.3, init_inv_frac_high=0.9)
    obs, _ = env.reset(seed=seed)
    policy.reset()
    st, done, n = [], False, 0
    while not done and n < n_steps:
        t0 = time.perf_counter()
        action, _ = policy.predict(obs, env=env, site_id=site, t=env._t_idx)
        st.append(time.perf_counter() - t0)
        obs, _, term, trunc, _ = env.step(int(action))
        done = term or trunc
        n += 1
    env.close()
    s = np.array(st)
    ep_s = float(s.mean()) * 360
    return {
        "kind": "oracle", "site": site, "scenario": scenario, "seed": seed,
        "episodes": "sampled", "decisions": int(s.size),
        "cell_seconds": round(ep_s * 10, 1),
        "cell_seconds_per_10ep": round(ep_s * 10, 1),
        "episode_mean_s": round(ep_s, 1),
        "decision_ms_median": round(float(np.median(s) * 1e3), 1),
        "decision_ms_mean": round(float(s.mean() * 1e3), 1),
        "decision_ms_p95": round(float(np.percentile(s, 95) * 1e3), 1),
        "decision_ms_max": round(float(s.max() * 1e3), 1),
        "projected_400_cells_hours": round(ep_s * 10 * FULL_GRID_CELLS / 3600.0, 1),
        "caveat": "UPPER BOUND -- sampled from the slowest (longest-horizon) "
                  "steps; the horizon recedes as the episode advances.",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--site", default="site5")
    ap.add_argument("--scenario", default="normal")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--oracle-sample-steps", type=int, default=120,
                    help="Oracle-MPC is sampled, not run to completion: a full "
                         "oracle cell can exceed an hour. 120 steps is enough "
                         "to establish the order of magnitude.")
    ap.add_argument("--output", type=Path,
                    default=Path("results/phase3r/g3_mpc_timing_pilot.json"))
    a = ap.parse_args()

    repo = a.repo.expanduser().resolve()
    sys.path.insert(0, str(repo / "src"))
    os.chdir(repo)
    from env.data_loader import load_site, train_test_split
    from env.telecom_env import TelecomEnv
    from baselines.s_S_policy import SSPolicy
    from baselines.mpc_policy import MPCDispatchB1Policy, OracleMPCPolicy

    args_common = (a.site, a.scenario, a.seed, a.episodes, repo / "data" / "processed",
                   load_site, train_test_split, TelecomEnv, SSPolicy,
                   MPCDispatchB1Policy, OracleMPCPolicy)

    print(f"Timing MPC (H=24) on {a.site}/{a.scenario}/seed{a.seed}, "
          f"{a.episodes} episodes ...", flush=True)
    rows = [time_one_cell("mpc", *args_common)]
    if not a.skip_oracle:
        print(f"Sampling Oracle-MPC (H=360), {a.oracle_sample_steps} decisions "
              f"-- much slower, ~2-4 min ...", flush=True)
        rows.append(sample_oracle(a.site, a.scenario, a.seed, a.oracle_sample_steps,
                                  repo / "data" / "processed", load_site,
                                  train_test_split, TelecomEnv, SSPolicy,
                                  OracleMPCPolicy))

    # Decide PER POLICY -- they differ by two orders of magnitude, so a combined
    # total would hide an affordable MPC behind an unaffordable Oracle-MPC.
    by = {r["kind"]: r["projected_400_cells_hours"] for r in rows}
    mpc_h = by.get("mpc"); ora_h = by.get("oracle")
    total_h = sum(by.values())

    def verdict(h):
        if h is None:  return "not measured"
        if h < 4:      return f"re-run FULL 400-cell grid ({h} h)"
        if h < 12:     return f"re-run FULL grid, schedule overnight ({h} h)"
        if h < 40:     return f"reduce to the 3 G4 sites, ~{h/10*3:.1f} h, disclose scope"
        return f"NOT AFFORDABLE at full scope ({h} h) -- retain analytic (s,S), disclose"

    decision = {"MPC (H=24)": verdict(mpc_h), "Oracle-MPC (H=360)": verdict(ora_h)}
    if mpc_h is not None and ora_h is not None:
        decision["ratio_oracle_over_mpc"] = round(ora_h / mpc_h, 1)

    report = {
        "status": "OK",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(repo),
        "full_grid_cells": FULL_GRID_CELLS,
        "cells": rows,
        "projected_total_hours_full_propagation": round(total_h, 2),
        "decision_rule": "per-policy: <4h full grid; 4-12h full grid overnight; 12-40h reduce to 3 G4 sites; >40h retain analytic (s,S) and disclose",
        "decision_per_policy": decision,
        "note": "Single-cell extrapolation. Solve time varies with state; treat as an order-of-magnitude estimate.",
    }
    out = a.output if a.output.is_absolute() else repo / a.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    for r in rows:
        print(f"  {r['kind']:7s} cell(10ep)={r['cell_seconds_per_10ep']:8.1f}s  "
              f"per-decision median={r['decision_ms_median']:7.2f}ms "
              f"p95={r['decision_ms_p95']:7.2f}ms  "
              f"-> 400 cells = {r['projected_400_cells_hours']:6.2f} h")
    print(f"\n  Combined full propagation: {total_h:.1f} h")
    print("  DECISION (per policy -- they differ by ~2 orders of magnitude):")
    for k, v in decision.items():
        print(f"    {k}: {v}")
    print(f"\n  Evidence saved: {out}")


if __name__ == "__main__":
    main()
