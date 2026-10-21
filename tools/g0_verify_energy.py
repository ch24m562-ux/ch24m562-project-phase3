#!/usr/bin/env python3
"""G0 energy-ledger audit for the HISTORICAL, UNMODIFIED TelecomEnv.

Install in repository as tools/g0_verify_energy.py, then run from any directory:
    python tools/g0_verify_energy.py --repo .

For an optional historical-results and tank audit:
    python tools/g0_verify_energy.py --repo . --master results/phase3/master_summary.csv

This verifies consistency of *implemented* energy and inventory bookkeeping.
It does NOT prove that the source dataset's discharge coefficient is a physically
correct efficiency; that is a distinct G0 modelling limitation (protocol §1.6).
It does NOT modify TelecomEnv, models, historical results or random-seed protocol.

v1.8.2: also validates violation_tank and the combined violation flag against
independently reconstructed conditions, so no constraint flag is merely trusted.
Clipping/overflow events are COUNTED, not auto-accepted: a non-zero count is an
investigation trigger even when the simulator handled it correctly.

v1.8.1: the SoC ledger check now compares against the CLIPPED SoC and asserts the
violation_soc flag separately.  The v1.8 form compared against the unclipped value
and would have hard-failed any legitimate clipping step -- the same false-rejection
class as the original HF-2 defect it was written to replace.
Successful execution writes a JSON evidence report. Failure exits nonzero.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import os

import numpy as np
import pandas as pd

TOL = 1e-6
SITES = ("site1", "site3", "site5")
SCENARIOS = ("normal", "delayed", "monsoon", "extreme")
SWEEP_SITES = ("site1", "site3", "site5", "site8")


def require(condition: bool, message: str):
    if not bool(condition):
        raise AssertionError(message)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def trajectory(site, scenario, seed, n_ep, data_dir, load_site, train_test_split, TelecomEnv):
    data, params = load_site(str(data_dir / f"{site}.csv"))
    _, test_data = train_test_split(data)
    require(len(test_data) == 360, f"{site}: expected 360 test hours, got {len(test_data)}")
    env = TelecomEnv(site_data=test_data, site_params=params, episode_len=360,
                     eval_mode=True, lead_scenario=scenario, seed=seed,
                     init_inv_frac_low=0.3, init_inv_frac_high=0.9)

    # Action RNG deliberately independent of the environment's own delivery RNG.
    action_rng = np.random.default_rng(seed + 100000)
    rows = []
    examples = {}
    try:
        for ep in range(n_ep):
            env.reset(seed=seed + ep)
            n_steps = 0
            sum_unmet = 0.0
            curtailed_steps = 0
            max_curtail = 0.0
            max_load_residual = 0.0
            max_source_residual = 0.0
            max_soc_residual = 0.0
            max_inventory_residual = 0.0
            stockouts = 0
            soc_clip_events = 0
            tank_overflow_events = 0
            while True:
                allowed = np.flatnonzero(env.get_action_mask())
                require(len(allowed) > 0, f"{site}/{scenario}: empty action mask")
                action = int(action_rng.choice(allowed))
                old_soc = float(env._soc)
                old_inv = float(env._inv_kwh)
                _, _, terminated, truncated, info = env.step(action)
                n_steps += 1

                # Reconstruct the load supply from the individual dispatch priority,
                # instead of defining 'served' as load - unmet (a circular check).
                load = float(info["p_load_kwh"])
                pv = float(info["p_pv_kwh"])
                grid = float(info["p_grid_kwh"])  # includes grid-to-battery
                dg = float(info["p_dg_kwh"])
                bat = float(info["p_bat_kwh"])  # + means charge, - discharge
                unmet = float(info["unmet_kwh"])
                require(np.all(np.isfinite([load,pv,grid,dg,bat,unmet])),
                        f"Non-finite energy at {site}/{scenario}/{ep}/{n_steps}")

                pv_load = min(pv, load)
                grid_load = min(grid, max(0.0, load - pv_load))
                bat_load = max(-bat, 0.0)
                sources_to_load = pv_load + grid_load + bat_load + dg
                load_err = sources_to_load + unmet - load
                max_load_residual = max(max_load_residual, abs(load_err))
                require(abs(load_err) <= TOL,
                        f"Load ledger: {site}/{scenario}/ep{ep}/step{n_steps}: {load_err}")
                require(-TOL <= unmet <= load + TOL, f"Invalid unmet: {site}/{scenario}")

                # Disentangle grid-to-load from grid-to-battery, and PV charging
                # from curtailed PV. 'grid' already includes grid battery charging.
                grid_charge = grid - grid_load
                pv_charge = max(bat, 0.0) - grid_charge
                curtailment = pv - pv_load - pv_charge
                require(grid_charge >= -TOL, f"Negative grid charge: {site}/{scenario}")
                require(pv_charge >= -TOL, f"Negative PV charge: {site}/{scenario}")
                require(-TOL <= curtailment <= pv + TOL,
                        f"PV curtailment outside [0,PV]: {site}/{scenario}/{curtailment}")
                source_err = pv + grid + dg - bat - curtailment - (load - unmet)
                max_source_residual = max(max_source_residual, abs(source_err))
                require(abs(source_err) <= TOL, f"Source ledger: {site}/{scenario}/{source_err}")

                # Check implemented SoC update and tank balance; these are checks
                # of simulator consistency, NOT a validation of eta_d semantics.
                delta_soc = (max(bat, 0.0) * env.eta_c
                             - max(-bat, 0.0) / env.eta_d) / env.bat_cap_kwh
                expected_raw = old_soc + delta_soc
                # The env CLIPS SoC to [SOC_MIN, SOC_MAX] and raises violation_soc
                # when it does (telecom_env.py:654-657).  Comparing against the
                # UNCLIPPED value would hard-fail a legitimate clipped step -- the
                # same failure mode as the original HF-2 defect.  Compare against
                # the clipped value, and separately assert the flag is consistent.
                expected_soc = float(np.clip(expected_raw, env.SOC_MIN, env.SOC_MAX))
                soc_err = float(info["soc"]) - expected_soc
                max_soc_residual = max(max_soc_residual, abs(soc_err))
                require(abs(soc_err) <= TOL,
                        f"SoC ledger: {site}/{scenario}/ep{ep}/step{n_steps}: {soc_err}")
                clipped = abs(expected_soc - expected_raw) > 1e-3
                require(bool(info.get("violation_soc", 0)) == clipped,
                        f"SoC clip flag inconsistent: {site}/{scenario}/ep{ep}/step{n_steps}")
                soc_clip_events += int(clipped)


                fuel = float(info["fuel_used_kwh"])
                delivery = float(info["delivery_kwh"])
                require(abs(dg - fuel) <= TOL,
                        f"DG dispatch exceeds accounted fuel: {site}/{scenario}/{dg - fuel}")
                expected_inv = min(env.tank_cap_kwh, old_inv - fuel + delivery)
                inv_err = float(info["inv_kwh"]) - expected_inv
                max_inventory_residual = max(max_inventory_residual, abs(inv_err))
                require(abs(inv_err) <= TOL,
                        f"Inventory ledger: {site}/{scenario}/ep{ep}/step{n_steps}: {inv_err}")
                # v1.8.2: validate violation_tank and the combined flag against
                # independently reconstructed conditions, so no constraint flag
                # is merely trusted.  Events are COUNTED, not auto-accepted.
                tank_over = (old_inv - fuel + delivery) > env.tank_cap_kwh + 1e-9
                require(bool(info.get("violation_tank", 0)) == tank_over,
                        f"Tank flag inconsistent: {site}/{scenario}/ep{ep}/step{n_steps}")
                require(bool(info.get("violation", 0)) ==
                        (bool(info.get("violation_soc", 0)) or bool(info.get("violation_tank", 0))),
                        f"Combined violation flag inconsistent: {site}/{scenario}/ep{ep}/step{n_steps}")
                tank_overflow_events += int(tank_over)
                require(env.min_fuel_kwh + TOL >= env.dg_rated_kw * env.DT_HOURS,
                        f"Generator minimum fuel threshold insufficient at {site}")
                require(-TOL <= dg <= env.dg_rated_kw * env.DT_HOURS + TOL,
                        f"DG exceeds rating at {site}/{scenario}")
                require(-TOL <= grid <= env.grid_max_kw * env.DT_HOURS + TOL,
                        f"Grid exceeds connection rating at {site}/{scenario}")
                require(env.SOC_MIN - TOL <= float(info['soc']) <= env.SOC_MAX + TOL,
                        f"SoC outside allowed bounds at {site}/{scenario}")
                require(-TOL <= float(info['inv_kwh']) <= env.tank_cap_kwh + TOL,
                        f"Inventory outside capacity at {site}/{scenario}")

                curtailed_steps += int(curtailment > TOL)
                max_curtail = max(max_curtail, curtailment)
                sum_unmet += unmet
                stockouts += int(info['stockout_flag'])
                if site == "site5" and scenario == "extreme":
                    snapshot = {"load":load, "unmet":unmet, "pv":pv,
                                "grid_total":grid, "dg":dg, "battery_net":bat,
                                "curtailment":curtailment}
                    if unmet > TOL:
                        examples.setdefault("supply_shortage", snapshot)
                    if unmet <= TOL:
                        examples.setdefault("fully_served", snapshot)
                    if pv > load + TOL:
                        examples.setdefault("excess_PV", snapshot)
                if terminated or truncated:
                    break

            stats = env.get_episode_stats()
            require(n_steps == 360, f"Wrong episode length at {site}/{scenario}: {n_steps}")
            require(int(stats['steps']) == n_steps,
                    f"Episode steps mismatch {site}/{scenario}/{ep}")
            require(abs(float(stats['EENS_kWh']) - sum_unmet) <= TOL,
                    f"EENS aggregation mismatch at {site}/{scenario}/{ep}")
            require(int(stats['stockout_events']) == stockouts,
                    f"Stockout aggregation mismatch at {site}/{scenario}/{ep}")
            rows.append({"site":site, "scenario":scenario, "seed":seed, "episode":ep,
                         "steps":n_steps, "EENS_kWh":sum_unmet,
                         "curtailed_steps":curtailed_steps,
                         "max_curtail_kWh":max_curtail,
                         "max_load_residual":max_load_residual,
                         "max_source_residual":max_source_residual,
                         "max_soc_residual":max_soc_residual,
                         "max_inventory_residual":max_inventory_residual,
                         "soc_clip_events":soc_clip_events,
                         "tank_overflow_events":tank_overflow_events,
                         "stockouts":stockouts})
    finally:
        env.close()
    return rows, examples


def data_audit(data_dir, master_path, load_site, train_test_split):
    tank_rows = []
    for j in range(1, 11):
        site = f"site{j}"
        p = data_dir / f"{site}.csv"
        if not p.exists():
            continue
        data, _ = load_site(str(p))
        tr, te = train_test_split(data)
        if len(tr) != 1080 or len(te) != 360:
            raise AssertionError(f"Bad train/test rows for {site}")
        ctrain, ctest = 72 * float(tr.load_kwh.mean()), 72 * float(te.load_kwh.mean())
        tank_rows.append({"site":site, "tank_train_kwh_eq":round(ctrain,6),
                          "tank_test_kwh_eq":round(ctest,6),
                          "test_minus_train_pct":round(100*(ctest/ctrain - 1),6)})
    out = {"tank_comparison":tank_rows}
    if master_path and master_path.exists():
        df = pd.read_csv(master_path)
        mandatory = ['policy','site','lead_scenario','seed','episode','steps',
                     'EENS_kWh','stockout_events','violations']
        missing = [c for c in mandatory if c not in df.columns]
        require(not missing, f"master_summary missing columns: {missing}")
        require(not df[mandatory].isna().any().any(), "Missing historical metrics")
        require(not df.duplicated(['policy','site','lead_scenario','seed','episode']).any(),
                "Duplicate historical episode keys")
        require(bool((df['steps'] == 360).all()), "Historical wrong episode length")
        require(bool((df['EENS_kWh'] >= -TOL).all()), "Historical negative EENS")
        out['historical_summary'] = {
            "source":str(master_path), "sha256":sha256(master_path),
            "episode_rows":int(len(df)), "policies":sorted(df.policy.unique().tolist()),
            "scenarios":sorted(df.lead_scenario.unique().tolist()),
            "stockout_events":int(df.stockout_events.sum()),
            "violation_events":int(df.violations.sum()),
            "seeds_by_policy":{p:sorted(map(int,g.seed.unique().tolist()))
                               for p,g in df.groupby('policy')},
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output', type=Path, default=Path('results/phase3r/g0_energy_audit.json'))
    parser.add_argument('--master', type=Path, default=Path('results/phase3/master_summary.csv'))
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    src = repo/'src'
    data_dir = repo/'data'/'processed'
    if not (src/'env'/'telecom_env.py').exists() or not (data_dir/'site5.csv').exists():
        raise SystemExit(f"Repository missing expected src/env/telecom_env.py or data/processed/site5.csv: {repo}")
    sys.path.insert(0, str(src))
    os.chdir(repo)
    from env.data_loader import load_site, train_test_split
    from env.telecom_env import TelecomEnv

    main_rows, examples = [], {}
    for site in SITES:
        for scen in SCENARIOS:
            rows, cases = trajectory(site, scen, 42, 3, data_dir,
                                      load_site, train_test_split, TelecomEnv)
            main_rows.extend(rows)
            examples.update(cases)
    sweep_rows = []
    for site in SWEEP_SITES:
        for scen in SCENARIOS:
            rows, _ = trajectory(site, scen, 42, 3, data_dir,
                                  load_site, train_test_split, TelecomEnv)
            sweep_rows.extend(rows)

    nmain = sum(x['steps'] for x in main_rows)
    nsweep = sum(x['steps'] for x in sweep_rows)
    require(nmain == 12960, f"Wrong primary audit step count {nmain}")
    require(nsweep == 17280, f"Wrong curtailment audit step count {nsweep}")
    master = args.master if args.master.is_absolute() else repo / args.master
    historical = data_audit(data_dir, master, load_site, train_test_split)
    report = {
        "status":"PASS", "audit_date_utc":datetime.now(timezone.utc).isoformat(),
        "environment":"UNMODIFIED Phase 3 TelecomEnv", "repository":str(repo),
        "file_hashes":{
            str(path.relative_to(repo)):sha256(path) for path in
            ([src/'env'/'telecom_env.py',src/'env'/'data_loader.py',
              src/'config_loader.py', src/'config'/'hparams.yaml'] +
             [data_dir/f'site{i}.csv' for i in range(1, 11)])
            if path.exists()
        },
        "scope":{
            "primary_sites":list(SITES), "curtailment_sites":list(SWEEP_SITES),
            "scenarios":list(SCENARIOS), "episodes_per_cell":3,
            "primary_steps":nmain, "curtailment_sweep_steps":nsweep,
        },
        "max_load_error":max(x['max_load_residual'] for x in main_rows+sweep_rows),
        "max_source_error":max(x['max_source_residual'] for x in main_rows+sweep_rows),
        "max_soc_error":max(x['max_soc_residual'] for x in main_rows+sweep_rows),
        "max_inventory_error":max(x['max_inventory_residual'] for x in main_rows+sweep_rows),
        "curtailment_fraction_sweep":sum(x['curtailed_steps'] for x in sweep_rows)/nsweep,
        "curtailment_fraction_site8":sum(x['curtailed_steps'] for x in sweep_rows if x['site']=='site8')/
                                      sum(x['steps'] for x in sweep_rows if x['site']=='site8'),
        "max_curtailment_kWh":max(x['max_curtail_kWh'] for x in sweep_rows),
        "stockouts_in_test_trajectories":sum(x['stockouts'] for x in main_rows+sweep_rows),
        "soc_clip_events":sum(x['soc_clip_events'] for x in main_rows+sweep_rows),
        "tank_overflow_events":sum(x['tank_overflow_events'] for x in main_rows+sweep_rows),
        "examples":examples,
        **historical,
        "limits":[
            "Tests energy-ledger consistency for the historical implementation only; does not validate source battery coefficient semantics.",
            "Checks randomly selected feasible actions, not policy-optimality or all possible action sequences.",
            "Does not reproduce historical RL policy checkpoints; separate G0 checkpoint replay required.",
            "Physical tank size and reward d_bar are derived independently from train/test slices as in original code.",
        ],
    }
    out = args.output if args.output.is_absolute() else repo / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f"PASS: primary {nmain:,} steps; curtailment sweep {nsweep:,} steps")
    print(f"Load/source/SoC/inventory maximum errors: "
          f"{report['max_load_error']:.3e} / {report['max_source_error']:.3e} / "
          f"{report['max_soc_error']:.3e} / {report['max_inventory_error']:.3e}")
    print(f"Curtailment in sweep: {report['curtailment_fraction_sweep']:.2%}; "
          f"site8: {report['curtailment_fraction_site8']:.2%}")
    if 'historical_summary' in report:
        print(f"Historical rows: {report['historical_summary']['episode_rows']:,}, "
              f"stockouts: {report['historical_summary']['stockout_events']}")
    else:
        print(f"Historical master CSV not found at {master}; dataset audit only")
    print(f"Evidence saved: {out}")


if __name__ == '__main__':
    main()
