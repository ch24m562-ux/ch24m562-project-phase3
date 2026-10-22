#!/usr/bin/env python3
"""G3 — B1*-Deployable (s,S) calibration.

FROZEN PROTOCOL (PHASE3R_PROTOCOL.md §6.3, frozen 2026-10-22)
-------------------------------------------------------------
Per site, independently:
  data        : days 1-45 only (df_train, 1080 h). THE TEST WINDOW IS NEVER OPENED.
  env         : TelecomEnv, eval_mode=False, episode_len=360, random start 0..720
  scenario    : normal (geometric lead time)
  init inv    : [0.20, 0.90]  -- the TRAINING convention, matching RLInv
  seeds       : 1001 and 2001, 5 episodes each = 10 per candidate
  reset       : seed + ep, ep = 0..4  (identical start index + initial inventory
                for EVERY candidate; delivery draws legitimately diverge)
  coarse      : s in 0.05..0.90 step 0.10, S in s+0.10..0.95 step 0.10  -> 45 pairs
  refine      : +/-0.05 around the coarse optimum, unchanged coordinates allowed,
                centre excluded, S >= s+0.10, bounds enforced -> <= 8 more
  objective   : minimise mean EENS over the 10 calibration episodes
  tie-break   : within 1% of the minimum -> lower mean diesel -> fewer orders

Also records, per site: how many candidates tie at the minimum, and the
mean/SD of the selected candidate AND the runner-up, so it is visible whether
the winner is actually separated from the field or is inside the noise.

Run from the repository root:
    python tools/tune_b1_star.py --repo .
"""
from __future__ import annotations

import argparse, json, sys, os, time
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import numpy as np


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


SCENARIO        = "normal"
EPISODE_LEN     = 360
INIT_LOW        = 0.20          # training convention
INIT_HIGH       = 0.90
TUNING_SEEDS    = (1001, 2001)  # deliberately OUTSIDE the historical ten
EPS_PER_SEED    = 5             # 5 + 5 = 10 calibration episodes
COARSE_STEP     = 0.10
REFINE_OFFSET   = 0.05
S_MIN, S_MAX    = 0.05, 0.90
SS_MAX          = 0.95
MIN_GAP         = 0.10          # S >= s + MIN_GAP
TIE_BAND        = 0.01          # 1%
Q_SMALL_EXPECTED = 0.30         # asserted against SSPolicy() at startup
Q_LARGE_EXPECTED = 0.60


def assert_fixed_order_quantities(SSPolicy):
    """The manifest claims the sweep held order SIZES fixed at 0.30/0.60.

    Verify that against the live SSPolicy defaults rather than trusting a
    hardcoded value -- otherwise the record could assert one pair of
    quantities while the policy used another.
    """
    d = SSPolicy()
    if abs(d.q_small_pct - Q_SMALL_EXPECTED) > 1e-12 or \
       abs(d.q_large_pct - Q_LARGE_EXPECTED) > 1e-12:
        raise AssertionError(
            f"Order-size defaults changed: SSPolicy() has "
            f"q_small_pct={d.q_small_pct}, q_large_pct={d.q_large_pct}; "
            f"this protocol froze {Q_SMALL_EXPECTED}/{Q_LARGE_EXPECTED}. "
            f"Refusing to run -- the manifest would misreport the sweep.")
    return {"q_small_pct": float(d.q_small_pct),
            "q_large_pct": float(d.q_large_pct),
            "verified_against": "live SSPolicy() defaults, not a hardcoded value"}


def coarse_grid():
    out = []
    for s in np.arange(S_MIN, S_MAX + 1e-9, COARSE_STEP):
        for S in np.arange(s + MIN_GAP, SS_MAX + 1e-9, COARSE_STEP):
            out.append((round(float(s), 4), round(float(S), 4)))
    return out


def refine_grid(s0, S0, already):
    out = []
    for ds in (-REFINE_OFFSET, 0.0, REFINE_OFFSET):
        for dS in (-REFINE_OFFSET, 0.0, REFINE_OFFSET):
            if ds == 0.0 and dS == 0.0:
                continue                      # centre already evaluated
            s, S = round(s0 + ds, 4), round(S0 + dS, 4)
            if not (S_MIN <= s <= S_MAX):      continue
            if not (s + MIN_GAP - 1e-9 <= S <= SS_MAX): continue
            if (s, S) in already:              continue
            out.append((s, S))
    return out


def evaluate_candidate(s_norm, S_norm, train_df, params,
                       TelecomEnv, SSPolicy, B1Policy):
    """Run the 10 frozen calibration episodes for one (s,S) candidate."""
    pol = B1Policy(ss_policy=SSPolicy(s_norm=s_norm, S_norm=S_norm))
    eens, diesel, orders = [], [], []
    for seed in TUNING_SEEDS:
        for ep in range(EPS_PER_SEED):
            env = TelecomEnv(site_data=train_df, site_params=params,
                             episode_len=EPISODE_LEN, eval_mode=False,
                             lead_scenario=SCENARIO, seed=seed + ep,
                             init_inv_frac_low=INIT_LOW,
                             init_inv_frac_high=INIT_HIGH)
            obs, _ = env.reset(seed=seed + ep)
            done = False
            while not done:
                obs, _, term, trunc, _ = env.step(int(pol.act(obs, env=env)))
                done = term or trunc
            st = env.get_episode_stats()
            eens.append(float(st["EENS_kWh"]))
            diesel.append(float(st["diesel_kWh"]))
            orders.append(float(st["orders_placed"]))
            env.close()
    e, d, o = np.array(eens), np.array(diesel), np.array(orders)
    return {"s": s_norm, "S": S_norm,
            "eens_mean": float(e.mean()), "eens_sd": float(e.std(ddof=1)),
            "eens_min": float(e.min()),   "eens_max": float(e.max()),
            "diesel_mean": float(d.mean()), "diesel_sd": float(d.std(ddof=1)),
            "orders_mean": float(o.mean()), "orders_sd": float(o.std(ddof=1)),
            "n_episodes": int(e.size),
            # Episode-level values retained: candidates share (seed, ep)
            # conditions, so PAIRED differences between candidates are valid.
            "eens_episodes": [float(x) for x in e],
            "diesel_episodes": [float(x) for x in d],
            "orders_episodes": [float(x) for x in o]}


def _apply_rule(results):
    """The frozen selection rule applied to a candidate set.

    Returns (winner, diagnostics).  Reports WHICH criterion actually decided,
    rather than asserting the tie-break was used whenever the band held more
    than one candidate.
    """
    best = min(r["eens_mean"] for r in results)
    band = best * (1.0 + TIE_BAND) if best > 0 else 0.0
    tied = [r for r in results if r["eens_mean"] <= band + 1e-12]
    exact = [r for r in results if abs(r["eens_mean"] - best) <= 1e-12]
    ordered = sorted(tied, key=lambda r: (r["diesel_mean"], r["orders_mean"]))
    winner = ordered[0]

    if len(tied) == 1:
        criterion = "EENS alone"
    else:
        d_best = min(r["diesel_mean"] for r in tied)
        d_tied = [r for r in tied if abs(r["diesel_mean"] - d_best) <= 1e-9]
        if len(d_tied) == 1:
            criterion = "diesel (within the 1% EENS band)"
        else:
            o_best = min(r["orders_mean"] for r in d_tied)
            o_tied = [r for r in d_tied if abs(r["orders_mean"] - o_best) <= 1e-9]
            criterion = ("orders (EENS and diesel both tied)" if len(o_tied) == 1
                         else "UNRESOLVED NUMERICAL TIE across "
                              f"{len(o_tied)} candidates -- first candidate in "
                              "deterministic enumeration order, selected only "
                              "after all prespecified performance criteria "
                              "remain tied")
    return winner, {
        "best_eens_mean": best,
        "candidates_in_1pct_band": len(tied),
        "candidates_exactly_at_minimum": len(exact),
        "selection_resolved_by": criterion,
        "saturated_at_zero": bool(best <= 1e-12),
    }


def select(results):
    """Winner, plus the runner-up defined as the candidate that would win if
    the winner were removed -- i.e. the SAME rule re-applied, not an EENS sort.
    """
    winner, tie = _apply_rule(results)
    rest = [r for r in results if r is not winner]
    runner = _apply_rule(rest)[0] if rest else None
    return winner, runner, tie


def reference_variants(site, train_df, params, TelecomEnv, SSPolicy, B1Policy):
    """B1-Original and B1-Analytic on the SAME ten calibration conditions.

    DIAGNOSTIC ONLY -- these never enter B1* candidate selection.

    B1-Original : SSPolicy() dataclass defaults -- what evaluate.py:494 and :519
                  actually construct for B1, MPC and Oracle-MPC.
    B1-Analytic : SSPolicy.from_site_params(...) -- the rule the thesis §4.5
                  Implementation Note DESCRIBES.  Because C_tank = 72*d_bar, the
                  d_bar term cancels and s = E[L]/72 for every site.
    """
    probe = TelecomEnv(site_data=train_df, site_params=params, episode_len=EPISODE_LEN,
                       eval_mode=False, lead_scenario=SCENARIO, seed=TUNING_SEEDS[0])
    analytic = SSPolicy.from_site_params(site_params=params, d_bar=probe.d_bar,
                                         lead_p=probe.lead_p,
                                         tank_cap_kwh=probe.tank_cap_kwh)
    probe.close()
    default = SSPolicy()
    out = {}
    for label, ss in (("B1_Original", default), ("B1_Analytic", analytic)):
        r = evaluate_candidate(ss.s_norm, ss.S_norm, train_df, params,
                               TelecomEnv, SSPolicy, B1Policy)
        r["source"] = ("SSPolicy() dataclass defaults -- as executed by "
                       "evaluate.py for B1/MPC/Oracle-MPC"
                       if label == "B1_Original" else
                       "SSPolicy.from_site_params() -- the rule documented in "
                       "thesis §4.5 but NOT used by B1/MPC/Oracle-MPC")
        out[label] = r
    return out


def tune_site(site, repo, TelecomEnv, SSPolicy, B1Policy,
              load_site, train_test_split):
    df, params = load_site(str(repo / "data" / "processed" / f"{site}.csv"))
    train_df, test_df = train_test_split(df)
    assert len(train_df) == 1080 and len(test_df) == 360, f"{site}: bad split"
    del test_df                       # the test window is never opened

    t0 = time.perf_counter()
    coarse = coarse_grid()
    res = [evaluate_candidate(s, S, train_df, params,
                              TelecomEnv, SSPolicy, B1Policy) for s, S in coarse]
    # Refinement is centred on the candidate the FROZEN RULE would pick from
    # the coarse grid -- not on the raw EENS minimum, which ignores the 1% band.
    c_best, c_tie = _apply_rule(res)
    refine = refine_grid(c_best["s"], c_best["S"], set(coarse))
    res += [evaluate_candidate(s, S, train_df, params,
                               TelecomEnv, SSPolicy, B1Policy) for s, S in refine]

    refs = reference_variants(site, train_df, params, TelecomEnv, SSPolicy, B1Policy)
    winner, runner, tie = select(res)

    # PAIRED winner-vs-runner-up difference.  Candidates share (seed, ep)
    # conditions, so episode i is a matched observation.  The previous version
    # divided by the winner's own SD, which is the SE of its mean -- not the SE
    # of the difference.  That statistic is withdrawn.
    paired = None
    if runner is not None:
        d = np.array(winner["eens_episodes"]) - np.array(runner["eens_episodes"])
        sd = float(d.std(ddof=1))
        paired = {
            "metric": "EENS (winner - runner_up), paired over the 10 calibration episodes",
            "mean_difference": float(d.mean()),
            "sd_of_differences": sd,
            "n_pairs": int(d.size),
            "all_differences": [float(x) for x in d],
            "identical_on_every_episode": bool(np.all(np.abs(d) < 1e-12)),
        }
        # EENS identity does NOT imply identical dispatch/ordering behaviour.
        # The stored diesel and order arrays let us check that separately.
        dd = np.array(winner["diesel_episodes"]) - np.array(runner["diesel_episodes"])
        do = np.array(winner["orders_episodes"]) - np.array(runner["orders_episodes"])
        paired["diesel_identical_on_every_episode"] = bool(np.all(np.abs(dd) < 1e-9))
        paired["orders_identical_on_every_episode"] = bool(np.all(np.abs(do) < 1e-9))
        paired["orders_mean_difference"] = float(do.mean())
        paired["behaviourally_identical"] = bool(
            paired["identical_on_every_episode"]
            and paired["diesel_identical_on_every_episode"]
            and paired["orders_identical_on_every_episode"])
        paired["interpretation"] = (
            "winner and runner-up are indistinguishable on all three recorded "
            "metrics" if paired["behaviourally_identical"] else
            "EENS matches but ordering/dispatch behaviour differs -- the "
            "configurations are NOT equivalent, only equally reliable here")
        if sd > 1e-12:
            paired["mean_over_sd_of_differences"] = round(float(d.mean() / sd), 3)

    return {
        "site": site,
        "selected": {"s_norm": winner["s"], "S_norm": winner["S"]},
        "selected_stats": winner,
        "runner_up": runner,
        "paired_winner_vs_runner_up": paired,
        "tie_info": tie,
        "reference_variants": refs,
        "vs_references_IN_SAMPLE": {
            "note": "IN-SAMPLE: B1* was selected using these same ten episodes. "
                    "Not evidence of held-out improvement -- that requires the "
                    "days 46-60 evaluation.",
            "B1_Original_eens_mean": refs["B1_Original"]["eens_mean"],
            "B1_Analytic_eens_mean": refs["B1_Analytic"]["eens_mean"],
            "B1_star_eens_mean": winner["eens_mean"],
        },
        "n_candidates": len(res),
        "n_coarse": len(coarse), "n_refine": len(refine),
        "coarse_optimum": {"s_norm": c_best["s"], "S_norm": c_best["S"]},
        "coarse_tie_info": c_tie,
        "refinement_moved_optimum": (c_best["s"], c_best["S"]) != (winner["s"], winner["S"]),
        "seconds": round(time.perf_counter() - t0, 1),
        "all_candidates": sorted(res, key=lambda r: (r["eens_mean"], r["diesel_mean"])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--sites", default="all",
                    help="'all' or a comma-separated list, e.g. site1,site5")
    ap.add_argument("--out", type=Path,
                    default=Path("results/phase3r/b1_star_calibration.json"))
    a = ap.parse_args()

    repo = a.repo.expanduser().resolve()
    sys.path.insert(0, str(repo / "src"))
    os.chdir(repo)
    from env.data_loader import load_site, train_test_split
    from env.telecom_env import TelecomEnv
    from baselines.s_S_policy import SSPolicy, B1Policy

    q_verified = assert_fixed_order_quantities(SSPolicy)
    print(f"order sizes verified against live defaults: "
          f"q_small={q_verified['q_small_pct']}, q_large={q_verified['q_large_pct']}")

    sites = ([f"site{i}" for i in range(1, 11)] if a.sites == "all"
             else [x.strip() for x in a.sites.split(",")])

    print(f"B1* calibration | {len(coarse_grid())} coarse candidates | "
          f"{len(TUNING_SEEDS)*EPS_PER_SEED} episodes each | training window only\n")
    out_sites, t0 = [], time.perf_counter()
    for site in sites:
        r = tune_site(site, repo, TelecomEnv, SSPolicy, B1Policy,
                      load_site, train_test_split)
        out_sites.append(r)
        w, t = r["selected_stats"], r["tie_info"]
        pw = r["paired_winner_vs_runner_up"]
        flag = "  <-- IDENTICAL to runner-up" if (pw and pw["identical_on_every_episode"]) else ""
        print(f"  {site:7s} s={w['s']:.2f} S={w['S']:.2f}  "
              f"EENS={w['eens_mean']:8.3f}+-{w['eens_sd']:6.3f}  "
              f"diesel={w['diesel_mean']:7.1f}  "
              f"tied@min={t['candidates_exactly_at_minimum']:2d}  "
              f"B1orig={r['reference_variants']['B1_Original']['eens_mean']:8.3f}  "
              f"B1anal={r['reference_variants']['B1_Analytic']['eens_mean']:8.3f}  "
              f"({r['seconds']:.0f}s){flag}")

    report = {
        "status": "OK",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(repo),
        "protocol": "PHASE3R_PROTOCOL.md §6.3 (frozen 2026-10-22)",
        "frozen_settings": {
            "scenario": SCENARIO, "episode_len": EPISODE_LEN,
            "eval_mode": False, "data": "df_train (days 1-45) ONLY",
            "init_inv_frac": [INIT_LOW, INIT_HIGH],
            "tuning_seeds": list(TUNING_SEEDS),
            "episodes_per_seed": EPS_PER_SEED,
            "objective": "min mean EENS",
            "tie_break": "within 1% EENS band -> lower diesel -> fewer orders",
            "final_fallback": "first candidate in deterministic enumeration "
                              "order, used ONLY when all prespecified "
                              "performance criteria remain tied",
            "coarse_step": COARSE_STEP, "refine_offset": REFINE_OFFSET,
            "constraints": f"S >= s+{MIN_GAP}, s in [{S_MIN},{S_MAX}], S <= {SS_MAX}",
        },
        "q_parameters": {**q_verified, "tuned": False,
                         "note": "Order SIZE is a fixed % of tank capacity. S "
                                 "selects between small and large via gap = S - inv "
                                 "crossing q_large_pct; it is NOT an order-up-to level."},
        "file_hashes": {str(p.relative_to(repo)): _sha256(p) for p in
                        [repo/"src"/"env"/"telecom_env.py",
                         repo/"src"/"baselines"/"s_S_policy.py",
                         repo/"src"/"config"/"hparams.yaml"] if p.exists()},
        "test_window_accessed": False,
        "sites": out_sites,
        "total_seconds": round(time.perf_counter() - t0, 1),
    }
    out = a.out if a.out.is_absolute() else repo / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Compact loader -- THREE labelled variants, not one
    variants = {
        "B1_Original": {r["site"]: {
            "s_norm": r["reference_variants"]["B1_Original"]["s"],
            "S_norm": r["reference_variants"]["B1_Original"]["S"]} for r in out_sites},
        "B1_Analytic": {r["site"]: {
            "s_norm": r["reference_variants"]["B1_Analytic"]["s"],
            "S_norm": r["reference_variants"]["B1_Analytic"]["S"]} for r in out_sites},
        "B1_star_Deployable": {r["site"]: r["selected"] for r in out_sites},
    }
    variants["_q_parameters_fixed"] = {**q_verified,
                                       "note": "NOT tuned in this study"}
    pfile = out.with_name("b1_star_params.json")
    pfile.write_text(json.dumps(variants, indent=2), encoding="utf-8")

    # Smoke test: can MPC and Oracle-MPC actually be constructed with these?
    from baselines.mpc_policy import MPCDispatchB1Policy, OracleMPCPolicy
    probe_site = out_sites[0]["site"]
    sp = variants["B1_star_Deployable"][probe_site]
    ss_loaded = SSPolicy(s_norm=sp["s_norm"], S_norm=sp["S_norm"])
    checks = []
    for name, ctor in (("MPC", lambda: MPCDispatchB1Policy(forecast_cache=None,
                                                           ss_policy=ss_loaded,
                                                           horizon=24, verbose=False)),
                       ("Oracle-MPC", lambda: OracleMPCPolicy(ss_policy=ss_loaded,
                                                              verbose=False))):
        p = ctor()
        got = getattr(p, "ss_policy", None) or getattr(p, "ss", None)
        ok = (got is not None and abs(got.s_norm - sp["s_norm"]) < 1e-12
              and abs(got.S_norm - sp["S_norm"]) < 1e-12)
        checks.append({"policy": name, "site": probe_site,
                       "expected": sp,
                       "observed": ({"s_norm": got.s_norm, "S_norm": got.S_norm}
                                    if got is not None else None),
                       "PASS": bool(ok)})
        if not ok:
            raise AssertionError(f"{name} did not retain the loaded (s,S)")
    report["propagation_smoke_test"] = checks
    report["WARNING_evaluate_py"] = (
        "evaluate.py:494 and :519 currently hardcode SSPolicy() defaults. It must "
        "be modified to load b1_star_params.json before propagation can take "
        "effect -- the smoke test proves the plumbing works, not that it is wired.")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n  full record : {out}")
    print(f"  params only : {pfile}")
    print(f"  total       : {report['total_seconds']:.0f}s")


if __name__ == "__main__":
    main()
