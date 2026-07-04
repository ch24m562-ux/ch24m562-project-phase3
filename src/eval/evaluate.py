"""    eval/evaluate.py — Unified evaluation for all policies

Works identically for:
  RL-Inv (MaskablePPO on TelecomEnv, Discrete(6))
  B0: RuleBasedPolicy (heuristic)
  B1: B1Policy (s,S + heuristic DG)
  B2: Track B (DispatchOnlyEnv + PPO dispatch, Discrete(2))

FIX (Bug 1): Track B env_factory now uses episode_len=720, matching Track A.
  Previously hardcoded to EVAL_EP_LEN=360 — all cost/diesel comparisons invalid.

FIX (Bug 2): RL policies were trained with VecNormalize(norm_obs=True).
  Uses VecNormObsWrapper: loads VecNormalize via VecNormalize.load() (proper
  SB3 path) then calls vn.normalize_obs() on each obs before passing to policy.

  Why NOT running through VecNormalize.step():
    DummyVecEnv calls env.reset() INSIDE vn.step() synchronously when done=True,
    before vn.step() returns. This wipes TelecomEnv.ep_info_log before
    get_episode_stats() can ever be called. The episode must run through the
    plain base env; normalization is applied obs-only via vn.normalize_obs().

  Track A pkl: 9D obs stats (runs/track_a/siteN_vecnormalize.pkl)
  Track B pkl: 6D obs stats (runs/track_b_fixed/siteN_vecnormalize.pkl)
  Baselines (B0/B1): no wrapper — not trained with VecNormalize.
"""
from __future__ import annotations

import os
import re
import time
from typing import Callable, List, Optional, Any
import numpy as np
import pandas as pd
import gymnasium as gym
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── VecNormObsWrapper (Bug 2 fix) ─────────────────────────────────────────────

class VecNormObsWrapper(gym.Wrapper):
    """
    Normalize observations using saved VecNormalize stats, without running
    the episode through VecNormalize.step().

    Approach:
      1. Load VecNormalize from pkl via VecNormalize.load() — proper SB3 path.
      2. Set training=False so obs_rms is frozen.
      3. In reset() and step(), call self._vn.normalize_obs(obs) to apply the
         same normalization the policy saw during training.
      4. The episode is driven by self.env.step() (plain TelecomEnv / wrapper),
         NOT by VecNormalize.step() — so ep_info_log is never auto-wiped.
      5. get_episode_stats() reads from the base env normally after the episode.

    This satisfies:
      - Proper SB3 VecNormalize.load() path (not manual formula).
      - Correct episode stats collection (env never auto-reset under us).
      - Separate pkls for Track A (9D) and Track B (6D).
    """

    def __init__(self, base_env: gym.Env, vecnorm_path: str):
        super().__init__(base_env)
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        # DummyVecEnv needed only to satisfy VecNormalize.load() interface.
        # We will NEVER call self._vn.step() — only self._vn.normalize_obs().
        _dummy = DummyVecEnv([lambda: base_env])
        # ── Shape check — catch obs dimension mismatches early ────────────
        vn_temp = VecNormalize.load(vecnorm_path, _dummy)
        if vn_temp.observation_space.shape != base_env.observation_space.shape:
            print(f"[WARN] VecNorm shape {vn_temp.observation_space.shape} "
                  f"!= env shape {base_env.observation_space.shape}")
        # ─────────────────────────────────────────────────────────────────
    
        self._vn: VecNormalize = vn_temp
        self._vn.training = False     # freeze: do not update running stats
        self._vn.norm_reward = False  # reward normalization off

        # Observation space now matches the normalized range the policy expects
        self.observation_space = self._vn.observation_space

    def _norm(self, obs: np.ndarray) -> np.ndarray:
        """Apply VecNormalize's own normalize_obs() to a single obs vector."""
        # normalize_obs expects shape (n_envs, obs_dim); we pass (1, obs_dim)
        normed = self._vn.normalize_obs(obs.reshape(1, -1))
        return normed[0].astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._norm(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._norm(obs), reward, terminated, truncated, info

    # Pass-through: action mask lives on the base env
    def get_action_mask(self) -> np.ndarray:
        cur = self.env
        for _ in range(8):
            if hasattr(cur, "get_action_mask"):
                return cur.get_action_mask()
            cur = getattr(cur, "env", None) or getattr(cur, "base_env", None)
            if cur is None:
                break
        raise AttributeError("No get_action_mask() found in env chain")

    # Pass-through: stats come from TelecomEnv which is NOT auto-reset here
    def get_episode_stats(self) -> dict:
        cur = self.env
        for _ in range(8):
            if hasattr(cur, "get_episode_stats"):
                return cur.get_episode_stats() or {}
            cur = getattr(cur, "env", None) or getattr(cur, "base_env", None)
            if cur is None:
                break
        return {}


# ── Metric computation ────────────────────────────────────────────────────────

def compute_metrics(ep_stats: dict, rc: dict) -> dict:
    diesel_kwh  = float(ep_stats.get("diesel_kWh", 0.0))
    grid_kwh    = float(ep_stats.get("grid_kWh",   0.0))
    alpha       = float(rc.get("alpha", 1.0))
    beta        = float(rc.get("beta",  0.4))
    steps       = int(ep_stats.get("steps", 720))
    outage_hrs  = int(ep_stats.get("outage_hours", 0))
    uptime_pct  = 100.0 * (1.0 - outage_hrs / max(steps, 1))
    return {
        "EENS_kWh":        float(ep_stats.get("EENS_kWh", 0.0)),
        "outage_hours":    outage_hrs,
        "uptime_pct":      round(uptime_pct, 4),
        "diesel_kWh":      diesel_kwh,
        "stockout_events": int(ep_stats.get("stockout_events", 0)),
        "orders_placed":   int(ep_stats.get("orders_placed", 0)),
        "violations":      int(ep_stats.get("violations", 0)),
        "mean_soc":        float(ep_stats.get("mean_soc", 0.0)),
        "mean_inv_pct":    float(ep_stats.get("mean_inv_pct", 0.0)),
        "min_inv_pct":     float(ep_stats.get("min_inv_pct", 0.0)),
        "dg_on_fraction":  float(ep_stats.get("dg_on_fraction", 0.0)),
        "grid_kWh":        grid_kwh,
        "cost_proxy":      float(alpha * diesel_kwh + beta * grid_kwh),
    }


# ── Policy action helpers ─────────────────────────────────────────────────────

def _predict_with_optional_mask(policy, obs, mask):
    if mask is None:
        a, _ = policy.predict(obs, deterministic=True)
        return int(np.asarray(a).item())
    try:
        a, _ = policy.predict(obs, action_masks=mask, deterministic=True)
        return int(np.asarray(a).item())
    except TypeError:
        a, _ = policy.predict(obs, deterministic=True)
        return int(np.asarray(a).item())


def _is_mpc_policy(policy) -> bool:
    """True for MPC/Oracle policies which need env-aware predict() signature."""
    return type(policy).__name__ in ("MPCDispatchB1Policy", "OracleMPCPolicy")


def _policy_act(policy, obs, env, site_id=None):
    # MPC policies need env access for site params, forecasts, and current step.
    # site_id is passed explicitly from the eval loop (args.site / loop variable)
    # since TelecomEnv does not store a site_id attribute on itself.
    if _is_mpc_policy(policy):
        base = env
        while hasattr(base, "env"):
            base = base.env
        t = getattr(base, "_t_idx", None)   # TelecomEnv attribute (local index into eval slice)
        try:
            a, _ = policy.predict(obs, env=env, site_id=site_id, t=t)
        except KeyError as e:
            # mpc_forecast raises KeyError on a forecast-cache miss by design
            # (see mpc_policy.py MPCDispatchB1Policy._get_forecasts) -- surface
            # this clearly at the point of failure rather than letting it
            # propagate as a bare traceback mid-grid-run.
            raise RuntimeError(
                f"[evaluate.py] MPC-forecast policy failed at site_id={site_id!r} "
                f"t={t}: {e}\n"
                f"This stops the run rather than silently falling back to "
                f"persistence -- re-check the --forecast_cache path and that "
                f"it was generated for this site/scenario before retrying."
            ) from e
        return int(a)

    mask = None
    if hasattr(env, "get_action_mask"):
        try:
            mask = env.get_action_mask()
        except Exception:
            mask = None
    if hasattr(policy, "predict"):
        return _predict_with_optional_mask(policy, obs, mask)
    if hasattr(policy, "act"):
        try:
            return int(policy.act(obs, env=env))
        except TypeError:
            return int(policy.act(obs, None, env))
    raise ValueError("Policy must implement .predict() (SB3) or .act() (baseline).")


def _get_stats_from_env(env, last_info: dict = None) -> dict:
    """Retrieve episode stats with two-layer fallback.

    Layer 1: env.get_episode_stats() — works when episode runs through plain env.
    Layer 2: last_info["episode_stats"] — works when VecNormalize auto-reset
             wiped ep_info_log before get_episode_stats() could be called.
             TelecomEnv.step() stashes stats into info on the terminal step.
    """
    # Layer 1: try env chain directly
    stats = {}
    if hasattr(env, "get_episode_stats"):
        try:
            stats = env.get_episode_stats() or {}
        except Exception:
            stats = {}

    if not stats:
        cur = env
        for _ in range(10):
            if hasattr(cur, "get_episode_stats"):
                try:
                    stats = cur.get_episode_stats() or {}
                except Exception:
                    stats = {}
                break
            if hasattr(cur, "env"):
                cur = cur.env
            elif hasattr(cur, "unwrapped"):
                cur = cur.unwrapped
            else:
                break

    # Layer 2: fallback to info["episode_stats"] stashed at terminal step
    if not stats and last_info and isinstance(last_info, dict):
        stats = last_info.get("episode_stats", {})

    return stats


# ── Evaluation core ─────────────────────────────────────────────────────────

def evaluate(
    policy,
    env_factory: Callable,
    n_episodes: int = 5,
    sites: Optional[List[str]] = None,
    leads: Optional[List[str]] = None,
    seed: int = 42,
    rc: Optional[dict] = None,
    verbose: bool = True,
    trace_out: str = "",
    meta: Optional[dict] = None,
) -> pd.DataFrame:
    if sites is None:
        sites = ["site1", "site7", "site5"]
    if leads is None:
        leads = ["normal", "delayed"]
    if rc is None:
        rc = {"alpha": 1.0, "beta": 0.4}

    rows = []
    t0 = time.time()

    for site in sites:
        for lead in leads:
            for ep in range(n_episodes):
                env = env_factory(site, lead, seed + ep)
                obs, info = env.reset(seed=seed + ep)

                if hasattr(policy, "reset"):
                    try:
                        policy.reset()
                    except Exception:
                        pass

                terminated = truncated = False
                last_info = {}
                _trace_this = trace_out and (ep == 0) and (site == sites[0]) and (lead == leads[0])
                _trace = {"inv_pct": [], "unmet_kwh": [], "dg_on": [],
                          "grid_avail": [], "soc": [],
                          "pipe_pct": [], "order_kwh": [],
                          "supplier_disrupted": [], "ewma_lead_h": [],
                          "delivery_in_hours": []}
                while not (terminated or truncated):
                    a = _policy_act(policy, obs, env, site_id=site)
                    obs, r, terminated, truncated, info = env.step(a)
                    last_info = info
                    if _trace_this:
                        _trace["inv_pct"].append(float(info.get("inv_pct", 0.0)))
                        _trace["unmet_kwh"].append(float(info.get("unmet_kwh", 0.0)))
                        _trace["dg_on"].append(float(info.get("dg_on", 0.0)))
                        # grid_avail derived from p_grid_kwh (>0 means grid available)
                        _trace["grid_avail"].append(1.0 if float(info.get("p_grid_kwh", 0.0)) > 0 else 0.0)
                        _trace["soc"].append(float(info.get("soc", 0.5)))
                        # pipeline and order fields for Fig 6
                        tank = float(info.get("tank_cap_kwh", 1.0))
                        _trace["pipe_pct"].append(float(info.get("pending_qty_kwh", 0.0)) / max(tank, 1e-9))
                        _trace["order_kwh"].append(float(info.get("order_qty_kwh", 0.0)))
                        # [CHANGE 11] RLInv-DA extension trace fields
                        _trace["supplier_disrupted"].append(float(info.get("supplier_disrupted", 0.0)))
                        _trace["ewma_lead_h"].append(float(info.get("ewma_lead_h", 0.0)))
                        _trace["delivery_in_hours"].append(float(info.get("delivery_in_hours", -1.0)))
                if _trace_this and trace_out:
                    os.makedirs(os.path.dirname(trace_out) or ".", exist_ok=True)
                    np.savez(trace_out, **{k: np.array(v) for k, v in _trace.items()})
                    print(f"[TRACE] Saved {len(_trace['inv_pct'])} steps → {trace_out}")

                ep_stats = _get_stats_from_env(env, last_info)
                metrics = compute_metrics(ep_stats, rc)
                row = {
                    "site": site,
                    "lead_scenario": lead,
                    "episode": ep,
                    "steps": int(ep_stats.get("steps", 0)),
                    "init_inv_frac": float(ep_stats.get("init_inv_frac", 0.6)),
                    **metrics,
                    **(meta or {}),
                }
                rows.append(row)

                if verbose:
                    print(f"[{site} | {lead} | ep={ep}] "
                          f"EENS={row['EENS_kWh']:.2f} "
                          f"stockouts={row['stockout_events']} "
                          f"diesel={row['diesel_kWh']:.1f} "
                          f"cost={row['cost_proxy']:.1f}")

                env.close()

    df = pd.DataFrame(rows)
    if verbose:
        print(f"Evaluation done: {len(df)} episodes in {time.time() - t0:.1f}s")
    return df


# ── H1 verdict helpers ──────────────────────────────────────────────────────

def _agg(df):
    return {
        "EENS_kWh":       float(df["EENS_kWh"].mean()),
        "stockout_events": float(df["stockout_events"].mean()),
        "cost_proxy":      float(df["cost_proxy"].mean()),
    }


def h1_verdict(df_rl_inv, df_track_b, eps=0.10):
    a = _agg(df_rl_inv)
    b = _agg(df_track_b)

    def rel_improve(x_ref, x_new):
        return (x_ref - x_new) / max(abs(x_ref), 1e-9)

    eens_impr = rel_improve(b["EENS_kWh"], a["EENS_kWh"])
    so_impr   = rel_improve(b["stockout_events"], a["stockout_events"])
    return {
        "confirmed": bool((eens_impr >= eps) and (so_impr >= eps)),
        "rejected":  bool((abs(eens_impr) <= eps) and (abs(so_impr) <= eps)),
        "eens_improvement": float(eens_impr),
        "stockout_improvement": float(so_impr),
        "rl_inv":  a,
        "track_b": b,
    }


def h1_verdict_by_lead(df_rl_inv, df_track_b, eps=0.10):
    rows = []
    for lead in sorted(
        set(df_rl_inv["lead_scenario"].unique()) |
        set(df_track_b["lead_scenario"].unique())
    ):
        a = df_rl_inv[df_rl_inv["lead_scenario"] == lead]
        b = df_track_b[df_track_b["lead_scenario"] == lead]
        if len(a) == 0 or len(b) == 0:
            continue
        v = h1_verdict(a, b, eps=eps)
        rows.append({
            "lead_scenario":      lead,
            "confirmed":          v["confirmed"],
            "rejected":           v["rejected"],
            "eens_improvement":   v["eens_improvement"],
            "stockout_improvement": v["stockout_improvement"],
            "rl_EENS":      v["rl_inv"]["EENS_kWh"],
            "b2_EENS":      v["track_b"]["EENS_kWh"],
            "rl_stockouts": v["rl_inv"]["stockout_events"],
            "b2_stockouts": v["track_b"]["stockout_events"],
            "rl_cost":      v["rl_inv"]["cost_proxy"],
            "b2_cost":      v["track_b"]["cost_proxy"],
        })
    return pd.DataFrame(rows)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from env.telecom_env import TelecomEnv
    from env.data_loader import load_site, train_test_split

    parser = argparse.ArgumentParser()
    parser.add_argument("--site",         required=True)
    parser.add_argument("--lead",         default="normal",
                        choices=["no_delay", "fast", "normal", "delayed",
                                 "monsoon", "very_delayed", "extreme", "multi"])
    parser.add_argument("--model_path",   default="")
    parser.add_argument("--vecnorm_path", default="",
                        help="path to vecnormalize.pkl (auto-detected if omitted)")
    parser.add_argument("--episodes",     type=int, default=5)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--out_csv",      default="")
    parser.add_argument("--policy_type",  default="rl", choices=["rl", "b0", "b1", "mpc", "oracle"])
    parser.add_argument("--forecast_cache", type=str, default="",
                        help="Path to forecast cache pkl for MPC (omit for persistence MPC)")
    parser.add_argument("--mpc_horizon",   type=int, default=24,
                        help="MPC planning horizon in hours (default 24)")
    parser.add_argument("--algo",         default="maskable", choices=["maskable", "ppo"])
    parser.add_argument("--env_type", default="track_a",
                        choices=["track_a", "track_b", "a5", "a6"])
    parser.add_argument("--episode_len",  type=int, default=360,
                        help="Evaluation episode length (default 360 = full test split)")
    parser.add_argument("--init_diesel_low",  type=float, default=0.3)
    parser.add_argument("--init_diesel_high", type=float, default=0.9)
    parser.add_argument("--trace_out", type=str, default="",
                        help="Save per-step trace for episode 0 to this .npz path.")
    # ── Phase 3 env args (pass-through to TelecomEnv) ────────────────────────
    parser.add_argument("--lead_dist",    type=str, default="geometric",
                        choices=["geometric", "lognormal", "weibull"],
                        help="Lead time distribution")
    parser.add_argument("--lead_sigma",   type=float, default=0.5,
                        help="Lognormal sigma shape parameter (default 0.5)")
    parser.add_argument("--lead_k", type=float, default=2.0,
                    help="Weibull shape parameter k (default 2.0, ignored for other dists)")
    parser.add_argument("--tank_scale",   type=float, default=1.0,
                        help="Tank capacity multiplier (1.0=72h base)")
    parser.add_argument("--stochastic_grid", action="store_true",
                        help="Use Markov chain grid outage instead of dataset")
    parser.add_argument("--no_time_enc",  action="store_true",
                        help="Disable time encoding ablation")
    parser.add_argument("--use_eta_obs",  action="store_true",
                        help="ETA-aware extension: agent sees delivery_remaining_n")
    parser.add_argument("--use_supplier_regime", action="store_true",
                        help="[CHANGE 11] Two-state Markov supplier regime (RLInv-DA)")
    parser.add_argument("--use_ewma_lead", action="store_true",
                        help="[CHANGE 11] EWMA lead-time estimate as obs dim 12 (RLInv-DA)")
    # ── Metadata columns written into every CSV row ───────────────────────────
    parser.add_argument("--policy_label",    type=str, default="")
    parser.add_argument("--train_scenario",  type=str, default="normal",
                        choices=["no_delay", "fast", "normal", "delayed",
                                 "monsoon", "very_delayed", "extreme", "multi"])
    parser.add_argument("--experiment_tag",  type=str, default="main")
    parser.add_argument("--train_steps",     type=int, default=0)
    args = parser.parse_args()

    # ── Auto-detect vecnorm_path ──────────────────────────────────────────────
    vecnorm_path = args.vecnorm_path
    if not vecnorm_path and args.policy_type == "rl" and args.model_path:
        # Strip known model filename suffixes to get the base path
        base = (args.model_path
                .replace("_final_model.zip", "").replace("_final_model", "")
                .replace("_best/best_model.zip", "").replace("_best/best_model", ""))
        # Phase 3 pattern: site5_s42_final.zip → site5_s42
        base = re.sub(r"_final\.zip$", "", base)
        base = re.sub(r"_final$",      "", base)
        # Try new Phase 3 suffix first (_vecnorm.pkl), then Phase 2 suffix (_vecnormalize.pkl)
        for suffix in ["_vecnorm.pkl", "_vecnormalize.pkl"]:
            candidate = base + suffix
            if os.path.exists(candidate):
                vecnorm_path = candidate
                print(f"[INFO] Auto-detected vecnorm_path: {vecnorm_path}")
                break
        if not vecnorm_path:
            print(f"[WARN] Could not auto-detect vecnorm_path for: {args.model_path}")

    # ── Load policy ───────────────────────────────────────────────────────────
    if args.policy_type == "rl":
        if not args.model_path:
            raise ValueError("RL policy requires --model_path")
        if args.algo == "maskable":
            from sb3_contrib import MaskablePPO
            policy = MaskablePPO.load(args.model_path)
        else:
            from stable_baselines3 import PPO
            policy = PPO.load(args.model_path)
    elif args.policy_type == "b0":
        from baselines.rule_based import RuleBasedPolicy
        policy = RuleBasedPolicy()
        vecnorm_path = ""   # baselines not trained with VecNormalize
    elif args.policy_type == "b1":
        from baselines.s_S_policy import B1Policy, SSPolicy
        policy = B1Policy(ss_policy=SSPolicy())
        vecnorm_path = ""   # baselines not trained with VecNormalize
    elif args.policy_type in ("mpc", "oracle"):
        from baselines.s_S_policy import SSPolicy
        from baselines.mpc_policy import MPCDispatchB1Policy, OracleMPCPolicy

        # Load forecast cache if provided (optional -- falls back to persistence)
        forecast_cache = None
        if args.forecast_cache and os.path.exists(args.forecast_cache):
            import pickle
            with open(args.forecast_cache, "rb") as fh:
                _raw = pickle.load(fh)
            # train_forecast.py saves {"cache": {...}, "horizon":, "train_days":, "sites":}
            # — unwrap to the bare {site_id: {t: {...}}} dict MPCDispatchB1Policy expects.
            forecast_cache = _raw["cache"] if isinstance(_raw, dict) and "cache" in _raw else _raw
            cache_horizon = _raw.get("horizon", args.mpc_horizon) if isinstance(_raw, dict) else args.mpc_horizon
            if cache_horizon != args.mpc_horizon:
                print(f"[WARN] cache horizon={cache_horizon} != --mpc_horizon={args.mpc_horizon}; using {args.mpc_horizon}")
            print(f"[INFO] Loaded forecast cache: {args.forecast_cache} ({len(forecast_cache)} sites)")
        elif args.forecast_cache:
            print(f"[WARN] forecast_cache not found: {args.forecast_cache} -- using persistence")
        else:
            print("[INFO] MPC using persistence forecast (MPC-H24-Persistence-B1)")

        # SSPolicy: same instance as B1 and A6 for fair comparison
        ss = SSPolicy()

        if args.policy_type == "oracle":
            policy = OracleMPCPolicy(ss_policy=ss, verbose=False)
            print("[INFO] Policy: Oracle-MPC-B1 (perfect forecast, H=360)")
        else:
            policy = MPCDispatchB1Policy(
                forecast_cache=forecast_cache,
                ss_policy=ss,
                horizon=args.mpc_horizon,
                verbose=False,
            )
            fc_label = "Forecast" if forecast_cache is not None else "Persistence"
            print(f"[INFO] Policy: MPC-H{args.mpc_horizon}-{fc_label}-B1")

        vecnorm_path = ""   # MPC/Oracle are not RL policies -- no VecNormalize

    # ── Env factory ───────────────────────────────────────────────────────────
    def env_factory(site: str, lead: str, seed: int):
        site_csv = f"data/processed/{site}.csv"

        if args.env_type == "track_b":
            # FIX (Bug 1): episode_len=720 so Track B matches Track A horizon
            from train.train_track_b import make_track_b_eval_env
            base_env = make_track_b_eval_env(
                site_csv, seed=seed, lead_scenario=lead,
                episode_len=args.episode_len,
                init_inv_frac_low=args.init_diesel_low,
                init_inv_frac_high=args.init_diesel_high,
            )
        elif args.env_type == "a6":
            # A6: RL controls DG only; ordering fixed by calibrated (s,S) policy.
            # make_a6_env builds A6Env (TelecomEnv subclass) + calibrates SSPolicy
            # from site_params. VecNormObsWrapper applied below — same pkl as RLInv
            # (both use 9D obs from TelecomEnv; A6Env does not change obs space).
            from env.data_loader import load_site, train_test_split
            from env.a6_env import make_a6_env
            df, params = load_site(site_csv)
            _df_train, df_test = train_test_split(df)
            base_env = make_a6_env(
                site_data=df_test,
                site_params=params,
                lead_scenario=lead,
                episode_len=args.episode_len,
                eval_mode=True,
                seed=seed,
                init_inv_frac_low=args.init_diesel_low,
                init_inv_frac_high=args.init_diesel_high,
            )
        else:
            from env.data_loader import load_site, train_test_split
            df, params = load_site(site_csv)
            _df_train, df_test = train_test_split(df)
            base_env = TelecomEnv(
                site_data=df_test, site_params=params,
                episode_len=args.episode_len,
                eval_mode=True, lead_scenario=lead, seed=seed,
                init_inv_frac_low=args.init_diesel_low,
                init_inv_frac_high=args.init_diesel_high,
                tank_scale=args.tank_scale,
                use_time_encoding=not args.no_time_enc,
                lead_distribution=args.lead_dist,
                lead_sigma=args.lead_sigma,
                lead_k=args.lead_k,
                use_stochastic_grid=args.stochastic_grid,
                use_eta_obs=args.use_eta_obs,
                use_supplier_regime=args.use_supplier_regime,
                use_ewma_lead=args.use_ewma_lead,
            )
            if args.env_type == "a5":
                from env.obs_wrappers import NoInvObsWrapper
                base_env = NoInvObsWrapper(base_env)

        # FIX (Bug 2): RL policies only — wrap with VecNormObsWrapper.
        # Uses VecNormalize.load() then vn.normalize_obs() on each obs.
        # Episode runs through plain env so get_episode_stats() works.
        # Track A pkl is 9D; Track B pkl is 6D — never mix them.
        # Baselines: vecnorm_path="" → no wrapper, plain env returned.
        if vecnorm_path and os.path.exists(vecnorm_path):
            return VecNormObsWrapper(base_env, vecnorm_path)
        if vecnorm_path and not os.path.exists(vecnorm_path):
            print(f"[WARN] vecnorm_path not found: {vecnorm_path}. "
                  f"Proceeding without normalization.")
        return base_env

    # ── Metadata merged into every CSV row ───────────────────────────────────
    meta = {
        "policy":           args.policy_label or args.algo,
        "seed":             args.seed,
        "train_scenario":   args.train_scenario,
        "experiment_tag":   args.experiment_tag,
        "train_steps":      args.train_steps,
        "init_low":         args.init_diesel_low,
        "init_high":        args.init_diesel_high,
        "episode_len":      args.episode_len,
    }

    # ── Run evaluation ────────────────────────────────────────────────────────
    df = evaluate(
        policy=policy,
        env_factory=env_factory,
        n_episodes=args.episodes,
        sites=[args.site],
        leads=[args.lead],
        seed=args.seed,
        verbose=True,
        trace_out=args.trace_out,
        meta=meta,
    )

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"Saved: {args.out_csv}")