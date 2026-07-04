"""train/train_rl_inv.py — Track A: RL-Inv (MaskablePPO)

Phase 3 changes from original:
  1. All hyperparameters read from hparams.yaml via config_loader
  2. --lead choices expanded to include "multi" and "very_delayed"
  3. MLflow experiment tracking added (logs params + metrics after each run)
  4. Experiment registry CSV logging added

All training logic, callback, model, vecenv setup — UNCHANGED from Phase 2.
"""
from __future__ import annotations

import os
import sys
import csv
import argparse
import subprocess
from datetime import datetime
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.data_loader import load_site, train_test_split
from env.telecom_env import TelecomEnv

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import mlflow
from callbacks import DetailedEvalCallback
from mlflow_rollout_callback import MlflowRolloutCallback

# ── All hyperparameters from hparams.yaml ─────────────────────────────────────
from config_loader import ppo_cfg, policy_cfg, train_cfg, env_cfg, registry_cfg


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def f(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return f


# ── VecNormalize sync callback (UNCHANGED) ────────────────────────────────────
def mask_fn(env) -> np.ndarray:
    return env.unwrapped.get_action_mask()


# ── make_env — same structure as original ────────────────────────────────────
def make_env(site_csv: str, seed: int, eval_mode: bool, lead_scenario: str,
             tank_scale: float = 1.0, use_time_encoding: bool = True,
             lead_distribution: str = "geometric",
             lead_sigma: float = 0.5,
             lead_k: float = 2.0,
             use_stochastic_grid: bool = False,
             use_eta_obs: bool = False,
             use_supplier_regime: bool = False,   # [CHANGE 11] RLInv-DA extension
             use_ewma_lead: bool = False):         # [CHANGE 11] RLInv-DA extension
    def _init():
        df, params = load_site(site_csv)
        df_train, df_test = train_test_split(df)
        data = df_test if eval_mode else df_train

        ep_len   = env_cfg["eval_episode_len"] if eval_mode else env_cfg["train_episode_len"]
        inv_low  = env_cfg["init_inv_frac_eval_low"]  if eval_mode else env_cfg["init_inv_frac_train_low"]
        inv_high = env_cfg["init_inv_frac_eval_high"] if eval_mode else env_cfg["init_inv_frac_train_high"]

        env = TelecomEnv(
            site_data=data,
            site_params=params,
            episode_len=ep_len,
            eval_mode=eval_mode,
            lead_scenario=lead_scenario,
            seed=seed,
            init_inv_frac_low=inv_low,
            init_inv_frac_high=inv_high,
            tank_scale=tank_scale,
            use_time_encoding=use_time_encoding,
            lead_distribution=lead_distribution,
            lead_sigma=lead_sigma,
            lead_k=lead_k,
            use_stochastic_grid=use_stochastic_grid,
            use_eta_obs=use_eta_obs,
            use_supplier_regime=use_supplier_regime,
            use_ewma_lead=use_ewma_lead,
        )
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)
        return env
    return _init


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _log_to_registry(record: dict):
    """Append one row to the flat CSV experiment registry."""
    path = registry_cfg["output_csv"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = registry_cfg["columns"]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(record)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site",       type=str,   default="site1")
    ap.add_argument("--all_sites",  action="store_true",
                    help="train all 10 sites sequentially")
    ap.add_argument("--lead",       type=str,   default="normal",
                    choices=["fast", "normal", "delayed", "very_delayed", "multi"])
    ap.add_argument("--timesteps",  type=int,   default=train_cfg["total_timesteps"])
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--logdir",     type=str,   default="runs/rlinv")
    ap.add_argument("--tag",        type=str,   default="phase3",
                    help="Experiment tag for MLflow and registry CSV")
    # ── Phase 3 additions ─────────────────────────────────────────────────────
    ap.add_argument("--gamma",      type=float, default=None,
                    help="Override PPO discount factor (default: from hparams.yaml). "
                         "Use for gamma sensitivity: 0.95, 0.99, 0.995")
    ap.add_argument("--tank_scale", type=float, default=1.0,
                    help="Tank capacity multiplier. 1.0=72h (base), 1.33=96h, 2.0=144h")
    ap.add_argument("--no_time_enc", action="store_true",
                    help="Disable time encoding ablation — zeros sin_h/cos_h in obs")
    ap.add_argument("--lead_dist",  type=str,   default="geometric",
                    choices=["geometric", "lognormal", "weibull"],
                    help="Lead time distribution. lognormal/weibull samples at order placement.")
    ap.add_argument("--lead_sigma", type=float, default=0.5,
                    help="Lognormal sigma shape parameter (default 0.5, ignored for geometric)")
    ap.add_argument("--lead_k",     type=float, default=2.0,
                    help="Weibull shape parameter k (default 2.0, ignored for other dists)")
    ap.add_argument("--stochastic_grid", action="store_true",
                    help="Use 2-state Markov chain for grid outage (fitted from site data).")
    ap.add_argument("--use_eta_obs", action="store_true",
                    help="ETA-aware EXTENSION only: agent sees delivery_remaining_n. "
                         "Default False for all main experiments. "
                         "Only use with --lead_dist lognormal --tag lognormal_eta.")
    ap.add_argument("--use_supplier_regime", action="store_true",
                    help="[CHANGE 11] Two-state Markov supplier regime (RLInv-DA extension).")
    ap.add_argument("--use_ewma_lead", action="store_true",
                    help="[CHANGE 11] EWMA lead-time estimate as obs dim 12 (RLInv-DA extension).")
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    all_sites = [f"site{i}" for i in range(1, 11)]
    sites = all_sites if args.all_sites else [args.site]

    # ── MLflow experiment — one experiment per tag ────────────────────────────
    # Set tracking URI explicitly so training always writes to the same
    # backend that `mlflow ui` reads from. Without this, MLflow may use
    # mlruns/ (file-based) instead of mlflow.db depending on version.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(args.tag)

    for site in sites:
        site_csv = f"data/processed/{site}.csv"

        # ── MLflow run — one run per (site, seed, lead) ───────────────────────
        run_name = f"{site}_lead{args.lead}_s{args.seed}"
        with mlflow.start_run(run_name=run_name):

            # Log all hyperparams from hparams.yaml + CLI overrides
            gamma_val = args.gamma if args.gamma is not None else ppo_cfg["gamma"]
            mlflow.log_params({
                "site":           site,
                "seed":           args.seed,
                "lead_scenario":  args.lead,
                "tag":            args.tag,
                "gamma":          gamma_val,
                "n_steps":        ppo_cfg["n_steps"],
                "batch_size":     ppo_cfg["batch_size"],
                "learning_rate":  ppo_cfg["learning_rate"],
                "n_envs":         train_cfg["n_envs"],
                "total_timesteps": args.timesteps,
                "net_arch":       str(policy_cfg["net_arch"]),
                "tank_scale":      args.tank_scale,
                "use_time_enc":    not args.no_time_enc,
                "lead_dist":       args.lead_dist,
                "lead_sigma":      args.lead_sigma,
                "lead_k":          args.lead_k,
                "stochastic_grid": args.stochastic_grid,
                "use_eta_obs":     args.use_eta_obs,
                "use_supplier_regime": args.use_supplier_regime,
                "use_ewma_lead":   args.use_ewma_lead,
                "git_commit":      _get_git_commit(),
            })

            # ── Training setup (UNCHANGED logic) ─────────────────────────────
            n_envs = train_cfg["n_envs"]

            vec_env = SubprocVecEnv([
                make_env(site_csv, args.seed + i, eval_mode=False,
                         lead_scenario=args.lead,
                         tank_scale=args.tank_scale,
                         use_time_encoding=not args.no_time_enc,
                         lead_distribution=args.lead_dist,
                         lead_sigma=args.lead_sigma,
                         lead_k=args.lead_k,
                         use_stochastic_grid=args.stochastic_grid,
                         use_eta_obs=args.use_eta_obs,
                         use_supplier_regime=args.use_supplier_regime,
                         use_ewma_lead=args.use_ewma_lead)
                for i in range(n_envs)
            ])
            vec_env = VecNormalize(
                vec_env, norm_obs=True, norm_reward=False,
                clip_obs=policy_cfg["obs_clip"]
            )

            eval_env = DummyVecEnv([
                make_env(site_csv, args.seed + 10_000, eval_mode=True,
                         lead_scenario=args.lead,
                         tank_scale=args.tank_scale,
                         use_time_encoding=not args.no_time_enc,
                         lead_distribution=args.lead_dist,
                         lead_sigma=args.lead_sigma,
                         lead_k=args.lead_k,
                         use_stochastic_grid=args.stochastic_grid,
                         use_eta_obs=args.use_eta_obs,
                         use_supplier_regime=args.use_supplier_regime,
                         use_ewma_lead=args.use_ewma_lead)
            ])
            eval_env = VecNormalize(
                eval_env, norm_obs=True, norm_reward=False,
                clip_obs=policy_cfg["obs_clip"]
            )
            eval_env.training = False

            policy_kwargs = dict(net_arch=policy_cfg["net_arch"])

            lr_schedule = (
                linear_schedule(ppo_cfg["learning_rate"])
                if ppo_cfg["lr_schedule"] == "linear"
                else ppo_cfg["learning_rate"]
            )

            model = MaskablePPO(
                "MlpPolicy",
                vec_env,
                learning_rate  = lr_schedule,
                n_steps        = ppo_cfg["n_steps"],
                batch_size     = ppo_cfg["batch_size"],
                gamma          = gamma_val,
                gae_lambda     = ppo_cfg["gae_lambda"],
                clip_range     = ppo_cfg["clip_range"],
                ent_coef       = ppo_cfg["ent_coef"],
                vf_coef        = ppo_cfg["vf_coef"],
                max_grad_norm  = ppo_cfg["max_grad_norm"],
                policy_kwargs  = policy_kwargs,
                verbose        = 1,
                tensorboard_log = args.logdir,
                seed           = args.seed,
            )

            eval_cb = DetailedEvalCallback(
                eval_env,
                site=site,
                seed=args.seed,
                best_model_save_path=os.path.join(args.logdir, f"{site}_best"),
                log_path=os.path.join(args.logdir, f"{site}_eval"),
                eval_freq=train_cfg["eval_freq"],
                n_eval_episodes=train_cfg["n_eval_episodes"],
                deterministic=True,
                render=False,
            )

            # ── Train ─────────────────────────────────────────────────────────
            start_time = datetime.now()
            #model.learn(total_timesteps=args.timesteps, callback=eval_cb)
            rollout_cb = MlflowRolloutCallback()
            model.learn(total_timesteps=args.timesteps, callback=[eval_cb, rollout_cb])
            wall_time_min = (datetime.now() - start_time).total_seconds() / 60

            # ── Save model + vecnorm ──────────────────────────────────────────
            model_path = os.path.join(args.logdir, f"{site}_s{args.seed}_final.zip")
            vn_path    = os.path.join(args.logdir, f"{site}_s{args.seed}_vecnorm.pkl")
            model.save(model_path)
            vec_env.save(vn_path)

            vec_env.close()
            eval_env.close()

            # ── Log training metrics to MLflow ────────────────────────────────
            # best_mean_reward is -inf if EvalCallback never fired (e.g. short
            # test runs where timesteps < eval_freq). Guard against passing -inf
            # to MLflow which causes the run to be marked FAILED silently.
            # MaskableEvalCallback may not expose best_mean_reward identically
            # to EvalCallback — use getattr with fallback chain
            best_reward = getattr(
                eval_cb, "best_mean_reward",
                getattr(eval_cb, "last_mean_reward", float("nan"))
            )
            mlflow.log_metrics({
                "best_eval_reward": float(best_reward) if np.isfinite(best_reward) else -9999.0,
                "wall_time_min":    round(wall_time_min, 1),
            })

            # Log saved model as MLflow artifact
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(vn_path)

            print(f"[RLInv] Done: {site} lead={args.lead} seed={args.seed} "
                  f"| best_reward={best_reward:.3f} "
                  f"| time={wall_time_min:.1f}min")
            print(f"[RLInv] Saved: {model_path}")

            # ── Log to flat CSV registry ──────────────────────────────────────
            _log_to_registry({
                "experiment_tag":  args.tag,
                "policy":          "RLInv" if args.lead != "multi" else "RLInv-Multi",
                "site":            site,
                "seed":            args.seed,
                "lead_scenario":   args.lead,
                "train_scenario":  args.lead,
                "EENS_kWh":        "",   # filled by evaluate.py after evaluation
                "diesel_kWh":      "",
                "uptime_pct":      "",
                "mean_inv_pct":    "",
                "orders_placed":   "",
                "stockout_events": "",
                "violations":      "",
                "timestamp":       datetime.now().isoformat(),
                "git_commit":      _get_git_commit(),
                "tank_scale":      args.tank_scale,
                "lead_dist":       args.lead_dist,
                "use_time_enc":    not args.no_time_enc,
            })


if __name__ == "__main__":
    main()