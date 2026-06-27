"""src/train/train_sensitivity.py — Reward sensitivity sweep (Phase 3)

Scientific basis:
  Controlled perturbation at ±50% (0.5× and 2×) is standard sensitivity analysis.
  Each variant varies ONE reward component while holding all others at base values.
  Economic grounding: lambda range covers GSMA tower downtime cost spectrum.

Variants (from hparams.yaml sensitivity section):
  lam_low:   lambda = 0.5×base (50)    — conservative, direct cost only
  lam_high:  lambda = 2×base (200)     — aggressive SLA breach scenario
  lam_vlow:  lambda = 0.25×base (25)   — below-market
  lam_vhigh: lambda = 5×base (500)     — severe regulatory penalty
  cliff_low: gamma_r = 1×lambda (100)  — cliff barely exceeds proportional
  cliff_high:gamma_r = 4×lambda (400)  — very strong cliff signal
  beta_low:  beta = 0.5×base (0.2)     — cheap grid state
  beta_high: beta = 2×base (0.8)       — expensive grid state
  mu_low:    mu = 0.25×base (5)        — weak violation penalty
  mu_high:   mu = 2.5×base (50)        — strong violation penalty

Run examples:
  # Single variant
  python src/train/train_sensitivity.py --variant lam_high --site site5 --seed 42

  # All variants on constrained sites
  python src/train/train_sensitivity.py --all_variants --sites constrained --seeds 42 123 777
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
import mlflow

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.data_loader import load_site, train_test_split
from env.telecom_env import TelecomEnv, RewardCoeffs

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from config_loader import ppo_cfg, policy_cfg, train_cfg, env_cfg, registry_cfg, cfg

# Load sensitivity variants from hparams.yaml
SENSITIVITY_VARIANTS = cfg.get("sensitivity", {})
CONSTRAINED_SITES = ["site5", "site10"]
ALL_SITES = [f"site{i}" for i in range(1, 11)]


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def f(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return f


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def mask_fn(env) -> np.ndarray:
    return env.unwrapped.get_action_mask()


# SyncNormEvalCallback is defined inline as SyncCb inside train_sensitivity_variant()
# to avoid circular imports from callbacks.py.


def make_env(site_csv: str, seed: int, eval_mode: bool,
             lead_scenario: str, reward_override: dict):
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
            reward_coeffs=reward_override,   # ← sensitivity override
        )
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)
        return env
    return _init


def train_sensitivity_variant(
    variant_name: str,
    site: str,
    seed: int,
    logdir: str,
    lead_scenario: str = "normal",
    timesteps: int | None = None,
):
    """Train one sensitivity variant on one site with one seed."""

    if variant_name not in SENSITIVITY_VARIANTS:
        raise ValueError(
            f"Variant '{variant_name}' not found in hparams.yaml sensitivity section.\n"
            f"Available: {list(SENSITIVITY_VARIANTS.keys())}"
        )

    # Get reward coefficients for this variant
    reward_override = dict(SENSITIVITY_VARIANTS[variant_name])
    base = SENSITIVITY_VARIANTS.get("base", {})

    site_csv = f"data/processed/{site}.csv"
    run_name = f"sensitivity_{variant_name}_{site}_s{seed}"
    run_dir  = os.path.join(logdir, variant_name, site)
    os.makedirs(run_dir, exist_ok=True)

    # Check if already done
    model_path = os.path.join(run_dir, f"{site}_s{seed}_final.zip")
    if os.path.exists(model_path):
        print(f"  SKIP {run_name} — model already exists")
        return

    print(f"\n[Sensitivity] {run_name}")
    print(f"  Reward override: {reward_override}")
    print(f"  vs base:         {base}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("phase3_sensitivity")

    with mlflow.start_run(run_name=run_name):
        # Log variant + base for comparison
        mlflow.log_params({
            "variant":       variant_name,
            "site":          site,
            "seed":          seed,
            "lead_scenario": lead_scenario,
            "alpha":         reward_override.get("alpha",   base.get("alpha", 1.0)),
            "beta":          reward_override.get("beta",    base.get("beta", 0.4)),
            "lam":           reward_override.get("lam",     base.get("lam", 100.0)),
            "gamma_r":       reward_override.get("gamma_r", base.get("gamma_r", 200.0)),
            "mu":            reward_override.get("mu",      base.get("mu", 20.0)),
            "gamma_ppo":     ppo_cfg["gamma"],
            "git_commit":    _get_git_commit(),
        })

        n_envs = train_cfg["n_envs"]

        vec_env = SubprocVecEnv([
            make_env(site_csv, seed + i, eval_mode=False,
                     lead_scenario=lead_scenario, reward_override=reward_override)
            for i in range(n_envs)
        ])
        vec_env = VecNormalize(
            vec_env, norm_obs=True, norm_reward=False,
            clip_obs=policy_cfg["obs_clip"]
        )

        eval_env = DummyVecEnv([
            make_env(site_csv, seed + 10_000, eval_mode=True,
                     lead_scenario=lead_scenario, reward_override=reward_override)
        ])
        eval_env = VecNormalize(
            eval_env, norm_obs=True, norm_reward=False,
            clip_obs=policy_cfg["obs_clip"]
        )
        eval_env.training    = False
        eval_env.norm_reward = False

        from stable_baselines3.common.callbacks import EvalCallback

        class SyncCb(EvalCallback):
            def _sync_vecnormalize(self):
                train_vn = self.model.get_vec_normalize_env()
                eval_vn  = self.eval_env
                if isinstance(train_vn, VecNormalize) and isinstance(eval_vn, VecNormalize):
                    eval_vn.obs_rms = train_vn.obs_rms; eval_vn.ret_rms = train_vn.ret_rms
                    eval_vn.training = False; eval_vn.norm_reward = False
            def _on_step(self):
                if self.eval_freq > 0 and (self.n_calls % self.eval_freq == 0):
                    self._sync_vecnormalize()
                return super()._on_step()

        model = MaskablePPO(
            "MlpPolicy", vec_env,
            learning_rate  = linear_schedule(ppo_cfg["learning_rate"]),
            n_steps        = ppo_cfg["n_steps"],
            batch_size     = ppo_cfg["batch_size"],
            gamma          = ppo_cfg["gamma"],
            gae_lambda     = ppo_cfg["gae_lambda"],
            clip_range     = ppo_cfg["clip_range"],
            ent_coef       = ppo_cfg["ent_coef"],
            vf_coef        = ppo_cfg["vf_coef"],
            max_grad_norm  = ppo_cfg["max_grad_norm"],
            policy_kwargs  = dict(net_arch=policy_cfg["net_arch"]),
            verbose        = 1,
            tensorboard_log= run_dir,
            seed           = seed,
        )

        eval_cb = SyncCb(
            eval_env,
            best_model_save_path = os.path.join(run_dir, f"{site}_best"),
            log_path             = os.path.join(run_dir, f"{site}_eval"),
            eval_freq            = train_cfg["eval_freq"],
            n_eval_episodes      = train_cfg["n_eval_episodes"],
            deterministic        = True, render = False,
        )

        total_steps = timesteps if timesteps is not None else train_cfg["total_timesteps"]
        mlflow.log_param("total_timesteps", total_steps)

        start_time = datetime.now()
        model.learn(total_timesteps=total_steps, callback=eval_cb)
        wall_time = (datetime.now() - start_time).total_seconds() / 60

        vn_path = os.path.join(run_dir, f"{site}_s{seed}_vecnorm.pkl")
        model.save(model_path)
        vec_env.save(vn_path)
        vec_env.close(); eval_env.close()

        best_reward = eval_cb.best_mean_reward
        mlflow.log_metrics({
            "best_eval_reward": float(best_reward) if np.isfinite(best_reward) else -9999.0,
            "wall_time_min":    round(wall_time, 1),
        })
        try:
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(vn_path)
        except Exception as e:
            print(f"[MLflow] Artifact logging skipped: {e}")

        print(f"  Done in {wall_time:.1f}min | best_reward={best_reward:.3f}")


def main():
    ap = argparse.ArgumentParser(
        description="Reward sensitivity sweep — 0.5× and 2× scaling of each component"
    )
    ap.add_argument("--variant",      type=str, default="lam_high",
                    help=f"One of: {list(SENSITIVITY_VARIANTS.keys())}")
    ap.add_argument("--all_variants", action="store_true",
                    help="Run ALL sensitivity variants sequentially")
    ap.add_argument("--site",         type=str, default="site5")
    ap.add_argument("--sites",        type=str, default="constrained",
                    help="constrained=site5+site10, all=all 10 sites, "
                         "or comma-separated e.g. site5,site2,site10")
    ap.add_argument("--seeds",        type=int, nargs="+",
                    default=[42, 123, 777],
                    help="Seeds to run (space-separated)")
    ap.add_argument("--lead",         type=str, default="normal",
                    choices=["fast", "normal", "delayed", "very_delayed", "multi"])
    ap.add_argument("--logdir",       type=str, default="runs/sensitivity")
    ap.add_argument("--timesteps",    type=int,
                    default=None,
                    help="Training timesteps (default: from hparams total_timesteps)")
    args = ap.parse_args()

    # Resolve site list
    if args.sites == "constrained":
        sites = CONSTRAINED_SITES
    elif args.sites == "all":
        sites = ALL_SITES
    else:
        sites = [s.strip() for s in args.sites.split(",")]

    # Resolve timesteps
    timesteps = args.timesteps if args.timesteps is not None else train_cfg["total_timesteps"]

    # Resolve variant list
    variants_to_run = (
        [k for k in SENSITIVITY_VARIANTS.keys() if k != "base"]
        if args.all_variants
        else [args.variant]
    )

    print(f"[Sensitivity] Running {len(variants_to_run)} variants × "
          f"{len(sites)} sites × {len(args.seeds)} seeds = "
          f"{len(variants_to_run) * len(sites) * len(args.seeds)} runs")
    print(f"  Variants:  {variants_to_run}")
    print(f"  Sites:     {sites}")
    print(f"  Seeds:     {args.seeds}")
    print(f"  Timesteps: {timesteps}")
    print()

    for variant in variants_to_run:
        for site in sites:
            for seed in args.seeds:
                train_sensitivity_variant(
                    variant_name  = variant,
                    site          = site,
                    seed          = seed,
                    logdir        = args.logdir,
                    lead_scenario = args.lead,
                    timesteps     = timesteps,
                )


if __name__ == "__main__":
    main()