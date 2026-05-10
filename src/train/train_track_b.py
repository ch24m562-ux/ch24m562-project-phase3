"""    train/train_track_b.py — Track B (B2): (s,S) ordering + PPO dispatch

- Base environment: TelecomEnv (unchanged), Discrete(6)
- Wrapper: DispatchOnlyEnv exposes:
    - action: Discrete(2) DG only
    - obs: 6-D (no inventory dimensions)
- Ordering: injected via SSPolicy calibrated analytically from (d_bar, lead_p, tank_cap)

Correctness note:
  Eval must share VecNormalize stats with training. Uses SyncNormEvalCallback.

FIX (Bug 3): In the --all_sites loop, site_csv previously used args.site
  (the CLI argument) instead of the loop variable `site`. This caused all
  three sites to train on the same CSV, with only save paths differing.
  Fixed: site_csv now uses the loop variable `site`.

FIX (make_track_b_eval_env): Added episode_len parameter defaulting to
  TRAIN_EP_LEN (720). Previously it called make_env(eval_mode=True) which
  hardcoded EVAL_EP_LEN=360, causing Track B evaluation to run for half the
  steps of Track A and making all cost/diesel comparisons invalid.
"""
from __future__ import annotations

import os
import sys
import argparse
from typing import Callable

import numpy as np
import mlflow

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.data_loader import load_site, train_test_split
from env.telecom_env import TelecomEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from wrappers.drop_inventory_obs import DispatchOnlyEnv
from baselines.s_S_policy import SSPolicy


# ── All hyperparameters from hparams.yaml ────────────────────────────────────
import csv
import subprocess
from datetime import datetime
from config_loader import ppo_cfg, policy_cfg, train_cfg, env_cfg, registry_cfg

# Keep as aliases for backward compat with make_track_b_eval_env default arg
TRAIN_EP_LEN = env_cfg["train_episode_len"]
EVAL_EP_LEN  = env_cfg["eval_episode_len"]


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def f(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return f


class SyncNormEvalCallback(EvalCallback):
    def _sync_vecnormalize(self):
        train_vn = self.model.get_vec_normalize_env()
        eval_vn = self.eval_env
        if train_vn is None:
            return
        if not isinstance(train_vn, VecNormalize):
            return
        if not isinstance(eval_vn, VecNormalize):
            return

        eval_vn.obs_rms = train_vn.obs_rms
        eval_vn.ret_rms = train_vn.ret_rms
        eval_vn.training = False
        eval_vn.norm_reward = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.n_calls % self.eval_freq == 0):
            self._sync_vecnormalize()
        return super()._on_step()


def _make_ss_order_fn(base_env: TelecomEnv) -> Callable:
    """Build the (s,S) order function from the base env's calibrated params."""
    ss = SSPolicy.from_site_params(
        site_params=base_env.params,
        d_bar=base_env.d_bar,
        lead_p=base_env.lead_p,
        tank_cap_kwh=base_env.tank_cap_kwh,
    )

    def order_fn(base_obs, _env):
        soc_n, inv_n, pending, pqty_n, pv_n, load_n, grid, sin_h, cos_h = base_obs.tolist()
        return int(ss.order_action(inv_n=inv_n, pending_flag=pending))

    return order_fn


def make_env(site_csv: str, seed: int, eval_mode: bool, lead_scenario: str):
    """Returns a callable (for SubprocVecEnv / DummyVecEnv) that builds the env."""
    def _init():
        df, params = load_site(site_csv)
        df_train, df_test = train_test_split(df)
        data = df_test if eval_mode else df_train

        base_env = TelecomEnv(
            site_data=data,
            site_params=params,
            episode_len=(env_cfg["eval_episode_len"] if eval_mode else env_cfg["train_episode_len"]),
            eval_mode=eval_mode,
            lead_scenario=lead_scenario,
            seed=seed,
        )

        env = DispatchOnlyEnv(base_env, order_fn=_make_ss_order_fn(base_env))
        env = Monitor(env)
        return env
    return _init


def make_track_b_eval_env(
    site_csv: str,
    seed: int,
    lead_scenario: str,
    episode_len: int = TRAIN_EP_LEN,
    init_inv_frac_low: float = 0.6,
    init_inv_frac_high: float = 0.6,
) -> DispatchOnlyEnv:
    """
    Build the exact Track-B evaluation env for use in evaluate.py.

    episode_len defaults to TRAIN_EP_LEN (720) so that Track B and Track A
    are evaluated over the same horizon. EVAL_EP_LEN (360) is only used
    inside the SyncNormEvalCallback during training for speed.

    init_inv_frac_low/high: initial inventory fraction range for stress
    testing (S1 scenario). Defaults to 0.6/0.6 (standard 60% start).
    """
    df, params = load_site(site_csv)
    _df_train, df_test = train_test_split(df)

    base_env = TelecomEnv(
        site_data=df_test,
        site_params=params,
        episode_len=episode_len,
        eval_mode=True,
        lead_scenario=lead_scenario,
        seed=seed,
        init_inv_frac_low=init_inv_frac_low,
        init_inv_frac_high=init_inv_frac_high,
    )

    env = DispatchOnlyEnv(base_env, order_fn=_make_ss_order_fn(base_env))
    env = Monitor(env)
    return env


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=str, default="site1")
    ap.add_argument("--all_sites", action="store_true")
    ap.add_argument("--lead", type=str, default="normal", choices=["fast", "normal", "delayed", "very_delayed", "multi"])
    ap.add_argument("--timesteps", type=int, default=train_cfg["total_timesteps"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logdir", type=str, default="runs/track_b")
    ap.add_argument("--tag", type=str, default="phase3")
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    sites = ["site1", "site7", "site5"] if args.all_sites else [args.site]

    for site in sites:
        # FIX (Bug 3): was `args.site` — used the CLI arg instead of the loop
        # variable, so --all_sites trained all three save paths on the same site.
        site_csv = f"data/processed/{site}.csv" if not site.endswith(".csv") else site

        vec_env = SubprocVecEnv([
            make_env(site_csv, args.seed + i, eval_mode=False, lead_scenario=args.lead)
            for i in range(train_cfg["n_envs"])
        ])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=policy_cfg["obs_clip"])

        eval_env = DummyVecEnv([make_env(site_csv, args.seed + 10_000, eval_mode=True, lead_scenario=args.lead)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=policy_cfg["obs_clip"])
        eval_env.training = False

        policy_kwargs = dict(net_arch=policy_cfg["net_arch"])

        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=linear_schedule(ppo_cfg["learning_rate"]),
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            gamma=ppo_cfg["gamma"],
            gae_lambda=ppo_cfg["gae_lambda"],
            clip_range=ppo_cfg["clip_range"],
            ent_coef=ppo_cfg["ent_coef"],
            vf_coef=ppo_cfg["vf_coef"],
            max_grad_norm=ppo_cfg["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=args.logdir,
            seed=args.seed,
        )

        eval_cb = SyncNormEvalCallback(
            eval_env,
            best_model_save_path=os.path.join(args.logdir, f"{site}_best"),
            log_path=os.path.join(args.logdir, f"{site}_eval"),
            eval_freq=train_cfg["eval_freq"],
            n_eval_episodes=train_cfg["n_eval_episodes"],
            deterministic=True,
            render=False,
        )

        model.learn(total_timesteps=args.timesteps, callback=eval_cb)

        model_path = os.path.join(args.logdir, f"{site}_s{args.seed}_final.zip")
        model.save(model_path)

        vn_path = os.path.join(args.logdir, f"{site}_s{args.seed}_vecnorm.pkl")
        vec_env.save(vn_path)

        vec_env.close()
        eval_env.close()

        best_reward = eval_cb.best_mean_reward
        print(f"[Track B] Saved: {model_path}")
        print(f"[Track B] Saved VecNormalize: {vn_path}")

        # MLflow logging
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(args.tag)
        with mlflow.start_run(run_name=f"{site}_trackb_s{args.seed}"):
            mlflow.log_params({
                "site": site, "seed": args.seed, "policy": "TrackB",
                "lead_scenario": args.lead, "gamma": ppo_cfg["gamma"],
                "n_envs": train_cfg["n_envs"], "total_timesteps": args.timesteps,
                "git_commit": _get_git_commit(),
            })
            mlflow.log_metrics({
                "best_eval_reward": float(best_reward) if np.isfinite(best_reward) else -9999.0,
            })
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(vn_path)


if __name__ == "__main__":
    main()