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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.data_loader import load_site, train_test_split
from env.telecom_env import TelecomEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from wrappers.drop_inventory_obs import DispatchOnlyEnv
from baselines.s_S_policy import SSPolicy


DEFAULT_TIMESTEPS = 576_000
N_ENVS = 8

TRAIN_EP_LEN = 720
EVAL_EP_LEN  = 720   # used only for the training-time EvalCallback (fast feedback)
                     # final evaluation always uses TRAIN_EP_LEN via make_track_b_eval_env

POLICY_NET = [256, 256]
LR_START = 3e-4


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
            episode_len=(EVAL_EP_LEN if eval_mode else TRAIN_EP_LEN),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=str, default="site1")
    ap.add_argument("--all_sites", action="store_true")
    ap.add_argument("--lead", type=str, default="normal", choices=["fast", "normal", "delayed"])
    ap.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logdir", type=str, default="runs/track_b")
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    sites = ["site1", "site7", "site5"] if args.all_sites else [args.site]

    for site in sites:
        # FIX (Bug 3): was `args.site` — used the CLI arg instead of the loop
        # variable, so --all_sites trained all three save paths on the same site.
        site_csv = f"data/processed/{site}.csv" if not site.endswith(".csv") else site

        vec_env = SubprocVecEnv([
            make_env(site_csv, args.seed + i, eval_mode=False, lead_scenario=args.lead)
            for i in range(N_ENVS)
        ])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        eval_env = DummyVecEnv([make_env(site_csv, args.seed + 10_000, eval_mode=True, lead_scenario=args.lead)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env.training = False

        policy_kwargs = dict(net_arch=POLICY_NET)

        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=linear_schedule(LR_START),
            n_steps=2048,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=args.logdir,
            seed=args.seed,
        )

        eval_cb = SyncNormEvalCallback(
            eval_env,
            best_model_save_path=os.path.join(args.logdir, f"{site}_best"),
            log_path=os.path.join(args.logdir, f"{site}_eval"),
            eval_freq=10_000,
            n_eval_episodes=3,
            deterministic=True,
            render=False,
        )

        model.learn(total_timesteps=args.timesteps, callback=eval_cb)

        model_path = os.path.join(args.logdir, f"{site}_final_model.zip")
        model.save(model_path)

        vn_path = os.path.join(args.logdir, f"{site}_vecnormalize.pkl")
        vec_env.save(vn_path)

        vec_env.close()
        eval_env.close()

        print(f"[Track B] Saved: {model_path}")
        print(f"[Track B] Saved VecNormalize: {vn_path}")


if __name__ == "__main__":
    main()
