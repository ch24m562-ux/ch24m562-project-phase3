"""train/train_ablation_a7.py — Ablation A7: Vanilla PPO (no safety mechanisms)

Identical to train_rl_inv.py EXCEPT:
  - Uses standard PPO instead of MaskablePPO
  - No ActionMasker wrapper (no action masking)
  - No Lagrangian penalty (not in base TelecomEnv reward anyway)
  - No recovery policy override

This directly tests H2: do safety mechanisms (action masking) reduce
constraint violations vs unconstrained RL?

Compare A7 results vs A1 (Full SC-PPO) on:
  - violations per episode
  - SoC violations
  - stockout events
  - EENS (reliability impact of constraint violations)

Run:
  python -m src.train.train_ablation_a7 --site site5 --lead normal --timesteps 200000 --seed 42 --logdir runs/ablation_a7

Evaluate after training:
  python -m src.eval.evaluate --site site5 --lead normal --policy_type rl --algo ppo
    --env_type track_a --model_path runs/ablation_a7/site5_final_model
    --episodes 5 --seed 42 --out_csv results/eval/ablation_a7_site5_normal.csv
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


# ── Hyperparameters — identical to train_rl_inv.py for fair comparison ───────

DEFAULT_TIMESTEPS = 200_000   # less than full 576K — ablation only needs site5
N_ENVS = 8

TRAIN_EP_LEN = 720
EVAL_EP_LEN  = 720

POLICY_NET = [256, 256]
LR_START   = 3e-4


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def f(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return f


# ── VecNormalize sync callback (same as train_rl_inv.py) ─────────────────────

class SyncNormEvalCallback(EvalCallback):
    def _sync_vecnormalize(self):
        train_vn = self.model.get_vec_normalize_env()
        eval_vn  = self.eval_env
        if not isinstance(train_vn, VecNormalize):
            return
        if not isinstance(eval_vn, VecNormalize):
            return
        eval_vn.obs_rms   = train_vn.obs_rms
        eval_vn.ret_rms   = train_vn.ret_rms
        eval_vn.training  = False
        eval_vn.norm_reward = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.n_calls % self.eval_freq == 0):
            self._sync_vecnormalize()
        return super()._on_step()


# ── Env factory — NO ActionMasker wrapper ────────────────────────────────────

def make_env(site_csv: str, seed: int, eval_mode: bool, lead_scenario: str,
             init_inv_frac_low: float = 0.6, init_inv_frac_high: float = 0.6):
    def _init():
        df, params = load_site(site_csv)
        df_train, df_test = train_test_split(df)
        data = df_test if eval_mode else df_train

        env = TelecomEnv(
            site_data=data,
            site_params=params,
            episode_len=(EVAL_EP_LEN if eval_mode else TRAIN_EP_LEN),
            eval_mode=eval_mode,
            lead_scenario=lead_scenario,
            seed=seed,
            init_inv_frac_low=init_inv_frac_low,
            init_inv_frac_high=init_inv_frac_high,
        )
        env = Monitor(env)
        # NOTE: NO ActionMasker here — this is the A7 ablation.
        # The policy will see the full Discrete(6) space with no masking.
        # This means it CAN try to run DG with no fuel, order when pending, etc.
        # Violations are tracked in ep_info_log via TelecomEnv's hard masking
        # in step() — the env still enforces physics, but the policy doesn't
        # get the mask signal to learn to avoid these.
        return env
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site",       type=str,   default="site5")
    ap.add_argument("--lead",       type=str,   default="normal",
                    choices=["fast", "normal", "delayed"])
    ap.add_argument("--timesteps",  type=int,   default=DEFAULT_TIMESTEPS)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--logdir",     type=str,   default="runs/ablation_a7")
    ap.add_argument("--init_diesel_low",  type=float, default=0.6)
    ap.add_argument("--init_diesel_high", type=float, default=0.6)
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    site = args.site
    site_csv = f"data/processed/{site}.csv"

    print(f"[A7 Ablation] Training Vanilla PPO (no masking) on {site} | "
          f"lead={args.lead} | timesteps={args.timesteps}")

    # ── Training env ─────────────────────────────────────────────────────────
    vec_env = SubprocVecEnv([
        make_env(site_csv, args.seed + i, eval_mode=False,
                 lead_scenario=args.lead,
                 init_inv_frac_low=args.init_diesel_low,
                 init_inv_frac_high=args.init_diesel_high)
        for i in range(N_ENVS)
    ])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # ── Eval env ─────────────────────────────────────────────────────────────
    eval_env = DummyVecEnv([
        make_env(site_csv, args.seed + 10_000, eval_mode=True,
                 lead_scenario=args.lead)
    ])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False

    # ── Model — standard PPO, no masking ─────────────────────────────────────
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
        policy_kwargs=dict(net_arch=POLICY_NET),
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

    # ── Save ─────────────────────────────────────────────────────────────────
    model_path = os.path.join(args.logdir, f"{site}_final_model.zip")
    vn_path    = os.path.join(args.logdir, f"{site}_vecnormalize.pkl")

    model.save(model_path)
    vec_env.save(vn_path)

    vec_env.close()
    eval_env.close()

    print(f"[A7] Saved: {model_path}")
    print(f"[A7] Saved VecNormalize: {vn_path}")
    print()
    print("Next — evaluate with:")
    print(f"  python -m src.eval.evaluate --site {site} --lead {args.lead} "
          f"--policy_type rl --algo ppo --env_type track_a "
          f"--model_path {model_path.replace('.zip','')} "
          f"--episodes 5 --seed 42 "
          f"--out_csv results/eval/ablation_a7_{site}_{args.lead}.csv")


if __name__ == "__main__":
    main()
