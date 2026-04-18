"""train/train_rl_inv_stress.py — Track A retrained under S1 stress conditions

Identical to train_rl_inv.py EXCEPT:
  - init_diesel_low / init_diesel_high forwarded to TelecomEnv
  - Default: low=0.10, high=0.20 (10-20% initial inventory)
  - Training lead: delayed (hardest scenario)

Scientific purpose:
  Tests whether a policy TRAINED under inventory scarcity outperforms
  a policy trained under standard conditions when EVALUATED under scarcity.

  If stress-trained policy >> standard-trained policy on S1 eval:
    → Proves inventory-aware training has genuine value (H3 strongest evidence)
  If similar:
    → RL-Inv generalises well out-of-distribution

Compare:
  Standard model:  runs/track_a/site5_final_model  evaluated under S1
  Stress model:    runs/stress_s1/site5_final_model evaluated under S1

Run:
  python -m src.train.train_rl_inv_stress --site site5 --lead delayed
    --timesteps 300000 --seed 42 --logdir runs/stress_s1
    --init_diesel_low 0.10 --init_diesel_high 0.20

Evaluate after training:
  python -m src.eval.evaluate --site site5 --lead delayed --policy_type rl
    --algo maskable --env_type track_a
    --model_path runs/stress_s1/site5_final_model
    --episodes 5 --seed 42
    --init_diesel_low 0.10 --init_diesel_high 0.20
    --out_csv results/eval/stress_trained_rlinv_site5_delayed.csv
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

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker


DEFAULT_TIMESTEPS = 300_000
N_ENVS = 8
TRAIN_EP_LEN = 720
EVAL_EP_LEN  = 720  # FIXED: must match training horizon for best-model selection
POLICY_NET   = [256, 256]
LR_START     = 3e-4


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def f(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return f


class SyncNormEvalCallback(EvalCallback):
    def _sync_vecnormalize(self):
        train_vn = self.model.get_vec_normalize_env()
        eval_vn  = self.eval_env
        if not isinstance(train_vn, VecNormalize):
            return
        if not isinstance(eval_vn, VecNormalize):
            return
        eval_vn.obs_rms     = train_vn.obs_rms
        eval_vn.ret_rms     = train_vn.ret_rms
        eval_vn.training    = False
        eval_vn.norm_reward = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.n_calls % self.eval_freq == 0):
            self._sync_vecnormalize()
        return super()._on_step()


def mask_fn(env) -> np.ndarray:
    return env.unwrapped.get_action_mask()


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
        env = ActionMasker(env, mask_fn)
        return env
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site",             type=str,   default="site5")
    ap.add_argument("--lead",             type=str,   default="delayed",
                    choices=["fast", "normal", "delayed"])
    ap.add_argument("--timesteps",        type=int,   default=DEFAULT_TIMESTEPS)
    ap.add_argument("--seed",             type=int,   default=42)
    ap.add_argument("--logdir",           type=str,   default="runs/stress_s1")
    ap.add_argument("--init_diesel_low",  type=float, default=0.10,
                    help="lower bound initial inventory fraction (default 0.10 = 10%)")
    ap.add_argument("--init_diesel_high", type=float, default=0.20,
                    help="upper bound initial inventory fraction (default 0.20 = 20%)")
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    site     = args.site
    site_csv = f"data/processed/{site}.csv"

    print(f"[Stress Train] MaskablePPO on {site} | lead={args.lead} | "
          f"init_inv=[{args.init_diesel_low},{args.init_diesel_high}] | "
          f"timesteps={args.timesteps}")

    # ── Training env — stress conditions ──────────────────────────────────────
    vec_env = SubprocVecEnv([
        make_env(site_csv, args.seed + i, eval_mode=False,
                 lead_scenario=args.lead,
                 init_inv_frac_low=args.init_diesel_low,
                 init_inv_frac_high=args.init_diesel_high)
        for i in range(N_ENVS)
    ])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # ── Eval env — also stress conditions (evaluate under what it trained on) ──
    eval_env = DummyVecEnv([
        make_env(site_csv, args.seed + 10_000, eval_mode=True,
                 lead_scenario=args.lead,
                 init_inv_frac_low=args.init_diesel_low,
                 init_inv_frac_high=args.init_diesel_high)
    ])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False

    # ── Model — identical architecture to Track A ─────────────────────────────
    model = MaskablePPO(
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

    print(f"[Stress Train] Saved: {model_path}")
    print(f"[Stress Train] Saved VecNormalize: {vn_path}")
    print()
    print("Now evaluate stress-trained model under S1 conditions:")
    print(f"  python -m src.eval.evaluate --site {site} --lead {args.lead} "
          f"--policy_type rl --algo maskable --env_type track_a "
          f"--model_path {model_path.replace('.zip','')} "
          f"--episodes 5 --seed 42 "
          f"--init_diesel_low {args.init_diesel_low} "
          f"--init_diesel_high {args.init_diesel_high} "
          f"--out_csv results/eval/stress_trained_rlinv_{site}_{args.lead}.csv")
    print()
    print("Compare against standard-trained model under same S1 conditions:")
    print(f"  python -m src.eval.evaluate --site {site} --lead {args.lead} "
          f"--policy_type rl --algo maskable --env_type track_a "
          f"--model_path runs/track_a/{site}_final_model "
          f"--episodes 5 --seed 42 "
          f"--init_diesel_low {args.init_diesel_low} "
          f"--init_diesel_high {args.init_diesel_high} "
          f"--out_csv results/eval/stress_S1_rlinv_{site}_{args.lead}_v2.csv")


if __name__ == "__main__":
    main()
