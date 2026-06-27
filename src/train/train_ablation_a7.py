"""train/train_ablation_a7.py — Ablation A7: Vanilla PPO (no safety mechanisms)

Identical to train_rl_inv.py EXCEPT:
  - Uses standard PPO instead of MaskablePPO
  - No ActionMasker wrapper (no action masking)
  - No Lagrangian penalty (not in base TelecomEnv reward anyway)
  - No recovery policy override

This directly tests H2: do safety mechanisms (action masking) reduce
constraint violations vs unconstrained RL?

Compare A7 results vs RLInv on:
  - violations per episode
  - SoC violations
  - stockout events
  - EENS (reliability impact of constraint violations)

Run:
  python -m src.train.train_ablation_a7 --site site5 --lead normal
    --timesteps 400000 --seed 42 --logdir runs/ablation_a7

Evaluate after training:
  python src/eval/evaluate.py --site site5 --lead normal
    --policy_type rl --algo ppo
    --model_path runs/ablation_a7/site5_s42_final.zip
    --episodes 10 --seed 42
    --out_csv results/phase3/a7/a7_site5_normal_s42.csv
"""
from __future__ import annotations

import os
import sys
import argparse
from typing import Callable

import mlflow
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import ppo_cfg, policy_cfg, train_cfg, env_cfg, registry_cfg
from env.data_loader import load_site, train_test_split
from env.telecom_env import TelecomEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback


# ── Hyperparameters — identical to train_rl_inv.py for fair comparison ───────

TRAIN_EP_LEN = env_cfg["train_episode_len"]
EVAL_EP_LEN  = env_cfg["eval_episode_len"]


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def f(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return f


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
        # Violations are tracked via TelecomEnv's hard masking in step() —
        # the env still enforces physics, but the policy doesn't get the mask
        # signal to learn to avoid these.
        return env
    return _init


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site",             type=str,   default="site5")
    ap.add_argument("--lead",             type=str,   default="normal",
                    choices=["fast", "normal", "delayed", "very_delayed", "multi"])
    ap.add_argument("--timesteps",        type=int,   default=train_cfg["total_timesteps"])
    ap.add_argument("--tag",              type=str,   default="phase3")
    ap.add_argument("--seed",             type=int,   default=42)
    ap.add_argument("--logdir",           type=str,   default="runs/ablation_a7")
    ap.add_argument("--init_diesel_low",  type=float, default=0.6)
    ap.add_argument("--init_diesel_high", type=float, default=0.6)
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(args.tag)
    site     = args.site
    site_csv = f"data/processed/{site}.csv"

    print(f"[A7 Ablation] Training Vanilla PPO (no masking) on {site} | "
          f"lead={args.lead} | timesteps={args.timesteps}")

    # ── Training env ─────────────────────────────────────────────────────────
    vec_env = SubprocVecEnv([
        make_env(site_csv, args.seed + i, eval_mode=False,
                 lead_scenario=args.lead,
                 init_inv_frac_low=args.init_diesel_low,
                 init_inv_frac_high=args.init_diesel_high)
        for i in range(train_cfg["n_envs"])
    ])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False,
                           clip_obs=policy_cfg["obs_clip"])

    # ── Eval env ─────────────────────────────────────────────────────────────
    eval_env = DummyVecEnv([
        make_env(site_csv, args.seed + 10_000, eval_mode=True,
                 lead_scenario=args.lead,
                 init_inv_frac_low=args.init_diesel_low,
                 init_inv_frac_high=args.init_diesel_high)
    ])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=policy_cfg["obs_clip"])
    eval_env.training    = False
    eval_env.norm_reward = False

    # ── Model — standard PPO, no masking ─────────────────────────────────────
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
        policy_kwargs=dict(net_arch=policy_cfg["net_arch"]),
        verbose=1,
        tensorboard_log=args.logdir,
        seed=args.seed,
    )

    # ── Standard EvalCallback — no action masking ─────────────────────────────
    # NOTE: Must use EvalCallback (not MaskableEvalCallback/DetailedEvalCallback)
    # because A7 env has no ActionMasker wrapper. MaskableEvalCallback raises
    # ValueError if env doesn't support action masking.
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.logdir, f"{site}_best"),
        log_path=os.path.join(args.logdir, f"{site}_eval"),
        eval_freq=train_cfg["eval_freq"],
        n_eval_episodes=train_cfg["n_eval_episodes"],
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_cb)

    # ── Save ─────────────────────────────────────────────────────────────────
    model_path = os.path.join(args.logdir, f"{site}_s{args.seed}_final.zip")
    vn_path    = os.path.join(args.logdir, f"{site}_s{args.seed}_vecnorm.pkl")

    model.save(model_path)
    vec_env.save(vn_path)
    vec_env.close()
    eval_env.close()

    # ── MLflow logging ────────────────────────────────────────────────────────
    best_reward = eval_cb.best_mean_reward
    with mlflow.start_run(run_name=f"{site}_A7_s{args.seed}"):
        mlflow.log_params({
            "site":            site,
            "seed":            args.seed,
            "policy":          "A7",
            "lead_scenario":   args.lead,
            "gamma":           ppo_cfg["gamma"],
            "n_envs":          train_cfg["n_envs"],
            "total_timesteps": args.timesteps,
            "git_commit":      _get_git_commit(),
        })
        mlflow.log_metrics({
            "best_eval_reward": float(best_reward) if np.isfinite(best_reward) else -9999.0,
        })
        try:
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(vn_path)
        except Exception as e:
            print(f"[MLflow] Artifact logging skipped: {e}")

    print(f"[A7] Saved: {model_path}")
    print(f"[A7] Saved VecNormalize: {vn_path}")
    print("\nNext — evaluate with:")
    print(f"  python src/eval/evaluate.py --site {site} --lead {args.lead} "
          f"--policy_type rl --algo ppo "
          f"--model_path {model_path} "
          f"--episodes 10 --seed {args.seed} "
          f"--out_csv results/phase3/a7/a7_{site}_{args.lead}_s{args.seed}.csv")


if __name__ == "__main__":
    main()