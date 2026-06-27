"""train/train_ablation_a5.py — Ablation A5: No Inventory State in Observation

Identical to train_rl_inv.py EXCEPT:
  - Obs indices [1,2,3] (inv_n, pending_flag, pending_qty_n) are zeroed
    via NoInvObsWrapper — policy cannot observe inventory state.
  - Action masking is KEPT (MaskablePPO + ActionMasker).
  - All other hyperparameters identical.

This directly tests H3: does explicit inventory state in the MDP improve
ordering decisions vs a policy that cannot observe inventory?

The policy still has an "order" action available — it just cannot
condition on current inventory level or pending orders when deciding
whether/how much to order. Compare vs A1 (Full SC-PPO):
  - If A1 EENS < A5 EENS → inventory state helps → H3 confirmed
  - If similar → ordering behaviour is not driven by inventory obs

Run:
  python -m src.train.train_ablation_a5 --site site5 --lead normal \\
    --timesteps 400000 --seed 42 --logdir runs/ablation_a5/site5/seed42

Evaluate after training:
  python -m src.eval.evaluate --site site5 --lead normal \\
    --policy_type rl --algo maskable --env_type a5 \\
    --model_path runs/ablation_a5/site5/seed42/site5_s{args.seed}_final \\
    --vecnorm_path runs/ablation_a5/site5/seed42/site5_vecnorm.pkl \\
    --episodes 30 --seed 42 \\
    --policy_label A5 --train_scenario normal --experiment_tag ablation_a5 \\
    --train_steps 400000 --init_diesel_low 0.3 --init_diesel_high 0.9 \\
    --out_csv results/ablation_a5/site5/seed42/eval_normal.csv
"""
from __future__ import annotations

import os
import sys
import argparse
from typing import Callable

import mlflow
import csv
import subprocess
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import ppo_cfg, policy_cfg, train_cfg, env_cfg, registry_cfg
from env.data_loader import load_site, train_test_split
from env.telecom_env import TelecomEnv
from env.obs_wrappers import NoInvObsWrapper

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


# ── Hyperparameters — identical to train_rl_inv.py for fair comparison ────────

# timesteps from hparams
N_ENVS            = 8

TRAIN_EP_LEN = env_cfg["train_episode_len"]
EVAL_EP_LEN  = env_cfg["eval_episode_len"]

# policy_net from hparams
# lr from hparams


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def f(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return f


# ── VecNormalize sync callback ────────────────────────────────────────────────

class SyncNormEvalCallback(EvalCallback):
    def _sync_vecnormalize(self):
        train_vn = self.model.get_vec_normalize_env()
        eval_vn  = self.eval_env
        if not isinstance(train_vn, VecNormalize):
            return
        if not isinstance(eval_vn, VecNormalize):
            return
        eval_vn.obs_rms    = train_vn.obs_rms
        eval_vn.ret_rms    = train_vn.ret_rms
        eval_vn.training   = False
        eval_vn.norm_reward = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.n_calls % self.eval_freq == 0):
            self._sync_vecnormalize()
        return super()._on_step()


# ── Action mask function (same as train_rl_inv.py) ────────────────────────────

def get_action_mask(env) -> np.ndarray:
    """Unwrap to TelecomEnv and return action mask."""
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base.get_action_mask()


# ── Env factory — WITH NoInvObsWrapper, WITH ActionMasker ────────────────────

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
        # A5: zero out inventory obs dims — policy cannot see inventory
        env = NoInvObsWrapper(env)
        env = Monitor(env)
        # Keep action masking — only inventory STATE is removed, not safety
        env = ActionMasker(env, get_action_mask)
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
    ap = argparse.ArgumentParser(
        description="A5 Ablation: MaskablePPO without inventory state in obs"
    )
    ap.add_argument("--site",              type=str,   default="site5")
    ap.add_argument("--lead",              type=str,   default="normal",
                    choices=["fast", "normal", "delayed", "very_delayed", "multi"])
    ap.add_argument("--timesteps",         type=int,   default=train_cfg["total_timesteps"])
    ap.add_argument("--tag", type=str, default="phase3")
    ap.add_argument("--seed",              type=int,   default=42)
    ap.add_argument("--logdir",            type=str,   default="runs/ablation_a5")
    ap.add_argument("--init_diesel_low",   type=float, default=0.6)
    ap.add_argument("--init_diesel_high",  type=float, default=0.6)
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(getattr(args, "tag", "phase3"))
    site     = args.site
    site_csv = f"data/processed/{site}.csv"

    print(f"[A5 Ablation] MaskablePPO, NO inventory obs | {site} | "
          f"lead={args.lead} | timesteps={args.timesteps} | seed={args.seed}")
    print(f"  Zeroed obs dims: [1]=inv_n  [2]=pending_flag  [3]=pending_qty_n")

    # ── Training env ──────────────────────────────────────────────────────────
    vec_env = SubprocVecEnv([
        make_env(site_csv, args.seed + i, eval_mode=False,
                 lead_scenario=args.lead,
                 init_inv_frac_low=args.init_diesel_low,
                 init_inv_frac_high=args.init_diesel_high)
        for i in range(train_cfg["n_envs"])
    ])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=policy_cfg["obs_clip"])

    # ── Eval env ──────────────────────────────────────────────────────────────
    eval_env = DummyVecEnv([
        make_env(site_csv, args.seed + 10_000, eval_mode=True,
                 lead_scenario=args.lead)
    ])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=policy_cfg["obs_clip"])
    eval_env.training   = False
    eval_env.norm_reward = False

    # ── Model — MaskablePPO, no inventory in obs ───────────────────────────────
    model = MaskablePPO(
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

    # ── Save ──────────────────────────────────────────────────────────────────
    model_path = os.path.join(args.logdir, f"{site}_s{args.seed}_final.zip")
    vn_path    = os.path.join(args.logdir, f"{site}_s{args.seed}_vecnorm.pkl")

    model.save(model_path)
    vec_env.save(vn_path)

    vec_env.close()
    eval_env.close()

    print(f"\n[A5] Saved model:      {model_path}")
    print(f"[A5] Saved VecNorm:    {vn_path}")
    print(f"\nNext — evaluate with:")
    print(f"  python -m src.eval.evaluate \\")
    print(f"    --site {site} --lead {args.lead} \\")
    print(f"    --policy_type rl --algo maskable --env_type a5 \\")
    print(f"    --model_path {model_path.replace('.zip','')} \\")
    print(f"    --vecnorm_path {vn_path} \\")
    print(f"    --episodes 30 --seed {args.seed} \\")
    print(f"    --policy_label A5 --train_scenario {args.lead} \\")
    print(f"    --experiment_tag ablation_a5 --train_steps {args.timesteps} \\")
    print(f"    --init_diesel_low 0.3 --init_diesel_high 0.9 \\")
    print(f"    --out_csv results/ablation_a5/{site}/seed{args.seed}/eval_{args.lead}.csv")


    # ── MLflow logging ────────────────────────────────────────────────────
    best_reward = eval_cb.best_mean_reward
    with mlflow.start_run(run_name=f"{site}_A5_s{args.seed}"):
        mlflow.log_params({
            "site": site, "seed": args.seed, "policy": "A5",
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