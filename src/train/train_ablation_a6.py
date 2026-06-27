"""train/train_ablation_a6.py — Ablation A6: RL controls DG only; ordering by (s,S) policy.

Same observation space, reward function, and environment dynamics as train_rl_inv.py EXCEPT:
  - Action space reduced to Discrete(2): RL agent picks DG {0=OFF, 1=ON} only.
  - Order action is computed at every step by calibrated SSPolicy (classical inventory).
  - Action masking is KEPT — DG=1 still masked when inventory < min_fuel.

Scientific purpose (H1 — cleaner ordering ablation than TrackB):
  RLInv:  RL controls BOTH DG dispatch AND ordering          (Discrete(6))
  TrackB: (s,S) ordering + SEPARATE PPO dispatch policy      (confounded: different
                                                              dispatch policy AND action space)
  A6:     (s,S) ordering + RL dispatch (Discrete(2), same obs/reward/dynamics)

  RLInv vs A6  →  cleaner H1 test: ordering mechanism is primary difference.
                   Dispatch is re-learned under Discrete(2) — not perfectly identical
                   to RLInv dispatch, but far less confounded than TrackB.

  Thesis language: "A6 provides a cleaner ordering ablation than TrackB; the dispatch
  policy is re-learned under a reduced action space, so the comparison isolates
  ordering better but not perfectly."

Run single site:
  python -m src.train.train_ablation_a6 --site site5 --lead normal \\
    --timesteps 400000 --seed 42 \\
    --logdir runs/ablation_a6/site5/seed42

Run all sites (sequentially):
  python -m src.train.train_ablation_a6 --all_sites --lead normal \\
    --timesteps 400000 --seed 42 \\
    --logdir runs/ablation_a6

Evaluate after training:
  python -m src.eval.evaluate --site site5 --lead normal \\
    --policy_type rl --algo maskable --env_type a6 \\
    --model_path runs/ablation_a6/site5/seed42/site5_s{args.seed}_final \\
    --vecnorm_path runs/ablation_a6/site5/seed42/site5_vecnorm.pkl \\
    --episodes 30 --seed 42 \\
    --policy_label A6 --train_scenario normal --experiment_tag ablation_a6 \\
    --train_steps 400000 --init_diesel_low 0.3 --init_diesel_high 0.9 \\
    --out_csv results/ablation_a6/site5/seed42/eval_normal.csv

NOTE on evaluate.py:
  If evaluate.py does not yet support --env_type a6, add the following
  in the env construction block (same place as the a5 branch):
    elif env_type == "a6":
        from env.a6_env import make_a6_env
        env = make_a6_env(site_data=data, site_params=params,
                          lead_scenario=lead_scenario, **env_kwargs)
"""
from __future__ import annotations

import os
import sys
import argparse
from typing import Callable, List

import mlflow
import csv
import subprocess
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import ppo_cfg, policy_cfg, train_cfg, env_cfg, registry_cfg
from env.data_loader import load_site, train_test_split
from env.a6_env import make_a6_env

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


# ── Hyperparameters — IDENTICAL to train_rl_inv.py and train_ablation_a5.py ──

# timesteps from hparams   # same as all_sites runs (convergence check confirmed sufficient)
# n_envs from hparams

TRAIN_EP_LEN = env_cfg["train_episode_len"]
EVAL_EP_LEN  = env_cfg["eval_episode_len"]

# policy_net from hparams
# lr from hparams

# All 10 sites — same order as all_sites runs
ALL_SITES: List[str] = [
    "site1", "site2", "site3", "site4", "site5",
    "site6", "site7", "site8", "site9", "site10",
]


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def f(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return f


# ── VecNormalize sync callback — identical to train_ablation_a5.py ────────────

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


# ── Action mask function — reads 2-bool mask from A6Env ──────────────────────

def get_action_mask(env) -> np.ndarray:
    """Unwrap to A6Env and return 2-bool DG mask."""
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base.get_action_mask()


# ── Env factory ───────────────────────────────────────────────────────────────

def make_env(
    site_csv: str,
    seed: int,
    eval_mode: bool,
    lead_scenario: str,
    init_inv_frac_low: float = 0.6,
    init_inv_frac_high: float = 0.6,
):
    def _init():
        df, params = load_site(site_csv)
        df_train, df_test = train_test_split(df)
        data = df_test if eval_mode else df_train

        # A6: use make_a6_env — calibrates SSPolicy from site_params automatically
        env = make_a6_env(
            site_data=data,
            site_params=params,
            lead_scenario=lead_scenario,
            episode_len=(EVAL_EP_LEN if eval_mode else TRAIN_EP_LEN),
            eval_mode=eval_mode,
            seed=seed,
            init_inv_frac_low=init_inv_frac_low,
            init_inv_frac_high=init_inv_frac_high,
        )
        env = Monitor(env)
        # ActionMasker: returns 2-bool mask matching Discrete(2) action space
        env = ActionMasker(env, get_action_mask)
        return env
    return _init


# ── Git commit helper ─────────────────────────────────────────────────────────

def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ── Training loop ─────────────────────────────────────────────────────────────

def train_site(site: str, args) -> None:
    site_csv = f"data/processed/{site}.csv"
    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[A6 Ablation] MaskablePPO — DG only, (s,S) ordering")
    print(f"  Site:        {site}")
    print(f"  Lead:        {args.lead}")
    print(f"  Timesteps:   {args.timesteps:,}")
    print(f"  Seed:        {args.seed}")
    print(f"  Logdir:      {logdir}")
    print(f"  Action space: Discrete(2) — DG only, order delegated to s,S")
    print(f"{'='*60}")

    # ── Training env ──────────────────────────────────────────────────────────
    vec_env = SubprocVecEnv([
        make_env(
            site_csv, args.seed + i, eval_mode=False,
            lead_scenario=args.lead,
            init_inv_frac_low=args.init_diesel_low,
            init_inv_frac_high=args.init_diesel_high,
        )
        for i in range(train_cfg["n_envs"])
    ])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=policy_cfg["obs_clip"])

    # ── Eval env ──────────────────────────────────────────────────────────────
    # CRITICAL: init inventory [0.3, 0.9] matches RLInv final evaluation range.
    # Using the same range ensures A6 vs RLInv comparison is apples-to-apples.
    eval_env = DummyVecEnv([
        make_env(site_csv, args.seed + 10_000, eval_mode=True,
                 lead_scenario=args.lead,
                 init_inv_frac_low=0.3, init_inv_frac_high=0.9)
    ])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=policy_cfg["obs_clip"])
    eval_env.training    = False
    eval_env.norm_reward = False

    # ── Model — MaskablePPO, Discrete(2) ─────────────────────────────────────
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
        tensorboard_log=logdir,
        seed=args.seed,
    )

    eval_cb = SyncNormEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(logdir, f"{site}_best"),
        log_path=os.path.join(logdir, f"{site}_eval"),
        eval_freq=train_cfg["eval_freq"],
        n_eval_episodes=train_cfg["n_eval_episodes"],
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_cb)

    # ── Save ──────────────────────────────────────────────────────────────────
    model_path = os.path.join(logdir, f"{site}_s{args.seed}_final.zip")
    vn_path    = os.path.join(logdir, f"{site}_s{args.seed}_vecnorm.pkl")

    model.save(model_path)
    vec_env.save(vn_path)

    vec_env.close()
    eval_env.close()

    # ── MLflow logging ────────────────────────────────────────────────────────
    best_reward = eval_cb.best_mean_reward
    with mlflow.start_run(run_name=f"{site}_A6_s{args.seed}"):
        mlflow.log_params({
            "site": site, "seed": args.seed, "policy": "A6",
            "lead_scenario": args.lead, "gamma": ppo_cfg["gamma"],
            "n_envs": train_cfg["n_envs"], "total_timesteps": args.timesteps,
            "git_commit": _get_git_commit(),
        })
        mlflow.log_metrics({
            "best_eval_reward": float(best_reward) if np.isfinite(best_reward) else -9999.0,
        })
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(vn_path)

    print(f"\n[A6] Saved model:   {model_path}")
    print(f"[A6] Saved VecNorm: {vn_path}")
    print(f"\nEvaluate with:")
    print(f"  python -m src.eval.evaluate \\")
    print(f"    --site {site} --lead {args.lead} \\")
    print(f"    --policy_type rl --algo maskable --env_type a6 \\")
    print(f"    --model_path {model_path.replace('.zip', '')} \\")
    print(f"    --vecnorm_path {vn_path} \\")
    print(f"    --episodes 30 --seed {args.seed} \\")
    print(f"    --policy_label A6 --train_scenario {args.lead} \\")
    print(f"    --experiment_tag ablation_a6 --train_steps {args.timesteps} \\")
    print(f"    --init_diesel_low 0.3 --init_diesel_high 0.9 \\")
    print(f"    --out_csv results/ablation_a6/{site}/seed{args.seed}/eval_{args.lead}.csv")


def main():
    ap = argparse.ArgumentParser(
        description="A6 Ablation: MaskablePPO DG-only + classical (s,S) ordering"
    )
    ap.add_argument("--site",             type=str,  default="site5",
                    help="site name, e.g. site5 (ignored if --all_sites)")
    ap.add_argument("--all_sites",        action="store_true",
                    help="train all 10 sites sequentially")
    ap.add_argument("--lead",             type=str,  default="normal",
                    choices=["fast", "normal", "delayed", "very_delayed", "multi"])
    ap.add_argument("--timesteps",        type=int,  default=train_cfg["total_timesteps"])
    ap.add_argument("--tag",              type=str,  default="phase3")
    ap.add_argument("--seed",             type=int,  default=42)
    ap.add_argument("--logdir",           type=str,  default="runs/ablation_a6")
    ap.add_argument("--init_diesel_low",  type=float, default=0.6,
                    help="TRAINING init inv low — keep 0.6 to match RLInv training")
    ap.add_argument("--init_diesel_high", type=float, default=0.6,
                    help="TRAINING init inv high — keep 0.6 to match RLInv training")
    # NOTE: eval callback always uses [0.3, 0.9] regardless of these args (hardcoded
    # in train_site() to match RLInv final evaluation range). These args only
    # affect training episode initialisation.
    args = ap.parse_args()

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(args.tag)

    sites = ALL_SITES if args.all_sites else [args.site]

    for site in sites:
        train_site(site, args)

    print(f"\n[A6] All done. Trained {len(sites)} site(s).")


if __name__ == "__main__":
    main()