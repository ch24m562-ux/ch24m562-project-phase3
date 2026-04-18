"""train_rlinv_phase3.py — Track A: RL-Inv (MaskablePPO) — Phase 3 param-clean

Phase 3 changes from original (ONLY these 3 changes — no other logic altered):
  1. All hyperparameters read from hparams.yaml via config_loader — no hardcoded values
  2. --lead choices expanded to include "multi" (multi-scenario training)
  3. Experiment registry logging added after each run (new — does not affect training)
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

# ── [CHANGE 1] All hyperparameters from hparams.yaml ─────────────────────────
from config_loader import ppo_cfg, policy_cfg, train_cfg, env_cfg, registry_cfg


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def f(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return f


# ── VecNormalize sync callback (UNCHANGED) ────────────────────────────────────
class SyncNormEvalCallback(EvalCallback):
    """EvalCallback that syncs VecNormalize stats from train env to eval env."""

    def _sync_vecnormalize(self):
        train_vn = self.model.get_vec_normalize_env()
        eval_vn  = self.eval_env
        if train_vn is None:
            return
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


def mask_fn(env) -> np.ndarray:
    return env.unwrapped.get_action_mask()


# ── make_env — SAME STRUCTURE AS ORIGINAL ────────────────────────────────────
# Original pattern: split outside, pass data slice to TelecomEnv.
# Only change: episode_len and init_inv_frac_* now read from hparams.
def make_env(site_csv: str, seed: int, eval_mode: bool, lead_scenario: str):
    def _init():
        df, params = load_site(site_csv)
        df_train, df_test = train_test_split(df)
        data = df_test if eval_mode else df_train

        # [CHANGE 1] episode_len and inv_frac from hparams — not hardcoded
        ep_len = env_cfg["eval_episode_len"] if eval_mode else env_cfg["train_episode_len"]
        inv_low  = env_cfg["init_inv_frac_eval_low"]  if eval_mode else env_cfg["init_inv_frac_train"]
        inv_high = env_cfg["init_inv_frac_eval_high"] if eval_mode else env_cfg["init_inv_frac_train"]

        env = TelecomEnv(
            site_data=data,
            site_params=params,
            episode_len=ep_len,
            eval_mode=eval_mode,
            lead_scenario=lead_scenario,
            seed=seed,
            init_inv_frac_low=inv_low,
            init_inv_frac_high=inv_high,
        )
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)
        return env
    return _init


# ── [CHANGE 3] Registry helper — new, does not affect training ────────────────
def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _log_to_registry(record: dict):
    path = registry_cfg["output_csv"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = registry_cfg["columns"]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(record)


# ── main — same flow as original, only constants replaced ────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site",      type=str,  default="site1")
    ap.add_argument("--all_sites", action="store_true",
                    help="train site1, site7, site5 sequentially")
    # [CHANGE 2] added "multi" as valid choice for Phase 3 multi-scenario training
    ap.add_argument("--lead",      type=str,  default="normal",
                    choices=["fast", "normal", "delayed", "very_delayed", "multi"],
                    help="lead time scenario. 'multi' samples from pool each episode")
    # [CHANGE 1] default timesteps from hparams
    ap.add_argument("--timesteps", type=int,  default=train_cfg["total_timesteps"])
    ap.add_argument("--seed",      type=int,  default=42)
    ap.add_argument("--logdir",    type=str,  default="runs/track_a")
    # [CHANGE 3] experiment tag for registry
    ap.add_argument("--tag",       type=str,  default="phase3",
                    help="Experiment tag written to registry CSV")
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    sites = ["site1", "site7", "site5"] if args.all_sites else [args.site]

    for site in sites:
        site_csv = f"data/processed/{site}.csv"

        # [CHANGE 1] n_envs from hparams (was N_ENVS=8 hardcoded, actual was 4)
        n_envs = train_cfg["n_envs"]

        vec_env = SubprocVecEnv([
            make_env(site_csv, args.seed + i, eval_mode=False, lead_scenario=args.lead)
            for i in range(n_envs)
        ])
        # [CHANGE 1] clip_obs from hparams (was 10.0 hardcoded)
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=policy_cfg["obs_clip"],
        )

        eval_env = DummyVecEnv([
            make_env(site_csv, args.seed + 10_000, eval_mode=True, lead_scenario=args.lead)
        ])
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=policy_cfg["obs_clip"],
        )
        eval_env.training = False

        # [CHANGE 1] policy network from hparams (was POLICY_NET=[256,256] hardcoded)
        policy_kwargs = dict(net_arch=policy_cfg["net_arch"])

        # [CHANGE 1] ALL PPO params from hparams — gamma=0.99 is now authoritative
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
            gamma          = ppo_cfg["gamma"],
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

        # [CHANGE 1] eval_freq and n_eval_episodes from hparams
        eval_cb = SyncNormEvalCallback(
            eval_env,
            best_model_save_path = os.path.join(args.logdir, f"{site}_best"),
            log_path             = os.path.join(args.logdir, f"{site}_eval"),
            eval_freq            = train_cfg["eval_freq"],
            n_eval_episodes      = train_cfg["n_eval_episodes"],
            deterministic        = True,
            render               = False,
        )

        model.learn(total_timesteps=args.timesteps, callback=eval_cb)

        # Save (UNCHANGED structure)
        model_path = os.path.join(args.logdir, f"{site}_final_model.zip")
        model.save(model_path)

        vn_path = os.path.join(args.logdir, f"{site}_vecnormalize.pkl")
        vec_env.save(vn_path)

        vec_env.close()
        eval_env.close()

        print(f"[Track A] Saved: {model_path}")
        print(f"[Track A] Saved VecNormalize: {vn_path}")

        # [CHANGE 3] Log to experiment registry (new — does not affect training)
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
        })


if __name__ == "__main__":
    main()
