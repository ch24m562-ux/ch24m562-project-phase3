"""
config_loader.py — load hparams.yaml once, use everywhere.

Usage in any script:
    from config_loader import cfg, env_cfg, ppo_cfg, reward_cfg

All constants previously hardcoded in telecom_env.py and train_rlinv.py
are now accessed via this loader. If you need to change a value, change it
in hparams.yaml only.
"""
from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any


def _find_hparams() -> Path:
    """Walk up from this file's location until hparams.yaml is found."""
    current = Path(__file__).resolve().parent
    for _ in range(5):  # search up to 5 levels
        candidate = current / "configs" / "hparams.yaml"
        if candidate.exists():
            return candidate
        candidate = current / "hparams.yaml"
        if candidate.exists():
            return candidate
        current = current.parent
    raise FileNotFoundError(
        "hparams.yaml not found. Place it in configs/ at the repo root."
    )


def _load() -> dict:
    path = _find_hparams()
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Module-level load (happens once on import) ───────────────
cfg: dict = _load()

# Convenient sub-sections — import these directly
env_cfg    = cfg["env"]
lead_cfg   = cfg["lead_time"]
reward_cfg = cfg["reward"]
ppo_cfg    = cfg["ppo"]
policy_cfg = cfg["policy"]
train_cfg  = cfg["training"]
eval_cfg   = cfg["evaluation"]
sS_cfg     = cfg["sS_policy"]
registry_cfg = cfg["registry"]


def get_lead_p(scenario: str) -> float:
    """Return geometric delivery probability for a named scenario."""
    if scenario not in lead_cfg:
        raise ValueError(
            f"Unknown lead scenario '{scenario}'. "
            f"Valid: {list(lead_cfg.keys())}"
        )
    return lead_cfg[scenario]["p"]


def get_multi_scenario_pool() -> list[str]:
    """Return list of scenarios to sample from during multi-scenario training."""
    return lead_cfg["multi_scenario_pool"]


if __name__ == "__main__":
    print("Loaded hparams.yaml successfully.")
    print()
    print(f"  gamma        = {ppo_cfg['gamma']}  "
          f"(effective horizon ~{1/(1-ppo_cfg['gamma']):.0f}h)")
    print(f"  n_envs       = {train_cfg['n_envs']}")
    print(f"  total_steps  = {train_cfg['total_timesteps']:,}")
    print(f"  net_arch     = {policy_cfg['net_arch']}")
    print(f"  lam          = {reward_cfg['lam']}")
    print(f"  gamma_r      = {reward_cfg['gamma_r']}")
    print(f"  train_len    = {env_cfg['train_len_steps']} steps "
          f"(= {env_cfg['train_days']} days × 24h)")
    print(f"  lead normal  = p={get_lead_p('normal'):.5f}  "
          f"(mean {lead_cfg['normal']['mean_hours']}h)")
    print(f"  lead delayed = p={get_lead_p('delayed'):.5f}  "
          f"(mean {lead_cfg['delayed']['mean_hours']}h)")
    print(f"  multi pool   = {get_multi_scenario_pool()}")
