"""    wrappers/drop_inventory_obs.py — DispatchOnlyEnv for Track B (B2)

Purpose:
  - Dispatch PPO sees a reduced observation vector WITHOUT inventory dimensions.
  - PPO action is DG-only (Discrete(2)).
  - (s,S) ordering is injected via an external order_fn.
  - Base TelecomEnv remains unchanged.

This implements the reviewer-preferred decomposition baseline fairly.
"""
from __future__ import annotations

from typing import Callable, Any, Optional, Tuple

import numpy as np
import gymnasium as gym


class DispatchOnlyEnv(gym.Env):
    """A wrapper env exposing DG-only control and non-inventory observations."""

    metadata = {"render_modes": ["text"]}

    # Keep indices from TelecomEnv obs:
    # [soc_n, inv_n, pending_flag, pending_qty_n, pv_n, load_n, grid, sin_h, cos_h]
    KEEP_IDX = (0, 4, 5, 6, 7, 8)  # 6-D: soc, pv, load, grid, sin, cos

    def __init__(self, base_env: Any, order_fn: Callable[[np.ndarray, Any], int]):
        super().__init__()
        self.base_env = base_env
        self.order_fn = order_fn

        self.action_space = gym.spaces.Discrete(2)  # 0=DG off, 1=DG on

        # Observation space mirrors base_env bounds but reduced dims
        low = np.full((len(self.KEEP_IDX),), -2.0, dtype=np.float32)
        high = np.full((len(self.KEEP_IDX),), 2.0, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        return self._drop(obs), info

    def step(self, action: int):
        # DG from action, order from order_fn (based on full base obs)
        base_obs = self.base_env._get_obs() if hasattr(self.base_env, "_get_obs") else None
        if base_obs is None:
            raise RuntimeError("base_env does not expose _get_obs(); required for Track B wrapper.")

        order = int(self.order_fn(base_obs, self.base_env))  # 0/1/2
        dg = int(action)

        combined = dg * 3 + order  # Discrete(6) expected by base env

        # Respect base env mask
        if hasattr(self.base_env, "get_action_mask"):
            mask6 = self.base_env.get_action_mask()
            if not bool(mask6[combined]):
                combined = 0  # safe fallback

        obs, reward, terminated, truncated, info = self.base_env.step(combined)
        return self._drop(obs), reward, terminated, truncated, info

    def _drop(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        return obs[list(self.KEEP_IDX)].astype(np.float32)

    def get_action_mask(self) -> np.ndarray:
        """Return 2-length mask (DG off always allowed; DG on allowed iff any DG-on action feasible in base env)."""
        mask2 = np.ones(2, dtype=bool)
        if hasattr(self.base_env, "get_action_mask"):
            m6 = self.base_env.get_action_mask()
            mask2[1] = bool(m6[3])  # dg=1, order=0
        return mask2

    def render(self):
        if hasattr(self.base_env, "render"):
            return self.base_env.render()

    def close(self):
        if hasattr(self.base_env, "close"):
            return self.base_env.close()
    
    def get_episode_stats(self):
        # Forward stats from the underlying TelecomEnv
        if hasattr(self.base_env, "get_episode_stats"):
            return self.base_env.get_episode_stats()
        return {}
    def close(self):
        if hasattr(self.base_env, "close"):
            self.base_env.close()
