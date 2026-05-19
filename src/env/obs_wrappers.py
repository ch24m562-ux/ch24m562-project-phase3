"""
env/obs_wrappers.py — Observation wrappers for ablation studies.

NoInvObsWrapper (used by A5 ablation):
  Zeros out inventory-related observation dimensions so the policy
  cannot observe or use inventory state. The observation space shape
  is UNCHANGED (still 9D) — inventory dims are set to 0.0 each step.

  Obs layout (TelecomEnv, 9D):
    [0] soc_n          ← kept
    [1] inv_n          ← ZEROED (inventory level)
    [2] pending_flag   ← ZEROED (in-transit order flag)
    [3] pending_qty_n  ← ZEROED (in-transit quantity)
    [4] pv_n           ← kept
    [5] load_n         ← kept
    [6] grid           ← kept
    [7] sin_h          ← kept
    [8] cos_h          ← kept

  The environment STILL tracks inventory internally and applies
  action masking correctly. The policy simply cannot see it.
  This tests H3: does knowing inventory level improve ordering decisions?

  Design note: zeroing rather than dropping keeps obs_space shape
  identical to RLInv, making the ablation a clean single-variable test.
  The policy receives a valid 9D obs — it just sees zeros for inv dims.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Indices in TelecomEnv obs that represent inventory state
INV_OBS_INDICES = [1, 2, 3]   # inv_n, pending_flag, pending_qty_n


class NoInvObsWrapper(gym.ObservationWrapper):
    """
    Zeros out inventory observation dimensions.
    Obs space shape and dtype are unchanged — compatible with
    the same VecNormalize stats as the full RLInv model.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Observation space is identical — same bounds, same shape
        # (we just always return 0 for inventory dims at runtime)
        assert isinstance(env.observation_space, spaces.Box), \
            "NoInvObsWrapper requires a Box observation space"
        assert (len(env.observation_space.shape) == 1
                and env.observation_space.shape[0] >= 10), \
            (f"Expected TelecomEnv obs with ≥10 dims (was 9D before Phase 3, "
             f"now 11D), got {env.observation_space.shape}")

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.copy()
        for idx in INV_OBS_INDICES:
            obs[idx] = 0.0
        return obs