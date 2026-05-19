"""
env/obs_wrappers.py — Observation wrappers for ablation studies.

NoInvObsWrapper (used by A5 ablation):
  Zeros out ALL inventory-related observation dimensions so the policy
  cannot observe or use any inventory state — direct or indirect.
  Obs space shape is UNCHANGED (11D) — zeroed dims set to 0.0 each step.

  Obs layout (TelecomEnv Phase 3, 11D):
    [0] soc_n               ← kept   (battery state — dispatch relevant)
    [1] inv_n               ← ZEROED (inventory level)
    [2] pending_flag        ← ZEROED (in-transit order flag)
    [3] pending_qty_n       ← ZEROED (in-transit quantity)
    [4] pv_n                ← kept
    [5] load_n              ← kept
    [6] grid                ← kept
    [7] sin_h               ← kept
    [8] cos_h               ← kept
    [9] hours_since_order_n ← ZEROED (indirect inventory timing info)
   [10] delivery_remaining_n← ZEROED (indirect delivery timing info)

  Why dims 9 and 10 must also be zeroed:
    hours_since_order_n  encodes elapsed time since the last order —
    indirect inventory information. A policy that sees this can infer
    whether a delivery is likely imminent. Zeroing it ensures H3 tests
    ONLY whether knowing inventory level matters, not order timing.
    delivery_remaining_n is always 0.0 in main experiments (use_eta_obs=False),
    but zeroed explicitly here for correctness under any configuration.

  The environment STILL tracks inventory internally and applies action
  masking correctly. The policy simply cannot observe it.
  Tests H3: does knowing inventory level improve ordering decisions?

  Design note: zeroing rather than dropping keeps obs_space shape
  identical to RLInv — a clean single-variable ablation.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ALL inventory-related indices in TelecomEnv 11D obs
# Dims 1,2,3 = direct inventory state
# Dims 9,10  = indirect timing info derived from ordering/delivery
INV_OBS_INDICES = [1, 2, 3, 9, 10]


class NoInvObsWrapper(gym.ObservationWrapper):
    """
    Zeros out all inventory observation dimensions (direct + indirect).
    Obs space shape and dtype are unchanged — compatible with
    the same VecNormalize stats as the full RLInv model.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box), \
            "NoInvObsWrapper requires a Box observation space"
        assert (len(env.observation_space.shape) == 1
                and env.observation_space.shape[0] >= 10), \
            (f"Expected TelecomEnv obs with ≥10 dims, "
             f"got {env.observation_space.shape}")

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.copy()
        for idx in INV_OBS_INDICES:
            obs[idx] = 0.0
        return obs