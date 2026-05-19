"""env/a6_env.py — A6 Ablation: RL controls DG only; ordering fixed by (s,S) policy.

Design:
  - Same observation space (9D), reward function, and environment dynamics as RLInv.
  - REDUCED action space: Discrete(2) — RL agent picks DG {0=OFF, 1=ON} only.
  - Order action computed at every step by calibrated SSPolicy (classical inventory).
  - get_action_mask() returns 2-bool mask (DG feasibility only).

Important: A6 is NOT perfectly identical to RLInv — the RL agent learns a dispatch
  policy under a different (smaller) action space. Gradient updates, exploration, and
  learned representations all differ because the agent never sees order outcomes.
  A6 is a CLEANER ordering ablation than TrackB, not a perfectly controlled one.

Scientific purpose (H1 — cleaner isolation than TrackB):
  RLInv:  RL controls BOTH DG dispatch AND ordering          (Discrete(6))
  TrackB: (s,S) ordering + SEPARATE PPO dispatch policy      (confounded: different
                                                              dispatch policy AND
                                                              different action space)
  A6:     (s,S) ordering + RLInv-style RL dispatch           (Discrete(2), same obs/
                                                              reward/dynamics as RLInv)

  RLInv vs A6  → cleaner H1 test: ordering mechanism is the primary difference,
                  though dispatch is learned under a reduced action space.
  RLInv vs TrackB → confounded: both ordering AND dispatch policy differ.

  Thesis framing: "A6 provides a cleaner ordering ablation than TrackB; the dispatch
  policy is re-learned under a reduced action space, so the comparison isolates
  ordering better but not perfectly."

Place this file at: env/a6_env.py
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces

try:
    from env.telecom_env import TelecomEnv
    from baselines.s_S_policy import SSPolicy
except ModuleNotFoundError:
    from src.env.telecom_env import TelecomEnv
    from src.baselines.s_S_policy import SSPolicy

class A6Env(TelecomEnv):
    """
    TelecomEnv subclass for ablation A6.

    Same observation space, reward function, and environment dynamics as TelecomEnv.
    Reduced action space: Discrete(2) — RL agent picks DG {0=OFF, 1=ON} only.
    Ordering is delegated to SSPolicy at every step.

    The dispatch policy is re-learned under this reduced action space, so A6 vs RLInv
    is a cleaner ordering ablation than TrackB, not a perfectly controlled experiment.
    Use "cleaner ordering ablation" in thesis language, not "perfectly controlled."
    """

    def __init__(
        self,
        ss_policy: Optional[SSPolicy] = None,
        **kwargs,
    ):
        """
        Args:
            ss_policy: Calibrated SSPolicy instance. If None, uses SSPolicy()
                       defaults (s=0.25, S=0.85). Always pass a calibrated
                       instance via make_a6_env() for correct behaviour.
            **kwargs:  All TelecomEnv constructor arguments passed through.
        """
        super().__init__(**kwargs)

        # Reduce action space: agent only picks DG ∈ {0, 1}
        self.action_space = spaces.Discrete(2)

        # s,S policy for ordering — calibrated per site in make_a6_env()
        self.ss_policy: SSPolicy = ss_policy if ss_policy is not None else SSPolicy()

        # Cache last obs so step() can read inv_n and pending_flag
        # 11D — matches Phase 3 TelecomEnv observation space
        self._last_obs: np.ndarray = np.zeros(
            self.observation_space.shape[0], dtype=np.float32
        )

    # ──────────────────────────────────────────────────────────────────────
    # Core override: intercept DG-only action, inject s,S order decision
    # ──────────────────────────────────────────────────────────────────────

    def step(self, action: Any) -> Tuple:
        """
        action: int ∈ {0, 1} — DG decision from RL agent.

        Internally:
          1. Read inv_n and pending_flag from cached last obs (normalised
             values — SSPolicy thresholds are also normalised fractions,
             so comparison is valid in normalised space).
          2. Get order action from SSPolicy.
          3. Reconstruct full Discrete(6) action: full_a = dg*3 + order.
          4. Pass to parent TelecomEnv.step().
          5. Log requested vs executed order with blocked/clipped status.

        Logging fields added to info:
          a6_dg_action       : int  — DG decision from RL agent {0,1}
          a6_ss_order_req    : int  — order requested by s,S policy {0,1,2}
          a6_order_executed  : int  — order that actually passed masking {0,1,2}
                                      (may differ from req if parent blocked it)
          a6_order_qty_kwh   : float — kWh actually ordered after clipping
          a6_order_blocked   : bool — True if parent masked order to 0
          a6_order_clipped   : bool — True if parent reduced qty (tank near full)

        Note: parent env's _apply_action_mask() can block or clip orders:
          - Blocked: pending order already in flight → order forced to 0
          - Clipped: requested qty > remaining tank space → qty reduced
        Both are physical constraints, not policy decisions. Logging them
        lets evaluation distinguish s,S intent from env-enforced outcomes.
        """
        dg = int(action) % 2  # safety clamp to {0,1}

        # _last_obs indices (normalised): [soc, inv_n, pending_flag, pending_qty_n, ...]
        inv_n        = float(self._last_obs[1])  # normalised inventory fraction
        pending_flag = float(self._last_obs[2])  # 0.0 or 1.0

        # s,S policy returns requested order ∈ {0, 1, 2}
        order_requested = self.ss_policy.order_action(inv_n=inv_n, pending_flag=pending_flag)

        # Reconstruct full action for parent env
        full_action = dg * 3 + order_requested

        obs, reward, terminated, truncated, info = super().step(full_action)
        self._last_obs = np.asarray(obs, dtype=np.float32)

        # ── Decode what actually happened inside parent._apply_action_mask() ──
        # Parent info already contains: order_qty_kwh, mask_info (order_blocked, order_clipped)
        mask_info      = info.get("mask_info", {})
        order_blocked  = bool(mask_info.get("order_blocked", False))
        order_clipped  = bool(mask_info.get("order_clipped", False))
        order_qty_kwh  = float(info.get("order_qty_kwh", 0.0))

        # Infer executed order level {0,1,2} from actual kWh placed.
        # PRIMARY TRUTH for analysis is a6_order_qty_kwh (actual kWh).
        # The level inference below is approximate: if clipping produces a value
        # near the small/large threshold boundary, classification may be ambiguous.
        # Always use a6_order_qty_kwh for quantitative analysis; use level only
        # for order-frequency counts where boundary cases are negligible.
        if order_blocked or order_qty_kwh < 1e-6:
            order_executed = 0
        elif order_qty_kwh <= self.q_small_kwh + 1e-6:
            order_executed = 1
        else:
            order_executed = 2

        # ── Append A6-specific fields to info ─────────────────────────────────
        info["a6_dg_action"]      = int(dg)
        info["a6_ss_order_req"]   = int(order_requested)   # what s,S asked for
        info["a6_order_executed"] = int(order_executed)    # what env actually did
        info["a6_order_qty_kwh"]  = float(order_qty_kwh)  # kWh actually ordered
        info["a6_order_blocked"]  = order_blocked          # env blocked entirely
        info["a6_order_clipped"]  = order_clipped          # env reduced qty

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple:
        obs, info = super().reset(**kwargs)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def get_action_mask(self) -> np.ndarray:
        """
        Return 2-bool mask for DG-only action space.

        dg=0 (indices 0,1,2 in parent): always feasible — index 0 is always True.
        dg=1 (indices 3,4,5 in parent): feasible if inventory >= min_fuel.

        We read the parent's 6-bool mask and project to 2-bool.
        """
        parent_mask = super().get_action_mask()  # shape (6,)
        dg0_ok = bool(parent_mask[0])            # dg=0, order=0 always True
        dg1_ok = bool(np.any(parent_mask[3:6]))  # dg=1 feasible if any dg=1 action unmasked
        return np.array([dg0_ok, dg1_ok], dtype=bool)


# ──────────────────────────────────────────────────────────────────────────────
# Factory function — mirrors pattern in train_ablation_a5.py
# ──────────────────────────────────────────────────────────────────────────────

def make_a6_env(
    site_data,
    site_params: Dict[str, Any],
    lead_scenario: str = "normal",
    ss_safety_k: float = 1.0,
    **env_kwargs,
) -> A6Env:
    """
    Build an A6Env with SSPolicy calibrated from site_params.

    Args:
        site_data:     DataFrame from load_site().
        site_params:   Dict from load_site() — must contain d_bar_kwh, tank_cap_kwh.
        lead_scenario: 'normal' or 'delayed' — determines geometric p for calibration.
        ss_safety_k:   Safety stock multiplier for s,S reorder point (default 1.0).
        **env_kwargs:  Passed to TelecomEnv constructor (episode_len, eval_mode, etc.)
    """
    LEAD_P = {
        "fast":         1.0 / 12.0,
        "normal":       1.0 / 24.0,
        "delayed":      1.0 / 48.0,
        "monsoon":      1.0 / 72.0,
        "very_delayed": 1.0 / 120.0,
    }
    lead_p = LEAD_P.get(lead_scenario, 1.0 / 24.0)

    # ── [FIX 4] Compute d_bar and tank_cap from actual site data ─────────────
    # data_loader does NOT populate d_bar_kwh or tank_cap_kwh in site_params.
    # Using site_params.get("d_bar_kwh", 3.0) would give wrong default (3.0 vs ~8.3).
    # Must derive from site_data["load_kwh"] — same calculation as TelecomEnv.__init__
    from config_loader import env_cfg
    d_bar    = float(site_data["load_kwh"].mean())
    tank_cap = float(env_cfg["tank_hours"]) * d_bar * float(env_kwargs.get("tank_scale", 1.0))

    ss = SSPolicy.from_site_params(
        site_params=site_params,
        d_bar=d_bar,
        lead_p=lead_p,
        tank_cap_kwh=tank_cap,
        safety_k=ss_safety_k,
    )

    return A6Env(
        ss_policy=ss,
        site_data=site_data,
        site_params=site_params,
        lead_scenario=lead_scenario,
        **env_kwargs,
    )