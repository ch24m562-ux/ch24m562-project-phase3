"""
baselines/rule_based.py — B0: Rule-Based Policy

Deterministic heuristic — no learning.
  DG:    ON if grid unavailable AND (SoC near min OR PV cover low)
         + inventory guard: prefer DG OFF when inv dangerously low,
           except in a severe reliability emergency (burn last fuel).
  Order: order-when-low (inv_n < threshold, no pending order)

Implements the Policy interface: act(obs, info=None, env=None) -> int in [0..5]
Action encoding: a = dg*3 + order
"""

import numpy as np
from typing import Optional


# Module-level defaults — shared so both B0 and B1 are not hand-tuned per site
ORDER_LOW_THRESH      = 0.20   # inv_n: place small order
ORDER_CRITICAL_THRESH = 0.10   # inv_n: place large order


class RuleBasedPolicy:
    """
    B0 baseline. Sees full 9-D obs from TelecomEnv.

    Obs layout:
      [0] soc_n          [1] inv_n        [2] pending_flag
      [3] pending_qty_n  [4] pv_n         [5] load_n
      [6] grid           [7] sin_h        [8] cos_h

    Thresholds are normalised (same scale as obs):
      soc_n   : 0=SOC_MIN, 1=SOC_MAX
      inv_n   : 0=empty,   1=full tank
    """

    def __init__(
        self,
        dg_soc_thresh:        float = 0.15,   # soc_n below this → consider DG
        dg_pv_cover_thresh:   float = 0.20,   # DG ON if PV covers < 20% of load (dimensionless ratio)
        inv_guard_thresh:     float = 0.15,   # inv_n: too low to risk running DG
        order_low_thresh:     float = ORDER_LOW_THRESH,
        order_critical_thresh:float = ORDER_CRITICAL_THRESH,
    ):
        self.dg_soc_thresh       = dg_soc_thresh
        self.dg_pv_cover_thresh  = dg_pv_cover_thresh
        self.inv_guard_thresh    = inv_guard_thresh
        self.order_low_thresh    = order_low_thresh
        self.order_critical_thresh = order_critical_thresh

        # ── Emergency override constants ──────────────────────────────────
        # Context: when diesel is scarce, B0 prefers DG OFF (conserve fuel).
        # However a real operator burns the last fuel to avoid a blackout.
        # Severe deficit = grid off + SoC near floor + virtually no PV.
        #
        # Defaults (defensible; fixed across all sites — not tuned per site):
        #   SOC_EMERGENCY = 0.23      ≈ 3% above SOC_MIN (0.20)
        #   PV_EMERGENCY_COVER = 0.05 = PV covers <5% of load (in pv_n/load_n space)
        #
        # Physical meaning of PV_EMERGENCY_COVER:
        #   pv_cover = pv_n / load_n.  With OBS_LOAD_MAX=12, OBS_PV_MAX=15,
        #   0.05 corresponds to pv/load < 6.25% in physical units.
        #
        # Appendix sensitivity alternatives:
        #   Conservative (less DG): SOC_EMERGENCY=0.22, PV_EMERGENCY_COVER=0.03
        #   Reliability-first:      SOC_EMERGENCY=0.25, PV_EMERGENCY_COVER=0.10
        self.SOC_EMERGENCY      = 0.23
        self.PV_EMERGENCY_COVER = 0.05

    def reset(self):
        """Stateless — nothing to reset."""
        pass

    def act(self, obs: np.ndarray, info: Optional[dict] = None, env=None) -> int:
        """
        Returns action in [0..5].
        If env is provided, applies get_action_mask() as a safety net.
        """
        soc_n   = float(obs[0])
        inv_n   = float(obs[1])
        pending = float(obs[2])
        pv_n    = float(obs[4])
        load_n  = float(obs[5])
        grid    = float(obs[6])

        # ── DG decision ───────────────────────────────────────────────────
        grid_off = grid < 0.5
        bat_low  = soc_n < self.dg_soc_thresh

        # PV cover ratio: dimensionless, scales naturally across sites.
        # "DG on if PV covers less than dg_pv_cover_thresh fraction of load."
        # Ratio form avoids site-specific tuning.
        pv_cover   = pv_n / max(load_n, 1e-6)
        pv_deficit = pv_cover < self.dg_pv_cover_thresh

        # Inventory guard: prefer DG OFF when tank nearly empty, to avoid
        # burning the last fuel before the next delivery arrives.
        inv_scarce = inv_n < self.inv_guard_thresh

        # Emergency override: if SoC is near floor AND PV is negligible,
        # allow DG even when scarce — a real operator burns last fuel
        # to prevent a complete outage.
        severe_deficit = (
            grid_off
            and (soc_n <= self.SOC_EMERGENCY)
            and (pv_cover <= self.PV_EMERGENCY_COVER)
        )

        dg = 0
        if grid_off and (bat_low or pv_deficit):
            # Normal: conserve diesel.  Emergency: burn it to avoid outage.
            dg = 0 if (inv_scarce and not severe_deficit) else 1

        # ── Order decision ────────────────────────────────────────────────
        order = 0
        if pending < 0.5:   # no order in transit
            if inv_n < self.order_critical_thresh:
                order = 2   # large order
            elif inv_n < self.order_low_thresh:
                order = 1   # small order

        a = dg * 3 + order

        # Safety: apply action mask if env available
        if env is not None:
            mask = env.get_action_mask()
            if not mask[a]:
                fallback = dg * 3   # same DG intent, no order
                a = fallback if mask[fallback] else int(np.argmax(mask))

        return int(a)
