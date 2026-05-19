"""    baselines/s_S_policy.py — Classical (s,S) Inventory Policy + Baseline B1

Used by:
  B1: (s,S) ordering + the SAME DG heuristic as B0 (for clean ablation)
  B2: (s,S) ordering injected into DispatchOnlyEnv; PPO controls DG only.

Notes:
  - Inventory is represented as normalized fraction inv_n ∈ [0,1].
  - Order actions are discrete: 0=none, 1=small, 2=large.

Important modelling note:
  With default (s=0.25, S=0.85) and q_large_pct=0.60, gaps after crossing s
  often exceed q_large, so large orders may dominate. Log order-size distribution
  in evaluation to interpret results correctly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict

import numpy as np


@dataclass
class SSPolicy:
    s_norm: float = 0.25
    S_norm: float = 0.85
    q_small_pct: float = 0.30
    q_large_pct: float = 0.60

    def order_action(self, inv_n: float, pending_flag: float) -> int:
        """Return 0/1/2 based on (s,S) and pending."""
        if pending_flag >= 0.5:
            return 0
        if inv_n >= self.s_norm:
            return 0

        gap = max(0.0, self.S_norm - inv_n)
        if gap >= self.q_large_pct:
            return 2
        if gap >= self.q_small_pct:
            return 1
        return 1  # if gap small but we crossed s, still place small

    @staticmethod
    def from_site_params(
        site_params: Dict[str, Any],
        d_bar: float,
        lead_p: float,
        tank_cap_kwh: Optional[float] = None,
        safety_k: float = 1.0,
    ) -> "SSPolicy":
        """Analytical calibration (textbook-style) to 'give Track B its best chance'.

        - Mean lead time (steps) for geometric: E[L] = 1/lead_p
        - Reorder point is proportional to expected demand during lead time with a safety factor.

        This is NOT empirical tuning; it is a closed-form inventory heuristic.
        """
        d_bar = float(max(d_bar, 1e-6))
        lead_p = float(max(lead_p, 1e-6))
        mean_L = 1.0 / lead_p

        # Tank capacity consistent with env (prefer explicit tank_cap_kwh from env)
        if tank_cap_kwh is None:
            tank_cap_kwh = float(site_params.get("tank_cap_kwh", 72.0 * d_bar))
        tank_cap_kwh = float(max(tank_cap_kwh, 1e-6))

        # Reorder point: demand during mean lead + safety cushion (in kWh)
        r_kwh = safety_k * d_bar * mean_L
        s_norm = float(np.clip(r_kwh / tank_cap_kwh, 0.05, 0.95))

        # Order-up-to level: keep high (but not full) to avoid frequent orders
        S_norm = float(np.clip(0.85, s_norm + 0.10, 0.95))

        return SSPolicy(s_norm=s_norm, S_norm=S_norm)

# -----------------------
# B1 baseline policy
# -----------------------

class B1Policy:
    """(s,S) ordering + SAME DG heuristic as B0 for controlled ablation."""

    # These match baselines/rule_based.py defaults
    OBS_PV_MAX = 15.0
    OBS_LOAD_MAX = 12.0

    def __init__(
        self,
        ss_policy: SSPolicy,
        dg_soc_thresh: float = 0.25,
        dg_pv_cover_thresh: float = 0.20,   # in pv_n/load_n space
        inv_guard_thresh: float = 0.15,
        soc_emergency: float = 0.23,         # matches B0 default
        pv_emergency_cover: float = 0.05,    # matches B0 default
        order_low_thresh: float = 0.20,
        order_critical_thresh: float = 0.10,
    ):
        self.ss = ss_policy
        self.dg_soc_thresh = float(dg_soc_thresh)
        self.dg_pv_cover_thresh = float(dg_pv_cover_thresh)
        self.inv_guard_thresh = float(inv_guard_thresh)
        self.soc_emergency = float(soc_emergency)
        self.pv_emergency_cover = float(pv_emergency_cover)
        self.order_low_thresh = float(order_low_thresh)
        self.order_critical_thresh = float(order_critical_thresh)

    def reset(self):
        return

    def act(self, obs: np.ndarray, env=None) -> int:
        # Slice first 9 dims — safe for 9D, 10D, or 11D obs (Phase 3 extended obs)
        o = np.asarray(obs, dtype=np.float32).flatten()
        soc_n, inv_n, pending, pqty_n, pv_n, load_n, grid, sin_h, cos_h = o[:9].tolist()

        grid_off = grid < 0.5
        bat_low = soc_n < self.dg_soc_thresh

        pv_cover = float(pv_n / max(load_n, 1e-6))
        pv_deficit = pv_cover < self.dg_pv_cover_thresh

        inv_scarce = inv_n < self.inv_guard_thresh

        severe_deficit = (
            grid_off
            and (soc_n <= self.soc_emergency)
            and (pv_cover <= self.pv_emergency_cover)
        )
        dg = 0
        if grid_off and (bat_low or pv_deficit):
            dg = 0 if (inv_scarce and not severe_deficit) else 1

        # Use s,S ordering
        order = self.ss.order_action(inv_n=inv_n, pending_flag=pending)

        a = dg * 3 + order

        # Respect env mask if available
        if env is not None and hasattr(env, "get_action_mask"):
            mask = env.get_action_mask()
            if not bool(mask[a]):
                a = 0
        return int(a)