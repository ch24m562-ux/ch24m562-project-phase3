# telecom_env.py — Gymnasium-compatible TelecomEnv (Phase 3 — param-clean)
# Key design: inventory is in kWh-equivalent (NOT liters). DG fuel use subtracts kWh.
# Action space: Discrete(6) with 6-bool action mask for MaskablePPO.
#
# Phase 3 changes from original:
#   1. Class-level constants now read from hparams.yaml via config_loader
#   2. Added self._hours_since_order + hours_since_order_n as 10th obs dim
#   3. Multi-scenario training: lead_scenario="multi" samples from pool at each reset()
#   4. train_end_idx fixed: uses env_cfg["train_len_steps"]=1080 (45 days)
#   5. tank_scale: multiplicative tank capacity scaling (sensitivity analysis)
#      tank_scale=1.0 → 72h (base), 1.33 → 96h, 2.0 → 144h
#   6. use_time_encoding: if False, zeros sin_h/cos_h → tests whether agent uses
#      diurnal solar pattern (time-encoding ablation)
#   7. lead_distribution: "geometric" (default, memoryless) or "lognormal" (realistic).
#      Lognormal samples delivery time at order placement; has memory — so
#      hours_since_order becomes informative about remaining wait.
#   8. Observation space upgraded 10D → 11D: added delivery_remaining_n (dim 11)
#      - geometric: always 0.0 (memoryless — remaining wait is unknown)
#      - lognormal: clip(remaining_hours / HOURS_ORDER_MAX, 0, 1) — informative
#   9. Stochastic grid outage: optional 2-state Markov chain replaces deterministic
#      dataset read. Fitted per-site from ITU data. Prevents agent memorising exact
#      outage timing. Default off (use_stochastic_grid=False) for backward compat.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import random

import gymnasium as gym
from gymnasium import spaces

from config_loader import env_cfg, reward_cfg, lead_cfg, get_multi_scenario_pool


# ── [CHANGE 1] Reward dataclass defaults now read from hparams.yaml ──────────
@dataclass
class RewardCoeffs:
    alpha:   float = 1.0     # overridden by hparams at instantiation
    beta:    float = 0.4
    lam:     float = 100.0
    gamma_r: float = 200.0
    mu:      float = 20.0


class TelecomEnv(gym.Env):
    """
    TelecomEnv (Phase 3)
    - Obs: 11D [SoC_n, Inv_n, pending_flag, pending_qty_n, PV_n, Load_n,
                Grid, sin_h, cos_h, hours_since_order_n, delivery_remaining_n]
      * dim 11 (delivery_remaining_n): 0.0 for geometric; informative for lognormal
      * sin_h/cos_h → 0.0 when use_time_encoding=False (ablation)
    - Action: Discrete(6) encoding DG(2) × order(3): a = dg*3 + order
    - Battery: NOT an RL action; handled by fixed priority rule.
    - Lead time: geometric(p) delivery or lognormal(mu, sigma).
    - lead_scenario="multi": samples from pool each episode reset (Phase 3).
    - tank_scale: scales tank capacity (e.g. 2.0 = 144h buffer instead of 72h).
    - use_stochastic_grid: if True, fits 2-state Markov chain from site data and
      samples new outage pattern each episode (prevents timing memorisation).
    """

    metadata = {"render_modes": ["text"], "render_fps": 4}

    # ── [CHANGE 1] Constants read from hparams.yaml ──────────────────────────
    DT_HOURS           = env_cfg["dt_hours"]
    SOC_MIN            = env_cfg["soc_min"]
    SOC_MAX            = env_cfg["soc_max"]
    TANK_HOURS         = env_cfg["tank_hours"]
    Q_SMALL_PCT        = env_cfg["q_small_pct"]
    Q_LARGE_PCT        = env_cfg["q_large_pct"]
    DG_MIN_FUEL_HRS    = env_cfg["dg_min_fuel_hrs"]
    DG_MIN_FUEL_SAFETY = env_cfg["dg_min_fuel_safety"]
    OBS_PV_MAX         = env_cfg["obs_pv_max"]
    OBS_LOAD_MAX       = env_cfg["obs_load_max"]

    # ── [CHANGE 3] Added "multi" to valid lead scenarios ─────────────────────
    LEAD_TIME_SCENARIOS = {
        "fast":         lead_cfg["fast"]["p"],
        "normal":       lead_cfg["normal"]["p"],
        "delayed":      lead_cfg["delayed"]["p"],
        "very_delayed": lead_cfg["very_delayed"]["p"],
        "monsoon":      lead_cfg["monsoon"]["p"],       # 72h — heavy monsoon
        "no_delay":     lead_cfg["no_delay"]["p"],      # instant — upper bound
        "extreme":      lead_cfg["extreme"]["p"],       # 14 days — disaster stress
    }
    MULTI_SCENARIO = "multi"   # sentinel — resolved at reset()

    # ── [CHANGE 2] hours_since_order normalisation cap ───────────────────────
    HOURS_ORDER_MAX = 72.0

    def __init__(
        self,
        site_data: pd.DataFrame,
        site_params: Dict[str, Any],
        episode_len: int = 720,
        eval_mode: bool = False,
        lead_scenario: str = "normal",
        reward_coeffs: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
        init_inv_frac_low: float = 0.6,
        init_inv_frac_high: float = 0.6,
        # ── Phase 3 additions ─────────────────────────────────────────────────
        tank_scale: float = 1.0,            # [CHANGE 5] scales tank capacity
        use_time_encoding: bool = True,     # [CHANGE 6] False → zeros sin_h/cos_h
        lead_distribution: str = "geometric",  # [CHANGE 7] "geometric"|"lognormal"
        lead_sigma: float = 0.5,            # [CHANGE 7] lognormal shape parameter
        use_stochastic_grid: bool = False,  # [CHANGE 9] Markov chain grid outage
    ):
        super().__init__()

        self.data = site_data.reset_index(drop=True)
        self.params = dict(site_params)
        self.episode_len = int(episode_len)
        self.eval_mode = bool(eval_mode)

        self.init_inv_frac_low  = float(np.clip(init_inv_frac_low,  0.0, 1.0))
        self.init_inv_frac_high = float(np.clip(init_inv_frac_high, 0.0, 1.0))
        if self.init_inv_frac_low > self.init_inv_frac_high:
            self.init_inv_frac_low = self.init_inv_frac_high

        # ── [CHANGE 5] Tank scale ─────────────────────────────────────────────
        self.tank_scale = float(max(tank_scale, 0.1))   # guard against zero

        # ── [CHANGE 6] Time encoding ablation ────────────────────────────────
        self.use_time_encoding = bool(use_time_encoding)

        # ── [CHANGE 7] Lead time distribution ────────────────────────────────
        _valid_dist = ("geometric", "lognormal")
        if lead_distribution not in _valid_dist:
            raise ValueError(
                f"lead_distribution must be one of {_valid_dist}, got '{lead_distribution}'"
            )
        self.lead_distribution = lead_distribution
        self.lead_sigma        = float(lead_sigma)

        # ── [CHANGE 9] Stochastic grid outage — 2-state Markov chain ─────────
        self.use_stochastic_grid = bool(use_stochastic_grid)
        if self.use_stochastic_grid:
            self._p_outage, self._p_restore = self._fit_markov_chain()
        else:
            self._p_outage  = 0.0
            self._p_restore = 1.0
        self._grid_state = True   # current Markov state (overwritten in reset)

        # ── [CHANGE 3] Validate lead_scenario — "multi" now allowed ──────────
        valid = list(self.LEAD_TIME_SCENARIOS.keys()) + [self.MULTI_SCENARIO]
        if lead_scenario not in valid:
            raise ValueError(
                f"lead_scenario must be one of {valid}, got {lead_scenario}"
            )
        self._lead_scenario_arg = lead_scenario
        # Resolve actual p immediately (overwritten at each reset() for "multi")
        if lead_scenario == self.MULTI_SCENARIO:
            self.lead_p = self.LEAD_TIME_SCENARIOS["normal"]  # placeholder
        else:
            self.lead_p = float(self.LEAD_TIME_SCENARIOS[lead_scenario])

        # RNG
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # Reward coefficients — [CHANGE 1] defaults now from hparams
        rc = reward_coeffs or {}
        self.rc = RewardCoeffs(
            alpha=rc.get("alpha",   reward_cfg["alpha"]),
            beta=rc.get("beta",     reward_cfg["beta"]),
            lam=rc.get("lam", rc.get("lambda", reward_cfg["lam"])),
            gamma_r=rc.get("gamma_r", reward_cfg["gamma_r"]),
            mu=rc.get("mu",         reward_cfg["mu"]),
        )

        # ---- Required site params (unchanged from original) ----
        self.bat_cap_kwh  = float(self.params["battery_capacity_kwh"])
        self.dg_rated_kw  = float(self.params["dg_power_kw"])
        self.grid_max_kw  = float(self.params.get("grid_power_kw", 8.0))

        self.eta_c = float(self.params.get("battery_charge_coeff",    0.95))
        self.eta_d = float(self.params.get("battery_discharge_coeff", 0.95))

        self.fuel_rate_L_per_kWh = float(self.params.get("fuel_rate_L_per_kWh", 0.30))

        self.p_ch_max_kw  = self.bat_cap_kwh / 2.0
        self.p_dis_max_kw = self.bat_cap_kwh / 2.0

        # ---- Tank capacity and order sizes (unchanged) ----
        if "load_kwh" not in self.data.columns:
            raise KeyError("site_data must contain 'load_kwh' column.")
        self.d_bar        = float(self.data["load_kwh"].mean())
        self.tank_cap_kwh = self.TANK_HOURS * self.d_bar * self.tank_scale  # [CHANGE 5]
        self.q_small_kwh  = self.Q_SMALL_PCT * self.tank_cap_kwh
        self.q_large_kwh  = self.Q_LARGE_PCT * self.tank_cap_kwh

        self.min_fuel_kwh = (
            self.dg_rated_kw * self.DG_MIN_FUEL_HRS * self.DG_MIN_FUEL_SAFETY
        )

        # ── [CHANGE 4] train_end_idx fixed — use 1080 (45×24) not min() ──────
        # Phase 2 bug: min(1080, len-episode_len) gave 720 when len=1440, ep=720
        # Phase 3 fix: train_end = min(1080, len) so the 45-day boundary is correct
        train_len = env_cfg["train_len_steps"]   # 1080 from hparams
        self.train_end_idx  = min(train_len, len(self.data) - self.episode_len)
        self.test_start_idx = min(train_len, len(self.data) - self.episode_len)

        # ---- Gym spaces — [CHANGE 8] shape (10,) → (11,) for delivery_remaining_n
        # dim 11: 0.0 for geometric (memoryless), informative for lognormal
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(11,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(6)

        # ---- Internal state (unchanged from original + hours_since_order) ----
        self._soc              = 0.5
        self._inv_kwh          = self.tank_cap_kwh * 0.6
        self._pending_flag     = 0
        self._pending_qty_kwh  = 0.0
        self._hours_since_order  = 0.0   # [CHANGE 2] counts hours since last order placed
        self._delivery_in_hours  = -1.0  # [CHANGE 7] lognormal: sampled countdown to arrival

        self._t_idx   = 0
        self._step_num = 0
        self._done    = True

        self.ep_rewards:  list[float]          = []
        self.ep_info_log: list[Dict[str, Any]] = []

    # ── Helpers: action encoding (UNCHANGED) ─────────────────────────────────
    @staticmethod
    def encode_action(dg: int, order: int) -> int:
        return int(dg) * 3 + int(order)

    @staticmethod
    def decode_action(action: Any) -> Tuple[int, int]:
        if isinstance(action, (int, np.integer)):
            a = int(action)
            return a // 3, a % 3
        arr = np.asarray(action).flatten()
        if arr.size == 1:
            a = int(arr[0])
            return a // 3, a % 3
        if arr.size >= 2:
            return int(arr[0]), int(arr[1])
        raise ValueError(f"Invalid action format: {action}")

    # ── [CHANGE 9] Markov chain fitter ───────────────────────────────────────
    def _fit_markov_chain(self) -> Tuple[float, float]:
        """
        Fit 2-state Markov chain from site data grid_available column.

        States: grid_on (True) ↔ grid_off (False)
        Transitions estimated by counting:
            p_outage  = P(off | on)  = n(on→off) / n(on)
            p_restore = P(on  | off) = n(off→on) / n(off)

        Uses FULL dataset (60 days) for stable estimates — more transitions
        gives lower variance in p estimates vs 45-day training split only.

        Stationary distribution:
            π_on  = p_restore / (p_outage + p_restore)
            π_off = p_outage  / (p_outage + p_restore)
        Mean outage duration  = 1 / p_restore  hours
        Mean inter-outage gap = 1 / p_outage   hours
        """
        grid = self.data["grid_available"].astype(bool).values
        on_mask  = grid[:-1] == True
        off_mask = grid[:-1] == False

        n_on       = int(on_mask.sum())
        n_off      = int(off_mask.sum())
        n_on_off   = int((on_mask  & (grid[1:] == False)).sum())
        n_off_on   = int((off_mask & (grid[1:] == True)).sum())

        p_outage  = float(n_on_off  / max(n_on,  1))
        p_restore = float(n_off_on  / max(n_off, 1))

        # Guard: if site has no outages at all, keep deterministic behaviour
        if p_outage == 0.0:
            self.use_stochastic_grid = False   # fall back silently

        return p_outage, p_restore

    # ── Gym API ───────────────────────────────────────────────────────────────
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # ── [CHANGE 3] Resolve lead_p for this episode ────────────────────────
        if self._lead_scenario_arg == self.MULTI_SCENARIO:
            pool     = get_multi_scenario_pool()   # from hparams.yaml
            scenario = random.choice(pool)
            self.lead_p = float(self.LEAD_TIME_SCENARIOS[scenario])
        # else: self.lead_p already set in __init__ and stays fixed

        # Start index selection (UNCHANGED)
        if self.eval_mode:
            self._t_idx = int(self.test_start_idx)
        else:
            max_start = max(0, self.train_end_idx - self.episode_len)
            self._t_idx = int(self.rng.integers(0, max(1, max_start + 1)))

        # Initial SoC (UNCHANGED)
        init_soc = float(self.params.get("init_soc", 0.5))
        self._soc = float(np.clip(init_soc, self.SOC_MIN, self.SOC_MAX))

        # Initial inventory (UNCHANGED)
        inv_frac = self.rng.uniform(self.init_inv_frac_low, self.init_inv_frac_high)
        self._inv_kwh = float(np.clip(inv_frac * self.tank_cap_kwh, 0.0, self.tank_cap_kwh))
        self._init_inv_frac = float(inv_frac)

        self._pending_flag    = 0
        self._pending_qty_kwh = 0.0
        self._hours_since_order    = 0.0
        self._delivery_in_hours    = -1.0   # [CHANGE 7] reset lognormal countdown
        self._had_stockout_pending = False
        self._emergency_arrivals   = 0

        # ── [CHANGE 9] Initialise Markov grid state from stationary dist ──────
        if self.use_stochastic_grid:
            denom = self._p_outage + self._p_restore
            p_on  = self._p_restore / max(denom, 1e-9)
            self._grid_state = bool(self.rng.random() < p_on)

        self._step_num = 0
        self._done     = False

        self.ep_rewards  = []
        self.ep_info_log = []

        return self._get_obs(), {}

    def step(self, action: Any):
        if self._done:
            raise RuntimeError("Call reset() before step().")

        dg_raw, order_raw = self.decode_action(action)

        # Exogenous data
        row        = self.data.iloc[self._t_idx]
        p_pv_kwh   = float(row["solar_kwh"])
        p_load_kwh = float(row["load_kwh"])

        # ── [CHANGE 9] Grid availability — Markov or dataset ─────────────────
        if self.use_stochastic_grid:
            # Transition current grid state
            if self._grid_state:   # currently on → may go off
                if self.rng.random() < self._p_outage:
                    self._grid_state = False
            else:                  # currently off → may restore
                if self.rng.random() < self._p_restore:
                    self._grid_state = True
            grid_avail = self._grid_state
        else:
            grid_avail = bool(row["grid_available"])

        # Hard masking (UNCHANGED)
        dg_on, order_qty_kwh, mask_info = self._apply_action_mask(dg_raw, order_raw)

        # Energy balance (UNCHANGED)
        p_bat_kwh, p_grid_kwh, p_dg_kwh, unmet_kwh, soc_violation = self._energy_balance(
            p_pv_kwh, p_load_kwh, grid_avail, dg_on
        )

        # DG fuel consumption (UNCHANGED)
        fuel_used_kwh   = min(p_dg_kwh, self._inv_kwh)
        self._inv_kwh   = max(0.0, self._inv_kwh - fuel_used_kwh)

        # Stockout check BEFORE delivery (UNCHANGED — original order preserved)
        inv_pre_delivery_kwh = self._inv_kwh
        stockout_flag = bool((unmet_kwh > 0.0) and (inv_pre_delivery_kwh <= 0.0))
        if stockout_flag and self._pending_flag == 1:
            self._had_stockout_pending = True  # stockout while waiting for delivery

        # ── Delivery logic — geometric or lognormal [CHANGE 7] ───────────────
        delivery_kwh = 0.0
        if self._pending_flag == 1:
            self._hours_since_order += 1.0
            # Determine if delivery arrives this step
            if self.lead_distribution == "lognormal":
                self._delivery_in_hours -= 1.0
                arrived = self._delivery_in_hours <= 0.0
            else:  # geometric (default) — memoryless Bernoulli trial
                arrived = bool(self.rng.random() < self.lead_p)

            if arrived:
                delivery_kwh       = float(self._pending_qty_kwh)
                self._inv_kwh      = self._inv_kwh + delivery_kwh
                if self._had_stockout_pending:
                    self._emergency_arrivals += 1  # delivery arrived AFTER stockout
                self._pending_flag         = 0
                self._pending_qty_kwh      = 0.0
                self._hours_since_order    = 0.0
                self._delivery_in_hours    = -1.0  # [CHANGE 7] reset countdown
                self._had_stockout_pending = False
        else:
            # No pending order: counter grows to signal long gap since last order
            self._hours_since_order += 1.0

        # Tank overflow clip (UNCHANGED)
        tank_overflow = False
        if self._inv_kwh > self.tank_cap_kwh:
            tank_overflow = True
            self._inv_kwh = self.tank_cap_kwh

        # ── Place new order [CHANGE 7] ───────────────────────────────────────
        if order_qty_kwh > 0.0:
            self._pending_flag    = 1
            self._pending_qty_kwh = float(order_qty_kwh)
            self._hours_since_order = 0.0
            if self.lead_distribution == "lognormal":
                # Sample delivery time so E[T] = 1/lead_p hours (matches geometric mean)
                # Lognormal: E[X] = exp(mu + s²/2) = mean_hours → mu = ln(mean)-s²/2
                mean_hours = 1.0 / max(self.lead_p, 1e-9)
                mu_log     = np.log(mean_hours) - 0.5 * self.lead_sigma ** 2
                sampled    = float(self.rng.lognormal(mu_log, self.lead_sigma))
                self._delivery_in_hours = max(1.0, sampled)  # at least 1h
            else:
                self._delivery_in_hours = -1.0  # geometric: no countdown needed

        violation = bool(soc_violation or tank_overflow)

        # Reward (UNCHANGED)
        reward = self._compute_reward(p_dg_kwh, p_grid_kwh, unmet_kwh, violation, stockout_flag)

        # Advance time (UNCHANGED)
        self._t_idx    = min(self._t_idx + 1, len(self.data) - 1)
        self._step_num += 1

        terminated = False
        truncated  = self._step_num >= self.episode_len
        if terminated or truncated:
            self._done = True

        # Info dict (UNCHANGED — all original keys preserved)
        info = {
            "soc":              self._soc,
            "inv_kwh":          self._inv_kwh,
            "inv_pct":          self._inv_kwh / max(self.tank_cap_kwh, 1e-9),
            "pending_flag":     self._pending_flag,
            "pending_qty_kwh":  self._pending_qty_kwh,
            "dg_on":            bool(dg_on),
            "order_qty_kwh":    float(order_qty_kwh),
            "p_pv_kwh":         p_pv_kwh,
            "p_load_kwh":       p_load_kwh,
            "p_grid_kwh":       p_grid_kwh,
            "p_dg_kwh":         p_dg_kwh,
            "p_bat_kwh":        p_bat_kwh,
            "unmet_kwh":        unmet_kwh,
            "delivery_kwh":     delivery_kwh,
            "fuel_used_kwh":    fuel_used_kwh,
            "fuel_used_L":      fuel_used_kwh * self.fuel_rate_L_per_kWh,
            "violation":        int(violation),
            "violation_soc":    int(soc_violation),
            "violation_tank":   int(tank_overflow),
            "stockout_flag":    int(stockout_flag),
            "mask_info":        mask_info,
            "step":             self._step_num,
            "reward":           float(reward),
            "tank_cap_kwh":     self.tank_cap_kwh,
            "d_bar":            self.d_bar,
            "hours_since_order":      self._hours_since_order,
            "emergency_arrival":      int(self._had_stockout_pending and delivery_kwh > 0),
            "cumul_emergency_arrivals": self._emergency_arrivals,
        }

        self.ep_rewards.append(float(reward))
        self.ep_info_log.append(info)

        if terminated or truncated:
            info["episode_stats"] = self.get_episode_stats()

        return self._get_obs(), float(reward), terminated, truncated, info

    # ── Masking (UNCHANGED) ───────────────────────────────────────────────────
    def _apply_action_mask(self, dg_raw: int, order_raw: int):
        mask_info = {"dg_blocked": False, "order_blocked": False, "order_clipped": False}

        dg_on = bool(dg_raw == 1)
        if dg_on and self._inv_kwh < self.min_fuel_kwh:
            dg_on = False
            mask_info["dg_blocked"] = True

        if order_raw == 0:
            order_qty_kwh = 0.0
        elif self._pending_flag == 1:
            order_qty_kwh = 0.0
            mask_info["order_blocked"] = True
        else:
            requested     = self.q_small_kwh if order_raw == 1 else self.q_large_kwh
            space         = self.tank_cap_kwh - self._inv_kwh
            order_qty_kwh = float(min(requested, max(0.0, space)))
            if order_qty_kwh + 1e-9 < requested:
                mask_info["order_clipped"] = True

        return dg_on, float(order_qty_kwh), mask_info

    def get_action_mask(self) -> np.ndarray:
        """6-length boolean mask for Discrete(6). idx = dg*3 + order. (UNCHANGED)"""
        mask = np.ones(6, dtype=bool)

        if self._inv_kwh < self.min_fuel_kwh:
            mask[3:] = False

        if self._pending_flag == 1:
            for idx in (1, 2, 4, 5):
                mask[idx] = False
        else:
            space = self.tank_cap_kwh - self._inv_kwh
            if space < self.q_small_kwh:
                mask[1] = False
                mask[4] = False
            if space < self.q_large_kwh:
                mask[2] = False
                mask[5] = False

        mask[0] = True
        return mask

    # ── Energy balance (UNCHANGED — full original with grid_max_kw etc.) ─────
    def _energy_balance(
        self,
        p_pv_kwh: float,
        p_load_kwh: float,
        grid_avail: bool,
        dg_on: bool,
    ):
        residual = float(p_load_kwh)

        pv_to_load = min(p_pv_kwh, residual)
        residual  -= pv_to_load
        pv_excess  = max(0.0, p_pv_kwh - pv_to_load)

        grid_to_load = 0.0
        if grid_avail and residual > 1e-9:
            grid_to_load = min(self.grid_max_kw * self.DT_HOURS, residual)
            residual    -= grid_to_load

        bat_dis = 0.0
        if residual > 1e-9:
            e_avail = (self._soc - self.SOC_MIN) * self.bat_cap_kwh * self.eta_d
            bat_dis  = min(self.p_dis_max_kw * self.DT_HOURS, max(0.0, e_avail), residual)
            residual -= bat_dis

        dg_to_load = 0.0
        if dg_on and (self._inv_kwh >= self.min_fuel_kwh) and residual > 1e-9:
            dg_to_load  = min(self.dg_rated_kw * self.DT_HOURS, residual)
            residual   -= dg_to_load

        unmet = max(0.0, residual)

        bat_ch_from_pv = 0.0
        if pv_excess > 1e-9 and self._soc < self.SOC_MAX:
            e_space        = (self.SOC_MAX - self._soc) * self.bat_cap_kwh
            bat_ch_from_pv = min(
                self.p_ch_max_kw * self.DT_HOURS,
                pv_excess,
                e_space / max(self.eta_c, 1e-9),
            )

        bat_ch_from_grid = 0.0
        if grid_avail and self._soc < self.SOC_MAX:
            e_space          = (self.SOC_MAX - self._soc) * self.bat_cap_kwh
            e_space_remaining = max(0.0, e_space - bat_ch_from_pv * self.eta_c)
            if e_space_remaining > 1e-9:
                grid_cap_remaining = max(
                    0.0, (self.grid_max_kw * self.DT_HOURS) - grid_to_load
                )
                if grid_cap_remaining > 1e-9:
                    bat_ch_from_grid = min(
                        self.p_ch_max_kw * self.DT_HOURS,
                        grid_cap_remaining,
                        e_space_remaining / max(self.eta_c, 1e-9),
                    )

        max_ch           = self.p_ch_max_kw * self.DT_HOURS
        bat_ch_from_pv   = min(bat_ch_from_pv,  max_ch)
        bat_ch_from_grid = min(bat_ch_from_grid, max(0.0, max_ch - bat_ch_from_pv))
        bat_ch_total     = bat_ch_from_pv + bat_ch_from_grid

        p_bat  = bat_ch_total - bat_dis
        e_ch   = max(p_bat, 0.0) * self.eta_c
        e_dis  = max(-p_bat, 0.0) / max(self.eta_d, 1e-9)
        new_soc = self._soc + (e_ch - e_dis) / max(self.bat_cap_kwh, 1e-9)

        clipped_soc  = float(np.clip(new_soc, self.SOC_MIN, self.SOC_MAX))
        soc_violated = abs(clipped_soc - new_soc) > 1e-3
        self._soc    = clipped_soc

        p_grid_total = grid_to_load + bat_ch_from_grid

        return float(p_bat), float(p_grid_total), float(dg_to_load), float(unmet), bool(soc_violated)

    # ── Reward (UNCHANGED) ────────────────────────────────────────────────────
    def _compute_reward(
        self,
        p_dg_kwh: float,
        p_grid_kwh: float,
        unmet_kwh: float,
        violation: bool,
        stockout_flag: bool,
    ) -> float:
        d = max(self.d_bar, 1e-6)
        r  = -self.rc.alpha   * (p_dg_kwh  / d)
        r += -self.rc.beta    * (p_grid_kwh / d)
        r += -self.rc.lam     * (unmet_kwh  / d)
        r += -self.rc.gamma_r * float(stockout_flag)
        r += -self.rc.mu      * float(violation)
        return float(r)

    # ── Observation — [CHANGE 2] 10-D: added hours_since_order_n ─────────────
    def _get_obs(self) -> np.ndarray:
        row    = self.data.iloc[self._t_idx]
        hour   = int(row.get("hour", 0))
        p_pv   = float(row["solar_kwh"])
        p_load = float(row["load_kwh"])
        grid   = float(row["grid_available"])

        soc_n  = (self._soc - self.SOC_MIN) / (self.SOC_MAX - self.SOC_MIN)
        inv_n  = self._inv_kwh / max(self.tank_cap_kwh, 1e-9)
        p_flag = float(self._pending_flag)
        pqty_n = self._pending_qty_kwh / max(self.tank_cap_kwh, 1e-9)
        pv_n   = p_pv   / self.OBS_PV_MAX
        load_n = p_load / self.OBS_LOAD_MAX
        sin_h  = np.sin(2.0 * np.pi * hour / 24.0)
        cos_h  = np.cos(2.0 * np.pi * hour / 24.0)

        # [CHANGE 2] 10th dimension — normalised by HOURS_ORDER_MAX
        hours_n = float(np.clip(self._hours_since_order / self.HOURS_ORDER_MAX, 0.0, 1.0))

        # [CHANGE 8] 11th dimension — delivery_remaining_n
        # Lognormal: informative (remaining hours until delivery / HOURS_ORDER_MAX)
        # Geometric: 0.0 always (memoryless — remaining wait is unknown)
        if self.lead_distribution == "lognormal" and self._pending_flag == 1:
            delivery_rem_n = float(np.clip(
                max(0.0, self._delivery_in_hours) / self.HOURS_ORDER_MAX, 0.0, 1.0
            ))
        else:
            delivery_rem_n = 0.0

        # [CHANGE 6] Time encoding ablation — zero sin/cos to remove diurnal signal
        if not self.use_time_encoding:
            sin_h = 0.0
            cos_h = 0.0

        return np.array(
            [soc_n, inv_n, p_flag, pqty_n, pv_n, load_n, grid,
             sin_h, cos_h, hours_n, delivery_rem_n],
            dtype=np.float32,
        )

    # ── Episode stats (UNCHANGED) ─────────────────────────────────────────────
    def get_episode_stats(self) -> Dict[str, Any]:
        if not self.ep_rewards:
            return {}
        log = self.ep_info_log
        return {
            "total_reward":   float(np.sum(self.ep_rewards)),
            "mean_reward":    float(np.mean(self.ep_rewards)),
            "steps":          int(self._step_num),
            "EENS_kWh":       float(np.sum([i["unmet_kwh"]   for i in log])),
            "outage_hours":   int(np.sum([i["unmet_kwh"] > 0 for i in log])),
            "diesel_kWh":     float(np.sum([i["p_dg_kwh"]    for i in log])),
            "grid_kWh":       float(np.sum([i["p_grid_kwh"]  for i in log])),
            "stockout_events":int(np.sum([i.get("stockout_flag", 0) for i in log])),
            "orders_placed":  int(np.sum([i["order_qty_kwh"] > 0   for i in log])),
            "violations":     int(np.sum([i["violation"]            for i in log])),
            "mean_soc":       float(np.mean([i["soc"]     for i in log])),
            "mean_inv_pct":   float(np.mean([i["inv_pct"] for i in log])),
            "min_inv_pct":    float(np.min( [i["inv_pct"] for i in log])),
            "dg_on_fraction":      float(np.mean([i["dg_on"]   for i in log])),
            "emergency_arrivals":  int(self._emergency_arrivals),
            "init_inv_frac":     float(getattr(self, "_init_inv_frac", 0.6)),
            "tank_scale":        float(self.tank_scale),
            "lead_distribution": self.lead_distribution,
            "stochastic_grid":   self.use_stochastic_grid,
        }

    # ── Render (UNCHANGED) ────────────────────────────────────────────────────
    def render(self):
        row = self.data.iloc[self._t_idx]
        print(
            f"t={self._step_num:4d} | SoC={self._soc:.3f} | "
            f"Inv={self._inv_kwh/max(self.tank_cap_kwh,1e-9):.1%}"
            f"({self._inv_kwh:.1f}kWh) | "
            f"Pend={'Y' if self._pending_flag else 'N'}"
            f"({self._pending_qty_kwh:.0f}kWh) | "
            f"PV={row['solar_kwh']:.2f} Load={row['load_kwh']:.2f} "
            f"Grid={'Y' if row['grid_available'] else 'N'} | "
            f"HrsSinceOrder={self._hours_since_order:.0f}"
        )

    def close(self):
        pass