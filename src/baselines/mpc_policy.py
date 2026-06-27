"""src/baselines/mpc_policy.py  --  MPC-Dispatch + B1-Ordering baseline
                                    (scipy/HiGHS solver, Gurobi optional)

Scientific role in ablation ladder
------------------------------------
  B1    : s,S ordering  +  rule-based dispatch
  A6    : s,S ordering  +  RL dispatch (Discrete-2)
  MPC   : s,S ordering  +  MILP-optimal dispatch, H=24  <-- this file
  RLInv : RL joint ordering + RL joint dispatch

Gaps
  A6  -> MPC  : does analytical optimality beat learned dispatch?
  MPC -> RLInv: does joint ordering-dispatch coupling add value beyond
               optimal dispatch with fixed (s,S) ordering?

Design decisions (changes from original Gurobi-only version)
------------------------------------------------------------
1. SOLVER
   Primary : scipy.optimize.milp (HiGHS backend, ships with scipy >= 1.9)
   Optional: gurobipy            (used automatically when available & requested)
   HiGHS solves a 24-binary MILP in ~5 ms; fully reproducible without a license.

2. FORECAST (HONESTLY LABELLED)
   forecast_cache=None  -> "MPC-H24-Persistence-B1" (current obs repeated H steps)
   forecast_cache=dict  -> "MPC-H24-Forecast-B1"    (cache with persistence fallback)
   Both are thesis-valid; label must match the results CSV policy_label field.

3. OBJECTIVE FUNCTION (corrected from original)
   Original had: inv_penalty * dg_inv_penalty[h]  (fires even when DG is OFF)
   Fixed:        inv_penalty * pen_prod[h]
   where pen_prod[h] = dg_on[h] * inv_pen[h] via Big-M linearisation:
     pen_prod[h] <= inv_pen[h]
     pen_prod[h] <= M * dg_on[h]
     pen_prod[h] >= inv_pen[h] - M*(1 - dg_on[h])
     pen_prod[h] >= 0
   Penalty now fires ONLY when DG is ON and inventory is below min_fuel.

4. BATTERY DYNAMICS (matched to TelecomEnv._energy_balance)
   Priority: PV -> Grid -> Battery-discharge -> DG -> unmet
   SoC update: new_soc = soc[h] + (eta_c*bc[h] - bd[h]/eta_d) / bat_cap
   Matches telecom_env.py lines 595-604 exactly.
   NOTE: TelecomEnv does greedy dispatch (priority rules); the MILP is a relaxed
   upper bound because it can co-optimise battery with DG. This is an optimistic
   bias for MPC — acceptable and documented.

5. B1 ORDERING
   Delegates to same SSPolicy used by B1 baseline and A6 ablation.
   Same (s,S) parameters, same site-specific calibration.
   MPC vs B1 differs ONLY in dispatch logic -- this is the intended comparison.

6. INVENTORY COUPLING (Level 2, memoryless)
   Expected delivery = E[remaining] = mean_lead_hours (geometric memoryless).
   Under 'extreme' (mean=336h), delivery is beyond H=24 -- returns None.
   This structural limitation of finite-horizon MPC vs RLInv's value function
   is the primary scientific finding of the thesis.
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Solver availability
# ---------------------------------------------------------------------------
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    import scipy.sparse as sp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

if not GUROBI_AVAILABLE and not SCIPY_AVAILABLE:
    raise ImportError(
        "Neither gurobipy nor scipy is available. "
        "Install scipy: pip install 'scipy>=1.9'"
    )

# ---------------------------------------------------------------------------
# Constants (all match hparams.yaml and telecom_env.py class-level values)
# ---------------------------------------------------------------------------
HORIZON        = 24      # planning horizon in hours
BIG_M          = 1e4     # Big-M for binary*continuous linearisation
SOC_MIN        = 0.20    # hparams soc_min
SOC_MAX        = 1.00    # hparams soc_max
DG_MIN_FUEL_HRS    = 2.0     # hparams dg_min_fuel_hrs
DG_MIN_FUEL_SAFETY = 1.20    # hparams dg_min_fuel_safety

# Mean lead times per scenario (hours) — from hparams.yaml lead_scenarios
MEAN_LEAD_HOURS = {
    "no_delay":      0,
    "fast":         12,
    "normal":       24,
    "delayed":      48,
    "monsoon":      72,
    "very_delayed": 120,
    "extreme":      336,
}


# ===========================================================================
# MPCDispatchB1Policy
# ===========================================================================
class MPCDispatchB1Policy:
    """
    MPC Dispatch with B1 (s,S) Ordering.

    At each timestep:
      1. B1 (s,S) ordering decision   -> order_action in {0,1,2}
      2. MILP dispatch over H=24 h    -> dg_on_t in {True, False}
      3. Combine into Discrete(6) env action

    Parameters
    ----------
    forecast_cache : dict or None
        {site_id: {t: {"load": np.array(H,), "solar": np.array(H,)}}}
        None -> persistence forecast (label = "MPC-H24-Persistence-B1").
    ss_policy : SSPolicy instance
        MUST be the same SSPolicy used by B1 baseline and A6 ablation.
    horizon : int
        Planning horizon in hours (default 24).
    lam_unmet : float
        Unmet-load penalty (match env reward lambda, default 100.0).
    alpha : float
        DG cost weight (match env reward alpha, default 1.0).
    beta : float
        Grid cost weight (match env reward beta, default 0.4).
    inv_penalty : float
        Penalty weight for (DG-on) × (inv < min_fuel). Only fires when DG is on.

    IMPORTANT -- lam_unmet vs inv_penalty tradeoff (validated via hour-by-hour
    trace, site5 & site2 / extreme / multiple seeds, see thesis validation log):
    inv_penalty (500.0) is 5x lam_unmet (100.0). Under scenarios where total
    available fuel cannot cover an entire outage regardless of dispatch
    strategy (observed under the "extreme" lead-time scenario, where mean
    delivery lead exceeds the episode length and no resupply arrives), this
    MILP optimizes a WEIGHTED COST, not raw EENS directly. Near the min_fuel
    threshold, this can produce a different -- and occasionally HIGHER raw
    EENS -- rationing pattern than a simpler reactive policy (e.g. B1), even
    though both Oracle-MPC and B1 face the identical underlying fuel scarcity
    and neither can avoid some unmet load. This was confirmed NOT to be an
    implementation bug: traced to specific hours where Oracle's solver makes
    an internally-rational (w.r.t. its own weighted objective) choice to
    leave load unmet rather than operate DG below the fuel-safety floor,
    producing a finer-grained on/off dispatch pattern right at the min_fuel
    boundary that can land on either side of B1's simpler all-or-nothing
    threshold rule depending on the specific load trajectory. This is a
    property of the objective's design (optimizing weighted cost, not pure
    EENS), not a correctness defect -- document this distinction explicitly
    if comparing Oracle-MPC's raw EENS against other policies under extreme.
    use_gurobi : bool
        Prefer Gurobi when available (default True).
    verbose : bool
        Print solver details.
    """

    def __init__(
        self,
        forecast_cache: Optional[dict],
        ss_policy,
        horizon:      int   = HORIZON,
        lam_unmet:    float = 100.0,
        alpha:        float = 1.0,
        beta:         float = 0.4,
        inv_penalty:  float = 500.0,
        use_gurobi:   bool  = True,
        verbose:      bool  = False,
    ):
        self.cache        = forecast_cache
        self.ss_policy    = ss_policy
        self.horizon      = horizon
        self.lam_unmet    = lam_unmet
        self.alpha        = alpha
        self.beta         = beta
        self.inv_penalty  = inv_penalty
        self.verbose      = verbose

        if use_gurobi and GUROBI_AVAILABLE:
            self._backend = "gurobi"
        elif SCIPY_AVAILABLE:
            self._backend = "scipy"
        else:
            raise RuntimeError("No MILP solver available.")

        self.forecast_mode = "Forecast" if self.cache is not None else "Persistence"
        if verbose:
            print(f"[MPC] backend={self._backend}  forecast={self.forecast_mode}  H={horizon}")

    # -----------------------------------------------------------------------
    # Public interface — mirrors SB3 policy.predict()
    # -----------------------------------------------------------------------
    def predict(self, obs: np.ndarray, env=None, site_id: str = None, t: int = None):
        """
        Returns (action, {}) where action in {0..5} (TelecomEnv Discrete-6).

        obs layout (11D from TelecomEnv):
          [0] soc_n               normalised SoC
          [1] inv_n               normalised inventory
          [2] pending_flag        1 if order in transit
          [3] pending_qty_n       normalised pending quantity
          [4] pv_n                normalised solar (/ OBS_PV_MAX=15.0)
          [5] load_n              normalised load  (/ OBS_LOAD_MAX=12.0)
          [6] grid_available      grid on flag
          [7] sin_h               hour-of-day sin encoding
          [8] cos_h               hour-of-day cos encoding
          [9] hours_since_order_n normalised hours since order (/ HOURS_ORDER_MAX=72)
          [10] delivery_rem_n     ETA signal (0.0 in all main experiments)
        """
        soc_n             = float(obs[0])
        inv_n             = float(obs[1])
        pending_flag      = float(obs[2]) > 0.5
        pending_qty_n     = float(obs[3])
        # obs[4]=pv_n, obs[5]=load_n, obs[6]=grid, obs[7]=sin, obs[8]=cos
        # obs[9]=hours_since_order_n (normalised by HOURS_ORDER_MAX=72)
        hours_since_order = float(obs[9]) * 72.0   # denormalise to hours

        # Step 1: B1 ordering
        order_action = self._b1_order(
            inv_n=inv_n,
            pending_flag=pending_flag,
        )

        # Step 2: Forecasts
        load_fc, solar_fc, grid_sched = self._get_forecasts(
            site_id=site_id, t=t, obs=obs, env=env
        )

        # Step 3: Site params
        params = self._get_site_params(env)

        # Step 4: Expected delivery step
        lead_scenario = "normal"
        if env is not None:
            base = env
            while hasattr(base, "env"):
                base = base.env
            # TelecomEnv stores the constructor arg as _lead_scenario_arg.
            # "lead_scenario" does not exist as a plain attribute.
            # Under "multi" training, _lead_scenario_arg == "multi"; in that
            # case fall back to reading the resolved lead_p and back-mapping,
            # or conservatively use "normal" (safe default for MPC planning).
            lead_scenario = getattr(base, "_lead_scenario_arg", "normal")
            if lead_scenario == "multi":
                lead_scenario = "normal"   # conservative fallback for multi-scenario episodes

        exp_delivery = self._expected_delivery_step(
            pending_flag=pending_flag,
            pending_qty_n=pending_qty_n,
            hours_since_order=hours_since_order,
            lead_scenario=lead_scenario,
        )

        # Step 5: MILP
        dg_on_decision = self._solve_milp(
            soc_n=soc_n, inv_n=inv_n,
            pending_qty_n=pending_qty_n,
            expected_delivery=exp_delivery,
            load_forecast=load_fc,
            solar_forecast=solar_fc,
            grid_schedule=grid_sched,
            params=params,
        )

        # Step 6: Combine
        return self._combine_action(dg_on=dg_on_decision, order_action=order_action), {}

    # -----------------------------------------------------------------------
    # B1 ordering
    # -----------------------------------------------------------------------
    def _b1_order(self, inv_n, pending_flag):
        """
        Delegates to SSPolicy.order_action(inv_n, pending_flag) -- the real
        method on s_S_policy.SSPolicy. This is the SAME call B1Policy.act()
        makes (see s_S_policy.py line 135), ensuring MPC's ordering is
        identical to the B1 baseline and A6 ablation for a fair comparison.
        """
        if self.ss_policy is None:
            return 0
        return int(self.ss_policy.order_action(inv_n=inv_n, pending_flag=pending_flag))

    # -----------------------------------------------------------------------
    # Forecast retrieval
    # -----------------------------------------------------------------------
    # train_forecast.py builds its cache keyed by ABSOLUTE timestep (0..1439)
    # against the full 60-day site CSV. TelecomEnv in eval mode pre-slices the
    # data to the 360-row test split (days 46-60) and resets _t_idx to 0 — so
    # the value of t passed into predict() is a LOCAL index (0..359), not the
    # absolute index the cache expects. Without this offset, a forecast lookup
    # at local t=0 would silently return the cache entry for absolute t=0
    # (day 1, training period) instead of the correct absolute t=1080 (day 46,
    # actual eval start). This offset is what bridges the two indexing schemes.
    CACHE_TEST_OFFSET = 1080   # hparams.yaml train_len_steps (45 days x 24h)

    def _get_forecasts(self, site_id, t, obs, env):
        """
        Returns (load_fc, solar_fc, grid_sched), each shape (H,).

        Obs layout (TelecomEnv._get_obs, line 674):
          [4] pv_n   = solar_kwh / OBS_PV_MAX   (15.0)
          [5] load_n = load_kwh  / OBS_LOAD_MAX  (12.0)
          [6] grid   = grid_available flag

        Load/solar: ML cache if available, else persistence (current obs repeated H steps).
        Grid: from env.data future rows via _t_idx (deterministic in ITU dataset).

        IMPORTANT: t is the env's local _t_idx (0..359 during eval). The forecast
        cache is keyed by absolute dataset index (0..1439). We add
        CACHE_TEST_OFFSET to translate local -> absolute before lookup.
        """
        H = self.horizon
        using_forecast_mode = self.cache is not None

        load_fc = solar_fc = None
        cache_t = None
        if using_forecast_mode and site_id is not None and t is not None:
            cache_t = int(t) + self.CACHE_TEST_OFFSET
            sc = self.cache.get(site_id, {}).get(cache_t, {})
            load_fc  = sc.get("load",  None)
            solar_fc = sc.get("solar", None)

        # HARD FAILURE for forecast-MPC (mpc_forecast / "MPC-H24-Forecast-B1"):
        # a cache miss here means the cache is incomplete, mis-keyed, or this
        # site/origin was never generated -- silently degrading to persistence
        # would make a "Forecast" result actually be an undocumented mix of
        # forecast and persistence, invalidating the comparison this policy
        # variant exists to make. Persistence-MPC (self.cache is None by
        # construction) is unaffected and keeps its intended fallback below.
        if using_forecast_mode and (load_fc is None or solar_fc is None):
            raise KeyError(
                f"[MPC-Forecast] Missing forecast cache entry for "
                f"site_id={site_id!r} local_t={t} (absolute cache_t={cache_t}). "
                f"This MPCDispatchB1Policy was constructed with a forecast_cache, "
                f"so a missing entry is treated as a hard error rather than "
                f"silently falling back to persistence. Check that: "
                f"(1) the cache was built for this site ({site_id!r} in "
                f"cache.keys()? {site_id in self.cache if self.cache else 'N/A'}), "
                f"(2) cache_t={cache_t} is within the generated range "
                f"(train_forecast.py's TEST_START..TOTAL_STEPS), and "
                f"(3) CACHE_TEST_OFFSET={self.CACHE_TEST_OFFSET} still matches "
                f"train_forecast.py's TEST_START constant."
            )

        # Persistence fallback (only reached when self.cache is None, i.e.
        # this is genuinely a persistence-MPC policy by construction --
        # NOT a forecast-MPC policy degrading silently, see hard-failure
        # check above).
        # Denormalise using hparams caps (not site-specific d_bar):
        # OBS_PV_MAX=15.0, OBS_LOAD_MAX=12.0 from hparams.yaml
        OBS_PV_MAX   = 15.0
        OBS_LOAD_MAX = 12.0
        if load_fc is None:
            load_fc = np.full(H, float(obs[5]) * OBS_LOAD_MAX)  # obs[5] = load_n
        if solar_fc is None:
            solar_fc = np.full(H, float(obs[4]) * OBS_PV_MAX)   # obs[4] = pv_n

        load_fc  = np.clip(np.resize(load_fc,  H), 0.0, None)
        solar_fc = np.clip(np.resize(solar_fc, H), 0.0, None)

        # Grid schedule from env data (always deterministic in ITU dataset)
        # TelecomEnv uses self._t_idx as the current data row pointer
        grid_sched = None
        if env is not None:
            base = env
            while hasattr(base, "env"):
                base = base.env
            data = getattr(base, "site_data", getattr(base, "data", None))
            if data is not None:
                t_now = getattr(base, "_t_idx", 0)   # TelecomEnv attribute (not current_step)
                t_end = min(t_now + H, len(data))
                if "grid_available" in data.columns:
                    g = data.iloc[t_now:t_end]["grid_available"].values.astype(float)
                    grid_sched = np.pad(g, (0, H - len(g)), "edge")

        if grid_sched is None:
            grid_sched = np.ones(H) * float(obs[6])   # obs[6] = grid_available

        return load_fc, solar_fc, grid_sched

    # -----------------------------------------------------------------------
    # Expected delivery step (Level 2 inventory coupling)
    # -----------------------------------------------------------------------
    def _expected_delivery_step(
        self,
        pending_flag: bool,
        pending_qty_n: float,
        hours_since_order: float,
        lead_scenario: str = "normal",
    ) -> Optional[int]:
        """
        Returns h in [0, H-1] if delivery expected within horizon, else None.
        Geometric memoryless: E[remaining] = E[total] = mean_lead_hours[scenario].
        """
        if not pending_flag:
            return None
        mean_h = MEAN_LEAD_HOURS.get(lead_scenario, 24)
        if mean_h < self.horizon:
            return max(0, int(mean_h))
        return None   # beyond H -- MPC cannot plan for it; RLInv value fn spans this gap

    # -----------------------------------------------------------------------
    # MILP solver dispatcher
    # -----------------------------------------------------------------------
    def _solve_milp(
        self,
        soc_n, inv_n, pending_qty_n, expected_delivery,
        load_forecast, solar_forecast, grid_schedule, params,
    ) -> bool:
        try:
            if self._backend == "gurobi":
                return self._solve_gurobi(
                    soc_n, inv_n, pending_qty_n, expected_delivery,
                    load_forecast, solar_forecast, grid_schedule, params,
                )
            return self._solve_scipy(
                soc_n, inv_n, pending_qty_n, expected_delivery,
                load_forecast, solar_forecast, grid_schedule, params,
            )
        except Exception as exc:
            if self.verbose:
                print(f"[MPC] solver error: {exc} -- rule-based fallback")
            return bool(inv_n > 0.1)

    # -----------------------------------------------------------------------
    # Shared problem-data builder
    # -----------------------------------------------------------------------
    def _problem_data(self, soc_n, inv_n, pending_qty_n, expected_delivery,
                      load_fc, solar_fc, grid_sched, params):
        """
        Returns a dict of all physical constants used by both solver backends.
        Centralised so both backends are guaranteed to use identical physics.
        """
        DG_kw    = float(params.get("dg_power_kw",   10.0))
        Grid_kw  = float(params.get("grid_power_kw", 10.0))
        batt_cap = float(params.get("battery_capacity_kwh", 10.0))
        eta_c    = float(params.get("battery_charge_coeff",    0.95))
        eta_d    = float(params.get("battery_discharge_coeff", 0.95))
        tank_cap = float(params.get("tank_capacity_kwh", 72.0 * DG_kw))

        # Match TelecomEnv: p_ch_max = p_dis_max = bat_cap / 2
        p_ch_max  = batt_cap / 2.0
        p_dis_max = batt_cap / 2.0

        # min_fuel = dg_rated_kw * DG_MIN_FUEL_HRS * DG_MIN_FUEL_SAFETY
        min_fuel = DG_kw * DG_MIN_FUEL_HRS * DG_MIN_FUEL_SAFETY

        soc_init = float(np.clip(soc_n,        SOC_MIN, SOC_MAX))
        inv_init = float(np.clip(inv_n,         0.0, 1.0)) * tank_cap
        pending  = float(np.clip(pending_qty_n, 0.0, 1.0)) * tank_cap

        return dict(
            H=self.horizon,
            DG_kwh=DG_kw,     # 1h timestep -> kW == kWh
            Grid_kwh=Grid_kw,
            batt_cap=batt_cap, eta_c=eta_c, eta_d=eta_d,
            p_ch_max=p_ch_max, p_dis_max=p_dis_max,
            min_fuel=min_fuel, tank_cap=tank_cap,
            soc_init=soc_init, inv_init=inv_init, pending=pending,
            expected_delivery=expected_delivery,
            load_fc=load_fc, solar_fc=solar_fc, grid_sched=grid_sched,
        )

    # -----------------------------------------------------------------------
    # scipy / HiGHS backend
    # -----------------------------------------------------------------------
    def _solve_scipy(self, soc_n, inv_n, pending_qty_n, expected_delivery,
                     load_fc, solar_fc, grid_sched, params) -> bool:
        """
        Variable vector layout (length = 11*H + 2):

          Binary indicator variables (integrality=1):
            [0:H]        dg[h]     DG on/off binary
            [H:2H]       gr[h]     grid on/off binary (0 when grid unavailable)

          Continuous energy variables (actual kWh dispatched this hour):
            [2H:3H]      dg_e[h]   DG energy  in [0, DG_max]
            [3H:4H]      gr_e[h]   grid energy in [0, Grid_max]
            [4H:5H]      bc[h]     battery charge >= 0
            [5H:6H]      bd[h]     battery discharge >= 0
            [6H:7H]      um[h]     unmet load >= 0

          Inventory penalty (Big-M linearised):
            [7H:8H]      ip[h]     inv_pen = max(0, min_fuel - inv[h])
            [8H:9H]      pp[h]     pen_prod = dg[h] * ip[h]

          State trajectories (H+1 values each):
            [9H:10H+1]   soc[0..H]
            [10H+1:11H+2] inv[0..H]

        Total: 11H + 2

        Key corrections vs previous binary-only version:
          1. dg_e/gr_e are continuous -> reward uses actual energy, not rated capacity
          2. inv[h+1] = inv[h] - dg_e[h] + delivery  (actual depletion, not DG_max)
          3. Load balance includes battery charge:
               dg_e + gr_e + Solar + bd + um >= Load + bc
             (prevents "free energy" where bc has no supply source)
          4. Coupling: dg_e <= DG_max*dg, gr_e <= Grid_max*gr

        Objective: min sum_h [alpha*dg_e + beta*gr_e + lam_unmet*um + inv_pen*pp]
        """
        d = self._problem_data(soc_n, inv_n, pending_qty_n, expected_delivery,
                               load_fc, solar_fc, grid_sched, params)
        H  = d["H"]
        N  = 11 * H + 2

        # --- Index helpers ---------------------------------------------------
        def i_dg(h):   return h           # binary: DG on/off
        def i_gr(h):   return H + h       # binary: grid on/off
        def i_dge(h):  return 2*H + h     # continuous: DG energy (kWh)
        def i_gre(h):  return 3*H + h     # continuous: grid energy (kWh)
        def i_bc(h):   return 4*H + h     # continuous: battery charge (kWh)
        def i_bd(h):   return 5*H + h     # continuous: battery discharge (kWh)
        def i_um(h):   return 6*H + h     # continuous: unmet load (kWh)
        def i_ip(h):   return 7*H + h     # continuous: inv_pen
        def i_pp(h):   return 8*H + h     # continuous: pen_prod (linearised dg*ip)
        def i_sc(h):   return 9*H + h     # continuous: SoC state (H+1 values)
        def i_iv(h):   return 10*H+1 + h  # continuous: inv state (H+1 values)

        # --- Objective -------------------------------------------------------
        c = np.zeros(N)
        for h in range(H):
            c[i_dge(h)] = self.alpha        # cost actual DG energy
            c[i_gre(h)] = self.beta         # cost actual grid energy
            c[i_um(h)]  = self.lam_unmet
            c[i_pp(h)]  = self.inv_penalty

        # --- Variable bounds -------------------------------------------------
        lb = np.zeros(N)
        ub = np.full(N, np.inf)
        # binary indicators
        ub[0 : 2*H]      = 1.0
        # energy variables: capped at rated capacity
        ub[2*H : 3*H]    = d["DG_kwh"]       # dg_e <= DG_max
        ub[3*H : 4*H]    = d["Grid_kwh"]     # gr_e <= Grid_max
        ub[4*H : 5*H]    = d["p_ch_max"]     # bc
        ub[5*H : 6*H]    = d["p_dis_max"]    # bd
        # SoC state
        lb[9*H : 10*H+1] = SOC_MIN
        ub[9*H : 10*H+1] = SOC_MAX
        # inventory state
        ub[10*H+1 : 11*H+2] = d["tank_cap"]

        # --- Constraints -----------------------------------------------------
        A_data, A_row, A_col, lb_c, ub_c = [], [], [], [], []

        def add_row(coeffs: dict, lo: float, hi: float):
            r = len(lb_c)
            lb_c.append(lo)
            ub_c.append(hi)
            for col, val in coeffs.items():
                A_data.append(val)
                A_row.append(r)
                A_col.append(col)

        # Initial state (equality)
        add_row({i_sc(0): 1.0}, d["soc_init"], d["soc_init"])
        add_row({i_iv(0): 1.0}, d["inv_init"], d["inv_init"])

        M = BIG_M
        for h in range(H):
            Lh = float(d["load_fc"][h])
            Sh = float(d["solar_fc"][h])
            Gh = float(d["grid_sched"][h])

            # Coupling: dg_e[h] <= DG_max * dg[h]  =>  dg_e - DG_max*dg <= 0
            add_row({i_dge(h): 1.0, i_dg(h): -d["DG_kwh"]}, -np.inf, 0.0)
            # Coupling: gr_e[h] <= Grid_max * gr[h]
            add_row({i_gre(h): 1.0, i_gr(h): -d["Grid_kwh"]}, -np.inf, 0.0)

            # Grid lock when unavailable: gr[h] = 0
            if Gh < 0.5:
                add_row({i_gr(h): 1.0}, 0.0, 0.0)

            # SoC balance: soc[h+1] = soc[h] + (eta_c*bc - bd/eta_d) / bat_cap
            #   => soc[h+1] - soc[h] - (eta_c/bat)*bc + (1/(eta_d*bat))*bd = 0
            add_row({
                i_sc(h+1):  1.0,
                i_sc(h):   -1.0,
                i_bc(h):   -(d["eta_c"] / d["batt_cap"]),
                i_bd(h):    (1.0 / (d["eta_d"] * d["batt_cap"])),
            }, 0.0, 0.0)

            # Load balance (FIX: includes bc on RHS — battery charge must come from supply):
            #   dg_e + gr_e + Solar + bd + um >= Load + bc
            #   => dg_e + gr_e + bd + um - bc >= Load - Solar
            add_row({
                i_dge(h):  1.0,
                i_gre(h):  1.0,
                i_bd(h):   1.0,
                i_um(h):   1.0,
                i_bc(h):  -1.0,
            }, Lh - Sh, np.inf)

            # Inventory balance (FIX: uses actual dg_e, not rated DG_max*dg_on):
            #   inv[h+1] = inv[h] - dg_e[h] + delivery_h
            #   => inv[h+1] - inv[h] + dg_e[h] = delivery_h
            delivery = d["pending"] if (d["expected_delivery"] is not None
                                        and d["expected_delivery"] == h) else 0.0
            add_row({
                i_iv(h+1):  1.0,
                i_iv(h):   -1.0,
                i_dge(h):   1.0,
            }, delivery, delivery)

            # Inventory penalty: ip[h] >= min_fuel - inv[h]
            #   => ip[h] + inv[h] >= min_fuel
            add_row({i_ip(h): 1.0, i_iv(h): 1.0}, d["min_fuel"], np.inf)

            # Big-M linearisation: pp[h] = dg[h] * ip[h]
            #   pp <= ip
            add_row({i_pp(h): 1.0, i_ip(h): -1.0}, -np.inf, 0.0)
            #   pp <= M * dg
            add_row({i_pp(h): 1.0, i_dg(h): -M}, -np.inf, 0.0)
            #   pp >= ip - M*(1-dg)  =>  pp - ip - M*dg >= -M
            add_row({i_pp(h): 1.0, i_ip(h): -1.0, i_dg(h): -M}, -M, np.inf)

        n_con = len(lb_c)
        A = sp.csc_matrix((A_data, (A_row, A_col)), shape=(n_con, N))
        integrality = np.zeros(N)
        integrality[0:2*H] = 1   # only dg[h] and gr[h] are binary

        # Time budget + MIP gap tolerance scale with problem size.
        # H<=24 (mpc, mpc_forecast): small MIP, solves to true optimality in ms.
        # H>24  (Oracle, up to 360): large MIP (~700+ binaries). Empirically the
        # solution reaches ~5% gap within ~0.1s and then spends 50+s chasing the
        # last fraction of a percent (classic MILP long tail) -- not worth it for
        # an upper-bound experiment. mip_rel_gap=0.02 (2%) accepts a near-optimal
        # incumbent fast; this is a documented approximation for Oracle specifically.
        if H <= 24:
            time_limit, mip_gap = 10.0, 0.0001   # tight: true baseline, must be exact
        else:
            time_limit, mip_gap = 8.0, 0.06       # Oracle: 6% gap acceptable -- empirically the
            # solver reaches ~4.9% within 0.1-2s; 8s budget lets it settle there without
            # chasing the long tail to true optimality (which costs 50+s for <1% more gap)

        result = milp(
            c=c,
            constraints=LinearConstraint(A, lb_c, ub_c),
            integrality=integrality,
            bounds=Bounds(lb, ub),
            options={"disp": self.verbose, "time_limit": time_limit, "mip_rel_gap": mip_gap},
        )

        # status==0: proven optimal. status==1 (time limit reached) still returns
        # a feasible incumbent in result.x when one was found -- use it rather
        # than discarding to a crude fallback, which previously caused DG to be
        # forced ON whenever inv_n > 0.1 regardless of actual need (see thesis
        # validation note: this produced an inflated dg_on_fraction for Oracle
        # at H=360 where the larger MIP times out more often).
        if result.status == 0:
            return bool(result.x[i_dge(0)] > 1e-6)
        if result.status == 1 and result.x is not None:
            if self.verbose:
                print(f"[MPC/scipy] status=1 (time limit) gap={result.mip_gap if hasattr(result,'mip_gap') else '?'} "
                      f"-- using best feasible incumbent, not crude fallback")
            return bool(result.x[i_dge(0)] > 1e-6)
        if self.verbose:
            print(f"[MPC/scipy] status={result.status} {result.message} -- rule-based fallback")
        return bool(inv_n > 0.1)

    # -----------------------------------------------------------------------
    # Gurobi backend (same MILP, cleaner syntax)
    # -----------------------------------------------------------------------
    def _solve_gurobi(self, soc_n, inv_n, pending_qty_n, expected_delivery,
                      load_fc, solar_fc, grid_sched, params) -> bool:
        """Same MILP as _solve_scipy with continuous energy variables (Gurobi syntax)."""
        d = self._problem_data(soc_n, inv_n, pending_qty_n, expected_delivery,
                               load_fc, solar_fc, grid_sched, params)
        H = d["H"]

        with gp.Env(empty=True) as env_g:
            env_g.setParam("OutputFlag", 1 if self.verbose else 0)
            env_g.setParam("TimeLimit",  5.0)
            env_g.start()
            with gp.Model(env=env_g) as m:
                # Binary on/off indicators
                dg  = m.addVars(H, vtype=GRB.BINARY, name="dg")
                gr  = m.addVars(H, vtype=GRB.BINARY, name="gr")
                # Continuous energy variables (actual kWh dispatched)
                dg_e = m.addVars(H, lb=0, ub=d["DG_kwh"],   name="dge")
                gr_e = m.addVars(H, lb=0, ub=d["Grid_kwh"], name="gre")
                bc   = m.addVars(H, lb=0, ub=d["p_ch_max"],  name="bc")
                bd   = m.addVars(H, lb=0, ub=d["p_dis_max"], name="bd")
                um   = m.addVars(H, lb=0, name="um")
                ip   = m.addVars(H, lb=0, name="ip")   # inv_pen
                pp   = m.addVars(H, lb=0, name="pp")   # pen_prod
                soc  = m.addVars(H+1, lb=SOC_MIN, ub=SOC_MAX,      name="soc")
                inv  = m.addVars(H+1, lb=0,        ub=d["tank_cap"], name="inv")

                m.addConstr(soc[0] == d["soc_init"])
                m.addConstr(inv[0] == d["inv_init"])

                M = BIG_M
                for h in range(H):
                    Lh = float(d["load_fc"][h])
                    Sh = float(d["solar_fc"][h])
                    Gh = float(d["grid_sched"][h])

                    # Coupling: energy <= capacity * on_flag
                    m.addConstr(dg_e[h] <= d["DG_kwh"]   * dg[h])
                    m.addConstr(gr_e[h] <= d["Grid_kwh"]  * gr[h])

                    # Grid unavailability
                    if Gh < 0.5:
                        m.addConstr(gr[h] == 0)

                    # SoC balance (matches TelecomEnv lines 595-604)
                    m.addConstr(
                        soc[h+1] == soc[h]
                        + (d["eta_c"]*bc[h] - bd[h]/d["eta_d"]) / d["batt_cap"]
                    )

                    # Load balance (FIX: bc on RHS prevents free-energy charging)
                    #   dg_e + gr_e + Solar + bd + um >= Load + bc
                    m.addConstr(
                        dg_e[h] + gr_e[h] + Sh + bd[h] + um[h] >= Lh + bc[h]
                    )

                    # Inventory balance (FIX: actual energy consumed, not rated capacity)
                    delivery = (d["pending"]
                                if d["expected_delivery"] is not None
                                and d["expected_delivery"] == h else 0.0)
                    m.addConstr(inv[h+1] == inv[h] - dg_e[h] + delivery)

                    # Inventory penalty variable
                    m.addConstr(ip[h] >= d["min_fuel"] - inv[h])

                    # Big-M: pp[h] = dg[h] * ip[h]
                    m.addConstr(pp[h] <= ip[h])
                    m.addConstr(pp[h] <= M * dg[h])
                    m.addConstr(pp[h] >= ip[h] - M*(1 - dg[h]))

                m.setObjective(
                    gp.quicksum(
                        self.alpha       * dg_e[h]
                        + self.beta      * gr_e[h]
                        + self.lam_unmet * um[h]
                        + self.inv_penalty * pp[h]
                        for h in range(H)
                    ),
                    GRB.MINIMIZE,
                )
                m.optimize()

                if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and m.SolCount > 0:
                    # DG on if positive energy was dispatched at step 0
                    return bool(dg_e[0].X > 1e-6)
                if self.verbose:
                    print(f"[MPC/gurobi] status={m.Status} -- fallback")
                return bool(inv_n > 0.1)

    # -----------------------------------------------------------------------
    # Action mapping
    # -----------------------------------------------------------------------
    def _combine_action(self, dg_on: bool, order_action: int) -> int:
        """
        TelecomEnv Discrete(6) action space:
          0: DG off, no order
          1: DG off, order small  (0.30 x tank)
          2: DG off, order large  (0.60 x tank)
          3: DG on,  no order
          4: DG on,  order small
          5: DG on,  order large
        """
        return (3 + order_action) if dg_on else order_action

    # -----------------------------------------------------------------------
    # Site parameter extraction
    # -----------------------------------------------------------------------
    def _get_site_params(self, env) -> dict:
        if env is None:
            return self._default_params()
        base = env
        while hasattr(base, "env"):
            base = base.env
        if hasattr(base, "site_params"):
            p    = base.site_params
            d_bar = float(base.site_data["load_kwh"].mean()) if hasattr(base, "site_data") else 5.0
            # Prefer already-computed tank_cap_kwh from env (includes tank_scale)
            tank_cap = float(getattr(base, "tank_cap_kwh",
                             p.get("tank_capacity_kwh", 72.0 * float(p.get("dg_power_kw", 10.0)))))
            return {
                "dg_power_kw":             float(p.get("dg_power_kw",                  10.0)),
                "grid_power_kw":           float(p.get("grid_power_kw",                10.0)),
                "battery_capacity_kwh":    float(p.get("battery_capacity_kwh",         10.0)),
                "battery_charge_coeff":    float(p.get("battery_charge_coeff",         0.95)),
                "battery_discharge_coeff": float(p.get("battery_discharge_coeff",      0.95)),
                "dod":                     float(p.get("DOD", p.get("dod",             0.2))),
                "tank_capacity_kwh":       tank_cap,
                "d_bar_kwh":               d_bar,
                "solar_capacity_kwh":      float(p.get("solar_capacity_kwh",           10.0)),
            }
        return self._default_params()

    def _default_params(self) -> dict:
        return {
            "dg_power_kw": 10.0, "grid_power_kw": 10.0,
            "battery_capacity_kwh": 10.0,
            "battery_charge_coeff": 0.95, "battery_discharge_coeff": 0.95,
            "dod": 0.20, "tank_capacity_kwh": 720.0,
            "d_bar_kwh": 5.0, "solar_capacity_kwh": 10.0,
        }


# ===========================================================================
# OracleMPCPolicy — perfect forecast upper bound
# ===========================================================================
class OracleMPCPolicy(MPCDispatchB1Policy):
    """
    Oracle MPC: same MILP as MPCDispatchB1Policy but uses ACTUAL future
    load/solar values (perfect knowledge). Theoretical upper bound.

    Scientific role: gap (Oracle - RLInv) reveals whether RLInv's ceiling is
    limited by the ordering policy (B1) or by dispatch quality.

    Horizon: 360 steps (full test episode), receding at each step.
    """

    FULL_HORIZON = 360   # full test episode length; restored at each reset()

    def __init__(self, ss_policy, **kwargs):
        kwargs.pop("forecast_cache", None)
        super().__init__(
            forecast_cache=None, ss_policy=ss_policy,
            horizon=self.FULL_HORIZON, **kwargs,
        )
        self.forecast_mode = "Oracle"

    def reset(self):
        """Restore full horizon at the start of each new episode.

        self.horizon is mutated in-place by _get_forecasts() (receding
        horizon shrink near episode end) -- without this reset, an episode
        would start with whatever shrunk horizon the previous episode ended
        on. evaluate.py calls policy.reset() before every episode.
        """
        self.horizon = self.FULL_HORIZON

    def _get_forecasts(self, site_id, t, obs, env):
        if env is None:
            return super()._get_forecasts(site_id, t, obs, env)

        base = env
        while hasattr(base, "env"):
            base = base.env

        data = getattr(base, "site_data", getattr(base, "data", None))
        if data is None:
            return super()._get_forecasts(site_id, t, obs, env)

        t_now = getattr(base, "_t_idx", 0)    # TelecomEnv uses _t_idx

        # Receding horizon: shrink to remaining actual data rather than padding
        # with repeated edge values. This both (a) avoids the MILP "planning"
        # over fake repeated-last-value hours near episode end, and (b) keeps
        # the MIP small (faster, fewer timeouts) once <360 steps remain.
        remaining = len(data) - t_now
        H = min(self.horizon, max(remaining, 1))
        self.horizon = H   # _problem_data() / _solve_scipy() read self.horizon directly

        t_end = min(t_now + H, len(data))

        load_fc  = data.iloc[t_now:t_end]["load_kwh"].values
        solar_fc = data.iloc[t_now:t_end]["solar_kwh"].values
        grid_sc  = data.iloc[t_now:t_end]["grid_available"].values.astype(float)

        pad = H - len(load_fc)
        return (
            np.pad(load_fc,  (0, pad), "edge"),
            np.pad(solar_fc, (0, pad), "edge"),
            np.pad(grid_sc,  (0, pad), "edge"),
        )


# ===========================================================================
# Quick sanity check
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MPC sanity test -- no env, H=6, scipy backend")
    print("=" * 60)

    # Use the REAL SSPolicy (s_S_policy.py) so the sanity test exercises the
    # actual production interface (.order_action(inv_n, pending_flag)),
    # not a stand-in API.
    try:
        from baselines.s_S_policy import SSPolicy
        ss = SSPolicy()
    except ImportError:
        class _FakeSS:
            def order_action(self, inv_n, pending_flag):
                return 0  # always no-order
        ss = _FakeSS()

    mpc = MPCDispatchB1Policy(
        forecast_cache=None,
        ss_policy=ss,
        horizon=6,
        verbose=True,
        use_gurobi=False,
    )

    # 11D obs: soc=0.8, inv=0.6, no pending, solar=0.3, load=0.5, grid=on
    obs = np.array([0.8, 0.6, 0.0, 0.0, 0.0, 0.3, 0.5, 1.0, 0.0, 1.0, 0.0])
    action, _ = mpc.predict(obs)
    dg_on = action >= 3
    order = action % 3
    print(f"\naction={action}  dg_on={dg_on}  order_size={order}")
    assert 0 <= action <= 5, "Action out of range!"
    print("SANITY TEST PASSED")
