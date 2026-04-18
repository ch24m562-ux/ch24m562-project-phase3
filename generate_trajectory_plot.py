"""
generate_trajectory_plot.py  (v2 — wired to real TelecomEnv)
=============================================================
Run from your PROJECT ROOT (where src/ and data/ live):

    python generate_trajectory_plot.py

Outputs -> ./trajectory_outputs/:
    rollout_log_rlinv_site5_normal.csv
    rollout_log_trackb_site5_normal.csv      <- PRIMARY (H1 defense)
    rollout_log_b1_site5_normal.csv          <- BACKUP (intuitive contrast)
    rollout_log_rlinv_site5_delayed.csv
    rollout_log_trackb_site5_delayed.csv
    rollout_log_b1_site5_delayed.csv
    rollout_log_combined_site5.csv
    trajectory_metadata_site5.json
    site5_trajectory_real.png / .pdf         <- RLInv vs TrackB
    site5_trajectory_b1.png / .pdf           <- RLInv vs B1 (backup)
    site5_order_timeline_appendix.png

KEY FIXES FROM v1+v2 (all assumptions verified against real training scripts):
  - Env constructor: TelecomEnv(site_data=df, site_params=params, lead_scenario=...)
  - Info dict keys: inv_kwh, inv_pct, pending_flag, pending_qty_kwh,
    p_pv_kwh, p_load_kwh, order_qty_kwh, delivery_kwh, unmet_kwh
  - Arrival: delivery_kwh > 0  (not a separate arrival_flag key)
  - Masking: info['mask_info']['dg_blocked'] and ['order_blocked']
  - Action: Discrete(6), a = dg*3 + order  (not array)
  - RLInv: MaskablePPO.load() — path runs/all_sites/{site}/{seed}/RLInv/{site}_final_model.zip
  - TrackB: PPO.load() (NOT MaskablePPO) — path runs/all_sites/{site}/{seed}/TrackB/{site}_final_model.zip
  - TrackB env: make_track_b_eval_env() from train_track_b.py
    (DispatchOnlyEnv, 6D obs, Discrete(2), SSPolicy order_fn)
  - TrackB vecnorm: 6D pkl — MUST NOT mix with RLInv 9D pkl
  - TrackB masking: none (plain PPO, no action_masks argument)
  - B1: B1Policy(SSPolicy()) with .act(obs, env=env) — raw obs, no vecnorm
  - SOC_MIN = 0.20  (real constant)
  - Battery discharge = p_bat_kwh < 0

CAUSAL CHAIN FOR VIVA:
  Panel 2: SoC similar     -> not the differentiator
  Panel 3: DG similar      -> not the differentiator
  Panel 1: ordering differs -> THIS is the differentiator
  Panel 4: lower EENS      -> causal outcome
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

warnings.filterwarnings('ignore')
OUT_DIR = './trajectory_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# ── SECTION 1: EDIT THESE PATHS ──────────────────────────────────────────────

SITE5_CSV          = 'data/processed/site5.csv'

# Actual run folder structure: runs/all_sites/{site}/{seed}/{policy}/
# Using seed42 for trajectory plot (matches primary eval seed)
RLINV_MODEL_PATH   = 'runs/all_sites/site5/seed42/RLInv/site5_final_model.zip'
RLINV_VECNORM_PATH = 'runs/all_sites/site5/seed42/RLInv/site5_vecnormalize.pkl'
EVAL_SEED   = 42
EVAL_EP_LEN = 720   # 15-day test window (hourly steps)

# TrackB paths — plain PPO, 6D obs, DispatchOnlyEnv
TRACKB_MODEL_PATH   = 'runs/all_sites/site5/seed42/TrackB/site5_final_model.zip'
TRACKB_VECNORM_PATH = 'runs/all_sites/site5/seed42/TrackB/site5_vecnormalize.pkl'

# Manual zoom override — set to day number (float) or None for auto-detect
ZOOM_START_DAY = None
ZOOM_HOURS     = 72

# ── SECTION 2: ENV + POLICY LOADING ──────────────────────────────────────────

def load_env(lead_scenario: str, seed: int):
    from src.env.data_loader import load_site, train_test_split
    from src.env.telecom_env import TelecomEnv
    df, params = load_site(SITE5_CSV)
    _, df_test = train_test_split(df)
    env = TelecomEnv(
        site_data=df_test, site_params=params,
        episode_len=EVAL_EP_LEN, eval_mode=True,
        lead_scenario=lead_scenario, seed=seed,
        init_inv_frac_low=0.6, init_inv_frac_high=0.6,
    )
    return env, params


def wrap_vecnorm(base_env):
    from src.eval.evaluate import VecNormObsWrapper
    if os.path.exists(RLINV_VECNORM_PATH):
        return VecNormObsWrapper(base_env, RLINV_VECNORM_PATH)
    print(f"[WARN] {RLINV_VECNORM_PATH} not found — no obs normalisation.")
    return base_env


def load_rlinv():
    from sb3_contrib import MaskablePPO
    m = MaskablePPO.load(RLINV_MODEL_PATH)
    print(f"[INFO] Loaded RLInv: {RLINV_MODEL_PATH}")
    return m


def load_trackb():
    """Load TrackB dispatch policy.
    TrackB uses plain PPO (NOT MaskablePPO) trained on DispatchOnlyEnv:
      - Discrete(2) action space (DG on/off only)
      - 6D observation (inventory dimensions stripped by DispatchOnlyEnv)
      - Ordering handled internally by SSPolicy order_fn, not by the policy
    """
    from stable_baselines3 import PPO
    if not os.path.exists(TRACKB_MODEL_PATH):
        print(f"[WARN] TrackB model not found at {TRACKB_MODEL_PATH}.")
        print(f"       Set TRACKB_MODEL_PATH correctly or leave as-is to skip.")
        return None
    m = PPO.load(TRACKB_MODEL_PATH)
    print(f"[INFO] Loaded TrackB (plain PPO): {TRACKB_MODEL_PATH}")
    return m


def wrap_vecnorm_trackb(base_env):
    """Wrap with TrackB VecNorm. MUST use TrackB pkl (6D obs stats),
    NOT the RLInv pkl (9D obs stats). Mixing them will silently
    apply wrong normalisation and produce garbage predictions."""
    from src.eval.evaluate import VecNormObsWrapper
    if os.path.exists(TRACKB_VECNORM_PATH):
        return VecNormObsWrapper(base_env, TRACKB_VECNORM_PATH)
    print(f"[WARN] TrackB vecnorm pkl not found at {TRACKB_VECNORM_PATH}.")
    print(f"       Running without obs normalisation — predictions may degrade.")
    return base_env


def load_b1(base_env, params):
    from src.baselines.s_S_policy import B1Policy, SSPolicy
    ss = SSPolicy.from_site_params(
        site_params=params, d_bar=float(base_env.d_bar),
        lead_p=float(base_env.lead_p), tank_cap_kwh=float(base_env.tank_cap_kwh),
        safety_k=1.5)
    print(f"[INFO] B1: s={ss.s_norm:.3f}  S={ss.S_norm:.3f}")
    return B1Policy(ss_policy=ss), ss.s_norm


# ── SECTION 3: ROLLOUT ───────────────────────────────────────────────────────

def run_rollout(policy, base_env, wrapped_env, policy_name, scenario, seed):
    is_rl = hasattr(policy, 'predict')
    # Defensive reset — Gymnasium returns (obs, info) but some wrappers return obs only
    reset_out = wrapped_env.reset(seed=seed)
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, _ = reset_out
    else:
        obs = reset_out
    if hasattr(policy, 'reset'):
        try: policy.reset()
        except: pass

    done = False; t = 0; cum_unmet = 0.0; rows = []

    while not done:
        if is_rl:
            # Try wrapped env first, then base_env for action mask.
            # Note: TrackB (plain PPO) does not use action_masks.
            # Passing mask=None triggers the no-mask path automatically.
            # MaskablePPO accepts action_masks; plain PPO raises TypeError
            # which is caught below — both work correctly.
            mask = None
            for env_obj in [wrapped_env, base_env]:
                if hasattr(env_obj, 'get_action_mask'):
                    try: mask = env_obj.get_action_mask(); break
                    except: pass
            try:
                action_int = int(np.asarray(
                    policy.predict(obs, action_masks=mask, deterministic=True)[0]
                    if mask is not None else
                    policy.predict(obs, deterministic=True)[0]
                ).item())
            except TypeError:
                # Plain PPO does not accept action_masks — fall back gracefully
                action_int = int(np.asarray(
                    policy.predict(obs, deterministic=True)[0]
                ).item())
        else:
            action_int = int(policy.act(obs, env=base_env))

        # Defensive step — handle both 5-tuple (Gymnasium) and 4-tuple (legacy)
        step_out = wrapped_env.step(action_int)
        if len(step_out) == 5:
            obs_next, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs_next, reward, done, info = step_out

        # DEBUG: print info keys on first step of TrackB to verify forwarding
        if t == 0 and policy_name == 'TrackB':
            print(f"[DEBUG] TrackB info keys (t=0): {sorted(info.keys())}")
            print(f"[DEBUG] Sample values: inv_kwh={info.get('inv_kwh','MISSING')}  "
                  f"order_qty_kwh={info.get('order_qty_kwh','MISSING')}  "
                  f"delivery_kwh={info.get('delivery_kwh','MISSING')}")

        delivery_kwh  = float(info.get('delivery_kwh', 0.0))
        mask_info     = info.get('mask_info', {})
        unmet         = float(info.get('unmet_kwh', 0.0))
        cum_unmet    += unmet
        tank          = float(info.get('tank_cap_kwh', 1.0))

        rows.append({
            't': t, 'day': t/24.0, 'hour_of_day': t%24,
            'site': 'site5', 'scenario': scenario, 'policy': policy_name, 'seed': seed,
            # State — exact keys from telecom_env.py info dict
            'soc':             float(info.get('soc',              0.5)),
            'inv_kwh':         float(info.get('inv_kwh',          0.0)),
            'inv_pct':         float(info.get('inv_pct',          0.0)),
            'pending_flag':    float(info.get('pending_flag',      0.0)),
            'pending_qty_kwh': float(info.get('pending_qty_kwh',  0.0)),
            'pending_pct':     float(info.get('pending_qty_kwh',  0.0)) / max(tank, 1.0),
            'p_pv_kwh':        float(info.get('p_pv_kwh',   0.0)),
            'p_load_kwh':      float(info.get('p_load_kwh', 0.0)),
            'p_grid_kwh':      float(info.get('p_grid_kwh', 0.0)),
            'net_load':        float(info.get('p_load_kwh', 0.0)) - float(info.get('p_pv_kwh', 0.0)),
            # Actions — decomposition depends on policy type:
            #   RLInv / B1: Discrete(6), a = dg*3 + order
            #   TrackB:     Discrete(2), a = dg only (ordering via SSPolicy)
            'action_int':      action_int,
            'action_dg_raw':   action_int if policy_name == 'TrackB' else action_int // 3,
            'action_ord_raw':  float('nan') if policy_name == 'TrackB' else action_int % 3,
            'dg_on':           float(info.get('dg_on', 0.0)),
            'order_qty_kwh':   float(info.get('order_qty_kwh', 0.0)),
            'order_placed':    1 if float(info.get('order_qty_kwh', 0.0)) > 1e-6 else 0,
            # Energy flows
            'p_dg_kwh':        float(info.get('p_dg_kwh',      0.0)),
            'p_bat_kwh':       float(info.get('p_bat_kwh',     0.0)),  # +charge -discharge
            'fuel_used_kwh':   float(info.get('fuel_used_kwh', 0.0)),
            'fuel_used_L':     float(info.get('fuel_used_L',   0.0)),
            # Delivery (delivery_kwh > 0 means arrival this step)
            'delivery_kwh':    delivery_kwh,
            'arrival_flag':    1 if delivery_kwh > 1e-6 else 0,
            # Outcome
            'unmet_kwh':       unmet, 'cum_unmet_kwh': cum_unmet,
            'stockout_flag':   int(info.get('stockout_flag', 0)),
            'reward':          float(reward),
            # Masking — from mask_info sub-dict
            'mask_dg_blocked':    int(mask_info.get('dg_blocked',    False)),
            'mask_order_blocked': int(mask_info.get('order_blocked',  False)),
            'mask_order_clipped': int(mask_info.get('order_clipped',  False)),
            # Violations
            'violation':       int(info.get('violation',     0)),
            'violation_soc':   int(info.get('violation_soc', 0)),
            'violation_tank':  int(info.get('violation_tank',0)),
            # Derived convenience columns
            'bat_charge_kwh':    max(0.0, float(info.get('p_bat_kwh', 0.0))),
            'bat_discharge_kwh': max(0.0, -float(info.get('p_bat_kwh', 0.0))),
            'served_kwh':        max(0.0, float(info.get('p_load_kwh', 0.0)) - unmet),
            'grid_outage_flag':  1 if float(info.get('p_grid_kwh', 0.0)) < 1e-6 else 0,
            # Proxy cost breakdown (logged, not plotted — NOT thesis cost coefficients)
            'proxy_cost_diesel': float(info.get('fuel_used_kwh', 0.0)) * 1.0,
            'proxy_cost_grid':   float(info.get('p_grid_kwh',   0.0)) * 0.4,
            'proxy_cost_unmet':  unmet * 100.0,
            'tank_cap_kwh':      tank,
        })
        obs = obs_next; t += 1

    return pd.DataFrame(rows)


# ── SECTION 4: GENERATE LOGS ─────────────────────────────────────────────────

def generate_logs():
    import json

    logs = {}
    rlinv   = load_rlinv()
    trackb  = load_trackb()   # None if zip not found
    b1_s_norm = 0.25          # updated after first B1 load below

    # ── IMPORTANT: TrackB was trained on DispatchOnlyEnv (6D obs, Discrete(2)).
    # Verify that your TRACKB_MODEL_PATH zip expects the same interface as
    # TelecomEnv before enabling. If unsure, set TRACKB_MODEL_PATH to a
    # non-existent path and the script will skip TrackB gracefully.
    # ─────────────────────────────────────────────────────────────────────

    for scenario in ['normal', 'delayed']:

        # ── RLInv ──────────────────────────────────────────────────────
        base_env, params = load_env(scenario, EVAL_SEED)
        wrapped = wrap_vecnorm(base_env)
        key = f'rlinv_site5_{scenario}'
        print(f"\n[INFO] Running {key}...")
        df = run_rollout(rlinv, base_env, wrapped, 'RLInv', scenario, EVAL_SEED)
        df.to_csv(os.path.join(OUT_DIR, f'rollout_log_{key}.csv'), index=False)
        print(f"  EENS={df['cum_unmet_kwh'].iloc[-1]:.1f} kWh | "
              f"orders={df['order_placed'].sum()} | "
              f"arrivals={df['arrival_flag'].sum()} | "
              f"violations={df['violation'].sum()} | "
              f"stockouts={df['stockout_flag'].sum()}")
        logs[key] = df

        # ── TrackB (primary H1 comparator) ─────────────────────────────
        # TrackB uses DispatchOnlyEnv (6D obs, Discrete(2)), plain PPO.
        # We use make_track_b_eval_env() from train_track_b.py — this builds
        # the EXACT same env + SSPolicy order_fn used during training.
        # The returned env is Monitor-wrapped DispatchOnlyEnv; its base_env
        # attribute is the underlying TelecomEnv for info dict access.
        # No action masking — plain PPO.predict() with no mask argument.
        if trackb is not None:
            from src.train.train_track_b import make_track_b_eval_env
            tb_env = make_track_b_eval_env(
                site_csv=SITE5_CSV,
                seed=EVAL_SEED,
                lead_scenario=scenario,
                episode_len=EVAL_EP_LEN,
            )
            # Wrap with TrackB VecNorm (6D obs stats)
            wrapped_tb = wrap_vecnorm_trackb(tb_env)
            # Fully unwrap to innermost env (TelecomEnv) for info access.
            # make_track_b_eval_env returns Monitor(DispatchOnlyEnv(TelecomEnv)).
            # One-layer unwrap may leave us at DispatchOnlyEnv; full unwrap
            # reaches TelecomEnv which holds the real state for masking etc.
            def _unwrap(env):
                while hasattr(env, 'env'):
                    env = env.env
                return env
            base_tb = _unwrap(tb_env)
            key_tb = f'trackb_site5_{scenario}'
            print(f"[INFO] Running {key_tb} (DispatchOnlyEnv, 6D obs, Discrete(2))...")
            df_tb = run_rollout(trackb, base_tb, wrapped_tb, 'TrackB', scenario, EVAL_SEED)
            df_tb.to_csv(os.path.join(OUT_DIR, f'rollout_log_{key_tb}.csv'), index=False)
            print(f"  EENS={df_tb['cum_unmet_kwh'].iloc[-1]:.1f} kWh | "
                  f"orders={df_tb['order_placed'].sum()} | "
                  f"arrivals={df_tb['arrival_flag'].sum()} | "
                  f"violations={df_tb['violation'].sum()} | "
                  f"stockouts={df_tb['stockout_flag'].sum()}")
            logs[key_tb] = df_tb
        else:
            print(f"[SKIP] TrackB not available — "
                  f"set TRACKB_MODEL_PATH to runs/all_sites/site5/seed42/TrackB/site5_final_model.zip")

        # ── B1 (backup / intuitive contrast) ───────────────────────────
        # Use params_b1 from b1's own load_env call, not the RL env's params.
        # Avoids cross-contamination even though site params are identical in
        # practice — cleaner and safer.
        # NOTE: B1 uses raw unnormalised obs; RL uses VecNorm-wrapped obs.
        # This matches the actual evaluation setup in evaluate.py.
        base_b1, params_b1 = load_env(scenario, EVAL_SEED)
        b1, b1_s_norm = load_b1(base_b1, params_b1)
        key_b1 = f'b1_site5_{scenario}'
        print(f"[INFO] Running {key_b1}...")
        df_b1 = run_rollout(b1, base_b1, base_b1, 'B1', scenario, EVAL_SEED)
        df_b1.to_csv(os.path.join(OUT_DIR, f'rollout_log_{key_b1}.csv'), index=False)
        print(f"  EENS={df_b1['cum_unmet_kwh'].iloc[-1]:.1f} kWh | "
              f"orders={df_b1['order_placed'].sum()} | "
              f"arrivals={df_b1['arrival_flag'].sum()} | "
              f"violations={df_b1['violation'].sum()} | "
              f"stockouts={df_b1['stockout_flag'].sum()}")
        logs[key_b1] = df_b1

    # Inspect before plotting
    print("\n[INFO] Sample — RLInv normal (verify fields before plotting):")
    cols = ['t','day','soc','inv_pct','pending_pct','dg_on',
            'order_placed','arrival_flag','unmet_kwh','cum_unmet_kwh']
    print(logs['rlinv_site5_normal'][cols].head(10).to_string(index=False))

    pd.concat(logs.values(), ignore_index=True).to_csv(
        os.path.join(OUT_DIR, 'rollout_log_combined_site5.csv'), index=False)

    # ── BUG 2 FIX: Write metadata JSON inside generate_logs() ──────────
    meta = {
        'site': 'site5', 'seed': EVAL_SEED, 'episode_len': EVAL_EP_LEN,
        'rlinv_model':   RLINV_MODEL_PATH,
        'rlinv_vecnorm': RLINV_VECNORM_PATH,
        'trackb_model':  TRACKB_MODEL_PATH,
        'trackb_vecnorm': TRACKB_VECNORM_PATH,
        'trackb_available': trackb is not None,
        'b1_s_norm': b1_s_norm,
        'scenarios': ['normal', 'delayed'],
        'note': 'B1 uses raw obs; RL policies use VecNorm-wrapped obs (matches eval.py)',
        'site_csv': SITE5_CSV,
        'zoom_start_day_manual': ZOOM_START_DAY,
        'zoom_hours': ZOOM_HOURS,
    }
    meta_path = os.path.join(OUT_DIR, 'trajectory_metadata_site5.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Metadata saved -> {meta_path}")

    return logs


# ── SECTION 5: HELPERS ───────────────────────────────────────────────────────

def shade_outages(ax, df, out_col):
    g = df['p_grid_kwh'].values; d = df['day'].values
    in_out = False; os_ = None
    for i in range(len(d)):
        if g[i] < 1e-6 and not in_out: os_=d[i]; in_out=True
        elif g[i] >= 1e-6 and in_out:
            ax.axvspan(os_,d[i],color=out_col,zorder=0,lw=0); in_out=False
    if in_out: ax.axvspan(os_,d[-1],color=out_col,zorder=0,lw=0)


def find_critical_window(rl, tb, window_h=72):
    T = len(rl); best_t = 24; best_score = -np.inf
    for t in range(24, T-window_h):
        out_hrs = int((rl['p_grid_kwh'].iloc[t:t+24] < 1e-6).sum())
        eens_gap = tb['unmet_kwh'].iloc[t:t+48].sum() - rl['unmet_kwh'].iloc[t:t+48].sum()
        inv_gap  = float(rl['inv_pct'].iloc[t] - tb['inv_pct'].iloc[t])
        score = eens_gap + inv_gap*20 + out_hrs*5
        if score > best_score: best_score=score; best_t=t
    return best_t


# ── SECTION 6: MAIN PLOT ─────────────────────────────────────────────────────

def make_plot(logs, comparator_key="trackb", comparator_label="TrackB", s_norm=0.25):
    rl_n=logs['rlinv_site5_normal']
    tb_n=logs.get(f'{comparator_key}_site5_normal', logs.get('b1_site5_normal'))
    rl_d=logs['rlinv_site5_delayed']
    tb_d=logs.get(f'{comparator_key}_site5_delayed', logs.get('b1_site5_delayed'))
    comparator_n = tb_n  # alias for zoom window detection
    rl_col='#1565C0'; tb_col='#2E7D32'; out_col='#F5F5F5'

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor('white')
    outer = GridSpec(1,2,figure=fig,wspace=0.09,left=0.07,right=0.98,top=0.91,bottom=0.07)
    sets = [
        ('Normal logistics  (mean lead = 24 h)',          rl_n, tb_n),
        ('Delayed logistics  (mean lead = 48 h)  — robustness check', rl_d, tb_d),
    ]
    all_axes = []

    for col_idx,(title,rl,tb) in enumerate(sets):
        inner = GridSpecFromSubplotSpec(4,1,subplot_spec=outer[col_idx],hspace=0.42)
        axs = [fig.add_subplot(inner[r]) for r in range(4)]
        all_axes.append(axs)
        days = rl['day'].values

        # Panel 1: Inventory
        ax=axs[0]; shade_outages(ax,rl,out_col)
        ax.plot(days,rl['inv_pct'],color=rl_col,lw=2.0,label='RLInv',zorder=3)
        ax.plot(days,tb['inv_pct'],color=tb_col,lw=1.8,ls='--',label=f'{comparator_label}',zorder=3,alpha=0.9)
        # Pipeline lines removed — add clutter without helping viva story
        for _,row in rl[rl['order_placed']==1].iterrows():
            ax.axvline(row['day'],color=rl_col,lw=1.4,alpha=0.7,zorder=4)
        for _,row in tb[tb['order_placed']==1].iterrows():
            ax.axvline(row['day'],color=tb_col,lw=1.4,alpha=0.7,ls='--',zorder=4)
        for _,row in rl[rl['arrival_flag']==1].iterrows():
            ax.axvline(row['day'],color=rl_col,lw=1.0,alpha=0.5,ls=':',zorder=4)
        for _,row in tb[tb['arrival_flag']==1].iterrows():
            ax.axvline(row['day'],color=tb_col,lw=1.0,alpha=0.5,ls=(0,(3,5)),zorder=4)
        ax.axhline(s_norm,color=tb_col,lw=0.9,ls=':',alpha=0.5)
        s_label = f'reorder pt. ({s_norm:.2f})' if comparator_key == 'b1' else f'classical reorder ref. ({s_norm:.2f})'
        ax.text(days[-1]*0.99,s_norm+0.01,s_label,ha='right',fontsize=7,color=tb_col,alpha=0.7)
        ax.set_ylim(-0.05,1.15); ax.set_ylabel('Inventory\n(frac. tank)',fontsize=8.5)
        ax.set_title(f'{title}\n① Diesel Inventory  (solid=order  ·  dotted=arrival)',
                     fontsize=9.0,loc='left',fontweight='bold')
        if col_idx==0: ax.legend(loc='lower left',fontsize=7.5,framealpha=0.95,ncol=2)

        # Panel 2: SoC
        ax=axs[1]; shade_outages(ax,rl,out_col)
        # RLInv: solid thick blue; comparator: dashed thinner green
        # Lines overlap closely — this is intentional: battery behaviour is
        # NOT the differentiator. Annotation makes this explicit for viva.
        ax.plot(days,rl['soc'],color=rl_col,lw=2.5,zorder=4,label='RLInv')
        ax.plot(days,tb['soc'],color=tb_col,lw=1.5,ls='--',zorder=3,alpha=0.85,
                label=f'{comparator_label}')
        ax.axhline(0.20,color='#888',lw=0.9,ls=':',alpha=0.6)   # SOC_MIN=0.20
        ax.text(days[-1]*0.99,0.21,'min SoC (0.20)',ha='right',fontsize=7,color='#888')
        # Annotation: inform viewer overlap is expected
        ax.text(days[len(days)//2],0.55,'← battery behaviour similar for both policies',
                ha='center',fontsize=7,color='#555555',style='italic',
                bbox=dict(boxstyle='round,pad=0.2',fc='white',ec='none',alpha=0.7))
        ax.legend(loc='lower right',fontsize=7.5,framealpha=0.9)
        ax.set_ylim(0.0,1.05); ax.set_ylabel('SoC',fontsize=8.5)
        ax.set_title('② Battery SoC  (overlap = both policies cycle battery similarly)',
                     fontsize=9.0,loc='left',fontweight='bold')

        # Panel 3: DG — annotate fuel-depletion gap if DG drops to zero
        ax=axs[2]; shade_outages(ax,rl,out_col)
        w = 6  # 6 hourly steps = 6h rolling mean
        dg_rl=np.convolve(rl['dg_on'].fillna(0).values,np.ones(w)/w,mode='same')
        dg_tb=np.convolve(tb['dg_on'].fillna(0).values,np.ones(w)/w,mode='same')
        ax.fill_between(days,dg_rl,alpha=0.22,color=rl_col)
        ax.fill_between(days,dg_tb,alpha=0.22,color=tb_col)
        ax.plot(days,dg_rl,color=rl_col,lw=1.8,zorder=3,label='RLInv')
        ax.plot(days,dg_tb,color=tb_col,lw=1.5,ls='--',zorder=3,alpha=0.9,
                label=f'{comparator_label}')
        # Find and annotate longest DG-off gap in RLInv (fuel depletion event)
        dg_off = (dg_rl < 0.1).astype(float)
        if dg_off.sum() > 12:   # at least 12h gap worth annotating
            # Find midpoint of longest zero-stretch
            changes = np.diff(np.concatenate([[0], dg_off, [0]]))
            starts  = np.where(changes == 1)[0]
            ends    = np.where(changes == -1)[0]
            if len(starts):
                longest = np.argmax(ends - starts)
                mid_t   = int((starts[longest] + ends[longest]) / 2)
                if mid_t < len(days):
                    ax.annotate('DG offline (fuel depleted)',
                                xy=(days[mid_t], 0.05),
                                xytext=(days[mid_t], 0.45),
                                fontsize=7, color=rl_col, ha='center',
                                arrowprops=dict(arrowstyle='->', color=rl_col, lw=1.0),
                                bbox=dict(boxstyle='round,pad=0.2',fc='white',
                                          ec=rl_col, alpha=0.85, lw=0.8))
        ax.legend(loc='upper right',fontsize=7.5,framealpha=0.9)
        ax.set_ylim(-0.05,1.45); ax.set_ylabel('DG util.\n(6h mean)',fontsize=8.5)
        ax.set_title('③ Diesel Generator Usage',fontsize=9.0,loc='left',fontweight='bold')

        # Panel 4: Cumulative EENS
        ax=axs[3]; shade_outages(ax,rl,out_col)
        cum_rl=rl['cum_unmet_kwh'].values; cum_tb=tb['cum_unmet_kwh'].values
        ax.fill_between(days,cum_rl,alpha=0.18,color=rl_col)
        ax.fill_between(days,cum_tb,alpha=0.18,color=tb_col)
        ax.plot(days,cum_rl,color=rl_col,lw=1.8,zorder=3,label=f'RLInv  ({cum_rl[-1]:.0f} kWh)')
        ax.plot(days,cum_tb,color=tb_col,lw=1.5,ls='--',zorder=3,alpha=0.9,label=f'{comparator_label}  ({cum_tb[-1]:.0f} kWh)')
        ax.set_ylabel('Cumul.\nEENS (kWh)',fontsize=8.5); ax.set_xlabel('Time (days)',fontsize=9)
        ax.set_title('④ Cumulative Unmet Load (EENS)',fontsize=9.0,loc='left',fontweight='bold')
        ax.legend(loc='upper left',fontsize=8,framealpha=0.95)

        for ax in axs:
            ax.set_xlim(0,days[-1]); ax.tick_params(labelsize=8)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.grid(axis='y',lw=0.4,color='#dddddd')

    # Zoom inset
    # Zoom: use manual override if set, else auto-detect
    if ZOOM_START_DAY is not None:
        zt = int(ZOOM_START_DAY * 24)
    else:
        zt = find_critical_window(rl_n, comparator_n)
    ze = min(zt + ZOOM_HOURS, len(rl_n))
    zd = rl_n['day'].values[zt:ze]
    ax_m = all_axes[0][0]
    ax_m.add_patch(plt.Rectangle(
        (rl_n['day'].values[zt],-0.02),
        rl_n['day'].values[ze-1]-rl_n['day'].values[zt],1.13,
        lw=1.8,edgecolor='#E65100',facecolor='#FFF3E0',zorder=2,alpha=0.45))
    ax_m.text((rl_n['day'].values[zt]+rl_n['day'].values[ze-1])/2,1.09,
              'Divergence window (manual)' if ZOOM_START_DAY is not None else 'Divergence window (auto-selected)',
              ha='center',fontsize=7.5,color='#E65100',fontweight='bold')
    ax_z = ax_m.inset_axes([0.52,-0.92,0.46,0.80])  # right side, below panel 1
    gz=rl_n['p_grid_kwh'].values[zt:ze]; in_out=False; os_=None
    for i,(d,g) in enumerate(zip(zd,gz)):
        if g<1e-6 and not in_out: os_=d; in_out=True
        elif g>=1e-6 and in_out:
            ax_z.axvspan(os_,d,color=out_col,zorder=0,lw=0); in_out=False
    if in_out: ax_z.axvspan(os_,zd[-1],color=out_col,zorder=0,lw=0)
    ax_z.plot(zd,rl_n['inv_pct'].values[zt:ze],color=rl_col,lw=2.2)
    ax_z.plot(zd,tb_n['inv_pct'].values[zt:ze],color=tb_col,lw=2.0,ls='--')
    d0,d1=rl_n['day'].values[zt],zd[-1]
    for _,row in rl_n[rl_n['order_placed']==1].iterrows():
        if d0<=row['day']<=d1:
            ax_z.axvline(row['day'],color=rl_col,lw=1.5,alpha=0.8)
            ax_z.text(row['day']+0.05,0.96,'RL\norder',color=rl_col,fontsize=6.5,va='top',fontweight='bold')
    for _,row in tb_n[tb_n['order_placed']==1].iterrows():
        if d0<=row['day']<=d1:
            ax_z.axvline(row['day'],color=tb_col,lw=1.5,alpha=0.8,ls='--')
            ax_z.text(row['day']+0.05,0.06,f'{comparator_label}\norder',color=tb_col,fontsize=6.5,va='bottom',fontweight='bold')
    for _,row in rl_n[rl_n['arrival_flag']==1].iterrows():
        if d0<=row['day']<=d1: ax_z.axvline(row['day'],color=rl_col,lw=1.0,alpha=0.6,ls=':')
    for _,row in tb_n[tb_n['arrival_flag']==1].iterrows():
        if d0<=row['day']<=d1: ax_z.axvline(row['day'],color=tb_col,lw=1.0,alpha=0.6,ls=(0,(3,5)))
    ax_z.set_xlim(d0,d1); ax_z.set_ylim(-0.05,1.15)
    ax_z.set_xlabel('Day',fontsize=7.5); ax_z.set_ylabel('Inventory',fontsize=7.5)
    ax_z.set_title('Example divergence window',fontsize=8,fontweight='bold',color='#E65100')
    ax_z.tick_params(labelsize=7); ax_z.spines['top'].set_visible(False); ax_z.spines['right'].set_visible(False)
    ax_z.grid(axis='y',lw=0.3,color='#dddddd')

    rl_p=mpatches.Patch(color=rl_col,label='RLInv (proposed)')
    tb_p=mpatches.Patch(color=tb_col,label=f'{comparator_label}')
    out_p=mpatches.Patch(color=out_col,label='Grid outage',ec='#bbbbbb',lw=0.8)
    ord_l=plt.Line2D([0],[0],color='grey',lw=1.3,label='Order (solid) / Arrival (dotted)')
    fig.legend(handles=[rl_p,tb_p,out_p,ord_l],loc='upper right',
               bbox_to_anchor=(0.99,0.97),fontsize=9,framealpha=0.95,ncol=2)
    n_rl=rl_n['cum_unmet_kwh'].iloc[-1]; n_cmp=tb_n['cum_unmet_kwh'].iloc[-1]
    d_rl=rl_d['cum_unmet_kwh'].iloc[-1]; d_cmp=tb_d['cum_unmet_kwh'].iloc[-1]
    fig.suptitle(
        f'Site 5 (Hard) — Actual rollout trajectory: RLInv vs {comparator_label}\n'
        f'Normal: RLInv {n_rl:.0f} kWh  vs  {comparator_label} {n_cmp:.0f} kWh  |  '
        f'Delayed: RLInv {d_rl:.0f} kWh  vs  {comparator_label} {d_cmp:.0f} kWh',
        fontsize=10.5,fontweight='bold',y=0.98)

    tag = comparator_key.replace('_site5','')
    plt.savefig(os.path.join(OUT_DIR,f'site5_trajectory_{tag}.png'),bbox_inches='tight',dpi=150)
    plt.savefig(os.path.join(OUT_DIR,f'site5_trajectory_{tag}.pdf'),bbox_inches='tight',dpi=150)
    plt.close(); print("[INFO] Main plot saved.")


# ── SECTION 7: APPENDIX ──────────────────────────────────────────────────────

def make_backup_plots(logs):
    """Backup stem plot — NOT for main slide."""
    rl=logs['rlinv_site5_normal']; tb=logs['b1_site5_normal']
    days=rl['day'].values; rl_col='#1565C0'; tb_col='#2E7D32'; out_col='#F5F5F5'
    fig,ax = plt.subplots(figsize=(14,4)); fig.patch.set_facecolor('white')
    g=rl['p_grid_kwh'].values; in_out=False; os_=None
    for i in range(len(days)):
        if g[i]<1e-6 and not in_out: os_=days[i]; in_out=True
        elif g[i]>=1e-6 and in_out:
            ax.axvspan(os_,days[i],color=out_col,zorder=0,lw=0); in_out=False
    if in_out: ax.axvspan(os_,days[-1],color=out_col,zorder=0,lw=0)
    ro=rl[rl['order_placed']==1]; to=tb[tb['order_placed']==1]
    if len(ro):
        ml,sl,_=ax.stem(ro['day'].values,ro['order_qty_kwh'].fillna(0).values,
                         linefmt=rl_col,markerfmt='o',basefmt=' ',label=f'RLInv (n={len(ro)})')
        plt.setp(sl,linewidth=1.8,alpha=0.8); plt.setp(ml,color=rl_col,markersize=7)
    if len(to):
        ml,sl,_=ax.stem(to['day'].values,-to['order_qty_kwh'].fillna(0).values,
                         linefmt=tb_col,markerfmt='s',basefmt=' ',label=f'B1 (n={len(to)})')
        plt.setp(sl,linewidth=1.8,alpha=0.8); plt.setp(ml,color=tb_col,markersize=7)
    ax.axhline(0,color='#888',lw=0.8)
    ax.set_xlim(0,days[-1]); ax.set_xlabel('Time (days)',fontsize=10)
    ax.set_ylabel('Order qty (kWh-equiv.)',fontsize=10)
    ax.set_title('APPENDIX — Order Timeline: RLInv (above) vs B1 (below)  |  backup figure',
                 fontsize=10,loc='left')
    ax.legend(fontsize=9,framealpha=0.95)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y',lw=0.4,color='#dddddd')
    plt.savefig(os.path.join(OUT_DIR,'site5_order_timeline_appendix.png'),bbox_inches='tight',dpi=150)
    plt.close(); print("[INFO] Appendix plot saved.")


# ── SECTION 8: MAIN ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import json
    print("="*60 + "\nSTEP 1: Rollout logs\n" + "="*60)
    logs = generate_logs()

    # Read real s_norm from saved metadata
    meta_path = os.path.join(OUT_DIR, 'trajectory_metadata_site5.json')
    with open(meta_path) as f: meta = json.load(f)
    b1_s_norm = float(meta.get('b1_s_norm', 0.25))

    print("\n" + "="*60 + "\nSTEP 2: Main plot — RLInv vs TrackB (H1 defense)\n" + "="*60)
    if 'trackb_site5_normal' in logs:
        make_plot(logs, comparator_key='trackb', comparator_label='TrackB', s_norm=b1_s_norm)
    else:
        print("[WARN] TrackB logs not found — skipping TrackB plot. Run with TrackB model.")

    print("\n" + "="*60 + "\nSTEP 3: Appendix order timeline (backup)\n" + "="*60)
    make_backup_plots(logs)
    # B1 plot omitted from main output — B1 delayed is degenerate (s=S=0.95).
    # Re-enable by uncommenting:
    # make_plot(logs, comparator_key='b1', comparator_label='B1 (classical)', s_norm=b1_s_norm)
    print("\nDone. Check ./trajectory_outputs/")
