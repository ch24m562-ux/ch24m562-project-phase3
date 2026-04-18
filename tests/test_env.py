# test_env.py — TelecomEnv v2 (telecom_env_v2 + charger cap fix)
# (Only minor robustness tweak in test_12_reward bound)

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

# Patch gymnasium if not installed (test environment may lack it)
try:
    import gymnasium as gym
except ImportError:
    import types, numpy as _np
    gym = types.ModuleType("gymnasium")
    class _Env:
        def reset(self, *, seed=None, options=None): pass
    class _spaces:
        class Discrete:
            def __init__(self, n): self.n = n
            def sample(self): return int(_np.random.randint(self.n))
        class Box:
            def __init__(self, low, high, shape, dtype=_np.float32):
                self.shape = shape; self.dtype = dtype
    gym.Env = _Env
    gym.spaces = _spaces()
    sys.modules["gymnasium"] = gym
    sys.modules["gymnasium.spaces"] = _spaces()

from src.env.telecom_env import TelecomEnv
from src.env.data_loader import load_site, train_test_split, compute_baseline_stats

DATA_DIR = "data/processed"


def make_env(site="site1", episode_len=720, seed=42, lead="normal"):
    df, params = load_site(f"{DATA_DIR}/{site}.csv")
    df_train, _ = train_test_split(df)
    return TelecomEnv(site_data=df_train, site_params=params,
                      episode_len=episode_len, lead_scenario=lead, seed=seed)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_01_instantiation():
    print("Test 01: Discrete(6) action space...", end=" ")
    env = make_env()
    assert hasattr(env.action_space, 'n') and env.action_space.n == 6
    assert env.observation_space.shape == (9,)
    print("PASS")


def test_02_reset():
    print("Test 02: Reset → valid 9-D obs...", end=" ")
    env = make_env()
    obs, _ = env.reset()
    assert obs.shape == (9,) and obs.dtype == np.float32
    assert not np.any(np.isnan(obs) | np.isinf(obs))
    print(f"PASS | obs={obs.round(3)}")


def test_03_decode_action():
    print("Test 03: decode_action handles int / array / scalar...", end=" ")
    for a, expect in [(0,(0,0)),(1,(0,1)),(2,(0,2)),(3,(1,0)),(4,(1,1)),(5,(1,2))]:
        assert TelecomEnv.decode_action(a) == expect
        assert TelecomEnv.decode_action(np.array([a])) == expect
        assert TelecomEnv.decode_action(np.int64(a)) == expect
    print("PASS")


def test_04_soc_lower_bound():
    print("Test 04: SoC never below SOC_MIN...", end=" ")
    env = make_env(); env.reset()
    for _ in range(200):
        _, _, term, trunc, _ = env.step(0)
        assert env._soc >= env.SOC_MIN - 1e-4, f"SoC={env._soc:.5f}"
        if term or trunc: break
    print("PASS")


def test_05_dg_min_fuel_kwh():
    print("Test 05: DG blocked; min_fuel is in kWh, not litres...", end=" ")
    env = make_env(); env.reset()
    expected = env.dg_rated_kw * env.DG_MIN_FUEL_HRS * env.DG_MIN_FUEL_SAFETY
    assert abs(env.min_fuel_kwh - expected) < 1e-6, f"{env.min_fuel_kwh} vs {expected}"
    env._inv_kwh = 0.0
    _, _, _, _, info = env.step(3)
    assert info["dg_on"] == False and info["mask_info"]["dg_blocked"] == True
    print(f"PASS | min_fuel={env.min_fuel_kwh:.2f} kWh")


def test_06_inventory_in_kwh():
    print("Test 06: Inventory depletes in kWh (not litres)...", end=" ")
    env = make_env(); env.reset()
    env._inv_kwh = env.tank_cap_kwh * 0.5
    env.data.loc[env._t_idx, "grid_available"] = False

    inv_before = env._inv_kwh
    _, _, _, _, info = env.step(3)
    fuel_kwh = info["fuel_used_kwh"]
    fuel_L   = info["fuel_used_L"]

    assert abs(fuel_kwh - info["p_dg_kwh"]) < 1e-6 or fuel_kwh == inv_before
    expected_inv = inv_before - fuel_kwh
    assert abs(env._inv_kwh - expected_inv) < 1e-6
    assert abs(fuel_L - fuel_kwh * env.fuel_rate_L_per_kWh) < 1e-6
    print(f"PASS | fuel_kwh={fuel_kwh:.3f}, fuel_L={fuel_L:.3f}")


def test_07_order_blocked_pending():
    print("Test 07: Order blocked when pending_flag=1...", end=" ")
    env = make_env(); env.reset()
    env._pending_flag = 1; env._pending_qty_kwh = env.q_large_kwh
    _, _, _, _, info = env.step(2)
    assert info["order_qty_kwh"] == 0.0
    assert info["mask_info"]["order_blocked"] == True
    print("PASS")


def test_08_geometric_delivery():
    print("Test 08: Geometric delivery clears pending_flag...", end=" ")
    env = make_env(lead="fast", seed=7); env.reset()
    env._inv_kwh = env.tank_cap_kwh * 0.2
    env._pending_flag = 0
    env.step(1)
    assert env._pending_flag == 1
    for _ in range(120):
        _, _, term, trunc, info = env.step(0)
        if info["delivery_kwh"] > 0:
            assert env._pending_flag == 0
            print(f"PASS | step={env._step_num}"); return
        if term or trunc: break
    raise AssertionError("No delivery in 120 steps (fast, p=1/12)")


def test_09_tank_cap_adaptive():
    print("Test 09: C = 72 × D̄ (site-adaptive)...", end=" ")
    e = make_env("site1"); h = make_env("site5")
    assert abs(e.tank_cap_kwh - 72 * e.d_bar) < 0.01
    assert h.tank_cap_kwh > e.tank_cap_kwh
    print(f"PASS | site1={e.tank_cap_kwh:.1f}kWh, site5={h.tank_cap_kwh:.1f}kWh")


def test_10_order_sizes():
    print("Test 10: q_small=30%C, q_large=60%C...", end=" ")
    env = make_env(); env.reset()
    assert abs(env.q_small_kwh - 0.30 * env.tank_cap_kwh) < 0.01
    assert abs(env.q_large_kwh - 0.60 * env.tank_cap_kwh) < 0.01
    print(f"PASS | small={env.q_small_kwh:.1f}, large={env.q_large_kwh:.1f}")


def test_11_episode_length():
    print("Test 11: Episode terminates at correct step...", end=" ")
    env = make_env(episode_len=48); env.reset()
    steps = 0
    while True:
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        steps += 1
        if term or trunc: break
    assert steps == 48, f"Expected 48, got {steps}"
    print(f"PASS | steps={steps}")


def test_12_reward():
    print("Test 12: Reward non-positive, finite, bounded...", end=" ")
    env = make_env(); env.reset()
    rs = []
    for _ in range(200):
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        assert np.isfinite(r) and r <= 1e-6, f"Bad reward: {r}"
        rs.append(r)
        if term or trunc: break
    assert min(rs) > -1e6, f"Reward too extreme: {min(rs)}"
    print(f"PASS | range=[{min(rs):.2f}, {max(rs):.2f}]")

# (rest of your tests unchanged)