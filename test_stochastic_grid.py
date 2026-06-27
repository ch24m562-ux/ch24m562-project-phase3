import sys, os
sys.path.insert(0, 'src')
os.chdir('C:/Users/dasja/projects/myproj')
from env.telecom_env import TelecomEnv
from env.obs_wrappers import NoInvObsWrapper
import pandas as pd, numpy as np

df = pd.read_csv('data/processed/site5.csv')
params = {'battery_capacity_kwh':10,'dg_power_kw':5,'grid_power_kw':8,
          'battery_charge_coeff':0.95,'battery_discharge_coeff':0.95,'init_soc':0.5}

# Fix 1: eval episode len
from config_loader import env_cfg
assert env_cfg['eval_episode_len'] == 360, 'eval_episode_len fix failed'
assert env_cfg['train_episode_len'] == 720, 'train_episode_len changed unexpectedly'
print('Fix 1 eval_episode_len: OK')

# Fix 2: A5 zeros dims 9,10
env = TelecomEnv(df, params)
wrapped = NoInvObsWrapper(env)
obs, _ = wrapped.reset()
assert obs[1] == 0.0 and obs[2] == 0.0 and obs[3] == 0.0, 'dims 1,2,3 not zeroed'
assert obs[9] == 0.0 and obs[10] == 0.0, 'dims 9,10 not zeroed'
assert obs[0] != 0.0, 'soc_n should not be zeroed'
print('Fix 2 A5 obs zeroing [1,2,3,9,10]: OK')

# Fix 3a: stochastic grid obs matches step
env_sg = TelecomEnv(df, params, use_stochastic_grid=True)
obs_sg, _ = env_sg.reset()
env_grid = obs_sg[6]
step_grid = float(env_sg._grid_state)
assert env_grid == step_grid, f'grid obs={env_grid} != _grid_state={step_grid}'
print('Fix 3a stochastic grid obs consistency: OK')

# Fix 3b: use_eta_obs=False gives delivery_remaining=0 always
env_ln = TelecomEnv(df, params, lead_distribution='lognormal', use_eta_obs=False)
obs_ln, _ = env_ln.reset()
for _ in range(10):
    obs_ln, _, _, _, _ = env_ln.step(1)
assert obs_ln[10] == 0.0, 'delivery_remaining_n should be 0 when use_eta_obs=False'
print('Fix 3b use_eta_obs=False: delivery_remaining=0: OK')

# Fix 3c: use_eta_obs=True gives non-zero delivery_remaining under lognormal
env_eta = TelecomEnv(df, params, lead_distribution='lognormal', use_eta_obs=True)
obs_eta, _ = env_eta.reset()
found_nonzero = False
for _ in range(20):
    obs_eta, _, _, _, _ = env_eta.step(1)
    if obs_eta[10] > 0:
        found_nonzero = True
        break
assert found_nonzero, 'delivery_remaining_n never non-zero with use_eta_obs=True'
print('Fix 3c use_eta_obs=True: delivery_remaining active: OK')

print()
print('All Stage 0 fixes verified.')