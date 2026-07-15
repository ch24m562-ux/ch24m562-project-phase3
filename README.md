# Energy-Aware Telecom Tower Management via Inventory-Inspired Reinforcement Learning

A constrained reinforcement learning (MaskablePPO) agent, RLInv, that
jointly controls diesel-generator dispatch and diesel-order replenishment
for solar-battery-diesel telecom towers, evaluated against classical and
decomposed baselines under uncertain delivery-logistics scenarios.

See `CANONICAL_ARTIFACTS.md` for which result files are the frozen source
of truth — always compute statistics from those, not from re-derived or
legacy files.

## Setup

Tested with **Python 3.12.12**.

```powershell
conda activate rlenv
pip install -r requirements.txt
```

> `requirements.txt` should be a minimal, portable list of the packages
> actually imported by this codebase (pandas, numpy, torch,
> stable-baselines3, sb3-contrib, gymnasium, matplotlib, PyYAML, mlflow)
> — not a full `pip freeze`/`conda list` dump. A full environment export
> can include machine-local build paths (e.g. Windows conda-forge
> `file:///C:/...` references) that break installation on any other
> machine, and pulls in unrelated packages (Jupyter, etc.) that aren't
> part of the actual pipeline. Regenerate it from a clean virtual
> environment containing only this project's direct dependencies.

## Data

Dataset: ITU / Zindi hybrid-energy challenge — 10 sites, 60 days each,
hourly resolution. Not included in this repo; place the raw CSVs under
`data/raw/`, then run:

```powershell
python src/utils/preprocess_data.py
```

This produces the per-site CSVs (e.g. `data/processed/site1.csv` ...
`site10.csv`) consumed by `src/env/data_loader.py`. See
`PREPROCESSING_THESIS_NOTES.md` for known data-quality caveats.

Each site is split chronologically (not randomly) into a 45-day training
block and a 15-day held-out test block — 10,800 training / 3,600 test
hourly records in total across all 10 sites.

## Repository structure

```
src/
  env/        telecom_env.py, data_loader.py, obs_wrappers.py, a6_env.py
  train/      train_rl_inv.py, train_track_b.py,
              train_ablation_a5.py / a6.py / a7.py, train_sensitivity.py
  eval/       evaluate.py            — unified evaluation for every policy
  baselines/  s_S_policy.py, rule_based.py, mpc_policy.py
  utils/      preprocess_data.py
hparams.yaml  central config for environment and PPO parameters
results/
  phase3/     per-episode registry CSVs (the raw evaluation output)
  traces/     step-level .npz traces for behavioural analysis
  figures/    generated plots
runs/         model checkpoints and VecNormalize stats — gitignored, see below
```

## Quick verification

Before training or evaluating anything, confirm the environment
constructs correctly across every delivery-delay scenario, using the
same data-loading path training itself uses:

```powershell
python -c "
import sys
sys.path.insert(0, 'src')
from env.data_loader import load_site
from env.telecom_env import TelecomEnv

df, params = load_site('data/processed/site5.csv')
for lead in ['no_delay','fast','normal','delayed','monsoon','very_delayed','extreme']:
    env = TelecomEnv(df, params, lead_scenario=lead)
    obs, _ = env.reset()
    assert obs.shape == (11,), f'Bad shape for {lead}'
    print(f'  {lead}: obs={obs.shape} OK')
print('All lead scenarios OK')
"
```

## Training

```powershell
python src/train/train_rl_inv.py --site site2 --seed 42 --lead normal
```

The central environment and PPO parameters (learning rate, network size,
total timesteps, etc.) are maintained in `hparams.yaml`; command-line
flags such as `--lead`, `--tank_scale`, `--lead_dist`, and `--gamma`
provide controlled, experiment-specific overrides on top of that shared
baseline — used for ablations and sensitivity sweeps.

This saves two files per run:
- `runs/rlinv/site2_s42_final.zip` — the trained policy
- `runs/rlinv/site2_s42_vecnorm.pkl` — the matching VecNormalize
  statistics (running mean/variance used to normalise observations at
  training time). **Both files are needed together at evaluation time**
  — if the VecNormalize file is missing or its path is wrong,
  `evaluate.py` currently continues silently without normalisation
  rather than raising an error, which will not reproduce the trained
  policy's real behaviour. Always pass `--vecnorm_path` explicitly and
  confirm it points at a file that actually exists.

Training is logged to MLflow (hyperparameters and wall-clock time,
`wall_time_min`); run `mlflow ui` from the project root and open
`localhost:5000` to inspect past runs.

## Evaluation

```powershell
python src/eval/evaluate.py --site site2 --lead normal --seed 42 `
    --episodes 10 --episode_len 360 `
    --policy_type rl --algo maskable --env_type track_a `
    --model_path "runs/rlinv/site2_s42_final.zip" `
    --vecnorm_path "runs/rlinv/site2_s42_vecnorm.pkl" `
    --policy_label rlinv --out_csv "results/phase3/rlinv/rlinv_site2_normal_s42.csv"
```

`evaluate.py` works identically for every policy in the comparison
(RLInv, B0, B1, MPC, Oracle-MPC, TrackB, A5, A6, A7) via the
`--policy_type` / `--env_type` flags — see the docstring at the top of
the file for the full flag reference.

## Model checkpoints

Trained model checkpoints (`runs/`) are not committed to this repository
— they're large, machine-specific, and gitignored. To reproduce a
result, retrain using the command above with the same site/seed, or
contact the author for the specific checkpoint you need.

> **Note:** an earlier version of this README linked a single Google
> Drive checkpoint (site5, one scenario) as "the" trained model. That
> was a single illustrative example from an early stage of the project,
> not the full evaluation campaign behind the thesis results — removed
> here to avoid confusion. If you specifically need that one checkpoint,
> ask the author.

## Acknowledgments

- Stable-Baselines3 & sb3-contrib — PPO and MaskablePPO implementations
- Gymnasium — environment interface
- ITU / Zindi — hybrid-energy dataset
- MLflow — training run tracking
