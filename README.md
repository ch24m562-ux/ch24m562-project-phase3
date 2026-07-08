# RL Project

## Environment
conda activate rlenv

## Install
pip install -r requirements.txt

## Train
python src/train.py

## Structure
src/        → code
notebooks/  → experiments
configs/    → hyperparameters
results/    → saved models/logs


## Trained Model Checkpoint
The main RL policy referenced in Chapter 5 results (site5, normal scenario, 400k timesteps) is not committed to this repo due to size (~X MB). Download it here:
[site5_s42_final.zip](https://drive.google.com/file/d/1UV_UsH-NkJ7rh7E2tAJBu5SRyWSU6-uE/view?usp=sharing)

Place it at `runs/full_site5_normal/site5_s42_final.zip` before running `evaluate.py` for reproduction.