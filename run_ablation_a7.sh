#!/usr/bin/env bash
# run_ablation_a7.sh — Vanilla PPO (no masking) across all 10 sites
#
# Purpose: H2 — does action masking improve stability?
# Tests: variance of EENS with vs without masking across all sites.
# Compare results against: results/all_sites/*/RLInv/eval_normal.csv
#
# After: python aggregate_results.py --scenario normal --include_a7

set -e

SITES="${SITES:-1 2 3 4 5 6 7 8 9 10}"
SEEDS="${SEEDS:-42 123 777}"
TRAIN_STEPS="${TRAIN_STEPS:-400000}"
EVAL_EPISODES="${EVAL_EPISODES:-30}"
TRAIN_SCENARIO="normal"
INIT_LOW=0.3
INIT_HIGH=0.9
DRY_RUN=false

[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true && echo "[DRY RUN]"

echo "A7 Ablation — Vanilla PPO (no masking)"
echo "Sites: $SITES | Seeds: $SEEDS | Steps: $TRAIN_STEPS"
echo ""

run() { echo ">>> $*"; $DRY_RUN || "$@"; }

for site_num in $SITES; do
  SITE="site${site_num}"
  SITE_CSV="data/processed/${SITE}.csv"
  [[ ! -f "$SITE_CSV" ]] && ! $DRY_RUN && echo "[SKIP] $SITE_CSV not found" && continue

  for seed in $SEEDS; do
    A7_DIR="runs/all_sites/${SITE}/seed${seed}/A7"
    A7_RES="results/all_sites/${SITE}/seed${seed}/A7"

    echo ""
    echo "========================================================"
    echo " A7 | ${SITE} | seed=${seed}"
    echo "========================================================"

    # ── Train A7 (vanilla PPO, no masking) ───────────────────────────────
    if [[ -f "${A7_DIR}/${SITE}_final_model.zip" ]] && ! $DRY_RUN; then
      echo "[SKIP TRAIN] A7 model exists"
    else
      run python -m src.train.train_ablation_a7 \
        --site "$SITE" --lead "$TRAIN_SCENARIO" \
        --timesteps "$TRAIN_STEPS" --seed "$seed" --logdir "$A7_DIR" \
        --init_diesel_low "$INIT_LOW" --init_diesel_high "$INIT_HIGH"
    fi

    # ── Evaluate A7 under normal only (masking comparison is normal regime) 
    A7_CSV="${A7_RES}/eval_normal.csv"
    if [[ -f "$A7_CSV" ]] && ! $DRY_RUN; then
      echo "[SKIP EVAL] $A7_CSV"
    else
      run mkdir -p "$A7_RES"
      run python -m src.eval.evaluate \
              --site "$SITE" --lead normal \
              --policy_type rl --algo ppo --env_type track_a \
              --model_path    "${A7_DIR}/${SITE}_final_model" \
              --vecnorm_path  "${A7_DIR}/${SITE}_vecnormalize.pkl" \
              --episodes "$EVAL_EPISODES" --seed "$seed" \
              --init_diesel_low "$INIT_LOW" --init_diesel_high "$INIT_HIGH" \
              --policy_label A7 --train_scenario "$TRAIN_SCENARIO" \
              --experiment_tag ablation_a7 --train_steps "$TRAIN_STEPS" \
              --out_csv "$A7_CSV"        
    fi

    echo "[DONE] A7 | ${SITE} | seed=${seed}"
  done
done

echo ""
echo "========================================================"
echo "A7 ablation complete."
echo "========================================================"
