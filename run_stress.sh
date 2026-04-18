#!/usr/bin/env bash
# run_stress.sh — Stress-trained RL-Inv (Track A only) across all 10 sites
#
# Purpose: H4 / robustness — train under delayed+scarce conditions,
#          evaluate under delayed+scarce. Answers reviewer: "did you
#          train under delayed conditions?"
#
# Compare against: standard RL-Inv evaluated under delayed
#   results/all_sites/*/RLInv/eval_delayed.csv  (normal-trained, delayed eval)
#   results/stress/*/RLInv/eval_delayed.csv      (stress-trained, delayed eval)
#
# After: python aggregate_results.py --scenario delayed --include_stress

set -e

SITES="${SITES:-1 2 3 4 5 6 7 8 9 10}"
SEEDS="${SEEDS:-42 123 777}"
TRAIN_STEPS=400000
EVAL_EPISODES=30
TRAIN_SCENARIO="delayed"
INIT_LOW=0.10
INIT_HIGH=0.20
DRY_RUN=false

[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true && echo "[DRY RUN]"

echo "Stress Run — RL-Inv trained under delayed+scarce conditions"
echo "Sites: $SITES | Seeds: $SEEDS | Steps: $TRAIN_STEPS"
echo "Train: lead=${TRAIN_SCENARIO}, init_inv=[${INIT_LOW}, ${INIT_HIGH}]"
echo ""

run() { echo ">>> $*"; $DRY_RUN || "$@"; }

for site_num in $SITES; do
  SITE="site${site_num}"
  SITE_CSV="data/processed/${SITE}.csv"
  [[ ! -f "$SITE_CSV" ]] && ! $DRY_RUN && echo "[SKIP] $SITE_CSV not found" && continue

  for seed in $SEEDS; do
    STRESS_DIR="runs/stress/${SITE}/seed${seed}/RLInv"
    STRESS_RES="results/stress/${SITE}/seed${seed}/RLInv"

    echo ""
    echo "========================================================"
    echo " STRESS | ${SITE} | seed=${seed}"
    echo "========================================================"

    # ── Train stress RL-Inv ───────────────────────────────────────────────
    if [[ -f "${STRESS_DIR}/${SITE}_final_model.zip" ]] && ! $DRY_RUN; then
      echo "[SKIP TRAIN] Stress model exists"
    else
      run python -m src.train.train_rl_inv_stress \
        --site "$SITE" --lead "$TRAIN_SCENARIO" \
        --timesteps "$TRAIN_STEPS" --seed "$seed" --logdir "$STRESS_DIR" \
        --init_diesel_low "$INIT_LOW" --init_diesel_high "$INIT_HIGH"
    fi

    # ── Evaluate stress model under delayed (primary comparison) ──────────
    STRESS_CSV_DELAYED="${STRESS_RES}/eval_delayed.csv"
    if [[ -f "$STRESS_CSV_DELAYED" ]] && ! $DRY_RUN; then
      echo "[SKIP EVAL] $STRESS_CSV_DELAYED"
    else
      run mkdir -p "$STRESS_RES"
      run python -m src.eval.evaluate \
        --site "$SITE" --lead delayed \
        --policy_type rl --algo maskable --env_type track_a \
        --model_path    "${STRESS_DIR}/${SITE}_final_model" \
        --vecnorm_path  "${STRESS_DIR}/${SITE}_vecnormalize.pkl" \
        --episodes "$EVAL_EPISODES" --seed "$seed" \
        --init_diesel_low "$INIT_LOW" --init_diesel_high "$INIT_HIGH" \
        --out_csv "$STRESS_CSV_DELAYED"
    fi

    # ── Also evaluate stress model under normal (OOD check) ───────────────
    STRESS_CSV_NORMAL="${STRESS_RES}/eval_normal.csv"
    if [[ -f "$STRESS_CSV_NORMAL" ]] && ! $DRY_RUN; then
      echo "[SKIP EVAL] $STRESS_CSV_NORMAL"
    else
      run python -m src.eval.evaluate \
        --site "$SITE" --lead normal \
        --policy_type rl --algo maskable --env_type track_a \
        --model_path    "${STRESS_DIR}/${SITE}_final_model" \
        --vecnorm_path  "${STRESS_DIR}/${SITE}_vecnormalize.pkl" \
        --episodes "$EVAL_EPISODES" --seed "$seed" \
        --init_diesel_low "$INIT_LOW" --init_diesel_high "$INIT_HIGH" \
        --out_csv "$STRESS_CSV_NORMAL"
    fi

    echo "[DONE] STRESS | ${SITE} | seed=${seed}"
  done
done

echo ""
echo "========================================================"
echo "Stress run complete."
echo "========================================================"
