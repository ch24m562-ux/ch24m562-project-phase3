#!/usr/bin/env bash
# run_all_sites.sh  v3 — 400K, normal+delayed, randomised init inventory
#
# Usage:
#   bash run_all_sites.sh                  # full run
#   bash run_all_sites.sh --dry-run        # print only
#   SITES="1 2 3" bash run_all_sites.sh    # subset
#
# Key changes from v2:
#   - timesteps 200K → 400K  (convergence)
#   - delayed eval ON by default alongside normal
#   - init_diesel randomised 0.3–0.9  (breaks deterministic episodes)
#
# After: python aggregate_results.py --scenario normal
#        python aggregate_results.py --scenario delayed

set -e

SITES="${SITES:-1 2 3 4 5 6 7 8 9 10}"
SEEDS="${SEEDS:-42 123 777}"
TRAIN_STEPS="${TRAIN_STEPS:-400000}"
EVAL_EPISODES="${EVAL_EPISODES:-30}"
EVAL_SCENARIOS="normal delayed"
TRAIN_SCENARIO="normal"
INIT_LOW=0.3
INIT_HIGH=0.9
DRY_RUN=false

[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true && echo "[DRY RUN]"

echo "Sites: $SITES | Seeds: $SEEDS | Steps: $TRAIN_STEPS"
echo "Eval scenarios: $EVAL_SCENARIOS | Init inv: [${INIT_LOW}, ${INIT_HIGH}]"
echo ""

run() { echo ">>> $*"; $DRY_RUN || "$@"; }

for site_num in $SITES; do
  SITE="site${site_num}"
  SITE_CSV="data/processed/${SITE}.csv"
  [[ ! -f "$SITE_CSV" ]] && ! $DRY_RUN && echo "[SKIP] $SITE_CSV not found" && continue

  for seed in $SEEDS; do
    RLINV_DIR="runs/all_sites/${SITE}/seed${seed}/RLInv"
    TRACKB_DIR="runs/all_sites/${SITE}/seed${seed}/TrackB"
    RLINV_RES="results/all_sites/${SITE}/seed${seed}/RLInv"
    TRACKB_RES="results/all_sites/${SITE}/seed${seed}/TrackB"

    echo ""
    echo "========================================================"
    echo " ${SITE} | seed=${seed}"
    echo "========================================================"

    # ── Train RL-Inv ──────────────────────────────────────────────────────
    if [[ -f "${RLINV_DIR}/${SITE}_final_model.zip" ]] && ! $DRY_RUN; then
      echo "[SKIP TRAIN] RL-Inv exists"
    else
      run python -m src.train.train_rl_inv \
        --site "$SITE" --lead "$TRAIN_SCENARIO" \
        --timesteps "$TRAIN_STEPS" --seed "$seed" --logdir "$RLINV_DIR"
    fi

    # ── Train Track-B ─────────────────────────────────────────────────────
    if [[ -f "${TRACKB_DIR}/${SITE}_final_model.zip" ]] && ! $DRY_RUN; then
      echo "[SKIP TRAIN] Track-B exists"
    else
      run python -m src.train.train_track_b \
        --site "$SITE" --lead "$TRAIN_SCENARIO" \
        --timesteps "$TRAIN_STEPS" --seed "$seed" --logdir "$TRACKB_DIR"
    fi

    # ── Evaluate both policies under each scenario ────────────────────────
    for scenario in $EVAL_SCENARIOS; do

      RLINV_CSV="${RLINV_RES}/eval_${scenario}.csv"
      if [[ -f "$RLINV_CSV" ]] && ! $DRY_RUN; then
        echo "[SKIP EVAL] $RLINV_CSV"
      else
        run mkdir -p "$RLINV_RES"
        run python -m src.eval.evaluate \
          --site "$SITE" --lead "$scenario" \
          --policy_type rl --algo maskable --env_type track_a \
          --model_path    "${RLINV_DIR}/${SITE}_final_model" \
          --vecnorm_path  "${RLINV_DIR}/${SITE}_vecnormalize.pkl" \
          --episodes "$EVAL_EPISODES" --seed "$seed" \
          --init_diesel_low "$INIT_LOW" --init_diesel_high "$INIT_HIGH" \
          --policy_label RLInv --train_scenario "$TRAIN_SCENARIO" \
          --experiment_tag main --train_steps "$TRAIN_STEPS" \
          --out_csv "$RLINV_CSV"
      fi

      TRACKB_CSV="${TRACKB_RES}/eval_${scenario}.csv"
      if [[ -f "$TRACKB_CSV" ]] && ! $DRY_RUN; then
        echo "[SKIP EVAL] $TRACKB_CSV"
      else
        run mkdir -p "$TRACKB_RES"
        run python -m src.eval.evaluate \
          --site "$SITE" --lead "$scenario" \
          --policy_type rl --algo ppo --env_type track_b \
          --model_path    "${TRACKB_DIR}/${SITE}_final_model" \
          --vecnorm_path  "${TRACKB_DIR}/${SITE}_vecnormalize.pkl" \
          --episodes "$EVAL_EPISODES" --seed "$seed" \
          --init_diesel_low "$INIT_LOW" --init_diesel_high "$INIT_HIGH" \
          --policy_label TrackB --train_scenario "$TRAIN_SCENARIO" \
          --experiment_tag main --train_steps "$TRAIN_STEPS" \
          --out_csv "$TRACKB_CSV"
      fi

      echo "[DONE] ${SITE} seed=${seed} scenario=${scenario}"
    done
  done
done

echo ""
echo "========================================================"
echo "Main run complete. Next:"
echo "  python aggregate_results.py --scenario normal"
echo "  python aggregate_results.py --scenario delayed"
echo "  bash run_ablation_a7.sh"
echo "  bash run_stress.sh"
echo "========================================================"

# ── Convergence spot-check: 576K on 3 sites, seed=42 only
# Skipped when SKIP_SPOTCHECK=1 (e.g. mini-test runs) ────────────────────
# Runs automatically after main loop. Compare final ep_rew_mean vs 400K runs.
# If gap < 5%: keep 400K for thesis (document as verified).
# If gap >= 5%: bump TRAIN_STEPS=576000 above and rerun (skip logic saves time).
if [[ "${SKIP_SPOTCHECK:-0}" != "1" ]]; then
echo ""
echo "========================================================"
echo " Convergence spot-check: 576K on site1, site5, site7"
echo "========================================================"

for spot_site in site1 site5 site7; do
  for policy in "train_rl_inv:RLInv" "train_track_b:TrackB"; do
    script="${policy%%:*}"
    label="${policy##*:}"
    SPOT_DIR="runs/convergence_check/${spot_site}/seed42/${label}"

    if [[ -f "${SPOT_DIR}/${spot_site}_final_model.zip" ]] && ! $DRY_RUN; then
      echo "[SKIP] Convergence check exists: ${spot_site} ${label}"
    else
      echo ""
      echo "--- 576K | ${spot_site} | ${label} ---"
      run python -m src.train.${script}         --site      "$spot_site"         --lead      normal         --timesteps 576000         --seed      42         --logdir    "$SPOT_DIR"
    fi
  done
done

echo ""
echo "========================================================"
echo "Spot-check done. Decision guide:"
echo "  Look at final ep_rew_mean in each log above."
echo "  Compare against 400K result in runs/all_sites/*/seed42/"
echo "  If improvement < 5% on all 3 sites: stay at 400K."
echo "  If improvement >= 5%: set TRAIN_STEPS=576000 and rerun."
echo "========================================================"
fi  # SKIP_SPOTCHECK
