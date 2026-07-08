# Run two more diagnostic traces to confirm sparsity is a property of the environment

# Trace 1: site5/seed777/monsoon (EWMA helped here)
python src/eval/evaluate.py `
  --site site5 --lead monsoon --seed 777 `
  --episodes 1 --episode_len 360 `
  --policy_type rl --algo maskable `
  --model_path runs/rlinv_da/ewma_regime/site5_s777_final.zip `
  --train_scenario normal `
  --lead_dist lognormal --lead_sigma 0.5 `
  --use_supplier_regime --use_ewma_lead `
  --policy_label rlinv_da_ewma `
  --trace_out results/traces/rlinv_da/ewma_site5_monsoon_s777_ep0.npz `
  --out_csv results/traces/rlinv_da/ewma_site5_monsoon_s777.csv

# Trace 2: site2/seed777/monsoon (base was better here)
python src/eval/evaluate.py `
  --site site2 --lead monsoon --seed 777 `
  --episodes 1 --episode_len 360 `
  --policy_type rl --algo maskable `
  --model_path runs/rlinv_da/ewma_regime/site2_s777_final.zip `
  --train_scenario normal `
  --lead_dist lognormal --lead_sigma 0.5 `
  --use_supplier_regime --use_ewma_lead `
  --policy_label rlinv_da_ewma `
  --trace_out results/traces/rlinv_da/ewma_site2_monsoon_s777_ep0.npz `
  --out_csv results/traces/rlinv_da/ewma_site2_monsoon_s777.csv
