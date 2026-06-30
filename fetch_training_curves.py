"""
fetch_training_curves.py
Pulls the full 49-point rollout/ep_rew_mean series for site2/site5/site7
(seed=999, normal scenario, training_curve_demo experiment) and saves
to CSV for the plot script to consume.
"""
import mlflow
import pandas as pd

client = mlflow.tracking.MlflowClient()
exps = client.search_experiments()
demo_exp = [e for e in exps if e.name == 'training_curve_demo'][0]
runs = client.search_runs(experiment_ids=[demo_exp.experiment_id], max_results=20)

rows = []
seen_sites = set()
for run in runs:
    if run.info.status != 'FINISHED':
        continue
    site = run.data.params.get('site')
    hist = client.get_metric_history(run.info.run_id, 'rollout/ep_rew_mean')
    if len(hist) == 0:
        continue
    # Keep only the first valid (non-empty) run per site -- skip duplicates
    if site in seen_sites:
        continue
    seen_sites.add(site)
    for h in sorted(hist, key=lambda x: x.step):
        rows.append({'site': site, 'step': h.step, 'reward': h.value})

df = pd.DataFrame(rows)
df.to_csv('results/training_curves/rollout_reward_hard_sites.csv', index=False)
print(f"Saved {len(df)} rows for sites: {sorted(seen_sites)}")
print(df.groupby('site')['reward'].agg(['first', 'last', 'count']))
