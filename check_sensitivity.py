import mlflow
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_ids=['19'])

variants = {}
for run in runs:
    p = run.data.params
    v = p.get('variant','')
    if v not in variants:
        variants[v] = {'lam': p.get('lam'), 'gamma_r': p.get('gamma_r'),
                       'beta': p.get('beta'), 'mu': p.get('mu'),
                       'gamma_ppo': p.get('gamma_ppo')}

print('VARIANT PARAMETER VALUES:')
for v, params in sorted(variants.items()):
    print(v, ':', params)

print()
from collections import Counter
counts = Counter(run.data.params.get('variant') for run in runs)
print('Runs per variant:', dict(sorted(counts.items())))
print('Sites:', sorted(set(run.data.params.get('site') for run in runs)))
print('Seeds:', sorted(set(run.data.params.get('seed') for run in runs)))
