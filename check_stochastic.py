import mlflow
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_ids=['22'], max_results=20)
print('Total stochastic grid runs:', len(runs))
for r in runs:
    site = r.data.params.get('site','?')
    seed = r.data.params.get('seed','?')
    eens = r.data.metrics.get('eval_EENS_kWh','?')
    tag  = r.data.params.get('tag','?')
    print('  site=%s seed=%s EENS=%s tag=%s' % (site, seed, eens, tag))
