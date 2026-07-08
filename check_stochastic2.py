import mlflow
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_ids=['22'], max_results=20)
for r in runs:
    site = r.data.params.get('site','?')
    seed = r.data.params.get('seed','?')
    eens = r.data.metrics.get('eval_EENS_kWh','?')
    policy = r.data.params.get('policy_type', r.data.params.get('policy','?'))
    scenario = r.data.params.get('lead_scenario', r.data.params.get('lead','?'))
    stoch = r.data.params.get('stochastic_grid','?')
    print('%s seed=%s scenario=%s policy=%s stoch=%s EENS=%s' % (site,seed,scenario,policy,stoch,eens))
