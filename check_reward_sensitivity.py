"""
check_reward_sensitivity.py
Apples-to-apples comparison: reward variants vs B1 baseline
on SAME sites (site2/site5/site7), SAME seeds (42/123/777),
SAME scenarios (normal/delayed/monsoon/extreme).
"""
import pandas as pd
import numpy as np

# Load reward sensitivity results
sens = pd.read_csv('results/sensitivity/reward_sensitivity_summary.csv')
print("Sensitivity variants:", sorted(sens.policy.unique()))
print("Scenarios:", sorted(sens.lead_scenario.unique()))
print()

# Load master_summary for B1 on same hard sites + same seeds
master = pd.read_csv('results/phase3/master_summary.csv')

HARD_SITES = ['site2', 'site5', 'site7']
SENS_SEEDS = [42, 123, 777]
SCENARIOS  = ['normal', 'delayed', 'monsoon', 'extreme']

b1_matched = master[
    (master.policy == 'b1') &
    (master.site.isin(HARD_SITES)) &
    (master.seed.isin(SENS_SEEDS))
].groupby('lead_scenario')['EENS_kWh'].mean()

print("B1 mean EENS -- hard sites only (site2/5/7), seeds 42/123/777:")
for sc in SCENARIOS:
    print(f"  {sc:<10}: {b1_matched.get(sc, float('nan')):.2f} kWh")
print()

# Compare against each sensitivity variant
print("APPLES-TO-APPLES: Reward variants vs B1 (same sites, same seeds)")
print(f"{'Variant':<15} {'normal':>8} {'delayed':>9} {'monsoon':>9} {'extreme':>9} {'beats B1 all scenarios?':>24}")
print("-" * 80)

all_variants_beat_b1 = True
pivot = sens.pivot_table(index='policy', columns='lead_scenario', values='mean')

for variant in sorted(pivot.index):
    row = []
    beats_all = True
    for sc in SCENARIOS:
        v_eens = pivot.loc[variant, sc] if sc in pivot.columns else float('nan')
        b1_eens = b1_matched.get(sc, float('nan'))
        row.append(f"{v_eens:>9.2f}")
        if v_eens >= b1_eens:
            beats_all = False
            all_variants_beat_b1 = False
    print(f"{variant:<15}" + "".join(row) + f"  {'YES' if beats_all else 'NO -- check'}")

print("-" * 80)
print(f"B1 (matched) " + "".join(f"{b1_matched.get(sc, float('nan')):>9.2f}" for sc in SCENARIOS))
print()
print(f"All variants beat B1 on hard sites: {'YES' if all_variants_beat_b1 else 'NO -- see table'}")
