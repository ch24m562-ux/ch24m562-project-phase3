import pandas as pd
import glob
import sys
sys.path.insert(0, "src")

from eval.evaluate import h1_verdict, h1_verdict_by_lead

df_a  = pd.concat([pd.read_csv(f) for f in glob.glob("results/eval/rl_inv_*_fixed.csv")])
df_b  = pd.concat([pd.read_csv(f) for f in glob.glob("results/eval/track_b_*_fixed.csv")])
df_b0 = pd.concat([pd.read_csv(f) for f in glob.glob("results/eval/b0_*_fixed.csv")])
df_b1 = pd.concat([pd.read_csv(f) for f in glob.glob("results/eval/b1_*_fixed.csv")])

cols = ["EENS_kWh", "diesel_kWh", "cost_proxy", "stockout_events", "mean_inv_pct", "dg_on_fraction"]

print("=" * 65)
print("FULL RESULTS TABLE (mean across 5 episodes)")
print("=" * 65)
for name, df in [
    ("RL-Inv  (Track A)", df_a),
    ("Track B (s,S+PPO)", df_b),
    ("B0 Rule-based",     df_b0),
    ("B1 s,S+heuristic",  df_b1),
]:
    print(f"\n--- {name} ---")
    print(df.groupby(["site", "lead_scenario"])[cols].mean().round(2).to_string())

print()
print("=" * 65)
print("H1 VERDICT — overall (all sites, all leads)")
print("=" * 65)
v = h1_verdict(df_a, df_b)
print(f"  confirmed  : {v['confirmed']}")
print(f"  rejected   : {v['rejected']}")
print(f"  EENS improvement (A over B) : {v['eens_improvement']*100:.1f}%")
print(f"  stockout improvement        : {v['stockout_improvement']*100:.1f}%")
print(f"  RL-Inv  mean EENS : {v['rl_inv']['EENS_kWh']:.2f} kWh")
print(f"  Track B mean EENS : {v['track_b']['EENS_kWh']:.2f} kWh")
print(f"  RL-Inv  mean cost : {v['rl_inv']['cost_proxy']:.2f}")
print(f"  Track B mean cost : {v['track_b']['cost_proxy']:.2f}")

print()
print("=" * 65)
print("H1 VERDICT — by lead scenario")
print("=" * 65)
print(h1_verdict_by_lead(df_a, df_b).to_string())

print()
print("=" * 65)
print("H1 VERDICT — by site")
print("=" * 65)
for site in ["site1", "site5", "site7"]:
    a_s = df_a[df_a.site == site]
    b_s = df_b[df_b.site == site]
    vs = h1_verdict(a_s, b_s)
    print(f"  {site}: EENS_impr={vs['eens_improvement']*100:+.1f}%  "
          f"confirmed={vs['confirmed']}  "
          f"A_EENS={vs['rl_inv']['EENS_kWh']:.1f}  "
          f"B_EENS={vs['track_b']['EENS_kWh']:.1f}")

print()
print("=" * 65)
print("SITE5 DETAIL — the only discriminating site")
print("=" * 65)
for lead in ["normal", "delayed"]:
    a5 = df_a[(df_a.site == "site5") & (df_a.lead_scenario == lead)][cols].mean().round(2)
    b5 = df_b[(df_b.site == "site5") & (df_b.lead_scenario == lead)][cols].mean().round(2)
    print(f"\n  lead={lead}")
    print(f"    Track A: EENS={a5['EENS_kWh']:.1f}  diesel={a5['diesel_kWh']:.1f}  "
          f"inv_pct={a5['mean_inv_pct']:.3f}  dg_frac={a5['dg_on_fraction']:.3f}")
    print(f"    Track B: EENS={b5['EENS_kWh']:.1f}  diesel={b5['diesel_kWh']:.1f}  "
          f"inv_pct={b5['mean_inv_pct']:.3f}  dg_frac={b5['dg_on_fraction']:.3f}")
    eens_impr = (b5["EENS_kWh"] - a5["EENS_kWh"]) / max(abs(b5["EENS_kWh"]), 1e-9)
    print(f"    EENS improvement (A over B): {eens_impr*100:+.1f}%")
