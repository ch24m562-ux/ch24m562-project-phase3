"""
compare_traces.py

Final check: do TrackB and A6 take the same actions (DG on/off, orders)
under Normal, or do different actions happen to produce the same aggregate
outcome? Run after generating both traces with --trace_out.
"""
import numpy as np

tb = np.load("check_trackb_site2_normal_s42.npz")
a6 = np.load("check_a6_site2_normal_s42.npz")

print("=" * 60)
print("Fields available in each trace:", list(tb.keys()))
print("=" * 60)

n = min(len(tb["dg_on"]), len(a6["dg_on"]))
print(f"Comparing first {n} steps (site2, normal, seed42)\n")

dg_match = (tb["dg_on"][:n] == a6["dg_on"][:n])
order_match = (tb["order_kwh"][:n] == a6["order_kwh"][:n])
inv_match = np.isclose(tb["inv_pct"][:n], a6["inv_pct"][:n], atol=1e-6)

print(f"DG on/off action match:     {dg_match.mean():.1%}  ({dg_match.sum()}/{n} steps)")
print(f"Order quantity match:       {order_match.mean():.1%}  ({order_match.sum()}/{n} steps)")
print(f"Inventory trajectory match: {inv_match.mean():.1%}  ({inv_match.sum()}/{n} steps)")

if not dg_match.all():
    first_diff = np.argmax(~dg_match)
    print(f"\nFirst DG action divergence at step {first_diff}:")
    print(f"  TrackB dg_on = {tb['dg_on'][first_diff]}")
    print(f"  A6     dg_on = {a6['dg_on'][first_diff]}")
    print(f"  (but inventory/EENS may still end up identical if this doesn't matter operationally)")
else:
    print("\n--> DG actions are IDENTICAL every single step.")

print()
if dg_match.mean() > 0.98 and order_match.mean() > 0.98:
    print("VERDICT: Actions genuinely converge -- this supports 'no bug, real")
    print("behavioural convergence under benign conditions' (90% hypothesis).")
elif dg_match.mean() < 0.5:
    print("VERDICT: Actions differ substantially. If EENS/diesel still matched")
    print("in the original aggregate data, that would be suspicious -- flag")
    print("for further investigation, do not close yet.")
else:
    print("VERDICT: Partial match -- inspect the specific divergence points above.")
