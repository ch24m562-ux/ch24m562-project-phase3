"""
plot_rlinv_da_behaviour.py
Behavioural comparison: inter-order gap by scenario, Base vs EWMA.
Reads: results/rlinv_da/rlinv_da_behaviour_summary.csv
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

df  = pd.read_csv("results/rlinv_da/rlinv_da_behaviour_summary.csv")
OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["normal","delayed","monsoon","extreme"]
SC_LABELS  = ["Normal\n(24h)","Delayed\n(48h)","Monsoon\n(72h)","Extreme\n(336h)"]
BG, BLUE, GREEN = "#FAFAFA", "#AEC6E8", "#90D4A0"

fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.8)
ax.set_axisbelow(True)

x    = np.arange(len(SCENARIOS))
w    = 0.35
base = [df[(df.scenario==sc)&(df.variant=="base")]["mean_gap"].values[0]
        if len(df[(df.scenario==sc)&(df.variant=="base")]) > 0 else 0
        for sc in SCENARIOS]
ewma = [df[(df.scenario==sc)&(df.variant=="ewma")]["mean_gap"].values[0]
        if len(df[(df.scenario==sc)&(df.variant=="ewma")]) > 0 else 0
        for sc in SCENARIOS]

ax.bar(x - w/2, base, w, label="RLInv (Base)", color=BLUE, edgecolor="white")
ax.bar(x + w/2, ewma, w, label="RLInv-DA (EWMA)", color=GREEN, edgecolor="white")

# Value labels
for xi, (b, e) in enumerate(zip(base, ewma)):
    ax.text(xi-w/2, b+1, f"{b:.0f}", ha="center", fontsize=8, color="#333333")
    ax.text(xi+w/2, e+1, f"{e:.0f}", ha="center", fontsize=8, color="#333333")

ax.set_xticks(x)
ax.set_xticklabels(SC_LABELS, fontsize=9.5)
ax.set_ylabel("Mean inter-order interval (hours)", fontsize=10)
ax.set_title("Behavioural Effect of EWMA-Based Delay Awareness on Replenishment Timing\n"
             "Mean inter-order interval across supplier-delay scenarios",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
for ext in ["pdf","png"]:
    out = OUT / f"fig_rlinv_da_behaviour.{ext}"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")
plt.close()
