import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

RAW = r"C:\Users\dasja\projects\myproj\results\all_results_final.csv"
CLEAN = r"C:\Users\dasja\projects\myproj\results\all_results_clean.csv"

def cohens_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    sp = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    return (y.mean() - x.mean()) / sp

df = pd.read_csv(RAW)
print("Raw rows:", len(df))

df = df.drop_duplicates().copy()
print("Clean rows:", len(df))
df.to_csv(CLEAN, index=False)
print("Saved:", CLEAN)

# ---------- constrained normal ----------
sub = df[df["site"].isin(["site5", "site10"]) & (df["lead_scenario"] == "normal")].copy()

seed_means = (
    sub.groupby(["policy", "seed"], as_index=False)
       .agg(EENS_kWh=("EENS_kWh", "mean"),
            uptime_pct=("uptime_pct", "mean"),
            diesel_kWh=("diesel_kWh", "mean"))
)

main = (
    seed_means.groupby("policy", as_index=False)
    .agg(EENS_mean=("EENS_kWh", "mean"),
         EENS_std=("EENS_kWh", "std"),
         uptime_mean=("uptime_pct", "mean"),
         diesel_mean=("diesel_kWh", "mean"))
    .sort_values("EENS_mean")
)

print("\n=== Main constrained-site normal summary ===")
print(main.round(3).to_string(index=False))

# ---------- pairwise ----------
target = seed_means.loc[seed_means["policy"] == "RLInv", "EENS_kWh"].values
comparators = ["TrackB", "A6", "B0", "B1", "A5", "A7"]

rows = []
for comp in comparators:
    y = seed_means.loc[seed_means["policy"] == comp, "EENS_kWh"].values
    t_res = ttest_ind(target, y, equal_var=False)
    rows.append({
        "Comparator": comp,
        "Comparator_EENS": y.mean(),
        "Comparator_excess_over_RLInv_pct": (y.mean() - target.mean()) / target.mean() * 100,
        "p_value": t_res.pvalue,
        "cohens_d": cohens_d(target, y),
    })

pairwise = pd.DataFrame(rows).sort_values("Comparator_EENS")
print("\n=== Pairwise RLInv vs comparator ===")
print(pairwise.round({
    "Comparator_EENS": 3,
    "Comparator_excess_over_RLInv_pct": 1,
    "p_value": 3,
    "cohens_d": 2
}).to_string(index=False))

# ---------- delayed ----------
sub2 = df[df["site"].isin(["site5", "site10"])].copy()

seed_means2 = (
    sub2.groupby(["policy", "lead_scenario", "seed"], as_index=False)
        .agg(EENS_kWh=("EENS_kWh", "mean"))
)

summary2 = (
    seed_means2.groupby(["policy", "lead_scenario"], as_index=False)
        .agg(EENS_mean=("EENS_kWh", "mean"))
)

delayed = summary2.pivot(index="policy", columns="lead_scenario", values="EENS_mean").reset_index()
delayed["Delta_pct"] = (delayed["delayed"] - delayed["normal"]) / delayed["normal"] * 100
delayed = delayed.sort_values("Delta_pct")

print("\n=== Delayed logistics table ===")
print(delayed.round(3).to_string(index=False))

# ---------- cross-site ----------
sub3 = df[df["lead_scenario"] == "normal"].copy()

seed_means3 = (
    sub3.groupby(["site", "policy", "seed"], as_index=False)
        .agg(EENS_kWh=("EENS_kWh", "mean"))
)

site_policy = (
    seed_means3.groupby(["site", "policy"], as_index=False)
        .agg(EENS_mean=("EENS_kWh", "mean"))
)

cross = site_policy.pivot(index="site", columns="policy", values="EENS_mean").reset_index()
cross["Comparator_excess_over_RLInv_pct"] = np.where(
    cross["RLInv"] > 0,
    (cross["B1"] - cross["RLInv"]) / cross["RLInv"] * 100,
    np.nan
)

print("\n=== Cross-site RLInv vs B1 ===")
print(cross[["site", "RLInv", "B1", "Comparator_excess_over_RLInv_pct"]]
      .round(3)
      .to_string(index=False))