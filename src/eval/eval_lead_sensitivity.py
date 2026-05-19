"""eval/eval_lead_sensitivity.py — Lead time sensitivity analysis (Phase 3)

Directly answers Reviewer Point 1:
  "A detailed analysis of the impact of delay in inventory will justify
   the proposed methodology."

Evaluates all trained policies across 6 lead time scenarios to produce
the EENS vs mean_lead_hours curve. No training needed — evaluation only.

Scientific story this script produces:
  - RLInv trained on normal: shows where advantage starts to erode
  - RLInv-Multi: shows whether domain randomisation closes the gap
  - TrackB / B1: show non-inventory baselines for comparison
  - The crossover point (μ_lead where RLInv advantage disappears) is the
    key scientific finding connecting to base-stock inventory theory

Usage:
  # Quick test (constrained sites, 3 seeds, 5 episodes)
  python src/eval/eval_lead_sensitivity.py ^
      --policies rlinv b1 ^
      --sites site5 site10 ^
      --seeds 42 123 777 ^
      --model_dir runs ^
      --episodes 5 ^
      --out_csv results/lead_sensitivity.csv

  # Full run (all policies, all seeds, plot)
  python src/eval/eval_lead_sensitivity.py ^
      --policies rlinv multi trackb b1 b0 ^
      --sites site5 site10 ^
      --seeds 42 123 777 7 13 21 99 314 500 999 ^
      --model_dir runs ^
      --episodes 10 ^
      --out_csv results/lead_sensitivity.csv ^
      --plot

Output CSV columns:
  policy, site, seed, lead_scenario, mean_lead_hours,
  EENS_kWh, diesel_kWh, stockout_events, uptime_pct,
  mean_inv_pct, min_inv_pct, cost_proxy, train_scenario, episode
"""
from __future__ import annotations

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import mlflow

from env.data_loader import load_site, train_test_split
from env.telecom_env import TelecomEnv
from eval.evaluate import evaluate, VecNormObsWrapper
from config_loader import env_cfg, reward_cfg

# ── Lead scenarios: (name, mean_lead_hours) ordered fastest → slowest ─────────
# All are defined in hparams.yaml lead_time section.
# no_delay = instant delivery — theoretical upper bound (perfect logistics)
LEAD_SCENARIOS_ORDERED = [
    ("no_delay",      0),    # instant — best case / theoretical upper bound
    ("fast",         12),    # 12h mean — urban India, nearby depot
    ("normal",       24),    # 24h mean — standard rural India (training baseline)
    ("delayed",      48),    # 48h mean — remote sites, minor supply disruption
    ("monsoon",      72),    # 72h mean — heavy monsoon season, rural India
    ("very_delayed", 120),   # 5 days  — severe monsoon / remote rural
    ("extreme",      336),   # 14 days — post-disaster supply disruption
]

# ── Policy metadata ────────────────────────────────────────────────────────────
# Each entry: (model_subdir, algo, train_scenario, is_baseline)
# model_subdir: subdirectory under model_dir where checkpoints are saved
# train_scenario: what lead the model was trained on (for CSV metadata)
POLICY_META = {
    "rlinv":  ("rlinv",        "maskable", "normal",  False),
    "multi":  ("rlinv_multi",  "maskable", "multi",   False),
    "trackb": ("trackb",       "ppo",      "normal",  False),
    "b1":     (None,           None,       "n/a",     True),
    "b0":     (None,           None,       "n/a",     True),
}


# ── Policy loaders ─────────────────────────────────────────────────────────────

def _load_rl_policy(model_path: str, algo: str):
    if algo == "maskable":
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(model_path)
    from stable_baselines3 import PPO
    return PPO.load(model_path)


def _load_baseline(name: str):
    if name == "b0":
        from baselines.rule_based import RuleBasedPolicy
        return RuleBasedPolicy()
    if name == "b1":
        from baselines.s_S_policy import B1Policy, SSPolicy
        return B1Policy(ss_policy=SSPolicy())
    raise ValueError(f"Unknown baseline: {name}")


# ── Env factory ────────────────────────────────────────────────────────────────

def _make_env_factory(site_csv: str, vecnorm_path: str,
                      ep_len: int, inv_low: float, inv_high: float):
    """Returns env_factory(site, lead, seed) compatible with evaluate()."""
    def factory(site: str, lead: str, seed: int) -> TelecomEnv:
        df, params = load_site(site_csv)
        _, df_test = train_test_split(df)
        env = TelecomEnv(
            site_data=df_test,
            site_params=params,
            episode_len=ep_len,
            eval_mode=True,
            lead_scenario=lead,
            seed=seed,
            init_inv_frac_low=inv_low,
            init_inv_frac_high=inv_high,
        )
        if vecnorm_path and os.path.exists(vecnorm_path):
            return VecNormObsWrapper(env, vecnorm_path)
        if vecnorm_path and not os.path.exists(vecnorm_path):
            print(f"  [WARN] vecnorm not found: {vecnorm_path} — running unnormalised")
        return env
    return factory


# ── Core evaluation loop ───────────────────────────────────────────────────────

def run_sensitivity(
    policies: list[str],
    sites: list[str],
    seeds: list[int],
    model_dir: str,
    leads: list[str],
    n_episodes: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate all (policy × site × seed × lead) combos. Returns combined DataFrame."""

    ep_len   = env_cfg["eval_episode_len"]
    inv_low  = env_cfg["init_inv_frac_eval_low"]
    inv_high = env_cfg["init_inv_frac_eval_high"]
    rc       = {"alpha": reward_cfg["alpha"], "beta": reward_cfg["beta"]}
    lead_to_hours = {name: hrs for name, hrs in LEAD_SCENARIOS_ORDERED}

    all_dfs = []

    for policy_name in policies:
        if policy_name not in POLICY_META:
            print(f"[WARN] Unknown policy '{policy_name}' — skipping")
            continue

        subdir, algo, train_scenario, is_baseline = POLICY_META[policy_name]

        for site in sites:
            site_csv = f"data/processed/{site}.csv"

            for seed in seeds:
                print(f"\n{'='*60}")
                print(f"[LeadSens] policy={policy_name}  site={site}  seed={seed}")

                # ── Load policy ───────────────────────────────────────────
                if is_baseline:
                    policy  = _load_baseline(policy_name)
                    vecnorm = ""
                else:
                    model_path = os.path.join(
                        model_dir, subdir, f"{site}_s{seed}_final.zip"
                    )
                    vecnorm = os.path.join(
                        model_dir, subdir, f"{site}_s{seed}_vecnorm.pkl"
                    )
                    if not os.path.exists(model_path):
                        print(f"  [SKIP] Model not found: {model_path}")
                        continue
                    policy = _load_rl_policy(model_path, algo)
                    print(f"  Loaded: {model_path}")

                # ── Env factory for this (site, seed) ────────────────────
                env_factory = _make_env_factory(
                    site_csv=site_csv,
                    vecnorm_path=vecnorm,
                    ep_len=ep_len,
                    inv_low=inv_low,
                    inv_high=inv_high,
                )

                meta = {
                    "policy":         policy_name,
                    "seed":           seed,
                    "train_scenario": train_scenario,
                    "experiment_tag": "lead_sensitivity",
                    "train_steps":    0,
                    "init_low":       inv_low,
                    "init_high":      inv_high,
                    "episode_len":    ep_len,
                }

                # ── Evaluate across all leads ─────────────────────────────
                df = evaluate(
                    policy=policy,
                    env_factory=env_factory,
                    n_episodes=n_episodes,
                    sites=[site],
                    leads=leads,
                    seed=seed,
                    rc=rc,
                    verbose=verbose,
                    meta=meta,
                )

                # Tag with mean lead hours for plotting
                df["mean_lead_hours"] = df["lead_scenario"].map(lead_to_hours)
                all_dfs.append(df)

    if not all_dfs:
        print("[WARN] No results collected — check model paths and policy names.")
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


# ── Aggregation helper ─────────────────────────────────────────────────────────

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over episodes and seeds per (policy, site, lead_scenario)."""
    grp = df.groupby(["policy", "site", "lead_scenario", "mean_lead_hours"])
    agg = grp.agg(
        EENS_mean       =("EENS_kWh",        "mean"),
        EENS_std        =("EENS_kWh",        "std"),
        EENS_min        =("EENS_kWh",        "min"),
        EENS_max        =("EENS_kWh",        "max"),
        diesel_mean     =("diesel_kWh",       "mean"),
        stockout_mean   =("stockout_events",  "mean"),
        uptime_mean     =("uptime_pct",       "mean"),
        cost_mean       =("cost_proxy",       "mean"),
        n               =("EENS_kWh",        "count"),
    ).reset_index()
    agg["EENS_sem"] = agg["EENS_std"] / np.sqrt(agg["n"].clip(lower=1))
    agg["EENS_ci95_lo"] = agg["EENS_mean"] - 1.96 * agg["EENS_sem"]
    agg["EENS_ci95_hi"] = agg["EENS_mean"] + 1.96 * agg["EENS_sem"]
    return agg.sort_values(["policy", "site", "mean_lead_hours"])


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_lead_sensitivity(agg_df: pd.DataFrame, out_path: str = "results/lead_sensitivity.png"):
    """EENS vs mean_lead_hours with 95% CI bands, one line per policy."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("[WARN] matplotlib not available — skipping plot")
        return

    sites = agg_df["site"].unique()
    n_sites = len(sites)

    fig, axes = plt.subplots(1, n_sites, figsize=(6 * n_sites, 5), sharey=False)
    if n_sites == 1:
        axes = [axes]

    # Colour palette — consistent across all plots in thesis
    COLOURS = {
        "rlinv":  "#1f77b4",   # blue
        "multi":  "#2ca02c",   # green
        "trackb": "#d62728",   # red
        "b1":     "#ff7f0e",   # orange
        "b0":     "#9467bd",   # purple
    }
    LABELS = {
        "rlinv":  "RLInv (normal)",
        "multi":  "RLInv-Multi",
        "trackb": "TrackB",
        "b1":     "B1 (s,S + rule)",
        "b0":     "B0 (rule only)",
    }

    # X-axis tick labels — human-readable lead time descriptions
    XTICK_LABELS = {
        0:   "0h\n(instant)",
        12:  "12h",
        24:  "24h\n(train)",
        48:  "48h",
        72:  "72h\n(monsoon)",
        120: "5d",
        336: "14d\n(disaster)",
    }

    for ax, site in zip(axes, sites):
        site_df = agg_df[agg_df["site"] == site]

        for policy in site_df["policy"].unique():
            pdata = site_df[site_df["policy"] == policy].sort_values("mean_lead_hours")
            colour = COLOURS.get(policy, "grey")
            label  = LABELS.get(policy, policy)

            ax.plot(
                pdata["mean_lead_hours"],
                pdata["EENS_mean"],
                marker="o", linewidth=2, color=colour, label=label,
            )
            ax.fill_between(
                pdata["mean_lead_hours"],
                pdata["EENS_ci95_lo"].clip(lower=0),
                pdata["EENS_ci95_hi"],
                alpha=0.15, color=colour,
            )

        # Mark the normal training lead time (24h) with a vertical dashed line
        ax.axvline(x=24, color="grey", linestyle="--", linewidth=1, alpha=0.6,
                   label="Training lead (24h)")

        ax.set_title(f"{site}", fontsize=13)
        ax.set_xlabel("Mean lead time (hours)", fontsize=11)
        ax.set_ylabel("EENS (kWh/episode)", fontsize=11)
        ax.set_xscale("log")                        # log scale — spans 0 to 336h
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        xticks = [h for h in [0, 12, 24, 48, 72, 120, 336] if h >= 0]
        ax.set_xticks(xticks)
        ax.set_xticklabels([
            "0h\n(instant)", "12h", "24h\n(train)",
            "48h", "72h\n(monsoon)", "5d", "14d\n(disaster)"
        ])
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "EENS vs Lead Time: RLInv degrades under delay — multi-scenario training closes gap",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved: {out_path}")
    plt.close()


# ── s* baseline comparison ─────────────────────────────────────────────────────

def compute_s_star_reorder_points(agg_df: pd.DataFrame, d_bar_kwh: float,
                                  tank_cap_kwh: float, safety_buffer_h: float = 48.0):
    """
    Compute classical optimal base-stock level s* for each lead time scenario.

    In the base-stock model with Poisson demand:
        s* = demand_rate × (mean_lead_time + safety_buffer)
           = d_bar × (μ_lead + safety_h)  [in kWh]
    Normalised as fraction of tank capacity.

    The safety buffer (48h) accounts for demand uncertainty during lead time.
    This is the theoretical reorder point the agent should be learning.
    """
    rows = []
    for _, g in agg_df.groupby("mean_lead_hours"):
        mean_lead_h = float(g["mean_lead_hours"].iloc[0])
        lead_name   = g["lead_scenario"].iloc[0]
        s_star_kwh  = d_bar_kwh * (mean_lead_h + safety_buffer_h)
        s_star_pct  = min(s_star_kwh / max(tank_cap_kwh, 1e-9), 1.0)
        rows.append({
            "lead_scenario":   lead_name,
            "mean_lead_hours": mean_lead_h,
            "s_star_kwh":      round(s_star_kwh, 2),
            "s_star_pct":      round(s_star_pct, 4),
            "d_bar_kwh":       round(d_bar_kwh, 3),
            "tank_cap_kwh":    round(tank_cap_kwh, 2),
        })
    return pd.DataFrame(rows).sort_values("mean_lead_hours")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Lead time sensitivity — EENS vs mean_lead_hours"
    )
    ap.add_argument("--policies",   nargs="+",
                    default=["rlinv", "b1"],
                    choices=list(POLICY_META.keys()),
                    help="Policies to evaluate")
    ap.add_argument("--sites",      nargs="+",
                    default=["site5", "site10"],
                    help="Sites to evaluate (default: constrained sites)")
    ap.add_argument("--all_sites",  action="store_true",
                    help="Evaluate all 10 sites")
    ap.add_argument("--seeds",      nargs="+", type=int,
                    default=[42, 123, 777])
    ap.add_argument("--model_dir",  type=str, default="runs",
                    help="Base directory containing policy subdirs")
    ap.add_argument("--episodes",   type=int, default=5,
                    help="Episodes per (site, lead, seed)")
    ap.add_argument("--leads",      nargs="+",
                    default=[s for s, _ in LEAD_SCENARIOS_ORDERED],
                    help="Lead scenarios to sweep (default: all 6)")
    ap.add_argument("--out_csv",    type=str,
                    default="results/lead_sensitivity.csv")
    ap.add_argument("--plot",       action="store_true",
                    help="Generate EENS vs lead_hours plot")
    ap.add_argument("--plot_path",  type=str,
                    default="results/lead_sensitivity.png")
    ap.add_argument("--s_star",     action="store_true",
                    help="Also compute classical s* reorder points for comparison")
    ap.add_argument("--mlflow",     action="store_true",
                    help="Log summary metrics to MLflow")
    ap.add_argument("--quiet",      action="store_true")
    args = ap.parse_args()

    if args.all_sites:
        args.sites = [f"site{i}" for i in range(1, 11)]

    print(f"\n{'='*60}")
    print(f"Lead Time Sensitivity Analysis")
    print(f"  Policies : {args.policies}")
    print(f"  Sites    : {args.sites}")
    print(f"  Seeds    : {args.seeds}")
    print(f"  Leads    : {args.leads}")
    print(f"  Episodes : {args.episodes}")
    print(f"{'='*60}\n")

    # ── Run ───────────────────────────────────────────────────────────────────
    df = run_sensitivity(
        policies=args.policies,
        sites=args.sites,
        seeds=args.seeds,
        model_dir=args.model_dir,
        leads=args.leads,
        n_episodes=args.episodes,
        verbose=not args.quiet,
    )

    if df.empty:
        print("[ERROR] No results — check model paths.")
        return

    # ── Save raw results ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\n[Saved] Raw results: {args.out_csv}  ({len(df)} rows)")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg = aggregate(df)
    agg_path = args.out_csv.replace(".csv", "_agg.csv")
    agg.to_csv(agg_path, index=False)
    print(f"[Saved] Aggregated : {agg_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n── EENS Summary (mean ± std over seeds×episodes) ─────────────────")
    for site in args.sites:
        print(f"\n  {site}:")
        tbl = agg[agg["site"] == site][
            ["policy", "lead_scenario", "mean_lead_hours",
             "EENS_mean", "EENS_std", "n"]
        ].copy()
        tbl["EENS_mean"] = tbl["EENS_mean"].round(1)
        tbl["EENS_std"]  = tbl["EENS_std"].round(1)
        print(tbl.to_string(index=False))

    # ── s* comparison ─────────────────────────────────────────────────────────
    if args.s_star:
        print("\n── Classical s* reorder points ───────────────────────────────")
        # Use site5 stats as representative (constrained site)
        try:
            from env.data_loader import load_site
            df5, p5 = load_site("data/processed/site5.csv")
            d_bar = float(df5["load_kwh"].mean())
            tank  = env_cfg["tank_hours"] * d_bar
            sstar = compute_s_star_reorder_points(agg, d_bar, tank)
            print(sstar.to_string(index=False))
            sstar.to_csv(args.out_csv.replace(".csv", "_sstar.csv"), index=False)
        except Exception as e:
            print(f"  [WARN] s* computation failed: {e}")

    # ── MLflow logging ────────────────────────────────────────────────────────
    if args.mlflow:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("lead_sensitivity")
        with mlflow.start_run(run_name="lead_sensitivity_sweep"):
            mlflow.log_param("policies",  str(args.policies))
            mlflow.log_param("sites",     str(args.sites))
            mlflow.log_param("n_seeds",   len(args.seeds))
            mlflow.log_param("episodes",  args.episodes)
            mlflow.log_artifact(args.out_csv)
            mlflow.log_artifact(agg_path)
            # Log key metric: EENS improvement RLInv vs B1 at normal lead
            try:
                rlinv_normal = agg[
                    (agg["policy"] == "rlinv") &
                    (agg["lead_scenario"] == "normal")
                ]["EENS_mean"].mean()
                b1_normal = agg[
                    (agg["policy"] == "b1") &
                    (agg["lead_scenario"] == "normal")
                ]["EENS_mean"].mean()
                if b1_normal > 0:
                    mlflow.log_metric(
                        "H1_eens_improvement_pct",
                        100 * (b1_normal - rlinv_normal) / b1_normal
                    )
            except Exception:
                pass
        print("[MLflow] Logged to lead_sensitivity experiment")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if args.plot:
        plot_lead_sensitivity(agg, out_path=args.plot_path)


if __name__ == "__main__":
    main()