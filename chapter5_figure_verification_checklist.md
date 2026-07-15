# Chapter 5 (Results) — Figure Verification Checklist
Internal working document — not part of the thesis. Generated while assembling
`chap-results-v6-DRAFT.tex`. Re-run this check any time master_summary.csv or a
sensitivity dataset changes.

| Figure | Source script | Dataset(s) | Regenerated this session? | Verified? |
|---|---|---|---|---|
| fig_eens_comparison | `plot_phase3_core_figures_v3.py` | `results/phase3/master_summary.csv` | Yes | ✅ Numbers cross-checked against `final_thesis_numbers.txt` pivot (rlinv/b1/trackb/oracle_mpc rows match Table primary_comparison exactly) |
| fig_ablation_h1 | `plot_phase3_core_figures_v3.py` | `results/phase3/master_summary.csv` | Yes | ✅ Bars match master_summary means; caption/text now cite paired-test stats from `hypothesis_matrix_hard_vs_all.csv`, not just CI overlap |
| fig_ablation_h2 | `plot_phase3_core_figures_v3.py` | `results/phase3/master_summary.csv` | Yes | ✅ Same as above; A6-vs-TrackB framing matches CANONICAL_ARTIFACTS.md H2 verdict |
| fig_ablation_h3 | `plot_phase3_core_figures_v3.py` | `results/phase3/master_summary.csv` | Yes | ✅ Matches CANONICAL_ARTIFACTS.md H3 verdict (p=0.058 extreme, not significant) |
| fig_cross_site | `results/plot_chapter7.py` (archived script, but this specific figure's numbers verified independently) | `results/phase3/master_summary.csv` | No (reused from repo) | ✅ Per-site RLInv-vs-B1 normal-logistics claim independently recomputed from master_summary.csv and confirmed true at all 10 sites |
| fig_trajectory | `plot_trajectory.py` | `results/traces/rlinv_site2_monsoon_s99_ep0.npz`, `b1_site2_monsoon_s99_ep0.npz` (raw per-episode traces, not a summary CSV) | Yes (re-run fresh this session) | ✅ One thing worth knowing: the legend text for EENS/order-count is written as literal strings in the script, not computed live from the array on every run. I recomputed both directly from the `.npz` arrays independently (`(order_kwh>0).sum()`, `unmet_kwh.sum()`) and confirmed exact match: RLInv 4 orders/0.00 kWh unmet, B1 1 order/140.63 kWh unmet. Also confirmed both trace files were added in the same commit as the script (2026-06-29), so no version-mismatch risk. |
| fig_reward_sensitivity | `plot_reward_sensitivity.py` | `results/sensitivity/reward_sensitivity_summary.csv` | No (reused from project folder) | ⚠️ Independent of master_summary.csv by design (separate sensitivity sweep) — confirmed script does NOT touch the deprecated `all_results_final.csv`. Not re-run this session. |
| fig_tank_sensitivity | `plot_tank_sensitivity.py` | `results/sensitivity/tank/tank_summary.csv` | Yes (re-run fresh this session) | ✅ `tank_summary.csv` values (215.554/283.553/67.999 at 24h ... 0.000/0.000/0.000 at 336h) match Table `tab:tank_sensitivity` in the chapter exactly. Script also has a hardcoded fallback if the CSV is missing — confirmed the CSV exists and is what actually loaded (fallback did not trigger). Raw per-episode CSVs behind the summary use site2/site5/site7, consistent with the updated hard-site definition. |
| fig_distributional_robustness | `plot_distributional_robustness.py` | `results/sensitivity/lognormal/`, `lognormal_sigma08/`, `weibull_k2/` | Yes (re-run fresh this session) | ✅ Confirms 4 curves (geometric + 2 lognormal + Weibull) per policy, matches caption |
| fig_lead_sensitivity | `plot_lead_sensitivity.py` (retitled copy for Results placement) | `results/lead_sensitivity.csv` | Yes (re-run fresh this session) | ✅ Advantage-over-B1 numbers (1.3 / 29.5 / 22.8 / 8.6 kWh at 24/72/120/336h) independently recomputed from the CSV and confirmed |
| fig_rlinv_da / fig_rlinv_da_behaviour | (RLInv-DA subsystem scripts) | RLInv-DA-specific eval runs | No | ⚠️ Marked "confirmed final" in your own tracker (Section K) — not independently re-verified this session, taken on trust from that prior sign-off |

## Known deprecated source (must never feed Chapter 5 numbers)
`results/archive/phase2_legacy/all_results_final.csv` + `plot_chapter7.py` (archived
copy) — n=3 seeds, site5/site10/normal-only, Phase 2. Confirmed **not** referenced by
any of the scripts above.

## Outstanding before final freeze
1. ~~Re-verify `fig_trajectory` and `fig_tank_sensitivity`~~ — done this pass, both check out (see rows above).
2. `fig_rlinv_da`/`fig_rlinv_da_behaviour` rely on an earlier sign-off rather than a
   fresh check this session; worth one more look given how much else has changed.
3. Both `plot_trajectory.py` and `plot_tank_sensitivity.py` docstrings say "For thesis
   Chapter 8 -- methodology justification" (same stale-labeling pattern found in
   `plot_lead_sensitivity.py`). Harmless for our purposes since we're placing these in
   Chapter 5 regardless, but worth a cleanup pass in the repo itself so the docstrings
   don't contradict where the figures actually end up.
