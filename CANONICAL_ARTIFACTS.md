# Canonical Artifacts — Phase 3 Final

This file tells any reader (human or AI assistant) exactly which files are
the source of truth, so results are never re-derived from stale data.

## Source of truth (use these)

| File | What it is |
|---|---|
| `results/phase3/master_summary.csv` | Raw episode-level results. 44,000 rows: 11 policies x 10 sites x 4 scenarios x 10 seeds x 10 episodes/seed. All statistics must be computed from this file, not hardcoded. |
| `results/phase3/final_thesis_numbers.txt` | Aggregated sanity-check report (means, oracle-vs-rlinv CIs, diesel/cost checks) derived from master_summary.csv. |
| `plot_phase3_core_figures_v3.py` | Canonical figure generation script (fig_eens_comparison, fig_ablation_h1/h2/h3). Reads master_summary.csv directly. Supersedes plot_chapter7.py. |
| `hypothesis_matrix_hard_vs_all.py` | Canonical H1/H2/H3 statistical comparison script (Welch's t-test + Cohen's d, seed-level), all-sites vs hard-sites (site2/site5/site7). |
| `results/tables/hypothesis_matrix_hard_vs_all.csv` | Output of the above script — frozen numbers for the thesis Hypothesis Evidence Matrix. |

## Hard sites definition

`site2`, `site5`, `site7` — sites where at least one policy's mean EENS
exceeds the constrained-site threshold in at least one scenario. Derived
in `final_thesis_numbers.txt` Section 3. NOTE: this differs from the
Phase 2 constrained-site definition (`site5`, `site10`), which no longer
applies after the Phase 3 evaluation-horizon audit changed the EENS scale.

## Deprecated / legacy (Phase 2 — do not use for Phase 3 conclusions)

| File | Status |
|---|---|
| `results/archive/phase2_legacy/plot_chapter7.py` | Phase 2 script. Reads `all_results_final.csv`, hardcodes `CONSTRAINED = ["site5","site10"]`, normal-scenario only. Superseded. |
| `results/archive/phase2_legacy/all_results_final.csv` | Phase 2 data (n=3 seeds, 5-7 policies, site5/site10/normal only). Do not merge with `master_summary.csv`. |

## Frozen hypothesis verdicts (as of this commit)

- **H1** (RLInv vs A6, ordering isolated): weak/inconsistent — significant only
  under Normal logistics; not significant Delayed/Monsoon; marginal Extreme.
  Confirmed identical conclusion on hard-sites-only subset (site2/5/7) — this
  is NOT a signal-dilution artifact.
- **H2** (A6 vs TrackB, masking isolated): strongly supported, scenario-conditional.
  Zero effect under Normal/Delayed; d=13-16 under Monsoon/Extreme. Interaction
  effect: masking only matters when ordering is externally decoupled (RLInv vs
  A7, both using joint ordering, shows no masking effect at all).
- **H3** (RLInv vs A5): not supported. Direction consistent, effect small
  (|d|<0.25), never significant.
