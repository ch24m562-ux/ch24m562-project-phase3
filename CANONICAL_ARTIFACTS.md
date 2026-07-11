# Canonical Artifacts — Phase 3 Final

This file tells any reader (human or AI assistant) exactly which files are
the source of truth, so results are never re-derived from stale data.

## Source of truth (use these)

| File | What it is |
|---|---|
| `results/phase3/master_summary.csv` | Raw episode-level results. 44,000 rows: 11 policies x 10 sites x 4 scenarios x 10 seeds x 10 episodes/seed. All statistics must be computed from this file, not hardcoded. |
| `results/phase3/final_thesis_numbers.txt` | Aggregated sanity-check report (means, oracle-vs-rlinv CIs, diesel/cost checks) derived from master_summary.csv. |
| `plot_phase3_core_figures_v3.py` | Canonical figure generation script (fig_eens_comparison, fig_ablation_h1/h2/h3). Reads master_summary.csv directly. Supersedes plot_chapter7.py. |
| `hypothesis_matrix_hard_vs_all.py` | Canonical H1/H2/H3 statistical comparison script (paired t-test + Cohen's d_z, seed-level), all-sites vs hard-sites (site2/site5/site7). |
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

## Statistical methodology (corrected)

Comparisons are **paired at the seed level** — the same 10 seeds are evaluated
under every policy, so seed 42 under RLInv and seed 42 under A6 are matched
observations, not independent samples. `hypothesis_matrix_hard_vs_all.py`
therefore uses `scipy.stats.ttest_rel` (paired t-test) on the within-seed
differences, with effect size Cohen's d_z = mean(diff) / std(diff), not a
two-sample Welch test with pooled-variance d. An earlier version of this
script incorrectly used `ttest_ind(equal_var=False)`, which understates
significance for this design (e.g. it reported Extreme as only "marginal",
p=0.064, when the correct paired test gives p=0.00035). Do not revert to
the independent-samples version.

Sign convention: d_z is signed and reported with an explicit "favours X"
label rather than an unsigned magnitude. When two policies are numerically
identical at every seed (paired difference has zero variance), the test
statistic is mathematically undefined and is reported as "IDENTICAL", not
as p=1.0/d=0.

## Frozen hypothesis verdicts (as of this commit — paired test)

- **H1** (RLInv vs A6, joint replenishment isolated — both share RL dispatch, paired):
  - Normal:  p=0.0175*,  d_z=-0.92  (favours RLInv)
  - Delayed: p=0.83 ns,  d_z=-0.07  (negligible)
  - Monsoon: p=0.41 ns,  d_z=+0.27  (favours A6, nominally — not significant)
  - Extreme: p=0.00035***, d_z=-1.76 (favours RLInv, large effect)

  **Conclusion:** The comparison between RLInv and A6 indicates that jointly
  learned replenishment provides additional benefit under Normal and Extreme
  logistics, while the difference is negligible under Delayed logistics and
  not distinguishable under Monsoon conditions. (Note: A6 is not an
  "ordering-only" ablation — it still shares RL dispatch with RLInv, differing
  only in whether replenishment is jointly learned or externally fixed via
  (s,S). The wording above reflects that the comparison isolates joint
  replenishment specifically, not "ordering" as a standalone mechanism.)
  The interpretation of H1 should therefore be scenario-dependent rather
  than a single global conclusion — do not write "H1 supported" or
  "H1 rejected" anywhere in the thesis; both are false simplifications of a
  conditional result. Identical qualitative pattern on hard-sites-only
  (site2/5/7) — not a signal-dilution artifact.

- **H2** (A6 vs TrackB, masking isolated, paired): no meaningful difference
  under Normal/Delayed (identical or ns); very large effect under Monsoon
  (d_z=-10.2) and Extreme (d_z=-4.6), both p<0.0001, favouring A6.
  **Conclusion (narrow form):** under external (s,S) ordering, action
  masking prevents severe dispatch-policy degradation under stressed
  logistics. Do NOT claim "the entire RLInv-TrackB gap is attributable to
  masking" — RLInv and TrackB also differ in ordering and in the
  ordering-dispatch interaction, and the A6-vs-TrackB effect size is not a
  clean decomposition of the RLInv-vs-TrackB effect size. The defensible
  claim is that A6-vs-TrackB shows masking accounts for a substantial part
  of the RLInv-TrackB gap under Monsoon/Extreme — not all of it.

- **H2-secondary** (RLInv vs A7, masking isolated with joint ordering held
  constant, paired): no significant difference in any scenario (Normal:
  identical on hard sites, ns on all-sites; Delayed/Monsoon: nominally
  favours A7 but ns, |d_z|<0.35; Extreme: ~0, ns). **Conclusion:** when
  ordering is already jointly learned, action masking has no measurable
  effect on mean EENS — masking's benefit (see H2 above) is conditional on
  ordering being externally decoupled, not a main effect of masking alone.
  Re-verified under the paired test; conclusion unchanged from the earlier
  (correct, since RLInv/A7 are always near-identical) independent-samples
  pass.

- **H3** (RLInv vs A5, paired): H3 provides limited evidence that explicit
  inventory observation improves performance. Although RLInv consistently
  achieves lower mean EENS than A5 in every scenario (|d_z| from 0.24 to
  0.69, largest and closest to significance under Extreme at p=0.058), the
  differences are small and do not reach statistical significance under the
  final evaluation protocol. Re-verified under the paired test; conclusion
  unchanged from the earlier independent-samples pass (this comparison was
  never borderline enough for the independent-vs-paired distinction to
  matter, unlike H1).

## Thesis naming convention (chapter headings vs internal labels)

H1/H2/H3 are useful internal shorthand (this document, the scripts, viva
prep) but readers of the thesis do not retain hypothesis numbers — they
retain what the mechanism does. In the thesis itself:

- Use descriptive names in chapter/section headings and in-text discussion:
  **Joint replenishment learning** (H1), **Action masking** (H2),
  **Inventory observability** (H3).
- Mention the H1/H2/H3 labels **once**, at first introduction (e.g. "the
  three ablations introduced in Chapter 3, hereafter referred to by their
  mechanism: joint replenishment learning, action masking, and inventory
  observability"), then use the descriptive names exclusively afterward.
- Do NOT rewrite the hypotheses themselves to match the evidence — the
  hypotheses stay exactly as originally formulated in Chapter 3. What
  changed across Phase 2 -> Phase 3 -> the paired-test correction is the
  *evidence*, not the *hypothesis*. Keep that distinction explicit in the
  thesis: state the hypothesis as originally written, then report what the
  final evidence shows relative to it.

## Discussion-chapter summary (deliberately non-evaluative wording)

| Component | Evidence | Contribution |
|---|---|---|
| Joint replenishment learning | Scenario-dependent | Moderate |
| Action masking | Strong under stressed logistics | Major |
| Inventory observability | Small | Secondary |

This table avoids subjective intensifiers ("important", "critical",
"dominant") — the scenario-conditional evidence above is what supports each
cell, not an editorial judgment layered on top of it.
