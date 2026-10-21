# PHASE3R_PROTOCOL.md
## Pre-registered experimental protocol — Phase 3 resubmission

| | |
|---|---|
| **Author** | Sangeeta Gupta (CH24M562) |
| **Protocol version** | **1.9 — audit record closed (design unchanged since 1.5)** |
| **Frozen on** | 2026-10-21 |
| **Amended** | **2026-10-22 — v1.8 historical simulator, dataset and reproducibility audit** (§1.4–1.8, §13, Appendices). Earlier design freeze retained; energy-audit script replaced with independently reconstructed checks.<br>**2026-10-22 — v1.9: repository G0 energy audit EXECUTED and PASSED (§1.4); ECG-1 dataset provenance RESOLVED (§1.9); role-based parameter table (§7.4b); battery claim discipline (§1.10).** |
| **Git commit at freeze** | 86e9632 |
| **Resubmission deadline** | 2026-11-11 |
| **Experiment freeze target** | 2026-11-04 (revised from 28 Oct after late start) |

**Status of this document.** This is a *pre-registration*. It is committed to the
repository **before** the first new Phase 3-R training run. Every design choice
below was made without sight of any **Phase 3-R result**. Changes after the freeze
date are recorded in §12 (Amendment log) with a date and reason — never silently.
**The 2026-10-22 v1.8 amendment is an evidence/provenance correction, NOT a
redesign:** the G3/G4/G5 hypotheses, sites, seeds, budgets, tuning rules,
comparison contrasts, and transfer split remain as frozen in v1.7. Historical
Phase 3 results are read-only, and G0 audit results do not count as new policy
performance results. **Do not modify historical `TelecomEnv` transitions or its
battery coefficients for any directly matched Phase 3-R primary comparison.**

**Why this exists.** Three independent instrumentation defects have already been
found in this codebase (Bug 1 episode-length mismatch, Bug 3 site-variable
shadowing, and the suspected TrackB action-override). All three shared one
property: *instrumentation that appeared to be recording something and was not.*
A fourth was found on 2026-10-04 (MLflow `start_run()` opened after
`model.learn()` in four trainers, producing zero-duration timings). The gates,
the sanity gate and the manifest below exist to make that class of defect
visible before it reaches a thesis table.

---

## 0. Decisions frozen at G0

| # | Decision | Frozen value | Source |
|---|---|---|---|
| D1 | G4 hyperparameters | Limited tuning: **3 configurations each for DQN and A2C only** (PPO: 0 — D22), on one development site, selected on a training-window validation slice. Configurations frozen in §7.5. | Mentor 2 |
| D2 | Hyperparameter commonality | Environment, reward, observation, action space, interaction budget and evaluation protocol identical. Algorithm-specific optimiser hyperparameters permitted and **not** forced identical. ⚠️ **This is not an equal-tuning design** — see D22. | Mentor 2 correction |
| D3 | G4 feasibility handling | Common environment-level handling for all unmasked arms (see §6.3 — already implemented, no new code) | Mentor 2, Option B |
| D4 | G4 arms | MaskablePPO (RLInv) / PPO-no-mask (A7 bridge) / DQN / A2C | Both mentors |
| D5 | TrackB retrain scope | Full 10 sites × 10 seeds, **conditional** on the three gates in §5.1 | Both mentors, after costing |
| D6 | Shared-policy budget | 400k matched-budget primary; 800k pre-declared sensitivity check | Mentor 2 |
| D7 | Transfer controls | scratch-50k and scratch-100k included | Mentor 2 |
| D8 | Site selection | From `site_classification.csv` by the stated rule in §6.2 | Both mentors |
| D9 | G3 selection objective | Training-window EENS; diesel and order count reported, never used for selection | Mentor 2 |
| D10 | Statistical standing | 3 seeds = screening only. 5 seeds = planned minimum for G4. Significance exploratory below adequate replication. | Mentor 2 Correction B |
| D11 | Core count | Measured automatically, recorded in the manifest. Does not gate the design. | Mentor 2 |
| D12 | Baseline Taxonomy & Fairness table | Mandatory addition to the methodology chapter (§6.5) | Mentor 2 |
| D13 | Scalability Limitations subsection | Mandatory, five dimensions, each with a named limitation (§9.7) | Mentor 2 |
| D14 | Highlighting + rebuttal appendix | Formal **exit criteria** for G6, not writing tasks (§11.6–11.8) | Resubmission instructions |
| D15 | Rebuttal format | Reuse the proven Phase 2 institutional format, not a generic table (§11.7) | Verified precedent |
| D16 | **Experimental Comparability Gate (ECG)** | Mandatory G0 deliverable alongside PSG. Full specification in **§13**. No result enters the thesis until **both** PSG and ECG pass. | Mentor 2 |
| D17 | Matched-subset comparison | New 3-site × 5-seed results are **never** compared against the 10×10 headline means. RLInv and A7 are recomputed on the identical subset. (§13.6) | Mentor 2 |
| D19 | G4 inferential standing | **Exploratory in its entirety.** Two prespecified focal comparisons under Normal — the **focal algorithm contrast** (A7 vs DQN) and the **principal controller comparison** (RLInv vs DQN), per D23. All others secondary/descriptive. Neither is presented as a confirmatory superiority test. (§10.8) | Mentor 2 |
| D20 | Pooling rule | Primary pooled estimate = unweighted mean of site-specific mean differences. **Never** a single t-test over pooled site-seed rows. (§10.5) | Mentor 2 |
| D22 | Tuning asymmetry named honestly | The design is **"submitted PPO configuration vs limited-tuned alternatives"**, *not* equal hyperparameter optimisation. PPO receives 0 configs, DQN and A2C 3 each. The asymmetry favours DQN/A2C and is disclosed as such. (§7.5, §13.7) | Mentor 2 |
| D23 | Focal contrasts | **Focal algorithm contrast: A7 vs DQN, Normal** (both unmasked). **Principal controller comparison: RLInv vs DQN, Normal.** (§10.8) | Mentor 2 |
| D24 | G5b reporting | Per-site **primary**, equal-site-weighted aggregate **secondary**. No pooled *d*<sub>z</sub>, no generalisation significance claim. (§9.6) | Mentor 2 |
| D21 | Pooled *d*<sub>z</sub> suppressed | Per-site *d*<sub>z</sub> at n=5 with all raw differences shown; **no pooled *d*<sub>z</sub>** across 3 sites (§10.6) | Small-n bias |
| D18 | Randomness protocol frozen | The evaluation randomness protocol is **not** improved during this revision. An improvement is a Category C change that would invalidate comparability with every existing result. (§13.5) | Verified 2026-10-21 |

### Open decisions (must be closed before the gate they affect)

| ID | Question | Blocks | Default if unanswered |
|---|---|---|---|
| **O1** | Development site for D1 tuning | G4 | site9 (see §6.4 rationale) |
| ~~O2~~ | ~~Does PPO-no-mask also receive a 3-config tuning pass?~~ | — | **CLOSED 2026-10-21 — see §13.7.** PPO ran at near-SB3-default settings and was never tuned. Tuning DQN/A2C while PPO stays at defaults therefore favours *them*, which is the safe direction. **No PPO-Tuned arm by default.** |
| **O3** | G5b fine-tune target: site5 only, or site5 + site4? | G5b | Both (cost is ~12 extra short runs) |

---

## 1. Policy Sanity Gate (PSG)

Implemented as `tools/policy_sanity_gate.py`. Run on **every** results CSV
before any number enters a thesis table, including the existing
`results/phase3/master_summary.csv`.

**Data-granularity rule (v1.8):** an *episode-summary* CSV contains EENS and
bounds aggregates but **not the per-step PV/grid/battery/DG allocations** needed
for HF-2, HF-2b or per-step overrides. The PSG must not claim these checks
passed using `master_summary.csv` alone. Apply all available summary-level
checks to that CSV; require linked per-step traces or the saved, hash-pinned
`g0_energy_audit.json` as separate evidence for step-level physics. Missing
step-level evidence is recorded as **NOT TESTED**, never a fabricated PASS.

Two severities, per Mentor 2's correction. My earlier draft wrongly encoded
behavioural expectations as invariants.

### 1.1 HARD FAIL — mathematical or physical invariant violated → run is invalid

| ID | Check |
|---|---|
| HF-1 | Energy balance: served + unserved = load, per step, within 1e-6 |
| HF-2 | **Load-side balance: `pv_to_load + grid_to_load + bat_discharge_to_load + dg_to_load + unmet = load`, within 1e-6.** See the correction note below |
| HF-2b | Episode EENS equals the sum of per-step `unmet` over the episode, within 1e-6 |
| HF-3 | SoC within `[soc_min, soc_max]` after every step |
| HF-4 | Inventory within `[0, tank_cap_kwh]` after every step |
| HF-5 | Executed action feasible under `TelecomEnv._apply_action_mask` semantics |
| HF-6 | Single-outstanding-order constraint never violated (`pending_flag` ≤ 1) |
| HF-7 | **Scientific configuration identity, conditional on the comparison's declared design.** Always required: `episode_len`, `init_low/high`, `train_scenario`, `tank_scale`, `lead_dist`, reward coefficients, dataset split. **Observation definition and action space are required only where the declared design holds them constant** — see the note below. Git commit is recorded as provenance (ECG-7), **not** hard-failed |

> **⚠️ Correction (v1.6) — this one would have broken the gate.** v1.5 specified
> HF-2 as a *source-side* identity, `PV + grid + DG + battery discharge = load +
> battery charge + curtailment`. **That equation cannot hold whenever load is
> not fully served** — which is exactly the Monsoon and Extreme behaviour this
> study exists to analyse. With demand 10 kWh and supply 9 kWh, the old HF-2
> would have hard-failed a physically valid high-EENS episode.
>
> The invariant is the **load-side** identity, which is what
> `TelecomEnv._energy_balance()` (`telecom_env.py:596–618`) actually constructs
> by successive subtraction: PV, then grid, then battery, then DG are each
> subtracted from the residual, and whatever remains is `unmet`. Served plus
> unmet equals load by construction, so HF-2 is a **regression check** on that
> construction rather than a physics test.
>
> A strict source-side balance **is** usable, but only in the net form derived
> and verified in §1.4 — a naive version double-counts, because the env's
> reported `p_grid_kwh` includes grid-to-battery charging.
>
> **2026-10-22 v1.8 clarification.** The incorrect HF-2 equation was a **validator
> specification error**, not a reason to discard historical EENS. A corrected audit
> now independently checks source allocations, curtailment, SoC, fuel and EENS (§1.4).
> Passing *accounting* identities confirms the implemented energy ledger; it **does
> not** validate external battery-parameter semantics (§1.6). Historical RL
> checkpoints need **no blanket retraining** for this equation correction.

### 1.2 WARNING — behavioural anomaly → mandatory investigation, does not auto-invalidate

| ID | Check | Trigger |
|---|---|---|
| WARN-FUEL-IDLE | High EENS with fuel available and no stockout | mean EENS > 10 kWh **and** `min_inv_pct` > 0.20 **and** `stockout_events` = 0 |
| WARN-UNIFORM-DG | Mechanism-like uniformity across sites | CV of `dg_on_fraction` across 10 sites < 0.05 while CV of EENS > 0.5 |
| WARN-SEVERITY-REVERSAL | Mean EENS not non-decreasing across normal → delayed → monsoon → extreme | any reversal (NOT a required property — finite-sample stochastic outcomes need not be monotonic) |
| WARN-DUPLICATE | Two policies numerically identical | > 50% of paired cells with \|ΔEENS\| < 1e-9 |
| WARN-DEGENERATE | Collapsed policy | any single action > 95% of evaluation steps |
| WARN-OVERRIDE | Action overridden by the environment | `dg_blocked` + `order_blocked` + `order_clipped` rate > 1% of steps; report the three separately, never only the total |

**Acceptance test for the PSG itself:** run it on the existing
`master_summary.csv`. It must raise WARN-FUEL-IDLE, WARN-UNIFORM-DG,
WARN-SEVERITY-REVERSAL and WARN-DUPLICATE on TrackB. **If it flags any other
policy, that policy is investigated before any gate proceeds.**

> **⚠️ Correction (v1.6).** v1.5 required identical observation and action
> spaces across *every* comparison. That would hard-fail the ablations by
> construction: **A5** removes five observation channels, **A6** uses
> `Discrete(2)`, and **TrackB** uses a 6-D observation *and* `Discrete(2)`.
> Those differences are the experiment, not a defect.
>
> Each comparison therefore declares which dimensions it holds constant, and
> HF-7 checks only those:
>
> | Comparison | Obs held constant? | Action space held constant? |
> |---|---|---|
> | RLInv vs A7 (masking) | ✅ yes — 11-D both | ✅ yes — `Discrete(6)` both |
> | RLInv vs DQN / A2C (G4) | ✅ yes — 11-D all | ✅ yes — `Discrete(6)` all |
> | RLInv vs A5 (observability) | ❌ **by design** | ✅ yes |
> | RLInv vs A6 / TrackB (decomposition) | ❌ by design (TrackB) | ❌ **by design** |
> | RLInv vs B1\*/MPC | ❌ n/a — not a learned policy | ❌ n/a |
>
> The declared design lives in the manifest (§2) as `comparison_class`, and the
> gate reads it. An *undeclared* mismatch is still a hard fail.

### 1.3 Required code change

`TelecomEnv` already returns `mask_info` with `dg_blocked`, `order_blocked` and
`order_clipped` per step (`telecom_env.py:544`), but `get_episode_stats()`
aggregates only `violation` (`telecom_env.py:753`). Note that
`violation = soc_violation OR tank_overflow` — it does **not** count
infeasible-action pressure. Add the three `mask_info` counters to the episode
stats so WARN-OVERRIDE has data.


### 1.4 Energy-accounting verification — corrected, independently reconstructed audit

**History of the finding.** v1.5 HF-2 incorrectly used total Load in an idealised
source-side equality and would falsely reject episodes with unmet load. v1.6
replaced it with a load-side identity. The uploaded v1.7
`g0_verify_energy.py`, however, was **not adequate evidence** for its claimed
passes: it defined `served = load − unmet` and then checked the resulting
identity (circular), did **not** implement the claimed 17,280-step sweep, and its
main-loop EENS check compared the final episode's EENS against zero.

**v1.8 replacement:** `tools/g0_verify_energy.py` (source artifact:
`g0_verify_energy_v1_8.py`) runs against the **historical, unchanged**
`TelecomEnv`. It reconstructs load allocation from separate reported source
quantities using the environment's PV → grid → battery → DG priority rule;
separates grid-to-load from grid-to-battery; reconstructs PV charging and
curtailment; and verifies EENS *after each episode*. The following are enforced
as actual `assert`/fail-closed checks (not display-only messages):

| Check | Independent audit assertion | Scope |
|---|---|---|
| Load-side HF-2 | `PV_to_load + Grid_to_load + Batt_discharge + DG + Unmet = Load` | Every step |
| Source-side & curtailment | `PV_available + Grid_total + DG − Batt_net − Curtail = Load − Unmet`; `0 ≤ Curtail ≤ PV_available` | Every step |
| HF-2b EENS | Episode `EENS_kWh` equals the sum of that episode's reported `unmet_kwh` | Every episode |
| Battery SoC bookkeeping | Recorded SoC matches the **implemented** coefficient convention, within tolerance | Every step |
| Diesel conservation | `fuel_used_kwh = p_dg_kwh` under standard 2.4-hour threshold | Every step |
| Inventory | `I_after = min(C_tank, I_before − fuel_used + delivery)` | Every step |
| Bounds and capacity | SoC, inventory, generator output, grid output and 360-step episode lengths | Every step / episode |
| Historical summary (if available) | No duplicate keys, all 360 steps, nonnegative EENS, stockout totals and per-policy seeds | Existing `master_summary.csv` |
| Train/test tank summary | Exact 45-day and 15-day mean-load tank capacities for all 10 sites | All site CSVs |

**Executed local diagnostic, 2026-10-22:** the replacement script was exercised
against the original `telecom_env.py`, `data_loader.py`, `hparams.yaml`,
all ten uploaded site CSVs and `master_summary.csv`. Because the local analysis
container lacks `gymnasium`, a **minimal Gymnasium base-class/spaces shim** was
used; the dispatch, reset, battery and inventory implementation itself was **not
modified**. The results were:

| Quantity | Observed local diagnostic |
|---|---:|
| Primary audit | 3 sites × 4 scenarios × 3 episodes = **12,960 steps** |
| Curtailment sweep | 4 sites (including site8) × 4 scenarios × 3 episodes = **17,280 steps** |
| Maximum independently reconstructed load/source residual | **1.776 × 10⁻¹⁵ kWh** each |
| Maximum SoC/inventory bookkeeping residual | **0** within machine precision |
| Fraction of curtailed steps, sweep | **16.46%** |
| Fraction of curtailed steps, site8 | **35.28%** |
| Historical master rows validated | **44,000**, with **0 stockout events** |

**Status:** local diagnostic **PASS**. **The official G0 energy-audit exit
criterion remains PENDING** until the updated script runs in the full research
repository with its installed Gymnasium dependencies and saves
`results/phase3r/g0_energy_audit.json` (and a run log) under the relevant commit.
Do not label the complete G0 gate as passed from this local diagnostic alone.

**Exact net accounting identity** with the original sign convention (`Bat_net > 0`
means charging):

`PV_available + Grid_total + DG − Bat_net − Curtail = Load − Unmet`.

Two non-obvious implementation details prevent naive checks: `p_grid_kwh`
**includes** grid-to-battery energy, and `p_dg_kwh` is the actual residual-limited
DG energy (not necessarily rated power whenever DG is ON).

**Scope boundary:** this verifies the internal energy ledger under the **historical
battery convention**. It does **not** certify that `eta_d=1.20` represents a
physically plausible efficiency, nor does it establish real-world EENS validity.
That issue is separately disclosed in §1.6 and §13.12.

### 1.5 Thesis Equation 3.1 must be corrected (documentation, not results)

The submitted thesis, page 28, states the power-balance constraint as:

> P<sub>t</sub><sup>Load</sup> = P<sub>t</sub><sup>PV</sup> − P<sub>t</sub><sup>Bat</sup> + P<sub>t</sub><sup>Grid</sup> + DG<sub>t</sub>·P<sub>rated</sub><sup>DG</sup>  (3.1)

It omits **three** things the implementation has: unserved load, curtailment,
and the fact that DG delivers `min(rated, residual)` rather than full rated
power. The implemented-energy-ledger replacement (validated in §1.4, **not**
a claim of physically calibrated battery conversion) is:

> **P<sub>t</sub><sup>Load</sup> − P<sub>t</sub><sup>Unserved</sup> = P<sub>t</sub><sup>PV,avail</sup> − P<sub>t</sub><sup>Curtail</sup> − P<sub>t</sub><sup>Bat</sup> + P<sub>t</sub><sup>Grid</sup> + P<sub>t</sub><sup>DG</sup>**, where P<sub>t</sub><sup>DG</sup> ≤ DG<sub>t</sub>·P<sub>rated</sub><sup>DG</sup>

**This is a documentation correction, not a model change.** The simulator
already behaves this way (§1.4); the thesis equation under-described it. The
correction is a **G6 revision**, highlighted like every other revised passage
(§11.5), and it strengthens the thesis — the stated model now matches the code,
which is the kind of thing a code viva probes.

**Classification under §13.6:** this is **not** a Category A/B/C experimental
change. Nothing in the environment is modified, so comparability with the
submitted results is untouched.



### 1.6 Historical-model audit — new facts discovered 2026-10-22

These findings were obtained by inspecting **actual historical code, original
ITU/Zindi CSVs, the per-episode Phase 3 master results, the evaluation script and
two site5 seed-42 checkpoints**, not merely the thesis narrative. They **do not
change the Phase 3-R registered research questions**. Treat a documented
implementation limitation differently from a result-affecting bug.

| ID | Finding | Evidence / how found | Interpretation / action |
|---|---|---|---|
| H-A | **No literal stockout during historical evaluation** | Sum of `stockout_events` in `master_summary.csv` is **0 in all 44,000 episodes** (11 policies × 4,000). `TelecomEnv` starts with positive inventory and requires 2.4 hours of DG energy as a start threshold, but can consume at most 1 hour per step. | Configured `γ_r=200` cliff did not trigger during **recorded evaluation**; do not attribute observed RL benefit to this term. `λ=100` unmet-load penalty *does* act. Normal training also has no obvious route to literal empty inventory under these settings, but evaluation records are not a training trace. |
| H-B | **Train/test tank capacity is not strictly fixed** | `TelecomEnv` recomputes `d_bar = mean(site_data.load_kwh)` and `C=72*d_bar` inside the constructor. Trainer passes first 45 days, evaluator last 15; recomputed from all 10 CSVs. | Test-time tank and 30/60% order amounts differ from training; max absolute difference **6.93% (site8)**; Hard site2 **−0.11%**, site5 **+0.29%**. Preserve existing convention for directly matched comparisons, measure/disclose it; do not state physical plant capacity is strictly fixed across the split. `d_bar` also changes reward normalisation. |
| H-C | **Battery coefficient provenance is confirmed, but semantics differ** | Raw `site information and grid outage.csv` provides `battery_charge_coefficient=0.35` and `battery_discharge_coefficient=1.20`; `preprocess_data.py` copies them unchanged. `TelecomEnv` uses `ΔSoC=(0.35·Echarge − Edischarge/1.20)/C_bat`. Public 2024 challenge scoring code uses `ΔSoC=(0.35·Echarge − 1.20·Edischarge)/C_bat` in its *different, quarter-hour* scoring formulation. | A material **battery-model interpretation limitation**. The historical model understates depletion relative to the source scoring convention for the same discharge, although that scoring formulation is **not a drop-in substitute** for the hourly inventory simulator. Preserve historical dynamics in G3/G4/G5; state exactly what was implemented, avoid claiming physically validated battery efficiency; optional alternative-model sensitivity is **separate**. |
| H-D | **2-hour DG rule is a fuel-start reserve, not a 2-step ON latch** | `TelecomEnv` masks DG start when inventory < `P_rated×2×1.2`; step logic does not force DG ON for the next hour. | Revise thesis Chapter 3 §3.7 and Chapter 4 description. Do **not** add a new commitment latch to historical G3/G4/G5. |
| H-E | **Energy-balance equation incomplete in submitted thesis** | Submitted Eq. 3.1 omits unmet, curtailment and actual DG output. Actual `_energy_balance()` subtracts sources sequentially and tracks residual unmet. | Fix Eq. 3.1 in revised thesis as §1.5; **not** a historical simulator change. |
| H-F | **Policy parameter count in submitted table incorrect** | Inspected original `site5_s42_final.zip` checkpoints: RLInv (MaskablePPO) and A7 (PPO) each have **139,527 total policy parameters** with **separate** actor and critic MLPs. Old table assumed one shared trunk and said 70,663. | Correct §7.4b, §8.3 and thesis Table 4.1. **139,527 is total training architecture**; inference uses actor-side parameters only. DQN is expected to have 70,406 online Q-network parameters; new A2C counts must be measured when trained. |
| H-G | **Selected historical checkpoint replay matches energy metrics** | Prior site5/seed42 RLInv and A7 replay on all 4 scenarios × 10 episodes matched stored EENS, diesel and grid metrics (**80/80 episodes**), using saved VecNormalize artefacts and reconstructed deterministic inference. Some minor inventory/order fields were not exact. | Evidence supports **selected primary-metric reproduction**, *not* all historical runs or bit-for-bit reproduction. Preserve the per-episode replay log in the repository and re-run through the official evaluation loader at G0 (§13.9). Retraining 30 PPO jobs for learning curves is not a substitute for the original-checkpoint replay. |
| H-H | **Scenario set is correct in historical core results** | `run_eval_grid.ps1` explicitly sets `normal, delayed, monsoon, extreme`; all 11 policy families have those labels in `master_summary.csv`, with 360 steps per episode. | YAML's generic `scenarios_to_eval` list is not proof of a wrong historical scenario run. The additional three MPC families need their own original batch-script provenance if claiming fully verified execution commands. |
| H-I | **MPC-Forecast does not use the usual ten seeds** | Compare `seed` sets in `master_summary.csv`. MPC-Forecast: 1–7, 42, 123, 777. Usual set: 42, 123, 777, 7, 13, 21, 99, 314, 500, 999. **4 seeds overlap: 7, 42, 123, 777**. | Do not report a full ten-seed *paired* MPC-Forecast comparison against RLInv from these data; use a matched 4-seed subset if pairing is essential or report unpaired/descriptive results. |
| H-J | **Preprocessor comments overstate grid randomisation** | `preprocess_data.py` documents an envisioned independent 168-hour phase randomisation and outage stress injection. Core `TelecomEnv` instead uses fixed grid traces with random training-window starts; normal evaluation has deterministic test trace. Four named primary scenarios vary fuel-delivery delays. | Describe the **actual** grid procedure, not unrealised preprocessing comments. Do not change the grid random process in primary comparisons. |

**Raw-data check:** the original load/solar/site-information CSVs and the ten
processed `site*.csv` files were reconciled, with no missing critical fields,
negative source/load values, duplicate time rows, or mismatch in their shared
processed fields. The source's lack of *diesel-delivery* observations remains an
explicit synthetic-overlay limitation, not a defect.

**External challenge-semantic evidence:**
- ITU/Zindi 2024 challenge staff discussion and scoring-code correction:
  https://zindi.world/competitions/smart-energy-supply-scheduling-for-green-telecom/discussions/22056
- Staff confirmation of scoring-function update:
  https://zindi.world/competitions/smart-energy-supply-scheduling-for-green-telecom/discussions/22167
- Published copy of corrected source scoring code (independent repository copy,
  **not** the project simulator):
  https://github.com/ggbaguidi/smart-uti/blob/main/errorMetric_ex.py
  (`calcSingleSiteScore`: when net source energy is negative, it multiplies
  by `disCoe`; `TelecomEnv` divides by the same 1.20 coefficient.)

### 1.7 G0 disposition — what passes and what remains pending

| Item | Disposition at this audit | Is original PPO retraining required? |
|---|---|---|
| Wrong original HF-2 and old energy-audit script | **Corrected specification and replacement code**; local accounting test PASS, official repo-run evidence pending | **No** |
| DG one-step fuel conservation at configured threshold | Code proof and local runtime verification pass | **No** |
| Original 44,000 episode results and 0 literal stockouts | Historical CSV verified; stockout term inactive in evaluation | **No**; correct reward interpretation |
| Tank split difference | Quantified 10/10 sites; preserve convention for all direct comparisons | **No blanket retrain**; potential sensitivity is separate |
| Battery-source discharge-factor discrepancy | **Confirmed model-interpretation difference**; do not claim physical validation | **No blanket retrain** if primary historical benchmark retained unchanged |
| RLInv/A7 checkpoint replay | Site5 seed42 primary energy metrics matched; full official reproducibility log pending | **No**; use original saved checkpoints |
| Physical accuracy under alternative battery convention | **Not established** by historical experiment | Requires separately labelled sensitivity or retraining **only if** making new corrected-physics claims |
| Overall PSG + ECG Gate G0 | **NOT YET COMPLETE**; requires §3 deliverables and official run logs | — |

**Reporting rule.** An internal bookkeeping PASS and an empirically reproducible
benchmark do not imply physically calibrated field performance. A battery
sensitivity run on **alternative physics** has a different comparator label
and must never be merged with historical EENS or used to imply an isolated
algorithm effect without matching every model's evaluation/training condition.

### 1.8 Mandatory thesis and viva correction list (G6, not new experiments)

1. **§3.2 Eq. 3.1:** replace incomplete balance with §1.5, explaining
   residual-limited DG output and unmet demand.
2. **§3.6 Eq. 3.4 / Table 3.1:** retain general conventional-efficiency notation
   *as a formal model* only if Chapter 4 explicitly gives the **actual historical
   update** using `eta_d=1.20` in the denominator. Name and explain the mismatch
   with source scoring's *discharge consumption multiplier*. Do **not** rewrite
   Eq. 3.4 to suggest the historical simulations used source-consistent dynamics.
3. **§3.7 and Chapter 4 §4.3:** replace the claimed minimum 2-hour continuous
   **runtime** with the actual minimum **start fuel reserve** (2.4 rated hours).
4. **§4.4 reward:** the `γ_r=200` stockout penalty is defined but did not fire in
   44,000 historical evaluation episodes. EENS/unmet-load penalty is distinct.
5. **§4.3 / §4.7:** disclose train/test `C_tank` and `d_bar` recalculation; call
   72-hour sizing a *slice-dependent modelling convention*, not an externally
   measured fixed tank capacity.
6. **§4.2/Table 4.1, §7.7, latency section:** correct RLInv and A7 total
   policy parameters to **139,527**; report actor-only inference size separately.
7. **§4.7 implementation-deviations table and §8.4 limitations:** add battery
   coefficient and DG commitment distinctions and emphasise conditional
   simulation conclusions rather than field-validated performance.
8. **Methods/results:** distinguish 11 historical evaluated families from the 9
   main-thesis families; qualify MPC-Forecast pairing; correct any claim that the
   primary logistics stress scenarios inject new grid-outage traces.
9. **G0 evidence appendix / rebuttal (if relevant):** provide a short factual
   change/provenance record and official saved G0 audit JSON, not a claim of
   physical-model correction or a new examiner-requested experiment.

**All G6 thesis passages changed after submission must be highlighted (§11.5).**
The examiner's explicitly requested breadth and scalability questions continue
to be answered by G3/G4/G5, **not** by turning this integrity audit into a new
research workstream.

---

## 2. Reproducibility manifest

Written automatically by every training and evaluation run to
`results/phase3r/manifest/<experiment_id>.json`. Every result row carries its
`experiment_id`.

```
experiment_id, git_commit, git_dirty_flag,
policy, algorithm, masking_method, feasibility_handling,
site, seed, training_steps, training_scenario,
reward_parameters, tank_capacity, tank_capacity_train_reference,
tank_capacity_test_actual, d_bar_train, d_bar_test,
battery_charge_coefficient, battery_discharge_coefficient, battery_update_convention,
lead_time_model, physics_fingerprint,
observation_definition, observation_dim, normalisation_file,
action_space, episode_len_train, episode_len_eval,
train_data_dates, test_data_dates,
wall_clock_min, n_envs, cpu_model, logical_cores, physical_cores, ram_gb,
python_version, sb3_version, sb3_contrib_version, numpy_version
```

Core count and CPU model are collected programmatically (closes D11).

---

## 3. G0 — Protocol, PSG, manifest, timing

**Deliverables**

1. This document, committed.
2. `tools/policy_sanity_gate.py` passing its §1.2 acceptance test.
3. `tools/manifest.py` writing §2 for every run.
4. `mask_info` counters added to `get_episode_stats()` (§1.3).
5. **ECG implementation (§13)** — `tools/comparability_gate.py`, the frozen
   reference-configuration file, and the reference-reproduction pilot (§13.8).
   **This is a G0 exit criterion, equal in standing to the PSG.**
6. **MLflow timing fix.** Move `mlflow.start_run()` to wrap `model.learn()` in
   `train_ablation_a5.py` (learn at 217, run at 246), `train_ablation_a6.py`
   (245 / 259), `train_ablation_a7.py` (181 / 194), `train_track_b.py`
   (232 / 250). `train_rl_inv.py` is already correct (178 / 289) and is the
   reference pattern.
7. `results/phase3r/training_cost.csv` from `harvest_wall_time.py`.
8. **Corrected historical energy audit:** `tools/g0_verify_energy.py` producing
   `results/phase3r/g0_energy_audit.json`, plus execution log and file hashes.
9. Historical results audit and disclosures as in **§1.6–1.8** (zero stockouts,
   slice-dependent tank, source-vs-implemented battery coefficient, no 2-hour DG
   ON latch, 11-policy provenance, MPC-Forecast pairing).
10. **Evaluator fail-closed checks:** RL `VecNormalize` must exist and match
    observation dimension; missing episode EENS or unexpected steps abort a run.

**Timing baseline already established** (from `mlflow.db`, logged
`wall_time_min` metric, 2026-10-04):

| Budget | n | min | mean | max |
|---|---|---|---|---|
| 400,000 steps | 240 | 8.2 | **11.2 min** | 85.2 |
| 500,000 steps | 100 | 12.6 | **14.2 min** | 17.8 |
| `phase3_normal` (the 10×10 RLInv campaign) | 100 | 9.5 | **10.8 min** | 14.9 |

Linearity check: 10.8 min at 400k predicts 13.5 min at 500k; observed 14.2.
Throughput ≈ 600 env-steps/s. **Unit cost for all planning below: 11 min per
400k PPO run.** The 85.2 min outlier sits in `phase3_gamma` and is to be
identified and explained in G0, not averaged in.

No timing exists for A5/A6/A7/TrackB (defect 4 above) — measured fresh in G5a.

### 1.9 Repository G0 energy audit — EXECUTED AND PASSED (v1.9)

v1.8 ran the audit under a Gymnasium **compatibility shim** and recorded the
official repository run as pending. **That run has now been executed** with real
Gymnasium 1.4.0 against the unmodified repository, using
`tools/g0_verify_energy.py` **v1.8.2**.

```
PASS: primary 12,960 steps; curtailment sweep 17,280 steps
Load/source/SoC/inventory max errors: 1.776e-15 / 1.776e-15 / 0.000e+00 / 0.000e+00
Curtailment in sweep: 16.46%; site8: 35.28%
Historical rows: 44,000, stockouts: 0
soc_clip_events: 0 | tank_overflow_events: 0
```

| Evidence | SHA-256 (first 16) |
|---|---|
| `results/phase3r/g0_energy_audit.json` | `21a2f1fb853117ed` |
| `src/env/telecom_env.py` | `e8052834f1e193e4` |
| `results/phase3/master_summary.csv` | `76a28c356893abcd` |

The repository figures match the shim run to two decimal places, which
**validates the shim** as well as the result.

**Script lineage:** v1.8 (reconstructed, fail-closed) → **v1.8.1** SoC ledger
compared against the **clipped** SoC with `violation_soc` asserted separately
(the v1.8 form compared against the unclipped value and would have hard-failed
any legitimate clipping step — the same false-rejection class as the original
HF-2 defect) → **v1.8.2** `violation_tank` and the combined `violation` flag
validated the same way. Clipping and overflow events are **counted, not
auto-accepted**: a non-zero count is an investigation trigger even when the
simulator handled it correctly.

#### ECG-1 dataset provenance — RESOLVED

The ten site-CSV SHA-256 hashes in the audit JSON differ from the copies held
outside the repository. Row-by-row numerical comparison of all ten:

| | Result |
|---|---|
| Shape | (1440, 23) in both, all ten sites |
| Columns | identical, all ten |
| **Max absolute difference, every numeric column, every site** | **exactly 0.0** |

Cause identified: `site5.csv` is 169,063 B in the repository and 170,504 B
outside — a difference of 1,441 bytes, with **1,441 CR characters** in the
external copy and **0** in the repository. 1,441 = 1,440 rows + 1 header, i.e.
exactly one byte per line. **CRLF vs LF, introduced in transfer.**

> **ECG-1 criterion (frozen):** dataset identity is established by **row-wise
> numerical comparison after newline normalisation**. Raw-byte SHA-256 is
> recorded for traceability but is **not** the identity criterion, because
> CRLF/LF conversion changes the digest without changing the data.

#### Still open in G0 after this

| Item | Status |
|---|---|
| Energy-ledger audit | ✅ **PASS**, repository-run |
| ECG-1 dataset identity | ✅ **RESOLVED** |
| **Native SB3 checkpoint replay** | ⬜ The 80/80 replay reconstructs the forward pass rather than calling `model.predict()`. A native SB3 replay remains required as final evidence |
| **`orders_placed` discrepancy** | ⬜ The replay matches EENS, diesel and grid exactly (`worst_diff` 0.0) but shows `orders_placed` differing by **1** (rlinv, delayed) and `mean_inv_pct` by up to 0.023. Likely benign — an order action issued where it was masked to zero has no energy consequence but a different count — **but an action-pathway discrepancy is the exact class of defect TrackB turned out to be, so it must be explained from the trace, not waved through** |
| PSG implementation + acceptance test | ⬜ |
| ECG implementation + frozen reference config | ⬜ |
| MLflow `start_run()` fix (4 trainers) | ⬜ |
| MPC/Oracle-MPC timing pilot (gates G3 scope) | ⬜ |

### 1.10 Battery interpretation — claim discipline (v1.9)

Two formulations were proposed during review and **both are rejected**:

| Rejected claim | Why |
|---|---|
| *"0.35 and 1.20 are C-rates"* | Plausible but **not established**. The published ITU/Zindi scoring code uses them directly as SoC multipliers (`coe`, `disCoe`). Describe them as **source scoring coefficients**, not proven C-rates and not conventional efficiencies. Do not put an unestablished interpretation in the thesis. |
| *"the battery model is identical for all policies, so rankings necessarily hold"* | **Logically invalid.** A uniform *rule* does not produce a uniform *effect*: policies charge and discharge at different times, so altered battery dynamics can affect them unequally. An exploratory site5 counterfactual showed a slight RLInv–A7 reversal under Extreme — a small diagnostic, not a robust finding, but sufficient to refute the guarantee. |

**Approved wording, to be used verbatim:**

> The historical Phase 3 results are reproducible under the implemented battery
> convention. The discrepancy with the source scoring convention limits physical
> interpretation and may affect both absolute EENS and relative policy
> performance under alternative battery dynamics. The historical primary
> benchmark will remain unchanged for Phase 3-R comparability; any
> battery-physics sensitivity analysis will be reported separately.

Direction, for accuracy: for the same 12 kWh discharge the implementation
deducts 12 ÷ 1.20 = **10 kWh** of stored energy while the source convention
deducts 12 × 1.20 = **14.4 kWh**. The implementation is therefore **more
optimistic** about discharge availability than the source convention.

**Exit criterion:** PSG passes its acceptance test; **ECG original-checkpoint
reproduction pilot passes (§13.9)**; manifest writes; timing table exists;
**updated `tools/g0_verify_energy.py` passes in the full repository and saves
the JSON evidence and console log**. Local diagnostic evidence (§1.4) is not a
substitute for the official repository G0 execution. In particular, fail if an
RL checkpoint has missing/mismatched `VecNormalize` statistics or an episode is
missing EENS rather than silently substituting defaults.

---

## 4. G1 — TrackB diagnostic

**No retraining. Replay of existing checkpoints only.**

| Item | Value |
|---|---|
| Sites | site1 (Easy), site3 (Medium), site5 (Hard) |
| Seeds | 42, 123 |
| Scenarios | normal, delayed, monsoon, extreme (all four) |
| Episodes | 10 per cell |

**Instrumentation** — log per step to `results/phase3r/g1_trackb_trace.csv`:

```
step, dg_requested, order_requested, combined_index,
mask6 (6 booleans), fallback_fired, 
dg_executed, order_executed,
inv_kwh, pending_flag, min_fuel_kwh, tank_headroom_kwh
```

**Hypothesis under test.** `DispatchOnlyEnv.step()`
(`src/wrappers/drop_inventory_obs.py:53–59`) fuses the dispatch and ordering
decisions into `combined = dg*3 + order`, and on infeasibility sets
`combined = 0` — **DG off *and* no order**. An infeasible *ordering* component
therefore destroys a feasible *dispatch* decision. This path exists in no other
policy; RLInv, A5, A6 and A7 act on the native `Discrete(6)` space or a clean
`Discrete(2)`.

**Two separate questions, two separate rules (restructured in v1.6).** v1.5
conflated them, requiring both conditions to confirm the defect — which would
have let a genuine defect escape simply by not matching the expected scenario
pattern.

**Question 1 — does the implementation defect exist?** Answered directly from
the requested-vs-executed traces, with no scenario thresholds:

> The **harmful component override** is confirmed by a step where
> `fallback_fired` AND **an actually requested, individually feasible nonzero
> component was cancelled**:
> (a) `dg_requested == 1` and `mask6[3] == True`, but `dg_executed == 0`;
> or (b) `order_requested > 0` and `mask6[order_requested] == True`,
> but `order_executed == 0`. Record both separately. Merely finding a
> feasible **DG-OFF** action is **not** evidence that a requested ON decision
> was destroyed. Any confirmed occurrence establishes the wrapper defect;
> its **causal contribution to EENS** is separately assessed from traces.

Report the rate, and its split by which component survived.

**Question 2 — does the defect explain the EENS collapse?** This is where the
scenario pattern is used, and it is now diagnostic rather than definitional:

- fallback firing rate by scenario — expected low under normal/delayed, high
  under monsoon/extreme;
- `dg_requested` rate approximately scenario-invariant while `dg_executed`
  collapses under monsoon/extreme.

The second is the informative one: it separates *the policy decided to run the
generator and was overruled* from *the policy stopped asking*.

**The two answers are reported separately.** A confirmed defect with weak
explanatory power still requires the fix; it just means something else also
contributes, and that something must be found.

**If Question 1 returns a zero rate** (no defect), investigate in this order,
1 day budget:
(i) VecNormalize `.pkl` dimension and whether stats actually loaded for those
runs; (ii) action entropy under monsoon (WARN-DEGENERATE); (iii) whether the
6-D observation can differ between delayed and monsoon at *t*=0 — it should not,
which is itself diagnostic. If still unresolved, **TrackB is removed from the
primary table** and the episode is written up as an implementation limitation.
Per Mentor 2: *do not rewrite the wrapper merely to improve TrackB's numbers.*

**Questions to ask of the output**
- Is the fallback rate bimodal by scenario or continuous?
- When it fires, which component was infeasible — DG, order, or both?
- Is there an absorbing state, or does it fire intermittently?

**Thesis sections affected either way:** Table 7.1, Table 7.2, Figure 7.1,
Figure 7.4, §7.2 narrative, §7.3.2, slides 8 and 10.

---

## 5. G2 — TrackB remediation (conditional on G1)

### 5.1 Three preconditions before any retraining

1. G1 confirms a real implementation issue per the §4 rule.
2. The corrected action-handling truth table is **unit-tested**.
3. A short corrected pilot (1 site, 1 seed, 50k steps) produces physically
   coherent action execution — *regardless of whether EENS improves*.

### 5.2 The truth table is not invented — it is read off the environment

**Finding (2026-10-21).** `TelecomEnv._apply_action_mask()`
(`telecom_env.py:544–564`) already implements exactly the semantics TrackB's
wrapper fails to implement. It handles the two components **independently**:

| Condition | Env behaviour | Flag set |
|---|---|---|
| DG requested on, `inv_kwh < min_fuel_kwh` | `dg_on → False`; **order untouched** | `dg_blocked` |
| Order requested, `pending_flag == 1` | `order_qty → 0`; **DG untouched** | `order_blocked` |
| Order requested, headroom < requested | order clipped to available headroom; **DG untouched** | `order_clipped` |
| Both feasible | executed as requested | — |

This is the reference semantics. The repair makes `DispatchOnlyEnv` defer to it
rather than pre-empt it with `combined = 0`. **This satisfies Mentor 2's
requirement that intended semantics be established before the repair — with
evidence from the existing codebase rather than a newly invented fallback.**

Unit tests: one per row above, plus the both-infeasible case, asserting the
executed `(dg, order)` pair and the flags.

### 5.3 Correction A compliance (Mentor 2)

The TrackB repair touches **`src/wrappers/drop_inventory_obs.py` only.**
`TelecomEnv` is **not** modified. This is a hard constraint: changing the
environment would invalidate the existing PPO/A7 runs that G4 reuses and make
the algorithm benchmark incomparable to the submitted campaign. Any proposed
change to `telecom_env.py` during G2 stops the gate and returns here.

### 5.4 Retrain

`DispatchOnlyEnv` is used at **training** time as well
(`train_track_b.py:119`), so corrected TrackB must be retrained — re-evaluating
old checkpoints is not valid.

| Item | Value |
|---|---|
| Scope | 10 sites × 10 seeds = 100 runs |
| Budget | 400k steps |
| Estimated cost | ~18 h serial at 11 min/run |
| Evaluation | 400 cells (10 sites × 10 seeds × 4 scenarios), ~2 h |

**Reporting rule.** Corrected TrackB is a **new row**, reported alongside the
original. The original numbers are not deleted. Thesis wording, per Mentor 2:

> During the revision-stage integrity audit, an action-handling issue was
> identified in the original TrackB wrapper. The affected comparator was
> corrected and re-evaluated; conclusions are based on the corrected
> implementation.

**Prohibited inference.** Corrected TrackB must **not** be used to claim masking
superiority. A6 and TrackB differ in observation design (11-D vs 6-D) as the
submitted thesis already states. RLInv vs A7 remains the clean masking
comparison.

**Removed from this gate.** My earlier draft asked "does corrected TrackB now
sit between B1 and RLInv, as theory predicts?" That is struck. Using a desired
ordering as a validation criterion is the reasoning error that lets a broken
comparator survive. The replacement question is: *does corrected TrackB exhibit
physically coherent action execution, and how does its EENS compare with B1, A6
and RLInv?*

### 5.5 Pre-committed position

H1 rests on A6 and H2 on A7, as the submitted thesis states explicitly.
A corrected or removed TrackB **narrows the evidence but overturns neither
hypothesis.** This sentence is written before the result arrives.

---

## 6. G3 — Strengthened classical baseline

### 6.1 Two baselines, not one

| Label | Tuning | Role |
|---|---|---|
| **B1\*-Deployable** | `(s,S)` tuned on the training window under the Normal scenario, frozen, evaluated unchanged on all four test scenarios | **Primary** strong classical comparator |
| **B1-OracleTune** | `(s,S)` tuned separately per logistics regime | Optimistic **regime-aware reference**, explicitly not deployable (conditional on time — see O3 class) |
| B1 (original) | Analytic `k_s = 1`, `S = 0.85` | Retained for traceability |

### 6.2 The symmetry argument (write this into the thesis)

> RLInv is trained on days 1–45 under the Normal lead-time scenario, then
> evaluated zero-shot on days 46–60 under all four scenarios. B1\*-Deployable is
> calibrated on days 1–45 under Normal, frozen, then evaluated on days 46–60
> under all four. **Both receive the same data access and the same selection
> discipline: the same chronological training window, the same calibration
> scenario, and no sight of test data during selection.**

⚠️ **The claim is about data access, not about representational parity
(v1.6).** They do **not** observe the same state at decision time — B1\* acts on
inventory level and the pending flag, while RLInv observes the full 11-D state —
and they do not have comparable model capacity. The earlier wording ("exactly
the same information") overstated this and is withdrawn.

The narrower claim is the one that answers "you tuned the baseline unfairly",
and it is the one that is true. It is enforced in the tuning script, not left to
discipline.

### 6.3 Grid and procedure

| Item | Value |
|---|---|
| Parameters | **Coarse-to-fine.** Pass 1: step 0.10 → **45 feasible pairs**. Pass 2: refine ±0.05 around the pass-1 optimum → ~9 more. Total ~54 per site |
| **Not** tuned initially | `q_small_pct`, `q_large_pct` — held at 0.30 / 0.60 (Mentor 2: change one thing at a time) |
| Data | **Training window only** (days 1–45). Test EENS never observed during selection. |
| Objective | Mean training-window EENS (D9) |
| Tie-break | Among candidates within 1% of the best EENS, choose the lower mean diesel; then the lower order count |
| Episodes per candidate | 10, fixed seeds |
| Scope | Per site |

**⚠️ Cost correction (v1.4).** The earlier "~1 h" estimate for G3 was wrong and
is withdrawn. Computed:

| Grid | Pairs | Episodes/site | **Total episodes** |
|---|---|---|---|
| Fine (step 0.05, as originally written) | 153 | 1,530 | **15,300** |
| **Coarse-to-fine (adopted)** | ~54 | ~540 | **~5,400** |

For scale: the *entire* submitted campaign was 44,000 episodes across 11
policies. A fine grid would have been 35% of that for one baseline sweep.

**The larger cost is the propagation, not the sweep.** Re-running MPC and
Oracle-MPC on B1\* means **400 cells × 10 episodes × 360 steps ≈ 1.44 M MILP
solves each**, and Oracle-MPC plans over H=360 rather than H=24.

**Mandatory pilot before committing G3:** time **one** MPC cell (10 episodes =
3,600 solves) and one Oracle-MPC cell. Then choose:

| If full propagation costs | Action |
|---|---|
| < ~4 h | Re-run both, as §6.3 propagation specifies |
| 4–12 h | Re-run **MPC only**; Oracle-MPC retains the analytic rule, disclosed |
| > 12 h | Re-run both on the three G4 sites only; disclose reduced scope |

Oracle-MPC is already described in the thesis as *"not a deployable policy… an
optimistic reference"*, so leaving it on the analytic calibration with an
explicit note is defensible if time-constrained. MPC is a genuine competitor and
should be re-run if at all affordable.

**Propagation.** B1, MPC, Oracle-MPC, TrackB and A6 all share the same `(s,S)`
rule (`train_track_b.py:87`, Table 4.4). Therefore:

- MPC and Oracle-MPC are evaluation-only → **re-run them with B1\* to the
  extent the §6.3 pilot shows affordable**, following the three-way decision
  rule there (both / MPC only / reduced site set). Leaving them on the analytic
  rule while B1\* is tuned would be unfair in the opposite direction, so
  whichever scope is chosen is **disclosed in the Baseline Taxonomy table
  (§6.4)** rather than left implicit.
- A6 and TrackB would require retraining to adopt it → **defer, and document
  explicitly** that they retain the analytic calibration.

### 6.4 Baseline Taxonomy and Fairness table (D12 — mandatory)

Added to the methodology chapter. Near-zero compute; it is the single most
efficient answer to *"missing proper baseline method"*, because it makes the
fairness argument auditable at a glance instead of scattered across prose.

One row per comparator, with columns:

| Comparator | Ordering mechanism | Dispatch mechanism | Information available | Training / tuning budget | Feasibility handling | Purpose in the evaluation |
|---|---|---|---|---|---|---|

Rows: B0, B1, **B1\*-Deployable**, B1-OracleTune (if run), MPC, Oracle-MPC,
TrackB (original + corrected), A5, A6, A7, **DQN**, **A2C**, RLInv.

The *training / tuning budget* and *feasibility handling* columns are the ones
that answer the fairness challenge, and they are the two the submitted Table 4.4
does not have.

**Reporting rule (Mentor 2).** Do not present B1\* as proof that RLInv is
superior. **If B1\* beats RLInv in some scenario, report it.** The purpose of
this gate is to demonstrate a credible, fairly constructed baseline — not to win
against one.

### 6.5 Pre-committed position

If B1\* substantially closes the 73% / 51% / 7% gaps, the contribution narrows
to *RL adds value specifically in the long-lead-time regime* — which is already
directionally what the H1 result says, and is a more defensible thesis.
**Table 7.1 and the percentage claims are revised in place. No contradictory
appendix.**

---

## 7. G4 — RL algorithm benchmark

### 7.1 Site selection (D8)

Rule, stated before looking at any result: *from `site_classification.csv`,
take one site per independently assigned difficulty class; within each class
choose the site whose profile is closest to the class median, excluding the
single `is_surplus` site as unrepresentative.*

| Class | Candidates | **Selected** | Why |
|---|---|---|---|
| Easy (score 1) | site1, site6, site8 | **site1** | site8 excluded (`is_surplus`, 151.6% solar coverage); site1 and site6 are near-identical (outage 20.0% both) |
| Medium (score 2) | site3, site4, site7, site9, site10 | **site3** | Closest to class median on all three axes (outage 32.8%, solar 56.0%, net load 2.58) |
| Hard (score 4) | site2, site5 | **site5** | Lowest battery autonomy relative to longest outage; highest outage event count (52 in 60 days) |

**Selection provenance:** `site_classification.csv` was computed from
60-day descriptive site characteristics, including the test window. The rule
was frozen before Phase 3-R algorithm results and uses no new policy outcome,
but it is **not** a train-window-only site-selection procedure. Do not assert
otherwise in the thesis or ECG description.

This is a **new** combination — not the convergence-check subset (1, 5, 7) nor
the sensitivity subset (2, 5, 7), and deliberately not the results-derived
"hard sites" set. PPO cells for site1/site3/site5 already exist in the main
10×10 campaign.

### 7.2 Seeds

Five, by a stated rule: **the first five in the order recorded in Table 6.2,
skipping any seed whose evaluation-episode block `[seed, seed+9]` would overlap
a block already selected.**

| Seed | Episode-seed block | |
|---|---|---|
| 42 | 42–51 | ✅ |
| 123 | 123–132 | ✅ |
| 777 | 777–786 | ✅ |
| 7 | 7–16 | ✅ |
| ~~13~~ | ~~13–22~~ | ❌ overlaps 7–16 — **skipped** |
| 21 | 21–30 | ✅ |

**Selected: `42, 123, 777, 7, 21`.**

**Why the skip clause exists (v1.6).** `evaluate.py` seeds each episode with
`seed + ep`. Seeds 7 and 13 therefore share episode-seeds 13, 14, 15, 16, so the
v1.5 selection would have produced 50 evaluation instances carrying only **46
distinct** evaluation conditions. This matters most for the deterministic
baselines, where the seed indexes evaluation randomness alone (§10.4). The skip
costs nothing — all ten seeds exist in the historical campaign, so any
five-subset remains matched-subset comparable (§13.8).

3 seeds are screening only (D10); 5 is the planned minimum for the final table.

### 7.3 Feasibility handling (D3) — no new code required

**Finding.** `TelecomEnv` applies component-wise feasibility handling inside
`step()` for *every* policy (§5.2). An unmasked algorithm acting on the native
`Discrete(6)` space therefore already receives identical, environment-level
handling. This is exactly what A7 does (`train_ablation_a7.py:84` — no
`ActionMasker`; the env still enforces physics).

So: **PPO-no-mask, DQN and A2C share identical feasibility semantics by
construction.** No handler is written, and — critically — **no TrackB wrapper
logic enters this gate** (Correction A).

| Arm | Masking | Feasibility |
|---|---|---|
| MaskablePPO (RLInv) | Native, logits → −∞ pre-softmax | Env component-wise |
| PPO-no-mask (A7) | None | Env component-wise |
| DQN | None | Env component-wise |
| A2C | None | Env component-wise |

MaskablePPO retains its native masking. Reported honestly as *the complete
proposed controller against alternative implementations*, not a perfectly
isolated algorithm-only comparison. The A7 bridge is what separates the masking
effect from the learning-algorithm effect.

### 7.4 A7 reuse — Mentor 2's critical checkpoint, verified

Audited `train_ablation_a7.py` against `train_rl_inv.py` on 2026-10-21:

| Dimension | Match? |
|---|---|
| Environment (`TelecomEnv`, same constructor args) | ✅ |
| Observation (11-D, no wrapper) | ✅ |
| Reward (base `TelecomEnv` reward, no Lagrangian in either) | ✅ |
| Episode lengths (`env_cfg` train 720 / eval 360) | ✅ |
| VecNormalize (`norm_obs=True, norm_reward=False`, same `obs_clip`) | ✅ |
| Hyperparameters (both from `ppo_cfg`) | ✅ |
| Invalid-action semantics (`_apply_action_mask`) | ✅ |
| Training budget (400k) | ✅ |
| **Differences** | PPO vs MaskablePPO; no `ActionMasker`; standard `EvalCallback` instead of `MaskableEvalCallback` |

**Verdict: A7 is reusable as the bridge arm.** The differences are precisely the
bridge's definition.

**A7 reuse is unconditional** (O2 closed, §13.7): PPO receives no tuning pass, so
A7-Original stands as the unmasked-PPO bridge arm at its submitted
configuration. Should a PPO-Tuned arm ever be introduced, it would be a
**separate labelled arm** run fresh — never a replacement for A7-Original.

### 7.4b Network architecture and checkpoint selection (frozen)

Without these frozen, an examiner can fairly ask whether a difference came from
the algorithm or from an uncontrolled implementation choice.

**Architecture — identical where the algorithm supports it:**

| Arm | Hidden layers | Heads | Trainable parameters |
|---|---|---|---|
| MaskablePPO (RLInv) | **Separate** actor and critic [256,256] ReLU MLPs | actor 256→6 (masked); critic 256→1 | **139,527 — measured checkpoint** |
| PPO-no-mask (A7) | **Separate** actor and critic [256,256] ReLU MLPs | actor 256→6; critic 256→1 | **139,527 — measured checkpoint** |
| A2C | Separate actor and critic [256,256] ReLU MLPs, if using the planned SB3 defaults | actor + critic | **139,527 expected; measure after training** |
| DQN | One online Q-network [256,256] ReLU | Q head 256→6 | **70,406 expected online parameters; measure after training**; separate target copy |

The earlier 70,663 for PPO assumed a **shared** actor/critic trunk. The saved
RLInv and A7 checkpoints use **two separate trunks**. Confirmed twice: by
checkpoint-tensor inspection, and by constructing the SB3 models directly —
`mlp_extractor.policy_net` and `mlp_extractor.value_net` are built separately
for `net_arch=[256,256]`.

**Role-based breakdown — this is the form to use in the thesis and the viva:**

| Arm | Decision network (inference) | Training-only apparatus | Total |
|---|---|---|---|
| MaskablePPO (RLInv) | actor **70,406** | critic 69,121 | **139,527** measured |
| PPO-no-mask (A7) | actor **70,406** | critic 69,121 | **139,527** measured |
| A2C | actor **70,406** expected | critic 69,121 | 139,527 — measure after training |
| DQN | online Q-net **70,406** expected | target copy 70,406 | 140,812 — measure after training |

**The decision networks have matched layer dimensions and identical parameter
counts (70,406); the totals differ by 0.9%.** PPO's critic and DQN's target
network are training-time apparatus with no inference role.

⚠️ **What this does and does not establish.** It supports a *capacity*
comparison. It does **not** prove identical computational complexity, nor that
neither algorithm holds an advantage — DQN performs a gradient step per
`train_freq` against a replay buffer while PPO runs 10 epochs over 8,192
transitions, which are very different compute profiles per environment step.
Report wall-clock separately (D2), and state the capacity claim in exactly the
terms above, no wider.

For DQN only the 70,406 online parameters are gradient-updated; the target is
synchronised by copy. Checkpoint-loaded counts are the source of truth
(§1.6 H-F); no retraining is needed to correct a table. Measure the DQN and A2C
counts from actual saved models once those runs exist.
Observation normalisation (`VecNormalize`, `norm_obs=True, norm_reward=False`)
is identical across all four arms.

**Checkpoint selection — verified, then frozen.** `run_eval_grid.ps1:134`
evaluates `{site}_s{seed}_final.zip`, i.e. the **final** checkpoint, for every
policy in the submitted campaign — not `best_model.zip`, despite
`best_model_save_path` being set by the callbacks. **DQN and A2C are therefore
also evaluated at their final checkpoint.** No arm uses validation-selected
weights. This removes the asymmetry Mentor 2 flagged between the RLInv and A7
callbacks: the callbacks differ, but the *evaluated artefact* does not.

### 7.5 Hyperparameter tuning (D1, D2)

| Item | Value |
|---|---|
| Budget | 3 configurations per algorithm |
| Development site | **site9** (Medium; in no existing subset; not a benchmark site) — closes **O1** |
| Data | site9 training window, validation on a held-out slice of it. **Test window never touched.** |
| Held common | Env, reward, observation, action space, interaction budget, evaluation protocol |
| Tuned per algorithm | Optimiser-specific hyperparameters only |
| **PPO / MaskablePPO** | **0 configurations** — retains the submitted configuration |

⚠️ **This is deliberately not an equal-tuning design (D22).** Describe it as
*"the submitted PPO configuration versus limited-tuned alternatives"*, never as
*"equal tuning opportunity"*. The asymmetry runs **in DQN's and A2C's favour**
(§13.7), which is the defensible direction, but it must be named accurately.

**The three configurations per algorithm, frozen now** (not merely the eligible
axes — Mentor 2's requirement). Shared across all: `net_arch` [256,256] ReLU,
γ=0.99, `VecNormalize` as §7.4b, 400k nominal budget, `n_envs`=4.

**DQN** (axes with no PPO equivalent, per D2):

| Config | `learning_rate` | `buffer_size` | `learning_starts` | `train_freq` | `target_update_interval` | `exploration_fraction` | `batch_size` |
|---|---|---|---|---|---|---|---|
| **D-1** baseline | 3e-4 | 100,000 | 5,000 | 4 | 1,000 | 0.10 | 256 |
| **D-2** slow target, more exploration | 1e-4 | 200,000 | 10,000 | 4 | 5,000 | 0.30 | 256 |
| **D-3** fast updates | 3e-4 | 50,000 | 2,000 | 1 | 500 | 0.20 | 256 |

**A2C:**

| Config | `learning_rate` | `n_steps` | optimiser | `gae_lambda` | `ent_coef` |
|---|---|---|---|---|---|
| **A-1** SB3-default-like | 7e-4 | 5 | RMSprop | 1.00 | 0.01 |
| **A-2** longer rollout | 3e-4 | 32 | RMSprop | 0.95 | 0.01 |
| **A-3** PPO-like rollout | 3e-4 | 128 | Adam | 0.95 | 0.01 |

**Selection protocol, frozen:**

| Element | Value |
|---|---|
| Development site | **site9** (Medium; in no existing subset; not a G4 benchmark site) |
| Training data | site9 **days 1–30** |
| Validation data | site9 **days 31–45** — the tail of the *training* window |
| Test data | **Never touched.** Days 46–60 are not read during tuning |
| Scenario | Normal |
| Tuning seeds | **999 and 500** — deliberately outside the five benchmark seeds |
| Selection criterion | Lowest mean validation EENS, averaged over the two tuning seeds |
| Tie-break | Lower mean diesel, then fewer orders |

Two tuning seeds rather than one: single-seed selection at this noise level is
unreliable, and the extra runs cost ~1 h.

Cost: **2 algorithms × 3 configs × 2 seeds = 12 runs ≈ 2–3 h.** PPO receives no
tuning pass (D22) and contributes no runs here.

### 7.6 Two research questions (kept separate)

- **RQ-A — Does the inventory-aware formulation depend on PPO?**
  Final EENS after the same 400,000 environment interactions.
- **RQ-B — Which algorithm learns most efficiently?**
  EENS at checkpoints **50k, 100k, 200k, 400k**.

> **⚠️ Blocking finding (v1.6): the historical artefacts for RQ-B do not
> exist.** `grep -rn "CheckpointCallback\|save_freq" src/` returns **nothing** —
> no trainer saves intermediate checkpoints. The trainers save only
> `_final.zip` and `_best/`. The eval callback writes `evaluations.npz` every
> 10k steps, but that records **rollout reward**, not test-grid EENS. So EENS at
> 50k/100k/200k **cannot be reconstructed for RLInv or A7 from what exists.**
>
> **Resolution — retrain RLInv and A7 with checkpointing on the G4 subset:**
> 3 sites × 5 seeds × 2 arms = **30 runs ≈ 5.5 h**, with a `CheckpointCallback`
> at 50k/100k/200k.
>
> **2026-10-22 correction — retraining is not original-checkpoint reproduction.**
> The 30 new runs are for the 50k/100k/200k EENS learning curves and an
> **independent training-reproducibility** check. Historical **evaluation
> reproducibility** instead replays an actual saved `_final.zip` together with
> its matching `_vecnorm.pkl` under the original evaluation seeds (§13.9).
> The selected site5/seed42 RLInv/A7 replay already matched EENS, diesel and
> grid on 80/80 episodes (§1.6 H-G); preserve its trace and repeat via the
> official evaluator. A fresh training run should **not** be expected to
> reproduce the historical episode values exactly.
>
> **The matched-subset table (§13.8) still uses the original stored cells.** The
> retrained runs supply the curves and the reproduction check — they do not
> silently replace the submitted results.
>
> **If this retrain is not done, RQ-B is reported for DQN and A2C only**, and
> must not be presented as a matched four-algorithm convergence comparison.

RQ-B is the more informative of the two, and the checkpointing above is what
makes it possible at all.

### 7.7 Metrics

Primary: EENS. Secondary: diesel, orders placed, DG-on fraction, violations
(`soc_violation OR tank_overflow`), the three override counters, seed variance,
**wall-clock training time** (per D2: 400k interactions is *not* equal
computational cost across algorithms), inference latency.

### 7.8 Cost

| Item | Runs | Est. |
|---|---|---|
| Tuning (2 algorithms × 3 configs × 2 seeds) | **12** | ~2–3 h |
| DQN main grid (3 sites × 5 seeds) | 15 | ~5–8 h (calibrate DQN pilot) |
| A2C main grid | 15 | ~3 h |
| **Total new DQN/A2C runs** | **42** | **~10–14 h**, indicative |

**Correction:** the older 36-run line counted three configs per algorithm
but accidentally omitted the second tuning seed already frozen in §7.5.
The separate **30 RLInv/A7 curve-retraining runs** (§7.6) are **not** part of
this 42-run table; they are included in §12.2.

⚠️ The DQN multiplier is the softest estimate in this document. **Calibrate with
one 50k DQN pilot before committing the grid.**

### 7.9 Implementation tasks

- `src/eval/evaluate.py:420` — `--algo` currently accepts only
  `{maskable, ppo}`. Add `dqn`, `a2c` and the corresponding loaders at line 482.
- New trainers `train_dqn.py`, `train_a2c.py`, cloned from
  `train_ablation_a7.py` (the correct no-mask reference), with
  `mlflow.start_run()` wrapping `model.learn()` per the G0 fix.
- Checkpoint callback at 50k / 100k / 200k for RQ-B.

### 7.10 What counts as sufficient evidence (Mentor 2)

G4 must **not** end as a table of four final EENS values. The revised report
requires all five of:

| # | Deliverable |
|---|---|
| 1 | **Performance table** — MaskablePPO / PPO-no-mask / DQN / A2C across the three sites and four scenarios |
| 2 | **Learning curves** — EENS and evaluation reward vs environment steps, showing convergence and sample efficiency (this is RQ-B) |
| 3 | **Variability** — seed-level performance, uncertainty, and any meaningful failures reported rather than smoothed away |
| 4 | **Computational comparison** — wall-clock training time and inference latency per algorithm |
| 5 | **Interpretation** — where an algorithm does better, a reasoned account of why, and whether the difference is consistent across sites and scenarios |

**Mandatory scope statement, to appear with the table:**

> This is a representative-subset study across three sites (one per difficulty
> class) and five seeds. It establishes comparative behaviour on that subset; it
> does not establish superiority across all ten sites.

**PPO does not have to win.** The examiner asked for comparative analysis, not
for proof that PPO is best. A fairly obtained result in which DQN outperforms
PPO under some condition is a valid answer to the feedback.

### 7.11 Pre-committed positions

- **Non-convergence.** If DQN has not converged by 400k, **report the learning
  curve and state non-convergence within the matched budget as the finding.** Do
  not extend DQN's budget alone — that destroys the matched-interaction fairness
  the comparison rests on. An extended-budget DQN run may appear as a separately
  labelled footnote, never in the main table.
- **DQN wins.** The contribution becomes: *the inventory-aware joint formulation
  is not tied to PPO; DQN may be preferable under condition X.* This directly
  answers the examiner and costs nothing.
- **Algorithm barely matters.** This is the strongest possible support for the
  thesis — the formulation, not the optimiser, carries the result.

---

## 8. G5a — Computational and training scalability

### 8.1 Frozen timing methodology (Mentor 2)

| Measurement | Protocol |
|---|---|
| RL policy inference | 100 warm-up calls, then **1,000 timed decisions** |
| B1 decision latency | Same observation sequence, same count |
| MPC | Time the complete optimisation call; **record HiGHS solve time separately** |
| Oracle-MPC | As MPC, labelled an offline reference |
| Statistics | **Median, p95, max**; mean reported separately |
| Hardware | CPU model, logical and physical cores, RAM, thread count, library versions |
| Training cost | Actual elapsed wall-clock per 400k run |

**Two rules that prevent an unfair comparison:**

1. Measure **both** policy-only decision latency **and** end-to-end controller
   latency. Do not compare an RL forward pass against MPC's full
   environment-plus-solver loop.
2. Benchmark MPC on a **representative set of real test states**, not one easy
   state. Solver difficulty varies substantially.

### 8.2 Training-cost measurement runs

No timing exists for A5/A6/A7/TrackB. After the G0 `start_run()` fix, run one
timed 400k job per policy on a single site/seed — 4 runs, ~1 h. This doubles as
a smoke test that the logging fix works before any gate depends on it.

### 8.3 Reported quantities

RLInv **total training-policy parameter count (139,527, measured saved
checkpoint)**; **actor-only** inference parameter count measured separately;
wall-clock per 400k run; per-decision inference latency for RLInv / B1 / MPC /
Oracle-MPC; model size on disk; total fleet training cost as a function of *N*.
The previously stated 70,663 in submitted Table 4.1 and protocol v1.7 was a
shared-trunk estimate, not the actual saved architecture (§1.6 H-F).

### 8.4 Out of scope, with reasons

- **The 1 → 2 → 5 → 10 → 25 → 50 replicated-environment sweep.** Running *N*
  independent environments costs *N* times as much by construction; it measures
  hardware, not method.
- **6^N as a headline.** True for a hypothetical centralised Cartesian action
  space, but this thesis never proposes one. Knocking down an architecture that
  was not built invites the obvious objection. It appears in **one paragraph** as
  motivation for the per-site design.

---

## 9. G5b — Cross-site generalisation and transfer

### 9.1 Split (stated before any run)

**Rule (two steps — the second was missing in v1.4 and is restored here):**

1. **Alternate within class.** Stratify by the independently assigned difficulty
   class; within each class order sites by ascending numeric ID; assign
   alternately starting with TRAIN.
2. **Balance to 5/5.** Step 1 yields 6 TRAIN / 4 HELD-OUT. Move the
   **highest-numbered TRAIN site from the largest class** (Medium) to HELD-OUT.

Derivation, verified against `site_classification.csv`:

| Class | Ascending IDs | After step 1 |
|---|---|---|
| Hard | site2, site5 | TRAIN: site2 · HELD: site5 |
| Medium | site3, site4, site7, site9, site10 | TRAIN: site3, site7, site10 · HELD: site4, site9 |
| Easy | site1, site6, site8 | TRAIN: site1, site8 · HELD: site6 |

Step 1 → TRAIN {1, 2, 3, 7, 8, 10}, HELD {4, 5, 6, 9}.
Step 2 → move **site10** (highest-numbered Medium in TRAIN) to HELD-OUT.

| | Sites | Composition |
|---|---|---|
| **TRAIN** | site1, site2, site3, site7, site8 | 2 Easy, 2 Medium, 1 Hard |
| **HELD-OUT** | site4, site5, site6, site9, site10 | 1 Easy, 3 Medium, 1 Hard |

Both sets contain all three difficulty classes. Held-out includes the canonical
Hard site (site5) — a genuinely demanding test.

**On the surplus site — there is no exception.** v1.4 carried a clause retaining
the `is_surplus` site (site8) in TRAIN. That clause is **removed**: site8 is the
third Easy site by ascending ID, so step 1 places it in TRAIN *by the rule
itself*. The desirable property — the shared policy has seen a surplus regime —
follows from the rule rather than from an override, which is a stronger
position. The property is **verified**, not imposed.

**Stated limitation, not to be overclaimed:** because every regime class appears
in TRAIN, the held-out sites are *interpolative*. This experiment tests
within-distribution generalisation across sites, **not** cross-region transfer.
Per Mentor 2, no claim of statistically established generalisation rests on
3 seeds alone.

### 9.2 Four arms (D7)

| Arm | Budget | Answers |
|---|---|---|
| Scratch-400k | 400k on the target site | Existing thesis baseline (already run) |
| **Scratch-50k / Scratch-100k** | 50k / 100k on the target site | **The control that makes transfer interpretable** |
| Zero-shot shared | 0 on target | Does a shared policy generalise? |
| Pretrained + fine-tune 50k / 100k | 50k / 100k on target | Does pretraining reduce adaptation cost? |

Without the scratch-50k/100k control, "pretrained + 50k performs well" is
uninterpretable — it cannot be distinguished from *any* policy trained 50k on
that site performing well. This is the single most important addition from
Mentor 2's review.

### 9.2b Frozen execution details (v1.6)

| Element | Frozen value |
|---|---|
| Training seeds | **42, 123, 777** — the first three in Table 6.2 order |
| Site sampling in the shared trainer | At each episode reset, sample uniformly at random from the five TRAIN sites, using the environment RNG |
| Per-site exposure | Episode count **per training site** recorded in the manifest (§2) and reported — 400k total across 5 sites is ~80k per site in expectation, not a guarantee |
| Zero-shot evaluation | **All five** held-out sites: site4, site5, site6, site9, site10 |
| Fine-tune + scratch controls | **site4 (Medium) and site5 (Hard) only** — closes O3 |
| Evaluation protocol | Identical to §13.5 — `seed + ep`, same episode indices for every arm |
| Normalisation | §9.4 — statistics from the five TRAIN sites' training windows only |

The asymmetry is deliberate and must be stated: **zero-shot is measured on all
five held-out sites; adaptation is measured on two.** Running the four-arm
adaptation study on all five would cost 60 runs rather than 24 for little extra
information, since site4 and site5 already span Medium and Hard.

### 9.3 Budget (D6)

- **Primary:** shared policy at **400k total** interactions vs site-specific
  400k. Matched budget.
- **Sensitivity:** shared policy at **800k**. Conditional on the 400k result and
  available compute — **but the decision rule is recorded here, before results.**
  **Trigger — fully prespecified, leaving nothing to be chosen after the fact:**

  | Element | Frozen value |
  |---|---|
  | Scenario | **Delayed** (one scenario only) |
  | Aggregate | **Equal-site-weighted mean over the five held-out sites**, each site first averaged over its 3 training seeds |
  | Comparator | Site-specific **scratch-400k** under the same scenario and aggregate |
  | Condition | Fire if the zero-shot aggregate exceeds the comparator by **>50% relative AND >2.0 kWh absolute** |

  **Why Delayed.** Normal is unusable as a trigger scenario: RLInv's Normal EENS
  is 0.019 kWh, so the 2.0 kWh floor could never be met and the trigger would
  never fire. Extreme is dominated by physical fuel scarcity, which constrains
  every policy and would mask an undertraining signal. Delayed carries material
  logistics stress at a scale where the absolute floor is meaningful
  (RLInv ≈ 3.6 kWh, B1 ≈ 13.4 kWh on the full grid). Chosen on these grounds
  **before** any G5b result exists.

  ⚠️ Both conditions must hold. The absolute floor prevents a large relative
  difference on a trivially small base from triggering an extra 1.5 h of
  compute.

⚠️ **Interpretation caveat (Mentor 2).** 400k total across five sites means each
site contributes roughly 80k — far less per-site exposure than a site-specific
400k policy. This is a meaningful *sample-efficiency* comparison, **not** equal
per-site exposure. State it plainly.

For scalability, also report **total training cost across all five sites**: a
shared 800k model may still be far cheaper than five independent 400k models
(800k vs 2,000k interactions).

### 9.4 Normalisation — the leakage trap

Observation normalisation statistics are fitted on the **five TRAIN sites'
training windows only**. Exactly those statistics are used when evaluating the
five unseen sites. Specifically prohibited: test-site statistics, held-out
trajectories, statistics refitted using unseen sites.

For fine-tuning: statistics are **updated using only the target site's training
window** (days 1–45), never its test window. Recorded in the manifest
(`normalisation_file`).

### 9.5 Cost

| Item | Runs | Est. |
|---|---|---|
| Shared 400k × 3 seeds | 3 | ~1 h |
| Shared 800k × 3 seeds (conditional) | 3 | ~1.5 h |
| Per fine-tune target: scratch-50k, scratch-100k, ft-50k, ft-100k × 3 seeds | 12 | ~1.5 h |
| Targets: site5 (Hard) + site4 (Medium) — **O3** | 24 | ~3 h |
| **Total** | **~30** | **~5–7 h** |

### 9.6b Reporting structure — per-site primary, aggregate secondary (D24)

My earlier draft said the five held-out sites are *"not a blocking factor to
average away"*. That was directionally right but over-corrected: a fleet-level
mean is a meaningful descriptive quantity for the scalability narrative. Four
reporting levels, in this order of primacy:

| Level | What to show | Interpretation |
|---|---|---|
| **1 — Individual held-out site** (primary) | Mean EENS ± SD over the 3 training seeds | Does transfer work on *this* site? |
| **2 — Individual site, paired arms** | The three seed-level differences, shown individually | Does fine-tuning beat scratch at the same budget? |
| **3 — Across five sites** (secondary) | Equal-site-weighted mean EENS and mean improvement | Descriptive performance over this held-out set |
| **4 — Generalisation claim** | An explicit limitation statement | No statistically established generalisation to a wider population of towers |

**Why both levels are needed.** If transfer works on four sites and fails badly
on the fifth, a single average conceals the deployment risk. Conversely, five
separate tables with no aggregate make the scalability discussion impossible.
Per-site is primary; the aggregate is descriptive support.

**Prohibited in G5b:**
- *d*<sub>z</sub> computed over all 15 site-seed cells as if they were 15 independent
  replicates. One shared-policy seed produces all five site outcomes, so they
  are **correlated through a common trained model** (§10.1).
- Any conventional significance test on 3 training seeds offered as evidence of
  generalisation.

Descriptive paired differences, per-site variability and the **across-site
range** are more informative here than a fragile p-value. The scratch and
fine-tuned arms must use explicitly matched evaluation conditions (§10.3).

### 9.7 Scalability Analysis and Limitations subsection (D13 — mandatory)

The action point is *"Investigate scalability **limitations**"* — not "report
scalability". The deliverable must therefore **name limits**, and a fast
inference number alone does not answer it.

Dedicated subsection, five rows, each pairing evidence with a stated limitation:

| Dimension | Evidence | Limitation to state |
|---|---|---|
| Inference | Policy latency, model size, parameter count | Deployment overhead; hardware assumptions |
| Training | 400k wall-clock; site-specific training cost | Retraining effort grows with tower count |
| Cross-site generalisation | Zero-shot performance on held-out sites | Site heterogeneity; distribution shift; held-out set is interpolative (§9.1) |
| Adaptation | Fine-tune 50k/100k vs scratch controls | Whether transfer genuinely reduces adaptation cost |
| **Multi-tower coordination** | **Architectural analysis only — not tested** | **Independent per-site policies do not optimise shared fuel trucks, shared depots or fleet-wide diesel allocation** |

**The fifth row is the one that must not be omitted.** Even a shared policy that
transfers across sites does **not** demonstrate coordinated optimisation of
shared logistics. State this as an untested scalability dimension rather than
implying that multi-site transfer solves multi-tower deployment. This is also
where the 6^N argument belongs (§8.4) — as motivation for why naïve centralised
scaling is undesirable, in one paragraph.

This answers the scalability comment **without** implementing multi-agent RL.

### 9.8 Pre-committed position

Every outcome answers the examiner. Good transfer → a strong deployment result.
Poor transfer → *site-specific retraining remains the principal scalability
limitation* — an honest, quantified answer to a question the submitted thesis
only deferred to future work.

---

## 10. Statistical protocol (D10)

Frozen before any new result exists. Ten items, matching the review checklist.

### 10.1 The three-level data structure

Every number in this study sits at one of three levels. Conflating them is the
primary statistical risk.

| Level | Object | Randomness it carries |
|---|---|---|
| **Episode** | One 360-step evaluation rollout, indexed `(site, scenario, seed, ep)` | Initial inventory + delivery draws |
| **Run** | One trained policy, indexed `(policy, site, seed)` | Training randomness (RL only) |
| **Cell** | One `(policy, site, scenario, seed)` = mean EENS over its 10 episodes | — |

**The cell is the observation unit.** Its metric is the mean EENS (kWh) over
that run's 10 evaluation episodes.

⚠️ **The observation unit is not automatically an independent training
replicate.** The two coincide in G4, where each `(site, seed)` is its own
training run. They do **not** coincide in G5b: one shared-policy training run
(one seed) is evaluated on all five held-out sites, so those five cells are
**correlated through a common trained model**. Independent replication in G5b is
therefore **3 (the training seeds)**, not 15. Any analysis must state which of
the two it is counting.

**Ten evaluation episodes from one checkpoint are not ten independent training
replicates.** They are repeated measurements of a single trained policy and are
averaged into the cell, never counted as *n*.

### 10.2 What *n* means in each experiment

| Experiment | Scope | Correct *n* |
|---|---|---|
| Existing H1/H2/H3 | 10 sites × 10 seeds | **n = 10** paired seed-level observations for the reported aggregate tests |
| G4 pilot | 3 sites × 3 seeds | **3 seeds per site.** Not n = 9 |
| G4 final | 3 sites × 5 seeds | **5 seeds per site.** Not n = 15. Site is handled separately (§10.5) |
| G3 B1\* | 10 sites × 10 seeds | 10 **evaluation-randomness** replicates per site — see §10.4 |
| G5b transfer | 5 held-out sites × 3 seeds | **3 training replicates**; the 5 sites are the generalisation target, **not** a blocking factor to average away |

### 10.3 Why pairing is justified here — the explicit argument

Mentor 2 is right that *"PPO seed 42 and DQN seed 42 are independent stochastic
training outcomes; the matching seed number is a reproducibility convention."*
The justification for pairing therefore cannot rest on training randomness. It
rests on the **evaluation conditions**, and the code audit (§13.5) establishes
that it holds:

For a given `(site, scenario, seed, ep)`, `evaluate.py:272–273` constructs and
reseeds the environment with `seed + ep` **independently of which policy is
being evaluated**. In `eval_mode` the start index is fixed, so the first RNG
draw is the initial inventory. Therefore **every policy faces the identical
site, identical scenario, identical load/solar/grid trace and identical initial
inventory** at each `(seed, ep)`.

⚠️ **Partially matched, not identical.** The delivery-arrival draws **do**
diverge once ordering behaviour diverges (§13.5), because the draw at
`telecom_env.py:428` is consumed only while an order is pending. So the
evaluation conditions are matched on site, scenario, exogenous traces and
initial inventory — **not** on the realised stochastic trajectory.

> **Statement to use in the thesis.** Pairing is on *partially matched
> evaluation conditions* indexed by seed — identical site, scenario, load/solar/
> grid trace and initial inventory — together with site. Delivery realisations
> diverge as a consequence of each policy's own ordering decisions, which is
> intended. Pairing is **not** a claim that the training randomness of two
> algorithms is matched, nor that their stochastic trajectories are identical.
> The block removes site, scenario and initial-condition variance;
> between-algorithm training variance and delivery divergence remain in the
> paired difference and are reported, not assumed away.

**Pairing strength is not uniform**, and this must be said:

| Comparison | Both arms carry training randomness? | Pairing strength |
|---|---|---|
| RLInv vs A6/A7 (existing H1/H2) | Yes, same PPO family | Strongest |
| RLInv vs DQN/A2C (G4) | Yes, but structurally different optimisers | Weaker — blocks evaluation conditions only |
| RLInv vs B1/B1\*/MPC (G3) | **No** — see §10.4 | Asymmetric variance components |

### 10.4 Variance decomposition differs by arm — verified

B0, B1, MPC and Oracle-MPC are **not RL-trained**. Their "seed" indexes
evaluation randomness only. Confirmed in `master_summary.csv`: all ten policies
carry the same ten seed labels, but for B1 on site5/monsoon the per-seed means
range **69.6 → 181.2 kWh** (variance ≈ 1154) — entirely from initial inventory
and delivery draws, since the policy itself is deterministic.

Consequences, to be stated wherever an RL arm is compared against a rule-based
or tuned arm:

- The paired difference carries **training variance on one side only**.
- The comparison remains valid — the block is the evaluation condition — but a
  pooled variance interpretation is not symmetric.
- For **B1\*** specifically: it is *tuned*, not RL-trained, so there is no
  training-seed replication at all. Report its variability as
  evaluation-randomness variability and label it as such. Do not present it as
  though it were a training replicate.

### 10.5 Primary estimand — prespecified pooling rule

**Do not pool all site-seed rows and run a single t-test.** For a comparison
between policies A and B under a given scenario:

1. Per site *i*, compute the 5 paired differences Δ<sub>i,s</sub> = EENS<sub>A,i,s</sub> − EENS<sub>B,i,s</sub> over seeds *s*.
2. Report **all five** Δ values per site — visible, not summarised away.
3. Site-specific estimate: Δ̄<sub>i</sub> with its SD over the 5 seeds.
4. **Primary pooled estimate: the unweighted mean of the three site-specific
   mean differences**, giving each site equal weight.

With three sites, a test on three site-means has df = 2 and is not informative.
**The pooled estimate is therefore reported as descriptive only.** Inference, to
the extent any is offered, is per site at n = 5.

### 10.6 Effect size

Paired Cohen's *d*<sub>z</sub> = Δ̄ / s<sub>Δ</sub>, where s<sub>Δ</sub> is the sample SD of the paired
differences. The formula stays identical to the submitted Phase 3 analysis so
the new work is methodologically continuous — **but the value, its uncertainty
and its *n* are computed from the new observations, never carried over.**

| Context | *d*<sub>z</sub> |
|---|---|
| Existing H1/H2/H3, n = 10 | Report as in the submitted thesis |
| G4, per site, n = 5 | Report **with n stated**, accompanied by all five raw Δ values |
| G4, pooled across sites | **Do not report a pooled *d*<sub>z</sub>** — at 3 sites the denominator is not meaningfully estimable |

At n = 5, *d*<sub>z</sub> is upward-biased and its confidence interval is very wide. It is
reported as a descriptive magnitude, never as the basis of a claim.

**Degenerate denominators.** When s<sub>Δ</sub> is zero or near-zero, *d*<sub>z</sub> explodes or is
undefined. In that case report **the raw mean difference** and mark *d*<sub>z</sub> as
`UNDEFINED` (exactly-zero variance) or `UNSTABLE` (s<sub>Δ</sub> below 1% of the pooled
EENS scale) — never a large number. This follows the convention already used in
`CANONICAL_ARTIFACTS.md`, which reports `IDENTICAL` rather than p=1.0/d=0 when
two policies match at every seed.

### 10.7 Confidence intervals

Per-site paired CIs at n = 5 use the t distribution with df = 4 (t ≈ 2.776) and
will be wide. **That width is the finding, not a presentation problem** — it is
reported as-is. No normal approximation at these sample sizes, and no CI on the
3-site pooled mean.

### 10.8 Multiple comparisons

G4 generates 4 pairwise contrasts × 4 scenarios = **16 comparisons**. Reporting
whichever is significant would be p-hacking.

**Frozen decision: the G4 analysis is EXPLORATORY in its entirety for
inferential purposes.** No confirmatory claim is made from it. This follows
directly from D10 and from the n = 5 sample size.

**Normal is the focal scenario** — chosen on principle, before any result: it is
the scenario all policies are *trained* on, so it isolates learning capability
without confounding by generalisation to unseen logistics, which is what RQ-A
asks.

**Two focal contrasts, because G4 contains two distinct questions (D23):**

| Role | Contrast | Question it answers |
|---|---|---|
| **Focal algorithm contrast** | **A7 vs DQN, Normal** | PPO vs DQN with **both unmasked** — the clean algorithm question |
| **Principal controller comparison** | **RLInv vs DQN, Normal** | The complete proposed controller vs a DQN implementation |
| Secondary | A7 vs A2C, Normal | PPO vs A2C, both unmasked |
| Secondary | RLInv vs A7, Normal | Masking contribution (already studied as H2) |
| Stress tests | All contrasts, other three scenarios | Robustness to logistics shift |

**Why A7 vs DQN is the focal *algorithm* contrast.** If RLInv beats DQN, an
examiner can fairly ask whether that is PPO or whether action masking explains
it. A7 vs DQN answers the narrower question cleanly, since both arms are
unmasked and share identical environment-level feasibility semantics (§7.3).
RLInv vs DQN is retained prominently as the practically relevant comparison —
but it is a **complete-controller** comparison and must be labelled as such,
never as an isolated PPO-vs-DQN test.

Because the study is exploratory (D19), **neither focal contrast is presented as
a confirmatory superiority test.**

If any contrast is later promoted to confirmatory, Holm–Bonferroni is applied
across the declared family and the promotion is recorded in §12.4 with its date.

### 10.9 Non-significance is not equivalence

A paired difference with p > 0.05 or a CI spanning zero **does not establish
that two algorithms perform the same.** At n = 5 the study is underpowered for
small effects by construction.

Required phrasing: *"no difference was detected at this sample size"* — never
*"the algorithms are equivalent"* or *"performance is the same."* This applies
equally to the existing H2 result (RLInv vs A7, |d<sub>z</sub>| ≤ 0.37), which the
submitted thesis already words correctly and which must not drift.

### 10.10 Sample size, escalation, and reporting limits

| Item | Value |
|---|---|
| G4 pilot | 3 seeds — implementation screening only, no reported inference |
| G4 final | **5 seeds**, the planned minimum (D10) |
| G5b | 3 seeds, explicitly exploratory; no claim of established generalisation |
| Existing evidence | 10 seeds — the contrast in replication must be stated wherever old and new results appear together |

**Withdrawn.** The earlier escalation rule ("expand if a paired CI spans zero
with \|d<sub>z</sub>\| > 0.5") is struck: at 3 seeds both quantities are unstable, so the
rule would fire on noise. It is replaced by the fixed 5-seed minimum plus the
**date-triggered** expansion below.

**Date-triggered expansion (pre-registered, outcome-independent).** If G4
completes and passes PSG and ECG **on or before 30 October** (≥ 5 days before
the 4 November freeze), extend from 3 sites to all 10 at 3 seeds (~60 runs,
~15 h). **The trigger is the calendar, never which algorithm is winning.**

⚠️ **Realistically unlikely.** The protocol is being frozen on 21 October with
G0 not yet started; G4 cannot begin before G1 and G3. The original "≥ 10 days"
threshold implied completion by 25 October, which is not achievable. This is
retained as a **contingency**, not a plan — if the calendar does not permit it,
the 3-site study stands on its own with the §7.10 scope statement, which is
sufficient to answer the examiner.

**Mandatory scope statement**, carried with every G4 table (also §7.10):

> This is a representative-subset study across three sites (one per difficulty
> class) and five seeds, analysed as an exploratory comparison. It establishes
> comparative behaviour on that subset; it does not establish superiority across
> all ten sites.

### 10.11 Matched-subset requirement

All G4 comparisons use RLInv and A7 **recomputed on exactly site1, site3, site5
and seeds 42, 123, 777, 7, 21** (§7.2), over the same evaluation episodes. The 10 × 10
headline figures are never a comparison target. Full specification in **§13.8**.

### 10.12 Recorded per run

Training seed, evaluation episode seed, site, scenario, initial inventory,
algorithm, actual `num_timesteps` — each as a **separate field** in the manifest
(§2), so any of them can be conditioned on afterwards rather than reconstructed.

## 11. Consistency obligations (G6)

### 11.1 Standardised claim wording

Adopted verbatim wherever the primary result is stated — abstract, Chapter 7,
Chapter 8, slides:

> RLInv achieved the lowest observed mean EENS among the primary policies in the
> reported table; paired statistical analysis established a significant
> advantage over A6 only under Normal and Extreme conditions.

This replaces three inconsistent strengths in the submitted work: "achieves the
lowest EENS across all four scenarios" (§7.2), "not statistically
distinguishable … under Delayed or Monsoon" (abstract), "statistically
competitive" (slide 8). Descriptive ranking and inferential evidence stay
separate.

### 11.2 Backward-impact audit

For **every** new result: *does this change an existing Phase 3 conclusion?*
Revise in place. No contradictory appendices.

### 11.3 Limitations to remove from §8.4 (now answered)

- "Single RL algorithm family" → answered by G4
- "No held-out-site transfer evaluation" → answered by G5b

Replace with whatever the new work actually reveals. **Retain and explicitly
add** the simulator-model limitations independently of the new experiments:
source-vs-implemented battery discharge coefficient, 2.4-hour DG reserve vs
2-step ON latch, slice-dependent tank/reward scaling and inactive literal
stockout-cliff in recorded evaluation. The G4/G5 results do **not** cure those
physical-model limitations (§1.8).

### 11.4 New sections

| Section | Content |
|---|---|
| New §7.7 | Algorithm comparison (RQ-A, RQ-B) |
| New §7.8 | Scalability: computational, training, deployment — **three separate statements, never a blanket "RLInv is scalable"** |
| §4.5 + §7.2 | B1\*-Deployable and B1-OracleTune |
| §6.3 or §7.2 | TrackB correction note |
| §8.4 / §8.5 | Revised limitations and future work |

### 11.5 Highlighting all revised content (D14 — EXIT CRITERION)

> *"All revised content should be highlighted in the report."* — resubmission
> instructions

This is a **compliance requirement**, not a formatting preference. A revision
that is scientifically complete but fails this can still fail the resubmission.

**Technical risk, and why it is scheduled first.** The thesis is 146 pages,
pdfTeX 1.40.23, LaTeX with `hyperref`. The common highlighting packages have
known interactions: `soul`'s `\hl{}` breaks badly across line and page
boundaries and can conflict with `hyperref`; `\colorbox` does not wrap at all.
Discovering this at page 140 on 9 November would be unrecoverable.

**Mandated approach — one convention, defined once:**

| Scope | Mechanism |
|---|---|
| Sentence / phrase level | A single `\revised{}` macro, defined once in the preamble so the whole convention can be re-coloured or disabled globally |
| Whole new sections, tables, figures | A vertical change bar in the margin (e.g. `changebar`), **not** a page of highlighted text — a fully highlighted page is unreadable in print |
| Changed numbers in existing tables | `\revised{}` on the cell, so the TrackB correction and any B1\* revisions are **visibly** marked |

**Prototype task, week 1, before any new prose is written:** compile a 2-page
sample exercising `\revised{}` across a line break, across a page break, inside
a table cell, inside a figure caption, and inside a section heading. Confirm it
survives, and that it remains legible in greyscale print. Only then write the
new chapters.

Marking changed table cells has a side benefit: the TrackB integrity disclosure
becomes visible rather than buried in prose.

### 11.6 Rebuttal report as an appendix (D14 — EXIT CRITERION)

> *"A rebuttal report should also be prepared and attached as an appendix in the
> report."*

Note **"as an appendix in the report"** — embedded in the thesis PDF, not a
separate file alongside it. The compiled PDF must be inspected to confirm the
appendix survived compilation.

### 11.7 Rebuttal format (D15) — reuse the proven institutional format

Do **not** invent a three-column table. `Rebuttal_Report__Phase2_Feedback.pdf`
used a format that was accepted, and Phase 1 used the same structure. Reuse it:

```
Thesis and Candidate Details
  1. Thesis Title            5. Previous Viva Date
  2. Student Name            6. Phase in Previous Viva  → Phase 3
  3. Roll Number             7. Mentor Name(s)
  4. Degree

Response to comments by Examiner 1
  Examiner Name: ...

  Comment N: <verbatim examiner text>
  Response:  <prose — what was done and why>
  Changes in Thesis:
     1. <specific change> (Section X.Y, Table/Figure Z)
     2. ...
```

The `Changes in Thesis` list carries **exact section, table and figure
numbers**. That is stricter than a generic mapping table and is what the Phase 2
rebuttal did. Fill the numbers only once the sections are final.

**Comment coverage (all seven feedback sentences get a numbered response):**

| # | Examiner text | Primary evidence |
|---|---|---|
| 1 | "analysis lacks breadth" | G3 + G4 + G5 jointly; revised discussion |
| 2 | "compare the performance of different RL techniques" | New §7.7 algorithm comparison |
| 3 | "study the scalability of this method" | New §7.8 scalability |
| 4 | "missing proper baseline method" | B1\*-Deployable + Baseline Taxonomy table (§6.4) + DQN/A2C as RL comparators |
| 5 | "needs comparative analysis and scalability considerations" | Same as 2 + 3, cross-referenced |
| 6 | Action point: "Explore comparative performance of RL techniques" | §7.7 |
| 7 | Action point: "Investigate scalability **limitations**" | §7.8, especially the limitations table (§9.7) |

### 11.8 Code-viva readiness (carried-over Phase 3 requirements)

The code viva follows **after** resubmission clearance, and
`phase3submission_instruction_class.docx` states: *code demo uses the README*,
*inference time and training time*, *code deployment on private cloud*, and
**"Examiner can ask to run the code during code viva."**

This revision adds ~8 new components (PSG, manifest, two trainers, the sweep
harness, the shared-multisite trainer, the latency bench, the repaired wrapper).
Repo runnability is therefore **not** automatically preserved.

| Item | Status |
|---|---|
| `README.md` updated for every new component, with exact run commands | Required |
| Fresh-clone reproduction test — clone, install, run one short job of each type | Required |
| Inference and training time reported | Satisfied by G5a — but must also reach the **deck** |
| Mentor pen signature on certificate | ✅ **Verified present** (certificate page, scanned image) |
| Certificate date | ⚠️ Currently reads "July 2026" — update to November 2026 |
| Signature image background | Minor: the scan carries a visible grey box; worth a transparent re-crop while recompiling |

### 11.9 Viva preparation (a first-class deliverable, not an afterthought)

Read the feedback categories carefully. Report = "well written". Presentation =
"good". Answers = "well". Performance = "good understanding". **Viva =
"satisfactory"** — the weakest word in the whole document. And *"analysis lacks
breadth"* is filed under **performance in Q&A**, not under the written work.

That combination suggests the gap was partly *live*: when probed on breadth —
other algorithms, scale — the answers did not extend. A fresh viva is expected.

| Deliverable | Note |
|---|---|
| Backup slides for **every new experiment** | The Phase 2 lesson (Comments #2, #11) recurs: evidence must be retrievable on the spot |
| Key numbers memorised for the new results | As was done for the five Phase 3 numbers |
| **Phase 1 + Phase 2 recap slide** | ⚠️ **Gap confirmed.** The submitted 23-slide deck has no recap slide; slide 14 addresses Phase 2 *feedback*, which is not the same thing. `phase3submission_instruction_class.docx` explicitly requires "In ppt recap of phase 1 and phase2 also" |
| **Training / inference time slide** | ⚠️ **Gap confirmed.** Slide 21 gives the training budget in steps but no wall-clock; the instructions require both times |
| Anticipated questions | "Why only three sites?", "Why not SAC?", "Why didn't you implement multi-agent RL?", "Is B1* really the strongest classical policy?", "What changed in TrackB and why?", **"Why is the source discharge coefficient 1.20?", "What does the zero stockout count mean?", "Why did the train/test tank capacity change?", "Is there really a two-hour DG ON minimum?"** |

### 11.10 Three scalability statements (keep distinct)

1. **Computational** — inference latency and model size.
2. **Training** — wall-clock per 400k site-specific policy.
3. **Deployment / generalisation** — zero-shot and fine-tuned performance on
   held-out sites.

One favourable latency number must not be used to claim overall scalability.

---

## 12. Execution order and amendment log

### 12.1 Priority (both mentors converge)

1. **G0** — protocol, PSG, manifest, timing fix
2. **G1** — TrackB diagnostic
3. **G3** — B1\* strengthened baseline (evaluation-only, overlaps G2)
4. **G4** — algorithm benchmark ← *examiner-requested*
5. **G5a** — measured timing
6. **G5b** — held-out-site transfer ← *examiner-requested*
7. **G2** — TrackB full retrain (background; conditional)
8. Date-triggered site expansion (§10)

**TrackB is integrity work, not an examiner requirement (Mentor 2).** G1/G2
must happen because they affect *existing* results, but they are not the
response to this feedback. If CPU time or development effort tightens,
**prioritise a defensible G4 and G5 over any additional TrackB variant.**

**Guard rail (Mentor 2).** The explicit examiner action points are *"explore
comparative performance of RL techniques"* and *"investigate scalability
limitations."* TrackB is an existing-results integrity issue that must be
investigated — but **it must not consume the revision.** If corrected TrackB
threatens G4 or G5b, remove it from the primary comparison, correct it at
reduced scope, explain why, and spend the compute on what was actually asked.

### 12.2 Total cost

| Gate | Training runs | Est. CPU |
|---|---|---|
| G1 | 0 — replay only | minutes |
| G2 | 100 (+400 eval cells) | ~18 h + ~2 h eval |
| G3 | 0 — evaluation only | **pilot-gated (§6.3)** |
| G4 — tuning | 12 | ~2–3 h |
| G4 — DQN | 15 | ~5.5–8 h |
| G4 — A2C | 15 | ~3 h |
| G4 — RLInv/A7 checkpoint retrain (§7.6) | 30 | ~5.5 h |
| G5a | 4 | ~1 h |
| G5b | 30 (incl. conditional 800k) | ~5–7 h |
| **Total (including conditional G2 and G5b 800k)** | **206** | **~42–50 h, plus G3, indicative** |

**Reconciled in v1.6.** The earlier "~190 runs / 40–47 h" was not derivable from
the gate list — Mentor 2's recount of ~170 was correct against v1.5. The figure
rises to 206 because v1.6 adds the 12 tuning runs (was 6) and the 30
checkpoint-retrain runs that RQ-B requires. **G3 remains excluded from the CPU
total** until its pilot resolves; the hours above are the known gates only.

Spread over the days to the 4 November freeze, with runs launching unattended,
this is roughly 3–4 h/day of machine time — still not the binding constraint
(§12.3).

### 12.3 The real constraint

**Compute is not the bottleneck. Implementation and writing time is.** Of ~45
CPU-hours, almost all runs unattended. What consumes the calendar is writing the
PSG, the manifest, the TrackB truth table and fix, the `(s,S)` harness, the
DQN/A2C trainers, the multi-site training loop — and then integrating all of it
into a 146-page thesis.

The risk is therefore **not** "we cannot afford the experiments." It is "we
build six new components in three weeks and one carries a TrackB-class defect we
do not catch." The PSG and the manifest exist for precisely that, which is why
G0 is not rushed despite producing no results.

**Per Mentor 2: begin drafting thesis sections and the rebuttal structure as
soon as this protocol is frozen, while experiments run.** Waiting for every
result before writing is the larger risk.

### 12.4 Amendment log

| Date | Section | Change | Reason |
|---|---|---|---|
| 2026-10-21 | — | Protocol v1.0 frozen | Initial pre-registration |
| **2026-10-22** | **§1.9, §1.10, §7.4b** | **v1.9 — audit record closed. Experiment design unchanged since v1.5.** Repository G0 energy audit executed with real Gymnasium against the unmodified repo: **PASS**, 12,960 + 17,280 steps, max ledger errors 1.78e-15, zero SoC-clip and tank-overflow events; evidence JSON hashed. Script advanced v1.8 → **v1.8.2** (SoC compared against clipped value with flag asserted; `violation_tank` and combined flag validated the same way). **ECG-1 provenance RESOLVED** — all ten CSVs numerically identical, max diff exactly 0.0; divergence is CRLF vs LF; identity criterion reworded to row-wise numerical comparison after newline normalisation. **Role-based parameter table added** — decision networks matched at 70,406, totals within 0.9%, with an explicit limit on what that establishes. **Battery claim discipline** — "C-rates" and "uniform across policies so rankings hold" both rejected with reasons; approved wording frozen. | Repository execution; row-by-row dataset comparison; SB3 model construction; mentor review of v1.8 |
| **2026-10-22** | **§1.4–1.8, §2, §3, §7.4b, §7.6, §7.8, §8.3, §11, §13, Appendix A/B** | **v1.8 audit-only amendment; experiment design unchanged.** Replaced circular/partial energy audit with fail-closed reconstruction and saved JSON; locally exercised 12,960 + 17,280 steps against historical environment (with Gymnasium compatibility shim), official repo run pending. Recorded 44,000-row historical zero-stockout result, measured 10-site train/test tank changes, battery 1.20 source-convention discrepancy, missing two-hour ON latch, 139,527 verified PPO parameters, and selected 80/80 checkpoint replay. Corrected G4 tuning budget to 12 and separated re-training from checkpoint replay. Added thesis/viva disclosure checklist and G0 fail-closed requirements. | Source code, raw CSVs, historical results, checkpoints, 2024 Zindi scoring reference; retrospective G0 integrity clarification, not post-hoc result selection |
| 2026-10-21 | §6.4 | Added Baseline Taxonomy and Fairness table (D12) | Mentor 2 |
| 2026-10-21 | §9.7 | Added Scalability Limitations subsection, incl. multi-tower coordination row (D13) | Mentor 2 |
| 2026-10-21 | §7.10 | Added sufficient-evidence requirements and the scope-boundary statement | Mentor 2 |
| 2026-10-21 | §11.5–11.9 | Highlighting and rebuttal appendix made exit criteria; rebuttal format fixed to the Phase 2 precedent; code-viva and viva-prep deliverables added | Resubmission instructions; gaps found 2026-10-21 |
| 2026-10-21 | **§13 (new)** | Experimental Comparability Gate added: reference configuration, Category A/B/C, ECG-1..7, matched-subset rule, reference-reproduction pilot, Compatibility Contract | Mentor 2 |
| 2026-10-21 | §0 (O2) | O2 closed — PPO was never tuned (§13.7); no PPO-Tuned arm by default | Config audit |
| 2026-10-21 | §11.1 | Claim wording tightened after verifying RLInv is numerically best in only 1 of 4 scenarios across the full 11-policy set (§13.3) | Data audit |
| 2026-10-21 | §1.1, §6.3, §7.5, §7.8, §9.3, §9.6b, §10.1, §10.3, §10.6, §10.8, §10.10, §13.7 | **v1.4 review corrections:** HF-7 relaxed to scientific-config identity; G3 grid made coarse-to-fine and cost estimate corrected (15,300 → ~5,400 episodes) with a mandatory MPC propagation pilot; tuning design renamed honestly (D22); G4 totals recomputed after O2 closure (54–57 → 36 runs); 800k trigger given an absolute floor; G5b four-level reporting (D24); observation unit vs independent replicate separated; pairing wording softened to *partially matched*; *d*<sub>z</sub> degenerate-denominator rule; focal contrasts split into algorithm vs controller (D23); expansion threshold made realistic | Mentor 2 review of v1.3 |
| 2026-10-21 | §1.1, §1.4, §1.5 | **v1.7:** energy-accounting verification **executed** on the real environment (12,960 + 17,280 steps) — load-side identity exact, curtailment never negative and ≤ PV, EENS = Σ unmet exact. **Simulator correct; HF-2 was a validator bug; no existing result affected; nothing to rerun.** Verified net source-side identity recorded; two validator gotchas documented (`p_grid_kwh` includes battery charging; DG delivers `min(rated, residual)`). **Thesis Eq 3.1 confirmed incomplete** — omits unserved, curtailment and the DG-partial-output fact; corrected form recorded as a G6 documentation revision, not a model change | Mentor 2 question on existing-result impact |
| 2026-10-21 | §1.1, §4, §6.2, §6.3, §7.2, §7.4b, §7.5, §7.6, §9.2b, §12.2, §13.9 | **v1.6 pre-freeze audit:** **HF-2 corrected to the load-side identity** (the source-side form would have hard-failed every high-EENS episode) + HF-2b added; HF-7 made conditional on declared comparison design (the ablations vary obs/action space *by design*); D1 aligned with D22; G4 seeds changed to 42/123/777/7/**21** to remove the 7↔13 episode-seed overlap; network architecture and parameter counts frozen; checkpoint convention verified as `_final.zip` and frozen; **the three tuning configurations per algorithm specified in full** with development split, tuning seeds and selection criterion; **RQ-B blocking finding — no intermediate checkpoints exist, resolved by a 30-run RLInv/A7 retrain that doubles as the §13.9 reproduction pilot**; TrackB confirmation split into defect-existence vs explanatory-power; B1\* fairness claim narrowed to data access; G5b execution details frozen; G3 propagation made conditional; budget reconciled to 206 runs | Mentor 2 pre-freeze audit of v1.5 |
| 2026-10-21 | §0 (D19), §7.4, §7.5, §9.1, §9.3, §12.2 | **v1.5 consistency pass:** D19 aligned with D23 (two focal comparisons); obsolete O2 conditional removed from the A7 reuse verdict; tuning cost corrected 9–12 → 6 runs; §12.2 totals reconciled (G4 54–57 → 36; G3 marked pilot-gated); **G5b split rule completed with the balancing step and verified to reproduce the stated sets**; surplus-site exception removed as unnecessary; 800k trigger scenario and aggregate prespecified | Mentor 2 review of v1.4 |
| 2026-10-21 | **§10 (rewritten)** | Three-level data structure; per-experiment *n*; explicit pairing justification; variance decomposition by arm type; prespecified pooling rule; *d*<sub>z</sub> policy at small n; CI method; **multiple-comparison policy (was absent)**; non-significance ≠ equivalence | Mentor 2 statistical review |

---

---

## 13. Experimental Comparability Gate (ECG)

**The distinction from the PSG, and why both are needed:**

- **PSG** asks: *is this policy behaving correctly?* Are the results physically
  and computationally plausible?
- **ECG** asks: *are two results scientifically comparable?* If they differ, can
  we identify exactly what changed?

A perfectly implemented DQN experiment can still produce an **invalid
comparison** if it uses a different reward, initial-inventory distribution,
evaluation scenario or normalisation procedure. PSG would pass it. ECG is what
catches it.

**No new result enters the thesis until both PSG and ECG pass.**

### 13.1 What this gate protects against

That the November resubmission compares results produced under different
experimental conditions and then attributes the difference to PPO, DQN, A2C,
inventory learning or scalability.

### 13.2 The frozen reference configuration

Committed as `config/phase3_reference.yaml`. **Verified against
`src/config/hparams.yaml` and the evaluation code on 2026-10-21 — not taken
from the thesis narrative alone.**

| Dimension | Reference value | Verified source |
|---|---|---|
| Dataset | ITU/Zindi 2024, 10 sites, hourly | `data/processed/site*.csv` |
| Temporal split | 45 d train (1,080 steps) / 15 d test | `hparams.yaml:42` |
| Episode length | 720 train / 360 eval | `env_cfg` |
| Seeds | 42, 123, 777, 7, 13, 21, 99, 314, 500, 999 | Table 6.2 + `master_summary` |
| Nominal budget | 400,000 timesteps | `hparams.yaml:127` |
| **Actual PPO budget** | **401,408** — see §13.3 | computed |
| Parallel envs | 4 (`SubprocVecEnv`) | `hparams.yaml:129` |
| Network | MLP [256, 256], ReLU | `hparams.yaml:120` |
| Learning rate | 3e-4, linear decay | `hparams.yaml:105` |
| γ | 0.99 | `hparams.yaml:109` |
| `n_steps` / `batch_size` | 2,048 / 256 | `hparams.yaml:107–108` |
| Reward coefficients | α=1.0, β=0.4, λ=100, γ_r=200, μ=20 | `hparams.yaml:188` |
| Init inventory — train | 0.20–0.90 | `hparams.yaml:45–46` |
| Init inventory — eval | 0.30–0.90 | `hparams.yaml:47–48` |
| Primary metric | EENS, kWh/episode | — |
| Scenarios | Normal, Delayed, Monsoon, Extreme | `run_eval_grid.ps1` + `master_summary.csv` (the generic YAML scenario-list differs) |
| Core historical CSV | **11** policy families × 10 sites × 10 seeds × 4 scenarios × 10 episodes = **44,000** episodes; thesis main-family subset = 9 | `master_summary.csv` + `run_eval_grid.ps1` (8 families; MPC runs separate) |
| Tank scaling caveat | `C_tank = 72 × mean(load)` evaluated from **each supplied slice**; train/test difference per site (§1.6 H-B) | `TelecomEnv.__init__`, trainer, evaluator, site CSVs |
| Battery-model convention | Historical `ΔSoC=(eta_c × E_charge − E_discharge/eta_d)/C`, with source `eta_d=1.20` | `TelecomEnv._energy_balance()`; §1.6 H-C |

### 13.3 Three audit findings (verified 2026-10-21)

**(a) The actual PPO training budget is 401,408 steps, not 400,000.**
`n_steps`=2,048 × `n_envs`=4 = 8,192 transitions per rollout; SB3 completes whole
rollouts, so ⌈400,000/8,192⌉ = 49 rollouts = **401,408**. DQN and A2C collect
transitions differently and will land on different actual counts. **ECG-3
therefore records actual `num_timesteps`, never the nominal budget.** The
difference is 0.35% and immaterial to performance — but it must be *recorded*,
because an unexplained discrepancy discovered by an examiner is worse than a
disclosed one.

**(b) RLInv is numerically best in only one of the four scenarios** once the
full evaluated policy set is considered:

| Scenario | Numerically lowest | Value | RLInv |
|---|---|---|---|
| Normal | **RLInv** | 0.019 | 0.019 |
| Delayed | A7 | 3.293 | 3.599 |
| Monsoon | A6 | 10.651 | 11.286 |
| Extreme | multi (RLInv-Multi) | 53.451 | 53.854 |

The submitted Table 7.1 claim — *"RLInv achieves the lowest EENS across all four
scenarios"* — is true **only within Table 7.1's six-policy set** (B0, B1, MPC,
TrackB, Oracle-MPC, RLInv), which excludes the ablations and the exploratory
multi-scenario arm. It is **not** true across the eleven policies actually
evaluated.

This is an existing wording vulnerability, independent of any new experiment,
and it becomes more exposed once DQN and A2C are added. §11.1's standardised
wording already handles it; the qualifier must be explicit and verifiable —
*"among the primary policies in Table 7.1"*, never an unqualified "lowest".

**(c) The evaluation pairing is stronger than feared — and must not be
disturbed.** See §13.5.

### 13.4 The seven checks

| Check | What is verified | If it fails |
|---|---|---|
| **ECG-1** Dataset identity | Site CSV hashes, chronological split, feature generation | Rebuild, or label as a different-dataset experiment |
| **ECG-2** Environment identity | Transition equations, reward, DG/battery physics, inventory and delivery rules, **including source-vs-implemented coefficient convention and slice-based `C_tank`/`d_bar`** | Re-run the affected reference policy; do not silently mix models |
| **ECG-3** Training budget | **Actual** environment interactions (§13.3a), plus training scenario | Correct the budget, or explicitly frame it as a different-budget comparison |
| **ECG-4** Observation & normalisation | Features, units, ordering, `VecNormalize` procedure, train-only statistics | Fix before evaluation; no test leakage |
| **ECG-5** Action semantics | Same six actions; same invalid-action consequences where a direct comparison is claimed | Cannot claim an isolated algorithm comparison |
| **ECG-6** Evaluation pairing | Same site, scenario, episode index, evaluation seed, initial inventory, exogenous traces (§13.5) | Re-evaluate under the common protocol, or analyse unpaired |
| **ECG-7** Results provenance | Checkpoint, code commit, YAML, dataset hash, evaluation script, raw metrics all traceable | **Quarantine the result** until provenance is established |

### 13.5 ECG-6 in detail — what is paired, what is not, and why we leave it alone

Mentor 2 is right that `seed=42` does not by itself guarantee identical
stochastic trajectories. Audit of the actual code:

`TelecomEnv` uses **one** RNG stream (`telecom_env.py:196`,
`np.random.default_rng(seed)`) for the episode start index (341), the initial
inventory draw (348), and the per-step delivery-arrival draw (428).

**What saves the pairing:** `evaluate.py:272–273` constructs a **fresh
environment and reseeds it for every episode** with `seed + ep`, and in
`eval_mode` the start index is fixed at 0 (`telecom_env.py:339`) with no RNG
draw. So the **first** draw of every evaluation episode is the initial-inventory
uniform. Therefore, for a given (site, scenario, seed, episode):

| Quantity | Matched across policies? |
|---|---|
| Load, solar, grid traces | ✅ Deterministic from `df_test`, start index 0 |
| Initial inventory | ✅ First RNG draw, before any divergence |
| Initial SoC | ✅ Fixed parameter |
| **Delivery arrival draws** | ❌ Diverge once ordering behaviour diverges |

Confirmed empirically: `init_inv_frac` has an identical mean (0.621) across
RLInv, A6 and TrackB in `master_summary.csv`.

The delivery draw at line 428 sits inside `if self._pending_flag == 1:`, so a
policy that orders at different times consumes a different number of draws and
every subsequent draw differs.

**This is correct behaviour, not a defect** (Mentor 2's own nuance): policies
*should* be free to order at different times, and forcing identical delivery
histories would contradict their decisions. The design is **paired on everything
exogenous and unpaired only on the consequences of the policy's own choices.**

**Rule (D18) — do not "improve" this.** Switching to pre-generated exogenous
lead-time streams indexed by order number would be defensible in a new study,
but here it is a **Category C** change: it would make every new result
incomparable with all 44,000 existing evaluation episodes. New experiments
**match the protocol as it stands, including this property.** The property is
disclosed in the thesis; it is not re-engineered three weeks before submission.

### 13.6 Category A / B / C — the comparability classification

| Category | Variables | Rule |
|---|---|---|
| **A — must match** | Site, train/test split, simulator physics, **the actual battery coefficient update**, DG start rule, reward definition and `d_bar`, **tank sizing from each slice**, logistics scenario, evaluation episode length, initial-state protocol, EENS calculation, evaluation randomness protocol | A mismatch **invalidates** a direct comparison |
| **B — may differ, must be disclosed** | Algorithm-specific hyperparameters, replay buffer, exploration schedule, rollout configuration, masking implementation, optimiser, network architecture, training runtime | Permitted **only** as the declared subject of the comparison |
| **C — intentionally different** | Joint vs decomposed action space, tuned vs original (s,S), shared vs site-specific training, zero-shot vs fine-tuned, 400k vs 800k | Creates a **new** comparison, never an interchangeable replacement |

**The governing rule:** a Category C difference produces a *new experimental
comparison*, not a drop-in replacement for an existing result.

Worked example: a DQN trained on site5 for 400k steps **may** be compared with
the existing site5 RLInv checkpoint — Category A matches, and the Category B
differences *are* the declared algorithm comparison. A DQN trained under a
revised reward or a different invalid-action penalty **may not**; that
difference cannot be attributed to the algorithm.

### 13.7 Per-experiment comparability requirements

| New experiment | Reference | Requirement |
|---|---|---|
| Corrected TrackB | Original TrackB, A6, RLInv | Identical training/evaluation protocol; **only** the documented wrapper defect changes |
| B1\* tuned (s,S) | Original B1, MPC, RLInv | Training-window tuning only; simulator and test scenarios unchanged |
| DQN / A2C | **RLInv-Original and A7-Original** | Same joint control task, reward, observations, evaluation; algorithm-specific differences declared as Category B |
| Computational scalability | RLInv, B1, MPC | Same hardware; representative test states; equivalent decision boundaries |
| Shared-site policy | Site-specific RLInv | Same held-out sites and scenarios; training-site exposure explicitly different (Category C) |
| Fine-tuning | Scratch controls | Same target site, adaptation steps, evaluation protocol, normalisation policy |

**On tuning, and the closure of O2.** Mentor 2's rule: *do not retune PPO and
then compare the new PPO against the old RLInv table as though nothing changed.*
Accordingly:

- **RLInv-Original and A7-Original are preserved at their submitted settings.**
- A7-Original serves as the unmasked-PPO reference **after** equivalence
  verification (done — §7.4).
- DQN and A2C are tuned on a separate development protocol (site9, §7.5).
- A **PPO-Tuned** arm, if ever added, is a **distinctly labelled fifth arm with
  its own results** — never a replacement for RLInv-Original.

**Config audit result (closes O2).** PPO's settings are SB3 defaults except
`batch_size` (64 → 256) and `ent_coef` (0.0 → 0.01), both pragmatic CPU/
exploration choices; slide 21 of the submitted deck records every other
hyperparameter as *"No — fixed"*, with only γ sensitivity-tested over
{0.95, 0.99, 0.995}. **PPO was therefore never tuned for this problem.** Giving
DQN and A2C a three-config budget while PPO stays at defaults advantages
*them* — the conservative direction. So **no PPO-Tuned arm by default.**

Required disclosure in §7.7: *"PPO ran at the submitted configuration, which is
close to Stable-Baselines3 defaults and was not tuned for this problem; DQN and
A2C each received a three-configuration tuning budget. Any residual advantage to
PPO is therefore not attributable to tuning effort."*
⚠️ Confirm the two deviations against the installed SB3 version during G0 — the
defaults were not verifiable in the audit environment.

### 13.8 Matched-subset rule (D17) — the single most likely mistake

The submitted RLInv Monsoon mean of **11.29 kWh is an average over 10 sites ×
10 seeds.** A new DQN result on 3 sites × 5 seeds **cannot** be compared against
it; the difference would conflate algorithm with site and seed composition.

**Required:** extract RLInv's and A7's results for **exactly** site1, site3,
site5 and seeds 42, 123, 777, 7, 21 (§7.2), over the same evaluation episodes, and
recompute. Then build, separately for each of the four scenarios:

| Algorithm | site1 | site3 | site5 | Matched-subset mean |
|---|---|---|---|---|
| RLInv-Original | from existing | from existing | from existing | recomputed |
| A7-Original | from existing | from existing | from existing | recomputed |
| DQN | new | new | new | computed |
| A2C | new | new | new | computed |

**The original 10-site headline table stays intact** unless a confirmed
implementation correction requires revision.

**Inference discipline:** report paired seed-level differences *within* sites,
plus site-specific uncertainty, plus descriptive pooled results. Do **not** treat
15 site-seed observations as 15 independent sites, nor 10 evaluation episodes
per seed as independent training replicates.

### 13.9 Reference-reproduction pilot — run before any expensive experiment

Take one existing RLInv checkpoint with its **matching** saved `VecNormalize`
statistics, run through the **official repository's evaluation loader** under the
historical settings, and compare **per-episode**, not only average, energy
metrics against the stored originals in `master_summary.csv`.

**Progress already made (2026-10-22):** saved site5 seed42 RLInv and A7
checkpoints (401,408 timesteps, 11-D input, six actions) were replayed in a
local audit, matching stored **EENS, diesel, and grid** on all
2 policies × 4 scenarios × 10 episodes = **80/80**. This was a targeted
inference reconstruction, **not** an official SB3-run execution trace and not
bit-for-bit equality of every inventory/order metric. Preserve a replay log;
repeat on the full repository before marking G0 COMPLETE.

Repeat for **B1** (no model checkpoint needed) under its historical parameters.
Missing `_vecnorm.pkl`, observation-shape mismatch or missing `EENS_kWh` must
stop evaluation, rather than triggering unnormalised inference or zero defaults.

**The 30 RLInv/A7 training jobs in §7.6 answer a different question:**
independent retraining with intermediate checkpoints supports *learning curves*,
not the authenticity/replay of historical `_final.zip` artefacts.

| Outcome | Action |
|---|---|
| Matches within tolerance | Evaluation pipeline is compatible — proceed |
| Does not match | **Stop.** Investigate before any new training run |

Exact matching requires the same random-stream protocol; otherwise compare under
a documented tolerance and inspect differences at the **episode** level, not only
at the mean.

✅ **Episode-level comparison is directly possible — verified (v1.6).**
`master_summary.csv` holds **44,000 rows = 4,400 cells × 10 episodes**, with an
`episode` column running 0–9 and exactly 10 rows per
`(policy, site, scenario, seed)`. It is a **per-episode** record, not a table of
cell averages, so per-episode reproduction can be checked against it directly
without recovering separate episode logs.

This costs minutes. It is far cheaper than discovering after 18 training runs
that the evaluation environment has drifted.

### 13.10 The Phase 3-R Compatibility Contract

1. The submitted Phase 3 results are a **read-only reference dataset**.
2. Every new run records a complete manifest (§2) **and a compatibility status**.
3. Every comparison uses a **matched subset** of existing results (§13.8).
4. Any environment change triggers an **impact assessment** naming which
   historical policies require re-evaluation or retraining.
5. New algorithm-specific tuning is **isolated from the final test data**.
   Historical `C_tank` and reward `d_bar` are still computed from each slice as
   the original simulator actually did; this limitation is documented (§1.6)
   and held *identically* across primary comparator evaluations, not silently
   changed under a train-only-tank patch.
6. Every new table states whether it is a **direct matched comparison**, a
   **corrected comparison**, or a **deliberately different experimental
   condition**.
7. **No new result enters the thesis until both PSG and ECG pass.**

### 13.11 First action, before any expensive run

Review together, as artefacts rather than as narrative: `master_summary.csv`,
the per-run manifests, `hparams.yaml`, the evaluation script, and the TrackB
wrapper. **The thesis states the intended design; these files state what
actually ran.** §13.3 is what that review has produced so far.

### 13.12 Frozen historical physics versus alternative-model sensitivity

**Primary G3/G4/G5 contract:** the simulator is frozen at the actual historical
transition/reward semantics (battery: divide discharge by 1.20; DG fuel reserve:
2.4 rated hours; **no** 2-step ON latch; `C_tank=72×d_bar` on the supplied
training/test slice). New DQN and A2C use precisely that same model so a
performance difference can be attributed to algorithm/control design within the
specified bounds. Correct **equations, explanatory text, telemetry, audit scripts,
fail-closed evaluation** without changing historical transition dynamics.

**Why:** Source code and the 2024 challenge scoring convention show a real
battery-interpretation difference (§1.6 H-C); changing it only for new arms
would confound the claimed algorithm effect with model physics. If a separate
source-consistent battery model is studied, assign an explicit `physics_version`
(e.g. `source_factor_sensitivity`), record a different ECG-2 fingerprint,
run **all** compared policies under that alternative model for evaluation, and
report it **outside** the historical G4/G5 main table. Frozen-policy evaluation
alone measures *deployment/robustness sensitivity*, not how policies would learn
if trained in the corrected model. Retrain the compared policies under corrected
physics only if making a corrected-model trained-controller claim.

**Original Phase 3 EENS interpretation:** reproducible under the implemented
benchmark is **not** a claim of universally valid physically calibrated EENS.
Neither the HF-2 validator defect nor the battery-source discrepancy by itself
establishes that the historical within-model controller comparison must be
withdrawn. G0 must disclose the limitation, and G6 must correct inconsistent
thesis equations/claims (§1.8).

## Appendix A — Code changes required

| File | Change | Gate |
|---|---|---|
| `tools/policy_sanity_gate.py` | New — §1 | G0 |
| `tools/manifest.py` | New — §2 | G0 |
| `src/env/telecom_env.py` | `get_episode_stats()` (line 753) to aggregate `mask_info` counters. **No other change permitted** (§5.3) | G0 |
| `src/train/train_ablation_a5.py` | `start_run()` to wrap `learn()` (217 / 246) | G0 |
| `src/train/train_ablation_a6.py` | Same (245 / 259) | G0 |
| `src/train/train_ablation_a7.py` | Same (181 / 194) | G0 |
| `src/train/train_track_b.py` | Same (232 / 250) | G0 |
| `src/wrappers/drop_inventory_obs.py` | Instrument `step()` for G1; then repair per §5.2 | G1, G2 |
| `src/baselines/s_S_policy.py` + new sweep script | B1\* grid search, training window only | G3 |
| `src/eval/evaluate.py` | `--algo` to accept `dqn`, `a2c` (420, 482) | G4 |
| `src/train/train_dqn.py`, `train_a2c.py` | New, cloned from `train_ablation_a7.py` | G4 |
| Checkpoint callback | 50k / 100k / 200k for RQ-B | G4 |
| `src/train/train_shared_multisite.py` | New — 5-site shared policy + fine-tuning | G5b |
| `tools/latency_bench.py` | New — §8.1 | G5a |
| `tools/comparability_gate.py` | New — ECG-1..7 (§13.4) | G0 |
| `tools/g0_verify_energy.py` | **Replace v1.7 script with the independent, fail-closed v1.8 audit** (§1.4). Local shim test PASS; official repo-run JSON + execution log **pending** | **G0 PENDING** |
| `src/eval/evaluate.py` | For RL models, hard fail on missing/mismatched VecNormalize; hard fail on missing EENS or non-360-step eval instead of zero defaults (§1.7) | G0 |
| Thesis §3.2, Eq 3.1 | Replace with implemented energy-balance form (§1.5); highlight as revised | G6 |
| Thesis §3.6 / §3.7 / §4.3 / §4.4 / §4.7 / §8.4 | Document source-vs-implemented battery discharge factor, fuel reserve vs minimum DG ON duration, stockout term inactive, train/test tank sizing, limitations (§1.8) | G6 |
| Thesis Table 4.1 and protocol §7.4b / §8.3 | Replace estimated 70,663 with **measured 139,527** training-policy parameters; distinguish actor-only inference size | G6 |
| `results/phase3r/g0_energy_audit.json` | New: per-episode accounting audit, historical CSV checks, 10-site tank table and source code hashes (§1.4) | G0 |
| `config/phase3_reference.yaml` | New — frozen reference configuration (§13.2) | G0 |
| `tools/reference_reproduction.py` | New — the pilot (§13.8) | G0 |
| LaTeX preamble | `\revised{}` macro + change-bar package; **2-page prototype first** (§11.5) | G6 |
| Thesis certificate | Date July 2026 → November 2026 | G6 |
| `README.md` | Run commands for all new components; fresh-clone test (§11.8) | G6 |
| Presentation deck | Add Phase 1 + Phase 2 recap slide; add training/inference time slide; backup slides for all new experiments (§11.9) | G6 |
| Rebuttal appendix | Phase 2 institutional format, embedded in the thesis PDF (§11.6–11.7) | G6 |

## Appendix B — Defects found to date

| # | Defect | Status | Found |
|---|---|---|---|
| 1 | TrackB evaluated at 360 steps vs Track A's 720 | Fixed pre-submission | Phase 3 |
| 2 | `--all_sites` used `args.site` instead of the loop variable | Fixed pre-submission | Phase 3 |
| 3 | `DispatchOnlyEnv` sets `combined = 0` on infeasibility, overriding both components | **Suspected — G1** | 2026-09-26 |
| 4 | `mlflow.start_run()` after `model.learn()` in four trainers → zero-duration timings | **Open — G0 fix** | 2026-10-04 |

All four motivated the PSG and manifest. **Additional 2026-10-22 audit
findings are documented separately in §1.6:** inadequate v1.7 energy-audit
instrumentation (now replaced), stockout **metric/penalty inactive** in the
historical evaluation (not an EENS aggregation bug), train/test slice-dependent
tank, source-vs-implemented battery parameter meaning, incorrect neural
parameter counts and inaccurate minimum DG-runtime claim. These are **not all
runtime defects** and should not be lumped together or claimed to require
retraining without an impact assessment.
