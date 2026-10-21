# G6 — Thesis correction checklist
### Findings from the G0 audit that require writing, not re-experimenting
Created 2026-10-22 · track against this file · tick as done

**None of these change a reported result.** They are places where the thesis
*describes* the model differently from how the code *behaves*, plus limitations
to disclose. Fixing them makes the description match the code — which matters
because the examiner can ask to run it in the code viva.

Every item below must also be **highlighted as revised content** (formal
resubmission requirement) and get a row in the **rebuttal appendix**.

---

## A. Model description does not match the implementation

| # | Where | What is wrong | What to write | Evidence |
|---|---|---|---|---|
| A1 | **§3.2, Eq 3.1** (p. 28) | States `P_Load = P_PV − P_Bat + P_Grid + DG·P_rated` as a constraint that "must be satisfied". Omits **unserved load**, **curtailment**, and asserts **full rated DG output** when the generator is on | `P_Load − P_Unserved = P_PV,avail − P_Curtail − P_Bat + P_Grid + P_DG`, where `P_DG ≤ DG_t·P_rated` | Verified numerically, protocol §1.4–1.5 |
| A2 | **§3.2 assumptions** (repeat in §3.7 if present) | Claims "a minimum DG up-time of 2 h whenever the generator is switched on". **Not implemented** — no up-time state exists in `telecom_env.py` | What exists is a 2-hour **fuel reserve** (`min_fuel = rated × 2.0 × 1.2`), not an enforced ON duration. Correct the claim; do not add the constraint | `grep` for up-time state returns nothing |
| A3 | **Table 4.1** (§4.2.5) | Parameter count given as **70,663**, assuming a shared actor/critic trunk. SB3 builds **separate** trunks | **139,527** total. Use the role-based breakdown: decision network (actor) 70,406 · critic 69,121 · total 139,527 | Confirmed by checkpoint inspection **and** by constructing the SB3 models |
| A4 | **§4.4 reward** | Does not disclose that the stockout-cliff term never activated | **Both** γ_r (stockout) and μ (violation) are **zero across all 44,000 evaluation episodes**. The effective evaluation reward was α·diesel + β·grid + λ·unmet only. State this; note training may differ (wider initial-inventory range, 720-step episodes) | `master_summary.csv`: `stockout_events` sum = 0, `violations` sum = 0 |

## B. Limitations to disclose (§4.7 deviations table and §8.4)

| # | Topic | What to state |
|---|---|---|
| B1 | **Battery coefficients** | Use the approved wording verbatim: *"The historical Phase 3 results are reproducible under the implemented battery convention. The discrepancy with the source scoring convention limits physical interpretation and may affect both absolute EENS and relative policy performance under alternative battery dynamics. The historical primary benchmark will remain unchanged for Phase 3-R comparability; any battery-physics sensitivity analysis will be reported separately."* **Do not** call 0.35/1.20 "C-rates" — unestablished. **Do not** claim uniform rules imply unchanged rankings — logically invalid |
| B2 | **Tank capacity** | `TelecomEnv` recomputes `C = 72 × mean(load)` from whatever slice it receives, so train and test tanks differ — up to **6.93%** (site8); site2 −0.11%, site5 +0.29%. `d_bar` also shifts, changing reward normalisation. Do not state plant capacity is fixed across the split |
| B3 | **Reproducibility scope** | The checkpoint replay covers **site5, seed 42, two policies, four scenarios** and matches on EENS, diesel and grid. It is a reproduction of principal energy metrics, **not** bit-for-bit across all recorded fields (`orders_placed` differs by 1 in one cell; `mean_inv_pct` by ≤0.023) |
| B4 | **Evaluation randomness** | Policies share site, scenario, exogenous traces and initial inventory per `(seed, episode)`, but **delivery draws diverge** once ordering behaviour diverges — intended, since policies order at different times. Pairing is on partially matched conditions |

## C. Claim-wording corrections

| # | Where | Fix |
|---|---|---|
| C1 | **§7.2** | "RLInv achieves the lowest EENS across all four scenarios" is true **only within Table 7.1's six-policy set**. Across all eleven evaluated policies RLInv is lowest in **one of four** (A7 lower under Delayed, A6 under Monsoon, multi under Extreme). Use the standardised sentence from protocol §11.1 everywhere — abstract, Ch.7, Ch.8, slides |
| C2 | **Wherever mpc_forecast appears** | Its seed set is `[1,2,3,4,5,6,7,42,123,777]`, different from the standard ten. Disclose |
| C3 | **§8.4** | Remove "Single RL algorithm family" and "No held-out-site transfer evaluation" once G4 and G5b are done; replace with whatever the new work reveals |

## D. Formal resubmission requirements

| # | Item |
|---|---|
| D1 | **Highlight all revised content** — one convention, `\revised{}` macro + margin change-bars. **Prototype on 2 pages first** (line break, page break, table cell, caption, heading; check greyscale) before writing new chapters |
| D2 | **Rebuttal appendix embedded in the thesis PDF**, in the proven Phase 2 format: candidate details → verbatim Comment → Response → numbered *Changes in Thesis* with exact section/table/figure numbers |
| D3 | **Certificate date** July 2026 → November 2026 (mentor signature already present and correct) |
| D4 | **Deck:** add a Phase 1 + Phase 2 recap slide (missing, and explicitly required); add a training/inference time slide (missing, explicitly required); backup slides for every new experiment |
| D5 | **README** updated for all new components + a fresh-clone reproduction test (examiner can ask to run the code) |

## E. New content from the experiments (not corrections)

| # | Item |
|---|---|
| E1 | New §7.7 — algorithm comparison (RQ-A final performance, RQ-B sample efficiency) with the mandatory scope statement |
| E2 | New §7.8 — scalability, as **three separate statements**: computational, training, deployment/generalisation. Never a blanket "RLInv is scalable" |
| E3 | §4.5 + §7.2 — B1\*-Deployable, plus the **Baseline Taxonomy and Fairness table** |
| E4 | §6.3 or §7.2 — TrackB correction note, if the defect is confirmed |
| E5 | Scalability **Limitations** subsection — five dimensions, each with a named limit, including the untested multi-tower coordination row |
