# Reviewer Comment #2 — Backup Slides Checklist
"The presentation can include more technical details and results,
at least as backup slides so that there is more clarity on the work executed"

## What the reviewer meant
During the Phase 2 viva, high-level results were shown without enough technical
depth. The examiner wanted to probe deeper but the slides didn't support it.
Backup slides = slides you don't present but can flip to when a specific question
is asked. Organised into 4 sections so you know immediately where to go.

---

## Main slides (already planned — not backup)
- [ ] Problem statement + motivation (why inventory coupling matters)
- [ ] System architecture (CMDP, state, action, reward)
- [ ] Ablation ladder table (11 policies, 4 scenarios)
- [ ] fig_lead_sensitivity (EENS vs lead time)
- [ ] fig_tank_sensitivity (RLInv advantage vs tank size)
- [ ] fig_trajectory (behavioural mechanism)
- [ ] Section 12: RLInv vs Oracle paired CI table
- [ ] Lognormal sensitivity verdict (PASS)

---

## Backup slides — Section A: Methodology

### B1. Environment details
- State vector (10 dimensions, what each means)
- Action space (Discrete 6 = 2 DG × 3 order)
- Reward function with exact weights
- Tank capacity, SOC limits, episode length

### B2. Reward weight justification
- Karnataka SERC tariff (Rs 9.5/kWh)
- IOCL diesel price (Rs 27.3/kWh effective)
- Lambda=100 economic grounding from GSMA data
- Sensitivity sweep results (all_results_clean.csv)

### B3. Baseline definitions table
- Columns: Policy | Ordering rule | Dispatch rule | Information used
- Covers: B0, B1, MPC, Oracle, TrackB, A5, A6, A7, RLInv, Multi

### B7. Training details
- MaskablePPO, 400k steps, 10 seeds per site
- Vecnorm, gamma=0.995, learning rate
- MLflow training curves (if produced)

---

## Backup slides — Section B: Experimental Results

### B4. Per-site breakdown table (merged with hard site characteristics)
- 10 sites × 4 scenarios, colour-coded by EENS
- Hard sites (site2/site5/site7): outage rate, solar coverage
- Shows hard sites dominate signal -- physical justification included
- Source: check_eens_sanity.py Section 6

### B5. Section 12 full results
- Paired CI table: all 4 scenarios, n_pairs=100
- Oracle vs RLInv advantage with confidence intervals
- Significance flags (normal/delayed/monsoon: YES; extreme: borderline)

### B6. Section 13 finding (Oracle vs MPC)
- Near-zero benefit of perfect information
- "Ordering bottleneck, not dispatch quality"
- Weighted objective explanation (inv_penalty=500 vs lam_unmet=100)

### B8. Lognormal sensitivity details
- Setup: 1200 runs, sigma=0.5, same mean as geometric
- Results table: rankings preserved all 4 scenarios
- Verdict: PASS. Mean |delta EENS|: rlinv=4.56, b1=7.62 kWh

---

## Backup slides — Section C: Sensitivity & Robustness

### B9. Tank sensitivity details
- 5 tank sizes (24/48/72/144/336h), 3 hard sites, monotone decrease
- Advantage vanishes at 336h — proves inventory coupling mechanism
- fig_tank_sensitivity (already committed)

### B10. MPC/Oracle investigation
- Why Oracle doesn't always beat MPC
- Weighted objective makes Oracle conservative on dispatch
- Hour-by-hour trace evidence from Oracle investigation session

### B11. Diesel/cost sanity check (Section 14)
- Two-group table: conservative (~190-199 kWh) vs reliability-first (~204-206 kWh)
- RLInv uses ~3-4% more diesel than B1
- EENS gains are structural, not purchased by diesel inflation

---

## Backup slides — Section D: Methodology Justification

### B13. How the three Phase 3 figures support the methodology (NEW)
One slide, three figures side by side:

| Figure | One-sentence takeaway |
|---|---|
| fig_lead_sensitivity | RLInv degrades more gracefully than B1 as delivery delays increase |
| fig_tank_sensitivity | RLInv's advantage vanishes when inventory is not a constraint, confirming the mechanism |
| fig_trajectory | RLInv maintains inventory through proactive ordering; B1 depletes and fails |

This is the strongest backup slide — three pieces of evidence, one visual, one story.

---

## Rules for building backup slides
1. Title each slide as a QUESTION, not a topic
   - Good: "Why was lambda=100 chosen?"
   - Bad:  "Reward sensitivity"
2. Note the evidence source on every slide
3. Number slides B1-B13 within their section groups
4. Main deck: 12-15 slides. Backup deck: 13 slides across 4 sections.

## Status
- All 13 backup slides have data/evidence already in the repo
- This is entirely a slide-building task — no new experiments needed
- Build during thesis writing week alongside the main presentation
