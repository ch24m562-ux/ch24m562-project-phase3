# MASTER_EXPERIMENT_LOG
**Version:** 1.0 (Frozen – Phase 3)

---

# Purpose

This document provides the canonical inventory of all experiments conducted in this thesis.

It distinguishes:

- the experiments required to validate the primary scientific contribution,
- robustness and sensitivity analyses performed to strengthen confidence in the proposed framework,
- exploratory extensions developed during Phase 3, and
- implementation artefacts that were developed but are not part of the primary experimental evaluation.

This document is the single reference for:

- Methodology Chapter
- Experimental Results Chapter
- Viva preparation
- Future journal publications

---

# PART 1 — Core Thesis Experiments

These experiments establish and validate the primary contribution of the thesis.

| ID | Experiment | Scientific Purpose | Canonical Evidence | Status | Thesis |
|----|------------|-------------------|-------------------|--------|--------|
| **EXP-01** | Primary Comparative Evaluation | Compare RLInv against rule-based, RL-based and optimisation-based baselines under four logistics scenarios. | `master_summary.csv`, `fig_eens_comparison_v4.png` | ✅ Frozen | Results |
| **EXP-02** | Ablation A5 – Inventory Observation Removed | Quantify the contribution of explicit inventory information within the observation space. | `fig_ablation_h3_v3.png` | ✅ Frozen | Results |
| **EXP-03** | Ablation A6 – Joint Replenishment Removed | Isolate the contribution of jointly learned replenishment while keeping the dispatch architecture unchanged through externally managed $(s,S)$ ordering. | `fig_ablation_h1_v3.png` | ✅ Frozen | Results |
| **EXP-04** | Ablation A7 – Action Masking Removed | Evaluate the contribution of action masking by removing masking while retaining the joint replenishment and dispatch architecture. | `fig_ablation_h2_v3.png` | ✅ Frozen | Results |
| **EXP-05** | Behavioural Policy Analysis | Explain how RLInv adapts ordering and dispatch decisions through representative policy trajectories. | Behaviour summaries, trajectory figures | ✅ Frozen | Results / Discussion |
| **EXP-06** | Optimisation Baseline Evaluation (MPC / Oracle-MPC) | Compare RLInv against receding-horizon optimisation using practical and oracle forecasts to distinguish optimisation limits from forecasting limits. | MPC evaluation summaries | ✅ Frozen | Results |

---

# MPC Implementation Note

The MPC framework was implemented with three forecast modes:

1. **Persistence Forecast MPC**
   - Practical deployable optimisation baseline.
   - Included in the primary thesis evaluation.

2. **Oracle-MPC**
   - Assumes perfect future information.
   - Included as an upper reference bound.

3. **Forecast-MPC**
   - Uses the learned 24-hour forecasting module (`forecast_cache_H24.pkl`).
   - Implemented and evaluated during development.
   - Not included in the primary thesis comparison because it did not alter the scientific conclusions.

---

# PART 2 — Robustness, Sensitivity and Reviewer Extensions

These experiments strengthen confidence in the proposed framework but do not define the primary contribution.

---

## A. Reward Sensitivity

| ID | Experiment | Scientific Purpose | Evidence | Motivation |
|----|------------|-------------------|----------|------------|
| **RS-01** | Reward Function Sensitivity | Evaluate robustness to reward coefficient selection and discount factor. | Reward sensitivity summaries | Reviewer feedback (Reward justification) |

---

## B. Infrastructure Sensitivity

| ID | Experiment | Scientific Purpose | Evidence | Motivation |
|----|------------|-------------------|----------|------------|
| **TS-01** | Tank Capacity Sensitivity | Evaluate robustness under different diesel storage capacities. | Tank sensitivity summaries | Reviewer feedback |

---

## C. Distributional Robustness

| ID | Experiment | Scientific Purpose | Evidence | Motivation |
|----|------------|-------------------|----------|------------|
| **DR-01** | Geometric Lead-Time Distribution | Training and evaluation baseline. | Main experiments | Reference |
| **DR-02** | Lognormal (σ = 0.5) | Evaluate robustness under heavy-tailed delivery delays. | Distribution robustness | Reviewer feedback |
| **DR-03** | Lognormal (σ = 0.8) | Evaluate robustness under severe delay variability. | Distribution robustness | Reviewer feedback |
| **DR-04** | Weibull (k = 2) | Evaluate robustness under increasing-hazard supplier behaviour. | Distribution robustness | Reviewer feedback |

---

## D. Additional Robustness

| ID | Experiment | Scientific Purpose | Evidence | Status |
|----|------------|-------------------|----------|--------|
| **RB-01** | Stochastic Grid Evaluation | Assess policy robustness under stochastic grid availability. | Stochastic grid summaries | Optional (Appendix / Methodology reference) |

---

# PART 3 — Exploratory Extension

These experiments explore future research directions and are intentionally separated from the primary scientific contribution.

| ID | Experiment | Scientific Purpose | Evidence | Motivation |
|----|------------|-------------------|----------|------------|
| **EXT-01** | RLInv-DA (Delay-Aware RL) | Investigate whether explicit estimation of supplier delay improves replenishment decisions beyond RLInv. | RLInv-DA evaluation | Reviewer feedback |

---

# Canonical Evaluation Protocol

Unless otherwise stated, all primary comparisons use:

| Item | Value |
|------|------|
| Telecom sites | 10 |
| Training seeds | 10 |
| Evaluation episodes | 10 per seed |
| Operating scenarios | Normal, Delayed, Monsoon, Extreme |
| Primary metrics | EENS, Diesel Consumption, Grid Energy, Ordering Behaviour, Inventory Dynamics |
| Statistical methodology | Canonical paired analysis (Phase 3) |

---

# Scientific Classification

## Primary Scientific Contribution

- RLInv formulation
- Primary comparative evaluation
- MPC comparison
- A5, A6 and A7 ablations
- Behavioural analysis

---

## Robustness Evidence

- Reward sensitivity
- Tank capacity sensitivity
- Distributional robustness
- Stochastic grid robustness

---

## Exploratory Extension

- RLInv-DA

---

# Canonical Data Sources

| Evidence | Canonical Source |
|-----------|-----------------|
| Primary evaluation | `master_summary.csv` |
| H1 | `fig_ablation_h1_v3.*` |
| H2 | `fig_ablation_h2_v3.*` |
| H3 | `fig_ablation_h3_v3.*` |
| Behaviour | Behaviour summaries + trajectory figures |
| Reward sensitivity | Reward sensitivity summaries |
| Tank sensitivity | Tank sensitivity summaries |
| Distribution robustness | Distribution summaries |
| MPC | MPC evaluation summaries |
| Oracle-MPC | Oracle evaluation summaries |
| RLInv-DA | RLInv-DA summaries |

---

# Freeze Status

**Phase 3 Results Frozen**

This document reflects the final experimental design used in the thesis.

Future modifications should only occur if:

- a reproducibility issue is discovered,
- a numerical error is identified, or
- additional journal experiments are conducted beyond the submitted thesis.

No further structural changes to the thesis should originate from this document.