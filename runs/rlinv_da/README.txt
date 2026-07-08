RLInv-DA Experiment
===================
Purpose: Reviewer Comments #9 and #10 exploratory extension.

Environment:
  Two-state supplier regime (Normal <-> Disrupted)
  Transition probabilities: p_disrupt=0.15, p_recover=0.30 (per order)
  Disrupted mean = 2x normal mean lead time

Lead distribution: Lognormal, sigma=0.5, matched scenario means

Training:
  Sites:  site2, site5, site7 (hard sites)
  Seeds:  42, 123, 777
  Steps:  400k per run
  Total:  18 runs (~3.6h)

Policies:
  RLInv-base-regime: supplier regime ON, EWMA OFF (logdir: base_regime/)
  RLInv-EWMA:        supplier regime ON, EWMA ON  (logdir: ewma_regime/)

Comparison metric: EENS under delayed and monsoon scenarios

Scientific question:
  Does adaptive lead-time estimation (EWMA) improve inventory decisions
  under persistent supplier disruptions, compared to a policy that
  experiences the same regime but without delay tracking?

Status: Exploratory extension only.
        Does NOT replace main thesis results.
        Results reported separately from core RLInv evaluation.
