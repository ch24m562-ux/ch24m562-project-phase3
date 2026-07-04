EV15b: Heavy-tail lognormal sensitivity evaluation
sigma     : 0.8
OutDir    : results/sensitivity/lognormal_sigma08
Companion : results/sensitivity/lognormal/ (EV15, sigma=0.5)
Policies  : rlinv, b1, mpc
Sites     : all 10
Seeds     : 42, 123, 777, 7, 13, 21, 99, 314, 500, 999
Episodes  : 10
Date      : 2026-07-02

Distribution properties vs EV15:
  sigma=0.5 (EV15):  P(delivery>2x mean)~5%,  99th pct ~2.9x mean
  sigma=0.8 (EV15b): P(delivery>2x mean)~10%, 99th pct ~4.8x mean
  Means are MATCHED across scenarios (same as EV15).

Do NOT mix these files with EV15 (lognormal/) results.
Report separately: EV15 = moderate variability, EV15b = heavy tail.
