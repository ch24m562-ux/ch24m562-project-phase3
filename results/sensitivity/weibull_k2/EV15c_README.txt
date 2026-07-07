EV15c: Weibull(k=2) lead-time sensitivity evaluation
lead_dist : weibull
lead_k    : 2
OutDir    : results/sensitivity/weibull_k2
Companion : results/sensitivity/lognormal/     (EV15,  sigma=0.5)
            results/sensitivity/lognormal_sigma08/ (EV15b, sigma=0.8)
Date      : 2026-07-04

Distribution properties at k=2 (Rayleigh distribution):
  Mean   : matches scenario mean exactly (by construction)
  CV     : ~0.52  (similar to lognormal sigma=0.5)
  P(T>2M): ~4.4%  (much lighter tail than geometric 13.5%)
  Hazard : increasing -- delivery more likely the longer you wait
  Models : SLA-governed supply chains

Interpretation:
  Weibull(k=2) is the OPTIMISTIC scenario -- less heavy-tailed / SLA-like increasing-hazard scenario.
  Lognormal sigma=0.8 is the PESSIMISTIC scenario -- heavy tail.
  Together they bracket the realistic space around geometric.

Do NOT mix with EV15 or EV15b results.
Report separately as EV15c.
