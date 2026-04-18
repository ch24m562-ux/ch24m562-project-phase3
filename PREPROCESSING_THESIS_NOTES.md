# PREPROCESSING OBSERVATIONS FOR THESIS
# Key Findings and How to Document Them

**Date:** Feb 20, 2026  
**Context:** ITU/Zindi dataset preprocessing completed  

---

## Observation 1: Partial Week in 60-Day Episodes

### **The Math**
- Template: 168 hours (1 week)
- Episode: 1440 hours (60 days)
- **1440 = 8 × 168 + 96** (not evenly divisible)
- Last 96 hours use only first 96 hours of template

### **Impact**
- Slight difference between template outage % and expanded outage %
- Difference is <1% and mathematically correct
- Now handled with exact analytical calculation (no warnings)

### **For Thesis (Chapter 5: Data Processing)**

> "The 168-hour outage template is expanded to 1440-hour episodes through cyclic repetition. Since 1440 = 8×168 + 96, the final 96 hours of each 60-day episode use only the first 96 hours of the weekly template. This partial week results in a slight (<1%) difference between the template outage percentage and the expanded pattern percentage, which is mathematically correct and does not affect policy learning."

---

## Observation 2: Renewable-Surplus Site (Site8)

### **The Data**
```
Site8:
- Mean load:  2.14 kWh/h
- Mean solar: 3.24 kWh/h
- Coverage:   151.6%
- Outage:     32.8%
- Classification: Easy (score=1)
```

### **What This Means**
- Solar generation exceeds load consumption on average
- Site is **energy-positive** (net producer)
- Represents oversized solar installation or very low load
- Diesel likely near-zero usage (except night + grid outage)

### **Is This Valid?**
**YES.** This can occur when:
- Telecom tower has very light load (2.14 kWh/h is small)
- Solar array is oversized for backup redundancy
- Good solar resource location

### **Modeling Implications**
- Battery frequently saturates (SoC → 1.0)
- Excess solar is curtailed (wasted) when battery full
- Diesel rarely needed (only night + outage overlap)
- **Great test case** for policy robustness

### **For Thesis (Chapter 5: Dataset Characteristics)**

> "One site (Site8) exhibits net-positive energy balance, with mean solar generation (3.24 kWh/h) exceeding mean load (2.14 kWh/h) by 51%. This represents an oversized solar installation relative to load, a scenario that can occur with telecom towers having redundant renewable capacity. Excess generation is modeled as curtailed when battery storage saturates. We include this site in the test set as a special evaluation case to assess policy behavior under energy-surplus conditions, where diesel usage should be near-zero."

### **For Thesis (Chapter 6: Results)**

**Add evaluation scenario:**
```
Table: Policy Performance on Renewable-Surplus Site (Site8)

Metric                  | RL-Inv | Track B | RB Baseline
------------------------|--------|---------|-------------
Diesel consumption (L)  | X.X    | X.X     | X.X
Battery cycles          | X.X    | X.X     | X.X
Solar curtailment (%)   | X.X    | X.X     | X.X
EENS (kWh)             | X.X    | X.X     | X.X
```

**Expected finding:** Diesel near-zero, high battery cycling, significant curtailment.

---

## Observation 3: Site Diversity (Training Set Selection)

### **Selected Training Sites**
```
Hard:   site5  (Solar: 34.0%, Outage: 48.3%, Score: 4)
Medium: site7  (Solar: 50.8%, Outage: 37.8%, Score: 2)
Easy:   site1  (Solar: 45.6%, Outage: 20.0%, Score: 1)
```

### **Rationale**
- **Outage range:** 20% → 48% (2.4× spread)
- **Solar range:** 34% → 51% (1.5× spread)
- **Load range:** 4.55 → 8.32 kWh/h (diverse)
- **Geographic diversity:** Ensures policy doesn't overfit to specific conditions

### **Why Not Site8?**
- Site8 is an outlier (151% solar coverage)
- Better as **test case** than training example
- Prevents training bias toward energy-surplus scenarios

### **Test Set (7 Sites)**
- site2, site3, site4, site6, **site8**, site9, site10
- Includes 1 surplus (site8), 2 hard (site2, site5), 5 medium/easy
- Good coverage of difficulty spectrum

### **For Thesis (Chapter 5: Experimental Setup)**

> "We select three training sites to maximize diversity: Site5 (hard: high outage, moderate solar), Site7 (medium: balanced), and Site1 (easy: low outage, moderate solar). This selection spans outage rates from 20% to 48% and solar coverage from 34% to 51%, ensuring the policy learns robust control strategies across varied operational conditions. Seven sites are held out for testing, including Site8 (renewable-surplus) as a special evaluation case."

---

## Observation 4: Load Variability Across Sites

### **Load Range**
```
Lowest:  site8 (2.14 kWh/h) — Small rural
Highest: site2 (9.66 kWh/h) — Large urban
Ratio: 4.5× difference
```

### **Implication**
- **Battery sizing varies:** 16.2 to 41.0 kWh across sites
- **DG sizing varies:** 10.8 to 18.0 kW across sites
- Policy must adapt to different battery-to-load ratios

### **For Thesis**
> "Sites exhibit significant load diversity, ranging from 2.14 to 9.66 kWh/h (mean hourly), representing small rural towers to large urban installations. This diversity tests the policy's ability to adapt control strategies based on site-specific battery capacity and DG ratings, which are included in the state representation."

---

## Summary: What Changed in Final Preprocessing

### **Fixes Applied**
1. ✅ **Outage mismatch:** Changed to exact analytical calculation
2. ✅ **Surplus flag:** Added `is_surplus` column to classification
3. ✅ **Special note:** Surplus sites flagged in output

### **No Changes Needed**
- ❌ Classification logic (correctly marks site8 as "Easy")
- ❌ Training site selection (site1, site5, site7 is optimal)
- ❌ Difficulty scoring (solar coverage doesn't need capping)

### **Thesis Documentation Added**
- Partial week explanation
- Renewable-surplus site handling
- Training set rationale
- Load diversity note

---

## Evaluation Plan Update

### **Standard Evaluation (All Test Sites)**
- 7 test sites: site2, site3, site4, site6, site8, site9, site10
- Metrics: EENS, diesel, cost, violations, uptime
- Report mean ± std

### **Special Evaluation: Renewable-Surplus**
- Focus on Site8 explicitly
- Additional metrics:
  - Solar curtailment %
  - Battery saturation hours
  - Diesel usage (should be near-zero)
  - Policy's handling of surplus

### **Thesis Table Template**

```
Table: Zero-Shot Generalization Performance

Test Site | Difficulty | EENS (kWh) | Diesel (L) | Cost (₹) | Violations
----------|------------|------------|------------|----------|------------
site2     | Hard       | X.X ± Y.Y  | A ± B      | C ± D    | E
site3     | Medium     | X.X ± Y.Y  | A ± B      | C ± D    | E
site4     | Medium     | X.X ± Y.Y  | A ± B      | C ± D    | E
site6     | Easy       | X.X ± Y.Y  | A ± B      | C ± D    | E
site8*    | Easy (surplus) | X.X ± Y.Y | A ± B   | C ± D    | E
site9     | Medium     | X.X ± Y.Y  | A ± B      | C ± D    | E
site10    | Medium     | X.X ± Y.Y  | A ± B      | C ± D    | E

*Renewable-surplus site (solar > load)
```

---

**END OF PREPROCESSING OBSERVATIONS**

Reference this document when:
1. Writing Chapter 5 (Methodology)
2. Writing Chapter 6 (Results)
3. Preparing viva defense
4. Responding to "why this site selection?" questions
