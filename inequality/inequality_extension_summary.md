# Inequality Extension — Summary

**Owner:** Lynnard  
**Status:** Complete

---

## What This Does

Extends Method 1 by replacing mean wage outcomes with within-occupation wage inequality measures. Instead of asking "did high-exposure occupations see lower mean wages?", we ask "did the wage spread *within* high-exposure occupations change?"

---

## Pipeline Changes

| File | Change |
|---|---|
| `01_process_oews.py` | Added `a_pct10`, `a_pct25`, `a_median`, `a_pct75`, `a_pct90` to output columns |
| `02_merge_exposure.py` | No code change — columns pass through automatically. Doc updated. |
| `03_inequality_did.py` | **New standalone script.** Run this after 01 and 02. |

**To run:**
```bash
python3 01_process_oews.py   # only if you haven't re-run after the percentile update
python3 02_merge_exposure.py # same
python3 03_inequality_did.py
```

Outputs go to `inequality_results/`.

---

## Specification

Identical to Method 1, outcome variable swapped:

```
log(w90/w10)_ot = α + β(dv_beta_o × post_t) + γ_o + δ_t + ε_ot
```

Three outcomes: `log_w90_10` (overall spread), `log_w90_50` (upper-tail), `log_w50_10` (lower-tail). Same sample filters, same clustered SEs, same treatment variable.

---

## Results

| Outcome | β | SE | p | |
|---|---|---|---|---|
| Log Wage Ratio (90/10) | +0.0228 | 0.0128 | 0.074 | * |
| Log Wage Ratio (90/50) | +0.0265 | 0.0094 | 0.005 | *** |
| Log Wage Ratio (50/10) | +0.0006 | 0.0098 | 0.950 | |

**The effect is entirely upper-tail.** High-exposure occupations saw the 90th percentile pull away from the median (significant at 1%), while the median-to-bottom relationship was unchanged. LLM exposure is **disequalizing within occupations**, driven by top earners capturing productivity gains. Moving from low to high exposure tercile widens the 90/50 ratio by ~1.3 percentage points.

---

## Notes for the Paper

- Pre-trend validation inherited from Method 4 event study — same occupations, same data
- `a_pct10` has more BLS suppression than `a_median`, causing a 149-observation gap between 90/50 and 50/10 regressions — mention in one sentence
- The pre-trend concern from Method 1 (tech-sector wage growth pre-ChatGPT) applies here too — note as limitation
- References: Acemoglu & Restrepo (2022) *Econometrica* for spec; Autor, Katz & Kearney (2008) *ReStat* for 90/50 vs 50/10 decomposition

## Notes for code

- The raw data path is currently tuned to work in the git repo. deleting "../" in:
- RAW_DIR = Path("../data/raw/oews") in 01_process_oews.py
- PROCESSED_DIR = Path("../data/processed") in 02_merge_exposure.py
- Can rewind it back to referencing data in the inequality folder, as it was initially designed to do. The raw data there were deleted to reduce repetitiveness.
