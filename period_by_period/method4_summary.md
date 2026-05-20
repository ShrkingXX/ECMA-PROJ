# Method 4: Period-by-Period Event Study
**LLMs and Labor Market Outcomes — Lynnard Z.**

---

## Why a Different Dataset

OEWS is annual (May snapshot) — too coarse for quarter-by-quarter dynamics. Method 4 uses **BLS CES** (monthly, aggregated to quarters) which allows estimating a separate treatment effect per quarter rather than pooling the entire post-ChatGPT period into one coefficient.

Because CES is industry-coded (not occupation-coded), the Eloundou exposure scores can't be used directly. Instead we use **Felten et al. (2021) AIIE** — an industry-level AI exposure index indexed by NAICS code, aggregated to CES supersectors.

---

## Data

| | Detail |
|---|---|
| **Outcome data** | BLS CES: 16 supersectors, 2017Q1–2026Q1, quarterly |
| **Exposure** | Felten AIIE aggregated to supersector level (mean of 4-digit NAICS within each supersector) |
| **Outcomes** | log employment, log avg hourly earnings, log wage bill |
| **Treatment timing** | ChatGPT launch = 2022Q4 (τ = 0) |
| **Reference period** | τ = −1 (2022Q3) — all β_τ relative to this quarter |

AIIE is standardized (mean ≈ 0). High exposure: Financial Activities (1.54), Information (1.27). Low exposure: Construction (−1.00), Mining (−1.01).

---

## Model

$$\log y_{it} = \alpha + \sum_\tau \beta_\tau \cdot (\text{AIIE}_i \times \mathbf{1}[t=\tau]) + \gamma_i + \delta_t + \varepsilon_{it}$$

- $i$ = supersector, $t$ = year×quarter
- $\gamma_i$ = industry fixed effects (absorb time-invariant level differences)
- $\delta_t$ = quarter fixed effects (absorb economy-wide shocks)
- $\beta_\tau$ = differential effect of high vs low AI exposure in quarter τ

Estimated via **TWFE OLS** (linearmodels, clustered SEs at supersector level). Pre-period $\beta_\tau \approx 0$ validates parallel trends; post-period $\beta_\tau$ shows dynamic treatment effects.

---

## Sample Specification

**Full sample (05, appendix):** 2017Q1–2026Q1. COVID (2020–2021) created large differential shocks across industries — Leisure/Hospitality collapsed while Finance/Information held steady — contaminating the pre-trend. Pre-period coefficients show visible trends unrelated to ChatGPT.

**Main spec (06):** 2021Q1–2026Q1. Drops pre-COVID and COVID years. Pre-period coefficients are much flatter, parallel trends validated. This is the headline result.

---

## Results Summary

**Log Wages (strongest result)**
- Pre-period: essentially flat (mean |β| = 0.002) ✓ parallel trends
- Post-period: immediate drop after 2022Q4, trough ~−0.008 at 2023Q2–Q3, gradual recovery toward zero by 2025
- τ = 1 and τ = 2 statistically significant (p < 0.05)
- Interpretation: high-AIIE industries saw relative wage compression post-ChatGPT, partially reversing by 2025

**Log Employment**
- Pre-period: mild downward trend (post-COVID normalization)
- Post-period: stable negative drift ~−0.007 to −0.010, persistent through 2025
- Not statistically significant but consistent direction

**Log Wage Bill**
- Combines both channels; most persistent negative effect (~−0.013 average)
- Stays negative even as wages recover — employment channel doing more work long-run

---

## Robustness

| Check | Result |
|---|---|
| Z-scored AIIE (07) | Identical shape, coefficients scaled by 0.756 (= SD of AIIE). Confirms results not sensitive to exposure scaling. |
| Leave-one-out (08) | All 13–14 supersector drops show same pattern. No single industry driving results. |

---

## Limitations

- Only 13–14 clusters → clustered SEs may be unreliable; treat CIs as indicative
- Industry-level aggregation loses within-supersector heterogeneity (e.g. Finance combines high and low AI-exposed roles)
- Felten AIIE measures pre-GPT-4 AI broadly, not LLM-specific exposure
- Mild pre-trend in employment/wage bill (post-COVID recovery artifact)

---

## Script Inventory

| Script | Purpose |
|---|---|
| `03_process_ces_v2.py` | Download + clean CES, output quarterly panel |
| `04_merge_felten.py` | Aggregate Felten AIIE to supersectors, merge onto CES |
| `05_event_study.py` | Full 2017 sample baseline (appendix) |
| `06_2021start.py` | **Main result** — 2021Q1 sample, aiie_score |
| `07_robustness_zscore.py` | Robustness — z-scored AIIE |
| `08_loo_robustness.py` | Robustness — leave-one-out |
