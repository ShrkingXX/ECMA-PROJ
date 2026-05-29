"""
================================================================================
INEQUALITY EXTENSION: DiD on Within-Occupation Wage Percentile Ratios
================================================================================
Project : LLMs and Labor Market Outcomes
Course  : Econometrics & ML — Spring 2025, University of Chicago
Person  : Lynnard (Inequality Extension)

PURPOSE
-------
Estimates the effect of LLM exposure on within-occupation wage inequality
using BLS OEWS wage percentile data. Extends Method 1 (OLS DiD) by replacing
mean wage outcomes with three log percentile ratio outcomes that capture
the spread of the wage distribution within each occupation × year cell.

RESEARCH QUESTION
-----------------
Do occupations with higher structural LLM task exposure experience compression
or widening of within-occupation wage inequality following ChatGPT's release?
  β < 0 → high-exposure occupations saw wage compression  (equalizing)
  β > 0 → high-exposure occupations saw wage widening     (disequalizing)

OUTCOMES
--------
  log_w90_10   ln(a_pct90 / a_pct10)  — overall within-occupation wage spread
  log_w90_50   ln(a_pct90 / a_median) — upper-tail inequality
  log_w50_10   ln(a_median / a_pct10) — lower-tail inequality

  The 90/50 vs 50/10 decomposition identifies whether LLM effects operate
  through the top or bottom of the within-occupation distribution, following
  Autor, Katz & Kearney (2008).

SPECIFICATION
-------------
  log(w90/w10)_ot = α + β(dv_beta_o × post_t) + γ_o + δ_t + ε_ot

  Identical to Method 1 except for the outcome variable. Occupation FEs (γ_o)
  absorb time-invariant inequality levels; year FEs (δ_t) absorb aggregate
  macro shocks. β is the coefficient of interest.

  Standard errors clustered at the occupation level.

SAMPLE FILTERS
--------------
  Same as Method 1:
    balanced         == 1
    suppressed_wage  == False
    exposure_residual== 0
    dv_beta.notna()
    log_wage.notna()

  Additional filter for inequality outcomes:
    a_pct10, a_median, a_pct90 all non-null (drops suppressed percentile cells)

PRE-TREND VALIDATION
--------------------
Parallel trends for the inequality outcomes is inherited from the Method 1
event study, which establishes that high- and low-exposure occupations come
from the same OEWS data with the same occupation and year FE structure.
The pre-trend concern flagged in Method 1 (τ=-3, τ=-2 significant for
log_wage) applies here too and should be noted as a limitation.

INPUTS
------
  data/processed/oews_exposure_merged.csv   ← from 02_merge_exposure.py
  Requires a_pct10, a_pct25, a_median, a_pct75, a_pct90 columns.
  These are produced by 01_process_oews.py — confirm you have run the
  updated version that includes percentile columns before running this.

OUTPUTS
-------
  inequality_results/table_m5_summary.csv           β table, 3 outcomes
  inequality_results/table_m5_economic_magnitudes.csv  scaled effect sizes
  inequality_results/table_m5_residual_diagnostics.csv residual checks
  inequality_results/figure_m5_residuals_log_w90_10.png
  inequality_results/figure_m5_residuals_log_w90_50.png
  inequality_results/figure_m5_residuals_log_w50_10.png
  inequality_results/figure_m5_coef_plot.png         coefficient plot

REFERENCES
----------
  Acemoglu & Restrepo (2022) Econometrica — percentile-ratio DiD specification
  Autor, Katz & Kearney (2008) ReStat     — 90/50 vs 50/10 decomposition
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from linearmodels.panel import PanelOLS

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(SCRIPT_DIR, "data", "processed", "oews_exposure_merged.csv")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "inequality_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Outcomes ──────────────────────────────────────────────────────────────────

OUTCOMES = {
    "log_w90_10": "Log Wage Ratio (90/10)",
    "log_w90_50": "Log Wage Ratio (90/50)",
    "log_w50_10": "Log Wage Ratio (50/10)",
}


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1: Load and Inspect the Data
# ══════════════════════════════════════════════════════════════════════════════
print("=== TASK 1: Load and Inspect the Data ===")

df_raw = pd.read_csv(DATA_PATH)
print(f"Shape after loading: {df_raw.shape}")

# ── Apply sample filters ──────────────────────────────────────────────────────
sample = df_raw.copy()

sample = sample[sample['balanced'] == 1]
print(f"\nAfter balanced==1: {len(sample)} rows, {sample['occ_code'].nunique()} unique occupations")

sample = sample[sample['suppressed_wage'] == False]
print(f"After suppressed_wage==False: {len(sample)} rows, {sample['occ_code'].nunique()} unique occupations")

sample = sample[sample['exposure_residual'] == 0]
print(f"After exposure_residual==0: {len(sample)} rows, {sample['occ_code'].nunique()} unique occupations")

sample = sample[sample['dv_beta'].notna()]
print(f"After dv_beta.notna(): {len(sample)} rows, {sample['occ_code'].nunique()} unique occupations")

sample = sample[sample['log_wage'].notna()]
print(f"After log_wage.notna(): {len(sample)} rows, {sample['occ_code'].nunique()} unique occupations")

# ── Verify percentile columns are present ─────────────────────────────────────
pct_cols = ['a_pct10', 'a_pct25', 'a_median', 'a_pct75', 'a_pct90']
missing_pct = [c for c in pct_cols if c not in sample.columns]
if missing_pct:
    raise ValueError(
        f"Missing percentile columns: {missing_pct}\n"
        "Re-run 01_process_oews.py with the updated version that includes "
        "wage percentile columns, then re-run 02_merge_exposure.py."
    )
print(f"\nAll percentile columns present: {pct_cols}")

# ── Convert percentile columns to numeric ─────────────────────────────────────
for col in pct_cols:
    sample[col] = pd.to_numeric(sample[col], errors='coerce')

# ── Structural verification ───────────────────────────────────────────────────
print("\n--- Structural Verification ---")

years_present = sorted(sample['year'].unique().tolist())
print(f"Years present: {years_present}")

post1_years = sorted(sample[sample['post'] == 1]['year'].unique().tolist())
post0_years = sorted(sample[sample['post'] == 0]['year'].unique().tolist())
print(f"post==1 years: {post1_years}")
print(f"post==0 years: {post0_years}")

if sample['dv_beta'].between(0, 1).all():
    print("PASS: dv_beta range — all values in [0, 1]")
else:
    n_out = (~sample['dv_beta'].between(0, 1)).sum()
    print(f"WARNING: {n_out} dv_beta values outside [0, 1]")

# ── Percentile suppression summary ────────────────────────────────────────────
print("\n--- Percentile Column Missingness (suppression) ---")
for col in pct_cols:
    n_null = sample[col].isna().sum()
    pct    = 100 * n_null / len(sample)
    print(f"  {col}: {n_null} NaN ({pct:.1f}%)")

# ── Summary statistics ─────────────────────────────────────────────────────────
print("\n--- Summary Statistics (clean sample) ---")
print(sample[['dv_beta', 'log_wage', 'log_emp', 'a_pct10', 'a_median', 'a_pct90']].describe(
    percentiles=[0.25, 0.5, 0.75]
).round(4).to_string())

sample['exposure_tercile'] = pd.qcut(sample['dv_beta'], q=3, labels=['Low', 'Mid', 'High'])
tercile_summary = sample.groupby('exposure_tercile', observed=True)['dv_beta'].agg(
    n='count', mean='mean', min='min', max='max'
).round(4)
print("\n--- Exposure by Tercile ---")
print(tercile_summary.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2: Construct Inequality Outcomes and Interaction Term
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== TASK 2: Construct Variables ===")

# ── Log percentile ratio outcomes ─────────────────────────────────────────────
sample['log_w90_10'] = np.log(sample['a_pct90']) - np.log(sample['a_pct10'])
sample['log_w90_50'] = np.log(sample['a_pct90']) - np.log(sample['a_median'])
sample['log_w50_10'] = np.log(sample['a_median']) - np.log(sample['a_pct10'])

for outcome, label in OUTCOMES.items():
    n_valid = sample[outcome].notna().sum()
    n_null  = sample[outcome].isna().sum()
    print(f"{label}: {n_valid} valid, {n_null} NaN (suppressed percentile cells dropped at regression stage)")

# ── DiD interaction term ──────────────────────────────────────────────────────
sample['dv_beta_x_post'] = sample['dv_beta'] * sample['post']

violations_pre = ((sample['post'] == 0) & (sample['dv_beta_x_post'] != 0)).sum()
if violations_pre > 0:
    print(f"WARNING: {violations_pre} pre-period rows have dv_beta_x_post != 0")
else:
    print("\nPASS: dv_beta_x_post == 0 for all pre-period rows")

# ── Within-occupation variation in inequality outcomes ─────────────────────────
print("\n--- Within-Occupation Variation in Inequality Outcomes ---")
for outcome, label in OUTCOMES.items():
    within_std = sample.groupby('occ_code')[outcome].std()
    zero_var   = (within_std == 0).sum()
    print(f"{label}: mean within-std={within_std.mean():.4f}, "
          f"min={within_std.min():.4f}, max={within_std.max():.4f}")
    if zero_var > 0:
        print(f"  WARNING: {zero_var} occupations have zero within-variation (silently dropped by PanelOLS)")

# ── Panel index ────────────────────────────────────────────────────────────────
panel = sample.set_index(['occ_code', 'year'])
n_entities = panel.index.get_level_values('occ_code').nunique()
n_periods  = panel.index.get_level_values('year').nunique()
print(f"\nPanel index set: {n_entities} unique occupations, {n_periods} unique time periods")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3: Run Three Parallel OLS DiD Regressions
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== TASK 3: Run Three Parallel OLS DiD Regressions ===")

results = {}

for outcome, label in OUTCOMES.items():
    print(f"\n--- Regression: {label} ---")

    cols_needed = [outcome, 'dv_beta_x_post']
    n_before    = len(panel)
    reg_data    = panel[cols_needed].copy().dropna(subset=cols_needed)
    n_dropped   = n_before - len(reg_data)
    print(f"Dropped {n_dropped} rows with NaN in {outcome} or dv_beta_x_post")

    model = PanelOLS(
        dependent=reg_data[outcome],
        exog=reg_data[['dv_beta_x_post']],
        entity_effects=True,
        time_effects=True,
    )
    res = model.fit(cov_type='clustered', cluster_entity=True)
    results[outcome] = res

    print(res.summary)

    beta  = res.params['dv_beta_x_post']
    se    = res.std_errors['dv_beta_x_post']
    tstat = res.tstats['dv_beta_x_post']
    pval  = res.pvalues['dv_beta_x_post']
    r2w   = res.rsquared_within
    nobs  = res.nobs
    n_ent = res.entity_info.total

    print(f"β (dv_beta_x_post): {beta:.4f}")
    print(f"SE (clustered):     {se:.4f}")
    print(f"t-statistic:        {tstat:.4f}")
    print(f"p-value:            {pval:.4f}")
    print(f"Within-R²:          {r2w:.4f}")
    print(f"N observations:     {nobs}")
    print(f"N entities:         {n_ent}")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 4: Summary Table and Economic Magnitudes
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== TASK 4: Summary Table and Economic Magnitudes ===")

# ── Summary table ─────────────────────────────────────────────────────────────
rows = []
for outcome, label in OUTCOMES.items():
    res   = results[outcome]
    beta  = res.params['dv_beta_x_post']
    se    = res.std_errors['dv_beta_x_post']
    tstat = res.tstats['dv_beta_x_post']
    pval  = res.pvalues['dv_beta_x_post']
    r2w   = res.rsquared_within
    nobs  = res.nobs

    stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.10 else ' '))

    rows.append({
        'outcome':   outcome,
        'label':     label,
        'beta':      round(beta,  4),
        'se':        round(se,    4),
        't_stat':    round(tstat, 4),
        'p_value':   round(pval,  4),
        'stars':     stars,
        'r2_within': round(r2w,   4),
        'n_obs':     nobs,
    })

summary_df = pd.DataFrame(rows).set_index('outcome')
print("\n--- β Summary Table ---")
print(summary_df.to_string())
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'table_m5_summary.csv'))

# ── Decomposition check: log(w90/w10) ≈ log(w90/w50) + log(w50/w10) ──────────
b_90_10 = summary_df.loc['log_w90_10', 'beta']
b_90_50 = summary_df.loc['log_w90_50', 'beta']
b_50_10 = summary_df.loc['log_w50_10', 'beta']
implied = b_90_50 + b_50_10
discrepancy = abs(b_90_10 - implied)

print(f"\n--- Decomposition Check ---")
print(f"β(90/10)                      = {b_90_10:.4f}")
print(f"β(90/50) + β(50/10) (implied) = {implied:.4f}")
print(f"Discrepancy                    = {discrepancy:.4f}")
if discrepancy > 0.01:
    print("WARNING: Decomposition identity violated — check for differential NaN dropping across outcomes.")
else:
    print("Decomposition check PASSED: β(90/10) ≈ β(90/50) + β(50/10)")

# ── Economic magnitudes ───────────────────────────────────────────────────────
std_dv_beta   = sample['dv_beta'].std()
tercile_means = sample.groupby('exposure_tercile', observed=True)['dv_beta'].mean()
mean_high     = tercile_means['High']
mean_low      = tercile_means['Low']

print(f"\n--- Economic Magnitudes ---")
print(f"std(dv_beta) = {std_dv_beta:.4f}")
print(f"mean(High tercile dv_beta) = {mean_high:.4f}, mean(Low tercile dv_beta) = {mean_low:.4f}")
print(f"High-Low gap = {mean_high - mean_low:.4f}")

mag_rows = []
for outcome, label in OUTCOMES.items():
    b    = summary_df.loc[outcome, 'beta']
    e1sd = b * std_dv_beta
    ehl  = b * (mean_high - mean_low)
    mag_rows.append({
        'outcome':                    outcome,
        'β (raw)':                    round(b,    4),
        'Effect per 1 SD exposure':   round(e1sd, 4),
        'Effect: High vs Low tercile':round(ehl,  4),
    })

mag_df = pd.DataFrame(mag_rows).set_index('outcome')
print(mag_df.to_string())
mag_df.to_csv(os.path.join(OUTPUT_DIR, 'table_m5_economic_magnitudes.csv'))


# ══════════════════════════════════════════════════════════════════════════════
# TASK 5: Residual Diagnostics
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== TASK 5: Residual Diagnostics ===")

print(f"\n--- Observation Counts ---")
for outcome, label in OUTCOMES.items():
    print(f"{label}: N = {results[outcome].nobs}")

print(f"\n--- Within-R² ---")
for outcome, label in OUTCOMES.items():
    print(f"{label}: within-R² = {results[outcome].rsquared_within:.4f}")
print("Note: within-R² measures fit after absorbing occupation and year FEs.")
print("Negative values are possible and do not indicate model failure.")

print("\n--- Residual Distribution ---")
diag_rows = []
for outcome, label in OUTCOMES.items():
    res      = results[outcome]
    residuals = res.resids.values
    r_mean   = np.mean(residuals)
    r_std    = np.std(residuals)
    r_skew   = stats.skew(residuals)
    r_kurt   = stats.kurtosis(residuals)

    print(f"\n{label}: mean={r_mean:.4f}, std={r_std:.4f}, "
          f"skewness={r_skew:.4f}, kurtosis={r_kurt:.4f}")

    if abs(r_skew) > 1.0:
        print(f"  WARNING: Residuals are skewed (skewness = {r_skew:.4f}).")
    else:
        print(f"  Residual skewness within acceptable range (skewness = {r_skew:.4f}).")

    diag_rows.append({
        'outcome':  outcome,
        'mean':     round(r_mean, 4),
        'std':      round(r_std,  4),
        'skewness': round(r_skew, 4),
        'kurtosis': round(r_kurt, 4),
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=30, density=True, alpha=0.6, color='steelblue', label='Residuals')
    x_range = np.linspace(residuals.min(), residuals.max(), 300)
    ax.plot(x_range, stats.norm.pdf(x_range, r_mean, r_std), 'r-', lw=2, label='Normal fit')
    ax.set_title(f"Residual Distribution: {label}")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f'figure_m5_residuals_{outcome}.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {fig_path}")

diag_df = pd.DataFrame(diag_rows).set_index('outcome')
diag_df.to_csv(os.path.join(OUTPUT_DIR, 'table_m5_residual_diagnostics.csv'))


# ══════════════════════════════════════════════════════════════════════════════
# TASK 6: Coefficient Plot
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== TASK 6: Coefficient Plot ===")

outcomes_list = list(OUTCOMES.keys())
labels_list   = list(OUTCOMES.values())
betas         = summary_df.loc[outcomes_list, 'beta'].values
ses           = summary_df.loc[outcomes_list, 'se'].values
pvals         = summary_df.loc[outcomes_list, 'p_value'].values
stars_list    = summary_df.loc[outcomes_list, 'stars'].str.strip().values

ci95 = 1.96  * ses
ci90 = 1.645 * ses
y    = np.array([2, 1, 0])

fig, ax = plt.subplots(figsize=(7, 3.5))
colors = ['#2166ac' if p < 0.05 else '#b0b0b0' for p in pvals]

for i in range(len(outcomes_list)):
    ax.plot([betas[i] - ci90[i], betas[i] + ci90[i]], [y[i], y[i]],
            color=colors[i], lw=3, solid_capstyle='round', zorder=2)
    ax.plot([betas[i] - ci95[i], betas[i] + ci95[i]], [y[i], y[i]],
            color=colors[i], lw=1.2, solid_capstyle='round', zorder=2)
    ax.scatter(betas[i], y[i], color=colors[i], s=60, zorder=3)
    if stars_list[i].strip():
        ax.text(betas[i] + ci95[i] + 0.002, y[i], stars_list[i],
                va='center', ha='left', fontsize=10, color=colors[i])

ax.axvline(0, color='black', lw=0.8, linestyle='--')
ax.set_yticks(y)
ax.set_yticklabels(labels_list, fontsize=11)
ax.set_xlabel('β  (effect of 1-unit increase in LLM exposure after ChatGPT)', fontsize=10)
ax.set_title(
    'OLS DiD: Effect of LLM Exposure on Within-Occupation Wage Inequality\n'
    '(TWFE, clustered SE at occupation level, 90% and 95% CI)',
    fontsize=10, pad=10
)
ax.set_xlim(betas.min() - ci95.max() - 0.015,
            betas.max() + ci95.max() + 0.025)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#2166ac', lw=2, label='p < 0.05'),
    Line2D([0], [0], color='#b0b0b0', lw=2, label='p ≥ 0.05'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='lower right')

plt.tight_layout()
coef_path = os.path.join(OUTPUT_DIR, 'figure_m5_coef_plot.png')
plt.savefig(coef_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Coefficient plot saved: {coef_path}")


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("""
=== INEQUALITY EXTENSION COMPLETE ===

Outcomes estimated:
  log_w90_10  — Log Wage Ratio (90/10): overall inequality
  log_w90_50  — Log Wage Ratio (90/50): upper-tail inequality
  log_w50_10  — Log Wage Ratio (50/10): lower-tail inequality

Tables saved to inequality_results/:
  table_m5_summary.csv                β, SE, t, p, stars for 3 outcomes
  table_m5_economic_magnitudes.csv    scaled effect sizes (1 SD, High vs Low)
  table_m5_residual_diagnostics.csv   mean, std, skewness, kurtosis

Plots saved to inequality_results/:
  figure_m5_residuals_log_w90_10.png
  figure_m5_residuals_log_w90_50.png
  figure_m5_residuals_log_w50_10.png
  figure_m5_coef_plot.png             coefficient plot with 90% and 95% CI

Interpretation:
  β < 0 → LLM exposure compressed within-occupation wages (equalizing)
  β > 0 → LLM exposure widened within-occupation wages (disequalizing)
  Compare 90/50 vs 50/10 to identify where in the distribution the effect hits.
""")
