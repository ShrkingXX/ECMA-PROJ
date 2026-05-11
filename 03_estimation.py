"""
================================================================================
ESTIMATION: All 4 Methods — LLMs and Labor Market Outcomes
================================================================================
Project : LLMs and Labor Market Outcomes
Course  : Econometrics & ML — Spring 2025, University of Chicago
Person  : C (Methods 1–2)  |  Person D (Methods 3–4)

INPUT
-----
  data/processed/oews_exposure_merged.csv   ← from 02_merge_exposure.py

METHODS
-------
  1. OLS Difference-in-Differences (Two-Way FE)          → Person C
  2. Shift-Share (Bartik) IV — 2SLS                      → Person C
  3. Double Lasso DiD (Conditional Parallel Trends)       → Person D
  4. Period-by-Period Event Study                         → Person D

OUTPUTS
-------
  results/table_m1_ols_did.csv          Main OLS DiD results (3 outcomes)
  results/table_m2_bartik_iv.csv        Bartik IV results (3 outcomes)
  results/table_m3_double_lasso.csv     Double Lasso results (add when controls ready)
  results/table_m4_event_study.csv      Event-study β̂τ + 95% CI (all outcomes)
  results/figure_event_study.png        Event-study plot (pre + post, all 3 outcomes)

SAMPLE FILTERS (applied once, shared across all methods)
---------------------------------------------------------
  balanced         == 1     occupation in all 6 years (2019–2024)
  suppressed_wage  == False BLS did not suppress the wage cell
  exposure_residual== 0     drop residual "All Other" aggregates
  dv_beta.notna()           drop the handful with no exposure score
  log_wage.notna()          drops Models (27-2090) 2021 suppression

  → 4,271 obs, 712 occupations × up to 6 years

IDENTIFICATION NOTES
---------------------
  Treatment variable : dv_beta  (pre-ChatGPT structural LLM exposure [0,1])
  Post indicator     : year > 2022  (OEWS May 2022 = last pre-treatment snapshot)
  DiD interaction    : dv_beta × post  (β is the coefficient of interest)

  Omitted event-time : τ = 0  (year 2022, last pre-treatment year)
  Pre-period τ       : −3 (2019), −2 (2020), −1 (2021)
  Post-period τ      : +1 (2023), +2 (2024)

  ⚠ PRE-TREND WARNING (from preliminary event-study):
    τ=-3: β≈+0.058, τ=-2: β≈+0.049  — both statistically significant
    → High-exposure occupations (tech) had faster wage growth BEFORE ChatGPT
    → Violates unconditional parallel trends
    → This motivates Method 3 (Double Lasso conditioning on education,
      offshorability, routine-task index) — see Section 3 below.

  Bartik IV logic    : dv_beta is pre-determined from O*NET task data
    collected before ChatGPT. However, high-exposure occupations may have
    unobserved characteristics correlated with post-2022 wage trends for
    non-LLM reasons (e.g., tech-sector demand boom). We instrument
    dv_beta × post with human_beta × post — human annotators rated the
    same tasks independently of the GPT-4 model. This removes measurement
    error in exposure and isolates the component of exposure predicted by
    both raters.

    First-stage F > 100 (human_beta and dv_beta correlate ≈ 0.88),
    confirming instrument strength. Exclusion restriction: human rater
    scores are uncorrelated with non-LLM post-2022 wage trends conditional
    on occupation and year FEs.

================================================================================
"""

from __future__ import annotations

import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS
from sklearn.linear_model import LassoCV
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_PATH   = Path("data/processed/oews_exposure_merged.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTCOMES = {
    "log_wage" : "Log Mean Annual Wage",
    "log_emp"  : "Log Total Employment",
    "log_wbill": "Log Wage Bill",
}

TREATMENT_COL  = "dv_beta"        # primary exposure
INSTRUMENT_COL = "human_beta"     # Bartik instrument
BASE_TAU       = 0                # omitted event-time (year 2022)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — DATA LOADING AND SAMPLE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def load_sample() -> pd.DataFrame:
    """Load merged panel and apply the shared estimation sample filters."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run 02_merge_exposure.py first."
        )

    df = pd.read_csv(DATA_PATH)

    # ── Apply sample filters ──────────────────────────────────────────────────
    mask = (
        (df["balanced"]          == 1)     &
        (df["suppressed_wage"]   == False) &
        (df["exposure_residual"] == 0)     &
        df["dv_beta"].notna()              &
        df["log_wage"].notna()
    )
    sample = df[mask].copy()

    n_occ  = sample["occ_code"].nunique()
    n_yr   = sample["year"].nunique()
    log.info(
        f"Estimation sample: {len(sample):,} obs  |  "
        f"{n_occ} occupations  |  {n_yr} years {sorted(sample['year'].unique())}"
    )

    # ── Interaction terms ─────────────────────────────────────────────────────
    sample["dv_beta_x_post"]     = sample["dv_beta"]     * sample["post"]
    sample["human_beta_x_post"]  = sample["human_beta"]  * sample["post"]

    # ── Event-time interaction dummies ────────────────────────────────────────
    for tau in sorted(sample["event_time"].unique()):
        if tau == BASE_TAU:
            continue
        col = f"dv_x_tau_{tau:+d}"
        sample[col] = sample["dv_beta"] * (sample["event_time"] == tau).astype(int)

    return sample


def twoway_demean(df: pd.DataFrame, cols: list[str],
                  entity: str = "occ_code", time: str = "year") -> pd.DataFrame:
    """
    Two-way within transformation (demean by occupation and year).
    Used for IV2SLS which doesn't support FEs natively in linearmodels v7.

    Formula: Y_ot_dm = Y_ot - Ȳ_o. - Ȳ_.t + Ȳ..
    """
    out = df[cols].copy().astype(float)
    grand = out.mean()
    entity_means = out.groupby(df[entity]).transform("mean")
    time_means   = out.groupby(df[time]).transform("mean")
    return out - entity_means - time_means + grand


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def summary_statistics(sample: pd.DataFrame) -> None:
    """Print summary statistics for the estimation sample."""
    log.info("\n" + "="*68 + "\nSUMMARY STATISTICS\n" + "="*68)

    one_per_occ = sample.drop_duplicates("occ_code")

    stats = pd.DataFrame({
        "dv_beta (exposure)": one_per_occ["dv_beta"].describe(),
        "log_wage (2022)": sample[sample["year"]==2022]["log_wage"].describe(),
        "log_emp (2022)":  sample[sample["year"]==2022]["log_emp"].describe(),
    }).round(3)
    print("\nOccupation-level summary (dv_beta) and 2022 outcomes:")
    print(stats.to_string())

    print("\nExposure by tercile (dv_beta):")
    one_per_occ = one_per_occ.copy()
    one_per_occ["tercile"] = pd.qcut(one_per_occ["dv_beta"], 3,
                                      labels=["Low","Mid","High"])
    print(one_per_occ.groupby("tercile")["dv_beta"].agg(
        n="count", mean="mean", min="min", max="max"
    ).to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# METHOD 1 — OLS DIFFERENCE-IN-DIFFERENCES (Two-Way FE)
# ═══════════════════════════════════════════════════════════════════════════════

def run_ols_did(sample: pd.DataFrame) -> pd.DataFrame:
    """
    Method 1: Two-way FE DiD.

    Equation (per proposal):
        log Y_ot = α + β(dv_beta_o × post_t) + γ_o + δ_t + ε_ot

    γ_o = occupation FE (absorbs time-invariant wage levels)
    δ_t = year FE (absorbs aggregate macro shocks)
    β   = coefficient of interest: differential change in outcome for
          a 1-unit increase in LLM exposure after ChatGPT release

    Standard errors: clustered at the occupation level.

    Run for all 3 outcomes: log_wage, log_emp, log_wbill.
    """
    log.info("\n" + "="*68)
    log.info("METHOD 1 — OLS DiD (Two-Way FE)")
    log.info("="*68)

    panel  = sample.set_index(["occ_code", "year"])
    rows   = []

    for outcome_col, outcome_label in OUTCOMES.items():
        if panel[outcome_col].isna().all():
            log.warning(f"  Skipping {outcome_col} — all NaN")
            continue

        sub = panel.dropna(subset=[outcome_col, "dv_beta_x_post"])

        mod = PanelOLS(
            sub[outcome_col],
            sub[["dv_beta_x_post"]],
            entity_effects=True,
            time_effects=True,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)

        b  = float(res.params["dv_beta_x_post"])
        se = float(res.std_errors["dv_beta_x_post"])
        t  = float(res.tstats["dv_beta_x_post"])
        p  = float(res.pvalues["dv_beta_x_post"])
        r2 = float(res.rsquared_within)

        log.info(
            f"  {outcome_label:30s}  β={b:+.4f}  se={se:.4f}  "
            f"t={t:+.3f}  p={p:.4f}  R²_within={r2:.4f}"
        )

        rows.append({
            "method":   "OLS DiD",
            "outcome":  outcome_col,
            "label":    outcome_label,
            "beta":     round(b, 4),
            "se":       round(se, 4),
            "t_stat":   round(t, 3),
            "p_value":  round(p, 4),
            "r2_within":round(r2, 4),
            "n_obs":    int(res.nobs),
        })

    results = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "table_m1_ols_did.csv"
    results.to_csv(out_path, index=False)
    log.info(f"\n  Saved → {out_path}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# METHOD 2 — SHIFT-SHARE (BARTIK) IV  /  2SLS
# ═══════════════════════════════════════════════════════════════════════════════

def run_bartik_iv(sample: pd.DataFrame) -> pd.DataFrame:
    """
    Method 2: Shift-Share IV (2SLS).

    Instrument construction (Bartik logic):
        Shift  = ChatGPT public release (binary: 1 if year > 2022)
        Share  = dv_beta_o  (pre-determined structural LLM exposure from O*NET,
                             collected before ChatGPT — truly exogenous to post-2022 trends)
        Z_ot   = dv_beta_o × post_t  = instrument

    We instrument Z_ot with an alternative measure of the same exposure:
        W_ot   = human_beta_o × post_t

    Rationale: GPT-4 self-ratings (dv_beta) and human annotator ratings
    (human_beta) independently measure the same underlying task exposure.
    Using the human ratings as an instrument for the model ratings removes
    any correlated measurement error in dv_beta that might be correlated
    with post-2022 tech-sector wage trends.

    Equivalently: this is a "classical measurement error IV" — the human
    and model ratings are two noisy measures of true task exposure.

    Implementation: IV2SLS on two-way demeaned data (manual within-transform)
    since linearmodels.iv.IV2SLS does not support entity/time effects.

    Reference: Goldsmith-Pinkham, Sorkin & Swift (2020) for shift-share
    validity conditions.
    """
    log.info("\n" + "="*68)
    log.info("METHOD 2 — Bartik IV / 2SLS")
    log.info("="*68)

    dm_cols = (
        list(OUTCOMES.keys()) +
        ["dv_beta_x_post", "human_beta_x_post"]
    )
    demeaned = twoway_demean(sample, dm_cols)
    demeaned.columns = [c + "_dm" for c in demeaned.columns]

    rows = []
    first_stage_reported = False

    for outcome_col, outcome_label in OUTCOMES.items():
        y_dm = demeaned[f"{outcome_col}_dm"]
        x_dm = demeaned[["dv_beta_x_post_dm"]]     # endogenous
        z_dm = demeaned[["human_beta_x_post_dm"]]  # instrument

        # Drop NaN rows (aligned across all three)
        valid = y_dm.notna() & x_dm.notna().all(axis=1) & z_dm.notna().all(axis=1)
        y_dm = y_dm[valid]
        x_dm = x_dm[valid]
        z_dm = z_dm[valid]

        # No exogenous regressors — FEs already absorbed by two-way demeaning
        res = IV2SLS(y_dm, None, x_dm, z_dm).fit(cov_type="robust")

        b  = float(res.params["dv_beta_x_post_dm"])
        se = float(res.std_errors["dv_beta_x_post_dm"])
        t  = float(res.tstats["dv_beta_x_post_dm"])
        p  = float(res.pvalues["dv_beta_x_post_dm"])

        # First-stage diagnostics (only need once — same for all outcomes)
        fs_f    = float(res.first_stage.diagnostics.loc["dv_beta_x_post_dm", "f.stat"])
        fs_pval = float(res.first_stage.diagnostics.loc["dv_beta_x_post_dm", "f.pval"])

        if not first_stage_reported:
            log.info(
                f"\n  First-stage F = {fs_f:.1f}  (p={fs_pval:.4f})  "
                f"[Stock-Yogo threshold: F > 10 for strong IV]\n"
            )
            first_stage_reported = True

        log.info(
            f"  {outcome_label:30s}  β={b:+.4f}  se={se:.4f}  "
            f"t={t:+.3f}  p={p:.4f}  first-stage F={fs_f:.1f}"
        )

        rows.append({
            "method":         "Bartik IV (2SLS)",
            "outcome":        outcome_col,
            "label":          outcome_label,
            "beta":           round(b, 4),
            "se":             round(se, 4),
            "t_stat":         round(t, 3),
            "p_value":        round(p, 4),
            "first_stage_f":  round(fs_f, 1),
            "first_stage_p":  round(fs_pval, 4),
            "n_obs":          int(len(y_dm)),
        })

    results = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "table_m2_bartik_iv.csv"
    results.to_csv(out_path, index=False)
    log.info(f"\n  Saved → {out_path}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# METHOD 3 — DOUBLE LASSO DiD (Conditional Parallel Trends)
# ═══════════════════════════════════════════════════════════════════════════════

def run_double_lasso(sample: pd.DataFrame,
                     controls: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Method 3: Double Lasso DiD (Belloni, Chernozhukov & Hansen 2014).

    MOTIVATION
    ----------
    Pre-trend test from the event study shows τ=-3 and τ=-2 coefficients are
    significantly positive: high-exposure occupations had faster wage growth
    2019–2021, before ChatGPT. This likely reflects the pre-2022 tech-sector
    boom, not LLM effects. Unconditional parallel trends fails.

    Solution: Condition on pre-determined occupation-level observables W_o
    (education requirements, routine task intensity, offshorability,
    demographic mix) that capture why tech and non-tech occupations had
    different pre-trends. Parallel trends is then plausible *conditional*
    on these controls.

    ALGORITHM (Post-Double-Selection Lasso, per Belloni et al. 2014)
    ----------------------------------------------------------------
    For each outcome Y (log_wage, log_emp, log_wbill):

    Step 1 — Lasso Y on W:
        Fit Lasso of outcome on all controls W_o (interacted with year).
        Record selected controls Ŝ_Y.

    Step 2 — Lasso treatment D on W:
        Fit Lasso of (dv_beta × post) on W.
        Record selected controls Ŝ_D.

    Step 3 — OLS DiD with union of selected controls:
        Run OLS DiD with two-way FE and controls Ŝ_Y ∪ Ŝ_D.
        β estimate from this regression is approximately unbiased even
        if the number of controls p is large relative to n.

    CONTROLS NEEDED (from 03_process_onet_controls.py — not yet run)
    ----------------------------------------------------------------
    The following occupation-level controls W_o should be added via script 03:
      - pct_bachelors_plus      : % workers with bachelor's or higher (O*NET)
      - routine_task_index      : Autor-Dorn (2013) RTI
      - offshorability_index    : Blinder-Krueger offshorability score
      - pct_female              : % female workers (ACS)
      - pct_white               : % white workers (ACS)
      - median_age              : median age of workers (ACS)

    CURRENT STATUS
    --------------
    This function runs a skeleton version using only the controls already
    in the merged dataset (none — controls not yet added). Once you run
    03_process_onet_controls.py and re-merge, pass the controls DataFrame
    as the `controls` argument.

    HOW TO CALL WHEN CONTROLS ARE READY
    ------------------------------------
    controls = pd.read_csv("data/processed/onet_controls.csv")
    results  = run_double_lasso(sample, controls=controls)
    """
    log.info("\n" + "="*68)
    log.info("METHOD 3 — Double Lasso DiD")
    log.info("="*68)

    if controls is None:
        log.warning(
            "  No controls provided. Run 03_process_onet_controls.py first,\n"
            "  then pass the controls DataFrame to this function.\n"
            "  Returning OLS DiD estimates as placeholder."
        )
        return pd.DataFrame()

    # ── Merge controls into sample ────────────────────────────────────────────
    control_cols = [c for c in controls.columns if c != "occ_code"]
    merged = sample.merge(controls[["occ_code"] + control_cols],
                          on="occ_code", how="left")

    rows = []

    for outcome_col, outcome_label in OUTCOMES.items():
        sub = merged.dropna(subset=[outcome_col] + control_cols).copy()

        # Build W matrix: control columns interacted with post
        # (allows control effects to differ pre vs. post)
        W_cols = []
        for w in control_cols:
            sub[f"{w}_x_post"] = sub[w] * sub["post"]
            W_cols += [w, f"{w}_x_post"]

        # ── Step 1: Lasso Y on W (after removing occ+year FE) ─────────────────
        dm_Y = twoway_demean(sub, [outcome_col] + W_cols)
        y_dm = dm_Y[outcome_col]
        W_dm = dm_Y[W_cols]

        lasso_Y = LassoCV(cv=5, max_iter=5000, random_state=42)
        lasso_Y.fit(W_dm, y_dm)
        selected_Y = [W_cols[i] for i, c in enumerate(lasso_Y.coef_) if abs(c) > 1e-8]

        # ── Step 2: Lasso D on W ──────────────────────────────────────────────
        dm_D = twoway_demean(sub, ["dv_beta_x_post"] + W_cols)
        D_dm = dm_D["dv_beta_x_post"]
        W_dm2 = dm_D[W_cols]

        lasso_D = LassoCV(cv=5, max_iter=5000, random_state=42)
        lasso_D.fit(W_dm2, D_dm)
        selected_D = [W_cols[i] for i, c in enumerate(lasso_D.coef_) if abs(c) > 1e-8]

        union_controls = sorted(set(selected_Y) | set(selected_D))
        log.info(
            f"  {outcome_col}: Lasso selected {len(selected_Y)} (Y) + "
            f"{len(selected_D)} (D) → union {len(union_controls)} controls"
        )

        # ── Step 3: OLS DiD with selected controls ────────────────────────────
        regressors = ["dv_beta_x_post"] + union_controls
        panel = sub.set_index(["occ_code", "year"])
        mod = PanelOLS(
            panel[outcome_col],
            panel[regressors],
            entity_effects=True,
            time_effects=True,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)

        b  = float(res.params["dv_beta_x_post"])
        se = float(res.std_errors["dv_beta_x_post"])
        t  = float(res.tstats["dv_beta_x_post"])
        p  = float(res.pvalues["dv_beta_x_post"])

        log.info(
            f"  {outcome_label:30s}  β={b:+.4f}  se={se:.4f}  "
            f"t={t:+.3f}  p={p:.4f}  n_controls={len(union_controls)}"
        )

        rows.append({
            "method":      "Double Lasso DiD",
            "outcome":     outcome_col,
            "label":       outcome_label,
            "beta":        round(b, 4),
            "se":          round(se, 4),
            "t_stat":      round(t, 3),
            "p_value":     round(p, 4),
            "n_controls":  len(union_controls),
            "n_obs":       int(res.nobs),
        })

    results = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "table_m3_double_lasso.csv"
    results.to_csv(out_path, index=False)
    log.info(f"\n  Saved → {out_path}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# METHOD 4 — PERIOD-BY-PERIOD EVENT STUDY
# ═══════════════════════════════════════════════════════════════════════════════

def run_event_study(sample: pd.DataFrame) -> pd.DataFrame:
    """
    Method 4: Period-by-period event study.

    Equation (per proposal):
        log Y_ot = α + Σ_τ≠0 β_τ (dv_beta_o × 1[t=τ]) + γ_o + δ_t + ε_ot

    τ indexes years relative to treatment: τ = year − 2022.
    Omitted category: τ = 0 (year 2022, last pre-treatment snapshot).

    Two uses:
    (a) Pre-trend validation: β_τ ≈ 0 for τ ∈ {−3,−2,−1} is necessary
        (not sufficient) for parallel trends.
    (b) Time path of treatment: β_τ for τ ∈ {+1,+2} shows whether effects
        are growing, stable, or reversing over time.

    ⚠ Pre-trend concern: preliminary estimates show β_-3 ≈ +0.058 and
      β_-2 ≈ +0.049 for log_wage — both significant. This means high-exposure
      occupations were already on higher wage trajectories before ChatGPT.
      Method 3 (Double Lasso) addresses this by conditioning on education,
      routine task index, and offshorability.

    Standard errors: clustered at the occupation level.
    """
    log.info("\n" + "="*68)
    log.info("METHOD 4 — Period-by-Period Event Study")
    log.info("="*68)

    event_times = sorted(t for t in sample["event_time"].unique() if t != BASE_TAU)
    interact_cols = [f"dv_x_tau_{tau:+d}" for tau in event_times]

    panel = sample.set_index(["occ_code", "year"])
    all_rows = []

    for outcome_col, outcome_label in OUTCOMES.items():
        sub = panel.dropna(subset=[outcome_col] + interact_cols)

        mod = PanelOLS(
            sub[outcome_col],
            sub[interact_cols],
            entity_effects=True,
            time_effects=True,
        )
        res = mod.fit(cov_type="clustered", cluster_entity=True)

        # Add the omitted base period (β = 0 by construction)
        rows_outcome = []
        for tau, col in zip(event_times, interact_cols):
            b  = float(res.params[col])
            se = float(res.std_errors[col])
            rows_outcome.append({
                "outcome":  outcome_col,
                "label":    outcome_label,
                "tau":      int(tau),
                "year":     int(tau + 2022),
                "beta":     round(b, 4),
                "se":       round(se, 4),
                "ci_lo":    round(b - 1.96*se, 4),
                "ci_hi":    round(b + 1.96*se, 4),
                "pre_post": "pre" if tau < 0 else "post",
                "n_obs":    int(res.nobs),
            })

        # Insert the omitted base period
        rows_outcome.append({
            "outcome":  outcome_col,
            "label":    outcome_label,
            "tau":      0,
            "year":     2022,
            "beta":     0.0,
            "se":       0.0,
            "ci_lo":    0.0,
            "ci_hi":    0.0,
            "pre_post": "base",
            "n_obs":    int(res.nobs),
        })

        rows_outcome.sort(key=lambda r: r["tau"])

        log.info(f"\n  {outcome_label}:")
        for r in rows_outcome:
            stars = (
                "***" if r["se"] > 0 and abs(r["beta"]/r["se"]) > 2.576 else
                "**"  if r["se"] > 0 and abs(r["beta"]/r["se"]) > 1.960 else
                "*"   if r["se"] > 0 and abs(r["beta"]/r["se"]) > 1.645 else
                ""
            )
            tag = "(BASE)" if r["tau"] == 0 else f"[{r['pre_post'].upper()}]"
            log.info(
                f"    τ={r['tau']:+d} ({r['year']}) {tag:6s}  "
                f"β={r['beta']:+.4f}  se={r['se']:.4f}  "
                f"95%CI=[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]{stars}"
            )

        all_rows.extend(rows_outcome)

    # ── Pre-trend formal test ──────────────────────────────────────────────────
    log.info("\n  Pre-trend summary:")
    results_df = pd.DataFrame(all_rows)
    for outcome_col in OUTCOMES:
        pre = results_df[
            (results_df["outcome"] == outcome_col) &
            (results_df["pre_post"] == "pre")
        ]
        any_sig = (abs(pre["beta"] / pre["se"].replace(0, np.nan)) > 1.96).any()
        status  = "⚠ SIGNIFICANT PRE-TRENDS DETECTED" if any_sig else "✓ Pre-trends not significant"
        log.info(f"    {outcome_col}: {status}")

    out_path = RESULTS_DIR / "table_m4_event_study.csv"
    results_df.to_csv(out_path, index=False)
    log.info(f"\n  Saved → {out_path}")
    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION — Event Study Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_event_study(es_results: pd.DataFrame) -> None:
    """
    Plot period-by-period β̂τ coefficients with 95% confidence intervals
    for all three outcome variables in a 3-panel figure.

    Pre-period τ ∈ {−3,−2,−1}: should cluster near zero if parallel trends hold.
    Post-period τ ∈ {+1,+2}: shows treatment effect trajectory.
    Base period τ=0 (2022): normalized to zero (shown as hollow dot).
    """
    outcomes = list(OUTCOMES.keys())
    labels   = list(OUTCOMES.values())

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    colors = {"pre": "#2166ac", "post": "#d6604d", "base": "black"}

    for idx, (outcome_col, outcome_label) in enumerate(OUTCOMES.items()):
        ax = fig.add_subplot(gs[0, idx])
        df = es_results[es_results["outcome"] == outcome_col].sort_values("tau")

        for _, row in df.iterrows():
            tau     = row["tau"]
            b       = row["beta"]
            ci_lo   = row["ci_lo"]
            ci_hi   = row["ci_hi"]
            period  = row["pre_post"]
            color   = colors[period]

            # Confidence interval bar
            if period != "base":
                ax.plot([tau, tau], [ci_lo, ci_hi],
                        color=color, linewidth=1.5, alpha=0.6, zorder=2)

            # Point estimate
            marker = "o" if period != "base" else "D"
            fill   = "full" if period != "base" else "none"
            ax.plot(tau, b, marker=marker, markersize=7,
                    color=color, fillstyle=fill,
                    markeredgecolor=color, zorder=3)

        # Formatting
        ax.axhline(0, color="black", linewidth=0.8, linestyle="-", zorder=1)
        ax.axvline(0.5, color="grey", linewidth=0.8, linestyle="--",
                   alpha=0.7, zorder=1,
                   label="ChatGPT release\n(Nov 2022)")

        ax.set_xticks(sorted(df["tau"].unique()))
        ax.set_xticklabels(
            [f"τ={t:+d}\n({t+2022})" for t in sorted(df["tau"].unique())],
            fontsize=7
        )
        ax.set_xlabel("Event time (τ = year − 2022)", fontsize=9)
        ax.set_ylabel("β̂τ (effect per unit dv_beta exposure)" if idx == 0 else "",
                      fontsize=9)
        ax.set_title(outcome_label, fontsize=10, fontweight="bold", pad=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        # Shade pre-period
        ax.axvspan(-3.4, -0.5, color="#e8f0fa", alpha=0.4, zorder=0)

        # Legend (first panel only)
        if idx == 0:
            from matplotlib.lines import Line2D
            legend_items = [
                Line2D([0],[0], color="#2166ac", marker="o", markersize=6,
                        label="Pre-period", linewidth=0),
                Line2D([0],[0], color="#d6604d", marker="o", markersize=6,
                        label="Post-period", linewidth=0),
                Line2D([0],[0], color="black", marker="D", markersize=6,
                        label="Base (τ=0, 2022)", linewidth=0,
                        fillstyle="none"),
                Line2D([0],[0], color="grey", linestyle="--",
                        label="ChatGPT release", linewidth=1),
            ]
            ax.legend(handles=legend_items, fontsize=7.5,
                      loc="upper right", framealpha=0.9)

    fig.suptitle(
        "Event Study: Effect of LLM Exposure on Labor Market Outcomes\n"
        r"$\hat{\beta}_\tau$ from $\log Y_{ot} = \alpha + \sum_{\tau\neq 0}"
        r"\beta_\tau(\text{dv\_beta}_o \times \mathbf{1}[t=\tau]) + \gamma_o + \delta_t + \varepsilon_{ot}$",
        fontsize=10, y=1.02
    )

    out_path = RESULTS_DIR / "figure_event_study.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info(f"\n  Event study figure saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-METHOD COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def comparison_table(m1: pd.DataFrame, m2: pd.DataFrame,
                     m3: pd.DataFrame | None = None) -> None:
    """Print a side-by-side comparison of β estimates across methods."""
    log.info("\n" + "="*68)
    log.info("CROSS-METHOD COMPARISON  (β coefficient on dv_beta × post)")
    log.info("="*68)

    frames = [m1, m2]
    if m3 is not None and not m3.empty:
        frames.append(m3)

    combined = pd.concat(frames, ignore_index=True)
    pivot = combined.pivot(index="label", columns="method",
                           values=["beta", "se"])

    print("\n  β estimates (standard errors in parentheses):\n")
    for outcome_label in OUTCOMES.values():
        row = combined[combined["label"] == outcome_label]
        parts = []
        for _, r in row.iterrows():
            stars = (
                "***" if r["p_value"] < 0.01 else
                "**"  if r["p_value"] < 0.05 else
                "*"   if r["p_value"] < 0.10 else ""
            )
            parts.append(f"  {r['method']}: {r['beta']:+.4f}{stars} ({r['se']:.4f})")
        log.info(f"\n  {outcome_label}:")
        for p in parts:
            log.info(p)

    log.info("\n  Significance: *** p<0.01  ** p<0.05  * p<0.10")
    log.info(
        "\n  Interpretation guide:"
        "\n  • β < 0 on log_wage → augmentation NOT occurring;"
        "  wage premium for high-exposure jobs shrinking"
        "\n  • β < 0 on log_emp  → displacement: employment falling in high-exposure occupations"
        "\n  • β < 0 on log_wbill→ check: composition bias vs. real total income loss"
        "\n  • If |β_IV| > |β_OLS|: IV removes attenuation bias from measurement error"
        "\n  • If |β_IV| < |β_OLS|: OLS was capturing pre-existing trends (tech-sector)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("=" * 68)
    log.info("ESTIMATION — LLMs AND LABOR MARKET OUTCOMES")
    log.info("=" * 68)

    # Load and build estimation sample
    sample = load_sample()
    summary_statistics(sample)

    # Run all methods
    m1_results = run_ols_did(sample)
    m2_results = run_bartik_iv(sample)

    # Method 3: Double Lasso — uncomment and pass controls when ready
    # controls   = pd.read_csv("data/processed/onet_controls.csv")
    # m3_results = run_double_lasso(sample, controls=controls)
    m3_results = run_double_lasso(sample, controls=None)   # placeholder

    m4_results = run_event_study(sample)

    # Plots
    plot_event_study(m4_results)

    # Cross-method comparison
    comparison_table(m1_results, m2_results)

    log.info("\n" + "="*68)
    log.info("ALL METHODS COMPLETE")
    log.info("="*68)
    log.info(f"\nResults saved to: {RESULTS_DIR}/")
    log.info("""
    Files produced:
      table_m1_ols_did.csv       Method 1: OLS DiD β for 3 outcomes
      table_m2_bartik_iv.csv     Method 2: IV 2SLS β for 3 outcomes
      table_m4_event_study.csv   Method 4: β̂τ per period for 3 outcomes
      figure_event_study.png     Event study plot (3-panel)

    TODO — add before final submission:
      1. Run 03_process_onet_controls.py to get education/RTI/offshorability controls
      2. Uncomment m3_results line in main() and re-run → table_m3_double_lasso.csv
      3. Add Felten et al. exposure scores as alternative measure robustness check
         (replace dv_beta with felten_score and re-run Methods 1 & 2)
    """)


if __name__ == "__main__":
    main()
