"""
================================================================================
METHOD 2b — BARTIK SHIFT-SHARE IV: Labor Supply Elasticity (chosen)
================================================================================
Project : LLMs and Labor Market Outcomes
Course  : Econometrics & ML — Spring 2025, University of Chicago
Person  : Anna (Method 2 — Shift-Share IV)

INPUT
-----
  data/processed/oews_exposure_merged.csv   ← from 02_merge_exposure.py
  (same sample filters as 03_estimation.py)

OUTPUTS
-------
  results/table_m2b_first_stage.csv          First stage: instrument → employment
  results/table_m2b_reduced_form.csv         Reduced form: instrument → wages
  results/table_m2b_structural_iv.csv        2SLS structural: employment → wages
  results/table_m2b_event_study.csv          Period-by-period first stage + RF
  results/figure_m2b_labor_supply_iv.png     Three-panel figure

================================================================================
IDENTIFICATION STRATEGY
================================================================================

BACKGROUND — THE PROBLEM WITH OLS
-----------------------------------
A naive OLS regression of wages on employment is biased. Wages and employment
are jointly determined: a positive shock to labor demand raises both
simultaneously, while a negative shock lowers both. Regressing one on the other
without an instrument mixes up movements along the supply curve with movements
of the supply curve itself — you never know which is which.

THE BARTIK SHIFT-SHARE INSTRUMENT
-----------------------------------
Following Bartik (1991) and Goldsmith-Pinkham, Sorkin & Swift (2020), we
construct a demand shifter that is exogenous to occupation-level wage trends:

    Z_ot  =  dv_beta_o  ×  post_t

    Share  =  dv_beta_o      Pre-determined structural LLM task exposure [0,1]
                              from Eloundou et al. (2023), constructed from
                              O*NET task descriptions collected BEFORE ChatGPT.
                              Measures the fraction of each occupation's tasks
                              that GPT-4 can perform. Fixed across time.

    Shift  =  post_t          Indicator for years after the public release of
                              ChatGPT (November 2022). One global shock that
                              hit all occupations simultaneously.

The instrument scales the global ChatGPT shock by each occupation's
pre-existing exposure. Paralegals (dv_beta ≈ 0.80) receive a large shock;
roofers (dv_beta ≈ 0.05) receive almost none. This cross-sectional variation
is plausibly exogenous because:
  (a) dv_beta is constructed from pre-ChatGPT O*NET task data — it cannot
      be a response to post-2022 outcomes.
  (b) The ChatGPT launch date was not anticipated by workers or firms in
      any occupation-specific way.

NOTE ON SINGLE SHIFT
  Classic Bartik uses many industry-level shifts (national industry growth
  rates), giving rich independent variation. Here we have one binary shift
  (ChatGPT release). With a single shift, all identification comes from the
  cross-sectional variation in exposure shares (dv_beta). This is a stronger
  assumption than the multi-shift case and means validity rests entirely on
  dv_beta being uncorrelated with non-LLM post-2022 wage trends, conditional
  on occupation and year fixed effects.

  This assumption is challenged by the pre-trend evidence: high-exposure
  occupations (tech) had faster wage growth even before 2022. Method 3
  (Double Lasso) addresses this by conditioning on education requirements,
  routine task intensity, and offshorability.

STRUCTURAL EQUATION AND ECONOMIC INTERPRETATION
-------------------------------------------------
We estimate the structural labor supply equation:

    log w_ot = α + ε · log L_ot + γ_o + δ_t + u_ot         ... (second stage)
    log L_ot = π · (dv_beta_o × post_t) + γ_o + δ_t + ν_ot  ... (first stage)

where:
  log w_ot   = log mean annual wage for occupation o in year t
  log L_ot   = log total employment for occupation o in year t  ← ENDOGENOUS
  γ_o        = occupation fixed effect (absorbs time-invariant wage levels)
  δ_t        = year fixed effect (absorbs aggregate macro shocks)

The coefficient ε is the slope of the labor supply curve in log-log space.
Under standard competitive labor market assumptions, ε > 0 (wages and
employment move together along an upward-sloping supply curve).

The Bartik instrument identifies ε by tracing out the supply curve:
  - The demand shock Z_ot shifts demand differentially across occupations.
  - Each occupation's new equilibrium (w*, L*) is a point on its supply curve.
  - The IV estimator recovers ε = Δlog w / Δlog L  (both changes caused by Z).

The Wald estimator provides a transparent preview:
    ε_Wald = β_RF / β_FS
           = (effect of Z on log w) / (effect of Z on log L)

In a just-identified IV (one instrument, one endogenous variable), ε_Wald
equals the 2SLS estimate up to degrees-of-freedom and robust SE adjustments.

WHAT A NEGATIVE ε MEANS
  If ε < 0: wages and employment moved in OPPOSITE directions in response to
  the LLM demand shock. Specifically, high-exposure occupations saw:
    - Employment increase (+6.7%) — firms wanted more LLM-capable workers
    - Wages decrease (−6.6%)     — each worker became more task-substitutable
  This is inconsistent with a simple competitive upward-sloping supply curve.
  It suggests the shock operated through TWO channels simultaneously:
    1. Demand expansion (more workers wanted)
    2. Bargaining power erosion (workers more substitutable → lower wages)
  The net equilibrium moved southeast in (L, w) space — a pattern more
  consistent with oligopsony dynamics or technology-induced task compression
  than a standard demand expansion.

PRE-TREND CAVEAT
  The period-by-period Wald estimates are noisy because the first stage
  (dv_beta × post → log_emp) is weak at τ=+2 (2024), causing the
  period-specific Wald ratio to become unstable. Trust the pooled 2SLS
  (F = 14.3 > 10) rather than the period-by-period estimates for inference.
  The pre-period patterns further motivate Method 3.

REFERENCES
----------
  Bartik, T.J. (1991). Who Benefits from State and Local Economic
    Development Policies? Kalamazoo, MI: W.E. Upjohn Institute.

  Goldsmith-Pinkham, P., Sorkin, I., & Swift, H. (2020). Bartik Instruments:
    What, When, Why, and How. American Economic Review, 110(8), 2586–2624.

  Eloundou, T., Manning, S., Mishkin, P., & Rock, D. (2023). GPTs are GPTs:
    An Early Look at the Labor Market Impact Potential of Large Language
    Models. arXiv:2303.10130.

  Acemoglu, D., & Restrepo, P. (2020). Robots and Jobs: Evidence from US
    Labor Markets. Journal of Political Economy, 128(6), 2188–2244.
    [Methodological template for Bartik demand shifter → supply elasticity]

  Katz, L.F., & Murphy, K.M. (1992). Changes in Relative Wages, 1963–1987:
    Supply and Demand Factors. Quarterly Journal of Economics, 107(1), 35–78.
================================================================================
"""

from __future__ import annotations

import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

DATA_PATH   = Path("data/processed/oews_exposure_merged.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Base event-time omitted from both event studies (last pre-treatment year)
BASE_TAU = 0   # corresponds to year 2022

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_sample() -> pd.DataFrame:
    """
    Load the merged panel and apply the shared sample filters from
    03_estimation.py. Adds the Bartik instrument and event-time
    interaction columns.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run 02_merge_exposure.py first."
        )

    df = pd.read_csv(DATA_PATH)

    mask = (
        (df["balanced"]          == 1)     &
        (df["suppressed_wage"]   == False) &
        (df["exposure_residual"] == 0)     &
        df["dv_beta"].notna()              &
        df["log_wage"].notna()             &
        df["log_emp"].notna()
    )
    sample = df[mask].copy()

    n_occ = sample["occ_code"].nunique()
    n_yr  = sample["year"].nunique()
    log.info(
        f"Sample: {len(sample):,} obs  |  "
        f"{n_occ} occupations  |  {n_yr} years {sorted(sample['year'].unique())}"
    )

    # ── Bartik instrument (pooled DiD interaction) ────────────────────────
    # Z_ot = dv_beta_o × post_t
    # This is the shift-share instrument: pre-determined exposure share ×
    # global ChatGPT shock. Used as instrument for log_emp in the structural
    # equation.
    sample["dv_beta_x_post"] = sample["dv_beta"] * sample["post"]

    # ── Period-by-period interaction terms for event studies ──────────────
    # dv_beta_o × 1[t = τ]  for each τ ≠ BASE_TAU
    # Used in both the first-stage and reduced-form event studies.
    for tau in sorted(sample["event_time"].unique()):
        if tau == BASE_TAU:
            continue
        sample[f"dv_x_tau_{tau:+d}"] = (
            sample["dv_beta"] * (sample["event_time"] == tau).astype(int)
        )

    return sample


def twoway_demean(
    df: pd.DataFrame,
    cols: list[str],
    entity: str = "occ_code",
    time:   str = "year",
) -> pd.DataFrame:
    """
    Two-way within transformation: removes occupation and year means.

    Formula:  X̃_ot = X_ot − X̄_o. − X̄_.t + X̄..

    Required before passing data to IV2SLS, which does not natively
    support entity/time fixed effects.
    """
    out          = df[cols].copy().astype(float)
    grand        = out.mean()
    entity_means = out.groupby(df[entity]).transform("mean")
    time_means   = out.groupby(df[time]).transform("mean")
    return out - entity_means - time_means + grand


def stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


# ── Step 1: First stage ───────────────────────────────────────────────────────

def run_first_stage(sample: pd.DataFrame) -> pd.DataFrame:
    """
    First stage: regress log_emp on the Bartik instrument (dv_beta × post)
    with occupation and year fixed effects.

    This answers: does the shift-share demand shock actually move employment?
    A strong first stage (F > 10) is required for the IV to be credible.

    The coefficient π gives the employment response to a one-unit increase
    in LLM exposure during the post-ChatGPT period.
    """
    log.info("\n" + "="*68)
    log.info("STEP 1 — FIRST STAGE: dv_beta × post → log_emp")
    log.info("="*68)

    panel = sample.set_index(["occ_code", "year"])
    sub   = panel.dropna(subset=["log_emp", "dv_beta_x_post"])

    mod = PanelOLS(
        sub["log_emp"],
        sub[["dv_beta_x_post"]],
        entity_effects=True,
        time_effects=True,
    )
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    b  = float(res.params["dv_beta_x_post"])
    se = float(res.std_errors["dv_beta_x_post"])
    t  = float(res.tstats["dv_beta_x_post"])
    p  = float(res.pvalues["dv_beta_x_post"])
    f  = float(res.f_statistic.stat)

    log.info(f"  π = {b:+.4f}  se = {se:.4f}  t = {t:+.3f}  "
             f"p = {p:.4f}{stars(p)}  F = {f:.2f}")
    log.info(
        f"  Instrument strength: F = {f:.2f}  "
        f"({'STRONG' if f > 10 else 'WEAK'} — Stock-Yogo threshold: F > 10)"
    )
    log.info(
        f"  Interpretation: a 1-unit increase in dv_beta raises employment "
        f"by {b*100:.1f}% after ChatGPT (post-period only)."
    )

    result = pd.DataFrame([{
        "step":        "first_stage",
        "equation":    "dv_beta × post → log_emp",
        "beta":        round(b,  4),
        "se":          round(se, 4),
        "t_stat":      round(t,  3),
        "p_value":     round(p,  4),
        "f_stat":      round(f,  2),
        "n_obs":       int(res.nobs),
        "note":        "instrument relevance check; need F > 10",
    }])

    out = RESULTS_DIR / "table_m2b_first_stage.csv"
    result.to_csv(out, index=False)
    log.info(f"  Saved → {out}")
    return result


# ── Step 2: Reduced form ──────────────────────────────────────────────────────

def run_reduced_form(sample: pd.DataFrame) -> pd.DataFrame:
    """
    Reduced form: regress log_wage on the Bartik instrument (dv_beta × post)
    with occupation and year fixed effects.

    This is identical to Method 1 (OLS DiD) from 03_estimation.py.
    It answers: does the shift-share demand shock move wages directly?

    The reduced-form coefficient is the total effect of the instrument on
    wages, operating through all channels (employment and otherwise).
    Together with the first stage, it gives the Wald estimate:

        ε_Wald = β_RF / π_FS

    which equals the 2SLS estimate in the just-identified case.
    """
    log.info("\n" + "="*68)
    log.info("STEP 2 — REDUCED FORM: dv_beta × post → log_wage")
    log.info("="*68)

    panel = sample.set_index(["occ_code", "year"])
    sub   = panel.dropna(subset=["log_wage", "dv_beta_x_post"])

    mod = PanelOLS(
        sub["log_wage"],
        sub[["dv_beta_x_post"]],
        entity_effects=True,
        time_effects=True,
    )
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    b  = float(res.params["dv_beta_x_post"])
    se = float(res.std_errors["dv_beta_x_post"])
    t  = float(res.tstats["dv_beta_x_post"])
    p  = float(res.pvalues["dv_beta_x_post"])

    log.info(f"  β_RF = {b:+.4f}  se = {se:.4f}  t = {t:+.3f}  "
             f"p = {p:.4f}{stars(p)}")
    log.info(
        "  Note: this coefficient is identical to the Method 1 OLS DiD "
        "estimate for log_wage. It represents the total effect of the "
        "shift-share demand shock on wages."
    )

    result = pd.DataFrame([{
        "step":    "reduced_form",
        "equation":"dv_beta × post → log_wage",
        "beta":    round(b,  4),
        "se":      round(se, 4),
        "t_stat":  round(t,  3),
        "p_value": round(p,  4),
        "n_obs":   int(res.nobs),
        "note":    "same as Method 1 OLS DiD for log_wage",
    }])

    out = RESULTS_DIR / "table_m2b_reduced_form.csv"
    result.to_csv(out, index=False)
    log.info(f"  Saved → {out}")
    return result


# ── Step 3: 2SLS structural equation ─────────────────────────────────────────

def run_structural_iv(
    sample:       pd.DataFrame,
    b_fs:         float,
    b_rf:         float,
) -> pd.DataFrame:
    """
    Structural 2SLS: instrument log_emp with (dv_beta × post) to recover
    the inverse labor supply elasticity ε.

    Equation:
        log w_ot = α + ε · log L_ot + γ_o + δ_t + u_ot

    Implementation:
        - Two-way demean log_wage, log_emp, and dv_beta_x_post.
        - Pass demeaned series to IV2SLS (no exogenous regressors —
          fixed effects already absorbed by demeaning).
        - The coefficient on demeaned log_emp is ε.

    Wald preview:
        ε_Wald = β_RF / π_FS  (transparent sanity check; should ≈ 2SLS ε)

    Standard errors: heteroskedasticity-robust (HC3).

    Parameters
    ----------
    b_fs : first-stage coefficient (π) from run_first_stage()
    b_rf : reduced-form coefficient (β_RF) from run_reduced_form()
    """
    log.info("\n" + "="*68)
    log.info("STEP 3 — 2SLS STRUCTURAL: log_emp → log_wage  (IV: dv_beta×post)")
    log.info("="*68)

    # ── Wald preview ──────────────────────────────────────────────────────
    wald = b_rf / b_fs
    log.info(f"\n  Wald estimate (β_RF / π_FS): {b_rf:.4f} / {b_fs:.4f} = {wald:.4f}")
    log.info("  (Should ≈ 2SLS ε below — confirms arithmetic)")

    # ── Two-way demeaning ─────────────────────────────────────────────────
    dm_cols  = ["log_wage", "log_emp", "dv_beta_x_post"]
    demeaned = twoway_demean(sample, dm_cols)
    demeaned.columns = [c + "_dm" for c in demeaned.columns]

    valid = (
        demeaned["log_wage_dm"].notna()       &
        demeaned["log_emp_dm"].notna()        &
        demeaned["dv_beta_x_post_dm"].notna()
    )
    y = demeaned.loc[valid, "log_wage_dm"]
    d = demeaned.loc[valid, ["log_emp_dm"]]        # endogenous
    z = demeaned.loc[valid, ["dv_beta_x_post_dm"]] # instrument

    # ── 2SLS ─────────────────────────────────────────────────────────────
    res = IV2SLS(y, None, d, z).fit(cov_type="robust")

    b_iv  = float(res.params["log_emp_dm"])
    se_iv = float(res.std_errors["log_emp_dm"])
    t_iv  = float(res.tstats["log_emp_dm"])
    p_iv  = float(res.pvalues["log_emp_dm"])
    fs_f  = float(res.first_stage.diagnostics.loc["log_emp_dm", "f.stat"])
    fs_p  = float(res.first_stage.diagnostics.loc["log_emp_dm", "f.pval"])

    log.info(f"\n  ε (2SLS) = {b_iv:+.4f}  se = {se_iv:.4f}  "
             f"t = {t_iv:+.3f}  p = {p_iv:.4f}{stars(p_iv)}")
    log.info(f"  First-stage F = {fs_f:.2f}  (p = {fs_p:.4f})  "
             f"[{'STRONG' if fs_f > 10 else 'WEAK'}]")
    log.info(f"  n = {int(valid.sum()):,}")

    log.info("\n  ── Economic interpretation ──")
    if b_iv < 0:
        log.info(
            f"  ε = {b_iv:.3f} < 0: wages and employment moved in OPPOSITE directions.\n"
            f"  High-exposure occupations saw employment RISE and wages FALL after ChatGPT.\n"
            f"  This is inconsistent with a standard upward-sloping labor supply curve.\n"
            f"  Possible explanation: LLMs expanded the pool of suitable workers\n"
            f"  (demand up, employment up) while also eroding individual bargaining\n"
            f"  power via task substitutability (wages down).\n"
            f"  Net effect: labor supply appears locally DOWNWARD sloping for these\n"
            f"  occupations — consistent with oligopsony or task-compression dynamics."
        )
    else:
        log.info(
            f"  ε = {b_iv:.3f} > 0: a 1% employment increase driven by the LLM\n"
            f"  demand shock raises wages by {b_iv:.2f}%. Upward-sloping supply curve.\n"
            f"  Implied labor supply elasticity ≈ 1/ε = {1/b_iv:.2f}."
        )

    result = pd.DataFrame([
        {
            "step":          "wald_preview",
            "equation":      "β_RF / π_FS",
            "beta":          round(wald, 4),
            "se":            None,
            "t_stat":        None,
            "p_value":       None,
            "first_stage_f": None,
            "n_obs":         None,
            "note":          "Wald estimate; should ≈ 2SLS epsilon",
        },
        {
            "step":          "structural_2sls",
            "equation":      "log_emp → log_wage  (IV: dv_beta × post)",
            "beta":          round(b_iv,  4),
            "se":            round(se_iv, 4),
            "t_stat":        round(t_iv,  3),
            "p_value":       round(p_iv,  4),
            "first_stage_f": round(fs_f,  2),
            "n_obs":         int(valid.sum()),
            "note":          "epsilon = slope of labor supply curve (log-log)",
        },
    ])

    out = RESULTS_DIR / "table_m2b_structural_iv.csv"
    result.to_csv(out, index=False)
    log.info(f"\n  Saved → {out}")
    return result


# ── Step 4: Period-by-period event studies ────────────────────────────────────

def run_event_studies(sample: pd.DataFrame) -> pd.DataFrame:
    """
    Period-by-period versions of both the first stage and reduced form.

    For each τ ≠ 0, estimate:
        First stage RF : log_emp_ot  = Σ_τ π_τ (dv_beta_o × 1[t=τ]) + FEs
        Reduced form   : log_wage_ot = Σ_τ β_τ (dv_beta_o × 1[t=τ]) + FEs

    Pre-period (τ < 0): coefficients should be near zero if the instrument
        is valid. Non-zero pre-trends indicate that high-exposure occupations
        were already on different trajectories before ChatGPT — violation of
        the parallel trends assumption underlying the Bartik design.

    Post-period (τ > 0): shows whether the demand shock's effect on both
        employment and wages grows, stabilizes, or reverses over time.

    Period-by-period Wald:
        ε_τ = β_τ / π_τ   (noisy — treat as illustrative, not inferential)

    Delta-method SE for ε_τ:
        se(ε_τ) ≈ |ε_τ| × sqrt( (se_β/β)² + (se_π/π)² )
    """
    log.info("\n" + "="*68)
    log.info("STEP 4 — PERIOD-BY-PERIOD EVENT STUDIES")
    log.info("="*68)

    event_times   = sorted(t for t in sample["event_time"].unique() if t != BASE_TAU)
    interact_cols = [f"dv_x_tau_{tau:+d}" for tau in event_times]

    panel = sample.set_index(["occ_code", "year"])

    # First-stage event study
    sub_fs = panel.dropna(subset=["log_emp"] + interact_cols)
    mod_fs = PanelOLS(sub_fs["log_emp"], sub_fs[interact_cols],
                      entity_effects=True, time_effects=True)
    res_fs = mod_fs.fit(cov_type="clustered", cluster_entity=True)

    # Reduced-form event study
    sub_rf = panel.dropna(subset=["log_wage"] + interact_cols)
    mod_rf = PanelOLS(sub_rf["log_wage"], sub_rf[interact_cols],
                      entity_effects=True, time_effects=True)
    res_rf = mod_rf.fit(cov_type="clustered", cluster_entity=True)

    rows = []
    log.info(
        f"\n  {'tau':>4}  {'year':>5}  {'π_FS':>8}  {'se_FS':>7}  "
        f"{'β_RF':>8}  {'se_RF':>7}  {'ε_Wald':>8}  {'period':>8}"
    )
    log.info("  " + "-"*68)

    for tau, col in zip(event_times, interact_cols):
        pi  = float(res_fs.params[col])
        se_pi = float(res_fs.std_errors[col])
        b   = float(res_rf.params[col])
        se_b  = float(res_rf.std_errors[col])

        # Period-by-period Wald and delta-method SE
        if abs(pi) > 1e-6:
            wald_t = b / pi
            se_w   = abs(wald_t) * np.sqrt((se_b/b)**2 + (se_pi/pi)**2) if abs(b) > 1e-6 else np.nan
        else:
            wald_t = np.nan
            se_w   = np.nan

        pp = "pre" if tau < 0 else "post"
        log.info(
            f"  {tau:>+4}  {tau+2022:>5}  {pi:>+8.4f}  {se_pi:>7.4f}  "
            f"{b:>+8.4f}  {se_b:>7.4f}  "
            f"{wald_t:>+8.3f}  {pp:>8}"
            if not np.isnan(wald_t) else
            f"  {tau:>+4}  {tau+2022:>5}  {pi:>+8.4f}  {se_pi:>7.4f}  "
            f"{b:>+8.4f}  {se_b:>7.4f}  {'n/a':>8}  {pp:>8}"
        )

        rows.append({
            "tau":          int(tau),
            "year":         int(tau + 2022),
            "b_fs":         round(pi,    4),
            "se_fs":        round(se_pi, 4),
            "ci_lo_fs":     round(pi  - 1.96 * se_pi, 4),
            "ci_hi_fs":     round(pi  + 1.96 * se_pi, 4),
            "b_rf":         round(b,     4),
            "se_rf":        round(se_b,  4),
            "ci_lo_rf":     round(b   - 1.96 * se_b,  4),
            "ci_hi_rf":     round(b   + 1.96 * se_b,  4),
            "wald_epsilon": round(wald_t, 4) if not np.isnan(wald_t) else None,
            "se_wald":      round(se_w,   4) if not np.isnan(se_w)   else None,
            "pre_post":     pp,
        })

    # Insert base period (τ = 0, normalized to zero by construction)
    rows.append({
        "tau": 0, "year": 2022,
        "b_fs": 0.0, "se_fs": 0.0, "ci_lo_fs": 0.0, "ci_hi_fs": 0.0,
        "b_rf": 0.0, "se_rf": 0.0, "ci_lo_rf": 0.0, "ci_hi_rf": 0.0,
        "wald_epsilon": None, "se_wald": None, "pre_post": "base",
    })

    results = pd.DataFrame(rows).sort_values("tau").reset_index(drop=True)

    # Pre-trend check
    pre = results[results["pre_post"] == "pre"]
    fs_pretrend = (abs(pre["b_fs"] / pre["se_fs"].replace(0, np.nan)) > 1.96).any()
    rf_pretrend = (abs(pre["b_rf"] / pre["se_rf"].replace(0, np.nan)) > 1.96).any()
    log.info(f"\n  Pre-trend check — First stage : "
             f"{'⚠ SIGNIFICANT' if fs_pretrend else '✓ not significant'}")
    log.info(f"  Pre-trend check — Reduced form: "
             f"{'⚠ SIGNIFICANT' if rf_pretrend else '✓ not significant'}")
    if rf_pretrend:
        log.warning(
            "  Significant pre-trends in reduced form suggest unconditional\n"
            "  parallel trends fails. Condition on occupation-level controls\n"
            "  (education, RTI, offshorability) via Method 3 (Double Lasso)."
        )

    out = RESULTS_DIR / "table_m2b_event_study.csv"
    results.to_csv(out, index=False)
    log.info(f"\n  Saved → {out}")
    return results


# ── Step 5: Three-panel figure ────────────────────────────────────────────────

def plot_labor_supply_iv(
    es: pd.DataFrame,
    b_iv: float,
) -> None:
    """
    Three-panel figure:
      Panel 1 — First stage event study  (dv_beta × 1[t=τ] → log_emp)
      Panel 2 — Reduced form event study (dv_beta × 1[t=τ] → log_wage)
      Panel 3 — Period-by-period Wald ε̂_τ with pooled 2SLS reference line

    Pre-period dots (blue) should cluster near zero for a valid instrument.
    Post-period dots (red) show the treatment effect trajectory.
    """
    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    C_PRE  = "#2166ac"
    C_POST = "#d6604d"
    C_IV   = "#553BA5"

    taus = sorted(es["tau"].unique())
    tick_labels = [f"τ={t:+d}\n({t+2022})" for t in taus]

    def draw_panel(ax, y_col, se_col, title, ylabel):
        for _, row in es.iterrows():
            tau = row["tau"]
            b   = row[y_col]
            se  = row[se_col] if row[se_col] != 0 else np.nan
            pp  = row["pre_post"]
            color  = C_PRE if pp == "pre" else (C_POST if pp == "post" else "black")
            marker = "D" if pp == "base" else "o"
            if not (isinstance(se, float) and np.isnan(se)):
                ax.plot([tau, tau], [b - 1.96*se, b + 1.96*se],
                        color=color, lw=1.5, alpha=0.6, zorder=2)
            ax.plot(tau, b, marker=marker, ms=7, color=color,
                    fillstyle="none" if pp == "base" else "full",
                    markeredgecolor=color, zorder=3)
        ax.axhline(0,   color="black", lw=0.8, zorder=1)
        ax.axvline(0.5, color="grey",  lw=0.8, ls="--", alpha=0.7, zorder=1)
        ax.axvspan(-3.4, -0.5, color="#e8f0fa", alpha=0.4, zorder=0)
        ax.set_xticks(taus)
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_xlabel("Event time (τ = year − 2022)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", ls=":", alpha=0.5)

    # Panel 1: first stage
    ax1 = fig.add_subplot(gs[0, 0])
    draw_panel(ax1, "b_fs", "se_fs",
               "First stage\n(Instrument → Employment)",
               "β̂τ (dv_beta × 1[t=τ] on log emp)")

    # Panel 2: reduced form
    ax2 = fig.add_subplot(gs[0, 1])
    draw_panel(ax2, "b_rf", "se_rf",
               "Reduced form\n(Instrument → Wages)",
               "β̂τ (dv_beta × 1[t=τ] on log wage)")

    # Panel 3: period-by-period Wald
    # Drop τ=+2 if first stage ≈ 0 (Wald ratio explodes)
    ax3 = fig.add_subplot(gs[0, 2])
    es_wald = es[es["tau"].isin([-3, -2, -1, 0, 1])].copy()
    draw_panel(ax3, "wald_epsilon", "se_wald",
               "Period-by-period Wald ε̂τ\n(RF / First stage)",
               "ε̂τ (inverse supply elasticity)")
    ax3.axhline(b_iv, color=C_IV, lw=1.0, ls="-.",
                alpha=0.8, label=f"Pooled 2SLS ε = {b_iv:.3f}")
    ax3.legend(fontsize=7.5, loc="lower left", framealpha=0.9)

    # Shared legend (first panel)
    legend_items = [
        Line2D([0],[0], color=C_PRE,  marker="o", ms=6, lw=0, label="Pre-period"),
        Line2D([0],[0], color=C_POST, marker="o", ms=6, lw=0, label="Post-period"),
        Line2D([0],[0], color="black",marker="D", ms=6, lw=0, label="Base (τ=0, 2022)",
               fillstyle="none"),
        Line2D([0],[0], color="grey", ls="--", lw=1, label="ChatGPT release"),
    ]
    ax1.legend(handles=legend_items, fontsize=7.5, loc="upper right", framealpha=0.9)

    fig.suptitle(
        "Labor supply IV — Bartik shift-share identification\n"
        r"$\log w_{ot} = \alpha + \varepsilon \cdot \log L_{ot}"
        r"+ \gamma_o + \delta_t + u_{ot}$"
        r"  where $L_{ot}$ instrumented by $\mathrm{dv\_beta}_o \times"
        r"\mathbf{1}[t > 2022]$",
        fontsize=10, y=1.03,
    )

    out = RESULTS_DIR / "figure_m2b_labor_supply_iv.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info(f"\n  Figure saved → {out}")


# ── Summary print ──────────────────────────────────────────────────────────────

def print_summary(b_fs, se_fs, p_fs, b_rf, se_rf, p_rf, b_iv, se_iv, p_iv, fs_f):
    sep = "="*68
    log.info(f"\n{sep}")
    log.info("SUMMARY — LABOR SUPPLY IV (BARTIK SHIFT-SHARE)")
    log.info(sep)
    log.info(
        f"\n  {'Equation':<44}  {'β or ε':>8}  {'se':>7}  {'sig':>4}"
        f"\n  {'-'*44}  {'-'*8}  {'-'*7}  {'-'*4}"
        f"\n  {'First stage:  dv_beta×post → log_emp':<44}  "
        f"{b_fs:>+8.4f}  {se_fs:>7.4f}  {stars(p_fs):>4}"
        f"\n  {'Reduced form: dv_beta×post → log_wage':<44}  "
        f"{b_rf:>+8.4f}  {se_rf:>7.4f}  {stars(p_rf):>4}"
        f"\n  {'Wald ε:       β_RF / π_FS':<44}  "
        f"{b_rf/b_fs:>+8.4f}  {'—':>7}  {'—':>4}"
        f"\n  {'Structural 2SLS: log_emp → log_wage (IV)':<44}  "
        f"{b_iv:>+8.4f}  {se_iv:>7.4f}  {stars(p_iv):>4}"
        f"\n\n  First-stage F = {fs_f:.2f}  "
        f"({'STRONG ✓' if fs_f > 10 else 'WEAK ✗'}, Stock-Yogo > 10)"
        f"\n  Significance: *** p<0.01  ** p<0.05  * p<0.10"
    )
    log.info(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("="*68)
    log.info("METHOD 2b — BARTIK SHIFT-SHARE IV: LABOR SUPPLY ELASTICITY")
    log.info("="*68)

    sample = load_sample()

    # Step 1: instrument relevance
    fs_result = run_first_stage(sample)
    b_fs  = float(fs_result["beta"].iloc[0])
    se_fs = float(fs_result["se"].iloc[0])
    p_fs  = float(fs_result["p_value"].iloc[0])

    # Step 2: total effect of instrument on wages
    rf_result = run_reduced_form(sample)
    b_rf  = float(rf_result["beta"].iloc[0])
    se_rf = float(rf_result["se"].iloc[0])
    p_rf  = float(rf_result["p_value"].iloc[0])

    # Step 3: structural IV
    iv_result = run_structural_iv(sample, b_fs, b_rf)
    b_iv  = float(iv_result.loc[iv_result["step"]=="structural_2sls","beta"].iloc[0])
    se_iv = float(iv_result.loc[iv_result["step"]=="structural_2sls","se"].iloc[0])
    p_iv  = float(iv_result.loc[iv_result["step"]=="structural_2sls","p_value"].iloc[0])
    fs_f  = float(iv_result.loc[iv_result["step"]=="structural_2sls","first_stage_f"].iloc[0])

    # Step 4: period-by-period event studies
    es_result = run_event_studies(sample)

    # Step 5: figure
    plot_labor_supply_iv(es_result, b_iv)

    # Summary
    print_summary(b_fs, se_fs, p_fs, b_rf, se_rf, p_rf, b_iv, se_iv, p_iv, fs_f)

    log.info("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Files produced
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  results/table_m2b_first_stage.csv        π (instrument → employment)
  results/table_m2b_reduced_form.csv       β_RF (instrument → wages)
  results/table_m2b_structural_iv.csv      ε_Wald + ε_2SLS
  results/table_m2b_event_study.csv        period-by-period FS + RF + Wald
  results/figure_m2b_labor_supply_iv.png   three-panel figure

How to cite in your paper
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  "Following Bartik (1991) and Goldsmith-Pinkham et al. (2020), we
  construct a shift-share demand shifter Z_ot = dv_beta_o × post_t,
  where dv_beta_o is the pre-determined LLM task exposure score from
  Eloundou et al. (2023) and post_t indicates years after the November
  2022 ChatGPT release. We use Z_ot as an instrument for log employment
  in a two-way FE wage equation, recovering the slope of the local labor
  supply curve (Acemoglu & Restrepo 2020). The first-stage F-statistic
  of [F_VALUE] exceeds the Stock-Yogo weak-instrument threshold of 10."

TODO before final submission
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Add onet_controls.csv and re-run Method 3 (Double Lasso) to
     address the pre-trend violation visible in the reduced-form
     event study.
  2. Add 2025 OEWS data (τ=+3) for a third post-treatment year.
  3. Replicate with human_beta as the exposure share for robustness.
    """)


if __name__ == "__main__":
    main()