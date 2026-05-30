"""
================================================================================
METHOD 4: Period-by-Period Event Study
================================================================================
Project : LLMs and Labor Market Outcomes
Course  : Econometrics & ML — Spring 2025, University of Chicago
Person  : Lynnard (Method 4 — Period-by-Period Event Study)

PURPOSE
-------
Estimates a two-way fixed effects (TWFE) OLS event study regression using
the CES quarterly panel merged with Felten et al. AIIE industry exposure scores.
Produces coefficient plots with confidence intervals for three outcomes:
  1. log employment
  2. log average hourly earnings (wages)
  3. log wage bill (employment × earnings)

MODEL
-----
  log y_it = α + Σ_τ β_τ·(aiie_i × 1[t=τ]) + γ_i + δ_t + ε_it

  i = supersector (industry), t = year×quarter
  γ_i = industry fixed effects       (absorbed by linearmodels)
  δ_t = time fixed effects           (absorbed by linearmodels)
  aiie_i = Felten AIIE score         (time-invariant, varies across industries)
  1[t=τ] = quarter dummy             (1 in quarter τ, 0 elsewhere)
  β_τ = coefficient of interest      (one per quarter, ~21 total)

REFERENCE PERIOD
----------------
τ = -1 (2022Q3) is omitted — all β_τ interpreted as relative to 2022Q3.
This is the standard convention: the quarter immediately before treatment.

IDENTIFICATION
--------------
Treatment: ChatGPT launched November 2022 → 2022Q4 is first treated quarter
           (τ = 0). Industries with higher AIIE scores are "more treated."
Pre-trend: β_τ ≈ 0 for τ < 0 validates parallel trends assumption.
Post-trend: β_τ for τ ≥ 0 shows dynamic treatment effects.

STANDARD ERRORS
---------------
Clustered at the supersector level (robust to serial correlation within
industry). NOTE: Only 16 clusters → standard clustered SEs may be
unreliable. Interpret confidence intervals with caution; treat as
indicative rather than exact.

SAMPLE RESTRICTIONS (defaults)
-------------------------------
  - Exclude Government (90): no earnings data; included for log_emp only
  - Exclude Manufacturing aggregate (30): double-counts SS 31 + SS 32
  - Exclude Trade aggregate (40): double-counts SS 41 + SS 42 + SS 43

OUTPUTS
-------
  results/2021start_log_emp.csv      coefficient table, employment
  results/2021start_log_wage.csv     coefficient table, wages
  results/2021start_log_wbill.csv    coefficient table, wage bill
  results/2021start_log_emp.png      event study plot, employment
  results/2021start_log_wage.png     event study plot, wages
  results/2021start_log_wbill.png    event study plot, wage bill
================================================================================
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


# ── Configuration ─────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR   = Path("results")

# Outcome variables and their labels
OUTCOMES = {
    "log_emp":   "Log Employment (Thousands)",
    "log_wage":  "Log Average Hourly Earnings",
    "log_wbill": "Log Wage Bill (Employment × AHE)",
}

# Reference quarter (omitted from interactions — all β_τ relative to this)
REFERENCE_TAU = -1   # 2022Q3

# Supersectors to EXCLUDE from all regressions (aggregates that double-count)
EXCLUDE_ALWAYS = {
    "30",   # Manufacturing total (= SS 31 + SS 32)
    "40",   # Trade/transport/utilities total (= SS 41 + 42 + 43)
}

# Supersectors to exclude from WAGE regressions only (no earnings data)
EXCLUDE_WAGE = {
    "90",   # Government
}

# Confidence interval level
CI_LEVEL = 0.95
CI_Z     = 1.96   # z for 95% CI (using normal approximation)

# Plot style
PLOT_STYLE = {
    "coef_color":    "#2c7bb6",
    "ci_color":      "#abd9e9",
    "zero_color":    "#d7191c",
    "pre_shade":     "#f7f7f7",
    "treatment_color": "#1a9641",
}


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Step 1: Load and prepare data ─────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load merged CES-Felten panel and prepare for estimation."""
    path = PROCESSED_DIR / "ces_felten_merged.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 04_merge_felten.py first."
        )

    df = pd.read_csv(path, dtype={"supersector_code": str})

    # Create a single time index: year×quarter integer (e.g. 20171 = 2017Q1)
    df["time_id"] = df["year"] * 10 + df["quarter"]

    log.info(
        f"Loaded: {len(df):,} rows, "
        f"{df['supersector_code'].nunique()} supersectors, "
        f"τ range: {df['event_time'].min()} to {df['event_time'].max()}"
    )
    return df


def build_interactions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Create one interaction column per quarter τ (excluding reference period).
    Column name format: inter_tm3 (τ=-3), inter_t0 (τ=0), inter_t2 (τ=2).

    Returns (df_with_interactions, list_of_interaction_colnames).
    """
    taus = sorted(df["event_time"].unique())
    taus = [t for t in taus if t != REFERENCE_TAU]   # drop reference period

    interaction_cols = []
    for tau in taus:
        # Column name: inter_tm3 for τ=-3, inter_t0, inter_t2, etc.
        sign = "m" if tau < 0 else ""
        col  = f"inter_t{sign}{abs(tau)}"
        df[col] = df["aiie_score"] * (df["event_time"] == tau).astype(float)
        interaction_cols.append((tau, col))

    log.info(
        f"Built {len(interaction_cols)} interaction terms "
        f"(τ from {taus[0]} to {taus[-1]}, reference τ={REFERENCE_TAU})"
    )
    return df, interaction_cols


# ── Step 2: Estimate TWFE event study ─────────────────────────────────────────

def estimate_event_study(
    df: pd.DataFrame,
    outcome: str,
    interaction_cols: list[tuple[int, str]],
    exclude_supersectors: set[str],
) -> pd.DataFrame:
    """
    Estimate TWFE OLS event study for one outcome variable.

    Uses linearmodels.PanelOLS with entity (industry) and time fixed effects,
    standard errors clustered at the entity (supersector) level.

    Returns DataFrame with columns:
      tau, coef, se, ci_low, ci_high, pval
    """
    # Sample restriction
    sample = df[~df["supersector_code"].isin(exclude_supersectors)].copy()
    sample = sample.dropna(subset=[outcome, "aiie_score"])

    n_obs = len(sample)
    n_ss  = sample["supersector_code"].nunique()
    log.info(
        f"\n  Outcome: {outcome} | "
        f"N={n_obs:,} | Supersectors={n_ss} | "
        f"Excluded: {exclude_supersectors}"
    )

    # Set panel index: (entity, time)
    sample = sample.set_index(["supersector_code", "time_id"])

    # Build formula: outcome ~ interactions (FEs absorbed)
    inter_names = [col for _, col in interaction_cols]
    formula = f"{outcome} ~ 1 + " + " + ".join(inter_names)

    # Estimate with two-way fixed effects
    model = PanelOLS.from_formula(
        formula + " + EntityEffects + TimeEffects",
        data=sample,
    )

    result = model.fit(
        cov_type="clustered",
        cluster_entity=True,   # cluster SEs at supersector level
    )

    log.info(f"  R² (within): {result.rsquared_within:.4f}")
    log.info(f"  N clusters : {n_ss} (interpret SEs cautiously)")

    # Extract coefficients
    rows = []
    for tau, col in     interaction_cols:
        if col not in result.params.index:
            continue
        coef  = result.params[col]
        se    = result.std_errors[col]
        pval  = result.pvalues[col]
        rows.append({
            "tau":      tau,
            "coef":     coef,
            "se":       se,
            "ci_low":   coef - CI_Z * se,
            "ci_high":  coef + CI_Z * se,
            "pval":     pval,
            "sig":      "*" if pval < 0.05 else ("†" if pval < 0.10 else ""),
        })

    # Add reference period (β = 0 by construction)
    rows.append({
        "tau": REFERENCE_TAU, "coef": 0.0, "se": 0.0,
        "ci_low": 0.0, "ci_high": 0.0, "pval": np.nan, "sig": "",
    })

    coef_df = pd.DataFrame(rows).sort_values("tau").reset_index(drop=True)
    return coef_df


# ── Step 3: Plot event study ───────────────────────────────────────────────────

def plot_event_study(
    coef_df: pd.DataFrame,
    outcome: str,
    outcome_label: str,
    save_path: Path,
) -> None:
    """
    Plot event study coefficients with 95% confidence intervals.

    Layout:
      - Pre-period shaded grey (τ < 0)
      - Treatment line at τ = -0.5 (between Q3 and Q4 2022)
      - Reference period (τ = -1) marked with dashed line at 0
      - Coefficients as filled circles, CIs as shaded band
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    taus     = coef_df["tau"].values
    coefs    = coef_df["coef"].values
    ci_low   = coef_df["ci_low"].values
    ci_high  = coef_df["ci_high"].values

    # ── Shading ───────────────────────────────────────────────────────────────

    # Pre-period shading (τ < 0, excluding reference)
    pre_taus = [t for t in taus if t < 0]
    if pre_taus:
        ax.axvspan(
            min(pre_taus) - 0.5, -0.5,
            alpha=0.08, color=PLOT_STYLE["pre_shade"],
            label="Pre-period", zorder=0
        )

    # ── Reference lines ───────────────────────────────────────────────────────

    # Zero line
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-", zorder=1)

    # Treatment line (between τ=-1 and τ=0)
    ax.axvline(
        -0.5,
        color=PLOT_STYLE["treatment_color"],
        linewidth=1.5, linestyle="--",
        label="ChatGPT launch (2022Q4)", zorder=2
    )

    # Reference period marker
    ax.axvline(
        REFERENCE_TAU,
        color="grey", linewidth=0.8, linestyle=":",
        label=f"Reference period (τ={REFERENCE_TAU})", zorder=2
    )

    # ── CI band ───────────────────────────────────────────────────────────────
    ax.fill_between(
        taus, ci_low, ci_high,
        alpha=0.25, color=PLOT_STYLE["ci_color"],
        label="95% CI", zorder=3
    )

    # ── Coefficients ──────────────────────────────────────────────────────────
    ax.plot(
        taus, coefs,
        color=PLOT_STYLE["coef_color"],
        linewidth=1.5, zorder=4
    )
    ax.scatter(
        taus, coefs,
        color=PLOT_STYLE["coef_color"],
        s=30, zorder=5, label="β_τ estimate"
    )

    # Mark reference period with open circle
    ax.scatter(
        [REFERENCE_TAU], [0],
        color="white", edgecolors=PLOT_STYLE["coef_color"],
        s=50, zorder=6
    )

    # ── X-axis labels: show year×quarter instead of τ ─────────────────────────
    # τ=0 is 2022Q4; τ=-4 is 2021Q4; τ=4 is 2023Q4, etc.
    def tau_to_label(tau):
        total_quarters = 2022 * 4 + 4 + tau   # absolute quarter index
        year = (total_quarters - 1) // 4
        q    = (total_quarters - 1) % 4 + 1
        return f"{year}\nQ{q}"

    # Show every 4th quarter label to avoid crowding
    tick_taus   = [t for t in taus if t % 4 == 0]
    tick_labels = [tau_to_label(t) for t in tick_taus]
    ax.set_xticks(tick_taus)
    ax.set_xticklabels(tick_labels, fontsize=7)

    # ── Labels and formatting ─────────────────────────────────────────────────
    ax.set_xlabel("Quarter", fontsize=11)
    ax.set_ylabel(f"β_τ  ({outcome_label})", fontsize=11)
    ax.set_title(
        f"Period-by-Period Event Study: {outcome_label}\n"
        f"Effect of Felten AIIE × Quarter on {outcome_label} "
        f"(relative to 2022Q3)",
        fontsize=11, fontweight="bold"
    )

    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    # Footnote
    fig.text(
        0.01, 0.01,
        "Note: Two-way FE OLS. Standard errors clustered at supersector level (N=13–14 clusters). "
        "Excludes Manufacturing (30) and Trade (40) aggregates to avoid double-counting.",
        fontsize=6.5, color="grey", va="bottom"
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Plot saved: {save_path}")


# ── Step 4: Summary table ─────────────────────────────────────────────────────

def print_summary(coef_df: pd.DataFrame, outcome: str) -> None:
    """Print pre/post coefficient summary to log."""
    pre  = coef_df[coef_df["tau"] < 0]
    post = coef_df[coef_df["tau"] >= 0]

    log.info(
        f"\n  {outcome} — coefficient summary:"
        f"\n    Pre-period  mean |β_τ|: {pre['coef'].abs().mean():.4f} "
        f"(want ≈ 0 for parallel trends)"
        f"\n    Post-period mean  β_τ : {post['coef'].mean():.4f}"
        f"\n    Post-period max   β_τ : {post['coef'].max():.4f}"
        f"\n    Post-period min   β_τ : {post['coef'].min():.4f}"
    )

    sig_post = post[post["pval"] < 0.05]
    if len(sig_post) > 0:
        log.info(
            f"    Significant post-period quarters (p<0.05): "
            f"{sig_post['tau'].tolist()}"
        )
    else:
        log.info("    No post-period quarters significant at p<0.05")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 68)
    log.info("METHOD 4: PERIOD-BY-PERIOD EVENT STUDY")
    log.info("=" * 68)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data()
    
    # 2021start: restrict pre-period to 2021Q1 onward
    # Drops 2017-2020 entirely for cleaner parallel trends
    df = df[df["year"] >= 2021].copy()
    log.info(
        f"Sample restricted to 2021Q1+: {len(df):,} rows, "
        f"τ range: {df['event_time'].min()} to {df['event_time'].max()}"
    )

    # Build interaction terms (done once, shared across outcomes)
    df, interaction_cols = build_interactions(df)

    # Estimate and plot for each outcome
    for outcome, label in OUTCOMES.items():
        log.info(f"\n{'─'*68}\nOutcome: {outcome}\n{'─'*68}")

        # Wage outcomes exclude Government (no earnings data)
        exclude = EXCLUDE_ALWAYS.copy()
        if outcome in ("log_wage", "log_wbill"):
            exclude |= EXCLUDE_WAGE

        # Estimate
        coef_df = estimate_event_study(df, outcome, interaction_cols, exclude)

        # Save coefficient table
        csv_path = RESULTS_DIR / f"2021start_{outcome}.csv"
        coef_df.to_csv(csv_path, index=False)
        log.info(f"  Coefficients saved: {csv_path}")

        # Summary
        print_summary(coef_df, outcome)

        # Plot
        png_path = RESULTS_DIR / f"2021start_{outcome}.png"
        plot_event_study(coef_df, outcome, label, png_path)

    log.info(f"\n{'='*68}")
    log.info("DONE — all outputs in results/")
    log.info(f"{'='*68}")
    log.info("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTERPRETING THE PLOTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pre-period (left of green line):
  β_τ ≈ 0 throughout → parallel trends holds → identification valid
  β_τ trending → pre-existing divergence → results less credible

Post-period (right of green line):
  β_τ < 0 → displacement: high-AIIE industries losing relative to low
  β_τ > 0 → augmentation: high-AIIE industries gaining
  β_τ ≈ 0 → no detectable effect yet (consistent with Acemoglu 2024)
  Growing magnitude → effects still accumulating (short window caveat)

ROBUSTNESS CHECKS TO RUN NEXT:
  2. Use aiie_zscore instead of aiie_score
  3. Drop one supersector at a time (leave-one-out)
  4. Move treatment date to 2023Q1 instead of 2022Q4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)


if __name__ == "__main__":
    main()
