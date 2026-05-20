"""
================================================================================
ROBUSTNESS CHECK: Leave-One-Out Event Study
================================================================================
Project : LLMs and Labor Market Outcomes
Course  : Econometrics & ML — Spring 2025, University of Chicago
Person  : Lynnard (Method 4 — Period-by-Period Event Study)

PURPOSE
-------
Runs the 2021start event study repeatedly, dropping one supersector at a time,
to verify that no single industry is driving the main results.

For each outcome, produces one overlay plot:
  - Thin grey lines: leave-one-out coefficient paths (one per dropped supersector)
  - Thick blue line: main result from 06_2021start.py (aiie_score, full sample)
  - Shaded band: range (min/max) across all leave-one-out runs

If all grey lines cluster tightly around the blue line → results are robust.
If one grey line diverges wildly → that supersector is influential.

SPEC
----
Identical to 06_2021start.py:
  - Sample: 2021Q1 onward
  - Treatment: aiie_score
  - Reference: τ = -1 (2022Q3)
  - Excluded always: Manufacturing (30), Trade (40) aggregates

OUTPUTS
-------
  results/loo_log_emp.png
  results/loo_log_wage.png
  results/loo_log_wbill.png
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

OUTCOMES = {
    "log_emp":   "Log Employment (Thousands)",
    "log_wage":  "Log Average Hourly Earnings",
    "log_wbill": "Log Wage Bill (Employment × AHE)",
}

REFERENCE_TAU = -1

EXCLUDE_ALWAYS = {"30", "40"}
EXCLUDE_WAGE   = {"90"}

CI_Z = 1.96

PLOT_STYLE = {
    "main_color":  "#2c7bb6",
    "loo_color":   "#aaaaaa",
    "range_color": "#dddddd",
    "treatment_color": "#1a9641",
}


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "ces_felten_merged.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 04_merge_felten.py first.")

    df = pd.read_csv(path, dtype={"supersector_code": str})
    df["time_id"] = df["year"] * 10 + df["quarter"]

    # 2021start restriction
    df = df[df["year"] >= 2021].copy()
    log.info(
        f"Loaded: {len(df):,} rows, "
        f"{df['supersector_code'].nunique()} supersectors, "
        f"τ range: {df['event_time'].min()} to {df['event_time'].max()}"
    )
    return df


def build_interactions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[int, str]]]:
    taus = sorted(df["event_time"].unique())
    taus = [t for t in taus if t != REFERENCE_TAU]

    interaction_cols = []
    for tau in taus:
        sign = "m" if tau < 0 else ""
        col  = f"inter_t{sign}{abs(tau)}"
        df[col] = df["aiie_score"] * (df["event_time"] == tau).astype(float)
        interaction_cols.append((tau, col))

    return df, interaction_cols


def estimate_one(
    df: pd.DataFrame,
    outcome: str,
    interaction_cols: list[tuple[int, str]],
    exclude_supersectors: set[str],
) -> pd.DataFrame:
    """Run TWFE OLS for one outcome, return coefficient DataFrame."""
    sample = df[~df["supersector_code"].isin(exclude_supersectors)].copy()
    sample = sample.dropna(subset=[outcome, "aiie_score"])

    if sample["supersector_code"].nunique() < 3:
        return pd.DataFrame()   # too few clusters, skip

    sample = sample.set_index(["supersector_code", "time_id"])

    inter_names = [col for _, col in interaction_cols]
    formula = f"{outcome} ~ 1 + " + " + ".join(inter_names)

    try:
        model  = PanelOLS.from_formula(
            formula + " + EntityEffects + TimeEffects",
            data=sample,
        )
        result = model.fit(cov_type="clustered", cluster_entity=True)
    except Exception as e:
        log.warning(f"  Estimation failed: {e}")
        return pd.DataFrame()

    rows = []
    for tau, col in interaction_cols:
        if col not in result.params.index:
            continue
        coef = result.params[col]
        se   = result.std_errors[col]
        rows.append({
            "tau":     tau,
            "coef":    coef,
            "ci_low":  coef - CI_Z * se,
            "ci_high": coef + CI_Z * se,
        })

    # Add reference period
    rows.append({"tau": REFERENCE_TAU, "coef": 0.0, "ci_low": 0.0, "ci_high": 0.0})
    return pd.DataFrame(rows).sort_values("tau").reset_index(drop=True)


# ── Leave-one-out loop ────────────────────────────────────────────────────────

def run_loo(
    df: pd.DataFrame,
    outcome: str,
    interaction_cols: list[tuple[int, str]],
    base_exclude: set[str],
) -> tuple[pd.DataFrame, list[tuple[str, pd.DataFrame]]]:
    """
    Run event study once with full sample (main result) and once per
    supersector dropped (leave-one-out).

    Returns (main_coefs, [(ss_code, loo_coefs), ...])
    """
    # Main result (no extra exclusions)
    log.info(f"  Main estimate...")
    main_coefs = estimate_one(df, outcome, interaction_cols, base_exclude)

    # Get supersectors actually in this regression
    sample = df[~df["supersector_code"].isin(base_exclude)].copy()
    sample = sample.dropna(subset=[outcome, "aiie_score"])
    supersectors = sorted(sample["supersector_code"].unique())

    log.info(f"  Running {len(supersectors)} leave-one-out iterations...")
    loo_results = []
    for ss in supersectors:
        exclude = base_exclude | {ss}
        ss_name = df.loc[df["supersector_code"] == ss, "supersector_name"].iloc[0]
        coefs = estimate_one(df, outcome, interaction_cols, exclude)
        if not coefs.empty:
            loo_results.append((ss, ss_name, coefs))
            log.info(f"    Dropped {ss} ({ss_name}): "
                     f"post mean β = {coefs[coefs['tau']>=0]['coef'].mean():.5f}")

    return main_coefs, loo_results


# ── Plot overlay ──────────────────────────────────────────────────────────────

def plot_loo_overlay(
    main_coefs: pd.DataFrame,
    loo_results: list[tuple[str, str, pd.DataFrame]],
    outcome: str,
    outcome_label: str,
    save_path: Path,
) -> None:
    """
    Overlay plot: all LOO paths in grey, main result in blue,
    min/max range as light grey band.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    taus = main_coefs["tau"].values

    # ── Range band (min/max across all LOO runs) ──────────────────────────────
    if loo_results:
        all_coefs = np.array([
            r[2].set_index("tau").reindex(taus)["coef"].values
            for r in loo_results
        ])
        range_low  = np.nanmin(all_coefs, axis=0)
        range_high = np.nanmax(all_coefs, axis=0)
        ax.fill_between(
            taus, range_low, range_high,
            alpha=0.3, color=PLOT_STYLE["range_color"],
            label="LOO range (min/max)", zorder=2
        )

    # ── Individual LOO paths ──────────────────────────────────────────────────
    for i, (ss_code, ss_name, coefs) in enumerate(loo_results):
        aligned = coefs.set_index("tau").reindex(taus)["coef"].values
        ax.plot(
            taus, aligned,
            color=PLOT_STYLE["loo_color"],
            linewidth=0.8, alpha=0.6, zorder=3,
            label="Drop one supersector" if i == 0 else "_nolegend_"
        )

    # ── Main result ───────────────────────────────────────────────────────────
    ax.plot(
        taus, main_coefs["coef"].values,
        color=PLOT_STYLE["main_color"],
        linewidth=2.5, zorder=5, label="Main result (06_2021start)"
    )
    ax.scatter(
        taus, main_coefs["coef"].values,
        color=PLOT_STYLE["main_color"],
        s=25, zorder=6
    )
    ax.scatter(
        [REFERENCE_TAU], [0],
        color="white", edgecolors=PLOT_STYLE["main_color"],
        s=50, zorder=7
    )

    # ── Reference lines ───────────────────────────────────────────────────────
    ax.axhline(0, color="black", linewidth=0.8, zorder=1)
    ax.axvline(
        -0.5,
        color=PLOT_STYLE["treatment_color"],
        linewidth=1.5, linestyle="--",
        label="ChatGPT launch (2022Q4)", zorder=4
    )
    ax.axvline(
        REFERENCE_TAU,
        color="grey", linewidth=0.8, linestyle=":",
        label=f"Reference period (τ={REFERENCE_TAU})", zorder=4
    )

    # ── X-axis labels ─────────────────────────────────────────────────────────
    def tau_to_label(tau):
        total_quarters = 2022 * 4 + 4 + tau
        year = (total_quarters - 1) // 4
        q    = (total_quarters - 1) % 4 + 1
        return f"{year}\nQ{q}"

    tick_taus   = [t for t in taus if t % 4 == 0]
    tick_labels = [tau_to_label(t) for t in tick_taus]
    ax.set_xticks(tick_taus)
    ax.set_xticklabels(tick_labels, fontsize=7)

    # ── Labels ────────────────────────────────────────────────────────────────
    n_loo = len(loo_results)
    ax.set_xlabel("Quarter", fontsize=11)
    ax.set_ylabel(f"β_τ  ({outcome_label})", fontsize=11)
    ax.set_title(
        f"Leave-One-Out Robustness: {outcome_label}\n"
        f"Each grey line drops one supersector (N={n_loo} iterations). "
        f"Blue = main result.",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.text(
        0.01, 0.01,
        "Note: Two-way FE OLS, 2021Q1+ sample, aiie_score. "
        "Each iteration drops one supersector from estimation. "
        "Grey band = min/max range across all LOO runs.",
        fontsize=6.5, color="grey", va="bottom"
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Plot saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 68)
    log.info("ROBUSTNESS: LEAVE-ONE-OUT EVENT STUDY")
    log.info("=" * 68)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df, interaction_cols = build_interactions(df)

    for outcome, label in OUTCOMES.items():
        log.info(f"\n{'─'*68}\nOutcome: {outcome}\n{'─'*68}")

        base_exclude = EXCLUDE_ALWAYS.copy()
        if outcome in ("log_wage", "log_wbill"):
            base_exclude |= EXCLUDE_WAGE

        main_coefs, loo_results = run_loo(
            df, outcome, interaction_cols, base_exclude
        )

        # Summary: how much do LOO results vary?
        if loo_results:
            post_mains = [
                r[2][r[2]["tau"] >= 0]["coef"].mean()
                for r in loo_results
            ]
            log.info(
                f"\n  LOO post-period mean β range: "
                f"{min(post_mains):.5f} to {max(post_mains):.5f}"
                f"\n  Main post-period mean β     : "
                f"{main_coefs[main_coefs['tau']>=0]['coef'].mean():.5f}"
            )

        png_path = RESULTS_DIR / f"loo_{outcome}.png"
        plot_loo_overlay(main_coefs, loo_results, outcome, label, png_path)

    log.info(f"\n{'='*68}")
    log.info("DONE — LOO plots saved in results/")
    log.info(f"{'='*68}")


if __name__ == "__main__":
    main()
