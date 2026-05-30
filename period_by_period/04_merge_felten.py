"""
================================================================================
DATASET 4: Merge CES Quarterly Panel with Felten et al. AIIE Exposure Scores
================================================================================
Project : LLMs and Labor Market Outcomes
Course  : Econometrics & ML — Spring 2025, University of Chicago
Person  : Lynnard (Method 4 — Period-by-Period Event Study)

PURPOSE
-------
Merges the CES quarterly panel (output of 03_process_ces_v2.py) with
Felten et al. (2021) industry-level AI exposure scores (AIIE).

Felten's AIIE scores are indexed by 4-digit NAICS codes. CES uses 2-digit
supersector codes. This script aggregates Felten's 4-digit scores to the
supersector level using an unweighted mean across constituent industries,
then merges onto the CES panel.

FELTEN AIIE SCORES — WHAT THEY MEAN
-------------------------------------
AIIE = AI Industry Exposure Index, from Felten, Raj & Seamans (2021).
Measures the degree to which AI capabilities (as of ~2019) overlap with
the tasks performed in each industry, based on O*NET task content and an
mTurk survey of AI application-ability relatedness.

Key difference from Eloundou et al.:
  - Felten: industry-level (NAICS), pre-GPT-4 AI broadly
  - Eloundou: occupation-level (SOC), GPT-4 specifically
  Both are pre-determined relative to ChatGPT launch — valid for identification.

AIIE is standardized (mean ~0, not bounded [0,1]).
  Positive = more AI-exposed than average industry
  Negative = less AI-exposed than average industry
Interpretation of β: effect per 1-unit increase in AIIE (≈ 1 SD).

NAICS → CES SUPERSECTOR MAPPING
---------------------------------
Felten 4-digit NAICS codes are grouped by their 2-digit prefix and mapped
to CES supersectors following BLS industry definitions:

  CES SS  NAICS prefixes  Description
  ------  --------------  -----------
  10      11, 21          Mining and logging
  20      23              Construction
  30      31, 32, 33      Manufacturing (all)
  31      33              Durable goods
  32      31, 32          Nondurable goods
  40      22,42,44,45,    Trade, transportation, and utilities
          48, 49
  41      42              Wholesale trade
  42      44, 45          Retail trade
  43      48, 49          Transportation and warehousing
  50      51              Information
  55      52, 53          Financial activities
  60      54, 55, 56      Professional and business services
  65      61, 62          Education and health services
  70      71, 72          Leisure and hospitality
  80      81              Other services
  90      99              Government

Aggregation: unweighted mean of AIIE across all 4-digit NAICS codes
within each supersector. This follows Felten et al.'s own aggregation
convention (no employment weights at this level of aggregation).

LIMITATION (flag in paper)
---------------------------
The NAICS→supersector aggregation loses within-supersector heterogeneity.
For example, supersector 55 (Financial Activities) combines securities
trading (high AI exposure) with insurance (lower). This is an inherent
limitation of matching industry-level exposure to CES supersector data
and should be noted in the Data section.

INPUTS
------
  data/processed/ces_quarterly_panel.csv   ← from 03_process_ces_v2.py
  data/raw/felten/AIOE_DataAppendix.csv    ← Felten et al. Appendix B

OUTPUTS
-------
  data/processed/ces_felten_merged.csv     ← main output (use for estimation)
  data/processed/felten_supersector_crosswalk.csv  ← aggregated AIIE per SS

MERGED PANEL STRUCTURE
----------------------
  [all columns from ces_quarterly_panel.csv, plus:]
  aiie_score       Felten AIIE score aggregated to supersector level
  n_naics          Number of 4-digit NAICS codes averaged into aiie_score
  aiie_zscore      Standardized AIIE (z-score across supersectors, mean=0 SD=1)
                   Useful for comparing coefficient magnitudes across methods
================================================================================
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
FELTEN_PATH   = Path("data/raw/felten/AIOE_DataAppendix.csv")

# Fallback: check project root if not in data/raw/felten/
FELTEN_FALLBACK = Path("AIOE_DataAppendix.csv")

# NAICS 2-digit prefix → CES supersector mapping
# Based on BLS industry definitions
NAICS_TO_SUPERSECTOR = {
    "11": "10",   # Agriculture/forestry/fishing → Mining & logging
    "21": "10",   # Mining → Mining & logging
    "22": "40",   # Utilities → Trade, transportation, utilities
    "23": "20",   # Construction
    "31": "32",   # Nondurable mfg (part 1)
    "32": "32",   # Nondurable mfg (part 2)
    "33": "31",   # Durable mfg
    "42": "41",   # Wholesale trade
    "44": "42",   # Retail trade (part 1)
    "45": "42",   # Retail trade (part 2)
    "48": "43",   # Transportation & warehousing (part 1)
    "49": "43",   # Transportation & warehousing (part 2)
    "51": "50",   # Information
    "52": "55",   # Finance & insurance → Financial activities
    "53": "55",   # Real estate → Financial activities
    "54": "60",   # Professional & technical services
    "55": "60",   # Management of companies
    "56": "60",   # Administrative & support services
    "61": "65",   # Educational services
    "62": "65",   # Health care & social assistance
    "71": "70",   # Arts, entertainment & recreation
    "72": "70",   # Accommodation & food services
    "81": "80",   # Other services
    "99": "90",   # Government (OES designations)
}

# Also map Manufacturing aggregate (SS 30) = all of 31+32+33
# This is handled separately since 31/32/33 already map to 31/32
MANUFACTURING_PREFIXES = {"31", "32", "33"}


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Step 1: Load and aggregate Felten AIIE scores ────────────────────────────

def load_felten(path: Path) -> pd.DataFrame:
    """Load Felten AIOE_DataAppendix.csv (Appendix B — industry scores)."""
    if not path.exists():
        if FELTEN_FALLBACK.exists():
            log.warning(f"Felten file not at {path}, using fallback: {FELTEN_FALLBACK}")
            path = FELTEN_FALLBACK
        else:
            raise FileNotFoundError(
                f"Felten AIIE file not found at {path} or {FELTEN_FALLBACK}\n"
                "Download AIOE_DataAppendix.xlsx from:\n"
                "  https://github.com/AIOE-Data/AIOE\n"
                "Extract Appendix B sheet as CSV and place at:\n"
                f"  {path}"
            )

    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Rename for clarity
    df = df.rename(columns={
        "NAICS":          "naics_code",
        "Industry Title": "industry_title",
        "AIIE":           "aiie_raw",
    })

    df["naics_code"] = df["naics_code"].str.strip()
    df["aiie_raw"]   = pd.to_numeric(df["aiie_raw"], errors="coerce")
    df["naics_2digit"] = df["naics_code"].str[:2]

    log.info(
        f"Felten AIIE: {len(df):,} industries loaded\n"
        f"  AIIE range: {df['aiie_raw'].min():.3f} to {df['aiie_raw'].max():.3f}\n"
        f"  AIIE mean : {df['aiie_raw'].mean():.3f}  std: {df['aiie_raw'].std():.3f}"
    )
    return df


def aggregate_to_supersector(felten: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 4-digit NAICS AIIE scores to CES supersector level.
    Supersector 30 (Manufacturing total) = mean of SS 31 + SS 32 codes.
    Returns one row per supersector with aiie_score and n_naics.
    """
    felten = felten.copy()
    felten["supersector_code"] = felten["naics_2digit"].map(NAICS_TO_SUPERSECTOR)

    unmapped = felten["supersector_code"].isna().sum()
    if unmapped > 0:
        log.warning(
            f"{unmapped} Felten industries could not be mapped to a supersector:"
        )
        log.warning(
            felten[felten["supersector_code"].isna()][
                ["naics_code", "industry_title"]
            ].to_string()
        )

    # Aggregate to supersector (unweighted mean)
    agg = (
        felten.dropna(subset=["supersector_code"])
        .groupby("supersector_code")
        .agg(
            aiie_score=("aiie_raw", "mean"),
            n_naics   =("naics_code", "count"),
        )
        .reset_index()
    )

    # Add Manufacturing aggregate (SS 30) = mean of all 31/32/33 NAICS
    mfg_mask = felten["naics_2digit"].isin(MANUFACTURING_PREFIXES)
    mfg_aiie = felten.loc[mfg_mask, "aiie_raw"].mean()
    mfg_n    = mfg_mask.sum()
    mfg_row  = pd.DataFrame([{
        "supersector_code": "30",
        "aiie_score":       mfg_aiie,
        "n_naics":          mfg_n,
    }])
    agg = pd.concat([agg, mfg_row], ignore_index=True)
    # Keep only the row with more industries if duplicates exist
    agg = agg.sort_values("n_naics", ascending=False)
    agg = agg.drop_duplicates(subset="supersector_code", keep="first")

    agg["supersector_code"] = agg["supersector_code"].astype(str)
    agg = agg.sort_values("supersector_code").reset_index(drop=True)

    # Z-score across supersectors (useful for cross-method comparisons)
    agg["aiie_zscore"] = (
        (agg["aiie_score"] - agg["aiie_score"].mean())
        / agg["aiie_score"].std()
    )

    log.info(f"\nAggregated AIIE scores ({len(agg)} supersectors):")
    print(agg[["supersector_code", "aiie_score", "aiie_zscore", "n_naics"]]
          .sort_values("aiie_score", ascending=False)
          .to_string(index=False))

    return agg


# ── Step 2: Merge onto CES panel ──────────────────────────────────────────────

def merge_onto_panel(
    panel: pd.DataFrame,
    ss_aiie: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-merge CES quarterly panel with supersector AIIE scores.
    supersector_code must match as string in both DataFrames.
    """
    panel = panel.copy()
    panel["supersector_code"] = panel["supersector_code"].astype(str)

    n_before = len(panel)
    merged = panel.merge(
        ss_aiie[["supersector_code", "aiie_score", "aiie_zscore", "n_naics"]],
        on="supersector_code",
        how="left",
        validate="m:1",
    )
    assert len(merged) == n_before, "Merge changed row count — check for duplicates"

    null_aiie = merged["aiie_score"].isna().sum()
    if null_aiie > 0:
        missing_ss = merged.loc[
            merged["aiie_score"].isna(), "supersector_code"
        ].unique()
        log.warning(
            f"{null_aiie} panel rows have no AIIE score. "
            f"Missing supersectors: {missing_ss}"
        )
    else:
        log.info("All panel rows matched to an AIIE score.")

    return merged


# ── Step 3: Sanity checks ─────────────────────────────────────────────────────

def print_sanity_checks(merged: pd.DataFrame) -> None:
    sep = "=" * 68
    log.info(f"\n{sep}\nSANITY CHECKS\n{sep}")

    # 1. Coverage
    log.info(
        f"\nMerged panel: {len(merged):,} rows"
        f"\n  Supersectors with AIIE : "
        f"{merged['aiie_score'].notna().groupby(merged['supersector_code']).all().sum()}"
        f"\n  Rows missing AIIE      : {merged['aiie_score'].isna().sum()}"
        f"\n  AIIE range in panel    : "
        f"{merged['aiie_score'].min():.3f} to {merged['aiie_score'].max():.3f}"
    )

    # 2. Face validity — high vs low exposure
    one_year = merged[merged["year"] == 2019].drop_duplicates("supersector_code")
    ranked = one_year[["supersector_name", "aiie_score"]].sort_values(
        "aiie_score", ascending=False
    )
    log.info("\nSupersectors ranked by AIIE (face validity check):")
    log.info("  High exposure (expect: Finance, Info, Professional services):")
    print(ranked.head(5).to_string(index=False))
    log.info("  Low exposure (expect: Construction, Manufacturing, Leisure):")
    print(ranked.tail(5).to_string(index=False))

    # 3. Treatment variable variation
    log.info(
        "\nAIIE variation across supersectors (needed for identification):"
    )
    print(
        merged.drop_duplicates("supersector_code")[
            ["supersector_name", "aiie_score", "aiie_zscore"]
        ].describe().round(3).to_string()
    )

    # 4. Sample event-study check: Information vs Construction
    log.info(
        "\nSample panel rows — Information (high AIIE) vs Construction (low AIIE):"
    )
    sample = merged[
        merged["supersector_code"].isin(["50", "20"])
        & merged["year"].isin([2021, 2022, 2023])
        & (merged["quarter"] == 1)
    ][["supersector_code", "supersector_name", "year", "quarter_label",
    "event_time", "post", "log_emp", "log_wage", "aiie_score"]]
    print(sample.sort_values(
        ["supersector_code", "year"]
    ).to_string(index=False))

    # 5. Reminder about exclusions for regression
    log.info("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SAMPLE RESTRICTIONS for estimation (05_event_study.py):
  - Exclude Government (supersector 90) from wage regressions
    (no avg_hourly_earnings data)
  - Exclude Manufacturing aggregate (30) from regressions if also
    using Durable (31) and Nondurable (32) — would double-count
  - Same for Trade aggregate (40) if using 41/42/43 separately
  - preliminary == 1: keep but flag as robustness check
  - Consider dropping 2020 (COVID shock) as robustness check
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 68)
    log.info("MERGE: CES QUARTERLY PANEL × FELTEN AIIE SCORES")
    log.info("=" * 68)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load CES panel
    ces_path = PROCESSED_DIR / "ces_quarterly_panel.csv"
    if not ces_path.exists():
        raise FileNotFoundError(
            f"{ces_path} not found. Run 03_process_ces_v2.py first."
        )
    panel = pd.read_csv(ces_path)
    log.info(
        f"CES panel: {len(panel):,} rows, "
        f"{panel['supersector_code'].nunique()} supersectors"
    )

    # Load and aggregate Felten
    felten = load_felten(FELTEN_PATH)
    ss_aiie = aggregate_to_supersector(felten)

    # Save supersector crosswalk for reference
    crosswalk_path = PROCESSED_DIR / "felten_supersector_crosswalk.csv"
    ss_aiie.to_csv(crosswalk_path, index=False)
    log.info(f"\nCrosswalk saved: {crosswalk_path}")

    # Merge
    log.info("\nMerging AIIE scores onto CES panel...")
    merged = merge_onto_panel(panel, ss_aiie)

    # Sanity checks
    print_sanity_checks(merged)

    # Save
    out_path = PROCESSED_DIR / "ces_felten_merged.csv"
    merged.to_csv(out_path, index=False)
    log.info(f"\nOutput saved: {out_path}  ({len(merged):,} rows)")

    log.info("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Column guide for estimation (05_event_study.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTCOMES:
  log_emp     ln(employment thousands)    ← OUTCOME 1
  log_wage    ln(avg hourly earnings)     ← OUTCOME 2
  log_wbill   ln(AHE × employment)        ← OUTCOME 3

TREATMENT:
  aiie_score   Felten AIIE (standardized, mean~0)  ← PRIMARY
  aiie_zscore  Z-scored across supersectors         ← for comparability

TIMING:
  event_time  τ = quarters since 2022Q4  (τ=0 is 2022Q4)
  post        1 = quarter >= 2022Q4

FIXED EFFECTS:
  supersector_code   industry FE
  year + quarter     time FEs (add as dummies or use two-way FE)

EVENT STUDY EQUATION:
  log_y_it = α + Σ_τ β_τ·(aiie_i × 1[t=τ]) + γ_i + δ_t + ε_it

  β_τ for τ < 0  →  pre-trend test (should be ≈ 0)
  β_τ for τ ≥ 0  →  dynamic treatment effects post-ChatGPT

Next step → run 05_event_study.py to estimate and plot the event study.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)


if __name__ == "__main__":
    main()
