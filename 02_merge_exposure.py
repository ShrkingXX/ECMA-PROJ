"""
================================================================================
DATASET 2: Merge OEWS Panel with Eloundou et al. LLM Exposure Scores
================================================================================
Project : LLMs and Labor Market Outcomes
Course  : Econometrics & ML — Spring 2025, University of Chicago
Person  : B (data lead)

PURPOSE
-------
Merges the OEWS national panel (output of 01_process_oews.py) with the
Eloundou et al. (2023) occupation-level LLM exposure scores (occ_level.csv).
Handles the O*NET-to-SOC code mapping, aggregates subspecialty scores,
and imputes exposure for unmatched codes via sibling-based averaging.

INPUTS
------
  data/processed/oews_national_panel.csv   ← from 01_process_oews.py
  data/raw/eloundou/occ_level.csv          ← from Eloundou et al. GitHub

ELOUNDOU EXPOSURE SCORES — WHAT THEY MEAN
------------------------------------------
The file has 6 exposure measures per O*NET occupation, representing the
fraction of tasks in that occupation that LLMs can perform, under three
different capability assumptions:

  Threshold | Meaning
  --------- | -------
  alpha     | LLMs can complete tasks using text I/O alone (narrow)
  beta      | LLMs + complementary tools (e.g., code interpreter, search)
  gamma     | LLMs have access to all possible information (broad)

  Source    | Meaning
  --------- | -------
  dv        | GPT-4 model self-ratings of task exposure
  human     | Human annotators' ratings of the same tasks

  Primary measure  : dv_beta   (model-rated, beta capability assumption)
  Robustness check : human_beta (human-rated, same threshold)

  Eloundou et al. use dv_beta as their headline exposure variable throughout
  the paper. We follow the same convention and use human_beta as our
  alternative-measure robustness check (Section 4, Person C's script).

O*NET → SOC CODE MAPPING
-------------------------
Eloundou's file uses O*NET codes (e.g., "11-1011.00", "11-1011.03").
OEWS uses 6-digit SOC codes (e.g., "11-1011").

Mapping: truncate O*NET code to first 7 characters → SOC prefix.

Some SOC codes have multiple O*NET subspecialties (e.g., "11-1011" has
both "Chief Executives" and "Chief Sustainability Officers"). We aggregate
by taking the employment-unweighted mean across subspecialties within each
SOC. This follows the approach in Eloundou et al.'s own aggregation.

HANDLING UNMATCHED CODES (89 OEWS codes not in Eloundou)
---------------------------------------------------------
Three categories:

1. SOC code split (OEWS aggregate = multiple O*NET codes):
   e.g., 31-1120 "Home Health and Personal Care Aides" →
         31-1121 (Home Health Aides, dv_beta=0.077) +
         31-1122 (Personal Care Aides, dv_beta=0.222)
   Strategy: impute as mean of constituent codes sharing the 5-char prefix.
   Flag: exposure_imputed = 1

2. Residual "All Other" categories:
   e.g., "Information and Record Clerks, All Other" (43-4199)
   These aggregate workers from many different detailed occupations.
   Strategy: impute from sibling codes at 5-char prefix level.
   Flag: exposure_imputed = 1, exposure_residual = 1
   NOTE: These should be EXCLUDED from your main regressions and used only
   in robustness checks, as their exposure scores are noisier.

3. Truly unmappable (no siblings found):
   Strategy: kept in panel with exposure = NaN, exposure_imputed = 0.
   These drop out naturally when you filter to non-null exposure.

OUTPUTS
-------
  data/processed/oews_exposure_merged.csv    ← main output (use this forward)
  data/processed/merge_diagnostics.csv       ← per-code match status report

MERGED PANEL STRUCTURE
----------------------
  [all columns from oews_national_panel.csv, plus:]
  dv_alpha         Eloundou alpha exposure, model-rated
  dv_beta          Eloundou beta exposure, model-rated    ← PRIMARY TREATMENT VAR
  dv_gamma         Eloundou gamma exposure, model-rated
  human_alpha      Eloundou alpha exposure, human-rated
  human_beta       Eloundou beta exposure, human-rated    ← ROBUSTNESS CHECK
  human_gamma      Eloundou gamma exposure, human-rated
  n_onet           Number of O*NET subspecialties averaged (1 = exact match)
  exposure_imputed 1 = exposure score was imputed from sibling SOC codes
  exposure_residual 1 = occupation is a residual "All Other" category
================================================================================
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
RAW_ELOUNDOU  = Path("data/raw/eloundou/occ_level.csv")

# Primary and robustness exposure measures
EXPOSURE_COLS = [
    "dv_alpha", "dv_beta", "dv_gamma",
    "human_alpha", "human_beta", "human_gamma",
]

# "All Other" and residual category patterns — these get a special flag
RESIDUAL_PATTERNS = [
    "all other", "not elsewhere classified", "n.e.c.",
    "miscellaneous", "except ", ", all other",
]


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Step 1: Load and aggregate Eloundou exposure scores ──────────────────────

def load_exposure(path: Path) -> pd.DataFrame:
    """
    Load the Eloundou occ_level.csv file, truncate O*NET codes to 6-digit
    SOC codes, and aggregate subspecialties by taking the mean within each SOC.

    Returns one row per SOC code with all 6 exposure measures.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Eloundou file not found: {path}\n"
            "Download from: https://github.com/openai/evals or the paper's\n"
            "supplementary materials at https://arxiv.org/abs/2303.10130\n"
            f"Place occ_level.csv in: {path.parent}/"
        )

    raw = pd.read_csv(path, dtype=str)
    log.info(f"Eloundou file: {len(raw):,} O*NET rows, columns: {list(raw.columns)}")

    # Rename for clarity
    raw = raw.rename(columns={
        "O*NET-SOC Code":    "onet_code",
        "Title":             "onet_title",
        "dv_rating_alpha":   "dv_alpha",
        "dv_rating_beta":    "dv_beta",
        "dv_rating_gamma":   "dv_gamma",
        "human_rating_alpha":"human_alpha",
        "human_rating_beta": "human_beta",
        "human_rating_gamma":"human_gamma",
    })

    # Truncate O*NET code "11-1011.00" → SOC code "11-1011"
    raw["soc_code"] = raw["onet_code"].str[:7]

    # Convert exposure columns to numeric
    for col in EXPOSURE_COLS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # Aggregate: mean across subspecialties within each SOC code
    # (follows Eloundou et al.'s own aggregation convention)
    agg = raw.groupby("soc_code").agg(
        **{col: (col, "mean") for col in EXPOSURE_COLS},
        n_onet=("onet_code", "count"),
    ).reset_index()

    n_multi = (agg["n_onet"] > 1).sum()
    log.info(
        f"Aggregated to {len(agg):,} SOC codes "
        f"({n_multi} had multiple O*NET subspecialties → averaged)"
    )
    return agg


# ── Step 2: Sibling-based imputation for unmatched OEWS codes ────────────────

def build_sibling_imputer(expo_agg: pd.DataFrame) -> dict[str, dict]:
    """
    Build a lookup: for any 7-char SOC code prefix, return the mean exposure
    across all Eloundou codes sharing the first 5 characters.

    Used to impute exposure for OEWS codes not directly in Eloundou.
    e.g., 31-1120 → mean of 31-1121 and 31-1122
    """
    expo_agg["prefix5"] = expo_agg["soc_code"].str[:5]
    sibling_lookup = {}
    for prefix, group in expo_agg.groupby("prefix5"):
        sibling_lookup[prefix] = {
            col: group[col].mean() for col in EXPOSURE_COLS
        }
        sibling_lookup[prefix]["n_onet"]    = len(group)
        sibling_lookup[prefix]["siblings"]  = group["soc_code"].tolist()
    return sibling_lookup


def is_residual(title: str) -> bool:
    """Return True if the occupation title looks like a residual 'All Other' category."""
    t = str(title).lower()
    return any(pat in t for pat in RESIDUAL_PATTERNS)


# ── Step 3: Merge and impute ──────────────────────────────────────────────────

def merge_and_impute(
    panel: pd.DataFrame,
    expo_agg: pd.DataFrame,
    sibling_lookup: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Left-merge OEWS panel with exposure scores.
    For unmatched codes, attempt sibling-based imputation.

    Returns (merged_panel, diagnostics_df).
    """
    n_panel = len(panel)

    # Direct merge
    merged = panel.merge(
        expo_agg.rename(columns={"soc_code": "occ_code"}),
        on="occ_code",
        how="left",
        validate="m:1",    # many panel rows per SOC code (one per year)
    )

    assert len(merged) == n_panel, "Merge changed row count — check for duplicates"

    # Initialise imputation flags
    merged["exposure_imputed"]  = 0
    merged["exposure_residual"] = 0

    # Identify unmatched occupations (missing dv_beta after direct merge)
    unmatched_occs = (
        merged[merged["dv_beta"].isna()]["occ_code"]
        .unique()
    )
    log.info(
        f"\nDirect merge: {panel['occ_code'].nunique() - len(unmatched_occs):,} "
        f"SOC codes matched directly, {len(unmatched_occs):,} unmatched"
    )

    diagnostics = []

    # Use the most recent year's title for residual detection.
    # Earlier years may carry legacy "All Other" titles for codes that were
    # later renamed (e.g., 13-2051 was "Financial and Investment Analysts,
    # Financial Risk Specialists, and Financial Specialists, All Other" in
    # 2019 but became "Financial and Investment Analysts" in 2020+).
    latest_title = (
        merged.sort_values("year")
        .groupby("occ_code")["occ_title"]
        .last()
    )

    for occ in unmatched_occs:
        prefix5 = occ[:5]
        title   = latest_title.get(occ, merged.loc[merged["occ_code"] == occ, "occ_title"].iloc[0])
        residual_flag = int(is_residual(title))

        if prefix5 in sibling_lookup:
            sib = sibling_lookup[prefix5]
            # Fill exposure columns from sibling mean
            for col in EXPOSURE_COLS:
                merged.loc[merged["occ_code"] == occ, col] = sib[col]
            merged.loc[merged["occ_code"] == occ, "n_onet"]            = sib["n_onet"]
            merged.loc[merged["occ_code"] == occ, "exposure_imputed"]  = 1
            merged.loc[merged["occ_code"] == occ, "exposure_residual"] = residual_flag

            diagnostics.append({
                "occ_code":         occ,
                "occ_title":        title,
                "match_status":     "sibling_imputed",
                "imputed_from":     str(sib["siblings"]),
                "dv_beta_imputed":  round(sib["dv_beta"], 4),
                "exposure_residual":residual_flag,
            })
            log.info(
                f"  Imputed {occ} ({title[:40]}) "
                f"← siblings {sib['siblings']} → dv_beta={sib['dv_beta']:.3f}"
            )
        else:
            diagnostics.append({
                "occ_code":         occ,
                "occ_title":        title,
                "match_status":     "unmatched_no_siblings",
                "imputed_from":     "",
                "dv_beta_imputed":  np.nan,
                "exposure_residual":residual_flag,
            })
            log.warning(
                f"  Unmatched (no siblings): {occ} ({title[:40]}) "
                "— will have NaN exposure"
            )

    # Also log the directly matched codes
    matched_occs = set(panel["occ_code"].unique()) - set(unmatched_occs)
    for occ in sorted(matched_occs):
        title = latest_title.get(occ, panel.loc[panel["occ_code"] == occ, "occ_title"].iloc[0])
        dv_b  = expo_agg.loc[expo_agg["soc_code"] == occ, "dv_beta"]
        diagnostics.append({
            "occ_code":         occ,
            "occ_title":        title,
            "match_status":     "direct",
            "imputed_from":     "",
            "dv_beta_imputed":  round(float(dv_b.iloc[0]), 4) if len(dv_b) else np.nan,
            "exposure_residual":int(is_residual(title)),
        })

    diagnostics_df = pd.DataFrame(diagnostics).sort_values(
        ["match_status", "occ_code"]
    ).reset_index(drop=True)

    return merged, diagnostics_df


# ── Step 4: Sanity checks ─────────────────────────────────────────────────────

def print_sanity_checks(merged: pd.DataFrame, diag: pd.DataFrame) -> None:
    sep = "=" * 68
    log.info(f"\n{sep}\nSANITY CHECKS\n{sep}")

    # 1. Match rate
    total_occs   = merged["occ_code"].nunique()
    matched      = diag[diag["match_status"] == "direct"]["occ_code"].nunique()
    imputed      = diag[diag["match_status"] == "sibling_imputed"]["occ_code"].nunique()
    unmatched    = diag[diag["match_status"] == "unmatched_no_siblings"]["occ_code"].nunique()
    null_expo    = merged["dv_beta"].isna().sum()

    log.info(
        f"\nCoverage across {total_occs:,} OEWS occupations:"
        f"\n  Direct match       : {matched:,} ({100*matched/total_occs:.1f}%)"
        f"\n  Sibling imputed    : {imputed:,} ({100*imputed/total_occs:.1f}%)"
        f"\n  Unmatched (NaN)    : {unmatched:,} ({100*unmatched/total_occs:.1f}%)"
        f"\n  Residual categories: "
        f"{diag['exposure_residual'].sum()} (flagged, exclude from main regressions)"
        f"\n  Panel rows with NaN exposure: {null_expo:,} of {len(merged):,}"
    )

    # 2. Exposure distribution — full sample vs. balanced panel only
    log.info("\nExposure (dv_beta) distribution — unique occupations, one value each:")
    occ_expo = (
        merged[merged["year"] == merged["year"].min()]
        [["occ_code", "dv_beta", "exposure_imputed", "balanced"]]
        .dropna(subset=["dv_beta"])
    )
    print(occ_expo["dv_beta"].describe().round(3).to_string())

    # 3. Focal occupations
    focal = {
        "15-1252": "Software Developers",
        "15-2051": "Data Scientists",
        "13-2051": "Financial Analysts",
        "23-2011": "Paralegals and Legal Assistants",
        "35-3023": "Fast Food and Counter Workers",
        "47-2061": "Construction Laborers",
        "31-1120": "Home Health and Personal Care Aides",
    }
    log.info("\nFocal occupation exposure scores:")
    cols = ["occ_code", "occ_title", "dv_beta", "human_beta",
            "exposure_imputed", "n_onet"]
    one_year = merged[merged["year"] == merged["year"].min()]
    for soc, label in focal.items():
        row = one_year[one_year["occ_code"] == soc]
        if row.empty:
            log.warning(f"  ✗  {soc} ({label}) — NOT in merged panel")
        else:
            r = row.iloc[0]
            imputed_note = " [IMPUTED]" if r["exposure_imputed"] else ""
            print(
                f"  ✓  {soc} {label:40s}"
                f"  dv_beta={r['dv_beta']:.3f}"
                f"  human_beta={r['human_beta']:.3f}"
                f"{imputed_note}"
            )

    # 4. High vs low exposure: does the ranking match prior expectations?
    log.info(
        "\nTop 10 highest dv_beta occupations (should be high-knowledge roles):"
    )
    top10 = (
        occ_expo.sort_values("dv_beta", ascending=False)
        .head(10)[["occ_code", "dv_beta"]]
        .merge(
            merged[["occ_code", "occ_title"]].drop_duplicates("occ_code"),
            on="occ_code"
        )
    )
    print(top10[["occ_code", "occ_title", "dv_beta"]].to_string(index=False))

    log.info(
        "\nTop 10 lowest dv_beta occupations (should be manual/physical roles):"
    )
    bot10 = (
        occ_expo.sort_values("dv_beta", ascending=True)
        .head(10)[["occ_code", "dv_beta"]]
        .merge(
            merged[["occ_code", "occ_title"]].drop_duplicates("occ_code"),
            on="occ_code"
        )
    )
    print(bot10[["occ_code", "occ_title", "dv_beta"]].to_string(index=False))

    # 5. dv_beta vs human_beta correlation (measure consistency check)
    corr = merged[["dv_beta", "human_beta"]].dropna().corr().iloc[0, 1]
    log.info(
        f"\nCorrelation dv_beta vs human_beta: {corr:.3f} "
        f"(expect ~0.5–0.7; both measure same concept via different raters)"
    )

    # 6. Reminder about residual categories
    residual_occs = diag[diag["exposure_residual"] == 1]["occ_code"].tolist()
    log.info(
        f"\nResidual 'All Other' categories ({len(residual_occs)} occupations):"
        f"\n  Recommend excluding from main regressions — add 'exposure_residual==0'"
        f"\n  filter in your estimation script."
        f"\n  These codes: {residual_occs[:10]}{'...' if len(residual_occs)>10 else ''}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 68)
    log.info("MERGE: OEWS PANEL × ELOUNDOU EXPOSURE SCORES")
    log.info("=" * 68)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load OEWS panel
    oews_path = PROCESSED_DIR / "oews_national_panel.csv"
    if not oews_path.exists():
        raise FileNotFoundError(
            f"{oews_path} not found. Run 01_process_oews.py first."
        )
    panel = pd.read_csv(oews_path)
    log.info(f"OEWS panel: {len(panel):,} rows, {panel['occ_code'].nunique():,} occupations")

    # Load and aggregate exposure scores
    expo_agg = load_exposure(RAW_ELOUNDOU)

    # Build sibling imputer
    sibling_lookup = build_sibling_imputer(expo_agg)

    # Merge + impute
    merged, diagnostics = merge_and_impute(panel, expo_agg, sibling_lookup)

    # Sanity checks
    print_sanity_checks(merged, diagnostics)

    # Save
    out_merged = PROCESSED_DIR / "oews_exposure_merged.csv"
    out_diag   = PROCESSED_DIR / "merge_diagnostics.csv"

    merged.to_csv(out_merged, index=False)
    diagnostics.to_csv(out_diag, index=False)

    log.info(f"\nOutputs saved:")
    log.info(f"  {out_merged}  ({len(merged):,} rows)")
    log.info(f"  {out_diag}   ({len(diagnostics):,} occupation-level match records)")

    log.info("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Column guide for Person C (estimation scripts)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTCOMES (from OEWS):
  log_wage, log_emp, log_wbill    three regression outcomes

TREATMENT (from Eloundou):
  dv_beta          PRIMARY exposure variable  [0,1]
  human_beta       Robustness check exposure  [0,1]

TIMING:
  post             1 = year > 2022  (post-ChatGPT)
  event_time       τ = year − 2022  (event-study period dummies)

FIXED EFFECTS:
  occ_code         occupation FE
  year             year FE  (or interact with controls for two-way FE)

SAMPLE RESTRICTIONS (apply in estimation):
  balanced == 1           balanced panel (required for Methods 3 & 4)
  suppressed_wage == 0    drop BLS-suppressed wage observations
  exposure_residual == 0  drop residual "All Other" categories
  dv_beta.notna()         drop occupations with no exposure score

DiD INTERACTION TERM:
  dv_beta × post          this is your β in the OLS DiD (Method 1)

Next step → run 03_process_onet_controls.py to add O*NET control variables.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)


if __name__ == "__main__":
    main()
