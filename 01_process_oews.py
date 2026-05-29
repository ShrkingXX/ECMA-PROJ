"""
================================================================================
DATASET 1: BLS Occupational Employment and Wage Statistics (OEWS)
================================================================================
PURPOSE
-------
Reads raw OEWS national Excel files (2019–2025), filters to national ×
detailed × cross-industry rows, handles BLS suppression flags, applies a
SOC vintage crosswalk for 2019, constructs three log outcomes, and outputs
a clean long-format panel ready to merge with Eloundou et al. exposure scores.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILTERS APPLIED (confirmed against actual 2019 file)
------------------------------------------------------
  area     = '99'             → U.S. national (not state / metro)
  o_group  = 'detailed'       → 6-digit SOC only; drops major/minor/broad
                                aggregates that would double-count workers
  i_group  = 'cross-industry' → all-industry aggregate; drops industry-
                                specific cuts (Mfg, Finance, etc.)
  → 789 occupation rows per year

SUPPRESSION (confirmed against actual 2019 file)
-------------------------------------------------
  a_mean  : flagged '*' for 4 of 789 rows (Actors, Dancers, Musicians,
            Misc. Entertainers) — kept in panel with a_mean = NaN and
            suppressed_wage = 1 so estimation scripts can handle explicitly.
  tot_emp : never suppressed at national/cross-industry level (0 of 789).
  h_mean  : flagged '*' for some rows — stored as NaN (not an outcome).

OUTPUTS
-------
  data/processed/oews_national_panel.csv   ← main long-format panel
  data/processed/oews_suppression_log.csv  ← rows where a_mean was suppressed
  data/processed/oews_crosswalk_log.csv    ← rows where SOC code was remapped

PANEL STRUCTURE (one row = one detailed occupation × one survey year)
----------------------------------------------------------------------
  occ_code         SOC 2018 code — PRIMARY MERGE KEY for Eloundou scores
  occ_title        BLS occupation label (original; 2019 titles may differ)
  year             OEWS survey reference year (May snapshot)
  tot_emp          Total employment (numeric)
  a_mean           Annual mean wage in USD (NaN if suppressed)
  h_mean           Hourly mean wage (NaN if suppressed or annual-only occ)
  wage_bill        a_mean × tot_emp
  log_wage         ln(a_mean)    ← OUTCOME 1
  log_emp          ln(tot_emp)   ← OUTCOME 2
  log_wbill        ln(wage_bill) ← OUTCOME 3  (composition-bias check)
  annual_only      1 = BLS reports no hourly wage for this occupation
  suppressed_wage  1 = a_mean withheld by BLS
  soc_remapped     1 = SOC code was reassigned via 2019→2018 crosswalk
  post             1 if year > 2022  (post-ChatGPT release Nov 2022)
  event_time       year − 2022  (τ for event-study dummies; −3 … +3)
  balanced         1 = occupation present in ALL years (needed for Methods 3/4)
================================================================================
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

RAW_DIR       = Path("data/raw/oews")
PROCESSED_DIR = Path("data/processed")

YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

# Last pre-treatment survey year.
# ChatGPT launched Nov 2022; OEWS reference period is May, so May 2022
# is the final clean pre-period. May 2023 is the first post-period snapshot.
TREATMENT_YEAR = 2022

# All BLS suppression flags (only '*' appears at national/cross-industry level
# in practice, but we guard against the full documented set)
SUPPRESSION_FLAGS = {"*", "**", "#", "~", "-", "–", "N/A", "NA"}

# ── SOC 2010 → SOC 2018 crosswalk (applied to 2019 data only) ────────────────
#
# Format: { old_2019_code : (new_2018_code, reason_note) }
#
# Source: BLS SOC 2018 revision crosswalk
#   https://www.bls.gov/soc/2018/soc_2010_to_2018_crosswalk.xlsx
#
# Only occupations that (a) changed codes AND (b) are in our analysis are listed.
# The full crosswalk has hundreds of entries — we apply a targeted subset.
#
SOC_CROSSWALK_2019 = {
    # 2019 code            new code    note
    "15-1256": ("15-1252", "Software Developers: 2019 bundled QA analysts; "
                            "2020+ split. 2019 wage/emp slightly broader."),
    "15-2098": ("15-2051", "Data Scientists: 2019 was residual 'all other' category; "
                            "2020+ became standalone. 2019 wage/emp slightly broader."),
    "13-2098": ("13-2051", "Financial Analysts: 2019 bundled risk specialists & "
                            "others; 2020+ standalone. 2019 wage/emp slightly broader."),
}

# Focal occupations for sanity-check printout
FOCAL_OCCS = {
    "15-1252": "Software Developers",
    "15-2051": "Data Scientists",
    "13-2051": "Financial Analysts",
    "23-2011": "Paralegals and Legal Assistants",
    "35-3023": "Fast Food and Counter Workers",
    "47-2061": "Construction Laborers",
    "31-1120": "Home Health and Personal Care Aides",
}


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_numeric_safe(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Convert a string Series that may contain BLS suppression flags to float.

    Returns
    -------
    numeric    : float Series  (suppressed / unparseable → NaN)
    suppressed : bool Series   (True where original value was a flag)
    """
    s = series.astype(str).str.strip()
    suppressed = s.isin(SUPPRESSION_FLAGS) | s.isin({"nan", ""})
    s_clean = s.copy()
    s_clean[suppressed] = np.nan
    s_clean = s_clean.str.replace(",", "", regex=False)   # remove thousands commas
    numeric = pd.to_numeric(s_clean, errors="coerce")
    return numeric, suppressed


def apply_soc_crosswalk(df: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For the 2019 file, recode legacy SOC 2010 codes to SOC 2018 equivalents.

    Returns
    -------
    df            : DataFrame with occ_code updated
    crosswalk_log : DataFrame recording every row that was remapped (for audit)
    """
    df["soc_remapped"] = 0
    remapped_rows = []

    if year != 2019:
        return df, pd.DataFrame()

    for old_code, (new_code, note) in SOC_CROSSWALK_2019.items():
        mask = df["occ_code"] == old_code
        if mask.any():
            original_title = df.loc[mask, "occ_title"].iloc[0]
            df.loc[mask, "occ_code"]    = new_code
            df.loc[mask, "soc_remapped"] = 1
            remapped_rows.append({
                "year":           year,
                "old_soc_code":   old_code,
                "new_soc_code":   new_code,
                "original_title": original_title,
                "note":           note,
            })
            log.info(
                f"  SOC crosswalk: {old_code} → {new_code}  "
                f"({original_title})"
            )
        else:
            log.warning(
                f"  SOC crosswalk: {old_code} not found in 2019 data. "
                "This is unexpected — check the file."
            )

    crosswalk_log = pd.DataFrame(remapped_rows)
    return df, crosswalk_log


def read_one_year(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load, filter, and clean OEWS data for a single survey year.

    Applies three filters (area / o_group / i_group), converts numerics,
    flags suppression, applies SOC crosswalk for 2019, adds derived columns.

    Returns (data_df, crosswalk_log_df).
    Both are empty DataFrames if the file is missing.
    """
    filepath = RAW_DIR / f"all_data_M_{year}.xlsx"

    if not filepath.exists():
        log.error(
            f"File not found: {filepath}\n"
            f"  Download: https://www.bls.gov/oes/special.requests/"
            f"oesm{str(year)[2:]}nat.zip\n"
            f"  Extract all_data_M_{year}.xlsx  →  {RAW_DIR}/"
        )
        return pd.DataFrame(), pd.DataFrame()

    log.info(f"  Reading {filepath.name} ...")

    # dtype=str preserves suppression flags ('*', '#', etc.)
    raw = pd.read_excel(filepath, dtype=str, sheet_name=0)
    raw.columns = raw.columns.str.strip().str.lower()

    # ── Column validation ─────────────────────────────────────────────────────
    required = {"area", "o_group", "i_group", "occ_code", "occ_title",
                "tot_emp", "a_mean", "h_mean"}
    missing_cols = required - set(raw.columns)
    if missing_cols:
        log.error(
            f"  Year {year}: missing columns {missing_cols}. "
            f"Columns found: {list(raw.columns)}"
        )
        return pd.DataFrame(), pd.DataFrame()

    total_rows = len(raw)

    # ── Three filters ─────────────────────────────────────────────────────────
    mask = (
        (raw["area"].str.strip()    == "99")             &  # U.S. national
        (raw["o_group"].str.strip() == "detailed")       &  # 6-digit SOC only
        (raw["i_group"].str.strip() == "cross-industry")    # all-industry agg
    )
    df = raw[mask].copy()

    log.info(
        f"  {total_rows:>7,} raw rows  →  {len(df):,} "
        f"national × detailed × cross-industry rows"
    )

    if df.empty:
        log.error(
            f"  Year {year}: 0 rows after filtering. "
            "Confirm the xlsx is the all_data (national) file, not state/metro."
        )
        return pd.DataFrame(), pd.DataFrame()

    # ── Standardize string columns ────────────────────────────────────────────
    df["occ_code"]  = df["occ_code"].str.strip()
    df["occ_title"] = df["occ_title"].str.strip()

    # ── SOC vintage crosswalk (2019 only) ─────────────────────────────────────
    df, crosswalk_log = apply_soc_crosswalk(df, year)

    # ── Convert numerics; flag suppression ────────────────────────────────────
    df["a_mean"],  df["suppressed_wage"] = to_numeric_safe(df["a_mean"])
    df["tot_emp"], _                     = to_numeric_safe(df["tot_emp"])
    df["h_mean"],  _                     = to_numeric_safe(df["h_mean"])

    n_supp = int(df["suppressed_wage"].sum())
    if n_supp > 0:
        suppressed_titles = df.loc[df["suppressed_wage"], "occ_title"].tolist()
        log.info(
            f"  Suppressed a_mean: {n_supp} rows (kept with NaN): "
            f"{suppressed_titles}"
        )

    # ── annual_only flag ──────────────────────────────────────────────────────
    # BLS 'annual' column = TRUE when the occupation reports no hourly wage
    if "annual" in df.columns:
        df["annual_only"] = (
            df["annual"].str.strip().str.upper() == "TRUE"
        ).astype(int)
    else:
        df["annual_only"] = 0

    # ── Derived outcomes ──────────────────────────────────────────────────────
    df["wage_bill"] = df["a_mean"] * df["tot_emp"]
    df["log_wage"]  = np.log(df["a_mean"])    # NaN propagates for suppressed rows
    df["log_emp"]   = np.log(df["tot_emp"])
    df["log_wbill"] = np.log(df["wage_bill"])

    # ── Treatment timing ──────────────────────────────────────────────────────
    df["year"]       = year
    df["post"]       = int(year > TREATMENT_YEAR)
    df["event_time"] = year - TREATMENT_YEAR   # τ: −3,−2,−1,0,+1,+2

    # ── Select output columns ─────────────────────────────────────────────────
    keep = [
        "year", "event_time", "post",
        "occ_code", "occ_title",
        "tot_emp", "a_mean", "h_mean",
        "wage_bill", "log_wage", "log_emp", "log_wbill",
        "annual_only", "suppressed_wage", "soc_remapped",
    ]
    return df[keep].reset_index(drop=True), crosswalk_log


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stack all years, add the balanced-panel flag, split out audit logs.

    Returns (panel, suppression_log, crosswalk_log).
    """
    frames, crosswalk_frames = [], []

    for year in YEARS:
        df, cw_log = read_one_year(year)
        if not df.empty:
            frames.append(df)
        if not cw_log.empty:
            crosswalk_frames.append(cw_log)

    if not frames:
        log.error(
            f"No data loaded. Place OEWS xlsx files in {RAW_DIR}/ and retry."
        )
        sys.exit(1)

    panel  = pd.concat(frames, ignore_index=True)
    n_years = panel["year"].nunique()
    log.info(
        f"\nStacked: {len(panel):,} rows across "
        f"{n_years} years {sorted(panel['year'].unique())}"
    )

    # ── Balanced-panel flag ───────────────────────────────────────────────────
    # Event study (Method 4) and Double Lasso DiD (Method 3) need every
    # occupation present in every year.
    year_counts   = panel.groupby("occ_code")["year"].count()
    balanced_occs = year_counts[year_counts == n_years].index
    panel["balanced"] = panel["occ_code"].isin(balanced_occs).astype(int)

    n_bal   = panel.loc[panel["balanced"] == 1, "occ_code"].nunique()
    n_total = panel["occ_code"].nunique()
    log.info(
        f"Balanced (all {n_years} years): {n_bal:,} of {n_total:,} occupations"
    )

    # ── Audit logs ────────────────────────────────────────────────────────────
    suppression_log = (
        panel[panel["suppressed_wage"] == 1]
        .copy()
        .sort_values(["occ_code", "year"])
    )
    crosswalk_log = (
        pd.concat(crosswalk_frames, ignore_index=True)
        if crosswalk_frames else pd.DataFrame()
    )

    panel = panel.sort_values(["occ_code", "year"]).reset_index(drop=True)
    return panel, suppression_log, crosswalk_log


# ── Sanity checks ─────────────────────────────────────────────────────────────

def print_sanity_checks(panel: pd.DataFrame) -> None:
    """Printed diagnostics to verify the panel before saving."""

    sep = "=" * 68
    log.info(f"\n{sep}\nSANITY CHECKS\n{sep}")

    # 1. Shape
    log.info(
        f"\nPanel shape      : {panel.shape}"
        f"\nYears            : {sorted(panel['year'].unique())}"
        f"\nTotal occupations: {panel['occ_code'].nunique():,}"
        f"\nBalanced occs    : "
        f"{panel.loc[panel['balanced']==1,'occ_code'].nunique():,}"
        f"\nSOC-remapped rows: {int(panel['soc_remapped'].sum())} (2019 only)"
    )

    # 2. Event-time distribution
    log.info("\nEvent-time (τ) distribution:")
    ev = panel.groupby("event_time").agg(
        survey_year   = ("year",       "first"),
        n_occupations = ("occ_code",   "count"),
        post          = ("post",        "first"),
    )
    print(ev.to_string())

    # 3. National aggregates by year
    log.info("\nNational aggregates by year (eyeball for plausibility):")
    agg = panel.groupby("year").agg(
        mean_wage  = ("a_mean",          "mean"),
        total_emp  = ("tot_emp",         "sum"),
        n_occ      = ("occ_code",        "count"),
        pct_supp   = ("suppressed_wage", "mean"),
    )
    agg["mean_wage"] = agg["mean_wage"].map("${:>10,.0f}".format)
    agg["total_emp"] = agg["total_emp"].map("{:>14,.0f}".format)
    agg["pct_supp"]  = (agg["pct_supp"] * 100).map("{:.1f}%".format)
    print(agg.to_string())

    # 4. Focal occupations — confirm all seven are present across all years
    log.info("\nFocal occupation check (all 7 from proposal):")
    for soc, title in FOCAL_OCCS.items():
        rows = panel[panel["occ_code"] == soc][
            ["year", "a_mean", "tot_emp", "log_wage", "event_time", "soc_remapped"]
        ]
        if rows.empty:
            log.warning(f"  ✗  {soc} ({title}) — NOT FOUND.")
        else:
            flag = "  [includes 2019 crosswalk row]" if rows["soc_remapped"].any() else ""
            print(f"\n  ✓  {soc} — {title}{flag}")
            print(rows.drop(columns="soc_remapped").to_string(index=False))

    # 5. Log-outcome distributions
    log.info("\nLog-outcome descriptive stats (non-suppressed rows):")
    clean = panel[panel["suppressed_wage"] == 0]
    print(
        clean[["log_wage", "log_emp", "log_wbill"]]
        .describe()
        .round(3)
        .to_string()
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 68)
    log.info("OEWS NATIONAL PANEL — PROCESSING")
    log.info("=" * 68)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    panel, suppression_log, crosswalk_log = build_panel()
    print_sanity_checks(panel)

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_panel    = PROCESSED_DIR / "oews_national_panel.csv"
    out_supp     = PROCESSED_DIR / "oews_suppression_log.csv"
    out_crosswalk= PROCESSED_DIR / "oews_crosswalk_log.csv"

    panel.to_csv(out_panel, index=False)
    suppression_log.to_csv(out_supp, index=False)
    if not crosswalk_log.empty:
        crosswalk_log.to_csv(out_crosswalk, index=False)

    log.info("\nOutputs saved:")
    log.info(f"  {out_panel}      ({len(panel):,} rows)")
    log.info(f"  {out_supp}   ({len(suppression_log):,} rows)")
    if not crosswalk_log.empty:
        log.info(f"  {out_crosswalk}   ({len(crosswalk_log):,} remapped codes)")



if __name__ == "__main__":
    main()
