"""
================================================================================
DATASET 3: BLS Current Employment Statistics (CES) — Quarterly Panel
================================================================================
Project : LLMs and Labor Market Outcomes
Course  : Econometrics & ML — Spring 2025, University of Chicago
Person  : Lynnard (Method 4 — Period-by-Period Event Study)

VERSION 2 CHANGES
-----------------
Fixed employment data loading. CES flat files are split:
  *.a files = Employment only (datatype 01)
  *.b files = Hours and Earnings (datatype 03) — does NOT include employment
Both files are needed per supersector and combined after loading.
Government (90) has only an 'a' file — earnings not collected by BLS.

PURPOSE
-------
Downloads BLS CES flat files, filters to relevant industries and datatypes,
aggregates monthly data to fiscal quarters, and outputs a clean long-format
quarterly panel ready to merge with Felten et al. AIIE industry exposure scores.

WHY CES INSTEAD OF OEWS
------------------------
OEWS is annual (May snapshot). Method 4 requires quarter-by-quarter variation
to identify the timing and dynamics of post-ChatGPT effects. CES provides
monthly employment and earnings at the industry level, which we aggregate
to quarters.

LIMITATION (flag in paper):
CES is industry-coded (supersector level), not occupation-coded (SOC).
The mapping to LLM exposure scores uses Felten et al.'s AIIE (industry-level)
scores rather than Eloundou et al.'s occupation-level scores. This is an
approximation and should be noted as a limitation in the Data section.

DATA SOURCE
-----------
BLS CES flat files: https://download.bls.gov/pub/time.series/ce/
No API key required. Files are tab-delimited.
NOTE: BLS blocks automated downloads (403). Manually download files into
data/raw/ces/ and the script will use the local cache automatically.

FILE STRUCTURE (per supersector)
---------------------------------
  *.a file  →  Employment (datatype 01), thousands
  *.b file  →  Hours and Earnings (datatype 03), average hourly earnings
  Both needed; combined after loading.
  Exception: Government (90) has 'a' only — no earnings collected.

SUPERSECTORS INCLUDED
---------------------
10  Mining and logging
20  Construction
30  Manufacturing
31  Durable Goods
32  Nondurable Goods
40  Trade, transportation, and utilities
41  Wholesale trade
42  Retail trade
43  Transportation and warehousing
50  Information
55  Financial activities
60  Professional and business services
65  Private education and health services
70  Leisure and hospitality
80  Other services
90  Government  ← employment only

NOTE: Supersector 44 (Utilities) has no standalone file — embedded in 40.

TREATMENT TIMING
----------------
ChatGPT launched November 2022 → 2022Q4 is the first treated quarter.
event_time = quarters since 2022Q4 (τ = 0 at 2022Q4).
post = 1 if quarter >= 2022Q4.

Pre-period:  2017Q1 – 2022Q3  (~23 quarters)
Post-period: 2022Q4 – 2026Q1  (~14 quarters, growing)

OUTPUTS
-------
  data/processed/ces_quarterly_panel.csv
  data/processed/ces_download_log.csv
================================================================================
"""

import logging
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
RAW_CES_DIR   = Path("data/raw/ces")
BLS_BASE_URL  = "https://download.bls.gov/pub/time.series/ce"

START_YEAR        = 2017
END_YEAR          = 2026
TREATMENT_YEAR    = 2022
TREATMENT_QUARTER = 4

SUPERSECTORS = {
    "10": "Mining and logging",
    "20": "Construction",
    "30": "Manufacturing",
    "31": "Durable goods",
    "32": "Nondurable goods",
    "40": "Trade, transportation, and utilities",
    "41": "Wholesale trade",
    "42": "Retail trade",
    "43": "Transportation and warehousing",
    "50": "Information",
    "55": "Financial activities",
    "60": "Professional and business services",
    "65": "Private education and health services",
    "70": "Leisure and hospitality",
    "80": "Other services",
    "90": "Government",
}

# Both files needed per supersector: a=employment, b=earnings
# Government has no b file
SUPERSECTOR_FILES = {
    "10": {
        "a": "ce.data.10a.MiningAndLogging.Employment",
        "b": "ce.data.10b.MiningAndLogging.AllEmployeeHoursAndEarnings",
    },
    "20": {
        "a": "ce.data.20a.Construction.Employment",
        "b": "ce.data.20b.Construction.AllEmployeeHoursAndEarnings",
    },
    "30": {
        "a": "ce.data.30a.Manufacturing.Employment",
        "b": "ce.data.30b.Manufacturing.AllEmployeeHoursAndEarnings",
    },
    "31": {
        "a": "ce.data.31a.ManufacturingDurableGoods.Employment",
        "b": "ce.data.31b.ManufacturingDurableGoods.AllEmployeeHoursAndEarnings",
    },
    "32": {
        "a": "ce.data.32a.ManufacturingNondurableGoods.Employment",
        "b": "ce.data.32b.ManufacturingNondurableGoods.AllEmployeeHoursAndEarnings",
    },
    "40": {
        "a": "ce.data.40a.TradeTransportationAndUtilities.Employment",
        "b": "ce.data.40b.TradeTransportationAndUtilities.AllEmployeeHoursAndEarnings",
    },
    "41": {
        "a": "ce.data.41a.WholesaleTrade.Employment",
        "b": "ce.data.41b.WholesaleTrade.AllEmployeeHoursAndEarnings",
    },
    "42": {
        "a": "ce.data.42a.RetailTrade.Employment",
        "b": "ce.data.42b.RetailTrade.AllEmployeeHoursAndEarnings",
    },
    "43": {
        "a": "ce.data.43a.TransportationAndWarehousingAndUtilities.Employment",
        "b": "ce.data.43b.TransportationAndWarehousingAndUtilities.AllEmployeeHoursAndEarnings",
    },
    "50": {
        "a": "ce.data.50a.Information.Employment",
        "b": "ce.data.50b.Information.AllEmployeeHoursAndEarnings",
    },
    "55": {
        "a": "ce.data.55a.FinancialActivities.Employment",
        "b": "ce.data.55b.FinancialActivities.AllEmployeeHoursAndEarnings",
    },
    "60": {
        "a": "ce.data.60a.ProfessionalBusinessServices.Employment",
        "b": "ce.data.60b.ProfessionalBusinessServices.AllEmployeeHoursAndEarnings",
    },
    "65": {
        "a": "ce.data.65a.EducationAndHealthCare.Employment",
        "b": "ce.data.65b.EducationAndHealthCare.AllEmployeeHoursAndEarnings",
    },
    "70": {
        "a": "ce.data.70a.LeisureAndHospitality.Employment",
        "b": "ce.data.70b.LeisureAndHospitality.AllEmployeeHoursAndEarnings",
    },
    "80": {
        "a": "ce.data.80a.OtherServices.Employment",
        "b": "ce.data.80b.OtherServices.AllEmployeeHoursAndEarnings",
    },
    "90": {
        "a": "ce.data.90a.Government.Employment",
        "b": None,   # Government earnings not collected
    },
}

MONTH_TO_QUARTER = {
    1: 1, 2: 1, 3: 1,
    4: 2, 5: 2, 6: 2,
    7: 3, 8: 3, 9: 3,
    10: 4, 11: 4, 12: 4,
}

DATATYPES = {
    "01": "employment",
    "03": "avg_hourly_earnings",
}


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_bls_file(filename: str, cache_dir: Path) -> str:
    """
    Download a BLS flat file if not cached locally.
    Returns local path string, or empty string on failure.
    NOTE: BLS often blocks automated downloads (403 Forbidden).
    If this happens, manually download the file and place in cache_dir.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / filename

    if local_path.exists():
        log.info(f"    Cache hit : {filename}")
        return str(local_path)

    url = f"{BLS_BASE_URL}/{filename}"
    log.info(f"    Downloading: {url}")

    headers = {"User-Agent": "Mozilla/5.0 (research project)"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            content = response.read()
        local_path.write_bytes(content)
        log.info(f"    Saved: {local_path} ({len(content):,} bytes)")
        time.sleep(1)
        return str(local_path)
    except Exception as e:
        log.warning(
            f"    Download failed: {e}\n"
            f"    → Manually download from {url}\n"
            f"    → Place in {cache_dir}/"
        )
        return ""


def parse_data_file(local_path: str, supersector_code: str,
                    expected_datatype: str) -> pd.DataFrame:
    """
    Read one CES flat file and return long-format monthly rows.
    Filters to: seasonally adjusted, expected datatype, years in range.
    """
    if not local_path or not Path(local_path).exists():
        log.warning(f"    File not found: {local_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(local_path, sep="\t", dtype=str)
    except Exception as e:
        log.error(f"    Could not parse {local_path}: {e}")
        return pd.DataFrame()

    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Monthly rows only (drop M13 annual averages)
    df = df[df["period"].str.match(r"^M\d{2}$", na=False)].copy()
    df["month"] = df["period"].str[1:].astype(int)
    df = df[df["month"] <= 12]

    # Year filter
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()

    # Preliminary flag
    if "footnote_codes" in df.columns:
        df["preliminary"] = (df["footnote_codes"].str.upper() == "P").astype(int)
    else:
        df["preliminary"] = 0

    # Extract datatype from series_id (last 2 chars)
    df["datatype_code"] = df["series_id"].str[-2:]

    # Filter to expected datatype only
    df = df[df["datatype_code"] == expected_datatype].copy()

    # Seasonally adjusted only (3rd character of series_id == 'S')
    df = df[df["series_id"].str[2] == "S"].copy()

    if df.empty:
        log.warning(
            f"    No rows after filtering for datatype {expected_datatype} "
            f"in {Path(local_path).name}"
        )
        return pd.DataFrame()

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["supersector_code"] = supersector_code

    return df[["supersector_code", "datatype_code",
               "year", "month", "value", "preliminary"]].reset_index(drop=True)


# ── Step 1: Load all supersectors ────────────────────────────────────────────

def load_all_supersectors() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each supersector, load employment from 'a' file and earnings from
    'b' file separately, then concatenate everything.
    """
    frames = []
    download_log = []

    for ss_code, files in SUPERSECTOR_FILES.items():
        ss_name = SUPERSECTORS[ss_code]
        log.info(f"\nSupersector {ss_code} — {ss_name}")

        # --- Employment (a file, datatype 01) ---
        a_filename = files["a"]
        a_path = fetch_bls_file(a_filename, RAW_CES_DIR)
        df_emp = parse_data_file(a_path, ss_code, expected_datatype="01")

        if df_emp.empty:
            log.warning(f"  No employment data for supersector {ss_code}")
            emp_status = "missing"
        else:
            log.info(f"  Employment rows: {len(df_emp):,}")
            emp_status = "ok"
            frames.append(df_emp)

        # --- Earnings (b file, datatype 03) ---
        b_filename = files["b"]
        if b_filename is None:
            log.info(f"  Earnings: not available (Government)")
            earn_status = "not_collected"
        else:
            b_path = fetch_bls_file(b_filename, RAW_CES_DIR)
            df_earn = parse_data_file(b_path, ss_code, expected_datatype="03")
            if df_earn.empty:
                log.warning(f"  No earnings data for supersector {ss_code}")
                earn_status = "missing"
            else:
                log.info(f"  Earnings rows  : {len(df_earn):,}")
                earn_status = "ok"
                frames.append(df_earn)

        download_log.append({
            "supersector_code": ss_code,
            "supersector_name": ss_name,
            "filename": f"{a_filename} + {b_filename or 'N/A'}",
            "emp_status": emp_status,
            "earn_status": earn_status,
        })

    if not frames:
        raise RuntimeError(
            "No CES data loaded.\n"
            "BLS likely blocked automated downloads.\n"
            f"Manually download files to {RAW_CES_DIR}/ and re-run."
        )

    combined = pd.concat(frames, ignore_index=True)
    log.info(f"\nTotal monthly rows loaded: {len(combined):,}")
    return combined, pd.DataFrame(download_log)


# ── Step 2: Aggregate monthly → quarterly ────────────────────────────────────

def aggregate_to_quarters(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot datatypes wide then average within each quarter.
    """
    wide = monthly.pivot_table(
        index=["supersector_code", "year", "month", "preliminary"],
        columns="datatype_code",
        values="value",
        aggfunc="first",
    ).reset_index()

    wide.columns.name = None
    wide = wide.rename(columns={
        "01": "employment",
        "03": "avg_hourly_earnings",
    })

    for col in ["employment", "avg_hourly_earnings"]:
        if col not in wide.columns:
            wide[col] = np.nan

    wide["quarter"] = wide["month"].map(MONTH_TO_QUARTER)

    quarterly = wide.groupby(
        ["supersector_code", "year", "quarter"]
    ).agg(
        employment          = ("employment",          "mean"),
        avg_hourly_earnings = ("avg_hourly_earnings", "mean"),
        n_months            = ("month",               "count"),
        preliminary         = ("preliminary",          "max"),
    ).reset_index()

    return quarterly


# ── Step 3: Derived columns + treatment timing ────────────────────────────────

def add_derived_columns(quarterly: pd.DataFrame) -> pd.DataFrame:
    df = quarterly.copy()

    df["log_emp"]   = np.log(df["employment"])
    df["log_wage"]  = np.log(df["avg_hourly_earnings"])
    df["log_wbill"] = np.log(df["avg_hourly_earnings"] * df["employment"])

    # event_time: τ = 0 at 2022Q4
    df["quarter_int"] = (
        (df["year"] - TREATMENT_YEAR) * 4
        + (df["quarter"] - TREATMENT_QUARTER)
    )
    df["event_time"]    = df["quarter_int"]
    df["post"]          = (df["quarter_int"] >= 0).astype(int)
    df["quarter_label"] = "Q" + df["quarter"].astype(str)
    df["supersector_name"] = df["supersector_code"].astype(str).map(SUPERSECTORS)

    df = df.sort_values(["supersector_code", "year", "quarter"]).reset_index(drop=True)

    return df[[
        "supersector_code", "supersector_name",
        "year", "quarter_label", "quarter",
        "event_time", "post",
        "employment", "avg_hourly_earnings",
        "log_emp", "log_wage", "log_wbill",
        "n_months", "preliminary",
    ]]


# ── Step 4: Sanity checks ─────────────────────────────────────────────────────

def print_sanity_checks(panel: pd.DataFrame) -> None:
    sep = "=" * 68
    log.info(f"\n{sep}\nSANITY CHECKS\n{sep}")

    log.info(f"\nPanel shape: {panel.shape}")
    log.info(f"Years: {panel['year'].min()}–{panel['year'].max()}")
    log.info(f"Supersectors: {panel['supersector_code'].nunique()}")
    log.info(f"Preliminary quarters: {panel['preliminary'].sum()}")

    log.info("\nNull counts by supersector:")
    null_summary = panel.groupby(
        ["supersector_code", "supersector_name"]
    ).agg(
        emp_nulls  =("employment",          lambda x: x.isna().sum()),
        earn_nulls =("avg_hourly_earnings", lambda x: x.isna().sum()),
        n_quarters =("year",                "count"),
    ).reset_index()
    print(null_summary.to_string(index=False))

    log.info("\nEvent-time distribution (first and last 5 periods):")
    ev = panel.groupby("event_time").agg(
        year    =("year",          "first"),
        quarter =("quarter_label", "first"),
        post    =("post",          "first"),
        n_obs   =("supersector_code", "count"),
    )
    print(pd.concat([ev.head(5), ev.tail(5)]).to_string())

    log.info("\nSample — Information sector (supersector 50):")
    info = panel[panel["supersector_code"] == 50].head(6)[
        ["year", "quarter_label", "event_time", "post",
         "employment", "avg_hourly_earnings", "log_emp", "log_wage"]
    ]
    print(info.to_string(index=False))

    log.info("\nSample — Government (supersector 90, employment only):")
    gov = panel[panel["supersector_code"] == 90].head(4)[
        ["year", "quarter_label", "employment", "avg_hourly_earnings"]
    ]
    print(gov.to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 68)
    log.info("CES QUARTERLY PANEL — PROCESSING (v2)")
    log.info("=" * 68)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CES_DIR.mkdir(parents=True, exist_ok=True)

    log.info("\nStep 1: Loading CES flat files...")
    monthly, download_log = load_all_supersectors()

    log.info("\nStep 2: Aggregating monthly → quarterly...")
    quarterly = aggregate_to_quarters(monthly)

    log.info("\nStep 3: Adding derived columns and treatment timing...")
    panel = add_derived_columns(quarterly)

    print_sanity_checks(panel)

    out_panel = PROCESSED_DIR / "ces_quarterly_panel.csv"
    out_log   = PROCESSED_DIR / "ces_download_log.csv"
    panel.to_csv(out_panel, index=False)
    download_log.to_csv(out_log, index=False)

    log.info(f"\nOutputs saved:")
    log.info(f"  {out_panel}  ({len(panel):,} rows)")
    log.info(f"  {out_log}")

    log.info("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Next step → run 04_merge_felten.py to add AIIE exposure scores
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTCOMES:
  log_emp     ln(employment thousands)     ← OUTCOME 1
  log_wage    ln(avg hourly earnings)      ← OUTCOME 2
  log_wbill   ln(AHE × employment)         ← OUTCOME 3

TIMING:
  event_time  τ = quarters since 2022Q4   (τ=0 is 2022Q4)
  post        1 = quarter >= 2022Q4

FIXED EFFECTS:
  supersector_code  industry FE
  year + quarter    time FEs

SAMPLE NOTES:
  Government (90)    employment only; exclude from wage regressions
  preliminary == 1   flag as robustness check
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)


if __name__ == "__main__":
    main()
