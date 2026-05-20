"""
================================================================================
METHOD 2B: Metro-level Shift-Share / Bartik Labor Supply IV
================================================================================
Project : LLMs and Labor Market Outcomes
Person  : Anna — Method 2

PURPOSE
-------
This script implements a more canonical shift-share/Bartik version of Method 2.
Instead of using only occupation-year variation,

    Z_ot = LLMExposure_o × Post_t,

we use pre-ChatGPT metro occupational composition to construct a local predicted
LLM shock:

    Bartik_m = Σ_o s_mo,pre × LLMExposure_o
    Z_mt     = Bartik_m × Post_t

where s_mo,pre is occupation o's employment share in metro m before ChatGPT.

The intuition is:
    - ChatGPT release is the common national/global shift.
    - Metros differ in pre-existing exposure because their occupational mixes
      differ before ChatGPT.
    - A metro with many workers in high-LLM-exposure occupations receives a
      larger predicted LLM demand shock after 2022.

Main equations
--------------
First stage:
    log L_mt = α + π Z_mt + γ_m + δ_t + u_mt

Reduced form:
    log w_mt = α + ρ Z_mt + γ_m + δ_t + v_mt

Structural IV / 2SLS:
    log w_mt = α + ε log L_mt + γ_m + δ_t + e_mt,
    where log L_mt is instrumented by Z_mt.

Interpretation:
    ε is an inverse labor supply elasticity: the wage change associated with a
    1% instrument-induced change in local employment.

INPUTS
------
1. Metro OEWS raw Excel files, one per year, in one of these locations:
       data/raw/oews_metro/all_data_M_{YEAR}.xlsx
       data/raw/oews_metro/all_data_{YEAR}.xlsx
       data/raw/oews/all_data_M_{YEAR}_metro.xlsx
       /mnt/data/all_data_M_{YEAR}.xlsx

   These should be metro/nonmetro OEWS files, not the national files. The script
   will error clearly if it only sees national rows.

2. Existing occupation exposure file from your repo:
       data/processed/oews_exposure_merged.csv
   or sandbox fallback:
       /mnt/data/oews_exposure_merged.csv

OUTPUTS
-------
    results/table_m2b_metro_shift_share_metro_panel.csv
    results/table_m2b_metro_shift_share_first_stage.csv
    results/table_m2b_metro_shift_share_reduced_form.csv
    results/table_m2b_metro_shift_share_structural_iv.csv
    results/table_m2b_metro_shift_share_event_study.csv
    results/figure_m2b_metro_shift_share_iv.png

Run:
    python 05_method2_metro_shift_share_iv.py

Notes:
    - Main exposure measure: dv_beta
    - Robustness exposure measure: human_beta
    - Pre-period shares use average employment over 2019--2022 by default.
================================================================================
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS


# ── Configuration ─────────────────────────────────────────────────────────────

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
BASE_YEAR = 2022
BASE_TAU = 0
PRE_SHARE_YEARS = [2019, 2020, 2021, 2022]

MAIN_EXPOSURE = "dv_beta"
ROBUSTNESS_EXPOSURE = "human_beta"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPOSURE_CANDIDATES = [
    Path("data/processed/oews_exposure_merged.csv"),
    Path("/mnt/data/oews_exposure_merged.csv"),
]

RAW_FILE_CANDIDATES = [
    lambda y: Path(f"data/raw/oews_metro/all_data_M_{y}.xlsx"),
    lambda y: Path(f"data/raw/oews_metro/all_data_{y}.xlsx"),
    lambda y: Path(f"data/raw/oews/all_data_M_{y}.xlsx"),
    lambda y: Path(f"data/raw/oews/all_data_M_{y}_metro.xlsx"),
    lambda y: Path(f"/mnt/data/all_data_M_{y}.xlsx"),
]

SUPPRESSION_FLAGS = {"*", "**", "#", "~", "-", "–", "N/A", "NA", "nan", ""}

# Targeted 2019 SOC 2010 → SOC 2018 crosswalk used in the current repo.
SOC_CROSSWALK_2019 = {
    "15-1256": ("15-1252", "Software Developers: 2019 bundled with QA analysts"),
    "15-2098": ("15-2051", "Data Scientists: 2019 residual category"),
    "13-2098": ("13-2051", "Financial Analysts: 2019 bundled category"),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def find_existing_path(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("None of these files exist:\n" + "\n".join(str(p) for p in candidates))


def find_raw_file(year: int) -> Path | None:
    for make_path in RAW_FILE_CANDIDATES:
        p = make_path(year)
        if p.exists():
            return p
    return None


def to_numeric_safe(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    s = series.astype(str).str.strip()
    suppressed = s.isin(SUPPRESSION_FLAGS)
    s_clean = s.copy()
    s_clean[suppressed] = np.nan
    s_clean = s_clean.str.replace(",", "", regex=False)
    numeric = pd.to_numeric(s_clean, errors="coerce")
    return numeric, suppressed | numeric.isna()


def apply_2019_soc_crosswalk(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df["soc_remapped"] = 0
    if year != 2019:
        return df
    for old_code, (new_code, _) in SOC_CROSSWALK_2019.items():
        mask = df["occ_code"] == old_code
        if mask.any():
            df.loc[mask, "occ_code"] = new_code
            df.loc[mask, "soc_remapped"] = 1
            log.info(f"  SOC crosswalk applied: {old_code} → {new_code}")
    return df


def load_exposure_lookup(exposure_col: str) -> pd.DataFrame:
    """Use the already merged national file as a clean SOC → exposure lookup."""
    path = find_existing_path(EXPOSURE_CANDIDATES)
    exposure = pd.read_csv(path, dtype={"occ_code": str})

    needed = ["occ_code", exposure_col]
    optional = [c for c in ["human_beta", "dv_beta", "exposure_residual", "exposure_imputed"] if c in exposure.columns]
    cols = sorted(set(needed + optional))

    missing = [c for c in needed if c not in exposure.columns]
    if missing:
        raise ValueError(f"Exposure file {path} is missing columns: {missing}")

    lookup = (
        exposure[cols]
        .drop_duplicates("occ_code")
        .copy()
    )

    if "exposure_residual" not in lookup.columns:
        lookup["exposure_residual"] = 0
    if "exposure_imputed" not in lookup.columns:
        lookup["exposure_imputed"] = 0

    lookup[exposure_col] = pd.to_numeric(lookup[exposure_col], errors="coerce")
    log.info(
        f"Exposure lookup: {len(lookup):,} occupations from {path} "
        f"using {exposure_col}"
    )
    return lookup[["occ_code", exposure_col, "exposure_residual", "exposure_imputed"]]


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Build metro × occupation × year panel
# ═══════════════════════════════════════════════════════════════════════════════

def read_one_metro_year(year: int) -> pd.DataFrame:
    path = find_raw_file(year)
    if path is None:
        raise FileNotFoundError(
            f"Could not find a metro OEWS Excel file for {year}. Expected one of:\n"
            + "\n".join(str(make_path(year)) for make_path in RAW_FILE_CANDIDATES)
        )

    log.info(f"Reading {year}: {path}")
    raw = pd.read_excel(path, dtype=str, sheet_name=0)
    raw.columns = raw.columns.str.strip().str.lower()

    required = {"area", "area_title", "o_group", "i_group", "occ_code", "occ_title", "tot_emp", "a_mean"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    # Keep detailed SOC × cross-industry rows. This mirrors the national pipeline,
    # except area is metro/nonmetro rather than area == 99.
    df = raw[
        (raw["o_group"].str.strip().str.lower() == "detailed")
        & (raw["i_group"].str.strip().str.lower() == "cross-industry")
        & (raw["area"].str.strip() != "99")
    ].copy()

    # If area_type exists, keep metropolitan rows when the coding is informative.
    # BLS files differ across years/vintages, so this is intentionally permissive.
    if "area_type" in df.columns:
        area_type = df["area_type"].astype(str).str.strip().str.lower()
        metro_like = area_type.isin({"4", "m", "metro", "metropolitan"})
        if metro_like.sum() > 0:
            df = df[metro_like].copy()

    if df.empty:
        raise ValueError(
            f"0 metro rows after filtering {path}. This is probably a national file, "
            "not the metro/nonmetro OEWS file."
        )

    df["area"] = df["area"].str.strip()
    df["area_title"] = df["area_title"].str.strip()
    df["occ_code"] = df["occ_code"].str.strip()
    df["occ_title"] = df["occ_title"].str.strip()
    df = apply_2019_soc_crosswalk(df, year)

    df["tot_emp"], emp_bad = to_numeric_safe(df["tot_emp"])
    df["a_mean"], wage_bad = to_numeric_safe(df["a_mean"])
    df["suppressed_emp"] = emp_bad.astype(int)
    df["suppressed_wage"] = wage_bad.astype(int)

    # Employment is needed for shares and first-stage outcomes; wage is needed for wage outcomes.
    df["wage_bill"] = df["a_mean"] * df["tot_emp"]
    df["year"] = year
    df["post"] = (df["year"] > BASE_YEAR).astype(int)
    df["event_time"] = df["year"] - BASE_YEAR

    keep = [
        "area", "area_title", "year", "post", "event_time",
        "occ_code", "occ_title", "tot_emp", "a_mean", "wage_bill",
        "suppressed_emp", "suppressed_wage", "soc_remapped",
    ]
    out = df[keep].reset_index(drop=True)
    log.info(
        f"  kept {len(out):,} metro×occupation rows | "
        f"metros={out['area'].nunique():,}, occupations={out['occ_code'].nunique():,}"
    )
    return out


def build_or_load_metro_occ_panel(exposure_col: str) -> pd.DataFrame:
    """Build metro×occupation×year data and merge exposure scores."""
    cached = Path(f"data/processed/oews_metro_exposure_merged_{exposure_col}.csv")
    if cached.exists():
        log.info(f"Loading cached metro exposure panel: {cached}")
        return pd.read_csv(cached, dtype={"area": str, "occ_code": str})

    frames = [read_one_metro_year(y) for y in YEARS]
    panel = pd.concat(frames, ignore_index=True)

    exposure = load_exposure_lookup(exposure_col)
    merged = panel.merge(exposure, on="occ_code", how="left", validate="m:1")

    # Main sample filters for occupation cells.
    # Keep employment even if wage is suppressed for share construction, but drop missing exposure.
    merged["valid_exposure"] = merged[exposure_col].notna() & (merged["exposure_residual"] == 0)
    merged["valid_emp_cell"] = merged["valid_exposure"] & merged["tot_emp"].notna() & (merged["tot_emp"] > 0)
    merged["valid_wage_cell"] = merged["valid_emp_cell"] & merged["a_mean"].notna() & (merged["a_mean"] > 0)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cached, index=False)
    log.info(f"Saved cached metro exposure panel → {cached}")
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — Construct Bartik exposure and metro-year outcomes
# ═══════════════════════════════════════════════════════════════════════════════

def construct_metro_year_panel(metro_occ: pd.DataFrame, exposure_col: str) -> pd.DataFrame:
    """
    Construct:
        Bartik_m = Σ_o s_mo,pre × Exposure_o
    and aggregate OEWS outcomes to metro×year.
    """
    df = metro_occ.copy()
    df[exposure_col] = pd.to_numeric(df[exposure_col], errors="coerce")

    # Use pre-period employment to calculate fixed shares.
    pre = df[
        df["year"].isin(PRE_SHARE_YEARS)
        & df["valid_emp_cell"]
    ].copy()

    if pre.empty:
        raise ValueError("No valid pre-period metro occupation cells available for share construction.")

    # Average/sum pre-period employment by metro×occupation. Summing over pre years
    # is equivalent to employment-weighted average shares and is less noisy than one year.
    base_occ = (
        pre.groupby(["area", "area_title", "occ_code"], as_index=False)
        .agg(base_emp=("tot_emp", "sum"), exposure=(exposure_col, "first"))
    )
    base_tot = (
        base_occ.groupby("area", as_index=False)["base_emp"]
        .sum()
        .rename(columns={"base_emp": "base_total_emp"})
    )
    base_occ = base_occ.merge(base_tot, on="area", how="left")
    base_occ["base_share"] = base_occ["base_emp"] / base_occ["base_total_emp"]
    base_occ["share_x_exposure"] = base_occ["base_share"] * base_occ["exposure"]

    bartik = (
        base_occ.groupby(["area", "area_title"], as_index=False)
        .agg(
            bartik_exposure=("share_x_exposure", "sum"),
            base_total_emp=("base_total_emp", "first"),
            n_base_occs=("occ_code", "nunique"),
        )
    )

    # Metro-year outcomes: aggregate all valid exposure occupations.
    # For wages, use only non-suppressed wage cells. This creates a wage-bill-weighted
    # average wage over the included occupation cells.
    valid = df[df["valid_emp_cell"]].copy()
    valid["emp_for_wage"] = np.where(valid["valid_wage_cell"], valid["tot_emp"], np.nan)
    valid["wbill_for_wage"] = np.where(valid["valid_wage_cell"], valid["wage_bill"], np.nan)

    metro_year = (
        valid.groupby(["area", "area_title", "year", "post", "event_time"], as_index=False)
        .agg(
            total_emp=("tot_emp", "sum"),
            wage_emp=("emp_for_wage", "sum"),
            wage_bill=("wbill_for_wage", "sum"),
            n_occ_cells=("occ_code", "nunique"),
            n_wage_cells=("valid_wage_cell", "sum"),
        )
    )

    metro_year = metro_year.merge(bartik, on=["area", "area_title"], how="inner")
    metro_year["avg_wage"] = metro_year["wage_bill"] / metro_year["wage_emp"]

    metro_year["log_emp"] = np.log(metro_year["total_emp"])
    metro_year["log_wage"] = np.log(metro_year["avg_wage"])
    metro_year["log_wbill"] = np.log(metro_year["wage_bill"])
    metro_year["z_bartik_post"] = metro_year["bartik_exposure"] * metro_year["post"]

    # Balanced metro panel restriction.
    n_years = len(YEARS)
    counts = metro_year.groupby("area")["year"].nunique()
    balanced_areas = counts[counts == n_years].index
    metro_year["balanced"] = metro_year["area"].isin(balanced_areas).astype(int)

    # Drop rows with insufficient wage coverage and unbalanced metros.
    sample = metro_year[
        (metro_year["balanced"] == 1)
        & metro_year["log_emp"].notna()
        & metro_year["log_wage"].notna()
        & (metro_year["wage_emp"] > 0)
    ].copy()

    out_path = RESULTS_DIR / f"table_m2b_metro_shift_share_metro_panel_{exposure_col}.csv"
    sample.to_csv(out_path, index=False)

    log.info(
        f"Metro-year sample using {exposure_col}: {len(sample):,} obs | "
        f"metros={sample['area'].nunique():,} | years={sorted(sample['year'].unique())}"
    )
    log.info(
        f"Bartik exposure distribution across metros:\n"
        f"{sample.drop_duplicates('area')['bartik_exposure'].describe().round(4).to_string()}"
    )
    log.info(f"Saved metro-year panel → {out_path}")
    return sample


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Estimation utilities
# ═══════════════════════════════════════════════════════════════════════════════

def twoway_demean(df: pd.DataFrame, cols: list[str], entity: str = "area", time: str = "year") -> pd.DataFrame:
    out = df[cols].astype(float).copy()
    grand = out.mean()
    entity_means = out.groupby(df[entity]).transform("mean")
    time_means = out.groupby(df[time]).transform("mean")
    return out - entity_means - time_means + grand


def fit_panel_ols(sample: pd.DataFrame, y_col: str, x_cols: list[str]) -> object:
    panel = sample.set_index(["area", "year"])
    sub = panel.dropna(subset=[y_col] + x_cols)
    mod = PanelOLS(sub[y_col], sub[x_cols], entity_effects=True, time_effects=True)
    return mod.fit(cov_type="clustered", cluster_entity=True)


def run_first_stage(sample: pd.DataFrame, exposure_col: str) -> pd.DataFrame:
    res = fit_panel_ols(sample, "log_emp", ["z_bartik_post"])
    b = float(res.params["z_bartik_post"])
    se = float(res.std_errors["z_bartik_post"])
    out = pd.DataFrame([{
        "exposure_measure": exposure_col,
        "equation": "first_stage",
        "outcome": "log_emp",
        "coefficient": b,
        "se": se,
        "t_stat": float(res.tstats["z_bartik_post"]),
        "p_value": float(res.pvalues["z_bartik_post"]),
        "n_obs": int(res.nobs),
        "n_metros": int(sample["area"].nunique()),
    }])
    path = RESULTS_DIR / f"table_m2b_metro_shift_share_first_stage_{exposure_col}.csv"
    out.to_csv(path, index=False)
    log.info(f"First stage: π={b:+.4f}, se={se:.4f}, p={out.loc[0, 'p_value']:.4f}")
    log.info(f"Saved → {path}")
    return out


def run_reduced_form(sample: pd.DataFrame, exposure_col: str) -> pd.DataFrame:
    res = fit_panel_ols(sample, "log_wage", ["z_bartik_post"])
    b = float(res.params["z_bartik_post"])
    se = float(res.std_errors["z_bartik_post"])
    out = pd.DataFrame([{
        "exposure_measure": exposure_col,
        "equation": "reduced_form",
        "outcome": "log_wage",
        "coefficient": b,
        "se": se,
        "t_stat": float(res.tstats["z_bartik_post"]),
        "p_value": float(res.pvalues["z_bartik_post"]),
        "n_obs": int(res.nobs),
        "n_metros": int(sample["area"].nunique()),
    }])
    path = RESULTS_DIR / f"table_m2b_metro_shift_share_reduced_form_{exposure_col}.csv"
    out.to_csv(path, index=False)
    log.info(f"Reduced form: ρ={b:+.4f}, se={se:.4f}, p={out.loc[0, 'p_value']:.4f}")
    log.info(f"Saved → {path}")
    return out


def run_structural_iv(sample: pd.DataFrame, exposure_col: str) -> pd.DataFrame:
    cols = ["log_wage", "log_emp", "z_bartik_post"]
    dm = twoway_demean(sample, cols)
    dm.columns = [c + "_dm" for c in dm.columns]
    valid = dm.notna().all(axis=1)
    y = dm.loc[valid, "log_wage_dm"]
    x = dm.loc[valid, ["log_emp_dm"]]
    z = dm.loc[valid, ["z_bartik_post_dm"]]

    clusters = sample.loc[valid, "area"]
    try:
        res = IV2SLS(y, None, x, z).fit(cov_type="clustered", clusters=clusters)
    except Exception:
        # Fallback for environments where clustered IV covariance fails.
        res = IV2SLS(y, None, x, z).fit(cov_type="robust")

    fs_diag = res.first_stage.diagnostics
    fs_f = float(fs_diag.loc["log_emp_dm", "f.stat"])
    fs_p = float(fs_diag.loc["log_emp_dm", "f.pval"])

    b = float(res.params["log_emp_dm"])
    se = float(res.std_errors["log_emp_dm"])
    out = pd.DataFrame([{
        "exposure_measure": exposure_col,
        "equation": "structural_iv",
        "outcome": "log_wage",
        "endogenous": "log_emp",
        "instrument": "bartik_exposure_x_post",
        "epsilon_inverse_labor_supply": b,
        "se": se,
        "t_stat": float(res.tstats["log_emp_dm"]),
        "p_value": float(res.pvalues["log_emp_dm"]),
        "first_stage_f": fs_f,
        "first_stage_p": fs_p,
        "n_obs": int(res.nobs),
        "n_metros": int(sample["area"].nunique()),
    }])
    path = RESULTS_DIR / f"table_m2b_metro_shift_share_structural_iv_{exposure_col}.csv"
    out.to_csv(path, index=False)
    log.info(
        f"Structural IV: ε={b:+.4f}, se={se:.4f}, p={out.loc[0, 'p_value']:.4f}, "
        f"first-stage F={fs_f:.2f}"
    )
    log.info(f"Saved → {path}")
    return out


def run_event_study(sample: pd.DataFrame, exposure_col: str) -> pd.DataFrame:
    df = sample.copy()
    event_times = sorted(t for t in df["event_time"].unique() if t != BASE_TAU)
    cols = []
    for tau in event_times:
        c = f"bartik_x_tau_{tau:+d}"
        df[c] = df["bartik_exposure"] * (df["event_time"] == tau).astype(int)
        cols.append(c)

    fs = fit_panel_ols(df, "log_emp", cols)
    rf = fit_panel_ols(df, "log_wage", cols)

    rows = []
    for tau, c in zip(event_times, cols):
        pi = float(fs.params[c])
        pi_se = float(fs.std_errors[c])
        rho = float(rf.params[c])
        rho_se = float(rf.std_errors[c])
        wald = rho / pi if abs(pi) > 1e-10 else np.nan
        # Approximate ratio SE ignoring covariance. Diagnostic only.
        wald_se = np.nan
        if abs(pi) > 1e-10 and rho != 0:
            wald_se = abs(wald) * np.sqrt((rho_se / rho) ** 2 + (pi_se / pi) ** 2)
        rows.append({
            "exposure_measure": exposure_col,
            "tau": int(tau),
            "year": int(BASE_YEAR + tau),
            "first_stage_pi": pi,
            "first_stage_se": pi_se,
            "reduced_form_rho": rho,
            "reduced_form_se": rho_se,
            "wald_rho_over_pi": wald,
            "wald_se_approx": wald_se,
        })

    rows.append({
        "exposure_measure": exposure_col,
        "tau": 0,
        "year": BASE_YEAR,
        "first_stage_pi": 0.0,
        "first_stage_se": 0.0,
        "reduced_form_rho": 0.0,
        "reduced_form_se": 0.0,
        "wald_rho_over_pi": np.nan,
        "wald_se_approx": np.nan,
    })

    out = pd.DataFrame(rows).sort_values("tau").reset_index(drop=True)
    path = RESULTS_DIR / f"table_m2b_metro_shift_share_event_study_{exposure_col}.csv"
    out.to_csv(path, index=False)
    log.info(f"Saved event-study table → {path}")
    return out


def plot_event_study(event_df: pd.DataFrame, structural_iv: pd.DataFrame, exposure_col: str) -> None:
    df = event_df.sort_values("tau").copy()
    eps = float(structural_iv.loc[0, "epsilon_inverse_labor_supply"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
    fig.suptitle(
        "Metro-level Shift-Share Labor Supply IV\n"
        r"$Z_{mt}=(\sum_o s_{mo,pre}\,LLMExposure_o)\times \mathbf{1}[t>2022]$"
        f"  | exposure={exposure_col}",
        fontsize=12,
        y=1.05,
    )

    specs = [
        ("first_stage_pi", "first_stage_se", "First stage\nBartik shock → log employment", r"$\pi_\tau$"),
        ("reduced_form_rho", "reduced_form_se", "Reduced form\nBartik shock → log wage", r"$\rho_\tau$"),
        ("wald_rho_over_pi", "wald_se_approx", "Period-by-period Wald\nReduced form / first stage", r"$\hat{\varepsilon}_\tau$"),
    ]

    for ax, (coef, se, title, ylabel) in zip(axes, specs):
        y = df[coef]
        err = 1.96 * df[se]
        ax.axhline(0, linewidth=1)
        ax.axvline(0.5, linestyle="--", linewidth=1)
        ax.errorbar(df["tau"], y, yerr=err, fmt="o", capsize=3)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Event time τ = year − 2022")
        ax.set_xticks(df["tau"])
        ax.set_xticklabels([f"τ={int(t):+d}\n({int(BASE_YEAR+t)})" for t in df["tau"]])
        ax.grid(True, alpha=0.3)
        if coef == "wald_rho_over_pi":
            ax.axhline(eps, linestyle="-.", linewidth=1, label=f"Pooled 2SLS ε = {eps:.3f}")
            ax.legend(fontsize=8)

    path = RESULTS_DIR / f"figure_m2b_metro_shift_share_iv_{exposure_col}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved figure → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_for_exposure(exposure_col: str) -> None:
    log.info("=" * 76)
    log.info(f"METRO SHIFT-SHARE IV using exposure measure: {exposure_col}")
    log.info("=" * 76)

    metro_occ = build_or_load_metro_occ_panel(exposure_col)
    sample = construct_metro_year_panel(metro_occ, exposure_col)

    first_stage = run_first_stage(sample, exposure_col)
    reduced_form = run_reduced_form(sample, exposure_col)
    structural_iv = run_structural_iv(sample, exposure_col)
    event = run_event_study(sample, exposure_col)
    plot_event_study(event, structural_iv, exposure_col)

    # Small console summary.
    log.info("\nSummary:")
    log.info(f"  First stage π: {first_stage.loc[0, 'coefficient']:+.4f}")
    log.info(f"  Reduced form ρ: {reduced_form.loc[0, 'coefficient']:+.4f}")
    log.info(f"  2SLS ε:        {structural_iv.loc[0, 'epsilon_inverse_labor_supply']:+.4f}")
    log.info(f"  First-stage F:{structural_iv.loc[0, 'first_stage_f']:.2f}")


def main() -> None:
    try:
        run_for_exposure(MAIN_EXPOSURE)
        # Robustness check if human_beta exists in exposure file.
        try:
            run_for_exposure(ROBUSTNESS_EXPOSURE)
        except Exception as e:
            log.warning(f"Skipping robustness exposure {ROBUSTNESS_EXPOSURE}: {e}")
    except FileNotFoundError as e:
        log.error(str(e))
        log.error(
            "\nTo use this script, download the OEWS metro/nonmetro all-data Excel files "
            "for 2019--2024 and place them in data/raw/oews_metro/."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
