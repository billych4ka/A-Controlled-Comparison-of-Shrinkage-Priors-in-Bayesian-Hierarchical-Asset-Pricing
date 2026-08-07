"""
src/data_pipeline/load.py

Robust loaders for the raw Ken French (factors, portfolios) and Goyal (macro)
data files. Each loader parses the file's real on-disk structure directly
(scanning for markers/headers rather than assuming fixed skiprows), recodes
Ken French's missing-value sentinels to NaN, and returns a clean,
date-indexed pandas DataFrame.

No date-range trimming, merging, or listwise deletion happens here -- that
is clean.py's responsibility. This module's only job is: raw file -> clean
DataFrame, one file at a time.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths / registry
# ---------------------------------------------------------------------------

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PORTFOLIOS_DIR = RAW_DIR / "portfolios"
FACTORS_DIR = RAW_DIR / "factors"
MACRO_DIR = RAW_DIR / "macro"

# Maps a universe name (used throughout the pipeline / results/ folder names)
# to its raw portfolio file. build_dataset.py loops over this dict so that
# adding a new universe never requires touching load.py again.
PORTFOLIO_FILES: dict[str, str] = {
    "size_bm_25": "25_Portfolios_BM_5x5.csv",
    "size_op_25": "25_Portfolios_OP_5x5.csv",
    "size_inv_25": "25_Portfolios_INV_5x5.csv",
    "size_bm_100": "100_Portfolios_BM_10x10.csv",
}

# Ken French's documented missing-value sentinels (confirmed in both the
# portfolio files' preambles and the momentum file's preamble).
FRENCH_SENTINELS = [-99.99, -999, -999.99]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _recode_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw string columns to numeric, then replace Ken French's
    missing-value sentinels with NaN.

    Order matters: the sentinels must be matched as numbers, not as raw
    strings (source rows have inconsistent whitespace padding, e.g.
    ' -99.9900', so a string-level .replace() would silently miss them).
    """
    numeric = df.apply(pd.to_numeric, errors="coerce")
    return numeric.replace(FRENCH_SENTINELS, np.nan)


def _parse_yyyymm_index(raw_dates: pd.Series) -> pd.DatetimeIndex:
    """
    Convert a column of 'YYYYMM' integers/strings into a monthly PeriodIndex
    converted to a Timestamp at month-start. Using PeriodIndex under the hood
    avoids any ambiguity about which day-of-month represents "the month".
    """
    return pd.PeriodIndex(raw_dates.astype(str), freq="M").to_timestamp()

# ---------------------------------------------------------------------------
# Factor files
# ---------------------------------------------------------------------------


def _load_ff5(path: Path) -> pd.DataFrame:
    """
    Parse F-F_Research_Data_5_Factors_2x3.csv.

    Structure: 3 preamble lines, blank line, header row
    (',Mkt-RF,SMB,HML,RMW,CMA,RF'), then the MONTHLY block (YYYYMM dates),
    then a blank line, an "Annual Factors" label, a repeated header, and the
    ANNUAL block (YYYY dates) -- which we deliberately exclude.
    """
    lines = path.read_text().splitlines()

    # Find the header row: first line starting with ',Mkt-RF'
    header_idx = next(i for i, ln in enumerate(lines) if ln.startswith(",Mkt-RF"))

    # Monthly data runs from header_idx+1 until the first blank line.
    data_lines = []
    for ln in lines[header_idx + 1 :]:
        if ln.strip() == "":
            break
        data_lines.append(ln)

    header = lines[header_idx].split(",")[1:]  # drop the empty date-column name
    rows = [ln.split(",") for ln in data_lines]

    df = pd.DataFrame(rows, columns=["date"] + header)
    df["date"] = _parse_yyyymm_index(df["date"])
    df = df.set_index("date")
    df = _recode_sentinels(df)

    # French factor files report values in percent (e.g. 5.08 == 5.08%).
    # Convert to decimal returns for consistency with everything downstream.
    df = df / 100.0
    return df

def _load_momentum(path: Path) -> pd.DataFrame:
    """
    Parse F-F_Momentum_Factor.csv.

    Structure: 12 preamble lines (incl. an explicit note that missing data
    are -99.99 or -999), blank line, header row (',Mom'), MONTHLY block,
    blank line, two-line "Annual Factors:" / "January-December" label,
    repeated header, ANNUAL block (excluded).

    The column is renamed Mom -> UMD to match the dissertation's notation.
    """
    lines = path.read_text().splitlines()

    header_idx = next(i for i, ln in enumerate(lines) if ln.startswith(",Mom"))

    data_lines = []
    for ln in lines[header_idx + 1 :]:
        if ln.strip() == "":
            break
        data_lines.append(ln)

    rows = [ln.split(",") for ln in data_lines]
    df = pd.DataFrame(rows, columns=["date", "Mom"])
    df["date"] = _parse_yyyymm_index(df["date"])
    df = df.set_index("date")
    df = _recode_sentinels(df)
    df = df / 100.0
    df = df.rename(columns={"Mom": "UMD"})
    return df

def load_factors() -> pd.DataFrame:
    """
    Load and merge FF5 (Mkt-RF, SMB, HML, RMW, CMA, RF) with UMD (momentum)
    on date. Returns decimal (not percent) returns, date-indexed, monthly.

    RF is included here (needed downstream to build excess returns) but is
    NOT one of the 6 asset-level factor predictors -- callers should drop
    it before using this frame as a predictor block.
    """
    ff5 = _load_ff5(FACTORS_DIR / "F-F_Research_Data_5_Factors_2x3.csv")
    umd = _load_momentum(FACTORS_DIR / "F-F_Momentum_Factor.csv")
    merged = ff5.join(umd, how="inner")
    return merged

# ---------------------------------------------------------------------------
# Macro file (Goyal)
# ---------------------------------------------------------------------------


def load_macro() -> pd.DataFrame:
    """
    Parse goyal_predictors.csv. No preamble, single header row, one row per
    month. The 'Index' column has thousand-separator commas inside quotes
    (e.g. "6,460.26"); missing values are the literal string 'NaN'.
    """
    path = MACRO_DIR / "goyal_predictors.csv"
    df = pd.read_csv(path, thousands=",", na_values=["NaN"])
    df = df.rename(columns={"yyyymm": "date"})
    df["date"] = _parse_yyyymm_index(df["date"])
    df = df.set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

# ---------------------------------------------------------------------------
# Macro file (Goyal)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Portfolio files (25/100 Size x {BM, OP, INV})
# ---------------------------------------------------------------------------


def load_portfolios(universe: str) -> pd.DataFrame:
    """
    Parse a Ken French sorted-portfolio file for the given universe key
    (one of PORTFOLIO_FILES). Extracts only the 'Average Value Weighted
    Returns -- Monthly' block (the file also contains an equal-weighted
    block, annual blocks, firm-count and market-cap blocks, and several
    characteristic-average blocks further down -- all excluded).

    Preamble length varies by file, so we scan for the block marker rather
    than assuming a fixed skiprows.
    """
    if universe not in PORTFOLIO_FILES:
        raise ValueError(
            f"Unknown universe '{universe}'. Expected one of {list(PORTFOLIO_FILES)}."
        )

    path = PORTFOLIOS_DIR / PORTFOLIO_FILES[universe]
    lines = path.read_text().splitlines()

    marker_idx = next(
        i for i, ln in enumerate(lines) if "Average Value Weighted Returns -- Monthly" in ln
    )
    header_idx = marker_idx + 1
    header = lines[header_idx].split(",")[1:]  # drop empty date-column name

    data_lines = []
    for ln in lines[header_idx + 1 :]:
        if ln.strip() == "":
            break
        data_lines.append(ln)

    rows = [ln.split(",") for ln in data_lines]
    df = pd.DataFrame(rows, columns=["date"] + header)
    df["date"] = _parse_yyyymm_index(df["date"])
    df = df.set_index("date")
    df = _recode_sentinels(df)
    df = df / 100.0  # percent -> decimal, consistent with load_factors()

    # Strip any stray whitespace left over from the fixed-width-style source
    # formatting (column names like ' ME1 BM2' with a leading space).
    df.columns = [c.strip() for c in df.columns]
    return df

def load_all_portfolios() -> dict[str, pd.DataFrame]:
    """Convenience: load every registered universe into a dict keyed by name."""
    return {universe: load_portfolios(universe) for universe in PORTFOLIO_FILES}

# ---------------------------------------------------------------------------
# Manual smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    factors = load_factors()
    print("factors:", factors.shape, factors.index.min(), "->", factors.index.max())
    print(factors.head(3))
    print()

    macro = load_macro()
    print("macro:", macro.shape, macro.index.min(), "->", macro.index.max())
    print(macro[["Index", "b/m", "tbl", "infl"]].head(3))
    print()

    for uni in PORTFOLIO_FILES:
        pf = load_portfolios(uni)
        print(f"{uni}: {pf.shape}, {pf.index.min()} -> {pf.index.max()}, "
              f"NaNs={int(pf.isna().sum().sum())}")
