"""
src/data_pipeline/features.py

Builds the per-asset predictor matrix f_i (T x K) and response vector r_i
(T x 1) for a given portfolio universe, matching the SUR structure in the
dissertation: for each asset i, f_i,t contains [intercept, 15 macro-type
predictors (9 macro + 6 factors), 8 asset-level predictors, 120
interactions] and predicts r_{i,t+1}.

Factors (Mkt-RF, SMB, HML, RMW, CMA, UMD) are grouped with the macro
predictors rather than the asset-level block: they are identical across
every asset each month, so structurally they behave as macro-type
variables, not genuine asset-level characteristics (which must vary by
both asset and time). Only the portfolio's own-history features are
genuinely asset-level.

All standardisation is expanding-window (point-in-time): at each date t,
a column is standardised using only the mean/std of that column's own
history up to and including t. This avoids look-ahead bias, consistent
with the predictive, no-look-ahead design used throughout the dissertation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_pipeline.clean import clean_universe

RAW_MACRO_COLUMNS = ["DY", "EP", "BM", "NTIS", "SVAR", "TBL", "CPI", "DFY", "TMS"]  # 9, from Goyal
FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]  # 6, shared across assets -> macro-type
MACRO_COLUMNS = RAW_MACRO_COLUMNS + FACTOR_COLUMNS  # 15 total, the combined "macro-type" block

ASSET_COLUMNS = [
    "lag1", "lag3", "lag6", "lag12", "roll_mean12", "roll_vol12",
    "roll_skew12", "mom_12_1",
]  # 8, genuinely asset-specific

ROLLING_WINDOW = 12
STABILISATION_WINDOW = 30  # months dropped at the start of each series, beyond what NaNs
              # alone would force -- see thesis \S4.3 standardisation deviation
              # note: an expanding-window mean/std is unstable for roughly its
              # first 24-36 observations, so the stabilisation-window is set at 30 rather
              # than just the ~13 months strictly required to clear all NaNs.

def expanding_standardise(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Point-in-time (expanding-window) standardisation: at each row t, use
    only the mean/std of the column's history up to and including t.
    Works for both a single Series and a multi-column DataFrame.
    """
    expanding = data.expanding()
    mean = expanding.mean()
    std = expanding.std()
    return (data - mean) / std

def construct_macro_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the 9 raw macro predictors from the Goyal columns:
    5 direct (BM, NTIS, SVAR, TBL, CPI/infl) + 4 constructed
    (DY, EP, Default Spread, Term Spread). Factors (Mkt-RF, SMB, HML, RMW,
    CMA, UMD) are handled separately in build_features(), since they come
    from the factors file, not the macro file.
    """
    out = pd.DataFrame(index=macro_df.index)

    # Direct columns
    out["BM"] = macro_df["b/m"]
    out["NTIS"] = macro_df["ntis"]
    out["SVAR"] = macro_df["svar"]
    out["TBL"] = macro_df["tbl"]
    out["CPI"] = macro_df["infl"]

    # Constructed columns
    out["DY"] = np.log(macro_df["D12"]) - np.log(macro_df["Index"].shift(1))
    out["EP"] = np.log(macro_df["E12"]) - np.log(macro_df["Index"])
    out["DFY"] = macro_df["BAA"] - macro_df["AAA"]
    out["TMS"] = macro_df["lty"] - macro_df["tbl"]

    return out[RAW_MACRO_COLUMNS]

def construct_own_history_features(excess_return: pd.Series) -> pd.DataFrame:
    """
    Build the 8 portfolio-history features for a single asset's own excess
    return series: lag 1/3/6/12, 12-month trailing rolling mean/vol/skew,
    and 12-1 momentum (cumulative return from t-11 to t-1, skipping the
    most recent month to avoid short-term reversal contamination, following
    Jegadeesh and Titman 1993).
    """
    out = pd.DataFrame(index=excess_return.index)
    out["lag1"] = excess_return.shift(1)
    out["lag3"] = excess_return.shift(3)
    out["lag6"] = excess_return.shift(6)
    out["lag12"] = excess_return.shift(12)
    out["roll_mean12"] = excess_return.rolling(ROLLING_WINDOW).mean()
    out["roll_vol12"] = excess_return.rolling(ROLLING_WINDOW).std()
    out["roll_skew12"] = excess_return.rolling(ROLLING_WINDOW).skew()

    # 12-1 momentum: skip the most recent month (shift(1) drops this
    # month's own return), then compound the following 11 months of
    # returns into one cumulative return.
    skipped = excess_return.shift(1)
    out["mom_12_1"] = (1 + skipped).rolling(ROLLING_WINDOW - 1).apply(
        lambda x: x.prod(), raw=True
    ) - 1

    return out[ASSET_COLUMNS]

def build_asset_panel(
    asset_excess_return: pd.Series,
    macro_std: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Assemble the full f_i (T x K) predictor matrix and r_i (T x 1) response
    for a single asset, given its own excess-return series and the
    already-standardised, shared 15-column macro block (9 macro + 6
    factors -- factors are shared across every asset, so structurally they
    belong with the macro predictors, not the asset-level block).

    Returns (F_i, r_i), with unusable rows dropped (see below).
    """
    own_history = construct_own_history_features(asset_excess_return)
    asset_std = expanding_standardise(own_history)[ASSET_COLUMNS]

    # 15 x 8 = 120 interaction terms, computed after standardising the base
    # variables, per asset (since asset_std differs by asset).
    interaction_cols = {}
    for macro_col in MACRO_COLUMNS:
        for asset_col in ASSET_COLUMNS:
            interaction_cols[f"{macro_col}_x_{asset_col}"] = (
                macro_std[macro_col] * asset_std[asset_col]
            )
    interactions = pd.DataFrame(interaction_cols, index=asset_excess_return.index)

    f_i = pd.concat([macro_std, asset_std, interactions], axis=1)
    f_i.insert(0, "intercept", 1.0)

    # response: r_{i,t+1}, i.e. next month's excess return
    r_i = asset_excess_return.shift(-1)
    r_i.name = "r_next"

    # Drop any row that isn't fully usable. This single dropna (rather than
    # a fixed iloc[ROLLING_WINDOW:] slice) correctly handles THREE distinct
    # sources of NaN at once: (1) the first ROLLING_WINDOW months, where the
    # lag/rolling windows aren't yet available; (2) one additional row after
    # that, where a column's raw value has just become available but its
    # expanding standard deviation still can't be computed from a single
    # observation (this affects lag12 and mom_12_1 specifically, since they
    # are the last columns to "start", and this edge case only becomes
    # visible once expanding-window standardisation -- rather than one-shot
    # global standardisation -- is used); and (3) the final month, where
    # r_{t+1} doesn't exist yet.
    combined = pd.concat([f_i, r_i], axis=1).dropna()
    f_i = combined.drop(columns="r_next")
    r_i = combined["r_next"]

    # Additionally enforce a fixed BURN_IN-month cutoff from the start of
    # the asset's own history, regardless of how many months the dropna
    # step above already removed. This guarantees every surviving row's
    # expanding-window standardisation has at least BURN_IN months of
    # accumulated history behind it, not just the bare minimum needed to
    # clear NaNs (see thesis standardisation-deviation note).
    cutoff_date = asset_excess_return.index[STABILISATION_WINDOW]
    f_i = f_i.loc[f_i.index >= cutoff_date]
    r_i = r_i.loc[r_i.index >= cutoff_date]

    return f_i, r_i

def build_features(
    universe: str, drop_columns: list[str] | None = None
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """
    Full orchestration for one universe: clean -> construct macro & factor
    blocks -> standardise -> build per-asset (F_i, r_i) panels.

    Returns a dict {asset_name: (F_i, r_i)}.
    """
    excess, factors, macro = clean_universe(universe, drop_columns=drop_columns)

    macro_features = construct_macro_features(macro)
    factors_std = expanding_standardise(factors)
    macro_std = pd.concat([expanding_standardise(macro_features), factors_std], axis=1)
    macro_std = macro_std[MACRO_COLUMNS]

    panels = {}
    for asset in excess.columns:
        f_i, r_i = build_asset_panel(excess[asset], macro_std)
        panels[asset] = (f_i, r_i)

    return panels

if __name__ == "__main__":
    panels = build_features("size_bm_25")
    print(f"number of assets: {len(panels)}")

    sample_asset = list(panels.keys())[0]
    f_i, r_i = panels[sample_asset]
    print(f"\nsample asset: {sample_asset}")
    print(f"F_i shape: {f_i.shape} (expect T x 144)")
    print(f"r_i shape: {r_i.shape}")
    print(f"F_i NaNs: {int(f_i.isna().sum().sum())}, r_i NaNs: {int(r_i.isna().sum())}")
    print(f"date range: {f_i.index.min()} -> {f_i.index.max()}")
    print(f"\ncolumns (first 16): {list(f_i.columns[:16])}")
    print(f"total columns: {len(f_i.columns)}")

