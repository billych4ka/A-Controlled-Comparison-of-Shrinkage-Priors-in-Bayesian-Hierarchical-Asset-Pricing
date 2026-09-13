"""
src/data_pipeline/clean.py

Takes the raw-but-parsed DataFrames from load.py and produces analysis-ready
excess returns and aligned predictor sources: subtracts RF to build excess
returns, aligns all sources to a common date range, and applies missing-data
handling (listwise deletion, with an optional column-drop escape hatch for
universes like size_bm_100 where a few structurally sparse columns would
otherwise gut the sample).
"""

from __future__ import annotations

import pandas as pd

from src.data_pipeline.load import load_factors, load_macro, load_portfolios

def build_excess_returns(portfolio_df: pd.DataFrame, factors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Subtract RF from every column of portfolio_df, aligned by date.
    factors_df must contain an 'RF' column (as returned by load_factors()).
    """
    common_dates = portfolio_df.index.intersection(factors_df.index)
    portfolios = portfolio_df.loc[common_dates]
    rf = factors_df.loc[common_dates, "RF"]
    return portfolios.sub(rf, axis=0)

def align_sample_start(*dfs: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """
    Trim every given DataFrame to the common date range shared by all of
    them: starts at the latest of their individual start dates, ends at the
    earliest of their individual end dates.
    """
    start = max(df.index.min() for df in dfs)
    end = min(df.index.max() for df in dfs)
    return tuple(df.loc[start:end] for df in dfs)

def handle_missing(df: pd.DataFrame, drop_columns: list[str] | None = None) -> pd.DataFrame:
    """
    Balanced-panel missing-data handling: optionally drop named columns
    first (e.g. structurally sparse corner portfolios), then listwise-delete
    any remaining row containing an NaN anywhere.
    """
    if drop_columns:
        df = df.drop(columns=drop_columns)
    return df.dropna()

def clean_universe(
    universe: str, drop_columns: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates the full cleaning step for one portfolio universe.
    Returns (excess_returns, factors, macro), all aligned to the same
    final date index.
    """
    portfolios = load_portfolios(universe)
    factors = load_factors()
    macro = load_macro()

    excess = build_excess_returns(portfolios, factors)
    factors_only = factors.drop(columns="RF")

    excess, factors_only, macro = align_sample_start(excess, factors_only, macro)
    excess = handle_missing(excess, drop_columns=drop_columns)

    factors_only = factors_only.loc[excess.index]
    macro = macro.loc[excess.index]

    return excess, factors_only, macro

if __name__ == "__main__":
    for uni in ["size_bm_25", "size_op_25", "size_inv_25"]:
        excess, factors_only, macro = clean_universe(uni)
        print(f"{uni}: excess={excess.shape}, factors={factors_only.shape}, "
              f"macro={macro.shape}, range={excess.index.min()} -> {excess.index.max()}, "
              f"NaNs={int(excess.isna().sum().sum())}")

