"""
Quality factor implementation.

Return on Equity (ROE): Net Income / Shareholders' Equity.
Based on Novy-Marx (2013) and Fama-French (2015).
"""

import numpy as np
import pandas as pd

from quant.data.fundamental import fetch_financial_statements
from quant.portfolio.rebalance import get_latest_available_fiscal_year


def get_quality_factor_for_backtest(
    tickers: list,
    date: str
) -> pd.Series:
    """
    백테스트용 퀄리티 팩터 계산.

    DART API에서 재무제표를 가져와 ROE 계산.

    Parameters:
        tickers: 종목 코드 리스트
        date: 기준일 (YYYY-MM-DD)

    Returns:
        ROE Series (높을수록 고품질)
    """
    # 해당 시점에 사용 가능한 회계연도
    fiscal_year = get_latest_available_fiscal_year(date)

    # 재무제표 데이터 가져오기
    financials = fetch_financial_statements(tickers, fiscal_year, use_cache=True)

    if len(financials) == 0:
        return pd.Series(dtype=float)

    # ROE 계산
    return get_quality_factor(financials)


def calculate_roe(
    net_income: pd.Series,
    equity: pd.Series,
    equity_prior: pd.Series = None
) -> pd.Series:
    """
    Calculate Return on Equity (ROE).

    ROE = Net Income / Average Equity

    Higher ROE = Higher quality (efficient capital utilization)

    Parameters:
        net_income: Net income by ticker
        equity: Current period total equity by ticker
        equity_prior: Prior period total equity (optional, for average)

    Returns:
        Series indexed by ticker with ROE values

    Example:
        >>> roe = calculate_roe(net_income, equity)
        >>> roe.head()
        005930    0.15   # Samsung: 15% ROE
        035420    0.22   # NAVER: 22% ROE (higher quality)
    """
    common_tickers = net_income.index.intersection(equity.index)

    if len(common_tickers) == 0:
        return pd.Series(dtype=float)

    ni = net_income.loc[common_tickers]
    eq = equity.loc[common_tickers]

    # Use average equity if prior period available
    if equity_prior is not None:
        eq_prior = equity_prior.reindex(common_tickers)
        avg_equity = (eq + eq_prior.fillna(eq)) / 2
    else:
        avg_equity = eq

    # Calculate ROE
    roe = ni / avg_equity

    # Handle edge cases
    roe = roe.replace([np.inf, -np.inf], np.nan)

    # Remove extreme values (ROE > 100% or < -100% is suspicious)
    roe = roe[(roe > -1) & (roe < 1)]

    return roe


def calculate_roa(
    net_income: pd.Series,
    assets: pd.Series
) -> pd.Series:
    """
    Calculate Return on Assets (ROA).

    ROA = Net Income / Total Assets

    Alternative quality metric.

    Parameters:
        net_income: Net income by ticker
        assets: Total assets by ticker

    Returns:
        Series indexed by ticker with ROA values
    """
    common_tickers = net_income.index.intersection(assets.index)

    if len(common_tickers) == 0:
        return pd.Series(dtype=float)

    ni = net_income.loc[common_tickers]
    ta = assets.loc[common_tickers]

    roa = ni / ta

    # Handle edge cases
    roa = roa.replace([np.inf, -np.inf], np.nan)

    return roa


def calculate_gross_profitability(
    revenue: pd.Series,
    cogs: pd.Series,
    assets: pd.Series
) -> pd.Series:
    """
    Calculate Gross Profitability (Novy-Marx, 2013).

    GP = (Revenue - COGS) / Total Assets

    Strong predictor of future returns.

    Parameters:
        revenue: Total revenue by ticker
        cogs: Cost of goods sold by ticker
        assets: Total assets by ticker

    Returns:
        Series indexed by ticker with gross profitability
    """
    common_tickers = revenue.index.intersection(cogs.index).intersection(assets.index)

    if len(common_tickers) == 0:
        return pd.Series(dtype=float)

    rev = revenue.loc[common_tickers]
    cost = cogs.loc[common_tickers]
    ta = assets.loc[common_tickers]

    gross_profit = rev - cost
    gp = gross_profit / ta

    # Handle edge cases
    gp = gp.replace([np.inf, -np.inf], np.nan)

    return gp


def get_quality_factor(
    financials: pd.DataFrame
) -> pd.Series:
    """
    Convenience function to calculate quality factor from financial data.

    Uses ROE as the primary quality metric.

    Parameters:
        financials: DataFrame with columns:
            - ticker
            - net_income
            - total_equity
            - total_equity_prior (optional)

    Returns:
        Series with ROE values
    """
    required_cols = ['net_income', 'total_equity']
    if not all(col in financials.columns for col in required_cols):
        return pd.Series(dtype=float)

    df = financials.set_index('ticker')
    net_income = df['net_income']
    equity = df['total_equity']

    equity_prior = df.get('total_equity_prior')

    return calculate_roe(net_income, equity, equity_prior)
