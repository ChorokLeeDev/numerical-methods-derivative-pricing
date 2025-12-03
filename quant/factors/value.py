"""
Value factor implementation.

Book-to-Market (B/M) ratio: Book Value of Equity / Market Capitalization.
Based on Fama-French (1992).
"""

import numpy as np
import pandas as pd

from quant.data.fundamental import fetch_financial_statements, calculate_book_value
from quant.data.price import get_market_cap
from quant.portfolio.rebalance import get_latest_available_fiscal_year


def get_value_factor_for_backtest(
    tickers: list,
    date: str
) -> pd.Series:
    """
    백테스트용 가치 팩터 계산.

    DART API에서 재무제표를 가져와 B/M ratio 계산.

    Parameters:
        tickers: 종목 코드 리스트
        date: 기준일 (YYYY-MM-DD)

    Returns:
        B/M ratio Series (높을수록 저평가)
    """
    # 해당 시점에 사용 가능한 회계연도 (look-ahead bias 방지)
    fiscal_year = get_latest_available_fiscal_year(date)

    # 재무제표 데이터 가져오기
    financials = fetch_financial_statements(tickers, fiscal_year, use_cache=True)

    if len(financials) == 0:
        return pd.Series(dtype=float)

    # Book Value (자본총계)
    book_values = calculate_book_value(financials)

    # Market Cap (시가총액)
    market_caps = get_market_cap(tickers, date, use_cache=True)

    if len(market_caps) == 0:
        return pd.Series(dtype=float)

    # B/M ratio 계산
    return calculate_book_to_market(book_values, market_caps)


def calculate_book_to_market(
    book_values: pd.Series,
    market_caps: pd.Series
) -> pd.Series:
    """
    Calculate Book-to-Market value factor.

    B/M = Total Equity (Book Value) / Market Capitalization

    High B/M = Value stocks (cheap relative to book value)
    Low B/M = Growth stocks (expensive relative to book value)

    Parameters:
        book_values: Total equity by ticker (from financial statements)
        market_caps: Market capitalization by ticker

    Returns:
        Series indexed by ticker with B/M ratios

    Example:
        >>> bm = calculate_book_to_market(book_values, market_caps)
        >>> bm.head()
        005930    0.85   # Samsung: slightly undervalued
        035420    0.25   # NAVER: growth stock (high premium)
    """
    # Align indices
    common_tickers = book_values.index.intersection(market_caps.index)

    if len(common_tickers) == 0:
        return pd.Series(dtype=float)

    bv = book_values.loc[common_tickers]
    mc = market_caps.loc[common_tickers]

    # Calculate B/M ratio
    bm_ratio = bv / mc

    # Handle edge cases
    bm_ratio = bm_ratio.replace([np.inf, -np.inf], np.nan)

    # Remove negative B/M (companies with negative equity)
    bm_ratio = bm_ratio[bm_ratio > 0]

    return bm_ratio


def calculate_earnings_to_price(
    earnings: pd.Series,
    market_caps: pd.Series
) -> pd.Series:
    """
    Calculate Earnings-to-Price ratio (inverse of P/E).

    E/P = Net Income / Market Capitalization

    Alternative value metric to B/M.

    Parameters:
        earnings: Net income by ticker
        market_caps: Market capitalization by ticker

    Returns:
        Series indexed by ticker with E/P ratios
    """
    common_tickers = earnings.index.intersection(market_caps.index)

    if len(common_tickers) == 0:
        return pd.Series(dtype=float)

    e = earnings.loc[common_tickers]
    mc = market_caps.loc[common_tickers]

    ep_ratio = e / mc

    # Handle edge cases
    ep_ratio = ep_ratio.replace([np.inf, -np.inf], np.nan)

    return ep_ratio


def get_value_factor(
    financials: pd.DataFrame,
    market_caps: pd.Series
) -> pd.Series:
    """
    Convenience function to calculate value factor from financial data.

    Parameters:
        financials: DataFrame with 'ticker' and 'total_equity' columns
        market_caps: Market caps indexed by ticker

    Returns:
        Series with B/M ratios
    """
    if 'total_equity' not in financials.columns:
        return pd.Series(dtype=float)

    book_values = financials.set_index('ticker')['total_equity']

    return calculate_book_to_market(book_values, market_caps)
