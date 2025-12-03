"""
Price data retrieval using FinanceDataReader.

Fetches historical price data for Korean stocks with caching support.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import FinanceDataReader as fdr

from .cache import save_to_cache, load_from_cache


def fetch_price_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    cache_days: int = 1
) -> pd.DataFrame:
    """
    Fetch adjusted daily price data for multiple tickers.

    Parameters:
        tickers: List of 6-digit stock codes
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        use_cache: Whether to use cached data
        cache_days: Cache expiration in days

    Returns:
        DataFrame with DatetimeIndex and ticker columns containing adjusted close prices

    Example:
        >>> prices = fetch_price_data(['005930', '000660'], '2024-01-01', '2024-12-01')
        >>> prices.head()
                       005930    000660
        Date
        2024-01-02   78000.0  180000.0
        2024-01-03   77500.0  179000.0
    """
    # Create cache key from tickers hash
    ticker_hash = hash(tuple(sorted(tickers)))
    cache_key = f"prices_{start_date}_{end_date}_{ticker_hash}"

    if use_cache:
        cached = load_from_cache(cache_key, max_age_days=cache_days)
        if cached is not None:
            return cached

    # Fetch data for each ticker
    price_data = {}

    for ticker in tickers:
        try:
            df = fdr.DataReader(ticker, start_date, end_date)
            if len(df) > 0:
                price_data[ticker] = df['Close']
        except Exception as e:
            # Skip tickers that fail to fetch
            print(f"Warning: Failed to fetch {ticker}: {e}")
            continue

    if len(price_data) == 0:
        return pd.DataFrame()

    # Combine into single DataFrame
    prices = pd.DataFrame(price_data)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    # Cache result
    if use_cache:
        save_to_cache(prices, cache_key)

    return prices


def calculate_returns(
    prices: pd.DataFrame,
    period: Literal['daily', 'weekly', 'monthly'] = 'daily'
) -> pd.DataFrame:
    """
    Calculate returns from price data.

    Parameters:
        prices: DataFrame with adjusted close prices (ticker columns)
        period: Return calculation period

    Returns:
        DataFrame of returns with same structure as input

    Example:
        >>> returns = calculate_returns(prices, period='monthly')
        >>> returns.head()
                       005930    000660
        Date
        2024-01-31    0.025    -0.015
        2024-02-29    0.018     0.032
    """
    if period == 'daily':
        returns = prices.pct_change()
    elif period == 'weekly':
        # Resample to weekly (Friday close)
        weekly = prices.resample('W-FRI').last()
        returns = weekly.pct_change()
    elif period == 'monthly':
        # Resample to month-end
        monthly = prices.resample('ME').last()
        returns = monthly.pct_change()
    else:
        raise ValueError(f"Unknown period: {period}")

    return returns.dropna(how='all')


def get_market_cap(
    tickers: list[str],
    date: str,
    use_cache: bool = True
) -> pd.Series:
    """
    Get market capitalization for tickers on a specific date.

    Uses pykrx for market cap data which is more reliable.

    Parameters:
        tickers: List of 6-digit stock codes
        date: Date string (YYYYMMDD or YYYY-MM-DD)

    Returns:
        Series indexed by ticker with market cap values in KRW
    """
    from pykrx import stock as pykrx_stock

    # Normalize date
    date = date.replace('-', '')

    cache_key = f"marketcap_{date}"
    if use_cache:
        cached = load_from_cache(cache_key, max_age_days=7)
        if cached is not None:
            return cached.set_index('ticker')['market_cap']

    # Get market cap data from pykrx
    try:
        df = pykrx_stock.get_market_cap_by_ticker(date)
        if len(df) == 0:
            return pd.Series(dtype=float)

        # Filter to requested tickers
        market_caps = df['시가총액']
        market_caps = market_caps[market_caps.index.isin(tickers)]

        # Cache
        if use_cache:
            cache_df = pd.DataFrame({
                'ticker': market_caps.index,
                'market_cap': market_caps.values
            })
            save_to_cache(cache_df, cache_key)

        return market_caps

    except Exception as e:
        print(f"Warning: Failed to fetch market cap for {date}: {e}")
        return pd.Series(dtype=float)


def get_price_with_volume(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Get full OHLCV data for a single ticker.

    Parameters:
        ticker: 6-digit stock code
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with columns: [Open, High, Low, Close, Volume]
    """
    df = fdr.DataReader(ticker, start_date, end_date)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def fill_missing_prices(
    prices: pd.DataFrame,
    method: Literal['ffill', 'drop'] = 'ffill'
) -> pd.DataFrame:
    """
    Handle missing prices in the data.

    Parameters:
        prices: Price DataFrame
        method: 'ffill' to forward-fill, 'drop' to drop columns with missing data

    Returns:
        Cleaned price DataFrame
    """
    if method == 'ffill':
        return prices.ffill().bfill()
    elif method == 'drop':
        # Drop columns with more than 10% missing
        threshold = len(prices) * 0.9
        return prices.dropna(axis=1, thresh=int(threshold))
    else:
        raise ValueError(f"Unknown method: {method}")
