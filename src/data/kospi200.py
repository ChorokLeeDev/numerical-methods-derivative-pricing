"""
KOSPI200 constituent management.

Uses pykrx to fetch KOSPI200 index constituents from KRX.
"""

from datetime import datetime, timedelta

import pandas as pd
from pykrx import stock as pykrx_stock

from .cache import save_to_cache, load_from_cache


def get_kospi200_tickers(
    reference_date: str = None,
    use_cache: bool = True
) -> list[str]:
    """
    Fetch KOSPI200 constituent tickers from KRX.

    Parameters:
        reference_date: YYYYMMDD or YYYY-MM-DD format (default: most recent business day)
        use_cache: Whether to use cached data (default: True)

    Returns:
        List of 6-digit stock codes (e.g., ['005930', '000660', ...])

    Example:
        >>> tickers = get_kospi200_tickers('20241201')
        >>> len(tickers)
        200
    """
    # Normalize date format
    if reference_date is None:
        reference_date = datetime.now().strftime('%Y%m%d')
    else:
        reference_date = reference_date.replace('-', '')

    # Try cache first
    cache_key = f"kospi200_{reference_date}"
    if use_cache:
        cached = load_from_cache(cache_key, max_age_days=7)
        if cached is not None:
            return cached['ticker'].tolist()

    # Fetch from KRX
    # KOSPI200 index code is "1028"
    tickers = pykrx_stock.get_index_portfolio_deposit_file("1028", reference_date)

    if len(tickers) == 0:
        # Try previous business days if no data
        date = datetime.strptime(reference_date, '%Y%m%d')
        for i in range(1, 10):
            prev_date = (date - timedelta(days=i)).strftime('%Y%m%d')
            tickers = pykrx_stock.get_index_portfolio_deposit_file("1028", prev_date)
            if len(tickers) > 0:
                break

    # Cache result
    if use_cache and len(tickers) > 0:
        df = pd.DataFrame({'ticker': tickers})
        save_to_cache(df, cache_key)

    return list(tickers)


def get_kospi200_changes(
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Get KOSPI200 constituent changes over a period.

    Compares constituents at monthly intervals to identify additions/removals.

    Parameters:
        start_date: Start date (YYYYMMDD or YYYY-MM-DD)
        end_date: End date (YYYYMMDD or YYYY-MM-DD)

    Returns:
        DataFrame with columns: [date, ticker, action]
        where action is 'add' or 'remove'
    """
    start_date = start_date.replace('-', '')
    end_date = end_date.replace('-', '')

    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')

    changes = []
    prev_tickers = None

    # Check at monthly intervals
    current = start
    while current <= end:
        date_str = current.strftime('%Y%m%d')
        current_tickers = set(get_kospi200_tickers(date_str, use_cache=True))

        if prev_tickers is not None and len(current_tickers) > 0:
            # Find additions
            added = current_tickers - prev_tickers
            for ticker in added:
                changes.append({
                    'date': date_str,
                    'ticker': ticker,
                    'action': 'add'
                })

            # Find removals
            removed = prev_tickers - current_tickers
            for ticker in removed:
                changes.append({
                    'date': date_str,
                    'ticker': ticker,
                    'action': 'remove'
                })

        if len(current_tickers) > 0:
            prev_tickers = current_tickers

        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1, day=1)
        else:
            current = current.replace(month=current.month + 1, day=1)

    return pd.DataFrame(changes)


def get_ticker_name(ticker: str) -> str:
    """
    Get company name for a ticker.

    Parameters:
        ticker: 6-digit stock code

    Returns:
        Company name in Korean
    """
    try:
        return pykrx_stock.get_market_ticker_name(ticker)
    except Exception:
        return ""


def get_all_tickers_with_names(reference_date: str = None) -> pd.DataFrame:
    """
    Get all KOSPI200 tickers with company names.

    Parameters:
        reference_date: YYYYMMDD format (default: most recent)

    Returns:
        DataFrame with columns: [ticker, name]
    """
    tickers = get_kospi200_tickers(reference_date)

    data = []
    for ticker in tickers:
        name = get_ticker_name(ticker)
        data.append({'ticker': ticker, 'name': name})

    return pd.DataFrame(data)
