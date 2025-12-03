"""
Financial statement data retrieval using OpenDartReader.

Fetches fundamental data from DART (Data Analysis, Retrieval and Transfer System).
DART API key required: https://opendart.fss.or.kr/
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .cache import save_to_cache, load_from_cache

# Load .env file from project root
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / '.env')


def get_dart_api_key() -> str:
    """
    Get DART API key from environment (.env file).

    Returns:
        API key string

    Raises:
        ValueError: If no API key found
    """
    api_key = os.environ.get('DART_API_KEY')
    if api_key:
        return api_key

    raise ValueError(
        "DART API key not found. Add DART_API_KEY to .env file."
    )


def _get_dart_reader():
    """Get OpenDartReader instance with API key."""
    import OpenDartReader
    api_key = get_dart_api_key()
    return OpenDartReader(api_key)


def fetch_financial_statements(
    tickers: list[str],
    year: int,
    report_type: str = 'annual',
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch financial statement data from DART.

    Parameters:
        tickers: List of 6-digit stock codes
        year: Fiscal year (e.g., 2023)
        report_type: 'annual' (11011) or 'quarterly' (11012, 11013, 11014)
        use_cache: Whether to use cached data

    Returns:
        DataFrame with columns:
        - ticker: Stock code
        - total_equity: Total shareholders' equity
        - total_assets: Total assets
        - net_income: Net income
        - revenue: Total revenue

    Note:
        DART API has daily limit of 20,000 calls. Use caching.
    """
    cache_key = f"financials_{year}_{report_type}"

    if use_cache:
        cached = load_from_cache(cache_key, max_age_days=30)
        if cached is not None:
            # Filter to requested tickers
            return cached[cached['ticker'].isin(tickers)]

    dart = _get_dart_reader()

    # Report type codes
    report_codes = {
        'annual': '11011',      # 사업보고서
        'q1': '11013',          # 1분기
        'q2': '11012',          # 반기
        'q3': '11014',          # 3분기
    }
    reprt_code = report_codes.get(report_type, '11011')

    results = []

    for ticker in tickers:
        try:
            # Fetch financial statements
            # fs_all: 전체 재무제표
            fs = dart.finstate_all(ticker, year, reprt_code)

            if fs is None or len(fs) == 0:
                continue

            # Extract key metrics
            # sj_div: BS=재무상태표, IS=손익계산서, CIS=포괄손익, CF=현금흐름, SCE=자본변동
            bs = fs[fs['sj_div'] == 'BS']  # Balance Sheet
            # 손익계산서: IS 또는 CIS (회사마다 다름)
            income_stmt = fs[fs['sj_div'] == 'IS']
            if len(income_stmt) == 0:
                income_stmt = fs[fs['sj_div'] == 'CIS']  # 포괄손익계산서

            # Parse values
            data = {'ticker': ticker, 'year': year}

            # Total Equity (자본총계) - from Balance Sheet
            equity_row = bs[bs['account_nm'].str.contains('자본총계', na=False)]
            if len(equity_row) > 0:
                data['total_equity'] = _parse_amount(equity_row.iloc[0]['thstrm_amount'])
                # Prior year equity
                if 'frmtrm_amount' in equity_row.columns:
                    data['total_equity_prior'] = _parse_amount(equity_row.iloc[0]['frmtrm_amount'])

            # Total Assets (자산총계) - from Balance Sheet
            assets_row = bs[bs['account_nm'].str.contains('자산총계', na=False)]
            if len(assets_row) > 0:
                data['total_assets'] = _parse_amount(assets_row.iloc[0]['thstrm_amount'])

            # Net Income (당기순이익) - from Income Statement
            # 지배기업 귀속 순이익 우선, 없으면 전체 당기순이익
            income_row = income_stmt[income_stmt['account_nm'].str.contains('지배기업.*당기순이익', na=False, regex=True)]
            if len(income_row) == 0:
                income_row = income_stmt[income_stmt['account_nm'].str.contains('당기순이익', na=False)]
            if len(income_row) > 0:
                data['net_income'] = _parse_amount(income_row.iloc[0]['thstrm_amount'])

            # Revenue (매출액) - from Income Statement
            revenue_row = income_stmt[income_stmt['account_nm'].str.contains('^매출액$|^영업수익$|^수익', na=False, regex=True)]
            if len(revenue_row) > 0:
                data['revenue'] = _parse_amount(revenue_row.iloc[0]['thstrm_amount'])

            results.append(data)

        except Exception as e:
            print(f"Warning: Failed to fetch {ticker} for {year}: {e}")
            continue

    df = pd.DataFrame(results)

    # Cache all results
    if use_cache and len(df) > 0:
        save_to_cache(df, cache_key)

    return df


def _parse_amount(value) -> float:
    """Parse amount string to float, handling Korean number format."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)

    # Remove commas and convert
    try:
        return float(str(value).replace(',', ''))
    except (ValueError, TypeError):
        return np.nan


def calculate_book_value(financials: pd.DataFrame) -> pd.Series:
    """
    Extract book value (total equity) from financial data.

    Parameters:
        financials: DataFrame from fetch_financial_statements

    Returns:
        Series indexed by ticker with book values
    """
    if 'total_equity' not in financials.columns:
        return pd.Series(dtype=float)

    return financials.set_index('ticker')['total_equity']


def calculate_roe_from_financials(financials: pd.DataFrame) -> pd.Series:
    """
    Calculate ROE from financial statement data.

    ROE = Net Income / Average Equity
    where Average Equity = (Current Equity + Prior Equity) / 2

    Parameters:
        financials: DataFrame from fetch_financial_statements

    Returns:
        Series indexed by ticker with ROE values
    """
    required_cols = ['net_income', 'total_equity', 'total_equity_prior']
    if not all(col in financials.columns for col in required_cols):
        # Fallback: use current equity only
        if 'net_income' in financials.columns and 'total_equity' in financials.columns:
            df = financials.set_index('ticker')
            return df['net_income'] / df['total_equity']
        return pd.Series(dtype=float)

    df = financials.set_index('ticker')
    avg_equity = (df['total_equity'] + df['total_equity_prior']) / 2
    roe = df['net_income'] / avg_equity

    # Handle edge cases
    roe = roe.replace([np.inf, -np.inf], np.nan)

    return roe


def get_available_years(ticker: str, start_year: int = 2015) -> list[int]:
    """
    Get years with available financial data for a ticker.

    Parameters:
        ticker: 6-digit stock code
        start_year: Earliest year to check

    Returns:
        List of years with data
    """
    from datetime import datetime
    current_year = datetime.now().year

    dart = _get_dart_reader()
    available = []

    for year in range(start_year, current_year + 1):
        try:
            fs = dart.finstate_all(ticker, year, '11011')
            if fs is not None and len(fs) > 0:
                available.append(year)
        except Exception:
            continue

    return available
