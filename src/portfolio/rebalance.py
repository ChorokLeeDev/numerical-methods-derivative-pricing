"""
Rebalancing schedule utilities.

Generate rebalancing dates and formation periods for backtesting.
"""

from datetime import datetime
from typing import Literal

import pandas as pd
from pandas.tseries.offsets import BMonthEnd, BQuarterEnd


def generate_rebalance_dates(
    start_date: str,
    end_date: str,
    frequency: Literal['monthly', 'quarterly', 'semi-annual', 'annual'] = 'monthly'
) -> list[str]:
    """
    Generate rebalancing dates.

    Parameters:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        frequency: Rebalancing frequency

    Returns:
        List of rebalance dates (last business day of each period)

    Example:
        >>> dates = generate_rebalance_dates('2024-01-01', '2024-06-30', 'monthly')
        >>> dates
        ['2024-01-31', '2024-02-29', '2024-03-29', '2024-04-30', '2024-05-31', '2024-06-28']
    """
    freq_map = {
        'monthly': 'BME',      # Business Month End
        'quarterly': 'BQE',    # Business Quarter End
        'semi-annual': '6BME', # Every 6 months
        'annual': 'BAE'        # Business Year End
    }

    if frequency not in freq_map:
        raise ValueError(f"Unknown frequency: {frequency}")

    freq = freq_map[frequency]

    if frequency == 'semi-annual':
        # Generate monthly then filter to every 6 months
        dates = pd.date_range(start=start_date, end=end_date, freq='BME')
        dates = dates[::6]  # Every 6th month
    else:
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    return [d.strftime('%Y-%m-%d') for d in dates]


def get_formation_period(
    rebalance_date: str,
    lookback_months: int = 12,
    skip_months: int = 1
) -> tuple[str, str]:
    """
    Calculate formation period for factor calculation.

    Parameters:
        rebalance_date: Portfolio formation date
        lookback_months: How far back to look
        skip_months: Most recent months to skip

    Returns:
        Tuple of (start_date, end_date) for formation period

    Example:
        >>> get_formation_period('2024-06-30', lookback_months=12, skip_months=1)
        ('2023-05-31', '2024-05-31')
    """
    ref_date = pd.to_datetime(rebalance_date)

    # End of formation period (skip recent months)
    end_date = ref_date - pd.DateOffset(months=skip_months)
    end_date = end_date + BMonthEnd(0)  # Adjust to month end

    # Start of formation period
    start_date = ref_date - pd.DateOffset(months=lookback_months)
    start_date = start_date + BMonthEnd(0)

    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def get_holding_period(
    rebalance_date: str,
    frequency: str = 'monthly'
) -> tuple[str, str]:
    """
    Get the holding period until next rebalance.

    Parameters:
        rebalance_date: Current rebalance date
        frequency: Rebalancing frequency

    Returns:
        Tuple of (start_date, end_date) for holding period
    """
    start = pd.to_datetime(rebalance_date) + pd.DateOffset(days=1)

    if frequency == 'monthly':
        end = start + BMonthEnd(1)
    elif frequency == 'quarterly':
        end = start + BQuarterEnd(1)
    elif frequency == 'semi-annual':
        end = start + pd.DateOffset(months=6)
        end = end + BMonthEnd(0)
    elif frequency == 'annual':
        end = start + pd.DateOffset(years=1)
        end = end + BMonthEnd(0)
    else:
        raise ValueError(f"Unknown frequency: {frequency}")

    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def get_fiscal_data_available_date(
    fiscal_year_end: str,
    lag_months: int = 3
) -> str:
    """
    Get the date when fiscal year data becomes available.

    Korean companies must file annual reports within 90 days of fiscal year end.

    Parameters:
        fiscal_year_end: Fiscal year end date (usually Dec 31)
        lag_months: Reporting lag in months (default: 3)

    Returns:
        Date when data is publicly available

    Example:
        >>> get_fiscal_data_available_date('2023-12-31')
        '2024-03-31'  # FY2023 data available after March 2024
    """
    fy_end = pd.to_datetime(fiscal_year_end)
    available_date = fy_end + pd.DateOffset(months=lag_months)
    return available_date.strftime('%Y-%m-%d')


def get_latest_available_fiscal_year(
    reference_date: str,
    lag_months: int = 3
) -> int:
    """
    Get the most recent fiscal year with available data.

    Parameters:
        reference_date: Current date
        lag_months: Reporting lag in months

    Returns:
        Fiscal year (e.g., 2023)

    Example:
        >>> get_latest_available_fiscal_year('2024-06-15')
        2023  # FY2023 data is available
        >>> get_latest_available_fiscal_year('2024-02-15')
        2022  # FY2023 not yet available, use FY2022
    """
    ref = pd.to_datetime(reference_date)

    # Go back by lag months to find the latest FY end
    cutoff = ref - pd.DateOffset(months=lag_months)

    # Return the year of that cutoff date
    return cutoff.year
