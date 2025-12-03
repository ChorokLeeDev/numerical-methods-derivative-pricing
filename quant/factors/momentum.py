"""
Momentum factor implementation.

12-1 month momentum: Cumulative return over months [-12, -2], skipping the most recent month.
Based on Jegadeesh & Titman (1993).
"""

import numpy as np
import pandas as pd


def calculate_momentum_12_1(
    returns: pd.DataFrame,
    reference_date: str,
    lookback_months: int = 12,
    skip_months: int = 1
) -> pd.Series:
    """
    Calculate 12-1 month momentum factor.

    Algorithm:
    1. Look back `lookback_months` from reference_date
    2. Skip the most recent `skip_months` (to avoid short-term reversal)
    3. Calculate cumulative return for the formation period

    Parameters:
        returns: DataFrame with monthly returns (DatetimeIndex, ticker columns)
        reference_date: YYYY-MM-DD format
        lookback_months: How far back to look (default: 12)
        skip_months: Most recent months to skip (default: 1)

    Returns:
        Series indexed by ticker with momentum scores (cumulative returns)

    Example:
        >>> mom = calculate_momentum_12_1(monthly_returns, '2024-06-30')
        >>> mom.head()
        005930    0.234   # Samsung gained 23.4% over formation period
        000660   -0.052   # SK Hynix lost 5.2%
    """
    ref_date = pd.to_datetime(reference_date)

    # Formation period: -12 months to -2 months (skip most recent month)
    end_date = ref_date - pd.DateOffset(months=skip_months)
    start_date = ref_date - pd.DateOffset(months=lookback_months)

    # Filter returns to formation period
    mask = (returns.index >= start_date) & (returns.index <= end_date)
    formation_returns = returns.loc[mask]

    if len(formation_returns) == 0:
        return pd.Series(dtype=float)

    # Calculate cumulative return: (1+r1)*(1+r2)*...*(1+rn) - 1
    cumulative = (1 + formation_returns).prod() - 1

    return cumulative


def calculate_momentum_simple(
    prices: pd.DataFrame,
    reference_date: str,
    lookback_days: int = 252,
    skip_days: int = 21
) -> pd.Series:
    """
    Calculate simple price momentum (using daily prices).

    Parameters:
        prices: DataFrame with daily prices (DatetimeIndex, ticker columns)
        reference_date: YYYY-MM-DD format
        lookback_days: Trading days to look back (default: 252 = 1 year)
        skip_days: Recent days to skip (default: 21 = 1 month)

    Returns:
        Series indexed by ticker with momentum scores
    """
    ref_date = pd.to_datetime(reference_date)

    # Filter to data before reference date
    prices = prices.loc[:ref_date]

    if len(prices) < lookback_days:
        return pd.Series(dtype=float)

    # Get prices at formation period boundaries
    current_idx = -skip_days - 1 if skip_days > 0 else -1
    start_idx = -lookback_days

    try:
        price_end = prices.iloc[current_idx]
        price_start = prices.iloc[start_idx]

        momentum = (price_end / price_start) - 1
        return momentum

    except IndexError:
        return pd.Series(dtype=float)


def rank_by_quantile(
    scores: pd.Series,
    n_quantiles: int = 5,
    ascending: bool = True
) -> pd.Series:
    """
    Rank stocks by scores into quantiles.

    Parameters:
        scores: Factor scores by ticker
        n_quantiles: Number of quantiles (default: 5 = quintiles)
        ascending: If True, low scores get low ranks (default: True)

    Returns:
        Series with quantile labels (1=lowest, n=highest)

    Example:
        >>> ranks = rank_by_quantile(momentum_scores, n_quantiles=5)
        >>> ranks.value_counts()
        1    40  # Bottom quintile
        2    40
        3    40
        4    40
        5    40  # Top quintile
    """
    # Drop NaN values
    valid_scores = scores.dropna()

    if len(valid_scores) == 0:
        return pd.Series(dtype=int)

    # Use pd.qcut for quantile-based ranking
    # labels=False gives 0-indexed, so we add 1
    try:
        quantiles = pd.qcut(
            valid_scores.rank(method='first'),
            q=n_quantiles,
            labels=range(1, n_quantiles + 1)
        )
        return quantiles.astype(int)
    except ValueError:
        # Fallback for edge cases with too few unique values
        return pd.Series(1, index=valid_scores.index)


def standardize_factor(scores: pd.Series) -> pd.Series:
    """
    Standardize factor scores to z-scores (mean=0, std=1).

    Parameters:
        scores: Raw factor scores

    Returns:
        Series with standardized scores
    """
    valid = scores.dropna()
    if len(valid) == 0 or valid.std() == 0:
        return scores * 0  # Return zeros

    return (scores - valid.mean()) / valid.std()


def winsorize_factor(
    scores: pd.Series,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99
) -> pd.Series:
    """
    Winsorize extreme values to reduce outlier impact.

    Parameters:
        scores: Factor scores
        lower_percentile: Lower bound percentile (default: 1%)
        upper_percentile: Upper bound percentile (default: 99%)

    Returns:
        Series with winsorized scores
    """
    lower = scores.quantile(lower_percentile)
    upper = scores.quantile(upper_percentile)

    return scores.clip(lower=lower, upper=upper)
