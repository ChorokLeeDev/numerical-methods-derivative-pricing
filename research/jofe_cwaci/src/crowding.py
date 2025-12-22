"""
Crowding Signal Computation

Simple crowding proxy based on trailing absolute returns.

Note: This proxy may capture momentum effects. We acknowledge this limitation
and include momentum controls in our robustness analysis.

Author: Chorok Lee (KAIST)
Date: December 2024
"""

import numpy as np
import pandas as pd
from typing import Optional


def compute_crowding_proxy(returns: pd.Series, window: int = 12) -> pd.Series:
    """
    Compute crowding proxy from factor returns.

    Proxy: Rolling absolute return normalized by historical median.

    C(t) = |Rolling_Return(t-window:t)| / Median(Historical)

    Intuition: High trailing absolute returns → more attention → more crowding

    Parameters
    ----------
    returns : pd.Series
        Factor returns (monthly)
    window : int
        Lookback window in months (default: 12)

    Returns
    -------
    pd.Series
        Crowding proxy (higher = more crowded)

    Note
    ----
    This proxy is admittedly simple and may capture momentum effects.
    See robustness section for momentum-controlled analysis.
    """
    # Rolling sum of returns (momentum-like)
    rolling_return = returns.rolling(window).sum()

    # Take absolute value (captures attention regardless of direction)
    abs_rolling = np.abs(rolling_return)

    # Normalize by expanding median
    median_return = abs_rolling.expanding().median()

    # Crowding proxy
    crowding = abs_rolling / median_return.replace(0, np.nan)

    return crowding


def compute_volatility_crowding(returns: pd.Series, window: int = 12) -> pd.Series:
    """
    Alternative crowding proxy based on realized volatility.

    High volatility periods often correspond to crowded trades unwinding.

    Parameters
    ----------
    returns : pd.Series
        Factor returns (monthly)
    window : int
        Lookback window in months

    Returns
    -------
    pd.Series
        Volatility-based crowding proxy
    """
    rolling_vol = returns.rolling(window).std()
    median_vol = rolling_vol.expanding().median()

    return rolling_vol / median_vol.replace(0, np.nan)


def compute_correlation_crowding(returns: pd.Series,
                                  market_returns: pd.Series,
                                  window: int = 12) -> pd.Series:
    """
    Crowding proxy based on correlation with market.

    High correlation with market → factor is crowded (everyone in same trade)

    Parameters
    ----------
    returns : pd.Series
        Factor returns
    market_returns : pd.Series
        Market returns (e.g., Mkt-RF)
    window : int
        Rolling window

    Returns
    -------
    pd.Series
        Correlation-based crowding proxy
    """
    rolling_corr = returns.rolling(window).corr(market_returns)

    # Transform to (0, inf) range
    # Higher absolute correlation → more crowding
    return np.abs(rolling_corr)


def classify_crowding_regime(crowding: pd.Series,
                              threshold: Optional[float] = None) -> pd.Series:
    """
    Classify periods into high/low crowding regimes.

    Parameters
    ----------
    crowding : pd.Series
        Crowding signal
    threshold : float, optional
        Classification threshold. Default: median.

    Returns
    -------
    pd.Series
        Boolean series (True = high crowding)
    """
    if threshold is None:
        threshold = crowding.median()

    return crowding > threshold


if __name__ == '__main__':
    # Test
    np.random.seed(42)
    n = 500
    returns = pd.Series(np.random.randn(n) * 0.05)

    crowding = compute_crowding_proxy(returns)
    print(f"Crowding proxy stats:")
    print(f"  Mean: {crowding.mean():.3f}")
    print(f"  Std: {crowding.std():.3f}")
    print(f"  NaN count: {crowding.isna().sum()}")
