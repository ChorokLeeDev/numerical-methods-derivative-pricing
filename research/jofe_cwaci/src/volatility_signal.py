"""
Volatility Signal Computation

Honest volatility proxy for signal-adaptive conformal prediction.
We explicitly acknowledge this is a volatility signal, not a crowding measure.

Author: Chorok Lee (KAIST)
Date: December 2024
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy import stats


def compute_volatility_signal(returns: pd.Series, window: int = 12) -> pd.Series:
    """
    Compute volatility signal from factor returns.

    Uses realized volatility as the primary signal.

    Parameters
    ----------
    returns : pd.Series
        Factor returns (monthly)
    window : int
        Lookback window in months (default: 12)

    Returns
    -------
    pd.Series
        Volatility signal normalized by historical median
    """
    rolling_vol = returns.rolling(window).std()
    median_vol = rolling_vol.expanding().median()

    return rolling_vol / median_vol.replace(0, np.nan)


def compute_absolute_return_signal(returns: pd.Series, window: int = 12) -> pd.Series:
    """
    Alternative signal based on trailing absolute returns.

    This is a momentum-magnitude measure that correlates with volatility.
    We include it for robustness but acknowledge it's essentially a volatility proxy.

    Parameters
    ----------
    returns : pd.Series
        Factor returns (monthly)
    window : int
        Lookback window in months

    Returns
    -------
    pd.Series
        Absolute return signal normalized by historical median
    """
    rolling_return = returns.rolling(window).sum()
    abs_rolling = np.abs(rolling_return)
    median_return = abs_rolling.expanding().median()

    return abs_rolling / median_return.replace(0, np.nan)


def analyze_signal_correlation(returns: pd.Series, window: int = 12) -> dict:
    """
    Analyze correlation between different signals and realized volatility.

    This provides transparency about what our signals actually measure.

    Parameters
    ----------
    returns : pd.Series
        Factor returns
    window : int
        Rolling window

    Returns
    -------
    dict with correlation statistics
    """
    # Compute signals
    vol_signal = compute_volatility_signal(returns, window)
    abs_ret_signal = compute_absolute_return_signal(returns, window)

    # Compute forward realized volatility (what we're trying to predict)
    forward_vol = returns.rolling(window).std().shift(-window)

    # Valid observations
    valid = vol_signal.notna() & abs_ret_signal.notna() & forward_vol.notna()

    results = {
        'vol_signal_vs_forward_vol': np.corrcoef(
            vol_signal[valid], forward_vol[valid]
        )[0, 1],
        'abs_ret_vs_forward_vol': np.corrcoef(
            abs_ret_signal[valid], forward_vol[valid]
        )[0, 1],
        'vol_signal_vs_abs_ret': np.corrcoef(
            vol_signal[valid], abs_ret_signal[valid]
        )[0, 1],
        'n_obs': valid.sum()
    }

    return results


def classify_volatility_regime(signal: pd.Series,
                                threshold: Optional[float] = None) -> pd.Series:
    """
    Classify periods into high/low volatility regimes.

    Parameters
    ----------
    signal : pd.Series
        Volatility signal
    threshold : float, optional
        Classification threshold. Default: median.

    Returns
    -------
    pd.Series
        Boolean series (True = high volatility)
    """
    if threshold is None:
        threshold = signal.median()

    return signal > threshold


def compute_coverage_standard_error(coverage: float, n: int) -> float:
    """
    Compute standard error of coverage estimate.

    SE = sqrt(p(1-p)/n)

    Parameters
    ----------
    coverage : float
        Estimated coverage rate
    n : int
        Number of observations

    Returns
    -------
    float
        Standard error
    """
    return np.sqrt(coverage * (1 - coverage) / n)


def test_coverage_difference(cov1: float, n1: int,
                              cov2: float, n2: int) -> Tuple[float, float]:
    """
    Two-proportion z-test for coverage difference.

    H0: cov1 = cov2

    Parameters
    ----------
    cov1, n1 : coverage and sample size for method 1
    cov2, n2 : coverage and sample size for method 2

    Returns
    -------
    z_stat, p_value
    """
    # Pooled proportion
    p_pool = (cov1 * n1 + cov2 * n2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

    if se == 0:
        return 0.0, 1.0

    z = (cov1 - cov2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p_value


if __name__ == '__main__':
    # Test
    np.random.seed(42)
    n = 500
    returns = pd.Series(np.random.randn(n) * 0.05)

    vol_signal = compute_volatility_signal(returns)
    print(f"Volatility signal stats:")
    print(f"  Mean: {vol_signal.mean():.3f}")
    print(f"  Std: {vol_signal.std():.3f}")
    print(f"  NaN count: {vol_signal.isna().sum()}")

    # Analyze correlations
    corr = analyze_signal_correlation(returns)
    print(f"\nSignal correlations:")
    for k, v in corr.items():
        print(f"  {k}: {v:.3f}")
