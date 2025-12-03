"""
GARCH Models for Time-Varying Volatility

# The Problem

Traditional volatility estimation assumes constant volatility:
    σ = std(returns)

But volatility clusters! High volatility follows high volatility.

# GARCH Solution (Bollerslev, 1986)

GARCH(1,1) models volatility as a process:

    σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}

where:
    ω = long-run variance weight
    α = shock persistence (how much yesterday's return matters)
    β = volatility persistence (how much yesterday's vol matters)
    ε_{t-1} = yesterday's return shock

# Intuition

- β close to 1: Volatility is very persistent (changes slowly)
- α large: Big shocks have big impact on future volatility
- α + β < 1: Process is stationary (vol reverts to long-run level)

# Industry Usage

- Every trading desk uses GARCH for volatility forecasting
- Options pricing: Better IV surface modeling
- Risk management: Time-varying VaR
- Portfolio optimization: Dynamic covariance updates
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False


@dataclass
class GARCHResult:
    """Results from GARCH estimation."""
    omega: float       # Long-run variance weight
    alpha: float       # Shock persistence
    beta: float        # Volatility persistence
    long_run_var: float  # ω / (1 - α - β)
    conditional_vol: np.ndarray  # Time series of conditional volatility
    forecasts: Optional[np.ndarray] = None


def fit_garch(
    returns: np.ndarray,
    p: int = 1,
    q: int = 1,
    dist: str = 'normal'
) -> GARCHResult:
    """
    Fit GARCH(p,q) model to returns.

    Parameters:
        returns: Array of returns (not percentage, e.g., 0.01 for 1%)
        p: Number of GARCH terms (volatility lags)
        q: Number of ARCH terms (return shock lags)
        dist: Distribution ('normal', 't', 'skewt')

    Returns:
        GARCHResult with estimated parameters

    Example:
        >>> result = fit_garch(daily_returns)
        >>> print(f"Long-run volatility: {np.sqrt(result.long_run_var) * np.sqrt(252):.1%}")
    """
    if not HAS_ARCH:
        return _fit_garch_simple(returns)

    # Scale returns to percentage for arch library
    returns_pct = returns * 100

    model = arch_model(returns_pct, vol='Garch', p=p, q=q, dist=dist)
    result = model.fit(disp='off')

    # Extract parameters (scaled back)
    omega = result.params['omega'] / 10000  # Convert back from pct²
    alpha = result.params.get('alpha[1]', 0)
    beta = result.params.get('beta[1]', 0)

    # Conditional volatility (scaled back)
    cond_vol = result.conditional_volatility / 100

    # Long-run variance
    persistence = alpha + beta
    if persistence < 1:
        long_run_var = omega / (1 - persistence)
    else:
        long_run_var = np.var(returns)

    return GARCHResult(
        omega=omega,
        alpha=alpha,
        beta=beta,
        long_run_var=long_run_var,
        conditional_vol=cond_vol.values
    )


def _fit_garch_simple(returns: np.ndarray) -> GARCHResult:
    """
    Simple GARCH(1,1) estimation without arch library.

    Uses moment matching for initial estimates.
    """
    n = len(returns)
    var_r = np.var(returns)

    # Initial parameter guesses (typical values)
    omega = var_r * 0.05  # 5% weight on long-run
    alpha = 0.10          # 10% shock persistence
    beta = 0.85           # 85% vol persistence

    # Simple variance targeting
    long_run_var = var_r

    # Calculate conditional variance series
    cond_var = np.zeros(n)
    cond_var[0] = var_r

    for t in range(1, n):
        cond_var[t] = omega + alpha * returns[t-1]**2 + beta * cond_var[t-1]

    return GARCHResult(
        omega=omega,
        alpha=alpha,
        beta=beta,
        long_run_var=long_run_var,
        conditional_vol=np.sqrt(cond_var)
    )


def forecast_volatility(
    garch_result: GARCHResult,
    returns: np.ndarray,
    horizon: int = 1
) -> np.ndarray:
    """
    Forecast future volatility using GARCH model.

    Parameters:
        garch_result: Fitted GARCH model
        returns: Historical returns
        horizon: Number of periods to forecast

    Returns:
        Array of forecasted volatilities
    """
    omega = garch_result.omega
    alpha = garch_result.alpha
    beta = garch_result.beta

    # Last conditional variance
    last_vol = garch_result.conditional_vol[-1]
    last_var = last_vol ** 2

    # Last shock
    last_shock = returns[-1] ** 2

    forecasts = np.zeros(horizon)

    # One-step ahead
    forecasts[0] = omega + alpha * last_shock + beta * last_var

    # Multi-step ahead (mean-reverts to long-run var)
    for h in range(1, horizon):
        # E[σ²_{t+h}] converges to long-run variance
        persistence = alpha + beta
        forecasts[h] = garch_result.long_run_var + \
                       (persistence ** h) * (forecasts[0] - garch_result.long_run_var)

    return np.sqrt(forecasts)


def dynamic_covariance(
    returns: pd.DataFrame,
    method: str = 'dcc'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate time-varying covariance using DCC-GARCH.

    Dynamic Conditional Correlation (Engle, 2002):
    1. Fit univariate GARCH to each asset
    2. Standardize returns by conditional vol
    3. Model time-varying correlation

    Parameters:
        returns: DataFrame of returns (T × N)
        method: 'dcc' or 'ewma'

    Returns:
        (conditional_vols, conditional_corrs)
        - conditional_vols: T × N array
        - conditional_corrs: T × N × N array (correlation matrices)
    """
    T, N = returns.shape

    # Fit GARCH to each series
    cond_vols = np.zeros((T, N))
    standardized = np.zeros((T, N))

    for i, col in enumerate(returns.columns):
        result = fit_garch(returns[col].values)
        cond_vols[:, i] = result.conditional_vol
        standardized[:, i] = returns[col].values / result.conditional_vol

    if method == 'ewma':
        # Exponentially weighted correlation
        lambda_param = 0.94
        cond_corrs = np.zeros((T, N, N))
        cond_corrs[0] = np.corrcoef(standardized[:30].T) if T > 30 else np.eye(N)

        for t in range(1, T):
            outer = np.outer(standardized[t], standardized[t])
            cond_corrs[t] = lambda_param * cond_corrs[t-1] + (1 - lambda_param) * outer
            # Ensure it's a correlation matrix
            diag = np.sqrt(np.diag(cond_corrs[t]))
            cond_corrs[t] = cond_corrs[t] / np.outer(diag, diag)

    else:  # Simple rolling correlation
        window = 60
        cond_corrs = np.zeros((T, N, N))
        for t in range(window, T):
            cond_corrs[t] = np.corrcoef(standardized[t-window:t].T)
        # Fill initial period
        for t in range(window):
            cond_corrs[t] = cond_corrs[window]

    return cond_vols, cond_corrs


def garch_var(
    returns: np.ndarray,
    alpha: float = 0.05,
    horizon: int = 1
) -> float:
    """
    Calculate VaR using GARCH volatility forecast.

    Parameters:
        returns: Historical returns
        alpha: VaR confidence level
        horizon: Forecast horizon

    Returns:
        VaR estimate (positive number = loss)
    """
    from scipy.stats import norm

    result = fit_garch(returns)
    forecast_vol = forecast_volatility(result, returns, horizon)

    # Parametric VaR with GARCH volatility
    z = norm.ppf(alpha)
    mean_return = np.mean(returns) * horizon

    var = -(mean_return + z * forecast_vol[0] * np.sqrt(horizon))
    return var


# Convenience function for pandas
def add_garch_volatility(df: pd.DataFrame, return_col: str) -> pd.DataFrame:
    """
    Add GARCH conditional volatility column to DataFrame.

    Parameters:
        df: DataFrame with returns
        return_col: Name of return column

    Returns:
        DataFrame with 'garch_vol' column added
    """
    result = fit_garch(df[return_col].values)
    df = df.copy()
    df['garch_vol'] = result.conditional_vol
    df['garch_vol_ann'] = result.conditional_vol * np.sqrt(252)  # Annualized
    return df
