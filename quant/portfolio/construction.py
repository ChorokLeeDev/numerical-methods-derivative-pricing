"""
Portfolio construction utilities.

Long-short portfolio construction based on factor quintiles.
"""

import numpy as np
import pandas as pd


def construct_long_short_portfolio(
    factor_scores: pd.Series,
    n_quantiles: int = 5,
    long_quantile: int = None,
    short_quantile: int = 1
) -> tuple[pd.Series, pd.Series]:
    """
    Construct long-short portfolio from factor scores.

    Parameters:
        factor_scores: Factor scores by ticker
        n_quantiles: Number of quantiles (default: 5 = quintiles)
        long_quantile: Which quantile to go long (default: highest)
        short_quantile: Which quantile to go short (default: 1 = lowest)

    Returns:
        Tuple of (long_weights, short_weights) as Series

    Example:
        >>> long_w, short_w = construct_long_short_portfolio(momentum_scores)
        >>> long_w.sum()  # Weights sum to 1
        1.0
        >>> short_w.sum()
        1.0
    """
    if long_quantile is None:
        long_quantile = n_quantiles

    # Remove NaN values
    valid_scores = factor_scores.dropna()

    if len(valid_scores) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Rank into quantiles
    try:
        quantiles = pd.qcut(
            valid_scores.rank(method='first'),
            q=n_quantiles,
            labels=range(1, n_quantiles + 1)
        )
    except ValueError:
        # Fallback for edge cases
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Long portfolio (top quantile)
    long_mask = quantiles == long_quantile
    long_tickers = valid_scores.index[long_mask].tolist()

    if len(long_tickers) > 0:
        long_weights = pd.Series(1.0 / len(long_tickers), index=long_tickers)
    else:
        long_weights = pd.Series(dtype=float)

    # Short portfolio (bottom quantile)
    short_mask = quantiles == short_quantile
    short_tickers = valid_scores.index[short_mask].tolist()

    if len(short_tickers) > 0:
        short_weights = pd.Series(1.0 / len(short_tickers), index=short_tickers)
    else:
        short_weights = pd.Series(dtype=float)

    return long_weights, short_weights


def construct_long_only_portfolio(
    factor_scores: pd.Series,
    n_quantiles: int = 5,
    top_quantiles: int = 1
) -> pd.Series:
    """
    Construct long-only portfolio from top factor quantile(s).

    Useful when short-selling is restricted (Korean market).

    Parameters:
        factor_scores: Factor scores by ticker
        n_quantiles: Number of quantiles
        top_quantiles: How many top quantiles to include (default: 1)

    Returns:
        Portfolio weights as Series
    """
    valid_scores = factor_scores.dropna()

    if len(valid_scores) == 0:
        return pd.Series(dtype=float)

    # Rank into quantiles
    try:
        quantiles = pd.qcut(
            valid_scores.rank(method='first'),
            q=n_quantiles,
            labels=range(1, n_quantiles + 1)
        )
    except ValueError:
        return pd.Series(dtype=float)

    # Select top quantile(s)
    min_quantile = n_quantiles - top_quantiles + 1
    selected_mask = quantiles >= min_quantile
    selected_tickers = valid_scores.index[selected_mask].tolist()

    if len(selected_tickers) > 0:
        return pd.Series(1.0 / len(selected_tickers), index=selected_tickers)

    return pd.Series(dtype=float)


def calculate_portfolio_weights(
    tickers: list[str],
    weighting: str = 'equal',
    market_caps: pd.Series = None
) -> pd.Series:
    """
    Calculate portfolio weights for a list of tickers.

    Parameters:
        tickers: List of stock codes
        weighting: 'equal' or 'value_weighted'
        market_caps: Required if weighting='value_weighted'

    Returns:
        Series indexed by ticker with weights summing to 1.0
    """
    if len(tickers) == 0:
        return pd.Series(dtype=float)

    if weighting == 'equal':
        return pd.Series(1.0 / len(tickers), index=tickers)

    elif weighting == 'value_weighted':
        if market_caps is None:
            raise ValueError("market_caps required for value weighting")

        caps = market_caps.reindex(tickers).dropna()
        if len(caps) == 0:
            return pd.Series(dtype=float)

        return caps / caps.sum()

    else:
        raise ValueError(f"Unknown weighting: {weighting}")


def calculate_portfolio_return(
    long_weights: pd.Series,
    short_weights: pd.Series,
    returns: pd.Series,
    leverage: float = 1.0
) -> float:
    """
    Calculate long-short portfolio return for a period.

    Portfolio Return = Long Return - Short Return

    Parameters:
        long_weights: Weights for long positions
        short_weights: Weights for short positions
        returns: Period returns for all tickers
        leverage: Leverage multiplier (default: 1x)

    Returns:
        Portfolio return as decimal
    """
    # Calculate long leg return
    long_return = 0.0
    if len(long_weights) > 0:
        long_returns = returns.reindex(long_weights.index).fillna(0)
        long_return = (long_weights * long_returns).sum()

    # Calculate short leg return
    short_return = 0.0
    if len(short_weights) > 0:
        short_returns = returns.reindex(short_weights.index).fillna(0)
        short_return = (short_weights * short_returns).sum()

    # Long-short return
    portfolio_return = (long_return - short_return) * leverage

    return portfolio_return


def get_quantile_portfolios(
    factor_scores: pd.Series,
    n_quantiles: int = 5
) -> dict[int, list[str]]:
    """
    Get tickers for each quantile portfolio.

    Useful for quintile spread analysis.

    Parameters:
        factor_scores: Factor scores by ticker
        n_quantiles: Number of quantiles

    Returns:
        Dict mapping quantile number to list of tickers
    """
    valid_scores = factor_scores.dropna()

    if len(valid_scores) == 0:
        return {}

    try:
        quantiles = pd.qcut(
            valid_scores.rank(method='first'),
            q=n_quantiles,
            labels=range(1, n_quantiles + 1)
        )
    except ValueError:
        return {}

    portfolios = {}
    for q in range(1, n_quantiles + 1):
        mask = quantiles == q
        portfolios[q] = valid_scores.index[mask].tolist()

    return portfolios
