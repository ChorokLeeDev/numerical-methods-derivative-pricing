"""
Composite factor utilities.

Combine multiple factors into a single composite score.
"""

import numpy as np
import pandas as pd

from .momentum import standardize_factor, winsorize_factor


def combine_factors(
    factor_scores: dict[str, pd.Series],
    weights: dict[str, float] = None
) -> pd.Series:
    """
    Combine multiple factor scores into composite score.

    Parameters:
        factor_scores: Dict mapping factor name to scores
        weights: Dict mapping factor name to weight (default: equal)

    Returns:
        Series with composite z-scores

    Example:
        >>> composite = combine_factors({
        ...     'momentum': mom_scores,
        ...     'value': bm_scores,
        ...     'quality': roe_scores
        ... })
    """
    if len(factor_scores) == 0:
        return pd.Series(dtype=float)

    # Default to equal weights
    if weights is None:
        weights = {name: 1.0 / len(factor_scores) for name in factor_scores}

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # Standardize each factor
    standardized = {}
    for name, scores in factor_scores.items():
        # Winsorize first to reduce outlier impact
        winsorized = winsorize_factor(scores)
        standardized[name] = standardize_factor(winsorized)

    # Combine using weighted average
    df = pd.DataFrame(standardized)

    # Get intersection of all indices (stocks with all factor scores)
    valid_mask = df.notna().all(axis=1)

    composite = pd.Series(0.0, index=df.index)
    for name, z_scores in standardized.items():
        weight = weights.get(name, 0)
        composite = composite.add(z_scores * weight, fill_value=0)

    # Only keep stocks with all factors
    composite = composite[valid_mask]

    return composite


def create_factor_dataframe(
    factor_scores: dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Create a DataFrame from multiple factor scores.

    Parameters:
        factor_scores: Dict mapping factor name to scores

    Returns:
        DataFrame with tickers as index and factors as columns
    """
    return pd.DataFrame(factor_scores)


def calculate_factor_correlations(
    factor_scores: dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Calculate correlations between factors.

    Useful for understanding factor redundancy.

    Parameters:
        factor_scores: Dict mapping factor name to scores

    Returns:
        Correlation matrix as DataFrame
    """
    df = create_factor_dataframe(factor_scores)
    return df.corr()


def calculate_factor_stats(
    factor_scores: dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Calculate summary statistics for each factor.

    Parameters:
        factor_scores: Dict mapping factor name to scores

    Returns:
        DataFrame with stats (mean, std, min, max, count)
    """
    stats = []
    for name, scores in factor_scores.items():
        valid = scores.dropna()
        stats.append({
            'factor': name,
            'count': len(valid),
            'mean': valid.mean(),
            'std': valid.std(),
            'min': valid.min(),
            'max': valid.max(),
            'median': valid.median()
        })

    return pd.DataFrame(stats).set_index('factor')
