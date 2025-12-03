"""
Feature engineering for ML-based factor prediction.

Converts raw factor data into ML-ready features with:
- Lagged features (past factor values)
- Market regime indicators
- Rolling statistics
"""

import numpy as np
import pandas as pd
from typing import Optional


def prepare_features(
    factor_history: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    target_horizon: int = 1
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for ML training.

    Parameters:
        factor_history: Dict of {factor_name: DataFrame with (date, ticker) as columns}
        returns: Forward returns DataFrame (date index, ticker columns)
        target_horizon: Months ahead to predict

    Returns:
        X: Feature DataFrame (index: (date, ticker))
        y: Target Series (forward returns)

    Example:
        >>> X, y = prepare_features(
        ...     {'momentum': mom_df, 'value': val_df},
        ...     forward_returns,
        ...     target_horizon=1
        ... )
    """
    # Convert factor history to long format and merge
    feature_frames = []

    for factor_name, factor_df in factor_history.items():
        # factor_df has dates as index, tickers as columns
        long = factor_df.stack().reset_index()
        long.columns = ['date', 'ticker', factor_name]
        feature_frames.append(long.set_index(['date', 'ticker']))

    if len(feature_frames) == 0:
        return pd.DataFrame(), pd.Series(dtype=float)

    # Merge all factors
    features = pd.concat(feature_frames, axis=1)

    # Align returns (also convert to long format)
    returns_long = returns.stack().reset_index()
    returns_long.columns = ['date', 'ticker', 'forward_return']
    returns_long = returns_long.set_index(['date', 'ticker'])

    # Shift target by horizon (we predict next month's return)
    # Actually, returns should already be forward-looking
    # So we just join features_t with returns_t

    # Merge features and target
    data = features.join(returns_long, how='inner')

    X = data.drop(columns=['forward_return'])
    y = data['forward_return']

    return X, y


def create_lagged_features(
    factor_df: pd.DataFrame,
    lags: list[int] = [1, 2, 3, 6, 12]
) -> pd.DataFrame:
    """
    Create lagged factor features.

    For each factor column, creates lagged versions.

    Parameters:
        factor_df: DataFrame with factors (date index, ticker or factor columns)
        lags: List of lag periods to create

    Returns:
        DataFrame with original and lagged features
    """
    result = factor_df.copy()

    for col in factor_df.columns:
        for lag in lags:
            result[f'{col}_lag{lag}'] = factor_df[col].shift(lag)

    return result


def add_market_features(
    features: pd.DataFrame,
    market_returns: pd.Series
) -> pd.DataFrame:
    """
    Add market regime features.

    Parameters:
        features: Feature DataFrame (date in index level 0)
        market_returns: Series of market returns (date index)

    Returns:
        Features with market regime indicators
    """
    result = features.copy()

    # Get unique dates from features
    dates = features.index.get_level_values('date').unique()

    # Calculate market features
    market_features = pd.DataFrame(index=dates)

    # Market return
    market_features['market_ret'] = market_returns.reindex(dates)

    # Market momentum (6-month cumulative)
    market_features['market_mom_6m'] = market_returns.rolling(6).sum().reindex(dates)

    # Market volatility (rolling 6-month)
    market_features['market_vol_6m'] = market_returns.rolling(6).std().reindex(dates)

    # Bull/bear regime (1 if positive momentum, 0 otherwise)
    market_features['bull_regime'] = (market_features['market_mom_6m'] > 0).astype(float)

    # Merge back to features
    result = result.reset_index()
    result = result.merge(market_features, left_on='date', right_index=True, how='left')
    result = result.set_index(['date', 'ticker'])

    return result


def add_rolling_stats(
    features: pd.DataFrame,
    factor_history: dict[str, pd.DataFrame],
    windows: list[int] = [3, 6, 12]
) -> pd.DataFrame:
    """
    Add rolling statistics of factors.

    Parameters:
        features: Feature DataFrame
        factor_history: Original factor DataFrames
        windows: Rolling window sizes

    Returns:
        Features with rolling statistics
    """
    result = features.copy()

    for factor_name, factor_df in factor_history.items():
        # Calculate cross-sectional stats at each date
        for window in windows:
            # Rolling mean of factor (time-series)
            roll_mean = factor_df.rolling(window).mean()
            roll_std = factor_df.rolling(window).std()

            # Convert to long format
            roll_mean_long = roll_mean.stack()
            roll_mean_long.index.names = ['date', 'ticker']

            roll_std_long = roll_std.stack()
            roll_std_long.index.names = ['date', 'ticker']

            result[f'{factor_name}_rollmean_{window}'] = roll_mean_long.reindex(result.index)
            result[f'{factor_name}_rollstd_{window}'] = roll_std_long.reindex(result.index)

    return result


def create_interaction_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction terms between factors.

    Parameters:
        features: Feature DataFrame with factor columns

    Returns:
        Features with interaction terms
    """
    result = features.copy()

    # Get base factor columns (not lagged or rolling)
    base_factors = [c for c in features.columns
                    if not any(x in c for x in ['lag', 'roll', 'market', 'regime'])]

    # Create pairwise interactions
    for i, f1 in enumerate(base_factors):
        for f2 in base_factors[i+1:]:
            result[f'{f1}_x_{f2}'] = features[f1] * features[f2]

    return result


def prepare_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.7,
    gap_months: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-series aware train/test split.

    Parameters:
        X: Features
        y: Target
        train_ratio: Ratio for training set
        gap_months: Gap between train and test to avoid leakage

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Get unique dates
    dates = X.index.get_level_values('date').unique().sort_values()

    n_dates = len(dates)
    train_end_idx = int(n_dates * train_ratio)

    train_dates = dates[:train_end_idx]
    test_dates = dates[train_end_idx + gap_months:]

    # Filter by dates
    train_mask = X.index.get_level_values('date').isin(train_dates)
    test_mask = X.index.get_level_values('date').isin(test_dates)

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    return X_train, X_test, y_train, y_test
