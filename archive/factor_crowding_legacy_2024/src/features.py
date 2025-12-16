"""
Feature Engineering for ML-Based Crowding Detection

Features for predicting:
1. Crowding regime (high/low)
2. Tail risk (crash probability)

Feature Categories:
- Return-based: momentum, reversal signals
- Volatility-based: realized vol, vol regime
- Correlation-based: factor correlations, dispersion
- Cross-sectional: relative performance across factors
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature generation."""
    return_windows: List[int] = None  # Momentum lookbacks
    vol_windows: List[int] = None     # Volatility windows
    corr_window: int = 36             # Correlation window

    def __post_init__(self):
        if self.return_windows is None:
            self.return_windows = [1, 3, 6, 12]
        if self.vol_windows is None:
            self.vol_windows = [6, 12, 36]


class FeatureEngineer:
    """
    Generate features for ML-based crowding and tail risk prediction.

    Usage:
        fe = FeatureEngineer()
        features = fe.generate_all_features(factor_returns)
        X, y = fe.create_ml_dataset(features, factor_returns, target='crash')
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    # ================================================================
    # RETURN-BASED FEATURES
    # ================================================================

    def compute_return_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute return-based features for each factor.

        Features:
        - Rolling returns (1M, 3M, 6M, 12M)
        - Return rank (cross-sectional)
        - Return z-score (time-series)
        """
        features = {}

        for window in self.config.return_windows:
            # Rolling cumulative return
            roll_ret = returns.rolling(window).mean()
            for col in returns.columns:
                features[f'{col}_ret_{window}m'] = roll_ret[col]

            # Cross-sectional rank (0 to 1)
            roll_rank = roll_ret.rank(axis=1, pct=True)
            for col in returns.columns:
                features[f'{col}_rank_{window}m'] = roll_rank[col]

        # Time-series z-score (36-month expanding)
        for col in returns.columns:
            expanding_mean = returns[col].expanding(min_periods=36).mean()
            expanding_std = returns[col].expanding(min_periods=36).std()
            features[f'{col}_zscore'] = (returns[col] - expanding_mean) / expanding_std

        return pd.DataFrame(features)

    # ================================================================
    # VOLATILITY FEATURES
    # ================================================================

    def compute_volatility_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volatility-based features.

        Features:
        - Rolling volatility (6M, 12M, 36M)
        - Vol ratio (short/long)
        - Vol regime (high/low relative to history)
        """
        features = {}

        for window in self.config.vol_windows:
            roll_vol = returns.rolling(window).std() * np.sqrt(12)  # Annualized
            for col in returns.columns:
                features[f'{col}_vol_{window}m'] = roll_vol[col]

        # Vol ratio (short-term / long-term)
        short_vol = returns.rolling(6).std()
        long_vol = returns.rolling(36).std()
        vol_ratio = short_vol / long_vol
        for col in returns.columns:
            features[f'{col}_vol_ratio'] = vol_ratio[col]

        # Vol regime (current vs expanding percentile)
        for col in returns.columns:
            vol_12m = returns[col].rolling(12).std()
            vol_pct = vol_12m.expanding(min_periods=36).apply(
                lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 1 else 0.5
            )
            features[f'{col}_vol_regime'] = vol_pct

        return pd.DataFrame(features)

    # ================================================================
    # CORRELATION FEATURES
    # ================================================================

    def compute_correlation_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute correlation-based features.

        Features:
        - Pairwise correlations (rolling)
        - Average correlation (crowding proxy)
        - Correlation dispersion
        """
        features = {}
        window = self.config.corr_window

        # Rolling pairwise correlations
        factors = returns.columns.tolist()

        for i, f1 in enumerate(factors):
            for f2 in factors[i+1:]:
                corr = returns[f1].rolling(window).corr(returns[f2])
                features[f'corr_{f1}_{f2}'] = corr

        # Average correlation across all pairs
        corr_cols = [c for c in features.keys() if c.startswith('corr_')]
        if corr_cols:
            corr_df = pd.DataFrame({k: features[k] for k in corr_cols})
            features['avg_correlation'] = corr_df.mean(axis=1)
            features['corr_dispersion'] = corr_df.std(axis=1)

        # Factor-specific average correlation
        for factor in factors:
            factor_corrs = [c for c in corr_cols if factor in c]
            if factor_corrs:
                corr_df = pd.DataFrame({k: features[k] for k in factor_corrs})
                features[f'{factor}_avg_corr'] = corr_df.mean(axis=1)

        return pd.DataFrame(features)

    # ================================================================
    # CROWDING PROXY FEATURES
    # ================================================================

    def compute_crowding_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute direct crowding proxy features.

        Features:
        - Sharpe ratio deviation from trend
        - Performance clustering
        - Momentum concentration
        """
        features = {}

        # Rolling Sharpe ratio
        for col in returns.columns:
            roll_mean = returns[col].rolling(36).mean()
            roll_std = returns[col].rolling(36).std()
            sharpe = roll_mean / roll_std * np.sqrt(12)
            features[f'{col}_sharpe_36m'] = sharpe

            # Sharpe deviation from expanding mean (crowding signal)
            sharpe_trend = sharpe.expanding(min_periods=60).mean()
            features[f'{col}_sharpe_deviation'] = sharpe - sharpe_trend

        # Momentum concentration (HHI of return ranks)
        roll_ret_12m = returns.rolling(12).mean()
        ranks = roll_ret_12m.rank(axis=1, pct=True)
        # HHI = sum of squared shares
        hhi = (ranks ** 2).sum(axis=1)
        features['momentum_concentration'] = hhi

        # Performance dispersion
        features['return_dispersion'] = returns.rolling(12).mean().std(axis=1)

        return pd.DataFrame(features)

    # ================================================================
    # AGGREGATE FEATURE GENERATION
    # ================================================================

    def generate_all_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from factor returns.

        Args:
            returns: DataFrame of factor returns (index=date, columns=factors)

        Returns:
            DataFrame of features (index=date, columns=feature names)
        """
        print("Generating return features...")
        return_features = self.compute_return_features(returns)

        print("Generating volatility features...")
        vol_features = self.compute_volatility_features(returns)

        print("Generating correlation features...")
        corr_features = self.compute_correlation_features(returns)

        print("Generating crowding features...")
        crowd_features = self.compute_crowding_features(returns)

        # Combine all features
        all_features = pd.concat([
            return_features,
            vol_features,
            corr_features,
            crowd_features
        ], axis=1)

        print(f"Total features: {len(all_features.columns)}")

        return all_features

    # ================================================================
    # TARGET CREATION
    # ================================================================

    def create_crash_targets(
        self,
        returns: pd.DataFrame,
        threshold_pct: float = 0.10,
        horizon: int = 1,
    ) -> pd.DataFrame:
        """
        Create crash (tail risk) targets for each factor.

        Args:
            returns: Factor returns
            threshold_pct: Bottom percentile to define crash (default 10%)
            horizon: Forward horizon in months (default 1)

        Returns:
            DataFrame of binary crash indicators (1 = crash, 0 = no crash)
        """
        targets = {}

        for col in returns.columns:
            # Forward return
            fwd_ret = returns[col].shift(-horizon)

            # Expanding threshold (no lookahead)
            crash_threshold = returns[col].expanding(min_periods=60).quantile(threshold_pct)

            # Binary crash indicator
            is_crash = (fwd_ret < crash_threshold).astype(int)
            targets[f'{col}_crash'] = is_crash

        # Aggregate crash (any factor crashes)
        target_df = pd.DataFrame(targets)
        target_df['any_crash'] = (target_df.sum(axis=1) > 0).astype(int)

        return target_df

    def create_crowding_targets(
        self,
        returns: pd.DataFrame,
        sharpe_window: int = 36,
        threshold_pct: float = 0.25,
    ) -> pd.DataFrame:
        """
        Create crowding regime targets (for supervised learning).

        Args:
            returns: Factor returns
            sharpe_window: Window for Sharpe calculation
            threshold_pct: Bottom percentile = "crowded"

        Returns:
            DataFrame of crowding indicators
        """
        targets = {}

        for col in returns.columns:
            # Rolling Sharpe
            roll_mean = returns[col].rolling(sharpe_window).mean()
            roll_std = returns[col].rolling(sharpe_window).std()
            sharpe = roll_mean / roll_std * np.sqrt(12)

            # Expanding percentile (crowded = underperforming vs history)
            sharpe_pct = sharpe.expanding(min_periods=60).apply(
                lambda x: (x.iloc[-1] <= x[:-1]).mean() if len(x) > 1 else 0.5
            )

            # Binary: crowded if in bottom quartile
            is_crowded = (sharpe_pct < threshold_pct).astype(int)
            targets[f'{col}_crowded'] = is_crowded

        # Aggregate crowding
        target_df = pd.DataFrame(targets)
        target_df['high_crowding'] = (target_df.mean(axis=1) > 0.5).astype(int)

        return target_df

    # ================================================================
    # ML DATASET CREATION
    # ================================================================

    def create_ml_dataset(
        self,
        features: pd.DataFrame,
        returns: pd.DataFrame,
        target_type: str = 'crash',
        factor: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create aligned feature matrix and target vector.

        Args:
            features: Feature DataFrame
            returns: Factor returns (for target creation)
            target_type: 'crash' or 'crowding'
            factor: Specific factor or None for aggregate

        Returns:
            (X, y) tuple of features and targets
        """
        # Create targets
        if target_type == 'crash':
            targets = self.create_crash_targets(returns)
            if factor:
                y = targets[f'{factor}_crash']
            else:
                y = targets['any_crash']
        else:
            targets = self.create_crowding_targets(returns)
            if factor:
                y = targets[f'{factor}_crowded']
            else:
                y = targets['high_crowding']

        # Align features and targets
        common_idx = features.index.intersection(y.dropna().index)
        X = features.loc[common_idx]
        y = y.loc[common_idx]

        # Drop rows with NaN features
        valid_mask = ~X.isna().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        print(f"Dataset: {len(X)} samples, {len(X.columns)} features")
        print(f"Target '{target_type}' positive rate: {y.mean():.1%}")

        return X, y


class WalkForwardCV:
    """
    Walk-forward cross-validation for time series.

    No lookahead bias - always train on past, test on future.
    """

    def __init__(
        self,
        train_size: int = 120,  # 10 years training
        test_size: int = 12,    # 1 year test
        step_size: int = 12,    # Annual refit
    ):
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size

    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits.

        Yields:
            (train_indices, test_indices) tuples
        """
        n = len(X)
        splits = []

        start = 0
        while start + self.train_size + self.test_size <= n:
            train_end = start + self.train_size
            test_end = train_end + self.test_size

            train_idx = np.arange(start, train_end)
            test_idx = np.arange(train_end, min(test_end, n))

            splits.append((train_idx, test_idx))
            start += self.step_size

        print(f"Walk-forward CV: {len(splits)} splits")
        return splits

    def get_split_dates(self, X: pd.DataFrame) -> List[Dict]:
        """Get date ranges for each split."""
        splits = self.split(X)
        split_info = []

        for train_idx, test_idx in splits:
            split_info.append({
                'train_start': X.index[train_idx[0]],
                'train_end': X.index[train_idx[-1]],
                'test_start': X.index[test_idx[0]],
                'test_end': X.index[test_idx[-1]],
            })

        return split_info


# ================================================================
# EXAMPLE USAGE
# ================================================================

if __name__ == '__main__':
    from pathlib import Path

    DATA_DIR = Path(__file__).parent.parent / 'data'

    # Load data
    factors = pd.read_parquet(DATA_DIR / 'ff_extended_factors.parquet')
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    factors = factors.drop(columns=['RF'], errors='ignore')

    # Generate features
    fe = FeatureEngineer()
    features = fe.generate_all_features(factors)

    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Date range: {features.index.min()} to {features.index.max()}")

    # Check per-factor crash rates
    print("\n" + "=" * 50)
    print("PER-FACTOR CRASH RATES (Bottom 10%)")
    print("=" * 50)

    crash_targets = fe.create_crash_targets(factors)
    for col in crash_targets.columns:
        if col != 'any_crash':
            rate = crash_targets[col].dropna().mean()
            print(f"  {col}: {rate:.1%}")

    print(f"\n  any_crash: {crash_targets['any_crash'].dropna().mean():.1%}")
    print("  (Note: any_crash is high because P(any of 8) ≈ 1-0.9^8 ≈ 57%)")

    # Create ML dataset for specific factor (more useful)
    print("\n" + "=" * 50)
    print("ML DATASET FOR MOMENTUM CRASHES")
    print("=" * 50)

    X, y = fe.create_ml_dataset(features, factors, target_type='crash', factor='Mom')

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Crash rate: {y.mean():.1%}")

    # Walk-forward CV splits
    cv = WalkForwardCV()
    splits = cv.get_split_dates(X)

    print(f"\nWalk-forward CV: {len(splits)} splits")
    print(f"  First: Train {splits[0]['train_start'].strftime('%Y-%m')} - {splits[0]['train_end'].strftime('%Y-%m')}")
    print(f"  Last:  Test  {splits[-1]['test_start'].strftime('%Y-%m')} - {splits[-1]['test_end'].strftime('%Y-%m')}")

    # Feature summary
    print("\n" + "=" * 50)
    print("FEATURE CATEGORIES")
    print("=" * 50)

    return_feats = [c for c in features.columns if '_ret_' in c or '_rank_' in c or '_zscore' in c]
    vol_feats = [c for c in features.columns if '_vol_' in c]
    corr_feats = [c for c in features.columns if 'corr' in c.lower()]
    crowd_feats = [c for c in features.columns if 'sharpe' in c or 'concentration' in c or 'dispersion' in c]

    print(f"  Return features:      {len(return_feats)}")
    print(f"  Volatility features:  {len(vol_feats)}")
    print(f"  Correlation features: {len(corr_feats)}")
    print(f"  Crowding features:    {len(crowd_feats)}")
    print(f"  Total:                {len(features.columns)}")
