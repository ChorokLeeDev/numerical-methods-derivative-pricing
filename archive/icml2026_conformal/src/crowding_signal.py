"""
Crowding Signal Computation

Compute factor crowding signals for use in crowding-aware conformal prediction.
Crowding is a leading indicator of distribution shift in financial markets.

For ICML 2026 submission.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class CrowdingSignal:
    """
    Compute crowding signals from factor return data.

    Crowding measures how "crowded" a factor strategy is based on:
    1. Return correlation patterns (co-movement)
    2. Volatility compression (crowded trades have lower vol)
    3. Momentum decay (alpha erosion over time)

    Key insight: High crowding predicts distribution shift → regime changes.
    """

    def __init__(
        self,
        correlation_window: int = 12,  # months
        volatility_window: int = 12,   # months
        momentum_windows: Tuple[int, ...] = (1, 3, 6, 12),
    ):
        self.correlation_window = correlation_window
        self.volatility_window = volatility_window
        self.momentum_windows = momentum_windows

    def compute_crowding(
        self,
        factor_returns: pd.Series,
        all_factors: pd.DataFrame,
        normalize: bool = True
    ) -> pd.Series:
        """
        Compute crowding signal for a single factor.

        Combines multiple crowding indicators:
        1. Cross-factor correlation (high = crowded)
        2. Return volatility ratio (low = crowded)
        3. Momentum decay (high recent vs long-term = crowded)

        Args:
            factor_returns: Returns of target factor
            all_factors: DataFrame of all factor returns (for correlation)
            normalize: Whether to normalize to [0, 1]

        Returns:
            Crowding signal series (higher = more crowded)
        """
        signals = []

        # 1. Cross-factor correlation signal
        corr_signal = self._correlation_crowding(factor_returns, all_factors)
        signals.append(('correlation', corr_signal))

        # 2. Volatility compression signal
        vol_signal = self._volatility_crowding(factor_returns)
        signals.append(('volatility', vol_signal))

        # 3. Momentum decay signal
        mom_signal = self._momentum_crowding(factor_returns)
        signals.append(('momentum', mom_signal))

        # Combine signals (equal weight)
        combined = pd.DataFrame({name: sig for name, sig in signals})

        # Forward fill NaN, then backfill remaining
        combined = combined.ffill().bfill()

        # Equal-weight combination
        crowding = combined.mean(axis=1)

        if normalize:
            # Rolling normalization to [0, 1] using expanding window
            crowding_min = crowding.expanding(min_periods=12).min()
            crowding_max = crowding.expanding(min_periods=12).max()
            crowding = (crowding - crowding_min) / (crowding_max - crowding_min + 1e-8)
            crowding = crowding.clip(0, 1)

        return crowding

    def _correlation_crowding(
        self,
        factor_returns: pd.Series,
        all_factors: pd.DataFrame
    ) -> pd.Series:
        """
        Higher correlation with other factors = more crowded.

        When a factor is crowded, it tends to move together with
        other crowded factors (co-momentum effect).
        """
        # Rolling correlation with all other factors
        correlations = []
        factor_name = factor_returns.name

        for col in all_factors.columns:
            if col != factor_name:
                corr = factor_returns.rolling(
                    self.correlation_window, min_periods=6
                ).corr(all_factors[col])
                correlations.append(corr.abs())  # Absolute correlation

        if correlations:
            # Average absolute correlation
            avg_corr = pd.concat(correlations, axis=1).mean(axis=1)
            return avg_corr
        else:
            return pd.Series(0.5, index=factor_returns.index)

    def _volatility_crowding(self, factor_returns: pd.Series) -> pd.Series:
        """
        Lower volatility relative to long-term = more crowded.

        Crowded trades compress volatility as everyone piles into
        similar positions. Volatility expansion often signals
        crowded exit (de-crowding).
        """
        # Short-term volatility
        short_vol = factor_returns.rolling(
            self.volatility_window // 2, min_periods=3
        ).std()

        # Long-term volatility
        long_vol = factor_returns.rolling(
            self.volatility_window * 2, min_periods=12
        ).std()

        # Ratio: low = crowded (compressed), high = de-crowding
        vol_ratio = short_vol / (long_vol + 1e-8)

        # Invert: we want high = crowded
        crowding = 1 / (vol_ratio + 0.5)

        return crowding

    def _momentum_crowding(self, factor_returns: pd.Series) -> pd.Series:
        """
        High short-term vs long-term momentum = more crowded.

        When short-term momentum is high relative to long-term,
        it suggests recent inflows (crowding build-up).
        """
        # Compute momentum at different horizons
        momentums = {}
        for window in self.momentum_windows:
            momentums[window] = factor_returns.rolling(
                window, min_periods=max(1, window // 2)
            ).sum()

        # Short-term vs long-term momentum ratio
        short_mom = momentums[min(self.momentum_windows)]
        long_mom = momentums[max(self.momentum_windows)]

        # Ratio (shifted to be positive)
        mom_diff = short_mom - long_mom

        # Normalize to roughly [0, 1] range
        mom_signal = mom_diff.rolling(36, min_periods=12).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8)
            if len(x) > 0 else 0.5
        )

        return mom_signal

    def compute_all_crowding(
        self,
        factors_df: pd.DataFrame,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Compute crowding signals for all factors.

        Args:
            factors_df: DataFrame with factor returns
            normalize: Whether to normalize each to [0, 1]

        Returns:
            DataFrame with crowding signal for each factor
        """
        crowding_signals = {}

        for factor in factors_df.columns:
            crowding_signals[factor] = self.compute_crowding(
                factors_df[factor],
                factors_df,
                normalize=normalize
            )

        return pd.DataFrame(crowding_signals)


def compute_regime_crowding(
    crowding: pd.Series,
    n_regimes: int = 3
) -> pd.Series:
    """
    Discretize crowding into regimes: Low, Medium, High.

    Uses rolling quantiles to define regime thresholds.
    """
    # Rolling quantiles for regime boundaries
    low_threshold = crowding.expanding(min_periods=24).quantile(1/n_regimes)
    high_threshold = crowding.expanding(min_periods=24).quantile(2/n_regimes)

    # Assign regimes
    regime = pd.Series('medium', index=crowding.index)
    regime[crowding <= low_threshold] = 'low'
    regime[crowding >= high_threshold] = 'high'

    return regime


if __name__ == '__main__':
    # Test with synthetic data
    np.random.seed(42)
    dates = pd.date_range('2000-01-01', periods=300, freq='ME')

    # Create synthetic factor returns
    factors = pd.DataFrame({
        'MKT': np.random.randn(300) * 0.05,
        'SMB': np.random.randn(300) * 0.03,
        'HML': np.random.randn(300) * 0.03,
        'Mom': np.random.randn(300) * 0.04,
    }, index=dates)

    # Compute crowding
    cs = CrowdingSignal()
    crowding = cs.compute_all_crowding(factors)

    print("Crowding Signal Statistics:")
    print(crowding.describe())

    print("\nCrowding Regimes (Mom factor):")
    regimes = compute_regime_crowding(crowding['Mom'])
    print(regimes.value_counts())
