"""
Factor Timing Strategy Based on Crowding Signals

Strategy:
- Long factors with positive residual (uncrowding)
- Short/underweight factors with negative residual (crowding)
- Compare vs equal-weight and momentum-of-factors benchmarks
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BacktestResult:
    """Container for backtest results."""
    strategy_name: str
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    returns: pd.Series
    weights: pd.DataFrame


class FactorTimingStrategy:
    """
    Factor timing based on crowding signals.

    Strategies:
    1. Equal weight (benchmark)
    2. Crowding-timed (overweight uncrowded, underweight crowded)
    3. Momentum-of-factors (benchmark)
    4. Crowding + Momentum (combined)
    """

    def __init__(
        self,
        lookback_window: int = 12,
        rebalance_freq: str = 'M',
        long_only: bool = True,
    ):
        self.lookback_window = lookback_window
        self.rebalance_freq = rebalance_freq
        self.long_only = long_only

    def equal_weight_strategy(
        self,
        factor_returns: pd.DataFrame,
    ) -> BacktestResult:
        """Simple equal-weight factor portfolio."""
        weights = pd.DataFrame(
            1.0 / len(factor_returns.columns),
            index=factor_returns.index,
            columns=factor_returns.columns
        )
        returns = (factor_returns * weights).sum(axis=1)

        return self._compute_metrics(returns, weights, 'Equal Weight')

    def momentum_of_factors_strategy(
        self,
        factor_returns: pd.DataFrame,
        lookback: int = 12,
    ) -> BacktestResult:
        """
        Momentum-of-factors: overweight recent winners.
        Classic factor timing benchmark.
        """
        # Compute trailing returns
        trailing_returns = factor_returns.rolling(lookback).mean()

        # Rank factors (higher = better momentum)
        ranks = trailing_returns.rank(axis=1, pct=True)

        # Convert ranks to weights
        if self.long_only:
            weights = ranks / ranks.sum(axis=1).values.reshape(-1, 1)
        else:
            # Long top half, short bottom half
            weights = (ranks - 0.5) * 2
            weights = weights / weights.abs().sum(axis=1).values.reshape(-1, 1)

        # Shift weights (no lookahead)
        weights = weights.shift(1).dropna()
        aligned_returns = factor_returns.loc[weights.index]

        returns = (aligned_returns * weights).sum(axis=1)

        return self._compute_metrics(returns, weights, 'Factor Momentum')

    def crowding_timed_strategy(
        self,
        factor_returns: pd.DataFrame,
        crowding_signals: Dict[str, pd.DataFrame],
        signal_col: str = 'residual',
    ) -> BacktestResult:
        """
        Crowding-timed strategy:
        - Overweight factors with positive residual (uncrowding)
        - Underweight factors with negative residual (crowding)
        """
        # Align signals with returns
        signal_df = self._align_signals(factor_returns, crowding_signals, signal_col)

        if signal_df is None or len(signal_df) == 0:
            raise ValueError("No aligned signals available")

        # Convert signals to weights
        # Higher signal (positive residual) = higher weight
        if self.long_only:
            # Shift to positive, then normalize
            shifted = signal_df - signal_df.min().min() + 0.01
            weights = shifted / shifted.sum(axis=1).values.reshape(-1, 1)
        else:
            # Use signal directly (positive = long, negative = short)
            weights = signal_df / signal_df.abs().sum(axis=1).values.reshape(-1, 1)

        # Shift weights (no lookahead)
        weights = weights.shift(1).dropna()
        aligned_returns = factor_returns.loc[weights.index]

        returns = (aligned_returns * weights).sum(axis=1)

        return self._compute_metrics(returns, weights, 'Crowding Timed')

    def crowding_ranked_strategy(
        self,
        factor_returns: pd.DataFrame,
        crowding_signals: Dict[str, pd.DataFrame],
    ) -> BacktestResult:
        """
        Simple ranking strategy:
        - Long factors in top tercile of crowding score (most uncrowded)
        - Avoid factors in bottom tercile (most crowded)
        """
        signal_df = self._align_signals(factor_returns, crowding_signals, 'residual')

        if signal_df is None or len(signal_df) == 0:
            raise ValueError("No aligned signals available")

        # Rank factors by crowding (higher residual = less crowded = better)
        ranks = signal_df.rank(axis=1, pct=True)

        # Top tercile gets weight, bottom gets zero
        weights = (ranks > 0.33).astype(float)
        weights = weights / weights.sum(axis=1).values.reshape(-1, 1)
        weights = weights.fillna(0)

        # Shift weights
        weights = weights.shift(1).dropna()
        aligned_returns = factor_returns.loc[weights.index]

        returns = (aligned_returns * weights).sum(axis=1)

        return self._compute_metrics(returns, weights, 'Crowding Ranked')

    def combined_strategy(
        self,
        factor_returns: pd.DataFrame,
        crowding_signals: Dict[str, pd.DataFrame],
        crowding_weight: float = 0.5,
    ) -> BacktestResult:
        """
        Combine crowding signal with momentum.
        """
        signal_df = self._align_signals(factor_returns, crowding_signals, 'residual')

        if signal_df is None:
            raise ValueError("No aligned signals")

        # Crowding score (normalized)
        crowding_ranks = signal_df.rank(axis=1, pct=True)

        # Momentum score
        mom_returns = factor_returns.rolling(12).mean()
        mom_ranks = mom_returns.rank(axis=1, pct=True)

        # Align
        common_idx = crowding_ranks.index.intersection(mom_ranks.index)
        crowding_aligned = crowding_ranks.loc[common_idx]
        mom_aligned = mom_ranks.loc[common_idx]

        # Combined score
        combined = crowding_weight * crowding_aligned + (1 - crowding_weight) * mom_aligned

        # Convert to weights
        weights = combined / combined.sum(axis=1).values.reshape(-1, 1)
        weights = weights.shift(1).dropna()

        aligned_returns = factor_returns.loc[weights.index]
        returns = (aligned_returns * weights).sum(axis=1)

        return self._compute_metrics(returns, weights, 'Crowding + Momentum')

    def _align_signals(
        self,
        factor_returns: pd.DataFrame,
        crowding_signals: Dict[str, pd.DataFrame],
        signal_col: str,
    ) -> Optional[pd.DataFrame]:
        """Align crowding signals with factor returns."""
        signal_series = {}

        for factor in factor_returns.columns:
            if factor in crowding_signals:
                sig = crowding_signals[factor]
                if signal_col in sig.columns:
                    signal_series[factor] = sig[signal_col]

        if not signal_series:
            return None

        signal_df = pd.DataFrame(signal_series)

        # Align with returns index
        common_idx = signal_df.index.intersection(factor_returns.index)
        return signal_df.loc[common_idx]

    def _compute_metrics(
        self,
        returns: pd.Series,
        weights: pd.DataFrame,
        strategy_name: str,
    ) -> BacktestResult:
        """Compute performance metrics."""
        returns = returns.dropna()

        if len(returns) == 0:
            raise ValueError("No returns to compute metrics")

        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / 12
        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        ann_vol = returns.std() * np.sqrt(12)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_dd = drawdowns.min()

        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

        return BacktestResult(
            strategy_name=strategy_name,
            total_return=total_return,
            annualized_return=ann_return,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            returns=returns,
            weights=weights,
        )


def run_full_backtest(
    factor_returns: pd.DataFrame,
    crowding_signals: Dict[str, pd.DataFrame],
) -> Dict[str, BacktestResult]:
    """Run all strategies and compare."""
    strategy = FactorTimingStrategy(long_only=True)

    results = {}

    # Benchmark: Equal weight
    results['equal_weight'] = strategy.equal_weight_strategy(factor_returns)

    # Benchmark: Factor momentum
    try:
        results['factor_momentum'] = strategy.momentum_of_factors_strategy(factor_returns)
    except Exception as e:
        print(f"Factor momentum failed: {e}")

    # Our strategy: Crowding timed
    try:
        results['crowding_timed'] = strategy.crowding_timed_strategy(
            factor_returns, crowding_signals
        )
    except Exception as e:
        print(f"Crowding timed failed: {e}")

    # Our strategy: Crowding ranked
    try:
        results['crowding_ranked'] = strategy.crowding_ranked_strategy(
            factor_returns, crowding_signals
        )
    except Exception as e:
        print(f"Crowding ranked failed: {e}")

    # Combined
    try:
        results['combined'] = strategy.combined_strategy(
            factor_returns, crowding_signals
        )
    except Exception as e:
        print(f"Combined failed: {e}")

    return results


def print_backtest_summary(results: Dict[str, BacktestResult]):
    """Print formatted backtest summary."""
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS: FACTOR TIMING STRATEGIES")
    print("=" * 80)

    print(f"\n{'Strategy':<25} {'Ann. Ret':>10} {'Ann. Vol':>10} {'Sharpe':>10} {'Max DD':>10}")
    print("-" * 65)

    for name, res in sorted(results.items(), key=lambda x: -x[1].sharpe_ratio):
        print(f"{res.strategy_name:<25} {res.annualized_return:>10.2%} "
              f"{res.annualized_volatility:>10.2%} {res.sharpe_ratio:>10.2f} "
              f"{res.max_drawdown:>10.2%}")

    # Compute alpha vs benchmark
    if 'equal_weight' in results and 'crowding_timed' in results:
        benchmark_sharpe = results['equal_weight'].sharpe_ratio
        strategy_sharpe = results['crowding_timed'].sharpe_ratio
        improvement = strategy_sharpe - benchmark_sharpe

        print(f"\n{'='*65}")
        print(f"Sharpe improvement vs Equal Weight: {improvement:+.2f}")
        print(f"{'='*65}")
