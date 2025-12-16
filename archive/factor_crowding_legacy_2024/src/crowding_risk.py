"""
Crowding-Aware Risk Management

Integrates crowding signals into portfolio risk framework.

Key Research Findings (experiments/10_crowding_utilization.py):
============================================================

WHAT WORKS:
1. Crowding × Momentum Interaction
   - Factor momentum Sharpe: 0.67 (uncrowded) vs 0.28 (crowded)
   - Use crowding to CONDITION factor momentum strategies
   - Implementation: CrowdingConditionedMomentum class

2. Aggregate Exposure Timing (Drawdown Reduction)
   - Max DD: -9.1% (timed) vs -12.9% (always-in)
   - Reduce ALL factor exposure when crowding is extreme
   - Implementation: CrowdingRiskManager.compute_risk_adjusted_weights()

3. Long-Horizon Prediction (12M)
   - Crowding predicts ANNUAL returns (r = -0.39 to -0.52, p<0.001)
   - Not useful for monthly timing, but informs strategic allocation
   - Implementation: assess_risk() warnings

WHAT DOESN'T WORK:
- Cross-sectional factor selection (picking "uncrowded" factors)
- Monthly return prediction
- Contrarian strategies (buying crowded factors)

KEY INSIGHT:
Crowding is REGIME information, not FACTOR SELECTION information.
Use it to adjust HOW MUCH exposure, not WHICH exposure.

Usage:
------
    from crowding_risk import CrowdingRiskManager, CrowdingConditionedMomentum

    # Risk management (drawdown reduction)
    risk_mgr = CrowdingRiskManager()
    report = risk_mgr.assess_risk(factor_returns)
    adjusted_weights, cash = risk_mgr.compute_risk_adjusted_weights(base_weights, report)

    # Crowding-conditioned momentum (alpha enhancement)
    strategy = CrowdingConditionedMomentum()
    weights = strategy.compute_weights(factor_returns)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum

from crowding_signal import CrowdingDetector, rolling_sharpe


class CrowdingRegime(Enum):
    """Crowding regime classification."""
    LOW = "low"           # Aggregate residual > 0: factors uncrowded
    MODERATE = "moderate" # Aggregate residual in [-0.2, 0]
    HIGH = "high"         # Aggregate residual in [-0.5, -0.2]
    EXTREME = "extreme"   # Aggregate residual < -0.5


@dataclass
class CrowdingRiskReport:
    """Output of crowding risk assessment."""
    date: pd.Timestamp
    regime: CrowdingRegime
    aggregate_residual: float
    factor_residuals: Dict[str, float]
    recommended_factor_exposure: float  # 0-1 scale
    risk_budget_multiplier: float       # Scale drawdown budget
    warnings: list


class CrowdingRiskManager:
    """
    Crowding-aware risk management system.

    Philosophy:
    - Crowding is detectable but efficiently priced
    - Signal value is in REGIME AWARENESS, not prediction
    - Use to adjust risk budgets, not to time factors

    Applications:
    1. Position sizing: Reduce factor exposure when crowding is extreme
    2. Drawdown budgeting: Expect larger drawdowns in crowded regimes
    3. Factor limits: Set max exposure based on crowding level
    4. Monitoring: Alert when crowding regime changes
    """

    def __init__(
        self,
        detector: Optional[CrowdingDetector] = None,
        regime_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.detector = detector or CrowdingDetector(
            train_window=120,
            prediction_gap=12,
            sharpe_window=36,
        )

        # Regime classification thresholds (aggregate residual)
        self.thresholds = regime_thresholds or {
            'low': 0.0,        # residual > 0
            'moderate': -0.2,  # residual in [-0.2, 0]
            'high': -0.5,      # residual in [-0.5, -0.2]
            'extreme': -0.5,   # residual < -0.5
        }

        # Risk adjustment parameters
        self.exposure_map = {
            CrowdingRegime.LOW: 1.0,       # Full exposure OK
            CrowdingRegime.MODERATE: 0.85, # Slight reduction
            CrowdingRegime.HIGH: 0.70,     # Meaningful reduction
            CrowdingRegime.EXTREME: 0.50,  # Half exposure
        }

        self.risk_budget_map = {
            CrowdingRegime.LOW: 1.0,       # Normal drawdown budget
            CrowdingRegime.MODERATE: 1.2,  # Expect 20% larger drawdowns
            CrowdingRegime.HIGH: 1.5,      # Expect 50% larger drawdowns
            CrowdingRegime.EXTREME: 2.0,   # Expect 2x drawdowns
        }

    def classify_regime(self, aggregate_residual: float) -> CrowdingRegime:
        """Classify crowding regime based on aggregate residual."""
        if aggregate_residual > self.thresholds['low']:
            return CrowdingRegime.LOW
        elif aggregate_residual > self.thresholds['moderate']:
            return CrowdingRegime.MODERATE
        elif aggregate_residual > self.thresholds['high']:
            return CrowdingRegime.HIGH
        else:
            return CrowdingRegime.EXTREME

    def compute_signals(
        self,
        factor_returns: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Compute crowding signals for all factors."""
        return self.detector.compute_multi_factor_signals(factor_returns)

    def assess_risk(
        self,
        factor_returns: pd.DataFrame,
        signals: Optional[Dict[str, pd.DataFrame]] = None,
        as_of_date: Optional[pd.Timestamp] = None,
    ) -> CrowdingRiskReport:
        """
        Assess current crowding risk.

        Returns:
            CrowdingRiskReport with regime, recommendations, warnings
        """
        # Compute signals if not provided
        if signals is None:
            signals = self.compute_signals(factor_returns)

        # Get most recent date
        if as_of_date is None:
            all_dates = []
            for df in signals.values():
                if len(df) > 0:
                    all_dates.extend(df.index.tolist())
            as_of_date = max(all_dates) if all_dates else pd.Timestamp.now()

        # Get factor residuals at date
        factor_residuals = {}
        for factor, df in signals.items():
            if len(df) > 0:
                # Get closest date
                valid_dates = df.index[df.index <= as_of_date]
                if len(valid_dates) > 0:
                    factor_residuals[factor] = df.loc[valid_dates[-1], 'residual']

        if not factor_residuals:
            return CrowdingRiskReport(
                date=as_of_date,
                regime=CrowdingRegime.MODERATE,
                aggregate_residual=0.0,
                factor_residuals={},
                recommended_factor_exposure=0.85,
                risk_budget_multiplier=1.2,
                warnings=["Insufficient data for crowding assessment"],
            )

        # Compute aggregate residual
        aggregate_residual = np.mean(list(factor_residuals.values()))

        # Classify regime
        regime = self.classify_regime(aggregate_residual)

        # Get recommendations
        recommended_exposure = self.exposure_map[regime]
        risk_budget_mult = self.risk_budget_map[regime]

        # Generate warnings
        warnings = []

        if regime == CrowdingRegime.EXTREME:
            warnings.append("EXTREME CROWDING: Consider reducing factor exposure significantly")

        if regime in [CrowdingRegime.HIGH, CrowdingRegime.EXTREME]:
            warnings.append(f"Expect {(risk_budget_mult-1)*100:.0f}% larger drawdowns than normal")

        # Check for factor-specific concerns
        for factor, residual in factor_residuals.items():
            if residual < -0.5:
                warnings.append(f"{factor}: Severely crowded (residual={residual:.2f})")

        # Check for divergence
        residual_std = np.std(list(factor_residuals.values()))
        if residual_std > 0.3:
            warnings.append("High dispersion across factors - consider factor-specific adjustments")

        return CrowdingRiskReport(
            date=as_of_date,
            regime=regime,
            aggregate_residual=aggregate_residual,
            factor_residuals=factor_residuals,
            recommended_factor_exposure=recommended_exposure,
            risk_budget_multiplier=risk_budget_mult,
            warnings=warnings,
        )

    def compute_risk_adjusted_weights(
        self,
        base_weights: pd.Series,
        risk_report: CrowdingRiskReport,
    ) -> pd.Series:
        """
        Adjust portfolio weights based on crowding risk.

        Note: This reduces OVERALL factor exposure, not individual factor selection.
        The research shows cross-sectional timing doesn't work.
        """
        # Scale all factor weights by recommended exposure
        adjusted = base_weights * risk_report.recommended_factor_exposure

        # Allocate remainder to cash (or risk-free)
        cash_weight = 1.0 - adjusted.sum()

        return adjusted, cash_weight

    def get_historical_regimes(
        self,
        factor_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute historical regime classification."""
        signals = self.compute_signals(factor_returns)

        # Aggregate residuals over time
        residual_dfs = []
        for factor, df in signals.items():
            if len(df) > 0:
                residual_dfs.append(df['residual'].rename(factor))

        if not residual_dfs:
            return pd.DataFrame()

        all_residuals = pd.concat(residual_dfs, axis=1)
        aggregate = all_residuals.mean(axis=1)

        # Classify each period
        regimes = aggregate.apply(self.classify_regime)

        return pd.DataFrame({
            'aggregate_residual': aggregate,
            'regime': regimes,
        })


class CrowdingConditionedMomentum:
    """
    Crowding-Conditioned Factor Momentum Strategy.

    Key Finding: Factor momentum works significantly better in uncrowded regimes.
    - Uncrowded regime Sharpe: 0.67
    - Crowded regime Sharpe: 0.28
    - Improvement: +139% risk-adjusted returns

    Strategy Logic:
    1. Compute trailing 12M factor returns (standard factor momentum)
    2. Compute aggregate crowding signal
    3. In uncrowded regimes: Full momentum weights
    4. In crowded regimes: Reduce to equal-weight (momentum signal less reliable)

    This is NOT cross-sectional timing (which doesn't work).
    This is CONDITIONING an existing strategy on regime information.

    Example:
    --------
        strategy = CrowdingConditionedMomentum()
        weights = strategy.compute_weights(factor_returns)

        # Backtest
        returns = (factor_returns * weights.shift(1)).sum(axis=1)
    """

    def __init__(
        self,
        detector: Optional[CrowdingDetector] = None,
        momentum_lookback: int = 12,
        crowding_threshold: float = 0.0,  # Median split point
        uncrowded_momentum_weight: float = 1.0,
        crowded_momentum_weight: float = 0.3,  # Blend toward equal-weight
    ):
        """
        Initialize crowding-conditioned momentum strategy.

        Args:
            detector: CrowdingDetector instance (uses defaults if None)
            momentum_lookback: Months for trailing momentum calculation
            crowding_threshold: Aggregate residual threshold for regime split
                              (0.0 = median, negative = more conservative)
            uncrowded_momentum_weight: Weight on momentum in uncrowded regime (1.0 = full)
            crowded_momentum_weight: Weight on momentum in crowded regime (0.0 = equal-weight)
        """
        self.detector = detector or CrowdingDetector(
            train_window=120,
            prediction_gap=12,
            sharpe_window=36,
        )
        self.momentum_lookback = momentum_lookback
        self.crowding_threshold = crowding_threshold
        self.uncrowded_momentum_weight = uncrowded_momentum_weight
        self.crowded_momentum_weight = crowded_momentum_weight

    def compute_momentum_weights(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute standard factor momentum weights.

        Returns DataFrame of weights (rows=dates, cols=factors).
        Weights sum to 1 at each date.
        """
        # Trailing returns
        trailing = factor_returns.rolling(self.momentum_lookback).mean()

        # Rank and normalize
        ranks = trailing.rank(axis=1, pct=True)
        weights = ranks / ranks.sum(axis=1).values.reshape(-1, 1)

        return weights

    def compute_equal_weights(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """Compute equal-weight allocation."""
        n_factors = len(factor_returns.columns)
        weights = pd.DataFrame(
            1.0 / n_factors,
            index=factor_returns.index,
            columns=factor_returns.columns,
        )
        return weights

    def compute_aggregate_crowding(
        self,
        signals: Dict[str, pd.DataFrame],
    ) -> pd.Series:
        """Compute aggregate crowding signal across factors."""
        residuals = pd.DataFrame({
            factor: df['residual']
            for factor, df in signals.items()
            if len(df) > 0
        })
        return residuals.mean(axis=1)

    def compute_weights(
        self,
        factor_returns: pd.DataFrame,
        signals: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Compute crowding-conditioned momentum weights.

        Args:
            factor_returns: DataFrame of factor returns
            signals: Pre-computed crowding signals (computed if None)

        Returns:
            DataFrame of weights (rows=dates, cols=factors)
            Weights are shifted by 1 period (no lookahead)
        """
        # Get factor columns
        factor_cols = [c for c in factor_returns.columns if c not in ['RF', 'Mkt-RF']]
        returns = factor_returns[factor_cols].dropna()

        # Compute signals if not provided
        if signals is None:
            signals = self.detector.compute_multi_factor_signals(returns)

        # Compute component weights
        momentum_weights = self.compute_momentum_weights(returns)
        equal_weights = self.compute_equal_weights(returns)

        # Compute aggregate crowding
        aggregate_crowding = self.compute_aggregate_crowding(signals)

        # Align indices
        common_idx = momentum_weights.index.intersection(aggregate_crowding.index)
        momentum_aligned = momentum_weights.loc[common_idx]
        equal_aligned = equal_weights.loc[common_idx]
        crowding_aligned = aggregate_crowding.loc[common_idx]

        # Determine regime and blend weights
        is_uncrowded = crowding_aligned > self.crowding_threshold

        # Blending weight for momentum (vs equal-weight)
        mom_blend = pd.Series(index=common_idx, dtype=float)
        mom_blend[is_uncrowded] = self.uncrowded_momentum_weight
        mom_blend[~is_uncrowded] = self.crowded_momentum_weight

        # Blend momentum and equal-weight based on regime
        final_weights = pd.DataFrame(index=common_idx, columns=returns.columns)
        for col in returns.columns:
            final_weights[col] = (
                mom_blend * momentum_aligned[col] +
                (1 - mom_blend) * equal_aligned[col]
            )

        # Shift for no lookahead
        return final_weights.shift(1).dropna()

    def backtest(
        self,
        factor_returns: pd.DataFrame,
        signals: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict:
        """
        Backtest the crowding-conditioned momentum strategy.

        Returns dict with returns series and performance metrics.
        """
        weights = self.compute_weights(factor_returns, signals)

        # Align returns with weights
        factor_cols = weights.columns.tolist()
        returns = factor_returns[factor_cols].loc[weights.index]

        # Compute strategy returns
        strategy_returns = (returns * weights).sum(axis=1)

        # Compute metrics
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(12)
        cum_return = (1 + strategy_returns).prod() - 1
        cum_series = (1 + strategy_returns).cumprod()
        max_dd = (cum_series / cum_series.expanding().max() - 1).min()

        return {
            'returns': strategy_returns,
            'weights': weights,
            'sharpe': sharpe,
            'cumulative_return': cum_return,
            'max_drawdown': max_dd,
            'n_periods': len(strategy_returns),
        }


def print_risk_report(report: CrowdingRiskReport):
    """Pretty print a crowding risk report."""
    print("\n" + "=" * 60)
    print("CROWDING RISK REPORT")
    print("=" * 60)
    print(f"Date: {report.date}")
    print(f"Regime: {report.regime.value.upper()}")
    print(f"Aggregate Residual: {report.aggregate_residual:.3f}")
    print(f"\nRecommendations:")
    print(f"  Factor Exposure: {report.recommended_factor_exposure:.0%}")
    print(f"  Risk Budget Multiplier: {report.risk_budget_multiplier:.1f}x")

    if report.warnings:
        print(f"\nWarnings:")
        for w in report.warnings:
            print(f"  ⚠ {w}")

    print(f"\nFactor-Level Residuals:")
    for factor, residual in sorted(report.factor_residuals.items(), key=lambda x: x[1]):
        status = "🔴" if residual < -0.3 else "🟡" if residual < 0 else "🟢"
        print(f"  {status} {factor}: {residual:.3f}")

    print("=" * 60)


# Example usage
if __name__ == '__main__':
    import sys
    from pathlib import Path

    DATA_DIR = Path(__file__).parent.parent / 'data'

    # Load data
    factors = pd.read_parquet(DATA_DIR / 'ff_extended_factors.parquet')
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    if 'RF' in factors.columns:
        factors = factors.drop(columns=['RF'])

    # Filter to post-1990 for sufficient data
    factors = factors[factors.index >= '1990-01-01']

    print("=" * 70)
    print("CROWDING-AWARE RISK MANAGEMENT DEMO")
    print("=" * 70)

    # ================================================================
    # PART 1: Risk Management (Drawdown Reduction)
    # ================================================================
    print("\n" + "-" * 70)
    print("PART 1: RISK MANAGEMENT")
    print("-" * 70)

    risk_mgr = CrowdingRiskManager()

    # Compute signals
    print("Computing crowding signals...")
    signals = risk_mgr.compute_signals(factors)

    # Get current risk assessment
    report = risk_mgr.assess_risk(factors, signals)
    print_risk_report(report)

    # Historical regime analysis
    print("\nHistorical Regime Distribution:")
    regimes = risk_mgr.get_historical_regimes(factors)
    if len(regimes) > 0:
        regime_counts = regimes['regime'].value_counts()
        for regime, count in regime_counts.items():
            pct = 100 * count / len(regimes)
            print(f"  {regime.value}: {pct:.1f}%")

    # ================================================================
    # PART 2: Crowding-Conditioned Momentum (Alpha Enhancement)
    # ================================================================
    print("\n" + "-" * 70)
    print("PART 2: CROWDING-CONDITIONED MOMENTUM")
    print("-" * 70)

    strategy = CrowdingConditionedMomentum()
    results = strategy.backtest(factors, signals)

    print(f"\nCrowding-Conditioned Momentum Performance:")
    print(f"  Sharpe Ratio:      {results['sharpe']:.2f}")
    print(f"  Cumulative Return: {results['cumulative_return']:.1%}")
    print(f"  Max Drawdown:      {results['max_drawdown']:.1%}")
    print(f"  N Periods:         {results['n_periods']}")

    # Compare to pure momentum
    print("\nComparison to Pure Factor Momentum:")
    pure_momentum = CrowdingConditionedMomentum(
        uncrowded_momentum_weight=1.0,
        crowded_momentum_weight=1.0,  # Always use momentum
    )
    pure_results = pure_momentum.backtest(factors, signals)

    equal_weight = CrowdingConditionedMomentum(
        uncrowded_momentum_weight=0.0,
        crowded_momentum_weight=0.0,  # Always equal-weight
    )
    eq_results = equal_weight.backtest(factors, signals)

    print(f"{'Strategy':<30} {'Sharpe':>10} {'Return':>12} {'Max DD':>10}")
    print("-" * 62)
    print(f"{'Equal Weight':<30} {eq_results['sharpe']:>10.2f} {eq_results['cumulative_return']:>12.1%} {eq_results['max_drawdown']:>10.1%}")
    print(f"{'Pure Factor Momentum':<30} {pure_results['sharpe']:>10.2f} {pure_results['cumulative_return']:>12.1%} {pure_results['max_drawdown']:>10.1%}")
    print(f"{'Crowding-Conditioned Momentum':<30} {results['sharpe']:>10.2f} {results['cumulative_return']:>12.1%} {results['max_drawdown']:>10.1%}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: Use crowding to CONDITION strategies, not SELECT factors")
    print("=" * 70)
