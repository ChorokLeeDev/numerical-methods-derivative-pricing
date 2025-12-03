"""
Extended Factor Backtest

Test crowding signal on full set of factors:
- MKT, SMB, HML, RMW, CMA (FF5)
- Mom, ST_Rev, LT_Rev (Momentum/Reversal)

Hypothesis: More factors = better diversification + more signal dispersion
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from crowding_signal import CrowdingDetector
from factor_timing import FactorTimingStrategy, run_full_backtest, print_backtest_summary

DATA_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_DIR = Path(__file__).parent.parent / 'paper' / 'figures'


def load_extended_factors():
    """Load extended factor data."""
    factors = pd.read_parquet(DATA_DIR / 'ff_extended_factors.parquet')

    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()

    # Drop RF (risk-free rate) - not a factor
    if 'RF' in factors.columns:
        factors = factors.drop(columns=['RF'])

    return factors


def classify_factors():
    """Classify factors as mechanical vs judgment."""
    return {
        'mechanical': ['Mom', 'ST_Rev', 'LT_Rev'],  # Price-based, unambiguous
        'judgment': ['HML', 'RMW', 'CMA'],  # Require accounting interpretation
        'hybrid': ['SMB', 'MKT'],  # Market-based
    }


def analyze_signal_dispersion(signals: dict) -> pd.DataFrame:
    """Analyze how much signals differ across factors."""
    residuals = {}
    for factor, df in signals.items():
        residuals[factor] = df['residual']

    residual_df = pd.DataFrame(residuals)

    # Cross-sectional stats at each time point
    stats = pd.DataFrame({
        'mean': residual_df.mean(axis=1),
        'std': residual_df.std(axis=1),
        'spread': residual_df.max(axis=1) - residual_df.min(axis=1),
    })

    print("\nSignal Dispersion Analysis:")
    print(f"  Average cross-sectional std: {stats['std'].mean():.3f}")
    print(f"  Average spread (max-min): {stats['spread'].mean():.3f}")
    print(f"  Dispersion over time varies: {stats['std'].std():.3f}")

    return stats


def test_mechanical_vs_judgment(
    factor_returns: pd.DataFrame,
    signals: dict,
):
    """
    Test if signal works better for mechanical factors.
    """
    classification = classify_factors()

    print("\n" + "=" * 70)
    print("MECHANICAL VS JUDGMENT FACTORS")
    print("=" * 70)

    for category, factors in classification.items():
        available = [f for f in factors if f in signals]
        if not available:
            continue

        print(f"\n{category.upper()} factors: {available}")

        # Average signal stats
        avg_residual = np.mean([signals[f]['residual'].mean() for f in available])
        pct_crowding = np.mean([(signals[f]['signal'] == 'crowding').mean() for f in available])

        print(f"  Avg residual: {avg_residual:.3f}")
        print(f"  % crowding signals: {100*pct_crowding:.1f}%")


def plot_factor_classification_results(
    factor_returns: pd.DataFrame,
    signals: dict,
    save_path: Path = None
):
    """
    Panel plot showing signal for mechanical vs judgment factors.
    """
    classification = classify_factors()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Panel 1: Mechanical factors
    ax1 = axes[0]
    mech_factors = [f for f in classification['mechanical'] if f in signals]
    for factor in mech_factors:
        df = signals[factor]
        ax1.plot(df.index, df['residual'], label=factor, alpha=0.7)

    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Prediction Residual', fontsize=12)
    ax1.set_title('Mechanical Factors (Price-Based): Crowding Signal',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Judgment factors
    ax2 = axes[1]
    judg_factors = [f for f in classification['judgment'] if f in signals]
    for factor in judg_factors:
        df = signals[factor]
        ax2.plot(df.index, df['residual'], label=factor, alpha=0.7)

    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Prediction Residual', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title('Judgment Factors (Accounting-Based): Crowding Signal',
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def run_long_short_strategy(
    factor_returns: pd.DataFrame,
    signals: dict,
):
    """
    Long-short strategy:
    - Long factors with highest residual (least crowded)
    - Short factors with lowest residual (most crowded)
    """
    # Align all signals
    signal_series = {}
    for factor in factor_returns.columns:
        if factor in signals:
            signal_series[factor] = signals[factor]['residual']

    if len(signal_series) < 4:
        print("Not enough factors for long-short")
        return None

    signal_df = pd.DataFrame(signal_series)

    # Rank factors each period
    ranks = signal_df.rank(axis=1, pct=True)

    # Top tercile = long, bottom tercile = short
    n_factors = len(signal_series)
    long_threshold = 0.67
    short_threshold = 0.33

    weights = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    weights[ranks >= long_threshold] = 1.0
    weights[ranks <= short_threshold] = -1.0

    # Normalize
    long_count = (weights == 1.0).sum(axis=1)
    short_count = (weights == -1.0).sum(axis=1)

    for idx in weights.index:
        if long_count[idx] > 0:
            weights.loc[idx, weights.loc[idx] == 1.0] = 1.0 / long_count[idx]
        if short_count[idx] > 0:
            weights.loc[idx, weights.loc[idx] == -1.0] = -1.0 / short_count[idx]

    # Shift for no lookahead
    weights = weights.shift(1).dropna()

    # Align returns
    common_idx = weights.index.intersection(factor_returns.index)
    weights_aligned = weights.loc[common_idx]
    returns_aligned = factor_returns.loc[common_idx, weights.columns]

    # Strategy returns
    strategy_returns = (returns_aligned * weights_aligned).sum(axis=1)

    return strategy_returns, weights_aligned


def main():
    print("=" * 70)
    print("EXTENDED FACTOR BACKTEST (8 FACTORS)")
    print("=" * 70)

    # Load data
    print("\n1. Loading extended factors...")
    factors = load_extended_factors()
    print(f"   Factors: {list(factors.columns)}")
    print(f"   Date range: {factors.index.min()} to {factors.index.max()}")

    # Use post-1990 for more recent crowding dynamics
    factors = factors[factors.index >= '1990-01-01']
    print(f"   Using data from 1990 onwards: {len(factors)} observations")

    # Compute signals
    print("\n2. Computing crowding signals...")
    detector = CrowdingDetector(
        train_window=120,
        prediction_gap=12,
        sharpe_window=36,
    )

    signals = detector.compute_multi_factor_signals(factors)

    for factor, df in signals.items():
        mean_res = df['residual'].mean()
        pct_crowd = 100 * (df['signal'] == 'crowding').mean()
        print(f"   {factor:<10}: mean residual = {mean_res:>7.3f}, "
              f"% crowding = {pct_crowd:>5.1f}%")

    # Dispersion analysis
    dispersion = analyze_signal_dispersion(signals)

    # Mechanical vs judgment
    test_mechanical_vs_judgment(factors, signals)

    # Plot classification
    print("\n3. Generating classification plot...")
    plot_factor_classification_results(factors, signals,
                                       OUTPUT_DIR / 'fig16_factor_classification.png')

    # Run backtests
    print("\n4. Running backtests...")
    backtest_results = run_full_backtest(factors, signals)
    print_backtest_summary(backtest_results)

    # Long-short strategy
    print("\n5. Testing long-short strategy...")
    ls_result = run_long_short_strategy(factors, signals)

    if ls_result is not None:
        ls_returns, ls_weights = ls_result
        ls_returns = ls_returns.dropna()

        n_years = len(ls_returns) / 12
        total_ret = (1 + ls_returns).prod() - 1
        ann_ret = (1 + total_ret) ** (1/n_years) - 1 if n_years > 0 else 0
        ann_vol = ls_returns.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        print(f"\n   Long-Short Strategy Results:")
        print(f"   Ann. Return: {ann_ret:.1%}")
        print(f"   Ann. Vol: {ann_vol:.1%}")
        print(f"   Sharpe: {sharpe:.2f}")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
