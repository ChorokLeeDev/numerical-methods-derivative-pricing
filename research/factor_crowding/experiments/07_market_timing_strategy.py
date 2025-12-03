"""
Market Timing Strategy Based on Aggregate Crowding

Key insight: When ALL factors are crowded, reduce overall factor exposure.
The signal is better for timing WHEN to be in factors, not WHICH factors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from crowding_signal import CrowdingDetector

DATA_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_DIR = Path(__file__).parent.parent / 'paper' / 'figures'


def load_data():
    factors = pd.read_parquet(DATA_DIR / 'ff_factors_monthly.parquet')
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    return factors


def aggregate_crowding_signal(signals: dict) -> pd.Series:
    """
    Aggregate crowding signals across factors.
    Returns: average residual (more negative = more aggregate crowding)
    """
    residuals = []
    for factor, df in signals.items():
        residuals.append(df['residual'].rename(factor))

    if not residuals:
        return pd.Series()

    residual_df = pd.concat(residuals, axis=1)
    return residual_df.mean(axis=1)


def market_timing_backtest(
    factor_returns: pd.DataFrame,
    signals: dict,
    threshold_low: float = -0.3,  # Heavy crowding
    threshold_high: float = 0.0,  # Light/no crowding
):
    """
    Market timing strategy:
    - When aggregate crowding is high (residual < threshold_low): 0% factor exposure
    - When aggregate crowding is low (residual > threshold_high): 100% exposure
    - Otherwise: scale linearly
    """
    # Aggregate signal
    agg_signal = aggregate_crowding_signal(signals)

    # Equal weight factor returns
    eq_returns = factor_returns.mean(axis=1)

    # Align
    common_idx = agg_signal.index.intersection(eq_returns.index)
    signal_aligned = agg_signal.loc[common_idx]
    returns_aligned = eq_returns.loc[common_idx]

    # Compute weight based on signal (lagged)
    def signal_to_weight(s):
        if s < threshold_low:
            return 0.0
        elif s > threshold_high:
            return 1.0
        else:
            # Linear interpolation
            return (s - threshold_low) / (threshold_high - threshold_low)

    weights = signal_aligned.shift(1).apply(signal_to_weight)
    weights = weights.dropna()

    returns_aligned = returns_aligned.loc[weights.index]

    # Strategy returns
    strategy_returns = returns_aligned * weights

    # Benchmark (always 100% invested)
    benchmark_returns = returns_aligned

    return strategy_returns, benchmark_returns, weights


def compute_metrics(returns: pd.Series, name: str) -> dict:
    """Compute performance metrics."""
    returns = returns.dropna()
    n_years = len(returns) / 12

    total_ret = (1 + returns).prod() - 1
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    cumulative = (1 + returns).cumprod()
    max_dd = (cumulative / cumulative.expanding().max() - 1).min()

    return {
        'name': name,
        'ann_return': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
    }


def plot_market_timing(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    weights: pd.Series,
    save_path: Path = None
):
    """Plot market timing strategy results."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), height_ratios=[2, 1, 1])

    # Panel 1: Cumulative returns
    ax1 = axes[0]
    strat_cum = (1 + strategy_returns).cumprod()
    bench_cum = (1 + benchmark_returns).cumprod()

    ax1.plot(bench_cum.index, bench_cum.values, 'gray', linewidth=2,
             label='Always Invested (Benchmark)')
    ax1.plot(strat_cum.index, strat_cum.values, 'blue', linewidth=2,
             label='Crowding-Timed')

    ax1.set_ylabel('Cumulative Return', fontsize=12)
    ax1.set_title('Market Timing: Reduce Exposure When Factors Are Crowded',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Panel 2: Factor exposure over time
    ax2 = axes[1]
    ax2.fill_between(weights.index, weights.values, alpha=0.5, color='blue')
    ax2.plot(weights.index, weights.values, 'b-', linewidth=1)
    ax2.set_ylabel('Factor Exposure', fontsize=12)
    ax2.set_title('Dynamic Factor Exposure Based on Crowding Signal',
                  fontsize=12, fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Drawdowns
    ax3 = axes[2]
    strat_dd = strat_cum / strat_cum.expanding().max() - 1
    bench_dd = bench_cum / bench_cum.expanding().max() - 1

    ax3.fill_between(bench_dd.index, bench_dd.values, alpha=0.3, color='gray',
                     label='Benchmark')
    ax3.fill_between(strat_dd.index, strat_dd.values, alpha=0.3, color='blue',
                     label='Crowding-Timed')
    ax3.set_ylabel('Drawdown', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_title('Drawdown Comparison', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def run_sensitivity_analysis(
    factor_returns: pd.DataFrame,
    signals: dict,
):
    """Test different threshold parameters."""
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: THRESHOLD PARAMETERS")
    print("=" * 70)

    thresholds = [
        (-0.5, -0.1),
        (-0.4, 0.0),
        (-0.3, 0.0),
        (-0.3, 0.1),
        (-0.2, 0.1),
        (-0.2, 0.2),
    ]

    results = []

    for t_low, t_high in thresholds:
        strat, bench, weights = market_timing_backtest(
            factor_returns, signals, t_low, t_high
        )
        strat_metrics = compute_metrics(strat, f'Timed ({t_low}, {t_high})')
        bench_metrics = compute_metrics(bench, 'Benchmark')

        avg_exposure = weights.mean()
        pct_out = (weights == 0).mean()

        results.append({
            'threshold_low': t_low,
            'threshold_high': t_high,
            'sharpe': strat_metrics['sharpe'],
            'ann_return': strat_metrics['ann_return'],
            'max_dd': strat_metrics['max_dd'],
            'avg_exposure': avg_exposure,
            'pct_out': pct_out,
            'bench_sharpe': bench_metrics['sharpe'],
        })

    results_df = pd.DataFrame(results)

    print(f"\n{'Thresholds':<15} {'Sharpe':>10} {'Ann Ret':>10} {'Max DD':>10} "
          f"{'Avg Exp':>10} {'% Out':>10}")
    print("-" * 65)

    for _, row in results_df.iterrows():
        print(f"({row['threshold_low']:.1f}, {row['threshold_high']:.1f})"
              f"{row['sharpe']:>12.2f} {row['ann_return']:>10.1%} "
              f"{row['max_dd']:>10.1%} {row['avg_exposure']:>10.1%} "
              f"{row['pct_out']:>10.1%}")

    print(f"\nBenchmark Sharpe: {results_df['bench_sharpe'].iloc[0]:.2f}")

    return results_df


def main():
    print("=" * 70)
    print("MARKET TIMING STRATEGY")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    factors = load_data()
    factor_cols = ['Mom', 'HML', 'SMB']
    factor_returns = factors[factor_cols].dropna()

    # Compute signals
    print("\n2. Computing crowding signals...")
    detector = CrowdingDetector(
        train_window=120,
        prediction_gap=12,
        sharpe_window=36,
    )
    signals = detector.compute_multi_factor_signals(factor_returns)

    # Aggregate signal analysis
    agg_signal = aggregate_crowding_signal(signals)
    print(f"\n   Aggregate signal stats:")
    print(f"   Mean: {agg_signal.mean():.3f}")
    print(f"   Std: {agg_signal.std():.3f}")
    print(f"   Min: {agg_signal.min():.3f}")
    print(f"   Max: {agg_signal.max():.3f}")

    # Run market timing backtest
    print("\n3. Running market timing backtest...")
    strategy_returns, benchmark_returns, weights = market_timing_backtest(
        factor_returns, signals,
        threshold_low=-0.3,
        threshold_high=0.0
    )

    strat_metrics = compute_metrics(strategy_returns, 'Crowding-Timed')
    bench_metrics = compute_metrics(benchmark_returns, 'Always Invested')

    print(f"\n   Results:")
    print(f"   {'Strategy':<20} {'Sharpe':>10} {'Ann. Ret':>12} {'Max DD':>12}")
    print(f"   {'-'*54}")
    print(f"   {bench_metrics['name']:<20} {bench_metrics['sharpe']:>10.2f} "
          f"{bench_metrics['ann_return']:>12.1%} {bench_metrics['max_dd']:>12.1%}")
    print(f"   {strat_metrics['name']:<20} {strat_metrics['sharpe']:>10.2f} "
          f"{strat_metrics['ann_return']:>12.1%} {strat_metrics['max_dd']:>12.1%}")

    print(f"\n   Average exposure: {weights.mean():.1%}")
    print(f"   % time fully out: {(weights == 0).mean():.1%}")

    # Plot
    print("\n4. Generating plots...")
    plot_market_timing(
        strategy_returns, benchmark_returns, weights,
        OUTPUT_DIR / 'fig15_market_timing.png'
    )

    # Sensitivity analysis
    run_sensitivity_analysis(factor_returns, signals)

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
