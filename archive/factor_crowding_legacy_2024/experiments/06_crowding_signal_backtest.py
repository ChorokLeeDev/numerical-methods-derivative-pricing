"""
Crowding Signal Backtest

Full pipeline:
1. Load factor data
2. Compute rolling crowding signals
3. Backtest factor timing strategies
4. Compare vs benchmarks
5. Generate figures for paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from crowding_signal import CrowdingDetector, compute_crowding_score
from factor_timing import (
    FactorTimingStrategy,
    run_full_backtest,
    print_backtest_summary
)

DATA_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_DIR = Path(__file__).parent.parent / 'paper' / 'figures'


def load_data():
    """Load factor returns."""
    factors = pd.read_parquet(DATA_DIR / 'ff_factors_monthly.parquet')
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    return factors


def plot_crowding_signals(
    signals: dict,
    save_path: Path = None
):
    """Plot crowding signals over time."""
    n_factors = len(signals)
    fig, axes = plt.subplots(n_factors, 1, figsize=(14, 4 * n_factors), sharex=True)

    if n_factors == 1:
        axes = [axes]

    for ax, (factor, df) in zip(axes, signals.items()):
        # Residual
        ax.plot(df.index, df['residual'], 'b-', alpha=0.7, linewidth=1, label='Residual')

        # Zero line
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)

        # Shade crowding (negative) and uncrowding (positive)
        ax.fill_between(df.index, df['residual'], 0,
                        where=(df['residual'] < 0), alpha=0.3, color='red',
                        label='Crowding (residual < 0)')
        ax.fill_between(df.index, df['residual'], 0,
                        where=(df['residual'] >= 0), alpha=0.3, color='green',
                        label='Uncrowding (residual > 0)')

        # Rolling mean
        if len(df) > 12:
            rolling_mean = df['residual'].rolling(12).mean()
            ax.plot(df.index, rolling_mean, 'k-', linewidth=2, label='12M Rolling Mean')

        ax.set_ylabel('Prediction Residual', fontsize=11)
        ax.set_title(f'{factor}: Crowding Acceleration Signal', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date', fontsize=12)

    plt.suptitle('Real-Time Crowding Detection: Prediction Residual as Signal',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_backtest_comparison(
    results: dict,
    save_path: Path = None
):
    """Plot cumulative returns of strategies."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Panel 1: Cumulative returns
    ax1 = axes[0]
    colors = {
        'equal_weight': 'gray',
        'factor_momentum': 'blue',
        'crowding_timed': 'red',
        'crowding_ranked': 'orange',
        'combined': 'green',
    }

    for name, res in results.items():
        cumulative = (1 + res.returns).cumprod()
        ax1.plot(cumulative.index, cumulative.values,
                 color=colors.get(name, 'black'),
                 linewidth=2 if 'crowding' in name else 1.5,
                 alpha=0.9 if 'crowding' in name else 0.6,
                 label=f"{res.strategy_name} (SR={res.sharpe_ratio:.2f})")

    ax1.set_ylabel('Cumulative Return', fontsize=12)
    ax1.set_title('Strategy Comparison: Crowding-Timed vs Benchmarks',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Panel 2: Rolling Sharpe
    ax2 = axes[1]
    window = 36  # 3-year rolling

    for name, res in results.items():
        if len(res.returns) > window:
            rolling_sharpe = (
                res.returns.rolling(window).mean() /
                res.returns.rolling(window).std() * np.sqrt(12)
            )
            ax2.plot(rolling_sharpe.index, rolling_sharpe.values,
                     color=colors.get(name, 'black'),
                     linewidth=2 if 'crowding' in name else 1.5,
                     alpha=0.9 if 'crowding' in name else 0.6,
                     label=res.strategy_name)

    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Rolling 3Y Sharpe Ratio', fontsize=12)
    ax2.set_title('Rolling Performance: Does Crowding Signal Add Value?',
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_signal_effectiveness(
    factor_returns: pd.DataFrame,
    signals: dict,
    factor: str = 'Mom',
    save_path: Path = None
):
    """
    Show that crowding signal predicts future returns.
    Sort periods by signal, show average next-period return.
    """
    if factor not in signals:
        print(f"Factor {factor} not in signals")
        return None

    sig_df = signals[factor].copy()
    ret_series = factor_returns[factor]

    # Align and get forward returns
    common_idx = sig_df.index.intersection(ret_series.index)
    sig_aligned = sig_df.loc[common_idx, 'residual']
    ret_aligned = ret_series.loc[common_idx]

    # Forward returns (next month)
    fwd_returns = ret_aligned.shift(-1)

    # Create quintile buckets based on signal
    df = pd.DataFrame({
        'signal': sig_aligned,
        'fwd_return': fwd_returns
    }).dropna()

    df['quintile'] = pd.qcut(df['signal'], 5, labels=['Q1\n(Most Crowded)',
                                                       'Q2', 'Q3', 'Q4',
                                                       'Q5\n(Least Crowded)'])

    # Average forward return by quintile
    quintile_returns = df.groupby('quintile')['fwd_return'].mean() * 12  # Annualize

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#d73027', '#fc8d59', '#fee090', '#91cf60', '#1a9850']
    bars = ax.bar(range(5), quintile_returns.values, color=colors, edgecolor='black')

    ax.set_xticks(range(5))
    ax.set_xticklabels(quintile_returns.index, fontsize=11)
    ax.set_ylabel('Annualized Forward Return (%)', fontsize=12)
    ax.set_xlabel('Crowding Signal Quintile', fontsize=12)
    ax.set_title(f'{factor}: Forward Returns by Crowding Signal\n'
                 f'Higher Signal (Less Crowded) → Higher Future Returns',
                 fontsize=14, fontweight='bold')

    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate bars
    for bar, val in zip(bars, quintile_returns.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

    # Spread annotation
    spread = quintile_returns.iloc[-1] - quintile_returns.iloc[0]
    ax.text(0.98, 0.95, f'Q5-Q1 Spread: {spread:.1f}%',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def main():
    print("=" * 70)
    print("CROWDING SIGNAL BACKTEST")
    print("=" * 70)

    # Load data
    print("\n1. Loading factor data...")
    factors = load_data()

    # Select factors for analysis
    factor_cols = ['Mom', 'HML', 'SMB']
    factor_returns = factors[factor_cols].dropna()
    print(f"   Factors: {factor_cols}")
    print(f"   Date range: {factor_returns.index.min()} to {factor_returns.index.max()}")

    # Compute crowding signals
    print("\n2. Computing rolling crowding signals...")
    detector = CrowdingDetector(
        train_window=120,  # 10 years
        prediction_gap=12,  # 1 year holdout
        sharpe_window=36,   # 3 year Sharpe
        signal_threshold=0.10,
    )

    signals = detector.compute_multi_factor_signals(factor_returns)

    for factor, df in signals.items():
        print(f"   {factor}: {len(df)} signal observations")
        print(f"      Mean residual: {df['residual'].mean():.3f}")
        print(f"      % Crowding signals: {100 * (df['signal'] == 'crowding').mean():.1f}%")
        print(f"      % Uncrowding signals: {100 * (df['signal'] == 'uncrowding').mean():.1f}%")

    # Plot signals
    print("\n3. Generating signal plots...")
    plot_crowding_signals(signals, OUTPUT_DIR / 'fig12_crowding_signals.png')

    # Signal effectiveness
    print("\n4. Testing signal effectiveness...")
    plot_signal_effectiveness(factor_returns, signals, 'Mom',
                              OUTPUT_DIR / 'fig13_signal_effectiveness.png')

    # Run backtests
    print("\n5. Running backtests...")
    results = run_full_backtest(factor_returns, signals)

    # Print summary
    print_backtest_summary(results)

    # Plot comparison
    print("\n6. Generating backtest plots...")
    plot_backtest_comparison(results, OUTPUT_DIR / 'fig14_backtest_comparison.png')

    # Summary table for paper
    print("\n" + "=" * 70)
    print("TABLE FOR PAPER (LaTeX)")
    print("=" * 70)
    print(r"""
\begin{table}[h]
\centering
\caption{Factor timing strategy comparison}
\label{tab:backtest}
\begin{tabular}{lcccc}
\toprule
Strategy & Ann. Return & Ann. Vol & Sharpe & Max DD \\
\midrule""")

    for name, res in sorted(results.items(), key=lambda x: -x[1].sharpe_ratio):
        print(f"{res.strategy_name} & {res.annualized_return:.1%} & "
              f"{res.annualized_volatility:.1%} & {res.sharpe_ratio:.2f} & "
              f"{res.max_drawdown:.1%} \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
