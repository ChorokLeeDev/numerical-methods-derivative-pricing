"""
Residual Analysis: Crowding Acceleration Detection

The prediction gap (actual - predicted) reveals accelerating crowding.
Model over-predicts = something accelerated decay beyond historical rates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

DATA_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_DIR = Path(__file__).parent.parent / 'paper' / 'figures'


def load_data():
    factors = pd.read_parquet(DATA_DIR / 'ff_factors_monthly.parquet')
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    return factors


def rolling_sharpe(returns, window=36):
    return (returns.rolling(window).mean() /
            returns.rolling(window).std() * np.sqrt(12))


def alpha_decay_model(t, K, lam):
    """α(t) = K / (1 + λt)"""
    return K / (1 + lam * t)


def compute_residuals(factors, factor_name='Mom',
                      train_end='2015-12-31',
                      test_start='2016-01-01'):
    """
    Compute prediction residuals for crowding analysis.
    """
    # Compute rolling Sharpe
    sharpe = rolling_sharpe(factors[factor_name]).dropna()
    sharpe = sharpe[sharpe.index >= '1995-01-01']

    # Split train/test
    train = sharpe[sharpe.index <= train_end]
    test = sharpe[sharpe.index >= test_start]

    # Prepare training data
    t_train = np.arange(len(train))
    y_train = train.values
    mask = y_train > 0
    t_train_pos, y_train_pos = t_train[mask], y_train[mask]

    # Fit model
    popt, _ = curve_fit(alpha_decay_model, t_train_pos, y_train_pos,
                        p0=[1.5, 0.01], maxfev=5000)
    K, lam = popt

    # Predict test period
    t_test_start = len(train)
    t_test = np.arange(t_test_start, t_test_start + len(test))
    y_pred = alpha_decay_model(t_test, K, lam)
    y_actual = test.values

    # Compute residuals
    residuals = y_actual - y_pred

    return {
        'test_index': test.index,
        'actual': y_actual,
        'predicted': y_pred,
        'residuals': residuals,
        'K': K,
        'lambda': lam,
    }


def plot_residuals(results, save_path=None):
    """
    Figure: Prediction residuals showing crowding acceleration.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

    test_index = results['test_index']
    actual = results['actual']
    predicted = results['predicted']
    residuals = results['residuals']

    # Top panel: Actual vs Predicted
    ax1 = axes[0]
    ax1.plot(test_index, actual, 'g-', linewidth=2, label='Actual')
    ax1.plot(test_index, predicted, 'r--', linewidth=2, label='Predicted')
    ax1.fill_between(test_index, actual, predicted, alpha=0.3, color='orange',
                     label='Prediction Gap')
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_ylabel('Rolling 36-Month Sharpe', fontsize=12)
    ax1.set_title('Momentum Factor: Out-of-Sample Prediction (2016-2024)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Residuals
    ax2 = axes[1]
    ax2.plot(test_index, residuals, 'b-', linewidth=1.5)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.7)
    ax2.fill_between(test_index, residuals, 0,
                     where=(residuals < 0), alpha=0.4, color='red',
                     label='Faster decay than predicted')
    ax2.fill_between(test_index, residuals, 0,
                     where=(residuals >= 0), alpha=0.4, color='green',
                     label='Slower decay than predicted')
    ax2.set_ylabel('Residual\n(Actual - Predicted)', fontsize=11)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title('Crowding Acceleration: Model Under-predicts Decay Post-2015',
                  fontsize=12)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Stats annotation
    mean_residual = np.mean(residuals)
    pct_negative = 100 * np.sum(residuals < 0) / len(residuals)
    ax2.text(0.98, 0.95,
             f'Mean residual: {mean_residual:.3f}\n'
             f'% negative: {pct_negative:.0f}%',
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_cumulative_residual(results, save_path=None):
    """
    Cumulative residual - shows persistent over-prediction.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    test_index = results['test_index']
    residuals = results['residuals']
    cumulative = np.cumsum(residuals)

    ax.plot(test_index, cumulative, 'b-', linewidth=2)
    ax.fill_between(test_index, cumulative, 0, alpha=0.3)
    ax.axhline(0, color='black', linestyle='--')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Residual', fontsize=12)
    ax.set_title('Cumulative Prediction Error: Persistent Over-estimation of Alpha',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.text(0.02, 0.05,
            'Declining cumulative residual indicates\n'
            'systematic over-prediction of remaining alpha.\n'
            'Interpretation: Crowding accelerated beyond\n'
            'historical rates after 2015.',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def main():
    print("Loading data...")
    factors = load_data()

    print("\nComputing residuals for momentum...")
    results = compute_residuals(factors, 'Mom')

    # Summary stats
    print(f"\n{'='*50}")
    print("RESIDUAL ANALYSIS: CROWDING ACCELERATION")
    print(f"{'='*50}")
    print(f"Model: α(t) = {results['K']:.2f} / (1 + {results['lambda']:.4f}t)")
    print(f"Mean predicted Sharpe: {np.mean(results['predicted']):.3f}")
    print(f"Mean actual Sharpe: {np.mean(results['actual']):.3f}")
    print(f"Mean residual: {np.mean(results['residuals']):.3f}")
    print(f"% periods with negative residual: "
          f"{100 * np.sum(results['residuals'] < 0) / len(results['residuals']):.1f}%")

    print("\nInterpretation:")
    print("- Negative residual = actual < predicted = faster decay than expected")
    print("- Persistent negative residuals indicate crowding acceleration")
    print("- This coincides with ETF proliferation and commission-free trading")

    # Generate figures
    print("\nGenerating figures...")
    plot_residuals(results, OUTPUT_DIR / 'fig7_residuals.png')
    plot_cumulative_residual(results, OUTPUT_DIR / 'fig8_cumulative_residual.png')

    print("\nDone!")


if __name__ == '__main__':
    main()
