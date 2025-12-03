"""
Out-of-Sample Prediction Test

Key question: Can we PREDICT future alpha decay?

Setup:
- Train: 1995-2015 (fit decay model)
- Test: 2016-2024 (predict Sharpe)
- Metric: Does model predict continued momentum decay?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

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


def out_of_sample_test(factors, factor_name='Mom',
                        train_end='2015-12-31',
                        test_start='2016-01-01'):
    """
    Train model on historical data, predict future.
    """
    # Compute rolling Sharpe
    sharpe = rolling_sharpe(factors[factor_name]).dropna()
    sharpe = sharpe[sharpe.index >= '1995-01-01']

    # Split train/test
    train = sharpe[sharpe.index <= train_end]
    test = sharpe[sharpe.index >= test_start]

    # Prepare training data (positive values only for fitting)
    t_train = np.arange(len(train))
    y_train = train.values
    mask = y_train > 0
    t_train_pos, y_train_pos = t_train[mask], y_train[mask]

    # Fit model on training data
    try:
        popt, _ = curve_fit(alpha_decay_model, t_train_pos, y_train_pos,
                            p0=[1.5, 0.01], maxfev=5000)
        K, lam = popt
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None

    # Predict test period
    t_test_start = len(train)
    t_test = np.arange(t_test_start, t_test_start + len(test))
    y_pred = alpha_decay_model(t_test, K, lam)
    y_actual = test.values

    # Metrics
    # Only compare where actual > 0 (avoid division issues)
    valid_mask = y_actual > 0
    if valid_mask.sum() < 10:
        valid_mask = np.ones(len(y_actual), dtype=bool)

    mse = mean_squared_error(y_actual[valid_mask], y_pred[valid_mask])
    rmse = np.sqrt(mse)

    # Directional accuracy: Did we predict decay direction correctly?
    actual_trend = np.polyfit(range(len(y_actual)), y_actual, 1)[0]
    pred_trend = np.polyfit(range(len(y_pred)), y_pred, 1)[0]
    direction_correct = (actual_trend < 0 and pred_trend < 0) or \
                       (actual_trend > 0 and pred_trend > 0)

    results = {
        'K': K,
        'lambda': lam,
        'train_r2': r2_score(y_train_pos, alpha_decay_model(t_train_pos, K, lam)),
        'test_rmse': rmse,
        'test_mean_actual': y_actual.mean(),
        'test_mean_pred': y_pred.mean(),
        'direction_correct': direction_correct,
        'train_dates': (train.index[0], train.index[-1]),
        'test_dates': (test.index[0], test.index[-1]),
        'train_sharpe': train,
        'test_sharpe': test,
        'test_pred': pd.Series(y_pred, index=test.index),
        't_train': t_train,
        't_test': t_test,
    }

    return results


def plot_out_of_sample(results, factor_name, save_path=None):
    """
    Figure: Out-of-sample prediction visualization
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Training period
    train_sharpe = results['train_sharpe']
    ax.plot(train_sharpe.index, train_sharpe.values,
            'b-', linewidth=1.5, alpha=0.7, label='Training (1995-2015)')

    # Test period - actual
    test_sharpe = results['test_sharpe']
    ax.plot(test_sharpe.index, test_sharpe.values,
            'g-', linewidth=2, label='Actual (2016-2024)')

    # Test period - predicted
    test_pred = results['test_pred']
    ax.plot(test_pred.index, test_pred.values,
            'r--', linewidth=2, label='Predicted')

    # Vertical line at split
    split_date = results['train_dates'][1]
    ax.axvline(split_date, color='black', linestyle=':', alpha=0.7)
    ax.text(split_date, ax.get_ylim()[1] * 0.9, ' Train/Test Split',
            fontsize=10, va='top')

    # Zero line
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)

    # Annotations
    K, lam = results['K'], results['lambda']
    ax.text(0.02, 0.98,
            f"Model: α(t) = {K:.2f} / (1 + {lam:.4f}t)\n"
            f"Train R² = {results['train_r2']:.3f}\n"
            f"Test RMSE = {results['test_rmse']:.3f}\n"
            f"Direction Correct: {results['direction_correct']}",
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rolling 36-Month Sharpe Ratio', fontsize=12)
    ax.set_title(f'{factor_name} Factor: Out-of-Sample Prediction\n'
                 f'Train: 1995-2015 → Predict: 2016-2024',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def compare_factors_oos(factors, save_path=None):
    """
    Compare out-of-sample performance across factors.
    """
    factor_list = ['Mom', 'HML', 'SMB']
    results_all = {}

    for factor in factor_list:
        if factor in factors.columns:
            results_all[factor] = out_of_sample_test(factors, factor)

    # Summary table
    print("\n" + "=" * 70)
    print("OUT-OF-SAMPLE PREDICTION RESULTS")
    print("=" * 70)
    print(f"\n{'Factor':<10}{'Train R²':>12}{'Test RMSE':>12}{'Actual Mean':>14}{'Pred Mean':>12}{'Direction':>12}")
    print("-" * 70)

    for factor, res in results_all.items():
        if res:
            print(f"{factor:<10}{res['train_r2']:>12.3f}{res['test_rmse']:>12.3f}"
                  f"{res['test_mean_actual']:>14.3f}{res['test_mean_pred']:>12.3f}"
                  f"{'✓' if res['direction_correct'] else '✗':>12}")

    print("=" * 70)

    # Multi-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (factor, res) in zip(axes, results_all.items()):
        if res is None:
            continue

        # Training
        ax.plot(res['train_sharpe'].index, res['train_sharpe'].values,
                'b-', alpha=0.5, linewidth=1, label='Train')

        # Test actual
        ax.plot(res['test_sharpe'].index, res['test_sharpe'].values,
                'g-', linewidth=2, label='Actual')

        # Test predicted
        ax.plot(res['test_pred'].index, res['test_pred'].values,
                'r--', linewidth=2, label='Predicted')

        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(res['train_dates'][1], color='black', linestyle=':', alpha=0.5)

        direction_symbol = '✓' if res['direction_correct'] else '✗'
        ax.set_title(f"{factor}\nRMSE={res['test_rmse']:.2f}, Dir={direction_symbol}",
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Rolling Sharpe')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Out-of-Sample Prediction: Train 1995-2015 → Test 2016-2024',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, results_all


def main():
    print("Loading data...")
    factors = load_data()

    # Single factor detailed analysis
    print("\n" + "=" * 50)
    print("MOMENTUM OUT-OF-SAMPLE TEST")
    print("=" * 50)

    results_mom = out_of_sample_test(factors, 'Mom')
    if results_mom:
        print(f"\nModel: α(t) = {results_mom['K']:.3f} / (1 + {results_mom['lambda']:.5f}t)")
        print(f"Train R²: {results_mom['train_r2']:.3f}")
        print(f"Test RMSE: {results_mom['test_rmse']:.3f}")
        print(f"Test Actual Mean Sharpe: {results_mom['test_mean_actual']:.3f}")
        print(f"Test Predicted Mean Sharpe: {results_mom['test_mean_pred']:.3f}")
        print(f"Direction Correct: {results_mom['direction_correct']}")

        plot_out_of_sample(results_mom, 'Momentum',
                          OUTPUT_DIR / 'fig5_momentum_oos.png')

    # Compare all factors
    print("\nComparing all factors...")
    compare_factors_oos(factors, OUTPUT_DIR / 'fig6_all_factors_oos.png')

    print("\nDone!")


if __name__ == '__main__':
    main()
