"""
Baseline Comparison: Hyperbolic vs Linear vs Exponential Decay

Key question: Does our model actually beat naive alternatives?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

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


# ===== MODELS =====

def hyperbolic_decay(t, K, lam):
    """α(t) = K / (1 + λt)"""
    return K / (1 + lam * t)


def linear_decay(t, a, b):
    """α(t) = a - b*t"""
    return a - b * t


def exponential_decay(t, K, lam):
    """α(t) = K * exp(-λt)"""
    return K * np.exp(-lam * t)


def fit_and_evaluate(t, y, model_func, p0, model_name):
    """Fit model and return metrics."""
    try:
        # Filter positive values for fitting (same as main analysis)
        mask = y > 0
        t_pos, y_pos = t[mask], y[mask]

        if len(t_pos) < 10:
            return None

        popt, _ = curve_fit(model_func, t_pos, y_pos, p0=p0, maxfev=10000)

        # Predict on all data
        y_pred = model_func(t, *popt)

        # Metrics on positive values only (fair comparison)
        r2 = r2_score(y_pos, model_func(t_pos, *popt))
        rmse = np.sqrt(mean_squared_error(y_pos, model_func(t_pos, *popt)))

        return {
            'model': model_name,
            'params': popt,
            'r2': r2,
            'rmse': rmse,
            'y_pred': y_pred,
        }
    except Exception as e:
        print(f"  {model_name} fit failed: {e}")
        return None


def compare_models_single_factor(factors, factor_name='Mom', start_date='1995-01-01'):
    """Compare all models on a single factor."""

    sharpe = rolling_sharpe(factors[factor_name]).dropna()
    sharpe = sharpe[sharpe.index >= start_date]

    t = np.arange(len(sharpe))
    y = sharpe.values

    results = {}

    # Hyperbolic
    res = fit_and_evaluate(t, y, hyperbolic_decay, [1.5, 0.01], 'Hyperbolic')
    if res:
        results['hyperbolic'] = res

    # Linear
    res = fit_and_evaluate(t, y, linear_decay, [1.0, 0.001], 'Linear')
    if res:
        results['linear'] = res

    # Exponential
    res = fit_and_evaluate(t, y, exponential_decay, [1.5, 0.005], 'Exponential')
    if res:
        results['exponential'] = res

    return results, sharpe


def compare_all_factors(factors):
    """Compare models across all factors."""

    factor_list = ['Mom', 'HML', 'SMB']
    all_results = {}

    print("\n" + "=" * 80)
    print("BASELINE COMPARISON: HYPERBOLIC vs LINEAR vs EXPONENTIAL")
    print("=" * 80)

    for factor in factor_list:
        if factor not in factors.columns:
            continue

        print(f"\n{factor}:")
        results, _ = compare_models_single_factor(factors, factor)
        all_results[factor] = results

        # Print comparison
        print(f"  {'Model':<15} {'R²':>10} {'RMSE':>10}")
        print(f"  {'-'*35}")
        for model_name, res in sorted(results.items(), key=lambda x: -x[1]['r2']):
            print(f"  {res['model']:<15} {res['r2']:>10.3f} {res['rmse']:>10.3f}")

    return all_results


def create_comparison_table(all_results):
    """Create table for paper."""

    print("\n" + "=" * 80)
    print("TABLE FOR PAPER (LaTeX format)")
    print("=" * 80)

    print(r"""
\begin{table}[h]
\centering
\caption{Model comparison: In-sample fit (1995-2024)}
\label{tab:baselines}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{Hyperbolic} & \multicolumn{2}{c}{Linear} & \multicolumn{2}{c}{Exponential} \\
Factor & $R^2$ & RMSE & $R^2$ & RMSE & $R^2$ & RMSE \\
\midrule""")

    for factor, results in all_results.items():
        hyp = results.get('hyperbolic', {'r2': 0, 'rmse': 0})
        lin = results.get('linear', {'r2': 0, 'rmse': 0})
        exp = results.get('exponential', {'r2': 0, 'rmse': 0})

        # Bold the best R²
        best_r2 = max(hyp['r2'], lin['r2'], exp['r2'])

        def fmt(val, is_best):
            if is_best:
                return f"\\textbf{{{val:.2f}}}"
            return f"{val:.2f}"

        print(f"{factor} & {fmt(hyp['r2'], hyp['r2']==best_r2)} & {hyp['rmse']:.2f} "
              f"& {fmt(lin['r2'], lin['r2']==best_r2)} & {lin['rmse']:.2f} "
              f"& {fmt(exp['r2'], exp['r2']==best_r2)} & {exp['rmse']:.2f} \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")


def plot_model_comparison(factors, factor_name='Mom', save_path=None):
    """Visual comparison of models."""

    results, sharpe = compare_models_single_factor(factors, factor_name)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Actual data
    ax.plot(sharpe.index, sharpe.values, 'b-', alpha=0.6, linewidth=1, label='Actual')

    # Models
    colors = {'hyperbolic': 'red', 'linear': 'green', 'exponential': 'orange'}
    styles = {'hyperbolic': '-', 'linear': '--', 'exponential': ':'}

    for model_name, res in results.items():
        label = f"{res['model']} (R²={res['r2']:.2f})"
        ax.plot(sharpe.index, res['y_pred'],
                color=colors[model_name], linestyle=styles[model_name],
                linewidth=2, label=label)

    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rolling 36-Month Sharpe', fontsize=12)
    ax.set_title(f'{factor_name}: Model Comparison\nHyperbolic vs Linear vs Exponential',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_cross_factor_panel(factors, save_path=None):
    """Panel plot comparing model fit across factors."""

    factor_list = ['Mom', 'HML', 'SMB']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, factor in zip(axes, factor_list):
        results, sharpe = compare_models_single_factor(factors, factor)

        # Actual
        ax.plot(sharpe.index, sharpe.values, 'b-', alpha=0.5, linewidth=1, label='Actual')

        # Hyperbolic (our model)
        if 'hyperbolic' in results:
            res = results['hyperbolic']
            ax.plot(sharpe.index, res['y_pred'], 'r-', linewidth=2,
                    label=f'Hyperbolic (R²={res["r2"]:.2f})')

        # Linear baseline
        if 'linear' in results:
            res = results['linear']
            ax.plot(sharpe.index, res['y_pred'], 'g--', linewidth=2,
                    label=f'Linear (R²={res["r2"]:.2f})')

        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.set_title(f'{factor}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Rolling Sharpe')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Model Comparison Across Factors: Hyperbolic vs Linear Baseline',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def main():
    print("Loading data...")
    factors = load_data()

    # Compare all models
    all_results = compare_all_factors(factors)

    # Create LaTeX table
    create_comparison_table(all_results)

    # Generate figures
    print("\nGenerating figures...")
    plot_model_comparison(factors, 'Mom', OUTPUT_DIR / 'fig9_model_comparison.png')
    plot_cross_factor_panel(factors, OUTPUT_DIR / 'fig10_cross_factor_panel.png')

    # Key insight
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    mom_results = all_results.get('Mom', {})
    if 'hyperbolic' in mom_results and 'linear' in mom_results:
        hyp_r2 = mom_results['hyperbolic']['r2']
        lin_r2 = mom_results['linear']['r2']
        improvement = hyp_r2 - lin_r2
        print(f"\nMomentum:")
        print(f"  Hyperbolic R² = {hyp_r2:.3f}")
        print(f"  Linear R²     = {lin_r2:.3f}")
        print(f"  Improvement   = {improvement:+.3f} ({100*improvement/lin_r2:.1f}% relative)")

        if improvement > 0.05:
            print("  → Hyperbolic model provides meaningful improvement over linear baseline")
        else:
            print("  → Hyperbolic model offers marginal improvement")

    print("\nDone!")


if __name__ == '__main__':
    main()
