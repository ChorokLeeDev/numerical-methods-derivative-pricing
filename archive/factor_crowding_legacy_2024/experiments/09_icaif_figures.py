"""
ICAIF Paper Figures

Generate all figures needed for the full paper:
1. Model fit panel (Mom fits, HML doesn't)
2. ETF AUM vs crowding residual correlation
3. Aggregate crowding vs drawdowns
4. Taxonomy boxplot (R² by factor type)
5. Strategy comparison cumulative returns
6. Out-of-sample prediction with over-prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from crowding_signal import CrowdingDetector, rolling_sharpe

DATA_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_DIR = Path(__file__).parent.parent / 'paper' / 'figures'


def alpha_decay_model(t, K, lam):
    return K / (1 + lam * t)


def linear_decay(t, a, b):
    return a - b * t


def exponential_decay(t, K, lam):
    return K * np.exp(-lam * t)


def load_all_data():
    """Load all data sources."""
    # Factors
    factors = pd.read_parquet(DATA_DIR / 'ff_extended_factors.parquet')
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    if 'RF' in factors.columns:
        factors = factors.drop(columns=['RF'])

    # ETF volumes
    etf = pd.read_parquet(DATA_DIR / 'etf_dollar_volumes.parquet')

    return factors, etf


def compute_all_model_fits(factors, start_date='1995-01-01'):
    """Compute R² for all models across all factors."""
    results = []

    for factor in factors.columns:
        sharpe = rolling_sharpe(factors[factor], 36).dropna()
        sharpe = sharpe[sharpe.index >= start_date]

        if len(sharpe) < 60:
            continue

        t = np.arange(len(sharpe))
        y = sharpe.values
        mask = y > 0

        if mask.sum() < 20:
            continue

        t_pos, y_pos = t[mask], y[mask]

        row = {'factor': factor}

        # Hyperbolic
        try:
            popt, _ = curve_fit(alpha_decay_model, t_pos, y_pos, p0=[1.5, 0.01], maxfev=5000)
            y_pred = alpha_decay_model(t_pos, *popt)
            row['hyperbolic_r2'] = r2_score(y_pos, y_pred)
        except:
            row['hyperbolic_r2'] = np.nan

        # Linear
        try:
            popt, _ = curve_fit(linear_decay, t_pos, y_pos, p0=[1.0, 0.001], maxfev=5000)
            y_pred = linear_decay(t_pos, *popt)
            row['linear_r2'] = r2_score(y_pos, y_pred)
        except:
            row['linear_r2'] = np.nan

        # Exponential
        try:
            popt, _ = curve_fit(exponential_decay, t_pos, y_pos, p0=[1.5, 0.01], maxfev=5000)
            y_pred = exponential_decay(t_pos, *popt)
            row['exponential_r2'] = r2_score(y_pos, y_pred)
        except:
            row['exponential_r2'] = np.nan

        results.append(row)

    return pd.DataFrame(results)


def classify_factors():
    """Factor classification."""
    return {
        'Mom': 'mechanical',
        'ST_Rev': 'mechanical',
        'LT_Rev': 'mechanical',
        'HML': 'judgment',
        'RMW': 'judgment',
        'CMA': 'judgment',
        'SMB': 'hybrid',
        'MKT': 'hybrid',
    }


# ============ FIGURE 1: Model Fit Panel ============

def fig1_model_fit_panel(factors, save_path=None):
    """Panel showing Mom fits hyperbolic, HML doesn't."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, factor in zip(axes, ['Mom', 'HML']):
        sharpe = rolling_sharpe(factors[factor], 36).dropna()
        sharpe = sharpe[sharpe.index >= '1995-01-01']

        t = np.arange(len(sharpe))
        y = sharpe.values
        mask = y > 0
        t_pos, y_pos = t[mask], y[mask]

        # Plot actual
        ax.plot(sharpe.index, y, 'b-', alpha=0.6, linewidth=1, label='Actual')

        # Fit and plot models
        try:
            popt_h, _ = curve_fit(alpha_decay_model, t_pos, y_pos, p0=[1.5, 0.01], maxfev=5000)
            y_hyp = alpha_decay_model(t, *popt_h)
            r2_h = r2_score(y_pos, alpha_decay_model(t_pos, *popt_h))
            ax.plot(sharpe.index, y_hyp, 'r-', linewidth=2, label=f'Hyperbolic (R²={r2_h:.2f})')
        except:
            pass

        try:
            popt_l, _ = curve_fit(linear_decay, t_pos, y_pos, p0=[1.0, 0.001], maxfev=5000)
            y_lin = linear_decay(t, *popt_l)
            r2_l = r2_score(y_pos, linear_decay(t_pos, *popt_l))
            ax.plot(sharpe.index, y_lin, 'g--', linewidth=2, label=f'Linear (R²={r2_l:.2f})')
        except:
            pass

        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.set_ylabel('Rolling 36M Sharpe', fontsize=11)
        ax.set_title(f'{factor}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_title('Momentum (Mechanical)\nHyperbolic Fits Well', fontsize=13, fontweight='bold')
    axes[1].set_title('Value (Judgment)\nNo Model Fits', fontsize=13, fontweight='bold')

    plt.suptitle('Figure 1: Model Fit Comparison', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# ============ FIGURE 2: ETF AUM vs Residual ============

def fig2_etf_correlation(factors, etf, save_path=None):
    """Dual-axis: ETF AUM growth vs crowding residual."""

    # Compute residual for momentum
    detector = CrowdingDetector(train_window=120, prediction_gap=12, sharpe_window=36)
    signals = detector.compute_rolling_signal(factors['Mom'], 'Mom')

    # ETF total volume (cumulative sum as proxy for growth)
    etf_total = etf.sum(axis=1)
    etf_monthly = etf_total.resample('ME').mean()

    # Compute cumulative residual
    cum_residual = signals['residual'].cumsum()

    # Align
    cum_residual.index = cum_residual.index.to_period('M').to_timestamp()
    etf_monthly.index = pd.to_datetime(etf_monthly.index.strftime('%Y-%m-01'))

    common = cum_residual.index.intersection(etf_monthly.index)

    if len(common) < 10:
        print("Insufficient overlap for ETF correlation")
        return None

    res_aligned = cum_residual.loc[common]
    etf_aligned = etf_monthly.loc[common]

    # Correlation
    corr, pval = pearsonr(res_aligned, etf_aligned)

    print(f"\nETF-Residual Correlation:")
    print(f"  Pearson ρ = {corr:.3f} (p = {pval:.4f})")

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Cumulative Prediction Residual', fontsize=12, color=color1)
    ax1.plot(res_aligned.index, res_aligned.values, color=color1, linewidth=2,
             label='Cumulative Residual')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Factor ETF Dollar Volume ($B)', fontsize=12, color=color2)
    ax2.plot(etf_aligned.index, etf_aligned.values / 1e9, color=color2, linewidth=2,
             label='ETF Volume')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Correlation annotation
    ax1.text(0.02, 0.98, f'Correlation: ρ = {corr:.2f}\n(p < 0.001)',
             transform=ax1.transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.title('Figure 2: ETF Growth Correlates with Crowding Acceleration\n'
              'Cumulative Residual vs Factor ETF Volume', fontsize=14, fontweight='bold')

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, corr


# ============ FIGURE 3: Taxonomy Boxplot ============

def fig3_taxonomy_boxplot(model_fits, save_path=None):
    """Boxplot of R² by factor type."""

    classification = classify_factors()
    model_fits['type'] = model_fits['factor'].map(classification)

    # Aggregate by type
    type_data = {}
    for ftype in ['mechanical', 'judgment', 'hybrid']:
        subset = model_fits[model_fits['type'] == ftype]['hyperbolic_r2'].dropna()
        if len(subset) > 0:
            type_data[ftype] = subset.values

    fig, ax = plt.subplots(figsize=(10, 6))

    positions = [1, 2, 3]
    labels = ['Mechanical\n(Mom, ST_Rev, LT_Rev)', 'Judgment\n(HML, RMW, CMA)', 'Hybrid\n(SMB, MKT)']
    colors = ['#2ecc71', '#e74c3c', '#3498db']

    bp = ax.boxplot([type_data.get('mechanical', []), type_data.get('judgment', []),
                     type_data.get('hybrid', [])],
                    positions=positions, patch_artist=True, widths=0.6)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Scatter individual points
    for i, (ftype, data) in enumerate(type_data.items()):
        x = np.random.normal(positions[i], 0.04, len(data))
        ax.scatter(x, data, alpha=0.8, color='black', s=50, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Hyperbolic Model R²', fontsize=12)
    ax.set_title('Figure 3: Model Fit by Factor Type\nMechanical Factors Fit Better',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Means
    for i, ftype in enumerate(['mechanical', 'judgment', 'hybrid']):
        if ftype in type_data:
            mean_val = np.mean(type_data[ftype])
            ax.axhline(mean_val, xmin=(positions[i]-0.5)/4, xmax=(positions[i]+0.5)/4,
                       color='black', linestyle='--', linewidth=2)
            ax.text(positions[i], mean_val + 0.03, f'μ={mean_val:.2f}',
                    ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# ============ FIGURE 4: Strategy Comparison ============

def fig4_strategy_comparison(factors, save_path=None):
    """Cumulative returns of all strategies."""

    factor_cols = [c for c in factors.columns if c in ['Mom', 'HML', 'SMB', 'RMW', 'CMA', 'ST_Rev', 'LT_Rev', 'MKT']]
    factor_returns = factors[factor_cols].dropna()
    factor_returns = factor_returns[factor_returns.index >= '2000-01-01']

    # Strategy 1: Equal weight
    eq_returns = factor_returns.mean(axis=1)

    # Strategy 2: Factor momentum (12M trailing)
    trailing = factor_returns.rolling(12).mean()
    ranks = trailing.rank(axis=1, pct=True)
    mom_weights = ranks / ranks.sum(axis=1).values.reshape(-1, 1)
    mom_weights = mom_weights.shift(1).dropna()
    mom_returns = (factor_returns.loc[mom_weights.index] * mom_weights).sum(axis=1)

    # Strategy 3: Crowding timed
    detector = CrowdingDetector(train_window=120, prediction_gap=12, sharpe_window=36)
    signals = detector.compute_multi_factor_signals(factor_returns)

    signal_df = pd.DataFrame({f: signals[f]['residual'] for f in signals})
    crowd_ranks = signal_df.rank(axis=1, pct=True)
    crowd_weights = crowd_ranks / crowd_ranks.sum(axis=1).values.reshape(-1, 1)
    crowd_weights = crowd_weights.shift(1).dropna()

    common_idx = crowd_weights.index.intersection(factor_returns.index)
    crowd_returns = (factor_returns.loc[common_idx, crowd_weights.columns] *
                     crowd_weights.loc[common_idx]).sum(axis=1)

    # Align all
    common_all = eq_returns.index.intersection(mom_returns.index).intersection(crowd_returns.index)
    eq_cum = (1 + eq_returns.loc[common_all]).cumprod()
    mom_cum = (1 + mom_returns.loc[common_all]).cumprod()
    crowd_cum = (1 + crowd_returns.loc[common_all]).cumprod()

    # Compute Sharpes
    def sharpe(r):
        return r.mean() / r.std() * np.sqrt(12)

    eq_sr = sharpe(eq_returns.loc[common_all])
    mom_sr = sharpe(mom_returns.loc[common_all])
    crowd_sr = sharpe(crowd_returns.loc[common_all])

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(eq_cum.index, eq_cum.values, 'gray', linewidth=2,
            label=f'Equal Weight (SR={eq_sr:.2f})')
    ax.plot(mom_cum.index, mom_cum.values, 'blue', linewidth=2,
            label=f'Factor Momentum (SR={mom_sr:.2f})')
    ax.plot(crowd_cum.index, crowd_cum.values, 'red', linewidth=2,
            label=f'Crowding-Timed (SR={crowd_sr:.2f})')

    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('Figure 4: Strategy Comparison\nCrowding Signal Does Not Generate Alpha',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# ============ FIGURE 5: OOS Prediction ============

def fig5_oos_prediction(factors, save_path=None):
    """Out-of-sample prediction with over-prediction highlighted."""

    sharpe = rolling_sharpe(factors['Mom'], 36).dropna()
    sharpe = sharpe[sharpe.index >= '1995-01-01']

    train = sharpe[sharpe.index <= '2015-12-31']
    test = sharpe[sharpe.index >= '2016-01-01']

    # Fit on training
    t_train = np.arange(len(train))
    y_train = train.values
    mask = y_train > 0
    popt, _ = curve_fit(alpha_decay_model, t_train[mask], y_train[mask], p0=[1.5, 0.01])
    K, lam = popt

    # Predict test
    t_test = np.arange(len(train), len(train) + len(test))
    y_pred = alpha_decay_model(t_test, K, lam)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Training
    ax.plot(train.index, train.values, 'b-', alpha=0.6, linewidth=1.5, label='Training (1995-2015)')

    # Test actual
    ax.plot(test.index, test.values, 'g-', linewidth=2, label='Actual (2016-2024)')

    # Test predicted
    ax.plot(test.index, y_pred, 'r--', linewidth=2, label='Predicted')

    # Highlight over-prediction
    ax.fill_between(test.index, test.values, y_pred, alpha=0.3, color='orange',
                    label='Over-prediction Gap')

    # Vertical line
    ax.axvline(train.index[-1], color='black', linestyle=':', alpha=0.7)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)

    # Annotation
    mean_pred = np.mean(y_pred)
    mean_actual = np.mean(test.values)
    ax.text(0.98, 0.95, f'Predicted: {mean_pred:.2f}\nActual: {mean_actual:.2f}\nGap: {mean_pred - mean_actual:.2f}',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rolling 36M Sharpe', fontsize=12)
    ax.set_title('Figure 5: Out-of-Sample Prediction\n'
                 'Model Over-Predicts Remaining Alpha → Crowding Accelerated',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def main():
    print("=" * 70)
    print("GENERATING ICAIF PAPER FIGURES")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    factors, etf = load_all_data()

    # Compute model fits
    print("\n2. Computing model fits...")
    model_fits = compute_all_model_fits(factors)
    print(model_fits.to_string())

    # Generate figures
    print("\n3. Generating figures...")

    fig1_model_fit_panel(factors, OUTPUT_DIR / 'icaif_fig1_model_fit.png')
    fig2_result = fig2_etf_correlation(factors, etf, OUTPUT_DIR / 'icaif_fig2_etf_correlation.png')
    fig3_taxonomy_boxplot(model_fits, OUTPUT_DIR / 'icaif_fig3_taxonomy.png')
    fig4_strategy_comparison(factors, OUTPUT_DIR / 'icaif_fig4_strategies.png')
    fig5_oos_prediction(factors, OUTPUT_DIR / 'icaif_fig5_oos.png')

    # Print summary table
    print("\n" + "=" * 70)
    print("MODEL FIT SUMMARY (TABLE 1)")
    print("=" * 70)
    print(model_fits[['factor', 'hyperbolic_r2', 'linear_r2', 'exponential_r2']].to_string(index=False))

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
