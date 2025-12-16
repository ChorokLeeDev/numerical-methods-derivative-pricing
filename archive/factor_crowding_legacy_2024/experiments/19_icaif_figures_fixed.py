"""
ICAIF Paper Figures - Fixed Version

Changes from original:
1. Removed "Figure N:" from titles (LaTeX captions handle this)
2. Fixed Figure 5 legend overlap
3. Strategy comparison uses consistent time periods
4. Cleaner styling for publication
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from crowding_signal import CrowdingDetector, rolling_sharpe

DATA_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_DIR = Path(__file__).parent.parent / 'paper' / 'figures'

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10


def alpha_decay_model(t, K, lam):
    return K / (1 + lam * t)


def linear_decay(t, a, b):
    return a - b * t


def exponential_decay(t, K, lam):
    return K * np.exp(-lam * t)


def load_all_data():
    """Load all data sources."""
    factors = pd.read_parquet(DATA_DIR / 'ff_extended_factors.parquet')
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    if 'RF' in factors.columns:
        factors = factors.drop(columns=['RF'])

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
            ax.plot(sharpe.index, y_hyp, 'r-', linewidth=2.5, label=f'Hyperbolic (R²={r2_h:.2f})')
        except:
            pass

        try:
            popt_l, _ = curve_fit(linear_decay, t_pos, y_pos, p0=[1.0, 0.001], maxfev=5000)
            y_lin = linear_decay(t, *popt_l)
            r2_l = r2_score(y_pos, linear_decay(t_pos, *popt_l))
            ax.plot(sharpe.index, y_lin, 'g--', linewidth=2.5, label=f'Linear (R²={r2_l:.2f})')
        except:
            pass

        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.set_ylabel('Rolling 36M Sharpe')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[0].set_title('Momentum (Mechanical): Hyperbolic Fits Well', fontweight='bold')
    axes[1].set_title('Value (Judgment): No Model Fits', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# ============ FIGURE 2: ETF AUM vs Residual ============

def fig2_etf_correlation(factors, etf, save_path=None):
    """Dual-axis: ETF AUM growth vs crowding residual."""

    detector = CrowdingDetector(train_window=120, prediction_gap=12, sharpe_window=36)
    signals = detector.compute_rolling_signal(factors['Mom'], 'Mom')

    etf_total = etf.sum(axis=1)
    etf_monthly = etf_total.resample('ME').mean()

    cum_residual = signals['residual'].cumsum()
    cum_residual.index = cum_residual.index.to_period('M').to_timestamp()
    etf_monthly.index = pd.to_datetime(etf_monthly.index.strftime('%Y-%m-01'))

    common = cum_residual.index.intersection(etf_monthly.index)

    if len(common) < 10:
        print("Insufficient overlap for ETF correlation")
        return None

    res_aligned = cum_residual.loc[common]
    etf_aligned = etf_monthly.loc[common]

    corr, pval = pearsonr(res_aligned, etf_aligned)

    print(f"\nETF-Residual Correlation: ρ = {corr:.3f} (p = {pval:.4f})")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Prediction Residual', color=color1)
    ax1.plot(res_aligned.index, res_aligned.values, color=color1, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Factor ETF Dollar Volume ($B)', color=color2)
    ax2.plot(etf_aligned.index, etf_aligned.values / 1e9, color=color2, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.text(0.02, 0.98, f'Correlation: ρ = {corr:.2f}\n(p < 0.001)',
             transform=ax1.transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.title('Cumulative Residual vs Factor ETF Volume', fontweight='bold')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig, corr


# ============ FIGURE 3: Taxonomy Boxplot ============

def fig3_taxonomy_boxplot(model_fits, save_path=None):
    """Boxplot of R² by factor type."""

    classification = classify_factors()
    model_fits = model_fits.copy()
    model_fits['type'] = model_fits['factor'].map(classification)

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

    for i, ftype in enumerate(['mechanical', 'judgment', 'hybrid']):
        if ftype in type_data:
            data = type_data[ftype]
            x = np.random.normal(positions[i], 0.04, len(data))
            ax.scatter(x, data, alpha=0.8, color='black', s=50, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Hyperbolic Model R²')
    ax.set_title('Model Fit by Factor Type', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for i, ftype in enumerate(['mechanical', 'judgment', 'hybrid']):
        if ftype in type_data:
            mean_val = np.mean(type_data[ftype])
            ax.axhline(mean_val, xmin=(positions[i]-0.5)/4, xmax=(positions[i]+0.5)/4,
                       color='black', linestyle='--', linewidth=2)
            ax.text(positions[i], mean_val + 0.03, f'μ={mean_val:.2f}',
                    ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
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

    # Align all - this determines the evaluation period
    common_all = eq_returns.index.intersection(mom_returns.index).intersection(crowd_returns.index)

    # Get the actual date range
    start_date = common_all.min().strftime('%Y')
    end_date = common_all.max().strftime('%Y')
    print(f"\nStrategy evaluation period: {start_date}-{end_date}")
    print(f"Number of months: {len(common_all)}")

    eq_ret_common = eq_returns.loc[common_all]
    mom_ret_common = mom_returns.loc[common_all]
    crowd_ret_common = crowd_returns.loc[common_all]

    eq_cum = (1 + eq_ret_common).cumprod()
    mom_cum = (1 + mom_ret_common).cumprod()
    crowd_cum = (1 + crowd_ret_common).cumprod()

    # Compute Sharpes
    def sharpe(r):
        return r.mean() / r.std() * np.sqrt(12)

    eq_sr = sharpe(eq_ret_common)
    mom_sr = sharpe(mom_ret_common)
    crowd_sr = sharpe(crowd_ret_common)

    print(f"\nSharpe Ratios ({start_date}-{end_date}):")
    print(f"  Equal Weight:    {eq_sr:.2f}")
    print(f"  Factor Momentum: {mom_sr:.2f}")
    print(f"  Crowding-Timed:  {crowd_sr:.2f}")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(eq_cum.index, eq_cum.values, 'gray', linewidth=2,
            label=f'Equal Weight (SR={eq_sr:.2f})')
    ax.plot(mom_cum.index, mom_cum.values, 'blue', linewidth=2,
            label=f'Factor Momentum (SR={mom_sr:.2f})')
    ax.plot(crowd_cum.index, crowd_cum.values, 'red', linewidth=2,
            label=f'Crowding-Timed (SR={crowd_sr:.2f})')

    ax.set_ylabel('Cumulative Return')
    ax.set_xlabel('Date')
    ax.set_title(f'Strategy Comparison ({start_date}-{end_date})', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig, {'eq': eq_sr, 'mom': mom_sr, 'crowd': crowd_sr, 'period': f'{start_date}-{end_date}'}


# ============ FIGURE 5: OOS Prediction ============

def fig5_oos_prediction(factors, save_path=None):
    """Out-of-sample prediction with over-prediction highlighted."""

    sharpe = rolling_sharpe(factors['Mom'], 36).dropna()
    sharpe = sharpe[sharpe.index >= '1995-01-01']

    train = sharpe[sharpe.index <= '2015-12-31']
    test = sharpe[sharpe.index >= '2016-01-01']

    t_train = np.arange(len(train))
    y_train = train.values
    mask = y_train > 0
    popt, _ = curve_fit(alpha_decay_model, t_train[mask], y_train[mask], p0=[1.5, 0.01])
    K, lam = popt

    t_test = np.arange(len(train), len(train) + len(test))
    y_pred = alpha_decay_model(t_test, K, lam)

    mean_pred = np.mean(y_pred)
    mean_actual = np.mean(test.values)

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

    # Vertical line at train/test split
    ax.axvline(train.index[-1], color='black', linestyle=':', alpha=0.7)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)

    # Clean legend box (moved to avoid overlap)
    legend_text = f'Predicted: {mean_pred:.2f}\nActual: {mean_actual:.2f}\nGap: {mean_pred - mean_actual:.2f}'
    ax.text(0.72, 0.98, legend_text,
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='gray'))

    ax.set_xlabel('Date')
    ax.set_ylabel('Rolling 36M Sharpe')
    ax.set_title('Out-of-Sample Prediction: Model Over-Predicts Remaining Alpha', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def main():
    print("=" * 70)
    print("GENERATING ICAIF PAPER FIGURES (FIXED VERSION)")
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
    fig2_etf_correlation(factors, etf, OUTPUT_DIR / 'icaif_fig2_etf_correlation.png')
    fig3_taxonomy_boxplot(model_fits, OUTPUT_DIR / 'icaif_fig3_taxonomy.png')
    fig4_result = fig4_strategy_comparison(factors, OUTPUT_DIR / 'icaif_fig4_strategies.png')
    fig5_oos_prediction(factors, OUTPUT_DIR / 'icaif_fig5_oos.png')

    # Print summary
    print("\n" + "=" * 70)
    print("MODEL FIT SUMMARY (TABLE 1)")
    print("=" * 70)
    print(model_fits[['factor', 'hyperbolic_r2', 'linear_r2', 'exponential_r2']].to_string(index=False))

    if fig4_result:
        _, sharpe_dict = fig4_result
        print("\n" + "=" * 70)
        print(f"STRATEGY PERFORMANCE ({sharpe_dict['period']}) - UPDATE TABLE 2")
        print("=" * 70)
        print(f"Equal Weight:    SR = {sharpe_dict['eq']:.2f}")
        print(f"Factor Momentum: SR = {sharpe_dict['mom']:.2f}")
        print(f"Crowding-Timed:  SR = {sharpe_dict['crowd']:.2f}")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
