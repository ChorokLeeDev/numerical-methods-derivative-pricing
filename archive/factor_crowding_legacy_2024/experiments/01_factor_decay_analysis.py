"""
Factor Decay Empirical Analysis

Research Question:
- Do factor returns decay over time as more capital discovers them?
- Can we model this decay with a game-theoretic equilibrium?

Data:
- Fama-French factor returns (1990-2024)
- Factor ETF AUM as crowding proxy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_DIR = Path(__file__).parent.parent / 'paper' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. Load Data
# =============================================================================

def load_factor_data():
    """Load Fama-French factor returns."""
    factors_daily = pd.read_parquet(DATA_DIR / 'ff_factors_daily.parquet')
    factors_monthly = pd.read_parquet(DATA_DIR / 'ff_factors_monthly.parquet')

    # Convert Period index to Timestamp if needed
    if isinstance(factors_monthly.index, pd.PeriodIndex):
        factors_monthly.index = factors_monthly.index.to_timestamp()
    if isinstance(factors_daily.index, pd.PeriodIndex):
        factors_daily.index = factors_daily.index.to_timestamp()

    return factors_daily, factors_monthly


def load_etf_data():
    """Load ETF data for crowding proxy."""
    prices = pd.read_parquet(DATA_DIR / 'etf_prices.parquet')
    volumes = pd.read_parquet(DATA_DIR / 'etf_dollar_volumes.parquet')
    aum = pd.read_parquet(DATA_DIR / 'etf_aum_current.parquet')

    return prices, volumes, aum


# =============================================================================
# 2. Compute Rolling Performance Metrics
# =============================================================================

def rolling_sharpe(returns, window=36, annualize=12):
    """
    Compute rolling Sharpe ratio.

    Args:
        returns: Monthly returns series
        window: Rolling window in months
        annualize: Annualization factor (12 for monthly)
    """
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()

    sharpe = (rolling_mean / rolling_std) * np.sqrt(annualize)
    return sharpe


def rolling_cumulative_return(returns, window=36):
    """Compute rolling cumulative return."""
    return returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)


def compute_factor_metrics(factors_monthly, window=36):
    """Compute performance metrics for all factors."""
    metrics = {}

    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
    available_cols = [c for c in factor_cols if c in factors_monthly.columns]

    for factor in available_cols:
        returns = factors_monthly[factor]
        metrics[factor] = {
            'sharpe': rolling_sharpe(returns, window),
            'cum_return': rolling_cumulative_return(returns, window),
            'volatility': returns.rolling(window).std() * np.sqrt(12),
        }

    return metrics


# =============================================================================
# 3. Visualize Factor Decay
# =============================================================================

def plot_factor_decay(metrics, save_path=None):
    """
    Figure 1: Rolling Sharpe ratio showing factor decay over time.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    factors_to_plot = ['HML', 'Mom', 'SMB', 'Mkt-RF']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    titles = ['Value (HML)', 'Momentum', 'Size (SMB)', 'Market']

    for ax, factor, color, title in zip(axes.flat, factors_to_plot, colors, titles):
        if factor in metrics:
            sharpe = metrics[factor]['sharpe'].dropna()

            # Filter to 1995+ for cleaner visualization
            sharpe = sharpe[sharpe.index >= '1995-01-01']

            ax.plot(sharpe.index, sharpe.values, color=color, linewidth=1.5)
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)

            # Add rolling mean
            rolling_mean = sharpe.rolling(12).mean()
            ax.plot(rolling_mean.index, rolling_mean.values, color=color,
                    linewidth=3, alpha=0.7, label='12-month MA')

            # Mark key events
            events = {
                '1997-01-01': 'Asian Crisis',
                '2000-03-01': 'Dot-com Peak',
                '2008-09-01': 'Financial Crisis',
                '2020-03-01': 'COVID Crash',
            }
            for date, label in events.items():
                if pd.to_datetime(date) in sharpe.index:
                    ax.axvline(pd.to_datetime(date), color='gray',
                               linestyle=':', alpha=0.5)

            ax.set_title(f'{title}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Rolling 36M Sharpe Ratio')
            ax.set_ylim(-2, 4)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Factor Performance Decay Over Time (1995-2024)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_decade_comparison(factors_monthly, save_path=None):
    """
    Figure 2: Compare factor Sharpe ratios by decade.
    """
    # Define decades
    decades = {
        '1990s': ('1990-01-01', '1999-12-31'),
        '2000s': ('2000-01-01', '2009-12-31'),
        '2010s': ('2010-01-01', '2019-12-31'),
        '2020s': ('2020-01-01', '2024-12-31'),
    }

    factors = ['HML', 'Mom', 'SMB']
    results = {f: {} for f in factors}

    for decade, (start, end) in decades.items():
        mask = (factors_monthly.index >= start) & (factors_monthly.index <= end)
        subset = factors_monthly[mask]

        for factor in factors:
            if factor in subset.columns:
                returns = subset[factor]
                sharpe = returns.mean() / returns.std() * np.sqrt(12)
                results[factor][decade] = sharpe

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(decades))
    width = 0.25

    for i, (factor, color) in enumerate(zip(factors, ['#e41a1c', '#377eb8', '#4daf4a'])):
        values = [results[factor].get(d, 0) for d in decades.keys()]
        ax.bar(x + i * width, values, width, label=factor, color=color, alpha=0.8)

    ax.set_xlabel('Decade', fontsize=12)
    ax.set_ylabel('Annualized Sharpe Ratio', fontsize=12)
    ax.set_title('Factor Sharpe Ratios by Decade: Evidence of Decay',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(decades.keys())
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, results


# =============================================================================
# 4. Game-Theoretic Model
# =============================================================================

def alpha_decay_model(t, K, lam):
    """
    Simple alpha decay model.

    α(t) = K / (1 + λ × t)

    Args:
        t: Time (normalized)
        K: Initial alpha capacity
        lam: Decay rate

    Returns:
        Expected alpha at time t
    """
    return K / (1 + lam * t)


def alpha_decay_exponential(t, K, lam):
    """
    Exponential decay model.

    α(t) = K × exp(-λ × t)
    """
    return K * np.exp(-lam * t)


def fit_decay_model(sharpe_series, model='hyperbolic'):
    """
    Fit decay model to rolling Sharpe ratio.

    Returns:
        Fitted parameters (K, λ) and R² score
    """
    # Clean data
    sharpe = sharpe_series.dropna()
    sharpe = sharpe[sharpe.index >= '1995-01-01']

    # Normalize time
    t = np.arange(len(sharpe))
    y = sharpe.values

    # Remove negative values for fitting (optional)
    mask = y > 0
    t_pos, y_pos = t[mask], y[mask]

    if len(t_pos) < 10:
        return None, None, None

    try:
        if model == 'hyperbolic':
            popt, _ = curve_fit(alpha_decay_model, t_pos, y_pos,
                                p0=[1.0, 0.01], maxfev=5000)
            y_pred = alpha_decay_model(t_pos, *popt)
        else:
            popt, _ = curve_fit(alpha_decay_exponential, t_pos, y_pos,
                                p0=[1.0, 0.01], maxfev=5000)
            y_pred = alpha_decay_exponential(t_pos, *popt)

        # R² score
        ss_res = np.sum((y_pos - y_pred) ** 2)
        ss_tot = np.sum((y_pos - np.mean(y_pos)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        return popt, r2, (t_pos, y_pos, y_pred)

    except Exception as e:
        print(f"Fitting failed: {e}")
        return None, None, None


def plot_model_fit(metrics, save_path=None):
    """
    Figure 3: Game-theoretic model fit to factor decay.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    factors = ['HML', 'Mom', 'SMB']
    titles = ['Value (HML)', 'Momentum', 'Size (SMB)']

    for ax, factor, title in zip(axes, factors, titles):
        if factor not in metrics:
            continue

        sharpe = metrics[factor]['sharpe']
        result = fit_decay_model(sharpe, model='hyperbolic')

        if result[0] is None:
            continue

        popt, r2, (t, y, y_pred) = result
        K, lam = popt

        # Plot actual
        ax.scatter(t, y, alpha=0.3, s=10, label='Actual', color='blue')

        # Plot fitted
        t_smooth = np.linspace(0, max(t), 100)
        y_smooth = alpha_decay_model(t_smooth, K, lam)
        ax.plot(t_smooth, y_smooth, 'r-', linewidth=2,
                label=f'Fitted: K={K:.2f}, λ={lam:.4f}')

        ax.set_title(f'{title}\n$R^2$ = {r2:.3f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (months since 1995)')
        ax.set_ylabel('Rolling Sharpe')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Game-Theoretic Model Fit: α(t) = K / (1 + λt)',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# 5. Crowding Proxy Analysis
# =============================================================================

def analyze_crowding_vs_alpha(factors_monthly, etf_volumes, save_path=None):
    """
    Figure 4: Crowding proxy vs factor alpha.
    """
    # Map ETFs to factors
    factor_etf_map = {
        'HML': ['VTV', 'IWD', 'VLUE'],  # Value ETFs
        'Mom': ['MTUM', 'PDP'],  # Momentum ETFs
        'SMB': ['IWM', 'VB'],  # Small cap ETFs
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (factor, etfs) in zip(axes, factor_etf_map.items()):
        # Get factor returns
        if factor not in factors_monthly.columns:
            continue

        factor_returns = factors_monthly[factor]

        # Get total dollar volume for factor's ETFs
        available_etfs = [e for e in etfs if e in etf_volumes.columns]
        if not available_etfs:
            continue

        etf_vol = etf_volumes[available_etfs].sum(axis=1)

        # Align dates
        common_idx = factor_returns.index.intersection(etf_vol.index)
        if len(common_idx) < 50:
            continue

        # Compute monthly returns and volumes
        factor_monthly = factor_returns.resample('M').apply(lambda x: (1+x).prod() - 1)
        etf_monthly = etf_vol.resample('M').sum()

        common_idx = factor_monthly.index.intersection(etf_monthly.index)
        factor_aligned = factor_monthly.loc[common_idx]
        etf_aligned = etf_monthly.loc[common_idx]

        # Rolling correlation
        window = 12
        rolling_corr = factor_aligned.rolling(window).corr(etf_aligned)

        ax.plot(rolling_corr.index, rolling_corr.values, linewidth=1.5)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_title(f'{factor} Returns vs ETF Volume\nCorrelation',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Rolling 12M Correlation')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Factor Returns vs Crowding Proxy (ETF Dollar Volume)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# 6. Summary Statistics
# =============================================================================

def print_summary_stats(factors_monthly, metrics):
    """Print summary statistics for the paper."""
    print("\n" + "=" * 70)
    print("FACTOR DECAY: SUMMARY STATISTICS")
    print("=" * 70)

    # By decade
    decades = {
        '1990s': ('1990-01-01', '1999-12-31'),
        '2000s': ('2000-01-01', '2009-12-31'),
        '2010s': ('2010-01-01', '2019-12-31'),
        '2020s': ('2020-01-01', '2024-12-31'),
    }

    factors = ['HML', 'Mom', 'SMB', 'Mkt-RF']

    print("\nAnnualized Sharpe Ratio by Decade:")
    print("-" * 50)

    header = f"{'Factor':<10}" + "".join([f"{d:<12}" for d in decades.keys()])
    print(header)
    print("-" * 50)

    for factor in factors:
        if factor not in factors_monthly.columns:
            continue

        row = f"{factor:<10}"
        for decade, (start, end) in decades.items():
            mask = (factors_monthly.index >= start) & (factors_monthly.index <= end)
            returns = factors_monthly[mask][factor]
            sharpe = returns.mean() / returns.std() * np.sqrt(12)
            row += f"{sharpe:>10.2f}  "
        print(row)

    # Model fit results
    print("\n\nModel Fit Results: α(t) = K / (1 + λt)")
    print("-" * 50)
    print(f"{'Factor':<10}{'K':>10}{'λ':>12}{'R²':>10}")
    print("-" * 50)

    for factor in ['HML', 'Mom', 'SMB']:
        if factor not in metrics:
            continue

        result = fit_decay_model(metrics[factor]['sharpe'])
        if result[0] is not None:
            K, lam = result[0]
            r2 = result[1]
            print(f"{factor:<10}{K:>10.3f}{lam:>12.5f}{r2:>10.3f}")

    print("\n" + "=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    print("Loading data...")
    factors_daily, factors_monthly = load_factor_data()
    prices, volumes, aum = load_etf_data()

    print(f"Factor data: {len(factors_monthly)} months")
    print(f"ETF data: {len(prices)} days")

    print("\nComputing metrics...")
    metrics = compute_factor_metrics(factors_monthly)

    print("\nGenerating figures...")

    # Figure 1: Factor decay over time
    plot_factor_decay(metrics, OUTPUT_DIR / 'fig1_factor_decay.png')

    # Figure 2: Decade comparison
    plot_decade_comparison(factors_monthly, OUTPUT_DIR / 'fig2_decade_comparison.png')

    # Figure 3: Model fit
    plot_model_fit(metrics, OUTPUT_DIR / 'fig3_model_fit.png')

    # Figure 4: Crowding analysis
    analyze_crowding_vs_alpha(factors_monthly, volumes, OUTPUT_DIR / 'fig4_crowding.png')

    # Summary stats
    print_summary_stats(factors_monthly, metrics)

    print("\nDone! Figures saved to:", OUTPUT_DIR)


if __name__ == '__main__':
    main()
