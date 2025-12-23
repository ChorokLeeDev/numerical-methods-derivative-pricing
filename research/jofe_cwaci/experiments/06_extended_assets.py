"""
Extended Asset Classes Experiment for CW-ACI Paper
===================================================
Tests CW-ACI on:
1. Cryptocurrency (BTC, ETH) - high volatility regime
2. S&P 500 Sector ETFs - equity market

This strengthens the paper by showing generalization beyond Fama-French factors.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Data Collection
# ============================================================================

def fetch_crypto_data(symbols=['BTC-USD', 'ETH-USD'], start='2017-01-01', end='2025-12-01'):
    """Fetch cryptocurrency daily returns"""
    print(f"Fetching crypto data: {symbols}")

    data = {}
    for symbol in symbols:
        try:
            df = yf.download(symbol, start=start, end=end, progress=False)
            if len(df) > 0:
                # Handle both old and new yfinance API
                if 'Adj Close' in df.columns:
                    prices = df['Adj Close']
                elif 'Close' in df.columns:
                    prices = df['Close']
                else:
                    # Multi-level columns
                    prices = df['Close'][symbol] if ('Close', symbol) in df.columns or 'Close' in df.columns.get_level_values(0) else df.iloc[:, 0]

                # Flatten if needed
                if hasattr(prices, 'columns'):
                    prices = prices.iloc[:, 0] if len(prices.columns) > 0 else prices

                returns = prices.pct_change().dropna()
                # Convert to monthly for consistency with factor data
                monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                data[symbol.replace('-USD', '')] = monthly
                print(f"  {symbol}: {len(monthly)} months")
        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")
            import traceback
            traceback.print_exc()

    return pd.DataFrame(data)


def fetch_sector_data(start='2000-01-01', end='2025-12-01'):
    """Fetch S&P 500 sector ETF returns"""
    # Sector ETFs (SPDR)
    sectors = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLE': 'Energy',
        'XLV': 'Healthcare',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLI': 'Industrials',
        'XLB': 'Materials',
        'XLU': 'Utilities',
    }

    print(f"Fetching sector ETF data...")

    data = {}
    for symbol, name in sectors.items():
        try:
            df = yf.download(symbol, start=start, end=end, progress=False)
            if len(df) > 0:
                # Handle both old and new yfinance API
                if 'Adj Close' in df.columns:
                    prices = df['Adj Close']
                elif 'Close' in df.columns:
                    prices = df['Close']
                else:
                    prices = df.iloc[:, 0]

                if hasattr(prices, 'columns'):
                    prices = prices.iloc[:, 0] if len(prices.columns) > 0 else prices

                returns = prices.pct_change().dropna()
                monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                data[symbol] = monthly
                print(f"  {symbol} ({name}): {len(monthly)} months")
        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")

    return pd.DataFrame(data)


# ============================================================================
# Crowding Signal (adapted for different assets)
# ============================================================================

def compute_crowding_signal(returns, window=12):
    """
    Compute crowding proxy based on return momentum and volatility.
    Higher values = more crowded (potential for reversal)
    """
    # Rolling return (momentum)
    rolling_return = returns.rolling(window).sum()

    # Absolute rolling return as crowding proxy
    abs_rolling = np.abs(rolling_return)

    # Normalize by expanding median
    median_return = abs_rolling.expanding().median()
    crowding = abs_rolling / median_return

    # Clip extreme values
    crowding = crowding.clip(0, 3)

    return crowding


# ============================================================================
# CW-ACI Implementation
# ============================================================================

def standard_conformal_coverage(returns, alpha=0.1, calib_size=60):
    """Standard conformal prediction coverage"""
    coverages = []

    for t in range(calib_size, len(returns)):
        # Calibration set
        calib = returns.iloc[t-calib_size:t].values

        # Nonconformity scores (absolute residuals from mean)
        scores = np.abs(calib - np.mean(calib))

        # Quantile threshold
        q = np.quantile(scores, 1 - alpha)

        # Test point
        test_val = returns.iloc[t]
        pred_mean = np.mean(calib)

        # Coverage
        covered = np.abs(test_val - pred_mean) <= q
        coverages.append(covered)

    return np.array(coverages)


def cwaci_coverage(returns, crowding, alpha=0.1, gamma=1.0, calib_size=60):
    """CW-ACI coverage with crowding weighting"""
    coverages = []

    # Normalize crowding
    crowding_mean = crowding.expanding().mean()
    crowding_std = crowding.expanding().std()

    for t in range(calib_size, len(returns)):
        # Calibration set
        calib = returns.iloc[t-calib_size:t].values

        # Nonconformity scores
        scores = np.abs(calib - np.mean(calib))

        # Current crowding (normalized)
        c_t = crowding.iloc[t]
        c_mean = crowding_mean.iloc[t]
        c_std = crowding_std.iloc[t]

        if c_std > 0:
            c_norm = (c_t - c_mean) / c_std
        else:
            c_norm = 0

        # Weight based on crowding (sigmoid)
        w = 1 / (1 + np.exp(-gamma * c_norm))

        # Adjust scores
        adjusted_scores = scores * (1 + w)

        # Quantile threshold
        q = np.quantile(adjusted_scores, 1 - alpha)

        # Test point
        test_val = returns.iloc[t]
        pred_mean = np.mean(calib)

        # Coverage
        covered = np.abs(test_val - pred_mean) <= q
        coverages.append(covered)

    return np.array(coverages)


def analyze_coverage_by_crowding(coverages, crowding, calib_size=60):
    """Analyze coverage by crowding regime"""
    crowding_aligned = crowding.iloc[calib_size:].values

    # Define regimes by terciles
    low_thresh = np.percentile(crowding_aligned, 33)
    high_thresh = np.percentile(crowding_aligned, 67)

    low_mask = crowding_aligned <= low_thresh
    mid_mask = (crowding_aligned > low_thresh) & (crowding_aligned <= high_thresh)
    high_mask = crowding_aligned > high_thresh

    results = {
        'overall': np.mean(coverages),
        'low_crowding': np.mean(coverages[low_mask]) if low_mask.sum() > 0 else np.nan,
        'mid_crowding': np.mean(coverages[mid_mask]) if mid_mask.sum() > 0 else np.nan,
        'high_crowding': np.mean(coverages[high_mask]) if high_mask.sum() > 0 else np.nan,
        'n_low': low_mask.sum(),
        'n_mid': mid_mask.sum(),
        'n_high': high_mask.sum()
    }

    return results


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(returns_df, asset_class_name, alpha=0.1, gamma=1.0, calib_size=60):
    """Run CW-ACI experiment on a set of assets"""

    print(f"\n{'='*60}")
    print(f"Experiment: {asset_class_name}")
    print(f"{'='*60}")
    print(f"Assets: {list(returns_df.columns)}")
    print(f"Period: {returns_df.index[0]} to {returns_df.index[-1]}")
    print(f"Observations: {len(returns_df)}")
    print(f"Alpha: {alpha}, Gamma: {gamma}, Calib size: {calib_size}")

    results = []

    for col in returns_df.columns:
        returns = returns_df[col].dropna()

        if len(returns) < calib_size + 12:
            print(f"  Skipping {col}: insufficient data ({len(returns)} obs)")
            continue

        # Compute crowding signal
        crowding = compute_crowding_signal(returns)

        # Standard CP
        scp_cov = standard_conformal_coverage(returns, alpha=alpha, calib_size=calib_size)
        scp_results = analyze_coverage_by_crowding(scp_cov, crowding, calib_size)

        # CW-ACI
        cwaci_cov = cwaci_coverage(returns, crowding, alpha=alpha, gamma=gamma, calib_size=calib_size)
        cwaci_results = analyze_coverage_by_crowding(cwaci_cov, crowding, calib_size)

        # Improvement
        improvement = cwaci_results['high_crowding'] - scp_results['high_crowding']

        results.append({
            'asset': col,
            'n_obs': len(returns),
            'scp_overall': scp_results['overall'],
            'scp_high_crowd': scp_results['high_crowding'],
            'cwaci_overall': cwaci_results['overall'],
            'cwaci_high_crowd': cwaci_results['high_crowding'],
            'improvement': improvement
        })

        print(f"\n  {col}:")
        print(f"    SCP:   Overall={scp_results['overall']:.1%}, High-crowd={scp_results['high_crowding']:.1%}")
        print(f"    CW-ACI: Overall={cwaci_results['overall']:.1%}, High-crowd={cwaci_results['high_crowding']:.1%}")
        print(f"    Improvement: {improvement:+.1%}")

    return pd.DataFrame(results)


def main():
    print("="*70)
    print("CW-ACI Extended Assets Experiment")
    print("="*70)

    all_results = {}

    # 1. Cryptocurrency
    print("\n[1/2] Fetching Cryptocurrency Data...")
    crypto_returns = fetch_crypto_data()
    if len(crypto_returns) > 0:
        crypto_results = run_experiment(
            crypto_returns,
            "Cryptocurrency (BTC, ETH)",
            calib_size=24  # Shorter calibration for crypto (less history)
        )
        all_results['crypto'] = crypto_results

    # 2. S&P 500 Sectors
    print("\n[2/2] Fetching S&P 500 Sector Data...")
    sector_returns = fetch_sector_data()
    if len(sector_returns) > 0:
        sector_results = run_experiment(
            sector_returns,
            "S&P 500 Sectors",
            calib_size=60
        )
        all_results['sectors'] = sector_results

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Extended Asset Classes")
    print("="*70)

    for asset_class, df in all_results.items():
        print(f"\n{asset_class.upper()}:")
        print(f"  Average SCP high-crowd coverage: {df['scp_high_crowd'].mean():.1%}")
        print(f"  Average CW-ACI high-crowd coverage: {df['cwaci_high_crowd'].mean():.1%}")
        print(f"  Average improvement: {df['improvement'].mean():+.1%}")

    # Save results
    output_dir = '/Users/i767700/Github/quant/research/jofe_cwaci/results'
    import os
    os.makedirs(output_dir, exist_ok=True)

    for name, df in all_results.items():
        df.to_csv(f'{output_dir}/extended_{name}_results.csv', index=False)
        print(f"\nSaved: {output_dir}/extended_{name}_results.csv")

    return all_results


if __name__ == "__main__":
    results = main()
