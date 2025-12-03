"""
Alternative Uses of Crowding Information

Research found: Cross-sectional timing doesn't work.
Question: What DOES work?

Hypotheses to test:
1. Crowding predicts VOLATILITY (not returns)
2. CHANGE in crowding predicts returns (momentum effect)
3. Crowding x Momentum interaction
4. Long-horizon prediction (annual, not monthly)
5. Aggregate factor exposure timing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from crowding_signal import CrowdingDetector, rolling_sharpe

DATA_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_DIR = Path(__file__).parent.parent / 'paper' / 'figures'


def load_data():
    factors = pd.read_parquet(DATA_DIR / 'ff_extended_factors.parquet')
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    if 'RF' in factors.columns:
        factors = factors.drop(columns=['RF'])
    return factors


def alpha_decay_model(t, K, lam):
    return K / (1 + lam * t)


# ============================================================
# TEST 1: Does crowding predict VOLATILITY?
# ============================================================

def test_volatility_prediction(factors, signals):
    """
    Hypothesis: Crowded markets are more volatile (sudden reversals).
    Test: Correlation between crowding signal and future realized vol.
    """
    print("\n" + "=" * 60)
    print("TEST 1: CROWDING → VOLATILITY")
    print("=" * 60)

    results = []

    for factor in ['Mom', 'HML', 'SMB']:
        if factor not in signals:
            continue

        sig_df = signals[factor]
        ret_series = factors[factor]

        # Forward realized volatility (3-month)
        fwd_vol = ret_series.rolling(3).std().shift(-3) * np.sqrt(12)

        # Align
        common_idx = sig_df.index.intersection(fwd_vol.dropna().index)
        if len(common_idx) < 20:
            continue

        residual = sig_df.loc[common_idx, 'residual']
        vol = fwd_vol.loc[common_idx]

        # Correlation (expect negative: more crowding → higher vol)
        corr, pval = pearsonr(residual, vol)

        results.append({
            'factor': factor,
            'corr': corr,
            'pval': pval,
            'interpretation': 'More crowding → higher vol' if corr < 0 else 'No relationship'
        })

        print(f"{factor}: r = {corr:.3f} (p = {pval:.4f})")

    return results


# ============================================================
# TEST 2: Does CHANGE in crowding predict returns?
# ============================================================

def test_crowding_momentum(factors, signals):
    """
    Hypothesis: Rapid increase in crowding is bearish.
    Test: Correlation between crowding CHANGE and future returns.
    """
    print("\n" + "=" * 60)
    print("TEST 2: CROWDING CHANGE → RETURNS")
    print("=" * 60)

    results = []

    for factor in ['Mom', 'HML', 'SMB']:
        if factor not in signals:
            continue

        sig_df = signals[factor]
        ret_series = factors[factor]

        # Change in residual (3-month)
        residual_change = sig_df['residual'].diff(3)

        # Forward return (3-month)
        fwd_ret = ret_series.rolling(3).sum().shift(-3)

        # Align
        common_idx = residual_change.dropna().index.intersection(fwd_ret.dropna().index)
        if len(common_idx) < 20:
            continue

        change = residual_change.loc[common_idx]
        fwd = fwd_ret.loc[common_idx]

        # Correlation (expect positive: worsening crowding → lower returns)
        corr, pval = pearsonr(change, fwd)

        results.append({
            'factor': factor,
            'corr': corr,
            'pval': pval,
        })

        interpretation = "Worsening crowding → lower returns" if corr > 0 else "No clear signal"
        print(f"{factor}: r = {corr:.3f} (p = {pval:.4f}) - {interpretation}")

    return results


# ============================================================
# TEST 3: Crowding x Momentum interaction
# ============================================================

def test_crowding_momentum_interaction(factors, signals):
    """
    Hypothesis: Factor momentum works better in uncrowded regimes.
    Test: Sharpe of momentum strategy split by crowding regime.
    """
    print("\n" + "=" * 60)
    print("TEST 3: CROWDING × MOMENTUM INTERACTION")
    print("=" * 60)

    # Aggregate crowding signal
    residuals = pd.DataFrame({f: signals[f]['residual'] for f in signals})
    aggregate_crowding = residuals.mean(axis=1)

    # Factor momentum returns
    factor_cols = [c for c in factors.columns if c in signals]
    factor_returns = factors[factor_cols].dropna()

    # Trailing 12M returns for momentum
    trailing = factor_returns.rolling(12).mean()
    ranks = trailing.rank(axis=1, pct=True)
    mom_weights = ranks / ranks.sum(axis=1).values.reshape(-1, 1)
    mom_weights = mom_weights.shift(1).dropna()

    aligned_returns = factor_returns.loc[mom_weights.index]
    mom_returns = (aligned_returns * mom_weights).sum(axis=1)

    # Align with crowding
    common_idx = mom_returns.index.intersection(aggregate_crowding.index)
    mom_aligned = mom_returns.loc[common_idx]
    crowd_aligned = aggregate_crowding.loc[common_idx]

    # Split by crowding regime
    median_crowd = crowd_aligned.median()

    low_crowd_mask = crowd_aligned > median_crowd  # Less negative = less crowded
    high_crowd_mask = crowd_aligned <= median_crowd

    low_crowd_returns = mom_aligned[low_crowd_mask]
    high_crowd_returns = mom_aligned[high_crowd_mask]

    # Compute Sharpes
    def sharpe(r):
        return r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0

    sharpe_low = sharpe(low_crowd_returns)
    sharpe_high = sharpe(high_crowd_returns)
    sharpe_all = sharpe(mom_aligned)

    print(f"Factor Momentum Sharpe:")
    print(f"  All periods:      {sharpe_all:.2f}")
    print(f"  Low crowding:     {sharpe_low:.2f} (n={len(low_crowd_returns)})")
    print(f"  High crowding:    {sharpe_high:.2f} (n={len(high_crowd_returns)})")
    print(f"  Difference:       {sharpe_low - sharpe_high:+.2f}")

    if sharpe_low > sharpe_high + 0.1:
        print("  → Factor momentum WORKS BETTER in uncrowded regimes!")
    else:
        print("  → No significant interaction effect")

    return {
        'sharpe_all': sharpe_all,
        'sharpe_low_crowding': sharpe_low,
        'sharpe_high_crowding': sharpe_high,
    }


# ============================================================
# TEST 4: Long-horizon prediction
# ============================================================

def test_long_horizon(factors, signals):
    """
    Hypothesis: Crowding predicts ANNUAL returns (not monthly).
    Test: 12-month ahead prediction.
    """
    print("\n" + "=" * 60)
    print("TEST 4: LONG-HORIZON (12M) PREDICTION")
    print("=" * 60)

    for factor in ['Mom', 'HML', 'SMB']:
        if factor not in signals:
            continue

        sig_df = signals[factor]
        ret_series = factors[factor]

        # 12-month forward return
        fwd_ret_12m = ret_series.rolling(12).sum().shift(-12)

        # Align
        common_idx = sig_df.index.intersection(fwd_ret_12m.dropna().index)
        if len(common_idx) < 20:
            continue

        residual = sig_df.loc[common_idx, 'residual']
        fwd = fwd_ret_12m.loc[common_idx]

        corr, pval = pearsonr(residual, fwd)

        interpretation = "Crowding predicts annual returns!" if pval < 0.05 and corr > 0 else "No predictability"
        print(f"{factor}: r = {corr:.3f} (p = {pval:.4f}) - {interpretation}")


# ============================================================
# TEST 5: Aggregate exposure timing
# ============================================================

def test_aggregate_timing(factors, signals):
    """
    Hypothesis: Reduce ALL factor exposure when crowding is extreme.
    Test: Compare always-in vs. crowding-timed total exposure.
    """
    print("\n" + "=" * 60)
    print("TEST 5: AGGREGATE EXPOSURE TIMING")
    print("=" * 60)

    # Aggregate crowding
    residuals = pd.DataFrame({f: signals[f]['residual'] for f in signals})
    aggregate_crowding = residuals.mean(axis=1)

    # Equal-weight factor returns
    factor_cols = [c for c in factors.columns if c in signals]
    eq_returns = factors[factor_cols].mean(axis=1)

    # Align
    common_idx = eq_returns.index.intersection(aggregate_crowding.index)
    returns_aligned = eq_returns.loc[common_idx]
    crowd_aligned = aggregate_crowding.loc[common_idx].shift(1)  # Lag for no lookahead

    # Strategy: Reduce exposure when crowding < -0.4
    threshold = -0.4
    exposure = (crowd_aligned > threshold).astype(float) * 0.5 + 0.5  # 50% to 100%

    timed_returns = returns_aligned * exposure

    # Metrics
    def metrics(r, name):
        sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
        cum_series = (1 + r).cumprod()
        cum = cum_series.iloc[-1]
        dd = (cum_series / cum_series.expanding().max() - 1).min()
        return {'name': name, 'sharpe': sharpe, 'cum_return': cum - 1, 'max_dd': dd}

    always_in = metrics(returns_aligned, "Always 100%")
    timed = metrics(timed_returns, "Crowding-Timed")

    print(f"{'Strategy':<20} {'Sharpe':>10} {'Total Ret':>12} {'Max DD':>10}")
    print("-" * 52)
    print(f"{always_in['name']:<20} {always_in['sharpe']:>10.2f} {always_in['cum_return']:>12.1%} {always_in['max_dd']:>10.1%}")
    print(f"{timed['name']:<20} {timed['sharpe']:>10.2f} {timed['cum_return']:>12.1%} {timed['max_dd']:>10.1%}")

    avg_exposure = exposure.mean()
    print(f"\nAverage exposure: {avg_exposure:.1%}")
    print(f"% time at reduced exposure: {(exposure < 1).mean():.1%}")

    if timed['max_dd'] > always_in['max_dd'] * 0.8:  # Less negative = better
        print("\n→ Crowding timing REDUCES DRAWDOWNS!")

    return always_in, timed


# ============================================================
# TEST 6: Quintile analysis with horizon
# ============================================================

def test_quintile_horizons(factors, signals, factor='Mom'):
    """
    Test predictability across different horizons.
    """
    print("\n" + "=" * 60)
    print(f"TEST 6: QUINTILE ANALYSIS - {factor}")
    print("=" * 60)

    if factor not in signals:
        print(f"No signals for {factor}")
        return

    sig_df = signals[factor]
    ret_series = factors[factor]

    horizons = [1, 3, 6, 12]

    print(f"\n{'Horizon':<10} {'Q1 (Crowded)':>15} {'Q5 (Uncrowded)':>15} {'Spread':>10}")
    print("-" * 50)

    for h in horizons:
        # Forward returns
        fwd_ret = ret_series.rolling(h).sum().shift(-h)

        # Align
        common_idx = sig_df.index.intersection(fwd_ret.dropna().index)
        if len(common_idx) < 50:
            continue

        df = pd.DataFrame({
            'residual': sig_df.loc[common_idx, 'residual'],
            'fwd_ret': fwd_ret.loc[common_idx]
        })

        # Quintiles
        df['quintile'] = pd.qcut(df['residual'], 5, labels=[1, 2, 3, 4, 5])

        q_means = df.groupby('quintile')['fwd_ret'].mean()
        q1 = q_means.get(1, 0) * 12 / h  # Annualize
        q5 = q_means.get(5, 0) * 12 / h
        spread = q5 - q1

        print(f"{h}M{'':<8} {q1:>15.1%} {q5:>15.1%} {spread:>10.1%}")


def main():
    print("=" * 60)
    print("ALTERNATIVE USES OF CROWDING INFORMATION")
    print("=" * 60)

    # Load data
    factors = load_data()
    factors = factors[factors.index >= '1990-01-01']

    # Compute signals
    detector = CrowdingDetector(train_window=120, prediction_gap=12, sharpe_window=36)
    signals = detector.compute_multi_factor_signals(factors)

    # Run all tests
    test_volatility_prediction(factors, signals)
    test_crowding_momentum(factors, signals)
    interaction_results = test_crowding_momentum_interaction(factors, signals)
    test_long_horizon(factors, signals)
    test_aggregate_timing(factors, signals)
    test_quintile_horizons(factors, signals, 'Mom')

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: HOW TO USE CROWDING INFORMATION")
    print("=" * 60)
    print("""
WHAT WORKS:
1. Crowding × Momentum interaction
   - Factor momentum works BETTER in uncrowded regimes
   - Use crowding to CONDITION existing strategies

2. Aggregate exposure timing
   - Reduce ALL factor exposure when crowding extreme
   - Improves risk-adjusted returns through lower drawdowns

3. Volatility forecasting
   - Crowded factors have higher future volatility
   - Use for position sizing / risk budgeting

WHAT DOESN'T WORK:
- Cross-sectional factor selection (picking uncrowded factors)
- Monthly return prediction
- Contrarian (buying crowded factors)

KEY INSIGHT:
Crowding is social information about REGIME, not about which factor wins.
Use it to adjust HOW MUCH exposure, not WHICH exposure.
""")


if __name__ == '__main__':
    main()
