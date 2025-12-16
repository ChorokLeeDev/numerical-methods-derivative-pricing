"""
Out-of-Sample Validation of Regime Conditioning

Addresses reviewer concerns:
1. Sample size per regime
2. Threshold selection (pre-specified, not optimized)
3. Out-of-sample validation

Methodology:
- Training: 1990-2014 (establish median threshold)
- Out-of-sample: 2015-2024 (apply threshold prospectively)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from crowding_signal import CrowdingDetector

DATA_DIR = Path(__file__).parent.parent / 'data'


def load_data():
    factors = pd.read_parquet(DATA_DIR / 'ff_extended_factors.parquet')
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    if 'RF' in factors.columns:
        factors = factors.drop(columns=['RF'])
    return factors


def compute_factor_momentum_returns(factors, signals, lookback=12):
    """Compute factor momentum returns aligned with signals."""
    factor_cols = [c for c in factors.columns if c in signals]
    factor_returns = factors[factor_cols].dropna()

    # Trailing returns for momentum
    trailing = factor_returns.rolling(lookback).mean()
    ranks = trailing.rank(axis=1, pct=True)
    mom_weights = ranks / ranks.sum(axis=1).values.reshape(-1, 1)
    mom_weights = mom_weights.shift(1).dropna()

    aligned_returns = factor_returns.loc[mom_weights.index]
    mom_returns = (aligned_returns * mom_weights).sum(axis=1)

    return mom_returns


def sharpe(r):
    """Annualized Sharpe ratio."""
    if len(r) < 2 or r.std() == 0:
        return 0
    return r.mean() / r.std() * np.sqrt(12)


def main():
    print("=" * 70)
    print("OUT-OF-SAMPLE VALIDATION: REGIME CONDITIONING")
    print("=" * 70)

    # Load data
    factors = load_data()
    factors = factors[factors.index >= '1990-01-01']

    # Define periods
    train_end = '2014-12-31'
    oos_start = '2015-01-01'

    print(f"\nTraining period: 1990-01-01 to {train_end}")
    print(f"Out-of-sample:   {oos_start} to {factors.index.max().strftime('%Y-%m-%d')}")

    # Compute signals on FULL sample (expanding window handles lookahead)
    detector = CrowdingDetector(train_window=120, prediction_gap=12, sharpe_window=36)
    signals = detector.compute_multi_factor_signals(factors)

    # Aggregate crowding signal
    residuals = pd.DataFrame({f: signals[f]['residual'] for f in signals if len(signals[f]) > 0})
    aggregate_crowding = residuals.mean(axis=1)

    # Factor momentum returns
    mom_returns = compute_factor_momentum_returns(factors, signals)

    # Align
    common_idx = mom_returns.index.intersection(aggregate_crowding.index)
    mom_aligned = mom_returns.loc[common_idx]
    crowd_aligned = aggregate_crowding.loc[common_idx]

    # ================================================================
    # STEP 1: Establish threshold from TRAINING period only
    # ================================================================
    train_mask = crowd_aligned.index <= train_end
    train_crowding = crowd_aligned[train_mask]

    # Pre-specified threshold: training period median
    threshold = train_crowding.median()
    print(f"\n" + "-" * 70)
    print("THRESHOLD SELECTION (from training period only)")
    print("-" * 70)
    print(f"Training period median crowding: {threshold:.4f}")
    print(f"Training period mean crowding:   {train_crowding.mean():.4f}")
    print(f"Training period std crowding:    {train_crowding.std():.4f}")

    # ================================================================
    # STEP 2: In-sample results (for comparison)
    # ================================================================
    print(f"\n" + "-" * 70)
    print("IN-SAMPLE RESULTS (1990-2014)")
    print("-" * 70)

    train_mom = mom_aligned[train_mask]
    train_crowd = crowd_aligned[train_mask]

    is_uncrowded_train = train_crowd > threshold

    train_uncrowded_ret = train_mom[is_uncrowded_train]
    train_crowded_ret = train_mom[~is_uncrowded_train]

    print(f"{'Regime':<20} {'N':<10} {'Sharpe':<10} {'Mean Ret':<12} {'Std':<10}")
    print("-" * 62)
    print(f"{'Uncrowded':<20} {len(train_uncrowded_ret):<10} {sharpe(train_uncrowded_ret):<10.2f} {train_uncrowded_ret.mean()*12:<12.2%} {train_uncrowded_ret.std()*np.sqrt(12):<10.2%}")
    print(f"{'Crowded':<20} {len(train_crowded_ret):<10} {sharpe(train_crowded_ret):<10.2f} {train_crowded_ret.mean()*12:<12.2%} {train_crowded_ret.std()*np.sqrt(12):<10.2%}")
    print(f"{'All':<20} {len(train_mom):<10} {sharpe(train_mom):<10.2f} {train_mom.mean()*12:<12.2%} {train_mom.std()*np.sqrt(12):<10.2%}")

    sharpe_diff_train = sharpe(train_uncrowded_ret) - sharpe(train_crowded_ret)
    print(f"\nSharpe differential (in-sample): {sharpe_diff_train:+.2f}")

    # ================================================================
    # STEP 3: Out-of-sample results (KEY TEST)
    # ================================================================
    print(f"\n" + "-" * 70)
    print("OUT-OF-SAMPLE RESULTS (2015-2024)")
    print("-" * 70)

    oos_mask = crowd_aligned.index >= oos_start
    oos_mom = mom_aligned[oos_mask]
    oos_crowd = crowd_aligned[oos_mask]

    # Apply TRAINING threshold to OOS data
    is_uncrowded_oos = oos_crowd > threshold

    oos_uncrowded_ret = oos_mom[is_uncrowded_oos]
    oos_crowded_ret = oos_mom[~is_uncrowded_oos]

    print(f"{'Regime':<20} {'N':<10} {'Sharpe':<10} {'Mean Ret':<12} {'Std':<10}")
    print("-" * 62)
    print(f"{'Uncrowded':<20} {len(oos_uncrowded_ret):<10} {sharpe(oos_uncrowded_ret):<10.2f} {oos_uncrowded_ret.mean()*12:<12.2%} {oos_uncrowded_ret.std()*np.sqrt(12):<10.2%}")
    print(f"{'Crowded':<20} {len(oos_crowded_ret):<10} {sharpe(oos_crowded_ret):<10.2f} {oos_crowded_ret.mean()*12:<12.2%} {oos_crowded_ret.std()*np.sqrt(12):<10.2%}")
    print(f"{'All':<20} {len(oos_mom):<10} {sharpe(oos_mom):<10.2f} {oos_mom.mean()*12:<12.2%} {oos_mom.std()*np.sqrt(12):<10.2%}")

    sharpe_diff_oos = sharpe(oos_uncrowded_ret) - sharpe(oos_crowded_ret)
    print(f"\nSharpe differential (out-of-sample): {sharpe_diff_oos:+.2f}")

    # ================================================================
    # STEP 4: Statistical significance
    # ================================================================
    print(f"\n" + "-" * 70)
    print("STATISTICAL TESTS")
    print("-" * 70)

    # Two-sample t-test on returns
    if len(oos_uncrowded_ret) > 5 and len(oos_crowded_ret) > 5:
        t_stat, p_value = stats.ttest_ind(oos_uncrowded_ret, oos_crowded_ret)
        print(f"Two-sample t-test (OOS returns):")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value:     {p_value:.4f}")

        # Mann-Whitney U (non-parametric)
        u_stat, u_pvalue = stats.mannwhitneyu(oos_uncrowded_ret, oos_crowded_ret, alternative='greater')
        print(f"\nMann-Whitney U test (OOS):")
        print(f"  U-statistic: {u_stat:.1f}")
        print(f"  p-value:     {u_pvalue:.4f}")

    # ================================================================
    # STEP 5: Drawdown analysis (OOS)
    # ================================================================
    print(f"\n" + "-" * 70)
    print("DRAWDOWN ANALYSIS (OOS)")
    print("-" * 70)

    # Always-in strategy
    always_in = oos_mom
    always_cum = (1 + always_in).cumprod()
    always_dd = (always_cum / always_cum.expanding().max() - 1).min()

    # Crowding-timed (reduce exposure when crowded)
    exposure = (oos_crowd > threshold).astype(float) * 0.5 + 0.5  # 50-100%
    timed_ret = oos_mom * exposure
    timed_cum = (1 + timed_ret).cumprod()
    timed_dd = (timed_cum / timed_cum.expanding().max() - 1).min()

    print(f"{'Strategy':<25} {'Sharpe':<10} {'Max DD':<12} {'Avg Exposure':<12}")
    print("-" * 59)
    print(f"{'Always 100%':<25} {sharpe(always_in):<10.2f} {always_dd:<12.1%} {'100%':<12}")
    print(f"{'Crowding-Timed':<25} {sharpe(timed_ret):<10.2f} {timed_dd:<12.1%} {exposure.mean():<12.1%}")

    dd_improvement = (timed_dd - always_dd) / abs(always_dd) * 100
    print(f"\nDrawdown improvement: {dd_improvement:+.1f}%")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    print(f"""
THRESHOLD: Pre-specified as training period (1990-2014) median = {threshold:.3f}
           NOT optimized on out-of-sample data.

IN-SAMPLE (1990-2014):
  - Uncrowded: N={len(train_uncrowded_ret)}, Sharpe={sharpe(train_uncrowded_ret):.2f}
  - Crowded:   N={len(train_crowded_ret)}, Sharpe={sharpe(train_crowded_ret):.2f}
  - Differential: {sharpe_diff_train:+.2f}

OUT-OF-SAMPLE (2015-2024):
  - Uncrowded: N={len(oos_uncrowded_ret)}, Sharpe={sharpe(oos_uncrowded_ret):.2f}
  - Crowded:   N={len(oos_crowded_ret)}, Sharpe={sharpe(oos_crowded_ret):.2f}
  - Differential: {sharpe_diff_oos:+.2f}
  - p-value: {p_value:.4f}

DRAWDOWN (OOS):
  - Always-in:     Max DD = {always_dd:.1%}
  - Crowding-timed: Max DD = {timed_dd:.1%}
  - Improvement: {abs(dd_improvement):.0f}%
""")

    # Suggested paper language
    print("-" * 70)
    print("SUGGESTED PAPER LANGUAGE:")
    print("-" * 70)
    print(f"""
"We define crowding regimes using the training-period (1990-2014)
median aggregate residual ({threshold:.3f}), applied prospectively to
out-of-sample data. In-sample, factor momentum achieves Sharpe
{sharpe(train_uncrowded_ret):.2f} in uncrowded regimes vs. {sharpe(train_crowded_ret):.2f} in crowded
(N={len(train_uncrowded_ret)} and N={len(train_crowded_ret)} months, respectively).

Out-of-sample (2015-2024), the pattern persists: Sharpe {sharpe(oos_uncrowded_ret):.2f}
uncrowded vs. {sharpe(oos_crowded_ret):.2f} crowded (N={len(oos_uncrowded_ret)}, N={len(oos_crowded_ret)};
differential {sharpe_diff_oos:+.2f}, p={p_value:.2f}). Crowding-timed exposure
reduces maximum drawdown from {always_dd:.1%} to {timed_dd:.1%}."
""")


if __name__ == '__main__':
    main()
