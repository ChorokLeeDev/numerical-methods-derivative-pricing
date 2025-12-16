"""
Comprehensive Test: Is There ANY Actionable Value in Crowding?

Tests:
1. Long-short portfolio (long uncrowded, short crowded)
2. Volatility prediction (crowding → future vol)
3. Tail risk prediction (crowding → future drawdowns)
4. Cross-sectional ranking each month
5. Crowding momentum (change in crowding)
6. Extreme crowding only (top/bottom decile)
7. Factor-specific effects (maybe works for some factors?)
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
    return factors


def sharpe(r):
    if len(r) < 12 or r.std() == 0:
        return np.nan
    return r.mean() / r.std() * np.sqrt(12)


def max_drawdown(r):
    cum = (1 + r).cumprod()
    running_max = cum.expanding().max()
    dd = (cum / running_max - 1)
    return dd.min()


def main():
    print("=" * 70)
    print("COMPREHENSIVE TEST: ACTIONABLE VALUE OF CROWDING")
    print("=" * 70)

    # Load data
    factors = load_data()
    rf = factors['RF'] if 'RF' in factors.columns else 0
    factors = factors.drop(columns=['RF'], errors='ignore')

    # Compute signals
    detector = CrowdingDetector(train_window=120, prediction_gap=12, sharpe_window=36)
    signals = detector.compute_multi_factor_signals(factors)

    # Build residual DataFrame
    residual_df = pd.DataFrame({f: signals[f]['residual'] for f in signals if len(signals[f]) > 0})

    # Align with factor returns
    common_idx = factors.index.intersection(residual_df.index)
    factors_aligned = factors.loc[common_idx]
    residuals_aligned = residual_df.loc[common_idx]

    # Use post-1985 as OOS (after model has enough training data)
    oos_start = '1986-01-01'
    oos_mask = factors_aligned.index >= oos_start

    factors_oos = factors_aligned[oos_mask]
    residuals_oos = residuals_aligned[oos_mask]

    print(f"\nOOS Period: {oos_start} to {factors_oos.index.max().strftime('%Y-%m-%d')}")
    print(f"N months: {len(factors_oos)}")
    print(f"Factors: {list(factors_oos.columns)}")

    results = {}

    # ================================================================
    # TEST 1: LONG-SHORT PORTFOLIO
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 1: LONG-SHORT PORTFOLIO (Long Uncrowded, Short Crowded)")
    print("=" * 70)

    # Each month: rank factors by residual, go long top half, short bottom half
    ls_returns = []

    for date in factors_oos.index[1:]:  # Skip first (need prior residual)
        prev_date = factors_oos.index[factors_oos.index.get_loc(date) - 1]

        if prev_date not in residuals_oos.index:
            continue

        # Prior month residuals
        resid = residuals_oos.loc[prev_date].dropna()
        if len(resid) < 4:
            continue

        # Rank: high residual = uncrowded = LONG
        ranks = resid.rank()
        median_rank = ranks.median()

        long_factors = ranks[ranks > median_rank].index.tolist()
        short_factors = ranks[ranks <= median_rank].index.tolist()

        # This month returns
        ret = factors_oos.loc[date]

        long_ret = ret[long_factors].mean() if long_factors else 0
        short_ret = ret[short_factors].mean() if short_factors else 0

        ls_returns.append({
            'date': date,
            'long_ret': long_ret,
            'short_ret': short_ret,
            'ls_ret': long_ret - short_ret,
        })

    ls_df = pd.DataFrame(ls_returns).set_index('date')

    print(f"\nLong-Short Strategy (monthly rebalance):")
    print(f"  Long leg Sharpe:  {sharpe(ls_df['long_ret']):.2f}")
    print(f"  Short leg Sharpe: {sharpe(ls_df['short_ret']):.2f}")
    print(f"  L-S Sharpe:       {sharpe(ls_df['ls_ret']):.2f}")
    print(f"  L-S Ann Return:   {ls_df['ls_ret'].mean() * 12:.2%}")
    print(f"  L-S Max DD:       {max_drawdown(ls_df['ls_ret']):.1%}")
    print(f"  Hit Rate:         {(ls_df['ls_ret'] > 0).mean():.1%}")

    # T-test: is L-S return significantly different from zero?
    t_stat, p_val = stats.ttest_1samp(ls_df['ls_ret'], 0)
    print(f"  t-stat: {t_stat:.2f}, p-value: {p_val:.4f}")

    results['long_short'] = {'sharpe': sharpe(ls_df['ls_ret']), 'p_value': p_val}

    # ================================================================
    # TEST 2: VOLATILITY PREDICTION
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 2: VOLATILITY PREDICTION")
    print("=" * 70)

    # Does crowding predict NEXT month's factor volatility?
    vol_results = []

    for factor in residuals_oos.columns:
        resid = residuals_oos[factor].dropna()
        ret = factors_oos[factor]

        # Realized vol = abs return (simple proxy)
        fwd_vol = ret.shift(-1).abs()

        # Align
        common = resid.index.intersection(fwd_vol.dropna().index)
        if len(common) < 50:
            continue

        r, p = stats.spearmanr(resid.loc[common], fwd_vol.loc[common])
        vol_results.append({'factor': factor, 'corr': r, 'p_value': p})

    vol_df = pd.DataFrame(vol_results)
    print("\nCrowding → Next Month Volatility (Spearman correlation):")
    print(f"{'Factor':<10} {'Corr':<10} {'p-value':<10} {'Significant':<10}")
    print("-" * 40)
    for _, row in vol_df.iterrows():
        sig = "***" if row['p_value'] < 0.01 else "**" if row['p_value'] < 0.05 else "*" if row['p_value'] < 0.1 else ""
        print(f"{row['factor']:<10} {row['corr']:<10.3f} {row['p_value']:<10.4f} {sig:<10}")

    print(f"\nMean correlation: {vol_df['corr'].mean():.3f}")
    print(f"Factors with significant (p<0.05) prediction: {(vol_df['p_value'] < 0.05).sum()}/{len(vol_df)}")

    results['vol_prediction'] = {'mean_corr': vol_df['corr'].mean(),
                                  'n_significant': (vol_df['p_value'] < 0.05).sum()}

    # ================================================================
    # TEST 3: TAIL RISK PREDICTION
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 3: TAIL RISK PREDICTION (Extreme Losses)")
    print("=" * 70)

    # Does high crowding predict factor crashes (bottom 10% returns)?
    tail_results = []

    for factor in residuals_oos.columns:
        resid = residuals_oos[factor].dropna()
        ret = factors_oos[factor]
        fwd_ret = ret.shift(-1)

        common = resid.index.intersection(fwd_ret.dropna().index)
        if len(common) < 50:
            continue

        resid_common = resid.loc[common]
        fwd_common = fwd_ret.loc[common]

        # Split by crowding
        threshold = resid_common.median()
        crowded_mask = resid_common < threshold

        # Probability of bottom 10% return
        bottom_10 = fwd_common.quantile(0.1)

        prob_crash_crowded = (fwd_common[crowded_mask] < bottom_10).mean()
        prob_crash_uncrowded = (fwd_common[~crowded_mask] < bottom_10).mean()

        tail_results.append({
            'factor': factor,
            'prob_crash_crowded': prob_crash_crowded,
            'prob_crash_uncrowded': prob_crash_uncrowded,
            'ratio': prob_crash_crowded / prob_crash_uncrowded if prob_crash_uncrowded > 0 else np.nan
        })

    tail_df = pd.DataFrame(tail_results)
    print("\nProbability of Bottom 10% Return by Crowding State:")
    print(f"{'Factor':<10} {'P(crash|crowd)':<15} {'P(crash|uncrowd)':<18} {'Ratio':<10}")
    print("-" * 53)
    for _, row in tail_df.iterrows():
        print(f"{row['factor']:<10} {row['prob_crash_crowded']:<15.1%} {row['prob_crash_uncrowded']:<18.1%} {row['ratio']:<10.2f}")

    print(f"\nMean crash probability ratio: {tail_df['ratio'].mean():.2f}x")
    print(f"(>1 means crowding increases crash risk)")

    results['tail_risk'] = {'mean_ratio': tail_df['ratio'].mean()}

    # ================================================================
    # TEST 4: EXTREME CROWDING ONLY
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 4: EXTREME CROWDING (Top/Bottom Quintile)")
    print("=" * 70)

    # Maybe effect only exists at extremes?
    extreme_results = []

    for factor in residuals_oos.columns:
        resid = residuals_oos[factor].dropna()
        ret = factors_oos[factor]
        fwd_ret = ret.shift(-1)

        common = resid.index.intersection(fwd_ret.dropna().index)
        if len(common) < 100:
            continue

        resid_common = resid.loc[common]
        fwd_common = fwd_ret.loc[common]

        # Quintiles
        q20 = resid_common.quantile(0.2)
        q80 = resid_common.quantile(0.8)

        extreme_crowded = fwd_common[resid_common < q20]
        extreme_uncrowded = fwd_common[resid_common > q80]
        middle = fwd_common[(resid_common >= q20) & (resid_common <= q80)]

        extreme_results.append({
            'factor': factor,
            'ret_extreme_crowded': extreme_crowded.mean() * 12,
            'ret_middle': middle.mean() * 12,
            'ret_extreme_uncrowded': extreme_uncrowded.mean() * 12,
            'spread': (extreme_uncrowded.mean() - extreme_crowded.mean()) * 12,
        })

    extreme_df = pd.DataFrame(extreme_results)
    print("\nAnnualized Returns by Crowding Quintile:")
    print(f"{'Factor':<10} {'Q1 (Crowded)':<14} {'Q2-Q4':<12} {'Q5 (Uncrowd)':<14} {'Spread':<10}")
    print("-" * 60)
    for _, row in extreme_df.iterrows():
        print(f"{row['factor']:<10} {row['ret_extreme_crowded']:<14.1%} {row['ret_middle']:<12.1%} "
              f"{row['ret_extreme_uncrowded']:<14.1%} {row['spread']:<+10.1%}")

    print(f"\nMean Q5-Q1 spread: {extreme_df['spread'].mean():+.1%}")

    results['extreme'] = {'mean_spread': extreme_df['spread'].mean()}

    # ================================================================
    # TEST 5: CROWDING MOMENTUM (Change in Crowding)
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 5: CROWDING MOMENTUM (Change in Crowding)")
    print("=" * 70)

    # Does CHANGE in crowding predict returns?
    # Hypothesis: crowding getting worse (more negative) → bad returns
    crowd_mom_results = []

    for factor in residuals_oos.columns:
        resid = residuals_oos[factor].dropna()
        ret = factors_oos[factor]

        # Change in crowding (3-month)
        crowd_change = resid.diff(3)
        fwd_ret = ret.shift(-1)

        common = crowd_change.dropna().index.intersection(fwd_ret.dropna().index)
        if len(common) < 50:
            continue

        r, p = stats.spearmanr(crowd_change.loc[common], fwd_ret.loc[common])
        crowd_mom_results.append({'factor': factor, 'corr': r, 'p_value': p})

    crowd_mom_df = pd.DataFrame(crowd_mom_results)
    print("\nChange in Crowding (3M) → Next Month Return:")
    print(f"{'Factor':<10} {'Corr':<10} {'p-value':<10}")
    print("-" * 30)
    for _, row in crowd_mom_df.iterrows():
        sig = "**" if row['p_value'] < 0.05 else "*" if row['p_value'] < 0.1 else ""
        print(f"{row['factor']:<10} {row['corr']:<+10.3f} {row['p_value']:<10.4f} {sig}")

    results['crowd_momentum'] = {'mean_corr': crowd_mom_df['corr'].mean()}

    # ================================================================
    # TEST 6: AGGREGATE TIMING (Different Threshold)
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 6: AGGREGATE TIMING (Reduce Exposure in Extreme Crowding)")
    print("=" * 70)

    # Equal-weight factor portfolio
    ew_returns = factors_oos.mean(axis=1)

    # Aggregate crowding
    agg_crowd = residuals_oos.mean(axis=1)

    # Strategy: reduce to 50% when aggregate crowding in bottom quartile
    threshold = agg_crowd.quantile(0.25)
    exposure = (agg_crowd > threshold).astype(float) * 0.5 + 0.5
    exposure = exposure.shift(1)  # No lookahead

    timed_returns = ew_returns * exposure.reindex(ew_returns.index).fillna(1)

    # Align
    common = ew_returns.index.intersection(timed_returns.dropna().index)
    ew_common = ew_returns.loc[common]
    timed_common = timed_returns.loc[common]

    print(f"\nEqual-weight factors vs. Crowding-timed:")
    print(f"  Always 100%:     Sharpe={sharpe(ew_common):.2f}, MaxDD={max_drawdown(ew_common):.1%}")
    print(f"  Crowding-timed:  Sharpe={sharpe(timed_common):.2f}, MaxDD={max_drawdown(timed_common):.1%}")
    print(f"  Avg exposure:    {exposure.mean():.1%}")

    results['agg_timing'] = {
        'sharpe_always': sharpe(ew_common),
        'sharpe_timed': sharpe(timed_common),
        'dd_always': max_drawdown(ew_common),
        'dd_timed': max_drawdown(timed_common),
    }

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: ACTIONABLE VALUE OF CROWDING")
    print("=" * 70)

    print(f"""
TEST                          RESULT                    ACTIONABLE?
────────────────────────────────────────────────────────────────────
1. Long-Short Portfolio       Sharpe={results['long_short']['sharpe']:.2f}, p={results['long_short']['p_value']:.2f}      {'YES' if results['long_short']['p_value'] < 0.05 else 'NO'}
2. Volatility Prediction      {results['vol_prediction']['n_significant']}/{len(vol_df)} factors sig         {'YES' if results['vol_prediction']['n_significant'] >= 3 else 'WEAK'}
3. Tail Risk Prediction       Ratio={results['tail_risk']['mean_ratio']:.2f}x              {'YES' if results['tail_risk']['mean_ratio'] > 1.2 else 'NO'}
4. Extreme Quintiles          Spread={results['extreme']['mean_spread']:+.1%}            {'YES' if abs(results['extreme']['mean_spread']) > 2 else 'NO'}
5. Crowding Momentum          Corr={results['crowd_momentum']['mean_corr']:.3f}               {'YES' if abs(results['crowd_momentum']['mean_corr']) > 0.05 else 'NO'}
6. Aggregate Timing           Sharpe {sharpe(timed_common):.2f} vs {sharpe(ew_common):.2f}     {'YES' if sharpe(timed_common) > sharpe(ew_common) + 0.1 else 'NO'}
""")

    # Any winners?
    actionable = []
    if results['long_short']['p_value'] < 0.05:
        actionable.append("Long-Short")
    if results['vol_prediction']['n_significant'] >= 3:
        actionable.append("Vol Prediction")
    if results['tail_risk']['mean_ratio'] > 1.3:
        actionable.append("Tail Risk")
    if abs(results['extreme']['mean_spread']) > 3:
        actionable.append("Extreme Quintiles")
    if sharpe(timed_common) > sharpe(ew_common) + 0.1:
        actionable.append("Aggregate Timing")

    if actionable:
        print(f"\n✓ ACTIONABLE USES FOUND: {', '.join(actionable)}")
    else:
        print(f"\n✗ NO CLEARLY ACTIONABLE USES FOUND")
        print("  Crowding information appears to be efficiently priced across all tested applications.")


if __name__ == '__main__':
    main()
