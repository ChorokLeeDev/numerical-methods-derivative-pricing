"""
Extended Out-of-Sample Validation

Use full historical data (1963-2024) with earlier training period
to get 30+ years of out-of-sample data.

Splits tested:
1. Train 1970-1990, OOS 1991-2024 (33 years OOS)
2. Train 1970-1985, OOS 1986-2024 (38 years OOS)
3. Rolling 10-year OOS windows
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


def compute_signals_and_momentum(factors, min_history=120):
    """Compute crowding signals and factor momentum returns."""
    detector = CrowdingDetector(train_window=120, prediction_gap=12, sharpe_window=36)
    signals = detector.compute_multi_factor_signals(factors)

    # Aggregate crowding
    residuals = pd.DataFrame({f: signals[f]['residual'] for f in signals if len(signals[f]) > 0})
    aggregate_crowding = residuals.mean(axis=1)

    # Factor momentum
    factor_cols = [c for c in factors.columns if c in signals]
    factor_returns = factors[factor_cols].dropna()

    trailing = factor_returns.rolling(12).mean()
    ranks = trailing.rank(axis=1, pct=True)
    mom_weights = ranks / ranks.sum(axis=1).values.reshape(-1, 1)
    mom_weights = mom_weights.shift(1).dropna()

    aligned_returns = factor_returns.loc[mom_weights.index]
    mom_returns = (aligned_returns * mom_weights).sum(axis=1)

    return aggregate_crowding, mom_returns


def sharpe(r):
    if len(r) < 12 or r.std() == 0:
        return np.nan
    return r.mean() / r.std() * np.sqrt(12)


def analyze_period(mom_returns, crowding, threshold, start, end, label):
    """Analyze a specific period."""
    mask = (mom_returns.index >= start) & (mom_returns.index <= end)
    mom = mom_returns[mask]
    crowd = crowding.reindex(mom.index)

    # Drop NaN
    valid = ~crowd.isna()
    mom = mom[valid]
    crowd = crowd[valid]

    if len(mom) < 24:
        return None

    is_uncrowded = crowd > threshold

    uncrowded_ret = mom[is_uncrowded]
    crowded_ret = mom[~is_uncrowded]

    if len(uncrowded_ret) < 12 or len(crowded_ret) < 12:
        return None

    sharpe_unc = sharpe(uncrowded_ret)
    sharpe_cro = sharpe(crowded_ret)

    # T-test
    if len(uncrowded_ret) > 5 and len(crowded_ret) > 5:
        t_stat, p_value = stats.ttest_ind(uncrowded_ret, crowded_ret)
    else:
        t_stat, p_value = np.nan, np.nan

    return {
        'period': label,
        'start': start,
        'end': end,
        'n_uncrowded': len(uncrowded_ret),
        'n_crowded': len(crowded_ret),
        'sharpe_uncrowded': sharpe_unc,
        'sharpe_crowded': sharpe_cro,
        'differential': sharpe_unc - sharpe_cro,
        'p_value': p_value,
    }


def main():
    print("=" * 70)
    print("EXTENDED OUT-OF-SAMPLE VALIDATION")
    print("Using 60+ years of data for longer OOS periods")
    print("=" * 70)

    # Load data
    factors = load_data()
    print(f"\nData range: {factors.index.min().strftime('%Y-%m')} to {factors.index.max().strftime('%Y-%m')}")
    print(f"Total months: {len(factors)}")

    # Compute signals (uses expanding window internally)
    print("\nComputing signals...")
    crowding, mom_returns = compute_signals_and_momentum(factors)

    print(f"Signal range: {crowding.index.min().strftime('%Y-%m')} to {crowding.index.max().strftime('%Y-%m')}")
    print(f"Momentum returns: {len(mom_returns)} months")

    results = []

    # ================================================================
    # TEST 1: Train 1975-1990, OOS 1991-2024 (33 years)
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Train 1975-1990, OOS 1991-2024")
    print("=" * 70)

    train_crowd = crowding[(crowding.index >= '1975-01-01') & (crowding.index <= '1990-12-31')]
    threshold_1 = train_crowd.median()
    print(f"Training threshold (1975-1990 median): {threshold_1:.4f}")

    # In-sample
    is_result = analyze_period(mom_returns, crowding, threshold_1,
                                '1975-01-01', '1990-12-31', 'IS 1975-1990')
    if is_result:
        results.append(is_result)
        print(f"\nIn-sample (1975-1990):")
        print(f"  Uncrowded: N={is_result['n_uncrowded']}, Sharpe={is_result['sharpe_uncrowded']:.2f}")
        print(f"  Crowded:   N={is_result['n_crowded']}, Sharpe={is_result['sharpe_crowded']:.2f}")
        print(f"  Differential: {is_result['differential']:+.2f}")

    # Out-of-sample
    oos_result = analyze_period(mom_returns, crowding, threshold_1,
                                 '1991-01-01', '2024-12-31', 'OOS 1991-2024')
    if oos_result:
        results.append(oos_result)
        print(f"\nOut-of-sample (1991-2024): {oos_result['n_uncrowded'] + oos_result['n_crowded']} months")
        print(f"  Uncrowded: N={oos_result['n_uncrowded']}, Sharpe={oos_result['sharpe_uncrowded']:.2f}")
        print(f"  Crowded:   N={oos_result['n_crowded']}, Sharpe={oos_result['sharpe_crowded']:.2f}")
        print(f"  Differential: {oos_result['differential']:+.2f}, p={oos_result['p_value']:.4f}")

    # ================================================================
    # TEST 2: Train 1975-1985, OOS 1986-2024 (38 years)
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Train 1975-1985, OOS 1986-2024")
    print("=" * 70)

    train_crowd_2 = crowding[(crowding.index >= '1975-01-01') & (crowding.index <= '1985-12-31')]
    threshold_2 = train_crowd_2.median()
    print(f"Training threshold (1975-1985 median): {threshold_2:.4f}")

    oos_result_2 = analyze_period(mom_returns, crowding, threshold_2,
                                   '1986-01-01', '2024-12-31', 'OOS 1986-2024')
    if oos_result_2:
        results.append(oos_result_2)
        print(f"\nOut-of-sample (1986-2024): {oos_result_2['n_uncrowded'] + oos_result_2['n_crowded']} months")
        print(f"  Uncrowded: N={oos_result_2['n_uncrowded']}, Sharpe={oos_result_2['sharpe_uncrowded']:.2f}")
        print(f"  Crowded:   N={oos_result_2['n_crowded']}, Sharpe={oos_result_2['sharpe_crowded']:.2f}")
        print(f"  Differential: {oos_result_2['differential']:+.2f}, p={oos_result_2['p_value']:.4f}")

    # ================================================================
    # TEST 3: Rolling 10-year OOS windows
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Rolling 10-Year OOS Windows")
    print("=" * 70)

    # Use 1975-1985 threshold throughout (pre-specified)
    print(f"Using fixed threshold from 1975-1985: {threshold_2:.4f}")

    decades = [
        ('1986-1995', '1986-01-01', '1995-12-31'),
        ('1996-2005', '1996-01-01', '2005-12-31'),
        ('2006-2015', '2006-01-01', '2015-12-31'),
        ('2016-2024', '2016-01-01', '2024-12-31'),
    ]

    print(f"\n{'Decade':<12} {'N_unc':<8} {'N_cro':<8} {'Sh_unc':<10} {'Sh_cro':<10} {'Diff':<10} {'p-value':<10}")
    print("-" * 68)

    decade_results = []
    for label, start, end in decades:
        result = analyze_period(mom_returns, crowding, threshold_2, start, end, label)
        if result:
            decade_results.append(result)
            results.append(result)
            print(f"{label:<12} {result['n_uncrowded']:<8} {result['n_crowded']:<8} "
                  f"{result['sharpe_uncrowded']:<10.2f} {result['sharpe_crowded']:<10.2f} "
                  f"{result['differential']:<+10.2f} {result['p_value']:<10.4f}")

    # ================================================================
    # TEST 4: Pooled analysis across all OOS decades
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 4: Pooled OOS Analysis (1986-2024)")
    print("=" * 70)

    # Combine all OOS data
    oos_start = '1986-01-01'
    oos_end = '2024-12-31'

    mask = (mom_returns.index >= oos_start) & (mom_returns.index <= oos_end)
    oos_mom = mom_returns[mask]
    oos_crowd = crowding.reindex(oos_mom.index).dropna()
    oos_mom = oos_mom.reindex(oos_crowd.index)

    is_uncrowded = oos_crowd > threshold_2

    unc_ret = oos_mom[is_uncrowded]
    cro_ret = oos_mom[~is_uncrowded]

    print(f"\nPooled OOS (1986-2024): {len(oos_mom)} months")
    print(f"  Uncrowded: N={len(unc_ret)}, Sharpe={sharpe(unc_ret):.2f}, Ann.Ret={unc_ret.mean()*12:.2%}")
    print(f"  Crowded:   N={len(cro_ret)}, Sharpe={sharpe(cro_ret):.2f}, Ann.Ret={cro_ret.mean()*12:.2%}")

    diff = sharpe(unc_ret) - sharpe(cro_ret)
    t_stat, p_val = stats.ttest_ind(unc_ret, cro_ret)
    print(f"\n  Sharpe Differential: {diff:+.2f}")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_val:.4f}")

    # Consistency check
    n_positive = sum(1 for r in decade_results if r['differential'] > 0)
    print(f"\n  Decades with positive differential: {n_positive}/{len(decade_results)}")

    # ================================================================
    # TEST 5: Bootstrap confidence interval
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 5: Bootstrap Confidence Interval for Sharpe Differential")
    print("=" * 70)

    n_bootstrap = 10000
    boot_diffs = []

    np.random.seed(42)
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx_unc = np.random.choice(len(unc_ret), size=len(unc_ret), replace=True)
        idx_cro = np.random.choice(len(cro_ret), size=len(cro_ret), replace=True)

        boot_unc = unc_ret.iloc[idx_unc]
        boot_cro = cro_ret.iloc[idx_cro]

        boot_diff = sharpe(boot_unc) - sharpe(boot_cro)
        boot_diffs.append(boot_diff)

    boot_diffs = np.array(boot_diffs)
    ci_low = np.percentile(boot_diffs, 2.5)
    ci_high = np.percentile(boot_diffs, 97.5)

    print(f"\nBootstrap results ({n_bootstrap} iterations):")
    print(f"  Mean differential: {np.mean(boot_diffs):+.2f}")
    print(f"  95% CI: [{ci_low:+.2f}, {ci_high:+.2f}]")
    print(f"  % of bootstrap samples with positive differential: {100*np.mean(boot_diffs > 0):.1f}%")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
EXTENDED OOS VALIDATION RESULTS:

1. Using 1975-1985 training threshold ({threshold_2:.3f}), applied to 38 years OOS:
   - Uncrowded Sharpe: {sharpe(unc_ret):.2f} (N={len(unc_ret)})
   - Crowded Sharpe:   {sharpe(cro_ret):.2f} (N={len(cro_ret)})
   - Differential:     {diff:+.2f}
   - p-value:          {p_val:.4f}

2. Consistency across decades:
   - {n_positive}/{len(decade_results)} decades show positive differential
   - Effect direction is {'CONSISTENT' if n_positive >= 3 else 'INCONSISTENT'}

3. Bootstrap 95% CI for Sharpe differential:
   - [{ci_low:+.2f}, {ci_high:+.2f}]
   - {'SIGNIFICANT' if ci_low > 0 else 'NOT SIGNIFICANT'} (CI {'excludes' if ci_low > 0 else 'includes'} zero)

4. Statistical interpretation:
   - p = {p_val:.4f} {'< 0.05 → SIGNIFICANT' if p_val < 0.05 else '>= 0.05 → NOT SIGNIFICANT'}
   - {100*np.mean(boot_diffs > 0):.0f}% of bootstrap samples show positive effect
""")

    # Effect size
    pooled_std = np.sqrt((unc_ret.var() * (len(unc_ret)-1) + cro_ret.var() * (len(cro_ret)-1)) /
                         (len(unc_ret) + len(cro_ret) - 2))
    cohens_d = (unc_ret.mean() - cro_ret.mean()) / pooled_std
    print(f"5. Effect size (Cohen's d): {cohens_d:.3f}")
    print(f"   Interpretation: {'Small' if abs(cohens_d) < 0.2 else 'Medium' if abs(cohens_d) < 0.5 else 'Large'} effect")


if __name__ == '__main__':
    main()
