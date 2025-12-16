"""
Out-of-Sample Tail Risk Validation

Clean OOS test:
- Train: 1980-2000 (establish crash threshold)
- OOS: 2001-2024 (apply and measure)
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


def main():
    print("=" * 70)
    print("OUT-OF-SAMPLE TAIL RISK VALIDATION")
    print("=" * 70)

    factors = load_data()
    factors = factors.drop(columns=['RF'], errors='ignore')

    # Compute signals
    detector = CrowdingDetector(train_window=120, prediction_gap=12, sharpe_window=36)
    signals = detector.compute_multi_factor_signals(factors)

    residual_df = pd.DataFrame({f: signals[f]['residual'] for f in signals if len(signals[f]) > 0})

    # Align
    common_idx = factors.index.intersection(residual_df.index)
    factors_aligned = factors.loc[common_idx]
    residuals_aligned = residual_df.loc[common_idx]

    # ================================================================
    # SPLIT: Train 1980-2000, OOS 2001-2024
    # ================================================================
    train_end = '2000-12-31'
    oos_start = '2001-01-01'

    train_mask = factors_aligned.index <= train_end
    oos_mask = factors_aligned.index >= oos_start

    print(f"\nTrain period: 1980 - 2000")
    print(f"OOS period:   2001 - 2024")
    print(f"Train months: {train_mask.sum()}")
    print(f"OOS months:   {oos_mask.sum()}")

    # ================================================================
    # ESTABLISH THRESHOLDS FROM TRAINING DATA ONLY
    # ================================================================
    print("\n" + "-" * 70)
    print("ESTABLISHING THRESHOLDS (Training Period Only)")
    print("-" * 70)

    thresholds = {}
    crash_thresholds = {}

    for factor in residuals_aligned.columns:
        train_resid = residuals_aligned.loc[train_mask, factor].dropna()
        train_ret = factors_aligned.loc[train_mask, factor]

        # Crowding threshold: training median
        thresholds[factor] = train_resid.median()

        # Crash threshold: training 10th percentile of returns
        crash_thresholds[factor] = train_ret.quantile(0.10)

        print(f"{factor}: crowding_thresh={thresholds[factor]:.3f}, crash_thresh={crash_thresholds[factor]:.2%}")

    # ================================================================
    # OUT-OF-SAMPLE TAIL RISK TEST
    # ================================================================
    print("\n" + "-" * 70)
    print("OUT-OF-SAMPLE TAIL RISK PREDICTION")
    print("-" * 70)

    oos_results = []

    for factor in residuals_aligned.columns:
        oos_resid = residuals_aligned.loc[oos_mask, factor].dropna()
        oos_ret = factors_aligned.loc[oos_mask, factor]

        # Forward return (next month)
        fwd_ret = oos_ret.shift(-1)

        # Align
        common = oos_resid.index.intersection(fwd_ret.dropna().index)
        if len(common) < 50:
            continue

        resid_common = oos_resid.loc[common]
        fwd_common = fwd_ret.loc[common]

        # Apply TRAINING thresholds to OOS data
        crowding_thresh = thresholds[factor]
        crash_thresh = crash_thresholds[factor]

        is_crowded = resid_common < crowding_thresh
        is_crash = fwd_common < crash_thresh

        # Crash probabilities
        n_crowded = is_crowded.sum()
        n_uncrowded = (~is_crowded).sum()

        crashes_when_crowded = (is_crash & is_crowded).sum()
        crashes_when_uncrowded = (is_crash & ~is_crowded).sum()

        prob_crash_crowded = crashes_when_crowded / n_crowded if n_crowded > 0 else 0
        prob_crash_uncrowded = crashes_when_uncrowded / n_uncrowded if n_uncrowded > 0 else 0

        ratio = prob_crash_crowded / prob_crash_uncrowded if prob_crash_uncrowded > 0 else np.nan

        # Chi-square test for independence
        contingency = [
            [crashes_when_crowded, n_crowded - crashes_when_crowded],
            [crashes_when_uncrowded, n_uncrowded - crashes_when_uncrowded]
        ]
        if min(min(row) for row in contingency) >= 5:
            chi2, p_value = stats.chi2_contingency(contingency)[:2]
        else:
            # Fisher's exact for small samples
            _, p_value = stats.fisher_exact(contingency)
            chi2 = np.nan

        oos_results.append({
            'factor': factor,
            'n_crowded': n_crowded,
            'n_uncrowded': n_uncrowded,
            'prob_crash_crowded': prob_crash_crowded,
            'prob_crash_uncrowded': prob_crash_uncrowded,
            'ratio': ratio,
            'p_value': p_value,
        })

    # Display results
    print(f"\n{'Factor':<10} {'N_cro':<8} {'N_unc':<8} {'P(crash|cro)':<14} {'P(crash|unc)':<14} {'Ratio':<8} {'p-value':<10}")
    print("-" * 82)

    for r in oos_results:
        sig = "***" if r['p_value'] < 0.01 else "**" if r['p_value'] < 0.05 else "*" if r['p_value'] < 0.1 else ""
        print(f"{r['factor']:<10} {r['n_crowded']:<8} {r['n_uncrowded']:<8} "
              f"{r['prob_crash_crowded']:<14.1%} {r['prob_crash_uncrowded']:<14.1%} "
              f"{r['ratio']:<8.2f} {r['p_value']:<10.4f} {sig}")

    # Summary statistics
    results_df = pd.DataFrame(oos_results)
    mean_ratio = results_df['ratio'].mean()
    n_significant = (results_df['p_value'] < 0.05).sum()
    n_positive = (results_df['ratio'] > 1).sum()

    print(f"\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"Mean crash probability ratio: {mean_ratio:.2f}x")
    print(f"Factors with ratio > 1: {n_positive}/{len(results_df)}")
    print(f"Factors significant (p<0.05): {n_significant}/{len(results_df)}")

    # ================================================================
    # POOLED ANALYSIS
    # ================================================================
    print(f"\n" + "-" * 70)
    print("POOLED ANALYSIS (All Factors Combined)")
    print("-" * 70)

    all_crashes_crowded = sum(r['prob_crash_crowded'] * r['n_crowded'] for r in oos_results)
    all_n_crowded = sum(r['n_crowded'] for r in oos_results)
    all_crashes_uncrowded = sum(r['prob_crash_uncrowded'] * r['n_uncrowded'] for r in oos_results)
    all_n_uncrowded = sum(r['n_uncrowded'] for r in oos_results)

    pooled_prob_crowded = all_crashes_crowded / all_n_crowded
    pooled_prob_uncrowded = all_crashes_uncrowded / all_n_uncrowded
    pooled_ratio = pooled_prob_crowded / pooled_prob_uncrowded

    print(f"Pooled P(crash | crowded):   {pooled_prob_crowded:.1%}")
    print(f"Pooled P(crash | uncrowded): {pooled_prob_uncrowded:.1%}")
    print(f"Pooled ratio:                {pooled_ratio:.2f}x")

    # ================================================================
    # ECONOMIC INTERPRETATION
    # ================================================================
    print(f"\n" + "-" * 70)
    print("ECONOMIC INTERPRETATION")
    print("-" * 70)
    print(f"""
OUT-OF-SAMPLE (2001-2024) FINDINGS:

1. Crowded factors are {pooled_ratio:.0%} more likely to experience extreme losses
   (defined as bottom 10% of returns)

2. This is NOT about predicting average returns (which are efficiently priced)
   This IS about predicting TAIL RISK (crowded positions unwind together)

3. Consistency: {n_positive}/{len(results_df)} factors show higher crash risk when crowded

4. Statistical significance: {n_significant}/{len(results_df)} factors significant at p<0.05

ACTIONABLE USE:
- Position sizing: Reduce exposure to crowded factors
- Stop-loss: Tighter stops on crowded positions
- Portfolio construction: Diversify away from crowded factors
""")

    # For paper
    print("-" * 70)
    print("FOR PAPER:")
    print("-" * 70)

    # Get top 3 most significant
    sorted_results = sorted(oos_results, key=lambda x: x['ratio'], reverse=True)[:3]
    for r in sorted_results:
        print(f"  {r['factor']}: {r['ratio']:.1f}x crash risk when crowded (p={r['p_value']:.3f})")


if __name__ == '__main__':
    main()
