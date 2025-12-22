"""
Experiment 1: Coverage Analysis

Core experiment comparing standard conformal prediction vs CW-ACI
on Fama-French factor returns.

Research Question: Does standard CP under-cover during high-crowding periods,
and does CW-ACI fix this?

Author: Chorok Lee (KAIST)
Date: December 2024
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from conformal import (
    StandardConformalPredictor,
    CrowdingWeightedACI,
    compute_coverage,
    compute_conditional_coverage
)
from crowding import compute_crowding_proxy, classify_crowding_regime


def run_coverage_analysis(
    returns: pd.Series,
    crowding: pd.Series,
    alpha: float = 0.1,
    cal_fraction: float = 0.5,
    verbose: bool = True
) -> dict:
    """
    Run coverage analysis comparing standard CP and CW-ACI.

    Parameters
    ----------
    returns : pd.Series
        Factor returns
    crowding : pd.Series
        Crowding signal
    alpha : float
        Miscoverage rate (default 0.1 for 90% coverage)
    cal_fraction : float
        Fraction of data used for calibration
    verbose : bool
        Print progress

    Returns
    -------
    dict with results for both methods
    """
    # Drop NaN
    valid = returns.notna() & crowding.notna()
    returns = returns[valid].values
    crowding = crowding[valid].values

    n = len(returns)
    cal_end = int(n * cal_fraction)

    # Split data
    y_cal, y_test = returns[:cal_end], returns[cal_end:]
    crowd_cal, crowd_test = crowding[:cal_end], crowding[cal_end:]

    # Use mean predictor (naive baseline)
    pred_cal = np.full_like(y_cal, np.mean(y_cal))
    pred_test = np.full_like(y_test, np.mean(y_cal))

    # Classify high/low crowding
    crowding_median = np.median(crowd_test)
    high_crowding = crowd_test > crowding_median

    results = {}

    # ===== Standard CP =====
    scp = StandardConformalPredictor(alpha=alpha)
    scp.fit(y_cal, pred_cal)
    lower_scp, upper_scp = scp.predict(pred_test)

    cov_scp = compute_conditional_coverage(y_test, lower_scp, upper_scp, high_crowding)

    results['standard'] = {
        'overall_coverage': cov_scp['overall'],
        'high_crowding_coverage': cov_scp['high'],
        'low_crowding_coverage': cov_scp['low'],
        'interval_width': scp.get_width(),
        'width_high': scp.get_width(),
        'width_low': scp.get_width()
    }

    # ===== CW-ACI =====
    cwaci = CrowdingWeightedACI(alpha=alpha, sensitivity=1.0)
    cwaci.fit(y_cal, pred_cal, crowd_cal)
    lower_cw, upper_cw, width_cw = cwaci.predict(pred_test, crowd_test)

    cov_cw = compute_conditional_coverage(y_test, lower_cw, upper_cw, high_crowding)

    results['cwaci'] = {
        'overall_coverage': cov_cw['overall'],
        'high_crowding_coverage': cov_cw['high'],
        'low_crowding_coverage': cov_cw['low'],
        'interval_width': np.mean(width_cw),
        'width_high': np.mean(width_cw[high_crowding]),
        'width_low': np.mean(width_cw[~high_crowding])
    }

    # Compute improvement
    results['improvement'] = {
        'high_crowding_coverage_gain': (
            results['cwaci']['high_crowding_coverage'] -
            results['standard']['high_crowding_coverage']
        ),
        'width_adaptation_ratio': (
            results['cwaci']['width_high'] / results['cwaci']['width_low']
        )
    }

    if verbose:
        print(f"\n  Standard CP:")
        print(f"    Overall: {results['standard']['overall_coverage']:.1%}")
        print(f"    High crowding: {results['standard']['high_crowding_coverage']:.1%}")
        print(f"    Low crowding: {results['standard']['low_crowding_coverage']:.1%}")
        print(f"  CW-ACI:")
        print(f"    Overall: {results['cwaci']['overall_coverage']:.1%}")
        print(f"    High crowding: {results['cwaci']['high_crowding_coverage']:.1%}")
        print(f"    Low crowding: {results['cwaci']['low_crowding_coverage']:.1%}")
        print(f"  Improvement: {results['improvement']['high_crowding_coverage_gain']:+.1%}")
        print(f"  Width ratio (high/low): {results['improvement']['width_adaptation_ratio']:.2f}")

    return results


def main():
    """Run coverage analysis on all factors."""
    print("="*70)
    print("COVERAGE ANALYSIS: Standard CP vs CW-ACI")
    print("="*70)

    # Load data
    sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
    from download_data import load_or_download_factors

    factors = load_or_download_factors()
    if factors is None:
        print("ERROR: Could not load factor data")
        return

    print(f"\nData period: {factors.index.min()} to {factors.index.max()}")
    print(f"Observations: {len(factors)}")

    # Factors to analyze (exclude Mkt-RF since it's the market)
    factor_names = ['SMB', 'HML', 'RMW', 'CMA', 'Mom']

    # Store all results
    all_results = {}

    for factor in factor_names:
        print(f"\n--- {factor} ---")

        if factor not in factors.columns:
            print(f"  Factor {factor} not found, skipping")
            continue

        returns = factors[factor]

        # Compute crowding proxy
        crowding = compute_crowding_proxy(returns, window=12)

        # Run analysis
        results = run_coverage_analysis(returns, crowding, alpha=0.1)
        all_results[factor] = results

    # ===== Summary Table =====
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)

    print("\n" + "-"*70)
    print(f"{'Factor':<8} | {'Standard CP':<30} | {'CW-ACI':<25} | Gain")
    print(f"{'':8} | {'Overall':>8} {'High':>10} {'Low':>10} | {'Overall':>8} {'High':>8} {'Low':>7} |")
    print("-"*70)

    for factor, res in all_results.items():
        std = res['standard']
        cw = res['cwaci']
        gain = res['improvement']['high_crowding_coverage_gain']

        print(f"{factor:<8} | {std['overall_coverage']:>7.1%} {std['high_crowding_coverage']:>10.1%} "
              f"{std['low_crowding_coverage']:>10.1%} | {cw['overall_coverage']:>7.1%} "
              f"{cw['high_crowding_coverage']:>8.1%} {cw['low_crowding_coverage']:>7.1%} | {gain:>+5.1%}")

    print("-"*70)

    # Average
    avg_std_overall = np.mean([r['standard']['overall_coverage'] for r in all_results.values()])
    avg_std_high = np.mean([r['standard']['high_crowding_coverage'] for r in all_results.values()])
    avg_std_low = np.mean([r['standard']['low_crowding_coverage'] for r in all_results.values()])
    avg_cw_overall = np.mean([r['cwaci']['overall_coverage'] for r in all_results.values()])
    avg_cw_high = np.mean([r['cwaci']['high_crowding_coverage'] for r in all_results.values()])
    avg_cw_low = np.mean([r['cwaci']['low_crowding_coverage'] for r in all_results.values()])
    avg_gain = np.mean([r['improvement']['high_crowding_coverage_gain'] for r in all_results.values()])

    print(f"{'Average':<8} | {avg_std_overall:>7.1%} {avg_std_high:>10.1%} "
          f"{avg_std_low:>10.1%} | {avg_cw_overall:>7.1%} "
          f"{avg_cw_high:>8.1%} {avg_cw_low:>7.1%} | {avg_gain:>+5.1%}")
    print("="*70)

    # Width adaptation
    print("\n--- Interval Width Adaptation ---")
    print(f"{'Factor':<8} | {'Standard Width':>14} | {'CW-ACI High':>12} | {'CW-ACI Low':>11} | {'Ratio':>6}")
    print("-"*60)
    for factor, res in all_results.items():
        std_w = res['standard']['interval_width']
        cw_high = res['cwaci']['width_high']
        cw_low = res['cwaci']['width_low']
        ratio = res['improvement']['width_adaptation_ratio']
        print(f"{factor:<8} | {std_w:>13.4f} | {cw_high:>11.4f} | {cw_low:>10.4f} | {ratio:>6.2f}")
    print("-"*60)

    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    under_coverage_factors = [f for f, r in all_results.items()
                             if r['standard']['high_crowding_coverage'] < 0.85]
    improved_factors = [f for f, r in all_results.items()
                       if r['improvement']['high_crowding_coverage_gain'] > 0.05]

    print(f"\n1. Standard CP under-covers during high crowding:")
    print(f"   - {len(under_coverage_factors)}/{len(all_results)} factors below 85% target")
    print(f"   - Average high-crowding coverage: {avg_std_high:.1%} (target: 90%)")
    print(f"   - Coverage gap: {0.90 - avg_std_high:.1%}")

    print(f"\n2. CW-ACI improves high-crowding coverage:")
    print(f"   - {len(improved_factors)}/{len(all_results)} factors improved by >5pp")
    print(f"   - Average improvement: {avg_gain:+.1%}")
    print(f"   - New high-crowding coverage: {avg_cw_high:.1%}")

    avg_ratio = np.mean([r['improvement']['width_adaptation_ratio'] for r in all_results.values()])
    print(f"\n3. Width adaptation:")
    print(f"   - Average high/low width ratio: {avg_ratio:.2f}")
    print(f"   - CW-ACI produces {(avg_ratio-1)*100:.0f}% wider intervals during high crowding")

    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    # Create results DataFrame
    rows = []
    for factor, res in all_results.items():
        rows.append({
            'factor': factor,
            'scp_overall': res['standard']['overall_coverage'],
            'scp_high': res['standard']['high_crowding_coverage'],
            'scp_low': res['standard']['low_crowding_coverage'],
            'cwaci_overall': res['cwaci']['overall_coverage'],
            'cwaci_high': res['cwaci']['high_crowding_coverage'],
            'cwaci_low': res['cwaci']['low_crowding_coverage'],
            'gain_high': res['improvement']['high_crowding_coverage_gain'],
            'width_ratio': res['improvement']['width_adaptation_ratio']
        })

    df_results = pd.DataFrame(rows)
    df_results.to_csv(results_dir / 'coverage_analysis.csv', index=False)
    print(f"\nResults saved to {results_dir / 'coverage_analysis.csv'}")


if __name__ == '__main__':
    main()
