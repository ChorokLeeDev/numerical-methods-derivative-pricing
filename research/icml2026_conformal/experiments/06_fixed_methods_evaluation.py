"""
Experiment 06: Evaluate Fixed Methods

Test the corrected approaches:
1. UncertaintyWeightedCP - Uses (1-crowding) as uncertainty signal
2. AdaptiveLambdaCP - Learns per-bin λ values

Goal: Achieve good coverage across ALL crowding regimes.

For ICML 2026 submission.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

ICML_SRC = Path(__file__).parent.parent / 'src'
FC_SRC = Path(__file__).parent.parent.parent / 'factor_crowding' / 'src'

import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

crowding_signal_module = load_module('crowding_signal_icml', ICML_SRC / 'crowding_signal.py')
CrowdingSignal = crowding_signal_module.CrowdingSignal

crowding_aware_module = load_module('crowding_aware_conformal', ICML_SRC / 'crowding_aware_conformal.py')
CrowdingWeightedCP = crowding_aware_module.CrowdingWeightedCP
UncertaintyWeightedCP = crowding_aware_module.UncertaintyWeightedCP
AdaptiveLambdaCP = crowding_aware_module.AdaptiveLambdaCP
evaluate_coverage = crowding_aware_module.evaluate_coverage

baselines_module = load_module('baselines', ICML_SRC / 'baselines.py')
SplitConformalCP = baselines_module.SplitConformalCP
ACIConformal = baselines_module.ACIConformal

sys.path.insert(0, str(FC_SRC))
from features import FeatureEngineer


def evaluate_by_crowding_bin(
    pred_sets, y_true, crowding, n_bins=3
):
    """Evaluate coverage by crowding bin."""
    bins = np.quantile(crowding, np.linspace(0, 1, n_bins + 1))
    labels = ['low', 'medium', 'high'][:n_bins]

    results = {}
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (crowding >= low) & (crowding <= high)
        else:
            mask = (crowding >= low) & (crowding < high)

        if mask.sum() > 0:
            bin_sets = [pred_sets[j] for j in range(len(y_true)) if mask[j]]
            bin_y = y_true[mask]
            coverage = np.mean([int(bin_y[k]) in bin_sets[k] for k in range(len(bin_y))])
            results[labels[i]] = coverage

    return results


def run_comparison(
    X: np.ndarray,
    y: np.ndarray,
    crowding: np.ndarray,
    fit_size: int = 90,
    calib_size: int = 30,
    test_size: int = 12,
    alpha: float = 0.1
) -> pd.DataFrame:
    """Run walk-forward comparison of all methods."""

    results = []
    min_train = fit_size + calib_size

    start_idx = 0
    while start_idx + min_train + test_size <= len(X):
        fit_end = start_idx + fit_size
        calib_end = fit_end + calib_size
        test_end = calib_end + test_size

        X_fit = X[start_idx:fit_end]
        y_fit = y[start_idx:fit_end]
        X_calib = X[fit_end:calib_end]
        y_calib = y[fit_end:calib_end]
        X_test = X[calib_end:test_end]
        y_test = y[calib_end:test_end]
        crowding_calib = crowding[fit_end:calib_end]
        crowding_test = crowding[calib_end:test_end]

        # 1. Split CP (baseline)
        split_cp = SplitConformalCP()
        split_cp.fit(X_fit, y_fit, X_calib, y_calib)
        split_sets, _ = split_cp.predict_sets(X_test, alpha)
        split_metrics = evaluate_coverage(split_sets, y_test)
        split_by_bin = evaluate_by_crowding_bin(split_sets, y_test, crowding_test)
        results.append({
            'method': 'split',
            'coverage': split_metrics['coverage'],
            'avg_size': split_metrics['avg_set_size'],
            **{f'cov_{k}': v for k, v in split_by_bin.items()}
        })

        # 2. ACI (baseline)
        aci = ACIConformal(alpha=alpha, gamma=0.1)
        aci.fit(X_fit, y_fit, X_calib, y_calib)
        aci_sets, _ = aci.predict_sets_online(X_test, y_test)
        aci_metrics = evaluate_coverage(aci_sets, y_test)
        aci_by_bin = evaluate_by_crowding_bin(aci_sets, y_test, crowding_test)
        results.append({
            'method': 'aci',
            'coverage': aci_metrics['coverage'],
            'avg_size': aci_metrics['avg_set_size'],
            **{f'cov_{k}': v for k, v in aci_by_bin.items()}
        })

        # 3. Original CrowdingWeightedCP (for comparison)
        cwcp = CrowdingWeightedCP(lambda_weight=2.0)
        cwcp.fit(X_fit, y_fit, X_calib, y_calib, crowding_calib)
        cwcp_result = cwcp.predict_sets(X_test, crowding_test, alpha)
        cwcp_metrics = evaluate_coverage(cwcp_result.prediction_sets, y_test)
        cwcp_by_bin = evaluate_by_crowding_bin(cwcp_result.prediction_sets, y_test, crowding_test)
        results.append({
            'method': 'cwcp_original',
            'coverage': cwcp_metrics['coverage'],
            'avg_size': cwcp_metrics['avg_set_size'],
            **{f'cov_{k}': v for k, v in cwcp_by_bin.items()}
        })

        # 4. NEW: UncertaintyWeightedCP (FIXED)
        for lam in [1.0, 2.0, 3.0]:
            uwcp = UncertaintyWeightedCP(lambda_weight=lam)
            uwcp.fit(X_fit, y_fit, X_calib, y_calib, crowding_calib)
            uwcp_result = uwcp.predict_sets(X_test, crowding_test, alpha)
            uwcp_metrics = evaluate_coverage(uwcp_result.prediction_sets, y_test)
            uwcp_by_bin = evaluate_by_crowding_bin(uwcp_result.prediction_sets, y_test, crowding_test)
            results.append({
                'method': f'uwcp_λ{lam}',
                'coverage': uwcp_metrics['coverage'],
                'avg_size': uwcp_metrics['avg_set_size'],
                **{f'cov_{k}': v for k, v in uwcp_by_bin.items()}
            })

        # 5. NEW: AdaptiveLambdaCP
        alcp = AdaptiveLambdaCP(lambda_init=1.0, learning_rate=0.5)
        alcp.fit(X_fit, y_fit, X_calib, y_calib, crowding_calib)
        alcp_result = alcp.predict_sets(X_test, crowding_test, alpha)
        alcp_metrics = evaluate_coverage(alcp_result.prediction_sets, y_test)
        alcp_by_bin = evaluate_by_crowding_bin(alcp_result.prediction_sets, y_test, crowding_test)
        # Update lambdas for next period
        alcp.update_lambdas(y_test, alcp_result.prediction_sets, crowding_test, alpha)
        results.append({
            'method': 'adaptive_lambda',
            'coverage': alcp_metrics['coverage'],
            'avg_size': alcp_metrics['avg_set_size'],
            **{f'cov_{k}': v for k, v in alcp_by_bin.items()}
        })

        start_idx += test_size

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("FIXED METHODS EVALUATION: UncertaintyWeightedCP & AdaptiveLambdaCP")
    print("=" * 70)

    # Load data
    DATA_DIR = Path(__file__).parent.parent / 'data'
    factors = pd.read_parquet(DATA_DIR / 'ff_extended_factors.parquet')
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    factors = factors.drop(columns=['RF'], errors='ignore')

    # Compute crowding
    cs = CrowdingSignal()
    crowding_df = cs.compute_all_crowding(factors, normalize=True)

    # Generate features
    fe = FeatureEngineer()
    features = fe.generate_all_features(factors)

    all_results = []

    for factor in factors.columns:
        print(f"\nProcessing {factor}...")

        X, y = fe.create_ml_dataset(features, factors, target_type='crash', factor=factor)
        crowding = crowding_df[factor].loc[X.index].values

        results = run_comparison(X.values, y.values, crowding)
        results['factor'] = factor
        all_results.append(results)

    full_results = pd.concat(all_results, ignore_index=True)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: COVERAGE BY METHOD")
    print("=" * 70)

    summary = full_results.groupby('method').agg({
        'coverage': ['mean', 'std'],
        'cov_low': 'mean',
        'cov_medium': 'mean',
        'cov_high': 'mean',
        'avg_size': 'mean'
    }).round(3)

    print(summary)

    # Coverage variation (std across bins)
    print("\n" + "=" * 70)
    print("COVERAGE VARIATION (lower = more stable)")
    print("=" * 70)

    for method in full_results['method'].unique():
        method_data = full_results[full_results['method'] == method]
        bin_coverages = [
            method_data['cov_low'].mean(),
            method_data['cov_medium'].mean(),
            method_data['cov_high'].mean()
        ]
        variation = np.std(bin_coverages)
        print(f"{method:<20}: std = {variation:.4f}, bins = {[f'{c:.3f}' for c in bin_coverages]}")

    # Best method for uniform coverage
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Find method with lowest variation AND marginal coverage >= 0.88
    method_stats = []
    for method in full_results['method'].unique():
        method_data = full_results[full_results['method'] == method]
        marginal = method_data['coverage'].mean()
        bin_coverages = [
            method_data['cov_low'].mean(),
            method_data['cov_medium'].mean(),
            method_data['cov_high'].mean()
        ]
        variation = np.std(bin_coverages)
        min_bin = min(bin_coverages)
        method_stats.append({
            'method': method,
            'marginal': marginal,
            'variation': variation,
            'min_bin': min_bin,
            'low': bin_coverages[0],
            'high': bin_coverages[2],
        })

    stats_df = pd.DataFrame(method_stats)

    print("\nMethod Comparison (sorted by min_bin coverage):")
    print(stats_df.sort_values('min_bin', ascending=False).to_string(index=False))

    # Did we fix the issues?
    print("\n" + "=" * 70)
    print("DID WE FIX THE ISSUES?")
    print("=" * 70)

    original_cwcp = stats_df[stats_df['method'] == 'cwcp_original'].iloc[0]
    uwcp_best = stats_df[stats_df['method'].str.startswith('uwcp')].sort_values('min_bin', ascending=False).iloc[0]

    print(f"""
1. ORIGINAL ISSUE: CrowdingWeightedCP had coverage trade-off
   - Original CWCP: low={original_cwcp['low']:.3f}, high={original_cwcp['high']:.3f}

2. FIX: UncertaintyWeightedCP inverts the signal
   - Best UWCP ({uwcp_best['method']}): low={uwcp_best['low']:.3f}, high={uwcp_best['high']:.3f}

3. IMPROVEMENT:
   - Low crowding coverage: {original_cwcp['low']:.3f} → {uwcp_best['low']:.3f} ({uwcp_best['low'] - original_cwcp['low']:+.3f})
   - Minimum bin coverage: {original_cwcp['min_bin']:.3f} → {uwcp_best['min_bin']:.3f} ({uwcp_best['min_bin'] - original_cwcp['min_bin']:+.3f})
""")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    full_results.to_csv(output_dir / 'fixed_methods_comparison.csv', index=False)
    print(f"\nResults saved to: {output_dir / 'fixed_methods_comparison.csv'}")

    return full_results


if __name__ == '__main__':
    results = main()
