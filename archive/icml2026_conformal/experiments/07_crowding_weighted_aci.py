"""
Experiment 07: Crowding-Weighted ACI (CW-ACI)

Test the hybrid approach that combines:
- Crowding-weighted nonconformity scores (from UWCP)
- ACI's online threshold adaptation

Hypothesis: This should outperform both pure ACI and static crowding methods.

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
CrowdingWeightedACI = crowding_aware_module.CrowdingWeightedACI
UncertaintyWeightedCP = crowding_aware_module.UncertaintyWeightedCP
evaluate_coverage = crowding_aware_module.evaluate_coverage

baselines_module = load_module('baselines', ICML_SRC / 'baselines.py')
SplitConformalCP = baselines_module.SplitConformalCP
ACIConformal = baselines_module.ACIConformal

sys.path.insert(0, str(FC_SRC))
from features import FeatureEngineer


def evaluate_by_crowding_bin(pred_sets, y_true, crowding, n_bins=3):
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
    """Run walk-forward comparison focusing on ACI vs CW-ACI."""

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

        # 2. ACI (strong baseline)
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

        # 3. NEW: Crowding-Weighted ACI (multiple λ values)
        for lam in [0.5, 1.0, 1.5, 2.0]:
            cwaci = CrowdingWeightedACI(alpha=alpha, gamma=0.1, lambda_weight=lam)
            cwaci.fit(X_fit, y_fit, X_calib, y_calib, crowding_calib)
            cwaci_sets, _ = cwaci.predict_sets_online(X_test, y_test, crowding_test)
            cwaci_metrics = evaluate_coverage(cwaci_sets, y_test)
            cwaci_by_bin = evaluate_by_crowding_bin(cwaci_sets, y_test, crowding_test)
            results.append({
                'method': f'cw_aci_λ{lam}',
                'coverage': cwaci_metrics['coverage'],
                'avg_size': cwaci_metrics['avg_set_size'],
                **{f'cov_{k}': v for k, v in cwaci_by_bin.items()}
            })

        # 4. UncertaintyWeightedCP (static, for comparison)
        uwcp = UncertaintyWeightedCP(lambda_weight=1.0)
        uwcp.fit(X_fit, y_fit, X_calib, y_calib, crowding_calib)
        uwcp_result = uwcp.predict_sets(X_test, crowding_test, alpha)
        uwcp_metrics = evaluate_coverage(uwcp_result.prediction_sets, y_test)
        uwcp_by_bin = evaluate_by_crowding_bin(uwcp_result.prediction_sets, y_test, crowding_test)
        results.append({
            'method': 'uwcp_static',
            'coverage': uwcp_metrics['coverage'],
            'avg_size': uwcp_metrics['avg_set_size'],
            **{f'cov_{k}': v for k, v in uwcp_by_bin.items()}
        })

        start_idx += test_size

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("CROWDING-WEIGHTED ACI (CW-ACI) EVALUATION")
    print("Hypothesis: ACI + Crowding Weighting > Pure ACI")
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
    print("SUMMARY: ACI vs CROWDING-WEIGHTED ACI")
    print("=" * 70)

    summary = full_results.groupby('method').agg({
        'coverage': ['mean', 'std'],
        'cov_low': 'mean',
        'cov_medium': 'mean',
        'cov_high': 'mean',
        'avg_size': 'mean'
    }).round(3)

    print("\n" + summary.to_string())

    # Coverage variation
    print("\n" + "=" * 70)
    print("COVERAGE STABILITY (std across bins, lower = better)")
    print("=" * 70)

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
        avg_size = method_data['avg_size'].mean()

        method_stats.append({
            'method': method,
            'marginal': marginal,
            'low': bin_coverages[0],
            'med': bin_coverages[1],
            'high': bin_coverages[2],
            'variation': variation,
            'min_bin': min_bin,
            'avg_size': avg_size
        })

        print(f"{method:<15}: std={variation:.4f}, bins=[{bin_coverages[0]:.3f}, {bin_coverages[1]:.3f}, {bin_coverages[2]:.3f}]")

    stats_df = pd.DataFrame(method_stats)

    # Key comparison: ACI vs best CW-ACI
    print("\n" + "=" * 70)
    print("KEY COMPARISON: ACI vs CROWDING-WEIGHTED ACI")
    print("=" * 70)

    aci_stats = stats_df[stats_df['method'] == 'aci'].iloc[0]
    cwaci_best = stats_df[stats_df['method'].str.startswith('cw_aci')].sort_values('min_bin', ascending=False).iloc[0]

    print(f"""
ACI (baseline):
  Marginal coverage: {aci_stats['marginal']:.3f}
  Coverage by bin:   low={aci_stats['low']:.3f}, med={aci_stats['med']:.3f}, high={aci_stats['high']:.3f}
  Bin variation:     {aci_stats['variation']:.4f}
  Min bin coverage:  {aci_stats['min_bin']:.3f}
  Avg set size:      {aci_stats['avg_size']:.3f}

Best CW-ACI ({cwaci_best['method']}):
  Marginal coverage: {cwaci_best['marginal']:.3f}
  Coverage by bin:   low={cwaci_best['low']:.3f}, med={cwaci_best['med']:.3f}, high={cwaci_best['high']:.3f}
  Bin variation:     {cwaci_best['variation']:.4f}
  Min bin coverage:  {cwaci_best['min_bin']:.3f}
  Avg set size:      {cwaci_best['avg_size']:.3f}

IMPROVEMENT:
  Min bin coverage:  {aci_stats['min_bin']:.3f} → {cwaci_best['min_bin']:.3f} ({cwaci_best['min_bin'] - aci_stats['min_bin']:+.3f})
  Variation:         {aci_stats['variation']:.4f} → {cwaci_best['variation']:.4f} ({cwaci_best['variation'] - aci_stats['variation']:+.4f})
""")

    # Does CW-ACI beat ACI?
    if cwaci_best['min_bin'] > aci_stats['min_bin'] and cwaci_best['marginal'] >= 0.85:
        print("CONCLUSION: CW-ACI IMPROVES over ACI!")
        print("  → Crowding weighting + ACI adaptation = better coverage uniformity")
    elif cwaci_best['marginal'] >= aci_stats['marginal'] - 0.02:
        print("CONCLUSION: CW-ACI is COMPARABLE to ACI")
        print("  → Crowding signal doesn't hurt, but doesn't significantly help either")
    else:
        print("CONCLUSION: Pure ACI remains BEST")
        print("  → Crowding signal may be adding noise rather than information")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    full_results.to_csv(output_dir / 'crowding_weighted_aci_comparison.csv', index=False)
    stats_df.to_csv(output_dir / 'cwaci_summary_stats.csv', index=False)
    print(f"\nResults saved to: {output_dir}")

    return full_results, stats_df


if __name__ == '__main__':
    results, stats = main()
