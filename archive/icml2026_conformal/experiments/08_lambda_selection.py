"""
Experiment 08: Principled λ Selection for CW-ACI

Test the cross-validation based λ selection method.

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
evaluate_coverage = crowding_aware_module.evaluate_coverage

baselines_module = load_module('baselines', ICML_SRC / 'baselines.py')
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


def run_lambda_selection_test(
    X: np.ndarray,
    y: np.ndarray,
    crowding: np.ndarray,
    fit_size: int = 90,
    calib_size: int = 30,
    test_size: int = 12,
    alpha: float = 0.1
) -> pd.DataFrame:
    """Test λ selection across walk-forward windows."""

    results = []
    min_train = fit_size + calib_size

    start_idx = 0
    window_count = 0

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

        # Select λ using cross-validation
        optimal_lambda, selection_stats = CrowdingWeightedACI.select_lambda(
            X_fit, y_fit, X_calib, y_calib, crowding_calib,
            alpha=alpha,
            gamma=0.1,
            lambda_candidates=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
            n_folds=3,
            criterion='min_variance'
        )

        # Test with selected λ
        cwaci_selected = CrowdingWeightedACI(alpha=alpha, gamma=0.1, lambda_weight=optimal_lambda)
        cwaci_selected.fit(X_fit, y_fit, X_calib, y_calib, crowding_calib)
        selected_sets, _ = cwaci_selected.predict_sets_online(X_test, y_test, crowding_test)
        selected_metrics = evaluate_coverage(selected_sets, y_test)
        selected_by_bin = evaluate_by_crowding_bin(selected_sets, y_test, crowding_test)

        # Compare with ACI baseline
        aci = ACIConformal(alpha=alpha, gamma=0.1)
        aci.fit(X_fit, y_fit, X_calib, y_calib)
        aci_sets, _ = aci.predict_sets_online(X_test, y_test)
        aci_metrics = evaluate_coverage(aci_sets, y_test)
        aci_by_bin = evaluate_by_crowding_bin(aci_sets, y_test, crowding_test)

        # Compare with fixed λ=0.5
        cwaci_fixed = CrowdingWeightedACI(alpha=alpha, gamma=0.1, lambda_weight=0.5)
        cwaci_fixed.fit(X_fit, y_fit, X_calib, y_calib, crowding_calib)
        fixed_sets, _ = cwaci_fixed.predict_sets_online(X_test, y_test, crowding_test)
        fixed_metrics = evaluate_coverage(fixed_sets, y_test)
        fixed_by_bin = evaluate_by_crowding_bin(fixed_sets, y_test, crowding_test)

        results.append({
            'window': window_count,
            'selected_lambda': optimal_lambda,
            # Selected λ results
            'selected_coverage': selected_metrics['coverage'],
            'selected_low': selected_by_bin.get('low', np.nan),
            'selected_high': selected_by_bin.get('high', np.nan),
            'selected_variance': np.var([selected_by_bin.get('low', 0.5),
                                         selected_by_bin.get('medium', 0.5),
                                         selected_by_bin.get('high', 0.5)]),
            # ACI results
            'aci_coverage': aci_metrics['coverage'],
            'aci_low': aci_by_bin.get('low', np.nan),
            'aci_high': aci_by_bin.get('high', np.nan),
            'aci_variance': np.var([aci_by_bin.get('low', 0.5),
                                    aci_by_bin.get('medium', 0.5),
                                    aci_by_bin.get('high', 0.5)]),
            # Fixed λ=0.5 results
            'fixed_coverage': fixed_metrics['coverage'],
            'fixed_low': fixed_by_bin.get('low', np.nan),
            'fixed_high': fixed_by_bin.get('high', np.nan),
            'fixed_variance': np.var([fixed_by_bin.get('low', 0.5),
                                      fixed_by_bin.get('medium', 0.5),
                                      fixed_by_bin.get('high', 0.5)]),
        })

        start_idx += test_size
        window_count += 1

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("PRINCIPLED λ SELECTION FOR CW-ACI")
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
    lambda_selections = []

    for factor in factors.columns:
        print(f"\nProcessing {factor}...")

        X, y = fe.create_ml_dataset(features, factors, target_type='crash', factor=factor)
        crowding = crowding_df[factor].loc[X.index].values

        results = run_lambda_selection_test(X.values, y.values, crowding)
        results['factor'] = factor
        all_results.append(results)

        # Track λ selections
        lambda_selections.extend(results['selected_lambda'].tolist())

    full_results = pd.concat(all_results, ignore_index=True)

    # Summary
    print("\n" + "=" * 70)
    print("λ SELECTION ANALYSIS")
    print("=" * 70)

    print(f"\nλ Selection Distribution:")
    print(pd.Series(lambda_selections).value_counts().sort_index())

    print(f"\nMost common λ: {pd.Series(lambda_selections).mode()[0]}")
    print(f"Mean λ: {np.mean(lambda_selections):.3f}")
    print(f"Median λ: {np.median(lambda_selections):.3f}")

    # Compare methods
    print("\n" + "=" * 70)
    print("COMPARISON: Selected λ vs Fixed λ=0.5 vs ACI")
    print("=" * 70)

    comparison = {
        'Method': ['ACI', 'CW-ACI (λ=0.5)', 'CW-ACI (selected)'],
        'Marginal': [
            full_results['aci_coverage'].mean(),
            full_results['fixed_coverage'].mean(),
            full_results['selected_coverage'].mean(),
        ],
        'Low Cov': [
            full_results['aci_low'].mean(),
            full_results['fixed_low'].mean(),
            full_results['selected_low'].mean(),
        ],
        'High Cov': [
            full_results['aci_high'].mean(),
            full_results['fixed_high'].mean(),
            full_results['selected_high'].mean(),
        ],
        'Variance': [
            full_results['aci_variance'].mean(),
            full_results['fixed_variance'].mean(),
            full_results['selected_variance'].mean(),
        ],
    }

    comparison_df = pd.DataFrame(comparison)
    print("\n" + comparison_df.to_string(index=False))

    # Does selection help?
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    aci_var = full_results['aci_variance'].mean()
    selected_var = full_results['selected_variance'].mean()
    fixed_var = full_results['fixed_variance'].mean()

    improvement_vs_aci = (aci_var - selected_var) / aci_var * 100
    improvement_vs_fixed = (fixed_var - selected_var) / fixed_var * 100

    print(f"""
Selected λ vs ACI:
  Variance reduction: {improvement_vs_aci:.1f}%

Selected λ vs Fixed λ=0.5:
  Variance reduction: {improvement_vs_fixed:.1f}%

Recommendation: {'Use CV selection' if improvement_vs_fixed > 5 else 'Fixed λ=0.5 is sufficient'}
""")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    full_results.to_csv(output_dir / 'lambda_selection_results.csv', index=False)
    comparison_df.to_csv(output_dir / 'lambda_selection_comparison.csv', index=False)
    print(f"\nResults saved to: {output_dir}")

    return full_results, comparison_df


if __name__ == '__main__':
    results, comparison = main()
