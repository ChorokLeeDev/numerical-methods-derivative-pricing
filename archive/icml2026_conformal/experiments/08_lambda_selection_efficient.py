"""
Experiment 08 (EFFICIENT): Principled λ Selection for CW-ACI

Full test on all 8 factors with optimizations:
- 5 λ candidates (0, 0.25, 0.5, 0.75, 1.0)
- 2 folds (not 3)
- Sample every 3rd walk-forward window
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


def run_efficient_test(
    X: np.ndarray,
    y: np.ndarray,
    crowding: np.ndarray,
    fit_size: int = 90,
    calib_size: int = 30,
    test_size: int = 12,
    alpha: float = 0.1,
    window_step: int = 3  # Sample every 3rd window
) -> pd.DataFrame:
    """Efficient walk-forward test sampling every nth window."""

    results = []
    min_train = fit_size + calib_size

    start_idx = 0
    window_count = 0

    while start_idx + min_train + test_size <= len(X):
        # Only process every nth window
        if window_count % window_step == 0:
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

            # Select λ using CV (efficient: 5 candidates, 2 folds)
            optimal_lambda, _ = CrowdingWeightedACI.select_lambda(
                X_fit, y_fit, X_calib, y_calib, crowding_calib,
                alpha=alpha,
                lambda_candidates=[0.0, 0.25, 0.5, 0.75, 1.0],
                n_folds=2,
                criterion='min_variance'
            )

            # Test with selected λ
            cwaci = CrowdingWeightedACI(alpha=alpha, gamma=0.1, lambda_weight=optimal_lambda)
            cwaci.fit(X_fit, y_fit, X_calib, y_calib, crowding_calib)
            sets, _ = cwaci.predict_sets_online(X_test, y_test, crowding_test)
            metrics = evaluate_coverage(sets, y_test)
            by_bin = evaluate_by_crowding_bin(sets, y_test, crowding_test)

            # Also test ACI for comparison
            aci = ACIConformal(alpha=alpha, gamma=0.1)
            aci.fit(X_fit, y_fit, X_calib, y_calib)
            aci_sets, _ = aci.predict_sets_online(X_test, y_test)
            aci_metrics = evaluate_coverage(aci_sets, y_test)
            aci_by_bin = evaluate_by_crowding_bin(aci_sets, y_test, crowding_test)

            results.append({
                'window': window_count,
                'selected_lambda': optimal_lambda,
                'cwaci_coverage': metrics['coverage'],
                'cwaci_variance': np.var([by_bin.get('low', 0.5), by_bin.get('medium', 0.5), by_bin.get('high', 0.5)]),
                'aci_coverage': aci_metrics['coverage'],
                'aci_variance': np.var([aci_by_bin.get('low', 0.5), aci_by_bin.get('medium', 0.5), aci_by_bin.get('high', 0.5)]),
            })

        start_idx += test_size
        window_count += 1

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("EFFICIENT λ SELECTION TEST")
    print("8 factors × 5 λs × 2 folds × sampled windows")
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

    for i, factor in enumerate(factors.columns):
        print(f"\n[{i+1}/{len(factors.columns)}] Processing {factor}...")

        X, y = fe.create_ml_dataset(features, factors, target_type='crash', factor=factor)
        crowding = crowding_df[factor].loc[X.index].values

        results = run_efficient_test(X.values, y.values, crowding)
        results['factor'] = factor
        all_results.append(results)

        # Track λ selections
        factor_lambdas = results['selected_lambda'].tolist()
        lambda_selections.extend(factor_lambdas)

        # Progress
        print(f"    λ selections: {dict(pd.Series(factor_lambdas).value_counts())}")

    full_results = pd.concat(all_results, ignore_index=True)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nλ Selection Distribution (n={len(lambda_selections)}):")
    lambda_dist = pd.Series(lambda_selections).value_counts().sort_index()
    for lam, count in lambda_dist.items():
        pct = count / len(lambda_selections) * 100
        print(f"  λ={lam:.2f}: {count} ({pct:.1f}%)")

    print(f"\nMost common λ: {pd.Series(lambda_selections).mode()[0]}")
    print(f"Mean λ: {np.mean(lambda_selections):.3f}")

    # Coverage comparison
    print("\n" + "=" * 70)
    print("COVERAGE COMPARISON: CV-Selected CW-ACI vs ACI")
    print("=" * 70)

    cwaci_cov = full_results['cwaci_coverage'].mean()
    cwaci_var = full_results['cwaci_variance'].mean()
    aci_cov = full_results['aci_coverage'].mean()
    aci_var = full_results['aci_variance'].mean()

    print(f"""
    Method              Coverage    Variance
    ─────────────────────────────────────────
    ACI                 {aci_cov:.1%}      {aci_var:.4f}
    CW-ACI (CV λ)       {cwaci_cov:.1%}      {cwaci_var:.4f}

    Variance reduction: {(aci_var - cwaci_var) / aci_var * 100:.1f}%
    """)

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    full_results.to_csv(output_dir / 'lambda_selection_efficient.csv', index=False)
    print(f"Results saved to: {output_dir / 'lambda_selection_efficient.csv'}")

    return full_results


if __name__ == '__main__':
    results = main()
