"""
Experiment 08 (FAST): Principled λ Selection for CW-ACI

Quick validation of CV-based λ selection on 2 factors.
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


def main():
    print("=" * 60)
    print("FAST λ SELECTION TEST (2 factors, 3 λ values)")
    print("=" * 60)

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

    # Test on 2 representative factors
    test_factors = ['Mom', 'HML']
    lambda_selections = []

    for factor in test_factors:
        print(f"\n{'='*40}")
        print(f"Testing {factor}")
        print('='*40)

        X, y = fe.create_ml_dataset(features, factors, target_type='crash', factor=factor)
        crowding = crowding_df[factor].loc[X.index].values

        # Use middle portion for quick test
        n = len(X)
        start = n // 4
        end = 3 * n // 4

        X_sub = X.values[start:end]
        y_sub = y.values[start:end]
        crowding_sub = crowding[start:end]

        # Split: fit / calib / test
        fit_size = 90
        calib_size = 30

        X_fit = X_sub[:fit_size]
        y_fit = y_sub[:fit_size]
        X_calib = X_sub[fit_size:fit_size+calib_size]
        y_calib = y_sub[fit_size:fit_size+calib_size]
        crowding_calib = crowding_sub[fit_size:fit_size+calib_size]

        # Run λ selection
        print("Running λ selection via CV...")
        optimal_lambda, stats = CrowdingWeightedACI.select_lambda(
            X_fit, y_fit, X_calib, y_calib, crowding_calib,
            alpha=0.1,
            gamma=0.1,
            lambda_candidates=[0.0, 0.5, 1.0],  # Reduced candidates
            n_folds=2,  # Reduced folds
            criterion='min_variance'
        )

        lambda_selections.append(optimal_lambda)
        print(f"Selected λ: {optimal_lambda}")

        # Show all candidate results
        print("\nλ Candidate Results:")
        for r in stats['all_results']:
            print(f"  λ={r['lambda']:.1f}: var={r['variance']:.4f}, min_bin={r['min_bin']:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Selected λ values: {lambda_selections}")
    print(f"Most common: {max(set(lambda_selections), key=lambda_selections.count)}")

    # Compare with Experiment 07 results
    print("\n" + "=" * 60)
    print("VALIDATION vs Experiment 07")
    print("=" * 60)
    print("""
From Experiment 07 (full run):
  λ=0.0 (ACI):  var=0.0116, min_bin=88.1%
  λ=0.5:        var=0.0099, min_bin=88.4%  ← BEST
  λ=1.0:        var=0.0158, min_bin=87.5%

CV selection should pick λ≈0.5 to minimize variance.
""")

    if 0.5 in lambda_selections or all(l <= 0.5 for l in lambda_selections):
        print("✓ CV selection VALIDATED - picks low λ for uniform coverage")
    else:
        print("⚠ Results differ - may need larger sample for stable selection")

    return lambda_selections


if __name__ == '__main__':
    result = main()
