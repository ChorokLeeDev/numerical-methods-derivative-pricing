"""
Experiment 02: Conditional Coverage Analysis

Key hypothesis: Crowding-aware methods achieve better coverage
specifically in HIGH CROWDING regimes where standard methods fail.

This is the core ICML contribution:
- Standard CP: Coverage varies with crowding level
- Crowding-Aware CP: Coverage stable across crowding levels

For ICML 2026 submission.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add paths
ICML_SRC = Path(__file__).parent.parent / 'src'
FC_SRC = Path(__file__).parent.parent.parent / 'factor_crowding' / 'src'

# Import modules directly with spec loader
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
CrowdingAdaptiveOnline = crowding_aware_module.CrowdingAdaptiveOnline

baselines_module = load_module('baselines', ICML_SRC / 'baselines.py')
SplitConformalCP = baselines_module.SplitConformalCP
ACIConformal = baselines_module.ACIConformal

sys.path.insert(0, str(FC_SRC))
from features import FeatureEngineer


def analyze_conditional_coverage(
    X: np.ndarray,
    y: np.ndarray,
    crowding: np.ndarray,
    fit_size: int = 90,
    calib_size: int = 30,
    test_size: int = 12,
    alpha: float = 0.1,
    n_crowding_bins: int = 3,
    lambda_values: list = [0.5, 1.0, 2.0, 5.0],
    beta_values: list = [0.25, 0.5, 1.0, 2.0]
) -> pd.DataFrame:
    """
    Analyze coverage by crowding level for different methods and hyperparameters.
    """
    results = []

    # Define crowding bins
    bin_boundaries = np.quantile(crowding, np.linspace(0, 1, n_crowding_bins + 1))
    bin_labels = ['low', 'medium', 'high'][:n_crowding_bins]

    def get_bin(c):
        for i in range(n_crowding_bins - 1):
            if c < bin_boundaries[i + 1]:
                return bin_labels[i]
        return bin_labels[-1]

    # Walk-forward splits
    min_train = fit_size + calib_size
    period_results = []

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

        # Get crowding bins for test samples
        test_bins = [get_bin(c) for c in crowding_test]

        # 1. Split Conformal (baseline)
        split_cp = SplitConformalCP()
        split_cp.fit(X_fit, y_fit, X_calib, y_calib)
        split_sets, _ = split_cp.predict_sets(X_test, alpha)

        for i, (y_true, pred_set, c_bin) in enumerate(zip(y_test, split_sets, test_bins)):
            period_results.append({
                'method': 'split',
                'lambda': np.nan,
                'beta': np.nan,
                'crowding_bin': c_bin,
                'covered': int(y_true in pred_set),
                'set_size': len(pred_set),
                'crowding': crowding_test[i]
            })

        # 2. ACI (baseline)
        aci = ACIConformal(alpha=alpha, gamma=0.1)
        aci.fit(X_fit, y_fit, X_calib, y_calib)
        aci_sets, _ = aci.predict_sets_online(X_test, y_test)

        for i, (y_true, pred_set, c_bin) in enumerate(zip(y_test, aci_sets, test_bins)):
            period_results.append({
                'method': 'aci',
                'lambda': np.nan,
                'beta': np.nan,
                'crowding_bin': c_bin,
                'covered': int(y_true in pred_set),
                'set_size': len(pred_set),
                'crowding': crowding_test[i]
            })

        # 3. CrowdingWeightedCP with different lambda values
        for lam in lambda_values:
            cwcp = CrowdingWeightedCP(lambda_weight=lam)
            cwcp.fit(X_fit, y_fit, X_calib, y_calib, crowding_calib)
            cwcp_result = cwcp.predict_sets(X_test, crowding_test, alpha)

            for i, (y_true, pred_set, c_bin) in enumerate(zip(y_test, cwcp_result.prediction_sets, test_bins)):
                period_results.append({
                    'method': 'crowding_weighted',
                    'lambda': lam,
                    'beta': np.nan,
                    'crowding_bin': c_bin,
                    'covered': int(y_true in pred_set),
                    'set_size': len(pred_set),
                    'crowding': crowding_test[i]
                })

        # 4. CAO with different beta values
        for beta in beta_values:
            cao = CrowdingAdaptiveOnline(alpha=alpha, gamma_base=0.1, beta=beta)
            cao.fit(X_fit, y_fit, X_calib, y_calib, crowding_calib)
            cao_sets, _ = cao.predict_sets_online(X_test, y_test, crowding_test)

            for i, (y_true, pred_set, c_bin) in enumerate(zip(y_test, cao_sets, test_bins)):
                period_results.append({
                    'method': 'cao',
                    'lambda': np.nan,
                    'beta': beta,
                    'crowding_bin': c_bin,
                    'covered': int(y_true in pred_set),
                    'set_size': len(pred_set),
                    'crowding': crowding_test[i]
                })

        start_idx += test_size

    return pd.DataFrame(period_results)


def main():
    print("=" * 70)
    print("CONDITIONAL COVERAGE ANALYSIS BY CROWDING LEVEL")
    print("=" * 70)

    # Load data
    DATA_DIR = Path(__file__).parent.parent / 'data'
    factors = pd.read_parquet(DATA_DIR / 'ff_extended_factors.parquet')
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    factors = factors.drop(columns=['RF'], errors='ignore')

    print(f"\nData: {factors.index.min().strftime('%Y-%m')} to {factors.index.max().strftime('%Y-%m')}")

    # Compute crowding signals
    cs = CrowdingSignal()
    crowding_df = cs.compute_all_crowding(factors, normalize=True)

    # Generate features
    fe = FeatureEngineer()
    features = fe.generate_all_features(factors)

    # Aggregate results across factors
    all_results = []

    for factor in factors.columns:
        print(f"\nProcessing {factor}...")

        X, y = fe.create_ml_dataset(features, factors, target_type='crash', factor=factor)
        crowding = crowding_df[factor].loc[X.index]

        results = analyze_conditional_coverage(
            X.values, y.values, crowding.values,
            alpha=0.1
        )
        results['factor'] = factor
        all_results.append(results)

    full_results = pd.concat(all_results, ignore_index=True)

    # Analyze conditional coverage
    print("\n" + "=" * 70)
    print("CONDITIONAL COVERAGE BY CROWDING LEVEL")
    print("=" * 70)

    # Baselines by crowding level
    print("\n### BASELINES ###")
    for method in ['split', 'aci']:
        method_data = full_results[full_results['method'] == method]
        coverage_by_bin = method_data.groupby('crowding_bin')['covered'].mean()
        size_by_bin = method_data.groupby('crowding_bin')['set_size'].mean()

        print(f"\n{method.upper()}:")
        for bin_name in ['low', 'medium', 'high']:
            if bin_name in coverage_by_bin.index:
                cov = coverage_by_bin[bin_name]
                size = size_by_bin[bin_name]
                gap = cov - 0.9
                status = "✓" if cov >= 0.9 else "✗"
                print(f"  {bin_name:<8} coverage: {cov:.3f} ({gap:+.3f}) size: {size:.2f} {status}")

    # CrowdingWeightedCP by lambda and crowding level
    print("\n### CROWDING-WEIGHTED CP (by λ) ###")
    cwcp_data = full_results[full_results['method'] == 'crowding_weighted']

    for lam in sorted(cwcp_data['lambda'].unique()):
        lam_data = cwcp_data[cwcp_data['lambda'] == lam]
        coverage_by_bin = lam_data.groupby('crowding_bin')['covered'].mean()

        print(f"\nλ = {lam}:")
        total_cov = lam_data['covered'].mean()
        print(f"  Overall: {total_cov:.3f} (target: 0.90)")
        for bin_name in ['low', 'medium', 'high']:
            if bin_name in coverage_by_bin.index:
                cov = coverage_by_bin[bin_name]
                gap = cov - 0.9
                status = "✓" if cov >= 0.9 else "✗"
                print(f"  {bin_name:<8} {cov:.3f} ({gap:+.3f}) {status}")

    # CAO by beta and crowding level
    print("\n### CAO (by β) ###")
    cao_data = full_results[full_results['method'] == 'cao']

    for beta in sorted(cao_data['beta'].unique()):
        beta_data = cao_data[cao_data['beta'] == beta]
        coverage_by_bin = beta_data.groupby('crowding_bin')['covered'].mean()

        print(f"\nβ = {beta}:")
        total_cov = beta_data['covered'].mean()
        print(f"  Overall: {total_cov:.3f} (target: 0.90)")
        for bin_name in ['low', 'medium', 'high']:
            if bin_name in coverage_by_bin.index:
                cov = coverage_by_bin[bin_name]
                gap = cov - 0.9
                status = "✓" if cov >= 0.9 else "✗"
                print(f"  {bin_name:<8} {cov:.3f} ({gap:+.3f}) {status}")

    # Coverage variation analysis
    print("\n" + "=" * 70)
    print("COVERAGE VARIATION (std across crowding bins)")
    print("=" * 70)

    print("\nGoal: Crowding-aware methods should have LOWER variation (more stable coverage)")

    # Split baseline
    split_data = full_results[full_results['method'] == 'split']
    split_var = split_data.groupby('crowding_bin')['covered'].mean().std()
    print(f"\nSplit (baseline): coverage std = {split_var:.4f}")

    # ACI baseline
    aci_data = full_results[full_results['method'] == 'aci']
    aci_var = aci_data.groupby('crowding_bin')['covered'].mean().std()
    print(f"ACI (baseline):   coverage std = {aci_var:.4f}")

    # Best CrowdingWeightedCP
    best_cwcp_var = float('inf')
    best_cwcp_lambda = None
    for lam in cwcp_data['lambda'].unique():
        lam_data = cwcp_data[cwcp_data['lambda'] == lam]
        var = lam_data.groupby('crowding_bin')['covered'].mean().std()
        if var < best_cwcp_var:
            best_cwcp_var = var
            best_cwcp_lambda = lam
    print(f"CrowdingWeightedCP (λ={best_cwcp_lambda}): coverage std = {best_cwcp_var:.4f}")

    # Best CAO
    best_cao_var = float('inf')
    best_cao_beta = None
    for beta in cao_data['beta'].unique():
        beta_data = cao_data[cao_data['beta'] == beta]
        var = beta_data.groupby('crowding_bin')['covered'].mean().std()
        if var < best_cao_var:
            best_cao_var = var
            best_cao_beta = beta
    print(f"CAO (β={best_cao_beta}): coverage std = {best_cao_var:.4f}")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS FOR ICML PAPER")
    print("=" * 70)

    # High crowding analysis (where we expect standard to fail)
    print("\n### HIGH CROWDING REGIME ANALYSIS ###")
    high_split = split_data[split_data['crowding_bin'] == 'high']['covered'].mean()
    high_aci = aci_data[aci_data['crowding_bin'] == 'high']['covered'].mean()

    best_cwcp_high = 0
    for lam in cwcp_data['lambda'].unique():
        lam_high = cwcp_data[(cwcp_data['lambda'] == lam) & (cwcp_data['crowding_bin'] == 'high')]['covered'].mean()
        if lam_high > best_cwcp_high:
            best_cwcp_high = lam_high
            best_lam_for_high = lam

    best_cao_high = 0
    for beta in cao_data['beta'].unique():
        beta_high = cao_data[(cao_data['beta'] == beta) & (cao_data['crowding_bin'] == 'high')]['covered'].mean()
        if beta_high > best_cao_high:
            best_cao_high = beta_high
            best_beta_for_high = beta

    print(f"""
In HIGH CROWDING regime (where distribution shift is most likely):
- Split (baseline): {high_split:.1%} coverage
- ACI (baseline):   {high_aci:.1%} coverage
- CrowdingWeightedCP (λ={best_lam_for_high}): {best_cwcp_high:.1%} coverage ({best_cwcp_high - high_split:+.1%} vs Split)
- CAO (β={best_beta_for_high}): {best_cao_high:.1%} coverage ({best_cao_high - high_aci:+.1%} vs ACI)
""")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    full_results.to_csv(output_dir / 'conditional_coverage.csv', index=False)
    print(f"Results saved to: {output_dir / 'conditional_coverage.csv'}")

    return full_results


if __name__ == '__main__':
    results = main()
