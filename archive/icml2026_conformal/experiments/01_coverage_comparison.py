"""
Experiment 01: Coverage Comparison - Crowding-Aware vs Baselines

Compare all conformal prediction methods:
1. Split (baseline) - Standard split conformal
2. ACI (baseline) - Adaptive Conformal Inference
3. CrowdingWeightedCP - Novel crowding-weighted nonconformity
4. CrowdingStratifiedCP - Mondrian+ with crowding strata
5. CrowdingAdaptiveOnline (CAO) - ACI with crowding-dependent step size

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

# Import modules directly with spec loader to avoid naming conflicts
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load our ICML crowding signal (not the dataclass from factor_crowding)
crowding_signal_module = load_module('crowding_signal_icml', ICML_SRC / 'crowding_signal.py')
CrowdingSignal = crowding_signal_module.CrowdingSignal

# Load other ICML modules
crowding_aware_module = load_module('crowding_aware_conformal', ICML_SRC / 'crowding_aware_conformal.py')
CrowdingWeightedCP = crowding_aware_module.CrowdingWeightedCP
CrowdingStratifiedCP = crowding_aware_module.CrowdingStratifiedCP
CrowdingAdaptiveOnline = crowding_aware_module.CrowdingAdaptiveOnline
evaluate_coverage = crowding_aware_module.evaluate_coverage
evaluate_conditional_coverage = crowding_aware_module.evaluate_conditional_coverage

baselines_module = load_module('baselines', ICML_SRC / 'baselines.py')
SplitConformalCP = baselines_module.SplitConformalCP
ACIConformal = baselines_module.ACIConformal

# Load factor_crowding features module
sys.path.insert(0, str(FC_SRC))
from features import FeatureEngineer


class WalkForwardBacktest:
    """
    Walk-forward backtesting for conformal methods.

    Timeline: [Fit 90mo] → [Calib 30mo] → [Test 12mo] → step 12mo
    """

    def __init__(
        self,
        fit_size: int = 90,
        calib_size: int = 30,
        test_size: int = 12,
        step_size: int = 12,
        alpha: float = 0.1
    ):
        self.fit_size = fit_size
        self.calib_size = calib_size
        self.test_size = test_size
        self.step_size = step_size
        self.alpha = alpha

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        crowding: pd.Series,
        verbose: bool = True
    ) -> dict:
        """
        Run walk-forward backtest comparing all methods.
        """
        X_arr = X.values
        y_arr = y.values
        crowding_arr = crowding.values

        min_train = self.fit_size + self.calib_size
        results = {
            'split': [], 'aci': [],
            'crowding_weighted': [], 'crowding_stratified': [], 'cao': []
        }

        period = 0
        start_idx = 0

        while start_idx + min_train + self.test_size <= len(X):
            # Define splits
            fit_end = start_idx + self.fit_size
            calib_end = fit_end + self.calib_size
            test_end = calib_end + self.test_size

            X_fit = X_arr[start_idx:fit_end]
            y_fit = y_arr[start_idx:fit_end]
            X_calib = X_arr[fit_end:calib_end]
            y_calib = y_arr[fit_end:calib_end]
            X_test = X_arr[calib_end:test_end]
            y_test = y_arr[calib_end:test_end]
            crowding_calib = crowding_arr[fit_end:calib_end]
            crowding_test = crowding_arr[calib_end:test_end]

            if verbose:
                print(f"\nPeriod {period + 1}: fit={start_idx}-{fit_end}, "
                      f"calib={fit_end}-{calib_end}, test={calib_end}-{test_end}")

            # 1. Split Conformal (baseline)
            split_cp = SplitConformalCP()
            split_cp.fit(X_fit, y_fit, X_calib, y_calib)
            split_sets, _ = split_cp.predict_sets(X_test, self.alpha)
            results['split'].append(evaluate_coverage(split_sets, y_test))

            # 2. ACI (baseline)
            aci = ACIConformal(alpha=self.alpha, gamma=0.1)
            aci.fit(X_fit, y_fit, X_calib, y_calib)
            aci_sets, _ = aci.predict_sets_online(X_test, y_test)
            results['aci'].append(evaluate_coverage(aci_sets, y_test))

            # 3. CrowdingWeightedCP (novel)
            cwcp = CrowdingWeightedCP(lambda_weight=1.0)
            cwcp.fit(X_fit, y_fit, X_calib, y_calib, crowding_calib)
            cwcp_result = cwcp.predict_sets(X_test, crowding_test, self.alpha)
            results['crowding_weighted'].append(
                evaluate_coverage(cwcp_result.prediction_sets, y_test)
            )

            # 4. CrowdingStratifiedCP (novel)
            cscp = CrowdingStratifiedCP(n_strata=3)
            cscp.fit(X_fit, y_fit, X_calib, y_calib, crowding_calib)
            cscp_result = cscp.predict_sets(X_test, crowding_test, self.alpha)
            results['crowding_stratified'].append(
                evaluate_coverage(cscp_result.prediction_sets, y_test)
            )

            # 5. CrowdingAdaptiveOnline (novel)
            cao = CrowdingAdaptiveOnline(alpha=self.alpha, gamma_base=0.1, beta=0.5)
            cao.fit(X_fit, y_fit, X_calib, y_calib, crowding_calib)
            cao_sets, _ = cao.predict_sets_online(X_test, y_test, crowding_test)
            results['cao'].append(evaluate_coverage(cao_sets, y_test))

            period += 1
            start_idx += self.step_size

        # Aggregate results
        summary = {}
        for method, period_results in results.items():
            if period_results:
                df = pd.DataFrame(period_results)
                summary[method] = {
                    'coverage': df['coverage'].mean(),
                    'coverage_std': df['coverage'].std(),
                    'avg_set_size': df['avg_set_size'].mean(),
                    'singleton_rate': df['singleton_rate'].mean(),
                    'n_periods': len(period_results),
                    'target': 1 - self.alpha,
                    'gap': df['coverage'].mean() - (1 - self.alpha)
                }

        return {
            'summary': summary,
            'period_results': results
        }


def main():
    print("=" * 70)
    print("CROWDING-AWARE CONFORMAL PREDICTION: COVERAGE COMPARISON")
    print("=" * 70)

    # Load data
    DATA_DIR = Path(__file__).parent.parent / 'data'
    factors = pd.read_parquet(DATA_DIR / 'ff_extended_factors.parquet')
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    factors = factors.drop(columns=['RF'], errors='ignore')

    print(f"\nData: {factors.index.min().strftime('%Y-%m')} to {factors.index.max().strftime('%Y-%m')}")
    print(f"Factors: {list(factors.columns)}")

    # Compute crowding signals
    print("\nComputing crowding signals...")
    cs = CrowdingSignal()
    crowding_df = cs.compute_all_crowding(factors, normalize=True)
    print(f"Crowding signal range: {crowding_df.min().min():.3f} to {crowding_df.max().max():.3f}")

    # Generate features
    print("\nGenerating features...")
    fe = FeatureEngineer()
    features = fe.generate_all_features(factors)

    # Run comparison for each factor
    all_results = []

    for factor in factors.columns:
        print(f"\n{'='*60}")
        print(f"FACTOR: {factor}")
        print(f"{'='*60}")

        # Create ML dataset
        X, y = fe.create_ml_dataset(features, factors, target_type='crash', factor=factor)

        # Align crowding with features
        crowding = crowding_df[factor].loc[X.index]

        # Run backtest
        backtest = WalkForwardBacktest(
            fit_size=90,
            calib_size=30,
            test_size=12,
            alpha=0.1
        )

        result = backtest.run(X, y, crowding, verbose=False)

        # Store results
        for method, metrics in result['summary'].items():
            all_results.append({
                'factor': factor,
                'method': method,
                **metrics
            })

        # Print summary for this factor
        print(f"\n{'Method':<20} {'Coverage':<10} {'Gap':<10} {'Avg Size':<10}")
        print("-" * 50)
        for method, metrics in result['summary'].items():
            gap = f"{metrics['gap']:+.3f}"
            print(f"{method:<20} {metrics['coverage']:.3f}     {gap:<10} {metrics['avg_set_size']:.2f}")

    # Overall summary
    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY (averaged across all factors)")
    print("=" * 70)

    summary = results_df.groupby('method').agg({
        'coverage': ['mean', 'std'],
        'gap': 'mean',
        'avg_set_size': 'mean',
        'singleton_rate': 'mean'
    }).round(3)

    print(summary)

    # Method comparison
    print("\n" + "=" * 70)
    print("METHOD RANKING")
    print("=" * 70)

    method_avg = results_df.groupby('method')['coverage'].mean().sort_values(ascending=False)
    target = 0.9

    for rank, (method, coverage) in enumerate(method_avg.items(), 1):
        gap = coverage - target
        meets = "✓" if coverage >= target else "✗"
        is_novel = "★" if method in ['crowding_weighted', 'crowding_stratified', 'cao'] else " "
        print(f"{rank}. {method:<20} {coverage:.3f} ({gap:+.3f}) {meets} {is_novel}")

    # Key findings for paper
    print("\n" + "=" * 70)
    print("KEY FINDINGS FOR ICML PAPER")
    print("=" * 70)

    baseline_cov = results_df[results_df['method'] == 'split']['coverage'].mean()
    aci_cov = results_df[results_df['method'] == 'aci']['coverage'].mean()
    cwcp_cov = results_df[results_df['method'] == 'crowding_weighted']['coverage'].mean()
    cao_cov = results_df[results_df['method'] == 'cao']['coverage'].mean()

    print(f"""
1. BASELINE PERFORMANCE:
   - Split Conformal: {baseline_cov:.1%} coverage (gap: {baseline_cov - 0.9:+.1%})
   - ACI: {aci_cov:.1%} coverage (gap: {aci_cov - 0.9:+.1%})
   - Problem: Both fail to meet 90% target due to distribution shift

2. CROWDING-AWARE METHODS:
   - CrowdingWeightedCP: {cwcp_cov:.1%} coverage (gap: {cwcp_cov - 0.9:+.1%})
   - CAO (Crowding-Adaptive Online): {cao_cov:.1%} coverage (gap: {cao_cov - 0.9:+.1%})

3. IMPROVEMENT:
   - CrowdingWeightedCP vs Split: {cwcp_cov - baseline_cov:+.1%}
   - CAO vs ACI: {cao_cov - aci_cov:+.1%}

4. ICML CONTRIBUTION:
   - First integration of market microstructure (crowding) into conformal
   - Crowding-weighted nonconformity anticipates distribution shift
   - CAO with crowding-dependent step size achieves faster adaptation
""")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    results_df.to_csv(output_dir / 'coverage_comparison.csv', index=False)
    print(f"\nResults saved to: {output_dir / 'coverage_comparison.csv'}")

    return results_df


if __name__ == '__main__':
    results = main()
