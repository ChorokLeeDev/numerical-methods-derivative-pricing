"""
Baseline Comparison Experiment for CW-ACI Paper
================================================
Compares CW-ACI against:
1. Standard Conformal Prediction (SCP)
2. GARCH-based prediction intervals
3. Bootstrap prediction intervals
4. Conformalized Quantile Regression (CQR)
5. Gibbs-Candes ACI (already in 05_calibrated_analysis.py)

This addresses the reviewer question: "Why not just use X instead?"

Author: Chorok Lee (KAIST)
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from conformal import StandardConformalPredictor, CrowdingWeightedACI
from crowding import compute_crowding_proxy


# ============================================================================
# Load Fama-French Data
# ============================================================================

def load_ff_data():
    """Load Fama-French factor data from local CSV"""
    data_path = Path(__file__).parent.parent / 'data' / 'ff_factors.csv'
    try:
        ff = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"Loaded FF data from {data_path}")
        return ff
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


# ============================================================================
# Baseline Methods
# ============================================================================

class GARCHInterval:
    """GARCH(1,1) based prediction intervals"""

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit_predict(self, returns, cal_fraction=0.5):
        """Fit GARCH and compute prediction intervals"""
        n = len(returns)
        cal_end = int(n * cal_fraction)

        coverages = []
        interval_widths = []

        # Use calibration set to estimate GARCH parameters
        for t in range(cal_end, n):
            calib = returns.iloc[:t].values

            # Simple GARCH(1,1) estimation with method of moments
            r2 = calib ** 2
            omega = np.var(calib) * 0.05  # unconditional variance portion
            alpha_g = 0.1
            beta_g = 0.85

            # Compute conditional variance recursively on recent data
            sigma2 = np.var(calib[-60:]) if len(calib) > 60 else np.var(calib)
            for r in calib[-min(20, len(calib)):]:
                sigma2 = omega + alpha_g * r**2 + beta_g * sigma2

            sigma = np.sqrt(sigma2)

            # Prediction interval (assuming t-distribution for heavy tails)
            t_df = 5  # degrees of freedom
            t_crit = stats.t.ppf(1 - self.alpha/2, t_df)
            pred_mean = np.mean(calib[-12:])  # Recent mean
            lower = pred_mean - t_crit * sigma
            upper = pred_mean + t_crit * sigma

            # Check coverage
            test_val = returns.iloc[t]
            covered = (lower <= test_val <= upper)
            coverages.append(covered)
            interval_widths.append(upper - lower)

        return np.array(coverages), np.array(interval_widths)


class BootstrapInterval:
    """Bootstrap prediction intervals using block bootstrap"""

    def __init__(self, alpha=0.1, n_bootstrap=500, block_size=12):
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.block_size = block_size

    def fit_predict(self, returns, cal_fraction=0.5):
        """Compute bootstrap prediction intervals"""
        n = len(returns)
        cal_end = int(n * cal_fraction)

        coverages = []
        interval_widths = []

        for t in range(cal_end, n):
            calib = returns.iloc[:t].values

            # Block bootstrap for time series
            n_blocks = max(1, len(calib) // self.block_size)
            boot_samples = []

            for _ in range(self.n_bootstrap):
                # Sample blocks with replacement
                block_starts = np.random.randint(0, len(calib) - self.block_size + 1, n_blocks)
                sample = np.concatenate([calib[s:s+self.block_size] for s in block_starts])
                boot_samples.append(sample[-1] if len(sample) > 0 else 0)

            # Prediction interval from empirical quantiles
            lower = np.percentile(boot_samples, self.alpha/2 * 100)
            upper = np.percentile(boot_samples, (1 - self.alpha/2) * 100)

            # Check coverage
            test_val = returns.iloc[t]
            covered = (lower <= test_val <= upper)
            coverages.append(covered)
            interval_widths.append(upper - lower)

        return np.array(coverages), np.array(interval_widths)


class ConformizedQuantileRegression:
    """Conformalized Quantile Regression (CQR) - Romano et al. 2019"""

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit_predict(self, returns, cal_fraction=0.5):
        """CQR prediction intervals"""
        n = len(returns)
        cal_end = int(n * cal_fraction)

        coverages = []
        interval_widths = []

        for t in range(cal_end, n):
            calib = returns.iloc[:t].values

            # Estimate quantiles (using historical data as proxy for quantile regression)
            q_low = np.quantile(calib, self.alpha/2)
            q_high = np.quantile(calib, 1 - self.alpha/2)

            # Conformalize: compute nonconformity scores
            # E_i = max(q_low - Y_i, Y_i - q_high)
            scores = np.maximum(q_low - calib, calib - q_high)

            # Quantile of scores for correction
            n_cal = len(scores)
            q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
            q_level = min(q_level, 1.0)
            correction = np.quantile(scores, q_level)

            lower = q_low - correction
            upper = q_high + correction

            # Check coverage
            test_val = returns.iloc[t]
            covered = (lower <= test_val <= upper)
            coverages.append(covered)
            interval_widths.append(upper - lower)

        return np.array(coverages), np.array(interval_widths)


# ============================================================================
# Conformal Methods (using existing implementation)
# ============================================================================

def run_standard_cp(returns, crowding, alpha=0.1, cal_fraction=0.5):
    """Run standard conformal prediction"""
    n = len(returns)
    cal_end = int(n * cal_fraction)

    y_cal = returns.iloc[:cal_end].values
    y_test = returns.iloc[cal_end:].values
    crowd_test = crowding.iloc[cal_end:].values

    # Simple mean prediction
    pred_cal = np.full_like(y_cal, np.mean(y_cal))
    pred_test = np.full_like(y_test, np.mean(y_cal))

    scp = StandardConformalPredictor(alpha=alpha)
    scp.fit(y_cal, pred_cal)
    lower, upper = scp.predict(pred_test)

    coverages = (y_test >= lower) & (y_test <= upper)
    widths = upper - lower

    return coverages, widths


def run_cwaci(returns, crowding, alpha=0.1, cal_fraction=0.5, gamma=1.0):
    """Run CW-ACI"""
    n = len(returns)
    cal_end = int(n * cal_fraction)

    y_cal = returns.iloc[:cal_end].values
    y_test = returns.iloc[cal_end:].values
    crowd_cal = crowding.iloc[:cal_end].values
    crowd_test = crowding.iloc[cal_end:].values

    # Simple mean prediction
    pred_cal = np.full_like(y_cal, np.mean(y_cal))
    pred_test = np.full_like(y_test, np.mean(y_cal))

    cwaci = CrowdingWeightedACI(alpha=alpha, sensitivity=gamma)
    cwaci.fit(y_cal, pred_cal, crowd_cal)
    lower, upper, widths = cwaci.predict(pred_test, crowd_test)

    coverages = (y_test >= lower) & (y_test <= upper)

    return coverages, widths


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_by_crowding(coverages, crowding, cal_fraction=0.5):
    """Analyze coverage by crowding regime"""
    n = len(crowding)
    cal_end = int(n * cal_fraction)
    crowding_aligned = crowding.iloc[cal_end:].values

    low_thresh = np.percentile(crowding_aligned, 33)
    high_thresh = np.percentile(crowding_aligned, 67)

    low_mask = crowding_aligned <= low_thresh
    high_mask = crowding_aligned > high_thresh

    return {
        'overall': np.mean(coverages),
        'low_crowd': np.mean(coverages[low_mask]) if low_mask.sum() > 0 else np.nan,
        'high_crowd': np.mean(coverages[high_mask]) if high_mask.sum() > 0 else np.nan
    }


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    print("="*70)
    print("CW-ACI Baseline Comparison Experiment")
    print("="*70)

    # Load data
    print("\nLoading Fama-French data...")
    ff_data = load_ff_data()
    if ff_data is None:
        print("ERROR: Could not load factor data")
        return

    print(f"Loaded {len(ff_data)} observations, {len(ff_data.columns)} factors")

    # Factors to test
    factors_to_test = ['SMB', 'HML', 'Mom']

    all_results = []

    for factor in factors_to_test:
        print(f"\n{'='*50}")
        print(f"Factor: {factor}")
        print(f"{'='*50}")

        if factor not in ff_data.columns:
            print(f"  Factor {factor} not found")
            continue

        returns = ff_data[factor].dropna()
        crowding = compute_crowding_proxy(returns, window=12)

        # Drop NaN from crowding computation
        valid = crowding.notna()
        returns = returns[valid]
        crowding = crowding[valid]

        print(f"  Valid observations: {len(returns)}")

        # Run all methods
        methods = {}

        # 1. GARCH
        print(f"\n  GARCH...")
        garch = GARCHInterval(alpha=0.1)
        cov, widths = garch.fit_predict(returns)
        methods['GARCH'] = {'coverages': cov, 'widths': widths}

        # 2. Bootstrap
        print(f"  Bootstrap...")
        boot = BootstrapInterval(alpha=0.1)
        cov, widths = boot.fit_predict(returns)
        methods['Bootstrap'] = {'coverages': cov, 'widths': widths}

        # 3. CQR
        print(f"  CQR...")
        cqr = ConformizedQuantileRegression(alpha=0.1)
        cov, widths = cqr.fit_predict(returns)
        methods['CQR'] = {'coverages': cov, 'widths': widths}

        # 4. Standard CP (using proper implementation)
        print(f"  Standard CP...")
        cov, widths = run_standard_cp(returns, crowding)
        methods['Standard CP'] = {'coverages': cov, 'widths': widths}

        # 5. CW-ACI (using proper implementation)
        print(f"  CW-ACI...")
        cov, widths = run_cwaci(returns, crowding, gamma=1.5)
        methods['CW-ACI'] = {'coverages': cov, 'widths': widths}

        # Analyze results
        for method_name, data in methods.items():
            results = analyze_by_crowding(data['coverages'], crowding)

            all_results.append({
                'factor': factor,
                'method': method_name,
                'overall': results['overall'],
                'low_crowd': results['low_crowd'],
                'high_crowd': results['high_crowd'],
                'avg_width': np.mean(data['widths'])
            })

            print(f"\n  {method_name}:")
            print(f"    Overall: {results['overall']:.1%}")
            print(f"    Low-crowd: {results['low_crowd']:.1%}")
            print(f"    High-crowd: {results['high_crowd']:.1%}")
            print(f"    Avg width: {np.mean(data['widths']):.4f}")

    # Summary table
    results_df = pd.DataFrame(all_results)

    print("\n" + "="*70)
    print("SUMMARY: Method Comparison (averaged across factors)")
    print("="*70)

    summary = results_df.groupby('method').agg({
        'overall': 'mean',
        'low_crowd': 'mean',
        'high_crowd': 'mean',
        'avg_width': 'mean'
    }).round(3)

    # Reorder methods logically
    method_order = ['GARCH', 'Bootstrap', 'CQR', 'Standard CP', 'CW-ACI']
    summary = summary.reindex([m for m in method_order if m in summary.index])

    print("\n" + summary.to_string())

    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    cwaci_high = summary.loc['CW-ACI', 'high_crowd'] if 'CW-ACI' in summary.index else 0
    scp_high = summary.loc['Standard CP', 'high_crowd'] if 'Standard CP' in summary.index else 0

    print(f"\n1. High-crowding coverage comparison:")
    for method in method_order:
        if method in summary.index:
            h = summary.loc[method, 'high_crowd']
            print(f"   {method:<12}: {h:.1%}")

    print(f"\n2. CW-ACI improvement over Standard CP (high-crowd): {(cwaci_high - scp_high)*100:+.1f}pp")

    # Trade-off analysis
    cwaci_width = summary.loc['CW-ACI', 'avg_width'] if 'CW-ACI' in summary.index else 0
    scp_width = summary.loc['Standard CP', 'avg_width'] if 'Standard CP' in summary.index else 0
    width_ratio = cwaci_width / scp_width if scp_width > 0 else 1

    print(f"\n3. Interval width trade-off:")
    print(f"   CW-ACI/SCP width ratio: {width_ratio:.2f}x")
    print(f"   (wider intervals for better high-crowding coverage)")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    results_df.to_csv(output_dir / 'baseline_comparison_results.csv', index=False)
    summary.to_csv(output_dir / 'baseline_comparison_summary.csv')
    print(f"\nSaved: {output_dir}/baseline_comparison_*.csv")

    return results_df, summary


if __name__ == "__main__":
    results_df, summary = main()
