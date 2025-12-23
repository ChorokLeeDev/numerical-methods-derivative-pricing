"""
Test CW-ACI: Does Crowding-Weighted Adaptive Conformal Inference Actually Work?

Key questions:
1. Does CW-ACI achieve target coverage (e.g., 90%)?
2. Does it improve over standard conformal prediction?
3. Are prediction intervals appropriately wider during high-crowding periods?

Run: python scripts/test_cwaci.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SimpleConformalPredictor:
    """Standard split conformal prediction baseline."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.calibration_scores = None

    def fit(self, y_cal, y_pred_cal):
        """Compute nonconformity scores on calibration set."""
        self.calibration_scores = np.abs(y_cal - y_pred_cal)
        return self

    def predict(self, y_pred_test):
        """Return prediction intervals."""
        if self.calibration_scores is None:
            raise ValueError("Must fit first")

        # Quantile of calibration scores
        q = np.quantile(self.calibration_scores, 1 - self.alpha)

        lower = y_pred_test - q
        upper = y_pred_test + q

        return lower, upper, q


class CrowdingWeightedACI:
    """
    Crowding-Weighted Adaptive Conformal Inference.

    Key idea: Weight nonconformity scores by crowding level.
    High crowding → higher weight → wider intervals.
    """

    def __init__(self, alpha=0.1, crowding_sensitivity=1.0):
        self.alpha = alpha
        self.crowding_sensitivity = crowding_sensitivity
        self.calibration_scores = None
        self.calibration_weights = None

    def _compute_weights(self, crowding):
        """Compute weights from crowding signal."""
        # Sigmoid weighting: high crowding → high weight
        normalized = (crowding - crowding.mean()) / (crowding.std() + 1e-8)
        weights = 1 / (1 + np.exp(-self.crowding_sensitivity * normalized))
        return weights

    def fit(self, y_cal, y_pred_cal, crowding_cal):
        """Compute weighted nonconformity scores."""
        self.calibration_scores = np.abs(y_cal - y_pred_cal)
        self.calibration_weights = self._compute_weights(crowding_cal)
        self.crowding_mean = crowding_cal.mean()
        self.crowding_std = crowding_cal.std()
        return self

    def predict(self, y_pred_test, crowding_test):
        """Return adaptive prediction intervals."""
        if self.calibration_scores is None:
            raise ValueError("Must fit first")

        # Compute test weights
        test_weights = self._compute_weights(crowding_test)

        # For each test point, compute weighted quantile
        lowers = []
        uppers = []
        widths = []

        for i, (pred, w_test) in enumerate(zip(y_pred_test, test_weights)):
            # Weight calibration scores by similarity to test crowding
            # Higher test crowding → wider intervals
            adjusted_scores = self.calibration_scores * (1 + w_test)

            q = np.quantile(adjusted_scores, 1 - self.alpha)

            lowers.append(pred - q)
            uppers.append(pred + q)
            widths.append(2 * q)

        return np.array(lowers), np.array(uppers), np.array(widths)


def compute_coverage(y_true, lower, upper):
    """Compute empirical coverage."""
    covered = (y_true >= lower) & (y_true <= upper)
    return covered.mean()


def compute_interval_width(lower, upper):
    """Compute average interval width."""
    return (upper - lower).mean()


def run_cwaci_test():
    """Main test of CW-ACI vs standard conformal prediction."""

    print("="*70)
    print("CW-ACI TEST: Does Crowding-Weighted Conformal Prediction Work?")
    print("="*70)

    # Load data
    base_path = Path(__file__).parent.parent
    data_path = base_path / "data" / "factor_crowding" / "ff_factors_monthly.parquet"

    if not data_path.exists():
        print(f"ERROR: Data not found at {data_path}")
        return None

    df = pd.read_parquet(data_path)
    print(f"\nLoaded: {len(df)} months of data")

    # Test on multiple factors
    factors = ['SMB', 'HML', 'Mom', 'CMA', 'RMW']
    available_factors = [f for f in factors if f in df.columns]

    results = []

    for factor in available_factors:
        print(f"\n{'='*60}")
        print(f"TESTING: {factor}")
        print("="*60)

        returns = df[factor].dropna().values

        # Compute crowding proxy (trailing 12-month absolute return)
        crowding = pd.Series(returns).rolling(12).apply(
            lambda x: np.abs(x).mean()
        ).values

        # Simple prediction: rolling mean
        predictions = pd.Series(returns).rolling(12).mean().shift(1).values

        # Align all arrays (drop NaN from rolling)
        valid_idx = ~(np.isnan(crowding) | np.isnan(predictions))
        returns_valid = returns[valid_idx]
        crowding_valid = crowding[valid_idx]
        predictions_valid = predictions[valid_idx]

        n = len(returns_valid)
        print(f"Valid observations: {n}")

        # Split: 50% calibration, 50% test
        cal_end = n // 2
        y_cal = returns_valid[:cal_end]
        y_pred_cal = predictions_valid[:cal_end]
        crowding_cal = crowding_valid[:cal_end]

        y_test = returns_valid[cal_end:]
        y_pred_test = predictions_valid[cal_end:]
        crowding_test = crowding_valid[cal_end:]

        print(f"Calibration: {len(y_cal)}, Test: {len(y_test)}")

        # Target coverage
        alpha = 0.1  # 90% coverage target

        # METHOD 1: Standard Conformal Prediction
        print("\n[1] Standard Conformal Prediction")
        scp = SimpleConformalPredictor(alpha=alpha)
        scp.fit(y_cal, y_pred_cal)
        lower_scp, upper_scp, width_scp = scp.predict(y_pred_test)

        coverage_scp = compute_coverage(y_test, lower_scp, upper_scp)
        avg_width_scp = compute_interval_width(lower_scp, upper_scp)

        print(f"  Coverage: {coverage_scp:.1%} (target: {1-alpha:.0%})")
        print(f"  Avg width: {avg_width_scp:.4f}")

        # METHOD 2: CW-ACI
        print("\n[2] Crowding-Weighted ACI")
        cwaci = CrowdingWeightedACI(alpha=alpha, crowding_sensitivity=1.0)
        cwaci.fit(y_cal, y_pred_cal, crowding_cal)
        lower_cw, upper_cw, widths_cw = cwaci.predict(y_pred_test, crowding_test)

        coverage_cw = compute_coverage(y_test, lower_cw, upper_cw)
        avg_width_cw = widths_cw.mean()

        print(f"  Coverage: {coverage_cw:.1%} (target: {1-alpha:.0%})")
        print(f"  Avg width: {avg_width_cw:.4f}")

        # KEY TEST: Does CW-ACI adapt to crowding?
        print("\n[3] Adaptive Behavior Test")

        # Split test set by crowding level
        crowding_median = np.median(crowding_test)
        high_crowd_idx = crowding_test > crowding_median
        low_crowd_idx = ~high_crowd_idx

        # Standard CP: same width regardless of crowding
        width_scp_high = (upper_scp[high_crowd_idx] - lower_scp[high_crowd_idx]).mean()
        width_scp_low = (upper_scp[low_crowd_idx] - lower_scp[low_crowd_idx]).mean()

        # CW-ACI: should be wider for high crowding
        width_cw_high = widths_cw[high_crowd_idx].mean()
        width_cw_low = widths_cw[low_crowd_idx].mean()

        print(f"  Standard CP width (high crowding): {width_scp_high:.4f}")
        print(f"  Standard CP width (low crowding):  {width_scp_low:.4f}")
        print(f"  CW-ACI width (high crowding): {width_cw_high:.4f}")
        print(f"  CW-ACI width (low crowding):  {width_cw_low:.4f}")

        # Coverage by crowding level
        coverage_scp_high = compute_coverage(y_test[high_crowd_idx],
                                              lower_scp[high_crowd_idx],
                                              upper_scp[high_crowd_idx])
        coverage_scp_low = compute_coverage(y_test[low_crowd_idx],
                                             lower_scp[low_crowd_idx],
                                             upper_scp[low_crowd_idx])

        coverage_cw_high = compute_coverage(y_test[high_crowd_idx],
                                             lower_cw[high_crowd_idx],
                                             upper_cw[high_crowd_idx])
        coverage_cw_low = compute_coverage(y_test[low_crowd_idx],
                                            lower_cw[low_crowd_idx],
                                            upper_cw[low_crowd_idx])

        print(f"\n  Coverage (high crowding): SCP={coverage_scp_high:.1%}, CW-ACI={coverage_cw_high:.1%}")
        print(f"  Coverage (low crowding):  SCP={coverage_scp_low:.1%}, CW-ACI={coverage_cw_low:.1%}")

        # Verdict
        print("\n  VERDICT:")
        adapts = width_cw_high > width_cw_low * 1.05  # At least 5% wider
        improves_coverage = abs(coverage_cw - 0.9) < abs(coverage_scp - 0.9)

        if adapts:
            print(f"  ✓ CW-ACI adapts to crowding (high/low ratio: {width_cw_high/width_cw_low:.2f})")
        else:
            print(f"  ✗ CW-ACI does NOT adapt to crowding")

        if improves_coverage:
            print(f"  ✓ CW-ACI improves coverage ({coverage_cw:.1%} vs {coverage_scp:.1%})")
        else:
            print(f"  ✗ CW-ACI does NOT improve coverage")

        results.append({
            'factor': factor,
            'n_test': len(y_test),
            'coverage_scp': coverage_scp,
            'coverage_cwaci': coverage_cw,
            'width_scp': avg_width_scp,
            'width_cwaci': avg_width_cw,
            'width_ratio_high_low': width_cw_high / width_cw_low if width_cw_low > 0 else np.nan,
            'coverage_scp_high': coverage_scp_high,
            'coverage_scp_low': coverage_scp_low,
            'coverage_cwaci_high': coverage_cw_high,
            'coverage_cwaci_low': coverage_cw_low,
            'adapts': adapts,
            'improves_coverage': improves_coverage
        })

    # Summary
    results_df = pd.DataFrame(results)

    print("\n" + "="*70)
    print("SUMMARY: CW-ACI TEST RESULTS")
    print("="*70)

    print("\n{:<8} {:>12} {:>12} {:>12} {:>10} {:>10}".format(
        "Factor", "SCP Cov", "CW-ACI Cov", "Target", "Adapts?", "Better?"
    ))
    print("-"*70)

    for _, row in results_df.iterrows():
        print("{:<8} {:>12.1%} {:>12.1%} {:>12.0%} {:>10} {:>10}".format(
            row['factor'],
            row['coverage_scp'],
            row['coverage_cwaci'],
            0.90,
            "YES" if row['adapts'] else "NO",
            "YES" if row['improves_coverage'] else "NO"
        ))

    # Overall verdict
    n_adapts = results_df['adapts'].sum()
    n_improves = results_df['improves_coverage'].sum()

    print("\n" + "="*70)
    print("OVERALL VERDICT")
    print("="*70)
    print(f"\nFactors where CW-ACI adapts to crowding: {n_adapts}/{len(results_df)}")
    print(f"Factors where CW-ACI improves coverage: {n_improves}/{len(results_df)}")

    if n_adapts >= len(results_df) // 2:
        print("\n✓ CW-ACI SHOWS ADAPTIVE BEHAVIOR - method has merit")
    else:
        print("\n✗ CW-ACI DOES NOT ADAPT CONSISTENTLY - method may need revision")

    if n_improves >= len(results_df) // 2:
        print("✓ CW-ACI IMPROVES COVERAGE - practical benefit demonstrated")
    else:
        print("✗ CW-ACI DOES NOT IMPROVE COVERAGE - benefit unclear")

    # Save results
    output_path = base_path / "results" / "cwaci_test_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return results_df


if __name__ == '__main__':
    run_cwaci_test()
