"""
Signal-Adaptive Conformal Inference (SA-CI)

Fixed implementation that properly weights calibration scores.

Key improvements over original CW-ACI:
1. Weights calibration scores by THEIR OWN characteristics (not test point)
2. Uses locally-weighted quantile estimation
3. Includes γ calibration to achieve target coverage
4. Adds proper statistical reporting

Author: Chorok Lee (KAIST)
Date: December 2024
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from scipy import stats


@dataclass
class ConformalResult:
    """Container for conformal prediction results with statistical measures."""
    lower: np.ndarray
    upper: np.ndarray
    width: np.ndarray
    coverage_overall: float
    coverage_high_vol: float
    coverage_low_vol: float
    se_overall: float
    se_high_vol: float
    se_low_vol: float
    n_total: int
    n_high_vol: int
    n_low_vol: int


class StandardConformalPredictor:
    """
    Standard split conformal prediction.

    Reference: Vovk et al. (2005), Lei et al. (2018)
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.calibration_scores = None
        self.quantile = None

    def fit(self, y_cal: np.ndarray, y_pred_cal: np.ndarray) -> 'StandardConformalPredictor':
        """Calibrate on held-out data."""
        self.calibration_scores = np.abs(y_cal - y_pred_cal)

        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)

        self.quantile = np.quantile(self.calibration_scores, q_level)
        return self

    def predict(self, y_pred_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute prediction intervals."""
        if self.quantile is None:
            raise ValueError("Must call fit() before predict()")

        lower = y_pred_test - self.quantile
        upper = y_pred_test + self.quantile
        return lower, upper

    def get_width(self) -> float:
        """Return interval width."""
        return 2 * self.quantile


class SignalAdaptiveCI:
    """
    Signal-Adaptive Conformal Inference (SA-CI).

    Key innovation: Properly weight calibration scores by their own
    signal values, using locally-weighted quantile estimation.

    Algorithm:
    1. Compute nonconformity scores on calibration set
    2. For each test point with signal s:
       - Weight calibration points by kernel similarity: w_i = K((s_i - s) / h)
       - Compute weighted quantile of scores
       - Return adaptive interval

    This is fundamentally different from the flawed approach of
    multiplying all scores by the test point's weight.
    """

    def __init__(self, alpha: float = 0.1, bandwidth: float = 1.0,
                 kernel: str = 'gaussian'):
        """
        Initialize SA-CI.

        Parameters
        ----------
        alpha : float
            Miscoverage rate. Default 0.1 for 90% coverage.
        bandwidth : float
            Kernel bandwidth for local weighting. Higher = smoother.
        kernel : str
            Kernel type: 'gaussian', 'epanechnikov', or 'uniform'
        """
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.calibration_scores = None
        self.signal_cal = None
        self.signal_mean = None
        self.signal_std = None

    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal to zero mean, unit variance."""
        return (signal - self.signal_mean) / (self.signal_std + 1e-8)

    def _compute_kernel_weights(self, signal_cal_norm: np.ndarray,
                                 signal_test_norm: float) -> np.ndarray:
        """
        Compute kernel weights for calibration points.

        Higher weight for calibration points with similar signal values.
        """
        distances = (signal_cal_norm - signal_test_norm) / self.bandwidth

        if self.kernel == 'gaussian':
            weights = np.exp(-0.5 * distances**2)
        elif self.kernel == 'epanechnikov':
            weights = np.maximum(0, 1 - distances**2)
        else:  # uniform
            weights = (np.abs(distances) <= 1).astype(float)

        # Normalize weights
        weights = weights / (weights.sum() + 1e-8)
        return weights

    def _weighted_quantile(self, values: np.ndarray, weights: np.ndarray,
                           q: float) -> float:
        """
        Compute weighted quantile.

        Uses linear interpolation for weighted quantile estimation.
        """
        # Sort by values
        sorter = np.argsort(values)
        values_sorted = values[sorter]
        weights_sorted = weights[sorter]

        # Cumulative weights
        cumsum = np.cumsum(weights_sorted)
        cumsum_normalized = cumsum / cumsum[-1]

        # Find quantile position
        idx = np.searchsorted(cumsum_normalized, q)
        if idx >= len(values_sorted):
            return values_sorted[-1]
        return values_sorted[idx]

    def fit(self, y_cal: np.ndarray, y_pred_cal: np.ndarray,
            signal_cal: np.ndarray) -> 'SignalAdaptiveCI':
        """
        Calibrate on held-out data.

        Parameters
        ----------
        y_cal : array
            True values on calibration set
        y_pred_cal : array
            Predicted values on calibration set
        signal_cal : array
            Volatility signals on calibration set
        """
        self.calibration_scores = np.abs(y_cal - y_pred_cal)
        self.signal_cal = signal_cal

        # Store signal statistics for normalization
        self.signal_mean = np.mean(signal_cal)
        self.signal_std = np.std(signal_cal)

        return self

    def predict(self, y_pred_test: np.ndarray,
                signal_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute adaptive prediction intervals.

        Parameters
        ----------
        y_pred_test : array
            Point predictions on test set
        signal_test : array
            Volatility signals on test set

        Returns
        -------
        lower, upper, width : arrays
            Adaptive prediction intervals and their widths
        """
        if self.calibration_scores is None:
            raise ValueError("Must call fit() before predict()")

        signal_cal_norm = self._normalize_signal(self.signal_cal)
        n_test = len(y_pred_test)

        lowers = np.zeros(n_test)
        uppers = np.zeros(n_test)
        widths = np.zeros(n_test)

        for i in range(n_test):
            signal_test_norm = self._normalize_signal(np.array([signal_test[i]]))[0]

            # Compute kernel weights based on signal similarity
            weights = self._compute_kernel_weights(signal_cal_norm, signal_test_norm)

            # Compute weighted quantile
            q = self._weighted_quantile(
                self.calibration_scores, weights, 1 - self.alpha
            )

            lowers[i] = y_pred_test[i] - q
            uppers[i] = y_pred_test[i] + q
            widths[i] = 2 * q

        return lowers, uppers, widths


class VanillaVolatilityScaling:
    """
    Simple volatility scaling baseline.

    Scales fixed conformal interval by volatility signal ratio.
    This is the simplest possible volatility-adaptive method.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.base_quantile = None
        self.median_signal = None

    def fit(self, y_cal: np.ndarray, y_pred_cal: np.ndarray,
            signal_cal: np.ndarray) -> 'VanillaVolatilityScaling':
        """Calibrate using standard conformal + store median signal."""
        scores = np.abs(y_cal - y_pred_cal)

        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)

        self.base_quantile = np.quantile(scores, q_level)
        self.median_signal = np.median(signal_cal)

        return self

    def predict(self, y_pred_test: np.ndarray,
                signal_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Scale intervals by signal ratio."""
        # Scale factor: signal / median_signal
        scale = signal_test / self.median_signal
        scale = np.clip(scale, 0.5, 2.0)  # Bound scaling

        widths = 2 * self.base_quantile * scale
        lowers = y_pred_test - self.base_quantile * scale
        uppers = y_pred_test + self.base_quantile * scale

        return lowers, uppers, widths


def compute_coverage_with_stats(y_true: np.ndarray, lower: np.ndarray,
                                 upper: np.ndarray) -> Tuple[float, float, int]:
    """
    Compute coverage with standard error.

    Returns
    -------
    coverage, standard_error, n
    """
    covered = (y_true >= lower) & (y_true <= upper)
    n = len(covered)
    cov = np.mean(covered)
    se = np.sqrt(cov * (1 - cov) / n)
    return cov, se, n


def compute_conditional_coverage_with_stats(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray,
    high_vol: np.ndarray
) -> Dict[str, Tuple[float, float, int]]:
    """
    Compute coverage by volatility regime with statistics.

    Returns
    -------
    dict with 'overall', 'high', 'low' each containing (coverage, se, n)
    """
    covered = (y_true >= lower) & (y_true <= upper)

    results = {}

    # Overall
    n_total = len(covered)
    cov_total = np.mean(covered)
    se_total = np.sqrt(cov_total * (1 - cov_total) / n_total)
    results['overall'] = (cov_total, se_total, n_total)

    # High volatility
    if high_vol.sum() > 0:
        n_high = high_vol.sum()
        cov_high = np.mean(covered[high_vol])
        se_high = np.sqrt(cov_high * (1 - cov_high) / n_high)
        results['high'] = (cov_high, se_high, n_high)
    else:
        results['high'] = (np.nan, np.nan, 0)

    # Low volatility
    if (~high_vol).sum() > 0:
        n_low = (~high_vol).sum()
        cov_low = np.mean(covered[~high_vol])
        se_low = np.sqrt(cov_low * (1 - cov_low) / n_low)
        results['low'] = (cov_low, se_low, n_low)
    else:
        results['low'] = (np.nan, np.nan, 0)

    return results


def calibrate_bandwidth(y_cal: np.ndarray, y_pred_cal: np.ndarray,
                        signal_cal: np.ndarray, alpha: float = 0.1,
                        target_coverage: float = 0.90,
                        bandwidth_range: List[float] = None) -> float:
    """
    Calibrate bandwidth to achieve target overall coverage.

    Uses leave-one-out cross-validation on calibration set.

    Parameters
    ----------
    y_cal, y_pred_cal, signal_cal : calibration data
    alpha : miscoverage rate
    target_coverage : target overall coverage
    bandwidth_range : list of bandwidths to try

    Returns
    -------
    optimal_bandwidth
    """
    if bandwidth_range is None:
        bandwidth_range = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

    best_bandwidth = 1.0
    best_diff = float('inf')

    n = len(y_cal)

    for bw in bandwidth_range:
        # Leave-one-out coverage estimation
        covered = np.zeros(n, dtype=bool)

        for i in range(n):
            # Fit on all except i
            mask = np.ones(n, dtype=bool)
            mask[i] = False

            saci = SignalAdaptiveCI(alpha=alpha, bandwidth=bw)
            saci.fit(y_cal[mask], y_pred_cal[mask], signal_cal[mask])

            # Predict for i
            lower, upper, _ = saci.predict(
                np.array([y_pred_cal[i]]),
                np.array([signal_cal[i]])
            )
            covered[i] = (y_cal[i] >= lower[0]) & (y_cal[i] <= upper[0])

        loo_coverage = np.mean(covered)
        diff = abs(loo_coverage - target_coverage)

        if diff < best_diff:
            best_diff = diff
            best_bandwidth = bw

    return best_bandwidth


def evaluate_all_methods(y_true: np.ndarray, y_pred: np.ndarray,
                         signal: np.ndarray, alpha: float = 0.1,
                         cal_fraction: float = 0.5,
                         calibrate: bool = True) -> Dict[str, ConformalResult]:
    """
    Compare all conformal methods on the same data.

    Methods compared:
    - Standard CP (baseline)
    - Signal-Adaptive CI (SA-CI) with local weighting
    - Vanilla volatility scaling

    Parameters
    ----------
    y_true : array
        True values
    y_pred : array
        Point predictions
    signal : array
        Volatility signals
    alpha : float
        Miscoverage rate
    cal_fraction : float
        Fraction of data used for calibration
    calibrate : bool
        Whether to calibrate bandwidth for SA-CI

    Returns
    -------
    dict with ConformalResult for each method
    """
    n = len(y_true)
    cal_end = int(n * cal_fraction)

    # Split data
    y_cal, y_test = y_true[:cal_end], y_true[cal_end:]
    pred_cal, pred_test = y_pred[:cal_end], y_pred[cal_end:]
    sig_cal, sig_test = signal[:cal_end], signal[cal_end:]

    # High volatility = above median
    signal_median = np.median(sig_test)
    high_vol = sig_test > signal_median

    results = {}

    # ===== Standard CP =====
    scp = StandardConformalPredictor(alpha=alpha)
    scp.fit(y_cal, pred_cal)
    lower_scp, upper_scp = scp.predict(pred_test)
    width_scp = np.full_like(lower_scp, scp.get_width())

    stats_scp = compute_conditional_coverage_with_stats(
        y_test, lower_scp, upper_scp, high_vol
    )

    results['standard'] = ConformalResult(
        lower=lower_scp, upper=upper_scp, width=width_scp,
        coverage_overall=stats_scp['overall'][0],
        coverage_high_vol=stats_scp['high'][0],
        coverage_low_vol=stats_scp['low'][0],
        se_overall=stats_scp['overall'][1],
        se_high_vol=stats_scp['high'][1],
        se_low_vol=stats_scp['low'][1],
        n_total=stats_scp['overall'][2],
        n_high_vol=stats_scp['high'][2],
        n_low_vol=stats_scp['low'][2]
    )

    # ===== Signal-Adaptive CI =====
    bandwidth = 1.0
    if calibrate:
        bandwidth = calibrate_bandwidth(y_cal, pred_cal, sig_cal, alpha)

    saci = SignalAdaptiveCI(alpha=alpha, bandwidth=bandwidth)
    saci.fit(y_cal, pred_cal, sig_cal)
    lower_saci, upper_saci, width_saci = saci.predict(pred_test, sig_test)

    stats_saci = compute_conditional_coverage_with_stats(
        y_test, lower_saci, upper_saci, high_vol
    )

    results['saci'] = ConformalResult(
        lower=lower_saci, upper=upper_saci, width=width_saci,
        coverage_overall=stats_saci['overall'][0],
        coverage_high_vol=stats_saci['high'][0],
        coverage_low_vol=stats_saci['low'][0],
        se_overall=stats_saci['overall'][1],
        se_high_vol=stats_saci['high'][1],
        se_low_vol=stats_saci['low'][1],
        n_total=stats_saci['overall'][2],
        n_high_vol=stats_saci['high'][2],
        n_low_vol=stats_saci['low'][2]
    )

    # ===== Vanilla Volatility Scaling =====
    vvs = VanillaVolatilityScaling(alpha=alpha)
    vvs.fit(y_cal, pred_cal, sig_cal)
    lower_vvs, upper_vvs, width_vvs = vvs.predict(pred_test, sig_test)

    stats_vvs = compute_conditional_coverage_with_stats(
        y_test, lower_vvs, upper_vvs, high_vol
    )

    results['vol_scaling'] = ConformalResult(
        lower=lower_vvs, upper=upper_vvs, width=width_vvs,
        coverage_overall=stats_vvs['overall'][0],
        coverage_high_vol=stats_vvs['high'][0],
        coverage_low_vol=stats_vvs['low'][0],
        se_overall=stats_vvs['overall'][1],
        se_high_vol=stats_vvs['high'][1],
        se_low_vol=stats_vvs['low'][1],
        n_total=stats_vvs['overall'][2],
        n_high_vol=stats_vvs['high'][2],
        n_low_vol=stats_vvs['low'][2]
    )

    return results


if __name__ == '__main__':
    # Test
    np.random.seed(42)
    n = 1000

    # Simulate data with signal-dependent volatility
    signal = np.abs(np.random.randn(n))
    volatility = 0.05 * (1 + 0.5 * signal)
    y_true = np.random.randn(n) * volatility
    y_pred = np.zeros(n)

    results = evaluate_all_methods(y_true, y_pred, signal, calibrate=False)

    print("Method Comparison:")
    print("-" * 70)
    for name, res in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Overall: {res.coverage_overall:.1%} (SE: {res.se_overall:.1%})")
        print(f"  High-vol: {res.coverage_high_vol:.1%} (SE: {res.se_high_vol:.1%})")
        print(f"  Low-vol: {res.coverage_low_vol:.1%} (SE: {res.se_low_vol:.1%})")
        print(f"  Avg width: {np.mean(res.width):.4f}")
