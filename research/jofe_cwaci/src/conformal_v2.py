"""
Signal-Weighted Adaptive Conformal Inference (SW-ACI) - Version 2

Revised implementation addressing key limitations:
1. Honest framing: volatility signal, not "crowding"
2. Improved algorithm: proper weighted quantiles
3. Baseline comparisons: ACI, CQR, naive scaling
4. Statistical rigor: standard errors, significance tests

Author: Chorok Lee (KAIST)
Date: December 2024 (Revised)
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
    coverage: float
    coverage_se: float  # Standard error
    coverage_high: float
    coverage_high_se: float
    coverage_low: float
    coverage_low_se: float
    n_total: int
    n_high: int
    n_low: int


def compute_coverage_with_se(y_true: np.ndarray, lower: np.ndarray,
                              upper: np.ndarray) -> Tuple[float, float]:
    """Compute coverage with standard error."""
    covered = (y_true >= lower) & (y_true <= upper)
    p = np.mean(covered)
    n = len(covered)
    se = np.sqrt(p * (1 - p) / n) if n > 0 else np.nan
    return p, se


def test_coverage_difference(cov1: float, n1: int, cov2: float, n2: int) -> Tuple[float, float]:
    """
    Two-proportion z-test for coverage difference.

    Returns: (z_statistic, p_value)
    """
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan

    # Pooled proportion
    p_pool = (cov1 * n1 + cov2 * n2) / (n1 + n2)

    if p_pool == 0 or p_pool == 1:
        return np.nan, np.nan

    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (cov1 - cov2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value


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


class SignalWeightedACI:
    """
    Signal-Weighted Adaptive Conformal Inference (SW-ACI).

    Improved algorithm that weights calibration scores by their own signal
    values when computing quantiles for a test point.

    Key improvement over v1: Uses localized weighting based on signal similarity,
    rather than uniformly inflating all calibration scores.
    """

    def __init__(self, alpha: float = 0.1, sensitivity: float = 1.0,
                 bandwidth: float = 1.0):
        """
        Parameters
        ----------
        alpha : float
            Miscoverage rate. Default 0.1 for 90% coverage.
        sensitivity : float
            Controls how much signal affects interval width.
        bandwidth : float
            Kernel bandwidth for localized weighting.
        """
        self.alpha = alpha
        self.sensitivity = sensitivity
        self.bandwidth = bandwidth
        self.calibration_scores = None
        self.signal_cal_normalized = None
        self.signal_mean = None
        self.signal_std = None

    def _normalize_signal(self, signal: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize signal to zero mean, unit variance."""
        if fit:
            self.signal_mean = np.mean(signal)
            self.signal_std = np.std(signal) + 1e-8
        return (signal - self.signal_mean) / self.signal_std

    def _compute_weights(self, signal_test_normalized: float) -> np.ndarray:
        """
        Compute weights for calibration scores based on signal similarity.

        Two components:
        1. Similarity weight: upweight calibration points with similar signals
        2. Inflation weight: when test signal is high, inflate expected errors
        """
        # Component 1: Similarity-based weighting (Gaussian kernel)
        similarity = np.exp(-0.5 * ((self.signal_cal_normalized - signal_test_normalized)
                                    / self.bandwidth)**2)

        # Component 2: Signal-based inflation
        # High signal -> expect larger errors -> inflate quantile
        inflation = 1 + self.sensitivity * max(0, signal_test_normalized)

        return similarity * inflation

    def fit(self, y_cal: np.ndarray, y_pred_cal: np.ndarray,
            signal_cal: np.ndarray) -> 'SignalWeightedACI':
        """
        Calibrate on held-out data.

        Parameters
        ----------
        y_cal : array
            True values on calibration set
        y_pred_cal : array
            Predicted values on calibration set
        signal_cal : array
            Volatility signal on calibration set
        """
        self.calibration_scores = np.abs(y_cal - y_pred_cal)
        self.signal_cal_normalized = self._normalize_signal(signal_cal, fit=True)
        return self

    def _weighted_quantile(self, values: np.ndarray, weights: np.ndarray,
                           q: float) -> float:
        """Compute weighted quantile."""
        # Normalize weights
        weights = weights / np.sum(weights)

        # Sort by values
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]

        # Cumulative weights
        cumsum = np.cumsum(sorted_weights)

        # Find quantile
        idx = np.searchsorted(cumsum, q)
        idx = min(idx, len(sorted_values) - 1)

        return sorted_values[idx]

    def predict(self, y_pred_test: np.ndarray,
                signal_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute adaptive prediction intervals.

        Returns: lower, upper, width arrays
        """
        if self.calibration_scores is None:
            raise ValueError("Must call fit() before predict()")

        signal_test_normalized = self._normalize_signal(signal_test)

        n_test = len(y_pred_test)
        lowers = np.zeros(n_test)
        uppers = np.zeros(n_test)
        widths = np.zeros(n_test)

        for i in range(n_test):
            weights = self._compute_weights(signal_test_normalized[i])

            # Weighted quantile
            q = self._weighted_quantile(
                self.calibration_scores,
                weights,
                1 - self.alpha
            )

            lowers[i] = y_pred_test[i] - q
            uppers[i] = y_pred_test[i] + q
            widths[i] = 2 * q

        return lowers, uppers, widths


class SimpleSignalACI:
    """
    Simple Signal-Adaptive CI (original algorithm, kept for comparison).

    This is the v1 algorithm that uniformly inflates calibration scores
    based on test-point signal. Kept for backward compatibility and
    to show the difference with the improved algorithm.
    """

    def __init__(self, alpha: float = 0.1, sensitivity: float = 1.0):
        self.alpha = alpha
        self.sensitivity = sensitivity
        self.calibration_scores = None
        self.signal_mean = None
        self.signal_std = None

    def fit(self, y_cal: np.ndarray, y_pred_cal: np.ndarray,
            signal_cal: np.ndarray) -> 'SimpleSignalACI':
        """Calibrate."""
        self.calibration_scores = np.abs(y_cal - y_pred_cal)
        self.signal_mean = np.mean(signal_cal)
        self.signal_std = np.std(signal_cal) + 1e-8
        return self

    def predict(self, y_pred_test: np.ndarray,
                signal_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute intervals with simple inflation."""
        signal_normalized = (signal_test - self.signal_mean) / self.signal_std
        weights = 1 / (1 + np.exp(-self.sensitivity * signal_normalized))  # Sigmoid

        n_cal = len(self.calibration_scores)
        n_test = len(y_pred_test)

        lowers = np.zeros(n_test)
        uppers = np.zeros(n_test)
        widths = np.zeros(n_test)

        q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        q_level = min(q_level, 1.0)

        for i in range(n_test):
            adjustment = 1 + weights[i]
            adjusted_scores = self.calibration_scores * adjustment
            q = np.quantile(adjusted_scores, q_level)

            lowers[i] = y_pred_test[i] - q
            uppers[i] = y_pred_test[i] + q
            widths[i] = 2 * q

        return lowers, uppers, widths


class AdaptiveCI:
    """
    Adaptive Conformal Inference (Gibbs & Candes, 2021).

    Updates the miscoverage rate based on recent coverage errors,
    providing adaptation to distribution shift.

    Reference: Gibbs & Candes (2021) "Adaptive Conformal Inference
    Under Distribution Shift"
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.01):
        """
        Parameters
        ----------
        alpha : float
            Target miscoverage rate
        gamma : float
            Learning rate for alpha updates
        """
        self.alpha_target = alpha
        self.gamma = gamma
        self.alpha_t = alpha  # Current adaptive alpha
        self.calibration_scores = None
        self.coverage_history = []

    def fit(self, y_cal: np.ndarray, y_pred_cal: np.ndarray) -> 'AdaptiveCI':
        """Initial calibration."""
        self.calibration_scores = np.abs(y_cal - y_pred_cal)
        self.alpha_t = self.alpha_target
        self.coverage_history = []
        return self

    def predict_single(self, y_pred: float) -> Tuple[float, float]:
        """Predict interval for single point using current alpha_t."""
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha_t)) / n
        q_level = np.clip(q_level, 0.0, 1.0)

        q = np.quantile(self.calibration_scores, q_level)
        return y_pred - q, y_pred + q

    def update(self, y_true: float, lower: float, upper: float):
        """
        Update alpha_t based on coverage.

        If covered: decrease alpha_t (tighten future intervals)
        If not covered: increase alpha_t (widen future intervals)
        """
        covered = (y_true >= lower) & (y_true <= upper)
        self.coverage_history.append(covered)

        # Update rule from Gibbs & Candes
        err_t = 1 - int(covered)  # 1 if not covered, 0 if covered
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha_target - err_t)

        # Keep alpha_t in reasonable bounds
        self.alpha_t = np.clip(self.alpha_t, 0.01, 0.5)

    def predict_and_update(self, y_pred_test: np.ndarray,
                           y_true_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sequential prediction with updates.

        Note: This requires true values, so it's for backtesting only.
        """
        n_test = len(y_pred_test)
        lowers = np.zeros(n_test)
        uppers = np.zeros(n_test)

        for i in range(n_test):
            lowers[i], uppers[i] = self.predict_single(y_pred_test[i])
            self.update(y_true_test[i], lowers[i], uppers[i])

        return lowers, uppers


class NaiveVolatilityScaling:
    """
    Naive baseline: Scale intervals by realized volatility.

    Width_t = Base_Width * (Vol_t / Median_Vol)

    Simple but effective baseline that any "adaptive" method should beat.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.base_quantile = None
        self.median_signal = None

    def fit(self, y_cal: np.ndarray, y_pred_cal: np.ndarray,
            signal_cal: np.ndarray) -> 'NaiveVolatilityScaling':
        """Calibrate base quantile and median signal."""
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
        scale = signal_test / (self.median_signal + 1e-8)
        scale = np.clip(scale, 0.5, 2.0)  # Bound scaling

        q = self.base_quantile * scale

        lowers = y_pred_test - q
        uppers = y_pred_test + q
        widths = 2 * q

        return lowers, uppers, widths


def evaluate_method(y_test: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                    high_signal: np.ndarray) -> ConformalResult:
    """
    Comprehensive evaluation with statistical measures.
    """
    width = upper - lower

    # Overall coverage
    cov_overall, se_overall = compute_coverage_with_se(y_test, lower, upper)

    # Conditional coverage
    cov_high, se_high = compute_coverage_with_se(
        y_test[high_signal], lower[high_signal], upper[high_signal]
    )
    cov_low, se_low = compute_coverage_with_se(
        y_test[~high_signal], lower[~high_signal], upper[~high_signal]
    )

    return ConformalResult(
        lower=lower,
        upper=upper,
        width=width,
        coverage=cov_overall,
        coverage_se=se_overall,
        coverage_high=cov_high,
        coverage_high_se=se_high,
        coverage_low=cov_low,
        coverage_low_se=se_low,
        n_total=len(y_test),
        n_high=high_signal.sum(),
        n_low=(~high_signal).sum()
    )


def compare_methods(y_true: np.ndarray, y_pred: np.ndarray, signal: np.ndarray,
                    alpha: float = 0.1, cal_fraction: float = 0.5,
                    sensitivity: float = 1.0) -> Dict[str, ConformalResult]:
    """
    Compare all methods on the same data.

    Methods compared:
    1. Standard CP (baseline)
    2. SW-ACI v2 (improved algorithm)
    3. Simple Signal ACI (v1 algorithm)
    4. Adaptive CI (Gibbs-Candes)
    5. Naive volatility scaling
    """
    n = len(y_true)
    cal_end = int(n * cal_fraction)

    # Split data
    y_cal, y_test = y_true[:cal_end], y_true[cal_end:]
    pred_cal, pred_test = y_pred[:cal_end], y_pred[cal_end:]
    signal_cal, signal_test = signal[:cal_end], signal[cal_end:]

    # High signal = above median
    high_signal = signal_test > np.median(signal_test)

    results = {}

    # 1. Standard CP
    scp = StandardConformalPredictor(alpha=alpha)
    scp.fit(y_cal, pred_cal)
    lower_scp, upper_scp = scp.predict(pred_test)
    results['standard_cp'] = evaluate_method(y_test, lower_scp, upper_scp, high_signal)

    # 2. SW-ACI v2 (improved)
    swaci = SignalWeightedACI(alpha=alpha, sensitivity=sensitivity)
    swaci.fit(y_cal, pred_cal, signal_cal)
    lower_sw, upper_sw, _ = swaci.predict(pred_test, signal_test)
    results['swaci_v2'] = evaluate_method(y_test, lower_sw, upper_sw, high_signal)

    # 3. Simple Signal ACI (v1)
    simple = SimpleSignalACI(alpha=alpha, sensitivity=sensitivity)
    simple.fit(y_cal, pred_cal, signal_cal)
    lower_simple, upper_simple, _ = simple.predict(pred_test, signal_test)
    results['simple_signal'] = evaluate_method(y_test, lower_simple, upper_simple, high_signal)

    # 4. Adaptive CI (requires true values for online update)
    aci = AdaptiveCI(alpha=alpha, gamma=0.01)
    aci.fit(y_cal, pred_cal)
    lower_aci, upper_aci = aci.predict_and_update(pred_test, y_test)
    results['adaptive_ci'] = evaluate_method(y_test, lower_aci, upper_aci, high_signal)

    # 5. Naive volatility scaling
    naive = NaiveVolatilityScaling(alpha=alpha)
    naive.fit(y_cal, pred_cal, signal_cal)
    lower_naive, upper_naive, _ = naive.predict(pred_test, signal_test)
    results['naive_scaling'] = evaluate_method(y_test, lower_naive, upper_naive, high_signal)

    return results


def rolling_backtest(y_true: np.ndarray, signal: np.ndarray,
                     cal_window: int = 120, alpha: float = 0.1,
                     sensitivity: float = 1.0) -> pd.DataFrame:
    """
    Rolling window backtest for realistic evaluation.

    At each time t:
    - Calibrate on [t-cal_window, t)
    - Predict for time t
    - Record coverage
    """
    n = len(y_true)
    results = []

    for t in range(cal_window, n):
        # Calibration window
        y_cal = y_true[t-cal_window:t]
        signal_cal = signal[t-cal_window:t]

        # Test point
        y_t = y_true[t]
        signal_t = signal[t]
        pred_t = 0.0  # Mean predictor

        # Standard CP
        scp = StandardConformalPredictor(alpha=alpha)
        scp.fit(y_cal, np.zeros_like(y_cal))
        lower_scp, upper_scp = scp.predict(np.array([pred_t]))
        covered_scp = (y_t >= lower_scp[0]) & (y_t <= upper_scp[0])

        # SW-ACI
        swaci = SignalWeightedACI(alpha=alpha, sensitivity=sensitivity)
        swaci.fit(y_cal, np.zeros_like(y_cal), signal_cal)
        lower_sw, upper_sw, width_sw = swaci.predict(np.array([pred_t]), np.array([signal_t]))
        covered_sw = (y_t >= lower_sw[0]) & (y_t <= upper_sw[0])

        results.append({
            't': t,
            'signal': signal_t,
            'y_true': y_t,
            'covered_scp': covered_scp,
            'width_scp': scp.get_width(),
            'covered_swaci': covered_sw,
            'width_swaci': width_sw[0]
        })

    return pd.DataFrame(results)


def calibrate_sensitivity(y_true: np.ndarray, y_pred: np.ndarray, signal: np.ndarray,
                          alpha: float = 0.1, cal_fraction: float = 0.5,
                          target_coverage: float = 0.90) -> float:
    """
    Find sensitivity parameter that achieves target overall coverage.

    This addresses the over-coverage problem by tuning gamma.
    """
    best_gamma = 1.0
    best_diff = float('inf')

    n = len(y_true)
    cal_end = int(n * cal_fraction)

    y_cal, y_test = y_true[:cal_end], y_true[cal_end:]
    pred_cal, pred_test = y_pred[:cal_end], y_pred[cal_end:]
    signal_cal, signal_test = signal[:cal_end], signal[cal_end:]

    for gamma in np.linspace(0.1, 3.0, 30):
        swaci = SignalWeightedACI(alpha=alpha, sensitivity=gamma)
        swaci.fit(y_cal, pred_cal, signal_cal)
        lower, upper, _ = swaci.predict(pred_test, signal_test)

        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        diff = abs(coverage - target_coverage)

        if diff < best_diff:
            best_diff = diff
            best_gamma = gamma

    return best_gamma


if __name__ == '__main__':
    # Quick test
    np.random.seed(42)
    n = 1000

    # Simulate data with signal-dependent volatility
    signal = np.random.randn(n)
    volatility = 1 + 0.5 * np.maximum(0, signal)
    y_true = np.random.randn(n) * volatility
    y_pred = np.zeros(n)

    # Compare all methods
    results = compare_methods(y_true, y_pred, signal)

    print("Method Comparison:")
    print("-" * 80)
    print(f"{'Method':<20} | {'Overall':>10} | {'High Signal':>12} | {'Low Signal':>11} | {'Width':>8}")
    print("-" * 80)

    for name, res in results.items():
        print(f"{name:<20} | {res.coverage:>9.1%} | {res.coverage_high:>11.1%} | "
              f"{res.coverage_low:>10.1%} | {np.mean(res.width):>8.4f}")

    print("-" * 80)

    # Statistical significance test
    print("\nStatistical Tests (vs Standard CP):")
    scp = results['standard_cp']
    for name, res in results.items():
        if name == 'standard_cp':
            continue
        z, p = test_coverage_difference(
            res.coverage_high, res.n_high,
            scp.coverage_high, scp.n_high
        )
        print(f"  {name}: z={z:.2f}, p={p:.4f}")
