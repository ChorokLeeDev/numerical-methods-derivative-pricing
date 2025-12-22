"""
Crowding-Weighted Adaptive Conformal Inference (CW-ACI)

Core implementation for JoFE paper.

Key idea: Standard conformal prediction assumes exchangeability and produces
fixed-width intervals. In factor markets, volatility clusters around high-crowding
periods, causing under-coverage. CW-ACI adapts interval width based on crowding
signals while maintaining approximate coverage.

Author: Chorok Lee (KAIST)
Date: December 2024
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class ConformalResult:
    """Container for conformal prediction results."""
    lower: np.ndarray
    upper: np.ndarray
    width: np.ndarray
    coverage: float
    coverage_high_crowding: float
    coverage_low_crowding: float


class StandardConformalPredictor:
    """
    Standard split conformal prediction.

    Reference: Vovk et al. (2005), Lei et al. (2018)

    Given calibration data (X_cal, y_cal) and predictions y_pred_cal:
    1. Compute nonconformity scores: s_i = |y_i - y_pred_i|
    2. Find quantile q = Quantile(s, 1-alpha)
    3. Prediction interval: [y_pred - q, y_pred + q]

    Guarantees P(y in C(x)) >= 1 - alpha under exchangeability.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Initialize conformal predictor.

        Parameters
        ----------
        alpha : float
            Miscoverage rate. Default 0.1 for 90% coverage.
        """
        self.alpha = alpha
        self.calibration_scores = None
        self.quantile = None

    def fit(self, y_cal: np.ndarray, y_pred_cal: np.ndarray) -> 'StandardConformalPredictor':
        """
        Calibrate on held-out data.

        Parameters
        ----------
        y_cal : array
            True values on calibration set
        y_pred_cal : array
            Predicted values on calibration set
        """
        self.calibration_scores = np.abs(y_cal - y_pred_cal)

        # Compute quantile with finite-sample correction
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)

        self.quantile = np.quantile(self.calibration_scores, q_level)

        return self

    def predict(self, y_pred_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute prediction intervals.

        Parameters
        ----------
        y_pred_test : array
            Point predictions on test set

        Returns
        -------
        lower, upper : arrays
            Lower and upper bounds of prediction intervals
        """
        if self.quantile is None:
            raise ValueError("Must call fit() before predict()")

        lower = y_pred_test - self.quantile
        upper = y_pred_test + self.quantile

        return lower, upper

    def get_width(self) -> float:
        """Return interval width (constant for standard CP)."""
        return 2 * self.quantile


class CrowdingWeightedACI:
    """
    Crowding-Weighted Adaptive Conformal Inference (CW-ACI).

    Key innovation: Weight nonconformity scores by crowding level during
    quantile computation. High crowding → higher weight → wider intervals.

    Algorithm:
    1. Compute nonconformity scores on calibration set
    2. Compute crowding weights: w_i = sigmoid(crowding_i)
    3. For each test point with crowding c:
       - Adjust scores: s_adj = s * (1 + w(c))
       - Compute weighted quantile
       - Return adaptive interval

    Properties:
    - Wider intervals during high crowding (uncertainty signal)
    - Narrower intervals during low crowding (confidence signal)
    - Maintains approximate coverage overall
    """

    def __init__(self, alpha: float = 0.1, sensitivity: float = 1.0):
        """
        Initialize CW-ACI.

        Parameters
        ----------
        alpha : float
            Miscoverage rate. Default 0.1 for 90% coverage.
        sensitivity : float
            Crowding sensitivity parameter. Higher = more adaptation.
        """
        self.alpha = alpha
        self.sensitivity = sensitivity
        self.calibration_scores = None
        self.crowding_mean = None
        self.crowding_std = None

    def _normalize_crowding(self, crowding: np.ndarray) -> np.ndarray:
        """Normalize crowding to zero mean, unit variance."""
        return (crowding - self.crowding_mean) / (self.crowding_std + 1e-8)

    def _compute_weight(self, crowding_normalized: np.ndarray) -> np.ndarray:
        """
        Compute weights from normalized crowding.

        Uses sigmoid function: w = 1 / (1 + exp(-sensitivity * c))

        This maps crowding to (0, 1) where:
        - High crowding → w ≈ 1
        - Low crowding → w ≈ 0
        - Average crowding → w ≈ 0.5
        """
        return 1 / (1 + np.exp(-self.sensitivity * crowding_normalized))

    def fit(self, y_cal: np.ndarray, y_pred_cal: np.ndarray,
            crowding_cal: np.ndarray) -> 'CrowdingWeightedACI':
        """
        Calibrate on held-out data.

        Parameters
        ----------
        y_cal : array
            True values on calibration set
        y_pred_cal : array
            Predicted values on calibration set
        crowding_cal : array
            Crowding signals on calibration set
        """
        self.calibration_scores = np.abs(y_cal - y_pred_cal)

        # Store crowding statistics for normalization
        self.crowding_mean = np.mean(crowding_cal)
        self.crowding_std = np.std(crowding_cal)

        # Store normalized calibration crowding
        self.crowding_cal_normalized = self._normalize_crowding(crowding_cal)
        self.weights_cal = self._compute_weight(self.crowding_cal_normalized)

        return self

    def predict(self, y_pred_test: np.ndarray,
                crowding_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute adaptive prediction intervals.

        Parameters
        ----------
        y_pred_test : array
            Point predictions on test set
        crowding_test : array
            Crowding signals on test set

        Returns
        -------
        lower, upper, width : arrays
            Adaptive prediction intervals and their widths
        """
        if self.calibration_scores is None:
            raise ValueError("Must call fit() before predict()")

        # Normalize test crowding using calibration statistics
        crowding_test_normalized = self._normalize_crowding(crowding_test)
        weights_test = self._compute_weight(crowding_test_normalized)

        n_cal = len(self.calibration_scores)
        n_test = len(y_pred_test)

        lowers = np.zeros(n_test)
        uppers = np.zeros(n_test)
        widths = np.zeros(n_test)

        for i in range(n_test):
            # Adjust calibration scores based on test point's crowding
            # Higher test crowding → multiply scores by larger factor → wider interval
            adjustment_factor = 1 + weights_test[i]
            adjusted_scores = self.calibration_scores * adjustment_factor

            # Compute quantile with finite-sample correction
            q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
            q_level = min(q_level, 1.0)

            q = np.quantile(adjusted_scores, q_level)

            lowers[i] = y_pred_test[i] - q
            uppers[i] = y_pred_test[i] + q
            widths[i] = 2 * q

        return lowers, uppers, widths

    def get_adaptation_ratio(self, crowding_test: np.ndarray) -> np.ndarray:
        """
        Compute how much wider intervals are relative to baseline.

        Returns ratio of CW-ACI width to standard CP width.
        """
        crowding_normalized = self._normalize_crowding(crowding_test)
        weights = self._compute_weight(crowding_normalized)
        return 1 + weights


def compute_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Compute empirical coverage rate."""
    covered = (y_true >= lower) & (y_true <= upper)
    return np.mean(covered)


def compute_conditional_coverage(y_true: np.ndarray, lower: np.ndarray,
                                  upper: np.ndarray, condition: np.ndarray) -> Dict[str, float]:
    """
    Compute coverage conditional on a binary condition.

    Parameters
    ----------
    y_true : array
        True values
    lower, upper : arrays
        Prediction interval bounds
    condition : array
        Boolean array (True = high crowding, False = low crowding)

    Returns
    -------
    dict with 'high' and 'low' coverage rates
    """
    covered = (y_true >= lower) & (y_true <= upper)

    return {
        'high': np.mean(covered[condition]) if condition.sum() > 0 else np.nan,
        'low': np.mean(covered[~condition]) if (~condition).sum() > 0 else np.nan,
        'overall': np.mean(covered)
    }


def evaluate_conformal_methods(y_true: np.ndarray, y_pred: np.ndarray,
                                crowding: np.ndarray, alpha: float = 0.1,
                                cal_fraction: float = 0.5) -> Dict[str, ConformalResult]:
    """
    Compare standard CP and CW-ACI on the same data.

    Parameters
    ----------
    y_true : array
        True values
    y_pred : array
        Point predictions
    crowding : array
        Crowding signals
    alpha : float
        Miscoverage rate
    cal_fraction : float
        Fraction of data used for calibration

    Returns
    -------
    dict with 'standard' and 'cwaci' ConformalResult objects
    """
    n = len(y_true)
    cal_end = int(n * cal_fraction)

    # Split data
    y_cal, y_test = y_true[:cal_end], y_true[cal_end:]
    pred_cal, pred_test = y_pred[:cal_end], y_pred[cal_end:]
    crowd_cal, crowd_test = crowding[:cal_end], crowding[cal_end:]

    # High crowding = above median
    crowding_median = np.median(crowd_test)
    high_crowding = crowd_test > crowding_median

    results = {}

    # Standard CP
    scp = StandardConformalPredictor(alpha=alpha)
    scp.fit(y_cal, pred_cal)
    lower_scp, upper_scp = scp.predict(pred_test)

    cov_scp = compute_conditional_coverage(y_test, lower_scp, upper_scp, high_crowding)

    results['standard'] = ConformalResult(
        lower=lower_scp,
        upper=upper_scp,
        width=np.full_like(lower_scp, scp.get_width()),
        coverage=cov_scp['overall'],
        coverage_high_crowding=cov_scp['high'],
        coverage_low_crowding=cov_scp['low']
    )

    # CW-ACI
    cwaci = CrowdingWeightedACI(alpha=alpha)
    cwaci.fit(y_cal, pred_cal, crowd_cal)
    lower_cw, upper_cw, width_cw = cwaci.predict(pred_test, crowd_test)

    cov_cw = compute_conditional_coverage(y_test, lower_cw, upper_cw, high_crowding)

    results['cwaci'] = ConformalResult(
        lower=lower_cw,
        upper=upper_cw,
        width=width_cw,
        coverage=cov_cw['overall'],
        coverage_high_crowding=cov_cw['high'],
        coverage_low_crowding=cov_cw['low']
    )

    return results


if __name__ == '__main__':
    # Quick test
    np.random.seed(42)
    n = 1000

    # Simulate data with crowding-dependent volatility
    crowding = np.random.randn(n)
    volatility = 1 + 0.5 * (crowding > 0)  # Higher vol when crowding high
    y_true = np.random.randn(n) * volatility
    y_pred = np.zeros(n)  # Naive predictor

    results = evaluate_conformal_methods(y_true, y_pred, crowding)

    print("Standard CP:")
    print(f"  Overall coverage: {results['standard'].coverage:.1%}")
    print(f"  High crowding coverage: {results['standard'].coverage_high_crowding:.1%}")
    print(f"  Low crowding coverage: {results['standard'].coverage_low_crowding:.1%}")

    print("\nCW-ACI:")
    print(f"  Overall coverage: {results['cwaci'].coverage:.1%}")
    print(f"  High crowding coverage: {results['cwaci'].coverage_high_crowding:.1%}")
    print(f"  Low crowding coverage: {results['cwaci'].coverage_low_crowding:.1%}")
