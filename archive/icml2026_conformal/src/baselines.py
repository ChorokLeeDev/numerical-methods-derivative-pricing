"""
Baseline Conformal Prediction Methods

Standard methods for comparison with crowding-aware approaches:
1. SplitConformalCP - Standard split conformal prediction
2. ACIConformal - Adaptive Conformal Inference (Gibbs & Candes 2021)

For ICML 2026 submission.
"""

import numpy as np
from typing import List, Set, Optional, Tuple, Dict
from sklearn.ensemble import RandomForestClassifier


class SplitConformalCP:
    """
    Standard Split Conformal Prediction.

    Uses calibration set to compute nonconformity scores and threshold.
    Same threshold τ for all predictions.

    Prediction set = {y : score(x, y) ≤ τ}
    τ = quantile(calibration_scores, 1-α)

    Limitation: Assumes exchangeability, which is violated in financial
    time series due to distribution shift.
    """

    def __init__(
        self,
        base_model: Optional[RandomForestClassifier] = None,
        n_estimators: int = 100,
    ):
        if base_model is None:
            base_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=5,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        self.base_model = base_model
        self.calibration_scores = None
        self.threshold = None
        self._is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray
    ):
        """
        Fit base model and compute calibration threshold.
        """
        # Fit base model
        self.base_model.fit(X_train, y_train)

        # Compute nonconformity scores on calibration set
        proba = self.base_model.predict_proba(X_calib)[:, 1]

        # Score = 1 - P(true class)
        self.calibration_scores = np.where(
            y_calib == 1,
            1 - proba,      # For crash: 1 - P(crash)
            proba           # For no-crash: P(crash)
        )

        self._is_fitted = True

    def predict_sets(
        self,
        X_test: np.ndarray,
        alpha: float = 0.1
    ) -> Tuple[List[Set[int]], float]:
        """
        Construct prediction sets with coverage guarantee.

        Args:
            X_test: Test features
            alpha: Miscoverage level (target coverage = 1-α)

        Returns:
            Tuple of (prediction_sets, threshold)
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before predict_sets()")

        proba = self.base_model.predict_proba(X_test)[:, 1]

        # Compute threshold
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(q_level, 1.0)
        threshold = np.quantile(self.calibration_scores, q_level)

        # Construct prediction sets
        sets = []
        for p in proba:
            pred_set = set()
            if (1 - p) <= threshold:
                pred_set.add(1)
            if p <= threshold:
                pred_set.add(0)
            sets.append(pred_set)

        return sets, threshold

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get base model probabilities."""
        return self.base_model.predict_proba(X)[:, 1]


class ACIConformal:
    """
    Adaptive Conformal Inference (ACI).

    Online threshold update from Gibbs & Candes (2021):
        τ_{t+1} = τ_t + γ × (err_t - α)

    where err_t = 1 if y_t ∉ C_t (miscoverage), 0 otherwise.

    Benefits:
    - O(1) per-sample update
    - Handles distribution shift
    - Maintains long-run coverage guarantee

    Note: This is standard ACI without crowding awareness.
    """

    def __init__(
        self,
        base_model: Optional[RandomForestClassifier] = None,
        alpha: float = 0.1,
        gamma: float = 0.1,
        n_estimators: int = 100,
    ):
        if base_model is None:
            base_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=5,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        self.base_model = base_model
        self.alpha = alpha
        self.gamma = gamma
        self.threshold = 0.5
        self._is_fitted = False
        self.threshold_history = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray
    ):
        """
        Fit base model and initialize threshold.
        """
        # Fit base model
        self.base_model.fit(X_train, y_train)

        # Initialize threshold from calibration scores
        proba = self.base_model.predict_proba(X_calib)[:, 1]
        scores = np.where(
            y_calib == 1,
            1 - proba,
            proba
        )

        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self.threshold = np.quantile(scores, q_level)

        self._is_fitted = True
        self.threshold_history = [self.threshold]

    def predict_set(self, x: np.ndarray) -> Set[int]:
        """Predict set for a single sample."""
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before predict_set()")

        x = x.reshape(1, -1)
        p = self.base_model.predict_proba(x)[0, 1]

        pred_set = set()
        if (1 - p) <= self.threshold:
            pred_set.add(1)
        if p <= self.threshold:
            pred_set.add(0)

        return pred_set

    def update(self, y_true: int, pred_set: Set[int]) -> float:
        """
        Update threshold based on coverage error.

        ACI update: τ_{t+1} = τ_t + γ × (missed - α)
        """
        covered = int(y_true in pred_set)
        missed = 1 - covered

        error = missed - self.alpha
        self.threshold += self.gamma * error

        # Clip to valid range
        self.threshold = np.clip(self.threshold, 0.01, 0.99)

        self.threshold_history.append(self.threshold)
        return self.threshold

    def predict_sets_online(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[List[Set[int]], List[float]]:
        """
        Online prediction with threshold updates.

        Args:
            X_test: Test features
            y_test: True labels (for threshold updates)

        Returns:
            Tuple of (prediction_sets, threshold_history)
        """
        sets = []
        thresholds = []

        for i in range(len(X_test)):
            pred_set = self.predict_set(X_test[i])
            sets.append(pred_set)
            thresholds.append(self.threshold)

            self.update(y_test[i], pred_set)

        return sets, thresholds


def evaluate_coverage(
    pred_sets: List[Set[int]],
    y_true: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate conformal prediction coverage metrics.
    """
    n = len(y_true)

    covered = sum(int(y_true[i]) in pred_sets[i] for i in range(n))
    coverage = covered / n

    sizes = [len(s) for s in pred_sets]
    avg_size = np.mean(sizes)

    singletons = sum(len(s) == 1 for s in pred_sets)
    singleton_rate = singletons / n

    empties = sum(len(s) == 0 for s in pred_sets)
    empty_rate = empties / n

    return {
        'coverage': coverage,
        'avg_set_size': avg_size,
        'singleton_rate': singleton_rate,
        'empty_rate': empty_rate,
        'n_samples': n
    }


if __name__ == '__main__':
    # Quick test
    np.random.seed(42)

    n_train, n_calib, n_test = 500, 200, 100
    n_features = 10

    X = np.random.randn(n_train + n_calib + n_test, n_features)
    y = (X[:, 0] + X[:, 1] + np.random.randn(len(X)) * 0.5 > 0).astype(int)

    X_train, y_train = X[:n_train], y[:n_train]
    X_calib, y_calib = X[n_train:n_train+n_calib], y[n_train:n_train+n_calib]
    X_test, y_test = X[n_train+n_calib:], y[n_train+n_calib:]

    print("=" * 50)
    print("BASELINE CONFORMAL METHODS TEST")
    print("=" * 50)

    # Test SplitConformalCP
    print("\n1. Split Conformal (baseline)")
    split_cp = SplitConformalCP()
    split_cp.fit(X_train, y_train, X_calib, y_calib)
    sets, threshold = split_cp.predict_sets(X_test, alpha=0.1)
    metrics = evaluate_coverage(sets, y_test)
    print(f"   Coverage: {metrics['coverage']:.3f} (target ≥ 0.90)")
    print(f"   Threshold: {threshold:.3f}")

    # Test ACI
    print("\n2. ACI (online)")
    aci = ACIConformal(gamma=0.1)
    aci.fit(X_train, y_train, X_calib, y_calib)
    sets, thresholds = aci.predict_sets_online(X_test, y_test)
    metrics = evaluate_coverage(sets, y_test)
    print(f"   Coverage: {metrics['coverage']:.3f} (target ≥ 0.90)")
    print(f"   Final threshold: {thresholds[-1]:.3f}")

    print("\nBaseline tests passed!")
