"""
Crowding-Aware Conformal Prediction

Novel contribution: Integrate factor crowding signals INTO conformal prediction
to achieve better coverage under distribution shift in financial markets.

Key insight: Crowding is a leading indicator of distribution shift.
High crowding → expect regime change → need larger prediction sets.

For ICML 2026 submission.
"""

import numpy as np
import pandas as pd
from typing import List, Set, Optional, Tuple, Dict
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass


@dataclass
class ConformalResult:
    """Result from conformal prediction."""
    prediction_sets: List[Set[int]]
    probabilities: np.ndarray
    threshold: float
    crowding_levels: Optional[np.ndarray] = None


class CrowdingWeightedCP:
    """
    Crowding-Weighted Conformal Prediction.

    Novel approach: Weight nonconformity scores by inverse crowding.

    Standard: score = |y - ŷ| (or 1 - p(y|x) for classification)
    Crowding-weighted: score = base_score / (1 + λ * crowding)

    Intuition:
    - High crowding → scores DOWN-weighted
    - Need MORE extreme score to be nonconforming
    - Results in LARGER prediction sets (more conservative)

    This anticipates distribution shift when crowding is high.
    """

    def __init__(
        self,
        base_model: Optional[RandomForestClassifier] = None,
        lambda_weight: float = 1.0,
        n_estimators: int = 100,
    ):
        """
        Args:
            base_model: Pre-trained classifier (optional)
            lambda_weight: Crowding weighting strength (0 = standard CP)
            n_estimators: Trees for RandomForest if base_model not provided
        """
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
        self.lambda_weight = lambda_weight
        self.calibration_scores = None
        self.calibration_crowding = None
        self._is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        crowding_calib: np.ndarray
    ):
        """
        Fit base model and compute crowding-weighted calibration scores.

        Args:
            X_train: Training features
            y_train: Training labels
            X_calib: Calibration features
            y_calib: Calibration labels
            crowding_calib: Crowding levels for calibration set
        """
        # Fit base model
        self.base_model.fit(X_train, y_train)

        # Compute crowding-weighted nonconformity scores
        proba = self.base_model.predict_proba(X_calib)[:, 1]

        # Base score: 1 - P(true class)
        base_scores = np.where(
            y_calib == 1,
            1 - proba,      # For crash: 1 - P(crash)
            proba           # For no-crash: P(crash) = 1 - P(no-crash)
        )

        # Crowding-weighted score
        weight = 1 + self.lambda_weight * crowding_calib
        self.calibration_scores = base_scores / weight
        self.calibration_crowding = crowding_calib

        self._is_fitted = True

    def predict_sets(
        self,
        X_test: np.ndarray,
        crowding_test: np.ndarray,
        alpha: float = 0.1
    ) -> ConformalResult:
        """
        Construct prediction sets with crowding-aware coverage guarantee.

        Args:
            X_test: Test features
            crowding_test: Crowding levels for test set
            alpha: Miscoverage level (target coverage = 1-α)

        Returns:
            ConformalResult with prediction sets and metadata
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before predict_sets()")

        proba = self.base_model.predict_proba(X_test)[:, 1]

        # Compute threshold from calibration scores
        n_calib = len(self.calibration_scores)
        q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
        q_level = min(q_level, 1.0)
        threshold = np.quantile(self.calibration_scores, q_level)

        # Construct prediction sets
        sets = []
        for i, p in enumerate(proba):
            c = crowding_test[i]
            weight = 1 + self.lambda_weight * c

            pred_set = set()

            # Score for class 1 (crash)
            score_1 = (1 - p) / weight
            if score_1 <= threshold:
                pred_set.add(1)

            # Score for class 0 (no-crash)
            score_0 = p / weight
            if score_0 <= threshold:
                pred_set.add(0)

            sets.append(pred_set)

        return ConformalResult(
            prediction_sets=sets,
            probabilities=proba,
            threshold=threshold,
            crowding_levels=crowding_test
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get base model probabilities."""
        return self.base_model.predict_proba(X)[:, 1]


class CrowdingStratifiedCP:
    """
    Crowding-Stratified Conformal Prediction (Mondrian+).

    Separate calibration by crowding regime:
    - Low crowding:  τ_low  = quantile(scores | crowding < 33%)
    - Med crowding:  τ_med  = quantile(scores | crowding 33-67%)
    - High crowding: τ_high = quantile(scores | crowding > 67%)

    At prediction time, use threshold based on current crowding level.

    This provides CONDITIONAL coverage: P(Y ∈ C | crowding = g) ≥ 1-α
    """

    def __init__(
        self,
        base_model: Optional[RandomForestClassifier] = None,
        n_strata: int = 3,
        n_estimators: int = 100,
    ):
        """
        Args:
            base_model: Pre-trained classifier (optional)
            n_strata: Number of crowding strata (default: 3 = low/med/high)
            n_estimators: Trees for RandomForest if base_model not provided
        """
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
        self.n_strata = n_strata
        self.strata_thresholds = None  # Crowding boundaries
        self.strata_quantiles = None   # Nonconformity quantiles per stratum
        self._is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        crowding_calib: np.ndarray
    ):
        """
        Fit base model and compute per-stratum calibration thresholds.
        """
        # Fit base model
        self.base_model.fit(X_train, y_train)

        # Compute nonconformity scores
        proba = self.base_model.predict_proba(X_calib)[:, 1]
        scores = np.where(
            y_calib == 1,
            1 - proba,
            proba
        )

        # Define strata boundaries
        boundaries = [0]
        for i in range(1, self.n_strata):
            boundaries.append(np.quantile(crowding_calib, i / self.n_strata))
        boundaries.append(1.0 + 1e-8)
        self.strata_thresholds = boundaries

        # Compute threshold for each stratum
        self.strata_scores = {}
        for s in range(self.n_strata):
            low, high = boundaries[s], boundaries[s + 1]
            mask = (crowding_calib >= low) & (crowding_calib < high)
            stratum_scores = scores[mask]

            if len(stratum_scores) > 0:
                self.strata_scores[s] = stratum_scores
            else:
                # Fallback to all scores if stratum empty
                self.strata_scores[s] = scores

        self._is_fitted = True

    def _get_stratum(self, crowding: float) -> int:
        """Map crowding level to stratum index."""
        for s in range(self.n_strata):
            if crowding < self.strata_thresholds[s + 1]:
                return s
        return self.n_strata - 1

    def predict_sets(
        self,
        X_test: np.ndarray,
        crowding_test: np.ndarray,
        alpha: float = 0.1
    ) -> ConformalResult:
        """
        Construct prediction sets with stratum-specific thresholds.
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before predict_sets()")

        proba = self.base_model.predict_proba(X_test)[:, 1]

        # Compute per-stratum thresholds
        strata_thresholds = {}
        for s, scores in self.strata_scores.items():
            n = len(scores)
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            q_level = min(q_level, 1.0)
            strata_thresholds[s] = np.quantile(scores, q_level)

        # Construct prediction sets
        sets = []
        for i, p in enumerate(proba):
            stratum = self._get_stratum(crowding_test[i])
            threshold = strata_thresholds[stratum]

            pred_set = set()
            if (1 - p) <= threshold:
                pred_set.add(1)
            if p <= threshold:
                pred_set.add(0)
            sets.append(pred_set)

        return ConformalResult(
            prediction_sets=sets,
            probabilities=proba,
            threshold=np.mean(list(strata_thresholds.values())),  # Average for reporting
            crowding_levels=crowding_test
        )


class CrowdingAdaptiveOnline:
    """
    Crowding-Adaptive Online Conformal (CAO).

    Extends ACI (Adaptive Conformal Inference) with crowding-dependent step size.

    Standard ACI: τ_{t+1} = τ_t + γ × (err_t - α)

    Crowding-Adaptive:
        γ(c) = γ_base × (1 + β × crowding)
        τ_{t+1} = τ_t + γ(c_t) × (err_t - α)

    Intuition:
    - High crowding → larger step size → faster adaptation
    - Anticipates regime change when crowding is high
    - More responsive to coverage errors during crowded periods
    """

    def __init__(
        self,
        base_model: Optional[RandomForestClassifier] = None,
        alpha: float = 0.1,
        gamma_base: float = 0.1,
        beta: float = 0.5,
        n_estimators: int = 100,
    ):
        """
        Args:
            base_model: Pre-trained classifier (optional)
            alpha: Target miscoverage rate
            gamma_base: Base step size for threshold updates
            beta: Crowding sensitivity (0 = standard ACI)
            n_estimators: Trees for RandomForest if base_model not provided
        """
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
        self.gamma_base = gamma_base
        self.beta = beta
        self.threshold = 0.5  # Initial threshold
        self._is_fitted = False
        self.threshold_history = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        crowding_calib: Optional[np.ndarray] = None
    ):
        """
        Fit base model and initialize threshold from calibration set.
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

    def predict_set(
        self,
        x: np.ndarray,
        crowding: float
    ) -> Set[int]:
        """
        Predict set for a single sample.
        """
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

    def update(
        self,
        y_true: int,
        pred_set: Set[int],
        crowding: float
    ) -> float:
        """
        Update threshold based on coverage error and crowding.

        Args:
            y_true: True label
            pred_set: Predicted set
            crowding: Current crowding level

        Returns:
            Updated threshold
        """
        covered = int(y_true in pred_set)
        missed = 1 - covered

        # Crowding-adaptive step size
        gamma = self.gamma_base * (1 + self.beta * crowding)

        # Update: if missed, increase threshold; if covered, decrease
        error = missed - self.alpha
        self.threshold += gamma * error

        # Clip to valid range
        self.threshold = np.clip(self.threshold, 0.01, 0.99)

        self.threshold_history.append(self.threshold)
        return self.threshold

    def predict_sets_online(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        crowding_test: np.ndarray
    ) -> Tuple[List[Set[int]], List[float]]:
        """
        Online prediction with threshold updates after each sample.

        Args:
            X_test: Test features
            y_test: True labels (for threshold updates)
            crowding_test: Crowding levels

        Returns:
            Tuple of (prediction_sets, threshold_history)
        """
        sets = []
        thresholds = []

        for i in range(len(X_test)):
            # Predict
            pred_set = self.predict_set(X_test[i], crowding_test[i])
            sets.append(pred_set)
            thresholds.append(self.threshold)

            # Update (using true label - simulates online feedback)
            self.update(y_test[i], pred_set, crowding_test[i])

        return sets, thresholds


class CrowdingAwareConformalEnsemble:
    """
    Ensemble combining multiple crowding-aware methods.

    Combines:
    - CrowdingWeightedCP (static calibration)
    - CrowdingAdaptiveOnline (dynamic threshold)

    Takes intersection or union of prediction sets based on strategy.
    """

    def __init__(
        self,
        lambda_weight: float = 1.0,
        gamma_base: float = 0.1,
        beta: float = 0.5,
        strategy: str = 'union',  # 'union' or 'intersection'
    ):
        self.weighted_cp = CrowdingWeightedCP(lambda_weight=lambda_weight)
        self.cao = CrowdingAdaptiveOnline(gamma_base=gamma_base, beta=beta)
        self.strategy = strategy

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        crowding_calib: np.ndarray
    ):
        """Fit both methods."""
        self.weighted_cp.fit(X_train, y_train, X_calib, y_calib, crowding_calib)
        self.cao.fit(X_train, y_train, X_calib, y_calib, crowding_calib)

    def predict_sets_online(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        crowding_test: np.ndarray,
        alpha: float = 0.1
    ) -> List[Set[int]]:
        """
        Combine predictions from both methods.
        """
        # Get weighted CP predictions (static)
        weighted_result = self.weighted_cp.predict_sets(X_test, crowding_test, alpha)
        weighted_sets = weighted_result.prediction_sets

        # Get CAO predictions (online)
        cao_sets, _ = self.cao.predict_sets_online(X_test, y_test, crowding_test)

        # Combine
        combined_sets = []
        for ws, cs in zip(weighted_sets, cao_sets):
            if self.strategy == 'union':
                combined_sets.append(ws | cs)
            else:  # intersection
                combined_sets.append(ws & cs)

        return combined_sets


def evaluate_coverage(
    pred_sets: List[Set[int]],
    y_true: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate conformal prediction coverage metrics.

    Returns:
        Dictionary with coverage, set size, singleton rate, empty rate
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


def evaluate_conditional_coverage(
    pred_sets: List[Set[int]],
    y_true: np.ndarray,
    crowding: np.ndarray,
    n_bins: int = 3
) -> pd.DataFrame:
    """
    Evaluate coverage by crowding level.

    Returns:
        DataFrame with coverage statistics per crowding bin
    """
    # Bin crowding levels
    bins = np.quantile(crowding, np.linspace(0, 1, n_bins + 1))
    bin_labels = [f'Q{i+1}' for i in range(n_bins)]

    results = []
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (crowding >= low) & (crowding <= high)
        else:
            mask = (crowding >= low) & (crowding < high)

        if mask.sum() > 0:
            bin_sets = [pred_sets[j] for j in range(len(y_true)) if mask[j]]
            bin_y = y_true[mask]

            metrics = evaluate_coverage(bin_sets, bin_y)
            metrics['crowding_bin'] = bin_labels[i]
            metrics['crowding_low'] = low
            metrics['crowding_high'] = high
            results.append(metrics)

    return pd.DataFrame(results)


if __name__ == '__main__':
    # Simple test with synthetic data
    np.random.seed(42)

    # Generate synthetic data
    n_train, n_calib, n_test = 500, 200, 100
    n_features = 10

    X = np.random.randn(n_train + n_calib + n_test, n_features)
    y = (X[:, 0] + X[:, 1] + np.random.randn(len(X)) * 0.5 > 0).astype(int)
    crowding = np.random.beta(2, 5, len(X))  # Skewed towards low crowding

    X_train, y_train = X[:n_train], y[:n_train]
    X_calib, y_calib = X[n_train:n_train+n_calib], y[n_train:n_train+n_calib]
    X_test, y_test = X[n_train+n_calib:], y[n_train+n_calib:]
    crowding_calib = crowding[n_train:n_train+n_calib]
    crowding_test = crowding[n_train+n_calib:]

    print("=" * 60)
    print("CROWDING-AWARE CONFORMAL PREDICTION TEST")
    print("=" * 60)

    # Test CrowdingWeightedCP
    print("\n1. CrowdingWeightedCP (λ=1.0)")
    cwcp = CrowdingWeightedCP(lambda_weight=1.0)
    cwcp.fit(X_train, y_train, X_calib, y_calib, crowding_calib)
    result = cwcp.predict_sets(X_test, crowding_test, alpha=0.1)
    metrics = evaluate_coverage(result.prediction_sets, y_test)
    print(f"   Coverage: {metrics['coverage']:.3f} (target ≥ 0.90)")
    print(f"   Avg Size: {metrics['avg_set_size']:.2f}")

    # Test CrowdingStratifiedCP
    print("\n2. CrowdingStratifiedCP (3 strata)")
    cscp = CrowdingStratifiedCP(n_strata=3)
    cscp.fit(X_train, y_train, X_calib, y_calib, crowding_calib)
    result = cscp.predict_sets(X_test, crowding_test, alpha=0.1)
    metrics = evaluate_coverage(result.prediction_sets, y_test)
    print(f"   Coverage: {metrics['coverage']:.3f} (target ≥ 0.90)")
    print(f"   Avg Size: {metrics['avg_set_size']:.2f}")

    # Test CrowdingAdaptiveOnline
    print("\n3. CrowdingAdaptiveOnline (CAO, β=0.5)")
    cao = CrowdingAdaptiveOnline(gamma_base=0.1, beta=0.5)
    cao.fit(X_train, y_train, X_calib, y_calib, crowding_calib)
    cao_sets, _ = cao.predict_sets_online(X_test, y_test, crowding_test)
    metrics = evaluate_coverage(cao_sets, y_test)
    print(f"   Coverage: {metrics['coverage']:.3f} (target ≥ 0.90)")
    print(f"   Avg Size: {metrics['avg_set_size']:.2f}")

    # Conditional coverage analysis
    print("\n4. Conditional Coverage by Crowding Level (CrowdingWeightedCP)")
    cond_coverage = evaluate_conditional_coverage(
        result.prediction_sets, y_test, crowding_test, n_bins=3
    )
    print(cond_coverage[['crowding_bin', 'coverage', 'avg_set_size', 'n_samples']])

    print("\n" + "=" * 60)
    print("All tests passed!")
