"""
Advanced Conformal Prediction Methods for Financial Time Series

Addresses exchangeability violation in financial data:
1. ACI (Adaptive Conformal Inference) - Online threshold updates
2. Adaptive Conformal - Exponentially weighted calibration
3. Regime-Specific (Mondrian) - Conditional coverage by market regime

For ICML 2026 submission.

References:
- Gibbs & Candes (2021) "Adaptive Conformal Inference Under Distribution Shift"
- Vovk et al. (2003) "Mondrian Conformal Prediction"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from crowding_ml import RandomForestModel, BaseMLModel


# ================================================================
# 1. ACI (ADAPTIVE CONFORMAL INFERENCE)
# ================================================================

class ACIConformalClassifier:
    """
    Adaptive Conformal Inference (Gibbs & Candes 2021).

    Online threshold update with O(1) per step:
        q_{t+1} = q_t + γ × (coverage_t - (1-α))

    Handles distribution shift without recomputing quantiles.
    Works with delayed label feedback (e.g., 1-month delay).
    """

    def __init__(
        self,
        base_model: Optional[BaseMLModel] = None,
        alpha: float = 0.1,
        gamma: float = 0.1,
        initial_threshold: float = 0.5
    ):
        """
        Args:
            base_model: Base classifier with predict_proba()
            alpha: Target miscoverage (0.1 = 90% coverage)
            gamma: Step size for threshold updates (0.05-0.2 recommended)
            initial_threshold: Starting threshold value
        """
        self.base_model = base_model or RandomForestModel()
        self.alpha = alpha
        self.gamma = gamma
        self.threshold = initial_threshold

        # Tracking
        self.threshold_history = [initial_threshold]
        self.coverage_history = []
        self.step_count = 0

    def fit_base_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit base model on training data."""
        self.base_model.fit(X_train, y_train)

    def predict_sets(self, X: np.ndarray) -> List[Set[int]]:
        """
        Generate prediction sets using current threshold.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            List of prediction sets
        """
        proba = self.base_model.predict_proba(X)

        sets = []
        for p in proba:
            pred_set = set()
            # Include crash if nonconformity <= threshold
            if (1 - p) <= self.threshold:
                pred_set.add(1)
            # Include no-crash if nonconformity <= threshold
            if p <= self.threshold:
                pred_set.add(0)
            sets.append(pred_set)

        return sets

    def update(self, y_true: int, pred_set: Set[int]) -> float:
        """
        Update threshold based on coverage feedback (O(1)).

        Called when true label is revealed (possibly with delay).

        Gibbs & Candes (2021) update rule:
        - If we MISS (y not in set): increase threshold to make sets larger
        - If we COVER (y in set): decrease threshold to make sets smaller

        Formula: q_{t+1} = q_t - γ × (err_t - α)
        where err_t = 1 if missed, 0 if covered

        Args:
            y_true: True label (0 or 1)
            pred_set: Prediction set that was made

        Returns:
            New threshold value
        """
        # Check if covered
        covered = int(y_true in pred_set)
        missed = 1 - covered
        self.coverage_history.append(covered)

        # ACI update:
        # - If missed (err=1), threshold increases: q += γ*(1 - α) > 0
        # - If covered (err=0), threshold decreases: q += γ*(0 - α) < 0
        error = missed - self.alpha
        self.threshold += self.gamma * error

        # Clip to valid range
        self.threshold = np.clip(self.threshold, 0.01, 0.99)

        self.threshold_history.append(self.threshold)
        self.step_count += 1

        return self.threshold

    def get_diagnostics(self) -> Dict:
        """Get diagnostic metrics."""
        if not self.coverage_history:
            return {}

        return {
            'steps': self.step_count,
            'current_threshold': self.threshold,
            'empirical_coverage': np.mean(self.coverage_history),
            'target_coverage': 1 - self.alpha,
            'coverage_gap': np.mean(self.coverage_history) - (1 - self.alpha),
            'threshold_std': np.std(self.threshold_history),
            'last_12_coverage': np.mean(self.coverage_history[-12:]) if len(self.coverage_history) >= 12 else None
        }


# ================================================================
# 2. ADAPTIVE CONFORMAL (EXPONENTIAL WEIGHTING)
# ================================================================

class AdaptiveConformalClassifier:
    """
    Adaptive Conformal with exponentially weighted calibration scores.

    Recent observations weighted more heavily to adapt to distribution shift.

    Weight formula: w_i = exp(-decay × (n - i))
    """

    def __init__(
        self,
        base_model: Optional[BaseMLModel] = None,
        alpha: float = 0.1,
        decay: float = 0.02,  # ~35 month half-life
        min_calib_size: int = 20
    ):
        """
        Args:
            base_model: Base classifier
            alpha: Target miscoverage
            decay: Exponential decay rate (higher = faster adaptation)
            min_calib_size: Minimum calibration samples required
        """
        self.base_model = base_model or RandomForestModel()
        self.alpha = alpha
        self.decay = decay
        self.min_calib_size = min_calib_size

        # Rolling calibration storage
        self.calib_scores = []
        self.calib_dates = []
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray
    ):
        """Fit base model and initialize calibration scores."""
        self.base_model.fit(X_train, y_train)
        self._add_calibration_scores(X_calib, y_calib)
        self.is_fitted = True

    def _add_calibration_scores(self, X: np.ndarray, y: np.ndarray):
        """Compute and store nonconformity scores."""
        proba = self.base_model.predict_proba(X)
        scores = np.where(y == 1, 1 - proba, proba)
        self.calib_scores.extend(scores.tolist())

    def update_calibration(self, X_new: np.ndarray, y_new: np.ndarray):
        """Add new calibration samples (online update)."""
        self._add_calibration_scores(X_new, y_new)

    def _get_weighted_threshold(self) -> float:
        """Compute threshold using exponentially weighted quantile."""
        n = len(self.calib_scores)
        if n < self.min_calib_size:
            return 0.5  # Fallback

        scores = np.array(self.calib_scores)

        # Exponential weights: recent samples weighted more
        weights = np.exp(-self.decay * np.arange(n - 1, -1, -1))
        weights = weights / weights.sum()

        # Sort scores and weights together
        sorted_idx = np.argsort(scores)
        sorted_scores = scores[sorted_idx]
        sorted_weights = weights[sorted_idx]

        # Cumulative weights
        cum_weights = np.cumsum(sorted_weights)

        # Find weighted quantile
        q_level = 1 - self.alpha
        idx = np.searchsorted(cum_weights, q_level)
        idx = min(idx, len(sorted_scores) - 1)

        return sorted_scores[idx]

    def predict_sets(self, X: np.ndarray, alpha: Optional[float] = None) -> List[Set[int]]:
        """Generate prediction sets using weighted threshold."""
        if not self.is_fitted:
            raise ValueError("Must call fit() first")

        alpha = alpha or self.alpha
        threshold = self._get_weighted_threshold()

        proba = self.base_model.predict_proba(X)

        sets = []
        for p in proba:
            pred_set = set()
            if (1 - p) <= threshold:
                pred_set.add(1)
            if p <= threshold:
                pred_set.add(0)
            sets.append(pred_set)

        return sets

    def get_threshold(self) -> float:
        """Get current weighted threshold."""
        return self._get_weighted_threshold()


# ================================================================
# 3. REGIME-SPECIFIC (MONDRIAN) CONFORMAL
# ================================================================

class RegimeDetector:
    """
    Detect market regimes for Mondrian conformal prediction.

    Supports:
    - Volatility regimes (rolling vol percentile)
    - Return regimes (bull/bear)
    - Correlation regimes
    """

    def __init__(self, method: str = 'volatility', window: int = 12):
        """
        Args:
            method: 'volatility', 'return', or 'correlation'
            window: Rolling window for regime calculation
        """
        self.method = method
        self.window = window
        self.thresholds = {}

    def fit(self, returns: pd.Series):
        """Fit regime thresholds on historical data."""
        if self.method == 'volatility':
            vol = returns.rolling(self.window).std()
            self.thresholds['low'] = vol.quantile(0.33)
            self.thresholds['high'] = vol.quantile(0.67)
        elif self.method == 'return':
            roll_ret = returns.rolling(self.window).mean()
            self.thresholds['low'] = roll_ret.quantile(0.33)
            self.thresholds['high'] = roll_ret.quantile(0.67)

    def predict(self, returns: pd.Series) -> pd.Series:
        """Classify each observation into a regime."""
        if self.method == 'volatility':
            vol = returns.rolling(self.window).std()
            regimes = pd.cut(
                vol,
                bins=[-np.inf, self.thresholds['low'], self.thresholds['high'], np.inf],
                labels=['Low_Vol', 'Normal_Vol', 'High_Vol']
            )
        elif self.method == 'return':
            roll_ret = returns.rolling(self.window).mean()
            regimes = pd.cut(
                roll_ret,
                bins=[-np.inf, self.thresholds['low'], self.thresholds['high'], np.inf],
                labels=['Bear', 'Normal', 'Bull']
            )
        else:
            regimes = pd.Series(['Normal'] * len(returns), index=returns.index)

        return regimes


class MondrianConformalClassifier:
    """
    Mondrian Conformal Prediction with regime-specific calibration.

    Achieves conditional coverage guarantee per regime:
        P(Y ∈ C(X) | regime = g) ≥ 1 - α  for all g
    """

    def __init__(
        self,
        base_model: Optional[BaseMLModel] = None,
        regime_detector: Optional[RegimeDetector] = None,
        alpha: float = 0.1
    ):
        """
        Args:
            base_model: Base classifier
            regime_detector: RegimeDetector instance
            alpha: Target miscoverage (per regime)
        """
        self.base_model = base_model or RandomForestModel()
        self.regime_detector = regime_detector or RegimeDetector()
        self.alpha = alpha

        # Per-regime calibration
        self.regime_scores = {}  # {regime: [scores]}
        self.regime_thresholds = {}  # {regime: threshold}
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        regimes_calib: np.ndarray
    ):
        """
        Fit base model and compute regime-specific thresholds.

        Args:
            X_train: Training features
            y_train: Training labels
            X_calib: Calibration features
            y_calib: Calibration labels
            regimes_calib: Regime labels for calibration data
        """
        # Fit base model
        self.base_model.fit(X_train, y_train)

        # Compute nonconformity scores
        proba = self.base_model.predict_proba(X_calib)
        scores = np.where(y_calib == 1, 1 - proba, proba)

        # Group scores by regime
        unique_regimes = np.unique(regimes_calib)
        for regime in unique_regimes:
            mask = regimes_calib == regime
            regime_scores = scores[mask]

            if len(regime_scores) < 5:
                print(f"Warning: Only {len(regime_scores)} samples in regime {regime}")
                continue

            self.regime_scores[regime] = regime_scores

            # Compute regime-specific threshold
            n = len(regime_scores)
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            q_level = min(q_level, 1.0)
            self.regime_thresholds[regime] = np.quantile(regime_scores, q_level)

        # Global fallback threshold
        self.global_threshold = np.quantile(scores, 1 - self.alpha)

        self.is_fitted = True

    def predict_sets(
        self,
        X: np.ndarray,
        regimes: np.ndarray
    ) -> List[Set[int]]:
        """
        Generate prediction sets using regime-specific thresholds.

        Args:
            X: Test features
            regimes: Regime labels for test data

        Returns:
            List of prediction sets
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() first")

        proba = self.base_model.predict_proba(X)

        sets = []
        for i, p in enumerate(proba):
            regime = regimes[i]

            # Get regime-specific threshold (or global fallback)
            threshold = self.regime_thresholds.get(regime, self.global_threshold)

            pred_set = set()
            if (1 - p) <= threshold:
                pred_set.add(1)
            if p <= threshold:
                pred_set.add(0)
            sets.append(pred_set)

        return sets

    def evaluate_by_regime(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        regimes_test: np.ndarray
    ) -> Dict:
        """Evaluate coverage per regime."""
        pred_sets = self.predict_sets(X_test, regimes_test)

        results = {'overall': {}, 'by_regime': {}}

        # Overall coverage
        overall_coverage = np.mean([
            int(y_test[i]) in pred_sets[i]
            for i in range(len(y_test))
        ])
        results['overall'] = {
            'coverage': overall_coverage,
            'target': 1 - self.alpha,
            'gap': overall_coverage - (1 - self.alpha),
            'avg_set_size': np.mean([len(s) for s in pred_sets])
        }

        # Per-regime coverage
        for regime in np.unique(regimes_test):
            mask = regimes_test == regime
            regime_sets = [pred_sets[i] for i in np.where(mask)[0]]
            regime_y = y_test[mask]

            if len(regime_y) == 0:
                continue

            coverage = np.mean([
                int(regime_y[i]) in regime_sets[i]
                for i in range(len(regime_y))
            ])

            results['by_regime'][regime] = {
                'coverage': coverage,
                'target': 1 - self.alpha,
                'gap': coverage - (1 - self.alpha),
                'n_samples': len(regime_y),
                'threshold': self.regime_thresholds.get(regime, self.global_threshold)
            }

        return results


# ================================================================
# WALK-FORWARD BACKTEST FOR ALL METHODS
# ================================================================

class AdvancedConformalBacktest:
    """
    Walk-forward backtesting for comparing conformal methods.

    Methods:
    - split: Standard split conformal (baseline)
    - aci: Adaptive Conformal Inference
    - adaptive: Exponentially weighted
    - mondrian: Regime-specific
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

    def run_all_methods(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        returns: pd.Series,  # For regime detection
        verbose: bool = True
    ) -> Dict:
        """
        Run walk-forward backtest for all conformal methods.

        Returns:
            Dict with results for each method
        """
        n = len(X)
        total_size = self.fit_size + self.calib_size + self.test_size

        # Results storage
        results = {
            'split': {'sets': [], 'actuals': [], 'dates': []},
            'aci': {'sets': [], 'actuals': [], 'dates': [], 'thresholds': []},
            'adaptive': {'sets': [], 'actuals': [], 'dates': []},
            'mondrian': {'sets': [], 'actuals': [], 'dates': [], 'regimes': []}
        }

        # Regime detector for Mondrian
        regime_detector = RegimeDetector(method='volatility', window=12)
        regime_detector.fit(returns)

        # ACI instance (persists across splits)
        aci = ACIConformalClassifier(alpha=self.alpha, gamma=0.1)

        start = 0
        split_num = 0

        while start + total_size <= n:
            # Split boundaries
            fit_end = start + self.fit_size
            calib_end = fit_end + self.calib_size
            test_end = calib_end + self.test_size

            # Data splits
            X_fit = X.iloc[start:fit_end].values
            y_fit = y.iloc[start:fit_end].values

            X_calib = X.iloc[fit_end:calib_end].values
            y_calib = y.iloc[fit_end:calib_end].values

            X_test = X.iloc[calib_end:test_end].values
            y_test = y.iloc[calib_end:test_end].values

            test_dates = X.index[calib_end:test_end]
            test_returns = returns.iloc[calib_end:test_end]

            # Regimes for Mondrian
            calib_returns = returns.iloc[fit_end:calib_end]
            regimes_calib = regime_detector.predict(calib_returns).values
            regimes_test = regime_detector.predict(test_returns).values

            # 1. Split Conformal (baseline)
            from conformal_ml import ConformalClassifier
            split_cf = ConformalClassifier()
            split_cf.fit(X_fit, y_fit, X_calib, y_calib)
            split_sets = split_cf.predict_sets(X_test, self.alpha)

            results['split']['sets'].extend(split_sets)
            results['split']['actuals'].extend(y_test)
            results['split']['dates'].extend(test_dates)

            # 2. ACI (online - update threshold with delayed feedback)
            if split_num == 0:
                aci.fit_base_model(np.vstack([X_fit, X_calib]), np.concatenate([y_fit, y_calib]))

            aci_sets = aci.predict_sets(X_test)
            results['aci']['sets'].extend(aci_sets)
            results['aci']['actuals'].extend(y_test)
            results['aci']['dates'].extend(test_dates)

            # Update ACI with test labels (simulating delayed feedback)
            for i in range(len(y_test)):
                aci.update(y_test[i], aci_sets[i])
            results['aci']['thresholds'].append(aci.threshold)

            # 3. Adaptive Conformal
            adaptive_cf = AdaptiveConformalClassifier(alpha=self.alpha, decay=0.02)
            adaptive_cf.fit(X_fit, y_fit, X_calib, y_calib)
            adaptive_sets = adaptive_cf.predict_sets(X_test)

            results['adaptive']['sets'].extend(adaptive_sets)
            results['adaptive']['actuals'].extend(y_test)
            results['adaptive']['dates'].extend(test_dates)

            # 4. Mondrian Conformal
            # Handle NaN regimes
            valid_calib = ~pd.isna(regimes_calib)
            valid_test = ~pd.isna(regimes_test)

            if valid_calib.sum() >= 10 and valid_test.sum() >= 1:
                mondrian_cf = MondrianConformalClassifier(alpha=self.alpha)
                mondrian_cf.fit(
                    X_fit, y_fit,
                    X_calib[valid_calib], y_calib[valid_calib],
                    regimes_calib[valid_calib]
                )
                mondrian_sets = mondrian_cf.predict_sets(X_test[valid_test], regimes_test[valid_test])

                # Pad with empty sets for invalid regime indices
                full_mondrian_sets = []
                valid_idx = 0
                for i in range(len(y_test)):
                    if valid_test[i]:
                        full_mondrian_sets.append(mondrian_sets[valid_idx])
                        valid_idx += 1
                    else:
                        full_mondrian_sets.append({0, 1})  # Uncertain

                results['mondrian']['sets'].extend(full_mondrian_sets)
            else:
                results['mondrian']['sets'].extend([{0, 1}] * len(y_test))

            results['mondrian']['actuals'].extend(y_test)
            results['mondrian']['dates'].extend(test_dates)
            results['mondrian']['regimes'].extend(regimes_test)

            if verbose and split_num % 5 == 0:
                print(f"Split {split_num}: processed")

            start += self.step_size
            split_num += 1

        # Compute metrics for each method
        metrics = {}
        for method in ['split', 'aci', 'adaptive', 'mondrian']:
            actuals = np.array(results[method]['actuals'])
            sets = results[method]['sets']

            coverage = np.mean([int(actuals[i]) in sets[i] for i in range(len(actuals))])
            avg_size = np.mean([len(s) for s in sets])
            singleton_rate = np.mean([len(s) == 1 for s in sets])

            metrics[method] = {
                'coverage': coverage,
                'target': 1 - self.alpha,
                'gap': coverage - (1 - self.alpha),
                'avg_set_size': avg_size,
                'singleton_rate': singleton_rate,
                'n_samples': len(actuals)
            }

        results['metrics'] = metrics

        if verbose:
            print("\n" + "=" * 60)
            print("RESULTS COMPARISON")
            print("=" * 60)
            print(f"\n{'Method':<12} {'Coverage':<10} {'Target':<10} {'Gap':<10} {'AvgSize':<10}")
            print("-" * 52)
            for method, m in metrics.items():
                gap_str = f"+{m['gap']:.3f}" if m['gap'] >= 0 else f"{m['gap']:.3f}"
                status = "✓" if m['coverage'] >= m['target'] else "✗"
                print(f"{method:<12} {m['coverage']:.3f}      {m['target']:.3f}      {gap_str:<10} {m['avg_set_size']:.2f}  {status}")

        return results


# ================================================================
# EXAMPLE USAGE
# ================================================================

if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from features import FeatureEngineer

    DATA_DIR = Path(__file__).parent.parent / 'data'

    print("=" * 60)
    print("ADVANCED CONFORMAL PREDICTION COMPARISON")
    print("=" * 60)

    # Load data
    factors = pd.read_parquet(DATA_DIR / 'ff_extended_factors.parquet')
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    factors = factors.drop(columns=['RF'], errors='ignore')

    print(f"\nData: {factors.index.min().strftime('%Y-%m')} to {factors.index.max().strftime('%Y-%m')}")

    # Test on Momentum factor
    factor = 'Mom'
    print(f"\nFactor: {factor}")

    # Generate features
    fe = FeatureEngineer()
    features = fe.generate_all_features(factors)
    X, y = fe.create_ml_dataset(features, factors, target_type='crash', factor=factor)

    # Get returns for regime detection
    returns = factors[factor]

    # Run backtest
    backtest = AdvancedConformalBacktest(
        fit_size=90,
        calib_size=30,
        test_size=12,
        alpha=0.1
    )

    results = backtest.run_all_methods(X, y, returns, verbose=True)

    # ACI threshold evolution
    if results['aci']['thresholds']:
        print(f"\nACI Threshold Evolution:")
        print(f"  Start: {results['aci']['thresholds'][0]:.3f}")
        print(f"  End:   {results['aci']['thresholds'][-1]:.3f}")
        print(f"  Std:   {np.std(results['aci']['thresholds']):.3f}")
