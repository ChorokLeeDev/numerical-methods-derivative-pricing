"""
Conformal Prediction for Factor Crash Prediction

Provides distribution-free uncertainty quantification with coverage guarantees.

Key classes:
- ConformalClassifier: Split conformal wrapper for binary classification
- ConformalWalkForwardBacktest: Walk-forward validation with conformal prediction

For ICML submission: Distribution-free uncertainty for financial tail risk.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score

from crowding_ml import RandomForestModel, BaseMLModel


@dataclass
class ConformalMetrics:
    """Metrics for evaluating conformal prediction performance."""
    coverage: float              # Empirical coverage (fraction in prediction set)
    target_coverage: float       # Target coverage (1 - alpha)
    avg_set_size: float          # Average prediction set size
    singleton_rate: float        # Fraction of singleton sets (confident)
    empty_rate: float            # Fraction of empty sets (should be ~0)
    crash_set_rate: float        # Fraction where crash (1) is in set
    base_auc: float              # Base model AUC for reference


class ConformalClassifier:
    """
    Split conformal prediction wrapper for binary classification.

    Provides prediction sets with coverage guarantee:
    P(true_label ∈ prediction_set) ≥ 1 - α

    Usage:
        cf = ConformalClassifier(RandomForestModel())
        cf.fit(X_train, y_train, X_calib, y_calib)
        sets = cf.predict_sets(X_test, alpha=0.1)  # 90% coverage
    """

    def __init__(self, base_model: Optional[BaseMLModel] = None):
        """
        Initialize conformal classifier.

        Args:
            base_model: Base classifier with fit() and predict_proba() methods.
                       If None, uses RandomForestModel by default.
        """
        self.base_model = base_model or RandomForestModel()
        self.nonconformity_scores = None
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray
    ) -> 'ConformalClassifier':
        """
        Fit base model and compute calibration scores.

        Args:
            X_train: Training features for base model
            y_train: Training labels for base model
            X_calib: Calibration features for conformal scores
            y_calib: Calibration labels for conformal scores

        Returns:
            self
        """
        # Step 1: Fit base model on training data
        self.base_model.fit(X_train, y_train)

        # Step 2: Compute nonconformity scores on calibration data
        self.nonconformity_scores = self._compute_nonconformity_scores(
            X_calib, y_calib
        )

        self.is_fitted = True
        return self

    def fit_prefit(
        self,
        X_calib: np.ndarray,
        y_calib: np.ndarray
    ) -> 'ConformalClassifier':
        """
        Calibrate conformal scores using pre-fitted base model.

        Args:
            X_calib: Calibration features
            y_calib: Calibration labels

        Returns:
            self
        """
        self.nonconformity_scores = self._compute_nonconformity_scores(
            X_calib, y_calib
        )
        self.is_fitted = True
        return self

    def _compute_nonconformity_scores(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Compute nonconformity scores on calibration set.

        Score = 1 - P(true class)
        - High score = model gave low probability to true class (nonconforming)
        - Low score = model confident in true class (conforming)
        """
        # Get probability of crash (class 1)
        proba = self.base_model.predict_proba(X)

        # Nonconformity score = 1 - P(true class)
        scores = np.where(
            y == 1,
            1 - proba,    # For crash: 1 - P(crash)
            proba         # For no-crash: P(crash) = 1 - P(no-crash)
        )

        return scores

    def _get_threshold(self, alpha: float) -> float:
        """
        Compute conformal threshold for given miscoverage level.

        Uses finite-sample correction: q = ceil((n+1)(1-α)) / n
        """
        n = len(self.nonconformity_scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(q_level, 1.0)  # Cap at 1.0

        return np.quantile(self.nonconformity_scores, q_level)

    def predict_sets(
        self,
        X: np.ndarray,
        alpha: float = 0.1
    ) -> List[Set[int]]:
        """
        Predict sets with coverage guarantee ≥ 1 - α.

        Args:
            X: Test features
            alpha: Miscoverage level (0.1 = 90% coverage target)

        Returns:
            List of prediction sets: {0}, {1}, {0,1}, or {}
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before predict_sets()")

        # Get base model probabilities
        proba = self.base_model.predict_proba(X)

        # Get conformal threshold
        threshold = self._get_threshold(alpha)

        # Construct prediction sets
        sets = []
        for p in proba:
            pred_set = set()

            # Check if crash (class 1) is conforming
            # Nonconformity for class 1 = 1 - p
            if (1 - p) <= threshold:
                pred_set.add(1)

            # Check if no-crash (class 0) is conforming
            # Nonconformity for class 0 = p
            if p <= threshold:
                pred_set.add(0)

            sets.append(pred_set)

        return sets

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return base model probabilities (for reference)."""
        return self.base_model.predict_proba(X)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        alpha: float = 0.1
    ) -> ConformalMetrics:
        """
        Evaluate conformal prediction performance.

        Args:
            X_test: Test features
            y_test: True labels
            alpha: Miscoverage level

        Returns:
            ConformalMetrics with coverage, set size, etc.
        """
        # Get prediction sets
        pred_sets = self.predict_sets(X_test, alpha)

        # Coverage: fraction of true labels in prediction sets
        coverage = np.mean([
            int(y_test[i]) in pred_sets[i]
            for i in range(len(y_test))
        ])

        # Set sizes
        set_sizes = [len(s) for s in pred_sets]
        avg_set_size = np.mean(set_sizes)

        # Singleton rate (confident predictions)
        singleton_rate = np.mean([len(s) == 1 for s in pred_sets])

        # Empty rate (should be ~0)
        empty_rate = np.mean([len(s) == 0 for s in pred_sets])

        # Crash in set rate
        crash_set_rate = np.mean([1 in s for s in pred_sets])

        # Base model AUC
        proba = self.base_model.predict_proba(X_test)
        base_auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else 0.5

        return ConformalMetrics(
            coverage=coverage,
            target_coverage=1 - alpha,
            avg_set_size=avg_set_size,
            singleton_rate=singleton_rate,
            empty_rate=empty_rate,
            crash_set_rate=crash_set_rate,
            base_auc=base_auc
        )


class ConformalWalkForwardBacktest:
    """
    Walk-forward backtesting with conformal prediction.

    Split structure: [Fit] → [Calibrate] → [Test]

    No lookahead bias - always train/calibrate on past, test on future.
    """

    def __init__(
        self,
        fit_size: int = 90,      # 7.5 years for fitting
        calib_size: int = 30,    # 2.5 years for calibration
        test_size: int = 12,     # 1 year test
        step_size: int = 12,     # Annual refit
        alpha: float = 0.1       # 90% coverage target
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
        base_model: Optional[BaseMLModel] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Run walk-forward backtest with conformal prediction.

        Args:
            X: Feature DataFrame (index=date)
            y: Target Series (index=date)
            base_model: Base classifier (default: RandomForestModel)
            verbose: Print progress

        Returns:
            Dict with predictions, sets, and metrics
        """
        n = len(X)
        total_size = self.fit_size + self.calib_size + self.test_size

        # Results storage
        all_sets = []
        all_actuals = []
        all_probas = []
        all_dates = []
        period_metrics = []

        start = 0
        split_num = 0

        while start + total_size <= n:
            # Define split boundaries
            fit_end = start + self.fit_size
            calib_end = fit_end + self.calib_size
            test_end = calib_end + self.test_size

            # Split data
            X_fit = X.iloc[start:fit_end].values
            y_fit = y.iloc[start:fit_end].values

            X_calib = X.iloc[fit_end:calib_end].values
            y_calib = y.iloc[fit_end:calib_end].values

            X_test = X.iloc[calib_end:test_end].values
            y_test = y.iloc[calib_end:test_end].values

            test_dates = X.index[calib_end:test_end]

            # Fit conformal classifier
            cf = ConformalClassifier(base_model or RandomForestModel())
            cf.fit(X_fit, y_fit, X_calib, y_calib)

            # Predict and evaluate
            pred_sets = cf.predict_sets(X_test, self.alpha)
            probas = cf.predict_proba(X_test)
            metrics = cf.evaluate(X_test, y_test, self.alpha)

            # Store results
            all_sets.extend(pred_sets)
            all_actuals.extend(y_test)
            all_probas.extend(probas)
            all_dates.extend(test_dates)

            period_metrics.append({
                'split': split_num,
                'test_start': test_dates[0],
                'test_end': test_dates[-1],
                **vars(metrics)
            })

            if verbose and split_num % 5 == 0:
                print(f"Split {split_num}: Coverage={metrics.coverage:.3f}, "
                      f"Avg Size={metrics.avg_set_size:.2f}, AUC={metrics.base_auc:.3f}")

            start += self.step_size
            split_num += 1

        # Compute aggregate metrics
        all_actuals = np.array(all_actuals)
        all_probas = np.array(all_probas)

        aggregate_coverage = np.mean([
            int(all_actuals[i]) in all_sets[i]
            for i in range(len(all_actuals))
        ])

        aggregate_metrics = ConformalMetrics(
            coverage=aggregate_coverage,
            target_coverage=1 - self.alpha,
            avg_set_size=np.mean([len(s) for s in all_sets]),
            singleton_rate=np.mean([len(s) == 1 for s in all_sets]),
            empty_rate=np.mean([len(s) == 0 for s in all_sets]),
            crash_set_rate=np.mean([1 in s for s in all_sets]),
            base_auc=roc_auc_score(all_actuals, all_probas) if len(np.unique(all_actuals)) > 1 else 0.5
        )

        if verbose:
            print(f"\n{'='*50}")
            print(f"AGGREGATE RESULTS (α={self.alpha})")
            print(f"{'='*50}")
            print(f"Coverage:      {aggregate_metrics.coverage:.3f} (target ≥ {1-self.alpha:.3f})")
            print(f"Avg Set Size:  {aggregate_metrics.avg_set_size:.2f}")
            print(f"Singleton %:   {aggregate_metrics.singleton_rate:.1%}")
            print(f"Empty %:       {aggregate_metrics.empty_rate:.1%}")
            print(f"Base AUC:      {aggregate_metrics.base_auc:.3f}")

        return {
            'sets': all_sets,
            'actuals': all_actuals,
            'probas': all_probas,
            'dates': all_dates,
            'period_metrics': pd.DataFrame(period_metrics),
            'aggregate_metrics': aggregate_metrics,
            'n_splits': split_num
        }


def evaluate_conformal_by_factor(
    factor_returns: pd.DataFrame,
    factor_list: Optional[List[str]] = None,
    fit_size: int = 90,
    calib_size: int = 30,
    test_size: int = 12,
    alpha: float = 0.1,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluate conformal prediction across multiple factors.

    Args:
        factor_returns: DataFrame of factor returns
        factor_list: List of factors to evaluate (default: all)
        fit_size: Months for fitting base model
        calib_size: Months for calibration
        test_size: Months for testing
        alpha: Miscoverage level
        verbose: Print progress

    Returns:
        DataFrame with per-factor conformal metrics
    """
    from features import FeatureEngineer

    if factor_list is None:
        factor_list = factor_returns.columns.tolist()

    results = []

    for factor in factor_list:
        if verbose:
            print(f"\n{'='*50}")
            print(f"FACTOR: {factor}")
            print(f"{'='*50}")

        # Generate features
        fe = FeatureEngineer()
        features = fe.generate_all_features(factor_returns)
        X, y = fe.create_ml_dataset(features, factor_returns, target_type='crash', factor=factor)

        # Run walk-forward conformal backtest
        backtest = ConformalWalkForwardBacktest(
            fit_size=fit_size,
            calib_size=calib_size,
            test_size=test_size,
            alpha=alpha
        )

        result = backtest.run(X, y, verbose=verbose)
        metrics = result['aggregate_metrics']

        results.append({
            'factor': factor,
            'coverage': metrics.coverage,
            'target_coverage': metrics.target_coverage,
            'coverage_gap': metrics.coverage - metrics.target_coverage,
            'avg_set_size': metrics.avg_set_size,
            'singleton_rate': metrics.singleton_rate,
            'empty_rate': metrics.empty_rate,
            'crash_set_rate': metrics.crash_set_rate,
            'base_auc': metrics.base_auc,
            'n_splits': result['n_splits'],
            'n_samples': len(result['actuals'])
        })

    return pd.DataFrame(results)


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
    print("CONFORMAL PREDICTION FOR FACTOR CRASH PREDICTION")
    print("=" * 60)

    # Load data
    factors = pd.read_parquet(DATA_DIR / 'ff_extended_factors.parquet')
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    factors = factors.drop(columns=['RF'], errors='ignore')

    print(f"\nData: {factors.index.min().strftime('%Y-%m')} to {factors.index.max().strftime('%Y-%m')}")
    print(f"Factors: {list(factors.columns)}")

    # Test on Momentum factor
    print("\n" + "=" * 60)
    print("MOMENTUM CRASH PREDICTION WITH CONFORMAL")
    print("=" * 60)

    fe = FeatureEngineer()
    features = fe.generate_all_features(factors)
    X, y = fe.create_ml_dataset(features, factors, target_type='crash', factor='Mom')

    # Run conformal backtest
    backtest = ConformalWalkForwardBacktest(
        fit_size=90,
        calib_size=30,
        test_size=12,
        alpha=0.1
    )

    result = backtest.run(X, y, verbose=True)

    # Print sample prediction sets
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTION SETS (last 20)")
    print("=" * 60)

    for i in range(-20, 0):
        date = result['dates'][i]
        pred_set = result['sets'][i]
        actual = int(result['actuals'][i])
        proba = result['probas'][i]
        in_set = "✓" if actual in pred_set else "✗"

        set_str = str(pred_set) if pred_set else "{}"
        print(f"{date.strftime('%Y-%m')}: Set={set_str:<8} Actual={actual} P(crash)={proba:.3f} {in_set}")
