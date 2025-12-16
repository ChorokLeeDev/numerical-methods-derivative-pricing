"""
ML-Based Crowding Detection and Tail Risk Prediction

Models:
1. Random Forest (baseline)
2. XGBoost (gradient boosting)
3. LSTM (temporal patterns)

All models use walk-forward CV to avoid lookahead bias.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, brier_score_loss, confusion_matrix
)

# Check for optional dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Run: pip install xgboost")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not installed. Run: pip install torch")


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    auc: float
    precision: float
    recall: float
    f1: float
    brier: float
    predictions: np.ndarray
    probabilities: np.ndarray
    actuals: np.ndarray


class BaseMLModel:
    """Base class for ML models."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)


class RandomForestModel(BaseMLModel):
    """Random Forest for crash/crowding prediction."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_leaf: int = 20,
        class_weight: str = 'balanced',
        random_state: int = 42,
    ):
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importances."""
        imp = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return imp


class XGBoostModel(BaseMLModel):
    """XGBoost for crash/crowding prediction."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        scale_pos_weight: float = None,  # Auto-calculated if None
        random_state: int = 42,
    ):
        super().__init__("XGBoost")
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)

        # Auto-calculate class weight if not provided
        scale_pos_weight = self.scale_pos_weight
        if scale_pos_weight is None:
            neg_count = (y == 0).sum()
            pos_count = (y == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
        )
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importances."""
        imp = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return imp


class LSTMModel(BaseMLModel):
    """LSTM for temporal crash/crowding prediction."""

    def __init__(
        self,
        input_size: int = None,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        sequence_length: int = 12,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        device: str = None,
    ):
        super().__init__("LSTM")
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None

    def _build_model(self, input_size: int):
        """Build LSTM network."""

        class LSTMNetwork(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                )
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 32),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # Use last timestep output
                last_output = lstm_out[:, -1, :]
                return self.fc(last_output)

        return LSTMNetwork(
            input_size, self.hidden_size, self.num_layers, self.dropout
        ).to(self.device)

    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None):
        """Create sequences for LSTM input."""
        sequences = []
        targets = []

        for i in range(self.sequence_length, len(X)):
            sequences.append(X[i - self.sequence_length:i])
            if y is not None:
                targets.append(y[i])

        X_seq = np.array(sequences)
        if y is not None:
            y_seq = np.array(targets)
            return X_seq, y_seq
        return X_seq

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y)

        if len(X_seq) < self.batch_size:
            print(f"Warning: Not enough samples ({len(X_seq)}) for LSTM training")
            self.is_fitted = False
            return self

        # Build model
        self.input_size = X.shape[1]
        self.model = self._build_model(self.input_size)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(1).to(self.device)

        # Class weights for imbalanced data
        pos_weight = torch.tensor([(y_seq == 0).sum() / max((y_seq == 1).sum(), 1)]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Actually use BCE since we have sigmoid in model
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            return np.full(len(X), 0.5)

        X_scaled = self.scaler.transform(X)
        X_seq = self._create_sequences(X_scaled)

        if len(X_seq) == 0:
            return np.full(len(X), 0.5)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            proba = self.model(X_tensor).cpu().numpy().flatten()

        # Pad beginning with 0.5 (no prediction for first sequence_length samples)
        full_proba = np.full(len(X), 0.5)
        full_proba[self.sequence_length:] = proba

        return full_proba


class ModelEvaluator:
    """Evaluate and compare ML models."""

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> ModelMetrics:
        """Compute evaluation metrics."""
        # Handle edge cases
        if len(np.unique(y_true)) < 2:
            return ModelMetrics(
                auc=0.5, precision=0.0, recall=0.0,
                f1=0.0, brier=1.0,
                predictions=y_pred, probabilities=y_proba, actuals=y_true
            )

        return ModelMetrics(
            auc=roc_auc_score(y_true, y_proba),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1=f1_score(y_true, y_pred, zero_division=0),
            brier=brier_score_loss(y_true, y_proba),
            predictions=y_pred,
            probabilities=y_proba,
            actuals=y_true,
        )

    @staticmethod
    def print_metrics(name: str, metrics: ModelMetrics):
        """Print formatted metrics."""
        print(f"\n{name}:")
        print(f"  AUC:       {metrics.auc:.3f}")
        print(f"  Precision: {metrics.precision:.3f}")
        print(f"  Recall:    {metrics.recall:.3f}")
        print(f"  F1:        {metrics.f1:.3f}")
        print(f"  Brier:     {metrics.brier:.3f}")


class WalkForwardBacktest:
    """
    Walk-forward backtesting for ML models.

    Trains on expanding/rolling window, tests on next period.
    No lookahead bias.
    """

    def __init__(
        self,
        train_size: int = 120,
        test_size: int = 12,
        step_size: int = 12,
        expanding: bool = True,
    ):
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.expanding = expanding

    def run(
        self,
        model: BaseMLModel,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run walk-forward backtest.

        Returns dict with predictions, metrics by period, and aggregate metrics.
        """
        n = len(X)
        all_preds = []
        all_proba = []
        all_actuals = []
        all_dates = []
        period_metrics = []

        start = 0
        split_num = 0

        while start + self.train_size + self.test_size <= n:
            # Define train/test periods
            if self.expanding:
                train_start = 0
            else:
                train_start = start

            train_end = start + self.train_size
            test_end = min(train_end + self.test_size, n)

            # Get data
            X_train = X.iloc[train_start:train_end].values
            y_train = y.iloc[train_start:train_end].values
            X_test = X.iloc[train_end:test_end].values
            y_test = y.iloc[train_end:test_end].values
            test_dates = X.index[train_end:test_end]

            # Train and predict
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)
            preds = (proba >= 0.5).astype(int)

            # Store results
            all_preds.extend(preds)
            all_proba.extend(proba)
            all_actuals.extend(y_test)
            all_dates.extend(test_dates)

            # Period metrics
            if len(np.unique(y_test)) > 1:
                period_auc = roc_auc_score(y_test, proba)
            else:
                period_auc = 0.5

            period_metrics.append({
                'split': split_num,
                'train_start': X.index[train_start],
                'train_end': X.index[train_end - 1],
                'test_start': test_dates[0],
                'test_end': test_dates[-1],
                'auc': period_auc,
                'n_test': len(y_test),
                'crash_rate': y_test.mean(),
            })

            if verbose and split_num % 10 == 0:
                print(f"  Split {split_num}: AUC={period_auc:.3f}, "
                      f"Test {test_dates[0].strftime('%Y-%m')} - {test_dates[-1].strftime('%Y-%m')}")

            start += self.step_size
            split_num += 1

        # Aggregate metrics
        all_preds = np.array(all_preds)
        all_proba = np.array(all_proba)
        all_actuals = np.array(all_actuals)

        aggregate_metrics = ModelEvaluator.evaluate(all_actuals, all_preds, all_proba)

        return {
            'predictions': pd.Series(all_preds, index=all_dates),
            'probabilities': pd.Series(all_proba, index=all_dates),
            'actuals': pd.Series(all_actuals, index=all_dates),
            'period_metrics': pd.DataFrame(period_metrics),
            'aggregate_metrics': aggregate_metrics,
        }


def run_model_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, BaseMLModel] = None,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Compare multiple models using walk-forward backtesting.

    Args:
        X: Feature DataFrame
        y: Target Series
        models: Dict of model name -> model instance
        verbose: Print progress

    Returns:
        Dict of model name -> backtest results
    """
    if models is None:
        models = {
            'RandomForest': RandomForestModel(),
        }
        if HAS_XGBOOST:
            models['XGBoost'] = XGBoostModel()
        if HAS_TORCH:
            models['LSTM'] = LSTMModel(sequence_length=6, epochs=30)

    backtester = WalkForwardBacktest(
        train_size=120,
        test_size=12,
        step_size=12,
        expanding=True,
    )

    results = {}

    for name, model in models.items():
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"Running {name}...")
            print('=' * 50)

        try:
            results[name] = backtester.run(model, X, y, verbose=verbose)

            if verbose:
                metrics = results[name]['aggregate_metrics']
                ModelEvaluator.print_metrics(f"{name} (Aggregate)", metrics)
        except Exception as e:
            print(f"Error running {name}: {e}")
            results[name] = None

    return results


# ================================================================
# EXAMPLE USAGE
# ================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from features import FeatureEngineer, WalkForwardCV

    DATA_DIR = Path(__file__).parent.parent / 'data'

    # Load data
    print("Loading data...")
    factors = pd.read_parquet(DATA_DIR / 'ff_extended_factors.parquet')
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    factors = factors.drop(columns=['RF'], errors='ignore')

    # Generate features
    print("Generating features...")
    fe = FeatureEngineer()
    features = fe.generate_all_features(factors)

    # Create dataset for Momentum crashes
    print("Creating ML dataset...")
    X, y = fe.create_ml_dataset(features, factors, target_type='crash', factor='Mom')

    print(f"\nDataset: {len(X)} samples, {len(X.columns)} features")
    print(f"Crash rate: {y.mean():.1%}")

    # Compare models
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Walk-Forward Backtest)")
    print("=" * 60)

    results = run_model_comparison(X, y, verbose=True)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<15} {'AUC':<8} {'Precision':<10} {'Recall':<8} {'F1':<8}")
    print("-" * 50)

    for name, res in results.items():
        if res is not None:
            m = res['aggregate_metrics']
            print(f"{name:<15} {m.auc:<8.3f} {m.precision:<10.3f} {m.recall:<8.3f} {m.f1:<8.3f}")

    # Feature importance (for tree-based models)
    print("\n" + "=" * 60)
    print("TOP 10 FEATURES (RandomForest)")
    print("=" * 60)

    rf_model = RandomForestModel()
    rf_model.fit(X.values, y.values)
    importance = rf_model.feature_importance(X.columns.tolist())
    print(importance.head(10).to_string(index=False))
