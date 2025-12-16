"""
Neural Network for Tail Risk Prediction

Architecture for predicting P(crash | features, factor_type)

Key features:
- Factor type embedding (captures heterogeneous tail risk)
- Temporal features (LSTM or attention)
- Walk-forward training (no lookahead)

For ICAIF 2025 submission.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class NNConfig:
    """Neural network configuration."""
    hidden_sizes: List[int] = None
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    early_stopping_patience: int = 10
    factor_embedding_dim: int = 8

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [64, 32]


class TailRiskMLP(nn.Module):
    """
    Multi-Layer Perceptron for tail risk prediction.

    Architecture:
    - Input: Features + Factor embedding
    - Hidden: Multiple dense layers with ReLU, Dropout, BatchNorm
    - Output: Sigmoid for P(crash)
    """

    def __init__(
        self,
        input_size: int,
        num_factors: int,
        config: NNConfig
    ):
        super().__init__()
        self.config = config

        # Factor embedding layer
        self.factor_embedding = nn.Embedding(
            num_factors,
            config.factor_embedding_dim
        )

        # Build hidden layers
        layers = []
        prev_size = input_size + config.factor_embedding_dim

        for hidden_size in config.hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            prev_size = hidden_size

        self.hidden = nn.Sequential(*layers)

        # Output layer
        self.output = nn.Linear(prev_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, factor_idx: torch.Tensor) -> torch.Tensor:
        # Factor embedding
        factor_emb = self.factor_embedding(factor_idx)

        # Concatenate features and embedding
        combined = torch.cat([x, factor_emb], dim=1)

        # Hidden layers
        h = self.hidden(combined)

        # Output
        return self.sigmoid(self.output(h)).squeeze()


class TailRiskLSTM(nn.Module):
    """
    LSTM-based model for sequential tail risk prediction.

    Architecture:
    - Input: Sequence of features over time
    - LSTM: Captures temporal dependencies
    - Dense: Final prediction layer
    """

    def __init__(
        self,
        input_size: int,
        num_factors: int,
        config: NNConfig,
        sequence_length: int = 12
    ):
        super().__init__()
        self.config = config
        self.sequence_length = sequence_length

        # Factor embedding
        self.factor_embedding = nn.Embedding(
            num_factors,
            config.factor_embedding_dim
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_sizes[0],
            num_layers=2,
            batch_first=True,
            dropout=config.dropout_rate
        )

        # Output layers
        lstm_out_size = config.hidden_sizes[0] + config.factor_embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_size, config.hidden_sizes[-1]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_sizes[-1], 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, factor_idx: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, sequence, features)
        lstm_out, _ = self.lstm(x)

        # Take last hidden state
        last_hidden = lstm_out[:, -1, :]

        # Factor embedding
        factor_emb = self.factor_embedding(factor_idx)

        # Combine and predict
        combined = torch.cat([last_hidden, factor_emb], dim=1)
        return self.fc(combined).squeeze()


class NeuralNetworkTrainer:
    """
    Walk-forward trainer for neural network models.

    Handles:
    - Feature scaling
    - Class imbalance weighting
    - Early stopping
    - Walk-forward validation
    """

    def __init__(
        self,
        model_class: type,
        config: Optional[NNConfig] = None,
        factor_names: Optional[List[str]] = None
    ):
        self.model_class = model_class
        self.config = config or NNConfig()
        self.factor_names = factor_names or []
        self.factor_to_idx = {f: i for i, f in enumerate(self.factor_names)}
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        factor_labels: pd.Series,
        fit_scaler: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for training."""
        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        # Convert factor labels to indices
        factor_idx = factor_labels.map(self.factor_to_idx).values

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y.values).to(self.device)
        factor_tensor = torch.LongTensor(factor_idx).to(self.device)

        return X_tensor, y_tensor, factor_tensor

    def _compute_class_weights(self, y: np.ndarray) -> torch.Tensor:
        """Compute class weights for imbalanced data."""
        pos_weight = (1 - y.mean()) / (y.mean() + 1e-8)
        return torch.tensor([pos_weight]).to(self.device)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        factor_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        factor_val: Optional[pd.Series] = None,
        verbose: bool = False
    ):
        """Train the model."""
        # Prepare training data
        X_t, y_t, f_t = self._prepare_data(X_train, y_train, factor_train, fit_scaler=True)

        # Initialize model
        self.model = self.model_class(
            input_size=X_train.shape[1],
            num_factors=len(self.factor_names),
            config=self.config
        ).to(self.device)

        # Loss with class weights
        pos_weight = self._compute_class_weights(y_train.values)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )

        # Create data loader
        dataset = TensorDataset(X_t, f_t, y_t)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = 0

            for batch_x, batch_f, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x, batch_f)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Validation
            if X_val is not None:
                val_loss = self._validate(X_val, y_val, factor_val, criterion)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

            if verbose and epoch % 10 == 0:
                val_str = f", val_loss={val_loss:.4f}" if X_val is not None else ""
                print(f"Epoch {epoch}: train_loss={epoch_loss/len(loader):.4f}{val_str}")

    def _validate(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        factor_val: pd.Series,
        criterion: nn.Module
    ) -> float:
        """Compute validation loss."""
        self.model.eval()
        X_v, y_v, f_v = self._prepare_data(X_val, y_val, factor_val, fit_scaler=False)

        with torch.no_grad():
            outputs = self.model(X_v, f_v)
            loss = criterion(outputs, y_v)

        return loss.item()

    def predict_proba(
        self,
        X: pd.DataFrame,
        factor_labels: pd.Series
    ) -> np.ndarray:
        """Predict crash probabilities."""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        factor_idx = factor_labels.map(self.factor_to_idx).values

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        factor_tensor = torch.LongTensor(factor_idx).to(self.device)

        with torch.no_grad():
            probs = self.model(X_tensor, factor_tensor)

        return probs.cpu().numpy()


class WalkForwardNNBacktest:
    """
    Walk-forward backtesting for neural network models.

    Multi-factor approach:
    - Pool all factors together
    - Factor embedding captures heterogeneous effects
    """

    def __init__(
        self,
        train_size: int = 120,
        test_size: int = 12,
        step_size: int = 12
    ):
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size

    def run(
        self,
        features_by_factor: Dict[str, pd.DataFrame],
        targets_by_factor: Dict[str, pd.Series],
        config: Optional[NNConfig] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Run walk-forward backtest across all factors.

        Args:
            features_by_factor: Dict mapping factor name to feature DataFrame
            targets_by_factor: Dict mapping factor name to target Series
            config: Neural network configuration
            verbose: Print progress

        Returns:
            Dict with predictions, actuals, and metrics
        """
        # Stack all factors
        all_features = []
        all_targets = []
        all_factor_labels = []
        all_dates = []

        for factor in features_by_factor:
            X = features_by_factor[factor]
            y = targets_by_factor[factor]

            # Align
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]

            all_features.append(X)
            all_targets.append(y)
            all_factor_labels.extend([factor] * len(X))
            all_dates.extend(X.index.tolist())

        # Combine
        X_all = pd.concat(all_features)
        y_all = pd.concat(all_targets)
        factor_labels = pd.Series(all_factor_labels, index=X_all.index)
        dates = pd.Series(all_dates, index=X_all.index)

        # Sort by date for walk-forward
        sort_idx = dates.argsort()
        X_sorted = X_all.iloc[sort_idx]
        y_sorted = y_all.iloc[sort_idx]
        factor_sorted = factor_labels.iloc[sort_idx]

        # Remove NaN
        valid_mask = ~X_sorted.isna().any(axis=1)
        X_sorted = X_sorted[valid_mask]
        y_sorted = y_sorted[valid_mask]
        factor_sorted = factor_sorted[valid_mask]

        # Unique dates for splitting
        unique_dates = X_sorted.index.unique().sort_values()
        n_dates = len(unique_dates)

        # Walk-forward splits
        predictions = []
        actuals = []
        pred_factors = []

        start = 0
        while start + self.train_size + self.test_size <= n_dates:
            train_dates = unique_dates[start:start + self.train_size]
            test_dates = unique_dates[start + self.train_size:start + self.train_size + self.test_size]

            # Split by date
            train_mask = X_sorted.index.isin(train_dates)
            test_mask = X_sorted.index.isin(test_dates)

            X_train = X_sorted[train_mask]
            y_train = y_sorted[train_mask]
            f_train = factor_sorted[train_mask]

            X_test = X_sorted[test_mask]
            y_test = y_sorted[test_mask]
            f_test = factor_sorted[test_mask]

            # Train model
            trainer = NeuralNetworkTrainer(
                model_class=TailRiskMLP,
                config=config or NNConfig(),
                factor_names=list(features_by_factor.keys())
            )

            trainer.fit(X_train, y_train, f_train)

            # Predict
            preds = trainer.predict_proba(X_test, f_test)

            predictions.extend(preds)
            actuals.extend(y_test.values)
            pred_factors.extend(f_test.values)

            start += self.step_size

            if verbose:
                print(f"Split {start//self.step_size}: train={len(train_dates)} months, test={len(test_dates)} months")

        # Compute metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        pred_factors = np.array(pred_factors)

        # Overall metrics
        overall_auc = roc_auc_score(actuals, predictions)
        overall_ap = average_precision_score(actuals, predictions)

        # Per-factor metrics
        factor_metrics = {}
        for factor in features_by_factor:
            mask = pred_factors == factor
            if mask.sum() > 0 and actuals[mask].sum() > 0:
                factor_metrics[factor] = {
                    'auc': roc_auc_score(actuals[mask], predictions[mask]),
                    'ap': average_precision_score(actuals[mask], predictions[mask]),
                    'n_samples': mask.sum(),
                    'crash_rate': actuals[mask].mean()
                }

        return {
            'predictions': predictions,
            'actuals': actuals,
            'factors': pred_factors,
            'overall_auc': overall_auc,
            'overall_ap': overall_ap,
            'factor_metrics': factor_metrics
        }


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
    print("NEURAL NETWORK TAIL RISK PREDICTION")
    print("=" * 60)

    # Load data
    factors = pd.read_parquet(DATA_DIR / 'ff_extended_factors.parquet')
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    factors = factors.drop(columns=['RF'], errors='ignore')

    print(f"\nData: {factors.index.min().strftime('%Y-%m')} to {factors.index.max().strftime('%Y-%m')}")
    print(f"Factors: {list(factors.columns)}")

    # Generate features
    fe = FeatureEngineer()
    all_features = fe.generate_all_features(factors)

    # Create per-factor datasets
    features_by_factor = {}
    targets_by_factor = {}

    for factor in factors.columns:
        X, y = fe.create_ml_dataset(all_features, factors, target_type='crash', factor=factor)
        features_by_factor[factor] = X
        targets_by_factor[factor] = y

    # Run walk-forward backtest
    print("\n" + "=" * 60)
    print("WALK-FORWARD BACKTEST")
    print("=" * 60)

    config = NNConfig(
        hidden_sizes=[64, 32],
        dropout_rate=0.3,
        epochs=30,
        early_stopping_patience=5
    )

    backtest = WalkForwardNNBacktest(
        train_size=120,
        test_size=12,
        step_size=12
    )

    results = backtest.run(
        features_by_factor,
        targets_by_factor,
        config=config,
        verbose=True
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nOverall AUC: {results['overall_auc']:.3f}")
    print(f"Overall Average Precision: {results['overall_ap']:.3f}")

    print(f"\n{'Factor':<10} {'AUC':<8} {'AP':<8} {'Crash%':<8} {'N':<8}")
    print("-" * 40)

    for factor, metrics in sorted(results['factor_metrics'].items()):
        print(f"{factor:<10} {metrics['auc']:.3f}    {metrics['ap']:.3f}    {metrics['crash_rate']*100:.1f}%     {metrics['n_samples']}")
