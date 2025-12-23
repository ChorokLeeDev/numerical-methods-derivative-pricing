"""
Temporal-MMD: Regime-Aware Domain Adaptation for Financial Time Series

Key Innovation: Instead of matching all source and target data uniformly,
we match distributions within similar market regimes.

This addresses a fundamental limitation of standard domain adaptation:
financial data exhibits regime-dependent distributions (bull/bear markets,
high/low volatility periods), and naive distribution matching across
different regimes can hurt performance.

Reference: Novel contribution for KDD 2026
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from enum import Enum


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = 0        # Positive momentum, low volatility
    BEAR = 1        # Negative momentum, high volatility
    HIGH_VOL = 2    # High volatility (either direction)
    LOW_VOL = 3     # Low volatility (either direction)


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor,
                    kernel_mul: float = 2.0, kernel_num: int = 5) -> torch.Tensor:
    """Multi-kernel MMD kernel computation."""
    n_samples = x.size(0) + y.size(0)
    total = torch.cat([x, y], dim=0)

    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    bandwidth = torch.sum(L2_distance) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / (bw + 1e-8)) for bw in bandwidth_list]
    return sum(kernel_val) / len(kernel_val)


def mmd_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard MMD loss."""
    if source.size(0) < 2 or target.size(0) < 2:
        return torch.tensor(0.0)

    batch_size = min(source.size(0), target.size(0))
    source = source[:batch_size]
    target = target[:batch_size]

    kernels = gaussian_kernel(source, target)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]

    return torch.mean(XX) + torch.mean(YY) - 2 * torch.mean(XY)


class RegimeDetector:
    """
    Detect market regimes from factor returns.

    Regimes based on:
    1. Momentum: Rolling return (positive = bull, negative = bear)
    2. Volatility: Rolling std (above median = high vol)
    """

    def __init__(self, lookback: int = 12, vol_threshold: float = 0.5):
        self.lookback = lookback
        self.vol_threshold = vol_threshold

    def detect_regimes(self, returns: np.ndarray) -> np.ndarray:
        """
        Classify each time point into a regime.

        Args:
            returns: Array of shape (T, n_features) or (T,)

        Returns:
            regime_labels: Array of shape (T,) with regime indices
        """
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        T = len(returns)
        regimes = np.zeros(T, dtype=np.int64)

        # Use first column (typically momentum) for regime detection
        r = returns[:, 0] if returns.shape[1] > 0 else returns.flatten()

        # Rolling statistics
        for t in range(self.lookback, T):
            window = r[t-self.lookback:t]
            mom = np.mean(window)
            vol = np.std(window)

            # Classify regime
            if mom > 0 and vol < np.median(np.abs(r[:t])):
                regimes[t] = MarketRegime.BULL.value
            elif mom < 0 and vol > np.median(np.abs(r[:t])):
                regimes[t] = MarketRegime.BEAR.value
            elif vol > np.median(np.abs(r[:t])):
                regimes[t] = MarketRegime.HIGH_VOL.value
            else:
                regimes[t] = MarketRegime.LOW_VOL.value

        # Fill early periods with most common regime
        if self.lookback > 0:
            regimes[:self.lookback] = np.bincount(regimes[self.lookback:]).argmax()

        return regimes

    def detect_regimes_simple(self, returns: np.ndarray) -> np.ndarray:
        """
        Simpler 2-regime classification: High vol vs Low vol.
        More robust with limited data.
        """
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        T = len(returns)
        regimes = np.zeros(T, dtype=np.int64)

        r = returns[:, 0] if returns.shape[1] > 0 else returns.flatten()

        # Rolling volatility
        for t in range(self.lookback, T):
            window = r[t-self.lookback:t]
            vol = np.std(window)

            # Simple binary: high vol (1) vs low vol (0)
            median_vol = np.median([np.std(r[max(0,i-self.lookback):i])
                                    for i in range(self.lookback, t+1)])
            regimes[t] = 1 if vol > median_vol else 0

        return regimes


class TemporalMMDLoss(nn.Module):
    """
    Temporal-MMD: Regime-conditional domain adaptation loss.

    Key innovation: Compute MMD separately for each market regime,
    then aggregate. This ensures we only match distributions
    within similar market conditions.

    Loss = Σ_r w_r * MMD(S_r, T_r)

    where:
    - r indexes regimes (bull, bear, high_vol, low_vol)
    - S_r, T_r are source/target samples in regime r
    - w_r is the weight for regime r (default: uniform)
    """

    def __init__(self, num_regimes: int = 2, regime_weights: Optional[List[float]] = None):
        super().__init__()
        self.num_regimes = num_regimes
        if regime_weights is None:
            regime_weights = [1.0 / num_regimes] * num_regimes
        self.regime_weights = regime_weights

    def forward(self, source_features: torch.Tensor, target_features: torch.Tensor,
                source_regimes: torch.Tensor, target_regimes: torch.Tensor) -> torch.Tensor:
        """
        Compute regime-conditional MMD loss.

        Args:
            source_features: (N_s, D) source domain features
            target_features: (N_t, D) target domain features
            source_regimes: (N_s,) regime labels for source
            target_regimes: (N_t,) regime labels for target

        Returns:
            Weighted sum of per-regime MMD losses
        """
        total_loss = torch.tensor(0.0, device=source_features.device)

        for regime in range(self.num_regimes):
            # Get samples from this regime
            source_mask = source_regimes == regime
            target_mask = target_regimes == regime

            source_r = source_features[source_mask]
            target_r = target_features[target_mask]

            # Need at least 2 samples from each domain
            if source_r.size(0) >= 2 and target_r.size(0) >= 2:
                regime_mmd = mmd_loss(source_r, target_r)
                total_loss = total_loss + self.regime_weights[regime] * regime_mmd

        return total_loss


class TemporalMMDNet(nn.Module):
    """
    Neural network with Temporal-MMD adaptation.

    Architecture:
        Input → Feature Extractor → [Regime-conditional MMD] → Task Head → Output
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.3,
                 num_regimes: int = 2):
        super().__init__()

        # Feature extractor
        layers = []
        current_dim = input_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.feature_dim = hidden_dim

        # Task classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

        # Temporal-MMD loss
        self.temporal_mmd = TemporalMMDLoss(num_regimes=num_regimes)
        self.num_regimes = num_regimes

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, features)."""
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features

    def predict(self, x) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            return torch.sigmoid(logits)


class TemporalMMDTrainer:
    """Training utilities for Temporal-MMD."""

    def __init__(self, model: TemporalMMDNet, device: str = 'cpu',
                 lr: float = 1e-3, weight_decay: float = 1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.task_criterion = nn.BCEWithLogitsLoss()
        self.regime_detector = RegimeDetector(lookback=12)

    def compute_regimes(self, features: np.ndarray) -> np.ndarray:
        """Detect regimes from features (uses first feature as proxy for returns)."""
        # Use first feature column as regime indicator
        return self.regime_detector.detect_regimes_simple(features[:, 0])

    def train_epoch(self, source_loader, target_loader,
                    source_regimes: np.ndarray, target_regimes: np.ndarray,
                    lambda_mmd: float = 1.0) -> dict:
        """Train one epoch with Temporal-MMD."""
        self.model.train()

        total_task_loss = 0
        total_mmd_loss = 0
        n_batches = 0

        # Convert regimes to tensors
        source_regime_tensor = torch.LongTensor(source_regimes)
        target_regime_tensor = torch.LongTensor(target_regimes)

        target_iter = iter(target_loader)
        source_idx = 0
        target_idx = 0

        for batch_idx, (source_x, source_y, _) in enumerate(source_loader):
            try:
                target_x, _, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_x, _, _ = next(target_iter)
                target_idx = 0

            source_x = source_x.to(self.device)
            source_y = source_y.to(self.device).float()
            target_x = target_x.to(self.device)

            # Get regime labels for this batch
            batch_size_s = source_x.size(0)
            batch_size_t = target_x.size(0)

            # Handle regime indexing (cycle through if needed)
            s_end = min(source_idx + batch_size_s, len(source_regime_tensor))
            t_end = min(target_idx + batch_size_t, len(target_regime_tensor))

            batch_source_regimes = source_regime_tensor[source_idx:s_end].to(self.device)
            batch_target_regimes = target_regime_tensor[target_idx:t_end].to(self.device)

            # Pad if needed
            if len(batch_source_regimes) < batch_size_s:
                batch_source_regimes = torch.cat([
                    batch_source_regimes,
                    source_regime_tensor[:batch_size_s - len(batch_source_regimes)].to(self.device)
                ])
            if len(batch_target_regimes) < batch_size_t:
                batch_target_regimes = torch.cat([
                    batch_target_regimes,
                    target_regime_tensor[:batch_size_t - len(batch_target_regimes)].to(self.device)
                ])

            source_idx = (source_idx + batch_size_s) % len(source_regime_tensor)
            target_idx = (target_idx + batch_size_t) % len(target_regime_tensor)

            # Forward pass
            source_logits, source_features = self.model(source_x)
            _, target_features = self.model(target_x)

            # Task loss
            task_loss = self.task_criterion(source_logits.squeeze(), source_y)

            # Temporal-MMD loss
            mmd = self.model.temporal_mmd(
                source_features, target_features,
                batch_source_regimes, batch_target_regimes
            )

            # Combined loss
            loss = task_loss + lambda_mmd * mmd

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_task_loss += task_loss.item()
            total_mmd_loss += mmd.item()
            n_batches += 1

        return {
            'task_loss': total_task_loss / max(n_batches, 1),
            'mmd_loss': total_mmd_loss / max(n_batches, 1)
        }

    def evaluate(self, loader) -> dict:
        """Evaluate on a dataset."""
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(self.device)
                preds = self.model.predict(x)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        from sklearn.metrics import roc_auc_score, accuracy_score

        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.5

        acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))

        return {'auc': auc, 'accuracy': acc}


# =============================================================================
# Additional Innovation: Adaptive Regime Weighting
# =============================================================================

class AdaptiveTemporalMMD(TemporalMMDLoss):
    """
    Extension: Learn regime weights automatically.

    Instead of uniform weights, learn which regimes are most important
    for domain adaptation.
    """

    def __init__(self, num_regimes: int = 2):
        super().__init__(num_regimes)
        # Learnable regime weights
        self.regime_weight_params = nn.Parameter(torch.ones(num_regimes))

    def forward(self, source_features: torch.Tensor, target_features: torch.Tensor,
                source_regimes: torch.Tensor, target_regimes: torch.Tensor) -> torch.Tensor:
        # Softmax to ensure weights sum to 1
        weights = torch.softmax(self.regime_weight_params, dim=0)

        total_loss = torch.tensor(0.0, device=source_features.device)

        for regime in range(self.num_regimes):
            source_mask = source_regimes == regime
            target_mask = target_regimes == regime

            source_r = source_features[source_mask]
            target_r = target_features[target_mask]

            if source_r.size(0) >= 2 and target_r.size(0) >= 2:
                regime_mmd = mmd_loss(source_r, target_r)
                total_loss = total_loss + weights[regime] * regime_mmd

        return total_loss


if __name__ == '__main__':
    print("Testing Temporal-MMD implementation...")

    # Test regime detection
    detector = RegimeDetector(lookback=12)
    returns = np.random.randn(100, 2) * 0.05
    regimes = detector.detect_regimes_simple(returns)
    print(f"Regime distribution: {np.bincount(regimes)}")

    # Test Temporal-MMD loss
    source = torch.randn(32, 64)
    target = torch.randn(32, 64)
    source_reg = torch.randint(0, 2, (32,))
    target_reg = torch.randint(0, 2, (32,))

    loss_fn = TemporalMMDLoss(num_regimes=2)
    loss = loss_fn(source, target, source_reg, target_reg)
    print(f"Temporal-MMD loss: {loss.item():.4f}")

    # Compare with standard MMD
    std_mmd = mmd_loss(source, target)
    print(f"Standard MMD loss: {std_mmd.item():.4f}")

    # Test model
    model = TemporalMMDNet(input_dim=30, num_regimes=2)
    x = torch.randn(32, 30)
    logits, features = model(x)
    print(f"Output shape: {logits.shape}, Features shape: {features.shape}")

    print("\nTemporal-MMD test passed!")
