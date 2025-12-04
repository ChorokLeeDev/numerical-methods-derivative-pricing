"""
Maximum Mean Discrepancy (MMD) for Domain Adaptation

Reference: Long et al. (2015) "Learning Transferable Features with Deep Adaptation Networks"

MMD measures the distance between two distributions in a reproducing kernel Hilbert space (RKHS).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor,
                    kernel_mul: float = 2.0, kernel_num: int = 5,
                    fix_sigma: float = None) -> torch.Tensor:
    """
    Compute Gaussian kernel matrix between x and y.

    Uses multiple kernel widths (MK-MMD) for better performance.
    """
    n_samples = x.size(0) + y.size(0)
    total = torch.cat([x, y], dim=0)

    # Compute pairwise distances
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1))
    )
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1))
    )
    L2_distance = ((total0 - total1) ** 2).sum(2)

    # Compute bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance) / (n_samples ** 2 - n_samples)

    # Multiple kernels
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # Compute kernel values
    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
    return sum(kernel_val) / len(kernel_val)


def mmd_loss(source: torch.Tensor, target: torch.Tensor,
             kernel_mul: float = 2.0, kernel_num: int = 5,
             fix_sigma: float = None) -> torch.Tensor:
    """
    Compute MMD loss between source and target distributions.

    MMD^2 = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]

    where x, x' ~ source and y, y' ~ target
    """
    batch_size = source.size(0)

    kernels = gaussian_kernel(
        source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma
    )

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]

    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
    return loss


class MMDNet(nn.Module):
    """
    Neural network with MMD regularization for domain adaptation.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        # Feature extractor
        layers = []
        current_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
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

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: Task prediction logits
            features: Extracted features (for MMD computation)
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features

    def predict(self, x) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            return torch.sigmoid(logits)


class MMDTrainer:
    """Training utilities for MMD-based domain adaptation."""

    def __init__(self, model: MMDNet, device: str = 'cpu',
                 lr: float = 1e-3, weight_decay: float = 1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.task_criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, source_loader, target_loader,
                    lambda_mmd: float = 1.0) -> dict:
        """
        Train one epoch with MMD regularization.
        """
        self.model.train()

        total_task_loss = 0
        total_mmd_loss = 0
        n_batches = 0

        target_iter = iter(target_loader)

        for source_x, source_y, _ in source_loader:
            try:
                target_x, _, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_x, _, _ = next(target_iter)

            source_x = source_x.to(self.device)
            source_y = source_y.to(self.device).float()
            target_x = target_x.to(self.device)

            # Forward pass
            source_logits, source_features = self.model(source_x)
            _, target_features = self.model(target_x)

            # Task loss (source only)
            task_loss = self.task_criterion(source_logits.squeeze(), source_y)

            # MMD loss
            mmd = mmd_loss(source_features, target_features)

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
            'task_loss': total_task_loss / n_batches,
            'mmd_loss': total_mmd_loss / n_batches
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

        auc = roc_auc_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))

        return {'auc': auc, 'accuracy': acc}


if __name__ == '__main__':
    print("Testing MMD implementation...")

    # Test MMD loss
    source = torch.randn(32, 64)
    target = torch.randn(32, 64)

    loss = mmd_loss(source, target)
    print(f"MMD loss (same distribution): {loss.item():.4f}")

    # Test with different distributions
    target_shifted = torch.randn(32, 64) + 2.0
    loss_shifted = mmd_loss(source, target_shifted)
    print(f"MMD loss (shifted distribution): {loss_shifted.item():.4f}")

    # Test network
    model = MMDNet(input_dim=30)
    x = torch.randn(32, 30)
    logits, features = model(x)
    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
    print("MMD test passed!")
