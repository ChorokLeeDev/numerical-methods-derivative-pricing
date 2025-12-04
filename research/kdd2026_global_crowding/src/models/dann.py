"""
Domain Adversarial Neural Network (DANN) for Cross-Region Crowding Detection

Reference: Ganin et al. (2016) "Domain-Adversarial Training of Neural Networks"

Architecture:
    Input → Feature Extractor → Task Head → Crowding Prediction
                              ↘ Domain Classifier → Region Label (adversarial)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from typing import Tuple, Optional


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL).
    Forward: identity
    Backward: negate gradient and scale by lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_


class FeatureExtractor(nn.Module):
    """Shared feature extractor for all regions."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(self, x):
        return self.network(x)


class TaskClassifier(nn.Module):
    """Crowding prediction head."""

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x)


class DomainClassifier(nn.Module):
    """Region classification head (adversarial)."""

    def __init__(self, input_dim: int, num_domains: int, hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_domains)
        )

    def forward(self, x):
        return self.network(x)


class DANN(nn.Module):
    """
    Domain Adversarial Neural Network for Crowding Detection.

    Loss = L_task - lambda * L_domain

    The adversarial training forces the feature extractor to learn
    region-invariant representations.
    """

    def __init__(self, input_dim: int, num_domains: int,
                 hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.3, lambda_init: float = 1.0):
        super().__init__()

        self.feature_extractor = FeatureExtractor(
            input_dim, hidden_dim, num_layers, dropout
        )
        self.task_classifier = TaskClassifier(
            self.feature_extractor.output_dim
        )
        self.domain_classifier = DomainClassifier(
            self.feature_extractor.output_dim, num_domains
        )
        self.grl = GradientReversalLayer(lambda_init)

        self.num_domains = num_domains

    def forward(self, x, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features (batch_size, input_dim)
            alpha: GRL lambda parameter (increases during training)

        Returns:
            task_output: Crowding prediction logits (batch_size, 1)
            domain_output: Domain classification logits (batch_size, num_domains)
        """
        self.grl.set_lambda(alpha)

        features = self.feature_extractor(x)
        task_output = self.task_classifier(features)

        # Apply gradient reversal before domain classifier
        reversed_features = self.grl(features)
        domain_output = self.domain_classifier(reversed_features)

        return task_output, domain_output

    def predict(self, x) -> torch.Tensor:
        """Predict crowding probability."""
        self.eval()
        with torch.no_grad():
            features = self.feature_extractor(x)
            logits = self.task_classifier(features)
            return torch.sigmoid(logits)

    def get_features(self, x) -> torch.Tensor:
        """Extract learned features (for analysis)."""
        self.eval()
        with torch.no_grad():
            return self.feature_extractor(x)


class DANNTrainer:
    """Training utilities for DANN."""

    def __init__(self, model: DANN, device: str = 'cpu',
                 lr: float = 1e-3, weight_decay: float = 1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.task_criterion = nn.BCEWithLogitsLoss()
        self.domain_criterion = nn.CrossEntropyLoss()

    def compute_alpha(self, epoch: int, max_epochs: int) -> float:
        """
        Compute GRL lambda using schedule from original DANN paper.
        Gradually increase from 0 to 1.
        """
        p = epoch / max_epochs
        return 2. / (1. + np.exp(-10 * p)) - 1

    def train_epoch(self, source_loader, target_loader, epoch: int,
                    max_epochs: int, lambda_domain: float = 1.0) -> dict:
        """
        Train one epoch.

        Source data has labels, target data is unlabeled (for domain adaptation).
        """
        self.model.train()

        alpha = self.compute_alpha(epoch, max_epochs)

        total_task_loss = 0
        total_domain_loss = 0
        n_batches = 0

        # Iterate over both loaders
        target_iter = iter(target_loader)

        for source_x, source_y, source_domain in source_loader:
            # Get target batch (cycle if needed)
            try:
                target_x, _, target_domain = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_x, _, target_domain = next(target_iter)

            # Move to device
            source_x = source_x.to(self.device)
            source_y = source_y.to(self.device).float()
            source_domain = source_domain.to(self.device)
            target_x = target_x.to(self.device)
            target_domain = target_domain.to(self.device)

            # Forward pass - source
            task_output, domain_output = self.model(source_x, alpha)
            task_loss = self.task_criterion(task_output.squeeze(), source_y)
            domain_loss_source = self.domain_criterion(domain_output, source_domain)

            # Forward pass - target (no task loss, only domain)
            _, domain_output_target = self.model(target_x, alpha)
            domain_loss_target = self.domain_criterion(domain_output_target, target_domain)

            # Combined loss
            domain_loss = (domain_loss_source + domain_loss_target) / 2
            loss = task_loss + lambda_domain * domain_loss

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_task_loss += task_loss.item()
            total_domain_loss += domain_loss.item()
            n_batches += 1

        return {
            'task_loss': total_task_loss / n_batches,
            'domain_loss': total_domain_loss / n_batches,
            'alpha': alpha
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


class CrowdingDataset(torch.utils.data.Dataset):
    """Dataset for crowding detection with domain labels."""

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 domain_id: int):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.domain_id = domain_id
        self.domain_labels = torch.full((len(features),), domain_id, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.domain_labels[idx]


def create_dann_dataloaders(data_dict: dict, batch_size: int = 32,
                            source_regions: list = ['US'],
                            target_regions: list = None) -> Tuple:
    """
    Create data loaders for DANN training.

    Args:
        data_dict: {region: (features, labels)}
        source_regions: Regions with labels (for task training)
        target_regions: Regions for domain adaptation

    Returns:
        source_loader, target_loader, region_to_id mapping
    """
    if target_regions is None:
        target_regions = [r for r in data_dict.keys() if r not in source_regions]

    all_regions = source_regions + target_regions
    region_to_id = {r: i for i, r in enumerate(all_regions)}

    # Source datasets
    source_datasets = []
    for region in source_regions:
        if region in data_dict:
            features, labels = data_dict[region]
            ds = CrowdingDataset(features, labels, region_to_id[region])
            source_datasets.append(ds)

    # Target datasets
    target_datasets = []
    for region in target_regions:
        if region in data_dict:
            features, labels = data_dict[region]
            ds = CrowdingDataset(features, labels, region_to_id[region])
            target_datasets.append(ds)

    source_dataset = torch.utils.data.ConcatDataset(source_datasets)
    target_dataset = torch.utils.data.ConcatDataset(target_datasets)

    source_loader = torch.utils.data.DataLoader(
        source_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    target_loader = torch.utils.data.DataLoader(
        target_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return source_loader, target_loader, region_to_id


if __name__ == '__main__':
    # Quick test
    print("Testing DANN implementation...")

    # Create dummy data
    input_dim = 30
    num_domains = 5
    batch_size = 32

    model = DANN(input_dim, num_domains)
    x = torch.randn(batch_size, input_dim)

    task_out, domain_out = model(x, alpha=0.5)

    print(f"Input shape: {x.shape}")
    print(f"Task output shape: {task_out.shape}")
    print(f"Domain output shape: {domain_out.shape}")
    print("DANN test passed!")
