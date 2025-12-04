"""
CDAN: Conditional Domain Adversarial Network
Reference: Long et al. (2018) "Conditional Adversarial Domain Adaptation"

Key idea: Condition domain discriminator on classifier predictions,
making domain-invariant features also class-discriminative.

This is a stronger baseline than standard DANN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class RandomizedMultilinearMap(nn.Module):
    """
    Randomized multilinear conditioning for CDAN.
    Projects (features, predictions) to a joint embedding.
    """
    def __init__(self, feature_dim: int, class_dim: int, output_dim: int = 1024):
        super().__init__()
        self.output_dim = output_dim
        # Random projection matrices (fixed, not learned)
        self.Rf = nn.Parameter(torch.randn(feature_dim, output_dim), requires_grad=False)
        self.Rg = nn.Parameter(torch.randn(class_dim, output_dim), requires_grad=False)

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f: Features (batch, feature_dim)
            g: Predictions/softmax (batch, class_dim)
        Returns:
            Joint embedding (batch, output_dim)
        """
        f_proj = torch.mm(f, self.Rf)  # (batch, output_dim)
        g_proj = torch.mm(g, self.Rg)  # (batch, output_dim)
        return f_proj * g_proj / np.sqrt(self.output_dim)


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal layer for adversarial training."""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_


class CDANNet(nn.Module):
    """
    Conditional Domain Adversarial Network.

    Architecture:
        Input → FeatureExtractor → [Classifier, DomainDiscriminator(f⊗g)]

    The domain discriminator receives the outer product of features and
    classifier predictions, making it condition on class information.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.3,
                 use_random_projection: bool = True):
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

        # Task classifier (binary)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

        # Conditional domain discriminator
        self.use_random_projection = use_random_projection
        if use_random_projection:
            # Use randomized multilinear map for efficiency
            self.conditioning = RandomizedMultilinearMap(hidden_dim, 2, output_dim=256)
            discriminator_input_dim = 256
        else:
            # Direct outer product (hidden_dim * 2 for binary classification)
            discriminator_input_dim = hidden_dim * 2

        self.domain_discriminator = nn.Sequential(
            nn.Linear(discriminator_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.grl = GradientReversalLayer(lambda_=1.0)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, features)."""
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features

    def get_conditional_embedding(self, features: torch.Tensor,
                                   logits: torch.Tensor) -> torch.Tensor:
        """Create conditional embedding for domain discriminator."""
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        # Create 2-class probability distribution
        g = torch.cat([1 - probs, probs], dim=1)  # (batch, 2)

        if self.use_random_projection:
            return self.conditioning(features, g)
        else:
            # Direct outer product (flattened)
            return torch.cat([features * (1 - probs), features * probs], dim=1)

    def domain_output(self, features: torch.Tensor,
                      logits: torch.Tensor) -> torch.Tensor:
        """Get domain discriminator output with gradient reversal."""
        cond_embed = self.get_conditional_embedding(features, logits)
        reversed_embed = self.grl(cond_embed)
        return self.domain_discriminator(reversed_embed)

    def predict(self, x) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            return torch.sigmoid(logits)


class CDANTrainer:
    """Training utilities for CDAN."""

    def __init__(self, model: CDANNet, device: str = 'cpu',
                 lr: float = 1e-3, weight_decay: float = 1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.task_criterion = nn.BCEWithLogitsLoss()
        self.domain_criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, source_loader, target_loader,
                    lambda_domain: float = 1.0) -> dict:
        """Train one epoch."""
        self.model.train()
        self.model.grl.set_lambda(lambda_domain)

        total_task_loss = 0
        total_domain_loss = 0
        n_batches = 0

        target_iter = iter(target_loader)

        for source_data in source_loader:
            # Handle different tuple lengths
            if len(source_data) == 3:
                source_x, source_y, _ = source_data
            else:
                source_x, source_y, _, _ = source_data

            try:
                target_data = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_data = next(target_iter)

            if len(target_data) == 3:
                target_x, _, _ = target_data
            else:
                target_x, _, _, _ = target_data

            source_x = source_x.to(self.device)
            source_y = source_y.to(self.device).float()
            target_x = target_x.to(self.device)

            batch_size_s = source_x.size(0)
            batch_size_t = target_x.size(0)

            # Forward pass
            source_logits, source_features = self.model(source_x)
            target_logits, target_features = self.model(target_x)

            # Task loss (source only)
            task_loss = self.task_criterion(source_logits.squeeze(), source_y)

            # Domain loss (both domains)
            source_domain_output = self.model.domain_output(source_features, source_logits)
            target_domain_output = self.model.domain_output(target_features, target_logits)

            source_domain_labels = torch.zeros(batch_size_s, 1).to(self.device)
            target_domain_labels = torch.ones(batch_size_t, 1).to(self.device)

            domain_loss = (
                self.domain_criterion(source_domain_output, source_domain_labels) +
                self.domain_criterion(target_domain_output, target_domain_labels)
            ) / 2

            # Combined loss
            loss = task_loss + lambda_domain * domain_loss

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_task_loss += task_loss.item()
            total_domain_loss += domain_loss.item()
            n_batches += 1

        return {
            'task_loss': total_task_loss / max(n_batches, 1),
            'domain_loss': total_domain_loss / max(n_batches, 1)
        }


class MCDNet(nn.Module):
    """
    MCD: Maximum Classifier Discrepancy
    Reference: Saito et al. (2018) "Maximum Classifier Discrepancy for Unsupervised Domain Adaptation"

    Key idea: Two classifiers with maximum discrepancy on target,
    then feature extractor minimizes this discrepancy.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
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

        # Two classifiers
        self.classifier1 = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logits1, logits2, features)."""
        features = self.feature_extractor(x)
        logits1 = self.classifier1(features)
        logits2 = self.classifier2(features)
        return logits1, logits2, features

    def predict(self, x) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits1, logits2, _ = self.forward(x)
            # Average predictions from both classifiers
            return (torch.sigmoid(logits1) + torch.sigmoid(logits2)) / 2


class MCDTrainer:
    """Training utilities for MCD with 3-step optimization."""

    def __init__(self, model: MCDNet, device: str = 'cpu',
                 lr: float = 1e-3, weight_decay: float = 1e-4):
        self.model = model.to(device)
        self.device = device

        # Separate optimizers for feature extractor and classifiers
        self.opt_F = torch.optim.Adam(
            model.feature_extractor.parameters(),
            lr=lr, weight_decay=weight_decay
        )
        self.opt_C = torch.optim.Adam(
            list(model.classifier1.parameters()) + list(model.classifier2.parameters()),
            lr=lr, weight_decay=weight_decay
        )

        self.task_criterion = nn.BCEWithLogitsLoss()

    def discrepancy(self, logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
        """L1 discrepancy between two classifier outputs."""
        return torch.mean(torch.abs(torch.sigmoid(logits1) - torch.sigmoid(logits2)))

    def train_epoch(self, source_loader, target_loader,
                    n_critic: int = 4) -> dict:
        """
        MCD 3-step training:
        1. Train F and C on source (minimize task loss)
        2. Train C to maximize discrepancy on target (fix F)
        3. Train F to minimize discrepancy on target (fix C)
        """
        self.model.train()

        total_task_loss = 0
        total_disc_loss = 0
        n_batches = 0

        target_iter = iter(target_loader)

        for source_data in source_loader:
            if len(source_data) == 3:
                source_x, source_y, _ = source_data
            else:
                source_x, source_y, _, _ = source_data

            try:
                target_data = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_data = next(target_iter)

            if len(target_data) == 3:
                target_x, _, _ = target_data
            else:
                target_x, _, _, _ = target_data

            source_x = source_x.to(self.device)
            source_y = source_y.to(self.device).float()
            target_x = target_x.to(self.device)

            # Step 1: Train F and C on source
            logits1, logits2, _ = self.model(source_x)
            task_loss = (
                self.task_criterion(logits1.squeeze(), source_y) +
                self.task_criterion(logits2.squeeze(), source_y)
            ) / 2

            self.opt_F.zero_grad()
            self.opt_C.zero_grad()
            task_loss.backward()
            self.opt_F.step()
            self.opt_C.step()

            # Step 2: Train C to maximize discrepancy (fix F)
            for _ in range(n_critic):
                with torch.no_grad():
                    features = self.model.feature_extractor(target_x)
                logits1 = self.model.classifier1(features)
                logits2 = self.model.classifier2(features)
                disc = -self.discrepancy(logits1, logits2)  # Maximize

                self.opt_C.zero_grad()
                disc.backward()
                self.opt_C.step()

            # Step 3: Train F to minimize discrepancy (fix C)
            features = self.model.feature_extractor(target_x)
            with torch.no_grad():
                logits1 = self.model.classifier1(features)
                logits2 = self.model.classifier2(features)
            disc = self.discrepancy(logits1.detach(), logits2.detach())

            # Re-compute with gradient through F
            logits1, logits2, _ = self.model(target_x)
            disc = self.discrepancy(logits1, logits2)

            self.opt_F.zero_grad()
            disc.backward()
            self.opt_F.step()

            total_task_loss += task_loss.item()
            total_disc_loss += disc.item()
            n_batches += 1

        return {
            'task_loss': total_task_loss / max(n_batches, 1),
            'discrepancy': total_disc_loss / max(n_batches, 1)
        }


if __name__ == '__main__':
    print("Testing CDAN implementation...")

    # Test CDAN
    model = CDANNet(input_dim=30, hidden_dim=64)
    x = torch.randn(32, 30)
    logits, features = model(x)
    print(f"CDAN output: logits {logits.shape}, features {features.shape}")

    # Test conditional embedding
    cond = model.get_conditional_embedding(features, logits)
    print(f"Conditional embedding: {cond.shape}")

    # Test domain output
    domain_out = model.domain_output(features, logits)
    print(f"Domain output: {domain_out.shape}")

    # Test MCD
    mcd_model = MCDNet(input_dim=30, hidden_dim=64)
    logits1, logits2, features = mcd_model(x)
    print(f"MCD output: logits1 {logits1.shape}, logits2 {logits2.shape}")

    print("\nCDAN/MCD test passed!")
