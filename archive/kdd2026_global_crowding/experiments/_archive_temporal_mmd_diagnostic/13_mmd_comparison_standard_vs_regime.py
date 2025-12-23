"""
Final Verification: Standard MMD vs. Regime-Conditional MMD

Test the hypothesis: Regime-conditioning HURTS performance because regimes don't transfer.
If regimes were helpful, T-MMD would outperform standard MMD.
If regimes hurt, standard MMD would outperform T-MMD.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

from models.temporal_mmd import TemporalMMDNet

DATA_DIR = ROOT_DIR / 'data' / 'global_factors'


def prepare_country_pair(source_name, target_name):
    """Prepare US→Country transfer data (same as 09_country_transfer_validation.py)."""

    source_file = DATA_DIR / f'{source_name.lower()}_factors.parquet'
    target_file = DATA_DIR / f'{target_name.lower()}_factors.parquet'

    if not source_file.exists() or not target_file.exists():
        return None

    source_df = pd.read_parquet(source_file)
    target_df = pd.read_parquet(target_file)

    # Align columns
    common_cols = list(set(source_df.columns) & set(target_df.columns))
    if len(common_cols) == 0:
        return None

    source_df = source_df[common_cols].dropna()
    target_df = target_df[common_cols].dropna()

    # Create features
    def make_features(df, windows=[5, 21]):
        features = pd.DataFrame(index=df.index)
        for col in df.columns:
            for w in windows:
                features[f'{col}_ret_{w}'] = df[col].rolling(w).mean().shift(1)
                features[f'{col}_vol_{w}'] = df[col].rolling(w).std().shift(1)
        return features.dropna()

    source_feat = make_features(source_df)
    target_feat = make_features(target_df)

    # Align indices
    common_idx = source_feat.index.intersection(target_feat.index)
    if len(common_idx) < 100:
        return None

    X_source = source_feat.loc[common_idx].values
    X_target = target_feat.loc[common_idx].values

    # Create targets: 1 if momentum is in bottom 10% (crash signal)
    if 'Mom' in source_df.columns:
        source_mom = source_df['Mom'].loc[common_idx]
        target_mom = target_df['Mom'].loc[common_idx]
    else:
        source_mom = source_df.iloc[:, 0].loc[common_idx]
        target_mom = target_df.iloc[:, 0].loc[common_idx]

    y_source = (source_mom < source_mom.rolling(252).quantile(0.1).shift(1)).astype(int).values
    y_target = (target_mom < target_mom.rolling(252).quantile(0.1).shift(1)).astype(int).values

    # Create regime labels
    vol_source = source_mom.rolling(63).std()
    median_vol_s = vol_source.rolling(252).median()
    regime_source = (vol_source > median_vol_s).astype(int).values

    vol_target = target_mom.rolling(63).std()
    median_vol_t = vol_target.rolling(252).median()
    regime_target = (vol_target > median_vol_t).astype(int).values

    # Clean NaNs
    mask = ~(np.isnan(y_source) | np.isnan(y_target) |
             np.isnan(regime_source) | np.isnan(regime_target))

    if mask.sum() < 100:
        return None

    return {
        'X_source': X_source[mask],
        'X_target': X_target[mask],
        'y_source': y_source[mask],
        'y_target': y_target[mask],
        'regime_source': regime_source[mask],
        'regime_target': regime_target[mask],
        'samples': mask.sum()
    }


def mmd_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard MMD loss computation."""
    from models.temporal_mmd import gaussian_kernel

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


def evaluate_mmd_variants(X_source, X_target, y_source, y_target, regime_source, regime_target):
    """Compare Standard MMD vs. Regime-Conditional MMD."""

    # Train/test split (70/30)
    n = len(X_source)
    train_idx = int(n * 0.7)

    X_train = X_source[:train_idx]
    y_train = y_source[:train_idx]
    X_test = X_source[train_idx:]
    y_test = y_source[train_idx:]

    # Standardize
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    X_target_std = scaler.transform(X_target)

    results = {}

    # Baseline: Random Forest
    try:
        rf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                    class_weight='balanced', random_state=42)
        rf.fit(X_train_std, y_train)
        rf_target_auc = roc_auc_score(y_target, rf.predict_proba(X_target_std)[:, 1])
        results['RF'] = rf_target_auc
    except:
        results['RF'] = np.nan

    # Test 1: Standard MMD (no regime conditioning)
    try:
        input_dim = X_train_std.shape[1]

        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, X, y):
                self.X = torch.FloatTensor(X)
                self.y = torch.LongTensor(y)

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        source_ds = SimpleDataset(X_train_std, y_train)
        source_loader = torch.utils.data.DataLoader(source_ds, batch_size=32, shuffle=True, drop_last=True)

        target_ds = SimpleDataset(X_target_std, y_target)
        target_loader = torch.utils.data.DataLoader(target_ds, batch_size=32, shuffle=True, drop_last=True)

        # Simple model for standard MMD
        class StandardMMDNet(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim=32):
                super().__init__()
                self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
                self.fc2 = torch.nn.Linear(hidden_dim, 1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                logits = self.fc2(x)
                features = x
                return logits, features

        model = StandardMMDNet(input_dim, hidden_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(20):
            model.train()
            target_iter = iter(target_loader)
            for source_x, source_y in source_loader:
                try:
                    target_x, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_x, _ = next(target_iter)

                source_logits, source_features = model(source_x)
                _, target_features = model(target_x)

                task_loss = criterion(source_logits.squeeze(), source_y.float())
                # Standard (global) MMD: all samples matched together
                mmd = mmd_loss(source_features, target_features)
                loss = task_loss + 0.5 * mmd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            standard_mmd_auc = roc_auc_score(y_target, model(torch.FloatTensor(X_target_std))[0].numpy().flatten())

        results['StandardMMD'] = standard_mmd_auc
    except Exception as e:
        print(f"    StandardMMD error: {e}")
        results['StandardMMD'] = np.nan

    # Test 2: Regime-Conditional MMD (T-MMD)
    try:
        class RegimeAwareDataset(torch.utils.data.Dataset):
            def __init__(self, X, y, regimes, domain_label):
                self.X = torch.FloatTensor(X)
                self.y = torch.LongTensor(y)
                self.regimes = torch.LongTensor(regimes)
                self.domain_labels = torch.full((len(X),), domain_label, dtype=torch.long)

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx], self.regimes[idx], self.domain_labels[idx]

        regime_train = regime_source[:train_idx]

        source_ds = RegimeAwareDataset(X_train_std, y_train, regime_train, 0)
        source_loader = torch.utils.data.DataLoader(source_ds, batch_size=32, shuffle=True, drop_last=True)

        target_ds = RegimeAwareDataset(X_target_std, y_target, regime_target, 1)
        target_loader = torch.utils.data.DataLoader(target_ds, batch_size=32, shuffle=True, drop_last=True)

        tmmd_model = TemporalMMDNet(input_dim, hidden_dim=32, num_layers=2, num_regimes=2)
        optimizer = torch.optim.Adam(tmmd_model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(20):
            tmmd_model.train()
            target_iter = iter(target_loader)
            for source_x, source_y, source_reg, _ in source_loader:
                try:
                    target_x, _, target_reg, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_x, _, target_reg, _ = next(target_iter)

                source_logits, source_features = tmmd_model(source_x)
                _, target_features = tmmd_model(target_x)

                task_loss = criterion(source_logits.squeeze(), source_y.float())
                # Regime-conditional MMD
                mmd = tmmd_model.temporal_mmd(source_features, target_features, source_reg, target_reg)
                loss = task_loss + 0.5 * mmd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        tmmd_model.eval()
        with torch.no_grad():
            tmmd_auc = roc_auc_score(y_target, tmmd_model.predict(torch.FloatTensor(X_target_std)).numpy().flatten())

        results['TemporalMMD'] = tmmd_auc
    except Exception as e:
        print(f"    TemporalMMD error: {e}")
        results['TemporalMMD'] = np.nan

    return results


def main():
    print("=" * 90)
    print("FINAL VERIFICATION: Standard MMD vs. Regime-Conditional MMD")
    print("=" * 90)
    print("\nHypothesis: If regimes don't transfer, Standard MMD should outperform T-MMD")
    print()

    country_pairs = [
        ('US', 'UK'),
        ('US', 'Japan'),
        ('US', 'Europe'),
        ('US', 'AsiaPac'),
    ]

    all_results = []

    print(f"{'Country Pair':<15} | {'RF Baseline':<12} | {'Standard MMD':<12} | {'T-MMD':<12} | {'Comparison':<30}")
    print("-" * 90)

    for source, target in country_pairs:
        print(f"\n[{source} → {target}]")

        data = prepare_country_pair(source, target)
        if data is None:
            print(f"  ✗ Could not prepare data")
            continue

        print(f"  Preparing ({data['samples']} samples)...")
        # Remove 'samples' key before unpacking
        data_clean = {k: v for k, v in data.items() if k != 'samples'}
        results = evaluate_mmd_variants(**data_clean)

        rf_auc = results.get('RF', np.nan)
        std_mmd = results.get('StandardMMD', np.nan)
        tmmd_auc = results.get('TemporalMMD', np.nan)

        comparison = ""
        if not np.isnan(std_mmd) and not np.isnan(tmmd_auc):
            if std_mmd > tmmd_auc:
                diff = (std_mmd - tmmd_auc) / tmmd_auc * 100
                comparison = f"StandardMMD +{diff:.1f}% better"
            else:
                diff = (tmmd_auc - std_mmd) / std_mmd * 100
                comparison = f"T-MMD +{diff:.1f}% better"

        print(f"  RF Baseline:    {rf_auc:.4f}")
        if not np.isnan(std_mmd):
            print(f"  Standard MMD:   {std_mmd:.4f}")
        else:
            print(f"  Standard MMD:   FAILED")
        if not np.isnan(tmmd_auc):
            print(f"  T-MMD:          {tmmd_auc:.4f}")
        else:
            print(f"  T-MMD:          FAILED")
        print(f"  {comparison}")

        all_results.append({
            'pair': f"{source}→{target}",
            'RF': rf_auc,
            'StandardMMD': std_mmd,
            'TemporalMMD': tmmd_auc,
            'comparison': comparison,
        })

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print("\nIf Standard MMD > T-MMD in most cases → Regime conditioning hurts")
    print("If T-MMD > Standard MMD in most cases → Regime conditioning helps")
    print()

    standard_wins = sum(1 for r in all_results
                       if not np.isnan(r['StandardMMD']) and not np.isnan(r['TemporalMMD'])
                       and r['StandardMMD'] > r['TemporalMMD'])
    tmmd_wins = sum(1 for r in all_results
                   if not np.isnan(r['StandardMMD']) and not np.isnan(r['TemporalMMD'])
                   and r['TemporalMMD'] > r['StandardMMD'])

    print(f"Standard MMD wins: {standard_wins} cases")
    print(f"T-MMD wins: {tmmd_wins} cases")

    if standard_wins > tmmd_wins:
        print("\n✓ HYPOTHESIS CONFIRMED: Standard MMD outperforms T-MMD")
        print("  Interpretation: Regime conditioning hurts because regimes don't transfer")
    else:
        print("\n✗ HYPOTHESIS REJECTED: T-MMD outperforms Standard MMD")
        print("  Interpretation: Regime conditioning helps despite transfer issues")

    return all_results


if __name__ == '__main__':
    results = main()
