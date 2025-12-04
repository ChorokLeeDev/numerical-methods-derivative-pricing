"""
KDD 2026 Experiment: Domain Adaptation for Cross-Region Transfer

Compares:
1. Naive RF (baseline)
2. DANN (Domain Adversarial Neural Network)
3. MMD (Maximum Mean Discrepancy)
4. Multi-source transfer
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Add paths
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

from models.dann import DANN, DANNTrainer, CrowdingDataset, create_dann_dataloaders
from models.mmd import MMDNet, MMDTrainer

DATA_DIR = ROOT_DIR / 'data' / 'global_factors'


def load_all_regions():
    """Load all regional factor data."""
    data = {}
    for path in DATA_DIR.glob('*_factors.parquet'):
        region = path.stem.replace('_factors', '').upper()
        df = pd.read_parquet(path)
        if len(df) > 100:
            data[region] = df
    return data


def create_features_lagged(returns: pd.DataFrame):
    """Create ML features using lagged data only."""
    features = pd.DataFrame(index=returns.index)
    for col in returns.columns:
        r = returns[col]
        features[f'{col}_ret_1m_lag1'] = r.shift(1)
        features[f'{col}_ret_3m_lag1'] = r.rolling(3).mean().shift(1)
        features[f'{col}_ret_12m_lag1'] = r.rolling(12).mean().shift(1)
        features[f'{col}_vol_3m_lag1'] = r.rolling(3).std().shift(1)
        features[f'{col}_vol_12m_lag1'] = r.rolling(12).std().shift(1)
    return features.dropna()


def create_crash_target_rolling(returns: pd.Series, window: int = 60,
                                 threshold_pct: float = 0.10):
    """Binary target using rolling quantile."""
    targets = pd.Series(index=returns.index, dtype=float)
    for i in range(window, len(returns)):
        historical = returns.iloc[i-window:i]
        threshold = historical.quantile(threshold_pct)
        targets.iloc[i] = 1 if returns.iloc[i] < threshold else 0
    return targets.dropna().astype(int)


def prepare_region_data(region_data: pd.DataFrame, factor: str = 'Mom',
                        use_factors: list = None):
    """Prepare features and labels for a region."""
    if use_factors is not None:
        # Use only specified factors for consistency across regions
        available = [f for f in use_factors if f in region_data.columns]
        if not available:
            return None, None
        region_data = region_data[available]

    if factor not in region_data.columns:
        if len(region_data.columns) > 0:
            factor = region_data.columns[0]
        else:
            return None, None

    features = create_features_lagged(region_data)
    target = create_crash_target_rolling(region_data[factor])

    common_idx = features.index.intersection(target.index)
    if len(common_idx) < 50:
        return None, None

    X = features.loc[common_idx]
    y = target.loc[common_idx].values

    return X, y


def get_common_factors(data: dict, min_regions: int = 5) -> list:
    """Get factors available in at least min_regions regions."""
    factor_counts = {}
    for region_data in data.values():
        for col in region_data.columns:
            factor_counts[col] = factor_counts.get(col, 0) + 1

    # Return factors available in at least min_regions
    common = [f for f, count in factor_counts.items() if count >= min_regions]

    # Ensure Mom and HML are included if available (they're in most regions)
    if 'Mom' not in common and any('Mom' in d.columns for d in data.values()):
        common.append('Mom')
    if 'HML' not in common and any('HML' in d.columns for d in data.values()):
        common.append('HML')

    return common if common else ['Mom', 'HML']


def standardize_features(X_train, X_test):
    """Standardize features using training set statistics."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


# =============================================================================
# EXPERIMENT 1: Baseline (Naive RF)
# =============================================================================

def experiment_naive_rf(data: dict, source_region: str = 'US'):
    """Baseline: Train RF on source, test on all regions."""
    print("\n" + "=" * 60)
    print("BASELINE: Naive Random Forest Transfer")
    print("=" * 60)

    if source_region not in data or 'Mom' not in data[source_region].columns:
        print(f"{source_region} data not available!")
        return None

    # Use common factors across all regions
    common_factors = get_common_factors(data)
    print(f"Using common factors: {common_factors}")

    # Prepare source data
    X_source_df, y_source = prepare_region_data(data[source_region], use_factors=common_factors)
    if X_source_df is None:
        print("Could not prepare source data!")
        return None
    feature_cols = X_source_df.columns.tolist()
    X_source = X_source_df.values

    # Train/test split on source
    train_size = int(len(X_source) * 0.7)
    X_train = X_source[:train_size]
    y_train = y_source[:train_size]
    X_test_source = X_source[train_size:]
    y_test_source = y_source[train_size:]

    # Standardize
    X_train_std, X_test_source_std = standardize_features(X_train, X_test_source)

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=5,
                                   min_samples_leaf=10,
                                   class_weight='balanced', random_state=42)
    model.fit(X_train_std, y_train)

    results = {}

    # Test on source
    y_prob = model.predict_proba(X_test_source_std)[:, 1]
    results[f'{source_region} (test)'] = roc_auc_score(y_test_source, y_prob)

    # Test on other regions
    for region, region_data in data.items():
        if region == source_region:
            continue

        X_target_df, y_target = prepare_region_data(region_data, use_factors=common_factors)
        if X_target_df is None:
            print(f"  {region}: Skipping (no data)")
            continue

        # Ensure same column order
        try:
            X_target = X_target_df[feature_cols].values
        except KeyError:
            print(f"  {region}: Skipping (missing features)")
            continue

        # Use same standardization (from source training data)
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_target_std = (X_target - mean) / std

        try:
            y_prob = model.predict_proba(X_target_std)[:, 1]
            results[region] = roc_auc_score(y_target, y_prob)
        except Exception as e:
            print(f"  {region}: Error - {e}")

    print(f"\n{'Region':<15} {'AUC':<10}")
    print("-" * 25)
    for region, auc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{region:<15} {auc:<10.3f}")

    return results


# =============================================================================
# EXPERIMENT 2: DANN
# =============================================================================

def experiment_dann(data: dict, source_region: str = 'US',
                    epochs: int = 50, lambda_domain: float = 1.0):
    """DANN: Domain adversarial transfer."""
    print("\n" + "=" * 60)
    print("DANN: Domain Adversarial Neural Network")
    print("=" * 60)

    # Use common factors
    common_factors = get_common_factors(data)

    # Prepare all data
    all_data = {}
    feature_cols = None
    for region, region_data in data.items():
        X_df, y = prepare_region_data(region_data, use_factors=common_factors)
        if X_df is None:
            continue
        if feature_cols is None:
            feature_cols = X_df.columns.tolist()
        try:
            X = X_df[feature_cols].values
            all_data[region] = (X, y)
        except KeyError:
            continue

    if source_region not in all_data:
        print(f"{source_region} not available!")
        return None

    print(f"Regions with data: {list(all_data.keys())}")

    # Split source into train/test
    X_source, y_source = all_data[source_region]
    train_size = int(len(X_source) * 0.7)

    X_train = X_source[:train_size]
    y_train = y_source[:train_size]
    X_test = X_source[train_size:]
    y_test = y_source[train_size:]

    # Standardize all data using source training stats
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    all_data_std = {}
    for region, (X, y) in all_data.items():
        X_std = (X - mean) / std
        all_data_std[region] = (X_std, y)

    # Update source with standardized and split data
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    # Create data loaders manually
    target_regions = [r for r in all_data_std.keys() if r != source_region]
    all_regions = [source_region] + target_regions
    region_to_id = {r: i for i, r in enumerate(all_regions)}

    # Source loader
    source_ds = CrowdingDataset(X_train_std, y_train, region_to_id[source_region])
    source_loader = torch.utils.data.DataLoader(
        source_ds, batch_size=32, shuffle=True, drop_last=True
    )

    # Target loader
    target_datasets = []
    for region in target_regions:
        X, y = all_data_std[region]
        ds = CrowdingDataset(X, y, region_to_id[region])
        target_datasets.append(ds)

    if not target_datasets:
        print("No target regions available!")
        return None

    target_dataset = torch.utils.data.ConcatDataset(target_datasets)
    target_loader = torch.utils.data.DataLoader(
        target_dataset, batch_size=32, shuffle=True, drop_last=True
    )

    # Initialize DANN
    input_dim = X_train_std.shape[1]
    num_domains = len(region_to_id)

    model = DANN(input_dim, num_domains, hidden_dim=64, num_layers=2)
    trainer = DANNTrainer(model, device='cpu', lr=1e-3)

    # Training
    print(f"\nTraining DANN for {epochs} epochs...")
    for epoch in range(epochs):
        metrics = trainer.train_epoch(
            source_loader, target_loader, epoch, epochs, lambda_domain
        )
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: task_loss={metrics['task_loss']:.4f}, "
                  f"domain_loss={metrics['domain_loss']:.4f}, alpha={metrics['alpha']:.2f}")

    # Evaluate on all regions
    results = {}

    # Source test set
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_std)
        y_prob = model.predict(X_tensor).numpy().flatten()
        results[f'{source_region} (test)'] = roc_auc_score(y_test, y_prob)

    # Target regions
    for region in target_regions:
        X, y = all_data_std[region]
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            y_prob = model.predict(X_tensor).numpy().flatten()
            try:
                results[region] = roc_auc_score(y, y_prob)
            except:
                pass

    print(f"\n{'Region':<15} {'AUC':<10}")
    print("-" * 25)
    for region, auc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{region:<15} {auc:<10.3f}")

    return results


# =============================================================================
# EXPERIMENT 3: MMD
# =============================================================================

def experiment_mmd(data: dict, source_region: str = 'US',
                   epochs: int = 50, lambda_mmd: float = 1.0):
    """MMD: Maximum Mean Discrepancy adaptation."""
    print("\n" + "=" * 60)
    print("MMD: Maximum Mean Discrepancy Adaptation")
    print("=" * 60)

    # Use common factors
    common_factors = get_common_factors(data)

    # Prepare all data (same as DANN)
    all_data = {}
    feature_cols = None
    for region, region_data in data.items():
        X_df, y = prepare_region_data(region_data, use_factors=common_factors)
        if X_df is None:
            continue
        if feature_cols is None:
            feature_cols = X_df.columns.tolist()
        try:
            X = X_df[feature_cols].values
            all_data[region] = (X, y)
        except KeyError:
            continue

    if source_region not in all_data:
        return None

    print(f"Regions with data: {list(all_data.keys())}")

    X_source, y_source = all_data[source_region]
    train_size = int(len(X_source) * 0.7)

    X_train = X_source[:train_size]
    y_train = y_source[:train_size]
    X_test = X_source[train_size:]
    y_test = y_source[train_size:]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    all_data_std = {}
    for region, (X, y) in all_data.items():
        X_std = (X - mean) / std
        all_data_std[region] = (X_std, y)

    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    # Create loaders
    target_regions = [r for r in all_data_std.keys() if r != source_region]

    source_ds = CrowdingDataset(X_train_std, y_train, 0)
    source_loader = torch.utils.data.DataLoader(
        source_ds, batch_size=32, shuffle=True, drop_last=True
    )

    target_datasets = [
        CrowdingDataset(all_data_std[r][0], all_data_std[r][1], i+1)
        for i, r in enumerate(target_regions)
    ]
    target_dataset = torch.utils.data.ConcatDataset(target_datasets)
    target_loader = torch.utils.data.DataLoader(
        target_dataset, batch_size=32, shuffle=True, drop_last=True
    )

    # Initialize MMD model
    input_dim = X_train_std.shape[1]
    model = MMDNet(input_dim, hidden_dim=64, num_layers=2)
    trainer = MMDTrainer(model, device='cpu', lr=1e-3)

    # Training
    print(f"\nTraining MMD for {epochs} epochs...")
    for epoch in range(epochs):
        metrics = trainer.train_epoch(source_loader, target_loader, lambda_mmd)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: task_loss={metrics['task_loss']:.4f}, "
                  f"mmd_loss={metrics['mmd_loss']:.4f}")

    # Evaluate
    results = {}

    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_std)
        y_prob = model.predict(X_tensor).numpy().flatten()
        results[f'{source_region} (test)'] = roc_auc_score(y_test, y_prob)

    for region in target_regions:
        X, y = all_data_std[region]
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            y_prob = model.predict(X_tensor).numpy().flatten()
            try:
                results[region] = roc_auc_score(y, y_prob)
            except:
                pass

    print(f"\n{'Region':<15} {'AUC':<10}")
    print("-" * 25)
    for region, auc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{region:<15} {auc:<10.3f}")

    return results


# =============================================================================
# EXPERIMENT 4: Multi-Source Transfer
# =============================================================================

def experiment_multisource(data: dict, source_regions: list = ['US', 'EUROPE'],
                           epochs: int = 50):
    """Multi-source: Train on multiple regions."""
    print("\n" + "=" * 60)
    print(f"MULTI-SOURCE: Training on {source_regions}")
    print("=" * 60)

    # Use common factors
    common_factors = get_common_factors(data)

    # Prepare all data
    all_data = {}
    feature_cols = None
    for region, region_data in data.items():
        X_df, y = prepare_region_data(region_data, use_factors=common_factors)
        if X_df is None:
            continue
        if feature_cols is None:
            feature_cols = X_df.columns.tolist()
        try:
            X = X_df[feature_cols].values
            all_data[region] = (X, y)
        except KeyError:
            continue

    print(f"Regions with data: {list(all_data.keys())}")

    # Combine source regions
    X_sources = []
    y_sources = []
    for region in source_regions:
        if region in all_data:
            X, y = all_data[region]
            # Use first 70% of each
            train_size = int(len(X) * 0.7)
            X_sources.append(X[:train_size])
            y_sources.append(y[:train_size])

    if not X_sources:
        print("No source data available!")
        return None

    X_train = np.vstack(X_sources)
    y_train = np.concatenate(y_sources)

    # Standardize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_std = (X_train - mean) / std

    # Train RF on combined sources
    model = RandomForestClassifier(n_estimators=100, max_depth=5,
                                   min_samples_leaf=10,
                                   class_weight='balanced', random_state=42)
    model.fit(X_train_std, y_train)

    # Evaluate on all regions
    results = {}
    target_regions = [r for r in all_data.keys() if r not in source_regions]

    for region in target_regions:
        X, y = all_data[region]
        X_std = (X - mean) / std
        try:
            y_prob = model.predict_proba(X_std)[:, 1]
            results[region] = roc_auc_score(y, y_prob)
        except:
            pass

    # Also evaluate on source test sets
    for region in source_regions:
        if region in all_data:
            X, y = all_data[region]
            train_size = int(len(X) * 0.7)
            X_test = X[train_size:]
            y_test = y[train_size:]
            X_test_std = (X_test - mean) / std
            y_prob = model.predict_proba(X_test_std)[:, 1]
            results[f'{region} (test)'] = roc_auc_score(y_test, y_prob)

    print(f"\n{'Region':<15} {'AUC':<10}")
    print("-" * 25)
    for region, auc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{region:<15} {auc:<10.3f}")

    return results


# =============================================================================
# MAIN: Compare All Methods
# =============================================================================

def main():
    print("=" * 70)
    print("KDD 2026: DOMAIN ADAPTATION EXPERIMENTS")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    data = load_all_regions()
    print(f"Loaded {len(data)} regions")

    # Run experiments
    results = {}

    results['Naive RF'] = experiment_naive_rf(data)
    results['DANN'] = experiment_dann(data, epochs=50, lambda_domain=0.5)
    results['MMD'] = experiment_mmd(data, epochs=50, lambda_mmd=0.5)
    results['Multi-Source'] = experiment_multisource(data, source_regions=['US', 'EUROPE'])

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: METHOD COMPARISON")
    print("=" * 70)

    # Get all regions
    all_regions = set()
    for method_results in results.values():
        if method_results:
            all_regions.update(method_results.keys())

    # Print comparison table
    regions = sorted([r for r in all_regions if '(test)' not in r])
    test_regions = sorted([r for r in all_regions if '(test)' in r])

    print(f"\n{'Method':<15}", end="")
    for region in regions:
        print(f"{region:<12}", end="")
    print(f"{'Avg':<10}")
    print("-" * (15 + 12 * len(regions) + 10))

    for method, method_results in results.items():
        if method_results is None:
            continue
        print(f"{method:<15}", end="")
        aucs = []
        for region in regions:
            auc = method_results.get(region, 0)
            print(f"{auc:<12.3f}", end="")
            if auc > 0:
                aucs.append(auc)
        avg_auc = np.mean(aucs) if aucs else 0
        print(f"{avg_auc:<10.3f}")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    if results['Naive RF'] and results['DANN']:
        rf_avg = np.mean([v for k, v in results['Naive RF'].items() if '(test)' not in k])
        dann_avg = np.mean([v for k, v in results['DANN'].items() if '(test)' not in k])
        improvement = (dann_avg - rf_avg) / rf_avg * 100
        print(f"\n1. DANN vs Naive RF: {improvement:+.1f}% average improvement")

    if results['Naive RF'] and results['MMD']:
        mmd_avg = np.mean([v for k, v in results['MMD'].items() if '(test)' not in k])
        improvement = (mmd_avg - rf_avg) / rf_avg * 100
        print(f"2. MMD vs Naive RF: {improvement:+.1f}% average improvement")

    if results['Naive RF'] and results['Multi-Source']:
        ms_avg = np.mean([v for k, v in results['Multi-Source'].items() if '(test)' not in k])
        improvement = (ms_avg - rf_avg) / rf_avg * 100
        print(f"3. Multi-Source vs Naive RF: {improvement:+.1f}% average improvement")

    # UK specific
    if all(r and 'UK' in r for r in results.values()):
        print(f"\n4. UK Results (hardest region):")
        for method, method_results in results.items():
            if method_results and 'UK' in method_results:
                print(f"   {method}: {method_results['UK']:.3f}")

    return results


if __name__ == '__main__':
    results = main()
