"""
KDD 2026: Ablation Study for Temporal-MMD

1. Effect of number of regimes (2 vs 4)
2. Effect of λ (MMD weight)
3. Effect of regime detection method
4. Visualization of regime distributions
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

from models.mmd import MMDNet, MMDTrainer
from models.temporal_mmd import TemporalMMDNet, TemporalMMDTrainer, RegimeDetector
from models.dann import CrowdingDataset

DATA_DIR = ROOT_DIR / 'data' / 'global_factors'
FIG_DIR = ROOT_DIR / 'paper' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_all_regions():
    data = {}
    for path in DATA_DIR.glob('*_factors.parquet'):
        region = path.stem.replace('_factors', '').upper()
        df = pd.read_parquet(path)
        if len(df) > 100:
            data[region] = df
    return data


def create_features_lagged(returns: pd.DataFrame):
    features = pd.DataFrame(index=returns.index)
    for col in returns.columns:
        r = returns[col]
        features[f'{col}_ret_1m_lag1'] = r.shift(1)
        features[f'{col}_ret_3m_lag1'] = r.rolling(3).mean().shift(1)
        features[f'{col}_ret_12m_lag1'] = r.rolling(12).mean().shift(1)
        features[f'{col}_vol_3m_lag1'] = r.rolling(3).std().shift(1)
        features[f'{col}_vol_12m_lag1'] = r.rolling(12).std().shift(1)
    return features.dropna()


def create_crash_target_rolling(returns: pd.Series, window: int = 60):
    targets = pd.Series(index=returns.index, dtype=float)
    for i in range(window, len(returns)):
        historical = returns.iloc[i-window:i]
        threshold = historical.quantile(0.10)
        targets.iloc[i] = 1 if returns.iloc[i] < threshold else 0
    return targets.dropna().astype(int)


def prepare_all_data(data: dict, common_factors: list):
    all_data = {}
    all_data_raw = {}
    feature_cols = None

    for region, region_data in data.items():
        available = [f for f in common_factors if f in region_data.columns]
        if not available:
            continue

        region_df = region_data[available]
        factor = 'Mom' if 'Mom' in available else available[0]

        features = create_features_lagged(region_df)
        target = create_crash_target_rolling(region_df[factor])

        common_idx = features.index.intersection(target.index)
        if len(common_idx) < 50:
            continue

        X_df = features.loc[common_idx]
        y = target.loc[common_idx].values

        if feature_cols is None:
            feature_cols = X_df.columns.tolist()

        try:
            X = X_df[feature_cols].values
            all_data[region] = (X, y)
            raw_returns = region_df.loc[common_idx].values
            all_data_raw[region] = (raw_returns, y)
        except:
            continue

    return all_data, all_data_raw, feature_cols


# =============================================================================
# ABLATION 1: Number of Regimes
# =============================================================================

def ablation_num_regimes(all_data_std, all_data_raw, source_region,
                         X_train_std, y_train, X_test_std, y_test):
    """Test effect of number of regimes (2 vs 4)."""
    print("\n" + "=" * 60)
    print("ABLATION 1: NUMBER OF REGIMES")
    print("=" * 60)

    results = {}

    for num_regimes in [2, 4]:
        print(f"\nTesting {num_regimes} regimes...")

        target_regions = [r for r in all_data_std.keys() if r != source_region]

        # Regime detection
        detector = RegimeDetector(lookback=12)
        source_raw = all_data_raw[source_region][0]
        train_size = len(X_train_std)

        if num_regimes == 2:
            source_regimes = detector.detect_regimes_simple(source_raw[:train_size])
            target_regimes_list = [detector.detect_regimes_simple(all_data_raw[r][0])
                                   for r in target_regions]
        else:
            source_regimes = detector.detect_regimes(source_raw[:train_size])
            target_regimes_list = [detector.detect_regimes(all_data_raw[r][0])
                                   for r in target_regions]

        target_regimes = np.concatenate(target_regimes_list)

        # Data loaders
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

        # Model
        input_dim = X_train_std.shape[1]
        model = TemporalMMDNet(input_dim, hidden_dim=64, num_layers=2,
                               num_regimes=num_regimes)
        trainer = TemporalMMDTrainer(model, device='cpu', lr=1e-3)

        # Train
        for epoch in range(50):
            trainer.train_epoch(source_loader, target_loader,
                               source_regimes, target_regimes, lambda_mmd=0.5)

        # Evaluate
        region_results = {}
        model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test_std)
            y_prob = model.predict(X_tensor).numpy().flatten()
            region_results[f'{source_region} (test)'] = roc_auc_score(y_test, y_prob)

        for region in target_regions:
            X, y = all_data_std[region]
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                y_prob = model.predict(X_tensor).numpy().flatten()
                try:
                    region_results[region] = roc_auc_score(y, y_prob)
                except:
                    pass

        results[num_regimes] = region_results

        # Print
        avg_auc = np.mean([v for k, v in region_results.items() if '(test)' not in k])
        print(f"  {num_regimes} regimes: Avg AUC = {avg_auc:.3f}")

    return results


# =============================================================================
# ABLATION 2: Lambda (MMD Weight)
# =============================================================================

def ablation_lambda(all_data_std, all_data_raw, source_region,
                    X_train_std, y_train, X_test_std, y_test):
    """Test effect of lambda (MMD weight)."""
    print("\n" + "=" * 60)
    print("ABLATION 2: LAMBDA (MMD WEIGHT)")
    print("=" * 60)

    results = {}
    lambdas = [0.1, 0.5, 1.0, 2.0]

    target_regions = [r for r in all_data_std.keys() if r != source_region]

    # Regime detection (fixed)
    detector = RegimeDetector(lookback=12)
    source_raw = all_data_raw[source_region][0]
    train_size = len(X_train_std)
    source_regimes = detector.detect_regimes_simple(source_raw[:train_size])
    target_regimes = np.concatenate([
        detector.detect_regimes_simple(all_data_raw[r][0])
        for r in target_regions
    ])

    # Data loaders
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

    for lam in lambdas:
        print(f"\nTesting λ = {lam}...")

        input_dim = X_train_std.shape[1]
        model = TemporalMMDNet(input_dim, hidden_dim=64, num_layers=2, num_regimes=2)
        trainer = TemporalMMDTrainer(model, device='cpu', lr=1e-3)

        for epoch in range(50):
            trainer.train_epoch(source_loader, target_loader,
                               source_regimes, target_regimes, lambda_mmd=lam)

        region_results = {}
        model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test_std)
            y_prob = model.predict(X_tensor).numpy().flatten()
            region_results[f'{source_region} (test)'] = roc_auc_score(y_test, y_prob)

        for region in target_regions:
            X, y = all_data_std[region]
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                y_prob = model.predict(X_tensor).numpy().flatten()
                try:
                    region_results[region] = roc_auc_score(y, y_prob)
                except:
                    pass

        results[lam] = region_results

        avg_auc = np.mean([v for k, v in region_results.items() if '(test)' not in k])
        print(f"  λ = {lam}: Avg AUC = {avg_auc:.3f}")

    return results


# =============================================================================
# VISUALIZATION: Regime Distributions
# =============================================================================

def visualize_regime_distributions(all_data_raw, all_data_std):
    """Visualize how regimes differ across regions."""
    print("\n" + "=" * 60)
    print("VISUALIZATION: REGIME DISTRIBUTIONS")
    print("=" * 60)

    detector = RegimeDetector(lookback=12)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    regions = ['US', 'UK', 'EUROPE', 'JAPAN', 'ASIAPAC', 'GLOBAL']

    for idx, region in enumerate(regions):
        if region not in all_data_raw:
            continue

        raw, y = all_data_raw[region]
        regimes = detector.detect_regimes_simple(raw)

        # Plot returns by regime
        ax = axes[idx]

        # Use first column as momentum proxy
        returns = raw[:, 0] if raw.ndim > 1 else raw

        low_vol_returns = returns[regimes == 0]
        high_vol_returns = returns[regimes == 1]

        ax.hist(low_vol_returns, bins=30, alpha=0.5, label='Low Vol Regime', density=True)
        ax.hist(high_vol_returns, bins=30, alpha=0.5, label='High Vol Regime', density=True)
        ax.set_title(f'{region}')
        ax.set_xlabel('Returns')
        ax.legend()

    plt.suptitle('Return Distributions by Regime Across Regions', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'regime_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIG_DIR / 'regime_distributions.png'}")

    # Also create regime proportion plot
    fig, ax = plt.subplots(figsize=(10, 6))

    regime_props = {}
    for region in regions:
        if region not in all_data_raw:
            continue
        raw, _ = all_data_raw[region]
        regimes = detector.detect_regimes_simple(raw)
        regime_props[region] = np.mean(regimes == 1)  # Proportion of high vol

    ax.bar(regime_props.keys(), regime_props.values())
    ax.set_ylabel('Proportion of High Volatility Regime')
    ax.set_title('Regime Composition by Region')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

    plt.savefig(FIG_DIR / 'regime_proportions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIG_DIR / 'regime_proportions.png'}")


def visualize_feature_space(all_data_std, all_data_raw):
    """t-SNE visualization of feature space by region and regime."""
    print("\n  Creating t-SNE visualization...")

    # Combine all data
    X_all = []
    labels_region = []
    labels_regime = []

    detector = RegimeDetector(lookback=12)
    regions = list(all_data_std.keys())

    for region in regions:
        X, y = all_data_std[region]
        raw, _ = all_data_raw[region]
        regimes = detector.detect_regimes_simple(raw)

        # Subsample for speed
        n_samples = min(200, len(X))
        idx = np.random.choice(len(X), n_samples, replace=False)

        X_all.append(X[idx])
        labels_region.extend([region] * n_samples)
        labels_regime.extend(regimes[idx])

    X_all = np.vstack(X_all)
    labels_regime = np.array(labels_regime)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X_all)

    # Plot by region
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color by region
    ax = axes[0]
    for region in regions:
        mask = np.array(labels_region) == region
        ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
                  label=region, alpha=0.6, s=20)
    ax.set_title('Feature Space by Region')
    ax.legend()

    # Color by regime
    ax = axes[1]
    colors = ['blue', 'red']
    labels = ['Low Vol', 'High Vol']
    for regime in [0, 1]:
        mask = labels_regime == regime
        ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
                  c=colors[regime], label=labels[regime], alpha=0.6, s=20)
    ax.set_title('Feature Space by Regime')
    ax.legend()

    plt.suptitle('t-SNE Visualization of Features', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'tsne_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIG_DIR / 'tsne_features.png'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("KDD 2026: ABLATION STUDY")
    print("=" * 70)

    # Load data
    data = load_all_regions()
    common_factors = ['HML', 'Mom']

    all_data, all_data_raw, feature_cols = prepare_all_data(data, common_factors)
    print(f"Regions: {list(all_data.keys())}")

    # Prepare source data
    source_region = 'US'
    X_source, y_source = all_data[source_region]
    train_size = int(len(X_source) * 0.7)

    X_train = X_source[:train_size]
    y_train = y_source[:train_size]
    X_test = X_source[train_size:]
    y_test = y_source[train_size:]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    all_data_std = {r: ((X - mean) / std, y) for r, (X, y) in all_data.items()}

    # Run ablations
    regime_results = ablation_num_regimes(
        all_data_std, all_data_raw, source_region,
        X_train_std, y_train, X_test_std, y_test
    )

    lambda_results = ablation_lambda(
        all_data_std, all_data_raw, source_region,
        X_train_std, y_train, X_test_std, y_test
    )

    # Visualizations
    visualize_regime_distributions(all_data_raw, all_data_std)
    visualize_feature_space(all_data_std, all_data_raw)

    # Summary
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)

    print("\n1. Number of Regimes:")
    for num_reg, results in regime_results.items():
        avg = np.mean([v for k, v in results.items() if '(test)' not in k])
        print(f"   {num_reg} regimes: {avg:.3f}")

    print("\n2. Lambda (MMD Weight):")
    for lam, results in lambda_results.items():
        avg = np.mean([v for k, v in results.items() if '(test)' not in k])
        print(f"   λ = {lam}: {avg:.3f}")

    # Best configuration
    best_lambda = max(lambda_results.keys(),
                      key=lambda l: np.mean([v for k, v in lambda_results[l].items()
                                            if '(test)' not in k]))
    best_regimes = max(regime_results.keys(),
                       key=lambda r: np.mean([v for k, v in regime_results[r].items()
                                             if '(test)' not in k]))

    print(f"\n** BEST CONFIGURATION **")
    print(f"   Regimes: {best_regimes}")
    print(f"   Lambda: {best_lambda}")

    return regime_results, lambda_results


if __name__ == '__main__':
    regime_results, lambda_results = main()
