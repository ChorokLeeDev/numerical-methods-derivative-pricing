"""
KDD 2026 Experiment: Temporal-MMD vs Standard MMD

Key Question: Does regime-aware domain adaptation outperform naive MMD?

Hypothesis: Financial data has regime-dependent distributions.
Matching distributions within similar regimes should improve transfer.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

from models.mmd import MMDNet, MMDTrainer
from models.temporal_mmd import TemporalMMDNet, TemporalMMDTrainer, RegimeDetector
from models.dann import CrowdingDataset

DATA_DIR = ROOT_DIR / 'data' / 'global_factors'


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


def create_crash_target_rolling(returns: pd.Series, window: int = 60,
                                 threshold_pct: float = 0.10):
    targets = pd.Series(index=returns.index, dtype=float)
    for i in range(window, len(returns)):
        historical = returns.iloc[i-window:i]
        threshold = historical.quantile(threshold_pct)
        targets.iloc[i] = 1 if returns.iloc[i] < threshold else 0
    return targets.dropna().astype(int)


def prepare_region_data(region_data: pd.DataFrame, use_factors: list):
    available = [f for f in use_factors if f in region_data.columns]
    if not available:
        return None, None
    region_data = region_data[available]

    factor = 'Mom' if 'Mom' in region_data.columns else region_data.columns[0]
    features = create_features_lagged(region_data)
    target = create_crash_target_rolling(region_data[factor])

    common_idx = features.index.intersection(target.index)
    if len(common_idx) < 50:
        return None, None

    return features.loc[common_idx], target.loc[common_idx].values


def get_common_factors(data: dict, min_regions: int = 5) -> list:
    factor_counts = {}
    for region_data in data.values():
        for col in region_data.columns:
            factor_counts[col] = factor_counts.get(col, 0) + 1
    common = [f for f, count in factor_counts.items() if count >= min_regions]
    if 'Mom' not in common:
        common.append('Mom')
    if 'HML' not in common:
        common.append('HML')
    return common


def experiment_standard_mmd(all_data_std: dict, source_region: str,
                            X_train_std: np.ndarray, y_train: np.ndarray,
                            X_test_std: np.ndarray, y_test: np.ndarray,
                            epochs: int = 50, lambda_mmd: float = 0.5):
    """Standard MMD baseline."""
    target_regions = [r for r in all_data_std.keys() if r != source_region]

    # Create loaders
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
    model = MMDNet(input_dim, hidden_dim=64, num_layers=2)
    trainer = MMDTrainer(model, device='cpu', lr=1e-3)

    # Train
    for epoch in range(epochs):
        trainer.train_epoch(source_loader, target_loader, lambda_mmd)

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

    return results


def experiment_temporal_mmd(all_data_std: dict, all_data_raw: dict,
                            source_region: str,
                            X_train_std: np.ndarray, y_train: np.ndarray,
                            X_test_std: np.ndarray, y_test: np.ndarray,
                            epochs: int = 50, lambda_mmd: float = 0.5):
    """Temporal-MMD (regime-aware)."""
    target_regions = [r for r in all_data_std.keys() if r != source_region]

    # Regime detection
    detector = RegimeDetector(lookback=12)

    # Detect regimes for source (training data)
    source_raw = all_data_raw[source_region][0]
    train_size = len(X_train_std)
    source_regimes = detector.detect_regimes_simple(source_raw[:train_size])

    # Detect regimes for all target regions
    target_regimes_all = []
    for region in target_regions:
        raw = all_data_raw[region][0]
        regimes = detector.detect_regimes_simple(raw)
        target_regimes_all.append(regimes)

    target_regimes = np.concatenate(target_regimes_all)

    # Create loaders
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
    model = TemporalMMDNet(input_dim, hidden_dim=64, num_layers=2, num_regimes=2)
    trainer = TemporalMMDTrainer(model, device='cpu', lr=1e-3)

    # Train
    for epoch in range(epochs):
        trainer.train_epoch(
            source_loader, target_loader,
            source_regimes, target_regimes,
            lambda_mmd
        )

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

    return results


def main():
    print("=" * 70)
    print("KDD 2026: TEMPORAL-MMD vs STANDARD MMD")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    data = load_all_regions()
    common_factors = get_common_factors(data)
    print(f"Using factors: {common_factors}")

    # Prepare data
    all_data = {}
    all_data_raw = {}  # For regime detection
    feature_cols = None

    for region, region_data in data.items():
        X_df, y = prepare_region_data(region_data, common_factors)
        if X_df is None:
            continue
        if feature_cols is None:
            feature_cols = X_df.columns.tolist()
        try:
            X = X_df[feature_cols].values
            all_data[region] = (X, y)
            # Store raw returns for regime detection
            available_factors = [f for f in common_factors if f in region_data.columns]
            raw_returns = region_data[available_factors].dropna().values
            all_data_raw[region] = (raw_returns[-len(y):], y)
        except:
            continue

    print(f"Regions with data: {list(all_data.keys())}")

    source_region = 'US'
    X_source, y_source = all_data[source_region]
    train_size = int(len(X_source) * 0.7)

    X_train = X_source[:train_size]
    y_train = y_source[:train_size]
    X_test = X_source[train_size:]
    y_test = y_source[train_size:]

    # Standardize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    all_data_std = {}
    for region, (X, y) in all_data.items():
        all_data_std[region] = ((X - mean) / std, y)

    # Run experiments multiple times for statistical significance
    n_runs = 3
    mmd_results_all = []
    temporal_results_all = []

    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")

        # Standard MMD
        print("\nStandard MMD:")
        mmd_results = experiment_standard_mmd(
            all_data_std, source_region,
            X_train_std, y_train, X_test_std, y_test,
            epochs=50, lambda_mmd=0.5
        )
        mmd_results_all.append(mmd_results)
        for region, auc in sorted(mmd_results.items(), key=lambda x: -x[1]):
            print(f"  {region}: {auc:.3f}")

        # Temporal-MMD
        print("\nTemporal-MMD:")
        temporal_results = experiment_temporal_mmd(
            all_data_std, all_data_raw, source_region,
            X_train_std, y_train, X_test_std, y_test,
            epochs=50, lambda_mmd=0.5
        )
        temporal_results_all.append(temporal_results)
        for region, auc in sorted(temporal_results.items(), key=lambda x: -x[1]):
            print(f"  {region}: {auc:.3f}")

    # Aggregate results
    print("\n" + "=" * 70)
    print("FINAL RESULTS (mean ± std over {} runs)".format(n_runs))
    print("=" * 70)

    regions = [r for r in mmd_results_all[0].keys() if '(test)' not in r]

    print(f"\n{'Region':<15} {'MMD':<20} {'Temporal-MMD':<20} {'Δ':<10}")
    print("-" * 65)

    total_improvement = []

    for region in regions:
        mmd_aucs = [r.get(region, 0) for r in mmd_results_all if r.get(region, 0) > 0]
        temp_aucs = [r.get(region, 0) for r in temporal_results_all if r.get(region, 0) > 0]

        if mmd_aucs and temp_aucs:
            mmd_mean = np.mean(mmd_aucs)
            mmd_std = np.std(mmd_aucs)
            temp_mean = np.mean(temp_aucs)
            temp_std = np.std(temp_aucs)
            delta = temp_mean - mmd_mean

            total_improvement.append(delta)

            print(f"{region:<15} {mmd_mean:.3f}±{mmd_std:.3f}       "
                  f"{temp_mean:.3f}±{temp_std:.3f}       {delta:+.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    avg_improvement = np.mean(total_improvement) if total_improvement else 0
    print(f"\n1. Average improvement: {avg_improvement:+.3f} ({avg_improvement*100:+.1f}%)")

    # UK specific
    uk_mmd = np.mean([r.get('UK', 0) for r in mmd_results_all if r.get('UK', 0) > 0])
    uk_temp = np.mean([r.get('UK', 0) for r in temporal_results_all if r.get('UK', 0) > 0])
    if uk_mmd > 0 and uk_temp > 0:
        uk_improvement = (uk_temp - uk_mmd) / uk_mmd * 100
        print(f"2. UK improvement: {uk_mmd:.3f} → {uk_temp:.3f} ({uk_improvement:+.1f}%)")

    # Conclusion
    print(f"\n** CONCLUSION **")
    if avg_improvement > 0.01:
        print(f"Temporal-MMD outperforms standard MMD by {avg_improvement*100:.1f}%")
        print("Regime-aware adaptation provides meaningful improvement.")
    elif avg_improvement > 0:
        print(f"Temporal-MMD shows marginal improvement (+{avg_improvement*100:.1f}%)")
    else:
        print("No significant improvement from regime-aware adaptation.")

    return mmd_results_all, temporal_results_all


if __name__ == '__main__':
    mmd_results, temporal_results = main()
