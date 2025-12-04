"""
KDD 2026 Experiment: Daily Data Scale-Up

Compare Monthly vs Daily data performance
Show that Temporal-MMD scales to larger datasets
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

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

from models.mmd import MMDNet, MMDTrainer
from models.temporal_mmd import TemporalMMDNet, TemporalMMDTrainer, RegimeDetector
from models.dann import CrowdingDataset

DAILY_DIR = ROOT_DIR / 'data' / 'daily_factors'


def load_daily_data():
    """Load daily factor ETF data."""
    data = {}

    us_path = DAILY_DIR / 'us_daily.parquet'
    intl_path = DAILY_DIR / 'intl_daily.parquet'

    if us_path.exists():
        data['US'] = pd.read_parquet(us_path)
        print(f"US Daily: {len(data['US'])} days, factors: {list(data['US'].columns)}")

    if intl_path.exists():
        data['INTL'] = pd.read_parquet(intl_path)
        print(f"INTL Daily: {len(data['INTL'])} days, factors: {list(data['INTL'].columns)}")

    return data


def create_features_daily(returns: pd.DataFrame, windows: list = [5, 21, 63]):
    """Create features for daily data with appropriate windows."""
    features = pd.DataFrame(index=returns.index)

    for col in returns.columns:
        r = returns[col]
        for w in windows:
            features[f'{col}_ret_{w}d_lag1'] = r.rolling(w).mean().shift(1)
            features[f'{col}_vol_{w}d_lag1'] = r.rolling(w).std().shift(1)

    return features.dropna()


def create_crash_target_daily(returns: pd.Series, window: int = 252,
                               threshold_pct: float = 0.10):
    """Rolling quantile crash target for daily data."""
    targets = pd.Series(index=returns.index, dtype=float)

    for i in range(window, len(returns)):
        historical = returns.iloc[i-window:i]
        threshold = historical.quantile(threshold_pct)
        targets.iloc[i] = 1 if returns.iloc[i] < threshold else 0

    return targets.dropna().astype(int)


def run_experiment(X_train, y_train, X_test, y_test,
                   X_target, y_target, target_regimes,
                   source_regimes, method='rf', epochs=30):
    """Run a single experiment."""

    if method == 'rf':
        model = RandomForestClassifier(n_estimators=100, max_depth=5,
                                       min_samples_leaf=10,
                                       class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)

        source_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        target_auc = roc_auc_score(y_target, model.predict_proba(X_target)[:, 1])

    elif method == 'mmd':
        source_ds = CrowdingDataset(X_train, y_train, 0)
        source_loader = torch.utils.data.DataLoader(
            source_ds, batch_size=64, shuffle=True, drop_last=True
        )

        target_ds = CrowdingDataset(X_target, y_target, 1)
        target_loader = torch.utils.data.DataLoader(
            target_ds, batch_size=64, shuffle=True, drop_last=True
        )

        model = MMDNet(X_train.shape[1], hidden_dim=64, num_layers=2)
        trainer = MMDTrainer(model, device='cpu', lr=1e-3)

        for _ in range(epochs):
            trainer.train_epoch(source_loader, target_loader, lambda_mmd=0.5)

        model.eval()
        with torch.no_grad():
            source_auc = roc_auc_score(y_test,
                model.predict(torch.FloatTensor(X_test)).numpy().flatten())
            target_auc = roc_auc_score(y_target,
                model.predict(torch.FloatTensor(X_target)).numpy().flatten())

    elif method == 'temporal_mmd':
        source_ds = CrowdingDataset(X_train, y_train, 0)
        source_loader = torch.utils.data.DataLoader(
            source_ds, batch_size=64, shuffle=True, drop_last=True
        )

        target_ds = CrowdingDataset(X_target, y_target, 1)
        target_loader = torch.utils.data.DataLoader(
            target_ds, batch_size=64, shuffle=True, drop_last=True
        )

        model = TemporalMMDNet(X_train.shape[1], hidden_dim=64, num_layers=2, num_regimes=2)
        trainer = TemporalMMDTrainer(model, device='cpu', lr=1e-3)

        for _ in range(epochs):
            trainer.train_epoch(source_loader, target_loader,
                              source_regimes, target_regimes, lambda_mmd=0.5)

        model.eval()
        with torch.no_grad():
            source_auc = roc_auc_score(y_test,
                model.predict(torch.FloatTensor(X_test)).numpy().flatten())
            target_auc = roc_auc_score(y_target,
                model.predict(torch.FloatTensor(X_target)).numpy().flatten())

    return source_auc, target_auc


def main():
    print("=" * 70)
    print("KDD 2026: DAILY DATA SCALE-UP EXPERIMENT")
    print("=" * 70)

    # Load data
    print("\nLoading daily data...")
    data = load_daily_data()

    if 'US' not in data or 'INTL' not in data:
        print("Missing data! Run data download first.")
        return

    # Get common factors
    common_factors = list(set(data['US'].columns) & set(data['INTL'].columns))
    print(f"\nCommon factors: {common_factors}")

    # Prepare features
    print("\nPreparing features...")

    us_data = data['US'][common_factors]
    intl_data = data['INTL'][common_factors]

    factor = 'Mom'  # Use momentum as target

    us_features = create_features_daily(us_data)
    us_target = create_crash_target_daily(us_data[factor])

    intl_features = create_features_daily(intl_data)
    intl_target = create_crash_target_daily(intl_data[factor])

    # Align
    us_idx = us_features.index.intersection(us_target.index)
    intl_idx = intl_features.index.intersection(intl_target.index)

    X_us = us_features.loc[us_idx].values
    y_us = us_target.loc[us_idx].values

    X_intl = intl_features.loc[intl_idx].values
    y_intl = intl_target.loc[intl_idx].values

    print(f"US: {len(X_us)} samples, {X_us.shape[1]} features")
    print(f"INTL: {len(X_intl)} samples, {X_intl.shape[1]} features")
    print(f"US crash rate: {y_us.mean():.1%}")
    print(f"INTL crash rate: {y_intl.mean():.1%}")

    # Train/test split
    train_size = int(len(X_us) * 0.7)
    X_train = X_us[:train_size]
    y_train = y_us[:train_size]
    X_test = X_us[train_size:]
    y_test = y_us[train_size:]

    # Standardize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std
    X_intl_std = (X_intl - mean) / std

    # Regime detection
    detector = RegimeDetector(lookback=63)  # ~3 months in trading days

    us_raw = us_data.loc[us_idx].values[:train_size]
    source_regimes = detector.detect_regimes_simple(us_raw)

    intl_raw = intl_data.loc[intl_idx].values
    target_regimes = detector.detect_regimes_simple(intl_raw)

    print(f"\nUS regime distribution: {np.bincount(source_regimes)}")
    print(f"INTL regime distribution: {np.bincount(target_regimes)}")

    # Run experiments
    print("\n" + "=" * 70)
    print("EXPERIMENTS")
    print("=" * 70)

    results = {}

    for method in ['rf', 'mmd', 'temporal_mmd']:
        print(f"\nRunning {method}...")

        source_auc, target_auc = run_experiment(
            X_train_std, y_train, X_test_std, y_test,
            X_intl_std, y_intl, target_regimes, source_regimes,
            method=method, epochs=30
        )

        results[method] = {'source': source_auc, 'target': target_auc}
        print(f"  US→US:   {source_auc:.3f}")
        print(f"  US→INTL: {target_auc:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: DAILY DATA RESULTS")
    print("=" * 70)

    print(f"\n{'Method':<20} {'US→US':<12} {'US→INTL':<12} {'Transfer':<12}")
    print("-" * 56)

    for method, aucs in results.items():
        transfer_eff = aucs['target'] / aucs['source'] * 100 if aucs['source'] > 0 else 0
        print(f"{method:<20} {aucs['source']:<12.3f} {aucs['target']:<12.3f} {transfer_eff:<12.1f}%")

    # Improvement
    rf_target = results['rf']['target']
    tmmd_target = results['temporal_mmd']['target']
    improvement = (tmmd_target - rf_target) / rf_target * 100

    print(f"\n** KEY FINDING **")
    print(f"Temporal-MMD vs RF: {improvement:+.1f}% improvement on transfer")
    print(f"Daily data: {len(X_us)} samples (vs ~600 monthly)")
    print(f"Scale increase: {len(X_us)//600:.0f}x")

    return results


if __name__ == '__main__':
    results = main()
