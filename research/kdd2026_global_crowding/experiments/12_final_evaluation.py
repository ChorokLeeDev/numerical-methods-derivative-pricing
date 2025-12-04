"""
KDD 2026: Final Evaluation with 3 Regimes + MCD Baseline

Improvements over 09_extended_evaluation.py:
1. Uses 3 regimes (ablation showed this is optimal)
2. Adds MCD baseline
3. Cleaner output for paper
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

from models.mmd import MMDNet, mmd_loss
from models.dann import DANN, DANNTrainer
from models.cdan import CDANNet, CDANTrainer, MCDNet, MCDTrainer
from models.temporal_mmd import TemporalMMDNet

DATA_DIR = ROOT_DIR / 'data'
N_SEEDS = 5
NUM_REGIMES = 3  # Optimal from ablation study


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


def create_3_regimes(values, method='quantile'):
    """Create 3 regimes from continuous values."""
    if method == 'quantile':
        q33 = np.percentile(values, 33)
        q66 = np.percentile(values, 66)
        regime = np.zeros(len(values), dtype=int)
        regime[(values > q33) & (values <= q66)] = 1
        regime[values > q66] = 2
    else:  # equal intervals
        regime = (values * 3).astype(int).clip(0, 2)
    return regime


def prepare_electricity_data():
    """Electricity with 3 regimes (morning/afternoon/evening)."""
    print("\n--- ELECTRICITY DOMAIN (3 regimes) ---")

    electricity = fetch_openml(name='electricity', version=1, as_frame=True, parser='auto')
    df = electricity.data.copy()
    target = electricity.target

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    target = target[df.index]

    # 3 regimes based on time of day
    period = df['period'].values
    regime = np.zeros(len(period), dtype=int)
    regime[(period >= 0.33) & (period < 0.67)] = 1
    regime[period >= 0.67] = 2

    X_source = df[['nswprice', 'nswdemand', 'transfer']].values
    X_target = df[['vicprice', 'vicdemand', 'transfer']].values
    y = (target == 'UP').astype(int).values

    regime_counts = [np.sum(regime == r) for r in range(3)]
    print(f"  Samples: {len(df)}")
    print(f"  Regimes: Morning={regime_counts[0]}, Afternoon={regime_counts[1]}, Evening={regime_counts[2]}")

    return X_source, X_target, y, y, regime, regime


def prepare_gas_sensor_data():
    """Gas sensor with 3 concentration regimes."""
    print("\n--- GAS SENSOR DOMAIN (3 regimes) ---")

    np.random.seed(43)
    n = 8000

    concentration = np.random.exponential(100, n)

    # Source: Fresh sensors
    sensor1_a = concentration * 1.0 + np.random.randn(n) * 10
    sensor2_a = concentration * 0.8 + np.random.randn(n) * 8
    sensor3_a = concentration * 1.2 + np.random.randn(n) * 12

    # Target: Sensors with drift
    drift = 0.9
    sensor1_b = concentration * 1.0 * drift + np.random.randn(n) * 15
    sensor2_b = concentration * 0.8 * drift + np.random.randn(n) * 12
    sensor3_b = concentration * 1.2 * drift + np.random.randn(n) * 18

    X_source = np.column_stack([sensor1_a, sensor2_a, sensor3_a])
    X_target = np.column_stack([sensor1_b, sensor2_b, sensor3_b])

    y_source = (concentration > np.percentile(concentration, 70)).astype(int)
    y_target = y_source.copy()

    # 3 regimes: low/medium/high concentration
    regime = create_3_regimes(concentration)

    regime_counts = [np.sum(regime == r) for r in range(3)]
    print(f"  Samples: {len(X_source)}")
    print(f"  Regimes: Low={regime_counts[0]}, Med={regime_counts[1]}, High={regime_counts[2]}")

    return X_source, X_target, y_source, y_target, regime, regime


def prepare_activity_data():
    """Activity recognition with 3 activity regimes."""
    print("\n--- ACTIVITY DOMAIN (3 regimes) ---")

    np.random.seed(44)
    n = 12000

    # 3 activities: sitting, walking, running
    activity = np.random.choice([0, 1, 2], n, p=[0.4, 0.35, 0.25])

    # Person A
    accel_x_a = np.random.randn(n) * 2
    accel_y_a = np.random.randn(n) * 2
    accel_z_a = 9.8 + np.random.randn(n) * 0.5

    accel_x_a[activity == 1] += np.sin(np.arange((activity == 1).sum()) * 0.5) * 3
    accel_x_a[activity == 2] += np.sin(np.arange((activity == 2).sum()) * 1.0) * 6

    # Person B (different dynamics)
    accel_x_b = np.random.randn(n) * 2.2
    accel_y_b = np.random.randn(n) * 1.8
    accel_z_b = 9.8 + np.random.randn(n) * 0.6

    accel_x_b[activity == 1] += np.sin(np.arange((activity == 1).sum()) * 0.5) * 3.5
    accel_x_b[activity == 2] += np.sin(np.arange((activity == 2).sum()) * 1.0) * 5.5

    X_source = np.column_stack([accel_x_a, accel_y_a, accel_z_a])
    X_target = np.column_stack([accel_x_b, accel_y_b, accel_z_b])

    y_source = (activity >= 1).astype(int)
    y_target = y_source.copy()

    # 3 regimes = 3 activities
    regime = activity

    regime_counts = [np.sum(regime == r) for r in range(3)]
    print(f"  Samples: {len(X_source)}")
    print(f"  Regimes: Sitting={regime_counts[0]}, Walking={regime_counts[1]}, Running={regime_counts[2]}")

    return X_source, X_target, y_source, y_target, regime, regime


def train_mmd(model, source_loader, target_loader, epochs=30, lambda_mmd=0.5):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        target_iter = iter(target_loader)
        for source_x, source_y, _, _ in source_loader:
            try:
                target_x, _, _, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_x, _, _, _ = next(target_iter)

            source_logits, source_features = model(source_x)
            _, target_features = model(target_x)

            task_loss = criterion(source_logits.squeeze(), source_y.float())
            mmd = mmd_loss(source_features, target_features)
            loss = task_loss + lambda_mmd * mmd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def train_temporal_mmd(model, source_loader, target_loader, epochs=30, lambda_mmd=0.5):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        target_iter = iter(target_loader)
        for source_x, source_y, source_reg, _ in source_loader:
            try:
                target_x, _, target_reg, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_x, _, target_reg, _ = next(target_iter)

            source_logits, source_features = model(source_x)
            _, target_features = model(target_x)

            task_loss = criterion(source_logits.squeeze(), source_y.float())
            mmd = model.temporal_mmd(source_features, target_features, source_reg, target_reg)
            loss = task_loss + lambda_mmd * mmd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def run_single_experiment(X_source, X_target, y_source, y_target,
                          regime_source, regime_target, seed=42):
    """Run one experiment with all 6 methods."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Train/test split
    n = len(X_source)
    train_idx = int(n * 0.7)

    X_train = X_source[:train_idx]
    y_train = y_source[:train_idx]
    regime_train = regime_source[:train_idx]

    # Standardize
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_target_std = scaler.transform(X_target)

    # Data loaders
    source_ds = RegimeAwareDataset(X_train_std, y_train, regime_train, 0)
    source_loader = torch.utils.data.DataLoader(source_ds, batch_size=64, shuffle=True, drop_last=True)

    target_ds = RegimeAwareDataset(X_target_std, y_target, regime_target, 1)
    target_loader = torch.utils.data.DataLoader(target_ds, batch_size=64, shuffle=True, drop_last=True)

    input_dim = X_train_std.shape[1]
    results = {}

    # 1. RF
    rf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                class_weight='balanced', random_state=seed)
    rf.fit(X_train_std, y_train)
    results['RF'] = roc_auc_score(y_target, rf.predict_proba(X_target_std)[:, 1])

    # 2. MMD
    mmd_model = MMDNet(input_dim, hidden_dim=64, num_layers=2)
    mmd_model = train_mmd(mmd_model, source_loader, target_loader, epochs=30)
    mmd_model.eval()
    with torch.no_grad():
        results['MMD'] = roc_auc_score(y_target,
            mmd_model.predict(torch.FloatTensor(X_target_std)).numpy().flatten())

    # 3. DANN
    dann_model = DANN(input_dim, num_domains=2, hidden_dim=64, num_layers=2)
    dann_trainer = DANNTrainer(dann_model, lr=1e-3)
    for epoch in range(30):
        dann_trainer.train_epoch(source_loader, target_loader, epoch, 30, lambda_domain=1.0)
    dann_model.eval()
    with torch.no_grad():
        results['DANN'] = roc_auc_score(y_target,
            dann_model.predict(torch.FloatTensor(X_target_std)).numpy().flatten())

    # 4. CDAN
    cdan_model = CDANNet(input_dim, hidden_dim=64, num_layers=2)
    cdan_trainer = CDANTrainer(cdan_model, lr=1e-3)
    for _ in range(30):
        cdan_trainer.train_epoch(source_loader, target_loader, lambda_domain=1.0)
    cdan_model.eval()
    with torch.no_grad():
        results['CDAN'] = roc_auc_score(y_target,
            cdan_model.predict(torch.FloatTensor(X_target_std)).numpy().flatten())

    # 5. MCD
    mcd_model = MCDNet(input_dim, hidden_dim=64, num_layers=2)
    mcd_trainer = MCDTrainer(mcd_model, lr=1e-3)
    for _ in range(30):
        mcd_trainer.train_epoch(source_loader, target_loader)
    mcd_model.eval()
    with torch.no_grad():
        results['MCD'] = roc_auc_score(y_target,
            mcd_model.predict(torch.FloatTensor(X_target_std)).numpy().flatten())

    # 6. T-MMD (3 regimes)
    tmmd_model = TemporalMMDNet(input_dim, hidden_dim=64, num_layers=2, num_regimes=NUM_REGIMES)
    tmmd_model = train_temporal_mmd(tmmd_model, source_loader, target_loader, epochs=30)
    tmmd_model.eval()
    with torch.no_grad():
        results['T-MMD'] = roc_auc_score(y_target,
            tmmd_model.predict(torch.FloatTensor(X_target_std)).numpy().flatten())

    return results


def main():
    print("=" * 70)
    print("KDD 2026: FINAL EVALUATION (3 REGIMES + MCD)")
    print("=" * 70)
    print(f"\nRegimes: {NUM_REGIMES} (optimal from ablation)")
    print(f"Seeds: {N_SEEDS}")
    print(f"Methods: RF, MMD, DANN, CDAN, MCD, T-MMD")

    # Prepare domains
    domains = {}

    domain_funcs = [
        ('Electricity', prepare_electricity_data),
        ('GasSensor', prepare_gas_sensor_data),
        ('Activity', prepare_activity_data),
    ]

    for name, func in domain_funcs:
        try:
            data = func()
            if data is not None:
                domains[name] = data
        except Exception as e:
            print(f"  {name} error: {e}")

    # Run experiments
    all_results = {}
    methods = ['RF', 'MMD', 'DANN', 'CDAN', 'MCD', 'T-MMD']

    for domain_name, data in domains.items():
        print(f"\n{'='*60}")
        print(f"Running {domain_name} ({N_SEEDS} seeds)...")
        print('='*60)

        domain_results = []
        for seed in range(N_SEEDS):
            print(f"  Seed {seed+1}/{N_SEEDS}...", end=' ')
            result = run_single_experiment(*data, seed=seed)
            domain_results.append(result)
            print(f"T-MMD: {result['T-MMD']:.3f}")

        # Compute statistics
        stats_dict = {}
        for method in methods:
            values = [r[method] for r in domain_results]
            stats_dict[method] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        all_results[domain_name] = stats_dict

    # Results Summary
    print("\n" + "=" * 70)
    print("RESULTS WITH STANDARD DEVIATIONS")
    print("=" * 70)

    print(f"\n{'Domain':<12}", end='')
    for m in methods:
        print(f"{m:<14}", end='')
    print()
    print("-" * 96)

    for domain, stats in all_results.items():
        print(f"{domain:<12}", end='')
        best_mean = max(s['mean'] for s in stats.values())
        for m in methods:
            mean = stats[m]['mean']
            std = stats[m]['std']
            if mean == best_mean:
                print(f"**{mean:.3f}**±{std:.3f} ", end='')
            else:
                print(f"{mean:.3f}±{std:.3f}  ", end='')
        print()

    # Statistical Significance
    print("\n" + "=" * 70)
    print("PAIRED T-TEST: T-MMD vs BASELINES (p-values)")
    print("=" * 70)

    print(f"\n{'Domain':<12} {'vs RF':<10} {'vs MMD':<10} {'vs DANN':<10} {'vs CDAN':<10} {'vs MCD':<10}")
    print("-" * 72)

    for domain, stats in all_results.items():
        print(f"{domain:<12}", end='')
        tmmd_values = stats['T-MMD']['values']

        for baseline in ['RF', 'MMD', 'DANN', 'CDAN', 'MCD']:
            baseline_values = stats[baseline]['values']
            _, p_val = stats.ttest_rel(tmmd_values, baseline_values) if hasattr(stats, 'ttest_rel') else (0, 1)
            try:
                from scipy import stats as scipy_stats
                _, p_val = scipy_stats.ttest_rel(tmmd_values, baseline_values)
            except:
                p_val = 1.0

            sig = '*' if p_val < 0.05 else ''
            sig = '**' if p_val < 0.01 else sig
            print(f"{p_val:.3f}{sig:<5}", end='')
        print()

    # Average improvement
    print("\n" + "=" * 70)
    print("AVERAGE IMPROVEMENT")
    print("=" * 70)

    for baseline in ['RF', 'MMD', 'DANN', 'CDAN', 'MCD']:
        improvements = []
        for domain, stats in all_results.items():
            tmmd = stats['T-MMD']['mean']
            base = stats[baseline]['mean']
            imp = (tmmd - base) / base * 100
            improvements.append(imp)
        print(f"  T-MMD vs {baseline}: {np.mean(improvements):+.2f}%")

    return all_results


if __name__ == '__main__':
    results = main()
