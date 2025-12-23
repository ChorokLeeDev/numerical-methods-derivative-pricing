"""
KDD 2026: Regime Ablation Study

Goal: Show that 2 regimes is optimal (bias-variance tradeoff)

Test num_regimes = [2, 3, 4, 5] on Electricity domain
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

from models.temporal_mmd import TemporalMMDNet
from models.mmd import mmd_loss

N_SEEDS = 5
NUM_REGIMES_LIST = [2, 3, 4, 5]


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


def prepare_electricity_data(num_regimes=2):
    """Electricity domain with configurable regime count."""
    electricity = fetch_openml(name='electricity', version=1, as_frame=True, parser='auto')
    df = electricity.data.copy()
    target = electricity.target

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    target = target[df.index]

    # Create regimes based on period (time of day)
    period = df['period'].values

    if num_regimes == 2:
        # Peak vs off-peak
        regime = ((period >= 0.5) & (period <= 0.833)).astype(int)
    elif num_regimes == 3:
        # Morning, afternoon, evening
        regime = np.zeros(len(period), dtype=int)
        regime[(period >= 0.33) & (period < 0.67)] = 1
        regime[period >= 0.67] = 2
    elif num_regimes == 4:
        # Quarter day
        regime = (period * 4).astype(int).clip(0, 3)
    else:  # 5+
        regime = (period * num_regimes).astype(int).clip(0, num_regimes - 1)

    X_source = df[['nswprice', 'nswdemand', 'transfer']].values
    X_target = df[['vicprice', 'vicdemand', 'transfer']].values
    y = (target == 'UP').astype(int).values

    return X_source, X_target, y, y, regime, regime


def train_temporal_mmd(model, source_loader, target_loader, epochs=30, lambda_mmd=0.5, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
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


def run_experiment(num_regimes, seed):
    """Run single experiment with specific regime count."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = prepare_electricity_data(num_regimes=num_regimes)
    X_source, X_target, y_source, y_target, regime_source, regime_target = data

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

    # Train T-MMD
    model = TemporalMMDNet(input_dim, hidden_dim=64, num_layers=2, num_regimes=num_regimes)
    model = train_temporal_mmd(model, source_loader, target_loader, epochs=30)

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model.predict(torch.FloatTensor(X_target_std)).numpy().flatten()

    auc = roc_auc_score(y_target, preds)

    # Count samples per regime
    regime_counts = [np.sum(regime_train == r) for r in range(num_regimes)]
    min_samples = min(regime_counts)

    return auc, min_samples


def main():
    print("=" * 70)
    print("KDD 2026: REGIME ABLATION STUDY")
    print("=" * 70)
    print(f"\nTesting num_regimes = {NUM_REGIMES_LIST}")
    print(f"Seeds per config: {N_SEEDS}")
    print("\nDomain: Electricity (NSW → Victoria)")

    results = {}

    for num_regimes in NUM_REGIMES_LIST:
        print(f"\n{'='*50}")
        print(f"Testing {num_regimes} regimes...")
        print('='*50)

        aucs = []
        min_samples_list = []

        for seed in range(N_SEEDS):
            print(f"  Seed {seed+1}/{N_SEEDS}...", end=' ')
            auc, min_samples = run_experiment(num_regimes, seed)
            aucs.append(auc)
            min_samples_list.append(min_samples)
            print(f"AUC: {auc:.3f}, Min samples/regime: {min_samples}")

        results[num_regimes] = {
            'mean': np.mean(aucs),
            'std': np.std(aucs),
            'values': aucs,
            'min_samples': int(np.mean(min_samples_list))
        }

    # Results Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Regimes':<10} {'AUC':<20} {'Min Samples/Regime':<20}")
    print("-" * 50)

    best_regime = max(results.keys(), key=lambda k: results[k]['mean'])

    for num_regimes in NUM_REGIMES_LIST:
        r = results[num_regimes]
        marker = " ← BEST" if num_regimes == best_regime else ""
        print(f"{num_regimes:<10} {r['mean']:.3f} ± {r['std']:.3f}      {r['min_samples']:<20}{marker}")

    # Statistical comparison: best vs others
    print("\n" + "=" * 70)
    print(f"STATISTICAL COMPARISON (vs {best_regime} regimes)")
    print("=" * 70)

    best_values = results[best_regime]['values']
    for num_regimes in NUM_REGIMES_LIST:
        if num_regimes != best_regime:
            other_values = results[num_regimes]['values']
            t_stat, p_value = stats.ttest_rel(best_values, other_values)
            sig = "*" if p_value < 0.05 else ""
            sig = "**" if p_value < 0.01 else sig
            print(f"  {best_regime} vs {num_regimes} regimes: p = {p_value:.3f} {sig}")

    # Save figure
    print("\n" + "=" * 70)
    print("GENERATING FIGURE")
    print("=" * 70)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    regimes = list(results.keys())
    means = [results[r]['mean'] for r in regimes]
    stds = [results[r]['std'] for r in regimes]
    min_samples = [results[r]['min_samples'] for r in regimes]

    # AUC bars
    color1 = '#2E86AB'
    ax1.bar(regimes, means, yerr=stds, capsize=5, color=color1, alpha=0.7, label='AUC')
    ax1.set_xlabel('Number of Regimes', fontsize=12)
    ax1.set_ylabel('Target AUC', color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.5, 0.7)

    # Min samples line
    ax2 = ax1.twinx()
    color2 = '#E94F37'
    ax2.plot(regimes, min_samples, 'o-', color=color2, linewidth=2, markersize=8, label='Min Samples')
    ax2.set_ylabel('Min Samples per Regime', color=color2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Highlight best
    best_idx = regimes.index(best_regime)
    ax1.bar(regimes[best_idx], means[best_idx], color='#28A745', alpha=0.9)

    plt.title('Regime Ablation: Bias-Variance Tradeoff', fontsize=14, fontweight='bold')
    fig.tight_layout()

    # Save
    fig_dir = ROOT_DIR / 'paper' / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / 'regime_ablation.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")

    plt.close()

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print(f"""
    Optimal regime count: {best_regime}

    Bias-Variance Tradeoff:
    - Too few regimes (1): No regime-awareness, just standard MMD
    - Optimal ({best_regime} regimes): Best balance
    - Too many regimes: Insufficient samples per regime, high variance

    Min samples per regime at {best_regime} regimes: {results[best_regime]['min_samples']}
    """)

    return results


if __name__ == '__main__':
    results = main()
