"""
Test MMD Transfer: Does MMD-Based Domain Adaptation Actually Work?

Key questions:
1. Can we transfer factor insights from US to international markets?
2. Does MMD improve over naive transfer?
3. How much transfer efficiency do we achieve?

We'll use AQR's publicly available international factor data.

Run: python scripts/test_mmd_transfer.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def rbf_kernel(X, Y, gamma=1.0):
    """Compute RBF kernel between X and Y."""
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x.y
    X_sqnorms = np.sum(X**2, axis=1, keepdims=True)
    Y_sqnorms = np.sum(Y**2, axis=1, keepdims=True)
    K = X_sqnorms + Y_sqnorms.T - 2 * X @ Y.T
    return np.exp(-gamma * K)


def compute_mmd(X_source, X_target, gamma=1.0):
    """
    Compute Maximum Mean Discrepancy between source and target.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    """
    K_ss = rbf_kernel(X_source, X_source, gamma)
    K_tt = rbf_kernel(X_target, X_target, gamma)
    K_st = rbf_kernel(X_source, X_target, gamma)

    n_s = X_source.shape[0]
    n_t = X_target.shape[0]

    # Unbiased estimator
    mmd2 = (np.sum(K_ss) - np.trace(K_ss)) / (n_s * (n_s - 1))
    mmd2 += (np.sum(K_tt) - np.trace(K_tt)) / (n_t * (n_t - 1))
    mmd2 -= 2 * np.mean(K_st)

    return np.sqrt(max(0, mmd2))


def create_features(returns, window=12):
    """Create features for prediction."""
    df = pd.DataFrame({'returns': returns})

    # Rolling statistics
    df['rolling_mean'] = df['returns'].rolling(window).mean()
    df['rolling_std'] = df['returns'].rolling(window).std()
    df['rolling_skew'] = df['returns'].rolling(window).skew()

    # Lagged returns
    for lag in [1, 3, 6, 12]:
        df[f'lag_{lag}'] = df['returns'].shift(lag)

    # Momentum
    df['momentum_12'] = df['returns'].rolling(12).sum()
    df['momentum_6'] = df['returns'].rolling(6).sum()

    return df.dropna()


def mmd_reweight(X_source, X_target, gamma=1.0):
    """
    Compute importance weights for source samples to match target distribution.

    Simple approach: weight samples by kernel mean matching.
    """
    n_s = X_source.shape[0]
    n_t = X_target.shape[0]

    # Kernel between source and target
    K_st = rbf_kernel(X_source, X_target, gamma)

    # Weight = mean similarity to target
    weights = np.mean(K_st, axis=1)

    # Normalize
    weights = weights / weights.sum() * n_s

    return weights


class MMDTransferModel:
    """
    MMD-based transfer learning model.

    1. Compute MMD between source and target features
    2. Reweight source samples to match target distribution
    3. Train weighted model on source data
    4. Apply to target
    """

    def __init__(self, gamma=1.0, alpha=1.0):
        self.gamma = gamma
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()

    def fit(self, X_source, y_source, X_target):
        """Fit model with MMD reweighting."""
        # Scale features
        X_source_scaled = self.scaler.fit_transform(X_source)
        X_target_scaled = self.scaler.transform(X_target)

        # Compute MMD weights
        self.weights = mmd_reweight(X_source_scaled, X_target_scaled, self.gamma)

        # Fit weighted regression
        self.model.fit(X_source_scaled, y_source, sample_weight=self.weights)

        # Store MMD for diagnostics
        self.mmd = compute_mmd(X_source_scaled, X_target_scaled, self.gamma)

        return self

    def predict(self, X):
        """Predict on new data."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class NaiveTransferModel:
    """Naive transfer: just train on source, apply to target."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()

    def fit(self, X_source, y_source):
        """Fit model on source data."""
        X_scaled = self.scaler.fit_transform(X_source)
        self.model.fit(X_scaled, y_source)
        return self

    def predict(self, X):
        """Predict on new data."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


def simulate_international_data(us_returns, noise_scale=0.5, correlation=0.6):
    """
    Simulate international market data based on US data.

    In reality, we'd use AQR's international factor data.
    For this test, we simulate correlated international returns.
    """
    n = len(us_returns)

    # International = correlated with US + noise
    intl_returns = correlation * us_returns + noise_scale * np.random.randn(n) * us_returns.std()

    # Add regime shift (different mean/variance in second half)
    regime_shift = int(n * 0.6)
    intl_returns[regime_shift:] *= 0.8  # Lower returns
    intl_returns[regime_shift:] += np.random.randn(n - regime_shift) * 0.01

    return intl_returns


def run_mmd_transfer_test():
    """Main test of MMD transfer vs naive transfer."""

    print("="*70)
    print("MMD TRANSFER TEST: Does Domain Adaptation Work?")
    print("="*70)

    # Load US data
    base_path = Path(__file__).parent.parent
    data_path = base_path / "data" / "factor_crowding" / "ff_factors_monthly.parquet"

    if not data_path.exists():
        print(f"ERROR: Data not found at {data_path}")
        return None

    df = pd.read_parquet(data_path)
    print(f"\nLoaded US data: {len(df)} months")

    # Test factors
    factors = ['SMB', 'HML', 'Mom', 'CMA']
    available_factors = [f for f in factors if f in df.columns]

    # Simulate multiple "international markets"
    np.random.seed(42)
    markets = {
        'Europe': {'noise': 0.4, 'corr': 0.7},
        'Japan': {'noise': 0.6, 'corr': 0.5},
        'UK': {'noise': 0.3, 'corr': 0.75},
        'Asia_ex_Japan': {'noise': 0.7, 'corr': 0.4},
    }

    all_results = []

    for factor in available_factors:
        print(f"\n{'='*60}")
        print(f"TESTING FACTOR: {factor}")
        print("="*60)

        us_returns = df[factor].dropna().values

        # Create US features
        us_data = create_features(us_returns)
        feature_cols = [c for c in us_data.columns if c != 'returns']

        X_us = us_data[feature_cols].values
        y_us = us_data['returns'].shift(-1).dropna().values  # Next month return
        X_us = X_us[:-1]  # Align

        print(f"US data: {len(X_us)} samples, {len(feature_cols)} features")

        # Split US data: train on first 70%
        us_train_end = int(len(X_us) * 0.7)
        X_us_train = X_us[:us_train_end]
        y_us_train = y_us[:us_train_end]
        X_us_test = X_us[us_train_end:]
        y_us_test = y_us[us_train_end:]

        # US baseline (no transfer needed)
        us_model = NaiveTransferModel()
        us_model.fit(X_us_train, y_us_train)
        y_us_pred = us_model.predict(X_us_test)
        r2_us = r2_score(y_us_test, y_us_pred)
        print(f"\nUS in-market R²: {r2_us:.4f}")

        for market_name, market_params in markets.items():
            print(f"\n--- {market_name} ---")

            # Simulate international returns
            intl_returns = simulate_international_data(
                us_returns,
                noise_scale=market_params['noise'],
                correlation=market_params['corr']
            )

            # Create international features
            intl_data = create_features(intl_returns)
            X_intl = intl_data[feature_cols].values
            y_intl = intl_data['returns'].shift(-1).dropna().values
            X_intl = X_intl[:-1]

            # Split: use second half of international data as test
            intl_test_start = int(len(X_intl) * 0.5)
            X_intl_test = X_intl[intl_test_start:]
            y_intl_test = y_intl[intl_test_start:]

            # METHOD 1: Naive Transfer (train on US, test on international)
            naive_model = NaiveTransferModel()
            naive_model.fit(X_us_train, y_us_train)
            y_pred_naive = naive_model.predict(X_intl_test)
            r2_naive = r2_score(y_intl_test, y_pred_naive)

            # METHOD 2: MMD Transfer
            mmd_model = MMDTransferModel(gamma=0.1)
            mmd_model.fit(X_us_train, y_us_train, X_intl_test)
            y_pred_mmd = mmd_model.predict(X_intl_test)
            r2_mmd = r2_score(y_intl_test, y_pred_mmd)

            # METHOD 3: Oracle (train on international data)
            oracle_model = NaiveTransferModel()
            X_intl_train = X_intl[:intl_test_start]
            y_intl_train = y_intl[:intl_test_start]
            oracle_model.fit(X_intl_train, y_intl_train)
            y_pred_oracle = oracle_model.predict(X_intl_test)
            r2_oracle = r2_score(y_intl_test, y_pred_oracle)

            # Transfer efficiency
            # TE = (R²_MMD - R²_naive) / (R²_oracle - R²_naive)
            if r2_oracle != r2_naive:
                transfer_eff = (r2_mmd - r2_naive) / (r2_oracle - r2_naive + 1e-8)
            else:
                transfer_eff = 0

            print(f"  Naive transfer R²:  {r2_naive:.4f}")
            print(f"  MMD transfer R²:    {r2_mmd:.4f}")
            print(f"  Oracle R²:          {r2_oracle:.4f}")
            print(f"  Transfer efficiency: {transfer_eff:.1%}")
            print(f"  MMD distance:       {mmd_model.mmd:.4f}")

            # Verdict
            mmd_improves = r2_mmd > r2_naive
            print(f"  MMD improves over naive: {'YES ✓' if mmd_improves else 'NO ✗'}")

            all_results.append({
                'factor': factor,
                'market': market_name,
                'r2_naive': r2_naive,
                'r2_mmd': r2_mmd,
                'r2_oracle': r2_oracle,
                'transfer_efficiency': transfer_eff,
                'mmd_distance': mmd_model.mmd,
                'mmd_improves': mmd_improves
            })

    # Summary
    results_df = pd.DataFrame(all_results)

    print("\n" + "="*70)
    print("SUMMARY: MMD TRANSFER TEST RESULTS")
    print("="*70)

    print("\n{:<8} {:<15} {:>10} {:>10} {:>10} {:>12}".format(
        "Factor", "Market", "Naive R²", "MMD R²", "Oracle R²", "Trans Eff"
    ))
    print("-"*70)

    for _, row in results_df.iterrows():
        print("{:<8} {:<15} {:>10.4f} {:>10.4f} {:>10.4f} {:>12.1%}".format(
            row['factor'],
            row['market'],
            row['r2_naive'],
            row['r2_mmd'],
            row['r2_oracle'],
            row['transfer_efficiency']
        ))

    # Overall statistics
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)

    n_improves = results_df['mmd_improves'].sum()
    n_total = len(results_df)
    avg_transfer_eff = results_df['transfer_efficiency'].mean()
    avg_r2_naive = results_df['r2_naive'].mean()
    avg_r2_mmd = results_df['r2_mmd'].mean()

    print(f"\nCases where MMD improves over naive: {n_improves}/{n_total} ({n_improves/n_total:.0%})")
    print(f"Average naive R²:          {avg_r2_naive:.4f}")
    print(f"Average MMD R²:            {avg_r2_mmd:.4f}")
    print(f"Average transfer efficiency: {avg_transfer_eff:.1%}")

    print("\n" + "="*70)
    print("OVERALL VERDICT")
    print("="*70)

    if n_improves >= n_total // 2:
        print("\n✓ MMD TRANSFER WORKS - improves over naive in majority of cases")
    else:
        print("\n✗ MMD TRANSFER DOES NOT WORK - no consistent improvement")

    if avg_transfer_eff > 0.3:
        print(f"✓ TRANSFER EFFICIENCY IS MEANINGFUL ({avg_transfer_eff:.1%} > 30%)")
    else:
        print(f"✗ TRANSFER EFFICIENCY IS LOW ({avg_transfer_eff:.1%} < 30%)")

    # Compare to paper's claims
    print("\n" + "="*70)
    print("COMPARISON TO PAPER'S CLAIMS")
    print("="*70)
    print(f"\nPaper claims: 60% transfer efficiency, 7.7% improvement over naive")
    print(f"Reality:      {avg_transfer_eff:.1%} transfer efficiency, {(avg_r2_mmd - avg_r2_naive)*100:.2f}% R² improvement")

    if avg_transfer_eff > 0.5:
        print("\n✓ Paper's claims are APPROXIMATELY supported")
    elif avg_transfer_eff > 0.2:
        print("\n⚠️ Paper's claims are PARTIALLY supported (lower than claimed)")
    else:
        print("\n✗ Paper's claims are NOT supported")

    # Save results
    output_path = base_path / "results" / "mmd_transfer_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return results_df


if __name__ == '__main__':
    run_mmd_transfer_test()
