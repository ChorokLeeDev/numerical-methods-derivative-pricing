"""
KDD 2026 Core Experiments

Three key research questions:
1. Cross-Region Transfer: Does US model work in other regions?
2. Regional Decay Rates: Is crowding slower in emerging markets?
3. Global Taxonomy: Does Mechanical vs Judgment hold globally?
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data' / 'global_factors'
FACTOR_CROWDING_DIR = Path(__file__).parent.parent.parent / 'factor_crowding'

# Add factor_crowding src to path
sys.path.insert(0, str(FACTOR_CROWDING_DIR / 'src'))


def load_all_regions():
    """Load all regional factor data."""
    data = {}
    for path in DATA_DIR.glob('*_factors.parquet'):
        region = path.stem.replace('_factors', '').upper()
        df = pd.read_parquet(path)
        if len(df) > 100:  # Need enough data
            data[region] = df
    return data


def rolling_sharpe(returns: pd.Series, window: int = 36) -> pd.Series:
    """Compute rolling annualized Sharpe ratio."""
    return (returns.rolling(window).mean() /
            returns.rolling(window).std() * np.sqrt(12))


def hyperbolic_decay(t, K, lam):
    """α(t) = K / (1 + λt)"""
    return K / (1 + lam * t)


def fit_decay_model(sharpe: pd.Series):
    """Fit hyperbolic decay and return R², K, λ."""
    sharpe = sharpe.dropna()
    if len(sharpe) < 50:
        return None, None, None

    t = np.arange(len(sharpe))
    y = sharpe.values

    # Only positive values
    mask = y > 0
    if mask.sum() < 30:
        return None, None, None

    t_pos, y_pos = t[mask], y[mask]

    try:
        popt, _ = curve_fit(hyperbolic_decay, t_pos, y_pos,
                           p0=[1.5, 0.01], bounds=([0, 0], [10, 0.5]))
        K, lam = popt

        # R²
        y_pred = hyperbolic_decay(t_pos, K, lam)
        ss_res = np.sum((y_pos - y_pred) ** 2)
        ss_tot = np.sum((y_pos - np.mean(y_pos)) ** 2)
        r2 = 1 - ss_res / ss_tot

        return r2, K, lam
    except:
        return None, None, None


def create_features(returns: pd.DataFrame, window: int = 12):
    """Create ML features from factor returns."""
    features = pd.DataFrame(index=returns.index)

    for col in returns.columns:
        r = returns[col]
        # Returns
        features[f'{col}_ret_1m'] = r
        features[f'{col}_ret_3m'] = r.rolling(3).mean()
        features[f'{col}_ret_12m'] = r.rolling(12).mean()
        # Volatility
        features[f'{col}_vol_3m'] = r.rolling(3).std()
        features[f'{col}_vol_12m'] = r.rolling(12).std()

    return features.dropna()


def create_crash_target(returns: pd.Series, threshold_pct: float = 0.10):
    """Binary target: 1 if return in bottom threshold_pct."""
    threshold = returns.quantile(threshold_pct)
    return (returns < threshold).astype(int)


# =============================================================================
# EXPERIMENT 1: Cross-Region Transfer Learning
# =============================================================================

def experiment_cross_region_transfer(data: dict):
    """
    Train on US, test on other regions.

    Question: Does a US-trained model generalize globally?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: CROSS-REGION TRANSFER LEARNING")
    print("=" * 70)
    print("\nTrain on US, test on each region")

    if 'US' not in data:
        print("US data not available!")
        return None

    us_data = data['US']

    # Use Mom for primary experiment
    if 'Mom' not in us_data.columns:
        print("Momentum not available in US data!")
        return None

    # Create US features and target
    us_features = create_features(us_data)
    us_target = create_crash_target(us_data['Mom'].reindex(us_features.index))

    # Align
    common_idx = us_features.index.intersection(us_target.index)
    us_features = us_features.loc[common_idx]
    us_target = us_target.loc[common_idx]

    # Train on US (first 80%)
    train_size = int(len(us_features) * 0.8)
    X_train = us_features.iloc[:train_size]
    y_train = us_target.iloc[:train_size]

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                   class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # Test on US (last 20%)
    X_test_us = us_features.iloc[train_size:]
    y_test_us = us_target.iloc[train_size:]
    us_auc = roc_auc_score(y_test_us, model.predict_proba(X_test_us)[:, 1])

    results = {'US (test)': us_auc}

    # Test on other regions
    for region, region_data in data.items():
        if region == 'US':
            continue

        if 'Mom' not in region_data.columns:
            continue

        # Create features (must have same columns as US)
        region_features = create_features(region_data)

        # Ensure same columns
        common_cols = [c for c in X_train.columns if c in region_features.columns]
        if len(common_cols) < len(X_train.columns) * 0.5:
            print(f"  {region}: Insufficient overlapping features, skipping")
            continue

        # Use only common columns
        region_features = region_features[common_cols]
        X_train_subset = X_train[common_cols]

        # Retrain on common columns
        model_subset = RandomForestClassifier(n_estimators=100, max_depth=10,
                                              class_weight='balanced', random_state=42)
        model_subset.fit(X_train_subset, y_train)

        # Create target for region
        region_target = create_crash_target(region_data['Mom'].reindex(region_features.index))
        common_idx = region_features.index.intersection(region_target.index)

        if len(common_idx) < 50:
            continue

        X_region = region_features.loc[common_idx]
        y_region = region_target.loc[common_idx]

        try:
            auc = roc_auc_score(y_region, model_subset.predict_proba(X_region)[:, 1])
            results[region] = auc
        except:
            pass

    # Print results
    print(f"\n{'Region':<15} {'AUC':<10} {'vs US':<10}")
    print("-" * 35)

    us_baseline = results.get('US (test)', 0.5)
    for region, auc in sorted(results.items(), key=lambda x: -x[1]):
        diff = auc - us_baseline
        diff_str = f"{diff:+.3f}" if region != 'US (test)' else "-"
        print(f"{region:<15} {auc:<10.3f} {diff_str:<10}")

    # Summary
    other_aucs = [v for k, v in results.items() if k != 'US (test)']
    if other_aucs:
        mean_transfer = np.mean(other_aucs)
        transfer_rate = mean_transfer / us_baseline if us_baseline > 0 else 0
        print(f"\nTransfer efficiency: {transfer_rate:.1%}")
        print(f"Mean AUC on other regions: {mean_transfer:.3f}")

    return results


# =============================================================================
# EXPERIMENT 2: Regional Decay Rate Comparison
# =============================================================================

def experiment_regional_decay_rates(data: dict):
    """
    Compare decay rates (λ) across regions.

    Hypothesis: Emerging markets have slower decay (lower λ).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: REGIONAL DECAY RATE COMPARISON")
    print("=" * 70)
    print("\nHypothesis: Less efficient markets → slower crowding (lower λ)")

    results = []

    for region, region_data in data.items():
        for factor in region_data.columns:
            sharpe = rolling_sharpe(region_data[factor], window=36)
            r2, K, lam = fit_decay_model(sharpe)

            if r2 is not None:
                results.append({
                    'region': region,
                    'factor': factor,
                    'r2': r2,
                    'K': K,
                    'lambda': lam,
                    'half_life': np.log(2) / lam if lam > 0 else np.inf
                })

    df = pd.DataFrame(results)

    if len(df) == 0:
        print("No decay models fitted successfully!")
        return None

    # Summary by region
    print(f"\n{'Region':<12} {'Factors':<8} {'Mean R²':<10} {'Mean λ':<10} {'Half-life':<12}")
    print("-" * 52)

    region_summary = df.groupby('region').agg({
        'r2': ['count', 'mean'],
        'lambda': 'mean',
        'half_life': 'mean'
    }).round(3)

    for region in region_summary.index:
        count = int(region_summary.loc[region, ('r2', 'count')])
        mean_r2 = region_summary.loc[region, ('r2', 'mean')]
        mean_lam = region_summary.loc[region, ('lambda', 'mean')]
        mean_hl = region_summary.loc[region, ('half_life', 'mean')]
        print(f"{region:<12} {count:<8} {mean_r2:<10.3f} {mean_lam:<10.4f} {mean_hl:<12.1f} months")

    # Focus on Momentum
    print("\n" + "-" * 52)
    print("MOMENTUM DECAY BY REGION:")
    print("-" * 52)

    mom_df = df[df['factor'] == 'Mom']
    for _, row in mom_df.sort_values('lambda').iterrows():
        print(f"  {row['region']:<12} λ={row['lambda']:.4f}, R²={row['r2']:.3f}, Half-life={row['half_life']:.1f}mo")

    # Key finding
    if len(mom_df) > 1:
        fastest = mom_df.loc[mom_df['lambda'].idxmax()]
        slowest = mom_df.loc[mom_df['lambda'].idxmin()]
        ratio = fastest['lambda'] / slowest['lambda'] if slowest['lambda'] > 0 else np.inf

        print(f"\n** KEY FINDING **")
        print(f"Fastest decay: {fastest['region']} (λ={fastest['lambda']:.4f})")
        print(f"Slowest decay: {slowest['region']} (λ={slowest['lambda']:.4f})")
        print(f"Ratio: {ratio:.1f}x faster")

    return df


# =============================================================================
# EXPERIMENT 3: Global Taxonomy Validation
# =============================================================================

def experiment_global_taxonomy(data: dict):
    """
    Test if Mechanical vs Judgment taxonomy holds globally.

    Mechanical: Mom (clear signal) → should show decay
    Judgment: HML, QMJ (interpretation needed) → less clear decay
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: GLOBAL TAXONOMY VALIDATION")
    print("=" * 70)
    print("\nMechanical factors (Mom, BAB): Should show hyperbolic decay (higher R²)")
    print("Judgment factors (HML, QMJ): Should NOT show clear decay (lower R²)")

    # Classify factors
    mechanical = ['Mom', 'BAB']
    judgment = ['HML', 'QMJ']

    results = []

    for region, region_data in data.items():
        for factor in region_data.columns:
            sharpe = rolling_sharpe(region_data[factor], window=36)
            r2, K, lam = fit_decay_model(sharpe)

            if r2 is not None:
                category = 'Mechanical' if factor in mechanical else 'Judgment' if factor in judgment else 'Other'
                results.append({
                    'region': region,
                    'factor': factor,
                    'category': category,
                    'r2': r2,
                    'lambda': lam
                })

    df = pd.DataFrame(results)

    if len(df) == 0:
        print("No results!")
        return None

    # Compare categories
    print(f"\n{'Category':<12} {'Count':<8} {'Mean R²':<10} {'Std R²':<10}")
    print("-" * 40)

    for cat in ['Mechanical', 'Judgment']:
        cat_df = df[df['category'] == cat]
        if len(cat_df) > 0:
            print(f"{cat:<12} {len(cat_df):<8} {cat_df['r2'].mean():<10.3f} {cat_df['r2'].std():<10.3f}")

    # Statistical test
    mech_r2 = df[df['category'] == 'Mechanical']['r2']
    judg_r2 = df[df['category'] == 'Judgment']['r2']

    if len(mech_r2) > 0 and len(judg_r2) > 0:
        from scipy import stats
        stat, pvalue = stats.mannwhitneyu(mech_r2, judg_r2, alternative='greater')

        print(f"\nMann-Whitney U test (Mechanical > Judgment):")
        print(f"  U-statistic: {stat:.1f}")
        print(f"  p-value: {pvalue:.4f}")
        print(f"  Significant: {'Yes' if pvalue < 0.05 else 'No'} (α=0.05)")

        diff = mech_r2.mean() - judg_r2.mean()
        print(f"\n** KEY FINDING **")
        print(f"Mechanical mean R²: {mech_r2.mean():.3f}")
        print(f"Judgment mean R²: {judg_r2.mean():.3f}")
        print(f"Difference: {diff:+.3f}")

        if pvalue < 0.05:
            print(f"→ Taxonomy CONFIRMED globally (p={pvalue:.4f})")
        else:
            print(f"→ Taxonomy NOT confirmed globally (p={pvalue:.4f})")

    # Detail by factor
    print("\n" + "-" * 50)
    print("DETAIL BY FACTOR:")
    print("-" * 50)

    factor_summary = df.groupby('factor').agg({
        'r2': ['count', 'mean', 'std'],
        'category': 'first'
    })

    for factor in factor_summary.index:
        cat = factor_summary.loc[factor, ('category', 'first')]
        count = int(factor_summary.loc[factor, ('r2', 'count')])
        mean_r2 = factor_summary.loc[factor, ('r2', 'mean')]
        std_r2 = factor_summary.loc[factor, ('r2', 'std')]
        print(f"  {factor:<6} ({cat:<10}): R²={mean_r2:.3f}±{std_r2:.3f} (n={count})")

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("KDD 2026 EXPERIMENTS: GLOBAL FACTOR CROWDING")
    print("=" * 70)

    # Load data
    print("\nLoading global factor data...")
    data = load_all_regions()

    print(f"\nLoaded {len(data)} regions:")
    for region, df in data.items():
        print(f"  {region}: {len(df)} months, factors: {list(df.columns)}")

    # Run experiments
    exp1_results = experiment_cross_region_transfer(data)
    exp2_results = experiment_regional_decay_rates(data)
    exp3_results = experiment_global_taxonomy(data)

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY FOR KDD 2026 PAPER")
    print("=" * 70)

    print("""
Key Findings:

1. CROSS-REGION TRANSFER
   - US-trained model performance on other regions
   - Transfer efficiency metric

2. REGIONAL DECAY RATES
   - Crowding speed varies by market efficiency
   - Half-life comparison across regions

3. TAXONOMY VALIDATION
   - Mechanical vs Judgment classification
   - Global consistency of the taxonomy
""")

    return {
        'transfer': exp1_results,
        'decay': exp2_results,
        'taxonomy': exp3_results
    }


if __name__ == '__main__':
    results = main()
