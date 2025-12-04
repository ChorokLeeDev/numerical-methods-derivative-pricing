"""
Experiment 15: ML vs Model-Residual Baseline Comparison

Compare crash prediction performance:
1. Baseline: Model-residual based signal (current paper approach)
2. ML: RandomForest with engineered features

Goal: Demonstrate ML adds value for ICAIF submission
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features import FeatureEngineer, WalkForwardCV
from crowding_ml import RandomForestModel
from crowding_signal import CrowdingDetector


def create_baseline_predictions(
    factor_returns: pd.DataFrame,
    factor: str,
    train_size: int = 120,
    test_size: int = 12,
) -> pd.DataFrame:
    """
    Create crash predictions using model-residual baseline.

    Approach: Use residual as predictor - negative residual = crowded = higher crash risk
    """
    detector = CrowdingDetector(
        train_window=train_size,
        prediction_gap=12,  # 1 year gap
        sharpe_window=36
    )

    # Compute crowding signal
    signal_df = detector.compute_rolling_signal(
        factor_returns[factor],
        factor_name=factor
    )

    if len(signal_df) == 0:
        return pd.DataFrame()

    # Create crash target (same as ML approach)
    fe = FeatureEngineer()
    crash_targets = fe.create_crash_targets(factor_returns, threshold_pct=0.10, horizon=1)

    # Align signal with crash target
    common_idx = signal_df.index.intersection(crash_targets.index)

    result = pd.DataFrame({
        'residual': signal_df.loc[common_idx, 'residual'],
        'crash_prob_baseline': -signal_df.loc[common_idx, 'residual'],  # Negative residual = higher crash prob
        'actual_crash': crash_targets.loc[common_idx, f'{factor}_crash']
    })

    # Normalize to 0-1 probability
    result['crash_prob_baseline'] = (
        result['crash_prob_baseline'] - result['crash_prob_baseline'].min()
    ) / (result['crash_prob_baseline'].max() - result['crash_prob_baseline'].min() + 1e-8)

    return result.dropna()


def run_walk_forward_comparison(
    factor_returns: pd.DataFrame,
    factor: str,
    train_size: int = 120,
    test_size: int = 12,
) -> dict:
    """
    Run walk-forward comparison of ML vs baseline.
    """
    # Generate ML features
    fe = FeatureEngineer()
    features = fe.generate_all_features(factor_returns)
    X, y = fe.create_ml_dataset(features, factor_returns, target_type='crash', factor=factor)

    # Get baseline predictions
    baseline_df = create_baseline_predictions(factor_returns, factor, train_size, test_size)

    if len(baseline_df) == 0:
        return None

    # Align ML and baseline data
    common_idx = X.index.intersection(baseline_df.index)
    X_aligned = X.loc[common_idx]
    y_aligned = y.loc[common_idx]
    baseline_aligned = baseline_df.loc[common_idx]

    # Walk-forward CV
    cv = WalkForwardCV(train_size=train_size, test_size=test_size, step_size=test_size)
    splits = cv.split(X_aligned)

    # Collect predictions
    ml_preds = []
    baseline_preds = []
    actuals = []

    for train_idx, test_idx in splits:
        # ML prediction
        model = RandomForestModel()
        X_train = X_aligned.iloc[train_idx]
        y_train = y_aligned.iloc[train_idx]
        X_test = X_aligned.iloc[test_idx]
        y_test = y_aligned.iloc[test_idx]

        model.fit(X_train, y_train)
        ml_prob = model.predict_proba(X_test)

        # Baseline prediction (already computed)
        baseline_prob = baseline_aligned.iloc[test_idx]['crash_prob_baseline'].values

        ml_preds.extend(ml_prob)
        baseline_preds.extend(baseline_prob)
        actuals.extend(y_test.values)

    # Compute metrics
    ml_preds = np.array(ml_preds)
    baseline_preds = np.array(baseline_preds)
    actuals = np.array(actuals)

    # AUC
    ml_auc = roc_auc_score(actuals, ml_preds)
    baseline_auc = roc_auc_score(actuals, baseline_preds)

    # Average Precision (better for imbalanced data)
    ml_ap = average_precision_score(actuals, ml_preds)
    baseline_ap = average_precision_score(actuals, baseline_preds)

    # Precision at recall=0.5
    def precision_at_recall(y_true, y_score, target_recall=0.5):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        idx = np.argmin(np.abs(recall - target_recall))
        return precision[idx]

    ml_prec_50 = precision_at_recall(actuals, ml_preds, 0.5)
    baseline_prec_50 = precision_at_recall(actuals, baseline_preds, 0.5)

    return {
        'factor': factor,
        'n_samples': len(actuals),
        'crash_rate': actuals.mean(),
        'ml_auc': ml_auc,
        'baseline_auc': baseline_auc,
        'auc_improvement': ml_auc - baseline_auc,
        'ml_avg_precision': ml_ap,
        'baseline_avg_precision': baseline_ap,
        'ap_improvement': ml_ap - baseline_ap,
        'ml_prec_at_50recall': ml_prec_50,
        'baseline_prec_at_50recall': baseline_prec_50,
    }


def main():
    print("=" * 70)
    print("ML vs MODEL-RESIDUAL BASELINE COMPARISON")
    print("=" * 70)

    # Load data
    DATA_DIR = Path(__file__).parent.parent / 'data'
    factors = pd.read_parquet(DATA_DIR / 'ff_extended_factors.parquet')
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    factors = factors.drop(columns=['RF'], errors='ignore')

    print(f"\nData: {factors.index.min().strftime('%Y-%m')} to {factors.index.max().strftime('%Y-%m')}")
    print(f"Factors: {list(factors.columns)}")

    # Run comparison for each factor
    print("\n" + "=" * 70)
    print("WALK-FORWARD COMPARISON (Train: 120mo, Test: 12mo)")
    print("=" * 70)

    results = []
    for factor in factors.columns:
        print(f"\nProcessing {factor}...")
        result = run_walk_forward_comparison(factors, factor)
        if result:
            results.append(result)

    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(results)

    print(f"\n{'Factor':<10} {'Crash%':<8} {'ML AUC':<10} {'Base AUC':<10} {'Δ AUC':<10} {'ML AP':<10} {'Base AP':<10}")
    print("-" * 78)

    for _, row in results_df.iterrows():
        delta = f"+{row['auc_improvement']:.3f}" if row['auc_improvement'] > 0 else f"{row['auc_improvement']:.3f}"
        print(f"{row['factor']:<10} {row['crash_rate']*100:<8.1f} {row['ml_auc']:<10.3f} {row['baseline_auc']:<10.3f} {delta:<10} {row['ml_avg_precision']:<10.3f} {row['baseline_avg_precision']:<10.3f}")

    # Aggregate statistics
    print("\n" + "=" * 70)
    print("AGGREGATE STATISTICS")
    print("=" * 70)

    ml_wins = (results_df['auc_improvement'] > 0).sum()
    total = len(results_df)

    print(f"\nML outperforms baseline: {ml_wins}/{total} factors ({ml_wins/total*100:.0f}%)")
    print(f"\nMean AUC:")
    print(f"  ML:       {results_df['ml_auc'].mean():.3f} ± {results_df['ml_auc'].std():.3f}")
    print(f"  Baseline: {results_df['baseline_auc'].mean():.3f} ± {results_df['baseline_auc'].std():.3f}")
    print(f"  Δ:        {results_df['auc_improvement'].mean():.3f}")

    print(f"\nMean Average Precision:")
    print(f"  ML:       {results_df['ml_avg_precision'].mean():.3f}")
    print(f"  Baseline: {results_df['baseline_avg_precision'].mean():.3f}")
    print(f"  Δ:        {results_df['ap_improvement'].mean():.3f}")

    # Statistical significance test
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(results_df['ml_auc'], results_df['baseline_auc'])

    print(f"\nPaired t-test (ML AUC vs Baseline AUC):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value:     {p_value:.4f}")
    print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT FOR ICAIF PAPER")
    print("=" * 70)

    if results_df['auc_improvement'].mean() > 0:
        print(f"""
ML-based approach improves crash prediction over model-residual baseline:
- Mean AUC improvement: +{results_df['auc_improvement'].mean():.3f}
- ML wins on {ml_wins}/{total} factors
- Statistical significance: p={p_value:.4f}

This justifies the ML extension for ICAIF 2025 submission.
""")
    else:
        print("""
Baseline performs comparably to ML. Consider:
1. Feature engineering improvements
2. Alternative ML models (XGBoost, Neural Networks)
3. Ensemble approaches
""")

    return results_df


if __name__ == '__main__':
    results = main()
