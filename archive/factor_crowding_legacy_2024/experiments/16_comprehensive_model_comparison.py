"""
Experiment 16: Comprehensive Model Comparison for ICAIF 2025

Compare all approaches for crash prediction:
1. Baseline: Model-residual signal (current paper)
2. RandomForest: Feature-based ML
3. XGBoost: Gradient boosting
4. Neural Network: MLP with factor embedding

Goal: Demonstrate ML adds value and identify best approach
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features import FeatureEngineer, WalkForwardCV
from crowding_ml import RandomForestModel, XGBoostModel
from crowding_signal import CrowdingDetector


def create_baseline_predictions(factor_returns, factor, X_index):
    """Create baseline predictions aligned with ML index."""
    detector = CrowdingDetector(train_window=120, prediction_gap=12, sharpe_window=36)
    signal_df = detector.compute_rolling_signal(factor_returns[factor], factor_name=factor)

    if len(signal_df) == 0:
        return None

    common_idx = X_index.intersection(signal_df.index)
    if len(common_idx) == 0:
        return None

    # Negative residual = higher crash prob
    residuals = signal_df.loc[common_idx, 'residual']
    probs = -residuals

    # Normalize to 0-1
    probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)

    return probs


def run_comparison(factor_returns, factor, train_size=120, test_size=12):
    """Run walk-forward comparison for one factor."""
    # Generate features
    fe = FeatureEngineer()
    features = fe.generate_all_features(factor_returns)
    X, y = fe.create_ml_dataset(features, factor_returns, target_type='crash', factor=factor)

    # Walk-forward CV
    cv = WalkForwardCV(train_size=train_size, test_size=test_size, step_size=test_size)
    splits = cv.split(X)

    # Baseline predictions (aligned)
    baseline_probs = create_baseline_predictions(factor_returns, factor, X.index)
    if baseline_probs is None:
        return None

    common_idx = X.index.intersection(baseline_probs.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    baseline_probs = baseline_probs.loc[common_idx]

    # Re-compute splits on aligned data
    splits = cv.split(X)

    # Collect predictions
    results = {
        'baseline': [],
        'rf': [],
        'xgb': [],
        'actuals': []
    }

    for train_idx, test_idx in splits:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Baseline (already computed)
        results['baseline'].extend(baseline_probs.iloc[test_idx].values)

        # RandomForest
        rf = RandomForestModel()
        rf.fit(X_train, y_train)
        results['rf'].extend(rf.predict_proba(X_test))

        # XGBoost
        xgb = XGBoostModel()
        xgb.fit(X_train, y_train)
        results['xgb'].extend(xgb.predict_proba(X_test))

        results['actuals'].extend(y_test.values)

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    # Compute metrics
    metrics = {}
    for model in ['baseline', 'rf', 'xgb']:
        metrics[model] = {
            'auc': roc_auc_score(results['actuals'], results[model]),
            'ap': average_precision_score(results['actuals'], results[model])
        }

    metrics['crash_rate'] = results['actuals'].mean()
    metrics['n_samples'] = len(results['actuals'])

    return metrics


def main():
    print("=" * 70)
    print("COMPREHENSIVE MODEL COMPARISON FOR ICAIF 2025")
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
    print("WALK-FORWARD BACKTEST RESULTS")
    print("=" * 70)

    all_results = []
    for factor in factors.columns:
        print(f"\nProcessing {factor}...")
        metrics = run_comparison(factors, factor)
        if metrics:
            metrics['factor'] = factor
            all_results.append(metrics)

    # Create summary table
    print("\n" + "=" * 70)
    print("AUC COMPARISON")
    print("=" * 70)

    print(f"\n{'Factor':<10} {'Crash%':<8} {'Baseline':<10} {'RF':<10} {'XGBoost':<10} {'Best':<10}")
    print("-" * 58)

    for r in all_results:
        aucs = {'Baseline': r['baseline']['auc'], 'RF': r['rf']['auc'], 'XGB': r['xgb']['auc']}
        best = max(aucs, key=aucs.get)
        print(f"{r['factor']:<10} {r['crash_rate']*100:<8.1f} {r['baseline']['auc']:<10.3f} {r['rf']['auc']:<10.3f} {r['xgb']['auc']:<10.3f} {best:<10}")

    # Aggregate statistics
    print("\n" + "=" * 70)
    print("AGGREGATE STATISTICS")
    print("=" * 70)

    df = pd.DataFrame(all_results)

    baseline_aucs = [r['baseline']['auc'] for r in all_results]
    rf_aucs = [r['rf']['auc'] for r in all_results]
    xgb_aucs = [r['xgb']['auc'] for r in all_results]

    print(f"\n{'Model':<12} {'Mean AUC':<12} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 54)
    print(f"{'Baseline':<12} {np.mean(baseline_aucs):<12.3f} {np.std(baseline_aucs):<10.3f} {np.min(baseline_aucs):<10.3f} {np.max(baseline_aucs):<10.3f}")
    print(f"{'RF':<12} {np.mean(rf_aucs):<12.3f} {np.std(rf_aucs):<10.3f} {np.min(rf_aucs):<10.3f} {np.max(rf_aucs):<10.3f}")
    print(f"{'XGBoost':<12} {np.mean(xgb_aucs):<12.3f} {np.std(xgb_aucs):<10.3f} {np.min(xgb_aucs):<10.3f} {np.max(xgb_aucs):<10.3f}")

    # Win count
    rf_wins = sum(1 for i in range(len(all_results)) if rf_aucs[i] > baseline_aucs[i])
    xgb_wins = sum(1 for i in range(len(all_results)) if xgb_aucs[i] > baseline_aucs[i])
    best_ml = sum(1 for i in range(len(all_results)) if max(rf_aucs[i], xgb_aucs[i]) > baseline_aucs[i])

    print(f"\nML vs Baseline:")
    print(f"  RF beats baseline:     {rf_wins}/{len(all_results)} factors")
    print(f"  XGBoost beats baseline: {xgb_wins}/{len(all_results)} factors")
    print(f"  Best ML beats baseline: {best_ml}/{len(all_results)} factors")

    # Statistical significance
    from scipy import stats

    # RF vs Baseline
    t_rf, p_rf = stats.ttest_rel(rf_aucs, baseline_aucs)
    # XGBoost vs Baseline
    t_xgb, p_xgb = stats.ttest_rel(xgb_aucs, baseline_aucs)

    print(f"\nStatistical Tests (paired t-test):")
    print(f"  RF vs Baseline:     t={t_rf:.3f}, p={p_rf:.4f} {'*' if p_rf < 0.05 else ''}")
    print(f"  XGBoost vs Baseline: t={t_xgb:.3f}, p={p_xgb:.4f} {'*' if p_xgb < 0.05 else ''}")

    # Average Precision comparison
    print("\n" + "=" * 70)
    print("AVERAGE PRECISION COMPARISON")
    print("=" * 70)

    print(f"\n{'Factor':<10} {'Baseline':<10} {'RF':<10} {'XGBoost':<10}")
    print("-" * 40)

    for r in all_results:
        print(f"{r['factor']:<10} {r['baseline']['ap']:<10.3f} {r['rf']['ap']:<10.3f} {r['xgb']['ap']:<10.3f}")

    baseline_aps = [r['baseline']['ap'] for r in all_results]
    rf_aps = [r['rf']['ap'] for r in all_results]
    xgb_aps = [r['xgb']['ap'] for r in all_results]

    print(f"\n{'Mean AP':<10} {np.mean(baseline_aps):<10.3f} {np.mean(rf_aps):<10.3f} {np.mean(xgb_aps):<10.3f}")

    # Key findings for paper
    print("\n" + "=" * 70)
    print("KEY FINDINGS FOR ICAIF 2025 PAPER")
    print("=" * 70)

    best_model = 'RandomForest' if np.mean(rf_aucs) > np.mean(xgb_aucs) else 'XGBoost'
    best_auc = max(np.mean(rf_aucs), np.mean(xgb_aucs))
    improvement = best_auc - np.mean(baseline_aucs)

    print(f"""
1. BASELINE PERFORMANCE
   - Model-residual approach: Mean AUC = {np.mean(baseline_aucs):.3f}
   - Based on hyperbolic decay model residuals

2. ML IMPROVEMENT
   - Best ML model: {best_model}
   - Mean AUC: {best_auc:.3f}
   - Improvement over baseline: +{improvement:.3f}
   - Wins on {best_ml}/{len(all_results)} factors

3. STATISTICAL SIGNIFICANCE
   - RF vs Baseline: p = {p_rf:.4f} {'(significant)' if p_rf < 0.05 else '(not significant)'}
   - XGBoost vs Baseline: p = {p_xgb:.4f} {'(significant)' if p_xgb < 0.05 else '(not significant)'}

4. RECOMMENDATION FOR PAPER
   - Use {best_model} as primary ML approach
   - Show comparison table with baseline
   - Highlight heterogeneous performance across factor types
""")

    return all_results


if __name__ == '__main__':
    results = main()
