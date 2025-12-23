"""
Experiment 05: Investigate Crowding-Shift Assumption Violation

Critical issue: Our assumption is that high crowding → poor coverage (distribution shift)
But empirically we found: high crowding → BETTER coverage (correlation = -0.901)

This experiment investigates:
1. Why is the relationship inverted?
2. Is our crowding signal measuring the wrong thing?
3. Can we reframe the contribution?

For ICML 2026 submission.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add paths
ICML_SRC = Path(__file__).parent.parent / 'src'
FC_SRC = Path(__file__).parent.parent.parent / 'factor_crowding' / 'src'

import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

crowding_signal_module = load_module('crowding_signal_icml', ICML_SRC / 'crowding_signal.py')
CrowdingSignal = crowding_signal_module.CrowdingSignal

baselines_module = load_module('baselines', ICML_SRC / 'baselines.py')
SplitConformalCP = baselines_module.SplitConformalCP

sys.path.insert(0, str(FC_SRC))
from features import FeatureEngineer


def analyze_crowding_components(factors: pd.DataFrame) -> pd.DataFrame:
    """Analyze individual components of crowding signal."""
    cs = CrowdingSignal()

    results = []

    for factor in factors.columns:
        # Get individual crowding components
        corr_signal = cs._correlation_crowding(factors[factor], factors)
        vol_signal = cs._volatility_crowding(factors[factor])
        mom_signal = cs._momentum_crowding(factors[factor])
        combined = cs.compute_crowding(factors[factor], factors, normalize=True)

        results.append({
            'factor': factor,
            'corr_mean': corr_signal.mean(),
            'corr_std': corr_signal.std(),
            'vol_mean': vol_signal.mean(),
            'vol_std': vol_signal.std(),
            'mom_mean': mom_signal.dropna().mean(),
            'mom_std': mom_signal.dropna().std(),
            'combined_mean': combined.mean(),
            'combined_std': combined.std(),
        })

    return pd.DataFrame(results)


def analyze_coverage_by_crowding_detailed(
    X: np.ndarray,
    y: np.ndarray,
    crowding: np.ndarray,
    fit_size: int = 90,
    calib_size: int = 30,
    test_size: int = 12,
    alpha: float = 0.1
) -> pd.DataFrame:
    """Detailed analysis of coverage vs crowding relationship."""

    results = []
    min_train = fit_size + calib_size

    start_idx = 0
    while start_idx + min_train + test_size <= len(X):
        fit_end = start_idx + fit_size
        calib_end = fit_end + calib_size
        test_end = calib_end + test_size

        X_fit = X[start_idx:fit_end]
        y_fit = y[start_idx:fit_end]
        X_calib = X[fit_end:calib_end]
        y_calib = y[fit_end:calib_end]
        X_test = X[calib_end:test_end]
        y_test = y[calib_end:test_end]
        crowding_test = crowding[calib_end:test_end]

        # Split CP
        split_cp = SplitConformalCP()
        split_cp.fit(X_fit, y_fit, X_calib, y_calib)
        sets, threshold = split_cp.predict_sets(X_test, alpha)

        # Record per-sample results
        for i in range(len(y_test)):
            covered = int(y_test[i]) in sets[i]
            results.append({
                'crowding': crowding_test[i],
                'covered': int(covered),
                'set_size': len(sets[i]),
                'y_true': y_test[i],
                'threshold': threshold,
            })

        start_idx += test_size

    return pd.DataFrame(results)


def investigate_inversion():
    """Main investigation of why assumption is inverted."""

    print("=" * 70)
    print("INVESTIGATING CROWDING-SHIFT ASSUMPTION VIOLATION")
    print("=" * 70)

    # Load data
    DATA_DIR = Path(__file__).parent.parent / 'data'
    factors = pd.read_parquet(DATA_DIR / 'ff_extended_factors.parquet')
    if isinstance(factors.index, pd.PeriodIndex):
        factors.index = factors.index.to_timestamp()
    factors = factors.drop(columns=['RF'], errors='ignore')

    print(f"\nData: {factors.index.min().strftime('%Y-%m')} to {factors.index.max().strftime('%Y-%m')}")

    # 1. Analyze crowding signal components
    print("\n" + "=" * 70)
    print("1. CROWDING SIGNAL COMPONENT ANALYSIS")
    print("=" * 70)

    component_df = analyze_crowding_components(factors)
    print("\nCrowding Signal Components by Factor:")
    print(component_df.round(3).to_string(index=False))

    # 2. Analyze coverage vs crowding for each factor
    print("\n" + "=" * 70)
    print("2. COVERAGE VS CROWDING BY FACTOR")
    print("=" * 70)

    cs = CrowdingSignal()
    crowding_df = cs.compute_all_crowding(factors, normalize=True)

    fe = FeatureEngineer()
    features = fe.generate_all_features(factors)

    factor_correlations = []

    for factor in factors.columns:
        X, y = fe.create_ml_dataset(features, factors, target_type='crash', factor=factor)
        crowding = crowding_df[factor].loc[X.index].values

        detailed_results = analyze_coverage_by_crowding_detailed(
            X.values, y.values, crowding
        )

        # Correlation between crowding and coverage
        corr = detailed_results['crowding'].corr(detailed_results['covered'])

        # Coverage by crowding tercile
        detailed_results['crowding_tercile'] = pd.qcut(
            detailed_results['crowding'], 3, labels=['low', 'med', 'high']
        )
        coverage_by_tercile = detailed_results.groupby('crowding_tercile')['covered'].mean()

        factor_correlations.append({
            'factor': factor,
            'corr_crowding_coverage': corr,
            'cov_low': coverage_by_tercile.get('low', np.nan),
            'cov_med': coverage_by_tercile.get('med', np.nan),
            'cov_high': coverage_by_tercile.get('high', np.nan),
            'n_samples': len(detailed_results),
        })

        print(f"\n{factor}:")
        print(f"  Corr(crowding, coverage): {corr:.3f}")
        print(f"  Coverage: low={coverage_by_tercile.get('low', 0):.3f}, "
              f"med={coverage_by_tercile.get('med', 0):.3f}, "
              f"high={coverage_by_tercile.get('high', 0):.3f}")

    corr_df = pd.DataFrame(factor_correlations)

    # 3. Key insight: What does negative correlation mean?
    print("\n" + "=" * 70)
    print("3. INTERPRETATION: WHY IS CORRELATION NEGATIVE?")
    print("=" * 70)

    avg_corr = corr_df['corr_crowding_coverage'].mean()
    n_negative = (corr_df['corr_crowding_coverage'] < 0).sum()

    print(f"""
Average correlation: {avg_corr:.3f}
Factors with negative correlation: {n_negative}/{len(factors.columns)}

INTERPRETATION:
- Negative correlation means: HIGH crowding → HIGH coverage
- This is OPPOSITE to our assumption

POSSIBLE EXPLANATIONS:

1. **Crowding signal is INVERTED**
   - Our crowding signal might actually measure "uncrowding" or "opportunity"
   - When signal is high, the factor is actually LESS risky

2. **Model confidence effect**
   - High crowding → more predictable returns → model is more confident
   - More confident model → better calibrated → better coverage

3. **Volatility effect**
   - High crowding correlates with low volatility periods
   - Low volatility → easier prediction → better coverage

4. **Selection bias**
   - Crashes may cluster in LOW crowding periods (after unwinding)
   - Low crowding = post-crash = high uncertainty = poor coverage
""")

    # 4. Test the volatility hypothesis
    print("\n" + "=" * 70)
    print("4. TESTING VOLATILITY HYPOTHESIS")
    print("=" * 70)

    # Compute rolling volatility
    volatility = factors.rolling(12).std()

    for factor in ['Mom', 'MKT', 'HML']:  # Key factors
        X, y = fe.create_ml_dataset(features, factors, target_type='crash', factor=factor)
        crowding = crowding_df[factor].loc[X.index]
        vol = volatility[factor].loc[X.index]

        # Correlation between crowding and volatility
        corr_crowd_vol = crowding.corr(vol)

        print(f"\n{factor}:")
        print(f"  Corr(crowding, volatility): {corr_crowd_vol:.3f}")

        if corr_crowd_vol < -0.3:
            print(f"  → High crowding = Low volatility (supports hypothesis)")
        elif corr_crowd_vol > 0.3:
            print(f"  → High crowding = High volatility (contradicts hypothesis)")
        else:
            print(f"  → Weak relationship")

    # 5. Alternative: What if we INVERT the crowding signal?
    print("\n" + "=" * 70)
    print("5. SOLUTION: INVERT THE CROWDING INTERPRETATION")
    print("=" * 70)

    print("""
PROPOSED REFRAMING:

Instead of: "High crowding → Distribution shift → Need larger sets"

Use: "High UNCERTAINTY (low crowding) → Need larger sets"

The crowding signal actually measures STABILITY:
- High crowding signal = Stable regime = Confident predictions
- Low crowding signal = Unstable regime = Need conservative sets

This INVERTS our λ-weighting:
- Original: score / (1 + λ × crowding)  → larger sets when crowding HIGH
- Fixed:    score × (1 + λ × crowding)  → larger sets when crowding LOW

OR equivalently:
- Use (1 - crowding) as the signal
""")

    # 6. Summary
    print("\n" + "=" * 70)
    print("6. RECOMMENDED FIX")
    print("=" * 70)

    print("""
OPTION A: Invert the weighting formula
   score_new = score × (1 + λ × crowding)

   This gives LARGER sets when crowding is LOW (where coverage is poor)

OPTION B: Use (1 - crowding) as uncertainty signal
   uncertainty = 1 - crowding
   score_weighted = score / (1 + λ × uncertainty)

OPTION C: Reframe the narrative
   - Don't claim "crowding predicts shift"
   - Instead: "Our signal identifies low-uncertainty regimes"
   - Contribution: Learn to be MORE confident when appropriate

RECOMMENDED: Option A or B + Option C (reframe narrative)
""")

    return corr_df


if __name__ == '__main__':
    results = investigate_inversion()
