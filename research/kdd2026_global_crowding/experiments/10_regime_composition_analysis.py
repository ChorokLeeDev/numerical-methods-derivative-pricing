"""
Regime Composition Analysis: Why does Europe fail while Japan succeeds?

This diagnostic script analyzes:
1. Regime distributions in US (source) vs Japan (works) vs Europe (fails)
2. Sample sizes per regime bucket
3. Whether regime definitions transfer meaningfully across countries
4. Hypothesis testing: data sparsity vs regime non-transfer
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

DATA_DIR = ROOT_DIR / 'data' / 'global_factors'


def load_country_data(country_name: str) -> pd.DataFrame:
    """Load factor data for a country."""
    file_path = DATA_DIR / f'{country_name.lower()}_factors.parquet'
    if not file_path.exists():
        return None
    return pd.read_parquet(file_path)


def extract_momentum_series(df: pd.DataFrame) -> np.ndarray:
    """Extract momentum/return series from dataframe."""
    if 'Mom' in df.columns:
        return df['Mom'].values
    else:
        # Use first column as proxy
        return df.iloc[:, 0].values


def detect_regimes_volatility(returns: np.ndarray, lookback: int = 12) -> np.ndarray:
    """
    Detect 2-regime classification: High vol vs Low vol.
    This is the method used in 09_country_transfer_validation.py
    """
    T = len(returns)
    regimes = np.zeros(T, dtype=np.int64)

    # Rolling volatility
    for t in range(lookback, T):
        window = returns[t-lookback:t]
        vol = np.std(window)

        # Simple binary: high vol (1) vs low vol (0)
        median_vol = np.median([np.std(returns[max(0,i-lookback):i])
                                for i in range(lookback, t+1)])
        regimes[t] = 1 if vol > median_vol else 0

    return regimes


def analyze_regime_composition(country_name: str, momentum_series: np.ndarray, regimes: np.ndarray) -> dict:
    """Analyze regime composition for a country."""

    n_total = len(regimes)
    n_regime0 = np.sum(regimes == 0)
    n_regime1 = np.sum(regimes == 1)

    pct_regime0 = 100 * n_regime0 / n_total
    pct_regime1 = 100 * n_regime1 / n_total

    # Volatility statistics by regime
    regime0_vol = np.std(momentum_series[regimes == 0]) if n_regime0 > 0 else np.nan
    regime1_vol = np.std(momentum_series[regimes == 1]) if n_regime1 > 0 else np.nan

    # Mean return by regime
    regime0_mean = np.mean(momentum_series[regimes == 0]) if n_regime0 > 0 else np.nan
    regime1_mean = np.mean(momentum_series[regimes == 1]) if n_regime1 > 0 else np.nan

    return {
        'country': country_name,
        'total_samples': n_total,
        'regime0_count': n_regime0,
        'regime1_count': n_regime1,
        'regime0_pct': pct_regime0,
        'regime1_pct': pct_regime1,
        'regime0_vol': regime0_vol,
        'regime1_vol': regime1_vol,
        'regime0_mean': regime0_mean,
        'regime1_mean': regime1_mean,
    }


def compare_regime_alignment(source_data: pd.DataFrame, target_data: pd.DataFrame,
                            source_regimes: np.ndarray, target_regimes: np.ndarray) -> dict:
    """
    Analyze how well regime labels align between source and target.
    Key insight: if regimes don't overlap, MMD will be computed on tiny samples.

    Properly aligns datasets by date before comparing regimes.
    """

    # Align by common dates
    common_idx = source_data.index.intersection(target_data.index)
    if len(common_idx) == 0:
        return None

    # Get aligned indices
    source_mask = source_data.index.isin(common_idx)
    target_mask = target_data.index.isin(common_idx)

    source_reg_aligned = source_regimes[source_mask]
    target_reg_aligned = target_regimes[target_mask]

    # Reorder to match dates
    source_order = source_data.index[source_mask].searchsorted(common_idx)
    target_order = target_data.index[target_mask].searchsorted(common_idx)

    source_reg_aligned = source_reg_aligned[source_order]
    target_reg_aligned = target_reg_aligned[target_order]

    n_total = len(common_idx)

    # For regime 0
    source_r0 = np.sum(source_reg_aligned == 0)
    target_r0 = np.sum(target_reg_aligned == 0)

    # For regime 1
    source_r1 = np.sum(source_reg_aligned == 1)
    target_r1 = np.sum(target_reg_aligned == 1)

    # Regime agreement
    agree = np.sum(source_reg_aligned == target_reg_aligned)
    agree_pct = 100 * agree / n_total

    return {
        'total_samples': n_total,
        'source_regime0': source_r0,
        'source_regime1': source_r1,
        'target_regime0': target_r0,
        'target_regime1': target_r1,
        'regime_agreement': agree_pct,
        'min_regime0_samples': min(source_r0, target_r0),
        'min_regime1_samples': min(source_r1, target_r1),
    }


def main():
    print("=" * 90)
    print("REGIME COMPOSITION ANALYSIS: Diagnosing Transfer Success/Failure")
    print("=" * 90)

    countries = ['US', 'Japan', 'Europe', 'AsiaPac']

    # Load data
    data = {}
    momentum_series = {}
    regimes = {}

    print("\n1. LOADING DATA AND DETECTING REGIMES")
    print("-" * 90)

    for country in countries:
        df = load_country_data(country)
        if df is None:
            print(f"  ✗ {country}: Could not load data")
            continue

        data[country] = df
        mom = extract_momentum_series(df)
        momentum_series[country] = mom
        regimes[country] = detect_regimes_volatility(mom, lookback=12)

        print(f"  ✓ {country}: Loaded {len(df)} samples")

    # Analyze regime compositions
    print("\n2. REGIME COMPOSITION BY COUNTRY")
    print("-" * 90)
    print(f"{'Country':<12} | {'Total':<8} | {'Regime-0':<12} | {'Regime-1':<12} | {'Alignment':<12}")
    print("-" * 90)

    compositions = {}
    for country in countries:
        if country not in regimes:
            continue

        comp = analyze_regime_composition(country, momentum_series[country], regimes[country])
        compositions[country] = comp

        print(f"{country:<12} | {comp['total_samples']:<8} | "
              f"{comp['regime0_count']:<4} ({comp['regime0_pct']:>5.1f}%) | "
              f"{comp['regime1_count']:<4} ({comp['regime1_pct']:>5.1f}%) | ")

    # Compare regime alignment between source and target
    print("\n3. REGIME ALIGNMENT: SOURCE vs TARGET (Critical for MMD Loss)")
    print("-" * 90)

    alignments = {}
    for target in ['UK', 'Japan', 'Europe', 'AsiaPac']:
        if 'US' not in regimes or target not in regimes:
            continue
        if 'US' not in data or target not in data:
            continue

        align = compare_regime_alignment(data['US'], data[target], regimes['US'], regimes[target])
        if align is None:
            print(f"\n[US → {target}]")
            print(f"  ✗ No common dates found")
            continue

        alignments[target] = align

        print(f"\n[US → {target}]")
        print(f"  Total samples: {align['total_samples']}")
        print(f"  US regime 0: {align['source_regime0']} samples → {target} regime 0: {align['target_regime0']} samples")
        print(f"  US regime 1: {align['source_regime1']} samples → {target} regime 1: {align['target_regime1']} samples")
        print(f"  Regime agreement: {align['regime_agreement']:.1f}%")
        print(f"  ⚠️  Minimum samples in regime 0: {align['min_regime0_samples']}")
        print(f"  ⚠️  Minimum samples in regime 1: {align['min_regime1_samples']}")

        # Diagnostic: are the regimes too sparse?
        if align['min_regime0_samples'] < 10 or align['min_regime1_samples'] < 10:
            print(f"  🔴 CRITICAL: One regime has <10 samples! MMD computation unreliable.")
        elif align['min_regime0_samples'] < 50 or align['min_regime1_samples'] < 50:
            print(f"  🟡 WARNING: One regime has <50 samples. Limited statistical power.")
        else:
            print(f"  🟢 OK: Both regimes have >50 samples.")

    # Volatility analysis
    print("\n4. VOLATILITY CHARACTERISTICS BY REGIME")
    print("-" * 90)
    print(f"{'Country':<12} | {'Regime-0 Vol':<15} | {'Regime-1 Vol':<15} | {'Vol Ratio (1/0)':<15}")
    print("-" * 90)

    for country in countries:
        if country not in compositions:
            continue

        comp = compositions[country]
        vol_ratio = comp['regime1_vol'] / comp['regime0_vol'] if comp['regime0_vol'] > 0 else np.nan

        print(f"{country:<12} | {comp['regime0_vol']:<15.4f} | "
              f"{comp['regime1_vol']:<15.4f} | {vol_ratio:<15.2f}")

    # Key insight: Check volatility regime separation
    print("\n5. HYPOTHESIS: REGIME SEPARATION AND TRANSFERABILITY")
    print("-" * 90)

    print("""
The regime detection is based on rolling volatility (high vol vs low vol).

Key question: Are the regime boundaries meaningful across countries?

For regime-conditional MMD to work, the algorithm needs:
a) Sufficient samples in each regime bucket (>50 recommended)
b) Meaningful separation between regimes (vol_ratio > 1.5)
c) Consistent regime definitions across source and target

If these conditions fail, regime-conditioning becomes counterproductive:
- Too few samples → noise dominates signal → MMD unreliable
- Poor separation → regimes don't partition meaningful structure → conditioning hurts
- Inconsistent definitions → source regimes don't predict target regimes → non-transfer
    """)

    # Analyze Japan success vs Europe failure
    print("\n6. COMPARATIVE ANALYSIS: Japan (+18.9%) vs Europe (-21.5%)")
    print("-" * 90)

    if alignments and 'Japan' in alignments and 'Europe' in alignments:
        japan_align = alignments['Japan']
        europe_align = alignments['Europe']

        print(f"\nJAPAN (Works: +18.9% improvement)")
        print(f"  Regime bucket sizes:")
        print(f"    Regime 0: min({japan_align['source_regime0']}, {japan_align['target_regime0']}) = {japan_align['min_regime0_samples']}")
        print(f"    Regime 1: min({japan_align['source_regime1']}, {japan_align['target_regime1']}) = {japan_align['min_regime1_samples']}")
        print(f"  Regime agreement: {japan_align['regime_agreement']:.1f}%")

        print(f"\nEUROPE (Fails: -21.5% degradation)")
        print(f"  Regime bucket sizes:")
        print(f"    Regime 0: min({europe_align['source_regime0']}, {europe_align['target_regime0']}) = {europe_align['min_regime0_samples']}")
        print(f"    Regime 1: min({europe_align['source_regime1']}, {europe_align['target_regime1']}) = {europe_align['min_regime1_samples']}")
        print(f"  Regime agreement: {europe_align['regime_agreement']:.1f}%")

        # Hypothesis test
        print(f"\n📊 HYPOTHESIS TEST: Data Sparsity")
        japan_sparsity = max(japan_align['min_regime0_samples'], japan_align['min_regime1_samples'])
        europe_sparsity = max(europe_align['min_regime0_samples'], europe_align['min_regime1_samples'])

        if japan_sparsity > europe_sparsity:
            print(f"  ✗ Data sparsity hypothesis FAILS: Japan has MORE samples ({japan_sparsity}) than Europe ({europe_sparsity})")
            print(f"    → This doesn't explain why Europe fails")
        else:
            print(f"  ✓ Data sparsity hypothesis PLAUSIBLE: Europe has fewer samples ({europe_sparsity}) than Japan ({japan_sparsity})")
            print(f"    → Could explain performance degradation")

    print("\n" + "=" * 90)
    print("END OF ANALYSIS")
    print("=" * 90)

    return compositions, alignments


if __name__ == '__main__':
    compositions, alignments = main()
