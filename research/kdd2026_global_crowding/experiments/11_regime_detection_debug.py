"""
Debug regime detection: Why is US producing all regime-0 with 0 regime-1?
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

DATA_DIR = ROOT_DIR / 'data' / 'global_factors'


def debug_regime_detection(country_name: str):
    """Detailed debugging of regime detection process."""

    file_path = DATA_DIR / f'{country_name.lower()}_factors.parquet'
    if not file_path.exists():
        print(f"Cannot find {file_path}")
        return

    df = pd.read_parquet(file_path)

    # Extract momentum series (same as in 09_country_transfer_validation.py)
    if 'Mom' in df.columns:
        source_mom = df['Mom']
    else:
        source_mom = df.iloc[:, 0]

    print(f"\n{'='*90}")
    print(f"REGIME DETECTION DEBUG: {country_name}")
    print(f"{'='*90}")

    print(f"\n1. MOMENTUM SERIES BASICS")
    print(f"  Total samples: {len(source_mom)}")
    print(f"  Date range: {source_mom.index[0]} to {source_mom.index[-1]}")
    print(f"  Mean momentum: {source_mom.mean():.6f}")
    print(f"  Std momentum: {source_mom.std():.6f}")
    print(f"  Min momentum: {source_mom.min():.6f}")
    print(f"  Max momentum: {source_mom.max():.6f}")

    print(f"\n2. ROLLING VOLATILITY CALCULATION")
    vol_source = source_mom.rolling(63).std()
    print(f"  Rolling volatility (window=63):")
    print(f"    Non-NaN values: {vol_source.notna().sum()} / {len(vol_source)}")
    print(f"    Mean: {vol_source.mean():.6f}")
    print(f"    Min: {vol_source.min():.6f}")
    print(f"    Max: {vol_source.max():.6f}")

    print(f"\n3. ROLLING MEDIAN OF VOLATILITY")
    median_vol_s = vol_source.rolling(252).median()
    print(f"  Rolling median of volatility (window=252):")
    print(f"    Non-NaN values: {median_vol_s.notna().sum()} / {len(median_vol_s)}")
    print(f"    Mean: {median_vol_s.mean():.6f}")
    print(f"    Min: {median_vol_s.min():.6f}")
    print(f"    Max: {median_vol_s.max():.6f}")

    print(f"\n4. REGIME ASSIGNMENT: vol_source > median_vol_s")
    regime_source = (vol_source > median_vol_s).astype(int)
    print(f"  Regime assignments (before NaN filtering):")
    print(f"    Regime 0 (vol <= median): {(regime_source == 0).sum()}")
    print(f"    Regime 1 (vol > median): {(regime_source == 1).sum()}")
    print(f"    NaN values: {regime_source.isna().sum()}")

    # Analyze NaN patterns
    print(f"\n5. NAN ANALYSIS")
    nan_from_vol = vol_source.isna()
    nan_from_median = median_vol_s.isna()
    print(f"  NaNs from rolling(63).std(): {nan_from_vol.sum()}")
    print(f"  NaNs from rolling(252).median(): {nan_from_median.sum()}")
    print(f"  Combined NaNs in regime calculation: {(nan_from_vol | nan_from_median).sum()}")

    # Now apply the complete mask like 09_country_transfer_validation.py does
    print(f"\n6. COMPLETE MASKING (like in 09_country_transfer_validation.py)")

    # Create features (same as 09_country_transfer_validation.py)
    def make_features(df, windows=[5, 21]):
        features = pd.DataFrame(index=df.index)
        for col in df.columns:
            for w in windows:
                features[f'{col}_ret_{w}'] = df[col].rolling(w).mean().shift(1)
                features[f'{col}_vol_{w}'] = df[col].rolling(w).std().shift(1)
        return features.dropna()

    source_feat = make_features(df)

    # Create targets (same as 09_country_transfer_validation.py)
    y_source = (source_mom < source_mom.rolling(252).quantile(0.1).shift(1)).astype(int).values

    # Clean NaNs
    mask = ~(np.isnan(y_source) | np.isnan(regime_source.astype(float)))

    print(f"  Samples after feature engineering dropna(): {len(source_feat)}")
    print(f"  Valid mask (non-NaN): {mask.sum()} / {len(mask)}")

    # Get the regime values after masking
    regime_after_mask = regime_source.values[mask]
    print(f"  Regime distribution after masking:")
    print(f"    Regime 0: {(regime_after_mask == 0).sum()}")
    print(f"    Regime 1: {(regime_after_mask == 1).sum()}")

    # Find the boundary: where does the 252 lookback start having data?
    print(f"\n7. CRITICAL PERIOD ANALYSIS")
    print(f"  First 312 periods (63+252) have NaNs in rolling calculations")
    print(f"  Actual valid data starts around index {nan_from_vol.sum() + nan_from_median.sum()}")

    # Sample some regime values after the NaN period
    valid_indices = np.where(~(nan_from_vol | nan_from_median))[0]
    if len(valid_indices) > 0:
        print(f"\n  First 10 valid regime values (after NaN period):")
        for i in range(min(10, len(valid_indices))):
            idx = valid_indices[i]
            vol_val = vol_source.iloc[idx]
            median_val = median_vol_s.iloc[idx]
            regime_val = regime_source.iloc[idx]
            print(f"    idx={idx}: vol={vol_val:.6f}, median={median_val:.6f}, regime={regime_val}")

    print(f"\n" + "="*90)


if __name__ == '__main__':
    for country in ['US', 'Japan', 'Europe']:
        debug_regime_detection(country)
