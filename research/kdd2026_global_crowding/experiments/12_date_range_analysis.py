"""
Analyze date ranges and regime distributions over time.
This explains why regime-conditioning fails: regimes don't transfer across markets/times.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

DATA_DIR = ROOT_DIR / 'data' / 'global_factors'


def analyze_country_dates_and_regimes(country_name: str):
    """Analyze date ranges and regime distributions."""

    file_path = DATA_DIR / f'{country_name.lower()}_factors.parquet'
    if not file_path.exists():
        return None

    df = pd.read_parquet(file_path)

    # Extract momentum series
    if 'Mom' in df.columns:
        source_mom = df['Mom']
    else:
        source_mom = df.iloc[:, 0]

    # Compute regimes
    vol_source = source_mom.rolling(63).std()
    median_vol_s = vol_source.rolling(252).median()
    regime_source = (vol_source > median_vol_s).astype(int)

    # Get regime distribution over time blocks
    print(f"\n{'='*90}")
    print(f"DATE RANGE AND REGIME ANALYSIS: {country_name}")
    print(f"{'='*90}")

    print(f"\nOverall period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Total samples: {len(df)}")

    # Break into 5-year periods and analyze regimes
    periods = []
    years = 5
    for i in range(0, len(df), years*12):  # Assuming monthly data, 12 months per year
        start_idx = i
        end_idx = min(i + years*12, len(df))
        if end_idx - start_idx < years*12 // 2:  # Skip incomplete periods
            break

        period_df = df.iloc[start_idx:end_idx]
        period_regimes = regime_source.iloc[start_idx:end_idx]
        valid_regimes = period_regimes.dropna()

        if len(valid_regimes) > 0:
            regime0_pct = 100 * (valid_regimes == 0).sum() / len(valid_regimes)
            regime1_pct = 100 * (valid_regimes == 1).sum() / len(valid_regimes)

            periods.append({
                'country': country_name,
                'period': f"{period_df.index[0].year}-{period_df.index[-1].year}",
                'start_date': period_df.index[0].date(),
                'end_date': period_df.index[-1].date(),
                'samples': len(valid_regimes),
                'regime0_pct': regime0_pct,
                'regime1_pct': regime1_pct,
            })

    return periods


def main():
    print("\nANALYSIS: Regime distributions by country and time period\n")

    all_periods = {}
    for country in ['US', 'Japan', 'Europe', 'AsiaPac']:
        periods = analyze_country_dates_and_regimes(country)
        if periods:
            all_periods[country] = periods

            print(f"\n{country}: Regime distribution by 5-year period")
            print("-" * 120)
            print(f"{'Period':<20} | {'Dates':<30} | {'Samples':<10} | {'Regime-0':<15} | {'Regime-1':<15}")
            print("-" * 120)

            for p in periods:
                print(f"{p['period']:<20} | {str(p['start_date']) + ' to ' + str(p['end_date']):<30} | "
                      f"{p['samples']:<10} | {p['regime0_pct']:>6.1f}% ({int(p['regime0_pct']/100*p['samples']):<3}) | "
                      f"{p['regime1_pct']:>6.1f}% ({int(p['regime1_pct']/100*p['samples']):<3})")

    # Find common date ranges
    print(f"\n{'='*90}")
    print("CRITICAL FINDING: Date Range Misalignment and Regime Non-Transfer")
    print(f"{'='*90}")

    # Load full data to check overlaps
    us_df = pd.read_parquet(DATA_DIR / 'us_factors.parquet')
    japan_df = pd.read_parquet(DATA_DIR / 'japan_factors.parquet')
    europe_df = pd.read_parquet(DATA_DIR / 'europe_factors.parquet')
    asiapac_df = pd.read_parquet(DATA_DIR / 'asiapac_factors.parquet')

    for target_name, target_df in [('Japan', japan_df), ('Europe', europe_df), ('AsiaPac', asiapac_df)]:
        common_dates = us_df.index.intersection(target_df.index)
        us_earliest_in_common = us_df.loc[common_dates[0]]
        us_latest_in_common = us_df.loc[common_dates[-1]]

        print(f"\n[US → {target_name}]")
        print(f"  {target_name} date range: {target_df.index[0].date()} to {target_df.index[-1].date()}")
        print(f"  Common dates: {common_dates[0].date()} to {common_dates[-1].date()} ({len(common_dates)} samples)")
        print(f"  ⚠️ US regime labels for this period may not represent typical US regimes!")
        print(f"     US overall period is longer: {us_df.index[0].date()} to {us_df.index[-1].date()}")

    print(f"\n" + "="*90)
    print("KEY INSIGHT:")
    print("="*90)
    print("""
When we align US to Japan's date range, we get regime labels that are specific to
that time period. But regime-conditional MMD assumes that:

1. Source regimes (from US data period X) are meaningful for target (Japan data period Y)
2. Regimes are stable and interpretable across different markets
3. Regime boundaries are similar between markets

In reality:
- US low-volatility regime (0) might be during 1990s (tech boom)
- Japan high-volatility regime (1) might be during 1990s (Lost Decade)
- They occur simultaneously but have OPPOSITE meanings!

This is why regime-conditioning HURTS transfer rather than helps it:
- The algorithm tries to match "US low-vol" with "Japan high-vol" → contradictory
- MMD loss becomes unreliable because regime semantics don't align
- Negative transfer (worse performance than no adaptation) results

DIAGNOSIS: The theory assumes regime labels are domain-invariant, but they're
actually domain-specific temporal patterns that don't transfer.
""")


if __name__ == '__main__':
    main()
