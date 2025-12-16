"""
Fetch Extended Factor Data from Ken French's Data Library

Includes:
- Fama-French 5 Factors
- Momentum
- Short-term Reversal
- Long-term Reversal
- Industry portfolios (for robustness)
"""

import pandas as pd
import pandas_datareader.data as web
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent


def fetch_ff5_factors():
    """Fetch Fama-French 5 factors + momentum."""
    print("Fetching Fama-French 5 Factors...")

    # FF5 factors
    ff5 = web.DataReader(
        'F-F_Research_Data_5_Factors_2x3',
        'famafrench',
        start='1963-01-01'
    )[0]  # Monthly data

    # Momentum
    print("Fetching Momentum...")
    mom = web.DataReader(
        'F-F_Momentum_Factor',
        'famafrench',
        start='1963-01-01'
    )[0]

    # Short-term reversal
    print("Fetching Short-term Reversal...")
    try:
        st_rev = web.DataReader(
            'F-F_ST_Reversal_Factor',
            'famafrench',
            start='1963-01-01'
        )[0]
        st_rev.columns = ['ST_Rev']
    except Exception as e:
        print(f"  ST Reversal not available: {e}")
        st_rev = None

    # Long-term reversal
    print("Fetching Long-term Reversal...")
    try:
        lt_rev = web.DataReader(
            'F-F_LT_Reversal_Factor',
            'famafrench',
            start='1963-01-01'
        )[0]
        lt_rev.columns = ['LT_Rev']
    except Exception as e:
        print(f"  LT Reversal not available: {e}")
        lt_rev = None

    # Combine
    factors = ff5.copy()

    # Add momentum
    factors = factors.join(mom, how='outer')

    # Add reversals if available
    if st_rev is not None:
        factors = factors.join(st_rev, how='outer')
    if lt_rev is not None:
        factors = factors.join(lt_rev, how='outer')

    # Convert to returns (divide by 100)
    factors = factors / 100

    # Clean column names
    factors.columns = [c.strip() for c in factors.columns]

    # Rename for clarity
    rename_map = {
        'Mkt-RF': 'MKT',
        'Mom   ': 'Mom',
        'WML': 'Mom',
    }
    factors = factors.rename(columns=rename_map)

    print(f"  Factors: {list(factors.columns)}")
    print(f"  Date range: {factors.index.min()} to {factors.index.max()}")

    return factors


def fetch_industry_portfolios():
    """Fetch industry portfolios for cross-sectional tests."""
    print("Fetching Industry Portfolios...")

    industries = web.DataReader(
        '10_Industry_Portfolios',
        'famafrench',
        start='1963-01-01'
    )[0]  # Monthly value-weighted returns

    industries = industries / 100
    industries.columns = [f'Ind_{c.strip()}' for c in industries.columns]

    print(f"  Industries: {list(industries.columns)}")

    return industries


def main():
    print("=" * 60)
    print("FETCHING EXTENDED FACTOR DATA")
    print("=" * 60)

    # Fetch factors
    factors = fetch_ff5_factors()

    # Fetch industries
    industries = fetch_industry_portfolios()

    # Save
    print("\nSaving data...")

    factors.to_parquet(DATA_DIR / 'ff_extended_factors.parquet')
    print(f"  Saved: ff_extended_factors.parquet ({len(factors)} rows)")

    industries.to_parquet(DATA_DIR / 'industry_portfolios.parquet')
    print(f"  Saved: industry_portfolios.parquet ({len(industries)} rows)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nExtended Factors ({len(factors.columns)} total):")
    for col in factors.columns:
        valid = factors[col].dropna()
        print(f"  {col:<10}: {valid.index.min()} to {valid.index.max()}, "
              f"mean={valid.mean()*12:.1%}/yr")

    print("\nDone!")


if __name__ == '__main__':
    main()
