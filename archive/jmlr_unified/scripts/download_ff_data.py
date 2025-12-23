"""
Download Fama-French factor data from Kenneth French Data Library.

This script downloads all factors needed for the JMLR paper:
- FF 5 factors (Mkt-RF, SMB, HML, RMW, CMA)
- Momentum (MOM)
- Short-term reversal (ST_Rev)
- Long-term reversal (LT_Rev)

Run: python scripts/download_ff_data.py
"""

import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def download_ff_factors():
    """Download and save FF factors from Kenneth French Data Library."""

    # Setup paths
    base_path = Path(__file__).parent.parent
    data_dir = base_path / "data" / "factor_crowding"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("DOWNLOADING FAMA-FRENCH FACTORS")
    print("="*60)

    try:
        import pandas_datareader.data as web
    except ImportError:
        print("ERROR: pandas_datareader not installed.")
        print("Run: pip install pandas-datareader")
        return None

    datasets = {}

    # Download FF 5 factors
    print("\n[1/4] Downloading FF 5 Factors...")
    try:
        ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='1963-07-01')
        datasets['ff5'] = ff5[0]  # Monthly data
        print(f"  ✓ Downloaded: {len(datasets['ff5'])} months")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None

    # Download momentum
    print("\n[2/4] Downloading Momentum Factor...")
    try:
        mom = web.DataReader('F-F_Momentum_Factor', 'famafrench', start='1963-07-01')
        datasets['mom'] = mom[0]
        print(f"  ✓ Downloaded: {len(datasets['mom'])} months")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        datasets['mom'] = None

    # Download short-term reversal
    print("\n[3/4] Downloading Short-Term Reversal Factor...")
    try:
        st_rev = web.DataReader('F-F_ST_Reversal_Factor', 'famafrench', start='1963-07-01')
        datasets['st_rev'] = st_rev[0]
        print(f"  ✓ Downloaded: {len(datasets['st_rev'])} months")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        datasets['st_rev'] = None

    # Download long-term reversal
    print("\n[4/4] Downloading Long-Term Reversal Factor...")
    try:
        lt_rev = web.DataReader('F-F_LT_Reversal_Factor', 'famafrench', start='1963-07-01')
        datasets['lt_rev'] = lt_rev[0]
        print(f"  ✓ Downloaded: {len(datasets['lt_rev'])} months")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        datasets['lt_rev'] = None

    # Combine all factors
    print("\n" + "="*60)
    print("COMBINING DATASETS")
    print("="*60)

    combined = datasets['ff5'].copy()

    if datasets['mom'] is not None:
        combined = combined.join(datasets['mom'], how='outer')

    if datasets['st_rev'] is not None:
        combined = combined.join(datasets['st_rev'], how='outer')

    if datasets['lt_rev'] is not None:
        combined = combined.join(datasets['lt_rev'], how='outer')

    # Rename columns for clarity
    col_mapping = {
        'Mkt-RF': 'Mkt-RF',
        'SMB': 'SMB',
        'HML': 'HML',
        'RMW': 'RMW',
        'CMA': 'CMA',
        'RF': 'RF',
        'Mom   ': 'Mom',
        'MOM': 'Mom',
        'ST_Rev': 'ST_Rev',
        'ST Rev': 'ST_Rev',
        'LT_Rev': 'LT_Rev',
        'LT Rev': 'LT_Rev'
    }

    combined.columns = [col_mapping.get(c.strip(), c.strip()) for c in combined.columns]

    # Remove duplicates if any
    combined = combined.loc[:, ~combined.columns.duplicated()]

    # Save as parquet
    output_path = data_dir / 'ff_factors_monthly.parquet'
    combined.to_parquet(output_path)

    # Also save as CSV for inspection
    csv_path = data_dir / 'ff_factors_monthly.csv'
    combined.to_csv(csv_path)

    print(f"\n✓ Saved to: {output_path}")
    print(f"✓ CSV copy: {csv_path}")
    print(f"\nData Summary:")
    print(f"  Periods: {len(combined)} months")
    print(f"  Date range: {combined.index[0]} to {combined.index[-1]}")
    print(f"  Columns: {combined.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(combined.head())
    print(f"\nLast 5 rows:")
    print(combined.tail())

    return combined


def verify_data():
    """Verify downloaded data."""
    base_path = Path(__file__).parent.parent
    data_path = base_path / "data" / "factor_crowding" / "ff_factors_monthly.parquet"

    if not data_path.exists():
        print(f"Data not found at {data_path}")
        return False

    df = pd.read_parquet(data_path)

    print("\n" + "="*60)
    print("DATA VERIFICATION")
    print("="*60)

    required_factors = ['SMB', 'HML', 'RMW', 'CMA', 'Mom']

    for factor in required_factors:
        if factor in df.columns:
            non_null = df[factor].notna().sum()
            print(f"  ✓ {factor}: {non_null} observations")
        else:
            print(f"  ✗ {factor}: MISSING")

    return True


if __name__ == '__main__':
    df = download_ff_factors()
    if df is not None:
        verify_data()
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE - Ready for analysis")
        print("="*60)
