"""
Fetch Global Factor Data from Ken French Data Library

Regions:
- US (baseline)
- Developed ex-US
- Europe
- Japan
- Asia Pacific ex-Japan
- Emerging Markets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas_datareader.data as web
    HAS_DATAREADER = True
except ImportError:
    HAS_DATAREADER = False
    print("Install pandas_datareader: pip install pandas-datareader")

DATA_DIR = Path(__file__).parent / 'global_factors'
DATA_DIR.mkdir(exist_ok=True)

# Region configurations
REGIONS = {
    'US': {
        'ff5': 'F-F_Research_Data_5_Factors_2x3',
        'mom': 'F-F_Momentum_Factor',
        'name': 'United States'
    },
    'Developed': {
        'ff5': 'Developed_5_Factors',
        'mom': 'Developed_Mom_Factor',
        'name': 'Developed ex-US'
    },
    'Europe': {
        'ff5': 'Europe_5_Factors',
        'mom': 'Europe_Mom_Factor',
        'name': 'Europe'
    },
    'Japan': {
        'ff5': 'Japan_5_Factors',
        'mom': 'Japan_Mom_Factor',
        'name': 'Japan'
    },
    'AsiaPac': {
        'ff5': 'Asia_Pacific_ex_Japan_5_Factors',
        'mom': 'Asia_Pacific_ex_Japan_Mom_Factor',
        'name': 'Asia Pacific ex-Japan'
    },
    'EM': {
        'ff5': 'Emerging_5_Factors',
        'mom': 'Emerging_Mom_Factor',
        'name': 'Emerging Markets'
    },
}


def fetch_region_factors(region: str, start: str = '1990-01-01') -> pd.DataFrame:
    """Fetch FF5 + Momentum for a single region."""
    if not HAS_DATAREADER:
        raise ImportError("pandas_datareader required")

    config = REGIONS[region]

    try:
        # Fetch FF5
        ff5 = web.DataReader(config['ff5'], 'famafrench', start=start)[0]
        ff5 = ff5 / 100  # Convert from percentage

        # Fetch Momentum
        mom = web.DataReader(config['mom'], 'famafrench', start=start)[0]
        mom = mom / 100
        mom.columns = ['Mom']

        # Combine
        factors = ff5.join(mom, how='inner')

        # Convert PeriodIndex to Timestamp
        if isinstance(factors.index, pd.PeriodIndex):
            factors.index = factors.index.to_timestamp()

        # Standardize column names
        factors.columns = [c.replace('-', '_').replace(' ', '_') for c in factors.columns]
        if 'Mkt_RF' in factors.columns:
            factors = factors.rename(columns={'Mkt_RF': 'MKT'})

        return factors

    except Exception as e:
        print(f"Error fetching {region}: {e}")
        return pd.DataFrame()


def fetch_all_regions(start: str = '1990-01-01', save: bool = True) -> dict:
    """Fetch data for all regions."""
    all_data = {}

    for region in REGIONS:
        print(f"Fetching {REGIONS[region]['name']}...")
        df = fetch_region_factors(region, start)

        if len(df) > 0:
            all_data[region] = df
            print(f"  ✓ {len(df)} months, {df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")

            if save:
                df.to_parquet(DATA_DIR / f'{region.lower()}_factors.parquet')
        else:
            print(f"  ✗ Failed")

    return all_data


def load_all_regions() -> dict:
    """Load previously saved regional data."""
    all_data = {}

    for region in REGIONS:
        path = DATA_DIR / f'{region.lower()}_factors.parquet'
        if path.exists():
            all_data[region] = pd.read_parquet(path)

    return all_data


def get_common_period(data: dict) -> tuple:
    """Find common date range across all regions."""
    start_dates = [df.index.min() for df in data.values()]
    end_dates = [df.index.max() for df in data.values()]

    return max(start_dates), min(end_dates)


def align_regions(data: dict) -> dict:
    """Align all regions to common date range."""
    start, end = get_common_period(data)

    aligned = {}
    for region, df in data.items():
        aligned[region] = df.loc[start:end]

    return aligned


if __name__ == '__main__':
    print("=" * 60)
    print("FETCHING GLOBAL FACTOR DATA")
    print("=" * 60)

    data = fetch_all_regions()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for region, df in data.items():
        print(f"\n{REGIONS[region]['name']}:")
        print(f"  Period: {df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")
        print(f"  Months: {len(df)}")
        print(f"  Factors: {list(df.columns)}")

    # Common period
    if len(data) > 1:
        start, end = get_common_period(data)
        print(f"\nCommon period: {start.strftime('%Y-%m')} to {end.strftime('%Y-%m')}")
