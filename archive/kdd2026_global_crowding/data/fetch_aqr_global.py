"""
Fetch Global Factor Data from AQR Data Library

Source: "Value and Momentum Everywhere" (Asness et al.)
URL: https://www.aqr.com/Insights/Datasets/Value-and-Momentum-Everywhere-Factors-Monthly

Regions available:
- USA
- UK
- Continental Europe
- Japan
- Other (includes Asia, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent / 'global_factors'
DATA_DIR.mkdir(exist_ok=True)

AQR_URLS = {
    'VME': 'https://images.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Value-and-Momentum-Everywhere-Factors-Monthly.xlsx',
    'BAB': 'https://images.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Monthly.xlsx',
    'QMJ': 'https://images.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Quality-Minus-Junk-Factors-Monthly.xlsx',
}


def download_aqr_file(url: str) -> BytesIO:
    """Download AQR Excel file."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return BytesIO(response.content)


def parse_vme_data(file: BytesIO) -> dict:
    """Parse Value and Momentum Everywhere data."""

    # Read Value factors
    df_val = pd.read_excel(file, sheet_name='VAL', skiprows=18, index_col=0)
    df_val.index = pd.to_datetime(df_val.index)
    df_val = df_val.dropna(how='all')

    # Read Momentum factors
    file.seek(0)
    df_mom = pd.read_excel(file, sheet_name='MOM', skiprows=18, index_col=0)
    df_mom.index = pd.to_datetime(df_mom.index)
    df_mom = df_mom.dropna(how='all')

    # Regions mapping
    regions = {
        'USA': 'US',
        'UK': 'UK',
        'Europe ex UK': 'Europe',
        'Japan': 'Japan',
        'Asia ex Japan': 'AsiaPac',
        'Global': 'Global'
    }

    # Combine into regional datasets
    result = {}

    for aqr_name, our_name in regions.items():
        if aqr_name in df_val.columns and aqr_name in df_mom.columns:
            df = pd.DataFrame({
                'HML': df_val[aqr_name],
                'Mom': df_mom[aqr_name]
            })
            df = df.dropna()
            df = df / 100  # Convert from percentage
            result[our_name] = df

    return result


def parse_bab_data(file: BytesIO) -> dict:
    """Parse Betting Against Beta data."""

    df = pd.read_excel(file, sheet_name='BAB Factors', skiprows=18, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.dropna(how='all')

    # Available regions in BAB
    regions = {
        'USA': 'US',
        'Global': 'Global',
        'Europe': 'Europe',
        'Pacific': 'AsiaPac',
    }

    result = {}
    for aqr_name, our_name in regions.items():
        if aqr_name in df.columns:
            bab = df[[aqr_name]].dropna()
            bab.columns = ['BAB']
            bab = bab / 100
            result[our_name] = bab

    return result


def parse_qmj_data(file: BytesIO) -> dict:
    """Parse Quality Minus Junk data."""

    df = pd.read_excel(file, sheet_name='QMJ Factors', skiprows=18, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.dropna(how='all')

    # Available regions
    regions = {
        'USA': 'US',
        'Global': 'Global',
        'Global Ex USA': 'GlobalExUS',
        'Europe': 'Europe',
        'Japan': 'Japan',
    }

    result = {}
    for aqr_name, our_name in regions.items():
        if aqr_name in df.columns:
            qmj = df[[aqr_name]].dropna()
            qmj.columns = ['QMJ']
            qmj = qmj / 100
            result[our_name] = qmj

    return result


def fetch_all_aqr_data(save: bool = True) -> dict:
    """Fetch all AQR datasets and combine by region."""

    all_data = {}

    # Fetch Value and Momentum Everywhere
    print("Fetching Value and Momentum Everywhere...")
    try:
        file = download_aqr_file(AQR_URLS['VME'])
        vme = parse_vme_data(file)
        print(f"  ✓ VME: {len(vme)} regions")

        for region, df in vme.items():
            if region not in all_data:
                all_data[region] = df
            else:
                all_data[region] = all_data[region].join(df, how='outer')
    except Exception as e:
        print(f"  ✗ VME failed: {e}")

    # Fetch BAB
    print("Fetching Betting Against Beta...")
    try:
        file = download_aqr_file(AQR_URLS['BAB'])
        bab = parse_bab_data(file)
        print(f"  ✓ BAB: {len(bab)} regions")

        for region, df in bab.items():
            if region not in all_data:
                all_data[region] = df
            else:
                all_data[region] = all_data[region].join(df, how='outer')
    except Exception as e:
        print(f"  ✗ BAB failed: {e}")

    # Fetch QMJ
    print("Fetching Quality Minus Junk...")
    try:
        file = download_aqr_file(AQR_URLS['QMJ'])
        qmj = parse_qmj_data(file)
        print(f"  ✓ QMJ: {len(qmj)} regions")

        for region, df in qmj.items():
            if region not in all_data:
                all_data[region] = df
            else:
                all_data[region] = all_data[region].join(df, how='outer')
    except Exception as e:
        print(f"  ✗ QMJ failed: {e}")

    # Save
    if save:
        for region, df in all_data.items():
            df.to_parquet(DATA_DIR / f'{region.lower()}_factors.parquet')
            print(f"Saved {region}: {len(df)} months, {list(df.columns)}")

    return all_data


def load_all_regions() -> dict:
    """Load saved regional data."""
    all_data = {}

    for path in DATA_DIR.glob('*_factors.parquet'):
        region = path.stem.replace('_factors', '').upper()
        all_data[region] = pd.read_parquet(path)

    return all_data


if __name__ == '__main__':
    print("=" * 60)
    print("FETCHING AQR GLOBAL FACTOR DATA")
    print("=" * 60)

    data = fetch_all_aqr_data()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for region, df in sorted(data.items()):
        print(f"\n{region}:")
        print(f"  Period: {df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")
        print(f"  Months: {len(df)}")
        print(f"  Factors: {list(df.columns)}")
        print(f"  Missing: {df.isna().sum().to_dict()}")
