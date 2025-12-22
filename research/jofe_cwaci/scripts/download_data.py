"""
Download Fama-French Factor Data

Downloads factor returns from Kenneth French's data library.

Author: Chorok Lee (KAIST)
Date: December 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def download_ff_factors():
    """
    Download Fama-French factors using pandas-datareader.

    Returns
    -------
    pd.DataFrame
        Monthly factor returns (Mkt-RF, SMB, HML, RMW, CMA, Mom)
    """
    try:
        from pandas_datareader import data as pdr

        # Five factors (full history from 1963)
        ff5 = pdr.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench',
                             start='1963-01-01')[0]

        # Momentum (full history)
        mom = pdr.DataReader('F-F_Momentum_Factor', 'famafrench',
                            start='1963-01-01')[0]

        # Combine
        factors = ff5.copy()
        # Find momentum column (may have trailing spaces)
        mom_col = [c for c in mom.columns if 'mom' in c.lower() or 'Mom' in c][0]
        factors['Mom'] = mom[mom_col]

        # Convert to decimal
        factors = factors / 100

        # Drop RF column (we don't need it)
        if 'RF' in factors.columns:
            factors = factors.drop('RF', axis=1)

        return factors

    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Falling back to local file if available...")
        return None


def load_or_download_factors(data_dir: Path = None) -> pd.DataFrame:
    """
    Load factors from cache or download if not available.

    Parameters
    ----------
    data_dir : Path
        Directory to store/load data

    Returns
    -------
    pd.DataFrame
        Factor returns
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'data'

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    cache_file = data_dir / 'ff_factors.csv'

    if cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        factors = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return factors

    print("Downloading Fama-French factors...")
    factors = download_ff_factors()

    if factors is not None:
        # Save cache
        factors.to_csv(cache_file)
        print(f"Saved to {cache_file}")

    return factors


def describe_data(factors: pd.DataFrame) -> None:
    """Print summary statistics."""
    print("\n" + "="*60)
    print("FAMA-FRENCH FACTOR DATA SUMMARY")
    print("="*60)

    print(f"\nPeriod: {factors.index.min()} to {factors.index.max()}")
    print(f"Observations: {len(factors)}")
    print(f"Factors: {list(factors.columns)}")

    print("\n--- Summary Statistics (Monthly Returns) ---")
    stats = factors.describe().T[['mean', 'std', 'min', 'max']]
    stats['sharpe'] = stats['mean'] / stats['std'] * np.sqrt(12)
    stats['ann_ret'] = stats['mean'] * 12
    stats['ann_vol'] = stats['std'] * np.sqrt(12)

    print(stats.round(4))

    print("\n--- Correlations ---")
    print(factors.corr().round(2))


if __name__ == '__main__':
    # Download and describe data
    factors = load_or_download_factors()

    if factors is not None:
        describe_data(factors)
    else:
        print("Failed to load factor data.")
