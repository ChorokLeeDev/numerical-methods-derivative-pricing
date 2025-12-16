"""
Data Collection for Factor Crowding Research

Sources:
1. Fama-French factor returns (Ken French Data Library)
2. Factor ETF data (Yahoo Finance)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent


# =============================================================================
# 1. Fama-French Factor Returns
# =============================================================================

def fetch_fama_french_factors():
    """
    Fetch Fama-French 5 factors + Momentum from Ken French's website.

    Returns daily and monthly factor returns:
    - Mkt-RF: Market excess return
    - SMB: Small minus Big (size)
    - HML: High minus Low (value)
    - RMW: Robust minus Weak (profitability)
    - CMA: Conservative minus Aggressive (investment)
    - Mom: Momentum (from separate file)
    """
    try:
        import pandas_datareader.data as web

        # Fama-French 5 Factors (daily)
        ff5_daily = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily',
                                    'famafrench',
                                    start='1990-01-01')[0]
        ff5_daily.index = pd.to_datetime(ff5_daily.index, format='%Y%m%d')
        ff5_daily = ff5_daily / 100  # Convert from percentage

        # Momentum factor (daily)
        mom_daily = web.DataReader('F-F_Momentum_Factor_daily',
                                   'famafrench',
                                   start='1990-01-01')[0]
        mom_daily.index = pd.to_datetime(mom_daily.index, format='%Y%m%d')
        mom_daily = mom_daily / 100
        mom_daily.columns = ['Mom']

        # Combine
        factors_daily = ff5_daily.join(mom_daily, how='inner')

        # Monthly version
        ff5_monthly = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                                     'famafrench',
                                     start='1990-01-01')[0]
        ff5_monthly = ff5_monthly / 100

        mom_monthly = web.DataReader('F-F_Momentum_Factor',
                                     'famafrench',
                                     start='1990-01-01')[0]
        mom_monthly = mom_monthly / 100
        mom_monthly.columns = ['Mom']

        factors_monthly = ff5_monthly.join(mom_monthly, how='inner')

        return factors_daily, factors_monthly

    except Exception as e:
        print(f"pandas_datareader failed: {e}")
        print("Falling back to direct download...")
        return fetch_fama_french_direct()


def fetch_fama_french_direct():
    """Direct download from Ken French's website."""

    base_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

    # FF5 daily
    import urllib.request
    import zipfile
    import io

    def download_and_parse(filename, skiprows=3):
        url = base_url + filename
        response = urllib.request.urlopen(url)
        zip_file = zipfile.ZipFile(io.BytesIO(response.read()))

        # Get the CSV inside
        csv_name = zip_file.namelist()[0]
        with zip_file.open(csv_name) as f:
            df = pd.read_csv(f, skiprows=skiprows, index_col=0)
            # Find where annual data starts (usually blank row or text)
            for i, idx in enumerate(df.index):
                try:
                    int(str(idx).replace(' ', ''))
                except:
                    df = df.iloc[:i]
                    break
            df.index = pd.to_datetime(df.index.astype(str), format='%Y%m%d', errors='coerce')
            df = df.dropna()
            df = df.apply(pd.to_numeric, errors='coerce') / 100
        return df

    factors_daily = download_and_parse("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")

    return factors_daily, None


# =============================================================================
# 2. Factor ETF Data (as crowding proxy)
# =============================================================================

FACTOR_ETFS = {
    # Momentum
    'MTUM': 'iShares MSCI USA Momentum Factor',
    'PDP': 'Invesco DWA Momentum',

    # Value
    'VTV': 'Vanguard Value',
    'IWD': 'iShares Russell 1000 Value',
    'VLUE': 'iShares MSCI USA Value Factor',

    # Size (Small Cap)
    'IWM': 'iShares Russell 2000',
    'VB': 'Vanguard Small-Cap',

    # Quality
    'QUAL': 'iShares MSCI USA Quality Factor',

    # Low Volatility
    'USMV': 'iShares MSCI USA Min Vol Factor',
    'SPLV': 'Invesco S&P 500 Low Volatility',

    # Multi-factor
    'LRGF': 'iShares MSCI USA Multifactor',
}


def fetch_etf_data(tickers=None, start='2010-01-01'):
    """
    Fetch ETF price and volume data from Yahoo Finance.

    Volume × Price ≈ Dollar volume (proxy for flows)
    """
    import yfinance as yf

    if tickers is None:
        tickers = list(FACTOR_ETFS.keys())

    data = {}
    for ticker in tickers:
        try:
            etf = yf.Ticker(ticker)
            hist = etf.history(start=start)
            if len(hist) > 0:
                data[ticker] = {
                    'price': hist['Close'],
                    'volume': hist['Volume'],
                    'dollar_volume': hist['Close'] * hist['Volume'],
                }
                print(f"✓ {ticker}: {len(hist)} days")
        except Exception as e:
            print(f"✗ {ticker}: {e}")

    return data


def fetch_etf_aum(tickers=None):
    """
    Fetch current AUM for factor ETFs.
    Note: Historical AUM requires paid data sources.
    """
    import yfinance as yf

    if tickers is None:
        tickers = list(FACTOR_ETFS.keys())

    aum_data = {}
    for ticker in tickers:
        try:
            etf = yf.Ticker(ticker)
            info = etf.info
            aum = info.get('totalAssets', None)
            if aum:
                aum_data[ticker] = {
                    'aum': aum,
                    'aum_billions': aum / 1e9,
                    'name': FACTOR_ETFS.get(ticker, ticker)
                }
                print(f"✓ {ticker}: ${aum/1e9:.1f}B")
        except Exception as e:
            print(f"✗ {ticker}: {e}")

    return pd.DataFrame(aum_data).T


# =============================================================================
# 3. Derived: Crowding Indicators
# =============================================================================

def calculate_crowding_proxy(factor_returns, etf_dollar_volume, window=60):
    """
    Calculate crowding proxy: Rolling correlation between
    factor returns and ETF flows (dollar volume).

    Intuition: When factor is crowded, flows and returns decouple
    (everyone already in, no marginal buyer).
    """
    # Align dates
    common_dates = factor_returns.index.intersection(etf_dollar_volume.index)

    factor_aligned = factor_returns.loc[common_dates]
    flows_aligned = etf_dollar_volume.loc[common_dates]

    # Rolling correlation
    crowding = factor_aligned.rolling(window).corr(flows_aligned)

    return crowding


# =============================================================================
# Main: Download and Save
# =============================================================================

def main():
    print("=" * 60)
    print("Fetching Factor Crowding Data")
    print("=" * 60)

    # 1. Fama-French factors
    print("\n1. Fama-French Factor Returns")
    print("-" * 40)
    try:
        factors_daily, factors_monthly = fetch_fama_french_factors()
        if factors_daily is not None:
            factors_daily.to_parquet(DATA_DIR / 'ff_factors_daily.parquet')
            print(f"   Saved: ff_factors_daily.parquet ({len(factors_daily)} days)")
        if factors_monthly is not None:
            factors_monthly.to_parquet(DATA_DIR / 'ff_factors_monthly.parquet')
            print(f"   Saved: ff_factors_monthly.parquet ({len(factors_monthly)} months)")
    except Exception as e:
        print(f"   Error: {e}")

    # 2. Factor ETF data
    print("\n2. Factor ETF Data")
    print("-" * 40)
    try:
        etf_data = fetch_etf_data()

        # Combine into DataFrames
        prices = pd.DataFrame({k: v['price'] for k, v in etf_data.items()})
        volumes = pd.DataFrame({k: v['dollar_volume'] for k, v in etf_data.items()})

        prices.to_parquet(DATA_DIR / 'etf_prices.parquet')
        volumes.to_parquet(DATA_DIR / 'etf_dollar_volumes.parquet')
        print(f"   Saved: etf_prices.parquet, etf_dollar_volumes.parquet")
    except Exception as e:
        print(f"   Error: {e}")

    # 3. Current ETF AUM
    print("\n3. Factor ETF AUM (Current)")
    print("-" * 40)
    try:
        aum = fetch_etf_aum()
        aum.to_parquet(DATA_DIR / 'etf_aum_current.parquet')
        print(f"   Saved: etf_aum_current.parquet")
        print(aum[['name', 'aum_billions']].to_string())
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == '__main__':
    main()
