"""
Data caching utilities using Parquet format.

Provides efficient caching for price and fundamental data to reduce API calls.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


# Default cache directory
CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'cache'


def get_cache_path(cache_key: str, cache_dir: str | Path = None) -> Path:
    """
    Get the full path for a cache file.

    Parameters:
        cache_key: Unique identifier for the cached data
        cache_dir: Cache directory (default: ./data/cache)

    Returns:
        Path object for the cache file
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_key}.parquet"


def save_to_cache(
    data: pd.DataFrame,
    cache_key: str,
    cache_dir: str | Path = None
) -> None:
    """
    Save DataFrame to Parquet file for caching.

    Parameters:
        data: DataFrame to cache
        cache_key: Unique identifier for the cached data
        cache_dir: Cache directory (default: ./data/cache)
    """
    cache_path = get_cache_path(cache_key, cache_dir)
    data.to_parquet(cache_path)


def load_from_cache(
    cache_key: str,
    cache_dir: str | Path = None,
    max_age_days: int = 1
) -> pd.DataFrame | None:
    """
    Load DataFrame from cache if exists and not expired.

    Parameters:
        cache_key: Unique identifier for the cached data
        cache_dir: Cache directory (default: ./data/cache)
        max_age_days: Maximum age in days before cache expires

    Returns:
        DataFrame if cache hit and not expired, None otherwise
    """
    cache_path = get_cache_path(cache_key, cache_dir)

    if not cache_path.exists():
        return None

    # Check file age
    file_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    age = datetime.now() - file_mtime

    if age > timedelta(days=max_age_days):
        return None

    return pd.read_parquet(cache_path)


def clear_cache(cache_dir: str | Path = None) -> int:
    """
    Clear all cached files.

    Parameters:
        cache_dir: Cache directory (default: ./data/cache)

    Returns:
        Number of files deleted
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        return 0

    count = 0
    for f in cache_dir.glob('*.parquet'):
        f.unlink()
        count += 1

    return count


def get_cache_info(cache_dir: str | Path = None) -> pd.DataFrame:
    """
    Get information about cached files.

    Parameters:
        cache_dir: Cache directory (default: ./data/cache)

    Returns:
        DataFrame with cache file info (name, size, age)
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        return pd.DataFrame(columns=['name', 'size_mb', 'age_hours'])

    info = []
    now = datetime.now()

    for f in cache_dir.glob('*.parquet'):
        stat = f.stat()
        age = now - datetime.fromtimestamp(stat.st_mtime)
        info.append({
            'name': f.stem,
            'size_mb': stat.st_size / (1024 * 1024),
            'age_hours': age.total_seconds() / 3600
        })

    return pd.DataFrame(info)
