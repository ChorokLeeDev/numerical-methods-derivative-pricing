"""
Rolling Crowding Acceleration Signal

Core idea: The gap between predicted and actual alpha reveals crowding dynamics.
- Negative residual = crowding accelerated (danger)
- Positive residual = crowding decelerated (opportunity)

This module builds a real-time crowding detector.
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class CrowdingSignal:
    """Container for crowding signal output."""
    date: pd.Timestamp
    factor: str
    actual_sharpe: float
    predicted_sharpe: float
    residual: float
    cumulative_residual: float
    model_params: Tuple[float, float]  # (K, lambda)
    signal: str  # 'crowding', 'uncrowding', 'neutral'


def alpha_decay_model(t: np.ndarray, K: float, lam: float) -> np.ndarray:
    """Hyperbolic decay: α(t) = K / (1 + λt)"""
    return K / (1 + lam * t)


def rolling_sharpe(returns: pd.Series, window: int = 36) -> pd.Series:
    """Compute rolling annualized Sharpe ratio."""
    return (returns.rolling(window).mean() /
            returns.rolling(window).std() * np.sqrt(12))


def fit_decay_model(sharpe: pd.Series) -> Optional[Tuple[float, float]]:
    """
    Fit hyperbolic decay model to Sharpe ratio series.
    Returns (K, lambda) or None if fitting fails.
    """
    t = np.arange(len(sharpe))
    y = sharpe.values

    # Only fit on positive values
    mask = y > 0
    if mask.sum() < 20:
        return None

    t_pos, y_pos = t[mask], y[mask]

    try:
        popt, _ = curve_fit(
            alpha_decay_model,
            t_pos, y_pos,
            p0=[1.5, 0.01],
            bounds=([0, 0], [10, 0.5]),
            maxfev=5000
        )
        return tuple(popt)
    except Exception:
        return None


class CrowdingDetector:
    """
    Real-time crowding acceleration detector.

    Strategy:
    1. At each month t, fit model on [t-train_window, t-gap]
    2. Predict [t-gap, t]
    3. Compute residual = actual - predicted
    4. Generate signal based on residual magnitude
    """

    def __init__(
        self,
        train_window: int = 120,  # 10 years
        prediction_gap: int = 12,  # 1 year holdout
        sharpe_window: int = 36,   # 3 years for Sharpe
        signal_threshold: float = 0.10,  # Residual threshold for signal
    ):
        self.train_window = train_window
        self.prediction_gap = prediction_gap
        self.sharpe_window = sharpe_window
        self.signal_threshold = signal_threshold

    def compute_rolling_signal(
        self,
        returns: pd.Series,
        factor_name: str = 'Factor'
    ) -> pd.DataFrame:
        """
        Compute rolling crowding signal for a factor.

        Returns DataFrame with:
        - actual_sharpe
        - predicted_sharpe
        - residual
        - cumulative_residual
        - signal
        """
        # Compute rolling Sharpe
        sharpe = rolling_sharpe(returns, self.sharpe_window).dropna()

        results = []
        min_start = self.train_window + self.prediction_gap + self.sharpe_window

        for i in range(min_start, len(sharpe)):
            # Training data: [i - train_window - prediction_gap, i - prediction_gap]
            train_end = i - self.prediction_gap
            train_start = max(0, train_end - self.train_window)

            train_sharpe = sharpe.iloc[train_start:train_end]

            # Fit model
            params = fit_decay_model(train_sharpe)
            if params is None:
                continue

            K, lam = params

            # Predict current period
            t_pred = len(train_sharpe)  # Time index for prediction
            predicted = alpha_decay_model(np.array([t_pred]), K, lam)[0]
            actual = sharpe.iloc[i]

            residual = actual - predicted

            # Determine signal
            if residual < -self.signal_threshold:
                signal = 'crowding'
            elif residual > self.signal_threshold:
                signal = 'uncrowding'
            else:
                signal = 'neutral'

            results.append({
                'date': sharpe.index[i],
                'factor': factor_name,
                'actual_sharpe': actual,
                'predicted_sharpe': predicted,
                'residual': residual,
                'K': K,
                'lambda': lam,
                'signal': signal,
            })

        df = pd.DataFrame(results)
        if len(df) > 0:
            df['cumulative_residual'] = df['residual'].cumsum()
            df = df.set_index('date')

        return df

    def compute_multi_factor_signals(
        self,
        factor_returns: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Compute crowding signals for multiple factors."""
        signals = {}
        for factor in factor_returns.columns:
            signal_df = self.compute_rolling_signal(
                factor_returns[factor],
                factor_name=factor
            )
            if len(signal_df) > 0:
                signals[factor] = signal_df
        return signals


def compute_crowding_score(signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate crowding signals into a single score per factor.

    Score = rolling mean of residual (negative = crowded)
    """
    scores = []

    for factor, df in signals.items():
        if len(df) == 0:
            continue

        # 12-month rolling average residual
        df['crowding_score'] = df['residual'].rolling(12).mean()

        # Z-score for comparability
        df['crowding_zscore'] = (
            (df['crowding_score'] - df['crowding_score'].mean()) /
            df['crowding_score'].std()
        )

        scores.append(df[['factor', 'crowding_score', 'crowding_zscore', 'signal']])

    if scores:
        return pd.concat(scores)
    return pd.DataFrame()
