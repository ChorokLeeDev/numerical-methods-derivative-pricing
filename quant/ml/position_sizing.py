"""
Confidence-based position sizing.

Key idea: Size positions based on prediction confidence
- High confidence → Larger position
- Low confidence → Smaller position

This naturally implements the Bayesian principle:
"Don't bet big when you're uncertain"
"""

import numpy as np
import pandas as pd
from typing import Optional


def confidence_weighted_portfolio(
    predictions: pd.DataFrame,
    n_long: int = 20,
    n_short: int = 20,
    confidence_col: str = 'confidence_normalized',
    return_col: str = 'q50',
    min_weight: float = 0.01,
    max_weight: float = 0.15
) -> tuple[pd.Series, pd.Series]:
    """
    Construct portfolio with confidence-weighted positions.

    High-confidence, high-return → Large long position
    High-confidence, low-return → Large short position
    Low-confidence → Small position (regardless of direction)

    Parameters:
        predictions: DataFrame with return predictions and confidence
        n_long: Number of long positions
        n_short: Number of short positions
        confidence_col: Column name for confidence scores
        return_col: Column name for return predictions (q50 = median)
        min_weight: Minimum weight per position
        max_weight: Maximum weight per position

    Returns:
        (long_weights, short_weights) as pandas Series

    Example:
        >>> long_w, short_w = confidence_weighted_portfolio(predictions)
        >>> portfolio_return = (long_w * returns).sum() - (short_w * returns).sum()
    """
    # Clean predictions
    preds = predictions.dropna(subset=[return_col, confidence_col])

    if len(preds) < n_long + n_short:
        # Not enough stocks, equal weight fallback
        n_half = len(preds) // 2
        sorted_preds = preds.sort_values(return_col, ascending=False)
        top = sorted_preds.head(n_half)
        bottom = sorted_preds.tail(n_half)

        long_weights = pd.Series(1.0 / len(top), index=top.index.get_level_values('ticker'))
        short_weights = pd.Series(1.0 / len(bottom), index=bottom.index.get_level_values('ticker'))
        return long_weights, short_weights

    # Rank by predicted return
    preds = preds.sort_values(return_col, ascending=False)

    # Select top/bottom stocks
    long_candidates = preds.head(n_long).copy()
    short_candidates = preds.tail(n_short).copy()

    # Weight by confidence
    long_weights = _confidence_to_weights(
        long_candidates[confidence_col],
        min_weight, max_weight
    )
    short_weights = _confidence_to_weights(
        short_candidates[confidence_col],
        min_weight, max_weight
    )

    # Extract ticker from multi-index
    if isinstance(long_weights.index, pd.MultiIndex):
        long_weights.index = long_weights.index.get_level_values('ticker')
        short_weights.index = short_weights.index.get_level_values('ticker')

    return long_weights, short_weights


def _confidence_to_weights(
    confidence: pd.Series,
    min_weight: float,
    max_weight: float
) -> pd.Series:
    """
    Convert confidence scores to portfolio weights.

    Uses softmax-like transformation to ensure weights sum to 1.
    """
    # Normalize confidence to [0, 1]
    conf = confidence.copy()
    if conf.std() > 0:
        conf = (conf - conf.min()) / (conf.max() - conf.min())
    else:
        conf = pd.Series(0.5, index=conf.index)

    # Apply temperature scaling (higher = more equal weights)
    temperature = 2.0
    exp_conf = np.exp(conf / temperature)

    # Initial weights
    weights = exp_conf / exp_conf.sum()

    # Clip to bounds
    weights = weights.clip(lower=min_weight, upper=max_weight)

    # Renormalize
    weights = weights / weights.sum()

    return weights


def kelly_criterion_adjustment(
    predictions: pd.DataFrame,
    win_rate: float = 0.55,
    avg_win: float = 0.05,
    avg_loss: float = 0.03,
    max_kelly_fraction: float = 0.5
) -> pd.DataFrame:
    """
    Apply Kelly criterion to adjust position sizes.

    Kelly formula: f* = (p * b - q) / b
    where:
        p = probability of win
        b = odds (avg_win / avg_loss)
        q = probability of loss = 1 - p

    Parameters:
        predictions: Predictions with confidence
        win_rate: Historical win rate
        avg_win: Average winning return
        avg_loss: Average losing return (positive number)
        max_kelly_fraction: Maximum fraction of Kelly to use

    Returns:
        Predictions with 'kelly_fraction' column
    """
    result = predictions.copy()

    # Basic Kelly
    odds = avg_win / avg_loss
    full_kelly = (win_rate * odds - (1 - win_rate)) / odds

    # Adjust by confidence
    # High confidence → closer to full Kelly
    # Low confidence → more conservative
    confidence = result['confidence_normalized'].fillna(0.5)

    # Scale Kelly by confidence
    kelly_fraction = full_kelly * confidence * max_kelly_fraction

    # Clip to reasonable bounds
    kelly_fraction = kelly_fraction.clip(lower=0.01, upper=max_kelly_fraction)

    result['kelly_fraction'] = kelly_fraction

    return result


def risk_parity_confidence_blend(
    predictions: pd.DataFrame,
    volatilities: pd.Series,
    confidence_weight: float = 0.5
) -> pd.Series:
    """
    Blend risk parity with confidence weighting.

    Standard risk parity: weight ∝ 1/volatility
    Confidence adjustment: weight × confidence

    Parameters:
        predictions: Predictions with confidence
        volatilities: Volatility per stock
        confidence_weight: How much to weight confidence vs risk parity

    Returns:
        Blended weights
    """
    # Risk parity weights
    inv_vol = 1.0 / volatilities.clip(lower=0.01)
    rp_weights = inv_vol / inv_vol.sum()

    # Confidence weights
    conf = predictions['confidence_normalized']
    conf_weights = conf / conf.sum()

    # Blend
    blended = (1 - confidence_weight) * rp_weights + confidence_weight * conf_weights

    # Renormalize
    blended = blended / blended.sum()

    return blended


def dynamic_factor_weights(
    factor_predictions: dict[str, pd.DataFrame],
    base_weights: dict[str, float] = None
) -> dict[str, float]:
    """
    Dynamically adjust factor weights based on prediction confidence.

    If momentum model has high confidence but value model has low confidence,
    increase momentum weight and decrease value weight.

    Parameters:
        factor_predictions: Dict of {factor_name: predictions_df}
        base_weights: Base equal weights

    Returns:
        Adjusted factor weights

    Example:
        >>> factor_preds = {
        ...     'momentum': mom_predictions,
        ...     'value': val_predictions
        ... }
        >>> weights = dynamic_factor_weights(factor_preds)
        >>> # {'momentum': 0.6, 'value': 0.4}  # if momentum more confident
    """
    if base_weights is None:
        base_weights = {f: 1.0/len(factor_predictions) for f in factor_predictions}

    # Get average confidence per factor
    avg_confidence = {}
    for factor_name, preds in factor_predictions.items():
        if 'confidence_normalized' in preds.columns:
            avg_confidence[factor_name] = preds['confidence_normalized'].mean()
        else:
            avg_confidence[factor_name] = 0.5

    # Convert confidence to weights
    total_conf = sum(avg_confidence.values())
    if total_conf > 0:
        conf_weights = {f: c / total_conf for f, c in avg_confidence.items()}
    else:
        conf_weights = base_weights

    # Blend base and confidence-based weights
    blend_factor = 0.5  # How much to let confidence affect weights
    adjusted_weights = {}
    for factor_name in factor_predictions:
        adjusted_weights[factor_name] = (
            (1 - blend_factor) * base_weights[factor_name] +
            blend_factor * conf_weights[factor_name]
        )

    # Normalize
    total = sum(adjusted_weights.values())
    adjusted_weights = {f: w/total for f, w in adjusted_weights.items()}

    return adjusted_weights
