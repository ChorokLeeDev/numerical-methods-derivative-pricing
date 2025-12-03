"""
Transaction cost modeling.

Models for estimating trading costs and portfolio turnover.
"""

import numpy as np
import pandas as pd


def calculate_transaction_costs(
    old_weights: pd.Series,
    new_weights: pd.Series,
    cost_bps: float = 10.0
) -> float:
    """
    Calculate transaction costs from portfolio rebalancing.

    Parameters:
        old_weights: Current portfolio weights
        new_weights: Target portfolio weights
        cost_bps: Round-trip cost in basis points (default: 10bps = 0.1%)

    Returns:
        Transaction cost as decimal (e.g., 0.001 = 0.1%)

    Example:
        >>> old = pd.Series({'A': 0.5, 'B': 0.5})
        >>> new = pd.Series({'A': 0.0, 'C': 1.0})
        >>> calculate_transaction_costs(old, new, cost_bps=10)
        0.001  # 100% turnover * 10bps = 10bps = 0.1%
    """
    turnover = calculate_turnover(old_weights, new_weights)
    cost_rate = cost_bps / 10000  # Convert bps to decimal

    return turnover * cost_rate


def calculate_turnover(
    old_weights: pd.Series,
    new_weights: pd.Series
) -> float:
    """
    Calculate portfolio turnover rate.

    Turnover = Sum of |new_weight - old_weight| / 2

    A turnover of 1.0 means 100% of the portfolio was traded.

    Parameters:
        old_weights: Current portfolio weights
        new_weights: Target portfolio weights

    Returns:
        Turnover as decimal (1.0 = 100% turnover)
    """
    # Get union of all tickers
    all_tickers = old_weights.index.union(new_weights.index)

    # Reindex to common universe, filling missing with 0
    old = old_weights.reindex(all_tickers).fillna(0)
    new = new_weights.reindex(all_tickers).fillna(0)

    # Calculate absolute changes
    changes = np.abs(new - old)

    # Divide by 2 because each trade is counted twice (sell + buy)
    turnover = changes.sum() / 2

    return turnover


def estimate_market_impact(
    trade_size: float,
    adv: float,
    impact_coefficient: float = 0.1
) -> float:
    """
    Estimate market impact of a trade.

    Uses square-root market impact model:
    Impact = coefficient * sqrt(trade_size / adv)

    Parameters:
        trade_size: Trade value in same units as ADV
        adv: Average daily volume in value terms
        impact_coefficient: Impact coefficient (default: 0.1 = 10%)

    Returns:
        Estimated price impact as decimal
    """
    if adv <= 0:
        return 0.0

    participation_rate = trade_size / adv
    impact = impact_coefficient * np.sqrt(participation_rate)

    return min(impact, 0.1)  # Cap at 10%


def calculate_implementation_shortfall(
    target_prices: pd.Series,
    execution_prices: pd.Series,
    weights: pd.Series
) -> float:
    """
    Calculate implementation shortfall.

    IS = Weighted average of (execution_price - target_price) / target_price

    Parameters:
        target_prices: Decision prices (when signal was generated)
        execution_prices: Actual execution prices
        weights: Portfolio weights

    Returns:
        Implementation shortfall as decimal
    """
    common = target_prices.index.intersection(execution_prices.index)
    common = common.intersection(weights.index)

    if len(common) == 0:
        return 0.0

    target = target_prices.loc[common]
    exec_p = execution_prices.loc[common]
    w = weights.loc[common]

    # Normalize weights
    w = w / w.sum()

    # Calculate per-stock shortfall
    shortfall = (exec_p - target) / target

    # Weighted average
    return (shortfall * w).sum()


def get_transaction_cost_schedule() -> dict:
    """
    Get typical transaction cost schedule for Korean stocks.

    Returns:
        Dict with cost components in basis points
    """
    return {
        'brokerage_commission': 1.5,  # 0.015%
        'exchange_fee': 0.3,          # KRX fee
        'securities_transaction_tax': 23.0,  # 0.23% (for selling only)
        'bid_ask_spread': 5.0,        # Estimated average spread
        'market_impact': 5.0,         # Estimated for liquid stocks
        'total_round_trip': 35.0      # ~0.35% for round trip
    }
