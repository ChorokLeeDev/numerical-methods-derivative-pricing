"""
Main backtesting engine.

Runs factor-based portfolio backtests with transaction costs.
"""

from typing import Callable, Literal

import numpy as np
import pandas as pd

from .construction import (
    construct_long_short_portfolio,
    calculate_portfolio_return,
    get_quantile_portfolios,
)
from .transaction import calculate_transaction_costs, calculate_turnover
from .rebalance import (
    generate_rebalance_dates,
    get_formation_period,
    get_latest_available_fiscal_year,
)


def run_backtest(
    prices: pd.DataFrame,
    factor_func: Callable[[pd.DataFrame, str], pd.Series],
    start_date: str,
    end_date: str,
    rebalance_freq: Literal['monthly', 'quarterly'] = 'monthly',
    n_quantiles: int = 5,
    transaction_cost_bps: float = 10.0,
    initial_capital: float = 100_000_000
) -> dict:
    """
    Run full factor backtest.

    Parameters:
        prices: DataFrame with daily prices (DatetimeIndex, ticker columns)
        factor_func: Function that takes (prices, date) and returns factor scores
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD)
        rebalance_freq: Rebalancing frequency
        n_quantiles: Number of quantiles for portfolio construction
        transaction_cost_bps: Transaction costs in basis points
        initial_capital: Starting capital in KRW

    Returns:
        Dict containing:
        - 'returns': Series of portfolio returns
        - 'nav': Series of net asset values
        - 'long_portfolios': Dict of long positions by date
        - 'short_portfolios': Dict of short positions by date
        - 'turnover': Series of turnover rates
        - 'transaction_costs': Series of transaction costs

    Example:
        >>> def momentum_factor(prices, date):
        ...     returns = prices.pct_change().resample('ME').last()
        ...     return calculate_momentum_12_1(returns, date)
        >>> results = run_backtest(prices, momentum_factor, '2020-01-01', '2024-12-01')
    """
    # Generate rebalance dates
    rebal_dates = generate_rebalance_dates(start_date, end_date, rebalance_freq)

    if len(rebal_dates) < 2:
        raise ValueError("Need at least 2 rebalance dates for backtesting")

    # Initialize result containers
    results = {
        'returns': [],
        'dates': [],
        'long_portfolios': {},
        'short_portfolios': {},
        'turnover': [],
        'transaction_costs': [],
        'factor_scores': {}
    }

    # Track previous weights for turnover calculation
    prev_long_weights = pd.Series(dtype=float)
    prev_short_weights = pd.Series(dtype=float)

    # Main backtest loop
    for i in range(len(rebal_dates) - 1):
        rebal_date = rebal_dates[i]
        next_rebal_date = rebal_dates[i + 1]

        # Get prices up to rebalance date (no look-ahead)
        rebal_dt = pd.to_datetime(rebal_date)
        available_prices = prices.loc[:rebal_dt]

        if len(available_prices) < 252:  # Need at least 1 year of data
            continue

        # Calculate factor scores
        try:
            factor_scores = factor_func(available_prices, rebal_date)
        except Exception as e:
            print(f"Warning: Factor calculation failed for {rebal_date}: {e}")
            continue

        if len(factor_scores) == 0:
            continue

        # Construct portfolio
        long_weights, short_weights = construct_long_short_portfolio(
            factor_scores, n_quantiles=n_quantiles
        )

        if len(long_weights) == 0 or len(short_weights) == 0:
            continue

        # Calculate turnover
        long_turnover = calculate_turnover(prev_long_weights, long_weights)
        short_turnover = calculate_turnover(prev_short_weights, short_weights)
        total_turnover = long_turnover + short_turnover

        # Transaction costs
        tx_cost = total_turnover * (transaction_cost_bps / 10000)

        # Get holding period returns
        holding_start = rebal_dt + pd.DateOffset(days=1)
        holding_end = pd.to_datetime(next_rebal_date)

        # Get prices for holding period
        holding_prices = prices.loc[holding_start:holding_end]

        if len(holding_prices) < 2:
            continue

        # Calculate period returns
        start_prices = holding_prices.iloc[0]
        end_prices = holding_prices.iloc[-1]
        period_returns = (end_prices / start_prices) - 1

        # Calculate portfolio return
        portfolio_return = calculate_portfolio_return(
            long_weights, short_weights, period_returns
        )

        # Subtract transaction costs
        net_return = portfolio_return - tx_cost

        # Store results
        results['returns'].append(net_return)
        results['dates'].append(next_rebal_date)
        results['long_portfolios'][rebal_date] = long_weights.to_dict()
        results['short_portfolios'][rebal_date] = short_weights.to_dict()
        results['turnover'].append(total_turnover)
        results['transaction_costs'].append(tx_cost)
        results['factor_scores'][rebal_date] = factor_scores

        # Update previous weights
        prev_long_weights = long_weights
        prev_short_weights = short_weights

    # Convert to Series
    if len(results['returns']) == 0:
        return results

    results['returns'] = pd.Series(
        results['returns'],
        index=pd.to_datetime(results['dates'])
    )
    results['turnover'] = pd.Series(
        results['turnover'],
        index=pd.to_datetime(results['dates'])
    )
    results['transaction_costs'] = pd.Series(
        results['transaction_costs'],
        index=pd.to_datetime(results['dates'])
    )

    # Calculate NAV
    nav = (1 + results['returns']).cumprod() * initial_capital
    results['nav'] = nav

    return results


def run_quantile_backtest(
    prices: pd.DataFrame,
    factor_func: Callable[[pd.DataFrame, str], pd.Series],
    start_date: str,
    end_date: str,
    n_quantiles: int = 5,
    rebalance_freq: str = 'monthly'
) -> pd.DataFrame:
    """
    Run backtest for all quantile portfolios.

    Useful for analyzing factor monotonicity (returns should increase with factor).

    Parameters:
        prices: Daily prices DataFrame
        factor_func: Factor calculation function
        start_date: Start date
        end_date: End date
        n_quantiles: Number of quantiles
        rebalance_freq: Rebalancing frequency

    Returns:
        DataFrame with columns Q1, Q2, ..., Qn containing monthly returns
    """
    rebal_dates = generate_rebalance_dates(start_date, end_date, rebalance_freq)

    quantile_returns = {f'Q{q}': [] for q in range(1, n_quantiles + 1)}
    dates = []

    for i in range(len(rebal_dates) - 1):
        rebal_date = rebal_dates[i]
        next_rebal_date = rebal_dates[i + 1]

        rebal_dt = pd.to_datetime(rebal_date)
        available_prices = prices.loc[:rebal_dt]

        if len(available_prices) < 252:
            continue

        # Calculate factor scores
        try:
            factor_scores = factor_func(available_prices, rebal_date)
        except Exception:
            continue

        if len(factor_scores) == 0:
            continue

        # Get quantile portfolios
        portfolios = get_quantile_portfolios(factor_scores, n_quantiles)

        if len(portfolios) != n_quantiles:
            continue

        # Get holding period returns
        holding_start = rebal_dt + pd.DateOffset(days=1)
        holding_end = pd.to_datetime(next_rebal_date)
        holding_prices = prices.loc[holding_start:holding_end]

        if len(holding_prices) < 2:
            continue

        start_prices = holding_prices.iloc[0]
        end_prices = holding_prices.iloc[-1]
        period_returns = (end_prices / start_prices) - 1

        # Calculate return for each quantile
        for q in range(1, n_quantiles + 1):
            tickers = portfolios[q]
            if len(tickers) == 0:
                quantile_returns[f'Q{q}'].append(np.nan)
                continue

            weights = pd.Series(1.0 / len(tickers), index=tickers)
            q_returns = period_returns.reindex(tickers).fillna(0)
            q_return = (weights * q_returns).sum()
            quantile_returns[f'Q{q}'].append(q_return)

        dates.append(next_rebal_date)

    return pd.DataFrame(quantile_returns, index=pd.to_datetime(dates))


def analyze_backtest_results(results: dict) -> pd.DataFrame:
    """
    Generate summary statistics from backtest results.

    Parameters:
        results: Dict from run_backtest

    Returns:
        DataFrame with performance metrics
    """
    returns = results['returns']

    if len(returns) == 0:
        return pd.DataFrame()

    # Calculate metrics
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    n_years = n_periods / 12  # Assuming monthly

    cagr = (1 + total_return) ** (1 / n_years) - 1
    volatility = returns.std() * np.sqrt(12)
    sharpe = (returns.mean() * 12 - 0.03) / volatility if volatility > 0 else 0

    # Drawdown
    nav = results['nav']
    running_max = nav.expanding().max()
    drawdown = (nav - running_max) / running_max
    max_dd = drawdown.min()

    # Turnover
    avg_turnover = results['turnover'].mean() if len(results['turnover']) > 0 else 0

    metrics = {
        'Total Return': f"{total_return:.2%}",
        'CAGR': f"{cagr:.2%}",
        'Volatility (Ann.)': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Max Drawdown': f"{max_dd:.2%}",
        'Avg Monthly Turnover': f"{avg_turnover:.2%}",
        'N Periods': n_periods
    }

    return pd.DataFrame([metrics]).T.rename(columns={0: 'Value'})
