"""
Performance metrics for portfolio evaluation.

Sharpe ratio, MDD, Information ratio, and more.
"""

import numpy as np
import pandas as pd


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
    annualization_factor: int = 12
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Sharpe = (Mean Return - Risk Free Rate) / Std(Return) * sqrt(annualization)

    Parameters:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate (default: 3%)
        annualization_factor: Periods per year (12 for monthly)

    Returns:
        Annualized Sharpe ratio

    Example:
        >>> sharpe = calculate_sharpe_ratio(monthly_returns, risk_free_rate=0.03)
        >>> print(f"Sharpe: {sharpe:.2f}")
        Sharpe: 1.25
    """
    if len(returns) == 0:
        return 0.0

    # Annualize mean return
    mean_return = returns.mean() * annualization_factor

    # Annualize volatility
    volatility = returns.std() * np.sqrt(annualization_factor)

    if volatility == 0:
        return 0.0

    sharpe = (mean_return - risk_free_rate) / volatility

    return sharpe


def calculate_max_drawdown(nav: pd.Series) -> tuple[float, str, str]:
    """
    Calculate maximum drawdown.

    Parameters:
        nav: Series of net asset values or cumulative returns

    Returns:
        Tuple of (max_drawdown, peak_date, trough_date)
        max_drawdown is negative (e.g., -0.25 = -25%)

    Example:
        >>> mdd, peak, trough = calculate_max_drawdown(nav_series)
        >>> print(f"MDD: {mdd:.2%} from {peak} to {trough}")
    """
    if len(nav) == 0:
        return 0.0, '', ''

    # Calculate running maximum
    running_max = nav.expanding().max()

    # Calculate drawdown
    drawdown = (nav - running_max) / running_max

    # Find maximum drawdown
    max_dd = drawdown.min()

    if max_dd == 0:
        return 0.0, '', ''

    # Find trough (lowest point)
    trough_idx = drawdown.idxmin()

    # Find peak (highest point before trough)
    peak_idx = nav.loc[:trough_idx].idxmax()

    peak_date = peak_idx.strftime('%Y-%m-%d') if hasattr(peak_idx, 'strftime') else str(peak_idx)
    trough_date = trough_idx.strftime('%Y-%m-%d') if hasattr(trough_idx, 'strftime') else str(trough_idx)

    return max_dd, peak_date, trough_date


def calculate_information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    annualization_factor: int = 12
) -> float:
    """
    Calculate Information Ratio vs benchmark.

    IR = Mean(Active Return) / Std(Active Return) * sqrt(annualization)

    Parameters:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns
        annualization_factor: Periods per year

    Returns:
        Annualized Information Ratio
    """
    # Align series
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)

    if len(common_idx) == 0:
        return 0.0

    port = portfolio_returns.loc[common_idx]
    bench = benchmark_returns.loc[common_idx]

    # Calculate active return
    active_return = port - bench

    if active_return.std() == 0:
        return 0.0

    # Annualized IR
    ir = (active_return.mean() / active_return.std()) * np.sqrt(annualization_factor)

    return ir


def calculate_cagr(
    start_value: float,
    end_value: float,
    years: float
) -> float:
    """
    Calculate Compound Annual Growth Rate.

    CAGR = (End/Start)^(1/years) - 1

    Parameters:
        start_value: Initial value
        end_value: Final value
        years: Number of years

    Returns:
        CAGR as decimal
    """
    if start_value <= 0 or years <= 0:
        return 0.0

    return (end_value / start_value) ** (1 / years) - 1


def calculate_sortino_ratio(
    returns: pd.Series,
    target_return: float = 0.0,
    annualization_factor: int = 12
) -> float:
    """
    Calculate Sortino Ratio (downside risk-adjusted return).

    Uses only downside deviation (returns below target).

    Parameters:
        returns: Series of returns
        target_return: Minimum acceptable return (default: 0)
        annualization_factor: Periods per year

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    # Calculate downside deviation
    downside_returns = returns[returns < target_return]

    if len(downside_returns) == 0:
        return float('inf')

    downside_std = downside_returns.std() * np.sqrt(annualization_factor)

    if downside_std == 0:
        return float('inf')

    mean_return = returns.mean() * annualization_factor

    return (mean_return - target_return) / downside_std


def calculate_calmar_ratio(
    returns: pd.Series,
    annualization_factor: int = 12
) -> float:
    """
    Calculate Calmar Ratio (return / max drawdown).

    Parameters:
        returns: Series of returns
        annualization_factor: Periods per year

    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0

    # Calculate NAV
    nav = (1 + returns).cumprod()

    # Get MDD
    mdd, _, _ = calculate_max_drawdown(nav)

    if mdd == 0:
        return float('inf')

    # Annualized return
    total_return = nav.iloc[-1] - 1
    years = len(returns) / annualization_factor
    cagr = calculate_cagr(1, 1 + total_return, years)

    return cagr / abs(mdd)


def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate win rate (% of positive returns).

    Parameters:
        returns: Series of returns

    Returns:
        Win rate as decimal (0.6 = 60%)
    """
    if len(returns) == 0:
        return 0.0

    return (returns > 0).sum() / len(returns)


def generate_performance_summary(
    returns: pd.Series,
    benchmark_returns: pd.Series = None,
    turnover: pd.Series = None,
    risk_free_rate: float = 0.03
) -> pd.DataFrame:
    """
    Generate comprehensive performance summary.

    Parameters:
        returns: Portfolio returns
        benchmark_returns: Optional benchmark returns
        turnover: Optional turnover series
        risk_free_rate: Annual risk-free rate

    Returns:
        DataFrame with performance metrics
    """
    if len(returns) == 0:
        return pd.DataFrame()

    # Basic metrics
    nav = (1 + returns).cumprod()
    total_return = nav.iloc[-1] - 1
    n_periods = len(returns)
    years = n_periods / 12

    metrics = {}

    # Returns
    metrics['Total Return'] = f"{total_return:.2%}"
    metrics['CAGR'] = f"{calculate_cagr(1, 1 + total_return, years):.2%}"
    metrics['Volatility (Ann.)'] = f"{returns.std() * np.sqrt(12):.2%}"

    # Risk-adjusted
    metrics['Sharpe Ratio'] = f"{calculate_sharpe_ratio(returns, risk_free_rate):.2f}"
    metrics['Sortino Ratio'] = f"{calculate_sortino_ratio(returns):.2f}"
    metrics['Calmar Ratio'] = f"{calculate_calmar_ratio(returns):.2f}"

    # Drawdown
    mdd, peak, trough = calculate_max_drawdown(nav)
    metrics['Max Drawdown'] = f"{mdd:.2%}"
    metrics['MDD Period'] = f"{peak} to {trough}"

    # Win rate
    metrics['Win Rate'] = f"{calculate_win_rate(returns):.2%}"
    metrics['Best Month'] = f"{returns.max():.2%}"
    metrics['Worst Month'] = f"{returns.min():.2%}"

    # Benchmark comparison
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        metrics['Information Ratio'] = f"{calculate_information_ratio(returns, benchmark_returns):.2f}"

    # Turnover
    if turnover is not None and len(turnover) > 0:
        metrics['Avg Monthly Turnover'] = f"{turnover.mean():.2%}"
        metrics['Annual Turnover'] = f"{turnover.mean() * 12:.2%}"

    metrics['N Periods'] = n_periods

    return pd.DataFrame([metrics]).T.rename(columns={0: 'Value'})
