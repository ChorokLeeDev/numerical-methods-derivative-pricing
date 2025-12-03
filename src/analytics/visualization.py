"""
Visualization utilities for factor backtesting.

Charts for cumulative returns, drawdowns, and factor analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure


def plot_cumulative_returns(
    returns_dict: dict[str, pd.Series],
    title: str = 'Cumulative Returns',
    figsize: tuple = (12, 6),
    log_scale: bool = False
) -> matplotlib.figure.Figure:
    """
    Plot cumulative returns for multiple strategies.

    Parameters:
        returns_dict: Dict mapping strategy name to returns series
        title: Chart title
        figsize: Figure size
        log_scale: Use log scale for y-axis

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, returns in returns_dict.items():
        cumulative = (1 + returns).cumprod()
        ax.plot(cumulative.index, cumulative.values, label=name, linewidth=1.5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale('log')

    # Add horizontal line at 1.0
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig


def plot_drawdown(
    nav: pd.Series,
    title: str = 'Drawdown',
    figsize: tuple = (12, 4)
) -> matplotlib.figure.Figure:
    """
    Plot drawdown over time.

    Parameters:
        nav: Series of net asset values or cumulative returns
        title: Chart title
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate drawdown
    running_max = nav.expanding().max()
    drawdown = (nav - running_max) / running_max * 100  # As percentage

    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color='red')
    ax.plot(drawdown.index, drawdown.values, color='darkred', linewidth=0.5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)

    # Mark maximum drawdown
    min_dd = drawdown.min()
    min_dd_date = drawdown.idxmin()
    ax.annotate(
        f'Max DD: {min_dd:.1f}%',
        xy=(min_dd_date, min_dd),
        xytext=(10, -20),
        textcoords='offset points',
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='black')
    )

    plt.tight_layout()
    return fig


def plot_quantile_returns(
    quantile_returns: pd.DataFrame,
    title: str = 'Factor Quintile Returns',
    figsize: tuple = (10, 6)
) -> matplotlib.figure.Figure:
    """
    Plot bar chart of cumulative returns by factor quantile.

    Parameters:
        quantile_returns: DataFrame with Q1, Q2, ..., Qn columns
        title: Chart title
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate cumulative returns for each quantile
    cumulative = {}
    for col in quantile_returns.columns:
        cumulative[col] = ((1 + quantile_returns[col]).prod() - 1) * 100

    # Create bar chart
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(cumulative)))
    bars = ax.bar(cumulative.keys(), cumulative.values(), color=colors)

    # Add value labels on bars
    for bar, val in zip(bars, cumulative.values()):
        height = bar.get_height()
        ax.annotate(
            f'{val:.1f}%',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords='offset points',
            ha='center',
            fontsize=10
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Quantile (Q1=Low, Q5=High)')
    ax.set_ylabel('Cumulative Return (%)')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_monthly_returns_heatmap(
    returns: pd.Series,
    title: str = 'Monthly Returns Heatmap',
    figsize: tuple = (12, 8)
) -> matplotlib.figure.Figure:
    """
    Plot monthly returns as a calendar heatmap.

    Parameters:
        returns: Monthly returns series
        title: Chart title
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    # Reshape returns to year x month matrix
    returns_df = returns.to_frame('return')
    returns_df['year'] = returns_df.index.year
    returns_df['month'] = returns_df.index.month

    pivot = returns_df.pivot(index='year', columns='month', values='return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(pivot.values * 100, cmap='RdYlGn', aspect='auto',
                   vmin=-10, vmax=10)

    # Set labels
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(12):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val * 100:.1f}',
                               ha='center', va='center', fontsize=8)

    ax.set_title(title, fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Return (%)')

    plt.tight_layout()
    return fig


def plot_rolling_metrics(
    returns: pd.Series,
    window: int = 12,
    title: str = 'Rolling Performance Metrics',
    figsize: tuple = (12, 8)
) -> matplotlib.figure.Figure:
    """
    Plot rolling Sharpe ratio and volatility.

    Parameters:
        returns: Monthly returns series
        window: Rolling window in months
        title: Chart title
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Rolling Sharpe ratio
    rolling_mean = returns.rolling(window).mean() * 12
    rolling_std = returns.rolling(window).std() * np.sqrt(12)
    rolling_sharpe = (rolling_mean - 0.03) / rolling_std

    axes[0].plot(rolling_sharpe.index, rolling_sharpe.values, color='blue')
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].axhline(y=1, color='green', linestyle='--', alpha=0.5)
    axes[0].set_ylabel(f'{window}M Rolling Sharpe')
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Rolling volatility
    axes[1].plot(rolling_std.index, rolling_std.values * 100, color='red')
    axes[1].set_ylabel(f'{window}M Rolling Volatility (%)')
    axes[1].set_xlabel('Date')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_factor_exposure_over_time(
    factor_scores: dict[str, pd.Series],
    title: str = 'Factor Exposure Over Time',
    figsize: tuple = (12, 6)
) -> matplotlib.figure.Figure:
    """
    Plot average factor scores over time.

    Parameters:
        factor_scores: Dict mapping date to factor scores Series
        title: Chart title
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    dates = sorted(factor_scores.keys())
    avg_scores = [factor_scores[d].mean() for d in dates]

    ax.plot(pd.to_datetime(dates), avg_scores, marker='o', markersize=3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Factor Score')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_turnover(
    turnover: pd.Series,
    title: str = 'Portfolio Turnover',
    figsize: tuple = (12, 4)
) -> matplotlib.figure.Figure:
    """
    Plot turnover over time with rolling average.

    Parameters:
        turnover: Monthly turnover series
        title: Chart title
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Bar chart of monthly turnover
    ax.bar(turnover.index, turnover.values * 100, alpha=0.6, label='Monthly')

    # Rolling average
    rolling_avg = turnover.rolling(6).mean() * 100
    ax.plot(rolling_avg.index, rolling_avg.values, color='red',
            linewidth=2, label='6M Rolling Avg')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Turnover (%)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig
