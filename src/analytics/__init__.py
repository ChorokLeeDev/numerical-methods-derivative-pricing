"""
Performance analytics and visualization.

Modules:
    - metrics: Performance metrics (Sharpe, MDD, IR, etc.)
    - visualization: Charts and plots
"""

from .metrics import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_information_ratio,
    calculate_cagr,
    generate_performance_summary,
)
from .visualization import (
    plot_cumulative_returns,
    plot_drawdown,
    plot_quantile_returns,
)

__all__ = [
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'calculate_information_ratio',
    'calculate_cagr',
    'generate_performance_summary',
    'plot_cumulative_returns',
    'plot_drawdown',
    'plot_quantile_returns',
]
