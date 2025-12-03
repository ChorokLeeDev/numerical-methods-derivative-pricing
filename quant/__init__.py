"""
quant - Quantitative Finance Library

A unified Python package combining:
- High-performance Rust optimizers (quant_core)
- Python ML models (GARCH, stress testing)
- Factor investing tools
- Portfolio construction and backtesting

Usage:
    from quant import (
        # Optimization (Rust)
        MeanVarianceOptimizer,
        BlackLitterman,
        ledoit_wolf,
        risk_parity,

        # ML (Python)
        fit_garch,
        run_stress_test,
    )
"""

__version__ = "0.1.0"

# =============================================================================
# Rust Core (quant_core) - compiled via maturin
# =============================================================================
try:
    from quant_core import (
        # Mean-Variance Optimizer
        MeanVarianceOptimizer,

        # Black-Litterman
        BlackLitterman,

        # Constrained Optimization
        min_variance_constrained,
        max_sharpe_constrained,

        # Advanced Optimization
        minimize_cvar,
        robust_optimize,
        multiperiod_optimize,
        estimate_factor_model,
        factor_min_variance,

        # Covariance Estimation
        sample_covariance,
        ledoit_wolf,
        shrink_to_identity,

        # Risk Decomposition
        mcr,
        ccr,
        pct,
        risk_parity,

        # VaR/CVaR
        parametric_var,
        historical_var,
        parametric_cvar,
        historical_cvar,

        # HRP
        hrp_weights,

        # Factor Risk Models
        estimate_factor_risk_model,
        factor_risk_decomposition,

        # EVT (Tail Risk)
        fit_gpd,
        evt_var,
        evt_es,
        hill_tail_index,
        tail_risk_analysis,
    )
    _RUST_AVAILABLE = True
except ImportError as e:
    _RUST_AVAILABLE = False
    _RUST_IMPORT_ERROR = str(e)

# =============================================================================
# Python Modules
# =============================================================================

# ML Models
from quant.ml.garch import (
    fit_garch,
    forecast_volatility,
    dynamic_covariance,
    garch_var,
    GARCHResult,
)

from quant.ml.stress_test import (
    run_stress_test,
    historical_stress_test,
    reverse_stress_test,
    sensitivity_analysis,
    generate_report as generate_stress_report,
    create_custom_scenario,
    StressScenario,
    StressTestResult,
    HISTORICAL_SCENARIOS,
)

from quant.ml.lgbm_model import (
    QuantileGBM,
    train_quantile_models,
)

from quant.ml.features import (
    prepare_features,
)

from quant.ml.position_sizing import (
    confidence_weighted_portfolio,
    kelly_criterion_adjustment,
)

# Data
from quant.data.price import (
    fetch_price_data,
    calculate_returns,
)

from quant.data.kospi200 import (
    get_kospi200_tickers,
)

# Factors
from quant.factors.momentum import (
    calculate_momentum_12_1,
    calculate_momentum_simple,
)

from quant.factors.value import (
    get_value_factor,
)

from quant.factors.quality import (
    get_quality_factor,
)

from quant.factors.composite import (
    combine_factors,
)

# Portfolio
from quant.portfolio.construction import (
    construct_long_short_portfolio,
    construct_long_only_portfolio,
)

from quant.portfolio.backtest import (
    run_backtest,
)

# Analytics
from quant.analytics.metrics import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
)


def check_rust_core():
    """Check if Rust core is available."""
    if _RUST_AVAILABLE:
        print("quant_core (Rust): Available")
        return True
    else:
        print(f"quant_core (Rust): Not available - {_RUST_IMPORT_ERROR}")
        print("Run 'cd quant-core && maturin develop --release' to build")
        return False


# Expose submodules
from quant import ml
from quant import data
from quant import factors
from quant import portfolio
from quant import analytics

__all__ = [
    # Version
    "__version__",
    "check_rust_core",

    # Rust Core - Optimization
    "MeanVarianceOptimizer",
    "BlackLitterman",
    "min_variance_constrained",
    "max_sharpe_constrained",
    "minimize_cvar",
    "robust_optimize",
    "multiperiod_optimize",
    "estimate_factor_model",
    "factor_min_variance",

    # Rust Core - Covariance
    "sample_covariance",
    "ledoit_wolf",
    "shrink_to_identity",

    # Rust Core - Risk
    "mcr",
    "ccr",
    "pct",
    "risk_parity",
    "parametric_var",
    "historical_var",
    "parametric_cvar",
    "historical_cvar",
    "hrp_weights",
    "estimate_factor_risk_model",
    "factor_risk_decomposition",
    "fit_gpd",
    "evt_var",
    "evt_es",
    "hill_tail_index",
    "tail_risk_analysis",

    # Python - ML
    "fit_garch",
    "forecast_volatility",
    "dynamic_covariance",
    "garch_var",
    "GARCHResult",
    "run_stress_test",
    "historical_stress_test",
    "reverse_stress_test",
    "sensitivity_analysis",
    "generate_stress_report",
    "create_custom_scenario",
    "StressScenario",
    "StressTestResult",
    "HISTORICAL_SCENARIOS",
    "QuantileGBM",
    "train_quantile_models",
    "prepare_features",
    "confidence_weighted_portfolio",
    "kelly_criterion_adjustment",

    # Python - Data
    "fetch_price_data",
    "calculate_returns",
    "get_kospi200_tickers",

    # Python - Factors
    "calculate_momentum_12_1",
    "calculate_momentum_simple",
    "get_value_factor",
    "get_quality_factor",
    "combine_factors",

    # Python - Portfolio
    "construct_long_short_portfolio",
    "construct_long_only_portfolio",
    "run_backtest",

    # Python - Analytics
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",

    # Submodules
    "ml",
    "data",
    "factors",
    "portfolio",
    "analytics",
]
