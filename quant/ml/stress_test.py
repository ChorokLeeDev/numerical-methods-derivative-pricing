"""
Portfolio Stress Testing

# What is Stress Testing?

Stress testing simulates portfolio performance under extreme scenarios:
- Historical: "What if 2008 happens again?"
- Hypothetical: "What if rates rise 300bp?"
- Reverse: "What scenario causes 20% loss?"

# Regulatory Requirement

Banks are REQUIRED to conduct stress tests:
- Dodd-Frank Act Stress Tests (DFAST)
- Comprehensive Capital Analysis and Review (CCAR)
- Basel III requirements

# Industry Usage

- Risk limit setting
- Capital planning
- Scenario analysis for investment decisions
- Client reporting
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime


@dataclass
class StressScenario:
    """Definition of a stress scenario."""
    name: str
    description: str
    # Factor shocks: {factor_name: shock_magnitude}
    factor_shocks: Dict[str, float] = field(default_factory=dict)
    # Historical period reference (optional)
    historical_start: Optional[str] = None
    historical_end: Optional[str] = None


@dataclass
class StressTestResult:
    """Results from stress testing."""
    scenario_name: str
    portfolio_return: float
    portfolio_vol_change: float
    var_impact: float
    worst_assets: List[Tuple[str, float]]
    best_assets: List[Tuple[str, float]]


# Pre-defined historical scenarios
HISTORICAL_SCENARIOS = {
    '2008_financial_crisis': StressScenario(
        name='2008 Financial Crisis',
        description='Lehman Brothers collapse, global credit crisis',
        factor_shocks={
            'market': -0.40,      # S&P down 40%
            'credit_spread': 0.06,  # Credit spreads widen 600bp
            'volatility': 0.50,   # VIX spikes 50 points
            'rates': -0.02,       # Rates fall 200bp (flight to safety)
        },
        historical_start='2008-09-01',
        historical_end='2009-03-31'
    ),
    '2020_covid_crash': StressScenario(
        name='2020 COVID Crash',
        description='Pandemic-induced market crash',
        factor_shocks={
            'market': -0.34,
            'credit_spread': 0.03,
            'volatility': 0.60,
            'rates': -0.015,
        },
        historical_start='2020-02-19',
        historical_end='2020-03-23'
    ),
    '2022_rate_hike': StressScenario(
        name='2022 Rate Hike',
        description='Fed aggressive rate increases',
        factor_shocks={
            'market': -0.20,
            'rates': 0.04,        # Rates rise 400bp
            'growth_value_spread': 0.25,  # Value outperforms growth
        },
        historical_start='2022-01-01',
        historical_end='2022-10-31'
    ),
    '1997_asian_crisis': StressScenario(
        name='1997 Asian Crisis',
        description='Asian currency and market crisis',
        factor_shocks={
            'em_markets': -0.50,
            'currency_em': -0.30,
            'market': -0.15,
        },
        historical_start='1997-07-01',
        historical_end='1997-12-31'
    ),
    '2011_eurozone_crisis': StressScenario(
        name='2011 Eurozone Crisis',
        description='European sovereign debt crisis',
        factor_shocks={
            'market': -0.18,
            'eu_spreads': 0.05,
            'financials': -0.30,
        },
        historical_start='2011-07-01',
        historical_end='2011-10-31'
    ),
}


def run_stress_test(
    weights: np.ndarray,
    factor_exposures: Dict[str, np.ndarray],
    scenario: StressScenario,
    asset_names: Optional[List[str]] = None
) -> StressTestResult:
    """
    Run stress test on portfolio.

    Parameters:
        weights: Portfolio weights (n_assets,)
        factor_exposures: Dict of {factor_name: exposures array (n_assets,)}
        scenario: Stress scenario to apply
        asset_names: Optional list of asset names

    Returns:
        StressTestResult with impact analysis

    Example:
        >>> exposures = {
        ...     'market': np.array([1.2, 0.8, 1.0]),  # Beta
        ...     'rates': np.array([-0.5, 0.2, 0.0])   # Duration effect
        ... }
        >>> result = run_stress_test(weights, exposures, HISTORICAL_SCENARIOS['2008_financial_crisis'])
    """
    n_assets = len(weights)
    if asset_names is None:
        asset_names = [f'Asset_{i}' for i in range(n_assets)]

    # Calculate asset-level impacts
    asset_returns = np.zeros(n_assets)

    for factor_name, shock in scenario.factor_shocks.items():
        if factor_name in factor_exposures:
            # Impact = exposure × shock
            asset_returns += factor_exposures[factor_name] * shock

    # Portfolio return
    portfolio_return = np.sum(weights * asset_returns)

    # Find worst and best performing assets
    asset_impacts = list(zip(asset_names, asset_returns))
    sorted_impacts = sorted(asset_impacts, key=lambda x: x[1])

    worst_5 = sorted_impacts[:5]
    best_5 = sorted_impacts[-5:][::-1]

    return StressTestResult(
        scenario_name=scenario.name,
        portfolio_return=portfolio_return,
        portfolio_vol_change=0.0,  # Would need full covariance
        var_impact=abs(portfolio_return) if portfolio_return < 0 else 0,
        worst_assets=worst_5,
        best_assets=best_5
    )


def historical_stress_test(
    weights: np.ndarray,
    historical_returns: pd.DataFrame,
    scenario: StressScenario
) -> StressTestResult:
    """
    Run stress test using actual historical returns.

    Parameters:
        weights: Portfolio weights
        historical_returns: DataFrame of historical returns (dates × assets)
        scenario: Scenario with historical_start and historical_end

    Returns:
        StressTestResult based on actual historical performance
    """
    if scenario.historical_start is None or scenario.historical_end is None:
        raise ValueError("Scenario must have historical dates for historical stress test")

    # Filter to scenario period
    mask = (historical_returns.index >= scenario.historical_start) & \
           (historical_returns.index <= scenario.historical_end)
    period_returns = historical_returns.loc[mask]

    if len(period_returns) == 0:
        raise ValueError(f"No data found for period {scenario.historical_start} to {scenario.historical_end}")

    # Calculate cumulative returns
    cum_returns = (1 + period_returns).prod() - 1

    # Portfolio return
    portfolio_return = np.sum(weights * cum_returns.values)

    # Asset impacts
    asset_impacts = list(zip(historical_returns.columns, cum_returns.values))
    sorted_impacts = sorted(asset_impacts, key=lambda x: x[1])

    return StressTestResult(
        scenario_name=scenario.name,
        portfolio_return=portfolio_return,
        portfolio_vol_change=0.0,
        var_impact=abs(portfolio_return) if portfolio_return < 0 else 0,
        worst_assets=sorted_impacts[:5],
        best_assets=sorted_impacts[-5:][::-1]
    )


def reverse_stress_test(
    weights: np.ndarray,
    factor_exposures: Dict[str, np.ndarray],
    target_loss: float,
    max_factor_shock: float = 0.5
) -> Dict[str, float]:
    """
    Find factor shocks that would cause a specific loss.

    "What scenario causes a 20% loss?"

    Parameters:
        weights: Portfolio weights
        factor_exposures: Factor exposure dict
        target_loss: Target portfolio loss (positive number, e.g., 0.20 for 20%)
        max_factor_shock: Maximum allowed factor shock

    Returns:
        Dict of {factor_name: shock} that causes approximately target_loss
    """
    # Simple approach: scale shocks proportionally
    # More sophisticated: optimization problem

    # Calculate sensitivity to each factor
    factor_sensitivities = {}
    for factor_name, exposures in factor_exposures.items():
        # Portfolio sensitivity = weighted sum of exposures
        sensitivity = np.sum(weights * exposures)
        factor_sensitivities[factor_name] = sensitivity

    # Total absolute sensitivity
    total_sensitivity = sum(abs(s) for s in factor_sensitivities.values())

    if total_sensitivity < 1e-10:
        return {f: 0.0 for f in factor_exposures.keys()}

    # Distribute shocks proportionally (negative direction)
    shocks = {}
    for factor_name, sensitivity in factor_sensitivities.items():
        # Shock direction: opposite of sensitivity (to cause loss)
        direction = -1 if sensitivity > 0 else 1
        # Shock magnitude: proportional to sensitivity contribution
        magnitude = (abs(sensitivity) / total_sensitivity) * target_loss / abs(sensitivity)
        magnitude = min(magnitude, max_factor_shock)
        shocks[factor_name] = direction * magnitude

    return shocks


def sensitivity_analysis(
    weights: np.ndarray,
    factor_exposures: Dict[str, np.ndarray],
    shock_sizes: List[float] = [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]
) -> pd.DataFrame:
    """
    Analyze portfolio sensitivity to each factor.

    Parameters:
        weights: Portfolio weights
        factor_exposures: Factor exposure dict
        shock_sizes: List of shock sizes to test

    Returns:
        DataFrame with portfolio return for each factor/shock combination
    """
    results = []

    for factor_name, exposures in factor_exposures.items():
        for shock in shock_sizes:
            port_return = np.sum(weights * exposures * shock)
            results.append({
                'factor': factor_name,
                'shock': shock,
                'portfolio_return': port_return
            })

    return pd.DataFrame(results).pivot(
        index='factor',
        columns='shock',
        values='portfolio_return'
    )


def generate_report(
    weights: np.ndarray,
    factor_exposures: Dict[str, np.ndarray],
    asset_names: Optional[List[str]] = None,
    scenarios: Optional[List[StressScenario]] = None
) -> str:
    """
    Generate stress test report.

    Parameters:
        weights: Portfolio weights
        factor_exposures: Factor exposures
        asset_names: Asset names
        scenarios: List of scenarios (default: all historical)

    Returns:
        Formatted report string
    """
    if scenarios is None:
        scenarios = list(HISTORICAL_SCENARIOS.values())

    lines = [
        "=" * 60,
        "PORTFOLIO STRESS TEST REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "SCENARIO ANALYSIS",
        "-" * 60,
    ]

    for scenario in scenarios:
        result = run_stress_test(weights, factor_exposures, scenario, asset_names)

        lines.append(f"\n{scenario.name}")
        lines.append(f"  Description: {scenario.description}")
        lines.append(f"  Portfolio Return: {result.portfolio_return:+.2%}")
        lines.append(f"  VaR Impact: {result.var_impact:.2%}")

        lines.append("  Worst Performers:")
        for name, ret in result.worst_assets[:3]:
            lines.append(f"    - {name}: {ret:+.2%}")

    lines.append("\n" + "=" * 60)
    lines.append("SENSITIVITY ANALYSIS")
    lines.append("-" * 60)

    sens_df = sensitivity_analysis(weights, factor_exposures)
    lines.append(sens_df.to_string())

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


def create_custom_scenario(
    name: str,
    description: str,
    **factor_shocks
) -> StressScenario:
    """
    Create a custom stress scenario.

    Example:
        >>> scenario = create_custom_scenario(
        ...     name="Rate Shock",
        ...     description="Sudden rate increase",
        ...     market=-0.10,
        ...     rates=0.03
        ... )
    """
    return StressScenario(
        name=name,
        description=description,
        factor_shocks=factor_shocks
    )
