# Quantitative Finance Methods Overview

## Portfolio Optimization vs Risk Management

| Aspect | Portfolio Optimization | Risk Management |
|--------|----------------------|-----------------|
| **Question** | "How should I allocate?" | "How bad could it get?" |
| **Output** | Weights (w₁, w₂, ...) | Risk measures (VaR, vol) |
| **Goal** | Maximize return/risk | Measure and control risk |
| **When Used** | Portfolio construction | Ongoing monitoring |

---

## Implementation Status

### Portfolio Optimization Methods

| Method | Status | Location | Description |
|--------|--------|----------|-------------|
| Mean-Variance (Markowitz) | ✅ | `optimizer/mean_variance.rs` | Classic risk-return tradeoff |
| Min Variance | ✅ | `optimizer/mean_variance.rs` | Minimum volatility portfolio |
| Max Sharpe | ✅ | `optimizer/mean_variance.rs` | Maximum risk-adjusted return |
| Black-Litterman | ✅ | `optimizer/black_litterman.rs` | Bayesian view blending |
| Constrained Opt | ✅ | `optimizer/constrained.rs` | Box and sector constraints |
| Min CVaR | ✅ | `optimizer/cvar_opt.rs` | Minimize expected tail loss |
| Robust Optimization | ✅ | `optimizer/robust.rs` | Worst-case optimization |
| Multi-Period | ✅ | `optimizer/multiperiod.rs` | Dynamic with transaction costs |
| Factor-Based | ✅ | `optimizer/factor_opt.rs` | PCA factor model optimization |

### Risk Measurement Methods

| Method | Status | Location | Description |
|--------|--------|----------|-------------|
| Sample Covariance | ✅ | `risk/covariance.rs` | Basic covariance estimation |
| Ledoit-Wolf Shrinkage | ✅ | `risk/shrinkage.rs` | Optimal shrinkage to reduce noise |
| Risk Decomposition | ✅ | `risk/decomposition.rs` | MCR, CCR, Risk Parity |
| VaR (Parametric) | ✅ | `risk/var.rs` | Value at Risk - Normal dist |
| VaR (Historical) | ✅ | `risk/var.rs` | Value at Risk - Historical sim |
| CVaR/ES | ✅ | `risk/var.rs` | Conditional VaR (tail average) |
| HRP | ✅ | `risk/hrp.rs` | Hierarchical Risk Parity |
| Factor Risk Models | ✅ | `risk/factor_model.rs` | Factor-based risk decomposition |
| EVT (Tail Risk) | ✅ | `risk/evt.rs` | Extreme Value Theory for tails |

### Python ML Modules

| Method | Status | Location | Description |
|--------|--------|----------|-------------|
| GARCH | ✅ | `src/ml/garch.py` | Time-varying volatility |
| Stress Testing | ✅ | `src/ml/stress_test.py` | Historical & hypothetical scenarios |

---

## Method Details

### 1. Min CVaR Optimization (`cvar_opt.rs`)

**Problem**: Variance penalizes upside and downside equally. CVaR focuses on tail losses.

**Math**:
```
CVaR_α = E[Loss | Loss > VaR_α]
       = (1/(1-α)) × ∫_{α}^{1} VaR_u du
```

**Use When**:
- Concerned about tail risk
- Fat-tailed return distributions
- Regulatory requirements (Basel III)

### 2. Robust Optimization (`robust.rs`)

**Problem**: Estimated parameters have uncertainty. What if they're wrong?

**Solution**: Optimize for worst-case within uncertainty set:
```
max   min     w'μ - λ × w'Σw
 w    μ∈U(μ̂)
```

**Use When**:
- Low confidence in return estimates
- Short estimation period
- High stakes decisions

### 3. Multi-Period Optimization (`multiperiod.rs`)

**Problem**: Single-period ignores transaction costs and future information.

**Solution**: Dynamic programming with rebalancing costs:
```
V_t(w) = max_{w'} [u(w') - c×||w'-w|| + E[V_{t+1}(w')]]
```

**Use When**:
- Significant transaction costs
- Long investment horizon
- Predictable return dynamics

### 4. Factor-Based Optimization (`factor_opt.rs`)

**Problem**: N stocks = N(N+1)/2 covariance parameters. Too many to estimate!

**Solution**: Assume K << N factors drive returns:
```
Σ = BΣ_fB' + D

where:
  B = N×K factor loadings
  Σ_f = K×K factor covariance (small!)
  D = diagonal idiosyncratic variance
```

**Use When**:
- Large portfolios (N > 50)
- Want to control factor exposures
- Need stable covariance estimates

### 5. GARCH Models (`garch.py`)

**Problem**: Volatility isn't constant - it clusters!

**Model**:
```
σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}

where:
  α = shock persistence
  β = volatility persistence
  α + β < 1 for stationarity
```

**Use When**:
- Forecasting volatility
- Dynamic VaR calculation
- Options pricing

### 6. Factor Risk Models (`factor_model.rs`)

**Purpose**: Decompose portfolio risk into systematic and idiosyncratic.

**Decomposition**:
```
σ²_p = Factor Risk + Specific Risk
     = w'BΣ_fB'w + w'Dw
```

**Output**:
- Factor contributions to risk
- Specific (diversifiable) risk
- Risk attribution by factor

### 7. Stress Testing (`stress_test.py`)

**Types**:
1. **Historical**: "What if 2008 happens again?"
2. **Hypothetical**: "What if rates rise 300bp?"
3. **Reverse**: "What scenario causes 20% loss?"

**Built-in Scenarios**:
- 2008 Financial Crisis
- 2020 COVID Crash
- 2022 Rate Hike
- 1997 Asian Crisis
- 2011 Eurozone Crisis

### 8. EVT - Extreme Value Theory (`evt.rs`)

**Problem**: Normal distributions underestimate tail risk dramatically.

**Solution**: Generalized Pareto Distribution (GPD) for tail:
```
For exceedances over threshold u:

F_u(y) ≈ GPD(ξ, β)

VaR_α = u + (β/ξ) × [(n/N_u × (1-α))^(-ξ) - 1]
```

**Key Parameter**:
- ξ > 0: Heavy tail (financial returns typically ξ ≈ 0.1-0.3)
- ξ = 0: Exponential tail
- ξ < 0: Bounded tail

**Use When**:
- Extreme quantiles (99.9%)
- Basel III capital calculations
- Fat-tailed distributions

---

## Python API Reference

```python
from quant_core import (
    # Basic Optimization
    MeanVarianceOptimizer,
    BlackLitterman,
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
    mcr,  # Marginal Contribution to Risk
    ccr,  # Component Contribution to Risk
    pct,  # Percentage Contribution to Risk
    risk_parity,  # Risk Parity weights

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
```

---

## Method Selection Guide

### By Portfolio Size

| Size | Recommended Methods |
|------|---------------------|
| Small (< 20 assets) | Mean-Variance, Sample Covariance |
| Medium (20-100) | Ledoit-Wolf, Factor Model |
| Large (> 100) | Factor-Based, HRP |

### By Risk Tolerance

| Risk Profile | Optimization | Risk Measure |
|--------------|--------------|--------------|
| Conservative | Min Variance, Min CVaR | CVaR, EVT |
| Balanced | Max Sharpe, Black-Litterman | VaR, Vol |
| Aggressive | Mean-Variance (high λ) | Vol |

### By Horizon

| Horizon | Optimization | Risk Model |
|---------|--------------|------------|
| Short (< 1 month) | Single-period | GARCH volatility |
| Medium (1-12 months) | Multi-period | Factor models |
| Long (> 1 year) | Strategic allocation | Stress testing |
