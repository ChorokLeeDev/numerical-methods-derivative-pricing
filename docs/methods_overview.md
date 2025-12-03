# Quantitative Methods Overview

A complete guide to portfolio optimization and risk management methods.

---

## Risk Management vs Optimization

| Aspect | Optimization | Risk Management |
|--------|-------------|-----------------|
| **Question** | "How should I allocate money?" | "How risky is my portfolio?" |
| **Output** | Weights (w₁, w₂, ..., wₙ) | Risk metrics (VaR, volatility) |
| **When** | Before investing | Before AND after investing |
| **Goal** | Find best portfolio | Understand & control risk |

**Simple Analogy**:
```
Optimization = "Which route should I take to work?"
Risk Management = "What are the dangers on each route?"

You need BOTH:
1. Risk Management tells you "Route A has traffic, Route B has construction"
2. Optimization uses that info to pick the best route
```

---

## Portfolio Optimization Methods

| Method | Status | Difficulty | Industry Use | File |
|--------|--------|------------|--------------|------|
| Mean-Variance (Markowitz) | ✅ | Basic | Universal | `optimizer/mean_variance.rs` |
| Black-Litterman | ✅ | Intermediate | Banks, Asset Managers | `optimizer/black_litterman.rs` |
| Risk Parity | ✅ | Intermediate | Bridgewater, Many Funds | `risk/decomposition.rs` |
| HRP (Hierarchical) | ✅ | Intermediate | Quant Funds | `risk/hrp.rs` |
| Constrained Optimization | ✅ | Intermediate | Everyone | `optimizer/constrained.rs` |
| Min CVaR Optimization | ✅ | Advanced | Risk-focused funds | `optimizer/cvar_opt.rs` |
| Robust Optimization | ✅ | Advanced | Academic/Research | `optimizer/robust.rs` |
| Multi-Period Optimization | ✅ | Advanced | Pension Funds | `optimizer/multiperiod.rs` |
| Factor-Based Optimization | ✅ | Advanced | Quant Funds | `optimizer/factor_opt.rs` |

---

## Risk Measurement Methods

| Method | Status | Difficulty | Industry Use | File |
|--------|--------|------------|--------------|------|
| Volatility (Std Dev) | ✅ | Basic | Universal | `risk/covariance.rs` |
| Covariance/Correlation | ✅ | Basic | Universal | `risk/covariance.rs` |
| Shrinkage Estimation | ✅ | Intermediate | Production Systems | `risk/shrinkage.rs` |
| VaR (Value at Risk) | ✅ | Intermediate | Banks (Required!) | `risk/var.rs` |
| CVaR (Expected Shortfall) | ✅ | Intermediate | Basel III Standard | `risk/var.rs` |
| Risk Decomposition (MCR/CCR) | ✅ | Intermediate | Portfolio Managers | `risk/decomposition.rs` |
| GARCH (Time-varying vol) | ✅ | Advanced | Trading Desks | `src/ml/garch.py` |
| Factor Risk Models | ✅ | Advanced | Barra, Axioma | `risk/factor_model.rs` |
| Stress Testing | ✅ | Advanced | Banks (Required!) | `src/ml/stress_test.py` |
| Tail Risk (Extreme Value) | ✅ | Advanced | Hedge Funds | `risk/evt.rs` |

---

## Method Details

### 1. Mean-Variance Optimization (Markowitz, 1952)

**The Foundation**: Nobel Prize winning work that started modern portfolio theory.

```
maximize    μ'w - (λ/2) w'Σw
subject to  Σw = 1
```

**Key Insight**: Diversification reduces risk without necessarily reducing return.

**Limitations**:
- Sensitive to input estimates (μ, Σ)
- Often produces extreme weights
- Assumes normal returns

---

### 2. Black-Litterman Model

**Problem Solved**: Mean-Variance needs expected returns, which are hard to estimate.

**Solution**: Start from market equilibrium, then adjust with your "views".

```
Prior (Market):     π = λΣw_mkt
Your Views:         Q = Pw + ε
Posterior:          μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹[(τΣ)⁻¹π + P'Ω⁻¹Q]
```

**Key Insight**: Blend market wisdom with your beliefs.

---

### 3. Risk Parity

**Problem Solved**: Traditional portfolios concentrate risk in volatile assets.

**Solution**: Allocate so each asset contributes equally to risk.

```
Goal: CCR₁ = CCR₂ = ... = CCRₙ
where CCR_i = w_i × (Σw)_i / σ_p
```

**Key Insight**: Balance risk, not dollars.

---

### 4. Hierarchical Risk Parity (HRP)

**Problem Solved**: Covariance matrix inversion amplifies estimation errors.

**Solution**: Use hierarchical clustering, no matrix inversion needed.

```
1. Cluster assets by correlation
2. Reorder to quasi-diagonal
3. Recursive bisection allocation
```

**Key Insight**: Respect correlation structure without inverting.

---

### 5. Constrained Optimization

**Problem Solved**: Real portfolios have constraints (no shorting, position limits).

**Solution**: Projected Gradient Descent onto feasible set.

```
minimize    (1/2) w'Σw - λ⁻¹μ'w
subject to  w'1 = 1
            w_min ≤ w ≤ w_max
```

---

### 6. Min CVaR Optimization (Advanced)

**Problem Solved**: Minimizing variance doesn't control tail risk.

**Solution**: Minimize expected loss in worst α% scenarios.

```
minimize    CVaR_α(w)
subject to  Σw = 1, w ≥ 0

CVaR can be reformulated as linear program:
minimize    γ + (1/αT) Σᵢ uᵢ
subject to  uᵢ ≥ -r'w - γ, uᵢ ≥ 0
```

**Key Insight**: Directly control what you care about (tail losses).

---

### 7. Robust Optimization (Advanced)

**Problem Solved**: Inputs (μ, Σ) are uncertain. What if they're wrong?

**Solution**: Optimize for worst case within uncertainty set.

```
maximize    min_{μ ∈ U} μ'w - (λ/2) w'Σw
where U = {μ : ||μ - μ̂||² ≤ κ}
```

**Key Insight**: Build portfolios that work even if estimates are wrong.

---

### 8. Multi-Period Optimization (Advanced)

**Problem Solved**: Single-period ignores future rebalancing.

**Solution**: Dynamic programming over multiple periods.

```
V_T(W) = U(W)  (terminal utility)
V_t(W) = max_w E[V_{t+1}(W × (1 + r'w))]
```

**Key Insight**: Today's decision affects tomorrow's opportunities.

---

### 9. Factor-Based Optimization (Advanced)

**Problem Solved**: Direct covariance estimation fails with many assets.

**Solution**: Model returns through factors.

```
r = Bf + ε

Covariance: Σ = BΣ_f B' + D
where:
  B = factor loadings
  Σ_f = factor covariance (small matrix)
  D = idiosyncratic variance (diagonal)
```

**Key Insight**: Risk comes from common factors, not individual stocks.

---

### 10. GARCH Models (Advanced)

**Problem Solved**: Volatility is not constant (clusters in time).

**Solution**: Model volatility as a process.

```
GARCH(1,1):
σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}

Intuition:
- High volatility yesterday → High volatility today
- Big shock yesterday → Higher volatility today
```

**Key Insight**: Yesterday's volatility predicts today's.

---

### 11. Factor Risk Models (Advanced)

**Problem Solved**: Estimating 500×500 covariance matrix needs millions of data points.

**Solution**: Decompose risk into factor risk + specific risk.

```
Total Risk = Factor Risk + Specific Risk
           = w'BΣ_f B'w + w'Dw

Factor Risk: Systematic, undiversifiable
Specific Risk: Diversifies away with many stocks
```

**Industry Standard**: Barra, Axioma, Bloomberg factor models.

---

### 12. Stress Testing (Advanced)

**Problem Solved**: Normal VaR misses extreme events (2008, 2020).

**Solution**: Simulate portfolio under historical crisis scenarios.

```
Scenarios:
- 2008 Financial Crisis
- 2020 COVID Crash
- 1997 Asian Crisis
- Custom: "What if rates rise 300bp?"
```

**Key Insight**: "What's my loss if 2008 happens again?"

---

### 13. Tail Risk / Extreme Value Theory (Advanced)

**Problem Solved**: Normal distribution underestimates tail events.

**Solution**: Model tails with Generalized Pareto Distribution.

```
For extreme losses (beyond threshold u):
P(X > x | X > u) ≈ (1 + ξ(x-u)/σ)^{-1/ξ}

ξ = shape parameter (tail heaviness)
```

**Key Insight**: Tails are fatter than normal; model them separately.

---

## Method Selection Guide

### By Portfolio Size

| Portfolio Size | Recommended Methods |
|----------------|---------------------|
| Small (< 20 assets) | Mean-Variance, Black-Litterman |
| Medium (20-100) | Risk Parity, HRP, Constrained |
| Large (100+) | Factor-Based, HRP |

### By Investment Horizon

| Horizon | Recommended Methods |
|---------|---------------------|
| Short (< 1 month) | GARCH, Min CVaR |
| Medium (1-12 months) | Mean-Variance, Risk Parity |
| Long (> 1 year) | Multi-Period, Factor-Based |

### By Risk Tolerance

| Risk Profile | Recommended Methods |
|--------------|---------------------|
| Conservative | Min CVaR, Robust Optimization |
| Moderate | Risk Parity, HRP |
| Aggressive | Mean-Variance (Max Sharpe) |

---

## References

1. Markowitz, H. (1952). "Portfolio Selection"
2. Black, F. & Litterman, R. (1992). "Global Portfolio Optimization"
3. Maillard et al. (2010). "The Properties of Equally Weighted Risk Contribution Portfolios"
4. López de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out-of-Sample"
5. Rockafellar & Uryasev (2000). "Optimization of Conditional Value-at-Risk"
6. Goldfarb & Iyengar (2003). "Robust Portfolio Selection Problems"
7. Engle, R. (1982). "Autoregressive Conditional Heteroscedasticity" (ARCH)
8. Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroscedasticity" (GARCH)
9. McNeil & Frey (2000). "Estimation of Tail-Related Risk Measures" (EVT)
