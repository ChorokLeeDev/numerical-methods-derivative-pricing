# Risk Management: Theory and Implementation

This document explains the risk management concepts implemented in `quant-core`.

---

## 1. Covariance Estimation & Shrinkage

### The Problem with Sample Covariance

When you estimate covariance from data:

```
Sample Covariance = True Covariance + Estimation Noise
```

This noise gets **amplified** by portfolio optimization:
- Mean-Variance optimizer is an "estimation error maximizer"
- Small errors in covariance → Large errors in weights
- When `n_assets > n_observations`: Matrix is **singular** (can't invert!)

### Shrinkage Solution

**Key Idea**: Combine two estimators:

| Estimator | Variance | Bias | When to use |
|-----------|----------|------|-------------|
| Sample Covariance (Σ_sample) | High | Low | Lots of data |
| Structured Target (F) | Low | High | Little data |

**Shrinkage Formula**:
```
Σ_shrunk = α × F + (1 - α) × Σ_sample

where:
  α = shrinkage intensity (0 to 1)
  F = structured target (e.g., constant correlation model)
```

### Ledoit-Wolf Method

The **optimal α** is derived analytically (no cross-validation needed!):

```
α* = (estimation error of sample) / (squared deviation from target)
```

**Target Matrix (Constant Correlation)**:
```
F_ij = { σ_i²           if i = j     (keep variances)
       { ρ̄ × σ_i × σ_j   if i ≠ j     (use average correlation)
```

**When to Use**:
- Few observations relative to assets
- Want stable covariance estimates
- Any production portfolio optimization

**Implementation**: `quant_core.ledoit_wolf(returns)` → `(shrunk_cov, alpha)`

---

## 2. Risk Decomposition: MCR, CCR, Risk Parity

### Why Decompose Risk?

A portfolio's risk comes from multiple sources. Risk decomposition answers:
**"Where is my risk coming from?"**

### Key Metrics

#### Marginal Contribution to Risk (MCR)

```
MCR_i = ∂σ_p / ∂w_i = (Σ × w)_i / σ_p
```

**Interpretation**: "If I increase asset i's weight by 1%, how much does portfolio volatility change?"

#### Component Contribution to Risk (CCR)

```
CCR_i = w_i × MCR_i
```

**Key Property**: `Σ CCR_i = σ_p` (they sum to total portfolio risk!)

**Interpretation**: "How much of total risk is 'caused' by asset i?"

#### Percentage Contribution to Risk

```
PCT_i = CCR_i / σ_p
```

Sums to 100%. Shows risk budget allocation.

### Risk Parity

**Goal**: Find weights where all assets contribute equally to risk.

```
CCR_1 = CCR_2 = ... = CCR_n = σ_p / n
```

**Why Risk Parity?**
- **Diversification**: No single asset dominates risk
- **Robust**: Doesn't need expected return estimates (Mean-Variance needs μ)
- **Popular**: Bridgewater's "All Weather" fund uses this approach

**Implementation**:
```python
from quant_core import risk_parity_weights, percentage_contribution_to_risk

weights = risk_parity_weights(cov_matrix)
pct_risk = percentage_contribution_to_risk(weights, cov_matrix)
# pct_risk ≈ [0.33, 0.33, 0.33] for 3 assets
```

---

## 3. Value at Risk (VaR) and Conditional VaR (CVaR)

### Value at Risk (VaR)

**Question**: "What is the maximum loss I can expect at a given confidence level?"

```
VaR_α = "With (1-α)% confidence, I won't lose more than VaR_α"

Example: VaR_0.05 = $1M means:
"With 95% confidence, I won't lose more than $1M in one day"
```

#### Calculation Methods

**1. Parametric (Normal) VaR**:
```
VaR_α = -μ + z_α × σ

where z_α is the α-quantile of standard normal:
  z_0.05 ≈ -1.645 (95% confidence)
  z_0.01 ≈ -2.326 (99% confidence)
```

**2. Historical VaR**:
Simply take the α-percentile of historical returns.

### Conditional VaR (CVaR / Expected Shortfall)

**Question**: "If I DO exceed VaR, what's my expected loss?"

```
CVaR_α = E[Loss | Loss > VaR_α]
```

### VaR vs CVaR Comparison

| Aspect | VaR | CVaR |
|--------|-----|------|
| What it measures | Threshold | Tail average |
| Extreme events | Ignores | Captures |
| Mathematical property | Not coherent | Coherent |
| Regulatory use | Basel II | Basel III (preferred) |

**Coherence** means the risk measure satisfies:
- **Monotonicity**: More risk = higher measure
- **Sub-additivity**: Diversification reduces risk
- **Positive homogeneity**: Doubling position doubles risk
- **Translation invariance**: Adding cash reduces risk

**VaR fails sub-additivity!** Two portfolios combined can have higher VaR than sum.

**Implementation**:
```python
from quant_core import parametric_var, historical_cvar

var_95 = parametric_var(returns, alpha=0.05)  # 95% VaR
cvar_95 = historical_cvar(returns, alpha=0.05)  # Expected Shortfall
# cvar_95 >= var_95 always
```

---

## 4. Hierarchical Risk Parity (HRP)

### Problems with Traditional Methods

| Method | Problem |
|--------|---------|
| Mean-Variance | Needs μ estimates (hard!), sensitive to errors |
| Risk Parity | Still needs covariance inversion or iteration |
| Both | Often produce concentrated portfolios |

### HRP Solution (López de Prado, 2016)

**Key Insight**: Use hierarchical clustering to:
1. Group similar assets together
2. Allocate within groups first, then between groups
3. **No matrix inversion needed!**

### The 3-Step Algorithm

#### Step 1: Tree Clustering

Convert correlation to distance:
```
d_ij = √((1 - ρ_ij) / 2)

Maps correlation [-1, 1] to distance [0, 1]:
  ρ = +1 (perfect positive) → d = 0
  ρ =  0 (uncorrelated)     → d = 0.707
  ρ = -1 (perfect negative) → d = 1
```

Apply hierarchical clustering (single linkage).

#### Step 2: Quasi-Diagonalization

Reorder assets so similar ones are adjacent. This makes the covariance matrix "quasi-diagonal" (block structure).

```
Before reordering:        After reordering:
┌─────────────┐           ┌─────────────┐
│ x x . . x . │           │ x x . . . . │
│ x x . . x . │           │ x x . . . . │
│ . . x x . x │    →      │ . . x x . . │
│ . . x x . x │           │ . . x x . . │
│ x x . . x . │           │ . . . . x x │
│ . . x x . x │           │ . . . . x x │
└─────────────┘           └─────────────┘
  Random order              Block structure
```

#### Step 3: Recursive Bisection

Split portfolio recursively:
```
For each cluster split:
  1. Calculate cluster variances (left vs right)
  2. Allocate inversely: w_left ∝ 1/var_left
  3. Recurse into each sub-cluster
```

### Why HRP Works

1. **No inversion**: Avoids amplifying estimation errors
2. **Hierarchical**: Respects correlation structure
3. **Diversified**: Naturally spreads weights across clusters
4. **Robust**: Works even when n > T (more assets than observations)

**Implementation**:
```python
from quant_core import hrp_weights

weights = hrp_weights(cov_matrix)
# Automatically diversified across correlation clusters
```

---

## 5. Method Comparison

| Method | Needs μ? | Needs Σ⁻¹? | Robustness | Diversification |
|--------|----------|------------|------------|-----------------|
| Mean-Variance | Yes | Yes | Low | Variable |
| Risk Parity | No | Iterative | Medium | High |
| HRP | No | No | High | High |
| Equal Weight | No | No | Highest | Forced equal |

### When to Use What

- **Mean-Variance**: When you have strong return forecasts
- **Risk Parity**: When you want risk-balanced allocation
- **HRP**: When you have many assets or few observations
- **Equal Weight**: Baseline benchmark

---

## 6. Constrained Optimization

### The Problem

Real portfolios have constraints that unconstrained Mean-Variance ignores:

| Constraint | Example | Why It Matters |
|------------|---------|----------------|
| Long-only | w ≥ 0 | Many funds can't short |
| Position limits | w ≤ 10% | Concentration risk |
| Sector limits | Tech ≤ 30% | Regulatory/risk management |
| Budget | Σw = 1 | Fully invested |

### Mathematical Formulation

```
minimize    (1/2) w' Σ w - λ⁻¹ μ' w    (risk - return tradeoff)
subject to  w' 1 = 1                    (budget constraint)
            w_min ≤ w ≤ w_max           (box constraints)
```

This is a **Quadratic Programming (QP)** problem.

### Solution: Projected Gradient Descent

**Algorithm**:
```
1. Initialize: w = equal weight (feasible starting point)
2. Loop:
   a. Compute gradient: g = Σw - λ⁻¹μ
   b. Update: w_new = w - α × g
   c. Project: w_new = Project(w_new) onto constraints
   d. Check convergence
3. Return optimal w
```

**Projection** ensures weights satisfy all constraints:
- Clip to [w_min, w_max]
- Normalize to sum to 1

### Why Projected Gradient Descent?

| Method | Pros | Cons |
|--------|------|------|
| Interior Point | Fast, exact | Complex to implement |
| Active Set | Handles equality constraints well | Can be slow |
| **Projected GD** | Simple, robust | May be slower |
| Commercial (Gurobi) | Best performance | Expensive license |

For portfolio problems (typically < 1000 assets), Projected GD is sufficient.

### Implementation

```python
from quant_core import min_variance_constrained, max_sharpe_constrained
import numpy as np

# Long-only minimum variance
weights = min_variance_constrained(cov_matrix)

# With position limits (max 15% per asset)
w_max = np.full(n_assets, 0.15)
weights = min_variance_constrained(cov_matrix, w_max=w_max)

# Maximum Sharpe with constraints
weights = max_sharpe_constrained(
    cov_matrix,
    expected_returns,
    risk_free_rate=0.02,
    w_min=np.zeros(n),  # Long-only
    w_max=np.full(n, 0.10)  # Max 10%
)
```

### Constrained vs Unconstrained Comparison

```
Example: 2 assets, one with 10% vol, one with 50% vol

Unconstrained Min-Var:
  w = [1.0, 0.0]  (100% in low-vol asset)

Constrained (max 60% per asset):
  w = [0.6, 0.4]  (forced diversification)
```

The constrained solution has higher variance but is more robust and practical.

---

## References

1. Ledoit, O., & Wolf, M. (2004). "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices"
2. Maillard, S., Roncalli, T., & Teïletche, J. (2010). "The Properties of Equally Weighted Risk Contribution Portfolios"
3. López de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out-of-Sample"
4. Rockafellar, R.T., & Uryasev, S. (2000). "Optimization of Conditional Value-at-Risk"
