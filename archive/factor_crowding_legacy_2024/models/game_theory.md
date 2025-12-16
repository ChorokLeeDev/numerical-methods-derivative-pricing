# Game-Theoretic Model of Factor Alpha Decay

## 1. Setup

### Players
- **N(t)** agents who have discovered a profitable signal at time t
- Each agent is small relative to market (price-taker individually)
- Collectively, agents affect prices through market impact

### Signal
- Signal $s$ predicts excess return $r$ with edge $\alpha$
- When undiscovered: $E[r | s] = \alpha_0 > 0$
- Total "alpha capacity" of the market: $K$

### Information Structure
- Agents discover signal sequentially (Poisson process with rate $\lambda$)
- Once discovered, agent trades optimally given others' behavior
- No coordination — Nash equilibrium

---

## 2. Single-Period Model

### Setup
- $N$ agents have discovered signal
- Each agent trades quantity $q_i$
- Market impact: $\Delta P = \gamma \sum_i q_i$ (Kyle's lambda)

### Agent's Problem
Each agent maximizes:
$$
\max_{q_i} \quad E[q_i \cdot (r - \gamma \sum_j q_j)]
$$

### Symmetric Nash Equilibrium
In symmetric equilibrium, $q_i = q^*$ for all $i$:
$$
q^* = \frac{\alpha_0}{2\gamma N}
$$

### Equilibrium Alpha per Agent
$$
\alpha_i = \frac{\alpha_0}{N} - \gamma q^* (N-1) = \frac{\alpha_0}{2N}
$$

### Key Result 1: Alpha Decay
$$
\boxed{\alpha(N) = \frac{\alpha_0}{2N} = \frac{K}{N}}
$$

where $K = \alpha_0 / 2$ is the effective alpha capacity.

**Interpretation**: Alpha per agent decays as $1/N$ — hyperbolic decay.

---

## 3. Dynamic Model with Entry

### Discovery Process
- Agents discover signal at rate $\lambda$ (Poisson)
- Expected number at time $t$: $E[N(t)] = \lambda t$ (for early times)

### Alpha Decay Over Time
Substituting $N(t) = \lambda t$:
$$
\alpha(t) = \frac{K}{1 + \lambda t}
$$

This is our **hyperbolic decay model**.

### Alternative: Capacity Constraint
If there's a maximum $\bar{N}$ agents (market capacity):
$$
N(t) = \bar{N}(1 - e^{-\lambda t})
$$

Then:
$$
\alpha(t) = \frac{K}{\bar{N}(1 - e^{-\lambda t})} \approx K \cdot e^{-\lambda t} \quad \text{(for large } t\text{)}
$$

This gives **exponential decay**.

---

## 4. Entry Equilibrium

### Setup
- Entry cost: $c$ (research cost, infrastructure, etc.)
- Agent enters if expected profit > cost

### Entry Condition
Agent $N+1$ enters if:
$$
\frac{K}{N+1} > c
$$

### Equilibrium Number of Agents
$$
N^* = \frac{K}{c}
$$

### Key Result 2: Long-Run Alpha
$$
\boxed{\alpha_{\infty} = c}
$$

**Interpretation**: In equilibrium, alpha equals entry cost. This is the "zero-profit condition" — the factor still works, but only enough to cover costs.

---

## 5. Testable Predictions

### Prediction 1: Hyperbolic Decay
$$
\text{Sharpe}_t = \frac{K}{1 + \lambda t}
$$

**Test**: Fit to rolling Sharpe ratio. We found $R^2 = 0.65$ for momentum.

### Prediction 2: Faster Decay for "Easy" Factors
- $\lambda$ higher for widely-published factors
- $\lambda$ higher for low-cost-to-implement factors

**Test**: Compare decay rates:
- Momentum (well-known) vs. exotic factors
- US (liquid) vs. emerging markets (illiquid)

### Prediction 3: Crowding Proxy Correlation
If $N(t)$ is proxied by ETF AUM:
$$
\text{Corr}(\alpha_t, 1/\text{AUM}_t) > 0
$$

---

## 6. Model Parameters from Data

### Momentum Factor Fit
From empirical analysis:
- $K = 1.66$ (initial Sharpe capacity)
- $\lambda = 0.015$ per month
- $R^2 = 0.65$

### Implied Dynamics
- **Half-life**: $t_{1/2} = 1/\lambda = 67$ months ≈ 5.5 years
- **Time to 90% decay**: $t_{90} = 9/\lambda ≈ 50$ years

### Interpretation
Momentum alpha decays with half-life of ~5 years. This matches:
- Jegadeesh & Titman (1993) publication → decay visible by late 1990s
- ~5-7 year lag from publication to crowding

---

## 7. Extensions (Future Work)

### 7.1 Heterogeneous Agents
- Different entry costs $c_i$
- Different information quality
- Results in continuous entry over time

### 7.2 Strategic Hiding
- Agents can trade less to hide signal
- Trade-off: less profit now vs. longer edge life
- Stackelberg extension

### 7.3 Multiple Factors
- Agents choose which factor to trade
- Competition across factors
- Equilibrium allocation of capital

### 7.4 Regime Dependence
- $\alpha_0$ varies with market regime
- Crowding effect stronger in certain regimes
- State-dependent decay rates

---

## 8. Connection to Literature

### Kyle (1985)
- Our $\gamma$ is Kyle's $\lambda$
- Multiple informed traders splits profit

### McLean & Pontiff (2016)
- 50% decay post-publication
- Our model: $\alpha(t)/\alpha_0 = 1/(1+\lambda t)$
- At $t = 1/\lambda$, decay is 50% ✓

### Grossman & Stiglitz (1980)
- Information has value only if not fully reflected
- Our equilibrium: some alpha remains (= cost)
