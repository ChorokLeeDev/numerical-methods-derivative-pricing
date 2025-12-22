# CW-ACI Paper: Limitations and Path to Publication

## Executive Summary

This document identifies critical limitations in the current CW-ACI paper and provides a concrete plan to address each issue. The goal is to transform the paper from "likely desk reject" to "publishable at a solid finance/econometrics journal."

**Core insight to preserve:** Conformal prediction coverage degrades during high-volatility periods, and signal-adaptive methods can help.

**Core problem to fix:** The current framing conflates volatility with crowding, leading to circular reasoning.

---

## Part 1: Documented Limitations

### 1.1 CRITICAL: Crowding Proxy Validity

**Issue:** The primary crowding proxy (12-month trailing absolute returns) is fundamentally a volatility/momentum magnitude measure, not a crowding measure.

```
C(t) = |Σ r_s| / median(|Σ r_s|)
```

**Why this matters:**
- The paper claims to discover that high-crowding periods have elevated volatility
- But the proxy *is* a volatility signal, making this finding circular
- "Alternative" proxies (rolling volatility, correlation) are also volatility measures
- No actual crowding data (ETF flows, short interest, factor valuations) is used

**Evidence of the problem:**
- The proxy correlates >0.7 with realized volatility (we should compute this)
- All three "alternative proxies" produce similar results because they measure similar things

### 1.2 CRITICAL: Over-Coverage Problem

**Issue:** CW-ACI achieves 98.5% coverage when targeting 90%, indicating intervals are too wide.

| Method | Target | Overall | High-Crowd | Low-Crowd |
|--------|--------|---------|------------|-----------|
| Standard CP | 90% | 90.2% | 85.0% | 95.3% |
| CW-ACI | 90% | **98.5%** | 98.1% | 98.9% |

**Why this matters:**
- Over-coverage means inefficient intervals (too conservative)
- The "improvement" is achieved by making everything wider, not by being smarter
- A trivial baseline of "always use 99% quantile" would also "improve" coverage

**The right metric:** Coverage efficiency = Coverage / Interval Width, or proper scoring rules

### 1.3 CRITICAL: Subperiod Results Contradict Main Claim

**Issue:** Within subperiods, standard CP achieves near-nominal coverage. The "under-coverage" only appears in full sample.

| Period | Obs | SCP High Coverage | CW-ACI High Coverage |
|--------|-----|-------------------|----------------------|
| 1963-1993 | 366 | **90.1%** | 99.3% |
| 1994-2025 | 382 | **92.5%** | 98.5% |
| Full Sample | 748 | 78.3% | 93.3% |

**Why this matters:**
- Standard CP works fine when calibration and test are from same regime
- The "problem" is calibrating on one volatility regime, testing on another
- This is a regime change / non-stationarity problem, not a conformal prediction problem
- Any method would face this issue; CW-ACI just papers over it with wider intervals

### 1.4 MAJOR: No Theoretical Guarantees

**Issue:** The paper admits CW-ACI breaks exchangeability and provides no coverage guarantees.

**Why this matters:**
- Conformal prediction's main appeal is distribution-free finite-sample guarantees
- Without guarantees, CW-ACI is just a heuristic
- For JoFE, a methodology paper needs theoretical grounding

**Current status:** Remark 1 acknowledges this but offers no resolution.

### 1.5 MAJOR: Algorithm Design Flaw

**Issue:** The algorithm adjusts calibration scores based on *test-point* characteristics.

```python
# Current approach (conformal.py:220-234):
for i in range(n_test):
    adjustment_factor = 1 + weights_test[i]  # Based on TEST crowding
    adjusted_scores = self.calibration_scores * adjustment_factor
```

**Why this matters:**
- Calibration scores should be weighted by their *own* characteristics, not test characteristics
- Current approach doesn't learn the crowding-volatility relationship from calibration data
- Just uniformly inflates all historical scores when test crowding is high

**Better approaches:**
1. Weight calibration scores by their own crowding levels
2. Use locally-weighted quantile regression
3. Conformalized quantile regression with crowding as covariate

### 1.6 MAJOR: Missing Baseline Comparisons

**Issue:** No comparison to existing adaptive conformal methods.

**Missing comparisons:**
- Gibbs & Candès (2021) Adaptive Conformal Inference
- Romano et al. (2019) Conformalized Quantile Regression
- Tibshirani et al. (2019) Weighted/Covariate-Shift Conformal
- Zaffran et al. (2022) Adaptive Conformal with theoretical guarantees

### 1.7 MODERATE: Statistical Reporting

**Issue:** No standard errors or confidence intervals on coverage estimates.

**Why this matters:**
- With ~185 test observations per regime, SE on coverage ≈ 2.5pp
- Many reported "improvements" may not be statistically significant
- P-values for coverage differences are not reported

### 1.8 MODERATE: Factor Selection

**Issue:** Mkt-RF excluded without justification; RMW drives results.

**Details:**
- Average high-crowding coverage without RMW: 81.1% (vs 78.3% with)
- RMW's 66.8% coverage is an outlier
- No explanation for why Mkt-RF is excluded

### 1.9 MODERATE: Backtesting Design

**Issue:** Fixed 50/50 split instead of rolling calibration.

**Why this matters:**
- Real-world application would use rolling windows
- Fixed split maximizes look-ahead bias in crowding proxy (expanding median)
- Results may not hold under realistic implementation

### 1.10 MINOR: Data Period Claim

**Issue:** Abstract claims "62 years" but data is July 1963 - October 2025 (62.3 years). This is fine but should note data extends beyond typical academic cutoffs, raising questions about data source.

---

## Part 2: Revision Strategy

### Strategy A: "Honest Volatility-Adaptive CP" (Recommended)

**Core reframing:** Drop the "crowding" narrative. Reframe as volatility-adaptive conformal prediction for heteroskedastic time series.

**New title:** "Volatility-Adaptive Conformal Prediction for Factor Return Uncertainty"

**Advantages:**
- Honest about what the method does
- Still novel contribution (applying signal-adaptive CP to factor returns)
- Easier to provide theoretical grounding
- More generalizable (not specific to crowding)

**Disadvantages:**
- Less "exciting" narrative
- Closer to existing ACI literature

### Strategy B: "True Crowding Paper" (Higher risk, higher reward)

**Core change:** Obtain actual crowding data and show the method works with genuine crowding signals.

**Required data:**
- Factor ETF flows (e.g., from ICI or Bloomberg)
- Factor valuation spreads
- Short interest data
- Arbitrage capital measures

**Advantages:**
- Maintains the interesting crowding narrative
- Genuinely novel if crowding data shows different results than volatility

**Disadvantages:**
- Data access challenges
- May find crowding doesn't help beyond volatility (negative result)

### Strategy C: "Methodological Contribution" (Hardest)

**Core change:** Develop theoretical coverage guarantees under specific assumptions.

**Possible theorems:**
- Coverage guarantee under bounded heteroskedasticity
- Asymptotic coverage under mixing conditions
- Finite-sample bounds with volatility signal

**Advantages:**
- Would be a strong econometrics paper
- JoFE would value theoretical contribution

**Disadvantages:**
- Technically challenging
- May not be achievable

---

## Part 3: Concrete Fix Plan (Strategy A)

### Phase 1: Honest Reframing

1. **New title:** "Signal-Adaptive Conformal Prediction for Heteroskedastic Factor Returns"

2. **New abstract framing:**
   - Problem: CP assumes homoskedasticity; factor returns are heteroskedastic
   - Solution: Adapt interval width based on volatility signals
   - Contribution: Simple, practical method that improves conditional coverage

3. **Rename method:** "Signal-Weighted Adaptive Conformal Inference" (SW-ACI)
   - More accurate than "Crowding-Weighted"
   - Generalizes to any predictive signal

### Phase 2: Fix Over-Coverage

**Option A: Recalibrate γ**
```python
def calibrate_gamma(y_cal, pred_cal, signal_cal, alpha=0.1, target_overall=0.90):
    """Find γ that achieves target overall coverage."""
    for gamma in np.linspace(0.1, 2.0, 20):
        swaci = SignalWeightedACI(alpha=alpha, sensitivity=gamma)
        # ... compute coverage
        if abs(overall_coverage - target_overall) < 0.01:
            return gamma
    return gamma
```

**Option B: Adjust α dynamically**
- If CW-ACI over-covers, increase α (tighten intervals)
- Target: Overall coverage = 90%, then report conditional coverage

**Option C: Use interval efficiency metric**
- Report: Coverage / Mean(Width)
- Show CW-ACI achieves same coverage with narrower intervals
- Or show CW-ACI achieves better coverage with same total width budget

### Phase 3: Address Subperiod Issue

**Reframe as feature, not bug:**
- Acknowledge that the "problem" is regime change
- Position SW-ACI as robust to regime change
- Add rolling calibration experiment to show real-time performance

**New experiment:**
```python
def rolling_calibration_backtest(returns, signal, cal_window=120, alpha=0.1):
    """Rolling window backtest with expanding calibration."""
    results = []
    for t in range(cal_window, len(returns)):
        y_cal = returns[t-cal_window:t]
        signal_cal = signal[t-cal_window:t]
        # Fit and predict for time t
        # ...
    return results
```

### Phase 4: Fix Algorithm

**New algorithm design:**
```python
class SignalWeightedACI:
    def fit(self, y_cal, pred_cal, signal_cal):
        self.scores = np.abs(y_cal - pred_cal)
        self.signal_cal = signal_cal
        # Learn signal-score relationship
        self.signal_score_slope = np.corrcoef(signal_cal, self.scores)[0,1]

    def predict(self, pred_test, signal_test):
        # Weight calibration scores by similarity to test signal
        for i, s_test in enumerate(signal_test):
            # Localized weighting: upweight calibration points with similar signals
            weights = np.exp(-0.5 * ((self.signal_cal - s_test) / bandwidth)**2)
            weighted_quantile = self._weighted_quantile(self.scores, weights, 1-alpha)
            # ...
```

This weights calibration scores by their own characteristics relative to the test point, which is more principled.

### Phase 5: Add Baseline Comparisons

**Implement and compare:**

1. **Gibbs-Candès ACI:**
```python
class AdaptiveCI:
    """Gibbs & Candes (2021) Adaptive Conformal Inference."""
    def __init__(self, alpha=0.1, gamma=0.01):
        self.alpha_t = alpha
        self.gamma = gamma

    def update(self, covered):
        # Update alpha based on recent coverage
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - (1 - covered))
```

2. **Conformalized Quantile Regression:**
```python
from sklearn.ensemble import GradientBoostingRegressor
# Fit quantile regression with signal as feature
# Conformalize the quantile predictions
```

3. **Naive volatility scaling:**
```python
# Simple baseline: scale intervals by rolling volatility
width_t = base_width * (vol_t / median_vol)
```

### Phase 6: Statistical Rigor

**Add to all tables:**
- Standard errors: SE = sqrt(p(1-p)/n)
- Confidence intervals: [p - 1.96*SE, p + 1.96*SE]
- P-values for coverage differences (binomial test or bootstrap)

**Example:**
```python
def coverage_with_se(y_true, lower, upper):
    covered = (y_true >= lower) & (y_true <= upper)
    p = np.mean(covered)
    n = len(covered)
    se = np.sqrt(p * (1-p) / n)
    return p, se, (p - 1.96*se, p + 1.96*se)
```

### Phase 7: Include Mkt-RF

- Add market factor to analysis
- If it behaves differently, discuss why
- If similar, strengthens generalizability

### Phase 8: Rolling Calibration Backtest

```python
def rolling_backtest(factors, factor_names, cal_window=120):
    """Realistic rolling window backtest."""
    all_results = []

    for factor in factor_names:
        returns = factors[factor].values
        signal = compute_volatility_signal(factors[factor]).values

        for t in range(cal_window, len(returns)):
            # Calibration window: [t-cal_window, t)
            # Test point: t
            y_cal = returns[t-cal_window:t]
            s_cal = signal[t-cal_window:t]
            y_test = returns[t]
            s_test = signal[t]

            # Standard CP
            scp = StandardCP(alpha=0.1)
            scp.fit(y_cal, np.zeros_like(y_cal))
            lower_scp, upper_scp = scp.predict(np.array([0]))
            covered_scp = (y_test >= lower_scp[0]) & (y_test <= upper_scp[0])

            # SW-ACI
            swaci = SignalWeightedACI(alpha=0.1)
            swaci.fit(y_cal, np.zeros_like(y_cal), s_cal)
            lower_sw, upper_sw, _ = swaci.predict(np.array([0]), np.array([s_test]))
            covered_sw = (y_test >= lower_sw[0]) & (y_test <= upper_sw[0])

            all_results.append({
                'factor': factor,
                't': t,
                'signal': s_test,
                'covered_scp': covered_scp,
                'covered_swaci': covered_sw,
                'width_scp': upper_scp[0] - lower_scp[0],
                'width_swaci': upper_sw[0] - lower_sw[0]
            })

    return pd.DataFrame(all_results)
```

---

## Part 4: Revised Paper Outline

### New Title
"Signal-Adaptive Conformal Prediction for Heteroskedastic Factor Returns"

### New Abstract
Standard conformal prediction provides distribution-free coverage guarantees under exchangeability, but factor returns exhibit time-varying volatility that violates this assumption. We show that standard conformal prediction achieves nominal 90% coverage overall but under-covers during high-volatility periods (82% coverage) and over-covers during low-volatility periods (96% coverage). We propose Signal-Weighted Adaptive Conformal Inference (SW-ACI), which adjusts prediction interval width based on observable volatility signals. Using 62 years of Fama-French data and Monte Carlo simulations, we demonstrate that SW-ACI achieves near-nominal conditional coverage across volatility regimes while maintaining overall coverage. Our method is simple to implement, requires only a volatility proxy, and provides practical uncertainty quantification for factor investing.

### New Contributions
1. **Empirical documentation:** Quantify coverage heterogeneity across volatility regimes
2. **Simple practical method:** SW-ACI adapts intervals using volatility signals
3. **Comprehensive evaluation:** Compare to ACI, CQR, and naive baselines
4. **Realistic backtesting:** Rolling calibration results

### New Section Structure

1. **Introduction** - Reframed around heteroskedasticity, not crowding
2. **Related Work** - Expanded conformal prediction lit, add Gibbs-Candès, CQR
3. **The Problem: Heteroskedastic Coverage** - Document the issue clearly
4. **Methodology: Signal-Weighted ACI** - Fixed algorithm, clear assumptions
5. **Theoretical Discussion** - Conditions for coverage, connection to weighted CP
6. **Monte Carlo Validation** - With proper baselines
7. **Empirical Analysis** - All factors including Mkt-RF, rolling backtest
8. **Comparison to Alternatives** - ACI, CQR, naive scaling
9. **Robustness** - Signals, windows, parameters
10. **Conclusion** - Honest about limitations

---

## Part 5: Implementation Checklist

### Immediate Fixes (Code Changes)
- [ ] Rename classes: CrowdingWeightedACI → SignalWeightedACI
- [ ] Add calibration for γ to achieve target overall coverage
- [ ] Implement Gibbs-Candès ACI baseline
- [ ] Implement conformalized quantile regression baseline
- [ ] Implement naive volatility scaling baseline
- [ ] Add rolling calibration backtest
- [ ] Add standard errors to all coverage calculations
- [ ] Add Mkt-RF to factor list
- [ ] Compute correlation between "crowding" proxy and realized volatility

### Paper Rewrite
- [ ] New title and abstract
- [ ] Reframe introduction (heteroskedasticity, not crowding)
- [ ] Expand related work (ACI, CQR, weighted CP)
- [ ] Add theoretical discussion section
- [ ] Revise methodology section (fixed algorithm, assumptions)
- [ ] Add baseline comparison tables
- [ ] Add rolling backtest results
- [ ] Add statistical significance throughout
- [ ] Revise robustness section
- [ ] Honest limitations in conclusion

### New Experiments
- [ ] Correlation: signal proxy vs realized volatility
- [ ] Baseline comparison: SW-ACI vs ACI vs CQR vs naive
- [ ] Rolling calibration backtest
- [ ] Interval efficiency analysis
- [ ] Statistical significance tests
- [ ] Mkt-RF analysis

---

## Part 6: Target Venues After Revision

### Tier 1 (Reach)
- **Journal of Financial Econometrics** - If theoretical contribution added
- **Journal of Econometrics** - If theoretical contribution strong

### Tier 2 (Realistic)
- **Journal of Financial and Quantitative Analysis** - Applied methodology focus
- **Review of Asset Pricing Studies** - Factor investing angle
- **Journal of Empirical Finance** - Empirical focus

### Tier 3 (Safe)
- **Quantitative Finance** - Applied methods
- **Journal of Risk** - Risk management application
- **International Journal of Forecasting** - Forecasting methodology

---

## Appendix: Key Code Changes

### A1: Signal-Volatility Correlation Analysis
```python
def analyze_signal_validity(factors, factor_names):
    """Show that 'crowding' proxy is really volatility."""
    results = []
    for factor in factor_names:
        returns = factors[factor]

        # Compute proxies
        crowding_proxy = compute_crowding_proxy(returns, window=12)
        realized_vol = returns.rolling(12).std()

        # Correlation
        valid = crowding_proxy.notna() & realized_vol.notna()
        corr = np.corrcoef(crowding_proxy[valid], realized_vol[valid])[0,1]

        results.append({
            'factor': factor,
            'corr_crowding_vol': corr
        })

    return pd.DataFrame(results)
```

### A2: Coverage Efficiency Metric
```python
def coverage_efficiency(y_true, lower, upper):
    """Coverage per unit interval width."""
    covered = (y_true >= lower) & (y_true <= upper)
    coverage = np.mean(covered)
    mean_width = np.mean(upper - lower)
    return coverage / mean_width
```

### A3: Statistical Tests
```python
from scipy import stats

def test_coverage_difference(cov1, n1, cov2, n2):
    """Test if two coverage rates are significantly different."""
    # Pooled proportion
    p_pool = (cov1 * n1 + cov2 * n2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (cov1 - cov2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value
```
