# Research Plan: CW-ACI for JoFE

**Target:** Journal of Financial Econometrics - Special Issue on Machine Learning
**Deadline:** March 1, 2026
**Timeline:** 14 months

---

## Research Question

**Primary:** Does standard conformal prediction systematically under-cover factor returns during high-crowding periods, and can crowding-weighted intervals fix this?

**Secondary:**
1. How much does coverage drop during high crowding?
2. How much wider should intervals be to achieve nominal coverage?
3. Is the effect consistent across different factors?
4. Is the effect robust to alternative crowding proxies?

---

## Hypothesis

**H1:** Standard conformal prediction under-covers during high-crowding periods
- Expected: Coverage drops from 90% to 70-80% when crowding is high
- Mechanism: Volatility clusters during crowding, violating exchangeability

**H2:** CW-ACI improves conditional coverage
- Expected: CW-ACI achieves 85-95% coverage during high crowding
- Mechanism: Wider intervals during high crowding capture volatility

**H3:** The effect is economically meaningful
- Expected: Portfolios using CW-ACI have better risk management
- Mechanism: Correct uncertainty quantification enables better hedging

---

## Data

### Primary Data
- **Source:** Kenneth French Data Library
- **Factors:** Mkt-RF, SMB, HML, RMW, CMA, Mom, ST_Rev, LT_Rev
- **Period:** July 1963 - Present (~750 months)
- **Frequency:** Monthly

### Crowding Proxies
1. **Primary:** Trailing 12-month absolute return (simple, transparent)
2. **Alternative 1:** Rolling volatility
3. **Alternative 2:** Market correlation
4. **Alternative 3:** ETF flows (post-2000 only)

---

## Methodology

### Standard Conformal Prediction (Baseline)
```
1. Split data: 50% calibration, 50% test
2. Compute nonconformity scores: s_i = |y_i - ŷ_i|
3. Find quantile: q = Quantile(s, 0.9)
4. Interval: [ŷ - q, ŷ + q]
```

### CW-ACI (Proposed)
```
1. Split data: 50% calibration, 50% test
2. Compute nonconformity scores: s_i = |y_i - ŷ_i|
3. Compute crowding weights: w_i = sigmoid(crowding_i)
4. For each test point with crowding c:
   a. Adjust scores: s_adj = s * (1 + w(c))
   b. Find quantile: q = Quantile(s_adj, 0.9)
   c. Interval: [ŷ - q, ŷ + q]
```

### Evaluation Metrics
1. **Overall coverage:** P(y ∈ [lower, upper])
2. **Conditional coverage:** Coverage during high/low crowding
3. **Interval width:** Average and distribution
4. **Width ratio:** CW-ACI width / Standard CP width by crowding level

---

## Experiments

### Experiment 1: Coverage Analysis (Core Result)
**Goal:** Document under-coverage of standard CP during high crowding

**Method:**
- Apply standard CP to each factor
- Split test period by crowding level (above/below median)
- Compare coverage in high vs low crowding periods

**Expected Result:**
- Overall coverage ≈ 90%
- High crowding coverage ≈ 70-80%
- Low crowding coverage ≈ 95-100%

### Experiment 2: CW-ACI Improvement
**Goal:** Show CW-ACI fixes under-coverage

**Method:**
- Apply CW-ACI to same data
- Compare conditional coverage to standard CP

**Expected Result:**
- CW-ACI achieves 85-95% coverage during high crowding
- Improvement of 15-25 percentage points

### Experiment 3: Monte Carlo Validation
**Goal:** Verify CW-ACI works under controlled conditions

**Method:**
- Simulate data with known crowding-volatility relationship
- Vary: sample size, crowding effect strength, sensitivity parameter
- Report coverage across 1000 replications

**Expected Result:**
- CW-ACI achieves near-nominal coverage across settings
- Standard CP under-covers when crowding effect is strong

### Experiment 4: Robustness
**Goal:** Verify results hold under alternative specifications

**Tests:**
- Alternative crowding proxies
- Different calibration/test splits
- Rolling vs expanding windows
- Different factors
- Subperiod analysis

### Experiment 5: Economic Significance (Optional)
**Goal:** Show CW-ACI improves portfolio risk management

**Method:**
- Construct hedging strategy based on prediction intervals
- Compare portfolio metrics: Sharpe, max drawdown, VaR

---

## Paper Structure

### Section 1: Introduction (2 pages)
- Conformal prediction for financial forecasting
- Problem: under-coverage during volatile periods
- Contribution: CW-ACI adapts to crowding signals

### Section 2: Related Work (1.5 pages)
- Conformal prediction basics
- Finance applications of CP
- Crowding and factor returns

### Section 3: Methodology (3 pages)
- 3.1 Standard Conformal Prediction
- 3.2 Crowding Signal Construction
- 3.3 Crowding-Weighted ACI
- 3.4 Coverage Properties

### Section 4: Monte Carlo Study (3 pages)
- 4.1 Simulation Design
- 4.2 Coverage Results
- 4.3 Sensitivity Analysis

### Section 5: Empirical Analysis (4 pages)
- 5.1 Data and Setup
- 5.2 Coverage by Crowding Regime
- 5.3 Interval Width Adaptation
- 5.4 Factor-by-Factor Results

### Section 6: Robustness (2 pages)
- Alternative crowding proxies
- Subperiod stability
- Sensitivity to parameters

### Section 7: Conclusion (1 page)

### Appendix
- A: Proofs/Technical Details
- B: Additional Tables
- C: Data Description

**Total: ~18 pages** (JoFE limit: 30 pages)

---

## Timeline

### Phase 1: Foundation (January 2025)
- [ ] Week 1: Set up clean codebase
- [ ] Week 2: Implement standard CP and CW-ACI
- [ ] Week 3: Run basic coverage analysis
- [ ] Week 4: Document initial findings

### Phase 2: Monte Carlo (February-March 2025)
- [ ] Design simulation framework
- [ ] Run Monte Carlo experiments
- [ ] Analyze coverage properties
- [ ] Write Section 4

### Phase 3: Empirical Analysis (April-May 2025)
- [ ] Complete factor-by-factor analysis
- [ ] Generate all tables and figures
- [ ] Write Section 5

### Phase 4: Theory & Robustness (June-July 2025)
- [ ] Formalize coverage properties
- [ ] Run all robustness checks
- [ ] Write Sections 3 and 6

### Phase 5: Writing (August-October 2025)
- [ ] Complete first draft
- [ ] Internal review
- [ ] Iterate on feedback

### Phase 6: Polish (November-December 2025)
- [ ] Final robustness checks
- [ ] Prepare replication package
- [ ] Format per JoFE guidelines

### Phase 7: Submit (January-February 2026)
- [ ] Final proofreading
- [ ] Write cover letter
- [ ] Submit by February 15, 2026

---

## Success Criteria

### Minimum Viable Paper
- [ ] Coverage analysis complete (Experiment 1-2)
- [ ] Monte Carlo validation (Experiment 3)
- [ ] Basic robustness (one alternative proxy)
- [ ] Clean, reproducible code

### Ideal Paper
- [ ] All experiments complete
- [ ] Multiple robustness checks
- [ ] Economic significance analysis
- [ ] Theoretical coverage discussion
- [ ] Professional figures and tables

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| CW-ACI doesn't work on other data | Low | High | We already tested on 5 factors |
| Reviewers question crowding proxy | Medium | Medium | Multiple proxies in robustness |
| Not novel enough | Medium | High | Focus on finance application |
| Deadline pressure | Medium | Medium | Start early, phase work |

---

## Key References

### Conformal Prediction
1. Vovk, Gammerman, Shafer (2005) - Algorithmic Learning in a Random World
2. Lei et al. (2018) - Distribution-Free Predictive Inference
3. Romano et al. (2019) - Conformalized Quantile Regression
4. Angelopoulos & Bates (2021) - A Gentle Introduction to CP

### Finance Applications
5. Fantazzini (2024) - ACI for Crypto VaR
6. Bastos (2024) - CP for Option Prices
7. Zaffran et al. (2022) - ACI under Distribution Shift

### Factor Investing
8. Fama & French (2015) - Five-Factor Model
9. DeMiguel et al. (2020) - Factor Crowding
10. McLean & Pontiff (2016) - Factor Decay

---

## Notes

This research plan is intentionally focused. We are NOT claiming:
- Theoretical coverage guarantees (we show empirical coverage)
- Universal applicability (we focus on factor returns)
- Optimal methodology (we show improvement over baseline)

We ARE claiming:
- Standard CP under-covers during high crowding (empirical fact)
- CW-ACI improves coverage (empirical fact)
- The effect is consistent across factors (empirical fact)

This is an honest, empirical contribution suitable for JoFE.
