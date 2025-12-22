# JoFE Transformation Guide

**Purpose:** Understand what JoFE papers look like and what transformation is needed

---

## Part 1: JoFE Paper Characteristics

Based on analysis of recent JoFE publications (2023-2024), here's what characterizes a JoFE paper:

### Structure

| Section | Typical Content | Length |
|---------|-----------------|--------|
| Introduction | Problem motivation, contribution summary | 2-3 pages |
| Literature Review | Focused, technical | 1-2 pages |
| **Theoretical Framework** | **Formal assumptions, definitions, theorems** | **5-10 pages** |
| **Monte Carlo / Simulation** | **Extensive validation under controlled conditions** | **5-8 pages** |
| Empirical Application | Real data demonstration | 5-8 pages |
| Robustness | Sensitivity analyses | 3-5 pages |
| Conclusion | Brief | 1 page |
| **Appendix** | **Complete proofs, technical details** | **10-20 pages** |

### Key Characteristics

**1. Theoretical Rigor**
- Formal theorems with explicit assumptions
- Proofs (in appendix) or reference to established results
- Clear statement of what assumptions are needed and when they fail

**Example from Pesaran & Yamagata (2024):**
> "Theorem 1: Under Assumptions 1-4, as N,T → ∞ with N/T² → 0, the test statistic J^α → N(0,1)"

**2. Monte Carlo Validation**
- Extensive simulation studies
- Multiple sample size combinations
- Comparison to existing methods
- Report size AND power

**Example:**
> "Table 2: Size properties of J^α test. We consider 9 sample combinations (T ∈ {60,120,240}, N ∈ {50,100,200})"

**3. Statistical Tests**
- Diebold-Mariano tests for forecast comparison
- Model Confidence Sets (MCS)
- HAC standard errors
- Multiple testing corrections

**4. Economic Significance**
- Not just statistical significance
- Utility-based metrics
- Transaction cost considerations
- Sharpe ratio comparisons

---

## Part 2: Recent JoFE ML Papers Analysis

### Paper 1: Volatility Forecasting with ML (Zhang et al., 2024)

**What made it publishable:**
- Novel finding: "universal volatility mechanism" across stocks
- Rigorous comparison: 7 models, 4 horizons, 93 stocks
- Statistical tests: Diebold-Mariano, Model Confidence Sets
- Economic validation: utility-based metrics with transaction costs
- Out-of-sample: 6-year test period, unseen stocks

**Methodology highlights:**
```
Models: SARIMA, HAR-D, OLS, LASSO, XGBoost, MLP, LSTM
Training schemes: Single (asset-specific), Universal (pooled), Augmented (+ market)
Evaluation: Statistical (RMSE, MAE) + Economic (realized utility)
```

### Paper 2: Testing for Alpha (Pesaran & Yamagata, 2024)

**What made it publishable:**
- Addresses real problem: GRS test fails when N > T
- 5 formal theorems with explicit assumptions
- Proofs establish asymptotic distribution
- Monte Carlo: 9 sample combinations, 7 competing tests
- Extensions: non-Gaussian errors, weak factors

**Rigor markers:**
- "Theorem 1: Under Assumptions 1-4..."
- "Size close to 5% nominal level even when N=200 and T=60"
- Comparison to state-of-the-art alternatives

### Paper 3: Intraday Predictability (Huddleston et al., 2023)

**What made it publishable:**
- "Largest study ever of 5-min equity returns"
- Clear economic significance: "Sharpe ratios of 0.98 after transaction costs"
- Ensemble methods with proper validation
- Cross-sectional analysis

---

## Part 3: Gap Analysis - Your Paper vs JoFE Standard

### What Your Paper Has

| Element | Status | Notes |
|---------|--------|-------|
| Novel contribution | ✓ | Unified framework for crowding |
| Empirical validation | ✓ | 61 years of data |
| Economic significance | ✓ | Sharpe improvement, hedging |
| Clear writing | ✓ | Well-organized |
| ML methods | ✓ | MMD, conformal prediction |

### What Your Paper Lacks

| Element | Status | Gap |
|---------|--------|-----|
| **Formal theorems** | ⚠️ | "Theorems" are propositions without rigorous proofs |
| **Monte Carlo validation** | ❌ | No simulation study |
| **Proper statistical tests** | ⚠️ | Need HAC SEs, DM tests, MCS |
| **Assumption verification** | ⚠️ | C ⊥ y\|x not rigorously tested |
| **Comparison to benchmarks** | ⚠️ | Need comparison to existing methods |
| **Code/data replication** | ⚠️ | JoFE requires this |

---

## Part 4: Transformation Roadmap

### Critical Changes (Must Do)

**1. Add Monte Carlo Section**
```
Section X: Monte Carlo Validation
- Simulate data with known decay parameters
- Test if hyperbolic model recovers true parameters
- Compare to exponential, linear alternatives
- Report bias, RMSE, coverage across sample sizes
```

**2. Formalize Assumptions**
```
Assumption 1 (Equilibrium Existence): ...
Assumption 2 (Decay Regularity): K(t) is continuously differentiable...
Assumption 3 (Conditional Independence): C ⊥ y | x
Proposition 1: Under Assumptions 1-2, there exists unique equilibrium...
```

**3. Add Proper Statistical Tests**
```
- HAC standard errors (Newey-West) for all regressions
- Diebold-Mariano tests for model comparison
- Model Confidence Sets for model selection
- Bootstrap confidence intervals
```

**4. Strengthen CW-ACI Theory**
```
Theorem (Coverage): Under Assumption 3, CW-ACI satisfies:
P(y_{n+1} ∈ C(x_{n+1})) ≥ 1 - α - O(1/√n)
Proof: [Formal proof following Romano et al. 2019]
```

### Major Restructuring

**Current structure:**
```
1. Introduction
2. Related Work
3. Background
4. Game Theory (weak)
5. US Empirical
6. Domain Adaptation
7. Conformal Prediction
8. Robustness
9. Conclusion
```

**JoFE structure:**
```
1. Introduction
2. Literature Review (shorter)
3. Theoretical Framework
   3.1 Model Setup and Assumptions
   3.2 Equilibrium Model of Factor Decay
   3.3 CW-ACI: Theory and Coverage Guarantee
4. Monte Carlo Validation (NEW)
   4.1 Simulation Design
   4.2 Decay Parameter Recovery
   4.3 Coverage Properties
5. Empirical Analysis
   5.1 US Factor Data
   5.2 Global Transfer
   5.3 Portfolio Application
6. Robustness
7. Conclusion
Appendix A: Proofs
Appendix B: Monte Carlo Details
Appendix C: Data Description
```

---

## Part 5: Specific JoFE Requirements

### From JoFE Guidelines

> "For empirical, experimental, and numerical papers, authors will be required to upload source files and relevant data files for editorial replicability review as a condition of acceptance."

**You need:**
- [ ] Clean Python code with requirements.txt
- [ ] Data download scripts (FF library is public)
- [ ] Jupyter notebooks reproducing all tables/figures
- [ ] README with execution instructions

### Statistical Standards

**From recent papers, JoFE expects:**

1. **Standard Errors:**
   - HAC (Newey-West) for time series
   - Clustered if panel data
   - Bootstrap for non-standard estimators

2. **Model Comparison:**
   - Diebold-Mariano tests
   - Encompassing tests
   - Model Confidence Sets (Hansen et al.)

3. **Significance:**
   - Report t-stats AND p-values
   - Adjust for multiple testing where appropriate
   - Economic significance alongside statistical

### Writing Style

**JoFE is more formal than JMLR:**

| JMLR style | JoFE style |
|------------|------------|
| "We show that..." | "Theorem 1 establishes that..." |
| "Our method works" | "Under Assumptions 1-3, the test has size α" |
| Informal proofs OK | Formal proofs required (appendix) |
| Applied focus | Methodological focus |

---

## Part 6: Example Transformations

### Before (Current Paper)

```latex
\begin{theorem}[Existence and Uniqueness of Equilibrium]
Consider the crowding game with investor payoff function...
There exists a unique equilibrium crowding path C*(t).
\end{theorem}

\textbf{Proof Sketch}: Define equilibrium condition...
```

### After (JoFE Standard)

```latex
\begin{assumption}[Regularity Conditions]
\label{ass:regularity}
(i) The intrinsic alpha process $K(t)$ is continuously differentiable
    with $K(t) > r_f$ for all $t \geq 0$.
(ii) Transaction costs satisfy $TC'(C) > 0$ for all $C > 0$.
(iii) Capital adjustment parameter $\kappa > 0$.
\end{assumption}

\begin{proposition}[Equilibrium Characterization]
\label{prop:equilibrium}
Under Assumption \ref{ass:regularity}, there exists a unique
equilibrium crowding path $C^*(t)$ satisfying:
\begin{equation}
C^*(t) = \frac{K(t) - r_f}{\lambda_0 + TC'(C^*(t))}
\end{equation}
Moreover, $C^*(t)$ is monotonically decreasing in $t$.
\end{proposition}

\noindent\textit{Proof.} See Appendix A. $\square$
```

### Before (R² Claims)

```latex
Out-of-sample R² reaches 55\%.
```

### After (JoFE Standard)

```latex
Table \ref{tab:forecast_comparison} reports out-of-sample performance.
The hyperbolic model achieves an $R^2_{OOS}$ of 0.12, compared to
0.08 for the exponential model and 0.03 for the random walk benchmark.
Diebold-Mariano tests confirm the hyperbolic model significantly
outperforms alternatives at the 5\% level (Table \ref{tab:dm_tests}).
Model Confidence Sets include only the hyperbolic specification
in 8 of 10 rolling windows.
```

---

## Part 7: Monte Carlo Section Template

**Add this section (5-8 pages):**

```latex
\section{Monte Carlo Validation}

We validate our econometric methodology through simulation.

\subsection{Design}

We generate synthetic factor returns according to:
\begin{equation}
r_{i,t} = \alpha_i(t) + \beta_i F_t + \epsilon_{i,t}
\end{equation}
where $\alpha_i(t) = K_i / (1 + \lambda_i t)$ follows the hypothesized
hyperbolic decay with known parameters $(K_i, \lambda_i)$.

We consider:
\begin{itemize}
\item Sample sizes: $T \in \{120, 240, 480, 720\}$ months
\item True decay rates: $\lambda \in \{0.05, 0.10, 0.20\}$
\item Error structures: i.i.d., AR(1), GARCH(1,1)
\item 1,000 replications per configuration
\end{itemize}

\subsection{Results}

Table \ref{tab:mc_results} reports bias and RMSE for $\hat{\lambda}$.
[Results showing estimator performs well across configurations]

\subsection{Coverage Properties}

For CW-ACI, we verify the 90\% coverage guarantee holds:
[Table showing empirical coverage across configurations]
```

---

## Part 8: Timeline for Transformation

### Phase 1 (Jan-Feb 2025): Foundation
- [ ] Add Monte Carlo section skeleton
- [ ] Reformulate theorems as propositions with assumptions
- [ ] Add HAC standard errors to all regressions

### Phase 2 (Mar-Apr 2025): Monte Carlo
- [ ] Implement simulation framework
- [ ] Run Monte Carlo experiments
- [ ] Document results

### Phase 3 (May-Jun 2025): Statistical Tests
- [ ] Add Diebold-Mariano tests
- [ ] Add Model Confidence Sets
- [ ] Add bootstrap CIs

### Phase 4 (Jul-Aug 2025): CW-ACI Theory
- [ ] Formalize coverage theorem
- [ ] Write rigorous proof
- [ ] Verify assumptions empirically

### Phase 5 (Sep-Oct 2025): Replication Package
- [ ] Clean code
- [ ] Write documentation
- [ ] Test reproducibility

### Phase 6 (Nov-Jan 2026): Polish
- [ ] Internal review
- [ ] Colleague feedback
- [ ] Final revisions

### Phase 7 (Feb 2026): Submit
- [ ] Format per OUP template
- [ ] Write cover letter
- [ ] Submit by Feb 15

---

## Part 9: Key References to Cite

### JoFE Papers to Reference

1. **Zhang et al. (2024)** - Volatility Forecasting with ML
   - Cite for ML methodology in financial econometrics

2. **Pesaran & Yamagata (2024)** - Testing for Alpha
   - Cite for alpha testing methodology

3. **Christensen et al. (2023)** - ML Approach to Volatility
   - Cite for ML + HAR model comparison

### Methodological References

4. **Diebold & Mariano (1995)** - Forecast comparison tests
5. **Hansen et al. (2011)** - Model Confidence Sets
6. **Newey & West (1987)** - HAC standard errors
7. **Romano et al. (2019)** - Conformal prediction
8. **Angelopoulos & Bates (2021)** - Conformal prediction tutorial
9. **Long et al. (2015)** - MMD domain adaptation

---

## Summary: What Makes a JoFE Paper

| Criterion | Your Paper Now | JoFE Standard | Gap |
|-----------|----------------|---------------|-----|
| **Theoretical rigor** | Informal | Formal theorems + proofs | Large |
| **Monte Carlo** | None | Extensive | Large |
| **Statistical tests** | Basic | HAC, DM, MCS | Medium |
| **Replication** | Partial | Complete package | Medium |
| **Writing style** | JMLR-ish | Econometric formal | Medium |

**Bottom line:** The gap is significant but closeable in 14 months. Focus on Monte Carlo and statistical rigor first—these are non-negotiable for JoFE.

---

## Appendix: JoFE Editorial Board

**Editors-in-Chief:**
- Eric Ghysels (UNC Chapel Hill) - Time series, volatility
- Andrew Patton (Duke) - Copulas, forecast evaluation

**Special Issue Editors (ML):**
- Wolfgang Karl Härdle (Humboldt) - Quantitative finance
- HONG Yongmiao (Chinese Academy) - Econometrics
- SHA Yezhou (Capital U) - Machine learning

**What they care about:**
- Methodological innovation
- Rigorous statistical treatment
- Economic relevance
- Reproducibility

---

**End of JoFE Transformation Guide**
