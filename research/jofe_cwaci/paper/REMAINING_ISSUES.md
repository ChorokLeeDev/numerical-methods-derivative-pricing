# Remaining Issues for JoFE Submission

## Moderate Issues (Should Fix Before Submission)

### 1. Missing Literature References
- **Conformalized Quantile Regression (CQR)** by Romano et al. (2019) - The standard approach for heteroskedastic CP. Should explain why simple scaling is preferred over CQR.
- **Mondrian Conformal Prediction** - Handles covariate-conditional coverage
- **Adaptive Conformal Inference (ACI)** by Gibbs & Candès (2021) - Currently cited but not compared empirically

**Fix**: Add a paragraph in Related Work comparing to CQR and explaining the tradeoffs.

### 2. Weak Related Work Section
- Only 1 paragraph on "Conformal Prediction in Finance" with single citation (Fantazzini 2024)
- Missing applications: VaR estimation, portfolio optimization, option pricing

**Fix**: Expand with 3-4 more finance application citations.

### 3. Robustness Bound is Too Loose
- Corollary after Theorem 3: "10% estimation error costs ≤18pp coverage"
- But empirically, Vol-Scaled CP achieves 90%+ with noisy realized volatility
- The bound is too conservative to be useful

**Fix**: Add a sentence acknowledging the bound is conservative and note that empirical performance is much better. Could add empirical calibration of the bound.

## Minor Issues (Nice to Have)

### 4. Title Could Be Stronger
Current: "Volatility-Adaptive Conformal Prediction for Factor Return Uncertainty"
Alternative: "Distribution-Free Prediction Intervals for Factor Returns: Why Simple Volatility Scaling Beats GARCH"

### 5. References.bib
- Verify all citations compile correctly
- Add any missing references for expanded Related Work

## Summary of Changes Made (Dec 2024)

### Critical Issues Fixed:
1. ✅ Inconsistent numbers across tables - Added clarifying notes
2. ✅ Monte Carlo contradicted claims - Redesigned simulation
3. ✅ Locally-Weighted CP removed - Streamlined paper focus

### Sections Added:
- Section 8: Comparison with GARCH Prediction Intervals
- Section 9: Out-of-Sample Validation
- Appendix D: Validation of I.I.D. Assumption

### Paper Statistics:
- Pages: 27
- Tables: 7
- Figures: 5 (need to regenerate fig4 for new MC)
- Theorems: 3 with proofs
