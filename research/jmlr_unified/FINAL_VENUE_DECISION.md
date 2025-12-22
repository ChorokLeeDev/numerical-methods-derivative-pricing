# Final Venue Decision: JoFE vs JMLR

**Date:** December 22, 2024
**Based on:** Tonight's empirical tests

---

## What Actually Works

| Component | Status | Evidence |
|-----------|--------|----------|
| CW-ACI | ✅ Works | 5/5 factors adapt, 3/5 improve coverage |
| Crowding for CMA/Mom | ✅ Works | p<0.05 after momentum control |
| Game theory | ❌ Dead | Circular reasoning |
| R² predictions | ❌ Dead | Negative OOS R² |
| MMD transfer | ❌ Dead | -318% transfer efficiency |

---

## Venue Comparison

### JMLR (Journal of Machine Learning Research)

**Impact Factor:** ~6.0
**Acceptance Rate:** ~10%
**Review Time:** 3-6 months

**What JMLR wants:**
- Novel ML methodology with theoretical guarantees
- Broad applicability beyond one domain
- Rigorous proofs
- Comparison to state-of-the-art

**Does your paper fit?**

| Requirement | Your Paper | Fit |
|-------------|------------|-----|
| Novel methodology | CW-ACI | ⚠️ Incremental |
| Theoretical guarantee | Empirical only | ❌ Weak |
| Broad applicability | Finance-specific | ❌ No |
| SOTA comparison | Basic | ⚠️ Partial |

**JMLR Assessment:**
- CW-ACI is a reasonable idea but **not novel enough** for JMLR
- Weighted conformal prediction exists (Romano et al., 2019)
- You're applying existing techniques, not inventing new ones
- **Probability of acceptance: 10-15%**

---

### JoFE (Journal of Financial Econometrics)

**Impact Factor:** 2.42
**Acceptance Rate:** ~15-20%
**Review Time:** 2-4 months
**Special Issue Deadline:** March 1, 2026

**What JoFE wants:**
- Methods applicable to finance
- Empirical validation on real data
- Economic interpretation
- Rigorous but applied

**Does your paper fit?**

| Requirement | Your Paper | Fit |
|-------------|------------|-----|
| Finance application | Factor returns | ✅ Yes |
| Empirical validation | 748 months FF data | ✅ Yes |
| Economic interpretation | Crowding → uncertainty | ✅ Yes |
| Methodological rigor | Need Monte Carlo | ⚠️ Doable |

**JoFE Assessment:**
- CW-ACI for factor return uncertainty is **novel in finance**
- Special issue on ML in Finance is **perfect fit**
- Empirical contribution valued
- **Probability of acceptance: 25-35%**

---

## My Recommendation: **JoFE**

### Why JoFE wins:

1. **Your contribution is empirical, not theoretical**
   - CW-ACI improves coverage in practice
   - You can't prove theoretical guarantees (C ⊥ y|x is questionable)
   - JoFE accepts empirical contributions; JMLR does not

2. **Special issue timing is perfect**
   - "Machine Learning in Financial Econometrics"
   - Deadline: March 1, 2026 (14 months)
   - Guest editors will appreciate finance + ML

3. **The novelty is in the application**
   - CW-ACI for factor return uncertainty is new
   - Showing standard CP under-covers during high crowding is new
   - This matters to JoFE readers, not JMLR readers

4. **Achievable scope**
   - JoFE needs Monte Carlo (doable)
   - JMLR needs theoretical proof (probably can't do)

---

## What JoFE Paper Looks Like

**Title:** "Crowding-Aware Conformal Prediction for Factor Return Uncertainty"

**Abstract:**
> Standard conformal prediction provides distribution-free coverage guarantees for prediction intervals. However, when applied to factor returns, we find empirical coverage drops to 67-77% during high-crowding periods, well below the nominal 90% target. We introduce Crowding-Weighted Adaptive Conformal Inference (CW-ACI), which produces wider prediction intervals when crowding signals are elevated. Using 62 years of Fama-French data, we show CW-ACI achieves 83-95% coverage during high-crowding periods while maintaining nominal coverage overall. The method adapts interval width by 20-25% based on crowding levels. Our results suggest that uncertainty quantification in factor investing should account for crowding dynamics.

**Contributions:**
1. Document under-coverage of standard CP during high crowding (empirical finding)
2. Propose CW-ACI that adapts to crowding signals (method)
3. Show CW-ACI improves coverage from ~80% to ~90% (validation)

**What's removed:**
- Game theory
- R² predictions
- MMD transfer
- Universal factor claims

---

## Revised Timeline for JoFE

### Phase 1: Rebuild Paper (Jan-Apr 2025)
- Rewrite around CW-ACI as main contribution
- Add Monte Carlo section (simulate data, verify coverage)
- Remove game theory framing entirely
- Focus empirical section on coverage analysis

### Phase 2: Strengthen Theory (May-Jul 2025)
- Formalize when CW-ACI works (assumptions)
- Add theoretical discussion (not proof) of coverage
- Compare to other adaptive conformal methods

### Phase 3: Polish (Aug-Dec 2025)
- Add robustness checks
- Prepare replication package
- Get colleague feedback

### Phase 4: Submit (Jan-Feb 2026)
- Format per JoFE guidelines
- Submit to special issue
- Buffer before March 1 deadline

---

## Risk Assessment

| Venue | Probability | Time to Decision | Fallback |
|-------|-------------|------------------|----------|
| JoFE Special Issue | 25-35% | ~4 months | Regular JoFE |
| Regular JoFE | 20-25% | ~4 months | Quant Finance |
| JMLR | 10-15% | ~5 months | JoFE |

**Recommended path:**
1. Submit to JoFE Special Issue (March 2026)
2. If rejected, revise and submit to regular JoFE (July 2026)
3. If rejected again, submit to Quantitative Finance (Dec 2026)

---

## Final Answer

**Submit to: Journal of Financial Econometrics - Special Issue on Machine Learning**

**Why not JMLR:**
- Your contribution is empirical, not theoretical
- CW-ACI is not novel enough as a pure ML method
- Finance-specific application limits broad appeal

**Why JoFE:**
- CW-ACI for factor returns is novel in finance
- Empirical contribution valued
- Special issue timing perfect
- Achievable rigor requirements

---

## What To Do Next

1. **Tonight:** Commit all results ✅
2. **This week:** Write new abstract focusing on CW-ACI
3. **January:** Restructure paper around CW-ACI
4. **February-April:** Add Monte Carlo section
5. **May-July:** Strengthen theory section
6. **August-December:** Polish and robustness
7. **January 2026:** Final revisions
8. **February 2026:** Submit to JoFE

---

**Bottom line:** You have a real contribution (CW-ACI works), but it's a finance contribution, not an ML contribution. JoFE is the right home for this work.
