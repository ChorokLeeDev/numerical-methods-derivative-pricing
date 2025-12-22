# Tonight's Findings: Reality Check

**Date:** December 22, 2024
**Status:** Critical issues discovered

---

## Executive Summary

Tonight's empirical tests reveal the paper's claims are **not supported by data**.

| Claim | Paper Says | Reality | Verdict |
|-------|------------|---------|---------|
| OOS R² | 45-63% | **NEGATIVE** | ❌ FALSE |
| Crowding predicts returns | All factors | **2 of 5 factors** | ⚠️ WEAK |
| Hyperbolic decay fits well | R²=60-70% | **R²=8-17%** | ❌ FALSE |

---

## Test 1: R² Audit Results

### What We Found

| Factor | Full Sample R² | Extrapolation R² | Predictive R² |
|--------|----------------|------------------|---------------|
| SMB | 11.7% | **-4.9%** | -10.0% |
| HML | 7.6% | **-235%** | -8.4% |
| RMW | ~0% | **-1.7%** | -3.1% |
| CMA | 4.2% | **-85%** | -3.8% |
| Mom | 16.9% | **-224%** | -11.4% |

### Interpretation

- **Full sample R² of 8-17%**: The hyperbolic model explains only 8-17% of Sharpe ratio variance, not 60-70%
- **Negative extrapolation R²**: The model predicts WORSE than just guessing the historical mean
- **Negative predictive R²**: Cannot predict next-month returns at all

### What Went Wrong?

The paper likely computed R² incorrectly. Possible issues:
1. Using cumulative returns instead of period returns
2. Computing in-sample fit and calling it "OOS"
3. Using wrong R² formula
4. Cherry-picking time periods

### Action Required

- [ ] Find where 45-63% R² came from in original code
- [ ] Either fix the methodology or remove the claims
- [ ] Be honest: model fit is ~10-15%, not 50-60%

---

## Test 2: Momentum Control Results

### What We Found

| Factor | Crowding-Only p | Controlled p | Survives? |
|--------|-----------------|--------------|-----------|
| SMB | 0.98 | 0.93 | ❌ NO |
| HML | 0.08 | 0.23 | ❌ NO |
| RMW | 0.07 | 0.08 | ❌ NO |
| CMA | 0.03 | **0.01** | ✅ YES |
| Mom | 0.04 | **0.04** | ✅ YES |

### Interpretation

- Only **CMA and Momentum** show crowding effects independent of momentum
- **SMB, HML, RMW** crowding effects may be spurious (just mean-reversion)
- The paper's claim that crowding predicts all factors is not supported

### Action Required

- [ ] Acknowledge that crowding effect is factor-specific
- [ ] Focus claims on CMA and Momentum only
- [ ] Or find better crowding proxy that works for all factors

---

## Test 3: Language Audit Results

### Summary

| Pattern | Count | Severity |
|---------|-------|----------|
| game-theoretic | 31 | HIGH - not actually game theory |
| guarantee | 36 | HIGH - many unverified |
| derive | 10 | HIGH - circular reasoning |
| prove | 9 | MEDIUM - some are citations |
| first to | 3 | MEDIUM - overstatement |

### Key Files to Fix

1. `sections/09_conclusion.tex` - 24 issues
2. `sections/02_related_work.tex` - 22 issues
3. `main.tex` - 14 issues
4. `sections/06_domain_adaptation.tex` - 13 issues

### Language Changes Needed

| Current | Replace With |
|---------|--------------|
| "derive hyperbolic decay" | "parameterize decay as hyperbolic" |
| "game-theoretic model" | "equilibrium model" |
| "prove coverage guarantee" | "show empirical coverage" |
| "first to derive" | "we contribute by modeling" |
| "mechanistic explanation" | "descriptive framework" |

---

## Implications for JoFE Submission

### Before Tonight

We thought we needed:
- Fix circular reasoning in theory
- Clarify R² terminology
- Add robustness

### After Tonight

We now know we need:
- **Complete empirical redo** - claims are false
- **Major scope reduction** - crowding works for only 2 factors
- **Honest reframing** - this is a descriptive model, not predictive
- **118 language edits** - every major claim needs revision

### Revised Assessment

| Aspect | Before | After | Change |
|--------|--------|-------|--------|
| Empirical validity | Assumed OK | NOT OK | Major redo |
| Theory validity | Weak | Weak | Same |
| Scope | 7 factors | 2 factors | Reduced |
| Language | Needs polish | Needs rewrite | Worse |
| Timeline to JoFE | 14 months | 14 months | Same but harder |

---

## What To Do Next

### Immediate (This Week)

1. [ ] Find the source of 45-63% R² claims in original code
2. [ ] Understand what was actually computed
3. [ ] Decide: fix methodology or change claims?

### Short-term (January 2025)

1. [ ] Rewrite empirical section with honest numbers
2. [ ] Focus on CMA and Momentum (where crowding works)
3. [ ] Remove claims about SMB, HML, RMW

### Medium-term (Feb-Mar 2025)

1. [ ] Find better crowding proxy (not return-based)
2. [ ] Re-test with improved proxy
3. [ ] If still doesn't work, pivot paper focus

### Long-term (Apr-Dec 2025)

1. [ ] Develop honest contribution
2. [ ] Add Monte Carlo (for whatever survives)
3. [ ] Prepare for JoFE submission

---

## The Honest Paper

**What the paper CAN claim (with tonight's data):**

1. Hyperbolic decay provides a reasonable parameterization (R² ~10-15%)
2. For CMA and Momentum factors, crowding signal has predictive power (p<0.05)
3. CW-ACI improves empirical coverage (need to verify this separately)
4. MMD enables global transfer (need to verify this separately)

**What the paper CANNOT claim:**

1. ❌ OOS R² of 45-63%
2. ❌ Crowding predicts all factors
3. ❌ "Derive" from game theory
4. ❌ "First" to do anything (need to verify)

---

## Files Created Tonight

| File | Purpose |
|------|---------|
| `results/r2_audit_results.csv` | R² audit data |
| `results/momentum_control_results.csv` | Momentum test data |
| `results/language_audit.txt` | All problematic language |
| `data/factor_crowding/ff_factors_monthly.parquet` | Real FF data |
| `data/factor_crowding/ff_factors_monthly.csv` | Real FF data (CSV) |

---

## Bottom Line

The paper needs more work than we thought. The good news:
- We now know the problems
- 14 months is enough time to fix them
- The core ideas may still be salvageable

The bad news:
- Empirical claims are false
- Theory is weaker than claimed
- Scope must be reduced

**Recommendation:** Take a few days to digest these findings. Then decide:
1. Fix the methodology and redo everything, or
2. Pivot to a different contribution entirely

Either way, the JoFE March 2026 deadline is still achievable, but the paper will look very different from its current form.
