# Pivot Rationale: From JMLR Unified Framework to JoFE CW-ACI Paper

**Date:** December 22, 2024
**Decision:** Abandon unified framework, focus on CW-ACI for JoFE

---

## Executive Summary

After rigorous empirical testing on December 22, 2024, we discovered that 3 of 4 major claims in the original paper are **not supported by data**. Only CW-ACI survives scrutiny. We are pivoting to a focused paper on CW-ACI for the Journal of Financial Econometrics.

---

## Original Paper: What We Claimed

**Title:** "Not All Factors Crowd Equally: A Game-Theoretic Model of Alpha Decay with Global Transfer and Risk Management"

**Four Contributions:**
1. Game-theoretic derivation of hyperbolic factor decay
2. 45-63% out-of-sample R² for decay prediction
3. MMD-based global transfer (60% efficiency)
4. CW-ACI with coverage guarantees

---

## Empirical Tests Conducted (December 22, 2024)

### Test 1: R² Audit
**Script:** `scripts/audit_r2_claims.py`
**Data:** 748 months of Fama-French factors (1963-2025)

| Factor | Paper Claims | Reality |
|--------|--------------|---------|
| SMB | 54% OOS R² | **-4.9%** |
| HML | 58% OOS R² | **-235%** |
| Mom | 61% OOS R² | **-224%** |

**Verdict:** R² claims are **FALSE**. Model predicts worse than naive mean.

### Test 2: Momentum Control
**Script:** `scripts/momentum_control_test.py`

| Factor | Crowding Survives Momentum Control? |
|--------|-------------------------------------|
| SMB | ❌ NO (p=0.93) |
| HML | ❌ NO (p=0.23) |
| RMW | ❌ NO (p=0.08) |
| CMA | ✅ YES (p=0.01) |
| Mom | ✅ YES (p=0.04) |

**Verdict:** Crowding effect only works for **2 of 5 factors**.

### Test 3: CW-ACI
**Script:** `scripts/test_cwaci.py`

| Factor | Adapts to Crowding? | Improves Coverage? |
|--------|---------------------|-------------------|
| SMB | ✅ YES (1.22x) | ⚠️ Over-covers |
| HML | ✅ YES (1.25x) | ✅ YES |
| Mom | ✅ YES (1.24x) | ✅ YES |
| CMA | ✅ YES (1.21x) | ⚠️ Over-covers |
| RMW | ✅ YES (1.19x) | ✅ YES |

**Key Finding:** Standard CP under-covers during high-crowding (67-77%). CW-ACI achieves 83-95%.

**Verdict:** CW-ACI **WORKS**. This is a real contribution.

### Test 4: MMD Transfer
**Script:** `scripts/test_mmd_transfer.py`

| Metric | Paper Claims | Reality |
|--------|--------------|---------|
| Transfer efficiency | 60% | **-318%** |
| Cases improved | Most | **7/16 (44%)** |

**Verdict:** MMD transfer **DOES NOT WORK** (except for Momentum factor).

---

## Summary: What Survives

| Claim | Status | Action |
|-------|--------|--------|
| Game-theoretic derivation | ☠️ DEAD | Circular reasoning, remove entirely |
| 45-63% OOS R² | ☠️ DEAD | False, remove entirely |
| Universal crowding effect | ☠️ DEAD | Only 2/5 factors, reduce scope |
| MMD global transfer | ☠️ DEAD | Doesn't work, remove entirely |
| **CW-ACI** | ✅ **ALIVE** | Works, make this the focus |

---

## Why We're Pivoting

### Reason 1: Intellectual Honesty
We cannot publish claims that are demonstrably false. The R² claims, in particular, are indefensible. Publishing them would damage credibility.

### Reason 2: Something Real Exists
CW-ACI actually works. Standard conformal prediction under-covers during high-crowding periods. CW-ACI fixes this. This is a genuine contribution worth publishing.

### Reason 3: Better Venue Fit
The surviving contribution (CW-ACI for factor returns) is:
- An empirical finding (not theoretical)
- Finance-specific (not general ML)
- Practically useful (not just academic)

This fits JoFE better than JMLR.

### Reason 4: Achievable Timeline
The JoFE Special Issue deadline (March 2026) is achievable with a focused paper. The original unified framework would require years to fix properly.

---

## What Changes

### OLD Paper Structure
```
1. Introduction
2. Related Work
3. Background
4. Game-Theoretic Model ← REMOVE
5. US Empirical Validation ← REWRITE
6. Domain Adaptation (MMD) ← REMOVE
7. CW-ACI ← KEEP & EXPAND
8. Robustness
9. Conclusion
```

### NEW Paper Structure
```
1. Introduction (CW-ACI motivation)
2. Related Work (conformal prediction in finance)
3. Methodology
   3.1 Standard Conformal Prediction
   3.2 Crowding-Weighted ACI
   3.3 Coverage Properties
4. Monte Carlo Validation (NEW)
5. Empirical Analysis
   5.1 Data and Setup
   5.2 Coverage During High/Low Crowding
   5.3 Adaptive Interval Width
6. Robustness
7. Conclusion
```

### OLD Title
"Not All Factors Crowd Equally: A Game-Theoretic Model of Alpha Decay with Global Transfer and Risk Management"

### NEW Title
"Crowding-Aware Conformal Prediction for Factor Return Uncertainty"

---

## What We Learned

1. **Test before you write.** The original paper made claims without rigorous out-of-sample testing.

2. **Simpler is better.** One contribution that works beats four that don't.

3. **Be honest about what R² means.** Model fit ≠ predictive power.

4. **Crowding proxies are tricky.** Return-based proxies may just capture momentum.

5. **Conformal prediction is robust.** CW-ACI works even when everything else fails.

---

## Files Being Archived

The following files document the original approach and tonight's analysis:

```
research/jmlr_unified/
├── jmlr_submission/          # Original paper (archived)
├── REVISION_PLAN.md          # Original revision plan
├── VENUE_ANALYSIS.md         # QF vs JoFE analysis
├── MASTER_REVISION_DOCUMENT.md
├── JOFE_TRANSFORMATION_GUIDE.md
├── TONIGHTS_FINDINGS.md      # Critical discoveries
├── FINAL_VENUE_DECISION.md   # JoFE decision
├── PIVOT_RATIONALE.md        # This document
├── scripts/                  # Test scripts
└── results/                  # Test results
```

---

## New Directory

All new work will be in:
```
research/jofe_cwaci/
```

This is a clean start with only what works.

---

## Commitment

We commit to:
1. Only claiming what we can demonstrate
2. Reporting honest R² and coverage numbers
3. Acknowledging limitations prominently
4. Making code and data available for replication

---

**Signed off:** December 22, 2024

The pivot is necessary. The path forward is clear. Let's build something we can be proud of.
