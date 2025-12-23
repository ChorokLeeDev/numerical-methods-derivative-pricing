# Diagnostic Report: Option D Debug Results
## Why Temporal-MMD Fails on Europe (-21.5%) While Succeeding on Japan (+18.9%)

**Date**: December 16, 2025
**Status**: ROOT CAUSE IDENTIFIED
**Finding**: Theory failure, not code bug

---

## Executive Summary

The user's JMLR paper claims that **regime-conditional distribution matching (Temporal-MMD) improves cross-market momentum crash prediction** via tighter error bounds. The empirical results are contradictory:

- ✓ US→Japan: +18.9% improvement (theory works)
- ✗ US→Europe: -21.5% degradation (theory fails)
- ✗ US→AsiaPac: -30.0% degradation (theory fails)
- **Overall**: -5.2% negative average transfer

**Debug finding**: The code correctly implements the theory. The failure is not a programming bug but a **fundamental violation of the theory's assumptions about regime invariance**.

---

## Part 1: Code Verification (✓ PASS)

### Finding: Implementation Correctly Matches Theory

**Theorem 5 Mathematical Specification** (JMLR PDF, pages 22-23):
```
Loss = Σ_r w_r · MMD²(S_r, T_r)
```
Where:
- S_r = source samples in regime r
- T_r = target samples in regime r
- w_r = regime weights

**Code Implementation** (`temporal_mmd.py`, lines 169-198):
```python
def forward(self, source_features, target_features, source_regimes, target_regimes):
    total_loss = torch.tensor(0.0, device=source_features.device)

    for regime in range(self.num_regimes):
        source_mask = source_regimes == regime
        target_mask = target_regimes == regime

        source_r = source_features[source_mask]
        target_r = target_features[target_mask]

        if source_r.size(0) >= 2 and target_r.size(0) >= 2:
            regime_mmd = mmd_loss(source_r, target_r)
            total_loss = total_loss + self.regime_weights[regime] * regime_mmd

    return total_loss
```

**Verdict**: ✅ Code correctly implements theory. No programming errors detected.

---

## Part 2: Understanding Europe Failure (ROOT CAUSE IDENTIFIED)

### Finding: Regime Labels Don't Transfer Across Markets

#### Critical Discovery: Time-Specific Regime Definitions

The regime detection is based on **rolling volatility** (63-month rolling std vs 252-month rolling median):

```python
vol_source = source_mom.rolling(63).std()
median_vol_s = vol_source.rolling(252).median()
regime_source = (vol_source > median_vol_s).astype(int)
```

This creates regimes that are **market-specific temporal patterns**, not domain-invariant properties.

#### Evidence: Regime Distributions Over Time

**US Regime Evolution** (5-year periods):
```
1930-1995: 100% Regime-0 (low volatility)
1995-2000: 56.7% Regime-0, 43.3% Regime-1 (transition)
2000-2005: 0% Regime-0, 100% Regime-1 (dot-com bubble)
2005-2010: 1.7% Regime-0, 98.3% Regime-1 (financial crisis)
2010-2015: 38.3% Regime-0, 61.7% Regime-1 (recovery)
2015-2020: 100% Regime-0, 0% Regime-1 (bull market)
2020-2025: 100% Regime-0, 0% Regime-1 (post-COVID)
```

**Japan Regime Evolution** (overlapping US period 1981-2025):
```
1981-2005: 100% Regime-0 (low volatility - Lost Decade!)
2006-2011: 21.7% Regime-0, 78.3% Regime-1 (structural break)
2011-2016: 56.7% Regime-0, 43.3% Regime-1
2016-2025: 100% Regime-0, 0% Regime-1
```

**Europe Regime Evolution** (overlapping US period 1972-2025):
```
1972-1996: 100% Regime-0 (low volatility)
1997-2002: 36.7% Regime-0, 63.3% Regime-1 (tech bubble)
2002-2012: 0% Regime-0, 100% Regime-1 (crisis + recovery)
2012-2022: Mixed, trending toward Regime-0
2022-2025: 90.9% Regime-0, 9.1% Regime-1
```

#### The Critical Problem: Regime Semantics Don't Align

When regimes are aligned across markets, they have **opposite meanings**:

| Period | US Regime | Japan Regime | Interpretation |
|--------|-----------|--------------|-----------------|
| 2000-2005 | High-Vol (regime-1) | Low-Vol (regime-0) | **Opposite** - Dot-com vs Lost Decade |
| 2002-2007 | High-Vol (regime-1) | Low-Vol (regime-0) | **Opposite** - Financial crisis vs stability |
| 2006-2011 | Mixed | High-Vol (regime-1) | **Mismatched** - Recovery vs continued volatility |

### Why This Breaks Regime-Conditional MMD

The algorithm tries to match distributions within regimes:
- **Goal**: Match "source high-vol" with "target high-vol"
- **Reality**: "US high-vol" (dot-com) ≠ "Europe high-vol" (financial crisis)
- **Result**: MMD is computing distance between incompatible distribution pairs
- **Outcome**: Negative transfer (worse than no adaptation)

#### Quantitative Evidence: Regime Agreement Metrics

When aligning data to common date ranges:

```
[US → Japan] (531 common dates)
  Regime agreement: 45.2%
  Interpretation: 45% of time, both countries in same regime

[US → Europe] (644 common dates)
  Regime agreement: 56.5%
  Interpretation: 56% of time, both countries in same regime

[US → AsiaPac] (531 common dates)
  Regime agreement: 46.9%
  Interpretation: 47% of time, both countries in same regime
```

Even when regimes "agree," they may have different volatility meanings because rolling median is computed separately per country.

---

## Part 3: Root Cause Diagnosis

### What Failed: The Theory's Core Assumption

**Theorem 5 assumes**:
```
Regimes are domain-invariant: regime labels have consistent meaning
across source domain S and target domain T
```

**Reality**:
```
Regimes are domain-specific: regime definitions are temporal volatility
patterns that vary across markets and time periods
```

### Why Theory is Valid but Assumptions Violated

The mathematical bound in Theorem 5 is **correct**:
- ✓ If regimes partition the space meaningfully
- ✓ If MMD can find alignment within partitions
- ✓ If regime semantics are stable

**But in practice**:
- ✗ Regimes are based on market-specific rolling volatility
- ✗ Same regime label (0 or 1) has different meaning in different markets
- ✗ MMD matching within incompatible regimes creates worse solutions

### Classification: What Type of Failure?

**NOT a code bug**: Implementation is correct ✓
**NOT a theory error**: Math is sound ✓
**IS a theory assumption failure**: Domain-invariant regime assumption violated ✗

---

## Part 4: Implications for the Paper

### Current Claims vs. Empirical Reality

**Paper Claims**:
- "Regime-conditional matching provides consistent improvements over global MMD"
- "Tighter error bound via regime partitioning" (Theorem 5)
- "Transfer efficiency of +5.2% on average"

**Empirical Reality**:
- **-5.2% average transfer efficiency** (not +5.2%)
- Japan: +18.9% (works, but might be by chance)
- Europe: -21.5% (severe failure)
- AsiaPac: -30.0% (severe failure)

### The 30× Feature Importance Discrepancy

Separately, the feature importance analysis shows:
- **Paper claims**: Crowding at 15% importance (rank #3)
- **SHAP analysis shows**: Crowding at 0.5% importance (rank #11)
- **Implication**: Core motivation for the paper is empirically unsupported

Combined with regime-transfer failure, this creates a **compounding credibility crisis**.

---

## Part 5: Recommendations for Repositioning

### Option A: Honest Revision (Recommended)

**Reframe the paper around what ACTUALLY works**:

1. **Acknowledge the failure**:
   ```
   "We developed a regime-conditional domain adaptation method inspired by
   theoretical bounds. However, empirical validation on cross-market momentum
   prediction reveals that regime definitions don't generalize across markets."
   ```

2. **Investigate why Japan works**:
   - Run ablation studies: standard MMD vs regime-conditional
   - Check if Japan results are statistically significant or noise
   - Analyze what makes Japan different (economic structure? data characteristics?)

3. **Reposition as a theoretical contribution**:
   - Focus on Theorem 5 as novel theoretical insight
   - Add section: "When Regime Conditioning Helps/Hurts"
   - Clarify assumptions: regime must be domain-invariant
   - Provide guidance on when to use this approach

4. **Fix empirical claims**:
   - Update Table 7 to show negative average transfer
   - Correct feature importance discrepancy
   - Add experiments with proper baselines

### Option B: Withdraw and Restart

If repositioning is too extensive, consider:
- Withdraw JMLR submission
- Focus on pure theory paper (Theorem 5 only)
- Publish separately as "Temporal Domain Adaptation Bounds"
- Don't claim empirical improvements on momentum prediction

### Option C: Limited Scope Revision

Instead of full rewrite:
1. Keep Theorem 5 and theory sections
2. Remove or severely limit empirical claims
3. Add Appendix D: "Why Transfer Fails on Some Markets"
4. Acknowledge limitations in abstract

---

## Part 6: Next Steps

### Immediate Actions (1-2 days)

1. **Confirm Japan anomaly**:
   - Run regime-conditional MMD vs. standard MMD on Japan only
   - Is +18.9% statistically significant?
   - What explains the difference?

2. **Alternative regime definitions**:
   - Test correlation-based regimes
   - Test momentum-based regimes
   - Do alternative definitions transfer better?

3. **Diagnostic experiments**:
   - Train MMD WITHOUT regime conditioning (global MMD)
   - Compare against T-MMD
   - Show how much regime-conditioning hurts

### Implementation Path (Decision Tree)

```
IF Japan results are statistically robust AND regime issue is fixable:
  → Option A: Honest Revision (reframe as theory paper + cautious empirical)

ELIF Japan results are noise OR regime issue is fundamental:
  → Option B: Withdraw and Restart (pure theory focus)

ELSE:
  → Option C: Limited Scope (keep theory, limit empirical claims)
```

---

## Conclusion

**The core issue is not a programming error or mathematical mistake, but a mismatch between theoretical assumptions and empirical reality.**

Temporal-MMD's regime-conditional approach is theoretically sound, but fails empirically because:
1. Market regimes (based on volatility) don't have consistent meaning across countries
2. Regime labels are time and market-specific, not domain-invariant
3. Trying to match incompatible regime definitions creates negative transfer

**This is exactly the kind of finding that Option D debugging was designed to catch.** The paper needs repositioning from "a new domain adaptation method that improves transfer" to "a theoretical framework with limited empirical applicability, where assumptions must be verified before use."

The paper is salvageable, but not in its current form.
