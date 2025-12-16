# Paper Ecosystem Clarification
## Three Completely Independent Paper Systems

**Date**: December 16, 2025
**Status**: Final clarification of all research papers across repositories

---

## 🗂️ Repository Structure

### Repository 1: `/quant/` (Main research - your current focus)
**Location**: `/Users/i767700/Github/quant/`
**3-paper system** (all interconnected via Temporal-MMD)

### Repository 2: `/ai-in-finance/` (Separate research)
**Location**: `/Users/i767700/Github/ai-in-finance/`
**Independent paper** (no connection to quant)

---

## 📋 Paper 1 System: QUANT Repository

### A. JMLR Paper: "Not All Factors Crowd Equally: Unified Framework"
**Status**: 🟡 Active (blocked by Temporal-MMD issues)
**Submission target**: JMLR (no deadline)
**Location**: `/quant/research/jmlr_unified/` (primary) + `/quant/research/factor-crowding-unified/` (duplicate?)

**3 Integrated Components**:
1. **Game-Theoretic Model** (Theorem 1)
   - Nash equilibrium → alpha decay formula: α(t) = K/(1+λt)
   - Novel theoretical contribution
   - ✅ Status: Complete and empirically supported

2. **Temporal-MMD (Regime-Conditional Domain Adaptation)** (Theorem 5)
   - Loss: Σ_r w_r · MMD²(S_r, T_r)
   - ❌ Status: Fails empirically (Europe -21.5%, Japan +18.9%, avg -5.2%)
   - ❌ Issue: Regimes are domain-specific, not domain-invariant
   - 🔴 **ACTION NEEDED**: Option A (replace with Standard MMD) vs Option B (remove)

3. **Conformal Prediction** (uncertainty quantification)
   - Distribution-free prediction intervals
   - Coverage guarantees
   - ✅ Status: Complete and working

**Affected by Temporal-MMD elimination?** ✅ YES - Options planned

---

### B. KDD 2026 Paper: "Mining Factor Crowding at Global Scale"
**Status**: 🔴 Debug complete, awaiting decision
**Submission target**: KDD 2026, Jeju (Deadline: Feb 8, 2026)
**Location**: `/quant/research/kdd2026_global_crowding/`

**Scope**:
- 6 regions × 10+ factors
- ML detection: LSTM vs XGBoost vs baseline RF
- Table 7: Cross-region transfer using Temporal-MMD

**Affected by Temporal-MMD elimination?** ✅ YES
- Current Table 7: Europe -21.5%, average -5.2%
- With Standard MMD: Europe +6.3%, average +7.7%
- Impact: +13% improvement in average transfer

---

### C. ICML 2026 Paper: "Conformal Prediction for Factor Crowding"
**Status**: 🟢 Independent, progressing
**Submission target**: ICML 2026, Seoul (Deadline: Jan 28, 2026)
**Location**: `/quant/research/icml2026_conformal/`

**Scope**:
- Distribution-free uncertainty quantification
- Coverage guarantees for crowding detection
- Comparison: Bayesian vs Bootstrap vs Conformal

**Affected by Temporal-MMD elimination?** ❌ NO - completely independent

---

## 🆕 Paper 2 System: AI-IN-FINANCE Repository

### "Causal Structure Changes Across Market Regimes: Evidence from Factor Returns"
**Status**: 🟢 Active, standalone
**Repository**: `/ai-in-finance/`
**Location**: `/ai-in-finance/chorok/v11_causal_factor_crowding/paper/arxiv/main.tex`
**Author**: Chorok Lee
**Date**: December 2025

**Scope**:
- Granger causality analysis between factors
- Student-t Hidden Markov Model for regime detection
- Focus: Does causal direction between factors change by regime?

**Key Findings**:
- Value → Size during crisis regimes (9-day lag, p=1.89e-5)
- Size → Value during crowding regimes (3-day lag, p=1.94e-4)
- No causal link during normal regimes
- Practical: Early warning 2 months before Lehman Brothers collapse

**Methods Used**:
- ✅ Granger causality (classical econometrics)
- ✅ Student-t HMM (regime detection)
- ❌ NO Temporal-MMD
- ❌ NO domain adaptation
- ❌ NO conformal prediction

**Affected by Temporal-MMD elimination?** ❌ NO - completely independent project

**Overlap with JMLR/KDD/ICML?**
- **Common topic**: Factor crowding regimes
- **Different angle**: Causal direction vs detection/adaptation/uncertainty
- **Verdict**: Complementary, not competing

---

## 📊 Comparison Matrix

| Aspect | JMLR | KDD | ICML | Causal |
|--------|------|-----|------|--------|
| **Repository** | quant | quant | quant | ai-in-finance |
| **Target venue** | JMLR | KDD 2026 | ICML 2026 | Not specified |
| **Deadline** | None | Feb 8 | Jan 28 | Unknown |
| **Main method** | Game theory + T-MMD + Conformal | ML detection + T-MMD | Conformal prediction | Granger + HMM |
| **Focus** | Why crowding happens | What crowding looks like | Uncertainty bounds | Which factor causes which |
| **T-MMD dependent?** | YES | YES | NO | NO |
| **Status** | Blocked | Debug done | OK | Independent |
| **Action** | Option A/B decision | Execute edits | Continue | None needed |

---

## 🎯 Impact Analysis

### Temporal-MMD Elimination Impact

**On JMLR paper**:
- ✅ Component 2 (domain adaptation) is simplified
- ✅ Or removed entirely (Option B)
- ✅ Components 1 & 3 unaffected

**On KDD paper**:
- ✅ Table 7 results improve significantly
- ✅ Section 4.2 method name/content changes
- ✅ Transfer results become more credible

**On ICML paper**:
- ✅ NO IMPACT (independent method)

**On Causal Structure paper**:
- ✅ NO IMPACT (different repository, different methods)

---

## ✅ Clarifications

### Question 1: "Is Causal Structure paper part of KDD?"
**Answer**: NO
- Different repository (ai-in-finance vs quant)
- Different methodology (Granger causality vs Temporal-MMD)
- Different focus (causal direction vs crowding detection)
- Separate authorship (Chorok Lee standalone vs team project)
- **Verdict**: Complementary research, not competing

### Question 2: "Does Temporal-MMD elimination affect Causal paper?"
**Answer**: NO
- Causal paper doesn't use Temporal-MMD at all
- Uses Student-t HMM instead (different method)
- Independently valuable regardless of T-MMD status

### Question 3: "Should we coordinate between projects?"
**Answer**: Not necessary for Temporal-MMD decision
- Each paper is independent
- Could be integrated into single unified submission later
- But don't block one on the other

### Question 4: "Why present Causal paper with quant projects?"
**Answer**:
- You confused it as part of KDD (reasonable, similar domain)
- Actually in separate ai-in-finance repo
- Not affected by your current elimination decision

---

## 🔴 Decision Points Still Open

### For JMLR Paper:
- [ ] Option A: Replace Temporal-MMD with Standard MMD
- [ ] Option B: Remove domain adaptation entirely
- [ ] Timeline: 1-2 weeks

### For KDD Paper:
- [ ] Confirm same option (A or B)
- [ ] Update Table 7 accordingly
- [ ] Timeline: 2-3 weeks before Feb 8 deadline

### For ICML Paper:
- [ ] Continue as planned (no changes needed)
- [ ] Timeline: Draft due soon (Jan 28 deadline)

### For ai-in-finance/Causal Paper:
- [ ] Submission target?
- [ ] Related venue to KDD/JMLR?
- [ ] Should be kept independent or integrated?

---

## 📝 Summary

**The user presented "Causal Structure Changes..." thinking it might be part of KDD/JMLR system.**

**Reality**:
- It's a completely separate paper in ai-in-finance repo
- Uses Granger causality + HMM (not Temporal-MMD)
- Focuses on causal direction (not crowding detection/transfer)
- No impact from Temporal-MMD elimination decision
- Could be valuable complementary paper, but independent

**Current action**:
- Proceed with Temporal-MMD elimination decision for quant papers
- Causal paper is unaffected and can continue independently
- Consider whether to integrate all three systems later (optional)

---

## Files for Reference

**In quant/research/**:
- PROJECT_OVERVIEW.md - Big picture
- PROJECT_DETAILS.md - Individual project analysis
- ELIMINATION_PLAN.md - Temporal-MMD removal options
- KDD_IMPACT_ANALYSIS.md - KDD paper specific
- LITERATURE_ANALYSIS.md - Novelty vs existing work

**In ai-in-finance/chorok/v11_causal_factor_crowding/**:
- paper/arxiv/main.tex - The Causal Structure paper (completely independent)

