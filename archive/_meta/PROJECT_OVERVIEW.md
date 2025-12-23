# Research Projects Overview
## Unified Research on Factor Crowding in Financial Markets

**Last Updated**: December 16, 2025
**Status**: Multiple projects in progress, need consolidation

---

## 🗂️ Project Portfolio

You have **5 research projects** in parallel:

### 1️⃣ **factor_crowding** (Legacy)
**Status**: ⛔ SUPERSEDED
**Path**: `/research/factor_crowding/`

**What it was**:
- Original exploratory project on factor crowding
- Basic crowding detection models
- US market analysis only

**Current role**: Historical reference only (no active work)

**To do**: Can be archived/removed once unified projects are finalized

---

### 2️⃣ **factor-crowding-unified**
**Status**: 🟡 ACTIVE (JMLR submission)
**Path**: `/research/factor-crowding-unified/`
**Target**: JMLR (Journal of Machine Learning Research)

**What it is**:
- Main unified paper combining 3 components:
  1. Game-theoretic model of alpha decay
  2. Regime-conditional domain adaptation (Temporal-MMD)
  3. Conformal prediction for risk management

**Current status**:
- Paper: `factor-crowding-unified/paper/` - WRITTEN (needs revision)
- Code: Fully implemented
- Issue: **Temporal-MMD empirical results problematic** (Europe -21.5% failure)

**What needs fixing**:
- ❌ Feature importance: Crowding claimed 15%, actually 0.5%
- ❌ Transfer efficiency: Claimed +5.2%, actually -5.2%
- ❌ Regime-conditional approach: Works on Japan, fails on Europe

**Next step**:
- **REPOSITIONING REQUIRED** (Option A recommended)
- Keep Theorem 5, fix empirical claims
- See: `kdd2026_global_crowding/experiments/FINAL_SUMMARY.md` for debug results

---

### 3️⃣ **icml2026_conformal**
**Status**: 🟢 ACTIVE (ICML 2026)
**Path**: `/research/icml2026_conformal/`
**Target**: ICML 2026 (Seoul, Korea) - Deadline Jan 28, 2026

**What it is**:
- Separate paper on Conformal Prediction for crowding
- Distribution-free uncertainty quantification
- Coverage guarantees for regime detection

**Current status**:
- Paper: Under development
- Code: Experimental
- Connection: Potentially could integrate with factor-crowding-unified if needed

**Note**: Independent project, not blocking other work

---

### 4️⃣ **jmlr_unified**
**Status**: 🟡 ACTIVE (JMLR submission)
**Path**: `/research/jmlr_unified/`
**Target**: JMLR (but seems to be duplicate/alternative to factor-crowding-unified)

**What it is**:
- Another version of unified framework paper
- Contains: game theory, domain adaptation, tail risk

**Confusion**: Appears to be **duplicate of factor-crowding-unified**?
- Both target JMLR
- Both have similar structure
- Need to **consolidate**

**Action needed**: Clarify which is the primary submission

---

### 5️⃣ **kdd2026_global_crowding**
**Status**: 🔴 PROBLEM IDENTIFIED
**Path**: `/research/kdd2026_global_crowding/`
**Target**: KDD 2026 (Jeju, Korea) - Deadline Feb 8, 2026

**What it is**:
- Global scale factor crowding analysis
- 6 regions × 10+ factors
- ML-based detection (LSTM/XGBoost vs model residuals)

**Current status**:
- Code: Implemented
- Experiments: Run and analyzed
- **Debug Investigation**: COMPLETE (Option D investigation Dec 16)

**Critical Issue Found**:
- Regime-conditional MMD fails on cross-market transfer
- Europe: -21.5% degradation
- Japan: +18.9% success
- Root cause: Domain-specific regime definitions don't transfer

**Debug Results**: See `/experiments/` directory
- FINAL_SUMMARY.md - Root cause analysis
- DIAGNOSTIC_REPORT.md - Detailed findings
- 09-13_*.py - Reproducible diagnostic scripts

**Implication**:
- Table 7 results are problematic
- Need to decide: revise claims or withdraw method

---

## 🔀 Relationship Map

```
factor-crowding-unified (JMLR) ┐
                                 ├─→ Both need repositioning based on
jmlr_unified (JMLR)             ┘   regime-conditional MMD debug findings
                                    (kdd2026_global_crowding investigation)

icml2026_conformal (ICML) ────→ Independent (conformal prediction angle)
                                Can potentially integrate later

kdd2026_global_crowding (KDD) ─→ Source of debug investigation
                                 Temporal-MMD failure revealed here
```

---

## 📋 Status Summary Table

| Project | Target | Status | Issue | Action |
|---------|--------|--------|-------|--------|
| factor_crowding | None | Legacy | Superseded | Archive/Remove |
| **factor-crowding-unified** | **JMLR** | 🟡 **BLOCKED** | T-MMD empirics wrong | **Reposition** |
| icml2026_conformal | ICML 2026 | 🟢 Progressing | None immediate | Continue |
| **jmlr_unified** | **JMLR** | 🟡 **UNCLEAR** | Duplicate? | **Consolidate** |
| **kdd2026_global_crowding** | **KDD 2026** | 🔴 **DEBUG DONE** | Found root cause | Decide next step |

---

## 🎯 Immediate Actions Required

### Priority 1: JMLR Paper Clarity (This Week)
1. **Consolidate**: Are factor-crowding-unified and jmlr_unified the same paper?
   - If YES: Keep one, delete other, update it
   - If NO: Clarify their different scope

2. **Fix JMLR Paper**: Based on kdd2026 debug findings
   - Option A (Recommended): Honest revision - keep Theorem 5, fix empirical claims
   - Option B: Theory-only - remove momentum crash prediction claims
   - Option C: Limited revision - downgrade empirical scope

3. **Key Changes Needed**:
   - Feature importance: 15% → 0.5% (Crowding)
   - Transfer efficiency: +5.2% → -5.2% (acknowledge negative transfer)
   - Regime-conditional method: Add conditions for when it works vs fails

### Priority 2: KDD 2026 Paper (Next 1-2 weeks)
1. **Decide on T-MMD**:
   - Keep with caveats? (add Europe failure case)
   - Replace with Standard MMD? (simpler, more robust)
   - Remove method, focus on empirics only?

2. **Update Table 7**:
   - Current: Shows regime-conditional success
   - Reality: Mixed results (Japan works, Europe fails)
   - Action: Be honest about conditional success

### Priority 3: File Organization (This Week)
1. **Archive superseded**:
   - factor_crowding → move to /archive/
   - Old experimental scripts → archive (already done in kdd2026)

2. **Clarify active**:
   - Keep only: factor-crowding-unified (or jmlr_unified), icml2026_conformal, kdd2026_global_crowding
   - Rename for clarity
   - Create master README

---

## 📁 Directory Organization Recommendations

```
/research/
├── README_PROJECT_OVERVIEW.md ← This file
├── RESEARCH_STATUS.md ← Weekly status tracker
│
├── 2_ACTIVE_PAPERS/ ← Main work
│   ├── jmlr_2026_unified_framework/
│   │   ├── paper/
│   │   ├── src/
│   │   ├── experiments/
│   │   └── README.md (Theorem 5 + empirical validation)
│   │
│   ├── kdd_2026_global_crowding/
│   │   ├── paper/
│   │   ├── src/
│   │   ├── experiments/
│   │   ├── DEBUG_INVESTIGATION/ (Dec 16 findings)
│   │   └── README.md
│   │
│   └── icml_2026_conformal_prediction/
│       ├── paper/
│       ├── src/
│       ├── experiments/
│       └── README.md
│
└── 3_ARCHIVE/
    └── factor_crowding/ (original/legacy)
    └── superseded_experiments/ (old KDD scripts)
```

---

## 🔍 Key Questions to Answer

### Q1: JMLR Paper Status
- **Question**: Are factor-crowding-unified and jmlr_unified two separate papers or one?
- **Why it matters**: Affects which to submit, what to update
- **Action**: Check both README files, decide consolidation strategy

### Q2: Regime-Conditional MMD
- **Question**: Should we keep Temporal-MMD in papers given the Europe failure?
- **Options**:
  - A) Keep with honest analysis of conditions (Japan works, Europe fails)
  - B) Remove method entirely, use Standard MMD
  - C) Keep in JMLR as theory, remove from KDD empirical claims
- **Recommendation**: A (honest analysis)

### Q3: Feature Importance
- **Question**: Why is crowding 30× lower than claimed in paper?
- **Impact**: Undermines core motivation
- **Action**:
  - Check if paper used different feature engineering
  - Fix Table 1 to show actual importance
  - Adjust narrative accordingly

### Q4: Deadlines
- **ICML 2026**: Jan 28, 2026 (6 weeks away)
- **KDD 2026**: Feb 8, 2026 (7 weeks away)
- **JMLR**: Rolling submission (no deadline)
- **Priority**: ICML first (tight deadline), then KDD, then JMLR

---

## 📊 Current File Locations

### JMLR/Factor-Crowding Unified
- **Paper**: `/factor-crowding-unified/paper/`
- **Experiments**: `/kdd2026_global_crowding/experiments/`
- **Debug findings**: `/kdd2026_global_crowding/experiments/FINAL_SUMMARY.md` ⭐

### ICML Conformal
- **Paper**: `/icml2026_conformal/paper/`
- **Code**: `/icml2026_conformal/src/`

### KDD Global Crowding
- **Paper**: `/kdd2026_global_crowding/paper/`
- **Experiments**: `/kdd2026_global_crowding/experiments/`
- **Debug session**: `/kdd2026_global_crowding/experiments/README_DIAGNOSTIC_SESSION.md` ⭐

---

## ✅ What's Complete

✅ **Option D Debug Investigation** (Dec 16, 2025)
- Root cause identified: Regime non-transfer
- Diagnostic scripts created (09-13)
- Detailed reports written
- **Action**: Review findings and decide repositioning strategy

✅ **Code Implementation**
- Theorem 5 implementation (correct)
- Temporal-MMD network (works)
- Standard MMD baseline (works)

✅ **Global Data Collection**
- 6 regions of data
- 10+ factors
- 1930-2025 history

❌ **Empirical Validation**
- Table 7 results are mixed/problematic
- Feature importance discrepancy (30×)
- Transfer efficiency negative on average

---

## 🚀 Next Steps (In Order)

### This Week
1. Read `/kdd2026_global_crowding/experiments/FINAL_SUMMARY.md`
2. Decide on JMLR paper repositioning (Option A/B/C)
3. Decide on JMLR vs jmlr_unified consolidation
4. Create action plan for edits

### Next Week
1. Begin manuscript revisions (whichever paper starts first)
2. Run updated experiments if changes needed
3. Prepare ICML submission (if targeting ICML)

### Week 3
1. Complete JMLR/KDD revisions
2. Submit ICML (if proceeding)
3. Finalize other submissions

---

## 📞 Questions for Clarification

Before starting edits, please clarify:

1. **JMLR Consolidation**: Keep factor-crowding-unified or jmlr_unified (or merge)?
2. **Deadline Priority**: ICML first or JMLR first?
3. **Regime-Conditional**: Keep with caveats (Option A) or remove (Option B)?
4. **Conformal Paper**: Continue in parallel or pause?

---

## 📄 Key Documents to Review

In order of importance:

1. **`/kdd2026_global_crowding/experiments/FINAL_SUMMARY.md`** ⭐⭐⭐
   - Executive summary of debug findings
   - Root cause explained
   - Repositioning recommendations

2. **`/kdd2026_global_crowding/experiments/DIAGNOSTIC_REPORT.md`**
   - Detailed technical analysis
   - Evidence and verification
   - Theory assessment

3. **`/kdd2026_global_crowding/experiments/README_DIAGNOSTIC_SESSION.md`**
   - Quick reference guide
   - How to run diagnostics
   - Key findings table

4. **`/factor-crowding-unified/README.md`**
   - Current JMLR paper scope
   - 3-component framework
   - What needs fixing

---

## Summary

**The Big Picture**: You're working on 3 conference papers (JMLR + KDD + ICML) on factor crowding in financial markets. A recent debug investigation found that one of the key methods (Temporal-MMD) has conditional validity (works in Japan, fails in Europe), which means the main papers need repositioning to be honest about these limitations.

**The Good News**: The theory is sound, the code is correct, the issue is just that empirical claims need to match reality.

**The Action**: Review the debug findings, decide how to reposition, and update the papers accordingly.

All files are organized, git-committed, and ready for the next phase of work.
