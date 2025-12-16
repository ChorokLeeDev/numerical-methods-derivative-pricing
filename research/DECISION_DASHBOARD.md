# Decision Dashboard
## All Projects Status & Pending Decisions

**Date**: December 16, 2025
**Status**: Projects organized, debug complete, awaiting decisions

---

## 📊 Executive Summary

**5 Research Projects across 2 repositories:**

| Project | Repository | Status | Deadline | Action |
|---------|-----------|--------|----------|--------|
| **JMLR Unified** | quant | 🟡 Blocked | None | ⏳ DECIDE: Option A/B |
| **KDD 2026** | quant | 🔴 Debug done | Feb 8 | ⏳ DECIDE: Option A/B |
| **ICML 2026** | quant | 🟢 OK | Jan 28 | ✅ Continue |
| **Causal Structure** | ai-in-finance | 🟢 OK | Unknown | ✅ Independent |
| **Legacy factor_crowding** | quant | ⛔ Superseded | None | ⏳ Archive? |

---

## 🎯 CRITICAL DECISIONS NEEDED THIS WEEK

### DECISION 1: Temporal-MMD Elimination Strategy
**Affects**: JMLR paper + KDD paper
**Status**: Two options analyzed

#### ✅ OPTION A: Replace with Standard MMD (RECOMMENDED)
**Approach**: Use Long et al. (2015) standard MMD instead of regime-conditional
**Pros**:
- Works consistently across all regions (Europe: -21.5% → +6.3%)
- Average improvement: -5.2% → +7.7%
- More robust, well-understood method
- Still maintains domain adaptation contribution

**Cons**:
- Less novel than regime-conditional approach
- Standard method (less exciting academically)

**Implementation**:
- JMLR: Section 4.2 rewrite, update Theorem 5 references
- KDD: Update Table 7 with Standard MMD results, Section 4.2 name change
- Code: Replace src/models/temporal_mmd.py with standard_mmd.py
- Time: 6-8 hours

**Paper outcomes**:
```
JMLR: Game theory (✅) + Standard MMD (⬇️ novelty) + Conformal (✅)
KDD: ML detection (✅) + Standard MMD (✅) + Global scope (✅)
```

---

#### ❌ OPTION B: Remove Domain Adaptation Entirely
**Approach**: Keep only ML detection and conformal prediction
**Pros**:
- Simpler paper structure
- No conditional success issues
- Cleaner focus on detection vs adaptation

**Cons**:
- Lose entire domain adaptation contribution
- Less complete research story
- KDD loses the transfer learning angle
- JMLR reduces to 2 components instead of 3

**Implementation**:
- JMLR: Delete Section 4.2, remove Theorem 5
- KDD: Delete Table 7, focus on region-specific models
- Time: 4-6 hours (less work)

**Paper outcomes**:
```
JMLR: Game theory (✅) + Conformal (✅) [no adaptation]
KDD: ML detection (✅) only [no transfer]
```

---

#### ⚠️ OPTION C: Keep with Honest Caveats (NOT RECOMMENDED)
**Approach**: Keep T-MMD but explicitly document conditional success
**Pros**:
- Maintains novel method
- Shows where it works (Japan) and fails (Europe)
- Educational value

**Cons**:
- Appears to oversell results
- Weaker contribution (conditional success is weak)
- Harder to explain empirically

**Not recommended**: Conditional success looks like you're hiding failures

---

### 🔴 **YOUR DECISION**: Which option (A, B, or C)?
- **A**: Better balance of novelty and robustness
- **B**: Simpler but less complete
- **C**: Not recommended (weak positioning)

**Timeline**: Decide by end of week to implement in next 2 weeks

---

## 📋 SECONDARY DECISIONS NEEDED

### DECISION 2: JMLR Paper Consolidation
**Question**: Are `factor-crowding-unified` and `jmlr_unified` the same paper?
**Status**: Appears to be duplicate versions

**Current state**:
- `/quant/research/factor-crowding-unified/` - Original version (older)
- `/quant/research/jmlr_unified/` - Submission version (newer, has .zip files)
- Both have same 3-component structure
- Both have similar paper outlines

**ACTION OPTIONS**:

**Option A1**: Keep jmlr_unified (primary)
- Delete factor-crowding-unified
- Use jmlr_unified for submission
- Action: Archive old version

**Option A2**: Keep factor-crowding-unified (working version)
- Delete jmlr_unified
- Use factor-crowding-unified for edits
- Action: Clean up submission folder

**Option A3**: Clarify distinction
- If genuinely different: document why
- If same: consolidate into one

**YOUR DECISION**: A1 or A2 or explain difference?

---

### DECISION 3: Legacy Project Cleanup
**Question**: What to do with `/quant/research/factor_crowding/`?
**Status**: ⛔ Superseded by factor-crowding-unified

**Current state**:
- Created ~2024
- Not active since November 2024
- Likely contains duplicated work

**ACTION OPTIONS**:

**Option B1**: Archive it
```bash
mkdir -p /archive/
mv factor_crowding /archive/factor_crowding_legacy
git commit -m "Archive legacy factor_crowding project"
```

**Option B2**: Keep for reference
- Leave in place
- Mark as archived in project structure

**YOUR DECISION**: B1 (archive) or B2 (keep)?

---

### DECISION 4: Literature & Novelty Status
**Question**: How to position each paper vs existing work?
**Status**: Analysis complete in LITERATURE_ANALYSIS.md

**Key findings**:
- Game-theoretic model: ⚠️ Compare with Hua & Sun (2024)
- Domain adaptation: 🟢 Novel application to finance
- Conformal prediction: ✅ Novel application to crowding
- Causal structure: 🟢 Completely independent (ai-in-finance)

**ACTION NEEDED**:
- Read Hua & Sun (2024) to confirm game-theory novelty
- Update references in all papers
- Clarify contribution vs existing work

**Timeline**: This week (can do in parallel)

---

### DECISION 5: Feature Importance Discrepancy
**Question**: Why does JMLR claim crowding is 15% importance but SHAP shows 0.5%?
**Status**: ⚠️ 30× overstatement found

**Current state**:
- JMLR text: "Crowding ranks 3rd, explaining 15% of variance"
- SHAP analysis: Crowding ranks 11th, explains 0.5% of variance
- Discrepancy: Major issue for credibility

**ACTION NEEDED**:
- Clarify which metric is correct
- Fix text to match reality
- Explain methodology difference
- Update all related claims

**Timeline**: This week (necessary for credibility)

---

## ✅ PROJECTS THAT DON'T NEED DECISIONS

### ✅ ICML 2026: "Conformal Prediction for Factor Crowding"
**Status**: 🟢 Independent, proceeding normally
**Action**: Continue as planned
**Timeline**: Paper draft due soon (Jan 28 deadline)
**No T-MMD impact**: Uses completely different method
**Next**: Focus on experimental validation & paper writing

### ✅ ai-in-finance Causal Paper: "Causal Structure Changes..."
**Status**: 🟢 Independent in ai-in-finance repo
**Action**: No changes needed
**Method**: Granger causality + Student-t HMM (not Temporal-MMD)
**Note**: Complementary to quant papers, but completely separate
**Next**: Define submission target & timeline for this paper

---

## 📅 Timeline by Decision

### THIS WEEK (Dec 16-22)
**URGENT**:
- [ ] Decide: Option A, B, or C for Temporal-MMD?
- [ ] Decide: Consolidate JMLR papers (A1, A2, or explain)?
- [ ] Decide: Archive legacy factor_crowding (B1 or B2)?

**IMPORTANT**:
- [ ] Read Hua & Sun (2024) for novelty comparison
- [ ] Fix feature importance discrepancy (15% vs 0.5%)
- [ ] Clarify ai-in-finance Causal paper status

### NEXT 2 WEEKS (Dec 23-Jan 6)
**IMPLEMENT**:
- [ ] Execute Temporal-MMD decision (6-8 hours if Option A, 4-6 if Option B)
- [ ] Update JMLR paper accordingly
- [ ] Update KDD paper accordingly
- [ ] Archive/consolidate as decided
- [ ] Complete ICML conformal draft

### WEEK 3-6 (Jan 7-Feb 8)
**POLISH & SUBMIT**:
- [ ] ICML submission (Jan 28 deadline)
- [ ] Final validation experiments
- [ ] KDD submission (Feb 8 deadline)
- [ ] JMLR rolling submission (flexible)

---

## 📊 Impact Matrix

| Decision | JMLR | KDD | ICML | Timeline |
|----------|------|-----|------|----------|
| T-MMD Option A | 🔴 Critical | 🔴 Critical | 🟢 None | Implement next 2w |
| T-MMD Option B | 🟡 Important | 🟡 Important | 🟢 None | Implement next 2w |
| Paper consolidation | 🔴 Critical | 🟢 None | 🟢 None | Decide this week |
| Legacy archival | 🟢 Optional | 🟢 None | 🟢 None | Decide this week |
| Literature review | 🟡 Important | 🟡 Important | 🟡 Important | Implement this week |
| Feature importance | 🔴 Critical | 🟡 Nice-to-have | 🟢 None | Fix this week |

---

## 🎯 Summary of Key Files

**For Decisions**:
- `ELIMINATION_PLAN.md` - Temporal-MMD removal options (A, B, C)
- `KDD_IMPACT_ANALYSIS.md` - How each option affects KDD
- `LITERATURE_ANALYSIS.md` - Novelty assessment & overlaps
- `PAPER_ECOSYSTEM_CLARIFICATION.md` - Independence of papers

**For Context**:
- `PROJECT_OVERVIEW.md` - Big picture of 5 projects
- `PROJECT_DETAILS.md` - Individual project analysis
- `README.md` (in /quant) - Current project status

**In Debug Folder** (`/kdd2026_global_crowding/experiments/`):
- `FINAL_SUMMARY.md` - Executive summary of T-MMD failure
- `DIAGNOSTIC_REPORT.md` - Detailed root cause analysis
- `09-13_*.py` - Reproducible diagnostic scripts

---

## 🚀 Next Immediate Steps

### TODAY:
1. Read ELIMINATION_PLAN.md carefully
2. Decide: A (Standard MMD), B (remove), or C (not recommended)?
3. Read PAPER_ECOSYSTEM_CLARIFICATION.md to confirm understanding

### TOMORROW:
4. Confirm JMLR paper consolidation choice
5. Confirm legacy project cleanup choice
6. Start reading Hua & Sun (2024)

### THIS WEEK:
7. Fix feature importance discrepancy
8. Update literature section based on novelty assessment
9. Begin implementing chosen option

---

## 📞 Questions to Answer

Before implementation, clarify:

1. **T-MMD elimination**: Option A, B, or C?
2. **JMLR consolidation**: A1 (keep jmlr_unified), A2 (keep factor-crowding-unified), or explain difference?
3. **Legacy cleanup**: B1 (archive) or B2 (keep)?
4. **Feature importance**: Should 15% be corrected to 0.5%?
5. **Causal paper**: Submit separately or integrate with KDD/JMLR later?

**Once these are decided, implementation can proceed systematically.**

