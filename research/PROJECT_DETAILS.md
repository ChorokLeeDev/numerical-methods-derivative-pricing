# Project Details & Current Status
## Individual Project Analysis

**Last Updated**: December 16, 2025

---

## Project 1: factor_crowding (LEGACY)

**Location**: `/research/factor_crowding/`
**Status**: ⛔ SUPERSEDED
**Created**: ~2024
**Last active**: ~November 2024

### What it contains
```
factor_crowding/
├── data/           - Original Fama-French data
├── docs/           - Documentation
├── literature/     - Paper references
├── models/         - Base crowding models
├── experiments/    - Initial experiments
└── paper/         - Early draft(s)
```

### Current issues
- No recent activity
- Superseded by factor-crowding-unified
- Likely duplicated work

### Recommendation
**ARCHIVE & REMOVE**
- Move to `/archive/factor_crowding_legacy/`
- Keep as historical reference only
- Don't maintain or update

### Action
```bash
# If you decide to archive:
mkdir -p ../archive/
mv factor_crowding ../archive/factor_crowding_legacy
git add -A && git commit -m "Archive legacy factor_crowding project"
```

---

## Project 2: factor-crowding-unified (MAIN JMLR PAPER)

**Location**: `/research/factor-crowding-unified/`
**Status**: 🟡 ACTIVE but BLOCKED
**Target**: JMLR (no deadline, rolling submission)
**Created**: 2024
**Last updated**: December 4, 2025

### Paper structure
```
Three integrated components:

1. GAME-THEORETIC MODEL
   - Explains alpha decay from Nash equilibrium
   - Main contribution: Theorem for crowding dynamics
   - Status: ✅ Complete

2. REGIME-CONDITIONAL DOMAIN ADAPTATION (Temporal-MMD)
   - Cross-market transfer of factor models
   - Theorem 5: Error bound formula
   - Status: 🟡 Code correct, but empirical results problematic

3. CONFORMAL PREDICTION
   - Uncertainty quantification for crowding
   - Risk management approach
   - Status: ✅ Mostly complete

```

### Paper locations
- **Main manuscript**: `/factor-crowding-unified/paper/` (LaTeX)
- **Experiments code**: In various locations
- **Latest version**: `jmlr_unified/jmlr_submission/` (see next project)

### Current status

#### ✅ Completed
- Game-theoretic framework (Theorem 1)
- Theorem 5 mathematical proof
- Conformal prediction methodology
- Code implementation (all working)
- Writing (all sections drafted)

#### 🟡 Issues Found (Dec 16)
1. **Temporal-MMD empirics**
   - Theory: Should improve transfer via regime conditioning
   - Reality: Japan +18.9% (works), Europe -21.5% (fails)
   - Average: -5.2% negative transfer (not +5.2% claimed)
   - Root cause: Regimes are domain-specific, not domain-invariant

2. **Feature importance discrepancy**
   - Claims: Crowding at 15% importance (Table 1, rank #3)
   - Reality: Crowding at 0.5% importance (SHAP, rank #11)
   - Discrepancy: 30× overstatement

3. **Empirical validation**
   - Table 7: Only shows Japan success, needs to address Europe failure
   - Method claims: Overstated without European failure explanation

### What needs fixing

**CRITICAL**:
1. ❌ Fix feature importance: 15% → 0.5%
2. ❌ Fix transfer efficiency: +5.2% → -5.2% OR explain regime conditions
3. ❌ Add section: "When Regime-Conditioning Helps vs. Hurts"
4. ❌ Update Table 7 with honest results

**IMPORTANT**:
1. Add Japan success case analysis (+18.9%)
2. Add Europe failure case analysis (-21.5%)
3. Explain regime semantic mismatch across markets
4. Provide guidance on checking regime transferability

### Debug investigation results
**Location**: `/kdd2026_global_crowding/experiments/FINAL_SUMMARY.md`

**Finding**: Regime definitions are market-specific (based on rolling volatility), not domain-invariant. When regimes have different meanings across markets, MMD matching within regimes creates negative transfer.

### Recommended action: **REPOSITION (Option A)**

**New frame**:
- Title: "Regime-Conditional Domain Adaptation: Theory and Empirical Limitations"
- Keep Theorem 5 (mathematically valid)
- Honest about conditions for success (Japan-like) vs failure (Europe-like)
- Add diagnostic framework for checking regime transferability
- Fix all empirical numbers to match reality

**Timeline**: 1-2 weeks to revise

---

## Project 3: jmlr_unified (JMLR SUBMISSION VERSION)

**Location**: `/research/jmlr_unified/`
**Status**: 🟡 ACTIVE
**Target**: JMLR
**Created**: ~2025
**Last updated**: December 4, 2025

### Current status
```
jmlr_unified/
├── jmlr_submission/
│   ├── main.tex (MAIN MANUSCRIPT)
│   ├── main.pdf
│   ├── sections/ (01-09 numbered sections)
│   ├── appendices/ (A-F appendices)
│   └── [submission package files]
├── paper/
├── src/ (code)
└── data/
```

### Issue: **Likely duplicate of factor-crowding-unified**

Both projects:
- Target JMLR
- Have same 3-component structure (game theory + domain adaptation + conformal)
- Cover same research
- Have similar paper outlines

### Questions
1. Is this the "final" version vs "draft" version?
2. Should we submit jmlr_unified instead of factor-crowding-unified?
3. Are they meant to be different papers?

### Action required: **CONSOLIDATE**

**Decision needed**:
- Keep jmlr_unified (more recent, has submission files)
- Delete factor-crowding-unified (older version)
- OR clarify they are genuinely different

**For now**: Treat jmlr_unified as the primary JMLR submission version
- But it ALSO needs the repositioning fixes from debug investigation
- Same fixes apply (feature importance, transfer efficiency, regime conditions)

---

## Project 4: icml2026_conformal

**Location**: `/research/icml2026_conformal/`
**Status**: 🟢 INDEPENDENT & PROGRESSING
**Target**: ICML 2026 (Seoul, Korea)
**Deadline**: January 28, 2026 (≈6 weeks)
**Created**: 2025
**Last updated**: ~November 2025

### Scope
```
Conformal Prediction for Factor Crowding:

1. CONFORMAL CROWDING DETECTION
   - Prediction sets with coverage guarantees
   - Regime classification with uncertainty bounds
   - Status: Code complete

2. CONFORMAL TAIL RISK
   - Distribution-free prediction intervals
   - Factor returns uncertainty quantification
   - Status: Experimental

3. CALIBRATION ANALYSIS
   - Comparison vs Bayesian and Bootstrap
   - Coverage verification
   - Status: In progress
```

### Current status

#### ✅ Complete
- Theoretical framework
- Conformal prediction implementation
- Basic experiments

#### 🟡 In progress
- Calibration analysis
- Coverage guarantees proof
- Extended experiments

#### ❌ Not started
- Submission-ready paper writing
- Final experiments
- Performance comparisons

### Relationship to other papers
- **INDEPENDENT**: Doesn't depend on regime-conditional MMD fixes
- **Can integrate later**: Conformal approach could combine with game-theory framework
- **No overlap**: Different method, different focus

### Timeline
- Paper draft: Due soon (target 2-3 weeks)
- Experiments: Ongoing
- Deadline: Jan 28, 2026

### Recommendation
**CONTINUE IN PARALLEL** - Not affected by JMLR/KDD fixes

---

## Project 5: kdd2026_global_crowding (PROBLEM IDENTIFIED)

**Location**: `/research/kdd2026_global_crowding/`
**Status**: 🔴 ROOT CAUSE IDENTIFIED
**Target**: KDD 2026 (Jeju, Korea)
**Deadline**: February 8, 2026 (≈7 weeks)
**Created**: 2024
**Last updated**: December 16, 2025 (Option D debug complete)

### Project structure
```
kdd2026_global_crowding/
├── paper/            - KDD manuscript
├── src/              - Core code
├── data/             - 6 regions of global data
├── experiments/      - Experimental scripts
│   ├── 01-08_*.py   - ARCHIVED (superseded)
│   ├── 09-13_*.py   - ACTIVE (diagnostic scripts) ⭐
│   ├── DEBUG_*.md   - Investigation reports ⭐
│   └── README_DIAGNOSTIC_SESSION.md - Guide ⭐
├── results/          - OUTPUT DATA
└── tests/            - Unit tests
```

### Scope
Global-scale factor crowding analysis:
- **6 regions**: US, UK, Japan, Europe, AsiaPac, Global
- **10+ factors**: Momentum, value, size, quality, etc.
- **Period**: 1930-2025 (95 years of data)
- **Method**: ML detection vs Model-based

### ✅ What's complete
- Global data collection & preprocessing
- ML model implementation (LSTM, XGBoost)
- Baseline models
- All experiments run
- Table 7 generated (with problems)

### 🔴 Critical issues found (Dec 16)

**The Problem**: Table 7 shows cross-market transfer results for Temporal-MMD
```
Original claim: "Regime-conditional MMD improves transfer"

Results:
  US→UK:       +10.9% ✓
  US→Japan:    +18.9% ✓ (GREAT!)
  US→Europe:   -21.5% ✗ (TERRIBLE!)
  US→AsiaPac:  -30.0% ✗ (TERRIBLE!)
  Average:     -5.2%  ✗ (NEGATIVE, not +5.2%!)
```

**Root Cause Found**:
- Regimes defined by rolling volatility (market-specific temporal patterns)
- "US high-vol" (dot-com crash, 2000) ≠ "Europe high-vol" (Euro crisis, 2008)
- Same regime label (0 or 1) has different meaning in different markets
- When MMD tries to match incompatible regimes → negative transfer
- This is a **theory assumption failure**, not a code bug

**Evidence**:
- Standard MMD (no regime conditioning) > T-MMD on Europe
- Standard MMD (0.608) vs T-MMD (0.581) on Europe → regime conditioning HURTS
- T-MMD (0.788) >> Standard MMD (0.565) on Japan → regime conditioning HELPS
- **Conclusion**: Regime transfer is conditional on market structure, not universal

### Decision needed: **What to do with Table 7?**

**Option A**: Keep T-MMD, but add caveats
- Add section: "Conditional Transfer: When Regime Matching Works"
- Show Japan success case (why it works)
- Show Europe failure case (why it fails)
- Explain regime semantic mismatch
- Honest paper about conditions for success

**Option B**: Remove T-MMD, use Standard MMD
- Simpler method, more robust
- Still shows good results (better than RF)
- Easier to explain
- But loses the novelty of regime-conditioning

**Option C**: Keep empirics, remove method claim
- Don't claim regime-conditioning helps in general
- Just show results, let reader decide
- Adds caveats about conditional success

### Debug investigation deliverables

**Location**: `/kdd2026_global_crowding/experiments/`

**Documents** (must read):
1. `FINAL_SUMMARY.md` ⭐ - Executive summary (10 min read)
2. `DIAGNOSTIC_REPORT.md` - Detailed analysis (30 min read)
3. `DEBUG_SESSION_CLEANUP.md` - What was archived and why
4. `README_DIAGNOSTIC_SESSION.md` - Quick reference

**Reproducible scripts** (all executable):
1. `09_country_transfer_validation.py` - Shows the problem
2. `10_regime_composition_analysis.py` - Analyzes regime distributions
3. `11_regime_detection_debug.py` - Verifies algorithm
4. `12_date_range_analysis.py` - Shows regime evolution over time
5. `13_mmd_comparison_standard_vs_regime.py` - Confirms hypothesis

**Archived** (historical):
- `_archive_superseded_experiments/` - Old scripts (01-08)

### What needs fixing for KDD submission

**MUST FIX**:
1. ❌ Update Table 7 caption to be honest about mixed results
2. ❌ Add section explaining Europe failure (regime non-transfer)
3. ❌ Add section analyzing Japan success (why regime matching worked)
4. ❌ Decide: Keep T-MMD (Option A) or switch to Standard MMD (Option B)

**SHOULD ADD**:
1. Figure showing regime distributions across markets
2. Figure showing Standard MMD vs T-MMD comparison
3. Discussion of when regime-conditioning is useful vs harmful
4. Guidance for practitioners on checking regime assumptions

### Timeline
- Debug investigation: ✅ COMPLETE (Dec 16)
- Decision on T-MMD: ⏳ PENDING (needs your call)
- Manuscript revision: 2-3 weeks
- Submission: Before Feb 8, 2026 (6 weeks available)

### Recommendation
**KEEP T-MMD with honest repositioning (Option A)**
- Novel method is interesting, even if conditional
- Japan success shows it can work
- Europe failure is valuable scientific finding (where it doesn't work)
- Honest paper is better than overselling results
- See: FINAL_SUMMARY.md for detailed rationale

---

## Summary Table

| Project | Target | Status | Issue | Action |
|---------|--------|--------|-------|--------|
| factor_crowding | None | ⛔ Legacy | Superseded | Archive |
| **factor-crowding-unified** | **JMLR** | 🟡 Blocked | T-MMD & feature importance | **Reposition** |
| **jmlr_unified** | **JMLR** | 🟡 Unclear | Duplicate of above? | **Consolidate** |
| icml2026_conformal | ICML 2026 | 🟢 OK | None | Continue |
| **kdd2026_global_crowding** | **KDD 2026** | 🔴 Debug done | T-MMD conditional | **Revise** |

---

## Priority Matrix

### THIS WEEK (Dec 16-22)
1. **Read** debug findings (/kdd2026_global_crowding/experiments/)
2. **Decide** JMLR vs jmlr_unified consolidation
3. **Plan** repositioning strategy (Option A/B/C)
4. **Archive** legacy projects if decided

### NEXT 2 WEEKS (Dec 23-Jan 6)
1. **Revise** JMLR paper (option chosen)
2. **Revise** KDD paper (option chosen)
3. **Complete** ICML conformal draft
4. **Prepare** all for submission

### WEEK 3-6 (Jan 7-Feb 8)
1. **Polish** all three papers
2. **Run** final validation experiments
3. **Submit** ICML (Jan 28)
4. **Submit** KDD (Feb 8)
5. **Prepare** JMLR for rolling submission

---

## Key Documents to Read (In Order)

1. **THIS DOCUMENT** (PROJECT_DETAILS.md) - Where you are now
2. **PROJECT_OVERVIEW.md** - Big picture (5 min)
3. **kdd2026_global_crowding/experiments/FINAL_SUMMARY.md** - Debug findings (10 min) ⭐
4. **kdd2026_global_crowding/experiments/DIAGNOSTIC_REPORT.md** - Details (30 min) ⭐
5. **factor-crowding-unified/README.md** - Current paper scope
6. **icml2026_conformal/README.md** - ICML paper scope

---

## Next Steps

1. ✅ **Read** this document and PROJECT_OVERVIEW.md
2. ⏳ **Read** the debug investigation reports (FINAL_SUMMARY.md)
3. ⏳ **Decide**:
   - How to reposition JMLR paper
   - Whether to consolidate jmlr_unified
   - Whether to keep T-MMD in KDD paper
   - Whether to archive legacy factor_crowding
4. ⏳ **Plan** manuscript edits
5. ⏳ **Execute** revisions

Everything is organized and ready. You just need to make the strategic decisions.
