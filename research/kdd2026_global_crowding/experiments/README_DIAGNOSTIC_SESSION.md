# Diagnostic Session: Option D Investigation
## December 16, 2025

**Status**: Complete ✅
**Finding**: Root cause identified - regime non-transfer due to domain-specific definitions
**Action**: Repositioning required - fundamental claim revision needed

---

## Quick Start

### For Understanding the Findings
1. Read: **FINAL_SUMMARY.md** (10 min overview)
2. Deep dive: **DIAGNOSTIC_REPORT.md** (full analysis)

### For Verification/Reproduction
Run the diagnostic scripts in order:
```bash
python3 09_country_transfer_validation.py   # Shows the problem
python3 10_regime_composition_analysis.py   # Analyzes regime distribution
python3 11_regime_detection_debug.py        # Verifies algorithm
python3 12_date_range_analysis.py           # Traces regime evolution
python3 13_mmd_comparison_standard_vs_regime.py  # Confirms hypothesis
```

---

## Files Structure

### 📋 Critical Documentation
- **FINAL_SUMMARY.md** ⭐ START HERE
  - Executive summary with findings
  - Root cause explanation
  - Repositioning recommendations
  - Decision tree for paper revision

- **DIAGNOSTIC_REPORT.md** ⭐ DETAILED ANALYSIS
  - Code verification results
  - Root cause deep-dive
  - Empirical evidence
  - Theory assessment
  - Implication for paper

- **DEBUG_SESSION_CLEANUP.md**
  - What was kept/archived and why
  - Files inventory
  - Key findings summary
  - Cleanup log

### 🔬 Active Diagnostic Scripts
All fully executable and reproducible:

- **09_country_transfer_validation.py**
  - Tests: US→UK, US→Japan, US→Europe, US→AsiaPac
  - Output: Country-specific transfer results
  - Purpose: Demonstrates the problem (Japan +18.9%, Europe -21.5%)

- **10_regime_composition_analysis.py**
  - Analyzes regime distributions across countries
  - Shows regime agreement metrics
  - Purpose: Identifies regime mismatch

- **11_regime_detection_debug.py**
  - Verifies regime detection algorithm
  - Traces rolling volatility calculations
  - Purpose: Confirms code matches theory

- **12_date_range_analysis.py**
  - Analyzes regime evolution over 5-year periods
  - Shows temporal regime patterns
  - Purpose: Reveals why regimes don't transfer (different meanings across time)

- **13_mmd_comparison_standard_vs_regime.py**
  - Direct comparison: Standard MMD vs Regime-Conditional MMD
  - Results: Europe favors Standard MMD, Japan favors T-MMD
  - Purpose: Validates hypothesis that regime transfer is market-dependent

### 📦 Archive Directory
- **_archive_superseded_experiments/**
  - Files 01-08 (original experiments)
  - Files 09, 10, 11, 12 variants (intermediate iterations)
  - See README.md in that directory
  - Purpose: Historical reference only

### 📄 Output Files
- **country_transfer_results.log** - Output from 09_country_transfer_validation.py
- **transfer_validation_output.log** - Earlier experiment outputs

---

## The Findings in One Page

### What We Discovered
```
Theory: Regime-conditional MMD should improve transfer via tighter bounds
Reality: Japan works (+18.9%), Europe fails (-21.5%), average is negative (-5.2%)

Root Cause: Regimes are domain-specific temporal patterns, not domain-invariant
- Regimes defined by: rolling volatility (63-month std vs 252-month median)
- Problem: "US high-vol" (dot-com) ≠ "Europe high-vol" (Euro crisis)
- Result: MMD matching incompatible regimes creates negative transfer

Code Status: ✅ Correct (matches Theorem 5)
Theory Status: ✅ Valid (math is sound)
Assumption Status: ❌ Failed (domain-invariant regime assumption violated)
```

### Paper Claims vs Reality
```
Claimed: "Transfer efficiency +5.2% on average"
Reality: -5.2% negative transfer on average

Claimed: "Crowding importance 15% (rank #3)"
Reality: Crowding importance 0.5% (rank #11)

Claimed: "Consistent improvements across markets"
Reality: Japan +18.9%, Europe -21.5%, Mixed results
```

---

## Repositioning Decision

### Recommended: Option A (Honest Revision)
**New framing**: "Regime-Conditional Domain Adaptation: Theory and Empirical Limitations"

**Keep**:
- ✅ Theorem 5 (valid theoretical contribution)
- ✅ Regime-conditional framework (novel approach)

**Change**:
- ❌ Remove overstated empirical claims
- ✅ Add Japan success case analysis
- ✅ Add Europe failure case analysis
- ✅ Include conditions for regime transferability
- ✅ Fix feature importance (Crowding 0.5%, not 15%)

**Result**: Credible paper that preserves theory while being honest about empirical limitations

### Alternative: Option B (Theory-Only)
Focus exclusively on Theorem 5, remove momentum prediction claims

### Alternative: Option C (Limited Revision)
Keep theory, downgrade empirical claims to "preliminary", acknowledge limitations

---

## Next Steps

1. **Review** FINAL_SUMMARY.md and DIAGNOSTIC_REPORT.md (today)
2. **Decide** which repositioning option (A, B, or C)
3. **Plan** manuscript revision scope
4. **Begin** writing new Introduction/Assumptions sections
5. **Use** diagnostic scripts for validation during revision
6. **Reference** this cleanup log when explaining changes

---

## Quick Reference: Running Diagnostics

```bash
# Show the problem (30 sec)
python3 09_country_transfer_validation.py

# Understand regime composition (2 min)
python3 10_regime_composition_analysis.py

# Verify algorithm correctness (1 min)
python3 11_regime_detection_debug.py

# See regime evolution over time (2 min)
python3 12_date_range_analysis.py

# Confirm hypothesis with method comparison (3 min)
python3 13_mmd_comparison_standard_vs_regime.py
```

All scripts are self-contained and reproducible.

---

## Investigation Summary

| Phase | Task | Status | Finding |
|-------|------|--------|---------|
| 1 | Code verification | ✅ Complete | Code correct, matches Theorem 5 |
| 2 | Regime analysis | ✅ Complete | Regimes are domain-specific, not invariant |
| 3 | Root cause diagnosis | ✅ Complete | Regime non-transfer for Europe, not for Japan |
| 4 | Hypothesis validation | ✅ Complete | Standard MMD > T-MMD on Europe confirms finding |
| 5 | Documentation | ✅ Complete | Reports, scripts, cleanup log created |
| **6** | **Repositioning** | ⏳ **Pending** | **User decision needed** |

---

## Key Contacts/Notes

- **Theory (Theorem 5)**: Sound and valid ✅
- **Code (temporal_mmd.py)**: Correct implementation ✅
- **Assumptions**: Domain-invariant regimes - empirically violated ❌
- **Paper claims**: Overstated, require revision ❌

**Status**: Ready for repositioning work
