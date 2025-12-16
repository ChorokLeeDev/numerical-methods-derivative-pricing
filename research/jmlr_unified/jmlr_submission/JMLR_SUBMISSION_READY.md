# JMLR Submission Package - READY FOR UPLOAD

## Status: ✓ COMPLETE

**Date:** December 16, 2025  
**Document:** Not All Factors Crowd Equally: A Game-Theoretic Model of Alpha Decay with Global Transfer and Risk Management  
**Author:** Chorok Lee (KAIST)

## Compiled Manuscript

**File:** `main.pdf`
- **Pages:** 62
- **Size:** 567 KB  
- **Format:** PDF 1.7 (JMLR compliant)
- **Status:** Ready for submission

## Source Files

Located in: `/research/jmlr_unified/jmlr_submission/`

```
├── main.pdf          ← SUBMISSION FILE (62 pages, 567 KB)
├── main.tex          ← LaTeX source code
├── jmlr2e.sty        ← JMLR formatting style
├── references.bib    ← Bibliography database
├── macros.tex        ← Custom notation definitions
│
├── sections/         ← Main document sections
│   ├── 01_introduction.tex       (1.1-1.8: Problem & Contributions)
│   ├── 02_related_work.tex       (2.1-2.4: Literature Review)
│   ├── 03_background.tex         (3.1-3.4: Theory & Methods)
│   ├── 04_game_theory.tex        (4.1-4.5: Game-Theoretic Model)
│   ├── 05_us_empirical.tex       (5.1-5.5: US Market Tests)
│   ├── 06_domain_adaptation.tex  (6.1-6.5: Temporal-MMD)
│   ├── 07_tail_risk_aci.tex      (7.1-7.6: Conformal Prediction)
│   ├── 08_robustness.tex         (8.1-8.4: Sensitivity Analysis)
│   └── 09_conclusion.tex         (9.1-9.4: Summary & Impact)
│
└── appendices/       ← Supporting material (A-F)
    ├── A_game_theory_proofs.tex
    ├── B_domain_adaptation_theory.tex
    ├── C_conformal_prediction_proofs.tex
    ├── D_data_documentation.tex
    ├── E_algorithm_pseudocode.tex
    └── F_supplementary_robustness.tex
```

## Corrections Applied

✓ **Removed location & date** from author affiliation  
✓ **Fixed whitespace corruption** from PDF extraction  
✓ **Escaped special characters** (underscores, ampersands in text)  
✓ **Fixed mathematical notation** (lambda characters, fractions)  
✓ **Corrected URLs** with `\url{}` wrapper  
✓ **Cleaned malformed expressions** in all sections  
✓ **Verified compilation** with pdflatex  

## Content Structure

### Three Core Contributions

1. **Game-Theoretic Model of Crowding Decay** (Section 4)
   - Hyperbolic decay: α(t) = K/(1 + λt)
   - Judgment factors decay 2.4× faster than mechanical factors
   - Validated on 61 years of Fama-French data (1963–2024)

2. **Regime-Conditional Domain Adaptation** (Section 6)
   - Temporal-MMD framework respecting financial market structure
   - Transfer efficiency: 43% → 69% across 7 developed markets

3. **Crowding-Weighted Conformal Prediction** (Section 7)
   - CW-ACI preserves coverage guarantees while incorporating crowding
   - Portfolio hedging: Sharpe ratio 0.68 → 1.03 (54% improvement)

## How to Submit

### Option 1: Direct Upload
1. Go to JMLR submission portal
2. Upload **main.pdf** as the compiled manuscript
3. Upload entire `jmlr_submission/` folder as supplementary material

### Option 2: Create Submission Archive
```bash
cd /Users/i767700/Github/quant/research/jmlr_unified/
zip -r jmlr_submission.zip jmlr_submission/
# Upload jmlr_submission.zip to JMLR
```

## Verification Checklist

- [x] Title page with author affiliation (location/date removed)
- [x] Abstract with 3 main contributions
- [x] Keywords (7 terms for indexing)
- [x] Sections 1-9 (Introduction through Conclusion)
- [x] Appendices A-F (Proofs, Data, Algorithms)
- [x] Bibliography (references.bib with natbib format)
- [x] Mathematical notation (properly formatted with LaTeX)
- [x] Tables and figures (properly formatted)
- [x] PDF compilation (no critical errors)
- [x] File naming conventions (JMLR standard)

## Key Statistics

- **Total Pages:** 62
- **Word Count:** ~30,000 words
- **Main Sections:** 9
- **Appendices:** 6 (A-F)
- **References:** ~50+
- **Theorems:** 6 formal theorems with proofs
- **Empirical Tests:** Validated on 61 years of financial data
- **Geographic Coverage:** 7 developed markets (US, UK, Japan, Germany, France, Canada, Australia)

## Notes for Reviewers

1. **GitHub Availability:** All code and data processing scripts available upon request
2. **Reproducibility:** Complete algorithms and data sources documented in appendices
3. **Robustness:** Multiple specification tests and sensitivity analyses in Section 8
4. **Interdisciplinary:** Integrates finance, machine learning, and game theory

## Final Status

**✓ READY FOR JMLR SUBMISSION**

The manuscript is complete, properly formatted, and ready for upload to the Journal of Machine Learning Research (JMLR) submission system.

---

**Next Steps:**
1. Navigate to JMLR submission portal: https://jmlr.org/
2. Create/login to author account
3. Submit manuscript: **main.pdf**
4. Provide author information with affiliation (KAIST)
5. Add 7 keywords for indexing
6. Upload supplementary material if needed

---
