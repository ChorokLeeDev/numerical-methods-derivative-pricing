# JMLR Submission Roadmap - Factor Crowding Research
## Target: September 30, 2026 | Status: Planning Complete ✅

**Author:** Chorok Lee (KAIST)
**GitHub:** https://github.com/ChorokLeeDev/factor-crowding
**Plan File:** `/Users/i767700/.claude/plans/cheerful-juggling-honey.md`

---

## 📍 Current Status (December 16, 2025)

### ✅ What's Complete
| Project | Status | Key Results |
|---------|--------|-------------|
| **Factor Crowding (ICAIF)** | 95% | Game theory model, RF AUC 0.623, 19 experiments |
| **Global Crowding (KDD)** | 100% | Temporal-MMD, 4 domains, KDD paper drafted |
| **Conformal Prediction (ICML)** | 95% | CW-ACI 89.8% coverage, 15% variance reduction |

### ⚠️ What Needs Work
1. Conformal coverage: 85.6% → 89.8% ✓ (FIXED in ICML research)
2. Formal proofs: Theorems 1-7 need rigorous proofs
3. Heterogeneity test: Mechanical vs judgment (qualitative → quantitative)
4. Integration: Three papers → One unified JMLR narrative

---

## 🎯 Strategy Decision

**Chosen:** **JMLR Unified** (50 pages, September 2026)

**Rationale:**
- Time available: 9 months ✓
- Emphasis: Equal weight on all three contributions ✓
- Single high-impact publication > 3 separate papers
- Material 95%+ ready, just needs integration

**Expected Outcome:**
- Acceptance probability: 60-70%
- Backup venues: TMLR, Annals of Applied Statistics
- Can still split if rejected

---

## 📅 9-Month Timeline (Phases 1-6)

```
Weeks 1-4   │ PHASE 1: Integration & Setup
Weeks 5-12  │ PHASE 2: Empirical Enhancement
Weeks 13-24 │ PHASE 3: Paper Writing (50 pages)
Weeks 25-28 │ PHASE 4: Theoretical Proofs
Weeks 29-32 │ PHASE 5: Code Release & Reproducibility
Weeks 33-36 │ PHASE 6: Final Submission

🎯 Submit: September 30, 2026
```

**Estimated Effort:** 220 hours total (6 hours/week average)

---

## 📋 Phase Checklists

### PHASE 1: Integration & Setup (Weeks 1-4) ⏳

**Week 1: Repository Setup**
- [ ] Create `/Users/i767700/Github/quant/research/jmlr_unified/` directory
- [ ] Download JMLR LaTeX template
- [ ] Create main.tex skeleton (9 sections)
- [ ] Merge 3 bibliographies → `references.bib` (100+ entries)
- [ ] Create unified notation table (α, K, λ, c, r, N, γ)

**Week 2: Figure & Table Planning**
- [ ] Select 20 figures from 29 candidates (importance × clarity)
- [ ] Unify figure styles (Times New Roman 10pt, 300 DPI, colorblind palette)
- [ ] Create `experiments/jmlr_figures.py` script
- [ ] Design 12 tables

**Week 3: Code Integration**
- [ ] Create `src/unified_analysis.py` (orchestrates 3 methods)
- [ ] Verify reproducibility: Run all 30+ experiments
- [ ] Create `requirements.txt` (unified dependencies)
- [ ] Write README for code structure

**Week 4: Theoretical Enhancement - Part 1**
- [ ] Formalize Theorem 1 (Hyperbolic decay) with regularity conditions
- [ ] State Lemmas 1-2 (Nash uniqueness, entry equilibrium)
- [ ] Begin proof sketches (full proofs in Phase 4)

**Deliverables:**
- ✅ Compilable LaTeX template
- ✅ 20 publication-ready figures
- ✅ Reproducible code pipeline
- ✅ Theorems 1-6 formally stated

---

### PHASE 2: Empirical Enhancement (Weeks 5-12) ⏳

**Week 5-6: Feature Importance Analysis**
- [ ] Implement SHAP values (top 20 features)
- [ ] Feature ablation: Returns-only, Vol-only, Correlation-only, Crowding-only
- [ ] Generate Figure 8: SHAP summary plot
- [ ] Create Table 5: Feature groups comparison

**Week 7-8: Heterogeneity Statistical Test**
- [ ] Mixed-effects regression: R² ~ factor_type + (1|factor)
- [ ] Bootstrap p-value (1000 samples)
- [ ] NEW Theorem 7: Heterogeneous decay rates (mechanical vs judgment)
- [ ] Generate Figure 9: Factor type comparison

**Week 9-10: Extended Validation**
- [ ] Pre-sample validation (1980-2000)
- [ ] Different crash thresholds (5%, 10%, 15%)
- [ ] Alternative crowding signals (ETF flows, analyst coverage)
- [ ] Create Table 6: Robustness checks

**Week 11-12: Ensemble & Portfolio Extensions**
- [ ] Stacked ensemble (RF + XGBoost + NN)
- [ ] Multi-factor portfolio tail risk (2^8 = 256 combinations)
- [ ] CW-ACI for portfolio VaR
- [ ] Generate Figure 10: Ensemble comparison

**Deliverables:**
- ✅ Section 4.5 extended (feature importance)
- ✅ Section 4.3 formalized (heterogeneity with p-values)
- ✅ Appendix D: Extended validation (5 pages)
- ✅ Section 8.4: Future work (portfolio prelims)

---

### PHASE 3: Paper Writing (Weeks 13-24) ⏳

**Week 13-14: Introduction + Background (9 pages)**
- [ ] Section 1: Introduction (4 pages)
  - Hook: "Momentum 10% in 1990s → 2% today. Why?"
  - Three questions framework
  - Preview of results (3 findings per question)
- [ ] Section 2: Background & Related Work (5 pages)
  - Post-publication decay, game theory, domain adaptation, conformal prediction

**Week 15-16: Game Theory + US Validation (14 pages)**
- [ ] Section 3: Game-Theoretic Model (6 pages)
  - Setup, Nash equilibrium, dynamic entry, alternatives
- [ ] Section 4: Empirical Validation: US Equity Factors (8 pages)
  - Model fit, heterogeneity, OOS, ML detection, trading

**Week 17-18: Tail Risk + Global Adaptation (13 pages)**
- [ ] Section 5: Tail Risk Prediction (5 pages)
  - Crash definition, OOS test, heterogeneous results (Reversal ≠ Momentum)
- [ ] Section 6: Global Domain Adaptation (8 pages)
  - Temporal-MMD, theory, global finance, multi-domain

**Week 19-20: Conformal UQ + Discussion (13 pages)**
- [ ] Section 7: Distribution-Free Uncertainty Quantification (8 pages)
  - ACI baseline, CW-ACI breakthrough, theory, empirical validation
- [ ] Section 8: Synthesis & Discussion (5 pages)
  - Unified framework, practical implications, limitations, future work
- [ ] Section 9: Conclusion (2 pages)

**Week 21-22: Appendices (15 pages)**
- [ ] Appendix A: Proofs (Theorems 1-7)
- [ ] Appendix B: Data description
- [ ] Appendix C: Hyperparameters
- [ ] Appendix D: Extended validation
- [ ] Appendix E: Feature importance
- [ ] Appendix F: Regime detection methods

**Week 23-24: Revision & Polish**
- [ ] Read-through for flow
- [ ] Notation consistency check
- [ ] Reference verification (100+ citations)
- [ ] Figure/table captions detailed
- [ ] Abstract refinement (250 words)
- [ ] Grammar/spell check

**Deliverables:**
- ✅ Complete draft v1 (50 pages main + 15 pages appendix)
- ✅ 20 figures integrated
- ✅ 12 tables complete
- ✅ 100+ references cited

---

### PHASE 4: Theoretical Proofs (Weeks 25-28) ⏳

**Week 25-26: Game Theory Proofs**
- [ ] Lemma 1: Nash equilibrium uniqueness
- [ ] Theorem 1: Hyperbolic decay derivation
- [ ] Corollary 1: Half-life formula
- [ ] Theorem 7: Heterogeneous decay rates
- [ ] Appendix A.1-A.2 (4 pages)

**Week 27: Domain Adaptation Proofs**
- [ ] Theorem 2: Variance reduction under regime imbalance
- [ ] Theorem 3: Target risk bound (extend Ben-David et al.)
- [ ] Appendix A.3 (3 pages)

**Week 28: Conformal Prediction Proofs**
- [ ] Theorem 4: Marginal coverage preservation (CW-ACI)
- [ ] Theorem 5: Coverage uniformity improvement
- [ ] Theorem 6: Regret bound
- [ ] Appendix A.4-A.5 (4 pages)

**Deliverables:**
- ✅ Appendix A: Complete proofs (11 pages total)
- ✅ All theorems rigorously proven

---

### PHASE 5: Code Release & Reproducibility (Weeks 29-32) ⏳

**Week 29-30: Code Cleanup**
- [ ] Refactor for clarity (modularize, type hints, docstrings)
- [ ] Create pip-installable package: `pip install factor-crowding`
- [ ] Write comprehensive README
- [ ] Add license (MIT or Apache 2.0)

**Week 31: Reproducibility Notebook**
- [ ] Create `notebooks/reproduce_jmlr_figures.ipynb`
- [ ] Reproduce all 20 figures
- [ ] Reproduce all 12 tables
- [ ] Add narrative explaining results
- [ ] Test runtime (<4 hours on laptop)

**Week 32: Docker Container (Optional)**
- [ ] Create Dockerfile
- [ ] Test on clean Ubuntu environment
- [ ] Push to DockerHub
- [ ] Instructions in README

**Deliverables:**
- ✅ https://github.com/ChorokLeeDev/factor-crowding updated
- ✅ Complete reproducibility package
- ✅ Docker container (optional)

---

### PHASE 6: Final Submission (Weeks 33-36) ⏳

**Week 33-34: Internal Review**
- [ ] Share draft with advisor/colleagues
- [ ] Incorporate feedback
- [ ] Revise draft v2
- [ ] Check JMLR author guidelines
- [ ] Verify LaTeX template compliance

**Week 35: Final Checks**
- [ ] Abstract: 250 words, self-contained ✓
- [ ] Introduction: Motivates all three questions ✓
- [ ] Conclusion: Answers all three questions ✓
- [ ] Figures: All referenced in text ✓
- [ ] Tables: All captions detailed ✓
- [ ] References: All complete (no "et al." in bib) ✓
- [ ] Appendix: All theorems proven ✓
- [ ] Code: Public repo link in paper ✓
- [ ] Compile: No LaTeX errors/warnings ✓
- [ ] PDF: Check fonts, figures render correctly ✓

**Week 36: Submit! (September 30, 2026)**
- [ ] JMLR submission portal account
- [ ] Upload PDF (main + appendix)
- [ ] Upload LaTeX source files
- [ ] Cover letter (1 page, suggest 5 reviewers)
- [ ] Supplementary materials (code, data statement)
- [ ] Submit! 🎉

**Deliverables:**
- ✅ Submission-ready PDF
- ✅ Cover letter
- ✅ Supplementary materials

---

## 🚀 This Week's Focus (Week 1, Dec 16-22, 2025)

### Day 1 (Today - December 16)
1. ✅ Create `/Users/i767700/Github/quant/research/jmlr_unified/` directory
2. ✅ Download JMLR LaTeX template from https://www.jmlr.org/format/format.html
3. ✅ Read JMLR author guidelines

### Day 2 (December 17)
1. Set up directory structure:
   ```bash
   mkdir -p paper/{figures,tables}
   mkdir -p src/{game_theory,domain_adaptation,conformal}
   mkdir -p experiments/jmlr
   mkdir -p notebooks
   mkdir -p data/processed
   ```
2. Copy code from 3 sub-projects
3. Create `main.tex` skeleton

### Day 3 (December 18)
1. Merge bibliographies:
   - `/Users/i767700/Github/quant/research/factor_crowding/paper/references.bib`
   - `/Users/i767700/Github/quant/research/kdd2026_global_crowding/paper/references.bib`
   - `/Users/i767700/Github/quant/research/icml2026_conformal/paper/references.bib`
2. Remove duplicates
3. Standardize citation format (author-year)

### Day 4 (December 19)
1. Create unified notation table (Table 1)
2. Write Section 1.1 (Introduction - first 2 pages)
3. Outline all 9 sections with subsection headers

### Day 5 (December 20)
1. Generate figure selection matrix (29 → 20)
2. Unify figure styles (fonts, colors, DPI)
3. Create `experiments/jmlr_figures.py`

### Day 6 (December 21)
1. Run all 30+ experiments to verify reproducibility
2. Document any issues
3. Create `requirements.txt` with versions

### Day 7 (December 22 - End of Week 1)
1. Review progress
2. Compile LaTeX (should compile with placeholder text)
3. Plan Week 2 tasks

**Estimated Time:** 12-15 hours (2 hours/day)

---

## 📊 Progress Tracking

### Overall Progress: 0% (0/220 hours)

| Phase | Progress | Time Spent | Time Remaining |
|-------|----------|------------|----------------|
| Phase 1: Integration | 0% | 0h / 40h | 40h |
| Phase 2: Empirical | 0% | 0h / 60h | 60h |
| Phase 3: Writing | 0% | 0h / 80h | 80h |
| Phase 4: Proofs | 0% | 0h / 20h | 20h |
| Phase 5: Code | 0% | 0h / 16h | 16h |
| Phase 6: Submission | 0% | 0h / 4h | 4h |

**Update this table weekly!**

---

## 📁 Key File Locations

### Current Research (Source Material)
```
/Users/i767700/Github/quant/research/
├── factor_crowding/               # US factors, game theory, ML
│   ├── paper/icaif2026_factor_crowding.tex
│   ├── src/{crowding_ml.py, conformal_ml.py, features.py}
│   └── experiments/{01-19}
├── kdd2026_global_crowding/       # Global, Temporal-MMD
│   ├── paper/kdd2026_temporal_mmd.tex
│   ├── src/models/{temporal_mmd.py, dann.py, cdan.py}
│   └── experiments/{01-12}
└── icml2026_conformal/            # CW-ACI, uncertainty quantification
    ├── paper/icml2026_crowding_conformal.tex
    ├── src/{crowding_aware_conformal.py, theory.py}
    └── experiments/{01-11}
```

### JMLR Unified (Target)
```
/Users/i767700/Github/quant/research/jmlr_unified/
├── paper/
│   ├── main.tex                   # 50 pages
│   ├── appendix_proofs.tex        # 15 pages
│   ├── references.bib             # 100+ entries
│   └── figures/                   # 20 figures
├── src/
│   ├── unified_analysis.py        # Orchestrates all 3 methods
│   ├── game_theory/
│   ├── domain_adaptation/
│   └── conformal/
├── experiments/
│   ├── jmlr_figures.py            # Generate all figures
│   ├── 20_feature_importance.py
│   └── 21_heterogeneity_test.py
├── notebooks/
│   └── reproduce_jmlr_figures.ipynb
└── README.md
```

### GitHub
- **Public Repo:** https://github.com/ChorokLeeDev/factor-crowding
- **Update frequency:** After each phase completion

---

## 🎯 Success Criteria

### Minimum Viable Paper (JMLR Acceptance)
- [x] Novel theoretical contribution (game theory → hyperbolic decay)
- [x] Comprehensive empirical validation (62 years, 8 factors, 7 regions, 4 domains)
- [x] Rigorous methodology (walk-forward, no lookahead, statistical tests)
- [x] Distribution-free guarantees (conformal prediction)
- [ ] Formal proofs (Theorems 1-7)
- [ ] Heterogeneity formalized (statistical test)
- [x] Code release (ready, needs documentation)

### Stretch Goals (Increase Impact)
- [ ] Deployed system (live dashboard)
- [ ] Multi-asset extension (crypto, commodities)
- [ ] Causal inference validation (IV estimation of λ)
- [ ] Portfolio applications (multi-factor tail risk)

---

## 🆘 Emergency Contacts & Resources

### JMLR Submission
- **Portal:** https://jmlr.org/author-info.html
- **Template:** https://www.jmlr.org/format/format.html
- **Guidelines:** https://www.jmlr.org/author-info.html#submission

### Reference Papers
- **Game Theory:** DeMiguel et al. (2021), Kyle (1985)
- **Domain Adaptation:** Long et al. (2015), Ganin et al. (2016)
- **Conformal Prediction:** Gibbs & Candes (2021), Vovk et al. (2005)
- **Factor Investing:** McLean & Pontiff (2016), Fama & French (2015)

### Backup Venues (if JMLR rejects)
1. **TMLR** (Transactions on Machine Learning Research) - faster review
2. **Annals of Applied Statistics** - econometrics focus
3. **KDD 2027** + **ICML 2027** + **AAAI 2027** (split into 3 papers)

---

## 📝 Weekly Review Template

**Week X (Dates):**

**Completed:**
- [ ] Task 1
- [ ] Task 2

**Blocked:**
- Issue 1: Description + resolution plan

**Next Week Focus:**
- Priority 1
- Priority 2

**Hours Spent:** X hours
**Cumulative:** Y / 220 hours

**Status:** On Track / Needs Attention / Blocked

---

## 🎉 Milestones

- [ ] **Week 4:** LaTeX compiles, 20 figures selected
- [ ] **Week 12:** All experiments complete, empirical validation done
- [ ] **Week 24:** Complete draft v1 (50 pages)
- [ ] **Week 28:** All proofs complete
- [ ] **Week 32:** Code released publicly
- [ ] **Week 36:** JMLR submission! 🚀

---

**Last Updated:** December 16, 2025
**Plan File:** `/Users/i767700/.claude/plans/cheerful-juggling-honey.md`
**Next Review:** December 23, 2025 (End of Week 1)
