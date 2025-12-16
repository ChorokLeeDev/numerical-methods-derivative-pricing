# Phase 3C: Internal Review Guide - Comprehensive Checklist

**Duration**: 1 week (Dec 20-27, 2025)
**Goal**: Verify manuscript quality, consistency, and readiness for final formatting

---

## Review Framework

This internal review will verify:
1. **Structural Integrity** - Sections flow logically
2. **Mathematical Rigor** - All theorems/proofs are correct
3. **Empirical Consistency** - Results match across sections
4. **Writing Quality** - Publication-ready prose
5. **Cross-References** - All citations, figures, tables correct
6. **Integration** - Three contributions properly connected

---

## Part 1: Manuscript Structure Review

### 1.1 Section-by-Section Content Verification

**Section 1: Introduction**
- [ ] Opening hook is compelling and motivates the problem
- [ ] 3 gaps clearly articulated
- [ ] 3 solutions previewed
- [ ] Notation introduced and explained
- [ ] Roadmap tells reader what to expect
- [ ] No forward references to undefined concepts

**Section 2: Related Work**
- [ ] Covers 4 literature streams (crowding, DA, CP, tail risk)
- [ ] Each contribution's novelty clearly contrasted
- [ ] 50+ citations properly formatted
- [ ] Positioning vs. existing work is clear
- [ ] Summary table compares approaches
- [ ] No strawmanning of prior work

**Section 3: Background**
- [ ] Financial notation is self-contained (no undefined variables)
- [ ] Factor definitions (mechanical vs. judgment) are explicit
- [ ] Game theory section is accessible
- [ ] MMD explanation is clear
- [ ] Conformal prediction foundation is complete
- [ ] Notation table (3.5) is comprehensive

**Section 4: Game Theory**
- [ ] Model setup is clearly motivated
- [ ] Derivation steps are logical and justified
- [ ] Theorems 1, 2, 3 are formally stated
- [ ] Assumptions are explicit
- [ ] Proof sketches give intuition (full proofs in Appendix A)
- [ ] Comparative statics section explains results
- [ ] Bridge to empirical validation is clear

**Section 5: US Empirical**
- [ ] Data description matches Appendix D
- [ ] Crowding proxy is well-motivated
- [ ] Model fitting methodology is clear
- [ ] Table 4 results are clearly explained
- [ ] Heterogeneity test (Theorem 7) is properly detailed
- [ ] OOS validation is convincing (55% R² is strong for finance)
- [ ] Sub-period analysis shows stability
- [ ] No look-ahead bias in cross-validation

**Section 6: Domain Adaptation**
- [ ] Problem formulation is clear
- [ ] Regime shift problem is well-articulated
- [ ] Temporal-MMD algorithm is explained
- [ ] Table 7 results are properly interpreted
- [ ] Theorem 5 proof sketch is intuitive
- [ ] Connection to game theory is made explicit
- [ ] 7 markets validation is comprehensive

**Section 7: Tail Risk & CW-ACI**
- [ ] Crash prediction model is well-motivated
- [ ] Table 8 results are clearly presented
- [ ] CW-ACI algorithm is step-by-step
- [ ] Theorem 6 coverage guarantee is explained
- [ ] Portfolio hedging application is realistic
- [ ] Table 9 & 10 results are properly interpreted
- [ ] Risk management implications are clear

**Section 8: Robustness**
- [ ] Robustness tests are comprehensive
- [ ] Alternative specifications are fairly compared
- [ ] Limitations are honestly discussed
- [ ] No overstatement of results
- [ ] Generalization to other assets is shown
- [ ] Future work is identified

**Section 9: Conclusion**
- [ ] Recaps 3 contributions accurately
- [ ] Impact is well-articulated
- [ ] Limitations are acknowledged
- [ ] Future directions are promising
- [ ] Closing remarks are inspiring

### 1.2 Logical Flow & Narrative Arc

- [ ] §1 → §2: Motivation flows to literature review
- [ ] §2 → §3: Gaps motivate background setup
- [ ] §3 → §4: Background enables game theory
- [ ] §4 → §5: Theory validated empirically on US data
- [ ] §5 → §6: US insights transferred globally
- [ ] §6 → §7: Transfer enables risk management application
- [ ] §7 → §8: Results robustness tested
- [ ] §8 → §9: Synthesis and future work

---

## Part 2: Mathematical Rigor Verification

### 2.1 Theorem Verification

**Theorem 1** (Appendix A)
- [ ] Statement is clear and self-contained
- [ ] Assumptions are explicit (A1-A4)
- [ ] Proof structure is logical
- [ ] Key steps are justified
- [ ] Conclusion follows from proof
- [ ] Economic interpretation is provided
- [ ] Section 4.3 correctly references this

**Theorem 2** (Appendix A)
- [ ] Statement gives 3 properties of decay rate
- [ ] Proof sketch matches full proof
- [ ] Comparative statics are correct
- [ ] Section 4.4 discusses implications

**Theorem 3** (Appendix A)
- [ ] Statement is precise (λ_J > λ_M)
- [ ] Assumptions (B1-B3) are reasonable
- [ ] Proof logic is sound
- [ ] Heterogeneity interpretation is clear
- [ ] Section 5 empirically validates this
- [ ] (This is "Theorem 7" in empirical context—terminology is consistent)

**Theorem 5** (Appendix B)
- [ ] Statement gives transfer bound
- [ ] Proof shows regime conditioning tightens bound
- [ ] Section 6 correctly references this
- [ ] Transfer efficiency results validate bound

**Theorem 6** (Appendix C)
- [ ] Statement gives coverage guarantee
- [ ] Conditional independence assumption is stated
- [ ] Lemma C.1 (exchangeability) supports main result
- [ ] Section 7 correctly applies this
- [ ] Conditional independence is verified empirically

### 2.2 Supporting Results

- [ ] Lemma C.1: Exchangeability preservation (explained correctly)
- [ ] Proposition B.1: MMD convergence (appropriate for appendix)
- [ ] Proposition C.1: Prediction set widths (intuitively sensible)
- [ ] Proposition C.2: Computational complexity (correct analysis)

### 2.3 Mathematical Notation Consistency

**Key Notation**:
- [ ] $K_i$ always means profitability scale
- [ ] $\lambda_i$ always means decay rate
- [ ] $C_i(t)$ always means crowding level
- [ ] $\alpha_i(t)$ always means alpha at time t
- [ ] Subscripts consistent (judgment J vs mechanical M)
- [ ] All variables defined before use
- [ ] No notation overloading

**Equation Consistency**:
- [ ] $\alpha(t) = K/(1+\lambda t)$ stated consistently
- [ ] MMD definition consistent across §6, Appendix B
- [ ] CW-ACI algorithm matches Algorithms C.1-C.4
- [ ] All mathematical statements are precise

---

## Part 3: Empirical Results Consistency

### 3.1 Within-Section Consistency

**Section 5 (US Empirical)**:
- [ ] Table 4 results are cited in text correctly
- [ ] Judgment λ = 0.173 ± 0.025 matches Table 4
- [ ] Mechanical λ = 0.072 ± 0.010 matches Table 4
- [ ] OOS R² = 0.55 stated as average (check: range is 0.45-0.63)
- [ ] Heterogeneity test p-value matches stated significance

**Section 6 (Domain Adaptation)**:
- [ ] Table 7 results correctly interpreted
- [ ] 43% baseline, 57% MMD, 64% Temporal-MMD (check against table)
- [ ] 65% average TE is correct (verify from Table 7)

**Section 7 (Tail Risk)**:
- [ ] Table 8 AUC = 0.833 for ensemble (check stacking result)
- [ ] Crowding is 3rd most important feature (check SHAP values)
- [ ] Table 9: Sharpe 0.67 → 1.03 (verify from backtest)
- [ ] Table 10: 60-70% loss reduction verified for crashes

### 3.2 Cross-Section Consistency

- [ ] Section 5 heterogeneity test is same as "Theorem 7"
- [ ] Section 6 transfer efficiency matches transfer bound
- [ ] Section 7 hedging uses correct CW-ACI algorithm
- [ ] All three contributions are independent (no circular logic)

### 3.3 Data Consistency with Appendix D

- [ ] FF factor data: 1963-2024 stated consistently
- [ ] 754 months (61 years × 12) is correct
- [ ] 7 countries listed in both §6 and Appendix D
- [ ] Crowding proxy definition matches §5 and Appendix D
- [ ] Feature count (70 features) matches Appendix D

---

## Part 4: Citation & Reference Verification

### 4.1 Citation Completeness

- [ ] 50+ citations are present
- [ ] All cited papers have (Author, Year) format
- [ ] No missing citations for major claims
- [ ] Key papers cited multiple times where appropriate:
  - [ ] Hua & Sun (crowding)
  - [ ] DeMiguel et al. (crowding)
  - [ ] He et al. (domain adaptation)
  - [ ] Angelopoulos & Bates (conformal prediction)
  - [ ] Fantazzini (conformal + finance)

### 4.2 Figure & Table References

**Figures Referenced**:
- [ ] Figures 1-21 are mentioned (verify count in sections)
- [ ] Each figure is referenced in text before/after
- [ ] Figure captions would include:
  - [ ] Figure 1: Factor crowding over time
  - [ ] Figures 8-12: Phase 2 results (tables/figures from empirical work)
  - [ ] Figures 13-21: Additional analysis figures

**Tables Referenced**:
- [ ] Table 1: Notation summary (Section 3)
- [ ] Table 2: Parameter estimates (Section 5)
- [ ] Table 3: Literature comparison (Section 2)
- [ ] Table 4: Estimated decay parameters (Section 5)
- [ ] Table 5: OOS R² by period (Section 5)
- [ ] Table 6: Sub-period analysis (Section 5)
- [ ] Table 7: Transfer efficiency (Section 6)
- [ ] Table 8: Crash prediction (Section 7)
- [ ] Table 9: Portfolio hedging (Section 7)
- [ ] Table 10: Crash events (Section 7)

---

## Part 5: Writing Quality & Clarity

### 5.1 Prose Quality

- [ ] No jargon without definition
- [ ] Sentences are clear and concise
- [ ] Paragraphs have topic sentences
- [ ] Transitions between paragraphs are smooth
- [ ] No contradictions within sections
- [ ] No circular logic
- [ ] Technical language is precise

### 5.2 Tone & Voice

- [ ] Tone is professional throughout
- [ ] Balance between technical and accessible
- [ ] Author confidence is appropriate (not overstated)
- [ ] Limitations are discussed honestly
- [ ] No excessive hedging ("might," "could," etc.)

### 5.3 Common Issues to Check

- [ ] No "we" when first author is anonymous (check JMLR style)
- [ ] Consistent use of past/present tense in results sections
- [ ] No unexplained abbreviations
- [ ] Acronyms defined on first use
- [ ] No repetition of the same point in different words
- [ ] No outdated references (all post-2015 are current)

### 5.4 Typo & Grammar Check

- [ ] Spelling is consistent (e.g., "grey" vs "gray")
- [ ] No grammatical errors
- [ ] Punctuation is correct
- [ ] No missing spaces around citations
- [ ] Equation formatting is consistent

---

## Part 6: Integration & Coherence

### 6.1 Three Contributions Are Connected

**Check narrative arc**:
- [ ] Game theory explains *why* crowding matters (causal mechanism)
- [ ] Domain adaptation shows *how* to transfer insights (transfer learning)
- [ ] Conformal prediction shows *how* to use signals (risk management)
- [ ] Each builds on previous (not separate papers)

### 6.2 Limitations Are Balanced

- [ ] Section 8.6 acknowledges limitations of each component
- [ ] No overclaiming of results
- [ ] Honest about data limitations (proxies, etc.)
- [ ] Realistic about practical applicability
- [ ] Appropriate suggestions for future work

### 6.3 Reproducibility

- [ ] Appendix D provides complete data documentation
- [ ] Appendix E provides algorithm pseudocode
- [ ] Python code snippets are shown
- [ ] All parameters and hyperparameters are specified
- [ ] Readme would explain repo structure

---

## Review Checklist Summary

### Must-Have (Critical)
- [ ] All theorems are correctly stated
- [ ] All proofs are complete and rigorous
- [ ] All empirical results are consistent
- [ ] No mathematical errors
- [ ] No contradictions
- [ ] 50+ citations present

### Should-Have (Important)
- [ ] All figures/tables referenced
- [ ] Writing is publication-ready
- [ ] Notation is consistent
- [ ] Sections flow logically
- [ ] Limitations are discussed

### Nice-to-Have (Polish)
- [ ] Creative examples
- [ ] Compelling narrative
- [ ] Excellent writing style
- [ ] Professional presentation

---

## Review Scoring

| Section | Critical Issues | Important Issues | Nice-to-Have | Status |
|---------|---|---|---|---|
| 1. Introduction | __ | __ | __ | |
| 2. Related Work | __ | __ | __ | |
| 3. Background | __ | __ | __ | |
| 4. Game Theory | __ | __ | __ | |
| 5. Empirical | __ | __ | __ | |
| 6. Domain Adapt | __ | __ | __ | |
| 7. Tail Risk | __ | __ | __ | |
| 8. Robustness | __ | __ | __ | |
| 9. Conclusion | __ | __ | __ | |
| **Appendices A-F** | __ | __ | __ | |

---

## How to Conduct Review

**Approach**:
1. Read each section thoroughly (2-3 times)
2. Mark issues as you find them
3. Check one category at a time (structure → math → empirics → writing)
4. Verify all cross-references with master checklist
5. Create list of revisions needed
6. Execute fixes in priority order (critical first)

**Timeline**:
- Day 1-2: §1-3 (Motivation & Background)
- Day 2-3: §4-5 (Game Theory & Empirics)
- Day 3-4: §6-7 (Domain Adaptation & Risk Management)
- Day 4-5: §8-9 (Robustness & Conclusion)
- Day 5-6: Appendices A-F (Proofs & Details)
- Day 6-7: Final verification & fixes

---

**Note**: This is a comprehensive framework for systematic internal review. Use this to ensure manuscript quality before final polish and submission.

