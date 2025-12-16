# JMLR Submission Checklist

**Paper Title**: Not All Factors Crowd Equally: A Game-Theoretic Model of Alpha Decay with Global Transfer and Risk Management

**Target Submission Date**: January 20, 2026

**Submission Deadline**: September 30, 2026

---

## Pre-Submission Quality Assurance

### Content Verification
- [ ] **Manuscript Complete**: All 9 sections written and formatted
- [ ] **Theorems**: All 6 theorems included with complete proofs
  - [ ] Theorem 1: Existence and uniqueness of Nash equilibrium
  - [ ] Theorem 2: Hyperbolic decay properties
  - [ ] Theorem 3: Heterogeneous decay across factor types
  - [ ] Theorem 5: Domain adaptation transfer bound
  - [ ] Theorem 6: Conformal prediction coverage guarantee
  - [ ] All supporting lemmas included
- [ ] **Algorithms**: All 8 algorithms with complete pseudocode
- [ ] **Empirical Results**: All tables and figures referenced
- [ ] **Appendices**: All 6 appendices complete
  - [ ] Appendix A: Game theory proofs
  - [ ] Appendix B: Domain adaptation theory
  - [ ] Appendix C: Conformal prediction proofs
  - [ ] Appendix D: Data documentation
  - [ ] Appendix E: Algorithm pseudocode
  - [ ] Appendix F: Supplementary robustness tests

### Technical Formatting
- [ ] **LaTeX Compilation**: main.pdf compiles without errors
- [ ] **Bibliography**: All 40+ references properly formatted in BibTeX
- [ ] **Cross-References**: All \label{} and \ref{} commands working
- [ ] **Notation**: All mathematical notation consistent via macros.tex
- [ ] **Page Count**: Manuscript within 55-70 pages (with figures/tables)
- [ ] **Abstract**: 150-250 words (typically ~240 words)
- [ ] **Keywords**: 5-6 keywords listed

### Writing Quality
- [ ] **Prose**: Publication-ready writing (no grammatical errors)
- [ ] **Clarity**: Explanations clear and accessible to ML/finance audience
- [ ] **Organization**: Logical flow from motivation through results
- [ ] **Consistency**: Term definitions consistent throughout
- [ ] **Tone**: Professional and objective throughout
- [ ] **Spelling**: No typos or spelling errors (use spell-check)

### Reproducibility
- [ ] **Data Availability Statement**: Prepared and ready
  - [ ] Kenneth French factor data source documented
  - [ ] International data sources listed
  - [ ] Instructions for obtaining data provided
- [ ] **Code Availability**: GitHub repository ready
  - [ ] All Python source code documented
  - [ ] Jupyter notebooks functional and executable
  - [ ] Requirements.txt with all dependencies
  - [ ] Installation instructions clear
- [ ] **Random Seeds**: All random number generation seeded for reproducibility
- [ ] **Parameter Settings**: All hyperparameters documented

---

## JMLR Portal Submission

### Account Setup
- [ ] **Create JMLR Account**: Register at https://jmlr.org
- [ ] **Verify Email**: Confirm email address
- [ ] **Complete Profile**: Full author information entered

### Document Preparation
- [ ] **Main PDF**: Generated and error-checked (main.pdf)
- [ ] **Supplementary Files**: Organized
  - [ ] Code files (src/)
  - [ ] Notebooks (notebooks/)
  - [ ] Data files (data/ - or link if too large)
  - [ ] Figure sources (high-resolution PDFs/EPS)
- [ ] **File Naming**: Use clear, descriptive names
- [ ] **File Sizes**: Check within portal limits
  - [ ] Main PDF: typically < 10 MB
  - [ ] Supplementary: all files < 50 MB total

### Cover Letter
- [ ] **Cover Letter Written**: 1-2 page letter prepared
- [ ] **Components Included**:
  - [ ] Paper title
  - [ ] Brief summary of contributions
  - [ ] Statement on originality and novelty
  - [ ] Significance for JMLR audience
  - [ ] Information about prior/concurrent submissions
  - [ ] Conflict of interest statement
- [ ] **Professional Tone**: Polite and respectful
- [ ] **Proofread**: No errors in cover letter

### Author Information
- [ ] **All Authors**: Names and affiliations listed
- [ ] **Corresponding Author**: Clearly indicated
- [ ] **Contact Information**: Email addresses provided
- [ ] **Author Biographies**: 100-150 words each (optional but recommended)
- [ ] **Author Pictures**: High-resolution headshots (optional)

### Conflict of Interest
- [ ] **Potential Conflicts Listed**: Any relevant financial or personal conflicts
- [ ] **Funding Sources**: All research funding disclosed
- [ ] **Data Sources**: Kenneth French data is public (no conflict)
- [ ] **Prior Work**: Any related published/submitted papers noted

### Special Statements
- [ ] **Data Availability**: Statement prepared
- [ ] **Code Availability**: GitHub repository URL provided
- [ ] **Reproducibility Statement**: How to reproduce results documented
- [ ] **Ethical Considerations**: Any relevant ethical issues addressed

---

## Submission Day Checklist

### Final Checks (2 hours before submission)
- [ ] **PDF Test**: Open and verify main.pdf renders correctly
- [ ] **All Pages**: Confirm all pages present (should be 55-70 pages)
- [ ] **Formatting**: Check headers, footers, page numbers
- [ ] **Figures**: Verify all figures display correctly
- [ ] **Tables**: Verify all tables display correctly
- [ ] **References**: Spot-check several bibliography entries

### Portal Upload
- [ ] **Log In**: Access JMLR portal with credentials
- [ ] **Create Submission**: Start new submission process
- [ ] **Enter Metadata**: Title, keywords, abstract
- [ ] **Upload PDF**: main.pdf successfully uploaded
- [ ] **Upload Supplementary**: All code/data files uploaded
  - [ ] GitHub URL provided or files attached
  - [ ] Format verified as readable
- [ ] **Enter Author Info**: All authors properly entered
- [ ] **Review Summary**: Proofread all entered information

### Final Submission
- [ ] **Cover Letter**: Uploaded or pasted in text field
- [ ] **Author Biographies**: Uploaded or included
- [ ] **Conflict Statement**: Completed
- [ ] **Data Availability**: Statement included
- [ ] **Ethics**: Any required ethics certifications completed
- [ ] **Preview**: Review complete submission one final time
- [ ] **SUBMIT**: Click submit button
- [ ] **Confirmation**: Save confirmation number and email

---

## Post-Submission

### Confirmation
- [ ] **Email Received**: Confirm receipt from JMLR within 24 hours
- [ ] **Confirmation Number**: Record and save
- [ ] **Submission Date**: Note official submission timestamp
- [ ] **Manuscript ID**: Record assigned ID

### Documentation
- [ ] **Save Evidence**: Screenshot/PDF of confirmation
- [ ] **Update GitHub**: Update repository with submission status
- [ ] **Create Release**: Tag GitHub release as "submitted-to-jmlr"
- [ ] **Update Status**: Update README with submission status

### Planning for Review
- [ ] **Review Timeline**: Expect 3-6 months for initial decision
- [ ] **Reviewer Prep**: Be ready with author responses document
- [ ] **Revision Strategy**: Plan potential revisions if required
- [ ] **Supplementary Resources**: Prepare additional materials if needed

---

## Important JMLR Guidelines

### Manuscript Requirements
- **Length**: Typically 15-50 pages for main text; supplementary material as appendices
- **Format**: PDF generated from LaTeX (.tex source files accepted)
- **Submission**: Electronic submission via JMLR portal
- **Language**: English (though international authors welcome)

### Citation Requirements
- **BibTeX**: All references must be in BibTeX format
- **Bibliography Style**: Use `\bibliographystyle{jmlr}` in LaTeX
- **Reference Count**: 40+ references is appropriate for a comprehensive paper

### Ethics and Reproducibility
- **Data Accessibility**: Must provide access to data or explain why not possible
- **Code Availability**: Strongly encouraged (GitHub link acceptable)
- **Reproducibility**: Should be able to reproduce main results with provided code
- **Ethical Approval**: Any required ethics approvals must be documented

### Author Responsibilities
- **Originality**: Must be original, not previously published
- **Current Submissions**: Must disclose if under review elsewhere
- **Copyright**: Authors retain copyright; JMLR gets publication rights
- **Financial Disclosure**: Must disclose any conflicts of interest

---

## Timeline Summary

| Date | Task | Status |
|------|------|--------|
| Dec 16 | Complete LaTeX formatting | ✅ Complete |
| Dec 20 | Fix remaining LaTeX errors | ⏳ In Progress |
| Jan 5 | Final proofreading | ⏳ Pending |
| Jan 10 | GitHub repository finalized | ⏳ Pending |
| Jan 15 | Supplementary materials prepared | ⏳ Pending |
| Jan 16 | Test PDF compilation | ⏳ Pending |
| Jan 17 | Create JMLR account | ⏳ Pending |
| Jan 18 | Prepare cover letter and author info | ⏳ Pending |
| Jan 19 | Final review and QA | ⏳ Pending |
| Jan 20 | **SUBMIT TO JMLR** | ⏳ Pending |

---

## Support Resources

### JMLR Resources
- **JMLR Website**: https://jmlr.org/
- **Submission Portal**: https://jmlr.org/submit/
- **FAQ**: https://jmlr.org/submissions/
- **Contact**: editor@jmlr.org

### LaTeX Resources
- **JMLR Template**: Available on JMLR website
- **Overleaf**: Can use Overleaf for online editing/compilation
- **TeX Documentation**: https://www.latex-project.org/

### Data Sources
- **Fama-French Factors**: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/
- **International Factors**: Kenneth French's library has worldwide data

---

**Last Updated**: December 16, 2025

**Prepared by**: Week 3 Submission Preparation

This checklist ensures all requirements are met for successful JMLR submission.
