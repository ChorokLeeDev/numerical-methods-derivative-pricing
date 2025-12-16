# JMLR Submission Guide - Week 3 Implementation

**Prepared for**: Not All Factors Crowd Equally Paper
**Target Submission**: January 20, 2026
**Deadline**: September 30, 2026

---

## Overview

This guide provides step-by-step instructions for submitting the paper to JMLR using the online submission portal.

### Key Facts About JMLR

- **Publication Model**: Open access, free to readers
- **Review Timeline**: 3-6 months for initial decision (peer review)
- **Acceptance Rate**: ~15-20% (highly selective)
- **Page Limit**: No strict limit, but 15-50 pages typical for main text
- **Rebuttal Process**: Authors can respond to reviewer comments
- **Publication**: Online first, then compiled into volumes

---

## Part 1: Pre-Submission Preparation (Weeks 1-2)

### Step 1: Final Manuscript Review

**Timeline**: Complete by January 10, 2026

```
Task Checklist:
□ Fix all remaining LaTeX errors in PDF compilation
□ Verify all 9 sections appear in correct order
□ Verify all 6 appendices included
□ Check all cross-references (\ref{} commands)
□ Verify all citations in bibliography
□ Proofread for typos and grammatical errors
□ Check figure quality and captions
□ Verify table formatting and alignment
□ Check notation consistency using macros.tex
□ Generate final main.pdf without warnings
```

**Command to compile**:
```bash
cd jmlr_submission
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
# Should produce: main.pdf (55-70 pages, no errors)
```

**Quality checks**:
- Open main.pdf and verify all content appears correctly
- Page count should be 55-70 pages
- Figures should render without distortion
- All references should be present and properly formatted

### Step 2: Prepare Supplementary Materials

**Timeline**: Complete by January 12, 2026

Create organized submission materials:

```
submission_package/
├── main.pdf                          # Main manuscript
├── cover_letter.txt                  # Cover letter (plain text or PDF)
├── author_biographies.txt            # Author bios
├── data_availability_statement.txt   # Data statement
├── conflict_of_interest.txt          # CoI declaration
│
├── supplementary/
│   ├── code/
│   │   ├── src/                     # Python source code
│   │   ├── notebooks/               # Jupyter notebooks
│   │   ├── requirements.txt
│   │   └── README.md
│   │
│   ├── data/
│   │   ├── raw/                     # Raw data (if < 50MB)
│   │   └── processed/               # Processed data
│   │
│   └── results/
│       ├── figures/                 # High-res figures (PDF/EPS)
│       └── tables/                  # Table definitions
│
└── README_SUBMISSION.md              # How to access materials
```

**GitHub Repository**:
- Create public GitHub repository: `https://github.com/USERNAME/factor-crowding-unified`
- Include all code, notebooks, and documentation
- Add clear README with installation and usage instructions
- Tag release as "submitted-to-jmlr-v1"

**File Size Check**:
```bash
# Check file sizes before uploading
du -h main.pdf
du -sh supplementary/

# JMLR limits:
# - Main PDF: < 10 MB ✓
# - All supplementary: < 50 MB ✓
# - GitHub: no limits, but include link instead of upload
```

### Step 3: Write Cover Letter

**Timeline**: Complete by January 15, 2026

Create a professional 1-2 page cover letter including:

1. **Opening**: Paper title and submission statement
2. **Summary**: Highlight three key contributions
3. **Novelty**: Distinguish from prior work (cite 3-4 key papers)
4. **Significance**: Why JMLR readers should care
5. **Statements**:
   - "This manuscript is original and not under review elsewhere"
   - Conflict of interest disclosure
   - Data availability statement
6. **Closing**: Professional sign-off

Use template: `JMLR_COVER_LETTER_TEMPLATE.md`

---

## Part 2: JMLR Portal Submission (Day of Submission)

### Step 1: Create JMLR Account

**Timeline**: 1 week before submission (January 13, 2026)

1. Visit: https://jmlr.org/submit/
2. Click "Register as new user"
3. Enter:
   - Full name
   - Email address
   - Password
   - Institution
   - Field of interest
4. Verify email address (check inbox)
5. Log in and complete profile

**Profile Information**:
- Full name: [Your Name]
- Email: [Your Email] (should be your institutional email)
- Institution: [Your Institution]
- Area: Machine Learning with applications to Finance

### Step 2: Start New Submission

1. Log into JMLR portal
2. Click "Submit Paper"
3. Select submission type: "Regular Paper Submission"

### Step 3: Enter Manuscript Information

**Title**:
```
Not All Factors Crowd Equally: A Game-Theoretic Model of
Alpha Decay with Global Transfer and Risk Management
```

**Authors** (in order of contribution):
```
Author 1
  Name: [Full Name]
  Email: [Email]
  Institution: [Institution]
  Country: [Country]

Author 2
  Name: [Full Name]
  Email: [Email]
  Institution: [Institution]
  Country: [Country]
```

**Designate Corresponding Author**: Select Author 1 (or appropriate author)

### Step 4: Abstract and Keywords

**Abstract**:
- Copy from manuscript (should be 150-250 words)
- Preview how it renders in portal

**Keywords** (5-6):
```
1. Factor Investing
2. Alpha Decay and Crowding
3. Game Theory & Equilibrium
4. Domain Adaptation & Transfer Learning
5. Conformal Prediction & Uncertainty Quantification
6. Portfolio Risk Management
```

### Step 5: Upload Files

**Main Manuscript**:
1. Click "Upload PDF"
2. Select `main.pdf` from your computer
3. Verify upload successful
4. Check file size displayed (should be < 10 MB)

**Supplementary Files**:
1. Option A: **Upload to Portal** (if < 50 MB total)
   - Create ZIP file with all supplementary materials
   - Upload ZIP file to portal
   - Provide README explaining contents

2. Option B: **Provide Link** (recommended)
   - Include GitHub repository URL: https://github.com/USERNAME/factor-crowding-unified
   - Include link in cover letter
   - Specify: "All code, notebooks, and supplementary materials available at GitHub"

**Recommended**: Use GitHub link (cleaner, easier to maintain)

### Step 6: Enter Cover Letter

**Method 1**: Upload PDF file
```bash
# Option 1: Upload pre-written cover letter PDF
# Select file: cover_letter.pdf
```

**Method 2**: Copy-paste text
```
- Click "Paste Cover Letter" text field
- Copy and paste cover_letter.txt
- Proofread formatting
```

**Recommended**: Copy-paste to ensure formatting consistency

### Step 7: Author Information

**Author Biographies** (optional but recommended):
- 100-150 words per author
- Include: Education, position, research interests, key publications
- Upload as: author_biographies.txt or paste in field

**Author Pictures** (optional):
- High-resolution headshots (200×200 pixels minimum)
- Format: JPEG or PNG
- Not required, but can be included

### Step 8: Declarations and Statements

**Originality Declaration**:
```
Check: "This manuscript is original and has not been
       previously published or submitted elsewhere"
```

**Conflict of Interest Statement**:
- Disclose any relevant financial interests or relationships
- Kenneth French data is public → no conflicts
- Research funding: [List any grants or sponsorships]

**Data Availability Statement**:
```
"All data used in this study are publicly available from
Kenneth French's Data Library at:
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/

Code and reproducibility materials are available at:
https://github.com/USERNAME/factor-crowding-unified
```

**Ethics Statement** (if applicable):
```
"This research uses publicly available financial data and
does not involve human subjects or sensitive data. No ethical
approval was required."
```

### Step 9: Review Before Submission

**Portal Review Checklist**:

```
□ Title: Correct and complete
□ All authors listed with correct affiliations
□ Corresponding author email is correct
□ Abstract appears correctly formatted
□ Keywords: 5-6 listed
□ PDF uploaded: main.pdf displays without errors
□ Cover letter: Present and complete
□ Supplementary materials: Either uploaded or link provided
□ Author biographies: Complete
□ Conflict of interest: Disclosed
□ Data availability: Statement provided
□ All required fields marked as complete
□ No error messages or warnings
```

**Final Quality Check**:
1. Click "Preview Submission"
2. Review all information as it will appear to editors
3. Verify formatting and content
4. Check for any missing required fields
5. Print preview as PDF backup

### Step 10: Submit

**Final Confirmation**:
1. Read submission confirmation message
2. Click "I agree to the terms and conditions"
3. Click **SUBMIT** button

**Post-Submission**:
1. **Save confirmation page**: Screenshot or print
2. **Record confirmation number**: Usually format "JMLR2026-XXXX"
3. **Record submission timestamp**: For your records
4. **Check email**: Confirmation email should arrive within 24 hours

---

## Part 3: Post-Submission (After January 20)

### Immediate Actions (Day 1-2)

```
□ Verify receipt email from JMLR
□ Save confirmation number: ________________
□ Save submission timestamp: ________________
□ Update GitHub repository: Add tag "submitted-to-jmlr"
□ Create file: SUBMISSION_RECORD.txt with:
  - Submission date
  - Confirmation number
  - JMLR manuscript ID (when received)
```

### Documentation

Create `SUBMISSION_RECORD.txt`:
```
JMLR SUBMISSION RECORD
======================

Paper: Not All Factors Crowd Equally
Submission Date: January 20, 2026
Confirmation Number: [From email]
JMLR Manuscript ID: [When assigned - usually within 1 week]

Email Confirmation:
- Receipt confirmed: [Date]
- Confirmation email: [Save PDF]

GitHub Repository:
- URL: https://github.com/USERNAME/factor-crowding-unified
- Tag: submitted-to-jmlr-v1
- Release: Published on [Date]

Contact Information:
- Corresponding Author: [Name]
- Email: [Email]
- Phone: [Phone]

Next Steps:
- Editorial office will assign to area editor
- Timeline: 3-6 months for initial decision
- Reviews typically: 2-4 reviewers
- Revision round (if invited): 2-3 months to resubmit
```

### Long-Term Planning

**Expected Timeline**:
```
Week 1:    Submission received and logged
Week 2:    Assigned to area editor
Week 3-4:  Area editor assigns reviewers
Month 2-4: Peer review process
Month 4:   Editorial decision meeting
Month 5:   Decision email sent (Accept/Minor/Major Revisions/Reject)

If Revisions Requested:
Month 5-7: Prepare response and revisions
Month 7:   Resubmit revised manuscript
Month 8:   Final decision
Month 9+:  Publication in next JMLR volume
```

**Preparation for Potential Reviews**:
1. Create document: "Potential Reviewer Questions"
2. Prepare responses to anticipated criticisms
3. Have additional robustness tests ready
4. Document all assumptions and limitations
5. Prepare author response letter template

---

## Troubleshooting Guide

### PDF Upload Issues

**Problem**: PDF upload fails or is corrupted
**Solution**:
```bash
# Check PDF integrity
pdfinfo main.pdf  # Should show valid metadata

# Re-generate if needed
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Verify file size
ls -lh main.pdf  # Should be < 10 MB
```

### Missing Bibliography Entries

**Problem**: Some references not appearing
**Solution**:
```bash
# Check references.bib file
grep "^@" references.bib | wc -l  # Should show 40+

# Verify bibtex compilation
bibtex main  # Check for warnings

# Re-run full compilation
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

### Cross-Reference Errors

**Problem**: \ref{} commands show "??"
**Solution**:
```bash
# Run LaTeX multiple times
pdflatex main.tex
pdflatex main.tex
pdflatex main.tex

# Check main.aux file for undefined references
grep "undefined" main.log
```

### Portal Authentication

**Problem**: Cannot log into JMLR portal
**Solution**:
1. Click "Forgot Password" and reset
2. Check that email is correct (institutional email preferred)
3. Verify email is not filtered as spam
4. Contact editor@jmlr.org if problem persists

---

## Key Contacts

**JMLR Editorial Office**:
- Email: editor@jmlr.org
- Website: https://jmlr.org/
- Submission Portal: https://jmlr.org/submit/

**For Technical Issues**:
- Subject: "[Technical Issue] Submission Portal Problem"
- Include: Confirmation number, error message

---

## Final Checklist for Submission Day

**Morning of January 20**:
- [ ] Verify main.pdf opens and looks correct
- [ ] All 9 sections present in order
- [ ] All 6 appendices included
- [ ] Bibliography has 40+ entries
- [ ] Page count is 55-70 pages
- [ ] No warning or error messages

**1 Hour Before**:
- [ ] Log into JMLR portal
- [ ] Navigate to submission form
- [ ] Have all files ready for upload
- [ ] Have cover letter text ready
- [ ] Have author information ready

**Submission**:
- [ ] Enter manuscript information
- [ ] Upload main.pdf
- [ ] Provide GitHub link for supplementary materials
- [ ] Paste cover letter
- [ ] Enter author information
- [ ] Review all fields
- [ ] Click SUBMIT
- [ ] Save confirmation number

**After Submission**:
- [ ] Verify confirmation email arrives
- [ ] Save all confirmation documents
- [ ] Update GitHub with submission tag
- [ ] Create SUBMISSION_RECORD.txt
- [ ] Notify co-authors of successful submission

---

## Success Indicators

You'll know you've successfully submitted when:

1. ✓ Portal shows "Submission Received" message
2. ✓ Confirmation email arrives within 24 hours
3. ✓ Confirmation email contains submission ID
4. ✓ You can view your submission in portal
5. ✓ GitHub repository is public and accessible
6. ✓ All supplementary materials are findable

---

**Estimated time for Part 2 (portal submission)**: 45-90 minutes

**Pro tip**: Don't rush. Take 10 minutes to carefully review before clicking SUBMIT.

---

*Last Updated: December 16, 2025*
*Ready for January 20, 2026 Submission*
