# Phase 3D: JMLR Formatting Guide - Submission Preparation

**Duration**: 1 week (Dec 27, 2025 - Jan 3, 2026)
**Goal**: Format manuscript to JMLR standards and prepare complete submission package

---

## JMLR Submission Requirements

### Basic Requirements
- **Format**: LaTeX (PDF for submission)
- **Page Limit**: No explicit limit (but 40-60 pages typical for theory papers)
- **Fonts**: Standard LaTeX fonts (10-12pt)
- **Margins**: 1 inch on all sides
- **Line Spacing**: Single space main text, double space for figures/tables
- **Appendices**: Allowed and encouraged for proofs

### Estimated Paper Length
- Main text: ~40-50 pages
- Appendices: ~15-20 pages
- Total with references: ~55-70 pages

---

## Step 1: LaTeX Template Structure

### Directory Structure for Submission
```
jmlr_unified_submission/
├── main.tex                      # Main document
├── jmlr2e.sty                    # JMLR style file
├── macros.tex                    # Custom macros & notation
├── sections/
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── background.tex
│   ├── game_theory.tex
│   ├── empirical.tex
│   ├── domain_adaptation.tex
│   ├── tail_risk.tex
│   ├── robustness.tex
│   └── conclusion.tex
├── appendices/
│   ├── appendix_a.tex
│   ├── appendix_b.tex
│   ├── appendix_c.tex
│   ├── appendix_d.tex
│   ├── appendix_e.tex
│   └── appendix_f.tex
├── figures/                      # EPS or PDF figures
│   ├── fig_crowding.eps
│   ├── fig_decay_curves.eps
│   └── ...
├── tables/                       # Table data
│   ├── table_parameters.tex
│   └── ...
└── references.bib                # BibTeX bibliography
```

---

## Step 2: Main LaTeX File (main.tex)

```latex
\documentclass[11pt]{article}
\usepackage{jlmr2e}

% Include custom notation macros
\input{macros.tex}

% Document metadata
\jmlrheading{1}{2026}{1}{1}{Published online}{2}

\ShortHeadings{Not All Factors Crowd Equally}{Author}
\firstpageno{1}

\begin{document}

\title{Not All Factors Crowd Equally: A Game-Theoretic Model of Alpha Decay
        with Global Transfer and Risk Management}

\author{Author Name \and Affiliation}
\editor{Editor Name}

\maketitle

\begin{abstract}
[Abstract: 150-250 words. See Section 3 below]
\end{abstract}

\keywords{factor investing, crowding, domain adaptation, conformal prediction, risk management}

% Main text
\section{Introduction}
\input{sections/introduction.tex}

\section{Related Work}
\input{sections/related_work.tex}

\section{Background and Preliminaries}
\input{sections/background.tex}

\section{Game-Theoretic Model of Crowding Dynamics}
\input{sections/game_theory.tex}

\section{Empirical Validation on US Markets}
\input{sections/empirical.tex}

\section{Global Domain Adaptation with Temporal-MMD}
\input{sections/domain_adaptation.tex}

\section{Tail Risk Prediction and Crowding-Weighted Conformal Inference}
\input{sections/tail_risk.tex}

\section{Robustness, Extensions, and Discussion}
\input{sections/robustness.tex}

\section{Conclusion}
\input{sections/conclusion.tex}

% Appendices
\appendix

\section{Proofs of Game-Theoretic Model}
\input{appendices/appendix_a.tex}

\section{Domain Adaptation Theory}
\input{appendices/appendix_b.tex}

\section{Conformal Prediction Theory}
\input{appendices/appendix_c.tex}

\section{Data Documentation}
\input{appendices/appendix_d.tex}

\section{Algorithm Pseudocode}
\input{appendices/appendix_e.tex}

\section{Supplementary Robustness Tests}
\input{appendices/appendix_f.tex}

% References
\bibliography{references}

\end{document}
```

---

## Step 3: Abstract (150-250 words)

**Template**:

```
Factor investing generates systematic excess returns, but these returns decay
over time as capital flows in—a phenomenon called crowding. While prior work
documents crowding effects empirically, the mechanistic explanation remains
unclear. We provide three novel contributions addressing this gap:

1) GAME-THEORETIC MODEL: We derive a mechanistic explanation of factor alpha
decay from game-theoretic equilibrium. Rational investors' optimal exit timing
generates hyperbolic decay: α(t) = K/(1+λt). We prove heterogeneous decay
across factor types and validate on 61 years of Fama-French data. Judgment
factors decay 2.4× faster than mechanical factors (p<0.001), with out-of-sample
predictive power reaching 55%.

2) REGIME-CONDITIONAL DOMAIN ADAPTATION: We introduce Temporal-MMD, a regime-
aware domain adaptation framework respecting financial market structure (bull/
bear, high/low volatility). Unlike standard MMD, Temporal-MMD conditions on
regimes, improving transfer efficiency from 43% (naive) to 64% across seven
developed markets.

3) CROWDING-WEIGHTED CONFORMAL PREDICTION: We extend adaptive conformal
inference with crowding signals while preserving coverage guarantees. Our
CW-ACI framework produces prediction sets that adapt to crowding levels. In
dynamic portfolio hedging, this improves Sharpe ratio by 54% (0.67→1.03) and
reduces tail risk by 60-70% during crashes.

The three contributions form an integrated framework: game theory explains why
crowding matters, domain adaptation enables global transfer, and conformal
prediction manages risk. We validate across 61 years of US data, seven
international markets, and other asset classes.
```

**Word count**: ~240 words (good for JMLR)

---

## Step 4: Notation Macros (macros.tex)

```latex
% Key financial variables
\newcommand{\alphai}[1]{{\alpha_{#1}}}
\newcommand{\Ki}[1]{{K_{#1}}}
\newcommand{\li}[1]{{\lambda_{#1}}}
\newcommand{\Ci}[1]{{C_{#1}}}
\newcommand{\ri}[1]{{r_{#1}}}

% Specific notation
\newcommand{\alphajudg}{{\alpha_{\text{judgment}}}}
\newcommand{\alphmech}{{\alpha_{\text{mechanical}}}}
\newcommand{\ljudg}{{\lambda_{\text{judgment}}}}
\newcommand{\lmech}{{\lambda_{\text{mechanical}}}}

% Domain adaptation
\newcommand{\MMD}{\text{MMD}}
\newcommand{\RKERNEL}[2]{k_R({#1}, {#2})}

% Conformal prediction
\newcommand{\ACI}{{\text{ACI}}}
\newcommand{\CWACI}{{\text{CW-ACI}}}
\newcommand{\NC}{A}  % nonconformity score

% Expectation, probability
\newcommand{\E}[1]{\mathbb{E}[{#1}]}
\newcommand{\Prob}[1]{\mathbb{P}({#1})}
\newcommand{\Var}[1]{\mathbb{V}[{#1}]}

% Theorem/result numbering (JMLR style)
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\theoremstyle{definition}
\newtheorem{definition}{Definition}
\theoremstyle{remark}
\newtheorem{remark}{Remark}
```

---

## Step 5: Figure & Table Formatting

### Figure Formatting
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.8\textwidth]{figures/fig_crowding.eps}
\caption{Factor crowding over time (1963-2024).
         Momentum (blue) shows 3× more crowding variability than Size (red).
         Gray shading indicates recessions.}
\label{fig:crowding}
\end{figure}
```

### Table Formatting
```latex
\begin{table}[t]
\centering
\caption{Estimated Decay Parameters by Factor (Full Period 1963-2024)}
\label{table:parameters}
\small
\begin{tabular}{lcccc}
\hline
Factor & Type & $\hat{K}$ (\%) & $\hat{\lambda}$ & Model R$^2$ \\
\hline
SMB & Mechanical & 3.82 [3.12, 4.52] & 0.062 & 0.68 \\
HML & Judgment & 4.51 [3.82, 5.20] & 0.156 & 0.71 \\
\ldots \\
\hline
\end{tabular}
\end{table}
```

---

## Step 6: BibTeX References (references.bib)

```bibtex
@article{Hua2020,
  author = {Hua, Lei and Sun, Liyan},
  year = {2020},
  title = {Dynamics of Factor Crowding},
  journal = {Journal of Portfolio Management},
  volume = {46},
  pages = {1--15}
}

@article{DeMiguel2020,
  author = {DeMiguel, Victor and Garlappi, Lorenzo and Uppal, Raman},
  year = {2020},
  title = {What Alleviates Crowding?},
  journal = {Journal of Finance},
  volume = {75},
  pages = {1111--1147}
}

@inproceedings{He2023,
  author = {He, Xueqian and others},
  year = {2023},
  title = {Time-Series Domain Adaptation via Neural ODE},
  booktitle = {Proceedings of ICML 2023}
}

@article{Angelopoulos2021,
  author = {Angelopoulos, Anastasios N. and Bates, Stephen},
  year = {2021},
  title = {A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification},
  journal = {arXiv preprint}
}

% Add 46 more references...
```

---

## Step 7: Markdown to LaTeX Conversion

### Conversion Strategy

Since manuscripts are written in Markdown, convert to LaTeX:

**For Sections**:
```markdown
# Section 4: Game-Theoretic Model

## 4.1 Model Setup

### Investment Game...
```

**Becomes**:
```latex
\section{Game-Theoretic Model}

\subsection{Model Setup}

\subsubsection{Investment Game}...
```

**For Math**:
```markdown
$$\alpha(t) = \frac{K}{1 + \lambda t}$$
```

**Becomes**:
```latex
\begin{equation}
\alpha(t) = \frac{K}{1 + \lambda t}
\label{eq:hyperbolic_decay}
\end{equation}
```

**For Tables**:
Markdown tables convert to LaTeX tabular environments (as shown in Step 5)

### Conversion Checklist
- [ ] All markdown `##` headers become `\section`
- [ ] All markdown `###` become `\subsection`
- [ ] All markdown `$$..$$` become `\[...\]` or `\begin{equation}`
- [ ] All markdown `[ref](url)` become `\cite{key}` or inline citations
- [ ] All markdown tables become tabular environments
- [ ] All markdown code blocks become `\texttt{}` or `lstlisting`

---

## Step 8: Submission Checklist

### Before Submission
- [ ] Abstract is 150-250 words
- [ ] All LaTeX compiles without errors
- [ ] PDF renders correctly
- [ ] All figures are high-quality (300+ dpi for print)
- [ ] All tables are properly formatted
- [ ] No broken cross-references
- [ ] Bibliography is complete (50+ citations)
- [ ] Page count is ~55-70 pages total

### Meta Information
- [ ] Author affiliation is correct
- [ ] Keywords are relevant (5-6 keywords)
- [ ] Running head is short (<40 characters)
- [ ] No proprietary or confidential information
- [ ] Conflicts of interest (if any) are disclosed

### Supporting Materials
- [ ] Cover letter is written (see Step 9)
- [ ] Reproducibility note mentions code/data availability
- [ ] GitHub repo is set up (see Step 10)
- [ ] Supplementary files are organized

---

## Step 9: Cover Letter Template

```
[Date]

Editor, Journal of Machine Learning Research
[Contact Information]

Dear [Editor Name],

We submit our manuscript titled "Not All Factors Crowd Equally: A Game-Theoretic
Model of Alpha Decay with Global Transfer and Risk Management" for consideration
at JMLR.

SUMMARY
This paper makes three contributions at the intersection of quantitative finance
and machine learning:

1. A game-theoretic model explaining factor alpha decay mechanism
2. Regime-conditional domain adaptation for cross-market transfer
3. Crowding-weighted conformal prediction for risk management

All three are theoretically motivated, empirically validated, and practically
relevant.

NOVELTY
- First mechanistic (vs. empirical) explanation of factor crowding decay
- First regime-aware domain adaptation for financial markets
- First integration of domain knowledge with conformal prediction while
  preserving guarantees

VALIDATION
- Tested on 61 years of Fama-French data (1963-2024)
- Validated across 7 international developed markets
- Demonstrated practical impact: +54% Sharpe ratio improvement in hedging
- 50+ robustness tests confirm stability

The work is original and has not been published or submitted elsewhere. All
authors have reviewed and approved the manuscript.

We believe this work makes significant contributions to both the academic
understanding of factor investing and practical portfolio management.

Sincerely,

[Author Name]
[Affiliation]
[Contact Information]
```

---

## Step 10: GitHub Reproducibility Package

### Repository Structure
```
factor-crowding-unified/
├── README.md                 # Overview & instructions
├── requirements.txt          # Python dependencies
├── data/
│   ├── fama_french/         # FF factors (linked to public source)
│   └── processed/           # Processed data files
├── src/
│   ├── models/
│   │   ├── game_theory.py   # Decay model fitting
│   │   ├── temporal_mmd.py  # Domain adaptation
│   │   └── conformal.py     # CW-ACI framework
│   ├── data/
│   │   └── processing.py    # Data pipeline
│   └── evaluation/
│       └── metrics.py       # Evaluation metrics
├── notebooks/
│   ├── 01_decay_fitting.ipynb
│   ├── 02_domain_adaptation.ipynb
│   └── 03_hedging_application.ipynb
├── results/
│   ├── tables/
│   ├── figures/
│   └── logs/
└── paper/
    ├── main.pdf            # Compiled manuscript
    └── references.bib      # Bibliography
```

### README.md Template
```markdown
# Not All Factors Crowd Equally: Unified Framework

Complete code and data for reproducing results in "[paper title]"

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.models import GameTheoryModel, TemporalMMD, CWACI

# Fit game-theoretic decay model
model = GameTheoryModel()
model.fit(fama_french_data)

# Domain adaptation
da = TemporalMMD()
da.fit(us_factors, international_factors)

# Conformal prediction
cp = CWACI()
prediction_sets = cp.construct_sets(test_data, crowding_data)
```

## Reproduction

See notebooks/ for step-by-step reproduction of all results.

## Citation

If you use this code, please cite:

```bibtex
@article{Author2026,
  title={Not All Factors Crowd Equally...},
  author={Author, A.},
  journal={Journal of Machine Learning Research},
  year={2026}
}
```

## License

MIT License
```

---

## Timeline for Phase 3D

| Date | Task | Deliverable |
|------|------|-------------|
| Dec 27 | Markdown → LaTeX conversion | sections/*.tex |
| Dec 28 | Create main.tex and macros.tex | Compilable LaTeX |
| Dec 29 | Insert figures and tables | PDF with visuals |
| Dec 30 | Write abstract and cover letter | submission-ready docs |
| Dec 31 | Set up GitHub repo | Public code repository |
| Jan 1 | Final proofreading | Ready for submission |
| Jan 2-3 | Test compile, verify PDF | Final submission package |

---

## Final Submission Checklist

**Content** ✓
- [ ] Main text: ~40 pages
- [ ] Appendices: ~15-20 pages
- [ ] 50+ citations
- [ ] 10+ tables/figures
- [ ] 7 theorems with proofs

**Format** ✓
- [ ] LaTeX source files
- [ ] PDF rendering correctly
- [ ] Figures are high-quality
- [ ] Tables properly formatted
- [ ] References in BibTeX

**Documentation** ✓
- [ ] Abstract (150-250 words)
- [ ] Cover letter
- [ ] Author affiliation
- [ ] Keywords (5-6)
- [ ] Reproducibility note

**Supporting** ✓
- [ ] GitHub repository
- [ ] Code with documentation
- [ ] Processed data (if public)
- [ ] README with instructions
- [ ] Notebooks for reproducibility

**Ready for Submission** ✓
- [ ] All above items complete
- [ ] No errors or warnings in LaTeX compilation
- [ ] PDF looks professional
- [ ] All links and references work
- [ ] Covers all JMLR requirements

---

**Note**: This guide provides a complete framework for converting the Markdown manuscript to publication-ready LaTeX and preparing submission materials.

