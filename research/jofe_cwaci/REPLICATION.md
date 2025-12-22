# Replication Package

**Paper:** Crowding-Aware Conformal Prediction for Factor Return Uncertainty
**Authors:** Chorok Lee (KAIST)
**Target Venue:** Journal of Financial Econometrics

---

## Overview

This replication package contains all code and instructions needed to reproduce the results in the paper. The analysis uses publicly available Fama-French factor data and can be run on any standard computing environment.

---

## Requirements

### Software
- Python 3.8+
- LaTeX (for compiling the paper)

### Python Packages
```
numpy>=1.21
pandas>=1.3
scipy>=1.7
matplotlib>=3.4
pandas-datareader>=0.10
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Directory Structure

```
jofe_cwaci/
├── src/                          # Core implementation
│   ├── conformal.py              # CW-ACI algorithm
│   └── crowding.py               # Crowding signal computation
├── experiments/                   # Reproducible experiments
│   ├── 01_coverage_analysis.py   # Main empirical results (Table 3, Figure 1)
│   ├── 02_monte_carlo.py         # Monte Carlo validation (Tables 1-2, Figure 3)
│   ├── 03_robustness.py          # Robustness checks (Section 6)
│   └── 04_generate_figures.py    # Publication figures
├── scripts/                       # Utility scripts
│   └── download_data.py          # Download Fama-French data
├── paper/                         # LaTeX manuscript
│   ├── main.tex                  # Main document
│   ├── references.bib            # Bibliography
│   └── figures/                  # Generated figures
├── data/                          # Data files (auto-generated)
├── results/                       # Output files
├── requirements.txt               # Python dependencies
├── run_all.py                    # Master replication script
└── REPLICATION.md                # This file
```

---

## Replication Instructions

### Quick Start (All Results)

Run the master script to reproduce all results:

```bash
python run_all.py
```

This will:
1. Download Fama-French factor data
2. Run the main coverage analysis (Table 3, Figure 1)
3. Run Monte Carlo validation (Tables 1-2, Figure 3)
4. Run all robustness checks (Section 6)
5. Generate all figures

Total runtime: approximately 10-15 minutes.

### Step-by-Step Replication

#### Step 1: Download Data
```bash
python scripts/download_data.py
```

Downloads Fama-French 5 factors and momentum from Kenneth French's data library.
Output: `data/ff_factors.csv`

#### Step 2: Main Coverage Analysis (Table 3, Figure 1)
```bash
python experiments/01_coverage_analysis.py
```

Reproduces the main empirical results:
- Standard CP vs CW-ACI coverage by factor
- Coverage by crowding regime (high/low)
- Interval width adaptation

Output: `results/coverage_analysis.csv`

#### Step 3: Monte Carlo Validation (Tables 1-2, Figure 3)
```bash
python experiments/02_monte_carlo.py
```

Runs 500 simulations to validate CW-ACI under controlled conditions:
- Main simulation (Table 1)
- Sensitivity to crowding effect strength (Table 2)
- Sample size robustness

Output: `results/monte_carlo_main.csv`, `results/monte_carlo_effects.csv`

#### Step 4: Robustness Analysis (Section 6)
```bash
python experiments/03_robustness.py
```

Comprehensive robustness checks:
- Alternative crowding proxies (volatility, correlation)
- Subperiod analysis (1963-1993 vs 1994-2025)
- Calibration split sensitivity
- Sensitivity parameter gamma
- Momentum control

Output: `results/robustness_*.csv`

#### Step 5: Generate Figures
```bash
python experiments/04_generate_figures.py
```

Creates publication-quality figures:
- Figure 1: Coverage comparison by factor
- Figure 2: Interval width adaptation
- Figure 3: Monte Carlo sensitivity analysis
- Figure 4: Robustness summary
- Figure 5: Coverage gap analysis

Output: `paper/figures/*.pdf`

---

## Key Results to Verify

### Main Finding (Table 3)
Standard CP high-crowding coverage: **78.3%** (vs 90% target)
CW-ACI high-crowding coverage: **93.3%**
Improvement: **+15.0 percentage points**

### Monte Carlo (Table 1)
Under controlled conditions (δ=0.5):
- Standard CP high-crowding: 85.0%
- CW-ACI high-crowding: 98.1%
- Improvement: +13.1pp

### Robustness
Results hold across:
- Alternative proxies (volatility: +20.2pp, correlation: +13.7pp)
- Subperiods (both show positive improvement)
- Calibration splits (range: +9.0pp to +15.0pp)
- Sensitivity parameters (all γ values show +14.5pp to +15.3pp)
- Momentum control (orthogonalized: +15.3pp)

---

## Data Sources

All data is publicly available from the Kenneth French Data Library:
- https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

Factors used:
- Fama-French 5 Factors (Mkt-RF, SMB, HML, RMW, CMA)
- Momentum Factor

Period: July 1963 - October 2025 (748 monthly observations)

---

## Compiling the Paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Contact

For questions about replication, please contact:
- Chorok Lee, KAIST

---

## License

This code is provided for academic replication purposes. Please cite the paper if you use this code in your research.
