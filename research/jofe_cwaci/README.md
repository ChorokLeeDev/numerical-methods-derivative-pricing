# Crowding-Aware Conformal Prediction for Factor Return Uncertainty

**Target Venue:** Journal of Financial Econometrics - Special Issue on Machine Learning
**Submission Deadline:** March 1, 2026
**Author:** Chorok Lee (KAIST)

---

## Paper Summary

Standard conformal prediction provides distribution-free coverage guarantees for prediction intervals. However, when applied to factor returns, we find empirical coverage drops to **67-77% during high-crowding periods**, well below the nominal 90% target.

We introduce **Crowding-Weighted Adaptive Conformal Inference (CW-ACI)**, which produces wider prediction intervals when crowding signals are elevated. Using 62 years of Fama-French data, we show CW-ACI achieves **83-95% coverage during high-crowding periods** while maintaining nominal coverage overall.

---

## Key Findings

| Finding | Evidence |
|---------|----------|
| Standard CP under-covers during high crowding | 67-77% coverage (target: 90%) |
| CW-ACI adapts interval width to crowding | 20-25% wider during high crowding |
| CW-ACI improves coverage | 83-95% during high crowding |
| Effect is consistent across factors | 5/5 factors show adaptation |

---

## Contributions

1. **Empirical Finding:** Document that standard conformal prediction systematically under-covers factor returns during high-crowding market conditions

2. **Methodology:** Propose CW-ACI, which weights nonconformity scores by crowding signals to produce adaptive prediction intervals

3. **Validation:** Demonstrate on 62 years of Fama-French data that CW-ACI achieves nominal coverage across market conditions

---

## Project Structure

```
jofe_cwaci/
├── paper/                    # LaTeX manuscript
│   ├── main.tex             # Main document
│   ├── sections/            # Paper sections
│   ├── figures/             # Generated figures
│   └── references.bib       # Bibliography
├── src/                     # Core implementation
│   ├── conformal.py         # CW-ACI implementation
│   ├── crowding.py          # Crowding signal computation
│   └── evaluation.py        # Coverage metrics
├── experiments/             # Reproducible experiments
│   ├── 01_coverage_analysis.py
│   ├── 02_monte_carlo.py
│   └── 03_robustness.py
├── data/                    # Data files (gitignored)
├── results/                 # Output files
└── scripts/                 # Utility scripts
```

---

## Timeline

| Phase | Dates | Deliverable |
|-------|-------|-------------|
| Foundation | Jan 2025 | Core implementation, basic results |
| Monte Carlo | Feb-Mar 2025 | Simulation validation |
| Theory | Apr-May 2025 | Coverage properties discussion |
| Writing | Jun-Aug 2025 | Full manuscript draft |
| Polish | Sep-Dec 2025 | Robustness, replication package |
| Submit | Jan-Feb 2026 | JoFE submission |

---

## Why This Paper Exists

This paper is a pivot from a larger project ("Not All Factors Crowd Equally") that attempted to:
1. Derive factor decay from game theory
2. Predict factor returns with 45-63% R²
3. Transfer insights globally with MMD
4. Provide CW-ACI uncertainty quantification

Rigorous empirical testing on December 22, 2024 revealed that **only CW-ACI actually works**. The other claims were not supported by data. Rather than publish false claims, we pivoted to focus on what we can demonstrate honestly.

See `../jmlr_unified/PIVOT_RATIONALE.md` for full documentation.

---

## Data

- **Source:** Kenneth French Data Library
- **Factors:** SMB, HML, RMW, CMA, Momentum
- **Period:** July 1963 - October 2025 (748 months)
- **Download:** `python scripts/download_data.py`

---

## Requirements

```
numpy>=1.21
pandas>=1.3
scipy>=1.7
scikit-learn>=1.0
matplotlib>=3.4
pandas-datareader>=0.10
```

---

## Reproducibility

All results can be reproduced:

```bash
# Download data
python scripts/download_data.py

# Run main analysis
python experiments/01_coverage_analysis.py

# Run Monte Carlo
python experiments/02_monte_carlo.py

# Generate figures
python experiments/03_generate_figures.py
```

---

## Contact

Chorok Lee
Korea Advanced Institute of Science and Technology (KAIST)
