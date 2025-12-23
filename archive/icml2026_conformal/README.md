# Conformal Prediction for Factor Crowding

**Target**: ICML 2026 (Seoul, Korea)
**Deadline**: January 28, 2026

## Overview

This project applies Conformal Prediction to factor crowding detection, providing distribution-free uncertainty quantification with coverage guarantees.

## Research Question

> Can we provide calibrated uncertainty estimates for crowding regime detection and tail risk prediction without making distributional assumptions?

## Key Contributions

1. **Conformal Crowding Detection**: Prediction sets with guaranteed coverage for regime classification
2. **Conformal Tail Risk**: Distribution-free prediction intervals for factor returns
3. **Calibration Analysis**: Comparison vs Bayesian and Bootstrap methods

## Why Conformal Prediction?

| Property | Benefit for Finance |
|----------|---------------------|
| Distribution-free | No Gaussian assumptions |
| Coverage guarantee | 90% interval contains truth 90% of time |
| Simple | Wrapper on any base model |
| Valid under covariate shift | Robust to market regime changes |

## Project Structure

```
icml2026_conformal/
├── README.md                    # This file
├── docs/
│   └── RESEARCH_PLAN.md         # Detailed research plan
├── data/
│   └── fetch_data.py            # Data collection (reuses factor_crowding)
├── src/
│   ├── conformal_crowding.py    # Conformal prediction for crowding
│   ├── conformal_regression.py  # Conformal intervals for returns
│   └── calibration.py           # Calibration metrics
├── experiments/
│   ├── 01_coverage_analysis.py  # Coverage guarantee verification
│   ├── 02_crowding_sets.py      # Prediction set analysis
│   ├── 03_tail_risk_intervals.py # Tail risk UQ
│   └── 04_comparison.py         # vs Bayesian, Bootstrap
├── paper/
│   ├── icml2026_conformal.tex   # Paper
│   └── figures/
└── tests/
```

## Dependencies

```python
# Core
numpy
pandas
scipy

# Conformal Prediction
mapie  # or crepes, conformist

# From factor_crowding
sys.path.append('../factor_crowding')
from src.crowding_signal import CrowdingDetector
```

## Timeline

| Week | Dates | Task |
|------|-------|------|
| 1 | Dec 9-15 | Setup + Conformal basics |
| 2 | Dec 16-22 | Crowding detection wrapper |
| 3 | Dec 23-29 | Tail risk conformal regression |
| 4 | Dec 30-Jan 5 | Coverage experiments |
| 5 | Jan 6-12 | Baseline comparisons |
| 6 | Jan 13-19 | Paper writing |
| 7 | Jan 20-27 | Review + Submit |

## ICML Topics Alignment

- Trustworthy machine learning (reliability, calibration)
- Probabilistic methods
- Application-driven machine learning (finance)

## References

- Vovk et al. (2005) "Algorithmic Learning in a Random World"
- Romano et al. (2019) "Conformalized Quantile Regression"
- Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction"
