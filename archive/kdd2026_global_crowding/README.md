# Mining Factor Crowding at Global Scale

**Target**: KDD 2026 (Jeju, Korea)
**Deadline**: February 8, 2026

## Overview

This project extends factor crowding analysis to global markets using ML-based detection, validating patterns across 6 regions and 10+ factors.

## Research Question

> Do factor crowding patterns discovered in US markets generalize globally, and can ML methods outperform model-based detection?

## Key Contributions

1. **Global Scale**: 6 regions × 10 factors = 60 factor-region pairs
2. **ML Detection**: LSTM/XGBoost vs model residual comparison
3. **Taxonomy Validation**: Mechanical/Judgment classification across regions
4. **Cross-Region Generalization**: Train on US → Predict other regions

## Data Sources

| Source | Data | Regions |
|--------|------|---------|
| Ken French | FF5 + Momentum | US, Developed, Europe, Japan, APAC, EM |
| AQR | QMJ, BAB | Global |

## Project Structure

```
kdd2026_global_crowding/
├── README.md                    # This file
├── docs/
│   └── RESEARCH_PLAN.md         # Detailed research plan
├── data/
│   ├── fetch_global_factors.py  # Ken French global
│   ├── fetch_aqr_factors.py     # AQR (QMJ, BAB)
│   └── global_factors/          # Parquet storage
├── src/
│   ├── crowding_ml.py           # ML crowding detection
│   ├── features.py              # Feature engineering
│   ├── models/
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   └── lstm_model.py
│   └── tail_risk_nn.py          # Neural network tail risk
├── experiments/
│   ├── 01_global_robustness.py  # 6 regions decay verification
│   ├── 02_taxonomy_expansion.py # QMJ, BAB classification
│   ├── 03_ml_crowding.py        # ML vs model residual
│   ├── 04_lstm_temporal.py      # LSTM temporal patterns
│   ├── 05_nn_tail_risk.py       # NN tail risk
│   └── 06_cross_region.py       # Cross-region generalization
├── paper/
│   ├── kdd2026_global.tex       # Paper
│   └── figures/
└── tests/
```

## Dependencies

```python
# Core
numpy
pandas
scipy

# ML
scikit-learn
xgboost
torch  # for LSTM

# Data
pandas_datareader
openpyxl  # for AQR Excel files

# From factor_crowding
sys.path.append('../factor_crowding')
from src.crowding_signal import CrowdingDetector
```

## Timeline

| Week | Dates | Task |
|------|-------|------|
| 1 | Dec 9-15 | Setup + Global data collection |
| 2 | Dec 16-22 | Feature engineering + baseline |
| 3 | Dec 23-29 | RF, XGBoost implementation |
| 4 | Dec 30-Jan 5 | LSTM temporal model |
| 5 | Jan 6-12 | NN tail risk + walk-forward |
| 6 | Jan 13-19 | Cross-region experiments |
| 7 | Jan 20-26 | Paper writing |
| 8 | Jan 27-Feb 7 | Review + Submit |

## Expected Results

### Global Robustness
| Region | Expected Momentum R² |
|--------|---------------------|
| US | 0.65 (baseline) |
| Developed | 0.50-0.60 |
| Europe | 0.45-0.55 |
| Japan | 0.40-0.50 |
| EM | 0.30-0.40 |

### Taxonomy Expansion
| Factor | Classification | Rationale |
|--------|---------------|-----------|
| BAB | Mechanical | "Buy low beta" is unambiguous |
| QMJ | Judgment | "Quality" requires interpretation |

## KDD Topics Alignment

- Data mining in finance
- Large-scale pattern discovery
- Time series analysis
- Predictive analytics
