# GitHub Repository Setup Guide

For: "Not All Factors Crowd Equally" JMLR Paper

---

## Repository Overview

**Repository Name**: `factor-crowding-unified`
**Visibility**: Public
**License**: MIT
**Purpose**: Complete reproducibility package for JMLR paper

---

## Repository Structure to Create

```
factor-crowding-unified/
├── README.md                              # Project overview
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore rules
│
├── paper/                                 # Paper and LaTeX files
│   ├── main.pdf                          # Compiled manuscript
│   ├── main.tex                          # Master document
│   ├── macros.tex                        # Notation
│   ├── jmlr2e.sty                        # Style file
│   ├── references.bib                    # Bibliography
│   ├── sections/                         # 9 main sections
│   └── appendices/                       # 6 appendices
│
├── src/                                   # Python source code
│   ├── __init__.py
│   ├── models/
│   │   ├── game_theory.py               # Hyperbolic decay model
│   │   ├── temporal_mmd.py              # Domain adaptation
│   │   ├── conformal.py                 # CW-ACI framework
│   │   └── utils.py                     # Helper functions
│   ├── data/
│   │   ├── loaders.py                   # Data loading utilities
│   │   ├── preprocessing.py             # Data processing
│   │   └── fama_french.py               # FF factor utilities
│   └── evaluation/
│       ├── metrics.py                   # Evaluation metrics
│       └── backtesting.py               # Portfolio backtesting
│
├── notebooks/                             # Jupyter notebooks
│   ├── 01_decay_fitting.ipynb           # Game theory validation
│   ├── 02_domain_adaptation.ipynb       # Transfer learning results
│   ├── 03_hedging_application.ipynb     # Portfolio application
│   └── 04_robustness_tests.ipynb        # Sensitivity analyses
│
├── data/                                  # Data directory
│   ├── raw/                             # Raw data
│   ├── processed/                       # Processed data
│   └── README.md                        # Data documentation
│
├── results/                               # Output results
│   ├── figures/                         # Generated figures
│   ├── tables/                          # Generated tables
│   └── logs/                            # Execution logs
│
├── tests/                                 # Unit tests
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_data.py
│   └── test_evaluation.py
│
└── docs/                                  # Additional documentation
    ├── INSTALLATION.md
    ├── USAGE.md
    ├── API_REFERENCE.md
    └── TROUBLESHOOTING.md
```

---

## Step-by-Step Setup Instructions

### 1. Initialize GitHub Repository

```bash
# Create new repository on GitHub.com (https://github.com/new)
# Name: factor-crowding-unified
# Description: "Unified framework for understanding factor crowding through game theory, domain adaptation, and conformal prediction"
# Add MIT license
# Add .gitignore for Python

# Clone locally
git clone https://github.com/USERNAME/factor-crowding-unified.git
cd factor-crowding-unified
```

### 2. Create Directory Structure

```bash
# Create all directories
mkdir -p paper/{sections,appendices}
mkdir -p src/{models,data,evaluation}
mkdir -p notebooks
mkdir -p data/{raw,processed}
mkdir -p results/{figures,tables,logs}
mkdir -p tests
mkdir -p docs
```

### 3. Copy Paper Files

```bash
# Copy LaTeX files to paper/
cp jmlr_submission/* paper/
cp jmlr_submission/sections/* paper/sections/
cp jmlr_submission/appendices/* paper/appendices/
```

### 4. Create Python Source Code Structure

Create `src/models/game_theory.py`:
```python
"""
Game-theoretic model of factor crowding and alpha decay.

Implements the hyperbolic decay model:
    α(t) = K / (1 + λt)
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator

class HyperbolicDecayModel(BaseEstimator):
    """Fit hyperbolic decay to factor returns."""

    def __init__(self):
        self.K_ = None
        self.lambda_ = None
        self.R2_ = None

    def fit(self, returns, time_indices=None):
        """Fit decay parameters to returns data."""
        if time_indices is None:
            time_indices = np.arange(len(returns))

        # Implement fitting logic (Levenberg-Marquardt)
        # ... (full implementation in actual repository)
        return self

    def predict(self, time_indices):
        """Predict alpha at given times."""
        if self.K_ is None:
            raise ValueError("Model not fitted yet")
        return self.K_ / (1 + self.lambda_ * time_indices)
```

### 5. Create Jupyter Notebooks Template

Create `notebooks/01_decay_fitting.ipynb`:
- Load Fama-French data
- Fit hyperbolic decay model
- Visualize results
- Compute statistics

### 6. Create Main README.md

```markdown
# Not All Factors Crowd Equally: Unified Framework

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Complete reproducibility package for "Not All Factors Crowd Equally: A Game-Theoretic Model of Alpha Decay with Global Transfer and Risk Management"

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running Examples

```python
from src.models import HyperbolicDecayModel

# Load data
from src.data import load_fama_french
returns = load_fama_french()

# Fit model
model = HyperbolicDecayModel()
model.fit(returns)

# Predict
predictions = model.predict(time_indices)
```

## Citation

If you use this code, please cite:

```bibtex
@article{Author2026,
  title={Not All Factors Crowd Equally: A Game-Theoretic Model of Alpha Decay with Global Transfer and Risk Management},
  author={Author, A.},
  journal={Journal of Machine Learning Research},
  year={2026},
  volume={TBD},
  pages={TBD}
}
```

## License

MIT License - see LICENSE file
```

---

## GitHub-Specific Setup

### Initialize Repository

```bash
# In the repository directory
git init
git add .
git commit -m "Initial commit: JMLR paper and reproducibility package"
git branch -M main
git remote add origin https://github.com/USERNAME/factor-crowding-unified.git
git push -u origin main
```

### Create Release (after paper acceptance)

```bash
git tag -a v1.0.0 -m "Official release for JMLR publication"
git push origin v1.0.0
```

### GitHub Pages Documentation (Optional)

```bash
# Create docs branch for GitHub Pages
git checkout --orphan gh-pages
mkdir -p docs/_posts
# Add index.md with project overview
git add .
git commit -m "Add GitHub Pages documentation"
git push origin gh-pages
```

---

## Files to Include

### Python Dependencies (requirements.txt)
```
numpy>=1.20.0
scipy>=1.7.0
scikit-learn>=1.0.0
pandas>=1.3.0
matplotlib>=3.4.0
jupyter>=1.0.0
```

### .gitignore
```
# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep

# Results
results/figures/*
results/tables/*
results/logs/*
!results/.gitkeep

# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
env/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

### LICENSE (MIT)
```
MIT License

Copyright (c) [YEAR] [Author Name]

Permission is hereby granted, free of charge, to any person obtaining a copy...
[Full MIT license text]
```

---

## Important Notes

1. **Data Availability**: Fama-French data is publicly available from Kenneth French's library
2. **Code Quality**: Include docstrings, type hints, and unit tests
3. **Reproducibility**: Ensure all random seeds are fixed
4. **Documentation**: Comprehensive README and in-code comments
5. **Version Control**: Clean commit history with descriptive messages

---

## Release Checklist

Before making repository public:

- [ ] All source code is documented
- [ ] All notebooks execute without errors
- [ ] README is complete and clear
- [ ] LICENSE file is present
- [ ] requirements.txt lists all dependencies
- [ ] .gitignore is configured
- [ ] No sensitive data is included
- [ ] Notebook outputs are cleared
- [ ] All paths are relative (no absolute paths)
- [ ] Tests pass locally
- [ ] Code follows PEP 8 style guide

---

## Post-Publication

After paper publication on JMLR:

1. Update README with paper URL
2. Add paper citation to repository
3. Create GitHub releases for different paper versions
4. Consider adding continuous integration (GitHub Actions)
5. Monitor for issues and update code as needed

---

This GitHub repository serves as the official reproducibility package for the JMLR paper and provides a complete, documented, and tested implementation of all methods.
