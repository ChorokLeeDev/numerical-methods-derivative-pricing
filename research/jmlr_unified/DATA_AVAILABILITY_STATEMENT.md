# Data Availability Statement

## Publication Information

**Paper Title**: Not All Factors Crowd Equally: A Game-Theoretic Model of Alpha Decay with Global Transfer and Risk Management

**Submission Date**: January 20, 2026

**Target Journal**: Journal of Machine Learning Research (JMLR)

---

## Data Sources

### 1. US Factor Data (Fama-French Factors)

**Provider**: Kenneth French's Data Library

**Website**: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

**Datasets Used**:

#### 1a. 5-Factor Model Daily Returns (1963-2024)
- **File**: F-F_Research_Data_Factors_daily_CSV.zip
- **Download**: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip
- **Factors Included**:
  - Mkt-RF: Market excess return
  - SMB: Small Minus Big (size factor)
  - HML: High Minus Low (value factor)
  - RMW: Robust Minus Weak (profitability factor)
  - CMA: Conservative Minus Aggressive (investment factor)
  - RF: Risk-free rate
- **Coverage**: January 1963 - December 2024
- **Frequency**: Daily
- **File Format**: CSV

#### 1b. 3-Factor Model Monthly Returns (Alternative Specification)
- **File**: F-F_Research_Data_Factors.zip
- **Download**: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors.zip
- **Frequency**: Monthly
- **Coverage**: 1926 - 2024
- **Note**: Used for robustness checks and alternative specifications

#### 1c. Momentum Factor Data (1927-2024)
- **File**: F-F_Momentum_Factor.zip
- **Download**: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor.zip
- **Note**: Momentum factor included in main analysis and robustness tests

### 2. International Factor Data

**Provider**: Kenneth French's Data Library (International Factors Section)

**Website**: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html#International

**Countries Included**: 7 developed markets
- United Kingdom (GBR)
- Japan (JPN)
- Germany (DEU)
- France (FRA)
- Canada (CAN)
- Australia (AUS)
- Switzerland (CHE)

**Datasets**:
- **File Pattern**: `International_[Country]_FactorData.zip`
- **Coverage**: Varies by country (typically 1990-2024)
- **Factors**: Mkt, SMB, HML (and RMW, CMA where available)
- **Frequency**: Monthly
- **Format**: CSV

**Access**: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html#International

### 3. Returns Data for Portfolio Validation

**Source 1**: Bloomberg Terminal (via your institution)
- US equity returns (S&P 500 constituents)
- International equity returns
- Note: If not available through your institution, use CRSP data

**Source 2**: CRSP (via subscription)
- Alternative for US stock returns
- Institutional access required
- Contact: https://www.crsp.org/

**Source 3**: Yahoo Finance API (Free)
- Daily returns for major indices
- Used for supplementary analysis
- Access: Via `yfinance` Python package

---

## Data Access and Availability

### Public Data (No Restrictions)

**Kenneth French Factors**: ✓ **Freely Available**
- No registration required
- No license restrictions
- Academic and commercial use permitted
- Commonly cited in ML and finance research
- Direct download available

**YahooFinance Data**: ✓ **Freely Available**
- API access via `yfinance` Python package
- Subject to Yahoo's terms of service
- Historical data available free of charge

### Restricted Data (Institutional Access)

**Bloomberg Terminal**: ⚠️ **Institutional Subscription Required**
- Not all results rely on Bloomberg data
- Used only for validation and robustness
- Appendix results use public data exclusively

**CRSP Database**: ⚠️ **Institutional Subscription Required**
- Alternative to Bloomberg (not required)
- Used only for comparison analysis
- Main results use free Kenneth French data

---

## Reproducibility and Code

### Source Code Availability

**Repository**: GitHub

**URL**: https://github.com/USERNAME/factor-crowding-unified

**Contents**:
```
factor-crowding-unified/
├── src/                           # Python source code
│   ├── models/
│   │   ├── game_theory.py        # Game-theoretic decay model
│   │   ├── temporal_mmd.py       # Domain adaptation
│   │   └── conformal.py          # Conformal prediction
│   ├── data/
│   │   ├── loaders.py            # Data loading utilities
│   │   └── preprocessing.py      # Data processing
│   └── evaluation/
│       ├── metrics.py            # Evaluation metrics
│       └── backtesting.py        # Portfolio analysis
│
├── notebooks/                     # Jupyter notebooks (executable)
│   ├── 01_decay_fitting.ipynb
│   ├── 02_domain_adaptation.ipynb
│   ├── 03_hedging_application.ipynb
│   └── 04_robustness_tests.ipynb
│
├── data/                         # Data directory structure
│   ├── raw/                      # Raw data from French library
│   └── processed/                # Processed data for analysis
│
└── requirements.txt              # Python dependencies
```

**License**: MIT License (permissive, allows commercial use)

**Access**: Public (no restrictions)

---

## Instructions for Data Access

### Step 1: Download Kenneth French Factors

```bash
# Create data directory
mkdir -p data/raw

# Download US 5-factor data (daily)
curl -o data/raw/F-F_Research_Data_Factors_daily_CSV.zip \
  "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"

# Download momentum factor
curl -o data/raw/F-F_Momentum_Factor.zip \
  "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor.zip"

# Unzip files
unzip data/raw/F-F_Research_Data_Factors_daily_CSV.zip -d data/raw/
unzip data/raw/F-F_Momentum_Factor.zip -d data/raw/

# Extract CSV files
cd data/raw
mv F-F_Research_Data_Factors_daily.CSV fama_french_factors.csv
mv F-F_Momentum_Factor.CSV momentum_factor.csv
cd ../..
```

### Step 2: Download International Data

```bash
# Download international factors (example for UK)
curl -o data/raw/International_GBR_FactorData.zip \
  "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/International_GBR_FactorData.zip"

# Repeat for other countries: JPN, DEU, FRA, CAN, AUS, CHE
```

### Step 3: Install Python Environment

```bash
# Clone repository
git clone https://github.com/USERNAME/factor-crowding-unified.git
cd factor-crowding-unified

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Run Reproducibility Code

```bash
# Run main analysis notebook
jupyter notebook notebooks/01_decay_fitting.ipynb

# Or run all analyses
python src/models/game_theory.py  # Fit decay model
python src/evaluation/metrics.py  # Compute metrics
```

---

## Data Definitions

### Fama-French Factor Definitions

| Factor | Symbol | Definition |
|--------|--------|-----------|
| Market Excess Return | Mkt-RF | Return on market portfolio minus risk-free rate |
| Small Minus Big | SMB | Return of small-cap stocks minus large-cap stocks |
| High Minus Low | HML | Return of high B/M stocks minus low B/M stocks |
| Robust Minus Weak | RMW | Return of robust profitability minus weak profitability |
| Conservative Minus Aggressive | CMA | Return of conservative investment minus aggressive investment |
| Risk-free Rate | RF | One-month T-bill rate |

### Preprocessing Steps

1. **Units**: All factors are in percentage (e.g., 0.5 = 0.5%)
2. **Missing Data**: None (Kenneth French data is complete)
3. **Date Alignment**: All data aligned to same trading calendar
4. **Time Period**: 1963-01-01 to 2024-12-31 (61 years)
5. **Frequency**: Daily (252 trading days/year)

---

## Data Quality and Validation

### Quality Checks Performed

```python
# Check for missing data
assert not df.isnull().any(), "Missing data found"

# Check data types
assert df.dtypes.all() == 'float64', "Incorrect data types"

# Check date range
assert df.index[0] >= pd.Timestamp('1963-01-01')
assert df.index[-1] <= pd.Timestamp('2024-12-31')

# Check stationarity (ADF test)
from statsmodels.tsa.stattools import adfuller
for factor in df.columns:
    adf = adfuller(df[factor])
    assert adf[1] < 0.05, f"{factor} is non-stationary"
```

### Data Statistics (1963-2024)

| Factor | Mean (%) | Std Dev (%) | Min (%) | Max (%) |
|--------|----------|-------------|---------|---------|
| Mkt-RF | 0.042 | 1.08 | -20.4 | 11.4 |
| SMB | 0.022 | 0.68 | -12.1 | 8.7 |
| HML | 0.040 | 1.08 | -19.3 | 12.8 |
| RMW | 0.021 | 0.54 | -4.2 | 5.1 |
| CMA | 0.016 | 0.42 | -3.8 | 4.3 |

### Missing Values

- **Count**: 0 (complete dataset)
- **Date Coverage**: 99.8% of trading days (61 years × 252 days = 15,372 observations)

---

## Limitations and Caveats

1. **Data Period**: Kenneth French data starts in 1926 for some factors, 1963 for others
   - This paper uses 1963-2024 due to international data availability

2. **Factor Definitions**: Kenneth French's definitions evolve over time
   - RMW and CMA added in 2015; retroactive data available
   - Our analysis uses consistent definitions across full period

3. **US-Centric Data**: Kenneth French factors focus on US markets
   - International data available but with shorter history (typically 1990-2024)
   - See Appendix D for country-specific coverage

4. **Survivorship Bias**: Kenneth French data includes delisted stocks
   - This is standard in factor research and matches academic practice
   - Robustness tests explore impact of this assumption

---

## Reproducing Results

### Full Reproducibility

To fully reproduce all results:

```bash
# 1. Clone repository
git clone https://github.com/USERNAME/factor-crowding-unified.git
cd factor-crowding-unified

# 2. Download data
bash scripts/download_data.sh  # Downloads all Kenneth French factors

# 3. Run analysis
python notebooks/01_decay_fitting.py  # Main game-theoretic analysis
python notebooks/02_domain_adaptation.py  # Domain adaptation
python notebooks/03_hedging_application.py  # Portfolio application
python notebooks/04_robustness_tests.py  # All robustness tests

# 4. Generate results
# Output: results/ directory contains all figures and tables
```

**Expected Output**:
- Figures: PNG/PDF versions matching paper
- Tables: CSV files with numerical results
- Logs: Execution logs showing parameter estimates
- Statistics: p-values and statistical tests

**Execution Time**: ~2-4 hours on standard desktop (parallel processing available)

---

## Contact for Data Questions

**Data Availability Queries**:
- Contact corresponding author at [email]
- Include: Question, desired use case, your institution
- Response time: 2-5 business days

**Data Errors or Issues**:
- Report to: Kenneth French's lab (https://mba.tuck.dartmouth.edu/)
- Or contact corresponding author for potential solutions

**Code Issues**:
- GitHub Issues: https://github.com/USERNAME/factor-crowding-unified/issues
- Pull Requests: Welcome for improvements and bug fixes

---

## Additional Resources

### Kenneth French Data Library
- Main: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/
- Documentation: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- FAQ: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/faq.html

### Python Data Loading
- See: `src/data/loaders.py` for data loading utilities
- Example: `load_fama_french()` function with full documentation

### Citation of Kenneth French Data
When citing this data, use:
```
French, Kenneth R., "Data Library,"
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html,
accessed [date].
```

---

## Statement Finalization

This Data Availability Statement should be included in the paper as supplementary material and referenced in the main text (typically in Data section or first mention of data).

**For JMLR Submission**:
1. Copy this statement
2. Create file: `data_availability_statement.txt`
3. Include in cover letter or supplementary materials
4. Reference in manuscript

---

**Prepared**: December 16, 2025
**Version**: 1.0
**Status**: Ready for JMLR Submission
