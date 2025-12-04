# KDD 2026 Research Plan: Mining Factor Crowding at Global Scale

## Executive Summary

Extend factor crowding analysis to global markets using ML-based detection, validating the mechanical/judgment taxonomy across 6 regions and 10+ factors.

---

## 1. Problem Statement

### Current Limitations

The existing factor crowding paper analyzes only US markets:

```
- 8 factors (US Fama-French only)
- No validation that patterns generalize globally
- Model-based detection (not ML)
```

### Research Questions

1. Does hyperbolic decay hold for factors outside the US?
2. Does the mechanical/judgment taxonomy generalize?
3. Can ML methods outperform model-based crowding detection?
4. Does a model trained on US data generalize to other regions?

---

## 2. Data Sources

### 2.1 Ken French International Factors

Available via `pandas_datareader`:

| Region | Dataset Code | Factors |
|--------|--------------|---------|
| US | `F-F_Research_Data_5_Factors_2x3` | MKT, SMB, HML, RMW, CMA |
| Developed ex-US | `Developed_5_Factors` | Same |
| Europe | `Europe_5_Factors` | Same |
| Japan | `Japan_5_Factors` | Same |
| Asia Pacific ex-Japan | `Asia_Pacific_ex_Japan_5_Factors` | Same |
| Emerging | `Emerging_5_Factors` | Same |

Plus momentum for each: `{Region}_Mom_Factor`

### 2.2 AQR Factor Data

Download from AQR Data Library (Excel files):

| Factor | URL | Description |
|--------|-----|-------------|
| QMJ | Quality Minus Junk | Profitability, safety, growth |
| BAB | Betting Against Beta | Long low-beta, short high-beta |

### 2.3 Data Schema

```python
REGIONS = ['US', 'Developed', 'Europe', 'Japan', 'APAC', 'EM']

FACTORS = {
    'core': ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'Mom'],
    'extended': ['QMJ', 'BAB', 'ST_Rev', 'LT_Rev'],
}

TAXONOMY = {
    'mechanical': ['Mom', 'ST_Rev', 'LT_Rev', 'BAB'],
    'judgment': ['HML', 'RMW', 'CMA', 'QMJ'],
    'hybrid': ['SMB', 'MKT'],
}
```

---

## 3. Methodology

### 3.1 Global Robustness Analysis

**Goal**: Verify hyperbolic decay model holds globally

**Method**:
1. Compute rolling 36-month Sharpe for each factor × region
2. Fit hyperbolic, linear, exponential decay models
3. Compare R² across regions

**Hypotheses**:
| Region | Momentum R² | Rationale |
|--------|-------------|-----------|
| US | 0.65 | Baseline (already validated) |
| Developed | 0.50-0.60 | Similar market structure |
| Europe | 0.45-0.55 | Slightly less crowded |
| Japan | 0.40-0.50 | Unique market dynamics |
| EM | 0.30-0.40 | Less efficient, slower crowding |

### 3.2 Taxonomy Expansion

**Goal**: Validate mechanical/judgment classification with new factors

**Hypotheses**:
| Factor | Expected Category | Signal Clarity |
|--------|-------------------|----------------|
| BAB | Mechanical | "Buy low beta" is unambiguous |
| QMJ | Judgment | "Quality" requires interpretation |

**Validation**:
- Mechanical factors: R² > 0.3 for hyperbolic decay
- Judgment factors: R² < 0.15

### 3.3 ML Crowding Detection

**Goal**: Outperform model residuals with learned representations

**Features**:
```python
features = {
    'returns': ['ret_1m', 'ret_3m', 'ret_6m', 'ret_12m'],
    'volatility': ['vol_1m', 'vol_3m', 'vol_realized'],
    'correlations': ['cross_factor_corr', 'within_factor_corr'],
    'momentum': ['factor_momentum', 'cross_momentum'],
    'regime': ['vix_level', 'market_state'],
}
```

**Models**:
1. **Random Forest**: Baseline, interpretable
2. **XGBoost**: Better performance, feature importance
3. **LSTM**: Temporal patterns, regime persistence

**Target**:
```python
# Binary classification
y = (actual_sharpe < predicted_sharpe - threshold).astype(int)
# 1 = crowded, 0 = not crowded

# Or regression
y = actual_sharpe - predicted_sharpe  # Residual
```

### 3.4 Cross-Region Generalization

**Goal**: Test transfer learning across regions

**Experiment**:
1. Train on US data (longest history)
2. Test on other regions without retraining
3. Compare to region-specific models

**Questions**:
- Does US model transfer to Developed markets?
- Is retraining needed for EM?
- Which features transfer best?

---

## 4. Experiments

### Experiment 1: Global Decay Analysis

```python
# experiments/01_global_robustness.py

for region in REGIONS:
    factors = load_regional_factors(region)
    for factor in factors.columns:
        sharpe = rolling_sharpe(factors[factor], window=36)

        # Fit models
        r2_hyperbolic = fit_hyperbolic(sharpe)
        r2_linear = fit_linear(sharpe)
        r2_exponential = fit_exponential(sharpe)

        results.append({
            'region': region,
            'factor': factor,
            'r2_hyperbolic': r2_hyperbolic,
            'r2_linear': r2_linear,
            'r2_exponential': r2_exponential,
        })

# Output: 6 regions × 10 factors × 3 models = 180 R² values
```

### Experiment 2: Taxonomy Validation

```python
# experiments/02_taxonomy_expansion.py

# Group factors by taxonomy
mechanical = ['Mom', 'ST_Rev', 'LT_Rev', 'BAB']
judgment = ['HML', 'RMW', 'CMA', 'QMJ']

# Compare R² distributions
mechanical_r2 = results[results.factor.isin(mechanical)].r2_hyperbolic
judgment_r2 = results[results.factor.isin(judgment)].r2_hyperbolic

# Statistical test
from scipy.stats import mannwhitneyu
stat, pvalue = mannwhitneyu(mechanical_r2, judgment_r2)
```

### Experiment 3: ML vs Model Residual

```python
# experiments/03_ml_crowding.py

# Prepare features and target
X, y = prepare_crowding_dataset(factors, signals)

# Walk-forward validation
for train_end in range(min_train, len(X) - test_size):
    X_train = X[:train_end]
    y_train = y[:train_end]
    X_test = X[train_end:train_end + test_size]
    y_test = y[train_end:train_end + test_size]

    # Train models
    rf = RandomForestClassifier().fit(X_train, y_train)
    xgb = XGBClassifier().fit(X_train, y_train)

    # Evaluate
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])

    # Compare to model residual
    residual_auc = roc_auc_score(y_test, -signals['residual'][test_idx])
```

### Experiment 4: LSTM Temporal Model

```python
# experiments/04_lstm_temporal.py

class CrowdingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(lstm_out[:, -1, :]))

# Sequence preparation
def create_sequences(X, y, seq_len=12):
    sequences = []
    for i in range(seq_len, len(X)):
        sequences.append((X[i-seq_len:i], y[i]))
    return sequences
```

### Experiment 5: Cross-Region Transfer

```python
# experiments/06_cross_region.py

# Train on US
us_model = XGBClassifier().fit(X_us, y_us)

# Test on other regions (zero-shot)
for region in ['Developed', 'Europe', 'Japan', 'APAC', 'EM']:
    X_region, y_region = load_region_data(region)

    # Zero-shot transfer
    auc_transfer = roc_auc_score(y_region, us_model.predict_proba(X_region)[:, 1])

    # Region-specific
    region_model = XGBClassifier().fit(X_region, y_region)
    auc_specific = cross_val_score(region_model, X_region, y_region, scoring='roc_auc')

    print(f"{region}: Transfer AUC={auc_transfer:.3f}, Specific AUC={auc_specific.mean():.3f}")
```

---

## 5. Paper Structure

### Title
"Mining Factor Crowding at Global Scale: ML-Based Detection Across 60 Factor-Region Pairs"

### Abstract (Draft)
```
Factor crowding—the erosion of alpha as strategies become popular—has been
documented primarily in US markets. We extend crowding analysis to a global
scale, examining 10 factors across 6 regions (60 factor-region pairs) over
30+ years. Our contributions are: (1) we validate that hyperbolic alpha
decay generalizes to developed markets but weakens in emerging markets;
(2) we confirm the mechanical/judgment taxonomy holds globally, with BAB
(Betting Against Beta) exhibiting mechanical crowding patterns; (3) we
demonstrate that ML methods (LSTM, XGBoost) outperform model-based residuals
for crowding detection with AUC improvements of 15-20%; and (4) we show that
models trained on US data transfer effectively to other developed markets
but require adaptation for emerging markets. Our work provides the first
large-scale empirical validation of factor crowding dynamics across global
markets.
```

### Sections
1. Introduction
2. Related Work
3. Data Description
4. Methodology
   - 4.1 Global Robustness Analysis
   - 4.2 Taxonomy Expansion
   - 4.3 ML Crowding Detection
   - 4.4 Cross-Region Generalization
5. Experiments and Results
6. Discussion
7. Conclusion

---

## 6. Expected Results

### Global R² Matrix

| Factor | US | Dev | EUR | JPN | APAC | EM |
|--------|-----|-----|-----|-----|------|-----|
| Mom | 0.65 | 0.55 | 0.50 | 0.45 | 0.40 | 0.35 |
| BAB | 0.45 | 0.40 | 0.38 | 0.35 | 0.30 | 0.25 |
| HML | 0.05 | 0.06 | 0.05 | 0.04 | 0.05 | 0.04 |
| QMJ | 0.08 | 0.07 | 0.06 | 0.05 | 0.05 | 0.04 |

### ML Performance

| Model | AUC | vs Residual |
|-------|-----|-------------|
| Residual (baseline) | 0.58 | - |
| Random Forest | 0.65 | +12% |
| XGBoost | 0.68 | +17% |
| LSTM | 0.70 | +21% |

### Cross-Region Transfer

| Target | Zero-shot AUC | Specific AUC | Transfer Rate |
|--------|---------------|--------------|---------------|
| Developed | 0.65 | 0.67 | 97% |
| Europe | 0.62 | 0.66 | 94% |
| Japan | 0.58 | 0.64 | 91% |
| EM | 0.52 | 0.61 | 85% |

---

## 7. Figures and Tables

### Figures
1. **Fig 1**: Global decay model fit (6-panel: one per region)
2. **Fig 2**: R² heatmap (factors × regions)
3. **Fig 3**: Taxonomy boxplot (mechanical vs judgment, global)
4. **Fig 4**: ML model comparison (ROC curves)
5. **Fig 5**: LSTM feature importance over time
6. **Fig 6**: Cross-region transfer performance

### Tables
1. **Table 1**: Data summary (regions, factors, time periods)
2. **Table 2**: R² by factor × region
3. **Table 3**: Taxonomy classification results
4. **Table 4**: ML model performance
5. **Table 5**: Cross-region transfer results

---

## 8. Risk & Mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Regional data too short | Medium | Focus on developed markets, note EM limitations |
| ML doesn't outperform | Low | Report negative result honestly, analyze why |
| Taxonomy doesn't generalize | Medium | Interesting finding either way |
| AQR data access issues | Low | Fallback to Ken French only |

---

## 9. Success Criteria

1. **Global validation**: Hyperbolic decay R² > 0.4 for momentum in developed markets
2. **Taxonomy**: Mechanical vs judgment R² difference significant (p < 0.01)
3. **ML improvement**: AUC > 0.65 for at least one ML method
4. **Transfer**: US model achieves > 90% of region-specific performance in developed markets

---

## 10. Implementation Notes

### Data Fetching Pattern

```python
# data/fetch_global_factors.py

import pandas_datareader.data as web

def fetch_region_factors(region: str) -> pd.DataFrame:
    """Fetch FF5 + Momentum for a region."""

    region_codes = {
        'US': ('F-F_Research_Data_5_Factors_2x3', 'F-F_Momentum_Factor'),
        'Developed': ('Developed_5_Factors', 'Developed_Mom_Factor'),
        'Europe': ('Europe_5_Factors', 'Europe_Mom_Factor'),
        'Japan': ('Japan_5_Factors', 'Japan_Mom_Factor'),
        'APAC': ('Asia_Pacific_ex_Japan_5_Factors', 'Asia_Pacific_ex_Japan_Mom_Factor'),
        'EM': ('Emerging_5_Factors', 'Emerging_Mom_Factor'),
    }

    ff5_code, mom_code = region_codes[region]

    ff5 = web.DataReader(ff5_code, 'famafrench', start='1990')[0] / 100
    mom = web.DataReader(mom_code, 'famafrench', start='1990')[0] / 100
    mom.columns = ['Mom']

    return ff5.join(mom, how='inner')
```

### AQR Data Handling

```python
# data/fetch_aqr_factors.py

import requests
import pandas as pd

AQR_URLS = {
    'QMJ': 'https://images.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Quality-Minus-Junk-Factors-Monthly.xlsx',
    'BAB': 'https://images.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Monthly.xlsx',
}

def fetch_aqr_factor(factor: str) -> pd.DataFrame:
    """Download and parse AQR factor data."""
    url = AQR_URLS[factor]
    df = pd.read_excel(url, sheet_name=f'{factor} Factors', skiprows=18)
    # Parse date column, select regions...
    return df
```

---

## 11. References

### Global Factor Studies
- Fama, French (2012). "Size, Value, and Momentum in International Stock Returns"
- Asness et al. (2013). "Value and Momentum Everywhere"
- Hou, Karolyi, Kho (2011). "What Factors Drive Global Stock Returns?"

### Factor Crowding
- McLean, Pontiff (2016). "Does Academic Research Destroy Stock Return Predictability?"
- DeMiguel et al. (2021). "What Alleviates Crowding in Factor Investing?"
- Our prior work (2024). "Not All Factors Crowd Equally"

### ML in Finance
- Gu, Kelly, Xiu (2020). "Empirical Asset Pricing via Machine Learning"
- Chen, Pelger, Zhu (2023). "Deep Learning in Asset Pricing"
