# ML Extension Plan for ICAIF 2025

## Timeline
- **Deadline:** July 18, 2025
- **Available time:** ~7 months

## Current Paper (Baseline)
- Game-theoretic model: α(t) = K/(1+λt)
- Crowding signal: Model residual (Actual - Predicted Sharpe)
- Tail risk: Statistical analysis of crash probabilities

## ML Extensions Required

### 1. ML-Based Crowding Detection

**Goal:** Replace model residuals with learned crowding representation

**Approach A: Supervised Classification**
```
Input Features:
- Rolling factor returns (1M, 3M, 6M, 12M)
- Factor correlations (rolling)
- Factor volatility regime
- ETF flow data (if available)
- Cross-factor momentum signals

Target:
- Binary: High/Low crowding regime
- Or: Crowding score (regression)

Models to try:
- Random Forest (baseline)
- Gradient Boosting (XGBoost/LightGBM)
- LSTM for temporal patterns
- Transformer for attention over time
```

**Approach B: Unsupervised Regime Detection**
```
- Hidden Markov Model (HMM) for regime switching
- Variational Autoencoder (VAE) for crowding representation
- Clustering on factor return patterns
```

### 2. Neural Network Tail Risk Prediction

**Goal:** Predict P(crash | crowding state, factor characteristics)

**Architecture:**
```
Input Layer:
- Crowding signal (from ML detector above)
- Factor type embedding (momentum, reversal, value, etc.)
- Recent return features
- Volatility features

Hidden Layers:
- 2-3 dense layers with ReLU
- Dropout for regularization
- Batch normalization

Output Layer:
- Sigmoid for P(crash)
- Or: Distribution parameters for full predictive distribution
```

**Training:**
- Binary cross-entropy loss
- Class weights for imbalanced data (crashes are rare)
- Walk-forward validation (no lookahead)

### 3. Comparison Framework

| Method | Crowding Detection | Tail Risk | ICAIF Fit |
|--------|-------------------|-----------|-----------|
| Baseline (current) | Model residual | Statistical | ❌ No AI |
| ML-Enhanced | Random Forest | Logistic | ⚠️ Basic ML |
| Deep Learning | LSTM/Transformer | Neural Net | ✓ Strong |

## Implementation Plan

### Phase 1: Data Preparation ✅ COMPLETE
- [x] Compile feature matrix (returns, correlations, vol) - 168 features
- [x] Create target labels (crash = bottom 10%)
- [x] Set up walk-forward CV splits
- [x] Handle class imbalance

### Phase 2: Crowding Detection ML ✅ COMPLETE
- [x] Implement Random Forest baseline
- [x] Implement Gradient Boosting (XGBoost)
- [x] Implement LSTM model
- [x] Compare detection accuracy vs model residuals

### Phase 3: Tail Risk Neural Network ✅ COMPLETE
- [x] Design network architecture (TailRiskMLP with factor embedding)
- [x] Implement training pipeline
- [x] Walk-forward backtesting
- [x] Compare to statistical baseline

### Phase 4: Paper Rewrite (TODO)
- [ ] New methodology section on ML approach
- [ ] Results comparing ML vs baseline
- [ ] Ablation studies
- [ ] Convert to ACM format

## Results Summary (Dec 4, 2025)

### Walk-Forward Backtest Results

| Model | Mean AUC | Std | vs Baseline |
|-------|----------|-----|-------------|
| Baseline (model-residual) | 0.530 | 0.042 | - |
| **RandomForest** | **0.623** | 0.033 | **+0.094** |
| XGBoost | 0.591 | 0.045 | +0.061 |
| Neural Network (MLP) | 0.540 | - | +0.010 |

### Per-Factor AUC

| Factor | Crash% | Baseline | RF | XGBoost |
|--------|--------|----------|-------|---------|
| MKT | 11.2% | 0.604 | 0.625 | 0.500 |
| SMB | 11.9% | 0.569 | 0.617 | 0.618 |
| HML | 13.3% | 0.527 | 0.593 | 0.573 |
| RMW | 13.8% | 0.486 | 0.645 | 0.618 |
| CMA | 12.1% | 0.502 | 0.639 | 0.587 |
| **Mom** | 12.1% | 0.469 | **0.690** | 0.659 |
| ST_Rev | 15.7% | 0.532 | 0.587 | 0.614 |
| LT_Rev | 13.3% | 0.549 | 0.589 | 0.563 |

### Key Findings

1. **RandomForest is the best ML approach**
   - Mean AUC: 0.623 (vs 0.530 baseline)
   - Wins on 8/8 factors
   - Statistically significant: p = 0.0072

2. **Neural Network underperforms RF**
   - Limited data (654 samples per factor) hurts deep learning
   - Still useful for paper discussion of architecture tradeoffs

3. **Momentum has highest predictability**
   - RF AUC 0.690 for Momentum crashes
   - Aligns with trend-following vs mean-reverting hypothesis

## Key ICAIF Alignment

Paper would now address:
- ✓ "AI-driven risk management"
- ✓ "Financial time series analysis"
- ✓ "Validation and calibration of financial models"
- ✓ "Risk modeling and risk management"

## File Structure

```
research/factor_crowding/
├── src/
│   ├── crowding_signal.py      # Current model-based
│   ├── crowding_ml.py          # ML crowding detection (RF, XGBoost, LSTM)
│   ├── tail_risk_nn.py         # Neural network (MLP with factor embedding)
│   └── features.py             # Feature engineering (168 features)
├── experiments/
│   ├── 15_ml_vs_baseline_comparison.py   # RF vs baseline comparison
│   ├── 16_comprehensive_model_comparison.py  # All models comparison
│   └── ...
└── paper/
    └── icaif2026_factor_crowding.tex  # Paper to update with ML section
```

## Dependencies

```python
# requirements.txt additions
torch>=2.0
scikit-learn>=1.3
xgboost>=2.0
```

## Success Criteria

1. ✅ ML crowding detection outperforms model residuals
2. ⚠️ NN tail risk prediction: AUC = 0.54 (below 0.65 target)
3. ⏳ Economic significance: Sharpe improvement or drawdown reduction
4. ✅ Clear comparison showing ML adds value

## Next Steps

1. **Update paper with ML methodology section**
   - Feature engineering description
   - Walk-forward validation protocol
   - RandomForest results

2. **Add ablation studies**
   - Feature importance analysis
   - Rolling window sensitivity

3. **Convert to ACM sigconf format**
   - ICAIF uses ACM template
