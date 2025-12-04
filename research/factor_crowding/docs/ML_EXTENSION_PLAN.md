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

### Phase 1: Data Preparation (Week 1-2)
- [ ] Compile feature matrix (returns, correlations, vol)
- [ ] Create target labels (crash = bottom 10%)
- [ ] Set up walk-forward CV splits
- [ ] Handle class imbalance

### Phase 2: Crowding Detection ML (Week 3-5)
- [ ] Implement Random Forest baseline
- [ ] Implement Gradient Boosting
- [ ] Implement LSTM model
- [ ] Compare detection accuracy vs model residuals

### Phase 3: Tail Risk Neural Network (Week 6-8)
- [ ] Design network architecture
- [ ] Implement training pipeline
- [ ] Walk-forward backtesting
- [ ] Compare to statistical baseline

### Phase 4: Paper Rewrite (Week 9-12)
- [ ] New methodology section on ML approach
- [ ] Results comparing ML vs baseline
- [ ] Ablation studies
- [ ] Convert to ACM format

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
│   ├── crowding_ml.py          # NEW: ML crowding detection
│   ├── tail_risk_nn.py         # NEW: Neural network
│   └── features.py             # NEW: Feature engineering
├── experiments/
│   ├── 15_ml_crowding.py       # ML crowding experiments
│   ├── 16_nn_tail_risk.py      # NN tail risk experiments
│   └── 17_model_comparison.py  # Baseline vs ML comparison
└── paper/
    └── icaif2025_ml_crowding.tex  # New paper version
```

## Dependencies to Add

```python
# requirements.txt additions
torch>=2.0
scikit-learn>=1.3
xgboost>=2.0
optuna  # hyperparameter tuning
```

## Success Criteria

1. ML crowding detection outperforms model residuals
2. NN tail risk prediction: AUC > 0.65
3. Economic significance: Sharpe improvement or drawdown reduction
4. Clear comparison showing ML adds value
