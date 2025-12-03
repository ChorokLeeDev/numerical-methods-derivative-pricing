# Not All Factors Crowd Equally: A Game-Theoretic Model of Alpha Decay

**KDD Singapore 2026 Submission**

## Paper Files

- `factor_crowding_kdd2026.tex` - LaTeX source
- `figures/` - All figures (PNG)

## Abstract

We present a game-theoretic model of factor alpha decay driven by strategy crowding. Our model predicts that alpha decays hyperbolically as α(t) = K/(1 + λt), where K represents initial alpha capacity and λ is the rate of strategy discovery. Using Fama-French factor data from 1963-2024, we find that the model fits momentum factor decay well (R² = 0.65, in-sample) and correctly predicts the direction of continued decay out-of-sample (2016-2024). However, the model systematically over-predicts remaining alpha (predicted: 0.30, actual: 0.15), suggesting crowding *accelerated* beyond historical rates after 2015—coinciding with the proliferation of factor ETFs and commission-free trading. Crucially, the model fails for value (HML) and size (SMB) factors, which exhibit different decay dynamics. We argue this reflects a fundamental distinction: *mechanical* factors (momentum) crowd quickly due to unambiguous signals, while *judgment-based* factors (value) crowd slowly due to definitional ambiguity.

## Key Results

### 1. Model

N-player game where agents compete for fixed alpha capacity:

```
α(t) = K / (1 + λt)

Parameters (Momentum):
- K = 1.65 (initial Sharpe capacity)
- λ = 0.0145 (discovery rate per month)
- Half-life = 69 months ≈ 5.7 years
```

### 2. In-Sample Fit (1995-2015)

| Factor | R² | Interpretation |
|--------|-----|----------------|
| Momentum | 0.65 | Good fit ✓ |
| HML | poor | Different dynamics |
| SMB | poor | Different dynamics |

### 3. Out-of-Sample Prediction (2016-2024)

| Metric | Value |
|--------|-------|
| Direction | Correct ✓ |
| Predicted Sharpe | 0.30 |
| Actual Sharpe | 0.15 |
| RMSE | 0.19 |

**Key insight**: The over-prediction reveals crowding *acceleration* post-2015.

### 4. Residual Analysis

```
Mean residual: -0.157
% negative: 72%
```

Persistent negative residuals indicate faster-than-predicted decay, coinciding with:
- Factor ETF proliferation
- Commission-free trading (Robinhood, etc.)
- Retail access to systematic strategies

### 5. Not All Factors Crowd Equally

| Factor Type | Example | Signal Clarity | Crowding Speed | Model Fit |
|-------------|---------|----------------|----------------|-----------|
| Mechanical | Momentum | Clear (buy winners) | Fast | Good |
| Judgment | Value | Ambiguous (what's cheap?) | Slow | Poor |

## Figures

1. `fig1_factor_decay.png` - Rolling Sharpe over time (all factors)
2. `fig2_decade_comparison.png` - Sharpe by decade
3. `fig3_model_fit.png` - Hyperbolic decay fit (momentum)
4. `fig4_crowding.png` - Factor returns vs ETF crowding proxy
5. `fig5_momentum_oos.png` - Out-of-sample prediction detail
6. `fig6_all_factors_oos.png` - Cross-factor OOS comparison
7. `fig7_residuals.png` - Prediction residuals (crowding acceleration)
8. `fig8_cumulative_residual.png` - Cumulative residual analysis

## Two Contributions

1. **Equilibrium model** explains long-term decay (R² = 0.65 in-sample)
2. **Prediction residual** detects crowding acceleration

The gap between predicted and actual is itself a signal.

## Compilation

```bash
cd paper/
pdflatex factor_crowding_kdd2026.tex
bibtex factor_crowding_kdd2026
pdflatex factor_crowding_kdd2026.tex
pdflatex factor_crowding_kdd2026.tex
```

## Deadline

**December 7, 2025** (KDD Singapore 2026 Workshop Papers)
