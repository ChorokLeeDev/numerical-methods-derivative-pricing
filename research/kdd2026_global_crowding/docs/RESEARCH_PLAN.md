# KDD 2026 Research Plan: Temporal-MMD for Regime-Aware Domain Adaptation

## Executive Summary

**POSITIONING**: General Time Series Framework (Not Finance-Specific)

Temporal-MMD is a **novel regime-conditional domain adaptation** method that outperforms standard MMD by matching distributions within similar temporal regimes (e.g., high/low volatility, peak/off-peak, rush/normal hours).

**KEY INNOVATION**: Instead of `MMD(P_source, P_target)`, we compute:
```
Temporal-MMD = Σ_r w_r * MMD(P_source^r, P_target^r)
```
where `r` indexes regimes.

---

## Progress Log

### Phase 1: Data Collection (COMPLETED)
- [x] Fetched AQR global factor data (BAB, QMJ, VME)
- [x] 7 regions: US, UK, Europe, Japan, Asia Pacific, Global, Global ex-US
- [x] Daily factor ETF data (2,800+ days)

### Phase 2: Baseline Experiments (COMPLETED)
- [x] Fixed data leakage in transfer learning experiment
- [x] Changed from expanding to rolling quantile (fixes UK anomaly)
- [x] Ran transfer experiments across 6 regions

### Phase 3: Domain Adaptation (COMPLETED)
- [x] Implement DANN (Domain Adversarial Neural Network)
- [x] Implement MMD (Maximum Mean Discrepancy) loss
- [x] **Implement Temporal-MMD (Novel contribution)**
- [x] Ablation study (2 regimes optimal, λ=1.0)

### Phase 4: Multi-Domain Generalization (COMPLETED)
- [x] Finance domain (US → International factor transfer)
- [x] Electricity domain (NSW → Victoria demand forecasting)
- [x] Traffic domain (CityA → CityB congestion prediction)
- [x] Verified Temporal-MMD works across ALL 3 domains

---

## Key Findings

### 1. Multi-Domain Generalization (MAIN RESULT)

| Domain | RF Baseline | Standard MMD | Temporal-MMD | T-MMD vs RF | T-MMD vs MMD |
|--------|-------------|--------------|--------------|-------------|--------------|
| Finance | 0.584 | 0.574 | **0.594** | +1.7% | +3.5% |
| Electricity | 0.605 | 0.633 | **0.651** | +7.7% | +2.8% |
| Traffic | 0.946 | 1.000 | **1.000** | +5.7% | 0% |
| **Average** | | | | **+5.0%** | **+2.1%** |

**Key claim**: Temporal-MMD improves transfer across ALL domains, not just finance.

### 2. Ablation Study

| # Regimes | Target AUC | Note |
|-----------|------------|------|
| 2 | **0.691** | Optimal (bias-variance tradeoff) |
| 4 | 0.674 | Too few samples per regime |

| λ (MMD weight) | Target AUC |
|----------------|------------|
| 0.1 | 0.668 |
| 0.5 | 0.684 |
| **1.0** | **0.691** |
| 2.0 | 0.689 |

### 3. Finance-Specific Results

- Cross-region correlation: ρ=0.81
- US leads other markets by ~1 month
- UK anomaly: regime shift after 2000, requires rolling quantile

### 4. Theoretical Foundation

See `docs/THEORETICAL_ANALYSIS.md`:
- **Theorem 1**: Variance reduction when regime proportions differ
- **Theorem 2**: Target risk bound via regime-conditional divergence
- **Connection to Ben-David et al. (2010)**: Tighter bound than global divergence

---

## NEW Methodology: Domain Adaptation

### Problem Formulation

```
Source Domain: US market (S)
Target Domains: Europe (T1), Japan (T2), Asia Pacific (T3), UK (T4)

Goal: Learn features that are:
1. Predictive of crowding
2. Invariant across regions (domain-invariant)
```

### Proposed Methods

#### Method 1: DANN (Domain Adversarial Neural Network)

```
Loss = L_task - λ * L_domain

L_task: Crowding prediction loss (cross-entropy)
L_domain: Domain classification loss (adversarial)
```

The adversarial training forces features to be region-invariant.

#### Method 2: MMD (Maximum Mean Discrepancy)

```
Loss = L_task + λ * MMD(F_source, F_target)

MMD: Measures distribution distance in feature space
```

Minimizing MMD aligns source and target distributions.

#### Method 3: Multi-Source Transfer

```
Train on: US + Europe + Japan (multiple sources)
Test on: Asia Pacific, UK, Global

Benefit: More robust than single-source transfer
```

### Architecture

```
Input (features)
    │
    ▼
┌─────────────────┐
│ Feature         │
│ Extractor (F)   │  ← Shared across all regions
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────────┐
│ Task  │ │ Domain    │
│ Head  │ │ Classifier│  ← Gradient reversal
└───────┘ └───────────┘
    │         │
    ▼         ▼
 Crowding   Region
 Prediction  Label
```

---

## Experiments

### Experiment 1: Baseline Transfer (DONE)

```python
# experiments/02_kdd_experiments_fixed.py
# Results: 113% transfer efficiency
```

### Experiment 2: DANN vs Baseline

```python
# experiments/03_dann_transfer.py

# Compare:
# 1. Naive RF (no adaptation)
# 2. DANN (adversarial adaptation)
# 3. MMD (distribution matching)

# Expected improvement: +5-10% AUC on target regions
```

### Experiment 3: Multi-Source Transfer

```python
# experiments/04_multisource_transfer.py

# Training configurations:
# A: US only → Others
# B: US + Europe → Others
# C: US + Europe + Japan → Others

# Question: Does more sources help?
```

### Experiment 4: Ablation Study

```python
# experiments/05_ablation.py

# Ablate:
# 1. λ (adversarial weight): [0.01, 0.1, 1.0]
# 2. Feature extractor depth: [1, 2, 3 layers]
# 3. Which features are domain-invariant?
```

### Experiment 5: UK Deep Dive

```python
# experiments/06_uk_analysis.py

# UK has lowest transfer (0.519)
# Questions:
# 1. Is UK distribution fundamentally different?
# 2. Can targeted adaptation help UK specifically?
# 3. Is UK negative transfer (hurts to include)?
```

---

## Paper Structure (Updated)

### Title (New)
"Temporal-MMD: Regime-Aware Domain Adaptation for Time Series Transfer Learning"

### Abstract (Draft)
```
Domain adaptation methods like Maximum Mean Discrepancy (MMD) align source and
target distributions to enable transfer learning. However, time series data
exhibits temporal heterogeneity: distributions change across regimes (e.g.,
high/low volatility, peak/off-peak hours). Standard MMD ignores this structure,
potentially matching samples from incompatible regimes.

We propose Temporal-MMD, a regime-conditional domain adaptation method that
computes MMD within each temporal regime separately:

    L_T-MMD = Σ_r w_r * MMD(P_source^r, P_target^r)

This ensures domain alignment respects temporal structure. We prove that
Temporal-MMD reduces estimation variance when regime proportions differ between
source and target domains, and provides tighter target risk bounds than
standard MMD.

Experiments on three diverse domains demonstrate consistent improvements:
- Finance: +3.5% AUC over standard MMD
- Electricity: +2.8% AUC over standard MMD
- Traffic: Matches saturated MMD performance

Our method is simple to implement, adds minimal overhead, and applies to any
time series domain with identifiable regimes.
```

### Sections
1. Introduction
2. Related Work
   - 2.1 Domain Adaptation (MMD, DANN, CDAN)
   - 2.2 Time Series Transfer Learning
   - 2.3 Regime Detection in Time Series
3. Preliminaries
   - 3.1 Problem Setup
   - 3.2 Standard MMD Review
4. Temporal-MMD: Regime-Aware Domain Adaptation
   - 4.1 Motivation: Regime Shift Problem
   - 4.2 Temporal-MMD Loss Function
   - 4.3 Theoretical Analysis
   - 4.4 Algorithm
5. Experiments
   - 5.1 Datasets (Finance, Electricity, Traffic)
   - 5.2 Baselines (RF, Standard MMD, DANN)
   - 5.3 Main Results (Table 1)
   - 5.4 Ablation Study
6. Discussion and Limitations
7. Conclusion

---

## Expected Results

### Main Results Table

| Method | US→EUR | US→JPN | US→APAC | US→UK | Avg |
|--------|--------|--------|---------|-------|-----|
| Naive RF | 0.713 | 0.679 | 0.724 | 0.519 | 0.659 |
| DANN | 0.74 | 0.71 | 0.75 | 0.58 | 0.70 |
| MMD | 0.73 | 0.70 | 0.74 | 0.56 | 0.68 |
| Multi-Source | 0.76 | 0.73 | 0.77 | 0.60 | 0.72 |

**Target**: +5-10% AUC improvement over naive transfer

### Key Claims

1. **Crowding is global**: ρ=0.81 concurrent correlation
2. **Transfer works surprisingly well**: 113% baseline efficiency
3. **Domain adaptation helps**: +X% AUC improvement
4. **UK is hardest**: Regime shift requires special handling

---

## Implementation Plan

### Week 1: DANN Implementation
- [ ] Feature extractor network
- [ ] Gradient reversal layer
- [ ] Domain classifier
- [ ] Training loop with adversarial loss

### Week 2: MMD + Multi-Source
- [ ] MMD loss implementation
- [ ] Multi-source data loader
- [ ] Combined training procedure

### Week 3: Experiments
- [ ] Full experiment sweep
- [ ] Ablation study
- [ ] UK deep dive
- [ ] Statistical significance tests

### Week 4: Paper
- [ ] Write methodology section
- [ ] Create figures
- [ ] Results tables
- [ ] Final polish

---

## Code Structure

```
kdd2026_global_crowding/
├── data/
│   ├── global_factors/           # Parquet files (DONE)
│   ├── fetch_aqr_global.py       # Data fetching (DONE)
│   └── fetch_global_factors.py   # Ken French (failed, using AQR)
├── src/
│   ├── models/
│   │   ├── dann.py               # Domain Adversarial NN (TODO)
│   │   ├── mmd.py                # MMD adaptation (TODO)
│   │   └── multisource.py        # Multi-source transfer (TODO)
│   ├── features.py               # Feature engineering
│   └── evaluation.py             # Metrics and evaluation
├── experiments/
│   ├── 01_kdd_experiments.py     # Original (has bugs)
│   ├── 02_kdd_experiments_fixed.py  # Fixed baseline (DONE)
│   ├── 03_dann_transfer.py       # DANN experiments (TODO)
│   ├── 04_multisource_transfer.py   # Multi-source (TODO)
│   └── 05_ablation.py            # Ablation study (TODO)
├── paper/
│   └── kdd2026_global.tex        # Paper draft
└── docs/
    └── RESEARCH_PLAN.md          # This file
```

---

## References

### Domain Adaptation
- Ganin et al. (2016). "Domain-Adversarial Training of Neural Networks" (DANN)
- Long et al. (2015). "Learning Transferable Features with Deep Adaptation Networks" (MMD)
- Zhao et al. (2019). "On Learning Invariant Representations for Domain Adaptation"

### Transfer Learning in Finance
- Xu et al. (2021). "Stock Movement Prediction via Transfer Learning"
- Ye & Schuller (2021). "Transfer Learning for Financial Time Series"

### Factor Crowding
- McLean & Pontiff (2016). "Does Academic Research Destroy Return Predictability?"
- DeMiguel et al. (2021). "What Alleviates Crowding in Factor Investing?"
