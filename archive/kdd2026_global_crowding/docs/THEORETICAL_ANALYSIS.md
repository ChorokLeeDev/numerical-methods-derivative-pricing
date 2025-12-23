# Theoretical Analysis: Why Temporal-MMD Works

## 1. Problem Setup

### Standard Domain Adaptation
Given:
- Source domain $\mathcal{D}_S = \{(x_i^s, y_i^s)\}_{i=1}^{n_s}$ with distribution $P_S$
- Target domain $\mathcal{D}_T = \{x_j^t\}_{j=1}^{n_t}$ with distribution $P_T$

Goal: Learn $f: \mathcal{X} \to \mathcal{Y}$ that minimizes target risk $\mathbb{E}_{P_T}[\ell(f(x), y)]$

### MMD-based Adaptation
Standard MMD minimizes:
$$\mathcal{L} = \mathcal{L}_{task}(f, \mathcal{D}_S) + \lambda \cdot \text{MMD}^2(P_S, P_T)$$

where:
$$\text{MMD}^2(P, Q) = \mathbb{E}_{x,x' \sim P}[k(x,x')] - 2\mathbb{E}_{x \sim P, y \sim Q}[k(x,y)] + \mathbb{E}_{y,y' \sim Q}[k(y,y')]$$

---

## 2. The Regime Shift Problem

### Assumption: Regime-Dependent Distributions

In time series domains (especially finance), the data generating process varies by **regime**:

$$P_S(x,y) = \sum_{r=1}^R \pi_S^r \cdot P_S^r(x,y)$$
$$P_T(x,y) = \sum_{r=1}^R \pi_T^r \cdot P_T^r(x,y)$$

where:
- $r \in \{1, ..., R\}$ are regimes (e.g., bull/bear, high/low volatility)
- $\pi_S^r, \pi_T^r$ are regime proportions
- $P_S^r, P_T^r$ are regime-conditional distributions

### Problem with Standard MMD

When $\pi_S \neq \pi_T$ (regime proportions differ), standard MMD tries to match:
$$P_S = \sum_r \pi_S^r P_S^r \quad \text{to} \quad P_T = \sum_r \pi_T^r P_T^r$$

This is **incorrect** because:
- If US has 45% high-volatility periods and UK has 16%
- Standard MMD will try to match US high-vol to UK low-vol
- Results in negative transfer

---

## 3. Temporal-MMD: Regime-Conditional Adaptation

### Proposed Loss Function

$$\mathcal{L}_{T-MMD} = \mathcal{L}_{task} + \lambda \sum_{r=1}^R w_r \cdot \text{MMD}^2(P_S^r, P_T^r)$$

Key difference: Match distributions **within each regime** separately.

### Theorem 1: Variance Reduction

**Statement**: Under regime-conditional matching, the estimation variance is reduced when regime proportions differ.

**Sketch of Proof**:

Standard MMD estimator variance:
$$\text{Var}[\widehat{\text{MMD}}^2(P_S, P_T)] = O\left(\frac{1}{n}\right)$$

Temporal-MMD estimator variance:
$$\text{Var}[\widehat{\text{MMD}}^2_{T}] = \sum_r w_r^2 \cdot \text{Var}[\widehat{\text{MMD}}^2(P_S^r, P_T^r)]$$

When $\pi_S^r \neq \pi_T^r$, the cross-regime terms in standard MMD add noise:
$$\text{Var}[\widehat{\text{MMD}}^2(P_S, P_T)] \geq \text{Var}[\widehat{\text{MMD}}^2_T] + \text{cross-regime noise}$$

The noise term is proportional to $|\pi_S - \pi_T|_2^2$.

### Theorem 2: Target Risk Bound

**Statement**: The target risk is bounded by:

$$\mathcal{R}_T(f) \leq \mathcal{R}_S(f) + \sum_r w_r \cdot d_{\mathcal{H}}(P_S^r, P_T^r) + \epsilon_{regime}$$

where:
- $\mathcal{R}_T(f), \mathcal{R}_S(f)$ are target and source risks
- $d_{\mathcal{H}}$ is the $\mathcal{H}$-divergence
- $\epsilon_{regime}$ is the regime labeling error

**Interpretation**:
- Minimizing regime-conditional divergence controls target risk
- Better regime detection → tighter bound

---

## 4. Why 2 Regimes is Optimal

### Empirical Observation
Ablation study showed:
- 2 regimes: AUC = 0.691
- 4 regimes: AUC = 0.674

### Theoretical Explanation

**Bias-Variance Tradeoff**:

More regimes → Lower bias (finer conditional matching)
                → Higher variance (fewer samples per regime)

Optimal $R$ balances:
$$R^* = \arg\min_R \left[ \text{Bias}(R) + \text{Var}(R) \right]$$

For financial data with ~600 monthly samples:
- 2 regimes: ~300 samples per regime ✓
- 4 regimes: ~150 samples per regime ✗ (insufficient)

---

## 5. Connection to Existing Theory

### Domain Adaptation Theory (Ben-David et al., 2010)

Standard bound:
$$\mathcal{R}_T(h) \leq \mathcal{R}_S(h) + d_{\mathcal{H}\Delta\mathcal{H}}(P_S, P_T) + \lambda^*$$

Our contribution: Replace global divergence with regime-conditional sum:
$$d_{\mathcal{H}\Delta\mathcal{H}}(P_S, P_T) \to \sum_r w_r \cdot d_{\mathcal{H}\Delta\mathcal{H}}(P_S^r, P_T^r)$$

This is tighter when $\pi_S \neq \pi_T$.

### Conditional Distribution Matching (Long et al., 2018)

Similar to CDAN (Conditional Domain Adversarial Networks), but:
- CDAN conditions on class labels $y$
- Temporal-MMD conditions on **regime labels** $r$

Our insight: In time series, the **temporal regime** is a more meaningful conditioning variable than class labels.

---

## 6. Algorithm Summary

```
Algorithm: Temporal-MMD Training

Input: Source data D_S, Target data D_T, regime detector g(·)

1. Detect regimes:
   r_S = g(D_S), r_T = g(D_T)

2. For each epoch:
   a. Sample batch from source: (x_S, y_S, r_S)
   b. Sample batch from target: (x_T, r_T)

   c. Compute features:
      f_S = FeatureExtractor(x_S)
      f_T = FeatureExtractor(x_T)

   d. Compute losses:
      L_task = CrossEntropy(Classifier(f_S), y_S)
      L_mmd = Σ_r w_r · MMD(f_S[r_S=r], f_T[r_T=r])

   e. Update: θ ← θ - α∇(L_task + λ·L_mmd)

Output: Adapted model f(·)
```

---

## 7. Limitations and Future Work

### Current Limitations
1. **Regime detection**: Assumes regimes are observable/detectable
2. **Discrete regimes**: Doesn't handle continuous regime transitions
3. **Same number of regimes**: Assumes source and target have same R

### Future Directions
1. **Learnable regime detection**: End-to-end regime learning
2. **Continuous regimes**: Soft regime assignments via attention
3. **Regime alignment**: Handle different regime structures across domains

---

## References

1. Ben-David, S., et al. (2010). "A theory of learning from different domains."
2. Long, M., et al. (2015). "Learning Transferable Features with Deep Adaptation Networks."
3. Long, M., et al. (2018). "Conditional Adversarial Domain Adaptation."
4. Ganin, Y., et al. (2016). "Domain-Adversarial Training of Neural Networks."
