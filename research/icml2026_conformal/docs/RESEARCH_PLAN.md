# ICML 2026 Research Plan: Conformal Prediction for Factor Crowding

## Executive Summary

Apply Conformal Prediction (CP) to factor crowding detection to provide distribution-free uncertainty quantification with guaranteed coverage.

---

## 1. Problem Statement

### Current Limitations

The existing factor crowding paper has a key weakness:

```
Regime effect: p = 0.62 (not statistically significant)
"We can't tell if the pattern is real or noise"
```

### Proposed Solution

Instead of point estimates with p-values, provide **prediction sets with coverage guarantees**:

```
Before: "Factor is crowded" (binary, uncertain)
After:  "Factor is {crowded} with 90% coverage" OR
        "Factor is {crowded, not_crowded} - uncertain"
```

---

## 2. Background: Conformal Prediction

### What is Conformal Prediction?

A framework for creating prediction sets/intervals with guaranteed coverage:

```
P(Y_new ∈ C(X_new)) ≥ 1 - α
```

For any user-specified α (e.g., 0.1 for 90% coverage).

### Key Properties

| Property | Description |
|----------|-------------|
| **Distribution-free** | No assumptions about data distribution |
| **Finite-sample valid** | Works for any sample size |
| **Model-agnostic** | Wraps any base predictor |
| **Exchangeability** | Only requires i.i.d. or exchangeable data |

### Types of Conformal Prediction

1. **Split Conformal**: Train/calibration split (simple, efficient)
2. **Full Conformal**: Leave-one-out (more powerful, expensive)
3. **Conformalized Quantile Regression (CQR)**: For regression intervals

---

## 3. Research Design

### 3.1 Conformal Crowding Classification

**Goal**: Prediction sets for crowding regime

**Method**: Split Conformal Classification

```python
# Base classifier: Any model predicting P(crowded | features)
# Conformal wrapper: Calibrate to achieve coverage

class ConformalCrowdingClassifier:
    def __init__(self, base_model, alpha=0.1):
        self.base_model = base_model
        self.alpha = alpha

    def calibrate(self, X_cal, y_cal):
        # Compute nonconformity scores on calibration set
        probs = self.base_model.predict_proba(X_cal)
        self.scores = 1 - probs[np.arange(len(y_cal)), y_cal]
        self.threshold = np.quantile(self.scores, 1 - self.alpha)

    def predict_set(self, X_test):
        probs = self.base_model.predict_proba(X_test)
        # Include class if 1 - prob <= threshold
        return {c for c in classes if 1 - probs[c] <= self.threshold}
```

**Output Types**:
- `{crowded}`: Confident crowded
- `{not_crowded}`: Confident not crowded
- `{crowded, not_crowded}`: Uncertain (both possible)

### 3.2 Conformal Tail Risk Regression

**Goal**: Prediction intervals for factor returns during crowded periods

**Method**: Conformalized Quantile Regression (CQR)

```python
# Train quantile regressors for q_lo and q_hi
# Calibrate intervals to achieve coverage

class ConformalTailRisk:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q_lo = QuantileRegressor(quantile=alpha/2)
        self.q_hi = QuantileRegressor(quantile=1-alpha/2)

    def fit(self, X_train, y_train):
        self.q_lo.fit(X_train, y_train)
        self.q_hi.fit(X_train, y_train)

    def calibrate(self, X_cal, y_cal):
        lo = self.q_lo.predict(X_cal)
        hi = self.q_hi.predict(X_cal)
        scores = np.maximum(lo - y_cal, y_cal - hi)
        self.correction = np.quantile(scores, 1 - self.alpha)

    def predict_interval(self, X_test):
        lo = self.q_lo.predict(X_test) - self.correction
        hi = self.q_hi.predict(X_test) + self.correction
        return lo, hi
```

### 3.3 Calibration Analysis

**Goal**: Compare calibration of different methods

**Metrics**:
- **Coverage**: Fraction of true values in interval
- **Interval width**: Average width (efficiency)
- **Conditional coverage**: Coverage by regime/factor type

**Baselines**:
- Bayesian posterior intervals
- Bootstrap percentile intervals
- Naive Gaussian intervals

---

## 4. Experiments

### Experiment 1: Coverage Verification

**Question**: Does conformal achieve nominal coverage?

**Method**:
1. Split data: Train (60%), Calibration (20%), Test (20%)
2. Train base model on training set
3. Calibrate on calibration set
4. Evaluate coverage on test set

**Expected Results**:
| Method | Target Coverage | Actual Coverage |
|--------|-----------------|-----------------|
| Conformal | 90% | 90% ± 2% |
| Bayesian | 90% | 85-95% (depends on prior) |
| Bootstrap | 90% | 80-88% (often under-covers) |

### Experiment 2: Prediction Set Analysis

**Question**: When is crowding detection uncertain?

**Analysis**:
- Size distribution of prediction sets
- Correlation with market volatility
- Factor-specific patterns

**Hypothesis**: Prediction sets are larger (more uncertain) during:
- Regime transitions
- High volatility periods
- For judgment factors (vs mechanical)

### Experiment 3: Tail Risk Intervals

**Question**: Can we provide calibrated intervals for crash risk?

**Method**:
1. Predict return intervals during crowded/uncrowded regimes
2. Verify coverage separately for each regime
3. Compare interval widths

**Expected Results**:
- Wider intervals in crowded regimes (higher uncertainty)
- Asymmetric intervals (more downside in crowded)

### Experiment 4: Method Comparison

**Question**: Does conformal outperform alternatives?

**Comparison**:
| Method | Coverage | Width | Computational Cost |
|--------|----------|-------|-------------------|
| Split Conformal | ✓ Exact | Moderate | Low |
| Full Conformal | ✓ Exact | Tight | High |
| Bayesian | ~ Approx | Tight | Medium |
| Bootstrap | ~ Approx | Variable | Medium |

---

## 5. Paper Structure

### Title
"Conformal Prediction for Factor Crowding: Distribution-Free Uncertainty Quantification in Financial Markets"

### Abstract (Draft)
```
We apply conformal prediction to factor crowding detection in financial
markets, providing distribution-free uncertainty quantification with
guaranteed coverage. Unlike traditional statistical inference that
requires distributional assumptions, conformal methods achieve valid
coverage under only exchangeability. We develop conformal wrappers for
crowding regime classification and tail risk prediction, demonstrating
that (1) conformal prediction achieves nominal coverage where traditional
methods fail, (2) prediction set size reflects underlying uncertainty
about crowding regimes, and (3) conformal intervals for tail risk adapt
appropriately to market conditions. Our work bridges the gap between
theoretical coverage guarantees and practical financial applications.
```

### Sections
1. Introduction
2. Background: Factor Crowding & Conformal Prediction
3. Methodology
   - 3.1 Conformal Crowding Classification
   - 3.2 Conformalized Tail Risk Regression
   - 3.3 Calibration Metrics
4. Experiments
   - 4.1 Coverage Verification
   - 4.2 Prediction Set Analysis
   - 4.3 Tail Risk Intervals
   - 4.4 Method Comparison
5. Related Work
6. Conclusion

---

## 6. Implementation Details

### Libraries

```python
# Conformal prediction
from mapie.classification import MapieClassifier
from mapie.regression import MapieQuantileRegressor

# Alternative: crepes
from crepes import ConformalClassifier, ConformalRegressor
```

### Data Pipeline

```python
# Reuse existing infrastructure
import sys
sys.path.append('../factor_crowding')
from src.crowding_signal import CrowdingDetector
from data.fetch_data import fetch_fama_french_factors

# Load and prepare data
factors_daily, factors_monthly = fetch_fama_french_factors()
detector = CrowdingDetector()
signals = detector.compute_multi_factor_signals(factors_monthly)
```

### Walk-Forward Validation

Critical for time series: Use expanding window, not random split

```python
def walk_forward_conformal(X, y, model, alpha=0.1, min_train=120):
    """
    Walk-forward conformal prediction with temporal order preserved.
    """
    results = []
    for t in range(min_train, len(X)):
        X_train = X[:t-12]  # Training data
        X_cal = X[t-12:t]    # Calibration (recent)
        X_test = X[t:t+1]    # Test point

        model.fit(X_train, y[:t-12])
        model.calibrate(X_cal, y[t-12:t])
        pred_set = model.predict_set(X_test)

        results.append({
            'date': X_test.index[0],
            'prediction_set': pred_set,
            'true_label': y[t],
            'covered': y[t] in pred_set
        })
    return pd.DataFrame(results)
```

---

## 7. Risk & Mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Exchangeability violated | Medium | Use adaptive conformal, test robustness |
| Sets too large (trivial) | Low | Report efficiency metrics, compare baselines |
| Limited novelty | Low | Focus on finance application, practical value |
| Coverage only marginal | Low | Analyze conditional coverage |

---

## 8. Success Criteria

1. **Coverage**: Conformal achieves 90% ± 2% nominal coverage
2. **Efficiency**: Sets smaller than naive intervals
3. **Practical value**: Prediction set size correlates with regime uncertainty
4. **Comparison**: Outperforms bootstrap on coverage, competitive with Bayesian on efficiency

---

## 9. References

### Conformal Prediction
- Vovk, Gammerman, Shafer (2005). "Algorithmic Learning in a Random World"
- Shafer, Vovk (2008). "A Tutorial on Conformal Prediction"
- Romano, Patterson, Candes (2019). "Conformalized Quantile Regression"
- Angelopoulos, Bates (2021). "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"

### Finance Applications
- Xu, Xie (2023). "Conformal Prediction for Financial Time Series"
- Gibbs, Candes (2021). "Adaptive Conformal Inference Under Distribution Shift"

### Factor Crowding (Our Prior Work)
- [Factor Crowding Paper] (2024). "Not All Factors Crowd Equally"
