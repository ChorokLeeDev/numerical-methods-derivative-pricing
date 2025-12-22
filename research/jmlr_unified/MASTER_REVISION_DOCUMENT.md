# Master Revision Document: JMLR → JoFE Transformation

**Created:** December 22, 2024
**Target Venue:** Journal of Financial Econometrics - Special Issue on Machine Learning
**Submission Deadline:** March 1, 2026
**Author:** Chorok Lee (KAIST)

---

## Part 1: Honest Assessment Summary

### What the Paper Claims vs Reality

| Claim | Reality | Severity |
|-------|---------|----------|
| "Derive hyperbolic decay from game theory" | Assumes K(t)=K₀/(1+γt), then "derives" hyperbolic form. Circular. | 🔴 CRITICAL |
| "First mechanistic explanation" | It's a parameterization, not a derivation from first principles | 🔴 CRITICAL |
| "OOS R² of 45-63%" | Likely model fit R², not predictive R². If predictive, implausible. | 🔴 CRITICAL |
| "Crowding proxy measures crowding" | Proxy = |trailing returns|. This is momentum, not crowding. | 🟠 HIGH |
| "C ⊥ y \| x for CW-ACI coverage" | If true, crowding has no predictive power beyond features. Contradicts paper premise. | 🟠 HIGH |
| "Judgment factors decay faster" | Classification is arbitrary. Why is momentum "judgment"? | 🟡 MEDIUM |
| "MMD enables global transfer" | Standard application of Long et al. (2015). Not novel. | 🟡 MEDIUM |
| "54% Sharpe improvement" | Backtest result. Likely overfitted. No transaction cost sensitivity. | 🟡 MEDIUM |

### What's Actually Good

1. **Core insight is valid:** Factors do decay at different rates. This is empirically documented.
2. **Unified framework is creative:** Connecting game theory → transfer → risk management is novel narrative.
3. **CW-ACI is interesting:** Using domain signals in conformal prediction is a real contribution.
4. **Data coverage is extensive:** 61 years of data, 7 international markets.
5. **Writing is clear:** Paper is well-organized and readable.

### What Needs Fundamental Rework

1. **Theory section:** Must be reframed or rebuilt from scratch
2. **Empirical methodology:** Need real data, proper R² computation, momentum controls
3. **CW-ACI assumption:** Need to either verify rigorously or weaken claims
4. **Factor classification:** Need objective criteria or remove claims

---

## Part 2: The Circular Reasoning Problem (Deep Dive)

### The Current "Derivation"

**Step 1:** Assume intrinsic alpha decays hyperbolically
```
K(t) = K₀ / (1 + γt)
```

**Step 2:** Model observed alpha as intrinsic minus crowding cost
```
α(t) = K(t) - λ₀C(t)
```

**Step 3:** Solve for equilibrium crowding
```
C*(t) = [K(t) - r_f] / λ₀
```

**Step 4:** Substitute back to get observed alpha
```
α(t) = K₀ / (1 + λ_eff·t)  where λ_eff = γ + crowding_effect
```

**The Problem:** Step 1 ASSUMES the hyperbolic form. The rest is algebraic manipulation, not derivation.

### What Real Game Theory Would Look Like

**Option A: Signaling Game**
- Investors have private signals about factor quality
- Entry/exit decisions reveal information
- Equilibrium generates endogenous decay
- Much harder to solve, but would be genuine derivation

**Option B: Dynamic Game with Learning**
- Investors learn about factor alpha over time
- As information diffuses, alpha decays
- Decay rate depends on information structure
- Can derive functional form from learning dynamics

**Option C: Mean Field Game**
- Continuum of investors making optimal allocation
- Each investor is infinitesimal, aggregate behavior matters
- Can derive decay from equilibrium conditions
- Requires solving HJB equation

### Realistic Fix for Your Timeline

You don't have time to develop real game theory. Instead:

**Reframe as "Equilibrium Model" not "Game-Theoretic Model":**

> "We propose an equilibrium model where competitive capital allocation generates hyperbolic factor decay. While we do not derive the functional form from strategic interaction, we show that hyperbolic decay: (1) has intuitive economic interpretation, (2) fits the data better than alternatives, and (3) enables useful predictions."

This is honest and still valuable.

---

## Part 3: The R² Problem (Deep Dive)

### What R² Probably Means in Your Code

Looking at `src/game_theory/crowding_signal.py`:

```python
def fit_decay_model(sharpe: pd.Series):
    """Fit hyperbolic decay model to Sharpe ratio series."""
    popt, _ = curve_fit(alpha_decay_model, t_pos, y_pos, ...)
```

This computes **model fit R²**: how well the hyperbolic curve fits historical Sharpe ratios.

Model fit R² of 60-70% is reasonable—you're fitting a smooth curve to noisy data.

### What the Paper Seems to Claim

Table 4 shows "OOS R²" of 45-63%. If this means:

**Interpretation A (Probably What You Mean):**
- Fit model on 1963-2000
- Evaluate fit quality on 2000-2024
- R² measures how well the fitted curve extrapolates

This is defensible but needs clarification.

**Interpretation B (What Readers Will Think):**
- Train model to predict next month's factor return
- Test on held-out data
- R² measures predictive accuracy for return forecasting

This would be extraordinary and implausible.

### How to Fix

1. **Clarify terminology:**
   - "Model Fit R²" for curve fitting quality
   - "Extrapolation R²" for out-of-period fit
   - "Predictive R²" for return forecasting (probably don't claim this)

2. **Add naive benchmarks:**
   - Historical mean benchmark
   - Random walk benchmark
   - Show your model beats these

3. **Report honest numbers:**
   - If predictive R² is actually 5-15%, report that
   - Lower but honest numbers are better than high but suspicious numbers

---

## Part 4: The Crowding Proxy Problem (Deep Dive)

### Current Proxy

```python
C_i(t) = |Return_{i,t-12:t}| / Median(Historical Returns)
```

### Why This is Problematic

This measures **momentum**, not crowding:
- High trailing returns → High C_i(t)
- But high trailing returns → Mean reversion → Lower future returns

You might be rediscovering mean reversion, not crowding effects.

### How to Distinguish Crowding from Momentum

**Test 1: Control for Momentum**
```
α_future = β₀ + β₁·Crowding + β₂·Momentum + ε
```
If β₁ remains significant after controlling for momentum, crowding has independent effect.

**Test 2: Use Non-Return-Based Proxies**
- ETF flows (for post-2000 period)
- 13F institutional holdings
- Short interest
- Google search volume for factor names

**Test 3: Placebo Test**
- Randomly permute crowding labels
- If results still hold, the effect is spurious

### Realistic Fix

1. Add momentum as explicit control variable
2. Show crowding effect survives
3. Acknowledge limitation prominently
4. Discuss ideal data (13F filings) as future work

---

## Part 5: The CW-ACI Contradiction (Deep Dive)

### The Assumption

Theorem 6 requires:
```
C ⊥ y | x   (crowding independent of outcomes given features)
```

### The Contradiction

**Paper's Main Argument:**
> "Crowding predicts factor returns! High crowding → crashes!"

**CW-ACI Requirement:**
> "Crowding has no predictive power beyond features x"

These cannot both be true.

### Resolution Options

**Option A: Crowding is in x**
- If crowding is already a feature, then C ⊥ y | x is trivially satisfied
- CW-ACI adds value by explicitly weighting by crowding
- But then "crowding predicts beyond features" claim is false

**Option B: Weaken the Guarantee**
- Change from "distribution-free coverage guarantee" to "empirical coverage"
- Report: "CW-ACI achieves 89.8% empirical coverage"
- Drop the theoretical guarantee claim

**Option C: Different Role for Crowding**
- Argue: Crowding affects prediction uncertainty, not point prediction
- Given x, crowding tells us about variance, not mean
- This is subtle but defensible

### Recommended Fix

Use Option B + Option C:

> "We conjecture that crowding primarily affects prediction uncertainty rather than point forecasts. Under this interpretation, CW-ACI uses crowding to modulate interval width while preserving approximate coverage. Empirical validation confirms 89.8% coverage, close to the nominal 90% target."

---

## Part 6: The Factor Classification Problem

### Current Classification

| Mechanical | Judgment |
|------------|----------|
| SMB (size) | HML (value) |
| RMW (profitability) | MOM (momentum) |
| CMA (investment) | ST_Rev (short-term reversal) |
| | LT_Rev (long-term reversal) |

### Why This is Arbitrary

**Momentum is a mechanical sort:**
```
MOM = Return(t-12, t-1)
```
There's no "judgment" involved—it's purely trailing returns.

**Value requires judgment?**
```
HML = Book-to-Market ratio
```
This is an accounting ratio, equally mechanical.

### Better Classification Schemes

**Option 1: By Publication Date**
| Early (pre-1995) | Late (post-2010) |
|------------------|------------------|
| SMB, HML, MOM | RMW, CMA |

Rationale: Earlier factors have been arbitraged longer.

**Option 2: By Turnover**
| Low Turnover | High Turnover |
|--------------|---------------|
| HML, SMB, RMW, CMA | MOM, ST_Rev, LT_Rev |

Rationale: High turnover = more trading = more capacity constraints.

**Option 3: By Capacity**
| High Capacity | Low Capacity |
|---------------|--------------|
| SMB, HML | MOM, ST_Rev |

Rationale: Small-cap and momentum have limited capacity.

### Recommended Fix

Either:
1. Use objective classification with clear criteria
2. Or drop classification and just report factor-by-factor results

---

## Part 7: 14-Month Revision Timeline

### Phase 1: Foundation (Dec 2024 - Feb 2025)

**Month 1 (December 2024 - Tonight + Next 2 Weeks):**
- [ ] Download real Fama-French data
- [ ] Verify all experiments run with real data
- [ ] Document what R² actually measures
- [ ] Create momentum control regression
- [ ] Identify all "derive" language in paper

**Month 2 (January 2025):**
- [ ] Rewrite Section 4 with honest framing
- [ ] Change "Theorems" to "Propositions"
- [ ] Add explicit model limitations paragraph
- [ ] Revise abstract and introduction

**Month 3 (February 2025):**
- [ ] Run full robustness battery
- [ ] Test alternative crowding proxies
- [ ] Test factor classification sensitivity
- [ ] Document all results

### Phase 2: Restructuring (Mar - Jun 2025)

**Month 4-5 (March-April 2025):**
- [ ] Decide: unified framework or single contribution?
- [ ] If unified: strengthen each component
- [ ] If single: develop CW-ACI deeply, move others to appendix

**Month 6-7 (May-June 2025):**
- [ ] Strengthen theoretical treatment
- [ ] Address CW-ACI assumption formally
- [ ] Add proper econometric tests (HAC SEs, etc.)
- [ ] Get feedback from 2-3 colleagues

### Phase 3: Polishing (Jul - Oct 2025)

**Month 8-9 (July-August 2025):**
- [ ] Complete rewrite of weak sections
- [ ] Ensure all claims are defensible
- [ ] Prepare replication package

**Month 10-11 (September-October 2025):**
- [ ] Internal review and iteration
- [ ] Send to co-authors/advisors for feedback
- [ ] Iterate based on feedback

### Phase 4: Submission (Nov 2025 - Feb 2026)

**Month 12-13 (November-December 2025):**
- [ ] Final polishing
- [ ] Format per JoFE guidelines (OUP LaTeX template)
- [ ] Prepare cover letter highlighting fit with special issue

**Month 14 (January-February 2026):**
- [ ] Final read-through
- [ ] Submit by February 15, 2026 (2 weeks before deadline)

---

## Part 8: Tonight's Starting Point

### Hour 1: Set Up Real Data Pipeline

```bash
cd /Users/i767700/Github/quant/research/jmlr_unified

# Create data download script
cat > scripts/download_ff_data.py << 'EOF'
"""Download Fama-French factor data from Kenneth French Data Library."""

import pandas as pd
import pandas_datareader.data as web
from pathlib import Path

def download_ff_factors():
    """Download and save FF factors."""
    data_dir = Path(__file__).parent.parent / "data" / "factor_crowding"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download FF 5 factors
    ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='1963-07-01')
    ff5_monthly = ff5[0]  # Monthly data

    # Download momentum
    mom = web.DataReader('F-F_Momentum_Factor', 'famafrench', start='1963-07-01')
    mom_monthly = mom[0]

    # Download short-term reversal
    st_rev = web.DataReader('F-F_ST_Reversal_Factor', 'famafrench', start='1963-07-01')
    st_rev_monthly = st_rev[0]

    # Download long-term reversal
    lt_rev = web.DataReader('F-F_LT_Reversal_Factor', 'famafrench', start='1963-07-01')
    lt_rev_monthly = lt_rev[0]

    # Combine all factors
    combined = ff5_monthly.join([mom_monthly, st_rev_monthly, lt_rev_monthly], how='outer')
    combined.columns = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'Mom', 'ST_Rev', 'LT_Rev']

    # Save
    output_path = data_dir / 'ff_factors_monthly.parquet'
    combined.to_parquet(output_path)
    print(f"Saved {len(combined)} months of data to {output_path}")
    print(f"Date range: {combined.index[0]} to {combined.index[-1]}")
    print(f"Columns: {combined.columns.tolist()}")

    return combined

if __name__ == '__main__':
    download_ff_factors()
EOF

# Run it
python scripts/download_ff_data.py
```

### Hour 2: Audit R² Claims

Create a diagnostic script:

```python
# scripts/audit_r2_claims.py
"""Audit what R² actually measures in this paper."""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path

def hyperbolic_decay(t, K, lam):
    """α(t) = K / (1 + λt)"""
    return K / (1 + lam * t)

def compute_model_fit_r2(y_actual, y_fitted):
    """R² for model fit (how well curve fits data)."""
    ss_res = np.sum((y_actual - y_fitted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - ss_res / ss_tot

def compute_predictive_r2(y_actual, y_predicted):
    """R² for prediction (how well we predict future values)."""
    # Same formula, but y_predicted comes from model trained on PAST data only
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - ss_res / ss_tot

def audit_factor(returns, factor_name, train_end_idx):
    """Audit R² for one factor."""

    # Compute rolling Sharpe (36-month window)
    window = 36
    rolling_mean = pd.Series(returns).rolling(window).mean()
    rolling_std = pd.Series(returns).rolling(window).std()
    sharpe = (rolling_mean / rolling_std * np.sqrt(12)).dropna()

    # Split train/test
    train_sharpe = sharpe.iloc[:train_end_idx]
    test_sharpe = sharpe.iloc[train_end_idx:]

    # Fit hyperbolic model on training data
    t_train = np.arange(len(train_sharpe))
    y_train = train_sharpe.values

    # Only fit on positive values
    mask = y_train > 0
    if mask.sum() < 20:
        return None

    try:
        popt, _ = curve_fit(
            hyperbolic_decay,
            t_train[mask], y_train[mask],
            p0=[1.5, 0.01],
            bounds=([0, 0], [10, 0.5]),
            maxfev=5000
        )
        K, lam = popt
    except:
        return None

    # Model fit R² (in-sample)
    y_fitted_train = hyperbolic_decay(t_train, K, lam)
    r2_fit = compute_model_fit_r2(y_train, y_fitted_train)

    # Extrapolation R² (out-of-sample)
    t_test = np.arange(len(train_sharpe), len(train_sharpe) + len(test_sharpe))
    y_test = test_sharpe.values
    y_predicted_test = hyperbolic_decay(t_test, K, lam)
    r2_extrap = compute_predictive_r2(y_test, y_predicted_test)

    # Naive benchmark: predict historical mean
    naive_pred = np.full_like(y_test, y_train.mean())
    r2_vs_naive = compute_predictive_r2(y_test, y_predicted_test) - compute_predictive_r2(y_test, naive_pred)

    return {
        'factor': factor_name,
        'K': K,
        'lambda': lam,
        'r2_model_fit': r2_fit,
        'r2_extrapolation': r2_extrap,
        'r2_vs_naive': r2_vs_naive,
        'train_periods': len(train_sharpe),
        'test_periods': len(test_sharpe)
    }

def main():
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "factor_crowding" / "ff_factors_monthly.parquet"
    if not data_path.exists():
        print(f"Data not found at {data_path}. Run download_ff_data.py first.")
        return

    df = pd.read_parquet(data_path)
    factors = ['SMB', 'HML', 'RMW', 'CMA', 'Mom', 'ST_Rev', 'LT_Rev']

    # Train on 1963-2000, test on 2000-2024
    train_end = df.index.get_loc(df.index[df.index.year <= 2000][-1])

    results = []
    for factor in factors:
        if factor in df.columns:
            result = audit_factor(df[factor].values, factor, train_end)
            if result:
                results.append(result)

    results_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("R² AUDIT RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("- r2_model_fit: How well hyperbolic curve fits training data (can be high)")
    print("- r2_extrapolation: How well model predicts test period (should be lower)")
    print("- r2_vs_naive: Improvement over naive mean prediction (key metric)")
    print("="*80)

    # Save
    output_path = Path(__file__).parent.parent / "results" / "r2_audit.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

if __name__ == '__main__':
    main()
```

### Hour 3: Create Momentum Control Test

```python
# scripts/momentum_control_test.py
"""Test if crowding effect survives controlling for momentum."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

def main():
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "factor_crowding" / "ff_factors_monthly.parquet"
    df = pd.read_parquet(data_path)

    factors = ['SMB', 'HML', 'RMW', 'CMA', 'Mom']

    results = []

    for factor in factors:
        if factor not in df.columns:
            continue

        returns = df[factor].dropna()

        # Compute crowding proxy (trailing 12-month absolute return)
        crowding = returns.rolling(12).apply(lambda x: np.abs(x).mean()).shift(1)

        # Compute momentum (trailing 12-month return)
        momentum = returns.rolling(12).mean().shift(1)

        # Future return (next month)
        future_return = returns.shift(-1)

        # Align all series
        data = pd.DataFrame({
            'future_return': future_return,
            'crowding': crowding,
            'momentum': momentum
        }).dropna()

        # Regression 1: Future return ~ Crowding only
        X1 = sm.add_constant(data['crowding'])
        model1 = sm.OLS(data['future_return'], X1).fit(cov_type='HAC', cov_kwds={'maxlags': 12})

        # Regression 2: Future return ~ Crowding + Momentum
        X2 = sm.add_constant(data[['crowding', 'momentum']])
        model2 = sm.OLS(data['future_return'], X2).fit(cov_type='HAC', cov_kwds={'maxlags': 12})

        results.append({
            'factor': factor,
            'crowding_only_coef': model1.params['crowding'],
            'crowding_only_pval': model1.pvalues['crowding'],
            'crowding_controlled_coef': model2.params['crowding'],
            'crowding_controlled_pval': model2.pvalues['crowding'],
            'momentum_coef': model2.params['momentum'],
            'momentum_pval': model2.pvalues['momentum'],
            'r2_without_momentum': model1.rsquared,
            'r2_with_momentum': model2.rsquared
        })

    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("MOMENTUM CONTROL TEST")
    print("="*80)
    print("\nQuestion: Does crowding effect survive after controlling for momentum?")
    print(results_df.to_string(index=False))

    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("- If crowding_controlled_pval < 0.05: Crowding has independent effect (GOOD)")
    print("- If crowding_controlled_pval > 0.05: Crowding effect may be spurious (BAD)")
    print("="*80)

    # Save
    output_path = Path(__file__).parent.parent / "results" / "momentum_control_test.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

if __name__ == '__main__':
    main()
```

### Hour 4: Document All "Derive" Language

```bash
# Find all instances of problematic language
cd /Users/i767700/Github/quant/research/jmlr_unified/jmlr_submission

echo "=== Instances of 'derive' ==="
grep -rn "derive" sections/ --include="*.tex"

echo ""
echo "=== Instances of 'first to' ==="
grep -rn -i "first to" sections/ --include="*.tex"

echo ""
echo "=== Instances of 'novel' ==="
grep -rn -i "novel" sections/ --include="*.tex"

echo ""
echo "=== Instances of 'prove' ==="
grep -rn -i "we prove" sections/ --include="*.tex"
```

---

## Part 9: Key Files Reference

### Created Documents
| File | Purpose |
|------|---------|
| `REVISION_PLAN.md` | Detailed problem analysis + overnight checklist |
| `VENUE_ANALYSIS.md` | QF vs JoFE comparison |
| `MASTER_REVISION_DOCUMENT.md` | This file - comprehensive guide |

### Key Source Files
| File | What to Check |
|------|---------------|
| `jmlr_submission/sections/04_game_theory.tex` | Circular reasoning |
| `jmlr_submission/sections/05_us_empirical.tex` | R² claims |
| `jmlr_submission/sections/07_tail_risk_aci.tex` | CW-ACI assumption |
| `src/game_theory/crowding_signal.py` | R² computation |
| `experiments/jmlr/02_heterogeneity_test.py` | Factor classification |

### Scripts to Create Tonight
| Script | Purpose |
|--------|---------|
| `scripts/download_ff_data.py` | Get real FF data |
| `scripts/audit_r2_claims.py` | Verify R² computation |
| `scripts/momentum_control_test.py` | Test crowding vs momentum |

---

## Part 10: Success Metrics

### By End of Tonight
- [ ] Real FF data downloaded and verified
- [ ] R² audit complete - understand what numbers mean
- [ ] Momentum control test run - know if crowding effect survives
- [ ] All "derive" language documented

### By End of January 2025
- [ ] Section 4 rewritten with honest framing
- [ ] Abstract and intro revised
- [ ] All empirical results verified with real data

### By March 2025
- [ ] Full robustness battery complete
- [ ] Paper internally consistent
- [ ] Ready for colleague feedback

### By Submission (Feb 2026)
- [ ] All claims defensible
- [ ] Replication package complete
- [ ] Formatted per JoFE requirements
- [ ] Cover letter written

---

## Appendix: The Honest Abstract (Draft)

**Current:**
> "We derive a mechanistic explanation of factor alpha decay from game-theoretic equilibrium..."

**Revised:**
> "Factor investing generates systematic excess returns, but these returns decay over time as capital flows in. We develop an equilibrium framework that parameterizes this decay as hyperbolic: α(t) = K/(1+λt). While we do not derive this functional form from first principles, we show it fits 61 years of Fama-French data well and enables useful predictions. We find that decay rates vary across factors, with momentum and reversal factors decaying faster than size and profitability factors. We apply MMD-based domain adaptation to transfer these insights to international markets, achieving 60% transfer efficiency across seven developed markets. Finally, we introduce Crowding-Weighted Adaptive Conformal Inference (CW-ACI), which produces prediction intervals that widen during high-crowding periods. In portfolio hedging applications, CW-ACI improves Sharpe ratio by 54% with empirical coverage of 89.8%."

---

**End of Master Document**

*Start with Hour 1 tonight. The rest will follow.*
