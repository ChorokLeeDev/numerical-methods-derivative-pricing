"""
Momentum Control Test

Critical question: Is the "crowding effect" just momentum/mean-reversion in disguise?

The paper's crowding proxy:
    C(t) = |Return(t-12:t)| / Median(Historical Returns)

This is essentially momentum. The test:
1. Regress future returns on crowding alone
2. Regress future returns on crowding + momentum
3. See if crowding effect survives

If crowding coefficient becomes insignificant after controlling for momentum,
the "crowding" story may be spurious.

Run: python scripts/momentum_control_test.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not installed. Install with: pip install statsmodels")


def compute_crowding_proxy(returns, window=12):
    """
    Compute crowding proxy as defined in the paper.
    C(t) = |Rolling Return| / Median(Historical)
    """
    rolling_return = returns.rolling(window).sum()
    median_return = rolling_return.expanding().median()
    crowding = np.abs(rolling_return) / median_return.replace(0, np.nan)
    return crowding


def compute_momentum(returns, window=12):
    """Compute momentum (trailing return)."""
    return returns.rolling(window).sum()


def run_regressions(df, factor_name, returns):
    """Run regression analysis for one factor."""

    print(f"\n{'='*60}")
    print(f"FACTOR: {factor_name}")
    print('='*60)

    # Compute variables
    crowding = compute_crowding_proxy(returns, window=12).shift(1)  # Lag by 1
    momentum = compute_momentum(returns, window=12).shift(1)  # Lag by 1
    future_return = returns.shift(-1)  # Next month return

    # Combine into DataFrame
    data = pd.DataFrame({
        'future_return': future_return,
        'crowding': crowding,
        'momentum': momentum,
        'volatility': returns.rolling(12).std().shift(1)
    }).dropna()

    print(f"Observations: {len(data)}")

    if not HAS_STATSMODELS:
        # Fallback: simple correlation analysis
        print("\n[Using correlation analysis - install statsmodels for full regression]")
        corr_crowd = data['future_return'].corr(data['crowding'])
        corr_mom = data['future_return'].corr(data['momentum'])
        print(f"Correlation(future_return, crowding): {corr_crowd:.4f}")
        print(f"Correlation(future_return, momentum): {corr_mom:.4f}")
        return {
            'factor': factor_name,
            'corr_crowding': corr_crowd,
            'corr_momentum': corr_mom
        }

    results = {'factor': factor_name}

    # REGRESSION 1: Future Return ~ Crowding Only
    print("\n[1] Regression: Future Return ~ Crowding")
    X1 = sm.add_constant(data['crowding'])
    try:
        model1 = sm.OLS(data['future_return'], X1).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
        results['crowding_only_coef'] = model1.params['crowding']
        results['crowding_only_tstat'] = model1.tvalues['crowding']
        results['crowding_only_pval'] = model1.pvalues['crowding']
        results['r2_crowding_only'] = model1.rsquared

        print(f"  Coefficient: {model1.params['crowding']:.6f}")
        print(f"  t-statistic: {model1.tvalues['crowding']:.4f}")
        print(f"  p-value: {model1.pvalues['crowding']:.4f}")
        print(f"  R²: {model1.rsquared:.4f}")

        if model1.pvalues['crowding'] < 0.05:
            print("  ✓ Crowding is significant at 5% level")
        else:
            print("  ✗ Crowding is NOT significant at 5% level")
    except Exception as e:
        print(f"  Error: {e}")
        results['crowding_only_coef'] = np.nan

    # REGRESSION 2: Future Return ~ Momentum Only
    print("\n[2] Regression: Future Return ~ Momentum")
    X2 = sm.add_constant(data['momentum'])
    try:
        model2 = sm.OLS(data['future_return'], X2).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
        results['momentum_only_coef'] = model2.params['momentum']
        results['momentum_only_tstat'] = model2.tvalues['momentum']
        results['momentum_only_pval'] = model2.pvalues['momentum']
        results['r2_momentum_only'] = model2.rsquared

        print(f"  Coefficient: {model2.params['momentum']:.6f}")
        print(f"  t-statistic: {model2.tvalues['momentum']:.4f}")
        print(f"  p-value: {model2.pvalues['momentum']:.4f}")
        print(f"  R²: {model2.rsquared:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
        results['momentum_only_coef'] = np.nan

    # REGRESSION 3: Future Return ~ Crowding + Momentum (THE KEY TEST)
    print("\n[3] Regression: Future Return ~ Crowding + Momentum (KEY TEST)")
    X3 = sm.add_constant(data[['crowding', 'momentum']])
    try:
        model3 = sm.OLS(data['future_return'], X3).fit(cov_type='HAC', cov_kwds={'maxlags': 12})

        results['crowding_controlled_coef'] = model3.params['crowding']
        results['crowding_controlled_tstat'] = model3.tvalues['crowding']
        results['crowding_controlled_pval'] = model3.pvalues['crowding']
        results['momentum_controlled_coef'] = model3.params['momentum']
        results['momentum_controlled_pval'] = model3.pvalues['momentum']
        results['r2_both'] = model3.rsquared

        print(f"  Crowding coefficient: {model3.params['crowding']:.6f}")
        print(f"  Crowding t-statistic: {model3.tvalues['crowding']:.4f}")
        print(f"  Crowding p-value: {model3.pvalues['crowding']:.4f}")
        print(f"  Momentum coefficient: {model3.params['momentum']:.6f}")
        print(f"  Momentum p-value: {model3.pvalues['momentum']:.4f}")
        print(f"  R²: {model3.rsquared:.4f}")

        print("\n  VERDICT:")
        if model3.pvalues['crowding'] < 0.05:
            print("  ✓ Crowding SURVIVES after controlling for momentum")
            print("  → Crowding has independent predictive power (GOOD for paper)")
        else:
            print("  ✗ Crowding DOES NOT survive after controlling for momentum")
            print("  → Crowding effect may be spurious (PROBLEM for paper)")

    except Exception as e:
        print(f"  Error: {e}")
        results['crowding_controlled_coef'] = np.nan

    # REGRESSION 4: Full model with volatility
    print("\n[4] Full Model: Future Return ~ Crowding + Momentum + Volatility")
    X4 = sm.add_constant(data[['crowding', 'momentum', 'volatility']])
    try:
        model4 = sm.OLS(data['future_return'], X4).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
        results['crowding_full_pval'] = model4.pvalues['crowding']
        results['r2_full'] = model4.rsquared

        print(f"  Crowding p-value: {model4.pvalues['crowding']:.4f}")
        print(f"  R²: {model4.rsquared:.4f}")
    except Exception as e:
        print(f"  Error: {e}")

    return results


def main():
    """Run momentum control test for all factors."""

    print("\n" + "="*70)
    print("MOMENTUM CONTROL TEST")
    print("Is 'crowding' just momentum in disguise?")
    print("="*70)

    # Load data
    base_path = Path(__file__).parent.parent
    data_path = base_path / "data" / "factor_crowding" / "ff_factors_monthly.parquet"

    if not data_path.exists():
        print(f"\nERROR: Data not found at {data_path}")
        print("Run download_ff_data.py first.")
        return

    df = pd.read_parquet(data_path)
    print(f"\nLoaded: {len(df)} months")

    # Factors to test
    factors = ['SMB', 'HML', 'RMW', 'CMA', 'Mom']
    available_factors = [f for f in factors if f in df.columns]

    all_results = []

    for factor in available_factors:
        result = run_regressions(df, factor, df[factor])
        all_results.append(result)

    # Summary
    results_df = pd.DataFrame(all_results)

    print("\n" + "="*70)
    print("SUMMARY: Does Crowding Survive Momentum Control?")
    print("="*70)

    if 'crowding_controlled_pval' in results_df.columns:
        print("\n{:<10} {:>15} {:>15} {:>15}".format(
            "Factor", "Crowding-Only p", "Controlled p", "Survives?"
        ))
        print("-"*60)

        for _, row in results_df.iterrows():
            crowding_only_p = row.get('crowding_only_pval', np.nan)
            controlled_p = row.get('crowding_controlled_pval', np.nan)
            survives = "YES ✓" if controlled_p < 0.05 else "NO ✗"

            print("{:<10} {:>15.4f} {:>15.4f} {:>15}".format(
                row['factor'],
                crowding_only_p if not np.isnan(crowding_only_p) else 0,
                controlled_p if not np.isnan(controlled_p) else 0,
                survives
            ))

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
    If crowding effect SURVIVES (p < 0.05 after momentum control):
    → Crowding has independent predictive power
    → The paper's story is supported
    → Can claim crowding is distinct from momentum

    If crowding effect DOES NOT survive (p > 0.05 after momentum control):
    → Crowding effect may be spurious
    → Need to revise claims or use different proxy
    → Consider: Is this just mean-reversion?

    RECOMMENDED ACTIONS:
    1. If survives: Add this analysis to Section 5 as robustness check
    2. If doesn't survive: Acknowledge limitation, test alternative proxies
    """)

    # Save results
    output_path = base_path / "results" / "momentum_control_results.csv"
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
