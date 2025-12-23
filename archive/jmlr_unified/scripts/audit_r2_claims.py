"""
Audit R² Claims in the Paper

This script investigates what the R² numbers actually measure:
- Model Fit R²: How well hyperbolic curve fits historical data
- Extrapolation R²: How well model predicts out-of-sample period
- Predictive R²: How well we predict next-period returns

Key finding: The paper likely reports Model Fit R², not Predictive R².
Model Fit R² of 60-70% is reasonable. Predictive R² of 60-70% would be implausible.

Run: python scripts/audit_r2_claims.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def hyperbolic_decay(t, K, lam):
    """Hyperbolic decay model: α(t) = K / (1 + λt)"""
    return K / (1 + lam * t)


def exponential_decay(t, K, lam):
    """Exponential decay model: α(t) = K * exp(-λt)"""
    return K * np.exp(-lam * t)


def compute_r2(y_actual, y_predicted):
    """Compute R² (coefficient of determination)."""
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    if ss_tot == 0:
        return 0
    return 1 - ss_res / ss_tot


def fit_decay_models(sharpe_series, model='hyperbolic'):
    """Fit decay model to Sharpe ratio series."""
    t = np.arange(len(sharpe_series))
    y = sharpe_series.values

    # Only fit on valid data
    mask = np.isfinite(y) & (y > -5) & (y < 5)  # Reasonable Sharpe range
    if mask.sum() < 20:
        return None

    t_valid, y_valid = t[mask], y[mask]

    try:
        if model == 'hyperbolic':
            popt, _ = curve_fit(
                hyperbolic_decay,
                t_valid, y_valid,
                p0=[1.0, 0.01],
                bounds=([0, 0], [10, 1.0]),
                maxfev=5000
            )
            y_fitted = hyperbolic_decay(t, *popt)
        else:
            popt, _ = curve_fit(
                exponential_decay,
                t_valid, y_valid,
                p0=[1.0, 0.01],
                bounds=([0, 0], [10, 1.0]),
                maxfev=5000
            )
            y_fitted = exponential_decay(t, *popt)

        return {
            'params': popt,
            'fitted': y_fitted,
            'r2': compute_r2(y_valid, y_fitted[mask])
        }
    except Exception as e:
        print(f"  Fitting error: {e}")
        return None


def audit_factor(returns, factor_name, train_end_pct=0.7):
    """
    Comprehensive R² audit for one factor.

    Returns multiple R² metrics:
    1. model_fit_r2: Full sample curve fitting
    2. in_sample_r2: Training period curve fitting
    3. extrapolation_r2: Test period using train parameters
    4. predictive_r2: Next-month return prediction (the hard one)
    """

    print(f"\n{'='*60}")
    print(f"AUDITING: {factor_name}")
    print('='*60)

    # Compute rolling Sharpe ratio (36-month window, annualized)
    window = 36
    rolling_mean = pd.Series(returns).rolling(window).mean()
    rolling_std = pd.Series(returns).rolling(window).std()
    sharpe = (rolling_mean / rolling_std * np.sqrt(12)).dropna()

    print(f"Rolling Sharpe computed: {len(sharpe)} periods")

    # Train/test split
    train_end_idx = int(len(sharpe) * train_end_pct)
    train_sharpe = sharpe.iloc[:train_end_idx]
    test_sharpe = sharpe.iloc[train_end_idx:]

    print(f"Train periods: {len(train_sharpe)}")
    print(f"Test periods: {len(test_sharpe)}")

    results = {'factor': factor_name}

    # 1. FULL SAMPLE MODEL FIT R²
    print("\n[1] Full Sample Model Fit...")
    full_fit = fit_decay_models(sharpe, model='hyperbolic')
    if full_fit:
        results['full_sample_r2'] = full_fit['r2']
        results['K_full'] = full_fit['params'][0]
        results['lambda_full'] = full_fit['params'][1]
        print(f"  Full sample R²: {full_fit['r2']:.4f}")
        print(f"  K={full_fit['params'][0]:.4f}, λ={full_fit['params'][1]:.4f}")
    else:
        results['full_sample_r2'] = np.nan

    # 2. IN-SAMPLE MODEL FIT R² (train only)
    print("\n[2] In-Sample Model Fit (Train Period Only)...")
    train_fit = fit_decay_models(train_sharpe, model='hyperbolic')
    if train_fit:
        results['in_sample_r2'] = train_fit['r2']
        results['K_train'] = train_fit['params'][0]
        results['lambda_train'] = train_fit['params'][1]
        print(f"  In-sample R²: {train_fit['r2']:.4f}")
    else:
        results['in_sample_r2'] = np.nan

    # 3. EXTRAPOLATION R² (test period using train params)
    print("\n[3] Extrapolation R² (Test Period, Train Parameters)...")
    if train_fit and len(test_sharpe) > 10:
        K, lam = train_fit['params']
        t_test = np.arange(len(train_sharpe), len(train_sharpe) + len(test_sharpe))
        y_pred_test = hyperbolic_decay(t_test, K, lam)
        y_actual_test = test_sharpe.values

        mask = np.isfinite(y_actual_test)
        if mask.sum() > 10:
            extrap_r2 = compute_r2(y_actual_test[mask], y_pred_test[mask])
            results['extrapolation_r2'] = extrap_r2
            print(f"  Extrapolation R²: {extrap_r2:.4f}")

            # Compare to naive benchmark (predict historical mean)
            naive_pred = np.full_like(y_actual_test, train_sharpe.mean())
            naive_r2 = compute_r2(y_actual_test[mask], naive_pred[mask])
            results['naive_r2'] = naive_r2
            print(f"  Naive (mean) R²: {naive_r2:.4f}")
            print(f"  Improvement over naive: {extrap_r2 - naive_r2:.4f}")
        else:
            results['extrapolation_r2'] = np.nan
    else:
        results['extrapolation_r2'] = np.nan

    # 4. TRUE PREDICTIVE R² (next-month return prediction)
    print("\n[4] True Predictive R² (Next-Month Return)...")
    # This is the hard one - can we predict next month's return?
    returns_series = pd.Series(returns)

    # Features: lagged returns, rolling stats
    features = pd.DataFrame({
        'lag1': returns_series.shift(1),
        'lag12': returns_series.shift(12),
        'rolling_mean_12': returns_series.rolling(12).mean().shift(1),
        'rolling_std_12': returns_series.rolling(12).std().shift(1),
    }).dropna()

    target = returns_series.loc[features.index]

    # Split
    split_idx = int(len(features) * train_end_pct)
    X_train = features.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    y_train = target.iloc[:split_idx]
    y_test = target.iloc[split_idx:]

    if len(X_test) > 10:
        # Simple prediction: rolling mean
        y_pred_test = X_test['rolling_mean_12'].values
        pred_r2 = compute_r2(y_test.values, y_pred_test)
        results['predictive_r2'] = pred_r2
        print(f"  Predictive R² (rolling mean): {pred_r2:.4f}")

        # This should be MUCH lower than model fit R²
        if pred_r2 > 0.3:
            print(f"  ⚠️ WARNING: Predictive R² > 30% is suspicious!")
    else:
        results['predictive_r2'] = np.nan

    return results


def main():
    """Run full R² audit."""

    print("\n" + "="*70)
    print("R² AUDIT: What Do The Numbers Actually Mean?")
    print("="*70)

    # Load data
    base_path = Path(__file__).parent.parent
    data_path = base_path / "data" / "factor_crowding" / "ff_factors_monthly.parquet"

    if not data_path.exists():
        print(f"\nERROR: Data not found at {data_path}")
        print("Run download_ff_data.py first.")
        return

    df = pd.read_parquet(data_path)
    print(f"\nLoaded data: {len(df)} months, columns: {df.columns.tolist()}")

    # Factors to audit
    factors = ['SMB', 'HML', 'RMW', 'CMA', 'Mom']
    available_factors = [f for f in factors if f in df.columns]

    print(f"\nAuditing factors: {available_factors}")

    results = []
    for factor in available_factors:
        result = audit_factor(df[factor].values, factor, train_end_pct=0.7)
        results.append(result)

    # Summary table
    results_df = pd.DataFrame(results)

    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)

    summary_cols = ['factor', 'full_sample_r2', 'in_sample_r2', 'extrapolation_r2', 'predictive_r2']
    available_cols = [c for c in summary_cols if c in results_df.columns]
    print(results_df[available_cols].to_string(index=False))

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
    full_sample_r2:    How well hyperbolic curve fits ALL data (can be 60-70%)
    in_sample_r2:      How well curve fits TRAINING data (can be 60-70%)
    extrapolation_r2:  How well TRAINED model predicts TEST period (should be lower)
    predictive_r2:     How well we predict NEXT-MONTH RETURNS (typically < 10%)

    KEY INSIGHT:
    - If the paper reports 45-63% R², it's likely MODEL FIT, not PREDICTION
    - Model fit R² of 60% is reasonable
    - Predictive R² of 60% would be Nobel Prize-worthy (suspicious)

    RECOMMENDATION:
    - Clarify in paper what R² measures
    - Use term "Model Fit R²" not "OOS R²" if that's what it is
    - Report predictive R² separately (it will be much lower)
    """)

    # Save results
    output_path = base_path / "results" / "r2_audit_results.csv"
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
