"""
Comprehensive Baseline Comparison for JoFE Paper Revision
==========================================================
Adds all missing baselines:
1. EWMA volatility scaling
2. Conformalized Quantile Regression (CQR)
3. Historical simulation (industry standard)

Author: Chorok Lee (KAIST)
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Data Loading
# ============================================================================

def load_ff_data():
    """Load Fama-French factor data"""
    data_path = Path(__file__).parent.parent / 'data' / 'ff_factors.csv'
    ff = pd.read_csv(data_path, index_col=0, parse_dates=True)
    return ff


# ============================================================================
# Baseline Methods
# ============================================================================

class StandardCP:
    """Standard split conformal prediction (fixed intervals)"""

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit_predict(self, returns, cal_fraction=0.5):
        n = len(returns)
        cal_end = int(n * cal_fraction)

        y_cal = returns.iloc[:cal_end].values
        y_test = returns.iloc[cal_end:].values

        pred = np.mean(y_cal)
        scores = np.abs(y_cal - pred)

        n_cal = len(scores)
        q_level = min(np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal, 1.0)
        q = np.quantile(scores, q_level)

        lower = pred - q
        upper = pred + q

        coverages = (y_test >= lower) & (y_test <= upper)
        widths = np.full(len(y_test), 2 * q)

        return {'coverages': coverages, 'widths': widths}


class VolatilityScaledCP:
    """Volatility-Scaled CP using realized volatility"""

    def __init__(self, alpha=0.1, vol_window=12):
        self.alpha = alpha
        self.vol_window = vol_window

    def fit_predict(self, returns, cal_fraction=0.5):
        n = len(returns)
        cal_end = int(n * cal_fraction)

        vol_signal = returns.rolling(self.vol_window).std()
        vol_signal = vol_signal / vol_signal.expanding().median()

        y_cal = returns.iloc[:cal_end].values
        y_test = returns.iloc[cal_end:].values
        vol_cal = vol_signal.iloc[:cal_end].values
        vol_test = vol_signal.iloc[cal_end:].values

        vol_cal = np.nan_to_num(vol_cal, nan=1.0)
        vol_test = np.nan_to_num(vol_test, nan=1.0)

        pred = np.mean(y_cal)
        scores = np.abs(y_cal - pred) / np.maximum(vol_cal, 0.1)

        n_cal = len(scores)
        q_level = min(np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal, 1.0)
        q = np.quantile(scores, q_level)

        lower = pred - q * vol_test
        upper = pred + q * vol_test

        coverages = (y_test >= lower) & (y_test <= upper)
        widths = upper - lower

        return {'coverages': coverages, 'widths': widths, 'vol_signal': vol_test}


class EWMAScaledCP:
    """
    EWMA-Scaled Conformal Prediction.

    Uses Exponentially Weighted Moving Average volatility (RiskMetrics style).
    This is simpler than GARCH and widely used in practice.
    """

    def __init__(self, alpha=0.1, ewma_lambda=0.94):
        self.alpha = alpha
        self.ewma_lambda = ewma_lambda  # RiskMetrics daily = 0.94

    def _compute_ewma_vol(self, returns):
        """Compute EWMA volatility"""
        lam = self.ewma_lambda
        n = len(returns)
        var = np.zeros(n)
        var[0] = returns.iloc[0]**2

        for t in range(1, n):
            var[t] = lam * var[t-1] + (1 - lam) * returns.iloc[t-1]**2

        return np.sqrt(var)

    def fit_predict(self, returns, cal_fraction=0.5):
        n = len(returns)
        cal_end = int(n * cal_fraction)

        # Compute EWMA volatility
        ewma_vol = self._compute_ewma_vol(returns)
        ewma_vol = ewma_vol / np.median(ewma_vol[:cal_end])  # Normalize

        y_cal = returns.iloc[:cal_end].values
        y_test = returns.iloc[cal_end:].values
        vol_cal = ewma_vol[:cal_end]
        vol_test = ewma_vol[cal_end:]

        pred = np.mean(y_cal)
        scores = np.abs(y_cal - pred) / np.maximum(vol_cal, 0.1)

        n_cal = len(scores)
        q_level = min(np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal, 1.0)
        q = np.quantile(scores, q_level)

        lower = pred - q * vol_test
        upper = pred + q * vol_test

        coverages = (y_test >= lower) & (y_test <= upper)
        widths = upper - lower

        return {'coverages': coverages, 'widths': widths}


class ConformizedQuantileRegression:
    """
    Conformalized Quantile Regression (CQR) - Romano et al. 2019

    Estimates quantiles directly and conformalized for finite-sample validity.
    """

    def __init__(self, alpha=0.1, window=60):
        self.alpha = alpha
        self.window = window

    def fit_predict(self, returns, cal_fraction=0.5):
        n = len(returns)
        cal_end = int(n * cal_fraction)

        y_cal = returns.iloc[:cal_end].values
        y_test = returns.iloc[cal_end:].values

        # Estimate quantiles from calibration data
        q_low = np.quantile(y_cal, self.alpha/2)
        q_high = np.quantile(y_cal, 1 - self.alpha/2)

        # CQR nonconformity scores: E_i = max(q_low - Y_i, Y_i - q_high)
        scores = np.maximum(q_low - y_cal, y_cal - q_high)

        n_cal = len(scores)
        q_level = min(np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal, 1.0)
        correction = np.quantile(scores, q_level)

        lower = q_low - correction
        upper = q_high + correction

        coverages = (y_test >= lower) & (y_test <= upper)
        widths = np.full(len(y_test), upper - lower)

        return {'coverages': coverages, 'widths': widths}


class HistoricalSimulation:
    """
    Historical Simulation VaR-style prediction intervals.

    Industry standard approach: use empirical quantiles from historical data.
    """

    def __init__(self, alpha=0.1, window=250):
        self.alpha = alpha
        self.window = window

    def fit_predict(self, returns, cal_fraction=0.5):
        n = len(returns)
        cal_end = int(n * cal_fraction)

        y_test = returns.iloc[cal_end:].values

        coverages = []
        widths = []

        for t in range(cal_end, n):
            # Use rolling window of historical returns
            start = max(0, t - self.window)
            hist = returns.iloc[start:t].values

            # Empirical quantiles
            lower = np.quantile(hist, self.alpha/2)
            upper = np.quantile(hist, 1 - self.alpha/2)

            actual = returns.iloc[t]
            coverages.append(lower <= actual <= upper)
            widths.append(upper - lower)

        return {'coverages': np.array(coverages), 'widths': np.array(widths)}


# ============================================================================
# Analysis
# ============================================================================

def analyze_by_volatility_regime(results, returns, cal_fraction=0.5, vol_window=12):
    """Analyze coverage by volatility regime"""
    n = len(returns)
    cal_end = int(n * cal_fraction)

    vol_signal = returns.rolling(vol_window).std()
    vol_test = vol_signal.iloc[cal_end:].values

    # Align coverages with vol_test
    coverages = results['coverages']
    if len(coverages) > len(vol_test):
        coverages = coverages[:len(vol_test)]
    elif len(coverages) < len(vol_test):
        vol_test = vol_test[:len(coverages)]

    vol_median = np.nanmedian(vol_test)
    high_vol_mask = vol_test > vol_median
    low_vol_mask = ~high_vol_mask & ~np.isnan(vol_test)

    widths = results['widths']
    if len(widths) > len(vol_test):
        widths = widths[:len(vol_test)]
    elif len(widths) < len(vol_test):
        widths = np.pad(widths, (0, len(vol_test) - len(widths)), constant_values=np.nan)

    n_high = high_vol_mask.sum()
    n_low = low_vol_mask.sum()

    high_cov = np.mean(coverages[high_vol_mask]) if n_high > 0 else np.nan
    low_cov = np.mean(coverages[low_vol_mask]) if n_low > 0 else np.nan

    se_high = np.sqrt(high_cov * (1 - high_cov) / n_high) if n_high > 0 and not np.isnan(high_cov) else np.nan
    se_low = np.sqrt(low_cov * (1 - low_cov) / n_low) if n_low > 0 and not np.isnan(low_cov) else np.nan

    return {
        'overall_coverage': np.mean(coverages),
        'high_vol_coverage': high_cov,
        'low_vol_coverage': low_cov,
        'overall_width': np.nanmean(widths),
        'high_vol_width': np.nanmean(widths[high_vol_mask]) if n_high > 0 else np.nan,
        'low_vol_width': np.nanmean(widths[low_vol_mask]) if n_low > 0 else np.nan,
        'n_high': n_high,
        'n_low': n_low,
        'se_high': se_high,
        'se_low': se_low
    }


# ============================================================================
# Main Experiment
# ============================================================================

def run_comprehensive_comparison():
    """Run comprehensive baseline comparison"""

    print("=" * 70)
    print("Comprehensive Baseline Comparison for JoFE Paper")
    print("=" * 70)

    ff = load_ff_data()
    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']

    print(f"Data: {ff.index[0]} to {ff.index[-1]} ({len(ff)} months)")

    methods = {
        'Standard CP': StandardCP(alpha=0.1),
        'Vol-Scaled CP': VolatilityScaledCP(alpha=0.1),
        'EWMA-Scaled CP': EWMAScaledCP(alpha=0.1),
        'CQR': ConformizedQuantileRegression(alpha=0.1),
        'Historical Sim': HistoricalSimulation(alpha=0.1),
    }

    all_results = []

    for factor in factors:
        print(f"\n{'='*50}")
        print(f"Factor: {factor}")
        print(f"{'='*50}")

        returns = ff[factor].dropna()

        for method_name, method in methods.items():
            print(f"\n  Running {method_name}...")

            try:
                results = method.fit_predict(returns)
                analysis = analyze_by_volatility_regime(results, returns)

                print(f"    Overall: {analysis['overall_coverage']:.1%}")
                print(f"    High-vol: {analysis['high_vol_coverage']:.1%} (±{analysis['se_high']:.1%})")
                print(f"    Low-vol: {analysis['low_vol_coverage']:.1%}")

                all_results.append({
                    'factor': factor,
                    'method': method_name,
                    **analysis
                })

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

    results_df = pd.DataFrame(all_results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Average Across All Factors")
    print("=" * 70)

    summary = results_df.groupby('method').agg({
        'overall_coverage': 'mean',
        'high_vol_coverage': 'mean',
        'low_vol_coverage': 'mean',
        'overall_width': 'mean',
    }).round(4)

    method_order = ['Standard CP', 'Historical Sim', 'CQR', 'EWMA-Scaled CP', 'Vol-Scaled CP']
    summary = summary.reindex([m for m in method_order if m in summary.index])

    print("\n" + summary.to_string())

    print("\n" + "=" * 70)
    print("HIGH-VOLATILITY COVERAGE COMPARISON")
    print("=" * 70)

    print("\n  Method              High-Vol Coverage    Gap from 90%")
    print("  " + "-" * 55)

    for method in method_order:
        if method in summary.index:
            cov = summary.loc[method, 'high_vol_coverage']
            gap = (cov - 0.90) * 100
            print(f"  {method:<20} {cov:>8.1%}         {gap:>+6.1f}pp")

    # Save
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    results_df.to_csv(output_dir / 'comprehensive_baselines.csv', index=False)
    summary.to_csv(output_dir / 'comprehensive_baselines_summary.csv')

    print(f"\nResults saved to {output_dir}/comprehensive_baselines*.csv")

    return results_df, summary


def generate_latex_table(results_df):
    """Generate LaTeX table for paper"""

    pivot = results_df.pivot(index='factor', columns='method', values='high_vol_coverage')

    col_order = ['Standard CP', 'Historical Sim', 'CQR', 'EWMA-Scaled CP', 'Vol-Scaled CP']
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    print("\n" + "=" * 70)
    print("LaTeX Table: Additional Baselines Comparison")
    print("=" * 70)

    latex = []
    latex.append(r"\begin{table}[H]")
    latex.append(r"\centering")
    latex.append(r"\caption{High-Volatility Coverage: Extended Baseline Comparison}")
    latex.append(r"\label{tab:extended_baselines}")
    latex.append(r"\begin{tabular}{l" + "c" * len(pivot.columns) + "}")
    latex.append(r"\toprule")

    header = "Factor & " + " & ".join(pivot.columns) + r" \\"
    latex.append(header)
    latex.append(r"\midrule")

    for factor in pivot.index:
        row = f"{factor}"
        for col in pivot.columns:
            val = pivot.loc[factor, col]
            if pd.notna(val):
                row += f" & {val:.1%}"
            else:
                row += " & ---"
        row += r" \\"
        latex.append(row)

    latex.append(r"\midrule")

    avg_row = r"\textbf{Average}"
    for col in pivot.columns:
        avg_row += f" & \\textbf{{{pivot[col].mean():.1%}}}"
    avg_row += r" \\"
    latex.append(avg_row)

    gap_row = r"\textbf{Gap from 90\%}"
    for col in pivot.columns:
        gap = (pivot[col].mean() - 0.90) * 100
        gap_row += f" & {gap:+.1f}pp"
    gap_row += r" \\"
    latex.append(gap_row)

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)
    print(latex_str)

    output_dir = Path(__file__).parent.parent / 'paper'
    with open(output_dir / 'extended_baselines_table.tex', 'w') as f:
        f.write(latex_str)

    print(f"\nSaved to {output_dir}/extended_baselines_table.tex")

    return latex_str


if __name__ == "__main__":
    results_df, summary = run_comprehensive_comparison()
    generate_latex_table(results_df)
