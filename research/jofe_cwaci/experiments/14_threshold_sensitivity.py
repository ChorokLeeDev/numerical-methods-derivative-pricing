"""
Sensitivity Analysis: Volatility Threshold Definition
======================================================
Tests robustness to different definitions of "high volatility":
- Median split (baseline)
- Tercile split (top 33%)
- Quartile split (top 25%)
- Top decile (top 10%)

Author: Chorok Lee (KAIST)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_ff_data():
    """Load Fama-French factor data"""
    data_path = Path(__file__).parent.parent / 'data' / 'ff_factors.csv'
    ff = pd.read_csv(data_path, index_col=0, parse_dates=True)
    return ff


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

        return coverages, vol_test


class StandardCP:
    """Standard CP for comparison"""

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

        coverages = (y_test >= pred - q) & (y_test <= pred + q)
        return coverages


def analyze_with_threshold(coverages, vol_signal, percentile):
    """
    Analyze coverage for observations above given volatility percentile.

    Parameters:
    -----------
    coverages : array
        Coverage indicators
    vol_signal : array
        Volatility signal aligned with coverages
    percentile : float
        Percentile threshold (e.g., 50 for median, 75 for top quartile)

    Returns:
    --------
    dict with coverage statistics
    """
    threshold = np.nanpercentile(vol_signal, percentile)
    high_vol_mask = vol_signal > threshold

    n_high = high_vol_mask.sum()
    if n_high == 0:
        return {'coverage': np.nan, 'n': 0, 'se': np.nan}

    cov = np.mean(coverages[high_vol_mask])
    se = np.sqrt(cov * (1 - cov) / n_high)

    return {
        'coverage': cov,
        'n': n_high,
        'se': se,
        'threshold_value': threshold
    }


def run_sensitivity_analysis():
    """Run sensitivity analysis across different threshold definitions"""

    print("=" * 70)
    print("Sensitivity Analysis: Volatility Threshold Definition")
    print("=" * 70)

    ff = load_ff_data()
    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']

    # Different threshold definitions
    thresholds = {
        'Median (>50%)': 50,
        'Tercile (>67%)': 67,
        'Quartile (>75%)': 75,
        'Decile (>90%)': 90
    }

    results = []

    for factor in factors:
        print(f"\n{factor}:")
        returns = ff[factor].dropna()

        # Vol-Scaled CP
        vs_cp = VolatilityScaledCP(alpha=0.1)
        vs_coverages, vol_signal = vs_cp.fit_predict(returns)

        # Standard CP
        std_cp = StandardCP(alpha=0.1)
        std_coverages = std_cp.fit_predict(returns)

        for thresh_name, percentile in thresholds.items():
            # Analyze Vol-Scaled CP
            vs_analysis = analyze_with_threshold(vs_coverages, vol_signal, percentile)
            std_analysis = analyze_with_threshold(std_coverages, vol_signal, percentile)

            print(f"  {thresh_name}: Vol-Scaled={vs_analysis['coverage']:.1%} (n={vs_analysis['n']}), "
                  f"Standard={std_analysis['coverage']:.1%}")

            results.append({
                'factor': factor,
                'threshold': thresh_name,
                'percentile': percentile,
                'vs_coverage': vs_analysis['coverage'],
                'vs_n': vs_analysis['n'],
                'vs_se': vs_analysis['se'],
                'std_coverage': std_analysis['coverage'],
                'std_n': std_analysis['n'],
                'std_se': std_analysis['se']
            })

    results_df = pd.DataFrame(results)

    # Summary by threshold
    print("\n" + "=" * 70)
    print("SUMMARY: Average Coverage by Threshold Definition")
    print("=" * 70)

    summary = results_df.groupby('threshold').agg({
        'vs_coverage': 'mean',
        'std_coverage': 'mean',
        'vs_n': 'mean'
    }).round(4)

    # Reorder
    order = ['Median (>50%)', 'Tercile (>67%)', 'Quartile (>75%)', 'Decile (>90%)']
    summary = summary.reindex(order)

    print("\n  Threshold          Vol-Scaled CP   Standard CP   Improvement   Avg N")
    print("  " + "-" * 70)

    for thresh in order:
        if thresh in summary.index:
            vs = summary.loc[thresh, 'vs_coverage']
            std = summary.loc[thresh, 'std_coverage']
            n = summary.loc[thresh, 'vs_n']
            improvement = (vs - std) * 100
            print(f"  {thresh:<20} {vs:>8.1%}       {std:>8.1%}      {improvement:>+6.1f}pp    {n:>6.0f}")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    results_df.to_csv(output_dir / 'threshold_sensitivity.csv', index=False)
    summary.to_csv(output_dir / 'threshold_sensitivity_summary.csv')

    print(f"\nResults saved to {output_dir}/threshold_sensitivity*.csv")

    return results_df, summary


def generate_latex_table(results_df):
    """Generate LaTeX table for paper"""

    # Pivot for summary
    pivot_vs = results_df.pivot(index='factor', columns='threshold', values='vs_coverage')
    pivot_std = results_df.pivot(index='factor', columns='threshold', values='std_coverage')

    col_order = ['Median (>50%)', 'Tercile (>67%)', 'Quartile (>75%)', 'Decile (>90%)']
    pivot_vs = pivot_vs[[c for c in col_order if c in pivot_vs.columns]]
    pivot_std = pivot_std[[c for c in col_order if c in pivot_std.columns]]

    print("\n" + "=" * 70)
    print("LaTeX Table: Sensitivity to Threshold Definition")
    print("=" * 70)

    # Create combined table showing Vol-Scaled CP coverage across thresholds
    latex = []
    latex.append(r"\begin{table}[H]")
    latex.append(r"\centering")
    latex.append(r"\caption{Sensitivity Analysis: Coverage by High-Volatility Definition}")
    latex.append(r"\label{tab:threshold_sensitivity}")
    latex.append(r"\begin{tabular}{lcccc}")
    latex.append(r"\toprule")
    latex.append(r"& \multicolumn{4}{c}{Vol-Scaled CP Coverage} \\")
    latex.append(r"\cmidrule(lr){2-5}")
    latex.append(r"Factor & $>$50\% & $>$67\% & $>$75\% & $>$90\% \\")
    latex.append(r"\midrule")

    for factor in pivot_vs.index:
        row = f"{factor}"
        for col in col_order:
            if col in pivot_vs.columns:
                val = pivot_vs.loc[factor, col]
                row += f" & {val:.1%}"
        row += r" \\"
        latex.append(row)

    latex.append(r"\midrule")

    avg_row = r"\textbf{Average}"
    for col in col_order:
        if col in pivot_vs.columns:
            avg_row += f" & \\textbf{{{pivot_vs[col].mean():.1%}}}"
    avg_row += r" \\"
    latex.append(avg_row)

    # Standard CP comparison
    latex.append(r"\midrule")
    std_row = r"Standard CP (avg)"
    for col in col_order:
        if col in pivot_std.columns:
            std_row += f" & {pivot_std[col].mean():.1%}"
    std_row += r" \\"
    latex.append(std_row)

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\begin{flushleft}")
    latex.append(r"\small\textit{Note:} Columns show coverage for observations above the indicated volatility percentile.")
    latex.append(r"\end{flushleft}")
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)
    print(latex_str)

    output_dir = Path(__file__).parent.parent / 'paper'
    with open(output_dir / 'threshold_sensitivity_table.tex', 'w') as f:
        f.write(latex_str)

    return latex_str


if __name__ == "__main__":
    results_df, summary = run_sensitivity_analysis()
    generate_latex_table(results_df)
