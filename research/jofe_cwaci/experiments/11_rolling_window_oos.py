"""
Out-of-Sample Rolling Window Analysis
======================================
True out-of-sample validation using expanding/rolling calibration windows.

This addresses the reviewer concern: "Is this just in-sample overfitting?"

Methodology:
- Start with minimum calibration window (e.g., 120 months = 10 years)
- At each time t, calibrate on data up to t-1, predict for t
- Track coverage over time and by volatility regime
- Compare expanding window vs fixed rolling window

Author: Chorok Lee (KAIST)
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_ff_data():
    """Load Fama-French factor data"""
    data_path = Path(__file__).parent.parent / 'data' / 'ff_factors.csv'
    ff = pd.read_csv(data_path, index_col=0, parse_dates=True)
    return ff


class RollingConformalPredictor:
    """
    Rolling/expanding window conformal prediction for true OOS evaluation.
    """

    def __init__(self, alpha=0.1, vol_window=12, min_cal_window=120):
        """
        Parameters
        ----------
        alpha : float
            Miscoverage rate (default 0.1 for 90% coverage)
        vol_window : int
            Window for volatility estimation (months)
        min_cal_window : int
            Minimum calibration window size (months)
        """
        self.alpha = alpha
        self.vol_window = vol_window
        self.min_cal_window = min_cal_window

    def run_expanding_window(self, returns, method='vol_scaled'):
        """
        Expanding window: calibrate on all data up to t-1, predict t.

        Parameters
        ----------
        returns : pd.Series
            Return series
        method : str
            'standard' or 'vol_scaled'

        Returns
        -------
        dict with predictions, coverages, widths, etc.
        """
        n = len(returns)
        vol_signal = returns.rolling(self.vol_window).std()
        vol_signal = vol_signal / vol_signal.expanding().median()

        # Start after minimum calibration window + vol window
        start_idx = self.min_cal_window + self.vol_window

        results = {
            'dates': [],
            'actual': [],
            'lower': [],
            'upper': [],
            'width': [],
            'covered': [],
            'vol_signal': [],
            'cal_size': []
        }

        for t in range(start_idx, n):
            # Calibration data: all data up to t-1
            y_cal = returns.iloc[:t].values
            vol_cal = vol_signal.iloc[:t].values

            # Test point
            y_test = returns.iloc[t]
            vol_test = vol_signal.iloc[t]

            # Handle NaN in volatility
            valid_cal = ~np.isnan(vol_cal)
            y_cal_valid = y_cal[valid_cal]
            vol_cal_valid = vol_cal[valid_cal]

            if len(y_cal_valid) < 30 or np.isnan(vol_test):
                continue

            # Point prediction: calibration mean
            pred = np.mean(y_cal_valid)

            if method == 'standard':
                # Standard CP: fixed interval
                scores = np.abs(y_cal_valid - pred)
                n_cal = len(scores)
                q_level = min(np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal, 1.0)
                q = np.quantile(scores, q_level)

                lower = pred - q
                upper = pred + q

            elif method == 'vol_scaled':
                # Volatility-scaled CP
                scores = np.abs(y_cal_valid - pred) / np.maximum(vol_cal_valid, 0.1)
                n_cal = len(scores)
                q_level = min(np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal, 1.0)
                q = np.quantile(scores, q_level)

                lower = pred - q * vol_test
                upper = pred + q * vol_test

            # Record results
            results['dates'].append(returns.index[t])
            results['actual'].append(y_test)
            results['lower'].append(lower)
            results['upper'].append(upper)
            results['width'].append(upper - lower)
            results['covered'].append(lower <= y_test <= upper)
            results['vol_signal'].append(vol_test)
            results['cal_size'].append(len(y_cal_valid))

        # Convert to arrays
        for key in results:
            results[key] = np.array(results[key])

        return results

    def run_rolling_window(self, returns, method='vol_scaled', window_size=120):
        """
        Fixed rolling window: calibrate on most recent `window_size` observations.

        Parameters
        ----------
        returns : pd.Series
            Return series
        method : str
            'standard' or 'vol_scaled'
        window_size : int
            Fixed calibration window size

        Returns
        -------
        dict with predictions, coverages, widths, etc.
        """
        n = len(returns)
        vol_signal = returns.rolling(self.vol_window).std()
        vol_signal = vol_signal / vol_signal.expanding().median()

        # Start after window + vol window
        start_idx = window_size + self.vol_window

        results = {
            'dates': [],
            'actual': [],
            'lower': [],
            'upper': [],
            'width': [],
            'covered': [],
            'vol_signal': [],
            'cal_size': []
        }

        for t in range(start_idx, n):
            # Calibration data: most recent window_size observations
            cal_start = t - window_size
            y_cal = returns.iloc[cal_start:t].values
            vol_cal = vol_signal.iloc[cal_start:t].values

            # Test point
            y_test = returns.iloc[t]
            vol_test = vol_signal.iloc[t]

            # Handle NaN
            valid_cal = ~np.isnan(vol_cal)
            y_cal_valid = y_cal[valid_cal]
            vol_cal_valid = vol_cal[valid_cal]

            if len(y_cal_valid) < 30 or np.isnan(vol_test):
                continue

            # Point prediction
            pred = np.mean(y_cal_valid)

            if method == 'standard':
                scores = np.abs(y_cal_valid - pred)
                n_cal = len(scores)
                q_level = min(np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal, 1.0)
                q = np.quantile(scores, q_level)
                lower = pred - q
                upper = pred + q

            elif method == 'vol_scaled':
                scores = np.abs(y_cal_valid - pred) / np.maximum(vol_cal_valid, 0.1)
                n_cal = len(scores)
                q_level = min(np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal, 1.0)
                q = np.quantile(scores, q_level)
                lower = pred - q * vol_test
                upper = pred + q * vol_test

            results['dates'].append(returns.index[t])
            results['actual'].append(y_test)
            results['lower'].append(lower)
            results['upper'].append(upper)
            results['width'].append(upper - lower)
            results['covered'].append(lower <= y_test <= upper)
            results['vol_signal'].append(vol_test)
            results['cal_size'].append(len(y_cal_valid))

        for key in results:
            results[key] = np.array(results[key])

        return results


def analyze_results(results, name=""):
    """Analyze rolling window results"""

    covered = results['covered']
    vol = results['vol_signal']
    widths = results['width']

    # Overall coverage
    overall_cov = np.mean(covered)

    # Coverage by volatility regime
    vol_median = np.nanmedian(vol)
    high_vol = vol > vol_median
    low_vol = ~high_vol

    high_vol_cov = np.mean(covered[high_vol]) if high_vol.sum() > 0 else np.nan
    low_vol_cov = np.mean(covered[low_vol]) if low_vol.sum() > 0 else np.nan

    # Standard errors
    n_high = high_vol.sum()
    n_low = low_vol.sum()
    se_high = np.sqrt(high_vol_cov * (1 - high_vol_cov) / n_high) if n_high > 0 else np.nan
    se_low = np.sqrt(low_vol_cov * (1 - low_vol_cov) / n_low) if n_low > 0 else np.nan

    # Average widths
    avg_width = np.mean(widths)
    high_vol_width = np.mean(widths[high_vol]) if high_vol.sum() > 0 else np.nan
    low_vol_width = np.mean(widths[low_vol]) if low_vol.sum() > 0 else np.nan

    return {
        'name': name,
        'n_predictions': len(covered),
        'overall_coverage': overall_cov,
        'high_vol_coverage': high_vol_cov,
        'low_vol_coverage': low_vol_cov,
        'se_high': se_high,
        'se_low': se_low,
        'n_high': n_high,
        'n_low': n_low,
        'avg_width': avg_width,
        'high_vol_width': high_vol_width,
        'low_vol_width': low_vol_width
    }


def analyze_by_decade(results, returns):
    """Analyze coverage by decade"""
    dates = pd.to_datetime(results['dates'])
    covered = results['covered']

    decades = {}
    for decade_start in [1970, 1980, 1990, 2000, 2010, 2020]:
        decade_end = decade_start + 10
        mask = (dates.year >= decade_start) & (dates.year < decade_end)
        if mask.sum() > 0:
            decades[f"{decade_start}s"] = np.mean(covered[mask])

    return decades


def run_full_analysis():
    """Run complete rolling window analysis"""

    print("=" * 70)
    print("Out-of-Sample Rolling Window Analysis")
    print("=" * 70)

    ff = load_ff_data()
    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']

    print(f"\nData: {ff.index[0]} to {ff.index[-1]} ({len(ff)} months)")
    print(f"Minimum calibration window: 120 months (10 years)")
    print(f"Volatility window: 12 months")

    predictor = RollingConformalPredictor(alpha=0.1, vol_window=12, min_cal_window=120)

    all_results = []

    for factor in factors:
        print(f"\n{'='*50}")
        print(f"Factor: {factor}")
        print(f"{'='*50}")

        returns = ff[factor].dropna()

        # Method 1: Expanding window, Standard CP
        print("\n  Expanding Window - Standard CP...")
        res_exp_std = predictor.run_expanding_window(returns, method='standard')
        analysis_exp_std = analyze_results(res_exp_std, "Expanding-Standard")
        print(f"    Overall: {analysis_exp_std['overall_coverage']:.1%}")
        print(f"    High-vol: {analysis_exp_std['high_vol_coverage']:.1%} (±{analysis_exp_std['se_high']:.1%})")
        print(f"    Low-vol: {analysis_exp_std['low_vol_coverage']:.1%}")

        # Method 2: Expanding window, Vol-Scaled CP
        print("\n  Expanding Window - Vol-Scaled CP...")
        res_exp_vs = predictor.run_expanding_window(returns, method='vol_scaled')
        analysis_exp_vs = analyze_results(res_exp_vs, "Expanding-VolScaled")
        print(f"    Overall: {analysis_exp_vs['overall_coverage']:.1%}")
        print(f"    High-vol: {analysis_exp_vs['high_vol_coverage']:.1%} (±{analysis_exp_vs['se_high']:.1%})")
        print(f"    Low-vol: {analysis_exp_vs['low_vol_coverage']:.1%}")

        # Method 3: Rolling window (120 months), Standard CP
        print("\n  Rolling Window (120mo) - Standard CP...")
        res_roll_std = predictor.run_rolling_window(returns, method='standard', window_size=120)
        analysis_roll_std = analyze_results(res_roll_std, "Rolling-Standard")
        print(f"    Overall: {analysis_roll_std['overall_coverage']:.1%}")
        print(f"    High-vol: {analysis_roll_std['high_vol_coverage']:.1%} (±{analysis_roll_std['se_high']:.1%})")
        print(f"    Low-vol: {analysis_roll_std['low_vol_coverage']:.1%}")

        # Method 4: Rolling window (120 months), Vol-Scaled CP
        print("\n  Rolling Window (120mo) - Vol-Scaled CP...")
        res_roll_vs = predictor.run_rolling_window(returns, method='vol_scaled', window_size=120)
        analysis_roll_vs = analyze_results(res_roll_vs, "Rolling-VolScaled")
        print(f"    Overall: {analysis_roll_vs['overall_coverage']:.1%}")
        print(f"    High-vol: {analysis_roll_vs['high_vol_coverage']:.1%} (±{analysis_roll_vs['se_high']:.1%})")
        print(f"    Low-vol: {analysis_roll_vs['low_vol_coverage']:.1%}")

        # Store results
        for analysis in [analysis_exp_std, analysis_exp_vs, analysis_roll_std, analysis_roll_vs]:
            all_results.append({
                'factor': factor,
                **analysis
            })

        # Decade analysis for vol-scaled expanding window
        decades = analyze_by_decade(res_exp_vs, returns)
        print(f"\n  Coverage by decade (Vol-Scaled, Expanding):")
        for decade, cov in decades.items():
            print(f"    {decade}: {cov:.1%}")

    # Create summary DataFrame
    results_df = pd.DataFrame(all_results)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: High-Volatility Coverage (Out-of-Sample)")
    print("=" * 70)

    summary = results_df.groupby('name').agg({
        'high_vol_coverage': 'mean',
        'low_vol_coverage': 'mean',
        'overall_coverage': 'mean',
        'avg_width': 'mean'
    }).round(4)

    print("\n" + summary.to_string())

    # Key comparison
    print("\n" + "=" * 70)
    print("KEY FINDING: Out-of-Sample High-Volatility Coverage")
    print("=" * 70)

    methods = ['Expanding-Standard', 'Expanding-VolScaled', 'Rolling-Standard', 'Rolling-VolScaled']
    print("\n  Method                    High-Vol Coverage    Gap from 90%")
    print("  " + "-" * 60)

    for method in methods:
        if method in summary.index:
            cov = summary.loc[method, 'high_vol_coverage']
            gap = (cov - 0.90) * 100
            print(f"  {method:<27} {cov:>8.1%}         {gap:>+6.1f}pp")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    results_df.to_csv(output_dir / 'rolling_window_oos_detailed.csv', index=False)
    summary.to_csv(output_dir / 'rolling_window_oos_summary.csv')

    print(f"\nResults saved to {output_dir}/rolling_window_oos_*.csv")

    return results_df, summary


def generate_latex_table(results_df):
    """Generate LaTeX table for paper"""

    print("\n" + "=" * 70)
    print("LaTeX Table: Out-of-Sample Rolling Window Results")
    print("=" * 70)

    # Pivot table for high-vol coverage
    pivot = results_df.pivot(index='factor', columns='name', values='high_vol_coverage')

    # Reorder columns
    col_order = ['Expanding-Standard', 'Rolling-Standard', 'Expanding-VolScaled', 'Rolling-VolScaled']
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    latex = []
    latex.append(r"\begin{table}[H]")
    latex.append(r"\centering")
    latex.append(r"\caption{Out-of-Sample High-Volatility Coverage: Rolling Window Analysis}")
    latex.append(r"\label{tab:oos_rolling}")
    latex.append(r"\begin{tabular}{lcccc}")
    latex.append(r"\toprule")
    latex.append(r"& \multicolumn{2}{c}{Standard CP} & \multicolumn{2}{c}{Vol-Scaled CP} \\")
    latex.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    latex.append(r"Factor & Expanding & Rolling & Expanding & Rolling \\")
    latex.append(r"\midrule")

    for factor in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']:
        if factor in pivot.index:
            row = f"{factor}"
            for col in col_order:
                if col in pivot.columns:
                    val = pivot.loc[factor, col]
                    row += f" & {val:.1%}"
            row += r" \\"
            latex.append(row)

    latex.append(r"\midrule")

    # Average row
    avg_row = r"\textbf{Average}"
    for col in col_order:
        if col in pivot.columns:
            avg_row += f" & \\textbf{{{pivot[col].mean():.1%}}}"
    avg_row += r" \\"
    latex.append(avg_row)

    # Gap row
    gap_row = r"\textbf{Gap from 90\%}"
    for col in col_order:
        if col in pivot.columns:
            gap = (pivot[col].mean() - 0.90) * 100
            gap_row += f" & {gap:+.1f}pp"
    gap_row += r" \\"
    latex.append(gap_row)

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)
    print(latex_str)

    # Save
    output_dir = Path(__file__).parent.parent / 'paper'
    with open(output_dir / 'oos_rolling_table.tex', 'w') as f:
        f.write(latex_str)

    print(f"\nSaved to {output_dir}/oos_rolling_table.tex")

    return latex_str


def generate_figure(results_df):
    """Generate comparison figure"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']

    # Left panel: Expanding window comparison
    ax1 = axes[0]
    x = np.arange(len(factors))
    width = 0.35

    std_vals = []
    vs_vals = []
    for factor in factors:
        std_data = results_df[(results_df['factor'] == factor) & (results_df['name'] == 'Expanding-Standard')]
        vs_data = results_df[(results_df['factor'] == factor) & (results_df['name'] == 'Expanding-VolScaled')]
        std_vals.append(std_data['high_vol_coverage'].values[0] if len(std_data) > 0 else np.nan)
        vs_vals.append(vs_data['high_vol_coverage'].values[0] if len(vs_data) > 0 else np.nan)

    ax1.bar(x - width/2, std_vals, width, label='Standard CP', color='#d62728', alpha=0.8)
    ax1.bar(x + width/2, vs_vals, width, label='Vol-Scaled CP', color='#2ca02c', alpha=0.8)
    ax1.axhline(y=0.90, color='black', linestyle='--', linewidth=1.5, label='90% Target')

    ax1.set_xlabel('Factor', fontsize=11)
    ax1.set_ylabel('High-Volatility Coverage', fontsize=11)
    ax1.set_title('(a) Expanding Window (True OOS)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(factors)
    ax1.set_ylim(0.6, 1.0)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Right panel: Method comparison (averaged)
    ax2 = axes[1]

    methods = ['Expanding-Standard', 'Rolling-Standard', 'Expanding-VolScaled', 'Rolling-VolScaled']
    method_labels = ['Expand.\nStd CP', 'Roll.\nStd CP', 'Expand.\nVol-Scaled', 'Roll.\nVol-Scaled']
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

    summary = results_df.groupby('name')['high_vol_coverage'].mean()
    vals = [summary.get(m, np.nan) for m in methods]

    bars = ax2.bar(range(len(methods)), vals, color=colors, alpha=0.8)
    ax2.axhline(y=0.90, color='black', linestyle='--', linewidth=1.5, label='90% Target')

    ax2.set_xlabel('Method', fontsize=11)
    ax2.set_ylabel('High-Volatility Coverage (Avg)', fontsize=11)
    ax2.set_title('(b) Method Comparison (Average Across Factors)', fontsize=12)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(method_labels, fontsize=9)
    ax2.set_ylim(0.6, 1.0)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, vals)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save
    output_dir = Path(__file__).parent.parent / 'paper' / 'figures'
    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig_oos_rolling.{ext}', dpi=300, bbox_inches='tight')

    print(f"\nFigure saved to {output_dir}/fig_oos_rolling.[pdf|png]")
    plt.close()


if __name__ == "__main__":
    results_df, summary = run_full_analysis()
    generate_latex_table(results_df)
    generate_figure(results_df)
