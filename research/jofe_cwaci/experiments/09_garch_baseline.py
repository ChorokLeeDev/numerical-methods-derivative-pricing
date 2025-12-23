"""
GARCH Baseline Comparison for JoFE Paper
==========================================
Rigorous comparison of volatility-scaled CP against GARCH(1,1) prediction intervals.

This addresses the key reviewer question: "Why not just use GARCH?"

GARCH is the standard approach in finance for modeling time-varying volatility.
We compare:
1. Standard CP (fixed intervals)
2. Volatility-Scaled CP (our method)
3. GARCH(1,1) prediction intervals

Key finding: GARCH and Vol-Scaled CP both achieve similar high-vol coverage,
but Vol-Scaled CP requires no distributional assumptions and provides
finite-sample coverage guarantees.

Author: Chorok Lee (KAIST)
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# GARCH modeling
from arch import arch_model

# ============================================================================
# Data Loading
# ============================================================================

def load_ff_data():
    """Load Fama-French factor data"""
    data_path = Path(__file__).parent.parent / 'data' / 'ff_factors.csv'
    ff = pd.read_csv(data_path, index_col=0, parse_dates=True)
    return ff


# ============================================================================
# GARCH Prediction Intervals (Proper Implementation)
# ============================================================================

class GARCHPredictor:
    """
    GARCH(1,1) based prediction intervals using Maximum Likelihood Estimation.

    Uses the arch package for proper GARCH estimation.
    Intervals are: μ ± z_{α/2} * σ_t where σ_t is GARCH conditional volatility.

    This is the standard finance approach for volatility-adjusted prediction intervals.
    """

    def __init__(self, alpha=0.1, dist='normal'):
        """
        Parameters
        ----------
        alpha : float
            Miscoverage rate (default 0.1 for 90% coverage)
        dist : str
            Distribution assumption: 'normal', 't', or 'skewt'
        """
        self.alpha = alpha
        self.dist = dist

    def fit_predict_rolling(self, returns, cal_fraction=0.5, refit_every=12):
        """
        Rolling GARCH prediction intervals.

        Parameters
        ----------
        returns : pd.Series
            Return series
        cal_fraction : float
            Fraction for initial calibration
        refit_every : int
            Refit GARCH every N periods (default: 12 months)

        Returns
        -------
        dict with coverages, widths, forecasted_vols
        """
        n = len(returns)
        cal_end = int(n * cal_fraction)

        coverages = []
        widths = []
        forecasted_vols = []
        lower_bounds = []
        upper_bounds = []

        # Track GARCH parameters
        last_fit_idx = 0
        garch_params = None

        for t in range(cal_end, n):
            # Get training data
            train_data = returns.iloc[:t]

            # Refit GARCH periodically or on first iteration
            if t == cal_end or (t - last_fit_idx) >= refit_every:
                try:
                    # Fit GARCH(1,1)
                    # Scale returns to percentage for numerical stability
                    train_scaled = train_data * 100

                    model = arch_model(
                        train_scaled,
                        vol='Garch',
                        p=1, q=1,
                        dist=self.dist,
                        mean='Constant'
                    )
                    result = model.fit(disp='off', show_warning=False)
                    garch_params = result
                    last_fit_idx = t

                except Exception as e:
                    # If GARCH fails, use simple rolling volatility
                    if garch_params is None:
                        garch_params = None

            # Forecast one-step-ahead volatility
            if garch_params is not None:
                try:
                    # Get conditional variance forecast
                    forecast = garch_params.forecast(horizon=1, reindex=False)
                    sigma_forecast = np.sqrt(forecast.variance.values[-1, 0]) / 100  # Unscale
                    mu_forecast = garch_params.params.get('mu', 0) / 100  # Unscale mean
                except:
                    # Fallback to rolling volatility
                    sigma_forecast = train_data.iloc[-60:].std() if len(train_data) > 60 else train_data.std()
                    mu_forecast = train_data.iloc[-12:].mean()
            else:
                # Fallback
                sigma_forecast = train_data.iloc[-60:].std() if len(train_data) > 60 else train_data.std()
                mu_forecast = train_data.iloc[-12:].mean()

            # Compute prediction interval
            if self.dist == 't' and garch_params is not None:
                # Student-t quantiles
                try:
                    nu = garch_params.params.get('nu', 5)  # degrees of freedom
                    z_crit = stats.t.ppf(1 - self.alpha/2, nu)
                except:
                    z_crit = stats.norm.ppf(1 - self.alpha/2)
            else:
                z_crit = stats.norm.ppf(1 - self.alpha/2)

            lower = mu_forecast - z_crit * sigma_forecast
            upper = mu_forecast + z_crit * sigma_forecast

            # Check coverage
            actual = returns.iloc[t]
            covered = (lower <= actual <= upper)

            coverages.append(covered)
            widths.append(upper - lower)
            forecasted_vols.append(sigma_forecast)
            lower_bounds.append(lower)
            upper_bounds.append(upper)

        return {
            'coverages': np.array(coverages),
            'widths': np.array(widths),
            'forecasted_vols': np.array(forecasted_vols),
            'lower': np.array(lower_bounds),
            'upper': np.array(upper_bounds)
        }


# ============================================================================
# Conformal Prediction Methods
# ============================================================================

class StandardCP:
    """Standard split conformal prediction (fixed intervals)"""

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit_predict(self, returns, cal_fraction=0.5):
        """Run standard CP"""
        n = len(returns)
        cal_end = int(n * cal_fraction)

        y_cal = returns.iloc[:cal_end].values
        y_test = returns.iloc[cal_end:].values

        # Point prediction: calibration mean
        pred = np.mean(y_cal)

        # Nonconformity scores
        scores = np.abs(y_cal - pred)

        # Quantile with finite-sample correction
        n_cal = len(scores)
        q_level = min(np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal, 1.0)
        q = np.quantile(scores, q_level)

        # Fixed intervals
        lower = pred - q
        upper = pred + q

        coverages = (y_test >= lower) & (y_test <= upper)
        widths = np.full(len(y_test), 2 * q)

        return {
            'coverages': coverages,
            'widths': widths,
            'lower': np.full(len(y_test), lower),
            'upper': np.full(len(y_test), upper)
        }


class VolatilityScaledCP:
    """
    Volatility-Scaled Conformal Prediction (our method).

    Key insight: Scale nonconformity scores by volatility to restore exchangeability.
    """

    def __init__(self, alpha=0.1, vol_window=12):
        self.alpha = alpha
        self.vol_window = vol_window

    def fit_predict(self, returns, cal_fraction=0.5):
        """Run volatility-scaled CP"""
        n = len(returns)
        cal_end = int(n * cal_fraction)

        # Compute volatility signal
        vol_signal = returns.rolling(self.vol_window).std()
        vol_signal = vol_signal / vol_signal.expanding().median()  # Normalize

        # Skip NaN periods
        valid_start = vol_signal.first_valid_index()
        if valid_start is None:
            valid_start = returns.index[self.vol_window]

        y_cal = returns.iloc[:cal_end].values
        y_test = returns.iloc[cal_end:].values
        vol_cal = vol_signal.iloc[:cal_end].values
        vol_test = vol_signal.iloc[cal_end:].values

        # Handle NaN in volatility
        vol_cal = np.nan_to_num(vol_cal, nan=1.0)
        vol_test = np.nan_to_num(vol_test, nan=1.0)

        # Point prediction
        pred = np.mean(y_cal)

        # STANDARDIZED nonconformity scores (key innovation)
        scores = np.abs(y_cal - pred) / np.maximum(vol_cal, 0.1)

        # Quantile with finite-sample correction
        n_cal = len(scores)
        q_level = min(np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal, 1.0)
        q = np.quantile(scores, q_level)

        # SCALED intervals
        lower = pred - q * vol_test
        upper = pred + q * vol_test

        coverages = (y_test >= lower) & (y_test <= upper)
        widths = upper - lower

        return {
            'coverages': coverages,
            'widths': widths,
            'lower': lower,
            'upper': upper,
            'vol_signal': vol_test
        }


# ============================================================================
# Volatility Regime Analysis
# ============================================================================

def analyze_by_volatility_regime(results, returns, cal_fraction=0.5, vol_window=12):
    """
    Analyze coverage by volatility regime (high vs low).

    High volatility = above median realized vol
    """
    n = len(returns)
    cal_end = int(n * cal_fraction)

    # Compute realized volatility on test period
    vol_signal = returns.rolling(vol_window).std()
    vol_test = vol_signal.iloc[cal_end:].values

    # High/low volatility thresholds
    vol_median = np.nanmedian(vol_test)
    high_vol_mask = vol_test > vol_median
    low_vol_mask = ~high_vol_mask & ~np.isnan(vol_test)

    coverages = results['coverages']
    widths = results['widths']

    analysis = {
        'overall_coverage': np.mean(coverages),
        'high_vol_coverage': np.mean(coverages[high_vol_mask]) if high_vol_mask.sum() > 0 else np.nan,
        'low_vol_coverage': np.mean(coverages[low_vol_mask]) if low_vol_mask.sum() > 0 else np.nan,
        'overall_width': np.mean(widths),
        'high_vol_width': np.mean(widths[high_vol_mask]) if high_vol_mask.sum() > 0 else np.nan,
        'low_vol_width': np.mean(widths[low_vol_mask]) if low_vol_mask.sum() > 0 else np.nan,
        'n_high': high_vol_mask.sum(),
        'n_low': low_vol_mask.sum(),
        'se_high': np.sqrt(np.mean(coverages[high_vol_mask]) * (1 - np.mean(coverages[high_vol_mask])) / high_vol_mask.sum()) if high_vol_mask.sum() > 0 else np.nan,
        'se_low': np.sqrt(np.mean(coverages[low_vol_mask]) * (1 - np.mean(coverages[low_vol_mask])) / low_vol_mask.sum()) if low_vol_mask.sum() > 0 else np.nan
    }

    return analysis


# ============================================================================
# Main Experiment
# ============================================================================

def run_garch_comparison():
    """Run comprehensive GARCH baseline comparison"""

    print("=" * 70)
    print("GARCH Baseline Comparison for JoFE Paper")
    print("=" * 70)

    # Load data
    print("\nLoading Fama-French data...")
    ff = load_ff_data()
    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']

    print(f"Data: {ff.index[0]} to {ff.index[-1]} ({len(ff)} months)")

    # Methods to compare
    methods = {
        'Standard CP': StandardCP(alpha=0.1),
        'Vol-Scaled CP': VolatilityScaledCP(alpha=0.1),
        'GARCH(1,1)-N': GARCHPredictor(alpha=0.1, dist='normal'),
        'GARCH(1,1)-t': GARCHPredictor(alpha=0.1, dist='t'),
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
                if 'GARCH' in method_name:
                    results = method.fit_predict_rolling(returns)
                else:
                    results = method.fit_predict(returns)

                # Analyze by volatility regime
                analysis = analyze_by_volatility_regime(results, returns)

                print(f"    Overall: {analysis['overall_coverage']:.1%}")
                print(f"    High-vol: {analysis['high_vol_coverage']:.1%} (±{analysis['se_high']:.1%})")
                print(f"    Low-vol: {analysis['low_vol_coverage']:.1%}")
                print(f"    Avg width: {analysis['overall_width']:.4f}")

                all_results.append({
                    'factor': factor,
                    'method': method_name,
                    **analysis
                })

            except Exception as e:
                print(f"    ERROR: {e}")
                all_results.append({
                    'factor': factor,
                    'method': method_name,
                    'overall_coverage': np.nan,
                    'high_vol_coverage': np.nan,
                    'low_vol_coverage': np.nan,
                    'overall_width': np.nan,
                    'high_vol_width': np.nan,
                    'low_vol_width': np.nan,
                    'n_high': np.nan,
                    'n_low': np.nan,
                    'se_high': np.nan,
                    'se_low': np.nan
                })

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: Average Across All Factors")
    print("=" * 70)

    summary = results_df.groupby('method').agg({
        'overall_coverage': 'mean',
        'high_vol_coverage': 'mean',
        'low_vol_coverage': 'mean',
        'overall_width': 'mean',
        'high_vol_width': 'mean',
        'low_vol_width': 'mean'
    }).round(4)

    # Reorder
    method_order = ['Standard CP', 'Vol-Scaled CP', 'GARCH(1,1)-N', 'GARCH(1,1)-t']
    summary = summary.reindex([m for m in method_order if m in summary.index])

    print("\n" + summary.to_string())

    # Statistical comparison
    print("\n" + "=" * 70)
    print("KEY FINDING: High-Volatility Coverage Comparison")
    print("=" * 70)

    print("\n  Method              High-Vol Coverage    Gap from 90%")
    print("  " + "-" * 55)

    for method in method_order:
        if method in summary.index:
            cov = summary.loc[method, 'high_vol_coverage']
            gap = (cov - 0.90) * 100
            print(f"  {method:<20} {cov:>8.1%}         {gap:>+6.1f}pp")

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    results_df.to_csv(output_dir / 'garch_comparison_detailed.csv', index=False)
    summary.to_csv(output_dir / 'garch_comparison_summary.csv')

    print(f"\nResults saved to {output_dir}/garch_comparison_*.csv")

    return results_df, summary


def generate_latex_table(results_df):
    """Generate LaTeX table for paper"""

    # Pivot for high-vol coverage
    pivot = results_df.pivot(index='factor', columns='method', values='high_vol_coverage')

    # Reorder columns
    col_order = ['Standard CP', 'GARCH(1,1)-N', 'GARCH(1,1)-t', 'Vol-Scaled CP']
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    # Add standard errors
    pivot_se = results_df.pivot(index='factor', columns='method', values='se_high')

    print("\n" + "=" * 70)
    print("LaTeX Table: High-Volatility Coverage by Method")
    print("=" * 70)

    latex = []
    latex.append(r"\begin{table}[H]")
    latex.append(r"\centering")
    latex.append(r"\caption{High-Volatility Coverage: Conformal Prediction vs GARCH}")
    latex.append(r"\label{tab:garch_comparison}")
    latex.append(r"\begin{tabular}{l" + "c" * len(pivot.columns) + "}")
    latex.append(r"\toprule")

    # Header
    header = "Factor & " + " & ".join(pivot.columns) + r" \\"
    latex.append(header)
    latex.append(r"\midrule")

    # Data rows
    for factor in pivot.index:
        row = f"{factor}"
        for col in pivot.columns:
            val = pivot.loc[factor, col]
            se = pivot_se.loc[factor, col]
            if pd.notna(val):
                row += f" & {val:.1%}"
            else:
                row += " & ---"
        row += r" \\"
        latex.append(row)

    latex.append(r"\midrule")

    # Average row
    avg_row = r"\textbf{Average}"
    for col in pivot.columns:
        avg_row += f" & \\textbf{{{pivot[col].mean():.1%}}}"
    avg_row += r" \\"
    latex.append(avg_row)

    # Gap from target row
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

    # Save to file
    output_dir = Path(__file__).parent.parent / 'paper'
    with open(output_dir / 'garch_table.tex', 'w') as f:
        f.write(latex_str)

    print(f"\nSaved to {output_dir}/garch_table.tex")

    return latex_str


# ============================================================================
# Generate Figure
# ============================================================================

def generate_comparison_figure(results_df):
    """Generate comparison figure for paper"""
    import matplotlib.pyplot as plt

    # Set up figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Colors
    colors = {
        'Standard CP': '#d62728',      # Red
        'Vol-Scaled CP': '#2ca02c',    # Green
        'GARCH(1,1)-N': '#1f77b4',     # Blue
        'GARCH(1,1)-t': '#9467bd'      # Purple
    }

    method_order = ['Standard CP', 'GARCH(1,1)-N', 'GARCH(1,1)-t', 'Vol-Scaled CP']
    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']

    # Left panel: High-vol coverage by factor
    ax1 = axes[0]
    x = np.arange(len(factors))
    width = 0.2

    for i, method in enumerate(method_order):
        method_data = results_df[results_df['method'] == method]
        values = [method_data[method_data['factor'] == f]['high_vol_coverage'].values[0]
                  for f in factors if len(method_data[method_data['factor'] == f]) > 0]

        if len(values) == len(factors):
            offset = (i - 1.5) * width
            bars = ax1.bar(x + offset, values, width, label=method, color=colors[method], alpha=0.8)

    ax1.axhline(y=0.90, color='black', linestyle='--', linewidth=1.5, label='90% Target')
    ax1.set_xlabel('Factor', fontsize=11)
    ax1.set_ylabel('High-Volatility Coverage', fontsize=11)
    ax1.set_title('(a) Coverage During High-Volatility Periods', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(factors)
    ax1.set_ylim(0.6, 1.0)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Right panel: Average coverage comparison
    ax2 = axes[1]

    summary = results_df.groupby('method').agg({
        'high_vol_coverage': 'mean',
        'low_vol_coverage': 'mean',
        'overall_coverage': 'mean'
    })
    summary = summary.reindex([m for m in method_order if m in summary.index])

    x = np.arange(len(summary))
    width = 0.25

    ax2.bar(x - width, summary['low_vol_coverage'], width, label='Low Vol', color='lightblue', edgecolor='blue')
    ax2.bar(x, summary['overall_coverage'], width, label='Overall', color='lightgray', edgecolor='gray')
    ax2.bar(x + width, summary['high_vol_coverage'], width, label='High Vol', color='salmon', edgecolor='red')

    ax2.axhline(y=0.90, color='black', linestyle='--', linewidth=1.5, label='90% Target')
    ax2.set_xlabel('Method', fontsize=11)
    ax2.set_ylabel('Coverage', fontsize=11)
    ax2.set_title('(b) Coverage by Volatility Regime (Average)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace('(1,1)', '\n(1,1)') for m in summary.index], fontsize=9)
    ax2.set_ylim(0.6, 1.0)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent.parent / 'paper' / 'figures'
    output_dir.mkdir(exist_ok=True)

    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig_garch_comparison.{ext}',
                    dpi=300, bbox_inches='tight')

    print(f"\nFigure saved to {output_dir}/fig_garch_comparison.[pdf|png]")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    results_df, summary = run_garch_comparison()
    generate_latex_table(results_df)
    generate_comparison_figure(results_df)
