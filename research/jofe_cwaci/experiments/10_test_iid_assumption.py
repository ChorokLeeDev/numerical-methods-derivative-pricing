"""
Test i.i.d. Assumption on Standardized Residuals
=================================================
Tests whether the standardized residuals ε_t = (Y_t - μ) / σ_t are i.i.d.,
which is required for Assumption 1 (Multiplicative Heteroskedasticity).

Tests performed:
1. Ljung-Box test for autocorrelation in ε_t
2. Ljung-Box test for autocorrelation in ε_t² (ARCH effects)
3. Engle's ARCH-LM test for remaining heteroskedasticity
4. Jarque-Bera test for normality (informational)
5. Runs test for independence

If standardized residuals are approximately i.i.d., the theoretical
guarantees in Section 5 apply.

Author: Chorok Lee (KAIST)
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera


def load_ff_data():
    """Load Fama-French factor data"""
    data_path = Path(__file__).parent.parent / 'data' / 'ff_factors.csv'
    ff = pd.read_csv(data_path, index_col=0, parse_dates=True)
    return ff


def compute_standardized_residuals(returns, vol_window=12):
    """
    Compute standardized residuals: ε_t = (Y_t - μ) / σ_t

    Parameters
    ----------
    returns : pd.Series
        Return series
    vol_window : int
        Rolling window for volatility estimation

    Returns
    -------
    standardized : pd.Series
        Standardized residuals
    """
    # Estimate mean (expanding window for consistency with CP calibration)
    mu = returns.expanding().mean()

    # Estimate volatility (rolling window)
    sigma = returns.rolling(vol_window).std()

    # Standardize
    standardized = (returns - mu) / sigma

    return standardized.dropna()


def runs_test(x):
    """
    Wald-Wolfowitz runs test for randomness/independence.

    Tests whether the sequence of positive/negative values is random.
    """
    # Convert to binary (above/below median)
    median = np.median(x)
    binary = (x > median).astype(int)

    # Count runs
    runs = 1
    for i in range(1, len(binary)):
        if binary[i] != binary[i-1]:
            runs += 1

    # Expected runs under independence
    n1 = np.sum(binary)
    n2 = len(binary) - n1
    n = len(binary)

    expected_runs = (2 * n1 * n2) / n + 1
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1))

    if var_runs > 0:
        z_stat = (runs - expected_runs) / np.sqrt(var_runs)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        z_stat = np.nan
        p_value = np.nan

    return {
        'runs': runs,
        'expected_runs': expected_runs,
        'z_stat': z_stat,
        'p_value': p_value
    }


def test_iid_assumption(returns, factor_name, vol_window=12):
    """
    Comprehensive test of i.i.d. assumption for standardized residuals.

    Returns dict with test results.
    """
    # Compute standardized residuals
    std_resid = compute_standardized_residuals(returns, vol_window)

    results = {
        'factor': factor_name,
        'n_obs': len(std_resid),
        'mean': std_resid.mean(),
        'std': std_resid.std(),
        'skewness': stats.skew(std_resid),
        'kurtosis': stats.kurtosis(std_resid)
    }

    # 1. Ljung-Box test for autocorrelation in levels
    # H0: No autocorrelation up to lag k
    lb_results = acorr_ljungbox(std_resid, lags=[1, 6, 12], return_df=True)
    results['lb_lag1_stat'] = lb_results.loc[1, 'lb_stat']
    results['lb_lag1_pval'] = lb_results.loc[1, 'lb_pvalue']
    results['lb_lag12_stat'] = lb_results.loc[12, 'lb_stat']
    results['lb_lag12_pval'] = lb_results.loc[12, 'lb_pvalue']

    # 2. Ljung-Box test for autocorrelation in squared residuals (ARCH effects)
    std_resid_sq = std_resid ** 2
    lb_sq_results = acorr_ljungbox(std_resid_sq, lags=[1, 6, 12], return_df=True)
    results['lb_sq_lag1_stat'] = lb_sq_results.loc[1, 'lb_stat']
    results['lb_sq_lag1_pval'] = lb_sq_results.loc[1, 'lb_pvalue']
    results['lb_sq_lag12_stat'] = lb_sq_results.loc[12, 'lb_stat']
    results['lb_sq_lag12_pval'] = lb_sq_results.loc[12, 'lb_pvalue']

    # 3. Engle's ARCH-LM test
    # H0: No ARCH effects (homoskedasticity in standardized residuals)
    try:
        arch_lm = het_arch(std_resid, nlags=12)
        results['arch_lm_stat'] = arch_lm[0]
        results['arch_lm_pval'] = arch_lm[1]
    except:
        results['arch_lm_stat'] = np.nan
        results['arch_lm_pval'] = np.nan

    # 4. Jarque-Bera test for normality
    jb_stat, jb_pval, jb_skew, jb_kurt = jarque_bera(std_resid)
    results['jb_stat'] = jb_stat
    results['jb_pval'] = jb_pval

    # 5. Runs test for independence
    runs = runs_test(std_resid.values)
    results['runs_stat'] = runs['z_stat']
    results['runs_pval'] = runs['p_value']

    # 6. First-order autocorrelation coefficient
    results['acf1'] = std_resid.autocorr(lag=1)
    results['acf1_se'] = 1 / np.sqrt(len(std_resid))

    return results, std_resid


def interpret_results(results_df):
    """Provide interpretation of test results"""

    print("\n" + "=" * 70)
    print("INTERPRETATION OF I.I.D. ASSUMPTION TESTS")
    print("=" * 70)

    # Significance level
    alpha = 0.05

    # Check autocorrelation in levels
    n_reject_lb = (results_df['lb_lag12_pval'] < alpha).sum()
    print(f"\n1. AUTOCORRELATION IN LEVELS (Ljung-Box, lag 12)")
    print(f"   H0: No autocorrelation in standardized residuals")
    print(f"   Factors rejecting H0 at 5%: {n_reject_lb}/6")
    if n_reject_lb == 0:
        print("   ✓ PASS: No significant autocorrelation detected")
    else:
        reject_factors = results_df[results_df['lb_lag12_pval'] < alpha]['factor'].tolist()
        print(f"   ⚠ CAUTION: Autocorrelation detected in {reject_factors}")

    # Check ARCH effects (autocorrelation in squared residuals)
    n_reject_arch = (results_df['lb_sq_lag12_pval'] < alpha).sum()
    print(f"\n2. ARCH EFFECTS (Ljung-Box on squared residuals, lag 12)")
    print(f"   H0: No autocorrelation in squared standardized residuals")
    print(f"   Factors rejecting H0 at 5%: {n_reject_arch}/6")
    if n_reject_arch == 0:
        print("   ✓ PASS: No remaining ARCH effects after volatility scaling")
    else:
        reject_factors = results_df[results_df['lb_sq_lag12_pval'] < alpha]['factor'].tolist()
        print(f"   ⚠ CAUTION: Remaining ARCH effects in {reject_factors}")

    # Check ARCH-LM test
    n_reject_archlm = (results_df['arch_lm_pval'] < alpha).sum()
    print(f"\n3. ENGLE'S ARCH-LM TEST")
    print(f"   H0: Homoskedasticity in standardized residuals")
    print(f"   Factors rejecting H0 at 5%: {n_reject_archlm}/6")
    if n_reject_archlm == 0:
        print("   ✓ PASS: Volatility scaling successfully removes heteroskedasticity")
    else:
        reject_factors = results_df[results_df['arch_lm_pval'] < alpha]['factor'].tolist()
        print(f"   ⚠ CAUTION: Remaining heteroskedasticity in {reject_factors}")

    # Runs test
    n_reject_runs = (results_df['runs_pval'] < alpha).sum()
    print(f"\n4. RUNS TEST FOR INDEPENDENCE")
    print(f"   H0: Sequence is random (independent)")
    print(f"   Factors rejecting H0 at 5%: {n_reject_runs}/6")
    if n_reject_runs == 0:
        print("   ✓ PASS: Standardized residuals appear independent")
    else:
        reject_factors = results_df[results_df['runs_pval'] < alpha]['factor'].tolist()
        print(f"   ⚠ CAUTION: Non-random patterns in {reject_factors}")

    # Summary
    print("\n" + "=" * 70)
    print("OVERALL ASSESSMENT")
    print("=" * 70)

    total_rejections = n_reject_lb + n_reject_arch + n_reject_archlm + n_reject_runs
    max_rejections = 4 * 6  # 4 tests × 6 factors

    print(f"\nTotal rejections: {total_rejections}/{max_rejections} ({100*total_rejections/max_rejections:.0f}%)")

    if total_rejections <= 2:
        print("\n✓ STRONG SUPPORT for i.i.d. assumption")
        print("  Standardized residuals are approximately i.i.d.")
        print("  Theoretical guarantees (Theorems 1-3) apply.")
    elif total_rejections <= 6:
        print("\n~ MODERATE SUPPORT for i.i.d. assumption")
        print("  Some deviations from i.i.d., but volatility scaling")
        print("  substantially reduces dependence structure.")
        print("  Theoretical guarantees approximately hold.")
    else:
        print("\n✗ WEAK SUPPORT for i.i.d. assumption")
        print("  Significant deviations from i.i.d. detected.")
        print("  Theoretical guarantees may not hold exactly.")


def generate_latex_table(results_df):
    """Generate LaTeX table for paper"""

    print("\n" + "=" * 70)
    print("LaTeX Table: I.I.D. Assumption Tests")
    print("=" * 70)

    latex = []
    latex.append(r"\begin{table}[H]")
    latex.append(r"\centering")
    latex.append(r"\caption{Tests of I.I.D. Assumption on Standardized Residuals}")
    latex.append(r"\label{tab:iid_tests}")
    latex.append(r"\begin{tabular}{lcccccc}")
    latex.append(r"\toprule")
    latex.append(r"& \multicolumn{2}{c}{Autocorrelation} & \multicolumn{2}{c}{ARCH Effects} & \multicolumn{2}{c}{Independence} \\")
    latex.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    latex.append(r"Factor & LB(12) & $p$-value & LB$^2$(12) & $p$-value & Runs & $p$-value \\")
    latex.append(r"\midrule")

    for _, row in results_df.iterrows():
        factor = row['factor']
        lb = row['lb_lag12_stat']
        lb_p = row['lb_lag12_pval']
        lb_sq = row['lb_sq_lag12_stat']
        lb_sq_p = row['lb_sq_lag12_pval']
        runs = row['runs_stat']
        runs_p = row['runs_pval']

        # Bold if significant at 5%
        lb_p_str = f"\\textbf{{{lb_p:.3f}}}" if lb_p < 0.05 else f"{lb_p:.3f}"
        lb_sq_p_str = f"\\textbf{{{lb_sq_p:.3f}}}" if lb_sq_p < 0.05 else f"{lb_sq_p:.3f}"
        runs_p_str = f"\\textbf{{{runs_p:.3f}}}" if runs_p < 0.05 else f"{runs_p:.3f}"

        latex.append(f"{factor} & {lb:.1f} & {lb_p_str} & {lb_sq:.1f} & {lb_sq_p_str} & {runs:.2f} & {runs_p_str} \\\\")

    latex.append(r"\midrule")

    # Count rejections
    n_lb = (results_df['lb_lag12_pval'] < 0.05).sum()
    n_sq = (results_df['lb_sq_lag12_pval'] < 0.05).sum()
    n_runs = (results_df['runs_pval'] < 0.05).sum()

    latex.append(f"Reject at 5\\% & \\multicolumn{{2}}{{c}}{{{n_lb}/6}} & \\multicolumn{{2}}{{c}}{{{n_sq}/6}} & \\multicolumn{{2}}{{c}}{{{n_runs}/6}} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\begin{tablenotes}")
    latex.append(r"\small")
    latex.append(r"\item Notes: LB(12) is Ljung-Box statistic at lag 12 testing for autocorrelation.")
    latex.append(r"LB$^2$(12) tests for ARCH effects (autocorrelation in squared residuals).")
    latex.append(r"Runs test examines independence of sign sequences.")
    latex.append(r"Bold $p$-values indicate rejection at 5\% level.")
    latex.append(r"Standardized residuals: $\hat{\epsilon}_t = (r_t - \bar{r}) / \hat{\sigma}_t$.")
    latex.append(r"\end{tablenotes}")
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)
    print(latex_str)

    # Save
    output_dir = Path(__file__).parent.parent / 'paper'
    with open(output_dir / 'iid_tests_table.tex', 'w') as f:
        f.write(latex_str)

    print(f"\nSaved to {output_dir}/iid_tests_table.tex")

    return latex_str


def main():
    print("=" * 70)
    print("Testing I.I.D. Assumption on Standardized Residuals")
    print("=" * 70)

    # Load data
    ff = load_ff_data()
    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']

    print(f"\nData: {ff.index[0]} to {ff.index[-1]} ({len(ff)} months)")
    print(f"Volatility window: 12 months (trailing)")

    all_results = []

    for factor in factors:
        print(f"\n--- {factor} ---")
        returns = ff[factor].dropna()

        results, std_resid = test_iid_assumption(returns, factor)
        all_results.append(results)

        print(f"  N = {results['n_obs']}")
        print(f"  Mean: {results['mean']:.4f}, Std: {results['std']:.4f}")
        print(f"  ACF(1): {results['acf1']:.3f} (SE: {results['acf1_se']:.3f})")
        print(f"  Ljung-Box(12): stat={results['lb_lag12_stat']:.1f}, p={results['lb_lag12_pval']:.3f}")
        print(f"  LB-Squared(12): stat={results['lb_sq_lag12_stat']:.1f}, p={results['lb_sq_lag12_pval']:.3f}")
        print(f"  ARCH-LM: stat={results['arch_lm_stat']:.1f}, p={results['arch_lm_pval']:.3f}")
        print(f"  Runs test: z={results['runs_stat']:.2f}, p={results['runs_pval']:.3f}")

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Detailed results table
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)

    display_cols = ['factor', 'acf1', 'lb_lag12_pval', 'lb_sq_lag12_pval',
                    'arch_lm_pval', 'runs_pval']
    print("\n" + results_df[display_cols].to_string(index=False))

    # Interpretation
    interpret_results(results_df)

    # Generate LaTeX table
    generate_latex_table(results_df)

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    results_df.to_csv(output_dir / 'iid_assumption_tests.csv', index=False)
    print(f"\nResults saved to {output_dir}/iid_assumption_tests.csv")

    return results_df


if __name__ == "__main__":
    results_df = main()
