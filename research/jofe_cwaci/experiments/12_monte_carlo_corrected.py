"""
Corrected Monte Carlo Validation
================================
Validates theoretical results under controlled conditions where
Assumption 1 (multiplicative heteroskedasticity) holds exactly.

Author: Chorok Lee (KAIST)
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)


def run_simulation(n=500, gamma=0.5, sigma_base=0.04, alpha=0.1, cal_fraction=0.5):
    """
    Run single Monte Carlo simulation.

    Parameters
    ----------
    n : int
        Number of observations
    gamma : float
        Volatility dispersion parameter
    sigma_base : float
        Base volatility
    alpha : float
        Miscoverage rate
    cal_fraction : float
        Fraction for calibration

    Returns
    -------
    dict with coverage results
    """
    # Generate volatility: σ_t = σ_base * exp(γ * z_t)
    z = np.random.randn(n)
    sigma = sigma_base * np.exp(gamma * z)

    # Generate returns: Y_t = σ_t * ε_t
    epsilon = np.random.randn(n)
    Y = sigma * epsilon

    # Split
    cal_end = int(n * cal_fraction)
    Y_cal, Y_test = Y[:cal_end], Y[cal_end:]
    sigma_cal, sigma_test = sigma[:cal_end], sigma[cal_end:]

    # Point prediction (mean)
    pred = np.mean(Y_cal)

    # Standard CP
    scores_std = np.abs(Y_cal - pred)
    n_cal = len(scores_std)
    q_level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    q_std = np.quantile(scores_std, q_level)

    covered_std = np.abs(Y_test - pred) <= q_std

    # Volatility-scaled CP (oracle: true σ known)
    scores_vs = np.abs(Y_cal - pred) / sigma_cal
    q_vs = np.quantile(scores_vs, q_level)

    covered_vs = np.abs(Y_test - pred) <= q_vs * sigma_test

    # High/low volatility in test period
    vol_median = np.median(sigma_test)
    high_vol = sigma_test > vol_median
    low_vol = ~high_vol

    return {
        'std_overall': np.mean(covered_std),
        'std_high_vol': np.mean(covered_std[high_vol]),
        'std_low_vol': np.mean(covered_std[low_vol]),
        'vs_overall': np.mean(covered_vs),
        'vs_high_vol': np.mean(covered_vs[high_vol]),
        'vs_low_vol': np.mean(covered_vs[low_vol]),
        'vol_ratio': np.mean(sigma_test[high_vol]) / np.mean(sigma_test[low_vol])
    }


def run_monte_carlo(n_sims=500, gammas=[0.0, 0.25, 0.5, 0.75, 1.0]):
    """Run full Monte Carlo study"""

    print("=" * 70)
    print("Monte Carlo Validation (Corrected)")
    print("=" * 70)
    print(f"\nSimulations per gamma: {n_sims}")
    print(f"Sample size: 500, Cal fraction: 50%")

    results = []

    for gamma in gammas:
        print(f"\nRunning gamma = {gamma}...")

        sim_results = []
        for _ in range(n_sims):
            res = run_simulation(gamma=gamma)
            sim_results.append(res)

        # Aggregate
        df = pd.DataFrame(sim_results)

        results.append({
            'gamma': gamma,
            'std_high_vol_mean': df['std_high_vol'].mean(),
            'std_high_vol_se': df['std_high_vol'].std() / np.sqrt(n_sims),
            'vs_high_vol_mean': df['vs_high_vol'].mean(),
            'vs_high_vol_se': df['vs_high_vol'].std() / np.sqrt(n_sims),
            'vol_ratio': df['vol_ratio'].mean()
        })

        print(f"  Standard CP high-vol: {df['std_high_vol'].mean():.1%} (SE: {df['std_high_vol'].std()/np.sqrt(n_sims):.1%})")
        print(f"  Vol-Scaled CP high-vol: {df['vs_high_vol'].mean():.1%} (SE: {df['vs_high_vol'].std()/np.sqrt(n_sims):.1%})")
        print(f"  Vol ratio (high/low): {df['vol_ratio'].mean():.2f}x")

    results_df = pd.DataFrame(results)

    # Print table
    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    print("\ngamma | Standard CP | Vol-Scaled CP | Vol Ratio")
    print("-" * 55)
    for _, row in results_df.iterrows():
        print(f"{row['gamma']:.2f}  | {row['std_high_vol_mean']:.1%} ({row['std_high_vol_se']:.1%}) | {row['vs_high_vol_mean']:.1%} ({row['vs_high_vol_se']:.1%}) | {row['vol_ratio']:.2f}x")

    # Save
    output_dir = Path(__file__).parent.parent / 'results'
    results_df.to_csv(output_dir / 'monte_carlo_corrected.csv', index=False)
    print(f"\nSaved to {output_dir}/monte_carlo_corrected.csv")

    return results_df


if __name__ == "__main__":
    results = run_monte_carlo()
