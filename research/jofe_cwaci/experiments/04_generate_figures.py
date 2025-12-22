"""
Generate Publication-Quality Figures

Figures for JoFE paper:
1. Coverage comparison by factor (main result)
2. Interval width adaptation
3. Monte Carlo sensitivity analysis
4. Robustness summary

Author: Chorok Lee (KAIST)
Date: December 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Style settings for publication
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def fig1_coverage_comparison(results_dir, figures_dir):
    """Figure 1: Coverage by factor and crowding regime."""
    df = pd.read_csv(results_dir / 'coverage_analysis.csv')

    factors = df['factor'].tolist()
    x = np.arange(len(factors))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    # High crowding coverage
    bars1 = ax.bar(x - width/2, df['scp_high'] * 100, width,
                   label='Standard CP', color='#E74C3C', alpha=0.8)
    bars2 = ax.bar(x + width/2, df['cwaci_high'] * 100, width,
                   label='CW-ACI', color='#3498DB', alpha=0.8)

    # Reference line at 90%
    ax.axhline(y=90, color='gray', linestyle='--', linewidth=1.5, label='Target (90%)')

    ax.set_ylabel('Coverage (%)')
    ax.set_xlabel('Factor')
    ax.set_title('High-Crowding Coverage: Standard CP vs CW-ACI')
    ax.set_xticks(x)
    ax.set_xticklabels(factors)
    ax.set_ylim(60, 105)
    ax.legend(loc='lower right')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(figures_dir / 'fig1_coverage_comparison.pdf')
    plt.savefig(figures_dir / 'fig1_coverage_comparison.png')
    plt.close()
    print("  Created: fig1_coverage_comparison.pdf")


def fig2_width_adaptation(results_dir, figures_dir):
    """Figure 2: Interval width adaptation by crowding level."""
    df = pd.read_csv(results_dir / 'coverage_analysis.csv')

    factors = df['factor'].tolist()
    x = np.arange(len(factors))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    # Calculate CW-ACI widths from the data
    # We need to compute these from the raw experiment
    # For now, use the width_ratio which tells us high/low ratio

    # Re-run to get actual widths
    sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from download_data import load_or_download_factors
    from conformal import StandardConformalPredictor, CrowdingWeightedACI
    from crowding import compute_crowding_proxy

    factors_data = load_or_download_factors()

    widths_data = []
    for factor in factors:
        returns = factors_data[factor]
        crowding = compute_crowding_proxy(returns, window=12)

        valid = returns.notna() & crowding.notna()
        returns_arr = returns[valid].values
        crowding_arr = crowding[valid].values

        n = len(returns_arr)
        cal_end = int(n * 0.5)

        y_cal, y_test = returns_arr[:cal_end], returns_arr[cal_end:]
        crowd_cal, crowd_test = crowding_arr[:cal_end], crowding_arr[cal_end:]
        pred_cal = np.full_like(y_cal, np.mean(y_cal))
        pred_test = np.full_like(y_test, np.mean(y_cal))

        high_crowding = crowd_test > np.median(crowd_test)

        # Standard CP width
        scp = StandardConformalPredictor(alpha=0.1)
        scp.fit(y_cal, pred_cal)
        scp_width = scp.get_width()

        # CW-ACI widths
        cwaci = CrowdingWeightedACI(alpha=0.1, sensitivity=1.0)
        cwaci.fit(y_cal, pred_cal, crowd_cal)
        _, _, cw_widths = cwaci.predict(pred_test, crowd_test)

        widths_data.append({
            'factor': factor,
            'scp': scp_width,
            'cwaci_high': np.mean(cw_widths[high_crowding]),
            'cwaci_low': np.mean(cw_widths[~high_crowding])
        })

    df_widths = pd.DataFrame(widths_data)

    # Plot
    x = np.arange(len(factors))
    width = 0.25

    bars1 = ax.bar(x - width, df_widths['scp'] * 100, width,
                   label='Standard CP', color='#95A5A6', alpha=0.8)
    bars2 = ax.bar(x, df_widths['cwaci_low'] * 100, width,
                   label='CW-ACI (Low Crowding)', color='#27AE60', alpha=0.8)
    bars3 = ax.bar(x + width, df_widths['cwaci_high'] * 100, width,
                   label='CW-ACI (High Crowding)', color='#E74C3C', alpha=0.8)

    ax.set_ylabel('Interval Width (%)')
    ax.set_xlabel('Factor')
    ax.set_title('Prediction Interval Width by Method and Crowding Regime')
    ax.set_xticks(x)
    ax.set_xticklabels(factors)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(figures_dir / 'fig2_width_adaptation.pdf')
    plt.savefig(figures_dir / 'fig2_width_adaptation.png')
    plt.close()
    print("  Created: fig2_width_adaptation.pdf")


def fig3_monte_carlo_sensitivity(results_dir, figures_dir):
    """Figure 3: Monte Carlo sensitivity to crowding effect strength."""
    df = pd.read_csv(results_dir / 'monte_carlo_effects.csv')

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df['crowding_effect'], df['scp_high_mean'] * 100, 'o-',
            color='#E74C3C', linewidth=2, markersize=8, label='Standard CP')
    ax.plot(df['crowding_effect'], df['cwaci_high_mean'] * 100, 's-',
            color='#3498DB', linewidth=2, markersize=8, label='CW-ACI')

    ax.axhline(y=90, color='gray', linestyle='--', linewidth=1.5, label='Target (90%)')

    ax.set_xlabel('Crowding Effect Strength (δ)')
    ax.set_ylabel('High-Crowding Coverage (%)')
    ax.set_title('Monte Carlo: Coverage vs Crowding Effect Strength')
    ax.set_ylim(75, 102)
    ax.legend(loc='lower left')

    # Add shaded region for gap
    ax.fill_between(df['crowding_effect'],
                    df['scp_high_mean'] * 100,
                    df['cwaci_high_mean'] * 100,
                    alpha=0.2, color='green')

    plt.tight_layout()
    plt.savefig(figures_dir / 'fig3_monte_carlo_sensitivity.pdf')
    plt.savefig(figures_dir / 'fig3_monte_carlo_sensitivity.png')
    plt.close()
    print("  Created: fig3_monte_carlo_sensitivity.pdf")


def fig4_robustness_summary(results_dir, figures_dir):
    """Figure 4: Summary of robustness checks."""
    # Compute average gains from robustness results
    robustness_data = []

    # 1. Alternative proxies
    df_prox = pd.read_csv(results_dir / 'robustness_proxies.csv')
    for proxy in df_prox['proxy'].unique():
        subset = df_prox[df_prox['proxy'] == proxy]
        gain = subset['gain_high'].mean() * 100
        robustness_data.append({'check': f'Proxy: {proxy}', 'gain': gain})

    # 2. Subperiods
    df_sub = pd.read_csv(results_dir / 'robustness_subperiods.csv')
    for period in ['early', 'late']:
        subset = df_sub[df_sub['period'] == period]
        gain = subset['gain_high'].mean() * 100
        label = '1963-1993' if period == 'early' else '1994-2025'
        robustness_data.append({'check': f'Period: {label}', 'gain': gain})

    # 3. Sensitivity parameter
    df_sens = pd.read_csv(results_dir / 'robustness_sensitivity.csv')
    for gamma in [0.5, 1.5, 2.0]:
        subset = df_sens[df_sens['gamma'] == gamma]
        gain = subset['gain_high'].mean() * 100
        robustness_data.append({'check': f'γ = {gamma}', 'gain': gain})

    df_rob = pd.DataFrame(robustness_data)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#3498DB' if 'baseline' in c.lower() or 'γ = 1' in c else '#95A5A6'
              for c in df_rob['check']]

    bars = ax.barh(df_rob['check'], df_rob['gain'], color=colors, alpha=0.8)

    ax.axvline(x=15.0, color='#E74C3C', linestyle='--', linewidth=2,
               label='Baseline (15.0pp)')

    ax.set_xlabel('High-Crowding Coverage Improvement (pp)')
    ax.set_title('Robustness: CW-ACI Improvement Across Specifications')
    ax.set_xlim(0, 25)

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.1f}pp',
                    xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(3, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9)

    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(figures_dir / 'fig4_robustness_summary.pdf')
    plt.savefig(figures_dir / 'fig4_robustness_summary.png')
    plt.close()
    print("  Created: fig4_robustness_summary.pdf")


def fig5_coverage_gap(results_dir, figures_dir):
    """Figure 5: Coverage gap (high vs low crowding) before and after CW-ACI."""
    df = pd.read_csv(results_dir / 'coverage_analysis.csv')

    factors = df['factor'].tolist()

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(factors))
    width = 0.35

    gap_scp = (df['scp_low'] - df['scp_high']) * 100
    gap_cwaci = (df['cwaci_low'] - df['cwaci_high']) * 100

    bars1 = ax.bar(x - width/2, gap_scp, width,
                   label='Standard CP', color='#E74C3C', alpha=0.8)
    bars2 = ax.bar(x + width/2, gap_cwaci, width,
                   label='CW-ACI', color='#3498DB', alpha=0.8)

    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    ax.set_ylabel('Coverage Gap: Low - High (pp)')
    ax.set_xlabel('Factor')
    ax.set_title('Coverage Disparity Between Low and High Crowding Periods')
    ax.set_xticks(x)
    ax.set_xticklabels(factors)
    ax.legend(loc='upper right')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(figures_dir / 'fig5_coverage_gap.pdf')
    plt.savefig(figures_dir / 'fig5_coverage_gap.png')
    plt.close()
    print("  Created: fig5_coverage_gap.pdf")


def main():
    """Generate all figures."""
    print("="*60)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("="*60)

    results_dir = Path(__file__).parent.parent / 'results'
    figures_dir = Path(__file__).parent.parent / 'paper' / 'figures'
    figures_dir.mkdir(exist_ok=True)

    print(f"\nResults directory: {results_dir}")
    print(f"Figures directory: {figures_dir}")

    print("\nGenerating figures...")
    fig1_coverage_comparison(results_dir, figures_dir)
    fig2_width_adaptation(results_dir, figures_dir)
    fig3_monte_carlo_sensitivity(results_dir, figures_dir)
    fig4_robustness_summary(results_dir, figures_dir)
    fig5_coverage_gap(results_dir, figures_dir)

    print("\n" + "="*60)
    print("COMPLETE: All figures saved to paper/figures/")
    print("="*60)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(figures_dir.glob('*.pdf')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
