"""
Generate Publication Figures for Volatility-Adaptive Conformal Prediction Paper

Creates clean, publication-ready figures for:
1. Main coverage comparison by volatility regime
2. Subperiod analysis (regime-change evidence)
3. Width adaptation visualization
4. Monte Carlo sensitivity analysis

Author: Chorok Lee (KAIST)
Date: December 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

# Style settings for publication
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Color palette (colorblind-friendly)
COLORS = {
    'standard': '#E74C3C',      # Red
    'vol_scaling': '#2ECC71',   # Green
    'saci': '#3498DB',          # Blue
    'target': '#7F8C8D',        # Gray
    'high_vol': '#E74C3C',      # Red
    'low_vol': '#3498DB',       # Blue
}


def load_results():
    """Load all results from CSV files."""
    results_dir = Path(__file__).parent.parent / 'results'

    main_df = pd.read_csv(results_dir / 'main_coverage_results.csv')
    subperiod_df = pd.read_csv(results_dir / 'subperiod_analysis.csv')
    mc_df = pd.read_csv(results_dir / 'monte_carlo_results.csv')

    return main_df, subperiod_df, mc_df


def figure1_coverage_comparison(main_df, output_dir):
    """
    Figure 1: Main coverage comparison by method and factor.

    Bar chart showing high-volatility coverage for each factor,
    comparing Standard CP, Vol-Scaled, and Locally-Weighted.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
    x = np.arange(len(factors))
    width = 0.25

    # Extract data
    std_cov = []
    vs_cov = []
    lw_cov = []
    std_se = []
    vs_se = []
    lw_se = []

    for factor in factors:
        std_row = main_df[(main_df['factor'] == factor) & (main_df['method'] == 'standard')]
        vs_row = main_df[(main_df['factor'] == factor) & (main_df['method'] == 'vol_scaling')]
        lw_row = main_df[(main_df['factor'] == factor) & (main_df['method'] == 'saci')]

        std_cov.append(std_row['coverage_high_vol'].values[0] * 100 if len(std_row) > 0 else 0)
        vs_cov.append(vs_row['coverage_high_vol'].values[0] * 100 if len(vs_row) > 0 else 0)
        lw_cov.append(lw_row['coverage_high_vol'].values[0] * 100 if len(lw_row) > 0 else 0)

        std_se.append(std_row['se_high_vol'].values[0] * 100 if len(std_row) > 0 else 0)
        vs_se.append(vs_row['se_high_vol'].values[0] * 100 if len(vs_row) > 0 else 0)
        lw_se.append(lw_row['se_high_vol'].values[0] * 100 if len(lw_row) > 0 else 0)

    # Plot bars with error bars
    bars1 = ax.bar(x - width, std_cov, width, label='Standard CP',
                   color=COLORS['standard'], alpha=0.8, yerr=std_se, capsize=3)
    bars2 = ax.bar(x, vs_cov, width, label='Vol-Scaled',
                   color=COLORS['vol_scaling'], alpha=0.8, yerr=vs_se, capsize=3)
    bars3 = ax.bar(x + width, lw_cov, width, label='Locally-Weighted',
                   color=COLORS['saci'], alpha=0.8, yerr=lw_se, capsize=3)

    # Target line
    ax.axhline(y=90, color=COLORS['target'], linestyle='--', linewidth=2,
               label='90% Target', zorder=0)

    # Formatting
    ax.set_xlabel('Factor')
    ax.set_ylabel('High-Volatility Coverage (%)')
    ax.set_title('Coverage During High-Volatility Periods by Method')
    ax.set_xticks(x)
    ax.set_xticklabels(factors)
    ax.set_ylim(55, 100)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    # Add average annotation
    avg_std = np.mean(std_cov)
    avg_vs = np.mean(vs_cov)
    ax.annotate(f'Avg: {avg_std:.1f}%', xy=(5.3, avg_std), fontsize=9, color=COLORS['standard'])
    ax.annotate(f'Avg: {avg_vs:.1f}%', xy=(5.3, avg_vs), fontsize=9, color=COLORS['vol_scaling'])

    plt.tight_layout()
    fig.savefig(output_dir / 'fig1_coverage_comparison.pdf')
    fig.savefig(output_dir / 'fig1_coverage_comparison.png')
    plt.close()

    print("Figure 1: Coverage comparison saved")


def figure2_subperiod_analysis(subperiod_df, output_dir):
    """
    Figure 2: Subperiod analysis showing regime-change effect.

    Grouped bar chart showing within-period vs cross-period coverage.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
    x = np.arange(len(factors))
    width = 0.25

    # Extract standard CP coverage for each period
    period1_cov = []
    period2_cov = []
    full_cov = []

    for factor in factors:
        std_df = subperiod_df[(subperiod_df['factor'] == factor) &
                              (subperiod_df['method'] == 'standard')]

        p1 = std_df[std_df['period'] == '1963-1993']['coverage_high_vol'].values
        p2 = std_df[std_df['period'] == '1994-2025']['coverage_high_vol'].values
        full = std_df[std_df['period'] == 'Full Sample']['coverage_high_vol'].values

        period1_cov.append(p1[0] * 100 if len(p1) > 0 else 0)
        period2_cov.append(p2[0] * 100 if len(p2) > 0 else 0)
        full_cov.append(full[0] * 100 if len(full) > 0 else 0)

    # Plot bars
    bars1 = ax.bar(x - width, period1_cov, width, label='Within 1963-1993',
                   color='#3498DB', alpha=0.8)
    bars2 = ax.bar(x, period2_cov, width, label='Within 1994-2025',
                   color='#9B59B6', alpha=0.8)
    bars3 = ax.bar(x + width, full_cov, width, label='Full Sample (Cross-Period)',
                   color='#E74C3C', alpha=0.8)

    # Target line
    ax.axhline(y=90, color=COLORS['target'], linestyle='--', linewidth=2,
               label='90% Target', zorder=0)

    # Formatting
    ax.set_xlabel('Factor')
    ax.set_ylabel('High-Volatility Coverage (%)')
    ax.set_title('Standard CP Coverage: Within-Period vs Cross-Period\n(Evidence for Regime-Change Interpretation)')
    ax.set_xticks(x)
    ax.set_xticklabels(factors)
    ax.set_ylim(55, 100)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    # Add annotation
    ax.annotate('Under-coverage appears\nonly in cross-period analysis',
                xy=(4.5, 68), fontsize=9, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig.savefig(output_dir / 'fig2_subperiod_analysis.pdf')
    fig.savefig(output_dir / 'fig2_subperiod_analysis.png')
    plt.close()

    print("Figure 2: Subperiod analysis saved")


def figure3_width_adaptation(main_df, output_dir):
    """
    Figure 3: Interval width adaptation by volatility regime.

    Shows how vol-scaled intervals adapt to volatility.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']

    # Left panel: Width comparison
    x = np.arange(len(factors))
    width = 0.35

    std_width = []
    vs_width_high = []
    vs_width_low = []

    for factor in factors:
        std_row = main_df[(main_df['factor'] == factor) & (main_df['method'] == 'standard')]
        vs_row = main_df[(main_df['factor'] == factor) & (main_df['method'] == 'vol_scaling')]

        std_width.append(std_row['avg_width'].values[0] if len(std_row) > 0 else 0)
        vs_width_high.append(vs_row['width_high'].values[0] if len(vs_row) > 0 else 0)
        vs_width_low.append(vs_row['width_low'].values[0] if len(vs_row) > 0 else 0)

    bars1 = ax1.bar(x - width/2, std_width, width, label='Standard CP (Fixed)',
                    color=COLORS['standard'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, vs_width_high, width, label='Vol-Scaled (High-Vol)',
                    color=COLORS['high_vol'], alpha=0.5, hatch='//')

    ax1.set_xlabel('Factor')
    ax1.set_ylabel('Interval Width')
    ax1.set_title('Interval Width: Standard vs Vol-Scaled')
    ax1.set_xticks(x)
    ax1.set_xticklabels(factors)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)

    # Right panel: Width ratio (high/low)
    width_ratios = [h/l if l > 0 else 0 for h, l in zip(vs_width_high, vs_width_low)]

    colors = [COLORS['vol_scaling'] if r > 1 else COLORS['standard'] for r in width_ratios]
    bars = ax2.bar(x, width_ratios, 0.6, color=colors, alpha=0.8)

    ax2.axhline(y=1.0, color=COLORS['target'], linestyle='--', linewidth=2,
                label='No Adaptation')

    ax2.set_xlabel('Factor')
    ax2.set_ylabel('Width Ratio (High-Vol / Low-Vol)')
    ax2.set_title('Vol-Scaled Width Adaptation Ratio')
    ax2.set_xticks(x)
    ax2.set_xticklabels(factors)
    ax2.set_ylim(0, 2.5)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)

    # Add average ratio annotation
    avg_ratio = np.mean(width_ratios)
    ax2.annotate(f'Avg ratio: {avg_ratio:.2f}x', xy=(4.5, 2.2), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    fig.savefig(output_dir / 'fig3_width_adaptation.pdf')
    fig.savefig(output_dir / 'fig3_width_adaptation.png')
    plt.close()

    print("Figure 3: Width adaptation saved")


def figure4_monte_carlo(mc_df, output_dir):
    """
    Figure 4: Monte Carlo sensitivity analysis.

    Line plot showing coverage vs volatility effect strength.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Aggregate by delta and method
    summary = mc_df.groupby(['delta', 'method']).agg({
        'coverage_high_vol': ['mean', 'std']
    }).reset_index()
    summary.columns = ['delta', 'method', 'mean', 'std']

    deltas = sorted(mc_df['delta'].unique())

    for method, color, label in [
        ('standard', COLORS['standard'], 'Standard CP'),
        ('vol_scaling', COLORS['vol_scaling'], 'Vol-Scaled'),
        ('saci', COLORS['saci'], 'Locally-Weighted')
    ]:
        method_data = summary[summary['method'] == method]
        means = [method_data[method_data['delta'] == d]['mean'].values[0] * 100
                 for d in deltas]
        stds = [method_data[method_data['delta'] == d]['std'].values[0] * 100
                for d in deltas]

        ax.plot(deltas, means, 'o-', color=color, label=label, linewidth=2, markersize=8)
        ax.fill_between(deltas,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=color, alpha=0.2)

    # Target line
    ax.axhline(y=90, color=COLORS['target'], linestyle='--', linewidth=2,
               label='90% Target')

    ax.set_xlabel('Volatility Effect Strength (δ)')
    ax.set_ylabel('High-Volatility Coverage (%)')
    ax.set_title('Monte Carlo: Coverage vs Volatility Effect Strength\n(Shaded regions show ±1 std)')
    ax.set_ylim(75, 100)
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)

    # Add annotation
    ax.annotate('Standard CP degrades\nas δ increases',
                xy=(0.7, 86), fontsize=9, style='italic',
                xytext=(0.5, 80),
                arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    fig.savefig(output_dir / 'fig4_monte_carlo.pdf')
    fig.savefig(output_dir / 'fig4_monte_carlo.png')
    plt.close()

    print("Figure 4: Monte Carlo saved")


def figure5_coverage_gap(main_df, output_dir):
    """
    Figure 5: Coverage gap visualization.

    Shows the gap between target and actual coverage.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
    x = np.arange(len(factors))

    # Extract coverage gaps (target - actual)
    std_gap = []
    vs_gap = []

    for factor in factors:
        std_row = main_df[(main_df['factor'] == factor) & (main_df['method'] == 'standard')]
        vs_row = main_df[(main_df['factor'] == factor) & (main_df['method'] == 'vol_scaling')]

        std_cov = std_row['coverage_high_vol'].values[0] * 100 if len(std_row) > 0 else 0
        vs_cov = vs_row['coverage_high_vol'].values[0] * 100 if len(vs_row) > 0 else 0

        std_gap.append(90 - std_cov)  # Positive = under-coverage
        vs_gap.append(90 - vs_cov)

    width = 0.35

    # Plot bars (negative = over-coverage, positive = under-coverage)
    bars1 = ax.bar(x - width/2, std_gap, width, label='Standard CP',
                   color=COLORS['standard'], alpha=0.8)
    bars2 = ax.bar(x + width/2, vs_gap, width, label='Vol-Scaled',
                   color=COLORS['vol_scaling'], alpha=0.8)

    # Zero line (perfect coverage)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Shaded regions
    ax.axhspan(0, 30, alpha=0.1, color='red', label='Under-coverage zone')
    ax.axhspan(-10, 0, alpha=0.1, color='green', label='Over-coverage zone')

    ax.set_xlabel('Factor')
    ax.set_ylabel('Coverage Gap (90% - Actual)')
    ax.set_title('Coverage Gap from 90% Target\n(Positive = Under-coverage)')
    ax.set_xticks(x)
    ax.set_xticklabels(factors)
    ax.set_ylim(-10, 30)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Add average annotations
    avg_std_gap = np.mean(std_gap)
    avg_vs_gap = np.mean(vs_gap)
    ax.annotate(f'Avg gap: {avg_std_gap:.1f}pp', xy=(5.2, avg_std_gap + 1),
                fontsize=9, color=COLORS['standard'])
    ax.annotate(f'Avg gap: {avg_vs_gap:.1f}pp', xy=(5.2, avg_vs_gap - 2),
                fontsize=9, color=COLORS['vol_scaling'])

    plt.tight_layout()
    fig.savefig(output_dir / 'fig5_coverage_gap.pdf')
    fig.savefig(output_dir / 'fig5_coverage_gap.png')
    plt.close()

    print("Figure 5: Coverage gap saved")


def figure6_method_summary(main_df, output_dir):
    """
    Figure 6: Summary comparison across all factors.

    Horizontal bar chart showing overall performance.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # Calculate averages
    methods = ['standard', 'saci', 'vol_scaling']
    labels = ['Standard CP', 'Locally-Weighted', 'Vol-Scaled']
    colors = [COLORS['standard'], COLORS['saci'], COLORS['vol_scaling']]

    avg_high = []
    avg_overall = []

    for method in methods:
        method_df = main_df[main_df['method'] == method]
        avg_high.append(method_df['coverage_high_vol'].mean() * 100)
        avg_overall.append(method_df['coverage_overall'].mean() * 100)

    y = np.arange(len(methods))
    height = 0.35

    bars1 = ax.barh(y - height/2, avg_high, height, label='High-Volatility',
                    color=colors, alpha=0.8)
    bars2 = ax.barh(y + height/2, avg_overall, height, label='Overall',
                    color=colors, alpha=0.4, hatch='//')

    # Target line
    ax.axvline(x=90, color=COLORS['target'], linestyle='--', linewidth=2)
    ax.text(90.5, 2.5, '90% Target', fontsize=9, color=COLORS['target'])

    ax.set_xlabel('Coverage (%)')
    ax.set_ylabel('Method')
    ax.set_title('Average Coverage by Method')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(70, 100)
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (h, o) in enumerate(zip(avg_high, avg_overall)):
        ax.text(h + 0.5, i - height/2, f'{h:.1f}%', va='center', fontsize=9)
        ax.text(o + 0.5, i + height/2, f'{o:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig6_method_summary.pdf')
    fig.savefig(output_dir / 'fig6_method_summary.png')
    plt.close()

    print("Figure 6: Method summary saved")


def main():
    """Generate all figures."""
    print("="*60)
    print("Generating Publication Figures")
    print("="*60)

    # Load results
    try:
        main_df, subperiod_df, mc_df = load_results()
        print(f"Loaded results: {len(main_df)} main rows, {len(subperiod_df)} subperiod rows")
    except FileNotFoundError as e:
        print(f"Error: Results not found. Run experiments first.")
        print(f"Missing: {e}")
        return

    # Output directory
    output_dir = Path(__file__).parent.parent / 'paper' / 'figures'
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Generate figures
    print("\nGenerating figures...")

    figure1_coverage_comparison(main_df, output_dir)
    figure2_subperiod_analysis(subperiod_df, output_dir)
    figure3_width_adaptation(main_df, output_dir)
    figure4_monte_carlo(mc_df, output_dir)
    figure5_coverage_gap(main_df, output_dir)
    figure6_method_summary(main_df, output_dir)

    print("\n" + "="*60)
    print("All figures generated successfully!")
    print("="*60)

    # List files
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*.pdf')):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
