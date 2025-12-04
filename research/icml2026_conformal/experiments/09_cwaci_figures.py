"""
Experiment 09: Generate CW-ACI Figures for ICML 2026 Paper

Updated figures based on CW-ACI results (Experiments 07-08).

Figures:
1. Conditional Coverage: CW-ACI vs ACI by crowding bin
2. λ Sensitivity: Coverage by λ across crowding bins
3. Marginal Coverage Comparison (bar chart)
4. Coverage Uniformity: Variance comparison
5. λ Selection Distribution (from CV experiment)

For ICML 2026 submission.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set style for paper-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150

# Colors (updated for CW-ACI focus)
COLORS = {
    'split': '#1f77b4',      # Blue
    'aci': '#ff7f0e',        # Orange
    'cwaci': '#2ca02c',      # Green
    'target': '#7f7f7f',     # Gray
    'low': '#3498db',        # Light blue
    'medium': '#f39c12',     # Yellow
    'high': '#e74c3c',       # Red
}


def load_cwaci_results():
    """Load CW-ACI experimental results from Experiment 07."""
    results_dir = Path(__file__).parent.parent / 'results'

    # Summary stats from Experiment 07
    summary = pd.read_csv(results_dir / 'cwaci_summary_stats.csv')

    # Detailed comparison data
    comparison = pd.read_csv(results_dir / 'crowding_weighted_aci_comparison.csv')

    # Lambda selection results from Experiment 08
    lambda_selection = pd.read_csv(results_dir / 'lambda_selection_efficient.csv')

    return summary, comparison, lambda_selection


def fig1_cwaci_conditional_coverage(summary: pd.DataFrame, save_path: Path):
    """
    Figure 1: Conditional Coverage by Crowding Level - CW-ACI vs ACI

    Main result showing CW-ACI achieves more uniform coverage.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = ['low', 'med', 'high']
    bin_labels = ['Low\n(< 33%)', 'Medium\n(33-67%)', 'High\n(> 67%)']
    x = np.arange(len(bins))
    width = 0.25

    # ACI (baseline)
    aci_data = summary[summary['method'] == 'aci'].iloc[0]
    aci_coverages = [aci_data['low'], aci_data['med'], aci_data['high']]
    ax.bar(x - width, aci_coverages, width, label='ACI', color=COLORS['aci'], alpha=0.8)

    # CW-ACI λ=0.5 (optimal)
    cwaci_05 = summary[summary['method'] == 'cw_aci_λ0.5'].iloc[0]
    cwaci_coverages = [cwaci_05['low'], cwaci_05['med'], cwaci_05['high']]
    ax.bar(x, cwaci_coverages, width, label='CW-ACI (λ=0.5)', color=COLORS['cwaci'], alpha=0.8)

    # Split (baseline)
    split_data = summary[summary['method'] == 'split'].iloc[0]
    split_coverages = [split_data['low'], split_data['med'], split_data['high']]
    ax.bar(x + width, split_coverages, width, label='Split CP', color=COLORS['split'], alpha=0.8)

    # Target line
    ax.axhline(y=0.9, color=COLORS['target'], linestyle='--', linewidth=2, label='Target (90%)')

    ax.set_xlabel('Crowding Level')
    ax.set_ylabel('Coverage')
    ax.set_title('Conditional Coverage by Crowding Level')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_ylim(0.8, 0.95)
    ax.legend(loc='lower right')

    # Annotate key improvement
    improvement = cwaci_coverages[0] - aci_coverages[0]
    ax.annotate(f'+{improvement*100:.1f}%',
                xy=(0, cwaci_coverages[0]),
                xytext=(-0.4, cwaci_coverages[0] + 0.02),
                fontsize=10, fontweight='bold', color=COLORS['cwaci'])

    plt.tight_layout()
    plt.savefig(save_path / 'fig1_cwaci_conditional_coverage.pdf', bbox_inches='tight')
    plt.savefig(save_path / 'fig1_cwaci_conditional_coverage.png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved: {save_path / 'fig1_cwaci_conditional_coverage.pdf'}")


def fig2_lambda_sensitivity(summary: pd.DataFrame, save_path: Path):
    """
    Figure 2: λ Sensitivity Analysis

    Shows how λ controls coverage distribution across crowding regimes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Extract λ values from method names
    lambda_methods = summary[summary['method'].str.startswith('cw_aci')]

    lambdas = []
    low_cov = []
    high_cov = []
    variances = []

    for _, row in lambda_methods.iterrows():
        # Parse λ from method name like 'cw_aci_λ0.5'
        lam = float(row['method'].split('λ')[1])
        lambdas.append(lam)
        low_cov.append(row['low'])
        high_cov.append(row['high'])
        variances.append(row['variation'])

    # Sort by lambda
    sorted_idx = np.argsort(lambdas)
    lambdas = [lambdas[i] for i in sorted_idx]
    low_cov = [low_cov[i] for i in sorted_idx]
    high_cov = [high_cov[i] for i in sorted_idx]
    variances = [variances[i] for i in sorted_idx]

    # Left panel: Coverage by bin vs λ
    ax1.plot(lambdas, low_cov, 'o-', color=COLORS['low'], label='Low crowding', markersize=8, linewidth=2)
    ax1.plot(lambdas, high_cov, '^-', color=COLORS['high'], label='High crowding', markersize=8, linewidth=2)
    ax1.axhline(y=0.9, color=COLORS['target'], linestyle='--', linewidth=2, label='Target (90%)')

    # Mark optimal λ
    ax1.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7)
    ax1.annotate('Optimal\nλ=0.5', xy=(0.5, 0.85), fontsize=9, ha='center')

    ax1.set_xlabel('λ (Crowding Weight)')
    ax1.set_ylabel('Coverage')
    ax1.set_title('(a) Coverage by Crowding Bin')
    ax1.legend(loc='lower left')
    ax1.set_ylim(0.84, 0.94)

    # Right panel: Variance vs λ
    ax2.plot(lambdas, variances, 's-', color='purple', markersize=8, linewidth=2)

    # Mark minimum variance
    min_idx = np.argmin(variances)
    ax2.scatter([lambdas[min_idx]], [variances[min_idx]], s=200, c='gold', marker='*',
                zorder=5, label=f'Min variance (λ={lambdas[min_idx]})')

    # Add ACI variance for reference
    aci_var = summary[summary['method'] == 'aci'].iloc[0]['variation']
    ax2.axhline(y=aci_var, color=COLORS['aci'], linestyle='--', linewidth=2, label=f'ACI ({aci_var:.4f})')

    ax2.set_xlabel('λ (Crowding Weight)')
    ax2.set_ylabel('Coverage Variance')
    ax2.set_title('(b) Coverage Uniformity')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path / 'fig2_cwaci_lambda_sensitivity.pdf', bbox_inches='tight')
    plt.savefig(save_path / 'fig2_cwaci_lambda_sensitivity.png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved: {save_path / 'fig2_cwaci_lambda_sensitivity.pdf'}")


def fig3_marginal_comparison(summary: pd.DataFrame, save_path: Path):
    """
    Figure 3: Marginal Coverage Comparison

    Bar chart showing all methods maintain ~90% marginal coverage.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Select methods to show
    methods_to_show = ['split', 'aci', 'cw_aci_λ0.5', 'cw_aci_λ1.0']
    labels = ['Split CP', 'ACI', 'CW-ACI\n(λ=0.5)', 'CW-ACI\n(λ=1.0)']
    colors = [COLORS['split'], COLORS['aci'], COLORS['cwaci'], '#27ae60']

    coverages = []
    min_bins = []

    for method in methods_to_show:
        row = summary[summary['method'] == method].iloc[0]
        coverages.append(row['marginal'])
        min_bins.append(row['min_bin'])

    x = np.arange(len(methods_to_show))
    width = 0.35

    # Marginal coverage
    bars1 = ax.bar(x - width/2, coverages, width, label='Marginal Coverage', color=colors, alpha=0.8)

    # Min-bin coverage
    bars2 = ax.bar(x + width/2, min_bins, width, label='Min-Bin Coverage',
                   color=colors, alpha=0.4, hatch='//')

    ax.axhline(y=0.9, color=COLORS['target'], linestyle='--', linewidth=2, label='Target (90%)')

    ax.set_xlabel('Method')
    ax.set_ylabel('Coverage')
    ax.set_title('Marginal vs Min-Bin Coverage')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.82, 0.95)
    ax.legend(loc='lower right')

    # Add value labels
    for i, (c, m) in enumerate(zip(coverages, min_bins)):
        ax.text(i - width/2, c + 0.005, f'{c*100:.1f}%', ha='center', fontsize=9)
        ax.text(i + width/2, m + 0.005, f'{m*100:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path / 'fig3_cwaci_marginal_comparison.pdf', bbox_inches='tight')
    plt.savefig(save_path / 'fig3_cwaci_marginal_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved: {save_path / 'fig3_cwaci_marginal_comparison.pdf'}")


def fig4_variance_comparison(summary: pd.DataFrame, save_path: Path):
    """
    Figure 4: Coverage Variance (Uniformity) Comparison

    Key result: CW-ACI achieves lowest variance.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Select methods
    methods = ['split', 'aci', 'cw_aci_λ0.5', 'cw_aci_λ1.0']
    labels = ['Split CP', 'ACI', 'CW-ACI (λ=0.5)', 'CW-ACI (λ=1.0)']
    colors = [COLORS['split'], COLORS['aci'], COLORS['cwaci'], '#27ae60']

    variances = []
    for method in methods:
        row = summary[summary['method'] == method].iloc[0]
        variances.append(row['variation'])

    x = np.arange(len(methods))
    bars = ax.bar(x, variances, color=colors, alpha=0.8)

    # Highlight best (lowest variance)
    min_idx = np.argmin(variances)
    bars[min_idx].set_edgecolor('gold')
    bars[min_idx].set_linewidth(3)

    ax.set_xlabel('Method')
    ax.set_ylabel('Coverage Variance')
    ax.set_title('Coverage Uniformity (Lower = Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # Add value labels
    for i, v in enumerate(variances):
        ax.text(i, v + 0.0005, f'{v:.4f}', ha='center', fontsize=9)

    # Add improvement annotation
    aci_var = variances[1]
    cwaci_var = variances[2]
    improvement = (aci_var - cwaci_var) / aci_var * 100
    ax.annotate(f'{improvement:.0f}% reduction\nvs ACI',
                xy=(2, cwaci_var),
                xytext=(2.5, cwaci_var + 0.005),
                fontsize=10, fontweight='bold', color=COLORS['cwaci'],
                arrowprops=dict(arrowstyle='->', color=COLORS['cwaci']))

    plt.tight_layout()
    plt.savefig(save_path / 'fig4_cwaci_variance_comparison.pdf', bbox_inches='tight')
    plt.savefig(save_path / 'fig4_cwaci_variance_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved: {save_path / 'fig4_cwaci_variance_comparison.pdf'}")


def fig5_lambda_selection(lambda_selection: pd.DataFrame, save_path: Path):
    """
    Figure 5: CV-based λ Selection Distribution

    Shows what λ values are selected by cross-validation.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    lambda_counts = lambda_selection['selected_lambda'].value_counts().sort_index()

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(lambda_counts)))

    bars = ax.bar(range(len(lambda_counts)), lambda_counts.values,
                  color=colors, alpha=0.8, edgecolor='black')

    ax.set_xlabel('λ Value')
    ax.set_ylabel('Selection Frequency')
    ax.set_title('CV-based λ Selection Distribution (n=120 windows)')
    ax.set_xticks(range(len(lambda_counts)))
    ax.set_xticklabels([f'{l:.2f}' for l in lambda_counts.index])

    # Add percentage labels
    total = lambda_counts.sum()
    for i, v in enumerate(lambda_counts.values):
        pct = v / total * 100
        ax.text(i, v + 1, f'{pct:.0f}%', ha='center', fontsize=9)

    # Add mean annotation
    mean_lambda = lambda_selection['selected_lambda'].mean()
    ax.axvline(x=mean_lambda * 4, color='red', linestyle='--', linewidth=2)  # Scale for x position
    ax.annotate(f'Mean: {mean_lambda:.2f}',
                xy=(0.5, 85), fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(save_path / 'fig5_lambda_selection.pdf', bbox_inches='tight')
    plt.savefig(save_path / 'fig5_lambda_selection.png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved: {save_path / 'fig5_lambda_selection.pdf'}")


def create_cwaci_table(summary: pd.DataFrame, save_path: Path):
    """Create LaTeX table for CW-ACI comparison."""

    methods = ['split', 'aci', 'cw_aci_λ0.5', 'cw_aci_λ1.0']
    labels = ['Split CP', 'ACI', 'CW-ACI ($\\lambda=0.5$)', 'CW-ACI ($\\lambda=1.0$)']

    rows = []
    for method, label in zip(methods, labels):
        row = summary[summary['method'] == method].iloc[0]
        rows.append({
            'method': label,
            'marginal': row['marginal'] * 100,
            'low': row['low'] * 100,
            'med': row['med'] * 100,
            'high': row['high'] * 100,
            'variance': row['variation'],
            'min_bin': row['min_bin'] * 100,
            'avg_size': row['avg_size']
        })

    latex = r"""\begin{table}[t]
\centering
\caption{Coverage comparison of conformal prediction methods across crowding regimes. CW-ACI ($\lambda=0.5$) achieves the lowest coverage variance while maintaining marginal coverage near the 90\% target.}
\label{tab:cwaci_results}
\begin{tabular}{lcccccc}
\toprule
Method & Marginal & Low & Med & High & Variance & Avg Size \\
\midrule
"""

    for r in rows:
        latex += f"{r['method']} & {r['marginal']:.1f}\\% & {r['low']:.1f}\\% & {r['med']:.1f}\\% & {r['high']:.1f}\\% & {r['variance']:.4f} & {r['avg_size']:.2f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(save_path / 'table_cwaci_results.tex', 'w') as f:
        f.write(latex)

    print(f"Saved: {save_path / 'table_cwaci_results.tex'}")


def main():
    print("=" * 70)
    print("GENERATING CW-ACI FIGURES FOR ICML 2026 PAPER")
    print("=" * 70)

    # Create figures directory
    figures_dir = Path(__file__).parent.parent / 'paper' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    summary, comparison, lambda_selection = load_cwaci_results()

    print(f"\nLoaded summary with {len(summary)} methods")
    print(f"Loaded comparison with {len(comparison)} rows")
    print(f"Loaded lambda selection with {len(lambda_selection)} windows")

    # Generate figures
    print("\nGenerating figures...")

    fig1_cwaci_conditional_coverage(summary, figures_dir)
    fig2_lambda_sensitivity(summary, figures_dir)
    fig3_marginal_comparison(summary, figures_dir)
    fig4_variance_comparison(summary, figures_dir)
    fig5_lambda_selection(lambda_selection, figures_dir)

    # Generate table
    print("\nGenerating LaTeX table...")
    create_cwaci_table(summary, figures_dir)

    print("\n" + "=" * 70)
    print("ALL CW-ACI FIGURES GENERATED")
    print("=" * 70)
    print(f"\nFigures saved to: {figures_dir}")


if __name__ == '__main__':
    main()
