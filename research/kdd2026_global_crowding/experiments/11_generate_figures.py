"""
KDD 2026: Generate Paper Figures

Figures:
1. Method diagram (Temporal-MMD vs standard MMD)
2. Main results comparison (4 domains × 5 methods)
3. Regime imbalance vs T-MMD improvement
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

FIG_DIR = Path(__file__).parent.parent / 'paper' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Results from 09_extended_evaluation.py
RESULTS = {
    'Finance': {'RF': 0.593, 'MMD': 0.590, 'DANN': 0.587, 'CDAN': 0.582, 'T-MMD': 0.594},
    'Electricity': {'RF': 0.608, 'MMD': 0.625, 'DANN': 0.606, 'CDAN': 0.583, 'T-MMD': 0.634},
    'GasSensor': {'RF': 0.996, 'MMD': 0.997, 'DANN': 0.997, 'CDAN': 0.997, 'T-MMD': 0.997},
    'Activity': {'RF': 0.692, 'MMD': 0.707, 'DANN': 0.707, 'CDAN': 0.706, 'T-MMD': 0.706},
}

STDS = {
    'Finance': {'RF': 0.004, 'MMD': 0.011, 'DANN': 0.006, 'CDAN': 0.006, 'T-MMD': 0.008},
    'Electricity': {'RF': 0.005, 'MMD': 0.027, 'DANN': 0.011, 'CDAN': 0.025, 'T-MMD': 0.014},
    'GasSensor': {'RF': 0.001, 'MMD': 0.000, 'DANN': 0.000, 'CDAN': 0.000, 'T-MMD': 0.000},
    'Activity': {'RF': 0.002, 'MMD': 0.001, 'DANN': 0.001, 'CDAN': 0.000, 'T-MMD': 0.001},
}

# Regime imbalance (minority class proportion)
REGIME_IMBALANCE = {
    'Finance': 0.45,      # ~45/55
    'Electricity': 0.33,  # 33% peak
    'GasSensor': 0.50,    # 50/50
    'Activity': 0.41,     # 41% stationary
}


def fig1_method_diagram():
    """Create method comparison diagram."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Colors
    source_color = '#3498db'
    target_color = '#e74c3c'
    regime_colors = ['#2ecc71', '#f39c12']

    # Left: Standard MMD
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Standard MMD', fontsize=14, fontweight='bold')

    # Source distribution (mixed regimes)
    np.random.seed(42)
    source_x = np.random.randn(50) * 0.8 + 3
    source_y = np.random.randn(50) * 1.5 + 5
    ax.scatter(source_x, source_y, c=source_color, alpha=0.6, s=30, label='Source')

    # Target distribution (mixed regimes)
    target_x = np.random.randn(50) * 0.8 + 7
    target_y = np.random.randn(50) * 1.5 + 5
    ax.scatter(target_x, target_y, c=target_color, alpha=0.6, s=30, label='Target')

    # Arrow showing global alignment
    ax.annotate('', xy=(6, 5), xytext=(4, 5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(5, 5.5, 'MMD', ha='center', fontsize=11)

    ax.legend(loc='upper left')

    # Right: Temporal-MMD
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Temporal-MMD (Ours)', fontsize=14, fontweight='bold')

    # Source - Regime 1 (high volatility)
    source_r1_x = np.random.randn(25) * 0.5 + 2.5
    source_r1_y = np.random.randn(25) * 0.8 + 7
    ax.scatter(source_r1_x, source_r1_y, c=regime_colors[0], alpha=0.6, s=30,
               marker='o', label='Source R1')

    # Source - Regime 2 (low volatility)
    source_r2_x = np.random.randn(25) * 0.5 + 2.5
    source_r2_y = np.random.randn(25) * 0.8 + 3
    ax.scatter(source_r2_x, source_r2_y, c=regime_colors[1], alpha=0.6, s=30,
               marker='s', label='Source R2')

    # Target - Regime 1
    target_r1_x = np.random.randn(25) * 0.5 + 7.5
    target_r1_y = np.random.randn(25) * 0.8 + 7
    ax.scatter(target_r1_x, target_r1_y, c=regime_colors[0], alpha=0.6, s=30,
               marker='o', edgecolors='black', linewidths=0.5)

    # Target - Regime 2
    target_r2_x = np.random.randn(25) * 0.5 + 7.5
    target_r2_y = np.random.randn(25) * 0.8 + 3
    ax.scatter(target_r2_x, target_r2_y, c=regime_colors[1], alpha=0.6, s=30,
               marker='s', edgecolors='black', linewidths=0.5)

    # Arrows showing regime-wise alignment
    ax.annotate('', xy=(6.5, 7), xytext=(3.5, 7),
                arrowprops=dict(arrowstyle='->', lw=2, color=regime_colors[0]))
    ax.text(5, 7.5, 'MMD₁', ha='center', fontsize=10, color=regime_colors[0])

    ax.annotate('', xy=(6.5, 3), xytext=(3.5, 3),
                arrowprops=dict(arrowstyle='->', lw=2, color=regime_colors[1]))
    ax.text(5, 3.5, 'MMD₂', ha='center', fontsize=10, color=regime_colors[1])

    # Legend
    r1_patch = mpatches.Patch(color=regime_colors[0], label='Regime 1 (High)')
    r2_patch = mpatches.Patch(color=regime_colors[1], label='Regime 2 (Low)')
    ax.legend(handles=[r1_patch, r2_patch], loc='upper left')

    plt.tight_layout()
    path = FIG_DIR / 'fig1_method_diagram.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()


def fig2_main_results():
    """Create main results comparison figure."""
    fig, ax = plt.subplots(figsize=(10, 6))

    domains = list(RESULTS.keys())
    methods = ['RF', 'MMD', 'DANN', 'CDAN', 'T-MMD']
    colors = ['#95a5a6', '#3498db', '#9b59b6', '#e67e22', '#27ae60']

    x = np.arange(len(domains))
    width = 0.15

    for i, method in enumerate(methods):
        values = [RESULTS[d][method] for d in domains]
        stds = [STDS[d][method] for d in domains]
        offset = (i - 2) * width
        bars = ax.bar(x + offset, values, width, label=method, color=colors[i],
                      yerr=stds, capsize=3, alpha=0.85)

        # Highlight T-MMD
        if method == 'T-MMD':
            for bar in bars:
                bar.set_edgecolor('black')
                bar.set_linewidth(1.5)

    ax.set_ylabel('Target AUC', fontsize=12)
    ax.set_xlabel('Domain', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc='upper right', ncol=5)
    ax.set_title('Domain Adaptation Performance Across 4 Domains', fontsize=14, fontweight='bold')

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = FIG_DIR / 'fig2_main_results.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()


def fig3_regime_imbalance():
    """Create regime imbalance vs T-MMD improvement figure."""
    fig, ax = plt.subplots(figsize=(8, 6))

    domains = list(RESULTS.keys())

    # Calculate T-MMD improvement over best baseline
    improvements = []
    imbalances = []

    for domain in domains:
        tmmd = RESULTS[domain]['T-MMD']
        best_baseline = max(RESULTS[domain]['RF'], RESULTS[domain]['MMD'],
                           RESULTS[domain]['DANN'], RESULTS[domain]['CDAN'])
        improvement = (tmmd - best_baseline) / best_baseline * 100
        improvements.append(improvement)

        # Imbalance = distance from 0.5
        imbalances.append(abs(0.5 - REGIME_IMBALANCE[domain]))

    # Plot
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    for i, domain in enumerate(domains):
        ax.scatter(imbalances[i], improvements[i], s=200, c=colors[i],
                   label=domain, zorder=5, edgecolors='black', linewidths=1)
        ax.annotate(domain, (imbalances[i], improvements[i]),
                    xytext=(10, 5), textcoords='offset points', fontsize=10)

    # Trend line
    z = np.polyfit(imbalances, improvements, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 0.2, 100)
    ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.7, label='Trend')

    ax.set_xlabel('Regime Imbalance (|0.5 - minority ratio|)', fontsize=12)
    ax.set_ylabel('T-MMD Improvement over Best Baseline (%)', fontsize=12)
    ax.set_title('T-MMD Benefits Increase with Regime Imbalance', fontsize=14, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlim(-0.02, 0.22)

    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    path = FIG_DIR / 'fig3_regime_imbalance.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()


def fig4_results_table():
    """Create publication-ready results table as figure."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    # Table data
    domains = ['Finance', 'Electricity', 'GasSensor', 'Activity', 'Average']
    methods = ['RF', 'MMD', 'DANN', 'CDAN', 'T-MMD']

    # Calculate averages
    avg_results = {}
    for method in methods:
        avg_results[method] = np.mean([RESULTS[d][method] for d in list(RESULTS.keys())])

    # Create table data
    table_data = []
    for domain in domains[:-1]:  # Exclude average
        row = [domain]
        best_val = max(RESULTS[domain].values())
        for method in methods:
            val = RESULTS[domain][method]
            std = STDS[domain][method]
            if val == best_val:
                row.append(f'**{val:.3f}**±{std:.3f}')
            else:
                row.append(f'{val:.3f}±{std:.3f}')
        table_data.append(row)

    # Average row
    avg_row = ['Average']
    best_avg = max(avg_results.values())
    for method in methods:
        val = avg_results[method]
        if val == best_avg:
            avg_row.append(f'**{val:.3f}**')
        else:
            avg_row.append(f'{val:.3f}')
    table_data.append(avg_row)

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Domain'] + methods,
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0'] * 6
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Highlight T-MMD column
    for i in range(len(table_data) + 1):
        table[(i, 5)].set_facecolor('#e8f5e9')

    plt.title('Table 1: Domain Adaptation Results (AUC)', fontsize=14, fontweight='bold', y=0.95)

    path = FIG_DIR / 'fig4_results_table.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close()


def main():
    print("=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)

    print("\n1. Method diagram...")
    fig1_method_diagram()

    print("\n2. Main results comparison...")
    fig2_main_results()

    print("\n3. Regime imbalance analysis...")
    fig3_regime_imbalance()

    print("\n4. Results table...")
    fig4_results_table()

    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED")
    print("=" * 60)
    print(f"\nSaved to: {FIG_DIR}")


if __name__ == '__main__':
    main()
