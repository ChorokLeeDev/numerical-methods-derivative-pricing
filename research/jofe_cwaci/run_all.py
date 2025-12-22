#!/usr/bin/env python3
"""
Master Replication Script

Runs all experiments to reproduce the results in:
"Crowding-Aware Conformal Prediction for Factor Return Uncertainty"

Author: Chorok Lee (KAIST)
Date: December 2024

Usage:
    python run_all.py
"""

import subprocess
import sys
from pathlib import Path
import time


def run_script(script_path: Path, description: str) -> bool:
    """Run a Python script and report status."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_path.name}")
    print('='*70)

    start_time = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=script_path.parent.parent,
        capture_output=False
    )

    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"\n[SUCCESS] {description} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n[FAILED] {description}")
        return False


def main():
    """Run all replication scripts."""
    print("="*70)
    print("MASTER REPLICATION SCRIPT")
    print("Crowding-Aware Conformal Prediction for Factor Return Uncertainty")
    print("="*70)

    project_dir = Path(__file__).parent
    scripts_dir = project_dir / 'scripts'
    experiments_dir = project_dir / 'experiments'

    # Track results
    results = []

    # Step 1: Download data
    results.append(run_script(
        scripts_dir / 'download_data.py',
        "Step 1: Download Fama-French Factor Data"
    ))

    # Step 2: Main coverage analysis
    results.append(run_script(
        experiments_dir / '01_coverage_analysis.py',
        "Step 2: Main Coverage Analysis (Table 3, Figure 1)"
    ))

    # Step 3: Monte Carlo validation
    results.append(run_script(
        experiments_dir / '02_monte_carlo.py',
        "Step 3: Monte Carlo Validation (Tables 1-2, Figure 3)"
    ))

    # Step 4: Robustness analysis
    results.append(run_script(
        experiments_dir / '03_robustness.py',
        "Step 4: Robustness Analysis (Section 6)"
    ))

    # Step 5: Generate figures
    results.append(run_script(
        experiments_dir / '04_generate_figures.py',
        "Step 5: Generate Publication Figures"
    ))

    # Summary
    print("\n" + "="*70)
    print("REPLICATION SUMMARY")
    print("="*70)

    steps = [
        "Data Download",
        "Coverage Analysis",
        "Monte Carlo",
        "Robustness",
        "Figures"
    ]

    all_success = True
    for step, success in zip(steps, results):
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {step}")
        if not success:
            all_success = False

    print("="*70)

    if all_success:
        print("\nALL STEPS COMPLETED SUCCESSFULLY")
        print("\nOutput files:")
        print("  - results/coverage_analysis.csv")
        print("  - results/monte_carlo_main.csv")
        print("  - results/monte_carlo_effects.csv")
        print("  - results/robustness_*.csv")
        print("  - paper/figures/*.pdf")
        print("\nTo compile the paper:")
        print("  cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex")
    else:
        print("\nSOME STEPS FAILED - please check the output above")
        sys.exit(1)


if __name__ == '__main__':
    main()
