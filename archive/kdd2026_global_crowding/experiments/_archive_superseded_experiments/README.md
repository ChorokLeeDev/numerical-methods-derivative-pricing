# Archived Superseded Experiments
## December 16, 2025

This directory contains experimental scripts that have been superseded by more targeted diagnostic scripts.

These files are kept for **historical reference only** and should not be used for new analysis.

## Archived Files

### Original KDD Experiments (01-08)
- `01_kdd_experiments.py` - Initial KDD baseline experiments
- `02_kdd_experiments_fixed.py` - Bug fix iteration
- `03_domain_adaptation.py` - Domain adaptation testing
- `04_temporal_mmd.py` - Temporal MMD implementation tests
- `05_ablation_study.py` - Ablation studies
- `06_daily_data.py` - Daily frequency data experiments
- `07_multi_domain.py` - Multi-domain transfer tests
- `08_full_comparison.py` - Full baseline comparisons

### Intermediate Experiments (09-12 variants)
- `09_extended_evaluation.py` - Extended evaluation (superseded by 09_country_transfer_validation.py)
- `10_regime_ablation.py` - Regime ablation (superseded by 10_regime_composition_analysis.py)
- `11_generate_figures.py` - Figure generation
- `12_final_evaluation.py` - Final evaluation (superseded by 13_mmd_comparison_standard_vs_regime.py)

## Why Archived

These experiments were conducted during the initial research phase and were replaced by:
1. More focused diagnostic scripts (10-13 series)
2. Better documented analysis procedures
3. Cleaner output and interpretation

## Use Case

- **Reference only**: If you need to understand experimental evolution
- **NOT for reproduction**: Use the numbered 09-13 scripts instead
- **NOT for validation**: Use DIAGNOSTIC_REPORT.md and FINAL_SUMMARY.md for current findings

## Current Active Scripts

See parent directory for current diagnostic scripts:
- `09_country_transfer_validation.py` - Main transfer validation
- `10_regime_composition_analysis.py` - Regime distribution analysis
- `11_regime_detection_debug.py` - Regime detection verification
- `12_date_range_analysis.py` - Temporal regime evolution
- `13_mmd_comparison_standard_vs_regime.py` - Method comparison
