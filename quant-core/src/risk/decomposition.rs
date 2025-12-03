//! Risk Decomposition: MCR, CCR, and Risk Parity
//!
//! # Why Decompose Risk?
//!
//! A portfolio's risk comes from multiple sources:
//! - Individual asset volatilities
//! - Correlations between assets
//! - Asset weights
//!
//! Risk decomposition tells us: "Where is my risk coming from?"
//!
//! # Key Metrics
//!
//! ## Marginal Contribution to Risk (MCR)
//!
//! ```text
//! MCR_i = ∂σ_p / ∂w_i = (Σ × w)_i / σ_p
//! ```
//!
//! **Interpretation**: If I increase asset i's weight by 1%, how much does
//! portfolio volatility increase?
//!
//! ## Component Contribution to Risk (CCR)
//!
//! ```text
//! CCR_i = w_i × MCR_i
//! ```
//!
//! **Key Property**: Σ CCR_i = σ_p (they sum to total portfolio risk!)
//!
//! **Interpretation**: How much of the total portfolio risk is "caused" by asset i?
//!
//! ## Risk Parity
//!
//! Find weights where all assets contribute equally to risk:
//! ```text
//! CCR_1 = CCR_2 = ... = CCR_n
//! ```
//!
//! **Why Risk Parity?**
//! - Diversification: No single asset dominates risk
//! - Robust: Doesn't require expected return estimates (Mean-Variance needs μ)
//! - Popular: Bridgewater's "All Weather" fund uses this approach

use nalgebra::{DMatrix, DVector};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Calculate portfolio volatility from weights and covariance
///
/// ```text
/// σ_p = √(w' × Σ × w)
/// ```
pub fn portfolio_volatility(weights: &DVector<f64>, cov: &DMatrix<f64>) -> f64 {
    let variance = (weights.transpose() * cov * weights)[(0, 0)];
    variance.sqrt()
}

/// Calculate Marginal Contribution to Risk
///
/// ```text
/// MCR = (Σ × w) / σ_p
/// ```
///
/// # Returns
/// Vector of MCR for each asset
pub fn marginal_contribution_to_risk(
    weights: &DVector<f64>,
    cov: &DMatrix<f64>,
) -> DVector<f64> {
    let sigma_p = portfolio_volatility(weights, cov);

    if sigma_p < 1e-10 {
        return DVector::zeros(weights.len());
    }

    let cov_w = cov * weights;
    cov_w / sigma_p
}

/// Calculate Component Contribution to Risk
///
/// ```text
/// CCR_i = w_i × MCR_i
/// ```
///
/// # Key Property
/// sum(CCR) = σ_p (total portfolio volatility)
///
/// # Returns
/// Vector of CCR for each asset
pub fn component_contribution_to_risk(
    weights: &DVector<f64>,
    cov: &DMatrix<f64>,
) -> DVector<f64> {
    let mcr = marginal_contribution_to_risk(weights, cov);
    weights.component_mul(&mcr)
}

/// Calculate Percentage Contribution to Risk
///
/// ```text
/// PCT_i = CCR_i / σ_p = (w_i × MCR_i) / σ_p
/// ```
///
/// # Returns
/// Vector of percentage contributions (sums to 1.0)
pub fn percentage_contribution_to_risk(
    weights: &DVector<f64>,
    cov: &DMatrix<f64>,
) -> DVector<f64> {
    let ccr = component_contribution_to_risk(weights, cov);
    let sigma_p = portfolio_volatility(weights, cov);

    if sigma_p < 1e-10 {
        return DVector::from_element(weights.len(), 1.0 / weights.len() as f64);
    }

    ccr / sigma_p
}

/// Risk Parity Optimizer
///
/// Find weights where all assets contribute equally to risk.
///
/// # Algorithm (Newton-Raphson)
///
/// We want: CCR_1 = CCR_2 = ... = CCR_n
///
/// Equivalently, minimize:
/// ```text
/// f(w) = Σ_i Σ_j (w_i × (Σw)_i - w_j × (Σw)_j)²
/// ```
///
/// Subject to: Σ w_i = 1, w_i > 0
///
/// We use iterative adjustment:
/// 1. Start with equal weights
/// 2. Calculate CCR for each asset
/// 3. Reduce weight of high-CCR assets, increase low-CCR assets
/// 4. Repeat until convergence
pub fn risk_parity_weights(
    cov: &DMatrix<f64>,
    max_iter: usize,
    tol: f64,
) -> DVector<f64> {
    let n = cov.nrows();

    // Start with inverse volatility weights (better starting point)
    let vol: Vec<f64> = (0..n).map(|i| cov[(i, i)].sqrt()).collect();
    let inv_vol: Vec<f64> = vol.iter().map(|v| 1.0 / v.max(1e-10)).collect();
    let inv_vol_sum: f64 = inv_vol.iter().sum();

    let mut weights = DVector::from_vec(
        inv_vol.iter().map(|iv| iv / inv_vol_sum).collect()
    );

    // Iterative risk parity optimization
    for iter in 0..max_iter {
        let pct_contrib = percentage_contribution_to_risk(&weights, cov);

        // Target: equal contribution (1/n for each)
        let target = 1.0 / n as f64;

        // Calculate adjustment: reduce over-contributing, increase under-contributing
        let mut max_diff = 0.0;
        for i in 0..n {
            let diff = (pct_contrib[i] - target).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        if max_diff < tol {
            break; // Converged
        }

        // Adjust weights: w_new = w_old × (target / pct_contrib)^0.5
        // Using square root for stability
        for i in 0..n {
            let adjustment = if pct_contrib[i] > 1e-10 {
                (target / pct_contrib[i]).sqrt()
            } else {
                2.0 // Double weight if contribution is negligible
            };
            weights[i] *= adjustment;
        }

        // Normalize to sum to 1
        let sum: f64 = weights.iter().sum();
        weights /= sum;

        // Diagnostic output for first few iterations
        if iter < 3 {
            let _max_pct = pct_contrib.max();
            let _min_pct = pct_contrib.min();
        }
    }

    weights
}

// ============ Python Bindings ============

/// Calculate MCR for each asset (Python)
#[pyfunction]
#[pyo3(name = "marginal_contribution_to_risk")]
pub fn mcr_py<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray1<f64>,
    cov: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let w = weights.as_array();
    let c = cov.as_array();

    let n = w.len();
    let weights_vec = DVector::from_row_slice(w.as_slice().unwrap());
    let cov_mat = DMatrix::from_row_slice(n, n, c.as_slice().unwrap());

    let mcr = marginal_contribution_to_risk(&weights_vec, &cov_mat);

    Ok(PyArray1::from_slice_bound(py, mcr.as_slice()))
}

/// Calculate CCR for each asset (Python)
#[pyfunction]
#[pyo3(name = "component_contribution_to_risk")]
pub fn ccr_py<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray1<f64>,
    cov: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let w = weights.as_array();
    let c = cov.as_array();

    let n = w.len();
    let weights_vec = DVector::from_row_slice(w.as_slice().unwrap());
    let cov_mat = DMatrix::from_row_slice(n, n, c.as_slice().unwrap());

    let ccr = component_contribution_to_risk(&weights_vec, &cov_mat);

    Ok(PyArray1::from_slice_bound(py, ccr.as_slice()))
}

/// Calculate percentage contribution to risk (Python)
#[pyfunction]
#[pyo3(name = "percentage_contribution_to_risk")]
pub fn pct_py<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray1<f64>,
    cov: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let w = weights.as_array();
    let c = cov.as_array();

    let n = w.len();
    let weights_vec = DVector::from_row_slice(w.as_slice().unwrap());
    let cov_mat = DMatrix::from_row_slice(n, n, c.as_slice().unwrap());

    let pct = percentage_contribution_to_risk(&weights_vec, &cov_mat);

    Ok(PyArray1::from_slice_bound(py, pct.as_slice()))
}

/// Calculate risk parity weights (Python)
///
/// # Arguments
/// * `cov` - Covariance matrix (n × n)
/// * `max_iter` - Maximum iterations (default: 100)
/// * `tol` - Convergence tolerance (default: 1e-6)
#[pyfunction]
#[pyo3(name = "risk_parity_weights", signature = (cov, max_iter=100, tol=1e-6))]
pub fn risk_parity_py<'py>(
    py: Python<'py>,
    cov: PyReadonlyArray2<f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let c = cov.as_array();
    let n = c.nrows();

    let cov_mat = DMatrix::from_row_slice(n, n, c.as_slice().unwrap());

    let weights = risk_parity_weights(&cov_mat, max_iter, tol);

    Ok(PyArray1::from_slice_bound(py, weights.as_slice()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_cov() -> DMatrix<f64> {
        // 3 assets with different volatilities and correlations
        DMatrix::from_row_slice(3, 3, &[
            0.04, 0.01, 0.005,   // Asset 1: 20% vol
            0.01, 0.09, 0.015,   // Asset 2: 30% vol
            0.005, 0.015, 0.01,  // Asset 3: 10% vol
        ])
    }

    #[test]
    fn test_portfolio_volatility() {
        let cov = create_test_cov();
        let weights = DVector::from_vec(vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);

        let vol = portfolio_volatility(&weights, &cov);

        // Should be positive and less than max individual vol
        assert!(vol > 0.0);
        assert!(vol < 0.30); // Less than 30% (Asset 2's vol)
    }

    #[test]
    fn test_ccr_sums_to_total() {
        let cov = create_test_cov();
        let weights = DVector::from_vec(vec![0.5, 0.3, 0.2]);

        let ccr = component_contribution_to_risk(&weights, &cov);
        let sigma_p = portfolio_volatility(&weights, &cov);

        // Key property: CCR sums to total portfolio volatility
        let ccr_sum: f64 = ccr.iter().sum();
        assert_relative_eq!(ccr_sum, sigma_p, epsilon = 1e-10);
    }

    #[test]
    fn test_pct_sums_to_one() {
        let cov = create_test_cov();
        let weights = DVector::from_vec(vec![0.4, 0.4, 0.2]);

        let pct = percentage_contribution_to_risk(&weights, &cov);
        let pct_sum: f64 = pct.iter().sum();

        assert_relative_eq!(pct_sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_risk_parity() {
        let cov = create_test_cov();

        let weights = risk_parity_weights(&cov, 100, 1e-6);

        // Weights should sum to 1
        let sum: f64 = weights.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);

        // All weights should be positive
        for w in weights.iter() {
            assert!(*w > 0.0);
        }

        // Check that percentage contributions are roughly equal
        let pct = percentage_contribution_to_risk(&weights, &cov);
        let target = 1.0 / 3.0;

        for p in pct.iter() {
            assert_relative_eq!(*p, target, epsilon = 0.01); // 1% tolerance
        }
    }

    #[test]
    fn test_risk_parity_with_different_vols() {
        // Assets with very different volatilities
        let cov = DMatrix::from_row_slice(2, 2, &[
            0.01, 0.0,    // Asset 1: 10% vol
            0.0, 0.25,    // Asset 2: 50% vol (5x higher)
        ]);

        let weights = risk_parity_weights(&cov, 100, 1e-6);

        // High-vol asset should have lower weight
        // With 50% vs 10% vol (5:1 ratio), weight ratio should be ~1:5
        assert!(weights[1] < weights[0]);
    }
}
