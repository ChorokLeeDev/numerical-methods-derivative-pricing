//! Constrained Portfolio Optimization
//!
//! Real portfolios have constraints:
//! - **Long-only**: w ≥ 0 (no shorting)
//! - **Position limits**: w_min ≤ w ≤ w_max
//! - **Sector limits**: Σ w_sector ≤ limit
//! - **Budget**: Σ w = 1 (fully invested)
//!
//! # Mathematical Formulation
//!
//! ```text
//! minimize    (1/2) w' Σ w - λ μ' w    (risk - return tradeoff)
//! subject to  w' 1 = 1                  (budget constraint)
//!             w_min ≤ w ≤ w_max         (box constraints)
//! ```
//!
//! This is a **Quadratic Programming (QP)** problem.
//!
//! # Solution Method
//!
//! We use **Projected Gradient Descent**:
//! 1. Start with feasible weights
//! 2. Take gradient step: w_new = w - α ∇f(w)
//! 3. Project onto constraint set
//! 4. Repeat until convergence
//!
//! ## Why Not Use a QP Solver Library?
//!
//! - Educational: You learn the algorithm
//! - Portable: No external dependencies
//! - Sufficient: Works well for portfolio problems (convex, well-conditioned)
//!
//! For production with thousands of assets, consider OSQP, CVXPY, or Gurobi.

use nalgebra::{DMatrix, DVector};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Constraints for portfolio optimization
#[derive(Clone)]
pub struct PortfolioConstraints {
    /// Minimum weight per asset (default: 0.0 for long-only)
    pub w_min: Vec<f64>,
    /// Maximum weight per asset (default: 1.0)
    pub w_max: Vec<f64>,
    /// Whether weights must sum to 1 (default: true)
    pub fully_invested: bool,
}

impl PortfolioConstraints {
    /// Create default long-only constraints
    pub fn long_only(n: usize) -> Self {
        Self {
            w_min: vec![0.0; n],
            w_max: vec![1.0; n],
            fully_invested: true,
        }
    }

    /// Create box constraints with custom bounds
    pub fn box_constrained(w_min: Vec<f64>, w_max: Vec<f64>) -> Self {
        Self {
            w_min,
            w_max,
            fully_invested: true,
        }
    }
}

/// Project weights onto the constraint set
///
/// Two-step projection:
/// 1. Clip to box constraints [w_min, w_max]
/// 2. Normalize to sum to 1 (if fully_invested)
///
/// For the sum-to-1 constraint with box constraints, we use
/// iterative proportional fitting (Dykstra's algorithm simplified).
fn project_weights(w: &DVector<f64>, constraints: &PortfolioConstraints) -> DVector<f64> {
    let n = w.len();
    let mut projected = w.clone();

    // Step 1: Clip to box constraints
    for i in 0..n {
        projected[i] = projected[i].clamp(constraints.w_min[i], constraints.w_max[i]);
    }

    // Step 2: Normalize to sum to 1 (preserving box constraints)
    if constraints.fully_invested {
        // Iterative adjustment to satisfy both constraints
        for _ in 0..100 {
            let sum: f64 = projected.iter().sum();
            let diff = sum - 1.0;

            if diff.abs() < 1e-10 {
                break;
            }

            // Distribute the difference proportionally
            // but respect box constraints
            let mut adjustment_capacity = 0.0;
            for i in 0..n {
                if diff > 0.0 {
                    // Need to reduce weights
                    adjustment_capacity += projected[i] - constraints.w_min[i];
                } else {
                    // Need to increase weights
                    adjustment_capacity += constraints.w_max[i] - projected[i];
                }
            }

            if adjustment_capacity < 1e-10 {
                break; // Can't adjust further
            }

            for i in 0..n {
                let capacity = if diff > 0.0 {
                    projected[i] - constraints.w_min[i]
                } else {
                    constraints.w_max[i] - projected[i]
                };
                let proportion = capacity / adjustment_capacity;
                projected[i] -= diff * proportion;
                projected[i] = projected[i].clamp(constraints.w_min[i], constraints.w_max[i]);
            }
        }
    }

    projected
}

/// Gradient of the objective function
///
/// For: f(w) = (1/2) w' Σ w - λ μ' w
/// Gradient: ∇f(w) = Σ w - λ μ
fn gradient(
    w: &DVector<f64>,
    cov: &DMatrix<f64>,
    expected_returns: Option<&DVector<f64>>,
    risk_aversion: f64,
) -> DVector<f64> {
    let risk_grad = cov * w;

    match expected_returns {
        Some(mu) => risk_grad - mu * (1.0 / risk_aversion),
        None => risk_grad,
    }
}

/// Objective function value
///
/// f(w) = (1/2) w' Σ w - λ⁻¹ μ' w
fn objective(
    w: &DVector<f64>,
    cov: &DMatrix<f64>,
    expected_returns: Option<&DVector<f64>>,
    risk_aversion: f64,
) -> f64 {
    let variance = (w.transpose() * cov * w)[(0, 0)];
    let risk_term = 0.5 * variance;

    match expected_returns {
        Some(mu) => risk_term - (mu.dot(w)) / risk_aversion,
        None => risk_term,
    }
}

/// Constrained Portfolio Optimizer using Projected Gradient Descent
///
/// # Algorithm
///
/// ```text
/// 1. Initialize: w = equal weight (feasible)
/// 2. For each iteration:
///    a. Compute gradient: g = Σw - λ⁻¹μ
///    b. Update: w_new = w - α × g
///    c. Project: w_new = Project(w_new)
///    d. Check convergence
/// 3. Return optimal weights
/// ```
///
/// # Arguments
/// * `cov` - Covariance matrix (n × n)
/// * `expected_returns` - Expected returns (optional, for max Sharpe)
/// * `constraints` - Portfolio constraints
/// * `risk_aversion` - Risk aversion parameter (higher = more conservative)
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
/// Optimal portfolio weights
pub fn optimize_constrained(
    cov: &DMatrix<f64>,
    expected_returns: Option<&DVector<f64>>,
    constraints: &PortfolioConstraints,
    risk_aversion: f64,
    max_iter: usize,
    tol: f64,
) -> DVector<f64> {
    let n = cov.nrows();

    // Initialize with equal weights (projected to satisfy constraints)
    let init = DVector::from_element(n, 1.0 / n as f64);
    let mut w = project_weights(&init, constraints);

    // Adaptive learning rate
    // Start with 1 / max eigenvalue approximation
    let trace = (0..n).map(|i| cov[(i, i)]).sum::<f64>();
    let mut alpha = 1.0 / (trace / n as f64 + 1e-6);

    let mut prev_obj = objective(&w, cov, expected_returns, risk_aversion);

    // Momentum for acceleration
    let mut velocity = DVector::zeros(n);
    let momentum = 0.9;

    for iter in 0..max_iter {
        // Compute gradient
        let grad = gradient(&w, cov, expected_returns, risk_aversion);

        // Update with momentum
        velocity = &velocity * momentum - &grad * alpha;
        let w_new = &w + &velocity;

        // Project onto constraints
        let w_projected = project_weights(&w_new, constraints);

        // Compute new objective
        let new_obj = objective(&w_projected, cov, expected_returns, risk_aversion);

        // Line search: reduce step size if objective increased
        if new_obj > prev_obj + 1e-10 {
            alpha *= 0.5;
            velocity = DVector::zeros(n); // Reset momentum
            if alpha < 1e-12 {
                break; // Step size too small
            }
            continue;
        }

        // Check convergence
        let weight_change: f64 = (&w_projected - &w).iter().map(|x| x.abs()).sum();
        if weight_change < tol {
            return w_projected;
        }

        // Accept step
        w = w_projected;
        prev_obj = new_obj;

        // Occasionally try larger step
        if iter % 10 == 0 {
            alpha *= 1.2;
        }
    }

    w
}

/// Minimum Variance Portfolio (constrained)
///
/// ```text
/// minimize    w' Σ w
/// subject to  Σ w = 1
///             w_min ≤ w ≤ w_max
/// ```
pub fn min_variance_constrained(
    cov: &DMatrix<f64>,
    w_min: Option<&[f64]>,
    w_max: Option<&[f64]>,
) -> DVector<f64> {
    let n = cov.nrows();

    let constraints = match (w_min, w_max) {
        (Some(mins), Some(maxs)) => PortfolioConstraints::box_constrained(
            mins.to_vec(),
            maxs.to_vec(),
        ),
        (Some(mins), None) => PortfolioConstraints {
            w_min: mins.to_vec(),
            w_max: vec![1.0; n],
            fully_invested: true,
        },
        (None, Some(maxs)) => PortfolioConstraints {
            w_min: vec![0.0; n],
            w_max: maxs.to_vec(),
            fully_invested: true,
        },
        (None, None) => PortfolioConstraints::long_only(n),
    };

    optimize_constrained(cov, None, &constraints, 1.0, 1000, 1e-8)
}

/// Maximum Sharpe Ratio Portfolio (constrained)
///
/// ```text
/// maximize    (μ' w - r_f) / √(w' Σ w)
/// subject to  Σ w = 1
///             w_min ≤ w ≤ w_max
/// ```
///
/// We convert this to a QP by fixing the return and minimizing variance.
pub fn max_sharpe_constrained(
    cov: &DMatrix<f64>,
    expected_returns: &DVector<f64>,
    risk_free_rate: f64,
    w_min: Option<&[f64]>,
    w_max: Option<&[f64]>,
) -> DVector<f64> {
    let n = cov.nrows();

    // Excess returns
    let excess = expected_returns - DVector::from_element(n, risk_free_rate);

    let constraints = match (w_min, w_max) {
        (Some(mins), Some(maxs)) => PortfolioConstraints::box_constrained(
            mins.to_vec(),
            maxs.to_vec(),
        ),
        _ => PortfolioConstraints::long_only(n),
    };

    // Use risk aversion = 2 as default (common in practice)
    optimize_constrained(cov, Some(&excess), &constraints, 2.0, 1000, 1e-8)
}

// ============ Python Bindings ============

/// Constrained minimum variance portfolio (Python)
///
/// # Arguments
/// * `cov` - Covariance matrix
/// * `w_min` - Minimum weights (default: 0 for all, long-only)
/// * `w_max` - Maximum weights (default: 1 for all)
///
/// # Returns
/// Optimal weights satisfying constraints
#[pyfunction]
#[pyo3(name = "min_variance_constrained", signature = (cov, w_min=None, w_max=None))]
pub fn min_variance_constrained_py<'py>(
    py: Python<'py>,
    cov: PyReadonlyArray2<f64>,
    w_min: Option<PyReadonlyArray1<f64>>,
    w_max: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let c = cov.as_array();
    let n = c.nrows();

    let cov_mat = DMatrix::from_row_slice(n, n, c.as_slice().unwrap());

    let w_min_slice: Option<Vec<f64>> = w_min.map(|w| w.as_array().to_vec());
    let w_max_slice: Option<Vec<f64>> = w_max.map(|w| w.as_array().to_vec());

    let weights = min_variance_constrained(
        &cov_mat,
        w_min_slice.as_deref(),
        w_max_slice.as_deref(),
    );

    Ok(PyArray1::from_slice_bound(py, weights.as_slice()))
}

/// Constrained maximum Sharpe ratio portfolio (Python)
///
/// # Arguments
/// * `cov` - Covariance matrix
/// * `expected_returns` - Expected returns vector
/// * `risk_free_rate` - Risk-free rate (default: 0.0)
/// * `w_min` - Minimum weights (default: 0 for all)
/// * `w_max` - Maximum weights (default: 1 for all)
#[pyfunction]
#[pyo3(name = "max_sharpe_constrained", signature = (cov, expected_returns, risk_free_rate=0.0, w_min=None, w_max=None))]
pub fn max_sharpe_constrained_py<'py>(
    py: Python<'py>,
    cov: PyReadonlyArray2<f64>,
    expected_returns: PyReadonlyArray1<f64>,
    risk_free_rate: f64,
    w_min: Option<PyReadonlyArray1<f64>>,
    w_max: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let c = cov.as_array();
    let n = c.nrows();

    let cov_mat = DMatrix::from_row_slice(n, n, c.as_slice().unwrap());
    let mu = DVector::from_row_slice(expected_returns.as_array().as_slice().unwrap());

    let w_min_slice: Option<Vec<f64>> = w_min.map(|w| w.as_array().to_vec());
    let w_max_slice: Option<Vec<f64>> = w_max.map(|w| w.as_array().to_vec());

    let weights = max_sharpe_constrained(
        &cov_mat,
        &mu,
        risk_free_rate,
        w_min_slice.as_deref(),
        w_max_slice.as_deref(),
    );

    Ok(PyArray1::from_slice_bound(py, weights.as_slice()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_cov() -> DMatrix<f64> {
        DMatrix::from_row_slice(3, 3, &[
            0.04, 0.01, 0.005,
            0.01, 0.09, 0.015,
            0.005, 0.015, 0.02,
        ])
    }

    #[test]
    fn test_project_weights_sums_to_one() {
        let w = DVector::from_vec(vec![0.5, 0.3, 0.4]); // Sums to 1.2
        let constraints = PortfolioConstraints::long_only(3);

        let projected = project_weights(&w, &constraints);
        let sum: f64 = projected.iter().sum();

        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_project_weights_respects_bounds() {
        let w = DVector::from_vec(vec![-0.1, 0.8, 0.5]); // Negative weight
        let constraints = PortfolioConstraints::long_only(3);

        let projected = project_weights(&w, &constraints);

        // All weights should be >= 0
        for w_i in projected.iter() {
            assert!(*w_i >= 0.0);
        }
    }

    #[test]
    fn test_min_variance_long_only() {
        let cov = create_test_cov();
        let weights = min_variance_constrained(&cov, None, None);

        // Weights should sum to 1
        let sum: f64 = weights.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);

        // All weights should be >= 0 (long-only)
        for w in weights.iter() {
            assert!(*w >= -1e-6); // Small tolerance for numerical errors
        }
    }

    #[test]
    fn test_min_variance_with_max_constraint() {
        let cov = create_test_cov();
        let w_max = vec![0.4, 0.4, 0.4]; // Max 40% per asset

        let weights = min_variance_constrained(&cov, None, Some(&w_max));

        // All weights should be <= 0.4
        for w in weights.iter() {
            assert!(*w <= 0.4 + 1e-6);
        }

        // Weights should sum to 1
        let sum: f64 = weights.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_max_sharpe_constrained() {
        let cov = create_test_cov();
        let mu = DVector::from_vec(vec![0.08, 0.12, 0.06]); // Asset 2 has highest return

        let weights = max_sharpe_constrained(&cov, &mu, 0.02, None, None);

        // Weights should sum to 1
        let sum: f64 = weights.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);

        // All weights should be >= 0
        for w in weights.iter() {
            assert!(*w >= -1e-6);
        }
    }

    #[test]
    fn test_constrained_vs_unconstrained() {
        // With tight constraints, solution should differ from unconstrained
        let cov = DMatrix::from_row_slice(2, 2, &[
            0.04, 0.0,   // Low vol asset
            0.0, 0.25,   // High vol asset
        ]);

        // Unconstrained min variance would put all weight on low-vol asset
        // But with max 60% constraint, it must diversify
        let w_max = vec![0.6, 0.6];
        let weights = min_variance_constrained(&cov, None, Some(&w_max));

        // Should have exactly 60% in low-vol (binding constraint)
        assert_relative_eq!(weights[0], 0.6, epsilon = 0.01);
        assert_relative_eq!(weights[1], 0.4, epsilon = 0.01);
    }
}
