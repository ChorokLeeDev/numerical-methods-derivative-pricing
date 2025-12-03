//! CVaR (Conditional Value at Risk) Portfolio Optimization
//!
//! # The Problem
//!
//! Mean-Variance minimizes variance, but variance:
//! - Penalizes upside and downside equally
//! - Doesn't focus on tail risk (extreme losses)
//!
//! # CVaR Solution
//!
//! Minimize the expected loss in the worst α% of scenarios.
//!
//! ```text
//! CVaR_α = E[Loss | Loss > VaR_α]
//!        = "Average loss when things go really bad"
//! ```
//!
//! # Mathematical Formulation (Rockafellar & Uryasev, 2000)
//!
//! Key insight: CVaR can be computed via:
//!
//! ```text
//! CVaR_α(w) = min_γ { γ + (1/α) E[max(-r'w - γ, 0)] }
//! ```
//!
//! With T scenarios, this becomes a linear program:
//!
//! ```text
//! minimize    γ + (1/αT) Σᵢ uᵢ
//! subject to  uᵢ ≥ -rᵢ'w - γ    (for each scenario i)
//!             uᵢ ≥ 0
//!             Σw = 1
//!             w ≥ 0              (long-only)
//! ```
//!
//! Since we don't have an LP solver, we use scenario-based gradient descent.

use nalgebra::{DMatrix, DVector};
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Calculate CVaR for a given portfolio
///
/// CVaR_α = average of worst α% returns (negated for loss)
pub fn calculate_cvar(returns: &[f64], alpha: f64) -> f64 {
    if returns.is_empty() || alpha <= 0.0 || alpha >= 1.0 {
        return f64::NAN;
    }

    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n_tail = ((alpha * returns.len() as f64).ceil() as usize).max(1);
    let tail_sum: f64 = sorted.iter().take(n_tail).sum();

    -tail_sum / (n_tail as f64)
}

/// Calculate portfolio returns from scenarios
fn portfolio_returns(weights: &DVector<f64>, scenarios: &DMatrix<f64>) -> Vec<f64> {
    let t = scenarios.nrows();
    (0..t)
        .map(|i| {
            let row = scenarios.row(i);
            weights.iter().zip(row.iter()).map(|(w, r)| w * r).sum()
        })
        .collect()
}

/// Gradient of CVaR with respect to weights
///
/// ∂CVaR/∂w = -E[r | r is in worst α%]
fn cvar_gradient(
    weights: &DVector<f64>,
    scenarios: &DMatrix<f64>,
    alpha: f64,
) -> DVector<f64> {
    let n = weights.len();
    let t = scenarios.nrows();

    // Calculate portfolio returns for each scenario
    let port_returns = portfolio_returns(weights, scenarios);

    // Find the worst α% scenarios
    let mut indexed: Vec<(usize, f64)> = port_returns.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let n_tail = ((alpha * t as f64).ceil() as usize).max(1);

    // Average gradient over tail scenarios
    let mut grad = DVector::zeros(n);
    for (idx, _) in indexed.iter().take(n_tail) {
        let row = scenarios.row(*idx);
        for j in 0..n {
            grad[j] -= row[j]; // Negative because we want to minimize CVaR (maximize return)
        }
    }
    grad / (n_tail as f64)
}

/// Project weights onto constraint set
fn project_simplex(w: &DVector<f64>, w_min: f64, w_max: f64) -> DVector<f64> {
    let n = w.len();
    let mut projected = w.clone();

    // Clip to bounds
    for i in 0..n {
        projected[i] = projected[i].clamp(w_min, w_max);
    }

    // Normalize to sum to 1
    for _ in 0..100 {
        let sum: f64 = projected.iter().sum();
        let diff = sum - 1.0;

        if diff.abs() < 1e-10 {
            break;
        }

        // Distribute difference
        let mut capacity = 0.0;
        for i in 0..n {
            capacity += if diff > 0.0 {
                projected[i] - w_min
            } else {
                w_max - projected[i]
            };
        }

        if capacity < 1e-10 {
            break;
        }

        for i in 0..n {
            let cap = if diff > 0.0 {
                projected[i] - w_min
            } else {
                w_max - projected[i]
            };
            projected[i] -= diff * (cap / capacity);
            projected[i] = projected[i].clamp(w_min, w_max);
        }
    }

    projected
}

/// Minimize CVaR using scenario-based gradient descent
///
/// # Arguments
/// * `scenarios` - Historical return scenarios (T × n matrix)
/// * `alpha` - CVaR confidence level (e.g., 0.05 for worst 5%)
/// * `w_min` - Minimum weight per asset
/// * `w_max` - Maximum weight per asset
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
/// Optimal portfolio weights that minimize CVaR
pub fn minimize_cvar(
    scenarios: &DMatrix<f64>,
    alpha: f64,
    w_min: f64,
    w_max: f64,
    max_iter: usize,
    tol: f64,
) -> DVector<f64> {
    let n = scenarios.ncols();

    // Initialize with equal weights
    let mut w = DVector::from_element(n, 1.0 / n as f64);
    w = project_simplex(&w, w_min, w_max);

    let mut learning_rate = 0.1;
    let mut prev_cvar = calculate_cvar(&portfolio_returns(&w, scenarios), alpha);

    for _iter in 0..max_iter {
        // Calculate gradient
        let grad = cvar_gradient(&w, scenarios, alpha);

        // Gradient descent step
        let w_new = &w - &grad * learning_rate;
        let w_projected = project_simplex(&w_new, w_min, w_max);

        // Calculate new CVaR
        let new_cvar = calculate_cvar(&portfolio_returns(&w_projected, scenarios), alpha);

        // Line search
        if new_cvar > prev_cvar + 1e-10 {
            learning_rate *= 0.5;
            if learning_rate < 1e-10 {
                break;
            }
            continue;
        }

        // Check convergence
        let change: f64 = (&w_projected - &w).iter().map(|x| x.abs()).sum();
        if change < tol {
            return w_projected;
        }

        w = w_projected;
        prev_cvar = new_cvar;

        // Occasionally increase learning rate
        if _iter % 20 == 0 {
            learning_rate *= 1.2;
        }
    }

    w
}

/// Mean-CVaR optimization
///
/// Trade off between expected return and CVaR:
///
/// ```text
/// maximize    μ'w - λ × CVaR_α(w)
/// ```
pub fn mean_cvar_optimization(
    scenarios: &DMatrix<f64>,
    expected_returns: &DVector<f64>,
    alpha: f64,
    risk_aversion: f64,
    w_min: f64,
    w_max: f64,
    max_iter: usize,
) -> DVector<f64> {
    let n = scenarios.ncols();

    let mut w = DVector::from_element(n, 1.0 / n as f64);
    w = project_simplex(&w, w_min, w_max);

    let mut learning_rate = 0.1;

    for _iter in 0..max_iter {
        // CVaR gradient
        let cvar_grad = cvar_gradient(&w, scenarios, alpha);

        // Combined gradient: -μ + λ × ∂CVaR/∂w
        let grad = -expected_returns + &cvar_grad * risk_aversion;

        let w_new = &w - &grad * learning_rate;
        let w_projected = project_simplex(&w_new, w_min, w_max);

        let change: f64 = (&w_projected - &w).iter().map(|x| x.abs()).sum();
        if change < 1e-8 {
            return w_projected;
        }

        w = w_projected;
    }

    w
}

// ============ Python Bindings ============

/// Minimize CVaR portfolio (Python)
///
/// Find weights that minimize expected shortfall (average loss in worst scenarios).
///
/// # Arguments
/// * `scenarios` - Historical returns matrix (T scenarios × n assets)
/// * `alpha` - Tail probability (default: 0.05 for worst 5%)
/// * `w_min` - Minimum weight (default: 0.0, long-only)
/// * `w_max` - Maximum weight (default: 1.0)
#[pyfunction]
#[pyo3(name = "minimize_cvar", signature = (scenarios, alpha=0.05, w_min=0.0, w_max=1.0))]
pub fn minimize_cvar_py<'py>(
    py: Python<'py>,
    scenarios: PyReadonlyArray2<f64>,
    alpha: f64,
    w_min: f64,
    w_max: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let s = scenarios.as_array();
    let (t, n) = (s.nrows(), s.ncols());

    let scenarios_mat = DMatrix::from_row_slice(t, n, s.as_slice().unwrap());

    let weights = minimize_cvar(&scenarios_mat, alpha, w_min, w_max, 1000, 1e-8);

    Ok(PyArray1::from_slice_bound(py, weights.as_slice()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_scenarios() -> DMatrix<f64> {
        // 10 scenarios, 3 assets
        DMatrix::from_row_slice(10, 3, &[
            0.05, 0.03, 0.02,   // Good scenario
            0.02, 0.01, 0.01,
            0.01, 0.00, 0.01,
            0.00, -0.01, 0.00,
            -0.01, -0.02, -0.01,
            -0.02, -0.03, -0.01,  // Bad scenario
            -0.03, -0.05, -0.02,  // Worse
            -0.05, -0.08, -0.03,  // Worst for asset 2
            0.03, 0.02, 0.01,
            0.02, 0.01, 0.00,
        ])
    }

    #[test]
    fn test_calculate_cvar() {
        let returns = vec![0.05, 0.02, 0.01, -0.01, -0.03, -0.05, -0.08, 0.03, 0.02, 0.01];
        let cvar = calculate_cvar(&returns, 0.20); // Worst 20% = 2 scenarios

        // Worst 2: -0.08, -0.05 → average = -0.065 → CVaR = 0.065
        assert!(cvar > 0.0); // CVaR is positive (loss)
    }

    #[test]
    fn test_minimize_cvar_sums_to_one() {
        let scenarios = create_scenarios();
        let weights = minimize_cvar(&scenarios, 0.20, 0.0, 1.0, 500, 1e-6);

        let sum: f64 = weights.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_minimize_cvar_avoids_risky_asset() {
        let scenarios = create_scenarios();
        // Asset 2 (index 1) has the worst tail performance (-0.08)

        let weights = minimize_cvar(&scenarios, 0.10, 0.0, 1.0, 500, 1e-6);

        // Should put less weight on asset 2 (highest tail risk)
        // Asset 3 (index 2) is most stable in bad scenarios
        assert!(weights[2] > weights[1] || (weights[1] - weights[2]).abs() < 0.1);
    }
}
