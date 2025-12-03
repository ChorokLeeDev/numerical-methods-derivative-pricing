//! Robust Portfolio Optimization
//!
//! # The Problem
//!
//! Traditional optimization assumes we know μ and Σ exactly.
//! In reality, these are **estimates with uncertainty**.
//!
//! What if our estimates are wrong? The portfolio could perform badly.
//!
//! # Robust Solution
//!
//! Instead of optimizing for the point estimate, optimize for the **worst case**
//! within an uncertainty set.
//!
//! ```text
//! Traditional: maximize μ̂'w - (λ/2) w'Σ̂w
//! Robust:      maximize min_{μ ∈ U} μ'w - (λ/2) w'Σ̂w
//! ```
//!
//! # Uncertainty Set
//!
//! We model uncertainty as an ellipsoid around our estimate:
//!
//! ```text
//! U = {μ : (μ - μ̂)'Σ_μ⁻¹(μ - μ̂) ≤ κ²}
//!
//! where:
//!   μ̂ = point estimate of expected returns
//!   Σ_μ = uncertainty in the estimate (often Σ/T)
//!   κ = size of uncertainty set (controls conservatism)
//! ```
//!
//! # Solution
//!
//! The worst-case μ in the uncertainty set is:
//!
//! ```text
//! μ_worst = μ̂ - κ × Σ_μ^{1/2} × w / ||Σ_μ^{1/2} w||
//! ```
//!
//! This leads to a penalized objective:
//!
//! ```text
//! maximize μ̂'w - κ × ||Σ_μ^{1/2} w|| - (λ/2) w'Σw
//! ```
//!
//! The κ√(w'Σ_μw) term penalizes portfolios with high estimation uncertainty.

use nalgebra::{DMatrix, DVector};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Robust Portfolio Optimizer
///
/// Optimizes for worst-case expected return within uncertainty set.
pub struct RobustOptimizer {
    /// Covariance matrix
    cov: DMatrix<f64>,
    /// Expected returns estimate
    mu: DVector<f64>,
    /// Uncertainty in return estimates (typically Σ/T or diagonal)
    sigma_mu: DMatrix<f64>,
    /// Number of assets
    n: usize,
}

impl RobustOptimizer {
    /// Create new robust optimizer
    pub fn new(
        cov: DMatrix<f64>,
        mu: DVector<f64>,
        n_observations: usize,
    ) -> Self {
        let n = cov.nrows();

        // Uncertainty in mean estimates: Σ_μ = Σ / T
        // (from standard error of mean: σ/√T → variance = σ²/T)
        let sigma_mu = &cov / (n_observations as f64);

        Self { cov, mu, sigma_mu, n }
    }

    /// Set custom uncertainty matrix
    pub fn with_uncertainty(mut self, sigma_mu: DMatrix<f64>) -> Self {
        self.sigma_mu = sigma_mu;
        self
    }

    /// Calculate robust objective function
    ///
    /// f(w) = -μ̂'w + κ × √(w'Σ_μw) + (λ/2) w'Σw
    fn objective(&self, w: &DVector<f64>, kappa: f64, risk_aversion: f64) -> f64 {
        // Expected return term
        let return_term = self.mu.dot(w);

        // Uncertainty penalty: κ × √(w'Σ_μw)
        let uncertainty_var = (w.transpose() * &self.sigma_mu * w)[(0, 0)];
        let uncertainty_penalty = kappa * uncertainty_var.sqrt();

        // Risk term: (λ/2) w'Σw
        let variance = (w.transpose() * &self.cov * w)[(0, 0)];
        let risk_term = 0.5 * risk_aversion * variance;

        // Minimize: -return + penalty + risk
        -return_term + uncertainty_penalty + risk_term
    }

    /// Calculate gradient of robust objective
    fn gradient(&self, w: &DVector<f64>, kappa: f64, risk_aversion: f64) -> DVector<f64> {
        // ∂/∂w[-μ'w] = -μ
        let return_grad = -&self.mu;

        // ∂/∂w[κ√(w'Σ_μw)] = κ × Σ_μw / √(w'Σ_μw)
        let sigma_mu_w = &self.sigma_mu * w;
        let uncertainty_var = (w.transpose() * &self.sigma_mu * w)[(0, 0)];
        let uncertainty_grad = if uncertainty_var > 1e-10 {
            &sigma_mu_w * (kappa / uncertainty_var.sqrt())
        } else {
            DVector::zeros(self.n)
        };

        // ∂/∂w[(λ/2)w'Σw] = λΣw
        let risk_grad = &self.cov * w * risk_aversion;

        return_grad + uncertainty_grad + risk_grad
    }

    /// Optimize with uncertainty aversion
    ///
    /// # Arguments
    /// * `kappa` - Uncertainty aversion (0 = ignore uncertainty, higher = more conservative)
    /// * `risk_aversion` - Risk aversion for variance term
    /// * `w_min` - Minimum weight
    /// * `w_max` - Maximum weight
    pub fn optimize(
        &self,
        kappa: f64,
        risk_aversion: f64,
        w_min: f64,
        w_max: f64,
        max_iter: usize,
    ) -> DVector<f64> {
        let mut w = DVector::from_element(self.n, 1.0 / self.n as f64);
        w = project_simplex(&w, w_min, w_max);

        let mut learning_rate = 0.1;
        let mut prev_obj = self.objective(&w, kappa, risk_aversion);

        for iter in 0..max_iter {
            let grad = self.gradient(&w, kappa, risk_aversion);

            let w_new = &w - &grad * learning_rate;
            let w_projected = project_simplex(&w_new, w_min, w_max);

            let new_obj = self.objective(&w_projected, kappa, risk_aversion);

            if new_obj > prev_obj + 1e-10 {
                learning_rate *= 0.5;
                if learning_rate < 1e-12 {
                    break;
                }
                continue;
            }

            let change: f64 = (&w_projected - &w).iter().map(|x| x.abs()).sum();
            if change < 1e-8 {
                return w_projected;
            }

            w = w_projected;
            prev_obj = new_obj;

            if iter % 20 == 0 {
                learning_rate *= 1.1;
            }
        }

        w
    }
}

/// Project onto simplex with box constraints
fn project_simplex(w: &DVector<f64>, w_min: f64, w_max: f64) -> DVector<f64> {
    let n = w.len();
    let mut projected = w.clone();

    for i in 0..n {
        projected[i] = projected[i].clamp(w_min, w_max);
    }

    for _ in 0..100 {
        let sum: f64 = projected.iter().sum();
        let diff = sum - 1.0;

        if diff.abs() < 1e-10 {
            break;
        }

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

/// Simplified robust optimization (worst-case mean)
///
/// Directly subtracts uncertainty penalty from expected returns:
///
/// ```text
/// μ_robust = μ̂ - κ × σ_μ
///
/// where σ_μ = diagonal of Σ_μ (standard errors)
/// ```
///
/// Then runs standard mean-variance with μ_robust.
pub fn robust_shrink_returns(
    mu: &DVector<f64>,
    cov: &DMatrix<f64>,
    n_observations: usize,
    kappa: f64,
) -> DVector<f64> {
    let n = mu.len();

    // Standard errors of mean estimates
    let se: Vec<f64> = (0..n)
        .map(|i| (cov[(i, i)] / n_observations as f64).sqrt())
        .collect();

    // Shrink returns toward zero
    let mut mu_robust = mu.clone();
    for i in 0..n {
        mu_robust[i] -= kappa * se[i];
    }

    mu_robust
}

// ============ Python Bindings ============

/// Robust portfolio optimization (Python)
///
/// Optimizes for worst-case returns within uncertainty set.
///
/// # Arguments
/// * `cov` - Covariance matrix
/// * `expected_returns` - Expected returns estimate
/// * `n_observations` - Number of observations used for estimation
/// * `kappa` - Uncertainty aversion (default: 1.0)
/// * `risk_aversion` - Risk aversion (default: 1.0)
/// * `w_min` - Minimum weight (default: 0.0)
/// * `w_max` - Maximum weight (default: 1.0)
#[pyfunction]
#[pyo3(name = "robust_optimize", signature = (cov, expected_returns, n_observations, kappa=1.0, risk_aversion=1.0, w_min=0.0, w_max=1.0))]
pub fn robust_optimize_py<'py>(
    py: Python<'py>,
    cov: PyReadonlyArray2<f64>,
    expected_returns: PyReadonlyArray1<f64>,
    n_observations: usize,
    kappa: f64,
    risk_aversion: f64,
    w_min: f64,
    w_max: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let c = cov.as_array();
    let n = c.nrows();

    let cov_mat = DMatrix::from_row_slice(n, n, c.as_slice().unwrap());
    let mu = DVector::from_row_slice(expected_returns.as_array().as_slice().unwrap());

    let optimizer = RobustOptimizer::new(cov_mat, mu, n_observations);
    let weights = optimizer.optimize(kappa, risk_aversion, w_min, w_max, 1000);

    Ok(PyArray1::from_slice_bound(py, weights.as_slice()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_robust_more_conservative() {
        let cov = DMatrix::from_row_slice(2, 2, &[
            0.04, 0.01,
            0.01, 0.09,
        ]);
        let mu = DVector::from_vec(vec![0.10, 0.15]); // Asset 2 has higher return

        // Non-robust (kappa=0) vs Robust (kappa=2)
        let opt_nonrobust = RobustOptimizer::new(cov.clone(), mu.clone(), 60);
        let w_nonrobust = opt_nonrobust.optimize(0.0, 1.0, 0.0, 1.0, 500);

        let opt_robust = RobustOptimizer::new(cov, mu, 60);
        let w_robust = opt_robust.optimize(2.0, 1.0, 0.0, 1.0, 500);

        // Robust should put less weight on high-return asset (more uncertain)
        // Asset 2 has higher variance → higher estimation uncertainty
        assert!(w_robust[1] <= w_nonrobust[1] + 0.1);
    }

    #[test]
    fn test_robust_weights_sum_to_one() {
        let cov = DMatrix::from_row_slice(3, 3, &[
            0.04, 0.01, 0.005,
            0.01, 0.09, 0.01,
            0.005, 0.01, 0.02,
        ]);
        let mu = DVector::from_vec(vec![0.08, 0.12, 0.06]);

        let opt = RobustOptimizer::new(cov, mu, 120);
        let w = opt.optimize(1.5, 1.0, 0.0, 1.0, 500);

        let sum: f64 = w.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_shrink_returns() {
        let mu = DVector::from_vec(vec![0.10, 0.15]);
        let cov = DMatrix::from_row_slice(2, 2, &[0.04, 0.0, 0.0, 0.16]);

        let mu_robust = robust_shrink_returns(&mu, &cov, 100, 2.0);

        // Higher variance asset should be shrunk more
        // Asset 1: σ=0.2, SE=0.2/√100=0.02, shrink=0.04
        // Asset 2: σ=0.4, SE=0.4/√100=0.04, shrink=0.08
        assert!(mu_robust[0] > mu_robust[1]); // After shrinkage, asset 1 might be higher
        assert!(mu_robust[0] < mu[0]); // Both are shrunk down
        assert!(mu_robust[1] < mu[1]);
    }
}
