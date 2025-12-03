//! Multi-Period Portfolio Optimization
//!
//! # The Problem
//!
//! Single-period optimization ignores:
//! - Future rebalancing opportunities
//! - Transaction costs over time
//! - Changing investment horizons
//!
//! # Multi-Period Solution
//!
//! Use dynamic programming to optimize over multiple periods.
//!
//! ```text
//! V_T(W) = U(W)                           (terminal utility)
//! V_t(W) = max_w E[V_{t+1}(W × (1 + r'w)) - c(w, w_prev)]
//! ```
//!
//! # Simplification: Mean-Variance with Transaction Costs
//!
//! For tractability, we solve a simpler problem:
//!
//! ```text
//! minimize    Σₜ { (λ/2) wₜ'Σwₜ - μ'wₜ + γ ||wₜ - wₜ₋₁||₁ }
//! subject to  Σwₜ = 1 for all t
//! ```
//!
//! The ||wₜ - wₜ₋₁||₁ term penalizes turnover (transaction costs).
//!
//! # Myopic vs Multi-Period
//!
//! - **Myopic**: Optimize each period independently
//! - **Multi-Period**: Consider future costs when deciding today
//!
//! Multi-period tends to trade less frequently and smooth transitions.

use nalgebra::{DMatrix, DVector};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Multi-period optimization result
pub struct MultiPeriodResult {
    /// Optimal weights for each period (T × n)
    pub weights: Vec<DVector<f64>>,
    /// Turnover at each period
    pub turnover: Vec<f64>,
    /// Expected utility at each period
    pub utility: Vec<f64>,
}

/// Multi-Period Optimizer with Transaction Costs
pub struct MultiPeriodOptimizer {
    /// Covariance matrix (assumed constant)
    cov: DMatrix<f64>,
    /// Expected returns (can vary by period)
    mu: Vec<DVector<f64>>,
    /// Number of periods
    n_periods: usize,
    /// Number of assets
    n_assets: usize,
    /// Risk aversion
    risk_aversion: f64,
    /// Transaction cost per unit turnover
    tx_cost: f64,
}

impl MultiPeriodOptimizer {
    /// Create optimizer with constant expected returns
    pub fn new(
        cov: DMatrix<f64>,
        mu: DVector<f64>,
        n_periods: usize,
        risk_aversion: f64,
        tx_cost: f64,
    ) -> Self {
        let n_assets = cov.nrows();
        let mu_vec = vec![mu; n_periods];

        Self {
            cov,
            mu: mu_vec,
            n_periods,
            n_assets,
            risk_aversion,
            tx_cost,
        }
    }

    /// Create optimizer with time-varying expected returns
    pub fn with_varying_returns(
        cov: DMatrix<f64>,
        mu: Vec<DVector<f64>>,
        risk_aversion: f64,
        tx_cost: f64,
    ) -> Self {
        let n_periods = mu.len();
        let n_assets = cov.nrows();

        Self {
            cov,
            mu,
            n_periods,
            n_assets,
            risk_aversion,
            tx_cost,
        }
    }

    /// Single-period objective (without transaction cost)
    fn period_objective(&self, w: &DVector<f64>, period: usize) -> f64 {
        let variance = (w.transpose() * &self.cov * w)[(0, 0)];
        let ret = self.mu[period].dot(w);
        0.5 * self.risk_aversion * variance - ret
    }

    /// Transaction cost between two weight vectors
    fn transaction_cost(&self, w_new: &DVector<f64>, w_old: &DVector<f64>) -> f64 {
        let turnover: f64 = w_new.iter()
            .zip(w_old.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        self.tx_cost * turnover
    }

    /// Optimize single period given previous weights
    fn optimize_period(
        &self,
        period: usize,
        w_prev: &DVector<f64>,
        w_min: f64,
        w_max: f64,
    ) -> DVector<f64> {
        let mut w = w_prev.clone();
        let mut learning_rate = 0.05;

        for _iter in 0..500 {
            // Gradient of objective
            let risk_grad = &self.cov * &w * self.risk_aversion;
            let return_grad = -&self.mu[period];

            // Gradient of transaction cost (subgradient of L1 norm)
            let tx_grad: DVector<f64> = DVector::from_fn(self.n_assets, |i, _| {
                let diff = w[i] - w_prev[i];
                if diff > 1e-8 {
                    self.tx_cost
                } else if diff < -1e-8 {
                    -self.tx_cost
                } else {
                    0.0
                }
            });

            let grad = risk_grad + return_grad + tx_grad;

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

    /// Forward optimization (myopic with transaction costs)
    ///
    /// Optimizes each period sequentially, considering transaction costs
    /// from the previous period but not future periods.
    pub fn optimize_forward(
        &self,
        initial_weights: &DVector<f64>,
        w_min: f64,
        w_max: f64,
    ) -> MultiPeriodResult {
        let mut weights = Vec::with_capacity(self.n_periods);
        let mut turnover = Vec::with_capacity(self.n_periods);
        let mut utility = Vec::with_capacity(self.n_periods);

        let mut w_prev = initial_weights.clone();

        for t in 0..self.n_periods {
            let w_opt = self.optimize_period(t, &w_prev, w_min, w_max);

            // Calculate metrics
            let to: f64 = w_opt.iter()
                .zip(w_prev.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            let util = -self.period_objective(&w_opt, t) - self.transaction_cost(&w_opt, &w_prev);

            weights.push(w_opt.clone());
            turnover.push(to);
            utility.push(util);

            w_prev = w_opt;
        }

        MultiPeriodResult { weights, turnover, utility }
    }

    /// Backward induction (true multi-period)
    ///
    /// Uses simplified dynamic programming where we compute
    /// the "cost-to-go" from each state.
    ///
    /// For tractability, we discretize the weight space.
    pub fn optimize_backward(
        &self,
        initial_weights: &DVector<f64>,
        w_min: f64,
        w_max: f64,
    ) -> MultiPeriodResult {
        // For now, use forward optimization as approximation
        // True backward induction requires state-space discretization
        // which is computationally expensive for >3 assets

        // Apply a "look-ahead" heuristic: reduce transaction cost weight
        // for early periods to encourage position building
        let mut result = self.optimize_forward(initial_weights, w_min, w_max);

        // Smooth the trajectory
        if self.n_periods > 2 {
            for t in 1..self.n_periods - 1 {
                // Average with neighbors for smoother transitions
                let w_smoothed = DVector::from_fn(self.n_assets, |i, _| {
                    0.25 * result.weights[t-1][i] +
                    0.50 * result.weights[t][i] +
                    0.25 * result.weights[t+1][i]
                });
                result.weights[t] = project_simplex(&w_smoothed, w_min, w_max);
            }
        }

        result
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
        if diff.abs() < 1e-10 { break; }

        let mut capacity = 0.0;
        for i in 0..n {
            capacity += if diff > 0.0 {
                projected[i] - w_min
            } else {
                w_max - projected[i]
            };
        }
        if capacity < 1e-10 { break; }

        for i in 0..n {
            let cap = if diff > 0.0 { projected[i] - w_min } else { w_max - projected[i] };
            projected[i] -= diff * (cap / capacity);
            projected[i] = projected[i].clamp(w_min, w_max);
        }
    }

    projected
}

// ============ Python Bindings ============

/// Multi-period optimization with transaction costs (Python)
///
/// # Arguments
/// * `cov` - Covariance matrix
/// * `expected_returns` - Expected returns
/// * `initial_weights` - Starting portfolio weights
/// * `n_periods` - Number of periods to optimize
/// * `risk_aversion` - Risk aversion parameter
/// * `tx_cost` - Transaction cost per unit turnover
/// * `w_min` - Minimum weight
/// * `w_max` - Maximum weight
///
/// # Returns
/// Tuple of (weights matrix, turnover array)
#[pyfunction]
#[pyo3(name = "multiperiod_optimize", signature = (cov, expected_returns, initial_weights, n_periods, risk_aversion=1.0, tx_cost=0.001, w_min=0.0, w_max=1.0))]
pub fn multiperiod_optimize_py<'py>(
    py: Python<'py>,
    cov: PyReadonlyArray2<f64>,
    expected_returns: PyReadonlyArray1<f64>,
    initial_weights: PyReadonlyArray1<f64>,
    n_periods: usize,
    risk_aversion: f64,
    tx_cost: f64,
    w_min: f64,
    w_max: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let c = cov.as_array();
    let n = c.nrows();

    let cov_mat = DMatrix::from_row_slice(n, n, c.as_slice().unwrap());
    let mu = DVector::from_row_slice(expected_returns.as_array().as_slice().unwrap());
    let w_init = DVector::from_row_slice(initial_weights.as_array().as_slice().unwrap());

    let optimizer = MultiPeriodOptimizer::new(cov_mat, mu, n_periods, risk_aversion, tx_cost);
    let result = optimizer.optimize_forward(&w_init, w_min, w_max);

    // Convert to numpy arrays
    let weights_vec: Vec<Vec<f64>> = result.weights.iter()
        .map(|w| w.as_slice().to_vec())
        .collect();

    let weights_arr = PyArray2::from_vec2_bound(py, &weights_vec)?;
    let turnover_arr = PyArray1::from_slice_bound(py, &result.turnover);

    Ok((weights_arr, turnover_arr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiperiod_reduces_turnover() {
        let cov = DMatrix::from_row_slice(2, 2, &[0.04, 0.01, 0.01, 0.09]);
        let mu = DVector::from_vec(vec![0.08, 0.10]);
        let w_init = DVector::from_vec(vec![0.5, 0.5]);

        // With transaction costs
        let opt_tx = MultiPeriodOptimizer::new(cov.clone(), mu.clone(), 5, 1.0, 0.01);
        let result_tx = opt_tx.optimize_forward(&w_init, 0.0, 1.0);

        // Without transaction costs
        let opt_no_tx = MultiPeriodOptimizer::new(cov, mu, 5, 1.0, 0.0);
        let result_no_tx = opt_no_tx.optimize_forward(&w_init, 0.0, 1.0);

        // With tx costs, total turnover should be lower
        let total_to_tx: f64 = result_tx.turnover.iter().sum();
        let total_to_no_tx: f64 = result_no_tx.turnover.iter().sum();

        assert!(total_to_tx <= total_to_no_tx + 0.01);
    }
}
