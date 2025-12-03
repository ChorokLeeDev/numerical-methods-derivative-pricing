//! Extreme Value Theory (EVT) for Tail Risk
//!
//! # The Problem
//!
//! Normal distributions dramatically underestimate tail risk.
//! Fat tails (kurtosis > 3) are the norm in financial returns.
//!
//! ```text
//! Example: Daily returns
//! - Normal predicts 4σ event: 1 in 31,574 days (122 years)
//! - Reality: Happens multiple times per decade!
//! ```
//!
//! # EVT Solution
//!
//! Extreme Value Theory provides mathematical tools for modeling tails:
//!
//! ## Peaks Over Threshold (POT) Method
//!
//! For exceedances over a high threshold u, the excess distribution
//! converges to a Generalized Pareto Distribution (GPD):
//!
//! ```text
//! F_u(y) = P(X - u ≤ y | X > u)
//!
//! GPD parameters:
//!   ξ (xi) = shape parameter (tail heaviness)
//!   β (beta) = scale parameter
//!
//! If ξ > 0: Heavy tail (Pareto-like)
//! If ξ = 0: Exponential tail
//! If ξ < 0: Bounded tail
//! ```
//!
//! ## VaR and ES from GPD
//!
//! ```text
//! VaR_α = u + (β/ξ) × [(n/N_u × (1-α))^(-ξ) - 1]
//!
//! ES_α = VaR_α/(1-ξ) + (β - ξu)/(1-ξ)
//! ```
//!
//! # Industry Usage
//!
//! - Basel III regulatory capital (99.9% confidence)
//! - Hedge fund risk management
//! - Insurance catastrophe modeling
//! - Operational risk (rare but severe events)

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Generalized Pareto Distribution parameters
#[derive(Debug, Clone)]
pub struct GPDParams {
    /// Shape parameter (ξ): controls tail heaviness
    /// ξ > 0: heavy tail, ξ = 0: exponential, ξ < 0: bounded
    pub xi: f64,
    /// Scale parameter (β): spread of the distribution
    pub beta: f64,
    /// Threshold used
    pub threshold: f64,
    /// Number of exceedances
    pub n_exceedances: usize,
    /// Total observations
    pub n_total: usize,
}

impl GPDParams {
    /// Calculate VaR at given confidence level
    ///
    /// VaR_α = u + (β/ξ) × [(n/N_u × (1-α))^(-ξ) - 1]
    pub fn var(&self, alpha: f64) -> f64 {
        let n = self.n_total as f64;
        let n_u = self.n_exceedances as f64;

        if self.xi.abs() < 1e-8 {
            // Exponential case (ξ → 0)
            self.threshold + self.beta * (n / n_u * (1.0 - alpha)).ln()
        } else {
            let p = n / n_u * (1.0 - alpha);
            self.threshold + (self.beta / self.xi) * (p.powf(-self.xi) - 1.0)
        }
    }

    /// Calculate Expected Shortfall (CVaR) at given confidence level
    ///
    /// ES_α = VaR_α/(1-ξ) + (β - ξu)/(1-ξ)
    pub fn expected_shortfall(&self, alpha: f64) -> f64 {
        let var = self.var(alpha);

        if self.xi >= 1.0 {
            // ES undefined for ξ ≥ 1 (infinite mean)
            f64::INFINITY
        } else {
            let one_minus_xi = 1.0 - self.xi;
            var / one_minus_xi + (self.beta - self.xi * self.threshold) / one_minus_xi
        }
    }
}

/// Fit GPD using Peaks Over Threshold method
///
/// # Arguments
/// * `returns` - Historical returns (negative values for losses)
/// * `threshold_quantile` - Quantile for threshold selection (e.g., 0.95)
///
/// # Returns
/// GPD parameters fitted to exceedances over threshold
pub fn fit_gpd(returns: &[f64], threshold_quantile: f64) -> GPDParams {
    let n = returns.len();

    // Convert to losses (positive values)
    let mut losses: Vec<f64> = returns.iter().map(|r| -r).collect();
    losses.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Find threshold
    let threshold_idx = ((n as f64) * threshold_quantile) as usize;
    let threshold = losses[threshold_idx.min(n - 1)];

    // Get exceedances
    let exceedances: Vec<f64> = losses.iter()
        .filter(|&&x| x > threshold)
        .map(|&x| x - threshold)
        .collect();

    let n_exceed = exceedances.len();

    if n_exceed < 10 {
        // Not enough exceedances for reliable estimation
        // Fall back to simple estimates
        return GPDParams {
            xi: 0.0,
            beta: exceedances.iter().sum::<f64>() / n_exceed as f64,
            threshold,
            n_exceedances: n_exceed,
            n_total: n,
        };
    }

    // Maximum Likelihood Estimation for GPD
    // Using Probability Weighted Moments (PWM) method - more stable
    let (xi, beta) = fit_gpd_pwm(&exceedances);

    GPDParams {
        xi,
        beta,
        threshold,
        n_exceedances: n_exceed,
        n_total: n,
    }
}

/// Fit GPD using Probability Weighted Moments
///
/// More stable than MLE for small samples
fn fit_gpd_pwm(exceedances: &[f64]) -> (f64, f64) {
    let n = exceedances.len() as f64;

    // Sort exceedances
    let mut sorted = exceedances.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate probability weighted moments
    // M_0 = mean(X)
    // M_1 = mean(X × (1 - F))
    let m0: f64 = sorted.iter().sum::<f64>() / n;

    let m1: f64 = sorted.iter().enumerate()
        .map(|(i, &x)| {
            let f_i = (i as f64 + 0.35) / n;  // Plotting position
            x * (1.0 - f_i)
        })
        .sum::<f64>() / n;

    // PWM estimators for GPD
    // ξ = 2 - M_0 / (M_0 - 2M_1)
    // β = 2M_0M_1 / (M_0 - 2M_1)

    let denom = m0 - 2.0 * m1;

    if denom.abs() < 1e-10 {
        // Fallback to exponential distribution
        return (0.0, m0);
    }

    let xi = 2.0 - m0 / denom;
    let beta = 2.0 * m0 * m1 / denom;

    // Constrain xi to reasonable range
    let xi = xi.clamp(-0.5, 1.0);
    let beta = beta.max(1e-8);

    (xi, beta)
}

/// Calculate tail index using Hill estimator
///
/// Hill estimator for heavy-tailed distributions:
/// α = 1/ξ = k / Σ(log(X_i) - log(X_k))
///
/// where X_1 ≥ X_2 ≥ ... ≥ X_n are order statistics
pub fn hill_estimator(returns: &[f64], k: usize) -> f64 {
    let mut losses: Vec<f64> = returns.iter()
        .map(|r| -r)
        .filter(|&x| x > 0.0)
        .collect();

    losses.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Descending

    if losses.len() <= k || k == 0 {
        return 0.0;
    }

    let x_k = losses[k - 1];
    if x_k <= 0.0 {
        return 0.0;
    }

    let log_x_k = x_k.ln();
    let sum_logs: f64 = losses[..k].iter()
        .map(|x| x.ln() - log_x_k)
        .sum();

    // Hill estimator returns tail index α = 1/ξ
    // Return ξ for consistency with GPD
    sum_logs / k as f64
}

/// EVT-based VaR calculation
///
/// More accurate than parametric VaR for extreme quantiles
pub fn evt_var(returns: &[f64], alpha: f64, threshold_quantile: f64) -> f64 {
    let gpd = fit_gpd(returns, threshold_quantile);
    gpd.var(alpha)
}

/// EVT-based Expected Shortfall
pub fn evt_es(returns: &[f64], alpha: f64, threshold_quantile: f64) -> f64 {
    let gpd = fit_gpd(returns, threshold_quantile);
    gpd.expected_shortfall(alpha)
}

/// Tail risk metrics combining multiple EVT approaches
#[derive(Debug)]
pub struct TailRiskMetrics {
    pub gpd_var_95: f64,
    pub gpd_var_99: f64,
    pub gpd_var_999: f64,
    pub gpd_es_95: f64,
    pub gpd_es_99: f64,
    pub gpd_es_999: f64,
    pub tail_index: f64,  // ξ from GPD
    pub hill_index: f64,  // ξ from Hill estimator
    pub threshold: f64,
    pub n_exceedances: usize,
}

/// Calculate comprehensive tail risk metrics
pub fn tail_risk_analysis(returns: &[f64]) -> TailRiskMetrics {
    let gpd = fit_gpd(returns, 0.95);

    // Hill estimator with top 5% of observations
    let k = (returns.len() as f64 * 0.05) as usize;
    let hill_xi = hill_estimator(returns, k.max(10));

    TailRiskMetrics {
        gpd_var_95: gpd.var(0.95),
        gpd_var_99: gpd.var(0.99),
        gpd_var_999: gpd.var(0.999),
        gpd_es_95: gpd.expected_shortfall(0.95),
        gpd_es_99: gpd.expected_shortfall(0.99),
        gpd_es_999: gpd.expected_shortfall(0.999),
        tail_index: gpd.xi,
        hill_index: hill_xi,
        threshold: gpd.threshold,
        n_exceedances: gpd.n_exceedances,
    }
}

// ============ Python Bindings ============

/// Fit GPD to returns and get parameters (Python)
///
/// # Arguments
/// * `returns` - Historical returns array
/// * `threshold_quantile` - Quantile for threshold (default 0.95)
///
/// # Returns
/// Tuple of (xi, beta, threshold, n_exceedances)
#[pyfunction]
#[pyo3(name = "fit_gpd", signature = (returns, threshold_quantile=0.95))]
pub fn fit_gpd_py(
    returns: PyReadonlyArray1<f64>,
    threshold_quantile: f64,
) -> PyResult<(f64, f64, f64, usize)> {
    let r = returns.as_array();
    let gpd = fit_gpd(r.as_slice().unwrap(), threshold_quantile);

    Ok((gpd.xi, gpd.beta, gpd.threshold, gpd.n_exceedances))
}

/// Calculate EVT-based VaR (Python)
///
/// # Arguments
/// * `returns` - Historical returns
/// * `alpha` - Confidence level (e.g., 0.99 for 99% VaR)
/// * `threshold_quantile` - Quantile for threshold selection
///
/// # Returns
/// VaR value (positive = loss)
#[pyfunction]
#[pyo3(name = "evt_var", signature = (returns, alpha=0.99, threshold_quantile=0.95))]
pub fn evt_var_py(
    returns: PyReadonlyArray1<f64>,
    alpha: f64,
    threshold_quantile: f64,
) -> PyResult<f64> {
    let r = returns.as_array();
    Ok(evt_var(r.as_slice().unwrap(), alpha, threshold_quantile))
}

/// Calculate EVT-based Expected Shortfall (Python)
///
/// # Arguments
/// * `returns` - Historical returns
/// * `alpha` - Confidence level (e.g., 0.99)
/// * `threshold_quantile` - Quantile for threshold selection
///
/// # Returns
/// ES value (positive = loss)
#[pyfunction]
#[pyo3(name = "evt_es", signature = (returns, alpha=0.99, threshold_quantile=0.95))]
pub fn evt_es_py(
    returns: PyReadonlyArray1<f64>,
    alpha: f64,
    threshold_quantile: f64,
) -> PyResult<f64> {
    let r = returns.as_array();
    Ok(evt_es(r.as_slice().unwrap(), alpha, threshold_quantile))
}

/// Calculate Hill tail index (Python)
///
/// # Arguments
/// * `returns` - Historical returns
/// * `k` - Number of top observations to use
///
/// # Returns
/// Tail index ξ (higher = heavier tail)
#[pyfunction]
#[pyo3(name = "hill_tail_index", signature = (returns, k=None))]
pub fn hill_tail_index_py(
    returns: PyReadonlyArray1<f64>,
    k: Option<usize>,
) -> PyResult<f64> {
    let r = returns.as_array();
    let n = r.len();
    let k = k.unwrap_or((n as f64 * 0.05) as usize).max(10);

    Ok(hill_estimator(r.as_slice().unwrap(), k))
}

/// Comprehensive tail risk analysis (Python)
///
/// # Returns
/// Tuple of (var_95, var_99, var_999, es_95, es_99, es_999, tail_index)
#[pyfunction]
#[pyo3(name = "tail_risk_analysis")]
pub fn tail_risk_analysis_py(
    returns: PyReadonlyArray1<f64>,
) -> PyResult<(f64, f64, f64, f64, f64, f64, f64)> {
    let r = returns.as_array();
    let metrics = tail_risk_analysis(r.as_slice().unwrap());

    Ok((
        metrics.gpd_var_95,
        metrics.gpd_var_99,
        metrics.gpd_var_999,
        metrics.gpd_es_95,
        metrics.gpd_es_99,
        metrics.gpd_es_999,
        metrics.tail_index,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_fat_tail_returns(n: usize, seed: u64) -> Vec<f64> {
        // Simple LCG for reproducibility
        let mut state = seed;
        let mut returns = Vec::with_capacity(n);

        for _ in 0..n {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let u1 = (state % 10000) as f64 / 10000.0;
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let u2 = (state % 10000) as f64 / 10000.0;

            // Box-Muller with occasional large moves
            let z = (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

            // Add fat tail by mixing with larger volatility occasionally
            let r = if u1 < 0.05 {
                z * 0.03 * 3.0  // 5% chance of 3x volatility
            } else {
                z * 0.01
            };

            returns.push(r);
        }

        returns
    }

    #[test]
    fn test_gpd_fit() {
        let returns = generate_fat_tail_returns(1000, 42);
        let gpd = fit_gpd(&returns, 0.95);

        // Should have reasonable parameters
        assert!(gpd.xi > -1.0 && gpd.xi < 2.0, "xi should be reasonable: {}", gpd.xi);
        assert!(gpd.beta > 0.0, "beta should be positive");
        assert!(gpd.n_exceedances > 0, "should have exceedances");
    }

    #[test]
    fn test_evt_var_increasing() {
        let returns = generate_fat_tail_returns(1000, 42);

        let var_95 = evt_var(&returns, 0.95, 0.90);
        let var_99 = evt_var(&returns, 0.99, 0.90);
        let var_999 = evt_var(&returns, 0.999, 0.90);

        // VaR should increase with confidence level
        assert!(var_99 > var_95, "99% VaR should exceed 95% VaR");
        assert!(var_999 > var_99, "99.9% VaR should exceed 99% VaR");
    }

    #[test]
    fn test_es_exceeds_var() {
        let returns = generate_fat_tail_returns(1000, 42);
        let gpd = fit_gpd(&returns, 0.95);

        // For any alpha, ES should exceed VaR
        for alpha in [0.95, 0.99, 0.999] {
            let var = gpd.var(alpha);
            let es = gpd.expected_shortfall(alpha);

            if gpd.xi < 1.0 {
                assert!(es >= var, "ES should exceed VaR at alpha={}", alpha);
            }
        }
    }

    #[test]
    fn test_hill_estimator() {
        let returns = generate_fat_tail_returns(1000, 42);
        let xi = hill_estimator(&returns, 50);

        // Should return a reasonable tail index
        assert!(xi.is_finite(), "Hill estimator should be finite");
    }
}
