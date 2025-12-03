//! Value at Risk (VaR) and Conditional VaR (CVaR / Expected Shortfall)
//!
//! # Value at Risk (VaR)
//!
//! VaR answers: "What is the maximum loss at a given confidence level?"
//!
//! ```text
//! VaR_α = "With (1-α)% confidence, I won't lose more than VaR_α"
//!
//! Example: VaR_0.05 = $1M means:
//! "With 95% confidence, I won't lose more than $1M in one day"
//! ```
//!
//! ## Calculation Methods
//!
//! **1. Parametric (Normal) VaR**:
//! ```text
//! VaR_α = μ - z_α × σ
//!
//! where z_α is the α-quantile of standard normal
//! z_0.05 ≈ -1.645 (95% confidence)
//! z_0.01 ≈ -2.326 (99% confidence)
//! ```
//!
//! **2. Historical VaR**:
//! Simply take the α-percentile of historical returns
//!
//! # Conditional VaR (CVaR / Expected Shortfall)
//!
//! CVaR answers: "If I DO exceed VaR, what's my expected loss?"
//!
//! ```text
//! CVaR_α = E[Loss | Loss > VaR_α]
//! ```
//!
//! ## Why CVaR > VaR?
//!
//! 1. VaR is a single point; CVaR captures the tail distribution
//! 2. VaR can miss "black swan" events; CVaR accounts for them
//! 3. CVaR is mathematically "coherent"; VaR is not
//!
//! **Coherence** means the risk measure satisfies:
//! - Monotonicity: More risk = higher measure
//! - Sub-additivity: Diversification reduces risk
//! - Positive homogeneity: Doubling position doubles risk
//! - Translation invariance: Adding cash reduces risk
//!
//! VaR fails sub-additivity! (Two portfolios combined can have higher VaR than sum)

use nalgebra::DVector;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Standard normal quantile (z-score)
///
/// Uses approximation formula for speed:
/// Abramowitz and Stegun approximation 26.2.23
fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }

    // Handle symmetry: if p > 0.5, use 1-p and negate result
    let (p_adj, sign) = if p > 0.5 {
        (1.0 - p, 1.0)
    } else {
        (p, -1.0)
    };

    // Rational approximation for 0 < p <= 0.5
    let t = (-2.0 * p_adj.ln()).sqrt();

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let numerator = c0 + c1 * t + c2 * t * t;
    let denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;

    sign * (t - numerator / denominator)
}

/// Parametric (Normal) VaR
///
/// ```text
/// VaR_α = -μ + z_α × σ
/// ```
///
/// # Arguments
/// * `returns` - Historical returns (to estimate μ and σ)
/// * `alpha` - Confidence level (e.g., 0.05 for 95% VaR)
/// * `holding_period` - Days to scale for (default: 1)
///
/// # Returns
/// VaR as a positive number (loss)
pub fn parametric_var(returns: &[f64], alpha: f64, holding_period: f64) -> f64 {
    if returns.is_empty() || alpha <= 0.0 || alpha >= 1.0 {
        return f64::NAN;
    }

    let n = returns.len() as f64;

    // Mean return
    let mu: f64 = returns.iter().sum::<f64>() / n;

    // Standard deviation
    let variance: f64 = returns.iter().map(|r| (r - mu).powi(2)).sum::<f64>() / (n - 1.0);
    let sigma = variance.sqrt();

    // Scale for holding period (assumes returns are i.i.d.)
    let mu_scaled = mu * holding_period;
    let sigma_scaled = sigma * holding_period.sqrt();

    // VaR = -(μ - z_α × σ) = -μ + z_α × σ
    // Note: z_α is negative for α < 0.5, so this gives positive VaR
    let z_alpha = norm_ppf(alpha);
    -(mu_scaled + z_alpha * sigma_scaled)
}

/// Historical VaR
///
/// Simply the α-percentile of historical returns (negated for loss)
///
/// # Arguments
/// * `returns` - Historical returns
/// * `alpha` - Confidence level (e.g., 0.05 for 95% VaR)
///
/// # Returns
/// VaR as a positive number (loss)
pub fn historical_var(returns: &[f64], alpha: f64) -> f64 {
    if returns.is_empty() || alpha <= 0.0 || alpha >= 1.0 {
        return f64::NAN;
    }

    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx = (alpha * returns.len() as f64).floor() as usize;
    let idx = idx.min(returns.len() - 1);

    -sorted[idx] // Negate to get loss
}

/// Parametric (Normal) CVaR / Expected Shortfall
///
/// For normal distribution:
/// ```text
/// CVaR_α = -μ + σ × φ(z_α) / α
///
/// where φ is the standard normal PDF
/// ```
///
/// # Arguments
/// * `returns` - Historical returns
/// * `alpha` - Confidence level
/// * `holding_period` - Days to scale for
pub fn parametric_cvar(returns: &[f64], alpha: f64, holding_period: f64) -> f64 {
    if returns.is_empty() || alpha <= 0.0 || alpha >= 1.0 {
        return f64::NAN;
    }

    let n = returns.len() as f64;
    let mu: f64 = returns.iter().sum::<f64>() / n;
    let variance: f64 = returns.iter().map(|r| (r - mu).powi(2)).sum::<f64>() / (n - 1.0);
    let sigma = variance.sqrt();

    let mu_scaled = mu * holding_period;
    let sigma_scaled = sigma * holding_period.sqrt();

    let z_alpha = norm_ppf(alpha);

    // Standard normal PDF at z_alpha
    let phi_z = (-0.5 * z_alpha * z_alpha).exp() / (2.0 * std::f64::consts::PI).sqrt();

    // CVaR for normal: E[X | X < VaR] = μ - σ × φ(z) / α
    // Negate for loss
    -(mu_scaled - sigma_scaled * phi_z / alpha)
}

/// Historical CVaR / Expected Shortfall
///
/// Average of returns below VaR (tail average)
///
/// ```text
/// CVaR_α = E[Return | Return ≤ VaR_α]
/// ```
pub fn historical_cvar(returns: &[f64], alpha: f64) -> f64 {
    if returns.is_empty() || alpha <= 0.0 || alpha >= 1.0 {
        return f64::NAN;
    }

    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n_tail = ((alpha * returns.len() as f64).ceil() as usize).max(1);
    let tail_sum: f64 = sorted.iter().take(n_tail).sum();

    -tail_sum / (n_tail as f64) // Negate for loss
}

/// Portfolio VaR using weights and asset returns
///
/// # Arguments
/// * `weights` - Portfolio weights
/// * `asset_returns` - Matrix of asset returns (T × n)
/// * `alpha` - Confidence level
/// * `method` - "parametric" or "historical"
#[allow(dead_code)]  // Reserved for future Python binding
pub fn portfolio_var(
    weights: &DVector<f64>,
    asset_returns: &[Vec<f64>],
    alpha: f64,
    method: &str,
) -> f64 {
    if asset_returns.is_empty() || weights.len() != asset_returns[0].len() {
        return f64::NAN;
    }

    // Calculate portfolio returns: r_p = Σ w_i × r_i
    let portfolio_returns: Vec<f64> = asset_returns
        .iter()
        .map(|period_returns| {
            weights
                .iter()
                .zip(period_returns.iter())
                .map(|(w, r)| w * r)
                .sum()
        })
        .collect();

    match method {
        "parametric" => parametric_var(&portfolio_returns, alpha, 1.0),
        "historical" => historical_var(&portfolio_returns, alpha),
        _ => parametric_var(&portfolio_returns, alpha, 1.0),
    }
}

// ============ Python Bindings ============

/// Calculate Parametric VaR (Python)
#[pyfunction]
#[pyo3(name = "parametric_var", signature = (returns, alpha=0.05, holding_period=1.0))]
pub fn parametric_var_py(
    returns: PyReadonlyArray1<f64>,
    alpha: f64,
    holding_period: f64,
) -> f64 {
    let r = returns.as_array();
    parametric_var(r.as_slice().unwrap(), alpha, holding_period)
}

/// Calculate Historical VaR (Python)
#[pyfunction]
#[pyo3(name = "historical_var", signature = (returns, alpha=0.05))]
pub fn historical_var_py(returns: PyReadonlyArray1<f64>, alpha: f64) -> f64 {
    let r = returns.as_array();
    historical_var(r.as_slice().unwrap(), alpha)
}

/// Calculate Parametric CVaR (Python)
#[pyfunction]
#[pyo3(name = "parametric_cvar", signature = (returns, alpha=0.05, holding_period=1.0))]
pub fn parametric_cvar_py(
    returns: PyReadonlyArray1<f64>,
    alpha: f64,
    holding_period: f64,
) -> f64 {
    let r = returns.as_array();
    parametric_cvar(r.as_slice().unwrap(), alpha, holding_period)
}

/// Calculate Historical CVaR (Python)
#[pyfunction]
#[pyo3(name = "historical_cvar", signature = (returns, alpha=0.05))]
pub fn historical_cvar_py(returns: PyReadonlyArray1<f64>, alpha: f64) -> f64 {
    let r = returns.as_array();
    historical_cvar(r.as_slice().unwrap(), alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_norm_ppf() {
        // Known values
        assert_relative_eq!(norm_ppf(0.5), 0.0, epsilon = 0.01);
        assert_relative_eq!(norm_ppf(0.05), -1.645, epsilon = 0.01);
        assert_relative_eq!(norm_ppf(0.01), -2.326, epsilon = 0.01);
        assert_relative_eq!(norm_ppf(0.95), 1.645, epsilon = 0.01);
    }

    #[test]
    fn test_parametric_var() {
        // Create returns with known mean and std
        let returns: Vec<f64> = vec![0.01, -0.02, 0.015, -0.005, 0.008, -0.01];

        let var_95 = parametric_var(&returns, 0.05, 1.0);

        // VaR should be positive (loss)
        assert!(var_95 > 0.0);

        // Higher confidence (lower alpha) should give higher VaR
        let var_99 = parametric_var(&returns, 0.01, 1.0);
        assert!(var_99 > var_95);
    }

    #[test]
    fn test_historical_var() {
        let returns: Vec<f64> = vec![0.05, 0.02, 0.01, -0.01, -0.03, -0.05, -0.08, 0.03, 0.04, 0.00];

        let var_10 = historical_var(&returns, 0.10);

        // 10% of 10 returns = 1st worst return = -0.08
        // VaR should be 0.08 (positive loss)
        assert_relative_eq!(var_10, 0.08, epsilon = 0.001);
    }

    #[test]
    fn test_cvar_greater_than_var() {
        let returns: Vec<f64> = vec![0.02, 0.01, -0.01, -0.02, -0.03, -0.05, -0.08, 0.015];

        let var = historical_var(&returns, 0.25);
        let cvar = historical_cvar(&returns, 0.25);

        // CVaR should be >= VaR (it's the tail average)
        assert!(cvar >= var);
    }

    #[test]
    fn test_parametric_cvar() {
        let returns: Vec<f64> = vec![0.01, -0.02, 0.015, -0.005, 0.008, -0.01, 0.02, -0.015];

        let var = parametric_var(&returns, 0.05, 1.0);
        let cvar = parametric_cvar(&returns, 0.05, 1.0);

        // CVaR should be >= VaR
        assert!(cvar >= var);
    }
}
