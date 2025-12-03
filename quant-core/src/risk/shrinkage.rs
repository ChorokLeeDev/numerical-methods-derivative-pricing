//! Shrinkage Covariance Estimators
//!
//! # The Problem with Sample Covariance
//!
//! Sample covariance suffers from estimation error:
//! - When n_assets > n_observations: Matrix is singular (can't invert)
//! - When n_assets ≈ n_observations: Extreme eigenvalues are exaggerated
//! - Leads to unstable portfolio weights (small input changes → large weight changes)
//!
//! # Shrinkage Solution
//!
//! Combine the sample covariance with a structured "target" matrix:
//!
//! ```text
//! Σ_shrunk = α × F + (1 - α) × Σ_sample
//!
//! where:
//!   α = shrinkage intensity (0 to 1)
//!   F = structured target (e.g., constant correlation model)
//!   Σ_sample = sample covariance
//! ```
//!
//! **Intuition**:
//! - High α: Trust the simple model (when data is scarce)
//! - Low α: Trust the sample (when data is abundant)
//! - Ledoit-Wolf finds the OPTIMAL α analytically (no cross-validation!)
//!
//! # References
//!
//! - Ledoit & Wolf (2004): "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices"
//! - Ledoit & Wolf (2003): "Improved Estimation of the Covariance Matrix of Stock Returns"

use nalgebra::DMatrix;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use super::covariance::sample_covariance;

/// Ledoit-Wolf Shrinkage Estimator
///
/// # The Math
///
/// **Step 1: Choose a Target Matrix (F)**
///
/// We use the "constant correlation" target:
/// ```text
/// F_ij = { σ_i²           if i = j     (keep diagonal variances)
///       { ρ̄ × σ_i × σ_j   if i ≠ j     (use average correlation for off-diagonal)
/// ```
///
/// **Step 2: Calculate Optimal Shrinkage Intensity (α)**
///
/// The optimal α minimizes the expected loss:
/// ```text
/// E[||Σ_shrunk - Σ_true||²]
/// ```
///
/// Ledoit-Wolf derived this formula:
/// ```text
/// α* = (sum of squared estimation errors) / (sum of squared deviations from target)
/// ```
///
/// **Why This Works**:
/// - When sample covariance has high estimation error → α* is high → shrink more
/// - When target is close to truth → α* is high → shrink more
/// - Balances bias-variance tradeoff automatically
///
/// # Arguments
/// * `returns` - T×n matrix of returns (T observations, n assets)
///
/// # Returns
/// * Shrunk covariance matrix (n×n)
/// * Optimal shrinkage intensity (α)
#[pyfunction]
#[pyo3(name = "ledoit_wolf")]
pub fn ledoit_wolf_py<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, f64)> {
    let ret = returns.as_array();
    let (t, n) = (ret.nrows(), ret.ncols());

    if t < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Need at least 2 observations",
        ));
    }

    // Convert to nalgebra matrix
    let returns_mat = DMatrix::from_row_slice(t, n, ret.as_slice().unwrap());

    let (shrunk_cov, alpha) = ledoit_wolf(&returns_mat);

    // Convert back to numpy
    let cov_vec: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| shrunk_cov[(i, j)]).collect())
        .collect();

    Ok((PyArray2::from_vec2_bound(py, &cov_vec)?, alpha))
}

/// Ledoit-Wolf shrinkage (pure Rust implementation)
pub fn ledoit_wolf(returns: &DMatrix<f64>) -> (DMatrix<f64>, f64) {
    let (t, n) = (returns.nrows(), returns.ncols());
    let t_f64 = t as f64;
    let _n_f64 = n as f64;  // Reserved for future use

    // Step 1: Calculate sample covariance
    let sample = sample_covariance(returns);

    // Step 2: Calculate means (for demeaning)
    let means: Vec<f64> = (0..n)
        .map(|j| returns.column(j).sum() / t_f64)
        .collect();

    // Create demeaned returns
    let mut demeaned = returns.clone();
    for j in 0..n {
        for i in 0..t {
            demeaned[(i, j)] -= means[j];
        }
    }

    // Step 3: Build the constant-correlation target matrix
    //
    // F_ij = { variance_i           if i = j
    //       { avg_corr × σ_i × σ_j  if i ≠ j

    // Extract variances (diagonal of sample covariance)
    let variances: Vec<f64> = (0..n).map(|i| sample[(i, i)]).collect();
    let std_devs: Vec<f64> = variances.iter().map(|v| v.sqrt()).collect();

    // Calculate average correlation
    let mut corr_sum = 0.0;
    let mut corr_count = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if std_devs[i] > 1e-10 && std_devs[j] > 1e-10 {
                let corr = sample[(i, j)] / (std_devs[i] * std_devs[j]);
                corr_sum += corr;
                corr_count += 1;
            }
        }
    }
    let avg_corr = if corr_count > 0 {
        corr_sum / (corr_count as f64)
    } else {
        0.0
    };

    // Build target matrix F
    let mut target = DMatrix::zeros(n, n);
    for i in 0..n {
        target[(i, i)] = variances[i]; // Diagonal: keep variances
        for j in (i + 1)..n {
            let off_diag = avg_corr * std_devs[i] * std_devs[j];
            target[(i, j)] = off_diag;
            target[(j, i)] = off_diag; // Symmetric
        }
    }

    // Step 4: Calculate optimal shrinkage intensity
    //
    // This is the clever part of Ledoit-Wolf:
    // α* = π̂ / δ̂
    //
    // where:
    // π̂ = sum of asymptotic variances of entries of sample covariance
    // δ̂ = sum of squared deviations of sample from target

    // Calculate π̂ (estimation error of sample covariance)
    // π_ij = Var(sample_ij) = E[(r_i*r_j - σ_ij)²]
    let mut pi_sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            let mut sum_sq = 0.0;
            for k in 0..t {
                let prod = demeaned[(k, i)] * demeaned[(k, j)];
                sum_sq += (prod - sample[(i, j)]).powi(2);
            }
            pi_sum += sum_sq / t_f64;
        }
    }
    let pi_hat = pi_sum;

    // Calculate δ̂ (squared Frobenius norm of difference)
    let mut delta_sq = 0.0;
    for i in 0..n {
        for j in 0..n {
            delta_sq += (sample[(i, j)] - target[(i, j)]).powi(2);
        }
    }

    // Calculate γ̂ (cross-term for off-diagonal shrinkage)
    // This accounts for the fact that off-diagonal elements of target
    // depend on variances which are also estimated
    let mut gamma_sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            gamma_sum += (target[(i, j)] - sample[(i, j)]).powi(2);
        }
    }

    // Optimal shrinkage intensity
    // α* = κ / T where κ = (π̂ - γ̂) / δ̂
    let kappa = (pi_hat - gamma_sum) / delta_sq;
    let alpha = (kappa / t_f64).clamp(0.0, 1.0); // Must be in [0, 1]

    // Step 5: Apply shrinkage
    // Σ_shrunk = α × F + (1 - α) × Σ_sample
    let shrunk = &target * alpha + &sample * (1.0 - alpha);

    (shrunk, alpha)
}

/// Shrinkage toward Identity Matrix
///
/// A simpler alternative when you want to shrink toward "uncorrelated assets"
///
/// ```text
/// F = μ × I  (identity scaled by average variance)
///
/// Σ_shrunk = α × μI + (1 - α) × Σ_sample
/// ```
///
/// **When to use**:
/// - Very few observations
/// - Want maximum regularization
/// - Don't trust the correlation structure at all
#[pyfunction]
#[pyo3(name = "shrink_to_identity", signature = (returns, alpha=None))]
pub fn shrink_to_identity_py<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<f64>,
    alpha: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let ret = returns.as_array();
    let (t, n) = (ret.nrows(), ret.ncols());

    if t < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Need at least 2 observations",
        ));
    }

    let returns_mat = DMatrix::from_row_slice(t, n, ret.as_slice().unwrap());
    let shrunk = shrink_to_identity(&returns_mat, alpha);

    let cov_vec: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| shrunk[(i, j)]).collect())
        .collect();

    Ok(PyArray2::from_vec2_bound(py, &cov_vec)?)
}

/// Shrinkage to scaled identity (pure Rust)
pub fn shrink_to_identity(returns: &DMatrix<f64>, alpha: Option<f64>) -> DMatrix<f64> {
    let n = returns.ncols();

    let sample = sample_covariance(returns);

    // Target: μ × I where μ = average variance
    let avg_var: f64 = (0..n).map(|i| sample[(i, i)]).sum::<f64>() / (n as f64);
    let mut target = DMatrix::zeros(n, n);
    for i in 0..n {
        target[(i, i)] = avg_var;
    }

    // Use provided alpha or default to 0.2 (mild shrinkage)
    let alpha = alpha.unwrap_or(0.2).clamp(0.0, 1.0);

    // Apply shrinkage
    &target * alpha + &sample * (1.0 - alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ledoit_wolf_basic() {
        // Create synthetic returns: 100 observations, 5 assets
        let t = 100;
        let n = 5;
        let mut returns_data = vec![0.0; t * n];

        // Simple random-like pattern (deterministic for reproducibility)
        for i in 0..t {
            for j in 0..n {
                returns_data[i * n + j] = ((i * n + j) as f64 * 0.01).sin() * 0.02;
            }
        }

        let returns = DMatrix::from_row_slice(t, n, &returns_data);
        let (shrunk, alpha) = ledoit_wolf(&returns);

        // Alpha should be in [0, 1]
        assert!(alpha >= 0.0 && alpha <= 1.0);

        // Matrix should be symmetric
        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(shrunk[(i, j)], shrunk[(j, i)], epsilon = 1e-10);
            }
        }

        // Diagonal should be positive (variances)
        for i in 0..n {
            assert!(shrunk[(i, i)] > 0.0);
        }
    }

    #[test]
    fn test_shrink_to_identity() {
        let t = 50;
        let n = 3;
        let mut returns_data = vec![0.0; t * n];

        for i in 0..t {
            for j in 0..n {
                returns_data[i * n + j] = ((i + j) as f64 * 0.1).cos() * 0.01;
            }
        }

        let returns = DMatrix::from_row_slice(t, n, &returns_data);
        let shrunk = shrink_to_identity(&returns, Some(0.5));

        // With alpha=0.5 shrinkage to identity, off-diagonal should be reduced
        let sample = sample_covariance(&returns);

        // Off-diagonal should be closer to 0 than in sample
        for i in 0..n {
            for j in (i + 1)..n {
                assert!(shrunk[(i, j)].abs() <= sample[(i, j)].abs() + 1e-10);
            }
        }
    }

    #[test]
    fn test_high_dimension() {
        // Edge case: more assets than observations
        // Sample covariance would be singular, but shrinkage should work
        let t = 10;
        let n = 20; // n > t !

        let mut returns_data = vec![0.0; t * n];
        for i in 0..t {
            for j in 0..n {
                returns_data[i * n + j] = ((i * j) as f64 * 0.05).sin() * 0.01;
            }
        }

        let returns = DMatrix::from_row_slice(t, n, &returns_data);
        let (shrunk, alpha) = ledoit_wolf(&returns);

        // Should still produce valid result
        assert!(alpha >= 0.0 && alpha <= 1.0);

        // Matrix should still be symmetric
        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(shrunk[(i, j)], shrunk[(j, i)], epsilon = 1e-10);
            }
        }
    }
}
