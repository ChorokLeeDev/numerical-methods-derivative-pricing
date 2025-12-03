//! Factor-Based Portfolio Optimization
//!
//! # The Problem
//!
//! Direct covariance estimation for 500 stocks requires:
//! - 500 × 500 = 250,000 covariance terms
//! - Need 500+ observations for a non-singular matrix
//! - Estimation error dominates
//!
//! # Factor Model Solution
//!
//! Assume returns are driven by K common factors (K << N):
//!
//! ```text
//! r = α + Bf + ε
//!
//! where:
//!   r = N×1 asset returns
//!   α = N×1 stock-specific returns (usually assumed 0)
//!   B = N×K factor loadings (exposures)
//!   f = K×1 factor returns
//!   ε = N×1 idiosyncratic returns (uncorrelated across stocks)
//! ```
//!
//! # Covariance Decomposition
//!
//! ```text
//! Σ = B Σ_f B' + D
//!
//! where:
//!   Σ_f = K×K factor covariance (small matrix!)
//!   D = N×N diagonal (idiosyncratic variance)
//! ```
//!
//! Instead of estimating N² terms, we estimate:
//! - N×K factor loadings
//! - K×K factor covariance
//! - N idiosyncratic variances
//!
//! For K=5 factors and N=500 stocks: 2500 + 25 + 500 ≈ 3000 parameters
//! vs 250,000 for full covariance!
//!
//! # Common Factors
//!
//! - **Fama-French**: Market, Size (SMB), Value (HML)
//! - **Carhart**: + Momentum (MOM)
//! - **Barra**: Industry, Style factors
//! - **PCA**: Statistical factors (no economic interpretation)

use nalgebra::{DMatrix, DVector};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Factor Risk Model
pub struct FactorModel {
    /// Factor loadings (N × K)
    pub loadings: DMatrix<f64>,
    /// Factor covariance (K × K)
    pub factor_cov: DMatrix<f64>,
    /// Idiosyncratic variance (diagonal, N × 1)
    pub idio_var: DVector<f64>,
    /// Number of assets
    pub n_assets: usize,
    /// Number of factors
    pub n_factors: usize,
}

impl FactorModel {
    /// Create factor model from components
    pub fn new(
        loadings: DMatrix<f64>,
        factor_cov: DMatrix<f64>,
        idio_var: DVector<f64>,
    ) -> Self {
        let n_assets = loadings.nrows();
        let n_factors = loadings.ncols();

        Self {
            loadings,
            factor_cov,
            idio_var,
            n_assets,
            n_factors,
        }
    }

    /// Estimate factor model using PCA
    ///
    /// Principal Component Analysis finds statistical factors.
    pub fn from_pca(returns: &DMatrix<f64>, n_factors: usize) -> Self {
        let (t, n) = (returns.nrows(), returns.ncols());

        // Center returns
        let means: Vec<f64> = (0..n)
            .map(|j| returns.column(j).sum() / t as f64)
            .collect();

        let mut centered = returns.clone();
        for j in 0..n {
            for i in 0..t {
                centered[(i, j)] -= means[j];
            }
        }

        // Covariance matrix
        let cov = (&centered.transpose() * &centered) / ((t - 1) as f64);

        // Eigendecomposition (simplified power iteration for top factors)
        let (eigenvalues, eigenvectors) = eigen_decomposition(&cov, n_factors);

        // Factor loadings: B = eigenvectors × sqrt(eigenvalues)
        let mut loadings = DMatrix::zeros(n, n_factors);
        for k in 0..n_factors {
            let scale = eigenvalues[k].sqrt();
            for i in 0..n {
                loadings[(i, k)] = eigenvectors[(i, k)] * scale;
            }
        }

        // Factor covariance (for PCA factors, this is diagonal with eigenvalues)
        let mut factor_cov = DMatrix::zeros(n_factors, n_factors);
        for k in 0..n_factors {
            factor_cov[(k, k)] = 1.0; // Normalized PCA factors
        }

        // Idiosyncratic variance: Var(ε) = Var(r) - Var(Bf)
        let factor_var = &loadings * &factor_cov * loadings.transpose();
        let idio_var = DVector::from_fn(n, |i, _| {
            (cov[(i, i)] - factor_var[(i, i)]).max(1e-8)
        });

        Self::new(loadings, factor_cov, idio_var)
    }

    /// Reconstruct full covariance matrix
    ///
    /// Σ = B Σ_f B' + D
    pub fn covariance(&self) -> DMatrix<f64> {
        let factor_part = &self.loadings * &self.factor_cov * self.loadings.transpose();

        let mut cov = factor_part;
        for i in 0..self.n_assets {
            cov[(i, i)] += self.idio_var[i];
        }

        cov
    }

    /// Calculate portfolio variance efficiently
    ///
    /// Instead of w'Σw, compute:
    /// w'BΣ_fB'w + w'Dw = (B'w)'Σ_f(B'w) + Σ w_i² D_i
    pub fn portfolio_variance(&self, weights: &DVector<f64>) -> f64 {
        // Factor exposure: f_exp = B'w
        let factor_exposure = self.loadings.transpose() * weights;

        // Factor variance: f_exp' Σ_f f_exp
        let factor_var = (factor_exposure.transpose() * &self.factor_cov * &factor_exposure)[(0, 0)];

        // Idiosyncratic variance: Σ w_i² D_i
        let idio_var: f64 = weights.iter()
            .zip(self.idio_var.iter())
            .map(|(w, d)| w * w * d)
            .sum();

        factor_var + idio_var
    }

    /// Portfolio factor exposures
    pub fn portfolio_exposures(&self, weights: &DVector<f64>) -> DVector<f64> {
        self.loadings.transpose() * weights
    }
}

/// Simple eigendecomposition using power iteration
fn eigen_decomposition(mat: &DMatrix<f64>, n_components: usize) -> (Vec<f64>, DMatrix<f64>) {
    let n = mat.nrows();
    let mut eigenvalues = Vec::with_capacity(n_components);
    let mut eigenvectors = DMatrix::zeros(n, n_components);

    let mut work_mat = mat.clone();

    for k in 0..n_components {
        // Power iteration for largest eigenvalue
        let mut v = DVector::from_fn(n, |i, _| if i == k { 1.0 } else { 0.1 });
        v /= v.norm();

        for _ in 0..100 {
            let v_new = &work_mat * &v;
            let norm = v_new.norm();
            if norm < 1e-10 {
                break;
            }
            let v_normalized = &v_new / norm;

            let diff: f64 = (&v_normalized - &v).iter().map(|x| x.abs()).sum();
            v = v_normalized;
            if diff < 1e-8 {
                break;
            }
        }

        // Eigenvalue: λ = v'Av / v'v
        let lambda = (&v.transpose() * &work_mat * &v)[(0, 0)];
        eigenvalues.push(lambda.max(1e-10));

        // Store eigenvector
        for i in 0..n {
            eigenvectors[(i, k)] = v[i];
        }

        // Deflate: A = A - λvv'
        work_mat -= &v * v.transpose() * lambda;
    }

    (eigenvalues, eigenvectors)
}

/// Factor-constrained optimization
///
/// Optimize with constraints on factor exposures:
///
/// ```text
/// minimize    w'Σw
/// subject to  Σw = 1
///             |B'w - target_exposure| ≤ tolerance
/// ```
pub fn factor_constrained_optimize(
    model: &FactorModel,
    target_exposures: Option<&DVector<f64>>,
    exposure_tolerance: f64,
    w_min: f64,
    w_max: f64,
    max_iter: usize,
) -> DVector<f64> {
    let n = model.n_assets;

    // Initialize
    let mut w = DVector::from_element(n, 1.0 / n as f64);
    w = project_simplex(&w, w_min, w_max);

    let mut learning_rate = 0.1;

    for _iter in 0..max_iter {
        // Gradient of variance
        let cov = model.covariance();
        let var_grad = &cov * &w * 2.0;

        // Penalty for factor exposure deviation (if target specified)
        let exposure_grad = if let Some(target) = target_exposures {
            let current_exp = model.portfolio_exposures(&w);
            let exp_diff = &current_exp - target;

            // Gradient: ∂||B'w - t||² / ∂w = 2B(B'w - t)
            &model.loadings * &exp_diff * 2.0 / exposure_tolerance
        } else {
            DVector::zeros(n)
        };

        let grad = var_grad + exposure_grad;

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

/// Project onto simplex
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
            capacity += if diff > 0.0 { projected[i] - w_min } else { w_max - projected[i] };
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

/// Estimate factor model using PCA (Python)
///
/// # Arguments
/// * `returns` - Historical returns matrix (T × N)
/// * `n_factors` - Number of factors to extract
///
/// # Returns
/// Tuple of (loadings, factor_cov, idio_var)
#[pyfunction]
#[pyo3(name = "estimate_factor_model")]
pub fn estimate_factor_model_py<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<f64>,
    n_factors: usize,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let r = returns.as_array();
    let (t, n) = (r.nrows(), r.ncols());

    let returns_mat = DMatrix::from_row_slice(t, n, r.as_slice().unwrap());
    let model = FactorModel::from_pca(&returns_mat, n_factors);

    // Convert to numpy
    let loadings_vec: Vec<Vec<f64>> = (0..model.n_assets)
        .map(|i| (0..model.n_factors).map(|j| model.loadings[(i, j)]).collect())
        .collect();

    let factor_cov_vec: Vec<Vec<f64>> = (0..model.n_factors)
        .map(|i| (0..model.n_factors).map(|j| model.factor_cov[(i, j)]).collect())
        .collect();

    Ok((
        PyArray2::from_vec2_bound(py, &loadings_vec)?,
        PyArray2::from_vec2_bound(py, &factor_cov_vec)?,
        PyArray1::from_slice_bound(py, model.idio_var.as_slice()),
    ))
}

/// Factor-based minimum variance optimization (Python)
#[pyfunction]
#[pyo3(name = "factor_min_variance", signature = (loadings, factor_cov, idio_var, w_min=0.0, w_max=1.0))]
pub fn factor_min_variance_py<'py>(
    py: Python<'py>,
    loadings: PyReadonlyArray2<f64>,
    factor_cov: PyReadonlyArray2<f64>,
    idio_var: PyReadonlyArray1<f64>,
    w_min: f64,
    w_max: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let l = loadings.as_array();
    let fc = factor_cov.as_array();
    let iv = idio_var.as_array();

    let (n, k) = (l.nrows(), l.ncols());

    let loadings_mat = DMatrix::from_row_slice(n, k, l.as_slice().unwrap());
    let factor_cov_mat = DMatrix::from_row_slice(k, k, fc.as_slice().unwrap());
    let idio_var_vec = DVector::from_row_slice(iv.as_slice().unwrap());

    let model = FactorModel::new(loadings_mat, factor_cov_mat, idio_var_vec);
    let weights = factor_constrained_optimize(&model, None, 1.0, w_min, w_max, 1000);

    Ok(PyArray1::from_slice_bound(py, weights.as_slice()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_factor_model_pca() {
        // Create synthetic returns with factor structure
        let t = 100;
        let n = 10;
        let k = 2;

        // True loadings
        let mut returns_data = vec![0.0; t * n];
        for i in 0..t {
            // Factor returns
            let f1 = ((i as f64) * 0.1).sin() * 0.02;
            let f2 = ((i as f64) * 0.15).cos() * 0.015;

            for j in 0..n {
                let loading1 = (j as f64 / n as f64) * 0.5 + 0.5;
                let loading2 = 1.0 - (j as f64 / n as f64) * 0.5;
                let idio = ((i * n + j) as f64 * 0.3).sin() * 0.005;

                returns_data[i * n + j] = loading1 * f1 + loading2 * f2 + idio;
            }
        }

        let returns = DMatrix::from_row_slice(t, n, &returns_data);
        let model = FactorModel::from_pca(&returns, k);

        assert_eq!(model.n_assets, n);
        assert_eq!(model.n_factors, k);
        assert_eq!(model.loadings.nrows(), n);
        assert_eq!(model.loadings.ncols(), k);
    }

    #[test]
    fn test_portfolio_variance_consistency() {
        // Factor model variance should match direct calculation
        let loadings = DMatrix::from_row_slice(3, 2, &[
            0.8, 0.2,
            0.6, 0.4,
            0.3, 0.7,
        ]);
        let factor_cov = DMatrix::from_row_slice(2, 2, &[0.04, 0.01, 0.01, 0.03]);
        let idio_var = DVector::from_vec(vec![0.001, 0.002, 0.0015]);

        let model = FactorModel::new(loadings, factor_cov, idio_var);

        let w = DVector::from_vec(vec![0.4, 0.35, 0.25]);

        // Via factor model
        let var_factor = model.portfolio_variance(&w);

        // Via full covariance
        let full_cov = model.covariance();
        let var_full = (w.transpose() * &full_cov * &w)[(0, 0)];

        assert_relative_eq!(var_factor, var_full, epsilon = 1e-10);
    }
}
