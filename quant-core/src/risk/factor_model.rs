//! Factor Risk Models
//!
//! # The Problem
//!
//! For N stocks, the covariance matrix has N(N+1)/2 unique elements.
//! For 500 stocks: 125,250 parameters!
//!
//! Estimation error dominates, and the matrix may be singular.
//!
//! # Factor Model Solution
//!
//! Assume returns are driven by K common factors (K << N):
//!
//! ```text
//! r_i = α_i + Σⱼ β_ij × f_j + ε_i
//!
//! where:
//!   r_i = return of asset i
//!   β_ij = loading of asset i on factor j
//!   f_j = return of factor j
//!   ε_i = idiosyncratic (stock-specific) return
//! ```
//!
//! # Covariance Decomposition
//!
//! ```text
//! Σ = B × Σ_f × B' + D
//!
//! where:
//!   B = N×K factor loadings
//!   Σ_f = K×K factor covariance
//!   D = N×N diagonal (idiosyncratic variance)
//! ```
//!
//! # Risk Decomposition
//!
//! Portfolio variance decomposes into:
//!
//! ```text
//! σ²_p = Factor Risk + Specific Risk
//!      = w'BΣ_fB'w + w'Dw
//! ```
//!
//! - **Factor Risk**: Systematic, undiversifiable
//! - **Specific Risk**: Diversifies away with many stocks
//!
//! # Industry Standard
//!
//! - Barra: Style + Industry factors
//! - Axioma: Similar approach
//! - Fama-French: Academic factor model

use nalgebra::{DMatrix, DVector};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Factor Risk Model for risk decomposition
pub struct FactorRiskModel {
    /// Factor loadings B (N × K)
    pub loadings: DMatrix<f64>,
    /// Factor covariance Σ_f (K × K)
    pub factor_cov: DMatrix<f64>,
    /// Idiosyncratic variance D (diagonal, stored as N×1)
    pub specific_var: DVector<f64>,
    /// Factor names (optional)
    pub factor_names: Vec<String>,
    /// Number of assets
    n_assets: usize,
    /// Number of factors
    n_factors: usize,
}

impl FactorRiskModel {
    /// Create factor risk model from components
    pub fn new(
        loadings: DMatrix<f64>,
        factor_cov: DMatrix<f64>,
        specific_var: DVector<f64>,
    ) -> Self {
        let n_assets = loadings.nrows();
        let n_factors = loadings.ncols();
        let factor_names = (0..n_factors).map(|i| format!("Factor_{}", i + 1)).collect();

        Self {
            loadings,
            factor_cov,
            specific_var,
            factor_names,
            n_assets,
            n_factors,
        }
    }

    /// Set factor names
    pub fn with_factor_names(mut self, names: Vec<String>) -> Self {
        self.factor_names = names;
        self
    }

    /// Reconstruct full covariance matrix
    ///
    /// Σ = BΣ_fB' + D
    pub fn full_covariance(&self) -> DMatrix<f64> {
        let factor_cov = &self.loadings * &self.factor_cov * self.loadings.transpose();

        let mut cov = factor_cov;
        for i in 0..self.n_assets {
            cov[(i, i)] += self.specific_var[i];
        }

        cov
    }

    /// Portfolio factor exposures
    ///
    /// exp = B'w
    pub fn portfolio_factor_exposures(&self, weights: &DVector<f64>) -> DVector<f64> {
        self.loadings.transpose() * weights
    }

    /// Portfolio total variance
    ///
    /// σ²_p = w'BΣ_fB'w + w'Dw
    pub fn portfolio_variance(&self, weights: &DVector<f64>) -> f64 {
        let factor_exp = self.portfolio_factor_exposures(weights);
        let factor_var = (&factor_exp.transpose() * &self.factor_cov * &factor_exp)[(0, 0)];

        let specific_var: f64 = weights.iter()
            .zip(self.specific_var.iter())
            .map(|(w, d)| w * w * d)
            .sum();

        factor_var + specific_var
    }

    /// Decompose portfolio risk into factor and specific
    pub fn risk_decomposition(&self, weights: &DVector<f64>) -> RiskDecomposition {
        let factor_exp = self.portfolio_factor_exposures(weights);
        let factor_var = (&factor_exp.transpose() * &self.factor_cov * &factor_exp)[(0, 0)];

        let specific_var: f64 = weights.iter()
            .zip(self.specific_var.iter())
            .map(|(w, d)| w * w * d)
            .sum();

        let total_var = factor_var + specific_var;
        let total_vol = total_var.sqrt();

        // Contribution of each factor
        let mut factor_contributions = vec![0.0; self.n_factors];
        for k in 0..self.n_factors {
            // Marginal contribution of factor k
            let mut contribution = 0.0;
            for l in 0..self.n_factors {
                contribution += factor_exp[k] * self.factor_cov[(k, l)] * factor_exp[l];
            }
            factor_contributions[k] = contribution;
        }

        RiskDecomposition {
            total_variance: total_var,
            total_volatility: total_vol,
            factor_variance: factor_var,
            specific_variance: specific_var,
            factor_contributions,
            factor_pct: factor_var / total_var,
            specific_pct: specific_var / total_var,
        }
    }

    /// Marginal contribution to risk from each asset
    ///
    /// MCR_i = ∂σ_p / ∂w_i
    pub fn marginal_contribution(&self, weights: &DVector<f64>) -> DVector<f64> {
        let total_var = self.portfolio_variance(weights);
        let total_vol = total_var.sqrt();

        if total_vol < 1e-10 {
            return DVector::zeros(self.n_assets);
        }

        // MCR = (BΣ_fB'w + Dw) / σ_p
        let factor_term = &self.loadings * &self.factor_cov * self.loadings.transpose() * weights;

        let mut specific_term = DVector::zeros(self.n_assets);
        for i in 0..self.n_assets {
            specific_term[i] = self.specific_var[i] * weights[i];
        }

        (factor_term + specific_term) / total_vol
    }
}

/// Risk decomposition results
#[derive(Debug)]
pub struct RiskDecomposition {
    pub total_variance: f64,
    pub total_volatility: f64,
    pub factor_variance: f64,
    pub specific_variance: f64,
    pub factor_contributions: Vec<f64>,
    pub factor_pct: f64,
    pub specific_pct: f64,
}

/// Estimate factor model using regression
///
/// For each asset: r_i = α_i + Σⱼ β_ij × f_j + ε_i
pub fn estimate_from_factors(
    asset_returns: &DMatrix<f64>,  // T × N
    factor_returns: &DMatrix<f64>, // T × K
) -> FactorRiskModel {
    let (t, n) = (asset_returns.nrows(), asset_returns.ncols());
    let k = factor_returns.ncols();

    // OLS for each asset: β_i = (F'F)⁻¹F'r_i
    let ftf = factor_returns.transpose() * factor_returns;
    let ftf_inv = ftf.clone().try_inverse().unwrap_or_else(|| {
        // Add small regularization if singular
        let reg = DMatrix::identity(k, k) * 1e-6;
        (ftf + reg).try_inverse().unwrap()
    });

    let mut loadings = DMatrix::zeros(n, k);
    let mut residuals = DMatrix::zeros(t, n);

    for i in 0..n {
        let r_i = asset_returns.column(i);
        let ftr = factor_returns.transpose() * r_i;
        let beta = &ftf_inv * ftr;

        for j in 0..k {
            loadings[(i, j)] = beta[j];
        }

        // Calculate residuals
        for tt in 0..t {
            let predicted: f64 = (0..k).map(|j| loadings[(i, j)] * factor_returns[(tt, j)]).sum();
            residuals[(tt, i)] = r_i[tt] - predicted;
        }
    }

    // Factor covariance
    let factor_means: Vec<f64> = (0..k)
        .map(|j| factor_returns.column(j).sum() / t as f64)
        .collect();

    let mut factor_demeaned = factor_returns.clone();
    for j in 0..k {
        for tt in 0..t {
            factor_demeaned[(tt, j)] -= factor_means[j];
        }
    }
    let factor_cov = (&factor_demeaned.transpose() * &factor_demeaned) / ((t - 1) as f64);

    // Specific variance (from residuals)
    let specific_var = DVector::from_fn(n, |i, _| {
        let col = residuals.column(i);
        let mean = col.sum() / t as f64;
        let var: f64 = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / ((t - 1) as f64);
        var
    });

    FactorRiskModel::new(loadings, factor_cov, specific_var)
}

// ============ Python Bindings ============

/// Estimate factor risk model from returns (Python)
///
/// # Arguments
/// * `asset_returns` - Asset returns matrix (T × N)
/// * `factor_returns` - Factor returns matrix (T × K)
///
/// # Returns
/// Tuple of (loadings, factor_cov, specific_var)
#[pyfunction]
#[pyo3(name = "estimate_factor_risk_model")]
pub fn estimate_factor_risk_model_py<'py>(
    py: Python<'py>,
    asset_returns: PyReadonlyArray2<f64>,
    factor_returns: PyReadonlyArray2<f64>,
) -> PyResult<(
    Bound<'py, numpy::PyArray2<f64>>,
    Bound<'py, numpy::PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let ar = asset_returns.as_array();
    let fr = factor_returns.as_array();

    let (t, n) = (ar.nrows(), ar.ncols());
    let k = fr.ncols();

    let asset_mat = DMatrix::from_row_slice(t, n, ar.as_slice().unwrap());
    let factor_mat = DMatrix::from_row_slice(t, k, fr.as_slice().unwrap());

    let model = estimate_from_factors(&asset_mat, &factor_mat);

    // Convert to numpy
    let loadings_vec: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..k).map(|j| model.loadings[(i, j)]).collect())
        .collect();

    let factor_cov_vec: Vec<Vec<f64>> = (0..k)
        .map(|i| (0..k).map(|j| model.factor_cov[(i, j)]).collect())
        .collect();

    Ok((
        numpy::PyArray2::from_vec2_bound(py, &loadings_vec)?,
        numpy::PyArray2::from_vec2_bound(py, &factor_cov_vec)?,
        PyArray1::from_slice_bound(py, model.specific_var.as_slice()),
    ))
}

/// Calculate portfolio risk decomposition (Python)
#[pyfunction]
#[pyo3(name = "factor_risk_decomposition")]
pub fn factor_risk_decomposition_py<'py>(
    _py: Python<'py>,
    weights: PyReadonlyArray1<f64>,
    loadings: PyReadonlyArray2<f64>,
    factor_cov: PyReadonlyArray2<f64>,
    specific_var: PyReadonlyArray1<f64>,
) -> PyResult<(f64, f64, f64, f64)> {
    let w = weights.as_array();
    let l = loadings.as_array();
    let fc = factor_cov.as_array();
    let sv = specific_var.as_array();

    let (n, k) = (l.nrows(), l.ncols());

    let weights_vec = DVector::from_row_slice(w.as_slice().unwrap());
    let loadings_mat = DMatrix::from_row_slice(n, k, l.as_slice().unwrap());
    let factor_cov_mat = DMatrix::from_row_slice(k, k, fc.as_slice().unwrap());
    let specific_var_vec = DVector::from_row_slice(sv.as_slice().unwrap());

    let model = FactorRiskModel::new(loadings_mat, factor_cov_mat, specific_var_vec);
    let decomp = model.risk_decomposition(&weights_vec);

    // Return (total_vol, factor_vol, specific_vol, factor_pct)
    Ok((
        decomp.total_volatility,
        decomp.factor_variance.sqrt(),
        decomp.specific_variance.sqrt(),
        decomp.factor_pct,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_risk_decomposition_sums() {
        let loadings = DMatrix::from_row_slice(3, 2, &[
            0.8, 0.2,
            0.6, 0.5,
            0.3, 0.7,
        ]);
        let factor_cov = DMatrix::from_row_slice(2, 2, &[0.04, 0.01, 0.01, 0.03]);
        let specific_var = DVector::from_vec(vec![0.001, 0.002, 0.0015]);

        let model = FactorRiskModel::new(loadings, factor_cov, specific_var);
        let weights = DVector::from_vec(vec![0.4, 0.35, 0.25]);

        let decomp = model.risk_decomposition(&weights);

        // Factor + Specific should equal Total
        assert_relative_eq!(
            decomp.factor_variance + decomp.specific_variance,
            decomp.total_variance,
            epsilon = 1e-10
        );

        // Percentages should sum to 1
        assert_relative_eq!(
            decomp.factor_pct + decomp.specific_pct,
            1.0,
            epsilon = 1e-10
        );
    }
}
