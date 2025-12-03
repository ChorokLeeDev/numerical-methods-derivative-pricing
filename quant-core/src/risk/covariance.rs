//! 공분산 행렬 추정
//!
//! ## 표본 공분산 (Sample Covariance)
//!
//! ```text
//! Σ_ij = (1/(T-1)) * Σ_t (r_it - μ_i)(r_jt - μ_j)
//! ```
//!
//! ## 주의사항
//!
//! - 자산 수 > 관측치 수 → 특이행렬 (역행렬 불가)
//! - 해결책: Shrinkage, Factor Model, Random Matrix Theory

use nalgebra::DMatrix;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// 표본 공분산 행렬 계산 (Python용)
///
/// # Arguments
/// * `returns` - 수익률 행렬 (T x n), T=기간, n=자산 수
///
/// # Returns
/// * 공분산 행렬 (n x n)
#[pyfunction]
#[pyo3(name = "sample_covariance")]
pub fn sample_covariance_py<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let ret = returns.as_array();
    let (t, n) = (ret.nrows(), ret.ncols());

    if t < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "최소 2개 이상의 관측치 필요",
        ));
    }

    // numpy → nalgebra
    let returns_mat = DMatrix::from_row_slice(t, n, ret.as_slice().unwrap());

    // 평균 계산 (각 자산별)
    let means: Vec<f64> = (0..n)
        .map(|j| returns_mat.column(j).sum() / (t as f64))
        .collect();

    // 편차 행렬 (demeaned)
    let mut demeaned = returns_mat.clone();
    for j in 0..n {
        for i in 0..t {
            demeaned[(i, j)] -= means[j];
        }
    }

    // 공분산 = X' * X / (T - 1)
    let cov = (&demeaned.transpose() * &demeaned) / ((t - 1) as f64);

    // nalgebra → numpy
    let cov_vec: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| cov[(i, j)]).collect())
        .collect();

    Ok(PyArray2::from_vec2_bound(py, &cov_vec)?)
}

/// 표본 공분산 행렬 (순수 Rust)
pub fn sample_covariance(returns: &DMatrix<f64>) -> DMatrix<f64> {
    let (t, n) = (returns.nrows(), returns.ncols());

    // 평균 계산
    let means: Vec<f64> = (0..n)
        .map(|j| returns.column(j).sum() / (t as f64))
        .collect();

    // 편차 행렬
    let mut demeaned = returns.clone();
    for j in 0..n {
        for i in 0..t {
            demeaned[(i, j)] -= means[j];
        }
    }

    // 공분산
    (&demeaned.transpose() * &demeaned) / ((t - 1) as f64)
}

/// 연율화 공분산 (일간 → 연간)
///
/// Σ_annual = Σ_daily * 252
pub fn annualize_covariance(cov: &DMatrix<f64>, periods_per_year: f64) -> DMatrix<f64> {
    cov * periods_per_year
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sample_covariance() {
        // 간단한 2자산, 4기간 테스트
        let returns = DMatrix::from_row_slice(
            4,
            2,
            &[
                0.01, 0.02, // t=1
                0.02, 0.01, // t=2
                -0.01, 0.03, // t=3
                0.00, 0.00, // t=4
            ],
        );

        let cov = sample_covariance(&returns);

        // 분산은 항상 양수
        assert!(cov[(0, 0)] > 0.0);
        assert!(cov[(1, 1)] > 0.0);

        // 공분산은 대칭
        assert_relative_eq!(cov[(0, 1)], cov[(1, 0)], epsilon = 1e-10);
    }

    #[test]
    fn test_annualize() {
        let cov = DMatrix::from_row_slice(2, 2, &[0.0001, 0.00005, 0.00005, 0.0002]);

        let annual = annualize_covariance(&cov, 252.0);

        assert_relative_eq!(annual[(0, 0)], 0.0001 * 252.0, epsilon = 1e-10);
    }
}
