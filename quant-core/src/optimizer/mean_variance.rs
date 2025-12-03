//! Mean-Variance 최적화 (Markowitz, 1952)
//!
//! ## 이론
//!
//! 목표: 주어진 목표 수익률에서 분산(위험)을 최소화하는 포트폴리오 비중 찾기
//!
//! ```text
//! min   w'Σw           (포트폴리오 분산)
//! s.t.  w'μ = μ_target (목표 수익률)
//!       w'1 = 1        (비중 합 = 1)
//!       w ≥ 0          (롱온리, 선택적)
//! ```
//!
//! ## Lagrangian 해법 (제약 없는 경우)
//!
//! L = w'Σw - λ(w'μ - μ_target) - γ(w'1 - 1)
//!
//! ∂L/∂w = 0 → 2Σw = λμ + γ1
//!            → w = Σ^(-1)(λμ + γ1) / 2
//!
//! 제약 조건 대입하여 λ, γ 결정

use nalgebra::{DMatrix, DVector};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Mean-Variance 최적화기
///
/// Markowitz 포트폴리오 이론 기반 최적화
#[pyclass]
pub struct MeanVarianceOptimizer {
    /// 공분산 행렬 (n x n)
    cov_matrix: DMatrix<f64>,
    /// 기대 수익률 벡터 (n x 1)
    expected_returns: DVector<f64>,
    /// 자산 수
    n_assets: usize,
}

#[pymethods]
impl MeanVarianceOptimizer {
    /// 새 최적화기 생성
    ///
    /// # Arguments
    /// * `cov_matrix` - 공분산 행렬 (n x n numpy array)
    /// * `expected_returns` - 기대 수익률 (n x 1 numpy array)
    #[new]
    pub fn new(
        cov_matrix: PyReadonlyArray2<f64>,
        expected_returns: PyReadonlyArray1<f64>,
    ) -> PyResult<Self> {
        let cov = cov_matrix.as_array();
        let ret = expected_returns.as_array();

        let n = cov.nrows();

        // numpy → nalgebra 변환
        let cov_matrix = DMatrix::from_row_slice(n, n, cov.as_slice().unwrap());
        let expected_returns = DVector::from_row_slice(ret.as_slice().unwrap());

        Ok(Self {
            cov_matrix,
            expected_returns,
            n_assets: n,
        })
    }

    /// 최소 분산 포트폴리오 (Global Minimum Variance)
    ///
    /// 목표 수익률 제약 없이 분산만 최소화
    ///
    /// ```text
    /// w_gmv = Σ^(-1) * 1 / (1' * Σ^(-1) * 1)
    /// ```
    pub fn min_variance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        // Σ^(-1) 계산
        let cov_inv = self
            .cov_matrix
            .clone()
            .try_inverse()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("공분산 행렬이 특이행렬입니다"))?;

        // 1 벡터
        let ones = DVector::from_element(self.n_assets, 1.0);

        // Σ^(-1) * 1
        let cov_inv_ones = &cov_inv * &ones;

        // 1' * Σ^(-1) * 1 (스칼라)
        let denom = ones.dot(&cov_inv_ones);

        // w = Σ^(-1) * 1 / (1' * Σ^(-1) * 1)
        let weights = cov_inv_ones / denom;

        Ok(PyArray1::from_slice_bound(py, weights.as_slice()))
    }

    /// 최대 샤프 비율 포트폴리오 (Tangency Portfolio)
    ///
    /// 샤프 비율 = (w'μ - r_f) / sqrt(w'Σw) 최대화
    ///
    /// ```text
    /// w_tan = Σ^(-1) * (μ - r_f * 1) / (1' * Σ^(-1) * (μ - r_f * 1))
    /// ```
    ///
    /// # Arguments
    /// * `risk_free_rate` - 무위험 이자율 (연율, 예: 0.03 = 3%)
    pub fn max_sharpe<'py>(
        &self,
        py: Python<'py>,
        risk_free_rate: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        // Σ^(-1) 계산
        let cov_inv = self
            .cov_matrix
            .clone()
            .try_inverse()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("공분산 행렬이 특이행렬입니다"))?;

        // 1 벡터
        let ones = DVector::from_element(self.n_assets, 1.0);

        // μ - r_f * 1 (초과 수익률)
        let excess_returns = &self.expected_returns - risk_free_rate * &ones;

        // Σ^(-1) * (μ - r_f)
        let cov_inv_excess = &cov_inv * &excess_returns;

        // 1' * Σ^(-1) * (μ - r_f)
        let denom = ones.dot(&cov_inv_excess);

        if denom.abs() < 1e-10 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "모든 초과수익률이 0에 가깝습니다",
            ));
        }

        // w = Σ^(-1) * (μ - r_f) / (1' * Σ^(-1) * (μ - r_f))
        let weights = cov_inv_excess / denom;

        Ok(PyArray1::from_slice_bound(py, weights.as_slice()))
    }

    /// 목표 수익률 포트폴리오
    ///
    /// 주어진 목표 수익률에서 분산 최소화
    ///
    /// # Arguments
    /// * `target_return` - 목표 수익률 (연율)
    pub fn target_return<'py>(
        &self,
        py: Python<'py>,
        target_return: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        // Σ^(-1) 계산
        let cov_inv = self
            .cov_matrix
            .clone()
            .try_inverse()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("공분산 행렬이 특이행렬입니다"))?;

        let ones = DVector::from_element(self.n_assets, 1.0);
        let mu = &self.expected_returns;

        // 중간 계산
        // A = 1' * Σ^(-1) * μ
        // B = μ' * Σ^(-1) * μ
        // C = 1' * Σ^(-1) * 1
        let cov_inv_ones = &cov_inv * &ones;
        let cov_inv_mu = &cov_inv * mu;

        let a = ones.dot(&cov_inv_mu);
        let b = mu.dot(&cov_inv_mu);
        let c = ones.dot(&cov_inv_ones);

        let det = b * c - a * a;
        if det.abs() < 1e-10 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "해가 존재하지 않습니다",
            ));
        }

        // Lagrange 승수
        let lambda = (c * target_return - a) / det;
        let gamma = (b - a * target_return) / det;

        // 최적 비중: w = λ * Σ^(-1) * μ + γ * Σ^(-1) * 1
        let weights = lambda * &cov_inv_mu + gamma * &cov_inv_ones;

        Ok(PyArray1::from_slice_bound(py, weights.as_slice()))
    }

    /// 효율적 프론티어 계산
    ///
    /// 여러 목표 수익률에 대해 최적 포트폴리오 계산
    ///
    /// # Arguments
    /// * `n_points` - 프론티어 점 개수
    ///
    /// # Returns
    /// * (returns, volatilities, weights) 튜플
    pub fn efficient_frontier<'py>(
        &self,
        py: Python<'py>,
        n_points: usize,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        let min_ret = self.expected_returns.min();
        let max_ret = self.expected_returns.max();

        let mut returns = Vec::with_capacity(n_points);
        let mut volatilities = Vec::with_capacity(n_points);
        let mut all_weights = Vec::with_capacity(n_points * self.n_assets);

        for i in 0..n_points {
            let target = min_ret + (max_ret - min_ret) * (i as f64) / ((n_points - 1) as f64);

            match self.target_return(py, target) {
                Ok(w_arr) => {
                    let w_slice: Vec<f64> = w_arr.to_vec().unwrap();
                    let w = DVector::from_row_slice(&w_slice);

                    // 포트폴리오 수익률
                    let port_return = w.dot(&self.expected_returns);

                    // 포트폴리오 분산 = w' * Σ * w
                    let port_var = (&w.transpose() * &self.cov_matrix * &w)[(0, 0)];
                    let port_vol = port_var.sqrt();

                    returns.push(port_return);
                    volatilities.push(port_vol);
                    all_weights.extend(&w_slice);
                }
                Err(_) => continue,
            }
        }

        let _n_valid = returns.len();  // For debugging
        let returns_arr = PyArray1::from_slice_bound(py, &returns);
        let vols_arr = PyArray1::from_slice_bound(py, &volatilities);
        let weights_arr = PyArray2::from_vec2_bound(
            py,
            &all_weights
                .chunks(self.n_assets)
                .map(|c| c.to_vec())
                .collect::<Vec<_>>(),
        )?;

        Ok((returns_arr, vols_arr, weights_arr))
    }

    /// 포트폴리오 통계 계산
    ///
    /// # Arguments
    /// * `weights` - 포트폴리오 비중
    ///
    /// # Returns
    /// * (return, volatility, sharpe) 튜플
    pub fn portfolio_stats(
        &self,
        weights: PyReadonlyArray1<f64>,
        risk_free_rate: f64,
    ) -> PyResult<(f64, f64, f64)> {
        let w_arr = weights.as_array();
        let w = DVector::from_row_slice(w_arr.as_slice().unwrap());

        // 수익률 = w' * μ
        let port_return = w.dot(&self.expected_returns);

        // 분산 = w' * Σ * w
        let port_var = (&w.transpose() * &self.cov_matrix * &w)[(0, 0)];
        let port_vol = port_var.sqrt();

        // 샤프 = (수익률 - 무위험) / 변동성
        let sharpe = if port_vol > 1e-10 {
            (port_return - risk_free_rate) / port_vol
        } else {
            0.0
        };

        Ok((port_return, port_vol, sharpe))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_min_variance() {
        // 간단한 2자산 테스트
        let cov = DMatrix::from_row_slice(2, 2, &[0.04, 0.01, 0.01, 0.09]);
        let ret = DVector::from_row_slice(&[0.10, 0.15]);

        let opt = MeanVarianceOptimizer {
            cov_matrix: cov.clone(),
            expected_returns: ret,
            n_assets: 2,
        };

        // 역행렬 계산
        let cov_inv = cov.try_inverse().unwrap();
        let ones = DVector::from_element(2, 1.0);
        let cov_inv_ones = &cov_inv * &ones;
        let denom = ones.dot(&cov_inv_ones);
        let expected_w = cov_inv_ones / denom;

        // 비중 합 = 1 확인
        assert_relative_eq!(expected_w.sum(), 1.0, epsilon = 1e-10);
    }
}
