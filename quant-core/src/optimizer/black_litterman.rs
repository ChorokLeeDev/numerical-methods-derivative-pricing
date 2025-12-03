//! Black-Litterman 모델 (1992)
//!
//! ## 핵심 아이디어
//!
//! "시장 균형 수익률 + 투자자 뷰" → 베이지안 업데이트로 결합
//!
//! ## 수학적 구조
//!
//! ### Prior (시장 균형)
//! ```text
//! π = δ * Σ * w_mkt
//!
//! π: 균형 기대수익률 (implied returns)
//! δ: 위험회피계수 (보통 2.5)
//! Σ: 공분산 행렬
//! w_mkt: 시장 비중 (시가총액 가중)
//! ```
//!
//! ### 투자자 뷰
//! ```text
//! P * μ = Q + ε,  ε ~ N(0, Ω)
//!
//! P: k×n 뷰 행렬 (어떤 자산에 대한 뷰인지)
//! Q: k×1 뷰 수익률 (예상 수익률)
//! Ω: k×k 뷰 불확실성 (확신도)
//! ```
//!
//! ### Posterior (베이지안 업데이트)
//! ```text
//! μ_BL = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) × [(τΣ)^(-1)π + P'Ω^(-1)Q]
//!
//! τ: 스케일링 파라미터 (보통 0.05)
//! ```
//!
//! ## 베이지안 해석
//!
//! - Prior: μ ~ N(π, τΣ)
//! - Likelihood: Q|μ ~ N(Pμ, Ω)
//! - Posterior: μ|Q ~ N(μ_BL, Σ_BL)

use nalgebra::{DMatrix, DVector};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Black-Litterman 모델
///
/// 시장 균형 + 투자자 뷰 = 최적 포트폴리오
#[pyclass]
pub struct BlackLitterman {
    /// 공분산 행렬 (n x n)
    cov_matrix: DMatrix<f64>,
    /// 시장 비중 (시가총액 가중)
    market_weights: DVector<f64>,
    /// 자산 수
    n_assets: usize,
    /// 위험회피계수 (delta)
    risk_aversion: f64,
    /// 스케일링 파라미터 (tau)
    tau: f64,
    /// 뷰 행렬 P (k x n)
    view_matrix: Option<DMatrix<f64>>,
    /// 뷰 수익률 Q (k x 1)
    view_returns: Option<DVector<f64>>,
    /// 뷰 불확실성 Omega (k x k)
    view_uncertainty: Option<DMatrix<f64>>,
}

#[pymethods]
impl BlackLitterman {
    /// 새 Black-Litterman 모델 생성
    ///
    /// # Arguments
    /// * `cov_matrix` - 공분산 행렬 (n x n)
    /// * `market_weights` - 시장 비중 (시가총액 가중)
    /// * `risk_aversion` - 위험회피계수 (기본값: 2.5)
    /// * `tau` - 스케일링 파라미터 (기본값: 0.05)
    #[new]
    #[pyo3(signature = (cov_matrix, market_weights, risk_aversion=2.5, tau=0.05))]
    pub fn new(
        cov_matrix: PyReadonlyArray2<f64>,
        market_weights: PyReadonlyArray1<f64>,
        risk_aversion: f64,
        tau: f64,
    ) -> PyResult<Self> {
        let cov = cov_matrix.as_array();
        let mkt = market_weights.as_array();

        let n = cov.nrows();

        let cov_matrix = DMatrix::from_row_slice(n, n, cov.as_slice().unwrap());
        let market_weights = DVector::from_row_slice(mkt.as_slice().unwrap());

        Ok(Self {
            cov_matrix,
            market_weights,
            n_assets: n,
            risk_aversion,
            tau,
            view_matrix: None,
            view_returns: None,
            view_uncertainty: None,
        })
    }

    /// 시장 균형 수익률 (Prior) 계산
    ///
    /// π = δ * Σ * w_mkt
    ///
    /// "시장이 효율적이라면, 현재 시장 비중에서 implied 되는 기대수익률"
    pub fn equilibrium_returns<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let pi = self.risk_aversion * (&self.cov_matrix * &self.market_weights);
        PyArray1::from_slice_bound(py, pi.as_slice())
    }

    /// 절대 뷰 추가
    ///
    /// "자산 i의 수익률이 q%일 것이다"
    ///
    /// # Arguments
    /// * `asset_index` - 자산 인덱스 (0-based)
    /// * `view_return` - 예상 수익률 (연율)
    /// * `confidence` - 확신도 (0-1, 높을수록 확신)
    pub fn add_absolute_view(
        &mut self,
        asset_index: usize,
        view_return: f64,
        confidence: f64,
    ) -> PyResult<()> {
        if asset_index >= self.n_assets {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "자산 인덱스 범위 초과",
            ));
        }

        // P 행: 해당 자산만 1, 나머지 0
        let mut p_row = vec![0.0; self.n_assets];
        p_row[asset_index] = 1.0;

        self.add_view_internal(p_row, view_return, confidence);
        Ok(())
    }

    /// 상대 뷰 추가
    ///
    /// "자산 i가 자산 j보다 q% 아웃퍼폼할 것이다"
    ///
    /// # Arguments
    /// * `long_index` - 롱 자산 인덱스
    /// * `short_index` - 숏 자산 인덱스
    /// * `relative_return` - 상대 수익률 (연율)
    /// * `confidence` - 확신도 (0-1)
    pub fn add_relative_view(
        &mut self,
        long_index: usize,
        short_index: usize,
        relative_return: f64,
        confidence: f64,
    ) -> PyResult<()> {
        if long_index >= self.n_assets || short_index >= self.n_assets {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "자산 인덱스 범위 초과",
            ));
        }

        // P 행: 롱 자산 +1, 숏 자산 -1
        let mut p_row = vec![0.0; self.n_assets];
        p_row[long_index] = 1.0;
        p_row[short_index] = -1.0;

        self.add_view_internal(p_row, relative_return, confidence);
        Ok(())
    }

    /// Posterior 기대수익률 계산
    ///
    /// μ_BL = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) × [(τΣ)^(-1)π + P'Ω^(-1)Q]
    pub fn posterior_returns<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let pi = self.risk_aversion * (&self.cov_matrix * &self.market_weights);

        // 뷰가 없으면 Prior 반환
        let (p, q, omega) = match (&self.view_matrix, &self.view_returns, &self.view_uncertainty) {
            (Some(p), Some(q), Some(omega)) => (p, q, omega),
            _ => return Ok(PyArray1::from_slice_bound(py, pi.as_slice())),
        };

        // τΣ
        let tau_sigma = self.tau * &self.cov_matrix;

        // (τΣ)^(-1)
        let tau_sigma_inv = tau_sigma
            .try_inverse()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("τΣ 역행렬 실패"))?;

        // Ω^(-1)
        let omega_inv = omega
            .clone()
            .try_inverse()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Ω 역행렬 실패"))?;

        // P'Ω^(-1)P
        let pt_omega_inv_p = p.transpose() * &omega_inv * p;

        // [(τΣ)^(-1) + P'Ω^(-1)P]^(-1)
        let posterior_cov = (&tau_sigma_inv + &pt_omega_inv_p)
            .try_inverse()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Posterior 공분산 역행렬 실패"))?;

        // (τΣ)^(-1)π + P'Ω^(-1)Q
        let term1 = &tau_sigma_inv * &pi;
        let term2 = p.transpose() * &omega_inv * q;

        // μ_BL
        let mu_bl = &posterior_cov * (term1 + term2);

        Ok(PyArray1::from_slice_bound(py, mu_bl.as_slice()))
    }

    /// 최적 비중 계산
    ///
    /// Posterior 수익률로 Mean-Variance 최적화
    pub fn optimal_weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mu_bl = self.posterior_returns(py)?;
        let mu_vec: Vec<f64> = mu_bl.to_vec().unwrap();
        let mu = DVector::from_row_slice(&mu_vec);

        // Σ^(-1)
        let cov_inv = self
            .cov_matrix
            .clone()
            .try_inverse()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("공분산 역행렬 실패"))?;

        // w* = (1/δ) * Σ^(-1) * μ_BL (비제약)
        let w_unnorm = (1.0 / self.risk_aversion) * (&cov_inv * &mu);

        // 비중 합 = 1로 정규화
        let w_sum = w_unnorm.sum();
        let weights = if w_sum.abs() > 1e-10 {
            w_unnorm / w_sum
        } else {
            w_unnorm
        };

        Ok(PyArray1::from_slice_bound(py, weights.as_slice()))
    }

    /// 뷰 초기화
    pub fn clear_views(&mut self) {
        self.view_matrix = None;
        self.view_returns = None;
        self.view_uncertainty = None;
    }

    /// 현재 뷰 개수
    pub fn num_views(&self) -> usize {
        self.view_returns.as_ref().map(|q| q.len()).unwrap_or(0)
    }
}

impl BlackLitterman {
    /// 내부: 뷰 추가 로직
    fn add_view_internal(&mut self, p_row: Vec<f64>, q_val: f64, confidence: f64) {
        // 불확실성 = (1 - confidence) * p'Σp
        // confidence가 높을수록 불확실성 낮음
        let p_vec = DVector::from_row_slice(&p_row);
        let view_var = (&p_vec.transpose() * &self.cov_matrix * &p_vec)[(0, 0)];
        let uncertainty = (1.0 - confidence.clamp(0.01, 0.99)) * view_var;

        match (&mut self.view_matrix, &mut self.view_returns, &mut self.view_uncertainty) {
            (Some(p), Some(q), Some(omega)) => {
                // 기존 뷰에 추가
                let k = p.nrows();
                let mut new_p = DMatrix::zeros(k + 1, self.n_assets);
                new_p.view_mut((0, 0), (k, self.n_assets)).copy_from(p);
                new_p.set_row(k, &nalgebra::RowDVector::from_row_slice(&p_row));

                let mut new_q = DVector::zeros(k + 1);
                new_q.view_mut((0, 0), (k, 1)).copy_from(q);
                new_q[k] = q_val;

                let mut new_omega = DMatrix::zeros(k + 1, k + 1);
                new_omega.view_mut((0, 0), (k, k)).copy_from(omega);
                new_omega[(k, k)] = uncertainty;

                *p = new_p;
                *q = new_q;
                *omega = new_omega;
            }
            _ => {
                // 첫 번째 뷰
                self.view_matrix = Some(DMatrix::from_row_slice(1, self.n_assets, &p_row));
                self.view_returns = Some(DVector::from_element(1, q_val));
                self.view_uncertainty = Some(DMatrix::from_element(1, 1, uncertainty));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equilibrium_returns() {
        // 2자산 테스트
        let cov = DMatrix::from_row_slice(2, 2, &[0.04, 0.01, 0.01, 0.09]);
        let mkt = DVector::from_row_slice(&[0.6, 0.4]);

        let bl = BlackLitterman {
            cov_matrix: cov.clone(),
            market_weights: mkt.clone(),
            n_assets: 2,
            risk_aversion: 2.5,
            tau: 0.05,
            view_matrix: None,
            view_returns: None,
            view_uncertainty: None,
        };

        // π = δ * Σ * w
        let expected_pi = 2.5 * (&cov * &mkt);

        // 비교
        assert!((expected_pi[0] - 0.085).abs() < 0.01);
    }
}
