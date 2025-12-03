//! # quant-core
//!
//! 포트폴리오 최적화 라이브러리 (Rust 구현)
//!
//! ## 주요 기능
//! - Mean-Variance 최적화 (Markowitz)
//! - Black-Litterman 모델
//! - Python 바인딩 (PyO3)
//!
//! ## 사용 예시 (Python)
//! ```python
//! from quant_core import MeanVarianceOptimizer, BlackLitterman
//!
//! optimizer = MeanVarianceOptimizer(cov_matrix, expected_returns)
//! weights = optimizer.max_sharpe(risk_free_rate=0.03)
//! ```

pub mod optimizer;
pub mod risk;

use pyo3::prelude::*;

/// Python 모듈 정의
#[pymodule]
fn quant_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Mean-Variance Optimizer
    m.add_class::<optimizer::MeanVarianceOptimizer>()?;

    // Black-Litterman
    m.add_class::<optimizer::BlackLitterman>()?;

    // Risk utilities
    m.add_function(wrap_pyfunction!(risk::sample_covariance_py, m)?)?;

    Ok(())
}
