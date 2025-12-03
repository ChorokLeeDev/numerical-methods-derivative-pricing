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

    // Constrained Optimization
    m.add_function(wrap_pyfunction!(optimizer::min_variance_constrained_py, m)?)?;
    m.add_function(wrap_pyfunction!(optimizer::max_sharpe_constrained_py, m)?)?;

    // Advanced Optimization
    m.add_function(wrap_pyfunction!(optimizer::minimize_cvar_py, m)?)?;
    m.add_function(wrap_pyfunction!(optimizer::robust_optimize_py, m)?)?;
    m.add_function(wrap_pyfunction!(optimizer::multiperiod_optimize_py, m)?)?;
    m.add_function(wrap_pyfunction!(optimizer::estimate_factor_model_py, m)?)?;
    m.add_function(wrap_pyfunction!(optimizer::factor_min_variance_py, m)?)?;

    // Risk utilities - Covariance Estimation
    m.add_function(wrap_pyfunction!(risk::sample_covariance_py, m)?)?;
    m.add_function(wrap_pyfunction!(risk::ledoit_wolf_py, m)?)?;
    m.add_function(wrap_pyfunction!(risk::shrink_to_identity_py, m)?)?;

    // Risk utilities - Risk Decomposition
    m.add_function(wrap_pyfunction!(risk::mcr_py, m)?)?;
    m.add_function(wrap_pyfunction!(risk::ccr_py, m)?)?;
    m.add_function(wrap_pyfunction!(risk::pct_py, m)?)?;
    m.add_function(wrap_pyfunction!(risk::risk_parity_py, m)?)?;

    // Risk utilities - VaR and CVaR
    m.add_function(wrap_pyfunction!(risk::parametric_var_py, m)?)?;
    m.add_function(wrap_pyfunction!(risk::historical_var_py, m)?)?;
    m.add_function(wrap_pyfunction!(risk::parametric_cvar_py, m)?)?;
    m.add_function(wrap_pyfunction!(risk::historical_cvar_py, m)?)?;

    // Risk utilities - Hierarchical Risk Parity
    m.add_function(wrap_pyfunction!(risk::hrp_weights_py, m)?)?;

    // Risk utilities - Factor Risk Models
    m.add_function(wrap_pyfunction!(risk::estimate_factor_risk_model_py, m)?)?;
    m.add_function(wrap_pyfunction!(risk::factor_risk_decomposition_py, m)?)?;

    // Risk utilities - Extreme Value Theory (EVT)
    m.add_function(wrap_pyfunction!(risk::fit_gpd_py, m)?)?;
    m.add_function(wrap_pyfunction!(risk::evt_var_py, m)?)?;
    m.add_function(wrap_pyfunction!(risk::evt_es_py, m)?)?;
    m.add_function(wrap_pyfunction!(risk::hill_tail_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(risk::tail_risk_analysis_py, m)?)?;

    Ok(())
}
