//! 리스크 추정 모듈
//!
//! - 표본 공분산 행렬 (Sample Covariance)
//! - Shrinkage 추정량 (Ledoit-Wolf, Identity)
//! - 리스크 분해 (MCR, CCR, Risk Parity)
//! - VaR/CVaR (Value at Risk, Expected Shortfall)
//! - Hierarchical Risk Parity (HRP)
//! - Factor Risk Models (Risk Decomposition by Factors)
//! - Extreme Value Theory (EVT) for Tail Risk

pub mod covariance;
mod shrinkage;
mod decomposition;
mod var;
mod hrp;
pub mod factor_model;
mod evt;

pub use covariance::sample_covariance_py;
pub use shrinkage::{ledoit_wolf_py, shrink_to_identity_py};
pub use decomposition::{mcr_py, ccr_py, pct_py, risk_parity_py};
pub use var::{parametric_var_py, historical_var_py, parametric_cvar_py, historical_cvar_py};
pub use hrp::hrp_weights_py;
pub use factor_model::{estimate_factor_risk_model_py, factor_risk_decomposition_py};
pub use evt::{fit_gpd_py, evt_var_py, evt_es_py, hill_tail_index_py, tail_risk_analysis_py};
