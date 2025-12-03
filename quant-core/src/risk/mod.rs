//! 리스크 추정 모듈
//!
//! - 표본 공분산 행렬
//! - (향후) Shrinkage 추정량, Factor Model 등

mod covariance;

pub use covariance::sample_covariance_py;
