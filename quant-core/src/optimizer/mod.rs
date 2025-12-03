//! 포트폴리오 최적화 모듈
//!
//! - Mean-Variance (Markowitz)
//! - Black-Litterman
//! - Constrained Optimization (Long-only, Box constraints)
//! - Min CVaR Optimization
//! - Robust Optimization
//! - Multi-Period Optimization
//! - Factor-Based Optimization

mod mean_variance;
mod black_litterman;
mod constrained;
mod cvar_opt;
mod robust;
mod multiperiod;
mod factor_opt;

pub use mean_variance::MeanVarianceOptimizer;
pub use black_litterman::BlackLitterman;
pub use constrained::{min_variance_constrained_py, max_sharpe_constrained_py};
pub use cvar_opt::minimize_cvar_py;
pub use robust::robust_optimize_py;
pub use multiperiod::multiperiod_optimize_py;
pub use factor_opt::{estimate_factor_model_py, factor_min_variance_py};
