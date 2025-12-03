//! Hierarchical Risk Parity (HRP)
//!
//! A modern portfolio allocation method by López de Prado (2016)
//!
//! # Why HRP?
//!
//! Traditional methods have problems:
//! - **Mean-Variance**: Needs μ estimates, sensitive to errors
//! - **Risk Parity**: Still needs covariance inversion or iteration
//! - **Both**: Often produce concentrated portfolios
//!
//! HRP solves these by:
//! - Not requiring expected return estimates
//! - Not requiring matrix inversion
//! - Naturally producing diversified portfolios
//!
//! # The Algorithm (3 Steps)
//!
//! ## Step 1: Tree Clustering
//!
//! Convert correlation matrix to distance matrix:
//! ```text
//! d_ij = √((1 - ρ_ij) / 2)
//! ```
//!
//! Then apply hierarchical clustering (single linkage).
//!
//! ## Step 2: Quasi-Diagonalization
//!
//! Reorder assets so similar ones are adjacent.
//! This makes the covariance matrix "quasi-diagonal" (block structure).
//!
//! ## Step 3: Recursive Bisection
//!
//! Split portfolio recursively:
//! ```text
//! For each cluster split:
//!   1. Calculate cluster variances (left vs right)
//!   2. Allocate inversely: w_left ∝ 1/var_left, w_right ∝ 1/var_right
//!   3. Recurse into each sub-cluster
//! ```
//!
//! # References
//!
//! - López de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out-of-Sample"
//! - Journal of Portfolio Management, 42(4), 59-69

use nalgebra::DMatrix;
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Correlation to distance transformation
///
/// ```text
/// d_ij = √((1 - ρ_ij) / 2)
/// ```
///
/// Maps correlation [-1, 1] to distance [0, 1]:
/// - ρ = 1 (perfect positive) → d = 0
/// - ρ = 0 (uncorrelated) → d = 0.707
/// - ρ = -1 (perfect negative) → d = 1
fn correlation_to_distance(corr: &DMatrix<f64>) -> DMatrix<f64> {
    let n = corr.nrows();
    let mut dist = DMatrix::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            dist[(i, j)] = ((1.0 - corr[(i, j)]) / 2.0).sqrt();
        }
    }

    dist
}

/// Covariance to correlation matrix
fn cov_to_corr(cov: &DMatrix<f64>) -> DMatrix<f64> {
    let n = cov.nrows();
    let mut corr = DMatrix::zeros(n, n);

    let std_devs: Vec<f64> = (0..n).map(|i| cov[(i, i)].sqrt()).collect();

    for i in 0..n {
        for j in 0..n {
            if std_devs[i] > 1e-10 && std_devs[j] > 1e-10 {
                corr[(i, j)] = cov[(i, j)] / (std_devs[i] * std_devs[j]);
            } else {
                corr[(i, j)] = if i == j { 1.0 } else { 0.0 };
            }
        }
    }

    corr
}

/// Single-linkage hierarchical clustering
///
/// Returns the order of assets (quasi-diagonalization)
fn single_linkage_order(dist: &DMatrix<f64>) -> Vec<usize> {
    let n = dist.nrows();

    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0];
    }

    // Track which clusters each item belongs to
    // clusters[i] = list of original indices in cluster i
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut active: Vec<bool> = vec![true; n];

    // Working distance matrix (we'll modify it during clustering)
    let mut work_dist = dist.clone();

    // Merge n-1 times to get single cluster
    for _ in 0..(n - 1) {
        // Find minimum distance between active clusters
        let mut min_dist = f64::MAX;
        let mut min_i = 0;
        let mut min_j = 0;

        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                if work_dist[(i, j)] < min_dist {
                    min_dist = work_dist[(i, j)];
                    min_i = i;
                    min_j = j;
                }
            }
        }

        // Merge clusters i and j into i
        let cluster_j = clusters[min_j].clone();
        clusters[min_i].extend(cluster_j);
        clusters[min_j].clear();
        active[min_j] = false;

        // Update distances using single linkage (minimum)
        for k in 0..n {
            if !active[k] || k == min_i {
                continue;
            }
            let new_dist = work_dist[(min_i, k)].min(work_dist[(min_j, k)]);
            work_dist[(min_i, k)] = new_dist;
            work_dist[(k, min_i)] = new_dist;
        }
    }

    // Find the final merged cluster (should be the only active one)
    for i in 0..n {
        if active[i] {
            return clusters[i].clone();
        }
    }

    // Fallback: return original order
    (0..n).collect()
}

/// Calculate cluster variance
///
/// For a subset of assets, calculate the variance of an equally-weighted portfolio
fn cluster_variance(cov: &DMatrix<f64>, indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }

    let n = indices.len();
    let weight = 1.0 / n as f64;

    let mut variance = 0.0;
    for &i in indices {
        for &j in indices {
            variance += weight * weight * cov[(i, j)];
        }
    }

    variance
}

/// Recursive bisection for weight allocation
///
/// This is the core of HRP:
/// 1. Split assets into two clusters
/// 2. Allocate inversely to cluster variance
/// 3. Recurse into each cluster
fn recursive_bisection(
    cov: &DMatrix<f64>,
    ordered_indices: &[usize],
    weights: &mut [f64],
    weight_factor: f64,
) {
    if ordered_indices.len() == 1 {
        weights[ordered_indices[0]] = weight_factor;
        return;
    }

    if ordered_indices.len() == 0 {
        return;
    }

    // Split in half
    let mid = ordered_indices.len() / 2;
    let left = &ordered_indices[..mid];
    let right = &ordered_indices[mid..];

    // Calculate cluster variances
    let var_left = cluster_variance(cov, left).max(1e-10);
    let var_right = cluster_variance(cov, right).max(1e-10);

    // Allocate inversely to variance
    let alpha = var_right / (var_left + var_right);

    // Recurse
    recursive_bisection(cov, left, weights, weight_factor * alpha);
    recursive_bisection(cov, right, weights, weight_factor * (1.0 - alpha));
}

/// Hierarchical Risk Parity weights
///
/// # Algorithm
/// 1. Convert covariance → correlation → distance
/// 2. Hierarchical clustering to get asset ordering
/// 3. Recursive bisection to allocate weights
///
/// # Arguments
/// * `cov` - Covariance matrix (n × n)
///
/// # Returns
/// * Portfolio weights (n × 1, sum to 1)
pub fn hrp_weights(cov: &DMatrix<f64>) -> Vec<f64> {
    let n = cov.nrows();

    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![1.0];
    }

    // Step 1: Correlation and distance
    let corr = cov_to_corr(cov);
    let dist = correlation_to_distance(&corr);

    // Step 2: Hierarchical clustering → quasi-diagonal ordering
    let ordered = single_linkage_order(&dist);

    // Step 3: Recursive bisection
    let mut weights = vec![0.0; n];
    recursive_bisection(cov, &ordered, &mut weights, 1.0);

    weights
}

// ============ Python Binding ============

/// Calculate HRP weights (Python)
///
/// Hierarchical Risk Parity portfolio optimization.
/// No expected returns needed, no matrix inversion.
///
/// # Arguments
/// * `cov` - Covariance matrix (n × n)
///
/// # Returns
/// * Portfolio weights (sum to 1)
#[pyfunction]
#[pyo3(name = "hrp_weights")]
pub fn hrp_weights_py<'py>(
    py: Python<'py>,
    cov: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let c = cov.as_array();
    let n = c.nrows();

    let cov_mat = DMatrix::from_row_slice(n, n, c.as_slice().unwrap());
    let weights = hrp_weights(&cov_mat);

    Ok(PyArray1::from_slice_bound(py, &weights))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_cov() -> DMatrix<f64> {
        // 4 assets with 2 clusters:
        // Assets 0,1 are correlated (tech stocks)
        // Assets 2,3 are correlated (utility stocks)
        // Low correlation between clusters
        DMatrix::from_row_slice(4, 4, &[
            0.04, 0.02, 0.005, 0.003,   // Asset 0
            0.02, 0.05, 0.004, 0.002,   // Asset 1
            0.005, 0.004, 0.02, 0.01,   // Asset 2
            0.003, 0.002, 0.01, 0.015,  // Asset 3
        ])
    }

    #[test]
    fn test_correlation_to_distance() {
        let corr = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);

        let dist = correlation_to_distance(&corr);

        // Self-distance should be 0
        assert_relative_eq!(dist[(0, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(dist[(1, 1)], 0.0, epsilon = 1e-10);

        // Correlation 0.5 → distance = sqrt((1-0.5)/2) = 0.5
        assert_relative_eq!(dist[(0, 1)], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_hrp_weights_sum_to_one() {
        let cov = create_test_cov();
        let weights = hrp_weights(&cov);

        let sum: f64 = weights.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hrp_weights_all_positive() {
        let cov = create_test_cov();
        let weights = hrp_weights(&cov);

        for w in &weights {
            assert!(*w > 0.0);
        }
    }

    #[test]
    fn test_hrp_diversification() {
        // High volatility asset should get less weight
        let cov = DMatrix::from_row_slice(2, 2, &[
            0.01, 0.0,   // Low vol asset
            0.0, 0.25,   // High vol asset (5x)
        ]);

        let weights = hrp_weights(&cov);

        // Lower vol should have higher weight
        assert!(weights[0] > weights[1]);
    }

    #[test]
    fn test_single_asset() {
        let cov = DMatrix::from_row_slice(1, 1, &[0.04]);
        let weights = hrp_weights(&cov);

        assert_eq!(weights.len(), 1);
        assert_relative_eq!(weights[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hrp_vs_equal_weight() {
        // For identical uncorrelated assets, HRP should give equal weights
        let cov = DMatrix::from_row_slice(3, 3, &[
            0.04, 0.0, 0.0,
            0.0, 0.04, 0.0,
            0.0, 0.0, 0.04,
        ]);

        let weights = hrp_weights(&cov);

        for w in &weights {
            // Should be roughly equal (1/3)
            assert_relative_eq!(*w, 1.0 / 3.0, epsilon = 0.01);
        }
    }
}
