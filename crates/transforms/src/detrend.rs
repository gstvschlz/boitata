//! Polynomial trend fitting and removal (detrending).
//!
//! Fits a low-order polynomial in the (x, y, z) coordinates by ordinary least squares
//! and returns residuals plus the coefficients, so the trend can be added back after
//! estimation/simulation of the residuals.

use crate::error::{Result, TransformError};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A fitted polynomial trend of a given degree (0, 1, or 2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trend {
    pub degree: usize,
    pub coeffs: Vec<f64>,
}

/// Build the polynomial basis row for a location at the given degree.
fn basis(p: &(f64, f64, f64), degree: usize) -> Vec<f64> {
    let (x, y, z) = *p;
    let mut b = vec![1.0];
    if degree >= 1 {
        b.extend_from_slice(&[x, y, z]);
    }
    if degree >= 2 {
        b.extend_from_slice(&[x * x, y * y, z * z, x * y, x * z, y * z]);
    }
    b
}

impl Trend {
    /// Evaluate the trend at a location.
    pub fn eval(&self, p: &(f64, f64, f64)) -> f64 {
        basis(p, self.degree)
            .iter()
            .zip(&self.coeffs)
            .map(|(b, c)| b * c)
            .sum()
    }
}

/// Fit a polynomial trend of `degree` (0–2) and return it with the residuals.
pub fn detrend(
    locations: &[(f64, f64, f64)],
    values: &[f64],
    degree: usize,
) -> Result<(Trend, Vec<f64>)> {
    let n = locations.len();
    if n != values.len() {
        return Err(TransformError::InvalidParameters("length mismatch".into()));
    }
    if degree > 2 {
        return Err(TransformError::InvalidParameters(
            "degree must be 0, 1, or 2".into(),
        ));
    }
    let p = basis(&locations[0], degree).len();
    if n < p {
        return Err(TransformError::InsufficientData(format!(
            "need ≥ {p} samples for degree {degree}"
        )));
    }

    // Design matrix A (n×p) and response b.
    let mut a = DMatrix::<f64>::zeros(n, p);
    let mut b = DVector::<f64>::zeros(n);
    for i in 0..n {
        let row = basis(&locations[i], degree);
        for j in 0..p {
            a[(i, j)] = row[j];
        }
        b[i] = values[i];
    }

    // Least-squares via SVD: robust to rank-deficient designs (e.g. 2D data where the
    // z-column is constant), returning the minimum-norm solution.
    let svd = a.svd(true, true);
    let coeffs = svd
        .solve(&b, 1e-12)
        .map_err(|e| TransformError::FittingFailed(format!("SVD solve failed: {e}")))?;

    let trend = Trend {
        degree,
        coeffs: coeffs.as_slice().to_vec(),
    };
    let residuals: Vec<f64> = (0..n)
        .map(|i| values[i] - trend.eval(&locations[i]))
        .collect();
    Ok((trend, residuals))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_linear_trend() {
        // value = 3 + 2x - y ; residuals should be ~0.
        let locs: Vec<(f64, f64, f64)> = (0..20).map(|i| (i as f64, (i % 5) as f64, 0.0)).collect();
        let vals: Vec<f64> = locs.iter().map(|p| 3.0 + 2.0 * p.0 - p.1).collect();
        let (trend, resid) = detrend(&locs, &vals, 1).unwrap();
        assert!(resid.iter().all(|r| r.abs() < 1e-6));
        // Coeff order: [1, x, y, z].
        assert!((trend.coeffs[0] - 3.0).abs() < 1e-6);
        assert!((trend.coeffs[1] - 2.0).abs() < 1e-6);
        assert!((trend.coeffs[2] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn degree_zero_is_mean() {
        let locs = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)];
        let vals = vec![1.0, 2.0, 3.0];
        let (trend, _resid) = detrend(&locs, &vals, 0).unwrap();
        assert!((trend.coeffs[0] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn add_back_reconstructs() {
        let locs: Vec<(f64, f64, f64)> = (0..15).map(|i| (i as f64, 2.0 * i as f64, 0.0)).collect();
        let vals: Vec<f64> = locs
            .iter()
            .map(|p| 1.0 + 0.5 * p.0 + 0.1 * p.1 * p.1)
            .collect();
        let (trend, resid) = detrend(&locs, &vals, 2).unwrap();
        for i in 0..locs.len() {
            let recon = trend.eval(&locs[i]) + resid[i];
            assert!((recon - vals[i]).abs() < 1e-6);
        }
    }
}
