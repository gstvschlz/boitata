//! Kriging estimators: Ordinary (OK), Simple (SK), and Indicator (IK).
//!
//! All variants use the pseudo-covariance `C(h) = C(0) − γ(h)` from a
//! [`variogram::Variogram`] (which may carry anisotropy), and solve the linear
//! system with an LU factorization (partial pivoting) via nalgebra.

use crate::Sample;
use crate::error::{EstimError, Result};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use variogram::Variogram;

/// Kriging variant.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Kind {
    /// Ordinary kriging: unknown but locally constant mean (Σλ = 1).
    Ordinary,
    /// Simple kriging with a known global mean.
    Simple { mean: f64 },
    /// Indicator kriging of `P(Z ≤ threshold)` (ordinary form on indicators).
    Indicator { threshold: f64 },
}

/// A single kriging estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Estimate {
    pub value: f64,
    /// Kriging variance (≥ 0).
    pub variance: f64,
    pub n_used: usize,
    pub weights: Vec<f64>,
}

/// Krige `target` from `samples` using `vg`.
///
/// `samples` are assumed already selected (see [`crate::search`]). For `Indicator`,
/// each sample value is mapped to `1.0` if `value ≤ threshold` else `0.0`, and the
/// result is clipped to `[0, 1]` (negative weights can push it outside).
pub fn krige(
    kind: Kind,
    target: &(f64, f64, f64),
    samples: &[Sample],
    vg: &Variogram,
) -> Result<Estimate> {
    let n = samples.len();
    if n == 0 {
        return Err(EstimError::InsufficientData("no samples".into()));
    }

    let z: Vec<f64> = match kind {
        Kind::Indicator { threshold } => samples
            .iter()
            .map(|s| if s.value <= threshold { 1.0 } else { 0.0 })
            .collect(),
        _ => samples.iter().map(|s| s.value).collect(),
    };

    let c0 = vg.total_sill();

    // Sample-to-sample covariance matrix and sample-to-target RHS.
    let ordinary = !matches!(kind, Kind::Simple { .. });
    let dim = if ordinary { n + 1 } else { n };

    let mut a = DMatrix::<f64>::zeros(dim, dim);
    let mut b = DVector::<f64>::zeros(dim);

    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = vg.cov_points(&samples[i].loc, &samples[j].loc);
        }
        b[i] = vg.cov_points(&samples[i].loc, target);
    }

    if ordinary {
        for i in 0..n {
            a[(i, n)] = 1.0;
            a[(n, i)] = 1.0;
        }
        a[(n, n)] = 0.0;
        b[n] = 1.0;
    }

    let lu = a.lu();
    let x = lu
        .solve(&b)
        .ok_or_else(|| EstimError::Singular("kriging matrix not invertible".into()))?;

    let weights: Vec<f64> = x.as_slice()[..n].to_vec();

    let value = match kind {
        Kind::Simple { mean } => mean + (0..n).map(|i| weights[i] * (z[i] - mean)).sum::<f64>(),
        Kind::Indicator { .. } => (0..n)
            .map(|i| weights[i] * z[i])
            .sum::<f64>()
            .clamp(0.0, 1.0),
        _ => (0..n).map(|i| weights[i] * z[i]).sum::<f64>(),
    };

    let sum_wc: f64 = (0..n).map(|i| weights[i] * b[i]).sum();
    let variance = if ordinary {
        let mu = x[n];
        (c0 - sum_wc - mu).max(0.0)
    } else {
        (c0 - sum_wc).max(0.0)
    };

    Ok(Estimate {
        value,
        variance,
        n_used: n,
        weights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use variogram::model::Model;

    fn samp(x: f64, y: f64, v: f64) -> Sample {
        Sample {
            loc: (x, y, 0.0),
            value: v,
            hole: None,
        }
    }

    #[test]
    fn ordinary_weights_sum_to_one() {
        let vg = Variogram::single(Model::Spherical, 1.0, 100.0);
        let samples = vec![
            samp(0.0, 0.0, 1.0),
            samp(50.0, 0.0, 2.0),
            samp(0.0, 50.0, 3.0),
        ];
        let est = krige(Kind::Ordinary, &(10.0, 10.0, 0.0), &samples, &vg).unwrap();
        let sum: f64 = est.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "weights sum {sum}");
        assert!(est.variance >= 0.0);
    }

    #[test]
    fn exact_interpolation_at_sample_zero_nugget() {
        // With zero nugget, kriging at a sample location returns that sample's value.
        let vg = Variogram::single(Model::Spherical, 1.0, 100.0);
        let samples = vec![
            samp(0.0, 0.0, 5.0),
            samp(50.0, 0.0, 2.0),
            samp(0.0, 50.0, 3.0),
        ];
        let est = krige(Kind::Ordinary, &(0.0, 0.0, 0.0), &samples, &vg).unwrap();
        assert!((est.value - 5.0).abs() < 1e-6, "got {}", est.value);
        assert!(est.variance < 1e-6, "variance {}", est.variance);
    }

    #[test]
    fn simple_kriging_returns_mean_far_away() {
        // Far from all data, SK returns the global mean and variance → sill.
        let vg = Variogram::single(Model::Spherical, 1.0, 100.0);
        let samples = vec![samp(0.0, 0.0, 5.0), samp(50.0, 0.0, 2.0)];
        let est = krige(
            Kind::Simple { mean: 3.0 },
            &(10000.0, 10000.0, 0.0),
            &samples,
            &vg,
        )
        .unwrap();
        assert!((est.value - 3.0).abs() < 1e-6);
        assert!((est.variance - vg.total_sill()).abs() < 1e-6);
    }

    #[test]
    fn indicator_kriging_gives_probability() {
        let vg = Variogram::single(Model::Spherical, 1.0, 100.0);
        // Values: below/above threshold 2.5.
        let samples = vec![
            samp(0.0, 0.0, 1.0),
            samp(50.0, 0.0, 4.0),
            samp(0.0, 50.0, 2.0),
        ];
        let est = krige(
            Kind::Indicator { threshold: 2.5 },
            &(10.0, 10.0, 0.0),
            &samples,
            &vg,
        )
        .unwrap();
        assert!(
            est.value >= -1e-6 && est.value <= 1.0 + 1e-6,
            "prob {}",
            est.value
        );
    }
}
