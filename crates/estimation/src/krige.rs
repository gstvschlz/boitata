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
    /// Lagrange multipliers times their drift terms at the target: μ for
    /// ordinary kriging, 0 for simple kriging; NaN where it does not apply.
    pub lagrange: f64,
    /// Variance of the estimated support C(v, v): the sill for points, the
    /// mean covariance within the block for blocks; NaN where it does not apply.
    pub support_variance: f64,
}

impl Estimate {
    /// Kriging efficiency (C(v, v) − σ²) / C(v, v): 1 when the estimate is
    /// exact, 0 or less when it is no better than the mean.
    pub fn efficiency(&self) -> f64 {
        (self.support_variance - self.variance) / self.support_variance
    }

    /// Slope of the regression of true on estimated values, Cov(Z, Z*) /
    /// Var(Z*); below 1 the estimate is conditionally biased.
    pub fn slope(&self) -> f64 {
        self.covariance() / self.estimate_variance()
    }

    /// Cov(Z, Z*) = C(v, v) − σ² − μ.
    fn covariance(&self) -> f64 {
        self.support_variance - self.variance - self.lagrange
    }

    /// Variance of the estimator, Var(Z*) = C(v, v) − σ² − 2μ; the further
    /// below C(v, v), the smoother the estimates.
    pub fn estimate_variance(&self) -> f64 {
        self.covariance() - self.lagrange
    }

    /// Sum of the negative weights (≤ 0); NaN for estimators without weights.
    pub fn negative_weight_sum(&self) -> f64 {
        if self.weights.is_empty() {
            return f64::NAN;
        }
        self.weights.iter().filter(|w| **w < 0.0).sum()
    }
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
        a[(i, i)] += samples[i].error_variance;
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
    let mu = if ordinary { x[n] } else { 0.0 };
    let variance = (c0 - sum_wc - mu).max(0.0);

    Ok(Estimate {
        value,
        variance,
        n_used: n,
        weights,
        lagrange: mu,
        support_variance: c0,
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
            error_variance: 0.0,
            domain: None,
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
    fn efficiency_and_slope() {
        let vg = Variogram::single(Model::Spherical, 1.0, 100.0);
        let samples = vec![
            samp(0.0, 0.0, 5.0),
            samp(50.0, 0.0, 2.0),
            samp(0.0, 50.0, 3.0),
            samp(60.0, 70.0, 1.0),
        ];
        let at_datum = krige(Kind::Ordinary, &(0.0, 0.0, 0.0), &samples, &vg).unwrap();
        assert!((at_datum.efficiency() - 1.0).abs() < 1e-9);
        assert!((at_datum.slope() - 1.0).abs() < 1e-9);

        let target = (90.0, 20.0, 0.0);
        let simple = krige(Kind::Simple { mean: 2.0 }, &target, &samples, &vg).unwrap();
        assert_eq!(simple.slope(), 1.0);

        let ok = krige(Kind::Ordinary, &target, &samples, &vg).unwrap();
        let w = &ok.weights;
        let cov: f64 = (0..4)
            .map(|i| w[i] * vg.cov_points(&samples[i].loc, &target))
            .sum();
        let var: f64 = (0..4)
            .flat_map(|i| (0..4).map(move |j| (i, j)))
            .map(|(i, j)| w[i] * w[j] * vg.cov_points(&samples[i].loc, &samples[j].loc))
            .sum();
        assert!((ok.slope() - cov / var).abs() < 1e-9);
        assert!((ok.estimate_variance() - var).abs() < 1e-9);
        assert!(ok.slope() < 1.0 && ok.efficiency() < 1.0);
    }

    #[test]
    fn measurement_error_blends_the_datum_with_its_neighbors() {
        let vg = Variogram::single(Model::Spherical, 1.0, 100.0);
        let others = vec![
            samp(50.0, 0.0, 2.0),
            samp(0.0, 50.0, 3.0),
            samp(60.0, 70.0, 1.0),
        ];
        let target = (0.0, 0.0, 0.0);
        for kind in [Kind::Ordinary, Kind::Simple { mean: 2.0 }] {
            let loo = krige(kind, &target, &others, &vg).unwrap();
            let with = |error_variance| {
                let datum = Sample {
                    error_variance,
                    ..samp(0.0, 0.0, 5.0)
                };
                let all: Vec<Sample> = std::iter::once(datum).chain(others.clone()).collect();
                krige(kind, &target, &all, &vg).unwrap()
            };
            assert!((with(0.0).value - 5.0).abs() < 1e-9);
            for e in [0.1, 0.5, 2.0] {
                let est = with(e);
                let k = loo.variance / (loo.variance + e);
                assert!((est.value - (loo.value + k * (5.0 - loo.value))).abs() < 1e-9);
                assert!((est.variance - loo.variance * (1.0 - k)).abs() < 1e-9);
                assert!(est.value > loo.value && est.value < 5.0);
            }
        }
    }

    #[test]
    fn negative_weight_sum_adds_the_negative_weights() {
        let vg = Variogram::single(Model::Gaussian, 1.0, 60.0);
        let samples = vec![
            samp(10.0, 0.0, 1.0),
            samp(20.0, 0.0, 2.0),
            samp(0.0, 15.0, 3.0),
            samp(-30.0, -5.0, 4.0),
        ];
        let est = krige(Kind::Ordinary, &(0.0, 0.0, 0.0), &samples, &vg).unwrap();
        let negative: f64 = est.weights.iter().filter(|w| **w < 0.0).sum();
        assert!(negative < 0.0, "{:?}", est.weights);
        assert_eq!(est.negative_weight_sum(), negative);
        let positive = krige(Kind::Ordinary, &(10.0, 1.0, 0.0), &samples[..1], &vg).unwrap();
        assert_eq!(positive.negative_weight_sum(), 0.0);
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
