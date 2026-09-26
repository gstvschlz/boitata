//! Simple (variogram-free) interpolators.
//!
//! Extends the bare [`crate::idw`] with the full option set: an anisotropic
//! distance, a maximum search distance `dmax`, and several estimators —
//! inverse-distance, nearest, moving average, moving median, and local
//! least-squares (local polynomial trend). Each returns an optional local
//! dispersion (`stdev`) alongside the estimate.

use crate::Sample;
use crate::error::{EstimError, Result};
use nalgebra::{DMatrix, DVector};
use variogram::aniso::{Anisotropy, euclidean};

/// Shared options for the simple interpolators.
#[derive(Debug, Clone)]
pub struct InterpOptions {
    /// Maximum (possibly anisotropic) distance a datum may contribute from.
    pub dmax: f64,
    /// Optional anisotropy for the distance metric.
    pub anisotropy: Option<Anisotropy>,
    /// Minimum neighbors required, else an error.
    pub min_samples: usize,
}

impl Default for InterpOptions {
    fn default() -> Self {
        Self {
            dmax: f64::INFINITY,
            anisotropy: None,
            min_samples: 1,
        }
    }
}

/// An interpolation result: value, optional local dispersion, and the neighbor
/// count actually used.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterpEstimate {
    pub value: f64,
    /// Local weighted standard deviation of the neighbor values (dispersion).
    pub stdev: Option<f64>,
    pub n_used: usize,
}

fn dist(target: &(f64, f64, f64), loc: &(f64, f64, f64), opts: &InterpOptions) -> f64 {
    match &opts.anisotropy {
        Some(a) => a.lag(target, loc),
        None => euclidean(target, loc),
    }
}

/// Neighbors (index, distance) within `dmax`, sorted by increasing distance.
fn within(target: &(f64, f64, f64), samples: &[Sample], opts: &InterpOptions) -> Vec<(usize, f64)> {
    let mut v: Vec<(usize, f64)> = samples
        .iter()
        .enumerate()
        .map(|(i, s)| (i, dist(target, &s.loc, opts)))
        .filter(|(_, d)| *d <= opts.dmax)
        .collect();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    v
}

/// Weighted local standard deviation of `values` about their weighted mean.
fn weighted_stdev(values: &[f64], weights: &[f64], mean: f64) -> f64 {
    let wsum: f64 = weights.iter().sum();
    if wsum <= 0.0 {
        return 0.0;
    }
    let var = values
        .iter()
        .zip(weights)
        .map(|(v, w)| w * (v - mean).powi(2))
        .sum::<f64>()
        / wsum;
    var.max(0.0).sqrt()
}

fn check_min(n: usize, opts: &InterpOptions) -> Result<()> {
    if n < opts.min_samples.max(1) {
        return Err(EstimError::SearchFailed(format!(
            "found {n} neighbors within dmax, need {}",
            opts.min_samples.max(1)
        )));
    }
    Ok(())
}

/// Inverse-distance weighting with exponent `power`, anisotropy and `dmax`.
pub fn inverse_distance(
    target: &(f64, f64, f64),
    samples: &[Sample],
    power: f64,
    opts: &InterpOptions,
) -> Result<InterpEstimate> {
    let neigh = within(target, samples, opts);
    check_min(neigh.len(), opts)?;
    if let Some(&(i, d)) = neigh.first()
        && d < 1e-12
    {
        return Ok(InterpEstimate {
            value: samples[i].value,
            stdev: Some(0.0),
            n_used: 1,
        });
    }
    let (mut num, mut den) = (0.0, 0.0);
    let mut vals = Vec::new();
    let mut ws = Vec::new();
    for &(i, d) in &neigh {
        let w = 1.0 / d.powf(power);
        num += w * samples[i].value;
        den += w;
        vals.push(samples[i].value);
        ws.push(w);
    }
    let value = num / den;
    Ok(InterpEstimate {
        value,
        stdev: Some(weighted_stdev(&vals, &ws, value)),
        n_used: neigh.len(),
    })
}

/// Nearest-neighbor within `dmax`.
pub fn nearest(
    target: &(f64, f64, f64),
    samples: &[Sample],
    opts: &InterpOptions,
) -> Result<InterpEstimate> {
    let neigh = within(target, samples, opts);
    check_min(neigh.len(), opts)?;
    let (i, _) = neigh[0];
    Ok(InterpEstimate {
        value: samples[i].value,
        stdev: None,
        n_used: 1,
    })
}

/// Moving average: equal-weight mean of neighbors within `dmax`.
pub fn moving_average(
    target: &(f64, f64, f64),
    samples: &[Sample],
    opts: &InterpOptions,
) -> Result<InterpEstimate> {
    let neigh = within(target, samples, opts);
    check_min(neigh.len(), opts)?;
    let vals: Vec<f64> = neigh.iter().map(|&(i, _)| samples[i].value).collect();
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let ws = vec![1.0; vals.len()];
    Ok(InterpEstimate {
        value: mean,
        stdev: Some(weighted_stdev(&vals, &ws, mean)),
        n_used: vals.len(),
    })
}

/// Moving median: robust median of neighbors within `dmax`.
pub fn moving_median(
    target: &(f64, f64, f64),
    samples: &[Sample],
    opts: &InterpOptions,
) -> Result<InterpEstimate> {
    let neigh = within(target, samples, opts);
    check_min(neigh.len(), opts)?;
    let mut vals: Vec<f64> = neigh.iter().map(|&(i, _)| samples[i].value).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = vals.len();
    let value = if m % 2 == 1 {
        vals[m / 2]
    } else {
        0.5 * (vals[m / 2 - 1] + vals[m / 2])
    };
    let mean = vals.iter().sum::<f64>() / m as f64;
    let ws = vec![1.0; m];
    Ok(InterpEstimate {
        value,
        stdev: Some(weighted_stdev(&vals, &ws, mean)),
        n_used: m,
    })
}

/// Local least-squares: fit a local polynomial trend of `degree` (0 = mean,
/// 1 = linear, 2 = quadratic) to the neighbors and evaluate it at `target`.
pub fn local_least_squares(
    target: &(f64, f64, f64),
    samples: &[Sample],
    degree: usize,
    opts: &InterpOptions,
) -> Result<InterpEstimate> {
    let neigh = within(target, samples, opts);
    check_min(neigh.len(), opts)?;
    // Local monomials in coordinates relative to the target.
    let basis = |p: &(f64, f64, f64)| -> Vec<f64> {
        let (x, y, z) = (p.0 - target.0, p.1 - target.1, p.2 - target.2);
        let mut m = vec![1.0];
        if degree >= 1 {
            m.extend_from_slice(&[x, y, z]);
        }
        if degree >= 2 {
            m.extend_from_slice(&[x * x, y * y, z * z, x * y, x * z, y * z]);
        }
        m
    };
    let p = basis(target).len();
    let n = neigh.len();
    if n < p {
        // Under-determined: fall back to the local mean.
        return moving_average(target, samples, opts);
    }
    let a = DMatrix::from_fn(n, p, |i, l| basis(&samples[neigh[i].0].loc)[l]);
    let b = DVector::from_fn(n, |i, _| samples[neigh[i].0].value);
    // Min-norm least squares via the SVD pseudo-inverse — robust to a
    // rank-deficient design (e.g. a degenerate axis in 2-D data).
    let pinv = a
        .clone()
        .pseudo_inverse(1e-12)
        .map_err(|e| EstimError::Singular(format!("local least-squares: {e}")))?;
    let beta = &pinv * &b;
    // Value at target = β₀ (constant term, since basis(target) = [1,0,0,...]).
    let value = beta[0];
    // Residual dispersion.
    let fitted = &a * &beta;
    let resid: Vec<f64> = (0..n).map(|i| b[i] - fitted[i]).collect();
    let ws = vec![1.0; n];
    let stdev = weighted_stdev(&resid, &ws, 0.0);
    Ok(InterpEstimate {
        value,
        stdev: Some(stdev),
        n_used: n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(x: f64, y: f64, v: f64) -> Sample {
        Sample {
            loc: (x, y, 0.0),
            value: v,
            hole: None,
            error_variance: 0.0,
            domain: None,
        }
    }

    #[test]
    fn idw_respects_dmax() {
        let samples = vec![s(0.0, 0.0, 10.0), s(100.0, 0.0, 20.0)];
        let opts = InterpOptions {
            dmax: 50.0,
            ..Default::default()
        };
        // Only the near datum is within dmax → estimate ≈ 10.
        let e = inverse_distance(&(10.0, 0.0, 0.0), &samples, 2.0, &opts).unwrap();
        assert!((e.value - 10.0).abs() < 1e-9, "value {}", e.value);
        assert_eq!(e.n_used, 1);
    }

    #[test]
    fn moving_average_and_median() {
        let samples = vec![s(1.0, 0.0, 1.0), s(2.0, 0.0, 2.0), s(3.0, 0.0, 9.0)];
        let opts = InterpOptions::default();
        let avg = moving_average(&(0.0, 0.0, 0.0), &samples, &opts).unwrap();
        assert!((avg.value - 4.0).abs() < 1e-9); // (1+2+9)/3
        let med = moving_median(&(0.0, 0.0, 0.0), &samples, &opts).unwrap();
        assert!((med.value - 2.0).abs() < 1e-9); // robust to the outlier 9
    }

    #[test]
    fn local_least_squares_recovers_plane() {
        // Data on z = 2x + 3y + 1; degree-1 local fit is exact at the target.
        let f = |x: f64, y: f64| 2.0 * x + 3.0 * y + 1.0;
        let samples: Vec<Sample> = [
            (0.0, 0.0),
            (10.0, 0.0),
            (0.0, 10.0),
            (10.0, 10.0),
            (5.0, 5.0),
        ]
        .iter()
        .map(|&(x, y)| s(x, y, f(x, y)))
        .collect();
        let e =
            local_least_squares(&(4.0, 6.0, 0.0), &samples, 1, &InterpOptions::default()).unwrap();
        assert!(
            (e.value - f(4.0, 6.0)).abs() < 1e-6,
            "value {} vs {}",
            e.value,
            f(4.0, 6.0)
        );
        assert!(e.stdev.unwrap() < 1e-6, "residual stdev {:?}", e.stdev);
    }

    #[test]
    fn nearest_within_dmax_errors_when_empty() {
        let samples = vec![s(100.0, 0.0, 5.0)];
        let opts = InterpOptions {
            dmax: 10.0,
            min_samples: 1,
            anisotropy: None,
        };
        assert!(nearest(&(0.0, 0.0, 0.0), &samples, &opts).is_err());
    }
}
