//! Trend fitting and removal (detrending).
//!
//! [`detrend`] fits a low-order polynomial in the (x, y, z) coordinates by ordinary
//! least squares and returns residuals plus the coefficients, so the trend can be
//! added back after estimation/simulation of the residuals. [`KernelTrend`] is a
//! smooth moving-window trend: an anisotropic Gaussian kernel average.

use crate::error::{Result, TransformError};
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use rayon::prelude::*;
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

/// Kernel support in bandwidths: weights beyond are below 4e-4 of the peak.
const CUTOFF: f64 = 4.0;

/// A smooth trend: the weighted average of the samples under an anisotropic
/// Gaussian kernel of standard deviation `bandwidth` along the major axis,
/// `ratios` times it along the semi-major and minor axes, rotated by
/// `rotation` (azimuth, dip, rake). Each sample carries one or more values;
/// category indicators give local proportions that sum to 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelTrend {
    pub bandwidth: f64,
    pub rotation: [f64; 3],
    pub ratios: [f64; 2],
    pub locations: Vec<(f64, f64, f64)>,
    pub values: Vec<Vec<f64>>,
    pub weights: Vec<f64>,
    /// Candidate bandwidths and their leave-one-out errors.
    pub bandwidths: Vec<f64>,
    pub scores: Vec<f64>,
}

/// Samples in the rotated frame scaled by the ratios, indexed for search.
struct Frame {
    matrix: Matrix3<f64>,
    ratios: [f64; 2],
    points: Vec<[f64; 3]>,
    tree: ImmutableKdTree<f64, 3>,
}

fn project(matrix: &Matrix3<f64>, ratios: [f64; 2], p: &(f64, f64, f64)) -> [f64; 3] {
    let v = matrix * Vector3::new(p.0, p.1, p.2);
    [v.x, v.y / ratios[0], v.z / ratios[1]]
}

impl Frame {
    fn new(locations: &[(f64, f64, f64)], rotation: [f64; 3], ratios: [f64; 2]) -> Result<Self> {
        let matrix = boitata_core::rotation_matrix(rotation[0], rotation[1], rotation[2]);
        let points: Vec<[f64; 3]> = locations
            .iter()
            .map(|p| project(&matrix, ratios, p))
            .collect();
        let tree = ImmutableKdTree::new_from_slice(&points)
            .map_err(|e| TransformError::InvalidParameters(format!("{e:?}")))?;
        Ok(Self {
            matrix,
            ratios,
            points,
            tree,
        })
    }

    /// Kernel average of `values` at `u`, leaving out sample `skip`; None
    /// without samples within the kernel support.
    fn average(
        &self,
        u: &[f64; 3],
        bandwidth: f64,
        values: &[Vec<f64>],
        weights: &[f64],
        skip: Option<usize>,
    ) -> Option<Vec<f64>> {
        let mut near: Vec<(usize, f64)> = self
            .tree
            .query(u)
            .within::<SquaredEuclidean<f64>>((CUTOFF * bandwidth).powi(2))
            .execute()
            .iter()
            .map(|r| (r.item as usize, r.distance))
            .filter(|&(j, _)| Some(j) != skip)
            .collect();
        near.sort_unstable_by_key(|&(j, _)| j);
        let mut sum = vec![0.0; values[0].len()];
        let mut total = 0.0;
        for (j, d2) in near {
            let w = weights[j] * (-0.5 * d2 / (bandwidth * bandwidth)).exp();
            total += w;
            for (s, v) in sum.iter_mut().zip(&values[j]) {
                *s += w * v;
            }
        }
        (total > 0.0).then(|| sum.into_iter().map(|s| s / total).collect())
    }

    /// Weighted mean squared leave-one-out error; a sample without
    /// neighbors is predicted by the global mean.
    fn loo_error(&self, bandwidth: f64, values: &[Vec<f64>], weights: &[f64], mean: &[f64]) -> f64 {
        let errors: Vec<f64> = (0..values.len())
            .into_par_iter()
            .map(|i| {
                let t = self.average(&self.points[i], bandwidth, values, weights, Some(i));
                let t = t.as_deref().unwrap_or(mean);
                let e: f64 = values[i].iter().zip(t).map(|(v, t)| (v - t).powi(2)).sum();
                weights[i] * e
            })
            .collect();
        errors.iter().sum::<f64>() / weights.iter().sum::<f64>()
    }
}

impl KernelTrend {
    /// Fits the trend of `values` (a row of one or more values per sample)
    /// at `locations`, with optional declustering `weights`. Given several
    /// `bandwidths`, keeps the one with the least leave-one-out error: the
    /// weighted mean squared difference between each sample and the trend
    /// of the others.
    pub fn fit(
        locations: &[(f64, f64, f64)],
        values: &[Vec<f64>],
        weights: Option<&[f64]>,
        bandwidths: &[f64],
        rotation: [f64; 3],
        ratios: [f64; 2],
    ) -> Result<Self> {
        let n = locations.len();
        let invalid = |m: &str| Err(TransformError::InvalidParameters(m.into()));
        if n == 0 {
            return Err(TransformError::InsufficientData("no samples".into()));
        }
        if values.len() != n || weights.is_some_and(|w| w.len() != n) {
            return invalid("one row of values and one weight per sample");
        }
        let k = values[0].len();
        if k == 0
            || values
                .iter()
                .any(|v| v.len() != k || v.iter().any(|x| !x.is_finite()))
        {
            return invalid("values must be finite, as many per sample");
        }
        let weights = weights.map_or_else(|| vec![1.0; n], <[f64]>::to_vec);
        if weights.iter().any(|w| !w.is_finite() || *w < 0.0) || weights.iter().sum::<f64>() <= 0.0
        {
            return invalid("weights must be finite, non-negative and not all zero");
        }
        if bandwidths.is_empty() || bandwidths.iter().any(|b| !b.is_finite() || *b <= 0.0) {
            return invalid("bandwidths must be positive");
        }
        if ratios.iter().any(|r| !r.is_finite() || *r <= 0.0)
            || rotation.iter().any(|a| !a.is_finite())
        {
            return invalid("ratios must be positive and angles finite");
        }
        let frame = Frame::new(locations, rotation, ratios)?;
        let total: f64 = weights.iter().sum();
        let mean: Vec<f64> = (0..k)
            .map(|c| {
                values
                    .iter()
                    .zip(&weights)
                    .map(|(v, w)| w * v[c])
                    .sum::<f64>()
                    / total
            })
            .collect();
        let scores: Vec<f64> = bandwidths
            .iter()
            .map(|&b| frame.loo_error(b, values, &weights, &mean))
            .collect();
        let best = (0..scores.len()).fold(0, |b, i| if scores[i] < scores[b] { i } else { b });
        Ok(Self {
            bandwidth: bandwidths[best],
            rotation,
            ratios,
            locations: locations.to_vec(),
            values: values.to_vec(),
            weights,
            bandwidths: bandwidths.to_vec(),
            scores,
        })
    }

    /// The trend at `targets`; None beyond four bandwidths of every sample.
    pub fn eval(&self, targets: &[(f64, f64, f64)]) -> Result<Vec<Option<Vec<f64>>>> {
        let frame = Frame::new(&self.locations, self.rotation, self.ratios)?;
        Ok(targets
            .par_iter()
            .map(|p| {
                let u = project(&frame.matrix, frame.ratios, p);
                frame.average(&u, self.bandwidth, &self.values, &self.weights, None)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rand_distr::StandardNormal;

    type Point = (f64, f64, f64);

    fn grid(n: usize, step: f64) -> Vec<Point> {
        (0..n * n)
            .map(|i| ((i % n) as f64 * step, (i / n) as f64 * step, 0.0))
            .collect()
    }

    fn fit(locs: &[Point], values: &[f64], bandwidths: &[f64]) -> KernelTrend {
        let rows: Vec<Vec<f64>> = values.iter().map(|&v| vec![v]).collect();
        KernelTrend::fit(locs, &rows, None, bandwidths, [30.0, 0.0, 0.0], [0.5, 1.0]).unwrap()
    }

    #[test]
    fn constant_field_gives_constant_trend() {
        let locs = grid(15, 10.0);
        let t = fit(&locs, &vec![2.5; locs.len()], &[8.0]);
        for v in t.eval(&grid(30, 5.0)).unwrap().into_iter().flatten() {
            assert!((v[0] - 2.5).abs() < 1e-12);
        }
    }

    #[test]
    fn recovers_linear_field_in_the_interior() {
        let locs = grid(41, 5.0);
        let values: Vec<f64> = locs.iter().map(|p| 1.0 + 0.3 * p.0 - 0.2 * p.1).collect();
        let t = fit(&locs, &values, &[6.0]);
        let inner: Vec<Point> = grid(21, 3.0)
            .iter()
            .map(|p| (p.0 + 70.0, p.1 + 70.0, 0.0))
            .collect();
        for (p, v) in inner.iter().zip(t.eval(&inner).unwrap()) {
            let exact = 1.0 + 0.3 * p.0 - 0.2 * p.1;
            assert!((v.unwrap()[0] - exact).abs() < 1e-3 * exact.abs().max(1.0));
        }
    }

    #[test]
    fn proportions_sum_to_one() {
        let mut rng = StdRng::seed_from_u64(3);
        let locs: Vec<Point> = (0..300)
            .map(|_| (rng.r#gen::<f64>() * 100.0, rng.r#gen::<f64>() * 100.0, 0.0))
            .collect();
        let rows: Vec<Vec<f64>> = locs
            .iter()
            .map(|p| {
                let c = ((p.0 + 20.0 * rng.r#gen::<f64>()) / 40.0) as usize;
                (0..3).map(|j| f64::from(u8::from(j == c.min(2)))).collect()
            })
            .collect();
        let weights: Vec<f64> = (0..300).map(|i| 0.5 + (i % 3) as f64).collect();
        let t = KernelTrend::fit(
            &locs,
            &rows,
            Some(&weights),
            &[5.0, 15.0],
            [0.0; 3],
            [1.0; 2],
        )
        .unwrap();
        for p in t.eval(&grid(25, 4.0)).unwrap().into_iter().flatten() {
            assert!(p.iter().all(|&x| (0.0..=1.0).contains(&x)));
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn noisier_data_pick_a_larger_bandwidth() {
        let locs = grid(30, 5.0);
        let candidates = [2.0, 4.0, 6.0, 9.0, 13.0, 20.0, 30.0];
        let chosen = |noise: f64| {
            let mut rng = StdRng::seed_from_u64(11);
            let values: Vec<f64> = locs
                .iter()
                .map(|p| {
                    let e: f64 = rng.sample(StandardNormal);
                    (p.0 / 25.0).sin() + (p.1 / 30.0).cos() + noise * e
                })
                .collect();
            fit(&locs, &values, &candidates).bandwidth
        };
        let (quiet, noisy) = (chosen(0.05), chosen(1.0));
        assert!(noisy > quiet, "{quiet} vs {noisy}");
    }

    #[test]
    fn identical_for_any_thread_count() {
        let mut rng = StdRng::seed_from_u64(7);
        let locs: Vec<Point> = (0..400)
            .map(|_| (rng.r#gen::<f64>() * 100.0, rng.r#gen::<f64>() * 100.0, 0.0))
            .collect();
        let values: Vec<f64> = (0..400).map(|_| rng.sample(StandardNormal)).collect();
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let t = fit(&locs, &values, &[3.0, 6.0, 12.0]);
                    (t.scores.clone(), t.eval(&grid(20, 5.0)).unwrap())
                })
        };
        assert_eq!(run(1), run(4));
    }

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
