//! Spatial bootstrap: uncertainty in global statistics of correlated data.
//!
//! Unconditional Gaussian values are drawn at the data locations with the
//! correlation of a normal-score variogram (one Cholesky factor, reused), turned
//! into uniform ranks, and read through the declustered distribution of the
//! data. Each draw gives one resampled data set and its statistics. With no
//! spatial correlation this is the classical bootstrap; with correlation the
//! resampled values cluster and the statistics spread more.

use nalgebra::{DMatrix, DVector};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use variogram::Variogram;

use crate::error::{Result, TransformError};
use crate::normal::phi;

/// Statistics of each bootstrap realization.
#[derive(Debug, Clone, PartialEq)]
pub struct Bootstrap {
    /// Mean of each realization.
    pub mean: Vec<f64>,
    /// Per realization, the quantile at each requested probability.
    pub quantiles: Vec<Vec<f64>>,
    /// Per realization, the fraction of values above each cutoff.
    pub above: Vec<Vec<f64>>,
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Seed of realization `k`, mixed from the user seed so nearby seeds do not
/// share realizations.
fn realization_seed(seed: u64, k: usize) -> u64 {
    splitmix(splitmix(seed) ^ k as u64)
}

/// `n` resampled data sets of `values` at `coords`, drawn with the correlation of
/// `variogram` (a normal-score variogram; only its shape matters) from the
/// distribution of `values` weighted by `weights`.
#[allow(clippy::too_many_arguments)]
pub fn spatial_bootstrap(
    coords: &[(f64, f64, f64)],
    values: &[f64],
    weights: Option<&[f64]>,
    variogram: &Variogram,
    n: usize,
    seed: u64,
    quantiles: &[f64],
    cutoffs: &[f64],
) -> Result<Bootstrap> {
    let invalid = |m: &str| TransformError::InvalidParameters(m.into());
    let m = values.len();
    if m == 0 {
        return Err(TransformError::InsufficientData("no samples".into()));
    }
    if coords.len() != m {
        return Err(invalid("coords and values differ in length"));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("values must be finite"));
    }
    let weights = weights.map_or_else(|| vec![1.0; m], <[f64]>::to_vec);
    if weights.len() != m
        || weights.iter().any(|w| !w.is_finite() || *w < 0.0)
        || weights.iter().sum::<f64>() <= 0.0
    {
        return Err(invalid(
            "weights must match values, be >= 0 and sum above 0",
        ));
    }
    if quantiles.iter().any(|p| !(0.0..=1.0).contains(p)) {
        return Err(invalid("quantiles must be within [0, 1]"));
    }
    if cutoffs.iter().any(|c| c.is_nan()) {
        return Err(invalid("cutoffs must not be NaN"));
    }
    let sill = variogram.total_sill();
    if !variogram.is_stationary() || !(sill.is_finite() && sill > 0.0) {
        return Err(invalid("the variogram needs a finite, positive sill"));
    }

    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let sorted: Vec<f64> = order.iter().map(|&i| values[i]).collect();
    let total: f64 = weights.iter().sum();
    let cdf: Vec<f64> = order
        .iter()
        .scan(0.0, |c, &i| {
            *c += weights[i] / total;
            Some(*c)
        })
        .collect();
    let draw = |u: f64| sorted[cdf.partition_point(|&c| c < u).min(m - 1)];

    let factor = cholesky(coords, variogram, sill)?;
    let realizations: Vec<(f64, Vec<f64>, Vec<f64>)> = (0..n)
        .into_par_iter()
        .map(|k| {
            let mut rng = StdRng::seed_from_u64(realization_seed(seed, k));
            let w = DVector::from_fn(m, |_, _| StandardNormal.sample(&mut rng));
            let y = &factor * w;
            let mut x: Vec<f64> = y.iter().map(|&y| draw(phi(y))).collect();
            x.sort_by(f64::total_cmp);
            let mean = x.iter().sum::<f64>() / m as f64;
            let q = quantiles.iter().map(|&p| quantile(&x, p)).collect();
            let above = cutoffs
                .iter()
                .map(|&c| (m - x.partition_point(|&v| v <= c)) as f64 / m as f64)
                .collect();
            (mean, q, above)
        })
        .collect();
    let mut out = Bootstrap {
        mean: Vec::with_capacity(n),
        quantiles: Vec::with_capacity(n),
        above: Vec::with_capacity(n),
    };
    for (mean, q, a) in realizations {
        out.mean.push(mean);
        out.quantiles.push(q);
        out.above.push(a);
    }
    Ok(out)
}

/// Lower Cholesky factor of the correlation between `coords`, with a growing
/// diagonal jitter when the matrix is numerically singular.
fn cholesky(coords: &[(f64, f64, f64)], variogram: &Variogram, sill: f64) -> Result<DMatrix<f64>> {
    let m = coords.len();
    let rows: Vec<Vec<f64>> = (0..m)
        .into_par_iter()
        .map(|i| {
            (0..m)
                .map(|j| variogram.cov_points(&coords[i], &coords[j]) / sill)
                .collect()
        })
        .collect();
    let c = DMatrix::from_fn(m, m, |i, j| rows[i][j]);
    for jitter in [0.0, 1e-10, 1e-8, 1e-6, 1e-4] {
        let mut a = c.clone();
        for i in 0..m {
            a[(i, i)] += jitter;
        }
        if let Some(l) = a.cholesky() {
            return Ok(l.l());
        }
    }
    Err(TransformError::FittingFailed(
        "the data correlation matrix is not positive definite (duplicate locations?)".into(),
    ))
}

/// Linearly interpolated quantile of sorted `x`.
fn quantile(x: &[f64], p: f64) -> f64 {
    let t = p * (x.len() - 1) as f64;
    let (i, f) = (t.floor() as usize, t.fract());
    match x.get(i + 1) {
        Some(next) => x[i] + f * (next - x[i]),
        None => x[i],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use variogram::Model;

    fn data(m: usize) -> (Vec<(f64, f64, f64)>, Vec<f64>) {
        let mut rng = StdRng::seed_from_u64(5);
        let coords = (0..m)
            .map(|_| (rng.gen_range(0.0..100.0), rng.gen_range(0.0..100.0), 0.0))
            .collect();
        let values = (0..m).map(|_| rng.gen_range(0.0..10.0)).collect();
        (coords, values)
    }

    fn spread(x: &[f64]) -> f64 {
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (x.len() - 1) as f64).sqrt()
    }

    fn sigma(values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt()
    }

    fn nugget() -> Variogram {
        Variogram {
            nugget: 1.0,
            structures: vec![],
            anisotropy: None,
        }
    }

    #[test]
    fn pure_nugget_matches_the_classical_bootstrap() {
        let (coords, values) = data(100);
        let b = spatial_bootstrap(&coords, &values, None, &nugget(), 2000, 1, &[], &[]).unwrap();
        let expected = sigma(&values) / 10.0;
        assert!((spread(&b.mean) / expected - 1.0).abs() < 0.08);
    }

    #[test]
    fn long_ranges_spread_the_mean_towards_the_data_spread() {
        let (coords, values) = data(100);
        let run = |range| {
            let v = Variogram::single(Model::Spherical, 1.0, range);
            spread(
                &spatial_bootstrap(&coords, &values, None, &v, 1000, 1, &[], &[])
                    .unwrap()
                    .mean,
            )
        };
        let (short, mid, long) = (run(5.0), run(60.0), run(1e6));
        assert!(short < mid && mid < long);
        assert!((long / sigma(&values) - 1.0).abs() < 0.1);
    }

    #[test]
    fn weights_shift_the_resampled_distribution() {
        let (coords, values) = data(100);
        let w: Vec<f64> = values
            .iter()
            .map(|&v| if v > 5.0 { 3.0 } else { 1.0 })
            .collect();
        let b = spatial_bootstrap(
            &coords,
            &values,
            Some(&w),
            &nugget(),
            500,
            1,
            &[0.5],
            &[5.0],
        )
        .unwrap();
        let mean = b.mean.iter().sum::<f64>() / 500.0;
        let above = b.above.iter().map(|a| a[0]).sum::<f64>() / 500.0;
        let wm = values.iter().zip(&w).map(|(v, w)| v * w).sum::<f64>() / w.iter().sum::<f64>();
        assert!((mean - wm).abs() < 0.1);
        assert!(above > 0.7 && b.quantiles[0][0] > 5.0);
    }

    #[test]
    fn deterministic_for_a_seed_and_any_thread_count() {
        let (coords, values) = data(60);
        let v = Variogram::single(Model::Exponential, 1.0, 30.0);
        let run = |threads, seed| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    spatial_bootstrap(&coords, &values, None, &v, 50, seed, &[0.1], &[3.0]).unwrap()
                })
        };
        assert_eq!(run(1, 3), run(8, 3));
        assert_ne!(run(1, 3), run(1, 4));
    }

    #[test]
    fn rejects_bad_input() {
        let (coords, values) = data(5);
        let v = nugget();
        let f = |c: &[_], x: &[f64], w: Option<&[f64]>, q: &[f64]| {
            spatial_bootstrap(c, x, w, &v, 5, 0, q, &[]).is_err()
        };
        assert!(f(&[], &[], None, &[]));
        assert!(f(&coords[..4], &values, None, &[]));
        assert!(f(&coords, &[f64::NAN; 5], None, &[]));
        assert!(f(&coords, &values, Some(&[0.0; 5]), &[]));
        assert!(f(&coords, &values, None, &[1.5]));
        let power = Variogram::single(Model::Power { exponent: 1.0 }, 1.0, 1.0);
        assert!(spatial_bootstrap(&coords, &values, None, &power, 5, 0, &[], &[]).is_err());
    }
}
