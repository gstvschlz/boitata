//! Turning Bands simulation (continuous Gaussian).
//!
//! Unconditional realizations are built
//! by summing many independent 1-D line processes, then made conditional by
//! kriging.
//!
//! ## Method
//!
//! For an isotropic 3-D correlogram `C₃(r)`, the covariance of the process
//! restricted to a line is `C₁(r) = d/dr[r·C₃(r)]` (Lantuéjoul). We simulate
//! each band as a stationary 1-D Gaussian vector with covariance `C₁` on a fine
//! grid (one dense Cholesky shared by all bands), project every 3-D point onto the band direction, and
//! average `1/√L` over `L` uniformly-oriented bands. Conditioning uses
//! `Zc = Zk(data) + [Zu − Zk(Zu@data)]` (kriging of the residual).
//!
//! Anisotropy is handled by running the bands in the space where the variogram
//! is isotropic.

use crate::Realization;
use crate::error::{Result, SimError};
use estimation::Sample;
use estimation::krige::{Kind, krige};
use estimation::search::{Search, SearchTree};
use nalgebra::{DMatrix, Matrix3, Vector3};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use transforms::normal_score;
use variogram::Variogram;

/// Turning-bands parameters.
#[derive(Debug, Clone)]
pub struct TurningBandsParams {
    /// Number of bands (lines). More bands → less striping; 300–1000 typical.
    pub n_bands: usize,
    /// Band discretization in the variogram's isotropic space (metres along
    /// the major axis); `None` is a fiftieth of the shortest range. The step
    /// widens if a band would need more than 4000 nodes.
    pub step: Option<f64>,
    /// Neighbourhood for conditioning by kriging.
    pub search: Search,
    /// RNG seed.
    pub seed: u64,
}

impl Default for TurningBandsParams {
    fn default() -> Self {
        Self {
            n_bands: 300,
            step: None,
            seed: 1,
            search: Search {
                min_samples: 1,
                max_samples: 32,
                ..Default::default()
            },
        }
    }
}

/// Unit correlogram `C₃(r)` of the (unit-sill) Gaussian variogram.
fn c3(vg: &Variogram, r: f64) -> f64 {
    vg.cov(r) / vg.total_sill()
}

/// 1-D turning-bands covariance `C₁(r) = d/dr[r·C₃(r)] = C₃(r) + r·C₃'(r)`.
fn c1(vg: &Variogram, r: f64) -> f64 {
    if r <= 0.0 {
        return 1.0;
    }
    let dr = (r * 1e-4).max(1e-7);
    let g = |x: f64| x * c3(vg, x);
    (g(r + dr) - g(r - dr)) / (2.0 * dr)
}

/// Lower Cholesky factor of the `C₁` covariance on `n` band nodes spaced
/// `step`; its leading `m × m` block factors the covariance of `m` nodes.
fn band_factor(vg: &Variogram, n: usize, step: f64) -> DMatrix<f64> {
    let mut k = DMatrix::<f64>::from_fn(n, n, |i, j| c1(vg, (i as f64 - j as f64).abs() * step));
    for i in 0..n {
        k[(i, i)] += 1e-9;
    }
    k.cholesky()
        .map(|c| c.l())
        .unwrap_or_else(|| DMatrix::identity(n, n))
}

/// One band process on the first `n` nodes of `factor`.
fn simulate_band(factor: &DMatrix<f64>, n: usize, rng: &mut StdRng) -> Vec<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let w: Vec<f64> = (0..n).map(|_| normal.sample(rng)).collect();
    (0..n)
        .map(|i| (0..=i).map(|j| factor[(i, j)] * w[j]).sum())
        .collect()
}

/// A uniformly-distributed unit direction on the sphere.
fn unit_direction(rng: &mut StdRng, normal: &Normal<f64>) -> (f64, f64, f64) {
    loop {
        let (x, y, z) = (normal.sample(rng), normal.sample(rng), normal.sample(rng));
        let n = (x * x + y * y + z * z).sqrt();
        if n > 1e-9 {
            return (x / n, y / n, z / n);
        }
    }
}

/// Band spacing: the given step, or a fiftieth of the shortest range.
fn band_step(vg: &Variogram, params: &TurningBandsParams) -> f64 {
    params.step.unwrap_or_else(|| {
        vg.structures
            .iter()
            .map(|s| s.range)
            .fold(f64::INFINITY, f64::min)
            .min(1e12)
            / 50.0
    })
}

/// Unconditional standard-Gaussian field at `points` with the correlogram of
/// `vg`, anisotropy included: bands run in the space where it is isotropic.
fn unconditional(
    points: &[(f64, f64, f64)],
    vg: &Variogram,
    params: &TurningBandsParams,
    rng: &mut StdRng,
) -> Vec<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let to_isotropic = vg
        .anisotropy
        .as_ref()
        .map_or_else(Matrix3::identity, |a| a.matrix());
    let isotropic = Variogram {
        anisotropy: None,
        ..vg.clone()
    };
    let points: Vec<Vector3<f64>> = points
        .iter()
        .map(|p| to_isotropic * Vector3::new(p.0, p.1, p.2))
        .collect();
    let (lo, hi) = points.iter().fold(
        (
            Vector3::repeat(f64::INFINITY),
            Vector3::repeat(f64::NEG_INFINITY),
        ),
        |(lo, hi), p| (lo.inf(p), hi.sup(p)),
    );
    const MAX_NODES: usize = 4000;
    let diameter = (hi - lo).norm();
    let step = band_step(&isotropic, params).max(diameter / (MAX_NODES - 2) as f64);
    let longest = ((diameter / step).ceil() as usize + 2).clamp(2, MAX_NODES);
    let factor = band_factor(&isotropic, longest, step);

    let l = params.n_bands.max(1);
    let mut acc = vec![0.0; points.len()];
    for _ in 0..l {
        let dir = unit_direction(rng, &normal);
        let dir = Vector3::new(dir.0, dir.1, dir.2);
        let proj: Vec<f64> = points.iter().map(|p| p.dot(&dir)).collect();
        let tmin = proj.iter().cloned().fold(f64::INFINITY, f64::min);
        let tmax = proj.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let n1 = (((tmax - tmin).max(step) / step).ceil() as usize + 2).clamp(2, longest);
        let band = simulate_band(&factor, n1, rng);
        for (a, &t) in acc.iter_mut().zip(&proj) {
            let x = (t - tmin) / step;
            let i0 = (x.floor() as usize).min(n1 - 2);
            let frac = x - i0 as f64;
            *a += band[i0] * (1.0 - frac) + band[i0 + 1] * frac;
        }
    }
    let inv = 1.0 / (l as f64).sqrt();
    acc.iter().map(|v| v * inv).collect()
}

/// Conditional Gaussian field over `grid` from already-Gaussian data values
/// (no normal-score / back-transform). Returns conditioned Gaussian scores.
///
/// Unconditional turning-bands field plus simple kriging of the data
/// residuals, `Zc = Zu + Zk(z − Zu@data)`, from the neighbours chosen by
/// `params.search`. Takes a caller-owned RNG so several fields can be driven
/// from one seed stream.
pub fn conditional_gaussian_field(
    data_locs: &[(f64, f64, f64)],
    gaussian_data: &[f64],
    grid: &[(f64, f64, f64)],
    vg: &Variogram,
    params: &TurningBandsParams,
    rng: &mut StdRng,
) -> Result<Vec<f64>> {
    if data_locs.len() != gaussian_data.len() {
        return Err(SimError::InvalidParameters("data length mismatch".into()));
    }
    if data_locs.is_empty() {
        return Err(SimError::InsufficientData("no conditioning data".into()));
    }
    if grid.is_empty() {
        return Ok(vec![]);
    }

    let mut all: Vec<(f64, f64, f64)> = data_locs.to_vec();
    all.extend_from_slice(grid);
    let zu = unconditional(&all, vg, params, rng);
    let nd = data_locs.len();
    let (zu_data, zu_grid) = zu.split_at(nd);

    let residuals: Vec<Sample> = data_locs
        .iter()
        .zip(gaussian_data)
        .zip(zu_data)
        .map(|((&loc, &z), &u)| Sample {
            loc,
            value: z - u,
            hole: None,
        })
        .collect();
    let tree = SearchTree::new(&residuals, &params.search, Some(vg));
    grid.par_iter()
        .zip(zu_grid)
        .map(|(g, &u)| {
            let chosen = tree.neighbors(g).unwrap_or_default();
            if chosen.is_empty() {
                return Ok(u);
            }
            let near: Vec<Sample> = chosen.iter().map(|&i| residuals[i].clone()).collect();
            let rk = krige(Kind::Simple { mean: 0.0 }, g, &near, vg)
                .map_err(|e| SimError::Estimation(e.to_string()))?
                .value;
            Ok(u + rk)
        })
        .collect()
}

/// A single conditional turning-bands realization over `grid`.
///
/// `vg_nscore` is the variogram of the normal scores (unit-sill Gaussian).
pub fn turning_bands(
    data_locs: &[(f64, f64, f64)],
    data_vals: &[f64],
    data_weights: Option<&[f64]>,
    grid: &[(f64, f64, f64)],
    vg_nscore: &Variogram,
    params: &TurningBandsParams,
) -> Result<Realization> {
    if data_locs.len() != data_vals.len() {
        return Err(SimError::InvalidParameters("data length mismatch".into()));
    }
    if data_locs.is_empty() {
        return Err(SimError::InsufficientData("no conditioning data".into()));
    }
    if grid.is_empty() {
        return Ok(Realization { values: vec![] });
    }

    // Normal-score transform → condition in Gaussian space → back-transform.
    let ns = normal_score::transform(data_vals, data_weights)
        .map_err(|e| SimError::Transform(e.to_string()))?;
    let mut rng = StdRng::seed_from_u64(params.seed);
    let scores =
        conditional_gaussian_field(data_locs, &ns.scores, grid, vg_nscore, params, &mut rng)?;
    let values = scores.iter().map(|&s| ns.table.back(s)).collect();
    Ok(Realization { values })
}

#[cfg(test)]
mod tests {
    use super::*;
    use variogram::model::Model;

    #[test]
    fn unconditional_field_follows_the_anisotropy() {
        use variogram::{Angles, Anisotropy};
        let aniso = Anisotropy::new(Angles {
            azimuth: 45.0,
            dip: 0.0,
            rake: 0.0,
            major: 1.0,
            semi: 0.25,
            minor: 1.0,
        })
        .unwrap();
        let vg = Variogram::single(Model::Exponential, 1.0, 40.0).with_anisotropy(aniso);
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let (major, minor) = ((s, s), (s, -s));
        let h = 10.0;
        let points = vec![
            (0.0, 0.0, 0.0),
            (h * major.0, h * major.1, 0.0),
            (h * minor.0, h * minor.1, 0.0),
        ];
        let params = TurningBandsParams {
            n_bands: 200,
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(9);
        let (mut along, mut across, runs) = (0.0, 0.0, 2000);
        for _ in 0..runs {
            let z = unconditional(&points, &vg, &params, &mut rng);
            along += (z[1] - z[0]).powi(2) / 2.0 / runs as f64;
            across += (z[2] - z[0]).powi(2) / 2.0 / runs as f64;
        }
        let expect = |d: f64| 1.0 - (-3.0 * d / 40.0).exp();
        assert!(
            (along - expect(h)).abs() < 0.08,
            "along {along} vs {}",
            expect(h)
        );
        assert!(
            (across - expect(h / 0.25)).abs() < 0.12,
            "across {across} vs {}",
            expect(h / 0.25)
        );
    }

    #[test]
    fn conditional_honors_data() {
        let data_locs = vec![
            (0.0, 0.0, 0.0),
            (100.0, 0.0, 0.0),
            (0.0, 100.0, 0.0),
            (100.0, 100.0, 0.0),
        ];
        let data_vals = vec![1.0, 5.0, 3.0, 8.0];
        let grid = vec![(0.0, 0.0, 0.0), (50.0, 50.0, 0.0)];
        let vg = Variogram::single(Model::Spherical, 1.0, 200.0);
        let params = TurningBandsParams {
            n_bands: 200,
            step: Some(5.0),
            seed: 42,
            ..Default::default()
        };
        let r = turning_bands(&data_locs, &data_vals, None, &grid, &vg, &params).unwrap();
        assert_eq!(r.values.len(), 2);
        // Grid node 0 coincides with a datum → simulated value ≈ that datum.
        assert!(
            (r.values[0] - 1.0).abs() < 0.3,
            "conditioned value {}",
            r.values[0]
        );
    }

    #[test]
    fn reproducible_with_seed() {
        let data_locs = vec![(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (0.0, 100.0, 0.0)];
        let data_vals = vec![1.0, 5.0, 3.0];
        let grid: Vec<(f64, f64, f64)> = (0..15).map(|i| (i as f64 * 6.0, 25.0, 0.0)).collect();
        let vg = Variogram::single(Model::Exponential, 1.0, 150.0);
        let params = TurningBandsParams {
            n_bands: 100,
            step: Some(5.0),
            seed: 7,
            ..Default::default()
        };
        let a = turning_bands(&data_locs, &data_vals, None, &grid, &vg, &params).unwrap();
        let b = turning_bands(&data_locs, &data_vals, None, &grid, &vg, &params).unwrap();
        assert_eq!(a.values, b.values);
    }

    #[test]
    fn unconditional_has_unit_variance() {
        // Sanity on the engine: the unconditional field is ~unit variance.
        let vg = Variogram::single(Model::Spherical, 1.0, 50.0);
        let params = TurningBandsParams {
            n_bands: 400,
            step: Some(2.0),
            seed: 3,
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(params.seed);
        let pts: Vec<(f64, f64, f64)> = (0..400)
            .map(|i| ((i % 20) as f64 * 10.0, (i / 20) as f64 * 10.0, 0.0))
            .collect();
        let zu = unconditional(&pts, &vg, &params, &mut rng);
        let m = zu.iter().sum::<f64>() / zu.len() as f64;
        let v = zu.iter().map(|x| (x - m).powi(2)).sum::<f64>() / zu.len() as f64;
        assert!(v > 0.5 && v < 1.6, "unconditional variance {v}");
    }
}
