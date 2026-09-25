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
//! grid (dense Cholesky), project every 3-D point onto the band direction, and
//! average `1/√L` over `L` uniformly-oriented bands. Conditioning uses
//! `Zc = Zk(data) + [Zu − Zk(Zu@data)]` (kriging of the residual).
//!
//! Anisotropy is not applied to the band geometry here — supply an isotropic
//! (unit-sill) Gaussian variogram.

use crate::Realization;
use crate::error::{Result, SimError};
use estimation::Sample;
use estimation::krige::{Kind, krige};
use nalgebra::DMatrix;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use transforms::nscore;
use variogram::Variogram;

/// Turning-bands parameters.
#[derive(Debug, Clone)]
pub struct TurningBandsParams {
    /// Number of bands (lines). More bands → less striping; 300–1000 typical.
    pub n_bands: usize,
    /// 1-D discretization step along each band (world units).
    pub step: f64,
    /// RNG seed.
    pub seed: u64,
}

impl Default for TurningBandsParams {
    fn default() -> Self {
        Self {
            n_bands: 300,
            step: 1.0,
            seed: 1,
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

/// Simulate one 1-D band process on `[0, n)` grid points with spacing `step`,
/// via dense Cholesky of the `C₁` Toeplitz covariance.
fn simulate_band(vg: &Variogram, n: usize, step: f64, rng: &mut StdRng) -> Vec<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut k = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            k[(i, j)] = c1(vg, ((i as f64 - j as f64) * step).abs());
        }
    }
    // Jitter for numerical PD safety.
    for i in 0..n {
        k[(i, i)] += 1e-9;
    }
    let chol = match k.cholesky() {
        Some(c) => c.l(),
        None => DMatrix::<f64>::identity(n, n),
    };
    let w = nalgebra::DVector::from_fn(n, |_, _| normal.sample(rng));
    (chol * w).iter().copied().collect()
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

/// Unconditional standard-Gaussian field at `points` (unit variance, covariance
/// ≈ `C₃`).
fn unconditional(
    points: &[(f64, f64, f64)],
    vg: &Variogram,
    params: &TurningBandsParams,
    rng: &mut StdRng,
) -> Vec<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let l = params.n_bands.max(1);
    let mut acc = vec![0.0; points.len()];
    for _ in 0..l {
        let dir = unit_direction(rng, &normal);
        let proj: Vec<f64> = points
            .iter()
            .map(|p| p.0 * dir.0 + p.1 * dir.1 + p.2 * dir.2)
            .collect();
        let tmin = proj.iter().cloned().fold(f64::INFINITY, f64::min);
        let tmax = proj.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let span = (tmax - tmin).max(params.step);
        let n1 = ((span / params.step).ceil() as usize + 2).clamp(2, 4000);
        let band = simulate_band(vg, n1, params.step, rng);
        for (a, &t) in acc.iter_mut().zip(&proj) {
            // Linear interpolation on the band grid.
            let x = (t - tmin) / params.step;
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
/// This is the reusable engine behind [`turning_bands`] and
/// [`crate::pgs`]: unconditional turning-bands field + kriging of the residual
/// (`Zc = Zk(data) + [Zu − Zk(Zu@data)]`). Takes a caller-owned RNG so multiple
/// fields can be driven from one seed stream.
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

    let cond_real: Vec<Sample> = data_locs
        .iter()
        .zip(gaussian_data)
        .map(|(&loc, &v)| Sample {
            loc,
            value: v,
            hole: None,
        })
        .collect();
    let cond_unc: Vec<Sample> = data_locs
        .iter()
        .zip(zu_data)
        .map(|(&loc, &v)| Sample {
            loc,
            value: v,
            hole: None,
        })
        .collect();

    let mut scores = Vec::with_capacity(grid.len());
    for (g, &zug) in grid.iter().zip(zu_grid) {
        let zk = krige(Kind::Simple { mean: 0.0 }, g, &cond_real, vg)
            .map_err(|e| SimError::Estimation(e.to_string()))?
            .value;
        let zuk = krige(Kind::Simple { mean: 0.0 }, g, &cond_unc, vg)
            .map_err(|e| SimError::Estimation(e.to_string()))?
            .value;
        scores.push(zk + (zug - zuk));
    }
    Ok(scores)
}

/// A single conditional turning-bands realization over `grid`.
///
/// `vg_nscore` is the variogram of the normal scores (unit-sill Gaussian).
pub fn turning_bands(
    data_locs: &[(f64, f64, f64)],
    data_vals: &[f64],
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
    let ns = nscore::transform(data_vals, None).map_err(|e| SimError::Transform(e.to_string()))?;
    let mut rng = StdRng::seed_from_u64(params.seed);
    let scores =
        conditional_gaussian_field(data_locs, &ns.scores, grid, vg_nscore, params, &mut rng)?;
    let values = scores.iter().map(|&s| ns.table.back(s)).collect();
    Ok(Realization { values })
}

/// `n` independent realizations (seeds `seed, seed+1, …`).
pub fn turning_bands_ensemble(
    data_locs: &[(f64, f64, f64)],
    data_vals: &[f64],
    grid: &[(f64, f64, f64)],
    vg_nscore: &Variogram,
    params: &TurningBandsParams,
    n: usize,
) -> Result<Vec<Realization>> {
    (0..n)
        .map(|k| {
            let p = TurningBandsParams {
                seed: params.seed.wrapping_add(k as u64),
                ..params.clone()
            };
            turning_bands(data_locs, data_vals, grid, vg_nscore, &p)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use variogram::model::Model;

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
            step: 5.0,
            seed: 42,
        };
        let r = turning_bands(&data_locs, &data_vals, &grid, &vg, &params).unwrap();
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
            step: 5.0,
            seed: 7,
        };
        let a = turning_bands(&data_locs, &data_vals, &grid, &vg, &params).unwrap();
        let b = turning_bands(&data_locs, &data_vals, &grid, &vg, &params).unwrap();
        assert_eq!(a.values, b.values);
    }

    #[test]
    fn unconditional_has_unit_variance() {
        // Sanity on the engine: the unconditional field is ~unit variance.
        let vg = Variogram::single(Model::Spherical, 1.0, 50.0);
        let params = TurningBandsParams {
            n_bands: 400,
            step: 2.0,
            seed: 3,
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
