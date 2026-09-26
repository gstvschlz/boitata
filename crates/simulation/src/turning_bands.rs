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
use crate::post::{ContinuousOptions, ContinuousSummary, continuous};
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

/// Bounding box of `points`, `(lo, hi)`.
pub fn bounds(points: &[(f64, f64, f64)]) -> ([f64; 3], [f64; 3]) {
    points.iter().fold(
        ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]),
        |(lo, hi), p| {
            let p = [p.0, p.1, p.2];
            (
                std::array::from_fn(|i| lo[i].min(p[i])),
                std::array::from_fn(|i| hi[i].max(p[i])),
            )
        },
    )
}

/// The simulated bands of one unconditional realization, laid out over a box
/// so the field can be evaluated at any point inside it, in any number of
/// pieces, with the same values.
#[derive(Debug, Clone)]
pub struct Bands {
    to_isotropic: Matrix3<f64>,
    step: f64,
    directions: Vec<Vector3<f64>>,
    origins: Vec<f64>,
    values: Vec<Vec<f64>>,
}

impl Bands {
    /// Bands covering the box `lo..hi` with the correlogram of `vg`, anisotropy
    /// included: bands run in the space where it is isotropic.
    pub fn new(
        lo: [f64; 3],
        hi: [f64; 3],
        vg: &Variogram,
        params: &TurningBandsParams,
        rng: &mut StdRng,
    ) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let to_isotropic = vg
            .anisotropy
            .as_ref()
            .map_or_else(Matrix3::identity, |a| a.matrix());
        let isotropic = Variogram {
            anisotropy: None,
            ..vg.clone()
        };
        let corners: Vec<Vector3<f64>> = (0..8)
            .map(|c| {
                let pick = |axis: usize| {
                    if c >> axis & 1 == 0 {
                        lo[axis]
                    } else {
                        hi[axis]
                    }
                };
                to_isotropic * Vector3::new(pick(0), pick(1), pick(2))
            })
            .collect();
        let (clo, chi) = corners.iter().fold(
            (
                Vector3::repeat(f64::INFINITY),
                Vector3::repeat(f64::NEG_INFINITY),
            ),
            |(lo, hi), p| (lo.inf(p), hi.sup(p)),
        );
        const MAX_NODES: usize = 4000;
        let diameter = (chi - clo).norm();
        let step = band_step(&isotropic, params).max(diameter / (MAX_NODES - 2) as f64);
        let longest = ((diameter / step).ceil() as usize + 2).clamp(2, MAX_NODES);
        let factor = band_factor(&isotropic, longest, step);
        let l = params.n_bands.max(1);
        let mut bands = Self {
            to_isotropic,
            step,
            directions: Vec::with_capacity(l),
            origins: Vec::with_capacity(l),
            values: Vec::with_capacity(l),
        };
        for _ in 0..l {
            let dir = unit_direction(rng, &normal);
            let dir = Vector3::new(dir.0, dir.1, dir.2);
            let (tmin, tmax) = corners
                .iter()
                .map(|p| p.dot(&dir))
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), t| {
                    (a.min(t), b.max(t))
                });
            let n1 = (((tmax - tmin).max(step) / step).ceil() as usize + 2).clamp(2, longest);
            bands.values.push(simulate_band(&factor, n1, rng));
            bands.directions.push(dir);
            bands.origins.push(tmin);
        }
        bands
    }

    /// Standard-Gaussian unconditional field at `points` (inside the box).
    pub fn field(&self, points: &[(f64, f64, f64)]) -> Vec<f64> {
        let inv = 1.0 / (self.directions.len() as f64).sqrt();
        points
            .par_iter()
            .map(|p| {
                let p = self.to_isotropic * Vector3::new(p.0, p.1, p.2);
                let mut sum = 0.0;
                for ((dir, &tmin), band) in
                    self.directions.iter().zip(&self.origins).zip(&self.values)
                {
                    let x = ((p.dot(dir) - tmin) / self.step).max(0.0);
                    let i0 = (x.floor() as usize).min(band.len() - 2);
                    let frac = x - i0 as f64;
                    sum += band[i0] * (1.0 - frac) + band[i0 + 1] * frac;
                }
                sum * inv
            })
            .collect()
    }
}

/// Unconditional standard-Gaussian field at `points`.
#[cfg(test)]
fn unconditional(
    points: &[(f64, f64, f64)],
    vg: &Variogram,
    params: &TurningBandsParams,
    rng: &mut StdRng,
) -> Vec<f64> {
    let (lo, hi) = bounds(points);
    Bands::new(lo, hi, vg, params, rng).field(points)
}

/// Simple kriging of `residuals` at every target added to `field`.
fn condition(
    targets: &[(f64, f64, f64)],
    field: Vec<f64>,
    residuals: &[Sample],
    tree: &SearchTree,
    vg: &Variogram,
) -> Result<Vec<f64>> {
    targets
        .par_iter()
        .zip(field)
        .map(|(g, u)| {
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

/// The data as search samples: neighbours are chosen, and high-grade
/// thresholds compared, on the data values rather than the residuals.
fn data(locs: &[(f64, f64, f64)], values: &[f64]) -> Vec<Sample> {
    locs.iter()
        .zip(values)
        .map(|(&loc, &v)| Sample::new(loc, v))
        .collect()
}

fn residuals(data_locs: &[(f64, f64, f64)], gaussian_data: &[f64], at_data: &[f64]) -> Vec<Sample> {
    data_locs
        .iter()
        .zip(gaussian_data)
        .zip(at_data)
        .map(|((&loc, &z), &u)| Sample {
            loc,
            value: z - u,
            hole: None,
            error_variance: 0.0,
            domain: None,
        })
        .collect()
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
    let (lo, hi) = bounds(&[data_locs, grid].concat());
    let bands = Bands::new(lo, hi, vg, params, rng);
    let residuals = residuals(data_locs, gaussian_data, &bands.field(data_locs));
    let tree = SearchTree::new(&data(data_locs, gaussian_data), &params.search, Some(vg));
    condition(grid, bands.field(grid), &residuals, &tree, vg)
}

/// `n` conditional turning-bands realizations (seeds `seed, seed + 1, …`)
/// prepared over a box, so they can be evaluated at any targets inside it —
/// all at once or chunk by chunk, with the same values.
pub struct TurningBandsEnsemble {
    table: transforms::NormalScoreTable,
    bands: Vec<Bands>,
    residuals: Vec<Vec<Sample>>,
    tree: SearchTree,
    vg: Variogram,
}

impl TurningBandsEnsemble {
    /// Normal-scores the data and simulates the bands of every realization
    /// over the box `lo..hi`, which must hold every target; the data are
    /// added to it. `vg_nscore` is the variogram of the normal scores.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        data_locs: &[(f64, f64, f64)],
        data_vals: &[f64],
        data_weights: Option<&[f64]>,
        lo: [f64; 3],
        hi: [f64; 3],
        vg_nscore: &Variogram,
        params: &TurningBandsParams,
        n: usize,
    ) -> Result<Self> {
        if data_locs.len() != data_vals.len() {
            return Err(SimError::InvalidParameters("data length mismatch".into()));
        }
        if data_locs.is_empty() {
            return Err(SimError::InsufficientData("no conditioning data".into()));
        }
        let ns = normal_score::transform(data_vals, data_weights)
            .map_err(|e| SimError::Transform(e.to_string()))?;
        let (dlo, dhi) = bounds(data_locs);
        let lo = std::array::from_fn(|i| lo[i].min(dlo[i]));
        let hi = std::array::from_fn(|i| hi[i].max(dhi[i]));
        let (bands, residuals): (Vec<Bands>, Vec<Vec<Sample>>) = (0..n)
            .into_par_iter()
            .map(|k| {
                let mut rng = StdRng::seed_from_u64(params.seed.wrapping_add(k as u64));
                let bands = Bands::new(lo, hi, vg_nscore, params, &mut rng);
                let residuals = residuals(data_locs, &ns.scores, &bands.field(data_locs));
                (bands, residuals)
            })
            .unzip();
        let tree = SearchTree::new(&data(data_locs, data_vals), &params.search, Some(vg_nscore));
        Ok(Self {
            table: ns.table,
            bands,
            residuals,
            tree,
            vg: vg_nscore.clone(),
        })
    }

    pub fn len(&self) -> usize {
        self.bands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bands.is_empty()
    }

    /// Realization `k` at `targets`, back-transformed to data values.
    pub fn realization(&self, k: usize, targets: &[(f64, f64, f64)]) -> Result<Vec<f64>> {
        let bands = self
            .bands
            .get(k)
            .ok_or_else(|| SimError::InvalidParameters(format!("no realization {k}")))?;
        let scores = condition(
            targets,
            bands.field(targets),
            &self.residuals[k],
            &self.tree,
            &self.vg,
        )?;
        Ok(scores.iter().map(|&s| self.table.back(s)).collect())
    }

    /// Summary of every realization at `targets`.
    pub fn summary(
        &self,
        targets: &[(f64, f64, f64)],
        options: &ContinuousOptions,
    ) -> Result<ContinuousSummary> {
        continuous(self.len(), options, |k| self.realization(k, targets))
    }
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
    if grid.is_empty() {
        return Ok(Realization { values: vec![] });
    }
    let (lo, hi) = bounds(grid);
    let ensemble = TurningBandsEnsemble::new(
        data_locs,
        data_vals,
        data_weights,
        lo,
        hi,
        vg_nscore,
        params,
        1,
    )?;
    Ok(Realization {
        values: ensemble.realization(0, grid)?,
    })
}

/// Summary columns of turning-bands realizations, streamed from the block
/// model file `input` to `output` chunk by chunk (the input columns are
/// kept): `mean`, `variance`, `p_above_<c>` and `mean_above_<c>` per cutoff,
/// `q<p>` per quantile. Memory is bounded by `rows` blocks plus the bands.
/// Returns the global mean and share above each cutoff of every realization.
#[allow(clippy::too_many_arguments)]
pub fn turning_bands_to_parquet(
    input: impl AsRef<std::path::Path>,
    output: impl AsRef<std::path::Path>,
    data_locs: &[(f64, f64, f64)],
    data_vals: &[f64],
    data_weights: Option<&[f64]>,
    vg_nscore: &Variogram,
    params: &TurningBandsParams,
    n: usize,
    options: &ContinuousOptions,
    rows: usize,
) -> Result<GlobalSummary> {
    let reader = ceres_io::BlockModelReader::open(&input)?;
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for chunk in reader.chunks(rows, Some(&[]))? {
        let (clo, chi) = bounds(&points(&chunk?));
        lo = std::array::from_fn(|i| lo[i].min(clo[i]));
        hi = std::array::from_fn(|i| hi[i].max(chi[i]));
    }
    let ensemble = TurningBandsEnsemble::new(
        data_locs,
        data_vals,
        data_weights,
        lo,
        hi,
        vg_nscore,
        params,
        n,
    )?;
    let options = ContinuousOptions {
        keep: false,
        ..options.clone()
    };
    let mut writer = ceres_io::BlockModelWriter::create(
        output,
        *reader.geometry(),
        reader.layout(),
        reader.crs(),
    )?;
    let mut global = GlobalSummary {
        realization_mean: vec![0.0; n],
        realization_above: vec![vec![0.0; n]; options.cutoffs.len()],
    };
    let mut total = 0.0;
    for chunk in reader.chunks(rows, None)? {
        let chunk = chunk?;
        let s = ensemble.summary(&points(&chunk), &options)?;
        let m = chunk.len() as f64;
        total += m;
        for k in 0..n {
            global.realization_mean[k] += s.realization_mean[k] * m;
            for (c, above) in global.realization_above.iter_mut().enumerate() {
                above[k] += s.realization_above[c][k] * m;
            }
        }
        let mut columns: Vec<(String, Vec<f64>)> =
            vec![("mean".into(), s.mean), ("variance".into(), s.variance)];
        for (c, cut) in options.cutoffs.iter().enumerate() {
            columns.push((format!("p_above_{cut}"), s.probability_above[c].clone()));
            columns.push((format!("mean_above_{cut}"), s.mean_above[c].clone()));
        }
        for (q, p) in options.quantiles.iter().zip(s.quantile_values) {
            columns.push((format!("q{q}"), p));
        }
        let mut out = chunk;
        for (name, values) in columns {
            let column = arrow_array::Float64Array::from_iter(
                values.into_iter().map(|v| (!v.is_nan()).then_some(v)),
            );
            out = out
                .with_column(&name, std::sync::Arc::new(column))
                .map_err(ceres_io::Error::from)?;
        }
        writer.write(&out)?;
    }
    writer.finish()?;
    global.realization_mean.iter_mut().for_each(|v| *v /= total);
    for above in &mut global.realization_above {
        above.iter_mut().for_each(|v| *v /= total);
    }
    Ok(global)
}

/// Per-realization results over a whole streamed model.
#[derive(Debug, Clone)]
pub struct GlobalSummary {
    /// Mean of each realization over all blocks.
    pub realization_mean: Vec<f64>,
    /// Share of blocks above each cutoff, `[cutoff][realization]`.
    pub realization_above: Vec<Vec<f64>>,
}

fn points(model: &ceres_core::BlockModel) -> Vec<(f64, f64, f64)> {
    model
        .centroids()
        .into_iter()
        .map(|[x, y, z]| (x, y, z))
        .collect()
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
        let (mut along, mut across, runs) = (0.0, 0.0, 1000);
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

    #[test]
    fn streamed_summaries_equal_the_in_memory_ones() {
        use ceres_core::{BlockModel, Geometry};
        let data_locs: Vec<_> = (0..40)
            .map(|i| {
                (
                    (i * 7 % 50) as f64 + 0.3,
                    (i * 11 % 40) as f64 + 0.6,
                    (i % 3) as f64,
                )
            })
            .collect();
        let data_vals: Vec<f64> = (0..40)
            .map(|i| 1.0 + (i as f64 * 0.37).sin().abs() * 5.0)
            .collect();
        let geometry = Geometry {
            origin: [0.0; 3],
            size: [5.0, 5.0, 2.0],
            count: [10, 8, 2],
            rotation: [0.0; 3],
        };
        let empty = arrow_array::RecordBatch::try_new_with_options(
            std::sync::Arc::new(arrow_schema::Schema::empty()),
            vec![],
            &arrow_array::RecordBatchOptions::new().with_row_count(Some(160)),
        )
        .unwrap();
        let model = BlockModel::regular(geometry, empty).unwrap();
        let input = std::env::temp_dir().join(format!("ceres-tb-{}.parquet", std::process::id()));
        ceres_io::write_block_model(&input, &model).unwrap();
        let vg = Variogram::single(Model::Spherical, 1.0, 20.0);
        let params = TurningBandsParams {
            n_bands: 60,
            ..Default::default()
        };
        let options = ContinuousOptions {
            cutoffs: vec![3.0],
            quantiles: vec![0.5],
            keep: false,
        };
        let grid = points(&model);
        let (lo, hi) = bounds(&grid);
        let whole =
            TurningBandsEnsemble::new(&data_locs, &data_vals, None, lo, hi, &vg, &params, 6)
                .unwrap()
                .summary(&grid, &options)
                .unwrap();
        for rows in [7, 160] {
            let output = input.with_extension(format!("{rows}.parquet"));
            let global = turning_bands_to_parquet(
                &input, &output, &data_locs, &data_vals, None, &vg, &params, 6, &options, rows,
            )
            .unwrap();
            let ceres_io::Stored::Blocks(back) = ceres_io::read_parquet(&output).unwrap() else {
                panic!("expected a block model")
            };
            let column = |name: &str| {
                use arrow_array::cast::AsArray;
                back.attributes()
                    .column_by_name(name)
                    .unwrap()
                    .as_primitive::<arrow_array::types::Float64Type>()
                    .values()
                    .to_vec()
            };
            assert_eq!(column("mean"), whole.mean);
            assert_eq!(column("p_above_3"), whole.probability_above[0]);
            assert_eq!(column("q0.5"), whole.quantile_values[0]);
            for (a, b) in global.realization_mean.iter().zip(&whole.realization_mean) {
                assert!((a - b).abs() < 1e-12);
            }
        }
    }
}
