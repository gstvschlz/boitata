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
//! grid (one dense Cholesky shared by all bands and realizations), project
//! every 3-D point onto the band direction, and average `1/√L` over `L`
//! uniformly-oriented bands. Conditioning uses
//! `Zc = Zk(data) + [Zu − Zk(Zu@data)]` (kriging of the residual).
//!
//! Anisotropy is handled by running the bands in the space where the variogram
//! is isotropic.
//!
//! The data are transformed as in SGS, within each domain and trend class;
//! every domain shares the bands, and a node's residuals are kriged, and its
//! score back-transformed, within its own domain.

use crate::error::{Result, SimError};
use crate::post::{BlockSupport, ContinuousOptions, ContinuousSummary, continuous};
use crate::sgs::{Domains, Realization, Transform, Transforms, Trend, data};
use estimation::Sample;
use estimation::krige::{Kind, krige};
use estimation::search::{Search, SearchTree};
use nalgebra::{Matrix3, Vector3};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use variogram::Variogram;

/// Turning-bands parameters.
#[derive(Debug, Clone)]
pub struct TurningBandsParams {
    /// Number of bands (lines). More bands → less striping; 300–1000 typical.
    pub n_bands: usize,
    /// Band discretization in the variogram's isotropic space (meters along
    /// the major axis); `None` is a fiftieth of the shortest range. The step
    /// widens if a band would need more than 4000 nodes.
    pub step: Option<f64>,
    /// Neighborhood for conditioning by kriging.
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

/// Rows of the lower Cholesky factor of the `C₁` covariance on `n` band
/// nodes spaced `step`; its leading `m` rows factor the covariance of `m`
/// nodes. The identity if the covariance is not positive definite.
///
/// Every entry subtracts its products in column order, so the factor does
/// not depend on the tiling or the thread count.
fn band_factor(vg: &Variogram, n: usize, step: f64) -> Vec<Vec<f64>> {
    const BLOCK: usize = 64;
    const TILE: usize = 256;
    let c: Vec<f64> = (0..n).map(|d| c1(vg, d as f64 * step)).collect();
    let mut rows: Vec<Vec<f64>> = (0..n)
        .map(|i| c[..=i].iter().rev().copied().collect())
        .collect();
    for j0 in (0..n).step_by(BLOCK) {
        let j1 = (j0 + BLOCK).min(n);
        let (head, tail) = rows.split_at_mut(j1);
        for j in j0..j1 {
            let (pivot, below) = head[j..].split_first_mut().expect("j < j1");
            let mut d = pivot[j] + 1e-9;
            for l in &pivot[..j] {
                d -= l * l;
            }
            if d.is_nan() || d <= 0.0 {
                return (0..n)
                    .map(|i| (0..=i).map(|k| f64::from(u8::from(i == k))).collect())
                    .collect();
            }
            pivot[j] = d.sqrt();
            for row in below {
                let row = std::slice::from_mut(row);
                subtract(&pivot[..j], row, 0, j);
                row[0][j] /= pivot[j];
            }
        }
        let pivots = &head[j0..j1];
        tail.par_chunks_mut(32).for_each(|chunk| {
            for k0 in (0..j0).step_by(TILE) {
                let k1 = (k0 + TILE).min(j0);
                for group in chunk.chunks_mut(8) {
                    for (j, p) in (j0..).zip(pivots) {
                        subtract(&p[k0..k1], group, k0, j);
                    }
                }
            }
            for group in chunk.chunks_mut(8) {
                for (j, p) in (j0..).zip(pivots) {
                    subtract(&p[j0..j], group, j0, j);
                    for row in group.iter_mut() {
                        row[j] /= p[j];
                    }
                }
            }
        });
    }
    rows
}

/// Subtracts from entry `j` of each of `rows` its products with `p`, the
/// entries `k0..` of a finished row, in order.
fn subtract(p: &[f64], rows: &mut [Vec<f64>], k0: usize, j: usize) {
    const R: usize = 8;
    let k1 = k0 + p.len();
    for group in rows.chunks_mut(R) {
        if group.len() == R {
            let r: [&[f64]; R] = std::array::from_fn(|g| &group[g][k0..k1]);
            let mut y: [f64; R] = std::array::from_fn(|g| group[g][j]);
            for (i, pk) in p.iter().enumerate() {
                for g in 0..R {
                    y[g] -= pk * r[g][i];
                }
            }
            for (row, y) in group.iter_mut().zip(y) {
                row[j] = y;
            }
        } else {
            for row in group {
                let mut y = row[j];
                for (pk, a) in p.iter().zip(&row[k0..k1]) {
                    y -= pk * a;
                }
                row[j] = y;
            }
        }
    }
}

/// The band processes of the white noises `noise`, each on its first
/// `w.len()` nodes: `L·w`, every row summed in order. Eight bands at a time
/// share each row of `factor`.
fn simulate_bands(factor: &[Vec<f64>], noise: &[Vec<f64>]) -> Vec<Vec<f64>> {
    const G: usize = 8;
    noise
        .par_chunks(G)
        .flat_map_iter(|group| {
            let mut out: Vec<Vec<f64>> =
                group.iter().map(|w| Vec::with_capacity(w.len())).collect();
            let n = group.iter().map(Vec::len).max().unwrap_or(0);
            let shared = match group.len() {
                G => group.iter().map(Vec::len).min().unwrap_or(0),
                _ => 0,
            };
            let interleaved: Vec<[f64; G]> = (0..shared)
                .map(|k| std::array::from_fn(|g| group[g][k]))
                .collect();
            for (i, row) in factor[..n].iter().enumerate() {
                if i < shared {
                    let mut y = [-0.0; G];
                    for (l, w) in row.iter().zip(&interleaved) {
                        for g in 0..G {
                            y[g] += l * w[g];
                        }
                    }
                    for (out, y) in out.iter_mut().zip(y) {
                        out.push(y);
                    }
                } else {
                    for (out, w) in out.iter_mut().zip(group).filter(|(_, w)| w.len() > i) {
                        out.push(row.iter().zip(w).map(|(l, w)| l * w).sum());
                    }
                }
            }
            out
        })
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

/// What every realization over one box shares: the band spacing, the
/// corners in isotropic space and the band factorization.
struct Layout {
    to_isotropic: Matrix3<f64>,
    corners: Vec<Vector3<f64>>,
    step: f64,
    factor: Vec<Vec<f64>>,
}

impl Layout {
    fn new(lo: [f64; 3], hi: [f64; 3], vg: &Variogram, params: &TurningBandsParams) -> Self {
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
        Self {
            to_isotropic,
            corners,
            step,
            factor: band_factor(&isotropic, longest, step),
        }
    }

    /// The bands of one realization, drawn from `rng`.
    fn bands(&self, n_bands: usize, rng: &mut StdRng) -> Bands {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let step = self.step;
        let (mut directions, mut origins, mut noise) = (vec![], vec![], vec![]);
        for _ in 0..n_bands.max(1) {
            let dir = unit_direction(rng, &normal);
            let dir = Vector3::new(dir.0, dir.1, dir.2);
            let (tmin, tmax) = self
                .corners
                .iter()
                .map(|p| p.dot(&dir))
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), t| {
                    (a.min(t), b.max(t))
                });
            let n1 =
                (((tmax - tmin).max(step) / step).ceil() as usize + 2).clamp(2, self.factor.len());
            noise.push((0..n1).map(|_| normal.sample(rng)).collect());
            directions.push(dir);
            origins.push(tmin);
        }
        Bands {
            to_isotropic: self.to_isotropic,
            step,
            directions,
            origins,
            values: simulate_bands(&self.factor, &noise),
        }
    }
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
        Layout::new(lo, hi, vg, params).bands(params.n_bands, rng)
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

/// Simple kriging of residuals at every target, of domain `domains[i]`,
/// added to `field`: `residual(k, domain)` is the residual of neighbor `k`
/// for a target of `domain`.
fn condition(
    targets: &[(f64, f64, f64)],
    domains: Option<&[u32]>,
    field: Vec<f64>,
    tree: &SearchTree,
    vg: &Variogram,
    residual: impl Fn(usize, Option<u32>) -> Sample + Sync,
) -> Result<Vec<f64>> {
    targets
        .par_iter()
        .zip(field)
        .enumerate()
        .map(|(i, (g, u))| {
            let domain = domains.map(|d| d[i]);
            let chosen = tree.neighbors_in(g, domain).unwrap_or_default();
            if chosen.is_empty() {
                return Ok(u);
            }
            let near: Vec<Sample> = chosen.iter().map(|&k| residual(k, domain)).collect();
            let rk = krige(Kind::Simple { mean: 0.0 }, g, &near, vg)
                .map_err(|e| SimError::Estimation(e.to_string()))?
                .value;
            Ok(u + rk)
        })
        .collect()
}

/// Conditional Gaussian field over `grid` from already-Gaussian data values
/// (no normal-score / back-transform). Returns conditioned Gaussian scores.
///
/// Unconditional turning-bands field plus simple kriging of the data
/// residuals, `Zc = Zu + Zk(z − Zu@data)`, from the neighbors chosen by
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
    let at_data = bands.field(data_locs);
    let data = data(data_locs, gaussian_data, vec![None; data_locs.len()], None);
    let tree = SearchTree::new(&data, &params.search, Some(vg));
    condition(grid, None, bands.field(grid), &tree, vg, |k, _| {
        Sample::new(data[k].loc, data[k].value - at_data[k])
    })
}

/// `n` conditional turning-bands realizations (seeds `seed, seed + 1, …`)
/// prepared over a box, so they can be evaluated at any targets inside it —
/// all at once or chunk by chunk, with the same values.
///
/// With domains, each has its own transform (see [`Transforms`]), a target
/// is conditioned by the data of its domain, and of others within the
/// search's soft boundaries, and back-transformed through its domain's; the
/// bands are shared. A datum of another domain enters the kriging as its
/// grade (and trend) transformed through the target's domain.
pub struct TurningBandsEnsemble {
    transforms: Transforms,
    data: Vec<Sample>,
    trend: Option<Vec<f64>>,
    bands: Vec<Bands>,
    at_data: Vec<Vec<f64>>,
    tree: SearchTree,
    vg: Variogram,
}

impl TurningBandsEnsemble {
    /// Transforms the data and simulates the bands of every realization
    /// over the box `lo..hi`, which must hold every target; the data are
    /// added to it. `vg_nscore` is the variogram of the normal scores;
    /// `data_holes` tag the data by drill hole for `max_per_hole`;
    /// `data_domains` are the domain codes of the data and `data_trend` the
    /// trend at the data with its number of classes.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        data_locs: &[(f64, f64, f64)],
        data_vals: &[f64],
        data_weights: Option<&[f64]>,
        data_holes: Option<&[u32]>,
        data_domains: Option<&[u32]>,
        data_trend: Option<(&[f64], usize)>,
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
        let holes = crate::holes(data_holes, data_locs.len())?;
        let transforms = Transforms::fit(data_vals, data_weights, data_domains, data_trend)?;
        let (dlo, dhi) = bounds(data_locs);
        let lo = std::array::from_fn(|i| lo[i].min(dlo[i]));
        let hi = std::array::from_fn(|i| hi[i].max(dhi[i]));
        let layout = Layout::new(lo, hi, vg_nscore, params);
        let (bands, at_data): (Vec<Bands>, Vec<Vec<f64>>) = (0..n)
            .into_par_iter()
            .map(|k| {
                let mut rng = StdRng::seed_from_u64(params.seed.wrapping_add(k as u64));
                let bands = layout.bands(params.n_bands, &mut rng);
                let at_data = bands.field(data_locs);
                (bands, at_data)
            })
            .unzip();
        let data = data(data_locs, data_vals, holes, data_domains);
        let tree = SearchTree::new(&data, &params.search, Some(vg_nscore));
        Ok(Self {
            transforms,
            data,
            trend: data_trend.map(|t| t.0.to_vec()),
            bands,
            at_data,
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

    /// The transform of each target, by its code in `domains`, checked
    /// against the data's domains and trend.
    fn transforms(
        &self,
        targets: usize,
        domains: Option<&[u32]>,
        trend: Option<&[f64]>,
    ) -> Result<Vec<&Transform>> {
        let invalid = |m: &str| Err(SimError::InvalidParameters(m.into()));
        let domained = self.data.first().is_some_and(|s| s.domain.is_some());
        match (domained, domains) {
            (true, None) => return invalid("the data have domains; give the targets' domains"),
            (false, Some(_)) => return invalid("target domains need data domains"),
            (_, Some(d)) if d.len() != targets => return invalid("one domain per node"),
            _ => {}
        }
        match (&self.trend, trend) {
            (Some(_), None) => return invalid("the data have a trend; give it at the targets"),
            (None, Some(_)) => return invalid("a trend at the targets needs one at the data"),
            (_, Some(t)) if t.len() != targets => return invalid("one trend value per node"),
            _ => {}
        }
        (0..targets)
            .map(|i| {
                let code = domains.map_or(0, |d| d[i]);
                self.transforms
                    .domains
                    .get(code as usize)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| {
                        SimError::InvalidParameters(format!("domain {code} has no samples"))
                    })
            })
            .collect()
    }

    /// Score of datum `j` for a target of `domain`: its own in its domain,
    /// its grade (and trend) transformed through `domain` in another.
    fn score(&self, j: usize, domain: Option<u32>) -> f64 {
        let s = &self.data[j];
        if s.domain == domain {
            return self.transforms.scores[j];
        }
        let code = domain.expect("another domain") as usize;
        let t = self.trend.as_ref().map_or(0.0, |t| t[j]);
        self.transforms.domains[code]
            .as_ref()
            .expect("checked")
            .forward(s.value, t)
    }

    /// Realization `k` at `targets`, of codes `domains` and trend `trend`
    /// when the data have them, back-transformed to data values. Each
    /// realization may take its own `domains`, as simulated domains do.
    pub fn realization(
        &self,
        k: usize,
        targets: &[(f64, f64, f64)],
        domains: Option<&[u32]>,
        trend: Option<&[f64]>,
    ) -> Result<Vec<f64>> {
        let bands = self
            .bands
            .get(k)
            .ok_or_else(|| SimError::InvalidParameters(format!("no realization {k}")))?;
        let transforms = self.transforms(targets.len(), domains, trend)?;
        let at_data = &self.at_data[k];
        let scores = condition(
            targets,
            domains,
            bands.field(targets),
            &self.tree,
            &self.vg,
            |j, domain| Sample::new(self.data[j].loc, self.score(j, domain) - at_data[j]),
        )?;
        Ok(scores
            .iter()
            .zip(transforms)
            .enumerate()
            .map(|(i, (&s, t))| t.back(s, trend.map_or(0.0, |t| t[i])))
            .collect())
    }

    /// Summary of every realization at `targets`, of codes `domains` and
    /// trend `trend` when the data have them.
    pub fn summary(
        &self,
        targets: &[(f64, f64, f64)],
        domains: Option<&[u32]>,
        trend: Option<&[f64]>,
        options: &ContinuousOptions,
    ) -> Result<ContinuousSummary> {
        continuous(self.len(), options, |k| {
            self.realization(k, targets, domains, trend)
        })
    }
}

/// A single conditional turning-bands realization over `grid`.
///
/// `vg_nscore` is the variogram of the normal scores (unit-sill Gaussian);
/// `data_holes` tag the data by drill hole for `max_per_hole`.
pub fn turning_bands(
    data_locs: &[(f64, f64, f64)],
    data_vals: &[f64],
    data_weights: Option<&[f64]>,
    data_holes: Option<&[u32]>,
    grid: &[(f64, f64, f64)],
    vg_nscore: &Variogram,
    params: &TurningBandsParams,
) -> Result<Realization> {
    turning_bands_in(
        data_locs,
        data_vals,
        data_weights,
        data_holes,
        None,
        None,
        grid,
        vg_nscore,
        params,
    )
}

/// As [`turning_bands`] with `domains` and a `trend`, as in
/// [`crate::sgs_in`]; the bands cover `grid` and the data.
#[allow(clippy::too_many_arguments)]
pub fn turning_bands_in(
    data_locs: &[(f64, f64, f64)],
    data_vals: &[f64],
    data_weights: Option<&[f64]>,
    data_holes: Option<&[u32]>,
    domains: Option<Domains>,
    trend: Option<Trend>,
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
        data_holes,
        domains.map(|d| d.0),
        trend.map(|t| (t.data, t.classes)),
        lo,
        hi,
        vg_nscore,
        params,
        1,
    )?;
    Ok(Realization {
        values: ensemble.realization(0, grid, domains.map(|d| d.1), trend.map(|t| t.nodes))?,
    })
}

/// Summary columns of turning-bands realizations, streamed from the block
/// model file `input` to `output` chunk by chunk (the input columns are
/// kept): `mean`, `variance`, `p_above_<c>` and `mean_above_<c>` per cutoff,
/// `q<p>` per quantile. Memory is bounded by `rows` blocks plus the bands.
/// Returns the global mean and share above each cutoff of every realization.
/// `domains` are the codes of the data and of every block, in file order;
/// `trend` is the trend at the data, its number of classes and the input
/// column holding it at the blocks. Each block is simulated at
/// `discretization` nodes per axis, as [`ceres_core::BlockModel::discretize`],
/// averaged by volume as in [`BlockSupport`]; `[1, 1, 1]` is its centroid.
/// A node takes the domain and trend of its block.
#[allow(clippy::too_many_arguments)]
pub fn turning_bands_to_parquet(
    input: impl AsRef<std::path::Path>,
    output: impl AsRef<std::path::Path>,
    data_locs: &[(f64, f64, f64)],
    data_vals: &[f64],
    data_weights: Option<&[f64]>,
    data_holes: Option<&[u32]>,
    domains: Option<Domains>,
    trend: Option<(&[f64], usize, &str)>,
    vg_nscore: &Variogram,
    params: &TurningBandsParams,
    n: usize,
    options: &ContinuousOptions,
    rows: usize,
    discretization: [usize; 3],
) -> Result<GlobalSummary> {
    let reader = ceres_io::BlockModelReader::open(&input)?;
    if domains.is_some_and(|d| d.1.len() != reader.len()) {
        return Err(SimError::InvalidParameters("one domain per block".into()));
    }
    if let Some((.., column)) = trend
        && !reader.column_names().iter().any(|c| c == column)
    {
        return Err(SimError::InvalidParameters(format!(
            "no trend column {column:?}"
        )));
    }
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for chunk in reader.chunks(rows, Some(&[]))? {
        let nodes = chunk?
            .discretize(discretization)
            .map_err(ceres_io::Error::from)?;
        let (clo, chi) = bounds(&points(&nodes));
        lo = std::array::from_fn(|i| lo[i].min(clo[i]));
        hi = std::array::from_fn(|i| hi[i].max(chi[i]));
    }
    let ensemble = TurningBandsEnsemble::new(
        data_locs,
        data_vals,
        data_weights,
        data_holes,
        domains.map(|d| d.0),
        trend.map(|t| (t.0, t.1)),
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
        let start = total as usize;
        let (nodes, owner, support) = block_nodes(&chunk, discretization)?;
        let codes: Option<Vec<u32>> =
            domains.map(|d| owner.iter().map(|&b| d.1[start + b]).collect());
        let at_nodes: Option<Vec<f64>> = trend
            .map(|(.., column)| {
                let at = float_column(&chunk, column)?;
                Ok::<_, SimError>(owner.iter().map(|&b| at[b]).collect())
            })
            .transpose()?;
        let s = continuous(n, &options, |k| {
            let r = ensemble.realization(k, &nodes, codes.as_deref(), at_nodes.as_deref())?;
            match &support {
                Some(s) => s.mean(&r),
                None => Ok(r),
            }
        })?;
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

/// The simulation nodes of the rows of `chunk`, `n` per axis, the row of each
/// and their averaging to the rows; the centroids when `n` is `[1, 1, 1]`.
fn block_nodes(
    chunk: &ceres_core::BlockModel,
    n: [usize; 3],
) -> Result<(Vec<(f64, f64, f64)>, Vec<usize>, Option<BlockSupport>)> {
    if n == [1, 1, 1] {
        return Ok((points(chunk), (0..chunk.len()).collect(), None));
    }
    use arrow_array::cast::AsArray;
    let fine = chunk.discretize(n).map_err(ceres_io::Error::from)?;
    let owner = fine
        .attributes()
        .column_by_name("block")
        .and_then(|c| c.as_primitive_opt::<arrow_array::types::UInt64Type>())
        .expect("discretize writes the block column")
        .values()
        .iter()
        .map(|&b| b as usize)
        .collect();
    let nodes = points(&fine);
    let support = BlockSupport::new(&nodes, Some(&fine.volumes()), chunk)?;
    Ok((nodes, owner, Some(support)))
}

/// Column `name` of `chunk` as f64; nulls are an error.
fn float_column(chunk: &ceres_core::BlockModel, name: &str) -> Result<Vec<f64>> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float32Type, Float64Type};
    let column = chunk
        .attributes()
        .column_by_name(name)
        .ok_or_else(|| SimError::InvalidParameters(format!("no trend column {name:?}")))?;
    let values: Vec<Option<f64>> = if let Some(c) = column.as_primitive_opt::<Float64Type>() {
        c.iter().collect()
    } else if let Some(c) = column.as_primitive_opt::<Float32Type>() {
        c.iter().map(|v| v.map(f64::from)).collect()
    } else {
        return Err(SimError::InvalidParameters(format!(
            "trend column {name:?} must be float"
        )));
    };
    values
        .into_iter()
        .collect::<Option<_>>()
        .ok_or_else(|| SimError::InvalidParameters(format!("trend column {name:?} has nulls")))
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
    fn band_factor_is_the_dense_cholesky_bit_for_bit() {
        let vg = Variogram::single(Model::Spherical, 1.0, 30.0);
        let (n, step) = (613, 0.7);
        let mut k = nalgebra::DMatrix::<f64>::from_fn(n, n, |i, j| {
            c1(&vg, (i as f64 - j as f64).abs() * step)
        });
        for i in 0..n {
            k[(i, i)] += 1e-9;
        }
        let l = k.cholesky().unwrap().l();
        let rows = band_factor(&vg, n, step);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), i + 1);
            for (j, v) in row.iter().enumerate() {
                assert_eq!(v.to_bits(), l[(i, j)].to_bits(), "({i}, {j})");
            }
        }
        let w: Vec<f64> = (0..n - 2).map(|i| (i as f64 * 0.37).sin()).collect();
        let noise: Vec<Vec<f64>> = (0..11)
            .map(|b| w[..w.len() - 5 * (b % 4)].to_vec())
            .collect();
        for (band, w) in simulate_bands(&rows, &noise).iter().zip(&noise) {
            assert_eq!(band.len(), w.len());
            for (i, v) in band.iter().enumerate() {
                let want: f64 = (0..=i).map(|j| l[(i, j)] * w[j]).sum();
                assert_eq!(v.to_bits(), want.to_bits());
            }
        }
    }

    #[test]
    fn wide_ensembles_follow_the_seed_not_the_thread_count() {
        let data: Vec<_> = (0..60)
            .map(|i| {
                (
                    ((i * 7919) % 5000) as f64,
                    ((i * 104_729) % 2000) as f64,
                    0.0,
                )
            })
            .collect();
        let vals: Vec<f64> = (0..60)
            .map(|i| 1.0 + (i as f64 * 0.37).sin().abs())
            .collect();
        let grid: Vec<_> = (0..200)
            .map(|i| ((i % 20) as f64 * 250.0, (i / 20) as f64 * 200.0, 0.0))
            .collect();
        let (lo, hi) = bounds(&grid);
        let vg = Variogram::single(Model::Spherical, 1.0, 400.0);
        let run = |threads, seed| {
            let params = TurningBandsParams {
                n_bands: 50,
                seed,
                ..Default::default()
            };
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let e = TurningBandsEnsemble::new(
                        &data, &vals, None, None, None, None, lo, hi, &vg, &params, 3,
                    )
                    .unwrap();
                    (0..3)
                        .map(|k| e.realization(k, &grid, None, None).unwrap())
                        .collect::<Vec<_>>()
                })
        };
        let a = run(1, 5);
        assert_eq!(a, run(7, 5));
        assert_eq!(a, run(1, 5));
        assert_ne!(a, run(7, 6));
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
        let r = turning_bands(&data_locs, &data_vals, None, None, &grid, &vg, &params).unwrap();
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
        let a = turning_bands(&data_locs, &data_vals, None, None, &grid, &vg, &params).unwrap();
        let b = turning_bands(&data_locs, &data_vals, None, None, &grid, &vg, &params).unwrap();
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
        let codes: Vec<u32> = (0..40).map(|i| i % 2).collect();
        let nodes: Vec<u32> = grid.iter().map(|p| u32::from(p.0 > 20.0)).collect();
        for domains in [None, Some((&codes[..], &nodes[..]))] {
            let whole = TurningBandsEnsemble::new(
                &data_locs,
                &data_vals,
                None,
                None,
                domains.map(|d| d.0),
                None,
                lo,
                hi,
                &vg,
                &params,
                6,
            )
            .unwrap()
            .summary(&grid, domains.map(|d| d.1), None, &options)
            .unwrap();
            for rows in [7, 160] {
                let output = input.with_extension(format!("{rows}.parquet"));
                let global = turning_bands_to_parquet(
                    &input,
                    &output,
                    &data_locs,
                    &data_vals,
                    None,
                    None,
                    domains,
                    None,
                    &vg,
                    &params,
                    6,
                    &options,
                    rows,
                    [1, 1, 1],
                )
                .unwrap();
                let ceres_io::Stored::Blocks(back) = ceres_io::read_parquet(&output).unwrap()
                else {
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
        let short = Some((&codes[..], &nodes[1..]));
        let output = input.with_extension("short.parquet");
        let r = turning_bands_to_parquet(
            &input,
            &output,
            &data_locs,
            &data_vals,
            None,
            None,
            short,
            None,
            &vg,
            &params,
            6,
            &options,
            7,
            [1, 1, 1],
        );
        assert!(r.is_err());
    }

    #[test]
    fn streamed_blocks_average_the_discretized_realizations() {
        use arrow_array::cast::AsArray;
        use ceres_core::{BlockModel, Geometry};
        let data_locs: Vec<_> = (0..30)
            .map(|i| ((i * 7 % 40) as f64 + 0.3, (i * 11 % 30) as f64 + 0.6, 1.0))
            .collect();
        let data_vals: Vec<f64> = (0..30)
            .map(|i| 1.0 + (i as f64 * 0.37).sin().abs() * 5.0)
            .collect();
        let data_trend: Vec<f64> = data_locs.iter().map(|p| p.0 / 10.0).collect();
        let geometry = Geometry {
            origin: [0.0; 3],
            size: [8.0, 6.0, 2.0],
            count: [5, 5, 1],
            rotation: [0.0; 3],
        };
        let trend: Vec<f64> = (0..25).map(|r| geometry.centroid(r)[0] / 10.0).collect();
        let columns = arrow_array::RecordBatch::try_from_iter([(
            "trend",
            std::sync::Arc::new(arrow_array::Float32Array::from_iter_values(
                trend.iter().map(|&t| t as f32),
            )) as arrow_array::ArrayRef,
        )])
        .unwrap();
        let model = BlockModel::regular(geometry, columns).unwrap();
        let input =
            std::env::temp_dir().join(format!("ceres-tb-blocks-{}.parquet", std::process::id()));
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
        let n = [2, 3, 1];
        let fine = model.discretize(n).unwrap();
        let nodes = points(&fine);
        let owner: Vec<usize> = fine.attributes()["block"]
            .as_primitive::<arrow_array::types::UInt64Type>()
            .values()
            .iter()
            .map(|&b| b as usize)
            .collect();
        let at_nodes: Vec<f64> = owner.iter().map(|&b| f64::from(trend[b] as f32)).collect();
        let support = BlockSupport::new(&nodes, Some(&fine.volumes()), &model).unwrap();
        let (lo, hi) = bounds(&nodes);
        for trended in [false, true] {
            let data_trend = trended.then_some((&data_trend[..], 3));
            let ensemble = TurningBandsEnsemble::new(
                &data_locs, &data_vals, None, None, None, data_trend, lo, hi, &vg, &params, 5,
            )
            .unwrap();
            let whole = continuous(5, &options, |k| {
                let at = trended.then_some(&at_nodes[..]);
                support.mean(&ensemble.realization(k, &nodes, None, at)?)
            })
            .unwrap();
            let stream = |rows: usize, threads: usize| {
                let output = input.with_extension(format!("{rows}-{threads}-{trended}.parquet"));
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| {
                        turning_bands_to_parquet(
                            &input,
                            &output,
                            &data_locs,
                            &data_vals,
                            None,
                            None,
                            None,
                            data_trend.map(|(t, c)| (t, c, "trend")),
                            &vg,
                            &params,
                            5,
                            &options,
                            rows,
                            n,
                        )
                    })
                    .unwrap();
                let ceres_io::Stored::Blocks(back) = ceres_io::read_parquet(&output).unwrap()
                else {
                    panic!("expected a block model")
                };
                back.attributes()["mean"]
                    .as_primitive::<arrow_array::types::Float64Type>()
                    .values()
                    .to_vec()
            };
            let mean = stream(4, 1);
            assert_eq!(mean, whole.mean);
            assert_eq!(stream(25, 3), mean);
        }
        let output = input.with_extension("missing.parquet");
        let missing = turning_bands_to_parquet(
            &input,
            &output,
            &data_locs,
            &data_vals,
            None,
            None,
            None,
            Some((&data_trend[..], 3, "nope")),
            &vg,
            &params,
            2,
            &options,
            4,
            n,
        );
        assert!(missing.is_err());
    }

    #[test]
    fn max_per_hole_caps_the_data_of_one_hole() {
        let locs: Vec<_> = (0..10).map(|i| (0.0, 0.0, i as f64)).collect();
        let vals: Vec<f64> = (0..10).map(|i| (i * 7 % 10) as f64).collect();
        let holes = vec![0; 10];
        let grid: Vec<_> = (0..20).map(|i| (1.0 + i as f64, 0.0, 4.4)).collect();
        let vg = Variogram::single(Model::Spherical, 1.0, 20.0);
        let run = |holes: Option<&[u32]>, max_samples, max_per_hole| {
            let params = TurningBandsParams {
                search: Search {
                    min_samples: 1,
                    max_samples,
                    radius: f64::INFINITY,
                    max_per_hole,
                    ..Default::default()
                },
                seed: 2,
                ..Default::default()
            };
            turning_bands(&locs, &vals, None, holes, &grid, &vg, &params)
                .unwrap()
                .values
        };
        let capped = run(Some(&holes), 8, Some(1));
        assert_eq!(capped, run(None, 1, None));
        assert_ne!(capped, run(None, 8, None));
        assert_eq!(run(Some(&holes), 8, None), run(None, 8, None));
    }

    use crate::sgs::tests::{Zoned, forward, mean, pick, pooled, zoned, zoned_grid, zoned_search};
    use transforms::normal_score;

    const LO: [f64; 3] = [-1.0, -1.0, -1.0];
    const HI: [f64; 3] = [101.0, 101.0, 3.0];

    fn zoned_params(soft: Option<estimation::Soft>, seed: u64) -> TurningBandsParams {
        TurningBandsParams {
            n_bands: 100,
            search: zoned_search(soft).swap_remove(1),
            seed,
            ..Default::default()
        }
    }

    /// The ensemble of `rows` of `z` over the box of the zoned data, with
    /// domain codes `codes` and the trend at the data when `trended`.
    fn ensemble(
        z: &Zoned,
        rows: &[usize],
        codes: Option<&[u32]>,
        trended: bool,
        params: &TurningBandsParams,
        n: usize,
    ) -> TurningBandsEnsemble {
        let trend = pick(&z.trend, rows);
        TurningBandsEnsemble::new(
            &pick(&z.locs, rows),
            &pick(&z.vals, rows),
            Some(&pick(&z.weights, rows)),
            Some(&pick(&z.holes, rows)),
            codes,
            trended.then_some((&trend[..], 3)),
            LO,
            HI,
            &crate::sgs::tests::vg(),
            params,
            n,
        )
        .unwrap()
    }

    fn reals(
        e: &TurningBandsEnsemble,
        grid: &[(f64, f64, f64)],
        nodes: Option<&[u32]>,
        trend: Option<&[f64]>,
    ) -> Vec<Vec<f64>> {
        (0..e.len())
            .map(|k| e.realization(k, grid, nodes, trend).unwrap())
            .collect()
    }

    #[test]
    fn one_label_everywhere_is_no_domains() {
        let z = zoned();
        let (grid, _, node_trend) = zoned_grid();
        let params = zoned_params(Some(estimation::Soft::All(5.0)), 1);
        // Without the contact holes, which put two samples at one location.
        let rows: Vec<usize> = (0..z.locs.len() - 10).collect();
        let (one, all) = (vec![0; rows.len()], vec![0; grid.len()]);
        for trended in [false, true] {
            let trend = trended.then_some(&node_trend[..]);
            let none = ensemble(&z, &rows, None, trended, &params, 3);
            let labeled = ensemble(&z, &rows, Some(&one), trended, &params, 3);
            assert_eq!(
                reals(&none, &grid, None, trend),
                reals(&labeled, &grid, Some(&all), trend)
            );
        }
    }

    #[test]
    fn a_hard_domain_is_simulated_as_if_alone() {
        let z = zoned();
        let (grid, _, node_trend) = zoned_grid();
        let all: Vec<usize> = (0..z.locs.len()).collect();
        let nodes = vec![0; grid.len()];
        let params = zoned_params(None, 6);
        for trended in [false, true] {
            let trend = trended.then_some(&node_trend[..]);
            let both = ensemble(&z, &all, Some(&z.codes), trended, &params, 2);
            let alone = ensemble(&z, &z.of(0), None, trended, &params, 2);
            assert_eq!(
                reals(&both, &grid, Some(&nodes), trend),
                reals(&alone, &grid, None, trend)
            );
        }
    }

    #[test]
    fn soft_data_enter_as_grades_through_the_node_domain() {
        let z = zoned();
        let (grid, nodes, node_trend) = zoned_grid();
        let all: Vec<usize> = (0..z.locs.len()).collect();
        for trended in [false, true] {
            let e = ensemble(&z, &all, Some(&z.codes), trended, &zoned_params(None, 1), 1);
            let into = [forward(&z, 0, trended), forward(&z, 1, trended)];
            for j in all.iter().copied() {
                for code in 0..2 {
                    let want = match z.codes[j] == code {
                        true => e.transforms.scores[j],
                        false => into[code as usize](z.vals[j], z.trend[j]),
                    };
                    assert!((e.score(j, Some(code)) - want).abs() < 1e-12);
                }
            }
        }
        // Soft moves only the nodes within its distance of another domain.
        let soft = 8.0;
        let run = |soft| {
            let e = ensemble(&z, &all, Some(&z.codes), true, &zoned_params(soft, 4), 1);
            e.realization(0, &grid, Some(&nodes), Some(&node_trend))
                .unwrap()
        };
        let (hard, softened) = (run(None), run(Some(estimation::Soft::All(soft))));
        let near = |i: usize| {
            (0..z.locs.len()).any(|j| {
                let (a, b) = (grid[i], z.locs[j]);
                let d = ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2)).sqrt();
                z.codes[j] != nodes[i] && d < soft + 1e-9
            })
        };
        let moved: Vec<bool> = (0..grid.len()).map(|i| hard[i] != softened[i]).collect();
        assert!(moved.iter().any(|&m| m));
        assert!((0..grid.len()).all(|i| !moved[i] || near(i)));
    }

    #[test]
    fn each_domain_reproduces_its_declustered_histogram() {
        let z = zoned();
        let per = 30;
        // Nodes 30 m apart, beyond the search radius of the data.
        let grid: Vec<(f64, f64, f64)> = (0..2 * per)
            .map(|i| (30.0 * (i % 10) as f64, 150.0 + 30.0 * (i / 10) as f64, 1.0))
            .collect();
        let nodes: Vec<u32> = (0..2 * per).map(|i| (i / per) as u32).collect();
        let params = TurningBandsParams {
            step: Some(1.0),
            ..zoned_params(None, 0)
        };
        let (lo, hi) = bounds(&grid);
        let e = TurningBandsEnsemble::new(
            &z.locs,
            &z.vals,
            Some(&z.weights),
            Some(&z.holes),
            Some(&z.codes),
            None,
            lo,
            hi,
            &crate::sgs::tests::vg(),
            &params,
            200,
        )
        .unwrap();
        let reals = reals(&e, &grid, Some(&nodes), None);
        for code in 0..2 {
            let rows = z.of(code);
            let (data, w) = (pick(&z.vals, &rows), pick(&z.weights, &rows));
            let pooled = pooled(&reals, code as usize * per..(code as usize + 1) * per);
            let (want, got) = (mean(&data, &w), mean(&pooled, &vec![1.0; pooled.len()]));
            assert!(
                (got / want - 1.0).abs() < 0.05,
                "domain {code}: {got} vs {want}"
            );
            // Quantiles of the scores through the domain's own table.
            let table = normal_score::transform(&data, Some(&w)).unwrap().table;
            for (q, want) in [
                (0.1, -1.281_551_565_545),
                (0.5, 0.0),
                (0.9, 1.281_551_565_545),
            ] {
                let got = table.forward(pooled[(q * pooled.len() as f64) as usize]);
                assert!((got - want).abs() < 0.1, "domain {code} q{q}: {got}");
            }
        }
    }

    #[test]
    fn domains_follow_the_seed_not_the_thread_count() {
        let z = zoned();
        let (grid, nodes, node_trend) = zoned_grid();
        let all: Vec<usize> = (0..z.locs.len()).collect();
        let run = |threads, seed| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let params = zoned_params(Some(estimation::Soft::All(8.0)), seed);
                    let e = ensemble(&z, &all, Some(&z.codes), true, &params, 2);
                    reals(&e, &grid, Some(&nodes), Some(&node_trend))
                })
        };
        let a = run(1, 3);
        assert_eq!(a, run(4, 3));
        assert_ne!(a, run(4, 4));
    }

    #[test]
    fn bad_target_domains_and_trends_are_errors() {
        let z = zoned();
        let all: Vec<usize> = (0..z.locs.len()).collect();
        let grid = vec![(1.0, 1.0, 1.0), (2.0, 2.0, 1.0)];
        let params = zoned_params(None, 1);
        let zoned = ensemble(&z, &all, Some(&z.codes), true, &params, 1);
        let plain = ensemble(&z, &all, None, false, &params, 1);
        let t = [0.1, 0.2];
        assert!(zoned.realization(0, &grid, Some(&[0, 1]), Some(&t)).is_ok());
        assert!(
            zoned
                .realization(0, &grid, Some(&[0, 2]), Some(&t))
                .is_err()
        );
        assert!(zoned.realization(0, &grid, Some(&[0]), Some(&t)).is_err());
        assert!(zoned.realization(0, &grid, None, Some(&t)).is_err());
        assert!(zoned.realization(0, &grid, Some(&[0, 1]), None).is_err());
        assert!(
            zoned
                .realization(0, &grid, Some(&[0, 1]), Some(&t[1..]))
                .is_err()
        );
        assert!(plain.realization(0, &grid, Some(&[0, 0]), None).is_err());
        assert!(plain.realization(0, &grid, None, Some(&t)).is_err());
    }
}
