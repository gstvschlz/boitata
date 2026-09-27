//! Empirical (experimental) variogram estimation.
//!
//! Classical, robust, covariance, correlogram and pairwise-relative estimators,
//! omnidirectional or directional (azimuth/dip cone with tolerance), all
//! reported in variogram form so any of them can be fitted. Values on a
//! regular grid are paired by index shifts instead of a search over pairs.

use crate::aniso::euclidean;
use crate::error::{Result, VarioError};
use ceres_core::{Geometry, block_frame};
use nalgebra::Vector3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Empirical-variogram estimator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Estimator {
    /// Matheron classical: `γ(h) = 1/(2N) Σ (zᵢ − zⱼ)²`.
    Matheron,
    /// Cressie–Hawkins robust estimator (down-weights outliers).
    CressieHawkins,
    /// `σ² − C(h)`, with `C(h)` the covariance of head and tail values about
    /// their own lag means and `σ²` the sample variance.
    Covariance,
    /// `1 − ρ(h)`, with `ρ(h)` the head/tail correlation at each lag.
    Correlogram,
    /// `1/(2N) Σ (zᵢ − zⱼ)² / ((zᵢ + zⱼ)/2)²`; needs non-negative values.
    PairwiseRelative,
}

/// Optional directional constraint (a cone about a unit direction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Direction {
    /// Azimuth in degrees (from North, clockwise).
    pub azimuth: f64,
    /// Dip in degrees (positive downward).
    pub dip: f64,
    /// Angular half-width tolerance in degrees.
    pub tolerance: f64,
    /// Optional lateral bandwidth (meters). Pairs beyond this offset are rejected.
    pub bandwidth: Option<f64>,
}

impl Direction {
    pub fn unit(&self) -> (f64, f64, f64) {
        let az = self.azimuth.to_radians();
        let dip = self.dip.to_radians();
        // East, North, Up. Dip is positive downward.
        let cosd = dip.cos();
        (cosd * az.sin(), cosd * az.cos(), -dip.sin())
    }
}

/// Where the values sit.
#[derive(Debug, Clone, Copy)]
pub enum Support<'a> {
    /// Scattered points; pairs come from an `O(n²)` sweep.
    Points(&'a [(f64, f64, f64)]),
    /// Cells of a regular grid, one strictly increasing cell index per value.
    /// Every pair at one index offset has the same separation, so each
    /// offset within `max_lag` and the direction cone is binned once and its
    /// pairs gathered by index shifts: `O(n · offsets)`, with the same pairs,
    /// bins and estimates as the sweep over the cell centers, save pairs
    /// exactly on a lag or cone boundary, which round-off can split there.
    Grid(&'a Geometry, &'a [u64]),
}

impl<'a> From<&'a [(f64, f64, f64)]> for Support<'a> {
    fn from(points: &'a [(f64, f64, f64)]) -> Self {
        Support::Points(points)
    }
}

impl<'a> From<&'a Vec<(f64, f64, f64)>> for Support<'a> {
    fn from(points: &'a Vec<(f64, f64, f64)>) -> Self {
        Support::Points(points)
    }
}

impl Support<'_> {
    fn len(&self) -> usize {
        match self {
            Support::Points(p) => p.len(),
            Support::Grid(_, cells) => cells.len(),
        }
    }
}

/// Lag-binning parameters.
#[derive(Debug, Clone)]
pub struct LagBins {
    pub max_lag: f64,
    pub lag_width: f64,
}

impl Default for LagBins {
    fn default() -> Self {
        Self {
            max_lag: 1000.0,
            lag_width: 50.0,
        }
    }
}

/// Experimental variogram: lag centers, semivariances, and pair counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experimental {
    pub lags: Vec<f64>,
    pub gammas: Vec<f64>,
    pub counts: Vec<usize>,
    /// C(h) for [`Estimator::Covariance`], ρ(h) for [`Estimator::Correlogram`],
    /// `None` for the other estimators.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub covariances: Option<Vec<f64>>,
}

/// Estimate an experimental variogram. `standardize` divides the classical,
/// robust and covariance estimates (and C(h)) by the sample variance so the
/// sill is 1; the correlogram and pairwise-relative estimates are
/// dimensionless already and are left as they are.
///
/// `O(n²)` over sample pairs; for large `n` this is the dominant cost.
pub fn experimental<'a>(
    locations: impl Into<Support<'a>>,
    values: &[f64],
    bins: &LagBins,
    estimator: Estimator,
    direction: Option<&Direction>,
    standardize: bool,
) -> Result<Experimental> {
    let locations = locations.into();
    check(locations, values, bins)?;
    let scale = scale(values, estimator, standardize)?;
    let (sums, counts) = sweep(locations, None, bins, direction, false, |i, j, _| {
        moments(values[i], values[j])
    });
    Ok(collect(bins, &sums, &counts, estimator, scale))
}

/// Estimate an experimental cross-variogram of `values` and `other`.
///
/// With `other_locations` `None` the variables are co-located (isotopic) and
/// both estimators apply: [`Estimator::Matheron`] gives
/// `γ₁₂(h) = 1/(2N) Σ (z₁ᵢ − z₁ⱼ)(z₂ᵢ − z₂ⱼ)` and [`Estimator::Covariance`]
/// gives `C₁₂(0) − C₁₂(h)`, `C₁₂(0)` being the sample covariance. Otherwise
/// `other` sits at its own locations (heterotopic): only the cross-covariance
/// is defined, and `C₁₂(0)` is estimated from the pairs closer than half a lag.
/// `C₁₂(h)` pairs `values` at the tail with `other` at the head `h` away;
/// directional cones are one-sided, so reversing the direction estimates
/// `C₁₂(−h) = C₂₁(h)`, and the omnidirectional estimate averages both.
/// `standardize` divides by `σ₁σ₂`.
#[allow(clippy::too_many_arguments)]
pub fn cross_experimental<'a>(
    locations: impl Into<Support<'a>>,
    values: &[f64],
    other_locations: Option<&[(f64, f64, f64)]>,
    other: &[f64],
    bins: &LagBins,
    estimator: Estimator,
    direction: Option<&Direction>,
    standardize: bool,
) -> Result<Experimental> {
    let locations = locations.into();
    check(locations, values, bins)?;
    if !matches!(estimator, Estimator::Matheron | Estimator::Covariance) {
        return Err(VarioError::InvalidParameters(
            "cross-variograms take the matheron or covariance estimator".into(),
        ));
    }
    let divisor = |v1: f64, v2: f64| {
        if !standardize {
            Ok(1.0)
        } else if v1 == 0.0 || v2 == 0.0 {
            Err(VarioError::InsufficientData("values are constant".into()))
        } else {
            Ok((v1 * v2).sqrt())
        }
    };
    let Some(heads) = other_locations else {
        if other.len() != values.len() {
            return Err(VarioError::InsufficientData(
                "other must have one value per location".into(),
            ));
        }
        let (m1, m2) = (mean(values), mean(other));
        let covariance = values
            .iter()
            .zip(other)
            .map(|(a, b)| (a - m1) * (b - m2))
            .sum::<f64>()
            / values.len() as f64;
        let scale = Scale {
            variance: covariance,
            divisor: divisor(variance(values), variance(other))?,
        };
        let pair = |t: usize, h: usize| {
            let (d1, d2) = (values[t] - values[h], other[t] - other[h]);
            let (t1, h2) = (values[t], other[h]);
            [d1 * d2, 0.0, t1, h2, t1 * h2, t1 * t1, h2 * h2, 0.0]
        };
        let (sums, counts) = sweep(locations, None, bins, direction, false, |i, j, side| {
            if side > 0.0 {
                pair(i, j)
            } else if side < 0.0 {
                pair(j, i)
            } else {
                let mut m = pair(i, j);
                add(&mut m, &pair(j, i));
                m.map(|x| x * 0.5)
            }
        });
        return Ok(collect(bins, &sums, &counts, estimator, scale));
    };
    let Support::Points(locations) = locations else {
        return Err(VarioError::InvalidParameters(
            "a grid cross-variogram needs co-located values".into(),
        ));
    };
    check(Support::Points(heads), other, bins)?;
    if estimator == Estimator::Matheron {
        return Err(VarioError::InvalidParameters(
            "the cross-variogram needs co-located values; use the covariance estimator".into(),
        ));
    }
    let tails = Support::Points(locations);
    let (mut sums, mut counts) = sweep(tails, Some(heads), bins, direction, true, |i, j, _| {
        let (t1, h2) = (values[i], other[j]);
        [0.0, 0.0, t1, h2, t1 * h2, t1 * t1, h2 * h2, 0.0]
    });
    let (near, n_near) = (sums.pop().unwrap_or_default(), counts.pop().unwrap_or(0));
    let unit = Scale {
        variance: 0.0,
        divisor: 1.0,
    };
    let Some((_, c0)) = finalize(&near, n_near, Estimator::Covariance, unit) else {
        return Err(VarioError::InsufficientData(
            "no pairs closer than half a lag to estimate the zero-lag cross-covariance".into(),
        ));
    };
    let scale = Scale {
        variance: c0,
        divisor: divisor(variance(values), variance(other))?,
    };
    Ok(collect(bins, &sums, &counts, estimator, scale))
}

/// Direct and cross experimental variograms of every pair of `variables`,
/// all at `locations`: entry `[i][j]` for `i <= j` holds one per direction,
/// or one omnidirectional when `directions` is empty, and the lower triangle
/// is `None`, the layout [`crate::fit_coregionalization`] takes. Each entry
/// is what [`experimental`] or [`cross_experimental`] gives for that pair.
pub fn experimental_set<'a>(
    locations: impl Into<Support<'a>>,
    variables: &[&[f64]],
    bins: &LagBins,
    estimator: Estimator,
    directions: &[Direction],
    standardize: bool,
) -> Result<Vec<Vec<Option<Vec<Experimental>>>>> {
    let locations = locations.into();
    if variables.is_empty() {
        return Err(VarioError::InsufficientData("no variables".into()));
    }
    let directions: Vec<Option<&Direction>> = match directions {
        [] => vec![None],
        d => d.iter().map(Some).collect(),
    };
    let entry = |i: usize, j: usize, d: Option<&Direction>| match i == j {
        true => experimental(locations, variables[i], bins, estimator, d, standardize),
        false => {
            let (a, b) = (variables[i], variables[j]);
            cross_experimental(locations, a, None, b, bins, estimator, d, standardize)
        }
    };
    let n = variables.len();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| match j < i {
                    true => Ok(None),
                    false => directions
                        .iter()
                        .map(|&d| entry(i, j, d))
                        .collect::<Result<_>>()
                        .map(Some),
                })
                .collect()
        })
        .collect()
}

/// Downhole experimental variogram: only pairs of samples in the same hole
/// (equal `holes` ids) are counted. Lag `k` gathers the pairs within half a
/// lag width of `k · lag_width`, so with the width set to the composite length
/// neighbors fall in the first lag, and each lag is the mean distance of its
/// pairs. `O(Σ nₕ²)` over the holes.
pub fn downhole(
    locations: &[(f64, f64, f64)],
    values: &[f64],
    holes: &[u32],
    bins: &LagBins,
    estimator: Estimator,
    standardize: bool,
) -> Result<Experimental> {
    check(Support::Points(locations), values, bins)?;
    if holes.len() != values.len() {
        return Err(VarioError::InsufficientData(
            "holes must have one id per location".into(),
        ));
    }
    let scale = scale(values, estimator, standardize)?;
    let n_bins = ((bins.max_lag / bins.lag_width).ceil() as usize).max(1);
    let mut sums = vec![[0.0f64; 8]; n_bins];
    let mut counts = vec![0usize; n_bins];
    let mut distances = vec![0.0; n_bins];
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by_key(|&i| holes[i]);
    for hole in order.chunk_by(|&a, &b| holes[a] == holes[b]) {
        for (k, &i) in hole.iter().enumerate() {
            for &j in &hole[k + 1..] {
                let dist = euclidean(&locations[i], &locations[j]);
                let k = (dist / bins.lag_width).round() as usize;
                if k == 0 || dist > bins.max_lag {
                    continue;
                }
                let idx = (k - 1).min(n_bins - 1);
                add(&mut sums[idx], &moments(values[i], values[j]));
                counts[idx] += 1;
                distances[idx] += dist;
            }
        }
    }
    let mut exp = collect(bins, &sums, &counts, estimator, scale);
    for lag in &mut exp.lags {
        let b = (*lag / bins.lag_width) as usize;
        *lag = distances[b] / counts[b] as f64;
    }
    Ok(exp)
}

fn check(locations: Support, values: &[f64], bins: &LagBins) -> Result<()> {
    if let Support::Grid(geometry, cells) = locations {
        geometry
            .validate()
            .map_err(|e| VarioError::InvalidParameters(e.to_string()))?;
        if cells.windows(2).any(|w| w[0] >= w[1])
            || cells.last().is_some_and(|&c| c >= geometry.cells())
            || cells.len() >= u32::MAX as usize
        {
            return Err(VarioError::InvalidParameters(
                "grid cells must be strictly increasing indices inside the grid".into(),
            ));
        }
    }
    if locations.len() != values.len() {
        return Err(VarioError::InsufficientData(
            "locations and values length mismatch".into(),
        ));
    }
    if locations.len() < 2 {
        return Err(VarioError::InsufficientData(
            "at least 2 samples required".into(),
        ));
    }
    if bins.lag_width <= 0.0 || bins.max_lag <= 0.0 {
        return Err(VarioError::InvalidParameters(
            "lag_width and max_lag must be positive".into(),
        ));
    }
    Ok(())
}

type Cone<'a> = (&'a Direction, (f64, f64, f64), f64);

/// Moment sums and pair counts per lag bin over the pairs of `tails` with
/// `heads`, or with the later `tails` when `heads` is `None`, within `max_lag`
/// and the direction cone. `terms(i, j, side)` gives a pair's moments, `side`
/// being `+1`/`−1` as `h = head − tail` points along or against the direction
/// (`0` when omnidirectional). Distinct `heads` keep the forward side only, and
/// with `near` one extra bin gathers every pair closer than half a lag. Grid
/// tails take neither `heads` nor `near`.
fn sweep<F>(
    tails: Support,
    heads: Option<&[(f64, f64, f64)]>,
    bins: &LagBins,
    direction: Option<&Direction>,
    near: bool,
    terms: F,
) -> (Vec<Moments>, Vec<usize>)
where
    F: Fn(usize, usize, f64) -> Moments + Sync,
{
    let n_bins = ((bins.max_lag / bins.lag_width).ceil() as usize).max(1);
    let dir = direction.map(|d| (d, d.unit(), d.tolerance.to_radians().cos()));
    match tails {
        Support::Points(tails) => {
            point_sweep(tails, heads, bins, n_bins, dir.as_ref(), near, terms)
        }
        Support::Grid(geometry, cells) => {
            grid_sweep(geometry, cells, bins, n_bins, dir.as_ref(), terms)
        }
    }
}

/// Lag bin and side of the separation `d` of length `dist`; `None` at zero,
/// beyond `max_lag`, or outside the cone or band. `one_sided` rejects the
/// backward side.
fn classify(
    d: (f64, f64, f64),
    dist: f64,
    bins: &LagBins,
    n_bins: usize,
    dir: Option<&Cone>,
    one_sided: bool,
) -> Option<(usize, f64)> {
    if dist == 0.0 || dist > bins.max_lag {
        return None;
    }
    let (dx, dy, dz) = d;
    let mut side = 0.0;
    if let Some((d, u, ct)) = dir {
        let inv = 1.0 / dist;
        let proj = (dx * u.0 + dy * u.1 + dz * u.2) * inv; // cos(angle)
        if proj.abs() < *ct || (one_sided && proj < 0.0) {
            return None;
        }
        if let Some(bw) = d.bandwidth {
            // Perpendicular offset from the direction line.
            let along = dx * u.0 + dy * u.1 + dz * u.2;
            let perp2 = (dx * dx + dy * dy + dz * dz) - along * along;
            if perp2.max(0.0).sqrt() > bw {
                return None;
            }
        }
        side = proj.signum();
    }
    Some((
        ((dist / bins.lag_width).floor() as usize).min(n_bins - 1),
        side,
    ))
}

fn point_sweep<F>(
    tails: &[(f64, f64, f64)],
    heads: Option<&[(f64, f64, f64)]>,
    bins: &LagBins,
    n_bins: usize,
    dir: Option<&Cone>,
    near: bool,
    terms: F,
) -> (Vec<Moments>, Vec<usize>)
where
    F: Fn(usize, usize, f64) -> Moments + Sync,
{
    let slots = n_bins + near as usize;
    let n = tails.len();

    // The O(n²) pair sweep runs over a *fixed* number of chunks, never one sized
    // by the machine's core count: floating-point addition is not associative, so
    // the summation order must be identical on any machine, at any thread count.
    // Each chunk accumulates its own bins in a fixed order and the partials are
    // merged sequentially in chunk-index order below.
    //
    // Row `i` does `n - i - 1` inner iterations, so contiguous chunks of `i`
    // would carry wildly unequal work. Chunk `c` instead takes the strided rows
    // `c, c + N_CHUNKS, c + 2·N_CHUNKS, …`, giving every chunk a near-equal mix
    // of long and short rows; N_CHUNKS is comfortably above any plausible core
    // count so the pool stays fed even where the balance is imperfect.
    const N_CHUNKS: usize = 64;

    let partials: Vec<(Vec<Moments>, Vec<usize>)> = (0..N_CHUNKS)
        .into_par_iter()
        .map(|chunk| {
            let mut sums = vec![[0.0f64; 8]; slots];
            let mut counts = vec![0usize; slots];
            for i in (chunk..n).step_by(N_CHUNKS) {
                let pi = tails[i];
                let (others, first) = heads.map_or((tails, i + 1), |h| (h, 0));
                for (j, pj) in others.iter().enumerate().skip(first) {
                    let d = (pj.0 - pi.0, pj.1 - pi.1, pj.2 - pi.2);
                    let dist = (d.0 * d.0 + d.1 * d.1 + d.2 * d.2).sqrt();
                    if near && dist <= 0.5 * bins.lag_width {
                        add(&mut sums[n_bins], &terms(i, j, 0.0));
                        counts[n_bins] += 1;
                    }
                    if let Some((idx, side)) = classify(d, dist, bins, n_bins, dir, heads.is_some())
                    {
                        add(&mut sums[idx], &terms(i, j, side));
                        counts[idx] += 1;
                    }
                }
            }
            (sums, counts)
        })
        .collect();

    // Sequential merge in chunk-index order — the deterministic fold.
    let mut sums = vec![[0.0f64; 8]; slots];
    let mut counts = vec![0usize; slots];
    for (chunk_sums, chunk_counts) in &partials {
        for b in 0..slots {
            add(&mut sums[b], &chunk_sums[b]);
            counts[b] += chunk_counts[b];
        }
    }
    (sums, counts)
}

/// The pairs of grid cells, by index offset. An offset `(di, dj, dk)` after
/// `(0, 0, 0)` in z-, y-, x-major order always leads to a later cell, so its
/// pairs are each tail with the head that far on, as in the point sweep. Work
/// is split into a fixed list of (offset, block of grid lines) tasks dealt
/// to a fixed number of chunks whose partial sums merge in chunk order: the
/// result does not depend on the thread count.
fn grid_sweep<F>(
    geometry: &Geometry,
    cells: &[u64],
    bins: &LagBins,
    n_bins: usize,
    dir: Option<&Cone>,
    terms: F,
) -> (Vec<Moments>, Vec<usize>)
where
    F: Fn(usize, usize, f64) -> Moments + Sync,
{
    const LINES: i64 = 1024;
    let [nx, ny, nz] = geometry.count.map(|c| c as i64);
    let reach: [i64; 3] = std::array::from_fn(|a| {
        ((bins.max_lag / geometry.size[a]).floor() as i64).min(geometry.count[a] as i64 - 1)
    });
    let frame = block_frame(geometry.rotation).transpose();
    let mut offsets = Vec::new();
    for dk in 0..=reach[2] {
        for dj in -reach[1]..=reach[1] {
            for di in -reach[0]..=reach[0] {
                if dk == 0 && (dj < 0 || (dj == 0 && di <= 0)) {
                    continue;
                }
                let local = [di, dj, dk];
                let h = frame * Vector3::from_fn(|a, _| local[a] as f64 * geometry.size[a]);
                let d = (h.x, h.y, h.z);
                let dist = (d.0 * d.0 + d.1 * d.1 + d.2 * d.2).sqrt();
                if let Some((bin, side)) = classify(d, dist, bins, n_bins, dir, false) {
                    offsets.push((local, bin, side));
                }
            }
        }
    }
    let slot = (cells.len() as u64 != geometry.cells()).then(|| {
        let mut slot = vec![u32::MAX; geometry.cells() as usize];
        for (row, &c) in cells.iter().enumerate() {
            slot[c as usize] = row as u32;
        }
        slot
    });
    let blocks = ((ny * nz) as usize).div_ceil(LINES as usize);
    let tasks = offsets.len() * blocks;
    // As in the point sweep: a fixed number of chunks, each taking every
    // N_CHUNKS-th task, merged in chunk order.
    const N_CHUNKS: usize = 64;
    let partials: Vec<(Vec<Moments>, Vec<usize>)> = (0..N_CHUNKS)
        .into_par_iter()
        .map(|chunk| {
            let mut sums = vec![[0.0f64; 8]; n_bins];
            let mut counts = vec![0usize; n_bins];
            for task in (chunk..tasks).step_by(N_CHUNKS) {
                let ([di, dj, dk], bin, side) = offsets[task / blocks];
                let first = (task % blocks) as i64 * LINES;
                let shift = di + nx * (dj + ny * dk);
                for line in first..(first + LINES).min(ny * nz) {
                    let (j, k) = (line % ny, line / ny);
                    if !(0..ny).contains(&(j + dj)) || !(0..nz).contains(&(k + dk)) {
                        continue;
                    }
                    for i in (-di).max(0)..nx - di.max(0) {
                        let tail = (i + nx * line) as usize;
                        let head = (tail as i64 + shift) as usize;
                        let (tail, head) = match &slot {
                            None => (tail, head),
                            Some(s) if s[tail] == u32::MAX || s[head] == u32::MAX => continue,
                            Some(s) => (s[tail] as usize, s[head] as usize),
                        };
                        add(&mut sums[bin], &terms(tail, head, side));
                        counts[bin] += 1;
                    }
                }
            }
            (sums, counts)
        })
        .collect();
    let mut sums = vec![[0.0f64; 8]; n_bins];
    let mut counts = vec![0usize; n_bins];
    for (chunk_sums, chunk_counts) in &partials {
        for b in 0..n_bins {
            add(&mut sums[b], &chunk_sums[b]);
            counts[b] += chunk_counts[b];
        }
    }
    (sums, counts)
}

fn collect(
    bins: &LagBins,
    sums: &[Moments],
    counts: &[usize],
    estimator: Estimator,
    scale: Scale,
) -> Experimental {
    let mut exp = Experimental {
        lags: Vec::new(),
        gammas: Vec::new(),
        counts: Vec::new(),
        covariances: None,
    };
    let mut covariances = Vec::new();
    for (b, (s, &c)) in sums.iter().zip(counts).enumerate() {
        if let Some((gamma, covariance)) = finalize(s, c, estimator, scale) {
            exp.lags.push((b as f64 + 0.5) * bins.lag_width);
            exp.gammas.push(gamma);
            exp.counts.push(c);
            covariances.push(covariance);
        }
    }
    if matches!(estimator, Estimator::Covariance | Estimator::Correlogram) {
        exp.covariances = Some(covariances);
    }
    exp
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn variance(values: &[f64]) -> f64 {
    let m = mean(values);
    values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / values.len() as f64
}

/// Per-pair terms summed in each lag bin: squared and root differences, head,
/// tail, head·tail, head², tail² and the relative squared difference.
pub(crate) type Moments = [f64; 8];

pub(crate) fn moments(head: f64, tail: f64) -> Moments {
    let diff = (head - tail).abs();
    let pair = head + tail;
    let relative = if pair > 0.0 {
        4.0 * diff * diff / (pair * pair)
    } else {
        0.0
    };
    [
        diff * diff,
        diff.sqrt(),
        head,
        tail,
        head * tail,
        head * head,
        tail * tail,
        relative,
    ]
}

pub(crate) fn add(sums: &mut Moments, terms: &Moments) {
    for (s, t) in sums.iter_mut().zip(terms) {
        *s += t;
    }
}

/// Sample variance and the divisor applied to γ and C(h), after checking the
/// values suit the estimator.
#[derive(Clone, Copy)]
pub(crate) struct Scale {
    variance: f64,
    divisor: f64,
}

pub(crate) fn scale(values: &[f64], estimator: Estimator, standardize: bool) -> Result<Scale> {
    if estimator == Estimator::PairwiseRelative && values.iter().any(|&v| v < 0.0) {
        return Err(VarioError::InvalidParameters(
            "pairwise-relative needs non-negative values".into(),
        ));
    }
    let variance = variance(values);
    let divide = standardize
        && !matches!(
            estimator,
            Estimator::Correlogram | Estimator::PairwiseRelative
        );
    if (divide || estimator == Estimator::Covariance) && variance == 0.0 {
        return Err(VarioError::InsufficientData("values are constant".into()));
    }
    let divisor = if divide { variance } else { 1.0 };
    Ok(Scale { variance, divisor })
}

/// γ of one lag bin and, for the covariance and correlogram estimators, C(h)
/// or ρ(h) (NaN otherwise). `None` for an empty bin or an undefined ρ(h).
pub(crate) fn finalize(
    sums: &Moments,
    count: usize,
    estimator: Estimator,
    scale: Scale,
) -> Option<(f64, f64)> {
    if count == 0 {
        return None;
    }
    let nf = count as f64;
    let [sq, root, head, tail, prod, head2, tail2, relative] = sums.map(|s| s / nf);
    let covariance = prod - head * tail;
    let (gamma, raw) = match estimator {
        Estimator::Matheron => (sq / 2.0, f64::NAN),
        Estimator::CressieHawkins => {
            // γ(h) = 0.5 · (1/N Σ |Δz|^0.5)^4 / (0.457 + 0.494/N + 0.045/N²)
            let denom = 0.457 + 0.494 / nf + 0.045 / (nf * nf);
            (0.5 * root.powi(4) / denom, f64::NAN)
        }
        Estimator::Covariance => (scale.variance - covariance, covariance),
        Estimator::Correlogram => {
            let spread = ((head2 - head * head) * (tail2 - tail * tail)).sqrt();
            if spread.is_nan() || spread <= 0.0 {
                return None;
            }
            let rho = covariance / spread;
            (1.0 - rho, rho)
        }
        Estimator::PairwiseRelative => (relative / 2.0, f64::NAN),
    };
    Some((gamma / scale.divisor, raw / scale.divisor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matheron_linear_trend() {
        // Points on a line with linear values; short lags → small γ, longer → larger.
        let locs: Vec<(f64, f64, f64)> = (0..10).map(|i| (i as f64 * 10.0, 0.0, 0.0)).collect();
        let vals: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let bins = LagBins {
            max_lag: 100.0,
            lag_width: 10.0,
        };
        let e = experimental(&locs, &vals, &bins, Estimator::Matheron, None, false).unwrap();
        assert!(!e.lags.is_empty());
        assert_eq!(e.lags.len(), e.gammas.len());
        // Monotone increasing for a linear field.
        for w in e.gammas.windows(2) {
            assert!(w[1] >= w[0] - 1e-9);
        }
    }

    #[test]
    fn robust_estimator_runs() {
        let locs: Vec<(f64, f64, f64)> = (0..8).map(|i| (i as f64, 0.0, 0.0)).collect();
        let vals = vec![1.0, 2.0, 1.5, 3.0, 2.5, 10.0, 2.0, 2.2]; // one outlier
        let bins = LagBins {
            max_lag: 8.0,
            lag_width: 1.0,
        };
        let e = experimental(&locs, &vals, &bins, Estimator::CressieHawkins, None, false).unwrap();
        assert!(!e.gammas.is_empty());
        assert!(e.gammas.iter().all(|g| g.is_finite() && *g >= 0.0));
    }

    #[test]
    fn directional_filters_pairs() {
        // Cross pattern; East-West direction should only pick East-West pairs.
        let locs = vec![
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (0.0, 10.0, 0.0),
            (0.0, 20.0, 0.0),
        ];
        let vals = vec![1.0, 2.0, 5.0, 9.0];
        let bins = LagBins {
            max_lag: 30.0,
            lag_width: 10.0,
        };
        let dir = Direction {
            azimuth: 90.0, // East
            dip: 0.0,
            tolerance: 10.0,
            bandwidth: None,
        };
        let e = experimental(&locs, &vals, &bins, Estimator::Matheron, Some(&dir), false).unwrap();
        // Only the East-West pair (0,0,0)-(10,0,0) qualifies.
        let total: usize = e.counts.iter().sum();
        assert_eq!(total, 1);
    }

    #[test]
    fn pair_counts_match_serial_expectation() {
        // 10 collinear points spaced 10 apart. A pair at distance 10k bins into
        // lag index k (a boundary distance rounds up), and there are 10 − k such
        // pairs; bin 0 stays empty, so the serial sweep yields exactly 9..=1.
        let locs: Vec<(f64, f64, f64)> = (0..10).map(|i| (i as f64 * 10.0, 0.0, 0.0)).collect();
        let vals: Vec<f64> = (0..10).map(|i| (i as f64 * 0.7).sin()).collect();
        let bins = LagBins {
            max_lag: 100.0,
            lag_width: 10.0,
        };
        let e = experimental(&locs, &vals, &bins, Estimator::Matheron, None, false).unwrap();
        assert_eq!(e.counts, vec![9, 8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(
            e.lags,
            vec![15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0]
        );
    }

    #[test]
    fn deterministic_across_thread_counts() {
        // The audited property: bit-identical output on any thread count. Run the
        // same estimation inside pools of several sizes and require exact equality
        // for every estimator, omnidirectional and directional.
        let n = 300;
        let locs: Vec<(f64, f64, f64)> = (0..n)
            .map(|i| {
                let t = i as f64;
                ((t * 7.3) % 97.0, (t * 3.1) % 83.0, (t * 1.7) % 13.0)
            })
            .collect();
        let vals: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.37).sin() * 10.0 + (i as f64 * 0.11).cos() * 3.0 + 20.0)
            .collect();
        let bins = LagBins {
            max_lag: 80.0,
            lag_width: 5.0,
        };
        let dir = Direction {
            azimuth: 45.0,
            dip: 10.0,
            tolerance: 25.0,
            bandwidth: Some(30.0),
        };

        for estimator in ALL {
            for direction in [None, Some(&dir)] {
                let reference =
                    experimental(&locs, &vals, &bins, estimator, direction, false).unwrap();
                for k in [1usize, 2, 3, 7] {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(k)
                        .build()
                        .unwrap();
                    let e = pool
                        .install(|| experimental(&locs, &vals, &bins, estimator, direction, false))
                        .unwrap();
                    assert_eq!(e.lags, reference.lags, "lags differ at {k} threads");
                    assert_eq!(e.gammas, reference.gammas, "gammas differ at {k} threads");
                    assert_eq!(e.counts, reference.counts, "counts differ at {k} threads");
                }
            }
        }
    }

    const ALL: [Estimator; 5] = [
        Estimator::Matheron,
        Estimator::CressieHawkins,
        Estimator::Covariance,
        Estimator::Correlogram,
        Estimator::PairwiseRelative,
    ];

    /// Moving sum of seeded white noise on a unit-spaced line: stationary,
    /// continuous at short lags, correlated up to the window width.
    fn field(n: usize, window: usize) -> (Vec<(f64, f64, f64)>, Vec<f64>) {
        let mut state = 42u64;
        let noise: Vec<f64> = (0..n + window)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect();
        let values = noise
            .windows(window)
            .take(n)
            .map(|w| w.iter().sum())
            .collect();
        ((0..n).map(|i| (i as f64, 0.0, 0.0)).collect(), values)
    }

    fn run(locs: &[(f64, f64, f64)], vals: &[f64], e: Estimator, std: bool) -> Vec<f64> {
        let bins = LagBins {
            max_lag: 60.0,
            lag_width: 1.0,
        };
        experimental(locs, vals, &bins, e, None, std)
            .unwrap()
            .gammas
    }

    #[test]
    fn covariance_at_zero_lag_is_the_variance() {
        let (locs, vals) = field(4000, 40);
        let var = variance(&vals);
        let bins = LagBins {
            max_lag: 60.0,
            lag_width: 1.0,
        };
        let e = experimental(&locs, &vals, &bins, Estimator::Covariance, None, false).unwrap();
        let c = e.covariances.unwrap();
        assert!(
            (c[0] / var - 1.0).abs() < 0.05,
            "C(0+) = {}, σ² = {var}",
            c[0]
        );
        for (g, c) in e.gammas.iter().zip(&c) {
            assert!((g + c - var).abs() < 1e-9);
        }
    }

    #[test]
    fn correlogram_is_one_minus_standardized_variogram() {
        let (locs, vals) = field(4000, 20);
        let rho = run(&locs, &vals, Estimator::Correlogram, false);
        let gamma = run(&locs, &vals, Estimator::Matheron, true);
        for (r, g) in rho.iter().zip(&gamma) {
            assert!((r - g).abs() < 0.1, "1 − ρ = {r}, γ/σ² = {g}");
        }
        let affine: Vec<f64> = vals.iter().map(|v| 3.0 * v - 7.0).collect();
        let shifted = run(&locs, &affine, Estimator::Correlogram, false);
        for (a, r) in shifted.iter().zip(&rho) {
            assert!((a - r).abs() < 1e-9);
        }
    }

    #[test]
    fn pairwise_relative_is_scale_invariant() {
        let (locs, vals) = field(500, 10);
        let vals: Vec<f64> = vals.iter().map(|v| v + 10.0).collect();
        let scaled: Vec<f64> = vals.iter().map(|v| 250.0 * v).collect();
        let a = run(&locs, &vals, Estimator::PairwiseRelative, false);
        let b = run(&locs, &scaled, Estimator::PairwiseRelative, false);
        for (a, b) in a.iter().zip(&b) {
            assert!((a - b).abs() <= 1e-12 * a.abs());
        }
        let negative = vec![-1.0; vals.len()];
        let bins = LagBins::default();
        let e = Estimator::PairwiseRelative;
        assert!(experimental(&locs, &negative, &bins, e, None, false).is_err());
    }

    #[test]
    fn standardize_divides_by_the_variance() {
        let (locs, vals) = field(500, 10);
        let var = variance(&vals);
        for e in &ALL[..3] {
            let raw = run(&locs, &vals, *e, false);
            for (s, r) in run(&locs, &vals, *e, true).iter().zip(&raw) {
                assert!((s - r / var).abs() < 1e-12);
            }
        }
        let vals: Vec<f64> = vals.iter().map(|v| v + 10.0).collect();
        for e in &ALL[3..] {
            assert_eq!(run(&locs, &vals, *e, true), run(&locs, &vals, *e, false));
        }
    }

    fn scattered(n: usize) -> (Vec<(f64, f64, f64)>, Vec<f64>) {
        let locs = (0..n)
            .map(|i| {
                let t = i as f64;
                ((t * 7.3) % 97.0, (t * 3.1) % 83.0, (t * 1.7) % 13.0)
            })
            .collect();
        let vals = (0..n)
            .map(|i| (i as f64 * 0.37).sin() * 10.0 + (i as f64 * 0.11).cos() * 3.0 + 20.0)
            .collect();
        (locs, vals)
    }

    fn cone(azimuth: f64) -> Direction {
        Direction {
            azimuth,
            dip: 0.0,
            tolerance: 20.0,
            bandwidth: None,
        }
    }

    #[test]
    fn cross_variogram_with_itself_is_the_direct_variogram() {
        let (locs, vals) = scattered(300);
        let bins = LagBins {
            max_lag: 80.0,
            lag_width: 5.0,
        };
        let dir = cone(45.0);
        for direction in [None, Some(&dir)] {
            let m = Estimator::Matheron;
            let direct = experimental(&locs, &vals, &bins, m, direction, false).unwrap();
            let cross =
                cross_experimental(&locs, &vals, None, &vals, &bins, m, direction, false).unwrap();
            assert_eq!(cross.gammas, direct.gammas);
            assert_eq!(cross.counts, direct.counts);
            assert_eq!(cross.lags, direct.lags);
        }
    }

    #[test]
    fn cross_variogram_of_an_affine_image_is_scaled() {
        let (locs, vals) = scattered(300);
        let other: Vec<f64> = vals.iter().map(|v| -2.5 * v + 3.0).collect();
        let bins = LagBins {
            max_lag: 80.0,
            lag_width: 5.0,
        };
        let m = Estimator::Matheron;
        let direct = experimental(&locs, &vals, &bins, m, None, false).unwrap();
        let cross = cross_experimental(&locs, &vals, None, &other, &bins, m, None, false).unwrap();
        for (c, g) in cross.gammas.iter().zip(&direct.gammas) {
            assert!((c + 2.5 * g).abs() <= 1e-9 * g.abs(), "{c} vs −2.5·{g}");
        }
    }

    #[test]
    fn cross_covariance_swaps_with_lag_reversal() {
        let (locs, vals) = field(600, 12);
        let other: Vec<f64> = vals[3..].to_vec();
        let (locs, vals) = (&locs[..other.len()], &vals[..other.len()]);
        let bins = LagBins {
            max_lag: 10.0,
            lag_width: 2.0,
        };
        let (forward, back) = (cone(90.0), cone(270.0));
        let tails: Vec<_> = locs.iter().step_by(2).copied().collect();
        let heads: Vec<_> = locs.iter().skip(1).step_by(2).copied().collect();
        let even: Vec<f64> = vals.iter().step_by(2).copied().collect();
        let odd: Vec<f64> = other.iter().skip(1).step_by(2).copied().collect();
        let cases = [
            (locs, vals, None, &other[..]),
            (&tails[..], &even[..], Some(&heads[..]), &odd[..]),
        ];
        let run = |a: &[(f64, f64, f64)], x: &[f64], b, y: &[f64], d: &Direction| {
            let c = Estimator::Covariance;
            cross_experimental(a, x, b, y, &bins, c, Some(d), false).unwrap()
        };
        for (l1, v1, l2, v2) in cases {
            let c12 = run(l1, v1, l2, v2, &forward);
            let c21 = match l2 {
                None => run(l1, v2, None, v1, &back),
                Some(l2) => run(l2, v2, Some(l1), v1, &back),
            };
            let reversed = run(l1, v1, l2, v2, &back).covariances.unwrap();
            assert_eq!(c12.counts, c21.counts);
            let (a, b) = (c12.covariances.unwrap(), c21.covariances.unwrap());
            for ((a, b), r) in a.iter().zip(&b).zip(&reversed) {
                assert!((a - b).abs() < 1e-12, "C12(h) = {a}, C21(−h) = {b}");
                assert!((a - r).abs() > 1e-2, "C12(h) = {a} vs C12(−h) = {r}");
            }
        }
    }

    #[test]
    fn cross_estimates_are_deterministic_across_thread_counts() {
        let (locs, vals) = scattered(300);
        let other: Vec<f64> = vals.iter().map(|v| (v * 0.3).cos()).collect();
        let bins = LagBins {
            max_lag: 80.0,
            lag_width: 5.0,
        };
        let dir = cone(45.0);
        let swapped: Vec<_> = locs.iter().map(|p| (p.1, p.0, p.2)).collect();
        let run = |l2, e, d| cross_experimental(&locs, &vals, l2, &other, &bins, e, d, false);
        for (l2, e) in [
            (None, Estimator::Matheron),
            (None, Estimator::Covariance),
            (Some(&swapped[..]), Estimator::Covariance),
        ] {
            for d in [None, Some(&dir)] {
                let reference = run(l2, e, d).unwrap();
                for k in [1usize, 2, 3, 7] {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(k)
                        .build()
                        .unwrap();
                    let got = pool.install(|| run(l2, e, d)).unwrap();
                    assert_eq!(got.gammas, reference.gammas, "gammas differ at {k} threads");
                    assert_eq!(got.counts, reference.counts, "counts differ at {k} threads");
                }
            }
        }
    }

    #[test]
    fn cross_rejects_what_it_cannot_estimate() {
        let (locs, vals) = field(50, 5);
        let bins = LagBins {
            max_lag: 10.0,
            lag_width: 1.0,
        };
        let near: Vec<_> = locs.iter().map(|p| (p.0, 0.5, 0.0)).collect();
        let far: Vec<_> = locs.iter().map(|p| (p.0, 0.6, 0.0)).collect();
        let run = |l2, e| cross_experimental(&locs, &vals, l2, &vals, &bins, e, None, false);
        assert!(run(None, Estimator::CressieHawkins).is_err());
        assert!(run(Some(&near[..]), Estimator::Matheron).is_err());
        assert!(run(Some(&near[..]), Estimator::Covariance).is_ok());
        assert!(run(Some(&far[..]), Estimator::Covariance).is_err());
        assert!(run(Some(&far[..10]), Estimator::Covariance).is_err());
    }

    fn grid(rotation: [f64; 3]) -> (Geometry, Vec<f64>, Vec<f64>) {
        let geometry = Geometry {
            origin: [100.0, 200.0, 50.0],
            size: [2.0, 3.0, 1.5],
            count: [17, 13, 4],
            rotation,
        };
        let (_, a) = field(geometry.cells() as usize, 9);
        let a: Vec<f64> = a.iter().map(|v| v + 10.0).collect();
        let b = a
            .iter()
            .enumerate()
            .map(|(i, v)| 0.6 * v + (i as f64 * 0.91).sin())
            .collect();
        (geometry, a, b)
    }

    fn centers(geometry: &Geometry, cells: &[u64]) -> Vec<(f64, f64, f64)> {
        cells
            .iter()
            .map(|&c| {
                let [x, y, z] = geometry.centroid(c);
                (x, y, z)
            })
            .collect()
    }

    fn agree(a: &Experimental, b: &Experimental) {
        assert_eq!(a.counts, b.counts);
        assert_eq!(a.lags, b.lags);
        for (x, y) in a.gammas.iter().zip(&b.gammas) {
            assert!((x - y).abs() <= 1e-10 * x.abs().max(1.0), "{x} vs {y}");
        }
    }

    #[test]
    fn grid_shifts_match_the_pair_sweep() {
        let bins = LagBins {
            max_lag: 19.7,
            lag_width: 2.3,
        };
        let dir = Direction {
            azimuth: 37.0,
            dip: 11.0,
            tolerance: 21.0,
            bandwidth: Some(4.1),
        };
        for rotation in [[0.0; 3], [30.0, 12.0, 5.0]] {
            let (geometry, a, b) = grid(rotation);
            let all: Vec<u64> = (0..geometry.cells()).collect();
            let masked: Vec<u64> = all.iter().copied().filter(|c| c % 7 != 3).collect();
            for cells in [&all, &masked] {
                let at = centers(&geometry, cells);
                let pick = |v: &[f64]| cells.iter().map(|&c| v[c as usize]).collect::<Vec<_>>();
                let (a, b) = (pick(&a), pick(&b));
                let on = Support::Grid(&geometry, cells);
                for d in [None, Some(&dir)] {
                    for e in ALL {
                        let by_grid = experimental(on, &a, &bins, e, d, false).unwrap();
                        agree(
                            &by_grid,
                            &experimental(&at, &a, &bins, e, d, false).unwrap(),
                        );
                    }
                    for e in [Estimator::Matheron, Estimator::Covariance] {
                        let by_grid = cross_experimental(on, &a, None, &b, &bins, e, d, true);
                        let by_pairs = cross_experimental(&at, &a, None, &b, &bins, e, d, true);
                        agree(&by_grid.unwrap(), &by_pairs.unwrap());
                    }
                }
            }
        }
    }

    #[test]
    fn a_set_holds_every_direct_and_cross_variogram() {
        let (locs, a) = scattered(200);
        let b: Vec<f64> = a.iter().map(|v| (v * 0.3).cos()).collect();
        let c: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - 4.0 * y).collect();
        let bins = LagBins {
            max_lag: 80.0,
            lag_width: 5.0,
        };
        let dirs = [cone(0.0), cone(90.0)];
        let m = Estimator::Matheron;
        let set = experimental_set(&locs, &[&a, &b, &c], &bins, m, &dirs, false).unwrap();
        let swapped = experimental_set(&locs, &[&c, &b, &a], &bins, m, &dirs, false).unwrap();
        for (d, dir) in dirs.iter().enumerate() {
            let direct = experimental(&locs, &b, &bins, m, Some(dir), false).unwrap();
            assert_eq!(set[1][1].as_ref().unwrap()[d].gammas, direct.gammas);
            for (i, j) in [(0, 1), (0, 2), (1, 2)] {
                let (x, y) = (&set[i][j].as_ref().unwrap()[d], &swapped[2 - j][2 - i]);
                assert_eq!(x.gammas, y.as_ref().unwrap()[d].gammas);
                assert!(set[j][i].is_none());
            }
        }
        let one = experimental_set(&locs, &[&a], &bins, m, &[], false).unwrap();
        let direct = experimental(&locs, &a, &bins, m, None, false).unwrap();
        assert_eq!(one[0][0].as_ref().unwrap()[0].gammas, direct.gammas);
        assert!(experimental_set(&locs, &[], &bins, m, &[], false).is_err());
    }

    #[test]
    fn grid_sets_are_deterministic_across_thread_counts() {
        let (geometry, a, b) = grid([20.0, 0.0, 0.0]);
        let cells: Vec<u64> = (0..geometry.cells()).collect();
        let bins = LagBins {
            max_lag: 15.0,
            lag_width: 1.5,
        };
        let dirs = [cone(0.0), cone(90.0)];
        let run = || {
            let on = Support::Grid(&geometry, &cells);
            let m = Estimator::Matheron;
            experimental_set(on, &[&a, &b], &bins, m, &dirs, false).unwrap()
        };
        let reference = run();
        for k in [1usize, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(k)
                .build()
                .unwrap();
            let got = pool.install(run);
            for (r, g) in reference.iter().flatten().zip(got.iter().flatten()) {
                for (r, g) in r.iter().flatten().zip(g.iter().flatten()) {
                    assert_eq!(r.gammas, g.gammas, "gammas differ at {k} threads");
                    assert_eq!(r.counts, g.counts, "counts differ at {k} threads");
                }
            }
        }
    }

    #[test]
    fn grids_reject_bad_cells_and_separate_heads() {
        let (geometry, a, _) = grid([0.0; 3]);
        let bins = LagBins::default();
        let m = Estimator::Matheron;
        let backwards: Vec<u64> = (0..geometry.cells()).rev().collect();
        assert!(
            experimental(
                Support::Grid(&geometry, &backwards),
                &a,
                &bins,
                m,
                None,
                false
            )
            .is_err()
        );
        let cells: Vec<u64> = (0..geometry.cells()).collect();
        let on = Support::Grid(&geometry, &cells);
        let heads = centers(&geometry, &cells);
        let c = Estimator::Covariance;
        assert!(cross_experimental(on, &a, Some(&heads), &a, &bins, c, None, false).is_err());
    }
}
