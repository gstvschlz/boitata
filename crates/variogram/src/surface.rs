//! Variogram surfaces: γ over a hemisphere of directions, and γ on a cut plane.
//!
//! [`variogram_surface`] samples a lattice of (azimuth, dip) directions and
//! reduces each one to a fitted range — the radius of a "range sphere", which is
//! the empirical counterpart of the anisotropy ellipsoid. [`plane_map`] answers
//! the same question restricted to one plane through the origin, as the polar
//! (angle × lag) grid a cut-plane heat map draws.
//!
//! Both share a single `O(n²)` pair sweep. Calling [`crate::experimental`] once
//! per output direction would repeat that sweep a few hundred times; instead
//! every pair is binned once into a fine (azimuth, dip, lag) lattice, and each
//! output direction then sums the lattice cells inside its tolerance cone — a
//! pass whose cost no longer depends on the sample count.
//!
//! Two consequences of aggregating first. The cone test is bidirectional, as in
//! [`crate::experimental`], so a lattice over the lower hemisphere covers every
//! direction. And `bandwidth` has no analogue here: a pair's perpendicular
//! offset from a direction line is not recoverable once the pair is binned, so
//! these functions take a cone angle only.

use rayon::prelude::*;

use crate::empirical::{Estimator, Experimental, LagBins};
use crate::error::{Result, VarioError};
use crate::fit::{Weighting, fit};
use crate::model::Model;

/// Angular size of a fine-lattice cell, in degrees. Small enough that a cone
/// boundary lands within a few degrees of the true one, coarse enough that each
/// sweep chunk's private lattice stays well under a megabyte.
const CELL_DEG: f64 = 4.5;

/// Sweep chunks for the pair histogram. Fewer than [`crate::experimental`]'s 64
/// because a chunk here holds a whole (direction × lag) lattice rather than a
/// single lag row; the count is still fixed, so the merge order — and with it
/// the floating-point result — does not depend on the thread count.
const HISTOGRAM_CHUNKS: usize = 16;

/// A direction whose cone caught fewer populated bins than this has no range
/// worth reporting.
const MIN_FIT_BINS: usize = 3;

/// …nor one that caught fewer pairs than this.
const MIN_FIT_PAIRS: usize = 30;

/// Direction-grid resolution and pair selection for a [`variogram_surface`].
#[derive(Debug, Clone)]
pub struct SurfaceParams {
    pub bins: LagBins,
    pub estimator: Estimator,
    /// Half-angle of the cone summed for each direction, in degrees.
    pub tolerance: f64,
    /// Azimuths sampled around the compass.
    pub azimuth_steps: usize,
    /// Dip bands from horizontal to vertical; the grid carries `dip_steps + 1`
    /// dips, from 0° to 90° inclusive.
    pub dip_steps: usize,
    /// Single structure fitted per direction to reduce its curve to a range.
    pub model: Model,
    pub weighting: Weighting,
}

impl Default for SurfaceParams {
    fn default() -> Self {
        Self {
            bins: LagBins::default(),
            estimator: Estimator::Matheron,
            tolerance: 22.5,
            azimuth_steps: 24,
            dip_steps: 6,
            model: Model::Spherical,
            weighting: Weighting::ByCount,
        }
    }
}

/// One sampled direction of a [`VariogramSurface`].
#[derive(Debug, Clone)]
pub struct SurfaceDirection {
    /// Degrees from North, clockwise.
    pub azimuth: f64,
    /// Degrees below horizontal.
    pub dip: f64,
    /// Fitted range, or `None` when the cone was too sparse to fit. Note the
    /// fitter caps a range at roughly `1.5 × max_lag`, so a direction whose true
    /// range exceeds the lag window saturates rather than reporting the truth.
    pub range: Option<f64>,
    /// γ per lag bin, `0.0` where `counts` is 0.
    pub gammas: Vec<f64>,
    pub counts: Vec<usize>,
}

/// γ over a hemisphere of directions, each reduced to a fitted range.
#[derive(Debug, Clone)]
pub struct VariogramSurface {
    /// Lag-bin centres, shared by every direction.
    pub lags: Vec<f64>,
    pub directions: Vec<SurfaceDirection>,
}

/// Resolution and pair selection for a [`plane_map`].
#[derive(Debug, Clone)]
pub struct PlaneMapParams {
    pub bins: LagBins,
    pub estimator: Estimator,
    /// Half-angle of the cone summed for each in-plane direction, in degrees.
    pub tolerance: f64,
    /// In-plane directions sampled over the half-turn `[0, π)`.
    pub angle_steps: usize,
    pub model: Model,
    pub weighting: Weighting,
}

impl Default for PlaneMapParams {
    fn default() -> Self {
        Self {
            bins: LagBins::default(),
            estimator: Estimator::Matheron,
            tolerance: 22.5,
            angle_steps: 36,
            model: Model::Spherical,
            weighting: Weighting::ByCount,
        }
    }
}

/// γ on a plane through the origin, as a polar (angle × lag) grid.
#[derive(Debug, Clone)]
pub struct PlaneMap {
    /// Lag-bin centres, shared by every angle.
    pub lags: Vec<f64>,
    /// In-plane angles in radians, measured from `u` toward `v`, spanning
    /// `[0, π)`. γ is symmetric under a half-turn, so a caller drawing the full
    /// disc mirrors these rows onto the opposite half.
    pub angles: Vec<f64>,
    /// Row-major γ: `angles.len()` rows of `lags.len()` values.
    pub gammas: Vec<f64>,
    /// Row-major pair counts, laid out like `gammas`.
    pub counts: Vec<usize>,
    /// Fitted range per angle — where the surface meets this plane.
    pub ranges: Vec<Option<f64>>,
}

/// Unit vector of an (azimuth, dip) direction in degrees, in (East, North, Up);
/// dip is positive downward. Matches `Direction::unit` in [`crate::empirical`].
pub fn unit_vector(azimuth: f64, dip: f64) -> (f64, f64, f64) {
    let az = azimuth.to_radians();
    let d = dip.to_radians();
    let cosd = d.cos();
    (cosd * az.sin(), cosd * az.cos(), -d.sin())
}

/// Inverse of [`unit_vector`]: azimuth in `[0, 360)` and dip in `[-90, 90]`,
/// both degrees. A zero-length vector reads as due north, horizontal.
pub fn azimuth_dip(v: (f64, f64, f64)) -> (f64, f64) {
    let n = norm(v);
    if n == 0.0 {
        return (0.0, 0.0);
    }
    let (e, north, up) = (v.0 / n, v.1 / n, v.2 / n);
    let dip = (-up).clamp(-1.0, 1.0).asin().to_degrees();
    let mut az = e.atan2(north).to_degrees();
    if az < 0.0 {
        az += 360.0;
    }
    (az, dip)
}

fn norm(v: (f64, f64, f64)) -> f64 {
    (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt()
}

/// Σ(Δz)², Σ|Δz|^½ and pair counts per (direction cell, lag bin) — the one
/// aggregate every direction and every cut plane is answered from.
struct PairHistogram {
    n_bins: usize,
    lag_width: f64,
    sum_sq: Vec<f64>,
    sum_root: Vec<f64>,
    counts: Vec<usize>,
    /// Unit vector at each direction cell's centre, in (East, North, Up).
    units: Vec<(f64, f64, f64)>,
}

impl PairHistogram {
    fn build(locations: &[(f64, f64, f64)], values: &[f64], bins: &LagBins) -> Result<Self> {
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

        let az_cells = (360.0 / CELL_DEG).round() as usize;
        let dip_cells = (90.0 / CELL_DEG).round() as usize;
        let n_bins = ((bins.max_lag / bins.lag_width).ceil() as usize).max(1);
        let slots = az_cells * dip_cells * n_bins;
        let n = locations.len();

        // Strided rows per chunk, merged sequentially below: the same
        // deterministic fold `experimental` uses, for the same reason.
        let partials: Vec<(Vec<f64>, Vec<f64>, Vec<usize>)> = (0..HISTOGRAM_CHUNKS)
            .into_par_iter()
            .map(|chunk| {
                let mut sum_sq = vec![0.0f64; slots];
                let mut sum_root = vec![0.0f64; slots];
                let mut counts = vec![0usize; slots];
                for i in (chunk..n).step_by(HISTOGRAM_CHUNKS) {
                    for j in (i + 1)..n {
                        let pi = locations[i];
                        let pj = locations[j];
                        let (mut dx, mut dy, mut dz) = (pj.0 - pi.0, pj.1 - pi.1, pj.2 - pi.2);
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                        if dist == 0.0 || dist > bins.max_lag {
                            continue;
                        }
                        // γ is symmetric under a half-turn, so fold every pair
                        // into the lower hemisphere and store it once.
                        if dz > 0.0 {
                            dx = -dx;
                            dy = -dy;
                            dz = -dz;
                        }
                        let inv = 1.0 / dist;
                        let dip = (-(dz * inv)).clamp(-1.0, 1.0).asin().to_degrees();
                        let mut az = dx.atan2(dy).to_degrees();
                        if az < 0.0 {
                            az += 360.0;
                        }
                        let ai = ((az / CELL_DEG) as usize).min(az_cells - 1);
                        let di = ((dip / CELL_DEG) as usize).min(dip_cells - 1);
                        let bin = ((dist / bins.lag_width).floor() as usize).min(n_bins - 1);
                        let slot = (di * az_cells + ai) * n_bins + bin;
                        let diff = (values[i] - values[j]).abs();
                        sum_sq[slot] += diff * diff;
                        sum_root[slot] += diff.sqrt();
                        counts[slot] += 1;
                    }
                }
                (sum_sq, sum_root, counts)
            })
            .collect();

        let mut sum_sq = vec![0.0f64; slots];
        let mut sum_root = vec![0.0f64; slots];
        let mut counts = vec![0usize; slots];
        for (chunk_sq, chunk_root, chunk_counts) in &partials {
            for s in 0..slots {
                sum_sq[s] += chunk_sq[s];
                sum_root[s] += chunk_root[s];
                counts[s] += chunk_counts[s];
            }
        }

        let mut units = Vec::with_capacity(az_cells * dip_cells);
        for di in 0..dip_cells {
            let dip = (di as f64 + 0.5) * CELL_DEG;
            for ai in 0..az_cells {
                let az = (ai as f64 + 0.5) * CELL_DEG;
                units.push(unit_vector(az, dip));
            }
        }

        Ok(Self {
            n_bins,
            lag_width: bins.lag_width,
            sum_sq,
            sum_root,
            counts,
            units,
        })
    }

    /// Sum every cell whose centre lies within the cone about `axis`, then
    /// finalise the result into (γ, count) per lag bin. `cos_tol` is the cosine
    /// of the cone's half-angle; the test is bidirectional, so a cell and its
    /// antipode are equally admitted.
    fn cone(&self, axis: (f64, f64, f64), cos_tol: f64, estimator: Estimator) -> ConeCurve {
        let mut sum_sq = vec![0.0f64; self.n_bins];
        let mut sum_root = vec![0.0f64; self.n_bins];
        let mut counts = vec![0usize; self.n_bins];
        for (cell, u) in self.units.iter().enumerate() {
            let proj = u.0 * axis.0 + u.1 * axis.1 + u.2 * axis.2;
            if proj.abs() < cos_tol {
                continue;
            }
            let base = cell * self.n_bins;
            for b in 0..self.n_bins {
                sum_sq[b] += self.sum_sq[base + b];
                sum_root[b] += self.sum_root[base + b];
                counts[b] += self.counts[base + b];
            }
        }
        let gammas = (0..self.n_bins)
            .map(|b| finalise(sum_sq[b], sum_root[b], counts[b], estimator))
            .collect();
        ConeCurve { gammas, counts }
    }

    fn lag_centres(&self) -> Vec<f64> {
        (0..self.n_bins)
            .map(|b| (b as f64 + 0.5) * self.lag_width)
            .collect()
    }
}

/// One direction's dense curve: γ and pair count per lag bin.
struct ConeCurve {
    gammas: Vec<f64>,
    counts: Vec<usize>,
}

/// γ from one lag bin's accumulated sums, using the same formulas
/// [`crate::experimental`] applies so the two agree bin for bin.
fn finalise(sum_sq: f64, sum_root: f64, count: usize, estimator: Estimator) -> f64 {
    if count == 0 {
        return 0.0;
    }
    match estimator {
        Estimator::Matheron => sum_sq / (2.0 * count as f64),
        Estimator::CressieHawkins => {
            let nf = count as f64;
            let mean_root = sum_root / nf;
            let numer = mean_root.powi(4);
            let denom = 0.457 + 0.494 / nf + 0.045 / (nf * nf);
            0.5 * numer / denom
        }
    }
}

/// Reduce one direction's curve to a range by fitting a single structure to it.
/// `None` when the cone is too sparse for a fit to mean anything — a tight cone
/// with no pairs is a normal outcome, not an error.
fn fit_range(lags: &[f64], curve: &ConeCurve, model: Model, weighting: Weighting) -> Option<f64> {
    let mut exp = Experimental {
        lags: Vec::new(),
        gammas: Vec::new(),
        counts: Vec::new(),
    };
    let mut pairs = 0usize;
    for (b, &lag) in lags.iter().enumerate() {
        let count = curve.counts[b];
        if count == 0 {
            continue;
        }
        exp.lags.push(lag);
        exp.gammas.push(curve.gammas[b]);
        exp.counts.push(count);
        pairs += count;
    }
    if exp.lags.len() < MIN_FIT_BINS || pairs < MIN_FIT_PAIRS {
        return None;
    }
    fit(&exp, model, weighting)
        .ok()
        .map(|f| f.variogram.structures[0].range)
}

fn check_tolerance(tolerance: f64) -> Result<f64> {
    if tolerance <= 0.0 || tolerance > 90.0 {
        return Err(VarioError::InvalidParameters(
            "tolerance must be in (0, 90] degrees".into(),
        ));
    }
    Ok(tolerance.to_radians().cos())
}

/// Sample γ over a hemisphere of directions, reducing each to a fitted range.
///
/// The returned grid is rectangular — `azimuth_steps` azimuths for each of
/// `dip_steps + 1` dips — so a caller can build a lattice mesh straight from it.
/// Every azimuth collapses onto the same direction at dip 90°, so that row
/// repeats; the reduction pass is cheap enough that de-duplicating it would cost
/// more in special cases than it saves.
pub fn variogram_surface(
    locations: &[(f64, f64, f64)],
    values: &[f64],
    params: &SurfaceParams,
) -> Result<VariogramSurface> {
    if params.azimuth_steps == 0 || params.dip_steps == 0 {
        return Err(VarioError::InvalidParameters(
            "azimuth_steps and dip_steps must be positive".into(),
        ));
    }
    let cos_tol = check_tolerance(params.tolerance)?;

    let hist = PairHistogram::build(locations, values, &params.bins)?;
    let lags = hist.lag_centres();

    let mut grid = Vec::with_capacity(params.azimuth_steps * (params.dip_steps + 1));
    for k in 0..=params.dip_steps {
        let dip = 90.0 * k as f64 / params.dip_steps as f64;
        for i in 0..params.azimuth_steps {
            grid.push((360.0 * i as f64 / params.azimuth_steps as f64, dip));
        }
    }

    let directions = grid
        .par_iter()
        .map(|&(azimuth, dip)| {
            let curve = hist.cone(unit_vector(azimuth, dip), cos_tol, params.estimator);
            let range = fit_range(&lags, &curve, params.model, params.weighting);
            SurfaceDirection {
                azimuth,
                dip,
                range,
                gammas: curve.gammas,
                counts: curve.counts,
            }
        })
        .collect();

    Ok(VariogramSurface { lags, directions })
}

/// Sample γ on the plane spanned by `u` and `v`, as a polar (angle × lag) grid.
///
/// The basis is orthonormalised, so callers may pass any two independent vectors
/// lying in the plane.
pub fn plane_map(
    locations: &[(f64, f64, f64)],
    values: &[f64],
    u: (f64, f64, f64),
    v: (f64, f64, f64),
    params: &PlaneMapParams,
) -> Result<PlaneMap> {
    if params.angle_steps == 0 {
        return Err(VarioError::InvalidParameters(
            "angle_steps must be positive".into(),
        ));
    }
    let cos_tol = check_tolerance(params.tolerance)?;
    let (u, v) = orthonormalise(u, v)?;

    let hist = PairHistogram::build(locations, values, &params.bins)?;
    let lags = hist.lag_centres();

    let angles: Vec<f64> = (0..params.angle_steps)
        .map(|i| std::f64::consts::PI * i as f64 / params.angle_steps as f64)
        .collect();

    let rows: Vec<(ConeCurve, Option<f64>)> = angles
        .par_iter()
        .map(|&a| {
            let (c, s) = (a.cos(), a.sin());
            let axis = (u.0 * c + v.0 * s, u.1 * c + v.1 * s, u.2 * c + v.2 * s);
            let curve = hist.cone(axis, cos_tol, params.estimator);
            let range = fit_range(&lags, &curve, params.model, params.weighting);
            (curve, range)
        })
        .collect();

    let mut gammas = Vec::with_capacity(angles.len() * lags.len());
    let mut counts = Vec::with_capacity(angles.len() * lags.len());
    let mut ranges = Vec::with_capacity(angles.len());
    for (curve, range) in rows {
        gammas.extend_from_slice(&curve.gammas);
        counts.extend_from_slice(&curve.counts);
        ranges.push(range);
    }

    Ok(PlaneMap {
        lags,
        angles,
        gammas,
        counts,
        ranges,
    })
}

/// Two perpendicular unit vectors spanning a cut plane, in (East, North, Up).
type PlaneBasis = ((f64, f64, f64), (f64, f64, f64));

/// Orthonormalise a plane basis, rejecting degenerate or parallel inputs.
fn orthonormalise(u: (f64, f64, f64), v: (f64, f64, f64)) -> Result<PlaneBasis> {
    let un = norm(u);
    if un < 1e-12 {
        return Err(VarioError::InvalidParameters(
            "plane basis vector u is degenerate".into(),
        ));
    }
    let u = (u.0 / un, u.1 / un, u.2 / un);
    let dot = u.0 * v.0 + u.1 * v.1 + u.2 * v.2;
    let w = (v.0 - dot * u.0, v.1 - dot * u.1, v.2 - dot * u.2);
    let wn = norm(w);
    if wn < 1e-12 {
        return Err(VarioError::InvalidParameters(
            "plane basis vectors u and v are parallel".into(),
        ));
    }
    Ok((u, (w.0 / wn, w.1 / wn, w.2 / wn)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::empirical::experimental;

    /// A deterministic planar grid whose values vary slowly along East and
    /// quickly along North, so continuity is long east–west and short
    /// north–south. All samples share z, so vertical directions hold no pairs.
    fn anisotropic_grid() -> (Vec<(f64, f64, f64)>, Vec<f64>) {
        let mut locations = Vec::new();
        let mut values = Vec::new();
        for ix in 0..14 {
            for iy in 0..14 {
                let x = ix as f64 * 10.0;
                let y = iy as f64 * 10.0;
                locations.push((x, y, 0.0));
                // Wavelength 400 m east, 40 m north.
                values.push(
                    (x / 400.0 * std::f64::consts::TAU).sin()
                        + (y / 40.0 * std::f64::consts::TAU).sin(),
                );
            }
        }
        (locations, values)
    }

    #[test]
    fn azimuth_dip_inverts_unit_vector() {
        for &(az, dip) in &[(0.0, 0.0), (35.0, 12.0), (270.0, 45.0), (123.0, 89.0)] {
            let (a, d) = azimuth_dip(unit_vector(az, dip));
            assert!((a - az).abs() < 1e-9, "azimuth {a} vs {az}");
            assert!((d - dip).abs() < 1e-9, "dip {d} vs {dip}");
        }
    }

    /// The lattice is only a regrouping of the same pairs: admitting every cell
    /// must reproduce `experimental`'s omnidirectional answer exactly.
    #[test]
    fn summing_every_cell_reproduces_the_omnidirectional_variogram() {
        let (locations, values) = anisotropic_grid();
        let bins = LagBins {
            max_lag: 60.0,
            lag_width: 10.0,
        };
        let hist = PairHistogram::build(&locations, &values, &bins).unwrap();
        // cos_tol = -1 admits every cell, so the cone becomes the whole sphere.
        let curve = hist.cone((0.0, 0.0, -1.0), -1.0, Estimator::Matheron);
        let reference =
            experimental(&locations, &values, &bins, Estimator::Matheron, None).unwrap();
        let centres = hist.lag_centres();

        let mut compared = 0;
        for (b, centre) in centres.iter().enumerate() {
            if curve.counts[b] == 0 {
                continue;
            }
            let k = reference
                .lags
                .iter()
                .position(|l| (l - centre).abs() < 1e-9)
                .expect("every populated bin appears in the reference");
            assert_eq!(
                curve.counts[b], reference.counts[k],
                "pair count at {centre}"
            );
            let tol = 1e-9 * reference.gammas[k].abs().max(1.0);
            assert!(
                (curve.gammas[b] - reference.gammas[k]).abs() < tol,
                "γ at {centre}: {} vs {}",
                curve.gammas[b],
                reference.gammas[k]
            );
            compared += 1;
        }
        assert!(
            compared >= 5,
            "expected several populated bins, saw {compared}"
        );
    }

    #[test]
    fn fitted_ranges_are_longest_along_the_continuous_axis() {
        let (locations, values) = anisotropic_grid();
        let params = SurfaceParams {
            bins: LagBins {
                max_lag: 80.0,
                lag_width: 8.0,
            },
            tolerance: 25.0,
            azimuth_steps: 8,
            dip_steps: 1,
            ..Default::default()
        };
        let surface = variogram_surface(&locations, &values, &params).unwrap();

        let at = |azimuth: f64| {
            surface
                .directions
                .iter()
                .find(|d| d.dip == 0.0 && (d.azimuth - azimuth).abs() < 1e-9)
                .unwrap_or_else(|| panic!("no direction at azimuth {azimuth}"))
        };
        let east = at(90.0).range.expect("east direction has pairs");
        let north = at(0.0).range.expect("north direction has pairs");
        assert!(
            east > north * 1.5,
            "east range {east} should clearly exceed north range {north}"
        );
    }

    /// Every sample shares a z, so a cone about the vertical catches nothing —
    /// which must read as "no range", not as a fitted number or an error.
    #[test]
    fn a_direction_without_pairs_reports_no_range() {
        let (locations, values) = anisotropic_grid();
        let params = SurfaceParams {
            bins: LagBins {
                max_lag: 80.0,
                lag_width: 8.0,
            },
            tolerance: 20.0,
            azimuth_steps: 4,
            dip_steps: 1,
            ..Default::default()
        };
        let surface = variogram_surface(&locations, &values, &params).unwrap();
        let vertical: Vec<_> = surface
            .directions
            .iter()
            .filter(|d| d.dip == 90.0)
            .collect();
        assert!(!vertical.is_empty(), "grid should carry the vertical row");
        for d in vertical {
            assert!(
                d.range.is_none(),
                "vertical range should be None, got {:?}",
                d.range
            );
            assert_eq!(d.counts.iter().sum::<usize>(), 0);
        }
    }

    #[test]
    fn plane_map_rows_span_the_half_turn_and_follow_the_anisotropy() {
        let (locations, values) = anisotropic_grid();
        let params = PlaneMapParams {
            bins: LagBins {
                max_lag: 80.0,
                lag_width: 8.0,
            },
            tolerance: 25.0,
            angle_steps: 4,
            ..Default::default()
        };
        // u = East, v = North: row 0 looks east, row 2 looks north.
        let map = plane_map(
            &locations,
            &values,
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            &params,
        )
        .unwrap();

        assert_eq!(map.angles.len(), 4);
        assert_eq!(map.ranges.len(), 4);
        assert_eq!(map.gammas.len(), 4 * map.lags.len());
        assert_eq!(map.counts.len(), 4 * map.lags.len());
        assert!((map.angles[0] - 0.0).abs() < 1e-12);
        assert!((map.angles[2] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);

        let east = map.ranges[0].expect("east row has pairs");
        let north = map.ranges[2].expect("north row has pairs");
        assert!(
            east > north * 1.5,
            "east range {east} should clearly exceed north range {north}"
        );
    }

    #[test]
    fn rejects_a_degenerate_plane_basis() {
        let (locations, values) = anisotropic_grid();
        let params = PlaneMapParams::default();
        // Parallel basis vectors span no plane.
        assert!(
            plane_map(
                &locations,
                &values,
                (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                &params
            )
            .is_err()
        );
        // A zero-length vector is no direction at all.
        assert!(
            plane_map(
                &locations,
                &values,
                (0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                &params
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_out_of_range_parameters() {
        let (locations, values) = anisotropic_grid();
        let bad_tolerance = SurfaceParams {
            tolerance: 0.0,
            ..Default::default()
        };
        assert!(variogram_surface(&locations, &values, &bad_tolerance).is_err());

        let no_azimuths = SurfaceParams {
            azimuth_steps: 0,
            ..Default::default()
        };
        assert!(variogram_surface(&locations, &values, &no_azimuths).is_err());

        let no_angles = PlaneMapParams {
            angle_steps: 0,
            ..Default::default()
        };
        assert!(
            plane_map(
                &locations,
                &values,
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                &no_angles
            )
            .is_err()
        );
    }
}
