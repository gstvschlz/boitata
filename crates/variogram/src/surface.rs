//! Variogram maps: γ on a cut plane, as the polar (angle × lag) grid a heat map
//! draws, each angle reduced to a fitted range.
//!
//! Calling [`crate::experimental`] once per angle would repeat the `O(n²)` pair
//! sweep dozens of times; instead every pair is binned once into a fine
//! (azimuth, dip, lag) lattice, and each angle then sums the lattice cells
//! inside its tolerance cone — a pass whose cost no longer depends on the
//! sample count.
//!
//! Two consequences of aggregating first. The cone test is bidirectional, as in
//! [`crate::experimental`], so a lattice over the lower hemisphere covers every
//! direction. And `bandwidth` has no analogue here: a pair's perpendicular
//! offset from a direction line is not recoverable once the pair is binned, so
//! maps take a cone angle only.

use rayon::prelude::*;

use crate::empirical::{
    Estimator, Experimental, LagBins, Moments, Scale, add, finalise, moments, scale,
};
use crate::error::{Result, VarioError};
use crate::fit::{Weighting, fit};
use crate::model::Model;

/// Angular size of a fine-lattice cell, in degrees. Small enough that a cone
/// boundary lands within a few degrees of the true one, coarse enough that each
/// sweep chunk's private lattice stays at a few megabytes.
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
    /// Row-major γ: `angles.len()` rows of `lags.len()` values; 0 where a bin
    /// holds no pairs, NaN where the correlogram is undefined.
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

/// Pair moments and counts per (direction cell, lag bin) — the one aggregate
/// every direction of a cut plane is answered from.
struct PairHistogram {
    n_bins: usize,
    lag_width: f64,
    sums: Vec<Moments>,
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
        let partials: Vec<(Vec<Moments>, Vec<usize>)> = (0..HISTOGRAM_CHUNKS)
            .into_par_iter()
            .map(|chunk| {
                let mut sums = vec![[0.0f64; 8]; slots];
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
                        add(&mut sums[slot], &moments(values[i], values[j]));
                        counts[slot] += 1;
                    }
                }
                (sums, counts)
            })
            .collect();

        let mut sums = vec![[0.0f64; 8]; slots];
        let mut counts = vec![0usize; slots];
        for (chunk_sums, chunk_counts) in &partials {
            for s in 0..slots {
                add(&mut sums[s], &chunk_sums[s]);
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
            sums,
            counts,
            units,
        })
    }

    /// Sum every cell whose centre lies within the cone about `axis`, then
    /// finalise the result into (γ, count) per lag bin. `cos_tol` is the cosine
    /// of the cone's half-angle; the test is bidirectional, so a cell and its
    /// antipode are equally admitted.
    fn cone(
        &self,
        axis: (f64, f64, f64),
        cos_tol: f64,
        estimator: Estimator,
        scale: Scale,
    ) -> ConeCurve {
        let mut sums = vec![[0.0f64; 8]; self.n_bins];
        let mut counts = vec![0usize; self.n_bins];
        for (cell, u) in self.units.iter().enumerate() {
            let proj = u.0 * axis.0 + u.1 * axis.1 + u.2 * axis.2;
            if proj.abs() < cos_tol {
                continue;
            }
            let base = cell * self.n_bins;
            for b in 0..self.n_bins {
                add(&mut sums[b], &self.sums[base + b]);
                counts[b] += self.counts[base + b];
            }
        }
        let gammas = (0..self.n_bins)
            .map(|b| match finalise(&sums[b], counts[b], estimator, scale) {
                Some((gamma, _)) => gamma,
                None if counts[b] == 0 => 0.0,
                None => f64::NAN,
            })
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

/// Reduce one direction's curve to a range by fitting a single structure to it.
/// `None` when the cone is too sparse for a fit to mean anything — a tight cone
/// with no pairs is a normal outcome, not an error.
fn fit_range(lags: &[f64], curve: &ConeCurve, model: Model, weighting: Weighting) -> Option<f64> {
    let mut exp = Experimental {
        lags: Vec::new(),
        gammas: Vec::new(),
        counts: Vec::new(),
        covariances: None,
    };
    let mut pairs = 0usize;
    for (b, &lag) in lags.iter().enumerate() {
        let count = curve.counts[b];
        if count == 0 || curve.gammas[b].is_nan() {
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
    let scale = scale(values, params.estimator, false)?;

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
            let curve = hist.cone(axis, cos_tol, params.estimator, scale);
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
        let values: Vec<f64> = values.iter().map(|v| v + 3.0).collect();
        let bins = LagBins {
            max_lag: 60.0,
            lag_width: 10.0,
        };
        let hist = PairHistogram::build(&locations, &values, &bins).unwrap();
        for estimator in [
            Estimator::Matheron,
            Estimator::CressieHawkins,
            Estimator::Covariance,
            Estimator::Correlogram,
            Estimator::PairwiseRelative,
        ] {
            let scale = scale(&values, estimator, false).unwrap();
            // cos_tol = -1 admits every cell, so the cone becomes the whole sphere.
            let curve = hist.cone((0.0, 0.0, -1.0), -1.0, estimator, scale);
            let reference =
                experimental(&locations, &values, &bins, estimator, None, false).unwrap();
            assert!(agree(&hist.lag_centres(), &curve, &reference) >= 5);
        }
    }

    fn agree(centres: &[f64], curve: &ConeCurve, reference: &Experimental) -> usize {
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
        compared
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
        for params in [
            PlaneMapParams {
                tolerance: 0.0,
                ..Default::default()
            },
            PlaneMapParams {
                angle_steps: 0,
                ..Default::default()
            },
            PlaneMapParams {
                estimator: Estimator::PairwiseRelative,
                ..Default::default()
            },
        ] {
            let (u, v) = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0));
            assert!(plane_map(&locations, &values, u, v, &params).is_err());
        }
    }
}
