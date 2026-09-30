//! Variogram volumes: γ on a 3D grid of lag vectors, and the principal axes of
//! continuity read over a set of directions on the sphere.
//!
//! The grid is the 3D counterpart of a variogram map: each pair adds to the
//! cell holding its lag vector and to the cell holding its opposite, so the
//! volume is symmetric through the origin. The axes come from the same pair
//! lattice as [`crate::surface::plane_map`]. A single structure is fitted to
//! the omnidirectional curve; in each of a near-uniform (Fibonacci) set of
//! directions, the lag where γ crosses the nugget plus half that structure's
//! sill is read off the cone's curve. Under geometric anisotropy those lags
//! trace an ellipsoid with the axes and ratios of the range ellipsoid, so the
//! ellipsoid closest to them gives the axes and their ratios — far steadier
//! than a range fitted per direction, whose sill and range trade off against
//! each other. One structure fitted to every direction's curve, each stretched
//! onto the major axis, then sets the major range.

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use rayon::prelude::*;

use boitata_core::{angles_from_axes, rotation_matrix};

use crate::aniso::Angles;
use crate::empirical::{Estimator, Experimental, LagBins, Moments, add, finalize, moments, scale};
use crate::error::{Result, VarioError};
use crate::fit::Weighting;
use crate::fit::fit;
use crate::model::Model;
use crate::surface::{PairHistogram, azimuth_dip, check_tolerance, populated};

/// Sweep chunks for the lag grid; fixed, so the merge order and the result do
/// not depend on the thread count.
const GRID_CHUNKS: usize = 8;

/// Cells from the origin to the edge of the lag grid, per axis. Keeps each
/// chunk's private grid near ten megabytes.
const MAX_HALF_CELLS: usize = 25;

/// Fewest fitted directions an ellipsoid (six unknowns) is solved from.
const MIN_DIRECTIONS: usize = 9;

/// Resolution and pair selection for a [`variogram_volume`].
#[derive(Debug, Clone)]
pub struct VolumeParams {
    /// `lag_width` is the cell size of the lag grid; `max_lag` its half-extent
    /// and the longest lag used in the directional fits.
    pub bins: LagBins,
    pub estimator: Estimator,
    /// Half-angle of the cone about each direction, in degrees.
    pub tolerance: f64,
    /// Directions read, spread over the lower half-sphere.
    pub directions: usize,
    /// Structure fitted to the omnidirectional curve.
    pub model: Model,
    pub weighting: Weighting,
}

impl Default for VolumeParams {
    fn default() -> Self {
        Self {
            bins: LagBins::default(),
            estimator: Estimator::Matheron,
            tolerance: 20.0,
            directions: 200,
            model: Model::Spherical,
            weighting: Weighting::ByCount,
        }
    }
}

/// γ over a cube of lag vectors, and the principal axes of continuity.
#[derive(Debug, Clone)]
pub struct VariogramVolume {
    /// Cell centers along each axis, symmetric about 0.
    pub lags: Vec<f64>,
    /// γ per cell, indexed `[ix][iy][iz]` (x slowest); NaN where a cell holds
    /// no pairs or the correlogram is undefined.
    pub gammas: Vec<f64>,
    /// Pair counts, laid out like `gammas`.
    pub counts: Vec<usize>,
    /// Directions read, as (azimuth, dip) in degrees, dip ≥ 0.
    pub directions: Vec<(f64, f64)>,
    /// Range per direction; `None` where the cone held too few pairs or γ
    /// stayed below half the sill within `max_lag`.
    pub ranges: Vec<Option<f64>>,
    /// Principal axes: azimuth and dip of the major axis, rake of the
    /// semi-major, and the major, semi-major and minor ranges.
    pub angles: Angles,
}

/// Near-uniform unit vectors over the lower half-sphere (dip ≥ 0), in
/// (East, North, Up). γ is symmetric under a half-turn, so these cover every
/// direction.
pub fn fibonacci_directions(n: usize) -> Vec<(f64, f64, f64)> {
    let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    (0..n)
        .map(|k| {
            let up = -(k as f64 + 0.5) / n as f64;
            let r = (1.0 - up * up).sqrt();
            let phi = golden * k as f64;
            (r * phi.cos(), r * phi.sin(), up)
        })
        .collect()
}

/// γ on a cube of lag vectors, and the principal axes of continuity.
pub fn variogram_volume(
    locations: &[(f64, f64, f64)],
    values: &[f64],
    params: &VolumeParams,
) -> Result<VariogramVolume> {
    if params.directions < MIN_DIRECTIONS {
        return Err(VarioError::InvalidParameters(format!(
            "directions must be at least {MIN_DIRECTIONS}"
        )));
    }
    let cos_tol = check_tolerance(params.tolerance)?;
    let scale = scale(values, params.estimator, false)?;
    let hist = PairHistogram::build(locations, values, &params.bins)?;
    let (lags, gammas, counts) = lag_grid(locations, values, &params.bins, params.estimator)?;

    let lag_centers = hist.lag_centers();
    let omni = hist.cone((0.0, 0.0, -1.0), -1.0, params.estimator, scale);
    let model = fit_model(populated(&lag_centers, &omni), params)?;
    let target = model.nugget + 0.5 * model.structures[0].sill;
    let units = fibonacci_directions(params.directions);
    let (curves, crossings): (Vec<_>, Vec<_>) = units
        .par_iter()
        .map(|&u| {
            let curve = populated(
                &lag_centers,
                &hist.cone(u, cos_tol, params.estimator, scale),
            );
            let crossing = curve
                .as_ref()
                .and_then(|c| crossing(c, model.nugget, target));
            (curve, crossing)
        })
        .unzip();
    let shape = principal_axes(&units, &crossings)?;

    // Pool every cone's curve on the major axis, each lag stretched by the
    // shape's major radius over its radius in that direction, and fit one
    // structure to the pool: its range is the major range.
    let to_major = rotation_matrix(shape.azimuth, shape.dip, shape.rake);
    let mut pool: Vec<(f64, f64, usize)> = Vec::new();
    for (&(x, y, z), curve) in units.iter().zip(&curves) {
        let Some(curve) = curve else { continue };
        let v = to_major * Vector3::new(x, y, z);
        let stretch = (v.x.powi(2)
            + (v.y * shape.major / shape.semi).powi(2)
            + (v.z * shape.major / shape.minor).powi(2))
        .sqrt();
        for ((&h, &g), &n) in curve.lags.iter().zip(&curve.gammas).zip(&curve.counts) {
            pool.push((h * stretch, g, n));
        }
    }
    pool.sort_by(|a, b| a.0.total_cmp(&b.0));
    let pooled = Experimental {
        lags: pool.iter().map(|p| p.0).collect(),
        gammas: pool.iter().map(|p| p.1).collect(),
        counts: pool.iter().map(|p| p.2).collect(),
        covariances: None,
    };
    let major = fit_model(Some(pooled), params)?.structures[0].range;
    let k = major / shape.major;
    let angles = Angles {
        major,
        semi: shape.semi * k,
        minor: shape.minor * k,
        ..shape
    };
    let ranges = crossings.iter().map(|c| c.map(|c| c * k)).collect();
    let directions = units.iter().map(|&u| azimuth_dip(u)).collect();
    Ok(VariogramVolume {
        lags,
        gammas,
        counts,
        directions,
        ranges,
        angles,
    })
}

fn fit_model(exp: Option<Experimental>, params: &VolumeParams) -> Result<crate::Variogram> {
    exp.and_then(|exp| fit(&exp, params.model, params.weighting).ok())
        .map(|f| f.variogram)
        .ok_or_else(|| VarioError::FittingFailed("too few pairs to fit a structure".into()))
}

/// First lag where a curve's γ reaches `target`, interpolated linearly from
/// the previous bin (from `(0, nugget)` before the first one).
fn crossing(exp: &Experimental, nugget: f64, target: f64) -> Option<f64> {
    let (mut h0, mut g0) = (0.0, nugget.min(target));
    for (&h, &g) in exp.lags.iter().zip(&exp.gammas) {
        if g >= target {
            let t = if g > g0 {
                (target - g0) / (g - g0)
            } else {
                1.0
            };
            return Some(h0 + t * (h - h0));
        }
        (h0, g0) = (h, g);
    }
    None
}

type Grid = (Vec<f64>, Vec<f64>, Vec<usize>);

fn lag_grid(
    locations: &[(f64, f64, f64)],
    values: &[f64],
    bins: &LagBins,
    estimator: Estimator,
) -> Result<Grid> {
    let w = bins.lag_width;
    let half = (bins.max_lag / w).ceil() as usize;
    if half > MAX_HALF_CELLS {
        return Err(VarioError::InvalidParameters(format!(
            "max_lag / lag_width must not exceed {MAX_HALF_CELLS}"
        )));
    }
    let side = 2 * half + 1;
    let cells = side * side * side;
    let h = half as f64;
    let cell = |d: (f64, f64, f64)| -> Option<usize> {
        let (i, j, k) = ((d.0 / w).round(), (d.1 / w).round(), (d.2 / w).round());
        if i.abs() > h || j.abs() > h || k.abs() > h {
            return None;
        }
        let (i, j, k) = ((i + h) as usize, (j + h) as usize, (k + h) as usize);
        Some((i * side + j) * side + k)
    };
    let n = locations.len();
    let partials: Vec<(Vec<Moments>, Vec<usize>)> = (0..GRID_CHUNKS)
        .into_par_iter()
        .map(|chunk| {
            let mut sums = vec![[0.0f64; 9]; cells];
            let mut counts = vec![0usize; cells];
            for i in (chunk..n).step_by(GRID_CHUNKS) {
                let pi = locations[i];
                for j in (i + 1)..n {
                    let pj = locations[j];
                    let d = (pj.0 - pi.0, pj.1 - pi.1, pj.2 - pi.2);
                    if d == (0.0, 0.0, 0.0) {
                        continue;
                    }
                    let Some(c) = cell(d) else { continue };
                    let m = cell((-d.0, -d.1, -d.2)).expect("the grid is symmetric");
                    add(&mut sums[c], &moments(values[i], values[j]));
                    add(&mut sums[m], &moments(values[j], values[i]));
                    counts[c] += 1;
                    counts[m] += 1;
                }
            }
            (sums, counts)
        })
        .collect();
    let mut sums = vec![[0.0f64; 9]; cells];
    let mut counts = vec![0usize; cells];
    for (chunk_sums, chunk_counts) in &partials {
        for c in 0..cells {
            add(&mut sums[c], &chunk_sums[c]);
            counts[c] += chunk_counts[c];
        }
    }
    let scale = scale(values, estimator, false)?;
    let gammas = sums
        .iter()
        .zip(&counts)
        .map(|(s, &count)| finalize(s, count, estimator, scale).map_or(f64::NAN, |(g, _)| g))
        .collect();
    let lags = (0..side).map(|i| (i as f64 - h) * w).collect();
    Ok((lags, gammas, counts))
}

/// The ellipsoid `uᵀ M u = 1 / r²` closest to the fitted ranges, in relative
/// terms (each row is scaled by `r²`, so long and short ranges weigh alike),
/// read as azimuth, dip, rake and three ranges.
pub fn principal_axes(units: &[(f64, f64, f64)], ranges: &[Option<f64>]) -> Result<Angles> {
    let rows: Vec<[f64; 6]> = units
        .iter()
        .zip(ranges)
        .filter_map(|(&(x, y, z), r)| {
            let r2 = r.filter(|r| r.is_finite() && *r > 0.0)?.powi(2);
            Some([x * x, y * y, z * z, 2.0 * x * y, 2.0 * x * z, 2.0 * y * z].map(|t| t * r2))
        })
        .collect();
    if rows.len() < MIN_DIRECTIONS {
        return Err(VarioError::InsufficientData(format!(
            "only {} directions had enough pairs for a range; widen tolerance or max_lag",
            rows.len()
        )));
    }
    let a = DMatrix::from_fn(rows.len(), 6, |i, j| rows[i][j]);
    let b = DVector::from_element(rows.len(), 1.0);
    let q = a
        .svd(true, true)
        .solve(&b, 1e-12)
        .map_err(|e| VarioError::FittingFailed(e.into()))?;
    let m = Matrix3::new(q[0], q[3], q[4], q[3], q[1], q[5], q[4], q[5], q[2]);
    let eig = m.symmetric_eigen();
    let mut order = [0, 1, 2];
    order.sort_by(|&i, &j| eig.eigenvalues[i].total_cmp(&eig.eigenvalues[j]));
    if eig.eigenvalues[order[0]] <= 0.0 {
        return Err(VarioError::FittingFailed(
            "the fitted ranges do not bound an ellipsoid; widen tolerance or max_lag".into(),
        ));
    }
    let axis = |k: usize| -> Vector3<f64> { eig.eigenvectors.column(order[k]).into_owned() };
    let mut major = axis(0);
    if major.z > 0.0 {
        major = -major;
    }
    let semi = axis(1);
    let [azimuth, dip, mut rake] = angles_from_axes(major.into(), semi.into());
    if rake >= 180.0 {
        rake -= 180.0;
    }
    let range = |k: usize| 1.0 / eig.eigenvalues[order[k]].sqrt();
    Ok(Angles {
        azimuth,
        dip,
        rake,
        major: range(0),
        semi: range(1),
        minor: range(2),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ellipsoid_ranges(units: &[(f64, f64, f64)], angles: &Angles) -> Vec<Option<f64>> {
        let r = rotation_matrix(angles.azimuth, angles.dip, angles.rake);
        units
            .iter()
            .map(|&(x, y, z)| {
                let v = r * Vector3::new(x, y, z);
                let s = (v.x / angles.major).powi(2)
                    + (v.y / angles.semi).powi(2)
                    + (v.z / angles.minor).powi(2);
                Some(1.0 / s.sqrt())
            })
            .collect()
    }

    #[test]
    fn principal_axes_invert_an_exact_ellipsoid() {
        let truth = Angles {
            azimuth: 40.0,
            dip: 30.0,
            rake: 20.0,
            major: 120.0,
            semi: 60.0,
            minor: 20.0,
        };
        let units = fibonacci_directions(100);
        let found = principal_axes(&units, &ellipsoid_ranges(&units, &truth)).unwrap();
        for (a, b) in [
            (found.azimuth, truth.azimuth),
            (found.dip, truth.dip),
            (found.rake, truth.rake),
            (found.major, truth.major),
            (found.semi, truth.semi),
            (found.minor, truth.minor),
        ] {
            assert!((a - b).abs() < 1e-6, "{found:?}");
        }
    }

    #[test]
    fn fibonacci_directions_are_unit_and_point_down() {
        let units = fibonacci_directions(50);
        assert_eq!(units.len(), 50);
        for (x, y, z) in units {
            assert!(((x * x + y * y + z * z).sqrt() - 1.0).abs() < 1e-12);
            assert!(z < 0.0);
        }
    }

    #[test]
    fn lag_grid_is_symmetric_through_the_origin() {
        let locations: Vec<_> = (0..40)
            .map(|i| {
                let t = i as f64;
                ((t * 7.3) % 50.0, (t * 3.1) % 40.0, (t * 1.7) % 20.0)
            })
            .collect();
        let values: Vec<f64> = (0..40).map(|i| (i as f64 * 0.37).sin()).collect();
        let bins = LagBins {
            max_lag: 30.0,
            lag_width: 10.0,
        };
        let (lags, gammas, counts) =
            lag_grid(&locations, &values, &bins, Estimator::Matheron).unwrap();
        let n = lags.len();
        assert_eq!(n, 7);
        assert_eq!(lags[3], 0.0);
        for c in 0..n * n * n {
            let m = n * n * n - 1 - c;
            assert_eq!(counts[c], counts[m]);
            assert!(gammas[c] == gammas[m] || (gammas[c].is_nan() && gammas[m].is_nan()));
        }
    }

    #[test]
    fn rejects_too_few_directions_and_too_fine_a_grid() {
        let locations = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)];
        let values = vec![1.0, 2.0, 3.0];
        let few = VolumeParams {
            directions: 3,
            ..Default::default()
        };
        assert!(variogram_volume(&locations, &values, &few).is_err());
        let fine = VolumeParams {
            bins: LagBins {
                max_lag: 100.0,
                lag_width: 1.0,
            },
            ..Default::default()
        };
        assert!(variogram_volume(&locations, &values, &fine).is_err());
    }
}
