//! Neighborhood statistics for points / block-model cells.
//!
//! Attributes local geostatistical metrics to any query location from a set of
//! samples (drillhole composites, point data): how isolated the location is
//! (distance to the nearest sample, mean distance to its `k` nearest) and the
//! local grade context (mean, variance, extrema, count within a radius). Useful
//! for confidence/classification passes, data-spacing maps, and QA — no
//! variogram required.

use rayon::prelude::*;
use variogram::aniso::{Anisotropy, euclidean};

use crate::Sample;
use crate::error::{EstimError, Result};

type Point = (f64, f64, f64);

/// Local statistics around a query location.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeighborhoodStats {
    /// Number of samples within `radius` (all samples if `radius` is infinite).
    pub n_within: usize,
    /// Number of distinct drillholes among the samples within `radius`. Samples
    /// with no hole tag each count as their own hole, so untagged point data
    /// degrades to `n_within`.
    pub n_holes: usize,
    /// Distance to the single nearest sample (`INF` if none).
    pub nearest_dist: f64,
    /// Mean distance to the `k` nearest samples (`INF` if none).
    pub mean_dist_knn: f64,
    /// Max distance among the `k` nearest (data-spacing proxy).
    pub max_dist_knn: f64,
    /// Mean of the `k` nearest sample values.
    pub value_mean: f64,
    /// Population variance of the `k` nearest sample values.
    pub value_var: f64,
    pub value_min: f64,
    pub value_max: f64,
    /// Inverse-distance-weighted mean of the `k` nearest (power 2), a smooth
    /// local level indicator.
    pub value_idw: f64,
}

impl NeighborhoodStats {
    fn empty() -> Self {
        Self {
            n_within: 0,
            n_holes: 0,
            nearest_dist: f64::INFINITY,
            mean_dist_knn: f64::INFINITY,
            max_dist_knn: f64::INFINITY,
            value_mean: f64::NAN,
            value_var: f64::NAN,
            value_min: f64::NAN,
            value_max: f64::NAN,
            value_idw: f64::NAN,
        }
    }
}

/// Compute neighborhood statistics at `target`.
///
/// `k` bounds how many nearest samples feed the value/distance summaries;
/// `radius` bounds `n_within` (use `f64::INFINITY` for no cap); `anisotropy`
/// (optional) sets the distance metric.
pub fn neighborhood_stats(
    target: &(f64, f64, f64),
    samples: &[Sample],
    k: usize,
    radius: f64,
    anisotropy: Option<&Anisotropy>,
) -> NeighborhoodStats {
    if samples.is_empty() || k == 0 {
        return NeighborhoodStats::empty();
    }
    let dist = |s: &Sample| match anisotropy {
        Some(a) => a.lag(target, &s.loc),
        None => euclidean(target, &s.loc),
    };
    let mut d: Vec<(f64, f64, Option<u32>)> =
        samples.iter().map(|s| (dist(s), s.value, s.hole)).collect();
    d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let n_within = d.iter().filter(|(dd, _, _)| *dd <= radius).count();
    let mut holes = std::collections::HashSet::new();
    let n_holes = d
        .iter()
        .filter(|(dd, _, _)| *dd <= radius)
        .filter(|(_, _, hole)| match hole {
            Some(h) => holes.insert(*h),
            None => true, // untagged samples each count as their own hole
        })
        .count();
    let nearest_dist = d[0].0;

    let kk = k.min(d.len());
    let knn = &d[..kk];
    let mean_dist_knn = knn.iter().map(|(dd, _, _)| dd).sum::<f64>() / kk as f64;
    let max_dist_knn = knn.last().unwrap().0;

    let vals: Vec<f64> = knn.iter().map(|(_, v, _)| *v).collect();
    let value_mean = vals.iter().sum::<f64>() / kk as f64;
    let value_var = vals.iter().map(|v| (v - value_mean).powi(2)).sum::<f64>() / kk as f64;
    let value_min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let value_max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // IDW(power 2) over the k nearest (exact hit → that value).
    let value_idw = {
        let mut num = 0.0;
        let mut den = 0.0;
        let mut exact = None;
        for &(dd, v, _) in knn {
            if dd < 1e-12 {
                exact = Some(v);
                break;
            }
            let w = 1.0 / (dd * dd);
            num += w * v;
            den += w;
        }
        exact.unwrap_or(if den > 0.0 { num / den } else { value_mean })
    };

    NeighborhoodStats {
        n_within,
        n_holes,
        nearest_dist,
        mean_dist_knn,
        max_dist_knn,
        value_mean,
        value_var,
        value_min,
        value_max,
        value_idw,
    }
}

/// Mean distance from each target to its nearest drill holes, one column per
/// entry of `n`: each hole is at the distance of its nearest sample, and the
/// `n` nearest holes are averaged. Samples beyond `radius` are ignored, and a
/// target with fewer than `n` holes in reach gets `INF`. Distances follow
/// `anisotropy` when given.
pub fn hole_distance(
    targets: &[Point],
    locs: &[Point],
    holes: &[u32],
    n: &[usize],
    radius: f64,
    anisotropy: Option<&Anisotropy>,
) -> Result<Vec<Vec<f64>>> {
    let invalid = |m: &str| Err(EstimError::InvalidParameters(m.into()));
    if locs.len() != holes.len() {
        return invalid("one hole per sample");
    }
    if n.contains(&0) {
        return invalid("n must be at least 1");
    }
    if radius.is_nan() || radius <= 0.0 {
        return invalid("radius must be positive");
    }
    let mut ids = holes.to_vec();
    ids.sort_unstable();
    ids.dedup();
    let dense: Vec<usize> = holes
        .iter()
        .map(|h| ids.binary_search(h).unwrap_or_default())
        .collect();
    let most = n.iter().copied().max().unwrap_or(0);
    let rows: Vec<Vec<f64>> = targets
        .par_iter()
        .map(|t| {
            let mut nearest = vec![f64::INFINITY; ids.len()];
            for (loc, &h) in locs.iter().zip(&dense) {
                let d = match anisotropy {
                    Some(a) => a.lag(t, loc),
                    None => euclidean(t, loc),
                };
                if d <= radius && d < nearest[h] {
                    nearest[h] = d;
                }
            }
            nearest.retain(|d| d.is_finite());
            if nearest.len() > most {
                nearest.select_nth_unstable_by(most, f64::total_cmp);
                nearest.truncate(most);
            }
            nearest.sort_unstable_by(f64::total_cmp);
            n.iter()
                .map(|&k| match nearest.get(..k) {
                    Some(d) => d.iter().sum::<f64>() / k as f64,
                    None => f64::INFINITY,
                })
                .collect()
        })
        .collect();
    Ok((0..n.len())
        .map(|j| rows.iter().map(|r| r[j]).collect())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use variogram::aniso::Angles;

    fn hd(
        targets: &[Point],
        locs: &[Point],
        holes: &[u32],
        n: &[usize],
        radius: f64,
    ) -> Vec<Vec<f64>> {
        hole_distance(targets, locs, holes, n, radius, None).unwrap()
    }

    const O: Point = (0.0, 0.0, 0.0);

    #[test]
    fn a_target_on_samples_from_n_holes_is_at_distance_zero() {
        let locs = [O, O, O, (9.0, 0.0, 0.0)];
        let d = hd(&[O], &locs, &[1, 2, 3, 4], &[3, 4], f64::INFINITY);
        assert_eq!(d[0][0], 0.0);
        assert!((d[1][0] - 2.25).abs() < 1e-12);
    }

    #[test]
    fn a_hole_counts_once_at_its_nearest_sample() {
        let locs = [
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
        ];
        let d = hd(&[O], &locs, &[7, 7, 7, 8], &[2, 3], f64::INFINITY);
        assert!((d[0][0] - 3.0).abs() < 1e-12);
        assert!(d[1][0].is_infinite());
        let split = hd(&[O], &locs, &[7, 7, 8, 8], &[2], f64::INFINITY);
        assert!((split[0][0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn radius_drops_far_holes_and_distance_grows_with_n() {
        let locs = [
            (1.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 4.0),
            (30.0, 0.0, 0.0),
        ];
        let d = hd(&[O], &locs, &[1, 2, 3, 4], &[1, 2, 3, 4], 10.0);
        assert!(d[0][0] <= d[1][0] && d[1][0] <= d[2][0]);
        assert!(d[3][0].is_infinite());
    }

    #[test]
    fn anisotropy_shortens_distance_along_the_major_axis() {
        let a = Anisotropy::new(Angles {
            azimuth: 0.0,
            dip: 0.0,
            rake: 0.0,
            major: 1.0,
            semi: 0.5,
            minor: 0.5,
        })
        .unwrap();
        let locs = [(0.0, 10.0, 0.0), (10.0, 0.0, 0.0)];
        let d = hole_distance(&[O], &locs, &[1, 2], &[1, 2], 100.0, Some(&a)).unwrap();
        assert!((d[0][0] - 10.0).abs() < 1e-9);
        assert!((d[1][0] - 15.0).abs() < 1e-9);
    }

    #[test]
    fn hole_distance_is_the_same_for_any_thread_count() {
        let locs: Vec<Point> = (0..300)
            .map(|i| ((i * 37 % 101) as f64, (i * 53 % 97) as f64, (i % 7) as f64))
            .collect();
        let holes: Vec<u32> = (0..300).map(|i| i / 5).collect();
        let targets: Vec<Point> = (0..200)
            .map(|i| ((i % 20) as f64 * 5.0, (i / 20) as f64 * 10.0, 3.0))
            .collect();
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| hd(&targets, &locs, &holes, &[1, 3, 5], 60.0))
        };
        let one = run(1);
        assert_eq!(one, run(4));
        assert!(one.iter().flatten().any(|d| d.is_finite()));
    }

    #[test]
    fn hole_distance_rejects_bad_input() {
        let locs = [O];
        assert!(hole_distance(&[], &locs, &[], &[1], 1.0, None).is_err());
        assert!(hole_distance(&[], &locs, &[1], &[0], 1.0, None).is_err());
        assert!(hole_distance(&[], &locs, &[1], &[1], f64::NAN, None).is_err());
    }

    fn s(x: f64, y: f64, v: f64) -> Sample {
        Sample {
            loc: (x, y, 0.0),
            value: v,
            hole: None,
            error_variance: 0.0,
            domain: None,
        }
    }

    #[test]
    fn basic_distances_and_stats() {
        let samples = vec![s(3.0, 4.0, 10.0), s(6.0, 8.0, 20.0), s(0.0, 0.0, 30.0)];
        let st = neighborhood_stats(&(0.0, 0.0, 0.0), &samples, 3, f64::INFINITY, None);
        assert_eq!(st.n_within, 3);
        assert!((st.nearest_dist - 0.0).abs() < 1e-9); // sample at origin
        assert!((st.value_min - 10.0).abs() < 1e-9);
        assert!((st.value_max - 30.0).abs() < 1e-9);
        assert!((st.value_mean - 20.0).abs() < 1e-9);
        // Exact hit at origin → IDW returns that value.
        assert!((st.value_idw - 30.0).abs() < 1e-9);
    }

    #[test]
    fn radius_limits_count_but_not_knn() {
        let samples = vec![s(1.0, 0.0, 1.0), s(2.0, 0.0, 2.0), s(100.0, 0.0, 3.0)];
        let st = neighborhood_stats(&(0.0, 0.0, 0.0), &samples, 3, 10.0, None);
        assert_eq!(st.n_within, 2); // only two within radius 10
        assert_eq!(3, samples.len()); // but knn still summarizes k=3
        assert!((st.max_dist_knn - 100.0).abs() < 1e-9);
    }

    #[test]
    fn n_holes_counts_distinct_tagged_holes_and_untagged_singletons() {
        let h = |x: f64, v: f64, hole: u32| Sample {
            loc: (x, 0.0, 0.0),
            value: v,
            hole: Some(hole),
            error_variance: 0.0,
            domain: None,
        };
        // Two samples of hole 1 and one of hole 2 within radius; hole 3 outside.
        let samples = vec![
            h(1.0, 1.0, 1),
            h(2.0, 2.0, 1),
            h(3.0, 3.0, 2),
            h(50.0, 4.0, 3),
        ];
        let st = neighborhood_stats(&(0.0, 0.0, 0.0), &samples, 4, 10.0, None);
        assert_eq!(st.n_within, 3);
        assert_eq!(st.n_holes, 2);

        // Untagged samples each count as their own hole → n_holes == n_within.
        let untagged = vec![s(1.0, 0.0, 1.0), s(2.0, 0.0, 2.0)];
        let st = neighborhood_stats(&(0.0, 0.0, 0.0), &untagged, 2, 10.0, None);
        assert_eq!(st.n_holes, st.n_within);
    }

    #[test]
    fn empty_is_safe() {
        let st = neighborhood_stats(&(0.0, 0.0, 0.0), &[], 5, 10.0, None);
        assert_eq!(st.n_within, 0);
        assert!(st.nearest_dist.is_infinite());
    }
}
