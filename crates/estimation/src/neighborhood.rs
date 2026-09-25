//! Neighborhood statistics for points / block-model cells.
//!
//! Attributes local geostatistical metrics to any query location from a set of
//! samples (drillhole composites, point data): how isolated the location is
//! (distance to the nearest sample, mean distance to its `k` nearest) and the
//! local grade context (mean, variance, extrema, count within a radius). Useful
//! for confidence/classification passes, data-spacing maps, and QA — no
//! variogram required.

use crate::Sample;
use variogram::aniso::{Anisotropy, euclidean};

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

#[cfg(test)]
mod tests {
    use super::*;

    fn s(x: f64, y: f64, v: f64) -> Sample {
        Sample {
            loc: (x, y, 0.0),
            value: v,
            hole: None,
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
