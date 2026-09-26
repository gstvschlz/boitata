//! Despiking: breaks ties in data values before a normal-score transform.
//!
//! Tied samples are ordered by the local average of their neighborhoods at
//! increasing radii, then by a seeded random draw. The averaged quantity is each
//! sample's mid-rank fraction, averaged over the variables, so several variables
//! share one ordering. Tied values then get tiny increasing offsets, so the
//! order statistics are kept.

use std::cmp::Ordering;

use kiddo::{ImmutableKdTree, SquaredEuclidean};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::error::{Result, TransformError};

/// Share of the gap to the next distinct value spanned by the offsets of a tie.
const SPREAD: f64 = 1e-4;

/// Untied copies of `columns` (one `Vec` per variable, one value per sample at
/// `coords`). Tied samples are ranked on the local average of the mean rank over
/// `radii` (in increasing order), then on a random draw from `seed`.
pub fn despike(
    coords: &[(f64, f64, f64)],
    columns: &[Vec<f64>],
    radii: &[f64],
    seed: u64,
) -> Result<Vec<Vec<f64>>> {
    let n = coords.len();
    let invalid = |m: &str| TransformError::InvalidParameters(m.into());
    if n == 0 || columns.is_empty() {
        return Err(TransformError::InsufficientData("no samples".into()));
    }
    if columns.iter().any(|c| c.len() != n) {
        return Err(invalid("every variable needs one value per sample"));
    }
    if columns.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid("values must be finite"));
    }
    if radii.iter().any(|r| !r.is_finite() || *r < 0.0) {
        return Err(invalid("radii must be finite and non-negative"));
    }
    let mut radii = radii.to_vec();
    radii.sort_by(f64::total_cmp);

    let orders: Vec<Vec<usize>> = columns.iter().map(|c| sorted(c)).collect();
    let mut tied = vec![false; n];
    let mut score = vec![0.0; n];
    for (c, order) in columns.iter().zip(&orders) {
        for group in groups(c, order) {
            let mid = (group.start + group.end - 1) as f64 / 2.0 / n as f64;
            for &i in &order[group.clone()] {
                score[i] += mid / columns.len() as f64;
                tied[i] |= group.len() > 1;
            }
        }
    }

    let points: Vec<[f64; 3]> = coords.iter().map(|&(x, y, z)| [x, y, z]).collect();
    let tree = ImmutableKdTree::<f64, 3>::new_from_slice(&points)
        .map_err(|e| TransformError::InvalidParameters(format!("{e:?}")))?;
    let mut rng = StdRng::seed_from_u64(seed);
    let draws: Vec<f64> = (0..n).map(|_| rng.r#gen()).collect();
    let keys: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            if !tied[i] {
                return vec![];
            }
            let mut key: Vec<f64> = radii
                .iter()
                .map(|r| {
                    let mut near: Vec<usize> = tree
                        .query(&points[i])
                        .within::<SquaredEuclidean<f64>>(r * r * (1.0 + 1e-12))
                        .execute()
                        .iter()
                        .map(|f| f.item as usize)
                        .collect();
                    near.sort_unstable();
                    near.iter().map(|&j| score[j]).sum::<f64>() / near.len() as f64
                })
                .collect();
            key.push(draws[i]);
            key
        })
        .collect();

    columns
        .iter()
        .zip(&orders)
        .map(|(c, order)| {
            let mut out = c.clone();
            let bounds: Vec<_> = groups(c, order).collect();
            for (g, group) in bounds.iter().enumerate() {
                if group.len() < 2 {
                    continue;
                }
                let v = c[order[group.start]];
                let gap = match (g.checked_sub(1), bounds.get(g + 1)) {
                    (_, Some(next)) => c[order[next.start]] - v,
                    (Some(prev), None) => v - c[order[bounds[prev].start]],
                    (None, None) => v.abs().max(1.0),
                };
                let mut members = order[group.clone()].to_vec();
                members.sort_by(|&a, &b| compare(&keys[a], &keys[b]).then(a.cmp(&b)));
                let m = members.len() as f64;
                for (k, &i) in members.iter().enumerate() {
                    out[i] = v + SPREAD * gap * k as f64 / m;
                }
            }
            let mut check = out.clone();
            check.sort_by(f64::total_cmp);
            if check.windows(2).any(|w| w[0] >= w[1]) {
                return Err(invalid("too many ties to separate in floating point"));
            }
            Ok(out)
        })
        .collect()
}

/// Radii for [`despike`]: 1, 2, 4 and 8 times the median distance from each
/// sample to its nearest distinct neighbor.
pub fn default_radii(coords: &[(f64, f64, f64)]) -> Vec<f64> {
    let points: Vec<[f64; 3]> = coords.iter().map(|&(x, y, z)| [x, y, z]).collect();
    let Ok(tree) = ImmutableKdTree::<f64, 3>::new_from_slice(&points) else {
        return vec![];
    };
    let k = std::num::NonZero::new(points.len().min(8)).expect("n >= 1");
    let mut spacing: Vec<f64> = points
        .par_iter()
        .filter_map(|p| {
            tree.query(p)
                .nearest_n::<SquaredEuclidean<f64>>(k)
                .execute()
                .iter()
                .map(|f| f.distance.sqrt())
                .find(|d| *d > 0.0)
        })
        .collect();
    if spacing.is_empty() {
        return vec![];
    }
    spacing.sort_by(f64::total_cmp);
    let median = spacing[spacing.len() / 2];
    [1.0, 2.0, 4.0, 8.0].iter().map(|f| f * median).collect()
}

fn sorted(values: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    order
}

/// Ranges of `order` holding equal values.
fn groups<'a>(
    values: &'a [f64],
    order: &'a [usize],
) -> impl Iterator<Item = std::ops::Range<usize>> + 'a {
    let mut start = 0;
    std::iter::from_fn(move || {
        if start == order.len() {
            return None;
        }
        let v = values[order[start]];
        let end = start + order[start..].partition_point(|&i| values[i] == v);
        let range = start..end;
        start = end;
        Some(range)
    })
}

fn compare(a: &[f64], b: &[f64]) -> Ordering {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.total_cmp(y))
        .find(|o| o.is_ne())
        .unwrap_or(Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Samples on a line; the right half is high-valued, with ties at the
    /// detection limit 0.1 spread over both halves and ties at 5.0.
    fn data() -> (Vec<(f64, f64, f64)>, Vec<f64>) {
        let mut rng = StdRng::seed_from_u64(1);
        let coords: Vec<_> = (0..200).map(|i| (i as f64, 0.0, 0.0)).collect();
        let values = (0..200)
            .map(|i| match i % 7 {
                0 => 0.1,
                3 if i >= 100 && i % 2 == 0 => 5.0,
                _ if i < 100 => rng.gen_range(0.2..1.0),
                _ => rng.gen_range(2.0..4.0),
            })
            .collect();
        (coords, values)
    }

    fn rank(values: &[f64]) -> Vec<usize> {
        let mut r = vec![0; values.len()];
        for (k, i) in sorted(values).into_iter().enumerate() {
            r[i] = k;
        }
        r
    }

    #[test]
    fn breaks_ties_and_keeps_the_order_of_untied_values() {
        let (coords, values) = data();
        let out = despike(&coords, std::slice::from_ref(&values), &[3.0, 10.0], 7).unwrap();
        let out = &out[0];
        let mut s = out.clone();
        s.sort_by(f64::total_cmp);
        assert!(s.windows(2).all(|w| w[0] < w[1]));
        for i in 0..200 {
            assert!((out[i] - values[i]).abs() <= SPREAD * 5.0);
            for j in 0..200 {
                if values[i] < values[j] {
                    assert!(out[i] < out[j]);
                }
            }
        }
    }

    #[test]
    fn tied_samples_in_high_neighborhoods_rank_higher() {
        let (coords, values) = data();
        let out = &despike(&coords, std::slice::from_ref(&values), &[3.0], 7).unwrap()[0];
        let low = (0..100).filter(|&i| values[i] == 0.1).map(|i| out[i]);
        let high = (100..200).filter(|&i| values[i] == 0.1).map(|i| out[i]);
        assert!(low.fold(f64::MIN, f64::max) < high.fold(f64::MAX, f64::min));
    }

    #[test]
    fn ties_break_the_same_way_across_variables() {
        let (coords, values) = data();
        let other: Vec<f64> = values.iter().map(|v| v * 10.0).collect();
        let out = despike(&coords, &[values.clone(), other], &[3.0, 10.0], 7).unwrap();
        assert_eq!(rank(&out[0]), rank(&out[1]));
    }

    #[test]
    fn deterministic_for_a_seed_and_any_thread_count() {
        let (coords, values) = data();
        let columns = [values];
        let run = |threads: usize, seed| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| despike(&coords, &columns, &[], seed).unwrap())
        };
        assert_eq!(run(1, 3), run(4, 3));
        assert_ne!(run(1, 3), run(1, 4));
    }

    #[test]
    fn default_radii_follow_the_spacing() {
        let (coords, _) = data();
        assert_eq!(default_radii(&coords), vec![1.0, 2.0, 4.0, 8.0]);
    }

    #[test]
    fn rejects_bad_input() {
        let c = [(0.0, 0.0, 0.0)];
        assert!(despike(&c, &[vec![f64::NAN]], &[], 0).is_err());
        assert!(despike(&c, &[vec![1.0, 2.0]], &[], 0).is_err());
        assert!(despike(&c, &[vec![1.0]], &[-1.0], 0).is_err());
        assert!(despike(&[], &[vec![]], &[], 0).is_err());
    }
}
