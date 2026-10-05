//! Scenario reduction: representative realizations by k-medoids.

use boitata_core::rng::realization_seed;
use rayon::prelude::*;

use crate::error::{Result, SimError};

/// Picks `n` representative rows of `realizations` (one row per realization)
/// by k-medoids on the Euclidean distance between rows. Targets that are not
/// finite in every realization are left out. A seeded k-medoids++ start is
/// refined by PAM swaps until no swap lowers the total distance. Returns the
/// medoids by cluster size, largest first, ties by index; identical for any
/// thread count.
pub fn select_realizations(realizations: &[Vec<f64>], n: usize, seed: u64) -> Result<Vec<usize>> {
    let m = realizations.len();
    if n == 0 || n > m {
        return Err(SimError::InvalidParameters(format!(
            "n must be in 1..={m}, got {n}"
        )));
    }
    let width = realizations[0].len();
    if realizations.iter().any(|r| r.len() != width) {
        return Err(SimError::InvalidParameters(
            "realizations differ in length".into(),
        ));
    }
    let columns: Vec<usize> = (0..width)
        .filter(|&t| realizations.iter().all(|r| r[t].is_finite()))
        .collect();
    if columns.is_empty() {
        return Err(SimError::InsufficientData(
            "no target is finite in every realization".into(),
        ));
    }
    let d: Vec<f64> = (0..m)
        .into_par_iter()
        .flat_map_iter(|i| {
            let (a, columns) = (&realizations[i], &columns);
            realizations.iter().map(move |b| {
                columns
                    .iter()
                    .map(|&t| (a[t] - b[t]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
        })
        .collect();
    let mut medoids = start(&d, m, n, seed);
    swap(&d, m, &mut medoids);
    let mut size = vec![0usize; n];
    for j in 0..m {
        size[nearest(&d, m, &medoids, j).0] += 1;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(size[i]), medoids[i]));
    Ok(order.into_iter().map(|i| medoids[i]).collect())
}

fn unit(seed: u64, step: u64) -> f64 {
    (realization_seed(seed, step) >> 11) as f64 / (1u64 << 53) as f64
}

/// k-medoids++: the first medoid uniform, each next drawn with probability
/// proportional to its squared distance to the nearest medoid so far.
fn start(d: &[f64], m: usize, n: usize, seed: u64) -> Vec<usize> {
    let mut medoids = vec![((unit(seed, 0) * m as f64) as usize).min(m - 1)];
    let mut near: Vec<f64> = (0..m).map(|j| d[medoids[0] * m + j].powi(2)).collect();
    while medoids.len() < n {
        let total: f64 = near.iter().sum();
        let pick = if total > 0.0 {
            let mut u = unit(seed, medoids.len() as u64) * total;
            (0..m)
                .filter(|&j| near[j] > 0.0)
                .find(|&j| {
                    u -= near[j];
                    u < 0.0
                })
                .unwrap_or_else(|| (0..m).rev().find(|&j| near[j] > 0.0).expect("total > 0"))
        } else {
            (0..m).find(|j| !medoids.contains(j)).expect("n <= m")
        };
        medoids.push(pick);
        for (j, v) in near.iter_mut().enumerate() {
            *v = v.min(d[pick * m + j].powi(2));
        }
    }
    medoids
}

/// Position in `medoids` of the medoid nearest to `j` (ties to the lower
/// realization index), with the distance.
fn nearest(d: &[f64], m: usize, medoids: &[usize], j: usize) -> (usize, f64) {
    medoids
        .iter()
        .enumerate()
        .map(|(i, &c)| (i, d[c * m + j]))
        .min_by(|a, b| a.1.total_cmp(&b.1).then(medoids[a.0].cmp(&medoids[b.0])))
        .expect("at least one medoid")
}

/// PAM swap: applies the best medoid/non-medoid exchange while it lowers the
/// total distance to the nearest medoid.
fn swap(d: &[f64], m: usize, medoids: &mut [usize]) {
    let k = medoids.len();
    loop {
        let (mut near, mut first, mut second) = (vec![0; m], vec![0.0; m], vec![0.0; m]);
        for j in 0..m {
            let (i, dn) = nearest(d, m, medoids, j);
            near[j] = i;
            first[j] = dn;
            second[j] = (0..k)
                .filter(|&o| o != i)
                .map(|o| d[medoids[o] * m + j])
                .fold(f64::INFINITY, f64::min);
        }
        let cost: f64 = first.iter().sum();
        let best = (0..m)
            .into_par_iter()
            .filter(|h| !medoids.contains(h))
            .map(|h| {
                let mut common = 0.0;
                let mut own = vec![0.0; k];
                for j in 0..m {
                    let dj = d[h * m + j];
                    let shared = (dj - first[j]).min(0.0);
                    common += shared;
                    own[near[j]] += dj.min(second[j]) - first[j] - shared;
                }
                (0..k)
                    .map(|i| (common + own[i], h, i))
                    .min_by(|a, b| a.0.total_cmp(&b.0))
                    .expect("k > 0")
            })
            .min_by(|a, b| a.0.total_cmp(&b.0).then((a.1, a.2).cmp(&(b.1, b.2))));
        match best {
            Some((delta, h, i)) if delta < -1e-12 * cost => medoids[i] = h,
            _ => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `per` realizations around each of `centers`, interleaved, with small
    /// seeded noise on `width` targets.
    fn clusters(centers: &[f64], per: usize, width: usize) -> Vec<Vec<f64>> {
        (0..per * centers.len())
            .map(|r| {
                let c = centers[r % centers.len()];
                (0..width)
                    .map(|t| c + 0.1 * (unit(r as u64, t as u64) - 0.5))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn separated_clusters_give_one_medoid_each() {
        let reals = clusters(&[0.0, 10.0, 25.0], 7, 30);
        for seed in 0..10 {
            let picked = select_realizations(&reals, 3, seed).unwrap();
            let mut groups: Vec<usize> = picked.iter().map(|&p| p % 3).collect();
            groups.sort_unstable();
            assert_eq!(groups, [0, 1, 2], "seed {seed}");
        }
    }

    #[test]
    fn larger_clusters_come_first_and_nan_targets_are_ignored() {
        let mut reals = clusters(&[0.0, 10.0], 5, 12);
        reals.extend(clusters(&[0.5], 4, 12));
        reals[3][4] = f64::NAN;
        let picked = select_realizations(&reals, 2, 1).unwrap();
        assert!(reals[picked[0]][0] < 5.0 && reals[picked[1]][0] > 5.0);
        let all = select_realizations(&reals, reals.len(), 0).unwrap();
        assert_eq!(all, (0..reals.len()).collect::<Vec<_>>());
    }

    #[test]
    fn the_result_is_fixed_by_the_seed_on_any_thread_count() {
        let reals: Vec<Vec<f64>> = (0..60)
            .map(|r| (0..40).map(|t| unit(r, t)).collect())
            .collect();
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| select_realizations(&reals, 6, 3).unwrap())
        };
        assert_eq!(run(1), run(4));
        assert_eq!(run(1), select_realizations(&reals, 6, 3).unwrap());
    }

    #[test]
    fn bad_input_is_an_error() {
        let reals = vec![vec![1.0, f64::NAN], vec![2.0, 3.0]];
        assert!(select_realizations(&reals, 0, 0).is_err());
        assert!(select_realizations(&reals, 3, 0).is_err());
        assert!(select_realizations(&[vec![1.0], vec![]], 1, 0).is_err());
        assert!(select_realizations(&[vec![f64::NAN], vec![1.0]], 1, 0).is_err());
        assert!(select_realizations(&reals, 1, 0).is_ok());
    }
}
