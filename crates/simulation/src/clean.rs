//! Cleaning categorical realizations by maximum a-posteriori selection
//! (MAPS; Deutsch 1998, Computers & Geosciences 24(6):551-562).
//!
//! Each node takes the category with the largest weighted count in a window
//! around it: closer nodes weigh more, nodes on a datum weigh `data_weight`
//! times more, and each category is scaled by its target proportion over its
//! proportion in the realization, which pulls proportions toward the target.
//! Every node is scored from the original realization, so the result does
//! not depend on the order of the nodes or on the threads.

use rayon::prelude::*;

use crate::error::{Result, SimError};
use crate::lattice::Lattice;

/// The realization `cats` over `lattice` cleaned in a window of `half`
/// nodes on each side of a node along each axis. `targets` are the target
/// proportions of the `k` categories, one row for every node or one per
/// node (`local`). Nodes whose centroid is in `data_locs` keep their
/// category and weigh `data_weight` (at least 1) in their neighbors'
/// windows. A tie keeps the node's category.
#[allow(clippy::too_many_arguments)]
pub fn maps(
    lattice: &Lattice,
    cats: &[usize],
    k: usize,
    half: [usize; 3],
    targets: &[f64],
    local: Option<&[Vec<f64>]>,
    data_locs: &[(f64, f64, f64)],
    data_weight: f64,
) -> Result<Vec<usize>> {
    let n = lattice.len();
    let invalid = |m: &str| Err(SimError::InvalidParameters(m.into()));
    if cats.len() != n {
        return invalid("one category per node");
    }
    if cats.iter().any(|&c| c >= k) {
        return invalid("a category outside 0..k");
    }
    if targets.len() != k
        || local.is_some_and(|rows| rows.len() != n || rows.iter().any(|r| r.len() != k))
    {
        return invalid("one target proportion per category");
    }
    if !(data_weight >= 1.0 && data_weight.is_finite()) {
        return invalid("data_weight must be at least 1");
    }
    let mut on_datum = vec![false; n];
    for (m, _) in crate::shared::datum_nodes(lattice, data_locs, None) {
        on_datum[m] = true;
    }
    let mut share = vec![0.0; k];
    for &c in cats {
        share[c] += 1.0 / n as f64;
    }
    let reach = half.map(|h| h as i32);
    let mut window = vec![];
    for dz in -reach[2]..=reach[2] {
        for dy in -reach[1]..=reach[1] {
            for dx in -reach[0]..=reach[0] {
                let d = [dx, dy, dz];
                let r = (0..3)
                    .map(|a| (d[a] as f64 / (half[a] + 1) as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if r < 1.0 {
                    window.push((d, 1.0 - r));
                }
            }
        }
    }
    Ok((0..n)
        .into_par_iter()
        .map(|m| {
            if on_datum[m] {
                return cats[m];
            }
            let target = local.map_or(targets, |rows| &rows[m]);
            let mut score = vec![0.0; k];
            for &(d, w) in &window {
                if let Some(j) = lattice.shifted(m, d) {
                    let c = cats[j];
                    score[c] += w * if on_datum[j] { data_weight } else { 1.0 };
                }
            }
            for (c, s) in score.iter_mut().enumerate() {
                *s *= if share[c] > 0.0 {
                    target[c] / share[c]
                } else {
                    0.0
                };
            }
            let best = score.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            match score[cats[m]] >= best {
                true => cats[m],
                false => score.iter().position(|&s| s >= best).unwrap_or(cats[m]),
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use boitata_core::Geometry;

    fn square(n: usize) -> Lattice {
        Lattice::regular(Geometry {
            origin: [0.0; 3],
            size: [1.0, 1.0, 1.0],
            count: [n, n, 1],
            rotation: [0.0; 3],
        })
    }

    #[test]
    fn an_isolated_node_joins_its_neighbors_and_a_datum_stays() {
        let l = square(9);
        let mut cats = vec![0; 81];
        cats[40] = 1;
        cats[10] = 1;
        let data = [(1.5, 1.5, 0.5)];
        let own = [79.0 / 81.0, 2.0 / 81.0];
        let out = maps(&l, &cats, 2, [1, 1, 0], &own, None, &data, 5.0).unwrap();
        assert_eq!(out[40], 0);
        assert_eq!(out[10], 1);
    }

    #[test]
    fn noise_drops_and_a_uniform_realization_is_kept() {
        let l = square(40);
        let cats: Vec<usize> = (0..1600)
            .map(|m| {
                let (x, y) = (m % 40, m / 40);
                usize::from(x >= 20) ^ usize::from((x * 7 + y * 13) % 11 == 0)
            })
            .collect();
        let flips = |c: &[usize]| {
            (0..1600)
                .filter(|&m| m % 40 != 39 && c[m] != c[m + 1])
                .count()
        };
        let out = maps(&l, &cats, 2, [2, 2, 0], &[0.5, 0.5], None, &[], 1.0).unwrap();
        assert!(
            flips(&out) < flips(&cats) / 3,
            "{} vs {}",
            flips(&out),
            flips(&cats)
        );
        let same = vec![1; 1600];
        assert_eq!(
            maps(&l, &same, 2, [2, 2, 0], &[0.5, 0.5], None, &[], 1.0).unwrap(),
            same
        );
    }

    #[test]
    fn the_proportion_term_pulls_toward_the_target() {
        let l = square(40);
        let cats: Vec<usize> = (0..1600)
            .map(|m| usize::from((m * 37 + m / 40 * 11) % 10 < 3))
            .collect();
        let share = |c: &[usize]| c.iter().filter(|&&c| c == 1).count() as f64 / 1600.0;
        let low = maps(&l, &cats, 2, [1, 1, 0], &[0.9, 0.1], None, &[], 1.0).unwrap();
        let high = maps(&l, &cats, 2, [1, 1, 0], &[0.5, 0.5], None, &[], 1.0).unwrap();
        assert!(
            share(&low) < share(&high),
            "{} vs {}",
            share(&low),
            share(&high)
        );
    }

    #[test]
    fn the_result_does_not_depend_on_the_threads() {
        let l = square(30);
        let cats: Vec<usize> = (0..900).map(|m| (m * 7919 % 13) % 3).collect();
        let with = |t: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
                .install(|| {
                    maps(&l, &cats, 3, [2, 2, 0], &[0.3, 0.3, 0.4], None, &[], 1.0).unwrap()
                })
        };
        assert_eq!(with(1), with(4));
    }
}
