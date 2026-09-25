//! Sequential Gaussian Simulation (SGS).
//!
//! Produces conditional realizations of a continuous variable:
//! 1. normal-score transform the conditioning data,
//! 2. visit grid nodes along a random path,
//! 3. at each node, simple-krige (mean 0) from nearby data + previously simulated
//!    nodes to get a conditional mean/variance, then draw from that Gaussian,
//! 4. add the drawn value to the conditioning set and continue,
//! 5. back-transform all simulated scores to data space.

use crate::error::{Result, SimError};
use estimation::Sample;
use estimation::krige::{Kind, krige};
use estimation::search::{Search, neighbors};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use transforms::nscore;
use variogram::Variogram;

/// SGS parameters.
#[derive(Debug, Clone)]
pub struct SgsParams {
    pub search: Search,
    /// RNG seed for a reproducible realization.
    pub seed: u64,
}

/// One conditional realization over the simulation grid.
#[derive(Debug, Clone)]
pub struct Realization {
    /// Simulated values in data space, one per grid node (same order as `grid`).
    pub values: Vec<f64>,
}

/// Run a single SGS realization.
///
/// `vg_nscore` is the variogram of the *normal scores* (unit-sill Gaussian variogram);
/// `data_weights` are declustering weights for the normal-score transform.
pub fn sgs(
    data_locs: &[(f64, f64, f64)],
    data_vals: &[f64],
    data_weights: Option<&[f64]>,
    grid: &[(f64, f64, f64)],
    vg_nscore: &Variogram,
    params: &SgsParams,
) -> Result<Realization> {
    if data_locs.len() != data_vals.len() {
        return Err(SimError::InvalidParameters("data length mismatch".into()));
    }
    if data_locs.is_empty() {
        return Err(SimError::InsufficientData("no conditioning data".into()));
    }
    if grid.is_empty() {
        return Ok(Realization { values: vec![] });
    }

    // 1. Normal-score transform.
    let ns = nscore::transform(data_vals, data_weights)
        .map_err(|e| SimError::Transform(e.to_string()))?;

    // Conditioning set (grows as nodes are simulated).
    let mut cond: Vec<Sample> = data_locs
        .iter()
        .zip(&ns.scores)
        .map(|(&loc, &score)| Sample {
            loc,
            value: score,
            hole: None,
        })
        .collect();

    // 2. Random path over grid nodes.
    let mut rng = StdRng::seed_from_u64(params.seed);
    let mut path: Vec<usize> = (0..grid.len()).collect();
    path.shuffle(&mut rng);

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut sim_scores = vec![f64::NAN; grid.len()];

    for &node in &path {
        let target = grid[node];

        // 3. Kriging from neighbors (simple kriging, mean 0 in Gaussian space).
        let found = neighbors(&target, &cond, &params.search, Some(vg_nscore));
        if let Some(&k) = found.iter().flatten().find(|&&k| cond[k].loc == target) {
            // A node on a datum takes its value and is not added again.
            sim_scores[node] = cond[k].value;
            continue;
        }
        let score = match found {
            Ok(idx) if !idx.is_empty() => {
                let selected: Vec<Sample> = idx.iter().map(|&k| cond[k].clone()).collect();
                let est = krige(Kind::Simple { mean: 0.0 }, &target, &selected, vg_nscore)
                    .map_err(|e| SimError::Estimation(e.to_string()))?;
                let sd = est.variance.max(0.0).sqrt();
                est.value + sd * normal.sample(&mut rng)
            }
            // No neighbors found: draw from the marginal (standard normal).
            _ => normal.sample(&mut rng),
        };

        sim_scores[node] = score;
        // 4. Add simulated node to the conditioning set.
        cond.push(Sample {
            loc: target,
            value: score,
            hole: None,
        });
    }

    // 5. Back-transform to data space.
    let values: Vec<f64> = sim_scores.iter().map(|&s| ns.table.back(s)).collect();
    Ok(Realization { values })
}

/// Run `n` independent realizations (seeds `seed, seed+1, …`) in parallel; the
/// result does not depend on the number of threads.
pub fn sgs_ensemble(
    data_locs: &[(f64, f64, f64)],
    data_vals: &[f64],
    data_weights: Option<&[f64]>,
    grid: &[(f64, f64, f64)],
    vg_nscore: &Variogram,
    params: &SgsParams,
    n: usize,
) -> Result<Vec<Realization>> {
    (0..n)
        .into_par_iter()
        .map(|k| {
            let p = SgsParams {
                search: params.search.clone(),
                seed: params.seed.wrapping_add(k as u64),
            };
            sgs(data_locs, data_vals, data_weights, grid, vg_nscore, &p)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use variogram::model::Model;

    #[test]
    fn realization_honors_data_at_data_locations() {
        // When a grid node coincides with a datum, the simulated value should match it
        // closely (zero-nugget variogram → near-exact conditioning).
        let data_locs = vec![
            (0.0, 0.0, 0.0),
            (100.0, 0.0, 0.0),
            (0.0, 100.0, 0.0),
            (100.0, 100.0, 0.0),
        ];
        let data_vals = vec![1.0, 5.0, 3.0, 8.0];
        let grid = vec![(0.0, 0.0, 0.0), (50.0, 50.0, 0.0)]; // first coincides with a datum
        let vg = Variogram::single(Model::Spherical, 1.0, 200.0);
        let params = SgsParams {
            search: Search {
                min_samples: 1,
                max_samples: 8,
                radius: f64::INFINITY,
                max_per_hole: None,
                octant: false,
            },
            seed: 42,
        };
        let real = sgs(&data_locs, &data_vals, None, &grid, &vg, &params).unwrap();
        assert_eq!(real.values.len(), 2);
        assert!(
            (real.values[0] - 1.0).abs() < 0.5,
            "conditioned value {}",
            real.values[0]
        );
    }

    #[test]
    fn reproducible_with_same_seed() {
        let data_locs = vec![(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (0.0, 100.0, 0.0)];
        let data_vals = vec![1.0, 5.0, 3.0];
        let grid: Vec<(f64, f64, f64)> = (0..20).map(|i| (i as f64 * 5.0, 25.0, 0.0)).collect();
        let vg = Variogram::single(Model::Exponential, 1.0, 150.0);
        let params = SgsParams {
            search: Search {
                min_samples: 1,
                max_samples: 12,
                radius: f64::INFINITY,
                max_per_hole: None,
                octant: false,
            },
            seed: 7,
        };
        let a = sgs(&data_locs, &data_vals, None, &grid, &vg, &params).unwrap();
        let b = sgs(&data_locs, &data_vals, None, &grid, &vg, &params).unwrap();
        assert_eq!(a.values, b.values);
    }

    #[test]
    fn different_seeds_differ() {
        let data_locs = vec![(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (0.0, 100.0, 0.0)];
        let data_vals = vec![1.0, 5.0, 3.0];
        let grid: Vec<(f64, f64, f64)> = (0..20).map(|i| (i as f64 * 5.0, 25.0, 0.0)).collect();
        let vg = Variogram::single(Model::Exponential, 1.0, 150.0);
        let mk = |seed| SgsParams {
            search: Search {
                min_samples: 1,
                max_samples: 12,
                radius: f64::INFINITY,
                max_per_hole: None,
                octant: false,
            },
            seed,
        };
        let a = sgs(&data_locs, &data_vals, None, &grid, &vg, &mk(1)).unwrap();
        let b = sgs(&data_locs, &data_vals, None, &grid, &vg, &mk(2)).unwrap();
        assert_ne!(a.values, b.values);
    }

    #[test]
    fn nodes_on_data_take_the_data_value() {
        let data_locs = vec![(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)];
        let data_vals = vec![1.0, 5.0, 3.0];
        let grid = vec![(10.0, 0.0, 0.0), (5.0, 0.0, 0.0), (15.0, 0.0, 0.0)];
        let vg = Variogram::single(Model::Spherical, 1.0, 30.0);
        let params = SgsParams {
            search: Search {
                min_samples: 1,
                max_samples: 8,
                radius: 50.0,
                max_per_hole: None,
                octant: false,
            },
            seed: 4,
        };
        let r = sgs(&data_locs, &data_vals, None, &grid, &vg, &params).unwrap();
        assert!((r.values[0] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn ensemble_does_not_depend_on_thread_count() {
        let data_locs: Vec<_> = (0..30)
            .map(|i| ((i * 7 % 50) as f64, (i * 11 % 50) as f64, 0.0))
            .collect();
        let data_vals: Vec<f64> = (0..30).map(|i| (i as f64).sqrt()).collect();
        let grid: Vec<_> = (0..100)
            .map(|i| ((i % 10) as f64 * 5.0, (i / 10) as f64 * 5.0, 0.0))
            .collect();
        let vg = Variogram::single(Model::Spherical, 1.0, 20.0);
        let params = SgsParams {
            search: Search {
                min_samples: 1,
                max_samples: 12,
                radius: 30.0,
                max_per_hole: None,
                octant: false,
            },
            seed: 9,
        };
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    sgs_ensemble(&data_locs, &data_vals, None, &grid, &vg, &params, 6).unwrap()
                })
        };
        let (one, many) = (run(1), run(6));
        for (a, b) in one.iter().zip(&many) {
            assert_eq!(a.values, b.values);
        }
    }

    #[test]
    fn ensemble_mean_approximates_kriging() {
        // The average of many SGS realizations should approach the kriging estimate.
        let data_locs = vec![
            (0.0, 0.0, 0.0),
            (100.0, 0.0, 0.0),
            (0.0, 100.0, 0.0),
            (100.0, 100.0, 0.0),
        ];
        let data_vals = vec![2.0, 2.0, 2.0, 2.0]; // constant field → mean should be ~2
        let grid = vec![(50.0, 50.0, 0.0)];
        let vg = Variogram::single(Model::Spherical, 1.0, 200.0);
        let params = SgsParams {
            search: Search {
                min_samples: 1,
                max_samples: 8,
                radius: f64::INFINITY,
                max_per_hole: None,
                octant: false,
            },
            seed: 100,
        };
        let ens = sgs_ensemble(&data_locs, &data_vals, None, &grid, &vg, &params, 50).unwrap();
        let mean = ens.iter().map(|r| r.values[0]).sum::<f64>() / ens.len() as f64;
        assert!((mean - 2.0).abs() < 0.2, "ensemble mean {mean}");
    }
}
