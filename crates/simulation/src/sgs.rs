//! Sequential Gaussian Simulation (SGS).
//!
//! Produces conditional realizations of a continuous variable:
//! 1. normal-score transform the conditioning data, within each domain,
//! 2. visit grid nodes along a random path,
//! 3. at each node, simple-krige (mean 0) from nearby data + previously simulated
//!    nodes to get a conditional mean/variance, then draw from that Gaussian;
//!    with several searches, a node uses the first that finds enough data,
//!    as a kriging pass would,
//! 4. back-transform the draw through the node's domain and add the node to
//!    the conditioning set,
//! 5. continue until every node is simulated.
//!
//! The search compares data values and the back-transformed values of
//! simulated nodes with a high-grade threshold, and kriging uses their scores.

use crate::error::{Result, SimError};
use estimation::Sample;
use estimation::krige::{Kind, krige};
use estimation::lva::LocalAnisotropy;
use estimation::search::{Search, SearchTree};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use transforms::{NormalScoreTable, normal_score};
use variogram::{Anisotropy, Variogram};

/// SGS parameters.
#[derive(Debug, Clone)]
pub struct SgsParams {
    /// Search passes: a node takes the first that finds `min_samples` among
    /// the data, or the last when none does.
    pub search: Vec<Search>,
    /// RNG seed for a reproducible realization.
    pub seed: u64,
}

/// One conditional realization over the simulation grid.
#[derive(Debug, Clone)]
pub struct Realization {
    /// Simulated values in data space, one per grid node (same order as `grid`).
    pub values: Vec<f64>,
}

/// Domain codes of the data and of the grid nodes.
pub type Domains<'a> = (&'a [u32], &'a [u32]);

/// Run a single SGS realization.
///
/// `vg_nscore` is the variogram of the *normal scores* (unit-sill Gaussian variogram);
/// `data_weights` are declustering weights for the normal-score transform;
/// `data_holes` tag the data by drill hole for `max_per_hole`, simulated
/// nodes belonging to none.
#[allow(clippy::too_many_arguments)]
pub fn sgs(
    data_locs: &[(f64, f64, f64)],
    data_vals: &[f64],
    data_weights: Option<&[f64]>,
    data_holes: Option<&[u32]>,
    grid: &[(f64, f64, f64)],
    vg_nscore: &Variogram,
    params: &SgsParams,
    local: Option<&LocalAnisotropy>,
) -> Result<Realization> {
    sgs_in(
        data_locs,
        data_vals,
        data_weights,
        data_holes,
        None,
        grid,
        vg_nscore,
        params,
        local,
    )
}

/// As [`sgs`] with `domains`: each domain has its own declustered
/// normal-score table, a node is back-transformed through its domain's, and
/// the search's soft boundaries apply to data and simulated nodes alike. A
/// node whose domain has no data is an error.
#[allow(clippy::too_many_arguments)]
pub fn sgs_in(
    data_locs: &[(f64, f64, f64)],
    data_vals: &[f64],
    data_weights: Option<&[f64]>,
    data_holes: Option<&[u32]>,
    domains: Option<Domains>,
    grid: &[(f64, f64, f64)],
    vg_nscore: &Variogram,
    params: &SgsParams,
    local: Option<&LocalAnisotropy>,
) -> Result<Realization> {
    simulate(
        data_locs,
        data_vals,
        data_weights,
        data_holes,
        domains,
        grid,
        vg_nscore,
        params,
        local,
        |_, _, _| {},
    )
}

/// SGS calling `used(node, neighbours, samples)` with the indices into
/// `samples` (data, then simulated nodes, valued in data units) each
/// kriged node was simulated from.
#[allow(clippy::too_many_arguments)]
fn simulate(
    data_locs: &[(f64, f64, f64)],
    data_vals: &[f64],
    data_weights: Option<&[f64]>,
    data_holes: Option<&[u32]>,
    domains: Option<Domains>,
    grid: &[(f64, f64, f64)],
    vg_nscore: &Variogram,
    params: &SgsParams,
    local: Option<&LocalAnisotropy>,
    mut used: impl FnMut(usize, &[usize], &[Sample]),
) -> Result<Realization> {
    if data_locs.len() != data_vals.len() {
        return Err(SimError::InvalidParameters("data length mismatch".into()));
    }
    if data_locs.is_empty() {
        return Err(SimError::InsufficientData("no conditioning data".into()));
    }
    if params.search.is_empty() {
        return Err(SimError::InvalidParameters(
            "need at least one search".into(),
        ));
    }
    let holes = crate::holes(data_holes, data_locs.len())?;
    check(domains, data_locs.len(), grid.len())?;
    if local.is_some_and(|l| l.len() != grid.len()) {
        return Err(SimError::InvalidParameters(
            "one local anisotropy per grid node".into(),
        ));
    }
    if grid.is_empty() {
        return Ok(Realization { values: vec![] });
    }

    // 1. Normal-score transform within each domain.
    let (mut scores, tables) = normal_scores(data_vals, data_weights, domains.map(|d| d.0))?;
    let node_domain = |node: usize| domains.map(|d| d.1[node]);

    // Conditioning set in data units (grows as nodes are simulated), with
    // the scores kriging uses alongside.
    let mut samples = data(data_locs, data_vals, holes, domains.map(|d| d.0));
    let mut trees: Vec<SearchTree> = params
        .search
        .iter()
        .map(|s| tree(&samples, s, vg_nscore, local))
        .collect();
    let last = trees.len() - 1;
    let passes: Vec<usize> = match last {
        0 => vec![0; grid.len()],
        _ => first_pass(&trees, grid, domains.map(|d| d.1), local)
            .into_iter()
            .map(|p| p.unwrap_or(last))
            .collect(),
    };

    // 2. Random path over grid nodes.
    let mut rng = StdRng::seed_from_u64(params.seed);
    let mut path: Vec<usize> = (0..grid.len()).collect();
    path.shuffle(&mut rng);

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut values = vec![f64::NAN; grid.len()];

    for &node in &path {
        let target = grid[node];
        let domain = node_domain(node);

        // 3. Kriging from neighbors (simple kriging, mean 0 in Gaussian space).
        let aniso = local.map(|l| l.anisotropy(node));
        let found = find(&trees[passes[node]], &target, domain, aniso.as_ref());
        let vg_node = aniso.map(|a| Variogram {
            anisotropy: Some(a),
            ..vg_nscore.clone()
        });
        let vg = vg_node.as_ref().unwrap_or(vg_nscore);
        if let Some(&k) = found.iter().flatten().find(|&&k| samples[k].loc == target) {
            // A node on a datum takes its value and is not added again.
            values[node] = samples[k].value;
            continue;
        }
        let score = match found {
            Ok(idx) if !idx.is_empty() => {
                used(node, &idx, &samples);
                let selected: Vec<Sample> = idx
                    .iter()
                    .map(|&k| Sample {
                        value: scores[k],
                        ..samples[k].clone()
                    })
                    .collect();
                let est = krige(Kind::Simple { mean: 0.0 }, &target, &selected, vg)
                    .map_err(|e| SimError::Estimation(e.to_string()))?;
                let sd = est.variance.max(0.0).sqrt();
                est.value + sd * normal.sample(&mut rng)
            }
            // No neighbors found: draw from the marginal (standard normal).
            _ => normal.sample(&mut rng),
        };

        // 4. Back-transform and add the node to the conditioning set.
        let value = tables[domain.unwrap_or(0) as usize]
            .as_ref()
            .expect("checked")
            .back(score);
        values[node] = value;
        let sample = Sample {
            loc: target,
            value,
            hole: None,
            error_variance: 0.0,
            domain,
        };
        for tree in &mut trees {
            tree.add(&sample);
        }
        samples.push(sample);
        scores.push(score);
    }
    Ok(Realization { values })
}

/// Lengths of `domains` and a datum in the domain of every node.
fn check(domains: Option<Domains>, data: usize, nodes: usize) -> Result<()> {
    let Some((of_data, of_nodes)) = domains else {
        return Ok(());
    };
    if of_data.len() != data {
        return Err(SimError::InvalidParameters("one domain per datum".into()));
    }
    if of_nodes.len() != nodes {
        return Err(SimError::InvalidParameters("one domain per node".into()));
    }
    let known: std::collections::HashSet<u32> = of_data.iter().copied().collect();
    match of_nodes.iter().find(|d| !known.contains(d)) {
        Some(d) => Err(SimError::InvalidParameters(format!(
            "domain {d} has no samples"
        ))),
        None => Ok(()),
    }
}

/// Normal scores of the data and the table of each domain code, one table
/// for all without `domains`.
fn normal_scores(
    values: &[f64],
    weights: Option<&[f64]>,
    domains: Option<&[u32]>,
) -> Result<(Vec<f64>, Vec<Option<NormalScoreTable>>)> {
    let transform = |rows: &[usize]| {
        let pick = |v: &[f64]| rows.iter().map(|&i| v[i]).collect::<Vec<_>>();
        normal_score::transform(&pick(values), weights.map(pick).as_deref())
            .map_err(|e| SimError::Transform(e.to_string()))
    };
    let Some(codes) = domains else {
        let ns = normal_score::transform(values, weights)
            .map_err(|e| SimError::Transform(e.to_string()))?;
        return Ok((ns.scores, vec![Some(ns.table)]));
    };
    let k = codes.iter().max().map_or(0, |&m| m as usize + 1);
    let mut rows = vec![vec![]; k];
    for (i, &c) in codes.iter().enumerate() {
        rows[c as usize].push(i);
    }
    let mut scores = vec![0.0; values.len()];
    let mut tables = Vec::with_capacity(k);
    for rows in &rows {
        if rows.is_empty() {
            tables.push(None);
            continue;
        }
        let ns = transform(rows)?;
        for (&i, s) in rows.iter().zip(ns.scores) {
            scores[i] = s;
        }
        tables.push(Some(ns.table));
    }
    Ok((scores, tables))
}

fn data(
    locs: &[(f64, f64, f64)],
    values: &[f64],
    holes: Vec<Option<u32>>,
    domains: Option<&[u32]>,
) -> Vec<Sample> {
    locs.iter()
        .zip(values)
        .zip(holes)
        .enumerate()
        .map(|(i, ((&loc, &value), hole))| Sample {
            hole,
            domain: domains.map(|d| d[i]),
            ..Sample::new(loc, value)
        })
        .collect()
}

/// The pass of every grid node: the first of `search` that finds
/// `min_samples` among the data, as in a kriging by passes; `None` where none
/// does, and SGS then uses the last. Simulated nodes play no part, so the map
/// is the same in every realization.
#[allow(clippy::too_many_arguments)]
pub fn sgs_passes(
    data_locs: &[(f64, f64, f64)],
    data_vals: &[f64],
    data_holes: Option<&[u32]>,
    domains: Option<Domains>,
    grid: &[(f64, f64, f64)],
    vg_nscore: &Variogram,
    search: &[Search],
    local: Option<&LocalAnisotropy>,
) -> Result<Vec<Option<usize>>> {
    if data_locs.len() != data_vals.len() {
        return Err(SimError::InvalidParameters("data length mismatch".into()));
    }
    if local.is_some_and(|l| l.len() != grid.len()) {
        return Err(SimError::InvalidParameters(
            "one local anisotropy per grid node".into(),
        ));
    }
    check(domains, data_locs.len(), grid.len())?;
    let holes = crate::holes(data_holes, data_locs.len())?;
    let data = data(data_locs, data_vals, holes, domains.map(|d| d.0));
    let trees: Vec<SearchTree> = search
        .iter()
        .map(|s| tree(&data, s, vg_nscore, local))
        .collect();
    Ok(first_pass(&trees, grid, domains.map(|d| d.1), local))
}

fn tree(
    samples: &[Sample],
    search: &Search,
    vg: &Variogram,
    local: Option<&LocalAnisotropy>,
) -> SearchTree {
    match local {
        Some(_) => SearchTree::new(
            samples,
            &Search {
                anisotropy: None,
                ..search.clone()
            },
            None,
        ),
        None => SearchTree::new(samples, search, Some(vg)),
    }
}

fn find(
    tree: &SearchTree,
    target: &(f64, f64, f64),
    domain: Option<u32>,
    local: Option<&Anisotropy>,
) -> estimation::Result<Vec<usize>> {
    match local {
        Some(a) => tree.neighbors_within(target, domain, a),
        None => tree.neighbors_in(target, domain),
    }
}

fn first_pass(
    trees: &[SearchTree],
    grid: &[(f64, f64, f64)],
    domains: Option<&[u32]>,
    local: Option<&LocalAnisotropy>,
) -> Vec<Option<usize>> {
    grid.par_iter()
        .enumerate()
        .map(|(i, target)| {
            let aniso = local.map(|l| l.anisotropy(i));
            let domain = domains.map(|d| d[i]);
            trees
                .iter()
                .position(|t| find(t, target, domain, aniso.as_ref()).is_ok_and(|f| !f.is_empty()))
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
            search: vec![Search {
                min_samples: 1,
                max_samples: 8,
                radius: f64::INFINITY,
                max_per_hole: None,
                octant: false,
                anisotropy: None,
                high_grade: None,
                soft: None,
            }],
            seed: 42,
        };
        let real = sgs(
            &data_locs, &data_vals, None, None, &grid, &vg, &params, None,
        )
        .unwrap();
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
            search: vec![Search {
                min_samples: 1,
                max_samples: 12,
                radius: f64::INFINITY,
                max_per_hole: None,
                octant: false,
                anisotropy: None,
                high_grade: None,
                soft: None,
            }],
            seed: 7,
        };
        let a = sgs(
            &data_locs, &data_vals, None, None, &grid, &vg, &params, None,
        )
        .unwrap();
        let b = sgs(
            &data_locs, &data_vals, None, None, &grid, &vg, &params, None,
        )
        .unwrap();
        assert_eq!(a.values, b.values);
    }

    #[test]
    fn different_seeds_differ() {
        let data_locs = vec![(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (0.0, 100.0, 0.0)];
        let data_vals = vec![1.0, 5.0, 3.0];
        let grid: Vec<(f64, f64, f64)> = (0..20).map(|i| (i as f64 * 5.0, 25.0, 0.0)).collect();
        let vg = Variogram::single(Model::Exponential, 1.0, 150.0);
        let mk = |seed| SgsParams {
            search: vec![Search {
                min_samples: 1,
                max_samples: 12,
                radius: f64::INFINITY,
                max_per_hole: None,
                octant: false,
                anisotropy: None,
                high_grade: None,
                soft: None,
            }],
            seed,
        };
        let a = sgs(&data_locs, &data_vals, None, None, &grid, &vg, &mk(1), None).unwrap();
        let b = sgs(&data_locs, &data_vals, None, None, &grid, &vg, &mk(2), None).unwrap();
        assert_ne!(a.values, b.values);
    }

    #[test]
    fn uniform_local_anisotropy_is_global_anisotropy() {
        let data_locs: Vec<_> = (0..40)
            .map(|i| {
                let jitter = (i as f64 * 0.618).fract();
                (
                    (i * 7 % 50) as f64 + jitter,
                    (i * 11 % 50) as f64 + jitter * 0.37,
                    0.0,
                )
            })
            .collect();
        let data_vals: Vec<f64> = (0..40).map(|i| (i as f64).sqrt()).collect();
        let grid: Vec<_> = (0..100)
            .map(|i| {
                let jitter = (i as f64 * 0.414).fract();
                (
                    (i % 10) as f64 * 5.0 + jitter,
                    (i / 10) as f64 * 5.0 + jitter * 0.71,
                    0.0,
                )
            })
            .collect();
        let local = LocalAnisotropy::new(
            grid.clone(),
            vec![[30.0, 0.0, 0.0]; 100],
            vec![[0.4, 1.0]; 100],
        )
        .unwrap();
        let base = Variogram::single(Model::Spherical, 1.0, 20.0);
        let global = Variogram {
            anisotropy: Some(local.anisotropy(0)),
            ..base.clone()
        };
        let params = SgsParams {
            search: vec![Search {
                min_samples: 1,
                max_samples: 200,
                radius: 12.0,
                ..Default::default()
            }],
            seed: 3,
        };
        let a = sgs(
            &data_locs,
            &data_vals,
            None,
            None,
            &grid,
            &base,
            &params,
            Some(&local),
        )
        .unwrap();
        let b = sgs(
            &data_locs, &data_vals, None, None, &grid, &global, &params, None,
        )
        .unwrap();
        for (x, y) in a.values.iter().zip(&b.values) {
            assert!((x - y).abs() < 1e-9, "{x} vs {y}");
        }
    }

    #[test]
    fn nodes_on_data_take_the_data_value() {
        let data_locs = vec![(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)];
        let data_vals = vec![1.0, 5.0, 3.0];
        let grid = vec![(10.0, 0.0, 0.0), (5.0, 0.0, 0.0), (15.0, 0.0, 0.0)];
        let vg = Variogram::single(Model::Spherical, 1.0, 30.0);
        let params = SgsParams {
            search: vec![Search {
                min_samples: 1,
                max_samples: 8,
                radius: 50.0,
                max_per_hole: None,
                octant: false,
                anisotropy: None,
                high_grade: None,
                soft: None,
            }],
            seed: 4,
        };
        let r = sgs(
            &data_locs, &data_vals, None, None, &grid, &vg, &params, None,
        )
        .unwrap();
        assert!((r.values[0] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn a_planar_grid_among_3d_data_simulates_quickly() {
        let data_locs: Vec<_> = (0..300)
            .map(|i| ((i * 37 % 101) as f64, (i * 53 % 97) as f64, (i % 7) as f64))
            .collect();
        let data_vals: Vec<f64> = (0..300).map(|i| (i * 29 % 23) as f64).collect();
        let grid: Vec<_> = (0..10_000)
            .map(|i| ((i % 100) as f64, (i / 100) as f64, 3.5))
            .collect();
        let vg = Variogram::single(Model::Spherical, 1.0, 30.0);
        let params = SgsParams {
            search: vec![Search {
                min_samples: 1,
                max_samples: 16,
                radius: 40.0,
                ..Default::default()
            }],
            seed: 5,
        };
        let start = std::time::Instant::now();
        sgs(
            &data_locs, &data_vals, None, None, &grid, &vg, &params, None,
        )
        .unwrap();
        let elapsed = start.elapsed().as_secs_f64();
        assert!(elapsed < 10.0, "{elapsed} s");
    }

    fn summary(
        data_locs: &[(f64, f64, f64)],
        data_vals: &[f64],
        grid: &[(f64, f64, f64)],
        vg: &Variogram,
        n: usize,
    ) -> crate::ContinuousSummary {
        let search = Search {
            min_samples: 1,
            max_samples: 8,
            radius: f64::INFINITY,
            ..Default::default()
        };
        crate::continuous(n, &Default::default(), |k| {
            let params = SgsParams {
                search: vec![search.clone()],
                seed: 100 + k as u64,
            };
            sgs(data_locs, data_vals, None, None, grid, vg, &params, None).map(|r| r.values)
        })
        .unwrap()
    }

    #[test]
    fn ensemble_mean_approximates_kriging() {
        let data_locs = vec![
            (0.0, 0.0, 0.0),
            (100.0, 0.0, 0.0),
            (0.0, 100.0, 0.0),
            (100.0, 100.0, 0.0),
        ];
        let data_vals = vec![2.0, 2.0, 2.0, 2.0];
        let vg = Variogram::single(Model::Spherical, 1.0, 200.0);
        let s = summary(&data_locs, &data_vals, &[(50.0, 50.0, 0.0)], &vg, 50);
        assert!((s.mean[0] - 2.0).abs() < 0.2, "ensemble mean {}", s.mean[0]);
    }

    #[test]
    fn far_from_data_the_ensemble_reproduces_the_histogram() {
        let data_locs: Vec<_> = (0..20)
            .map(|i| ((i * 7 % 20) as f64, (i * 3 % 20) as f64, 0.0))
            .collect();
        let data_vals: Vec<f64> = (1..=20).map(f64::from).collect();
        let vg = Variogram::single(Model::Spherical, 1.0, 10.0);
        let s = summary(&data_locs, &data_vals, &[(1e4, 1e4, 0.0)], &vg, 400);
        assert!((s.mean[0] - 10.5).abs() < 0.8, "mean {}", s.mean[0]);
        assert!(
            (s.variance[0] / 33.25 - 1.0).abs() < 0.25,
            "variance {}",
            s.variance[0]
        );
    }

    #[test]
    fn max_per_hole_caps_the_data_of_one_hole() {
        let locs: Vec<_> = (0..10).map(|i| (0.0, 0.0, i as f64)).collect();
        let vals: Vec<f64> = (0..10).map(|i| (i * 7 % 10) as f64).collect();
        let holes = vec![0; 10];
        let grid = vec![(3.0, 0.0, 4.4)];
        let vg = Variogram::single(Model::Spherical, 1.0, 20.0);
        let run = |holes: Option<&[u32]>, max_samples, max_per_hole| {
            let search = Search {
                min_samples: 1,
                max_samples,
                radius: f64::INFINITY,
                max_per_hole,
                ..Default::default()
            };
            let params = SgsParams {
                search: vec![search],
                seed: 2,
            };
            sgs(&locs, &vals, None, holes, &grid, &vg, &params, None)
                .unwrap()
                .values
        };
        let capped = run(Some(&holes), 8, Some(1));
        assert_eq!(capped, run(None, 1, None));
        assert_ne!(capped, run(None, 8, None));
        assert_eq!(run(Some(&holes), 8, None), run(None, 8, None));
    }

    /// Clustered holes of 5 samples down z, the grid around them in 2D.
    fn passes_case() -> (
        Vec<(f64, f64, f64)>,
        Vec<f64>,
        Vec<(f64, f64, f64)>,
        Vec<Search>,
    ) {
        let locs: Vec<_> = (0..150)
            .map(|i| {
                let h = i / 5;
                let (x, y) = if h < 20 {
                    ((h * 7 % 20) as f64 * 1.3, (h * 3 % 20) as f64 * 1.1)
                } else {
                    (
                        40.0 + (h * 13 % 10) as f64 * 5.0,
                        50.0 + (h % 10) as f64 * 3.0,
                    )
                };
                (x, y, (i % 5) as f64)
            })
            .collect();
        let vals: Vec<f64> = (0..150).map(|i| ((i * 29 % 23) as f64).powf(1.5)).collect();
        let grid: Vec<_> = (0..40 * 40)
            .map(|i| ((i % 40) as f64 * 2.5, (i / 40) as f64 * 2.5, 2.0))
            .collect();
        let pass = |radius, min_samples| Search {
            min_samples,
            max_samples: 12,
            radius,
            max_per_hole: Some(3),
            high_grade: Some(estimation::HighGrade {
                threshold: 60.0,
                radius: 6.0,
            }),
            ..Default::default()
        };
        (
            locs,
            vals,
            grid,
            vec![pass(10.0, 8), pass(25.0, 4), pass(40.0, 2)],
        )
    }

    #[test]
    fn pass_map_is_the_kriging_pass_map() {
        let (locs, vals, grid, search) = passes_case();
        let vg = Variogram::single(Model::Spherical, 1.0, 30.0);
        let data: Vec<Sample> = locs
            .iter()
            .zip(&vals)
            .enumerate()
            .map(|(i, (&l, &v))| Sample::with_hole(l, v, (i / 5) as u32))
            .collect();
        let kriged = estimation::by_pass(grid.len(), &search, |s, remaining| {
            let at: Vec<_> = remaining.iter().map(|&i| grid[i]).collect();
            Ok(estimation::estimate_many(
                &at,
                None,
                &data,
                s,
                Some(&vg),
                |t, n| krige(Kind::Ordinary, t, n, &vg),
            ))
        })
        .unwrap();
        let want: Vec<_> = kriged.iter().map(|k| k.as_ref().map(|k| k.0)).collect();
        let holes: Vec<u32> = (0..150).map(|i| i / 5).collect();
        let got = sgs_passes(&locs, &vals, Some(&holes), None, &grid, &vg, &search, None).unwrap();
        assert_eq!(got, want);
        for p in [Some(0), Some(1), Some(2), None] {
            assert!(got.contains(&p), "no node in pass {p:?}");
        }
    }

    #[test]
    fn passes_never_taken_change_nothing() {
        let (locs, vals, grid, _) = passes_case();
        let vg = Variogram::single(Model::Spherical, 1.0, 30.0);
        let wide = Search {
            min_samples: 1,
            max_samples: 12,
            radius: f64::INFINITY,
            ..Default::default()
        };
        let run = |search: Vec<Search>| {
            let params = SgsParams { search, seed: 9 };
            sgs(&locs, &vals, None, None, &grid, &vg, &params, None)
                .unwrap()
                .values
        };
        let one = run(vec![wide.clone()]);
        let two = run(vec![
            wide.clone(),
            Search {
                radius: 5.0,
                ..wide
            },
        ]);
        assert_eq!(one, two);
    }

    #[test]
    fn passes_follow_the_seed_not_the_thread_count() {
        let (locs, vals, grid, search) = passes_case();
        let vg = Variogram::single(Model::Spherical, 1.0, 30.0);
        let run = |threads, seed| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let params = SgsParams {
                        search: search.clone(),
                        seed,
                    };
                    sgs(&locs, &vals, None, None, &grid, &vg, &params, None)
                        .unwrap()
                        .values
                })
        };
        let a = run(1, 3);
        assert_eq!(a, run(4, 3));
        assert_eq!(a, run(1, 3));
        assert_ne!(a, run(4, 4));
    }

    type Point = (f64, f64, f64);

    struct Zoned {
        locs: Vec<Point>,
        vals: Vec<f64>,
        weights: Vec<f64>,
        holes: Vec<u32>,
        codes: Vec<u32>,
    }

    /// Domain 0 west of x = 50, domain 1 east and richer, both richer to the
    /// north where the weights are higher; holes of 3 samples down z, and
    /// holes on the contact logging both domains at every sample.
    fn zoned() -> Zoned {
        use rand::Rng;
        let mut rng = StdRng::seed_from_u64(8);
        let mut z = Zoned {
            locs: vec![],
            vals: vec![],
            weights: vec![],
            holes: vec![],
            codes: vec![],
        };
        for h in 0..60 {
            let x = if h < 40 { 0.0 } else { 50.0 } + rng.r#gen::<f64>() * 49.0;
            let y: f64 = rng.r#gen::<f64>() * 100.0;
            let code = u32::from(x >= 50.0);
            for k in 0..3 {
                let g: f64 = rng.sample(rand_distr::StandardNormal);
                z.locs.push((x, y, k as f64));
                z.vals
                    .push((1.5 * f64::from(code) + y / 50.0 + 0.5 * g).exp());
                z.weights.push(if y > 50.0 { 3.0 } else { 1.0 });
                z.holes.push(h);
                z.codes.push(code);
            }
        }
        for h in 0..5 {
            for code in 0..2 {
                z.locs.push((50.0, 10.0 + 20.0 * h as f64, 1.0));
                z.vals.push(if code == 0 { 0.5 } else { 20.0 });
                z.weights.push(1.0);
                z.holes.push(100 + h);
                z.codes.push(code);
            }
        }
        z
    }

    /// Nodes on a 4 m grid at z = 1 and on the contact holes, each in both
    /// domains.
    fn zoned_grid() -> (Vec<Point>, Vec<u32>) {
        let mut grid: Vec<Point> = (0..25 * 25)
            .map(|i| {
                (
                    (i % 25) as f64 * 4.0 + 2.0,
                    (i / 25) as f64 * 4.0 + 2.0,
                    1.0,
                )
            })
            .collect();
        let mut codes: Vec<u32> = grid.iter().map(|p| u32::from(p.0 >= 50.0)).collect();
        for h in 0..5 {
            for code in 0..2 {
                grid.push((50.0, 10.0 + 20.0 * h as f64, 1.0));
                codes.push(code);
            }
        }
        (grid, codes)
    }

    fn zoned_search(soft: Option<estimation::Soft>) -> Vec<Search> {
        let pass = |radius, min_samples| Search {
            min_samples,
            max_samples: 12,
            radius,
            max_per_hole: Some(2),
            high_grade: Some(estimation::HighGrade {
                threshold: 12.0,
                radius: 6.0,
            }),
            soft: soft.clone(),
            ..Default::default()
        };
        vec![pass(12.0, 6), pass(40.0, 2)]
    }

    fn distance(a: &Point, b: &Point) -> f64 {
        ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2)).sqrt()
    }

    /// Neighbours of the other domain used; asserts every neighbour obeys the
    /// high-grade rule in data units and is in the node's domain or within
    /// `soft`.
    fn other_domain_used(soft: Option<f64>, seed: u64) -> usize {
        let z = zoned();
        let (grid, nodes) = zoned_grid();
        let search = zoned_search(soft.map(estimation::Soft::All));
        let vg = Variogram::single(Model::Spherical, 1.0, 25.0);
        let params = SgsParams { search, seed };
        let (mut other, mut high, mut simulated) = (0, 0, 0);
        simulate(
            &z.locs,
            &z.vals,
            Some(&z.weights),
            Some(&z.holes),
            Some((&z.codes, &nodes)),
            &grid,
            &vg,
            &params,
            None,
            |node, idx, samples| {
                for &k in idx {
                    let (s, d) = (&samples[k], distance(&grid[node], &samples[k].loc));
                    if s.value > 12.0 {
                        assert!(d <= 6.0, "high grade {} at {d}", s.value);
                        high += 1;
                    }
                    if s.domain != Some(nodes[node]) {
                        assert!(d < soft.unwrap_or(0.0), "other domain at {d}");
                        other += 1;
                    }
                    simulated += usize::from(k >= z.locs.len());
                }
            },
        )
        .unwrap();
        assert!(high > 0 && simulated > 0);
        other
    }

    #[test]
    fn hard_boundaries_use_only_the_node_domain() {
        assert_eq!(other_domain_used(None, 1), 0);
    }

    #[test]
    fn soft_boundaries_reach_strictly_within_the_distance() {
        assert!(other_domain_used(Some(8.0), 2) > 0);
        assert!(other_domain_used(Some(20.0), 3) > 0);
    }

    #[test]
    fn one_label_everywhere_is_no_domains() {
        let z = zoned();
        let (grid, _) = zoned_grid();
        let vg = Variogram::single(Model::Spherical, 1.0, 25.0);
        let search = zoned_search(Some(estimation::Soft::All(5.0)));
        // Without the contact holes, which put two samples at one location.
        let n = z.locs.len() - 10;
        let (one, all) = (vec![0; n], vec![0; grid.len()]);
        let domains = Some((&one[..], &all[..]));
        for seed in 0..3 {
            let params = SgsParams {
                search: search.clone(),
                seed,
            };
            let run = |domains| {
                sgs_in(
                    &z.locs[..n],
                    &z.vals[..n],
                    Some(&z.weights[..n]),
                    Some(&z.holes[..n]),
                    domains,
                    &grid,
                    &vg,
                    &params,
                    None,
                )
                .unwrap()
                .values
            };
            assert_eq!(run(None), run(domains));
        }
        let passes = |domains| {
            sgs_passes(
                &z.locs[..n],
                &z.vals[..n],
                Some(&z.holes[..n]),
                domains,
                &grid,
                &vg,
                &search,
                None,
            )
            .unwrap()
        };
        assert_eq!(passes(None), passes(domains));
    }

    #[test]
    fn each_domain_reproduces_its_declustered_histogram() {
        let z = zoned();
        let per = 30;
        let grid: Vec<Point> = (0..2 * per)
            .map(|i| (1e4 * (1 + i / per) as f64 + 100.0 * i as f64, 1e4, 0.0))
            .collect();
        let nodes: Vec<u32> = (0..2 * per).map(|i| (i / per) as u32).collect();
        let vg = Variogram::single(Model::Spherical, 1.0, 25.0);
        let reals: Vec<Vec<f64>> = (0..100)
            .map(|seed| {
                let params = SgsParams {
                    search: zoned_search(None),
                    seed,
                };
                sgs_in(
                    &z.locs,
                    &z.vals,
                    Some(&z.weights),
                    None,
                    Some((&z.codes, &nodes)),
                    &grid,
                    &vg,
                    &params,
                    None,
                )
                .unwrap()
                .values
            })
            .collect();
        let mean = |v: &[f64], w: &[f64]| {
            v.iter().zip(w).map(|(v, w)| v * w).sum::<f64>() / w.iter().sum::<f64>()
        };
        for code in 0..2 {
            let rows: Vec<usize> = (0..z.vals.len()).filter(|&i| z.codes[i] == code).collect();
            let data: Vec<f64> = rows.iter().map(|&i| z.vals[i]).collect();
            let w: Vec<f64> = rows.iter().map(|&i| z.weights[i]).collect();
            let range = code as usize * per..(code as usize + 1) * per;
            let mut pooled: Vec<f64> = reals
                .iter()
                .flat_map(|r| r[range.clone()].to_vec())
                .collect();
            let (want, naive) = (mean(&data, &w), mean(&data, &vec![1.0; data.len()]));
            let got = mean(&pooled, &vec![1.0; pooled.len()]);
            assert!((want / naive - 1.0).abs() > 0.1, "weights change nothing");
            assert!(
                (got / want - 1.0).abs() < 0.05,
                "domain {code}: {got} vs {want}"
            );
            pooled.sort_by(f64::total_cmp);
            let table = normal_score::transform(&data, Some(&w)).unwrap().table;
            for (q, z) in [
                (0.1, -1.281_551_565_545),
                (0.5, 0.0),
                (0.9, 1.281_551_565_545),
            ] {
                let (got, want) = (pooled[(q * pooled.len() as f64) as usize], table.back(z));
                assert!(
                    (got / want).ln().abs() < 0.1,
                    "domain {code} q{q}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn on_the_contact_a_node_takes_its_own_domain_datum() {
        let z = zoned();
        let (grid, nodes) = zoned_grid();
        let vg = Variogram::single(Model::Spherical, 1.0, 25.0);
        for soft in [None, Some(estimation::Soft::All(f64::INFINITY))] {
            let params = SgsParams {
                search: zoned_search(soft),
                seed: 5,
            };
            let r = sgs_in(
                &z.locs,
                &z.vals,
                Some(&z.weights),
                Some(&z.holes),
                Some((&z.codes, &nodes)),
                &grid,
                &vg,
                &params,
                None,
            )
            .unwrap();
            for (i, &code) in nodes.iter().enumerate().skip(625) {
                assert_eq!(r.values[i], if code == 0 { 0.5 } else { 20.0 });
            }
        }
    }

    #[test]
    fn domains_follow_the_seed_not_the_thread_count() {
        let z = zoned();
        let (grid, nodes) = zoned_grid();
        let vg = Variogram::single(Model::Spherical, 1.0, 25.0);
        let run = |threads, seed| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let params = SgsParams {
                        search: zoned_search(Some(estimation::Soft::All(8.0))),
                        seed,
                    };
                    sgs_in(
                        &z.locs,
                        &z.vals,
                        Some(&z.weights),
                        Some(&z.holes),
                        Some((&z.codes, &nodes)),
                        &grid,
                        &vg,
                        &params,
                        None,
                    )
                    .unwrap()
                    .values
                })
        };
        let a = run(1, 3);
        assert_eq!(a, run(4, 3));
        assert_ne!(a, run(4, 4));
    }

    #[test]
    fn a_node_domain_without_data_is_an_error() {
        let z = zoned();
        let grid = vec![(1.0, 1.0, 1.0), (2.0, 2.0, 1.0)];
        let vg = Variogram::single(Model::Spherical, 1.0, 25.0);
        let search = zoned_search(None);
        let params = SgsParams {
            search: search.clone(),
            seed: 1,
        };
        let run = |nodes: &[u32], codes: &[u32]| {
            sgs_in(
                &z.locs,
                &z.vals,
                None,
                None,
                Some((codes, nodes)),
                &grid,
                &vg,
                &params,
                None,
            )
        };
        assert!(run(&[0, 1], &z.codes).is_ok());
        assert!(run(&[0, 2], &z.codes).is_err());
        assert!(run(&[0], &z.codes).is_err());
        assert!(run(&[0, 1], &z.codes[1..]).is_err());
        let domains = Some((&z.codes[..], &[0, 2][..]));
        let passes = sgs_passes(&z.locs, &z.vals, None, domains, &grid, &vg, &search, None);
        assert!(passes.is_err());
    }
}
