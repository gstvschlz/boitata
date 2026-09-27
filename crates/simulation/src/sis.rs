//! Sequential Indicator Simulation (SIS) for categorical variables (facies).
//!
//! At each node along a random path, the conditional probability of each category is
//! obtained by ordinary kriging of that category's indicator, corrected to a valid
//! distribution (clamped to `[0,1]`, renormalized to sum 1). A category is drawn and
//! added as hard data for subsequent nodes. With local proportions, simple kriging
//! of the indicators minus their local proportions replaces ordinary kriging, so
//! each node's probabilities are drawn towards its own proportions.

use crate::error::{Result, SimError};
use estimation::Sample;
use estimation::krige::{Kind, krige};
use estimation::search::{Search, SearchTree};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use variogram::Variogram;

/// SIS parameters.
#[derive(Debug, Clone)]
pub struct SisParams {
    pub search: Search,
    pub seed: u64,
}

/// One categorical realization over the grid (category index per node).
#[derive(Debug, Clone)]
pub struct CategoricalRealization {
    pub categories: Vec<usize>,
}

/// Run a single SIS realization over `n_categories` facies.
///
/// `data_cats[k]` is the category (0-based) at `data_locs[k]`. `variograms[c]` is the
/// indicator variogram for category `c`. `data_holes` tag the data by drill
/// hole for `max_per_hole`, simulated nodes belonging to none. `proportions`
/// holds local proportions of the categories at the data and at the grid, one
/// row each, rescaled to sum 1.
#[allow(clippy::too_many_arguments)]
pub fn sis(
    data_locs: &[(f64, f64, f64)],
    data_cats: &[usize],
    data_holes: Option<&[u32]>,
    grid: &[(f64, f64, f64)],
    n_categories: usize,
    variograms: &[Variogram],
    params: &SisParams,
    proportions: Option<(&[Vec<f64>], &[Vec<f64>])>,
) -> Result<CategoricalRealization> {
    if data_locs.len() != data_cats.len() {
        return Err(SimError::InvalidParameters("data length mismatch".into()));
    }
    if data_locs.is_empty() {
        return Err(SimError::InsufficientData("no conditioning data".into()));
    }
    if params.search.clamps() {
        return Err(SimError::InvalidParameters(
            "categories cannot be clamped at a high-grade threshold".into(),
        ));
    }
    if variograms.len() != n_categories {
        return Err(SimError::InvalidParameters(
            "need one variogram per category".into(),
        ));
    }
    for &c in data_cats {
        if c >= n_categories {
            return Err(SimError::InvalidParameters(format!(
                "category {c} ≥ n_categories {n_categories}"
            )));
        }
    }
    let holes = crate::holes(data_holes, data_locs.len())?;
    let (mut means, local) = match proportions {
        Some((at_data, at_grid)) => {
            if at_data.len() != data_locs.len() || at_grid.len() != grid.len() {
                return Err(SimError::InvalidParameters(
                    "one row of proportions per datum and per node".into(),
                ));
            }
            let close = |rows: &[Vec<f64>]| {
                rows.iter()
                    .map(|r| closed(r, n_categories))
                    .collect::<Result<Vec<_>>>()
            };
            (close(at_data)?, Some(close(at_grid)?))
        }
        None => (vec![], None),
    };
    if grid.is_empty() {
        return Ok(CategoricalRealization { categories: vec![] });
    }

    // Marginal proportions (fallback when a node has no neighbors).
    let mut marg = vec![0.0f64; n_categories];
    for &c in data_cats {
        marg[c] += 1.0;
    }
    for m in &mut marg {
        *m /= data_cats.len() as f64;
    }

    // Conditioning categories (grow as nodes are simulated).
    let mut all: Vec<Sample> = data_locs
        .iter()
        .zip(data_cats)
        .zip(holes)
        .map(|((&loc, &c), hole)| Sample {
            loc,
            value: c as f64,
            hole,
            error_variance: 0.0,
            domain: None,
        })
        .collect();
    let mut tree = SearchTree::new(&all, &params.search, Some(&variograms[0]));

    let mut rng = StdRng::seed_from_u64(params.seed);
    let mut path: Vec<usize> = (0..grid.len()).collect();
    path.shuffle(&mut rng);

    let mut out = vec![0usize; grid.len()];
    let kind = match local {
        Some(_) => Kind::Simple { mean: 0.0 },
        None => Kind::Ordinary,
    };

    for &node in &path {
        let target = grid[node];
        let prior = local.as_ref().map_or(&marg, |l| &l[node]);

        let found = tree.neighbors(&target);
        if let Some(&k) = found.iter().flatten().find(|&&k| all[k].loc == target) {
            // A node on a datum takes its category and is not added again.
            out[node] = all[k].value as usize;
            continue;
        }
        let probs = match found {
            Ok(idx) if !idx.is_empty() => {
                let mut p = vec![0.0f64; n_categories];
                for c in 0..n_categories {
                    // Indicator = 1 where the neighbor's category equals c,
                    // less its local proportion.
                    let mean = |k: usize| means.get(k).map_or(0.0, |m: &Vec<f64>| m[c]);
                    let ind: Vec<Sample> = idx
                        .iter()
                        .map(|&k| Sample {
                            loc: all[k].loc,
                            value: f64::from(u8::from(all[k].value.round() as usize == c))
                                - mean(k),
                            hole: None,
                            error_variance: 0.0,
                            domain: None,
                        })
                        .collect();
                    let offset = if local.is_some() { prior[c] } else { 0.0 };
                    let est = krige(kind, &target, &ind, &variograms[c])
                        .map(|e| e.value + offset)
                        .unwrap_or(prior[c]);
                    p[c] = est.clamp(0.0, 1.0);
                }
                normalize(&mut p, prior);
                p
            }
            _ => prior.clone(),
        };

        let cat = draw_category(&probs, &mut rng);
        out[node] = cat;

        let sample = Sample {
            loc: target,
            value: cat as f64,
            hole: None,
            error_variance: 0.0,
            domain: None,
        };
        tree.add(&sample);
        all.push(sample);
        if let Some(l) = &local {
            means.push(l[node].clone());
        }
    }

    Ok(CategoricalRealization { categories: out })
}

/// Normalize probabilities to sum 1; fall back to marginals if degenerate.
fn normalize(p: &mut [f64], marg: &[f64]) {
    let s: f64 = p.iter().sum();
    if s <= 0.0 {
        p.copy_from_slice(marg);
    } else {
        for x in p.iter_mut() {
            *x /= s;
        }
    }
}

/// `row` of `k` proportions rescaled to sum 1.
fn closed(row: &[f64], k: usize) -> Result<Vec<f64>> {
    let s: f64 = row.iter().sum();
    if row.len() != k || row.iter().any(|p| !p.is_finite() || *p < 0.0) || s <= 0.0 {
        return Err(SimError::InvalidParameters(format!(
            "proportions must be {k} finite, non-negative values, not all zero"
        )));
    }
    Ok(row.iter().map(|p| p / s).collect())
}

/// Draw a category index from a probability vector using `rng`.
fn draw_category(probs: &[f64], rng: &mut StdRng) -> usize {
    let u: f64 = rng.r#gen::<f64>();
    let mut cum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if u <= cum {
            return i;
        }
    }
    probs.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use variogram::model::Model;

    fn ind_vg() -> Variogram {
        Variogram::single(Model::Spherical, 0.25, 100.0)
    }

    #[test]
    fn honors_categorical_data() {
        // Two facies; a grid node on a datum should take that datum's category.
        let data_locs = vec![(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (0.0, 100.0, 0.0)];
        let data_cats = vec![0, 1, 0];
        let grid = vec![(0.0, 0.0, 0.0)]; // coincides with category-0 datum
        let vgs = vec![ind_vg(), ind_vg()];
        let params = SisParams {
            search: Search {
                min_samples: 1,
                max_samples: 8,
                radius: f64::INFINITY,
                max_per_hole: None,
                octant: false,
                anisotropy: None,
                high_grade: None,
                soft: None,
            },
            seed: 1,
        };
        let real = sis(&data_locs, &data_cats, None, &grid, 2, &vgs, &params, None).unwrap();
        assert_eq!(real.categories[0], 0);
    }

    #[test]
    fn reproducible_and_valid_categories() {
        let data_locs = vec![(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (50.0, 100.0, 0.0)];
        let data_cats = vec![0, 1, 2];
        let grid: Vec<(f64, f64, f64)> = (0..15).map(|i| (i as f64 * 8.0, 40.0, 0.0)).collect();
        let vgs = vec![ind_vg(), ind_vg(), ind_vg()];
        let params = SisParams {
            search: Search {
                min_samples: 1,
                max_samples: 10,
                radius: f64::INFINITY,
                max_per_hole: None,
                octant: false,
                anisotropy: None,
                high_grade: None,
                soft: None,
            },
            seed: 5,
        };
        let a = sis(&data_locs, &data_cats, None, &grid, 3, &vgs, &params, None).unwrap();
        let b = sis(&data_locs, &data_cats, None, &grid, 3, &vgs, &params, None).unwrap();
        assert_eq!(a.categories, b.categories);
        assert!(a.categories.iter().all(|&c| c < 3));
    }

    #[test]
    fn max_per_hole_caps_the_data_of_one_hole() {
        let locs: Vec<_> = (0..10).map(|i| (0.0, 0.0, i as f64)).collect();
        let cats: Vec<usize> = (0..10).map(|i| i % 2).collect();
        let holes = vec![0; 10];
        let grid = vec![(3.0, 0.0, 4.4)];
        let vgs = vec![ind_vg(), ind_vg()];
        let run = |holes: Option<&[u32]>, max_samples, max_per_hole| {
            let search = Search {
                min_samples: 1,
                max_samples,
                radius: f64::INFINITY,
                max_per_hole,
                ..Default::default()
            };
            let params = SisParams { search, seed: 2 };
            sis(&locs, &cats, holes, &grid, 2, &vgs, &params, None)
                .unwrap()
                .categories
        };
        assert_eq!(run(Some(&holes), 8, Some(1)), run(None, 1, None));
        assert_eq!(run(Some(&holes), 8, None), run(None, 8, None));
    }

    #[test]
    fn local_proportions_are_reproduced_on_average() {
        // Far from the few data, realizations follow the proportions alone.
        let local = |x: f64| vec![0.8 * x / 300.0, 0.8 * (1.0 - x / 300.0), 0.2];
        let data_locs: Vec<_> = (0..3).map(|i| (-1e3 - i as f64, 0.0, 0.0)).collect();
        let data_cats = vec![0, 1, 2];
        let at_data = vec![vec![1.0; 3]; 3];
        let grid: Vec<_> = (0..300).map(|i| (i as f64 + 0.5, 0.0, 0.0)).collect();
        let at_grid: Vec<_> = grid.iter().map(|l| local(l.0)).collect();
        let vg = Variogram::single(Model::Spherical, 0.2, 20.0);
        let vgs = vec![vg.clone(), vg.clone(), vg];
        let run = |seed, proportions| {
            let search = Search {
                max_samples: 12,
                radius: 60.0,
                ..Default::default()
            };
            let params = SisParams { search, seed };
            sis(
                &data_locs,
                &data_cats,
                None,
                &grid,
                3,
                &vgs,
                &params,
                proportions,
            )
            .unwrap()
            .categories
        };
        let n = 200;
        let mut freq = [[0.0; 3]; 3];
        for seed in 0..n {
            for (i, c) in run(seed, Some((&at_data, &at_grid)))
                .into_iter()
                .enumerate()
            {
                freq[i / 100][c] += 1.0 / (100 * n) as f64;
            }
        }
        for (bin, f) in freq.iter().enumerate() {
            let x = 100.0 * bin as f64 + 50.0;
            for (got, want) in f.iter().zip(local(x)) {
                assert!(
                    (got - want).abs() < 0.06,
                    "bin {bin}: {f:?} vs {:?}",
                    local(x)
                );
            }
        }
        let with = Some((&at_data[..], &at_grid[..]));
        assert_eq!(run(9, with), run(9, with));
        assert_ne!(run(9, with), run(9, None));
        let short = Some((&at_data[1..], &at_grid[..]));
        let params = SisParams {
            search: Search::default(),
            seed: 0,
        };
        assert!(sis(&data_locs, &data_cats, None, &grid, 3, &vgs, &params, short).is_err());
    }
}
