//! Sequential Indicator Simulation (SIS) for categorical variables (facies).
//!
//! At each node along a random path, the conditional probability of each category is
//! obtained by ordinary kriging of that category's indicator, corrected to a valid
//! distribution (clamped to `[0,1]`, renormalized to sum 1). A category is drawn and
//! added as hard data for subsequent nodes.

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
/// indicator variogram for category `c`.
pub fn sis(
    data_locs: &[(f64, f64, f64)],
    data_cats: &[usize],
    grid: &[(f64, f64, f64)],
    n_categories: usize,
    variograms: &[Variogram],
    params: &SisParams,
) -> Result<CategoricalRealization> {
    if data_locs.len() != data_cats.len() {
        return Err(SimError::InvalidParameters("data length mismatch".into()));
    }
    if data_locs.is_empty() {
        return Err(SimError::InsufficientData("no conditioning data".into()));
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
        .map(|(&loc, &c)| Sample {
            loc,
            value: c as f64,
            hole: None,
        })
        .collect();
    let mut tree = SearchTree::new(&all, &params.search, Some(&variograms[0]));

    let mut rng = StdRng::seed_from_u64(params.seed);
    let mut path: Vec<usize> = (0..grid.len()).collect();
    path.shuffle(&mut rng);

    let mut out = vec![0usize; grid.len()];

    for &node in &path {
        let target = grid[node];

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
                    // Indicator = 1 where the neighbor's category equals c.
                    let ind: Vec<Sample> = idx
                        .iter()
                        .map(|&k| Sample {
                            loc: all[k].loc,
                            value: if all[k].value.round() as usize == c {
                                1.0
                            } else {
                                0.0
                            },
                            hole: None,
                        })
                        .collect();
                    let est = krige(Kind::Ordinary, &target, &ind, &variograms[c])
                        .map(|e| e.value)
                        .unwrap_or(marg[c]);
                    p[c] = est.clamp(0.0, 1.0);
                }
                normalize(&mut p, &marg);
                p
            }
            _ => marg.clone(),
        };

        let cat = draw_category(&probs, &mut rng);
        out[node] = cat;

        let sample = Sample {
            loc: target,
            value: cat as f64,
            hole: None,
        };
        tree.add(&sample);
        all.push(sample);
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
            },
            seed: 1,
        };
        let real = sis(&data_locs, &data_cats, &grid, 2, &vgs, &params).unwrap();
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
            },
            seed: 5,
        };
        let a = sis(&data_locs, &data_cats, &grid, 3, &vgs, &params).unwrap();
        let b = sis(&data_locs, &data_cats, &grid, 3, &vgs, &params).unwrap();
        assert_eq!(a.categories, b.categories);
        assert!(a.categories.iter().all(|&c| c < 3));
    }
}
