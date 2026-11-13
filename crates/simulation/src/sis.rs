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
use rayon::prelude::*;
use variogram::Variogram;

/// Largest batch of nodes searched and kriged in parallel.
const BATCH: usize = 512;

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
    simulate(
        data_locs,
        data_cats,
        data_holes,
        grid,
        n_categories,
        variograms,
        params,
        proportions,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn simulate(
    data_locs: &[(f64, f64, f64)],
    data_cats: &[usize],
    data_holes: Option<&[u32]>,
    grid: &[(f64, f64, f64)],
    n_categories: usize,
    variograms: &[Variogram],
    params: &SisParams,
    proportions: Option<(&[Vec<f64>], &[Vec<f64>])>,
    batch: Option<usize>,
) -> Result<CategoricalRealization> {
    let Inputs {
        samples: mut all,
        mut means,
        local,
        marg,
    } = inputs(
        data_locs,
        data_cats,
        data_holes,
        grid.len(),
        n_categories,
        variograms,
        &params.search,
        proportions,
    )?;
    if grid.is_empty() {
        return Ok(CategoricalRealization { categories: vec![] });
    }
    let mut tree = SearchTree::new(&all, &params.search, Some(&variograms[0]));

    let mut rng = StdRng::seed_from_u64(params.seed);
    let mut path: Vec<usize> = (0..grid.len()).collect();
    path.shuffle(&mut rng);

    let on_datum = crate::sgs::datum_index(data_locs);
    let mut out = vec![0usize; grid.len()];
    let kind = match local {
        Some(_) => Kind::Simple { mean: 0.0 },
        None => Kind::Ordinary,
    };
    let (reps, groups) = variogram_groups(variograms);
    let local_rows = local.as_deref();
    let most = params.search.max_samples;

    // The weights of a neighborhood depend on the neighbor locations and the
    // variogram, not on the indicator values.
    let step = |all: &[Sample],
                means: &[Vec<f64>],
                tree: &SearchTree,
                node: usize,
                earlier: Option<&[usize]>|
     -> Option<Step> {
        let target = grid[node];
        let prior: &[f64] = match local_rows {
            Some(rows) => &rows[node],
            None => &marg,
        };
        let found = tree.neighbors(&target);
        if let Some(earlier) = earlier {
            // Stale when a node simulated since `all` would join the search:
            // within the farthest neighbor of a full search, or within the
            // radius of one that is not.
            let d = |p: &(f64, f64, f64)| tree.distance(&target, p);
            let reach = match &found {
                Ok(idx) if idx.len() >= most => {
                    idx.iter().map(|&k| d(&all[k].loc)).fold(0.0, f64::max)
                }
                _ => params.search.radius,
            };
            if earlier.iter().any(|&j| d(&grid[j]) <= reach * (1.0 + 1e-9)) {
                return None;
            }
        }
        let datum = match &found {
            Ok(idx) => idx.iter().copied().find(|&k| all[k].loc == target),
            Err(_) => on_datum.get(&crate::sgs::location_key(&target)).copied(),
        };
        if let Some(k) = datum {
            return Some(Step::Datum(all[k].value as usize));
        }
        let idx = match found {
            Ok(idx) if !idx.is_empty() => idx,
            _ => return Some(Step::Marginal(prior.to_vec())),
        };
        let zeros: Vec<Sample> = idx
            .iter()
            .map(|&k| Sample {
                loc: all[k].loc,
                value: 0.0,
                hole: None,
                error_variance: 0.0,
                domain: None,
            })
            .collect();
        let mut p = vec![0.0; n_categories];
        for (g, cats) in groups.iter().enumerate() {
            match krige(kind, &target, &zeros, &variograms[reps[g]]) {
                Ok(est) => {
                    for &c in cats {
                        let z_at = |k: usize| {
                            f64::from(u8::from(all[k].value.round() as usize == c))
                                - means.get(k).map_or(0.0, |m| m[c])
                        };
                        let value = match kind {
                            Kind::Simple { mean } => {
                                mean + est
                                    .weights
                                    .iter()
                                    .zip(&idx)
                                    .map(|(w, &k)| w * (z_at(k) - mean))
                                    .sum::<f64>()
                            }
                            _ => est
                                .weights
                                .iter()
                                .zip(&idx)
                                .map(|(w, &k)| w * z_at(k))
                                .sum::<f64>(),
                        };
                        let offset = if local_rows.is_some() { prior[c] } else { 0.0 };
                        p[c] = (value + offset).clamp(0.0, 1.0);
                    }
                }
                Err(_) => {
                    for &c in cats {
                        p[c] = prior[c].clamp(0.0, 1.0);
                    }
                }
            }
        }
        normalize(&mut p, prior);
        Some(Step::Probs(p))
    };

    let mut start = 0;
    while start < path.len() {
        // A batch is searched and kriged in parallel against the nodes
        // simulated before it. Stale nodes are redone in path order, so the
        // realization is the sequential one for any batch size.
        let size = batch.unwrap_or(all.len() / (8 * most).max(1));
        let nodes = &path[start..(start + size.clamp(1, BATCH)).min(path.len())];
        let steps: Vec<Option<Step>> = nodes
            .par_iter()
            .enumerate()
            .map(|(t, &node)| step(&all, &means, &tree, node, Some(&nodes[..t])))
            .collect();
        for (&node, found) in nodes.iter().zip(steps) {
            let found = match found {
                Some(found) => found,
                None => step(&all, &means, &tree, node, None).expect("never stale alone"),
            };
            let cat = match found {
                // A node on a datum takes its category and is not added again.
                Step::Datum(cat) => {
                    out[node] = cat;
                    continue;
                }
                Step::Marginal(probs) | Step::Probs(probs) => draw_category(&probs, &mut rng),
            };
            out[node] = cat;
            let sample = Sample {
                loc: grid[node],
                value: cat as f64,
                hole: None,
                error_variance: 0.0,
                domain: None,
            };
            tree.add(&sample);
            all.push(sample);
            if let Some(rows) = &local {
                means.push(rows[node].clone());
            }
        }
        start += nodes.len();
    }

    Ok(CategoricalRealization { categories: out })
}

/// Categories of a batch of realizations along one shared multigrid path.
pub struct SharedCategories {
    /// Node-major: the categories of node `m` are `cats[m * batch..][..batch]`.
    cats: Vec<u8>,
    batch: usize,
}

impl SharedCategories {
    pub fn len(&self) -> usize {
        self.batch
    }

    pub fn is_empty(&self) -> bool {
        self.batch == 0
    }

    /// Realization `b` of the batch, one category per lattice node.
    pub fn realization(&self, b: usize) -> Vec<usize> {
        self.cats
            .iter()
            .skip(b)
            .step_by(self.batch)
            .map(|&c| c as usize)
            .collect()
    }
}

/// A node's probabilities in every realization: `constant[c]` plus, for each
/// earlier node `nodes[j]`, `weights[group[c]][j]` where that node holds `c`.
/// `None` weights for a group whose system failed: its categories keep
/// their constant, the node's prior.
struct SisPlan {
    constant: Vec<f64>,
    nodes: Vec<u32>,
    weights: Vec<Option<Vec<f64>>>,
}

/// Realizations `realizations` of SIS over `shared.lattice` along one
/// multigrid path shared by every realization, as [`crate::sgs_shared`] is
/// for SGS. The inputs are those of [`sis`], with local proportions at the
/// data and at every lattice node. A node's neighbors and indicator-kriging
/// weights are found once for the batch; each realization draws its own
/// categories from its seed and the node, so it does not depend on the
/// batch, the chunking or the threads.
#[allow(clippy::too_many_arguments)]
pub fn sis_shared(
    data_locs: &[(f64, f64, f64)],
    data_cats: &[usize],
    data_holes: Option<&[u32]>,
    n_categories: usize,
    variograms: &[Variogram],
    proportions: Option<(&[Vec<f64>], &[Vec<f64>])>,
    shared: &crate::shared::SharedSgs,
    realizations: std::ops::Range<usize>,
) -> Result<SharedCategories> {
    use crate::shared::{CHUNK, Context, datum_nodes, fallback, located, neighbors, waves};
    if let Some(why) = crate::shared::shared_unsupported(shared.search, false, false) {
        return Err(SimError::InvalidParameters(format!(
            "a shared path does not support {why}"
        )));
    }
    let [search] = shared.search else {
        return Err(SimError::InvalidParameters("SIS takes one search".into()));
    };
    if n_categories > u8::MAX as usize + 1 {
        return Err(SimError::InvalidParameters(
            "at most 256 categories on a shared path".into(),
        ));
    }
    if realizations.is_empty() {
        return Err(SimError::InvalidParameters(
            "need at least one realization".into(),
        ));
    }
    let lattice = shared.lattice;
    let n = lattice.len();
    let Inputs {
        samples,
        means,
        local,
        marg,
    } = inputs(
        data_locs,
        data_cats,
        data_holes,
        n,
        n_categories,
        variograms,
        search,
        proportions,
    )?;
    let vg = &variograms[0];
    let trees = vec![crate::sgs::tree(&samples, search, vg, None)];
    let templates = vec![crate::lattice::Template::new(
        lattice,
        trees[0].frame(),
        search.radius,
    )?];
    let top = shared
        .levels
        .unwrap_or_else(|| crate::lattice::default_levels(lattice, &templates[0]));
    let on_datum = datum_nodes(lattice, data_locs, None);
    let mut skip = vec![false; n];
    for &(m, _) in &on_datum {
        skip[m] = true;
    }
    let (path, rank) = crate::lattice::multigrid_path(lattice, top, shared.seed, &skip)?;
    drop(skip);
    let batch = realizations.len();
    let keys: Vec<u64> = realizations
        .map(|k| boitata_core::rng::realization_seed(shared.seed, k as u64))
        .collect();
    let mut cats = vec![0u8; n * batch];
    for &(m, k) in &on_datum {
        cats[m * batch..(m + 1) * batch].fill(data_cats[k] as u8);
    }
    let ctx = Context {
        fallback: fallback(&samples, shared.search, vg),
        lattice,
        samples,
        scores: vec![],
        trees,
        templates,
        search: shared.search,
        vg,
        node_domains: None,
        rank,
        marginal: 0.0,
    };
    let kind = match local {
        Some(_) => Kind::Simple { mean: 0.0 },
        None => Kind::Ordinary,
    };
    let (reps, groups) = variogram_groups(variograms);
    let mut group = vec![0usize; n_categories];
    for (g, members) in groups.iter().enumerate() {
        for &c in members {
            group[c] = g;
        }
    }
    let prior = |m: usize| -> &[f64] { local.as_ref().map_or(&marg, |rows| &rows[m]) };
    let plan = |m: usize| -> SisPlan {
        let target = lattice.location(m);
        let near = neighbors(&ctx, m);
        let p = prior(m);
        if near.is_empty() {
            return SisPlan {
                constant: p.to_vec(),
                nodes: vec![],
                weights: vec![None; reps.len()],
            };
        }
        let selected = located(&ctx, &near);
        let weights: Vec<Option<Vec<f64>>> = reps
            .iter()
            .map(|&r| {
                krige(kind, &target, &selected, &variograms[r])
                    .ok()
                    .map(|e| e.weights)
            })
            .collect();
        let mut constant = vec![0.0; n_categories];
        let mut nodes = vec![];
        for (c, value) in constant.iter_mut().enumerate() {
            let Some(w) = &weights[group[c]] else {
                *value = p[c];
                continue;
            };
            for (&(_, is_node, i), &wj) in near.iter().zip(w) {
                let i = i as usize;
                let mean = match is_node {
                    true => local.as_ref().map_or(0.0, |rows| rows[i][c]),
                    false => means.get(i).map_or(0.0, |row| row[c]),
                };
                let on = !is_node && data_cats[i] == c;
                *value += wj * (f64::from(u8::from(on)) - mean);
            }
            if local.is_some() {
                *value += p[c];
            }
        }
        let mut kept = vec![];
        for (j, &(_, is_node, i)) in near.iter().enumerate() {
            if is_node {
                nodes.push(i);
                kept.push(j);
            }
        }
        let weights = weights
            .into_iter()
            .map(|w| w.map(|w| kept.iter().map(|&j| w[j]).collect()))
            .collect();
        SisPlan {
            constant,
            nodes,
            weights,
        }
    };
    for chunk in path.chunks(CHUNK) {
        let plans: Vec<SisPlan> = chunk.par_iter().map(|&m| plan(m as usize)).collect();
        for wave in waves(chunk, &ctx.rank, |t| plans[t].nodes.iter().copied()) {
            let drawn: Vec<Vec<u8>> = wave
                .par_iter()
                .map(|&t| {
                    let m = chunk[t] as usize;
                    let plan = &plans[t];
                    keys.iter()
                        .enumerate()
                        .map(|(b, &key)| {
                            let mut probs = plan.constant.clone();
                            for (c, value) in probs.iter_mut().enumerate() {
                                if let Some(w) = &plan.weights[group[c]] {
                                    for (&node, &wj) in plan.nodes.iter().zip(w) {
                                        if cats[node as usize * batch + b] as usize == c {
                                            *value += wj;
                                        }
                                    }
                                }
                                *value = value.clamp(0.0, 1.0);
                            }
                            normalize(&mut probs, prior(m));
                            let u = unit(boitata_core::rng::realization_seed(key, m as u64));
                            pick(&probs, u) as u8
                        })
                        .collect()
                })
                .collect();
            for (&t, v) in wave.iter().zip(drawn) {
                let m = chunk[t] as usize;
                cats[m * batch..(m + 1) * batch].copy_from_slice(&v);
            }
        }
    }
    Ok(SharedCategories { cats, batch })
}

/// The checked inputs of one SIS run: the data as samples, their rows of
/// local proportions (empty without them), the rows at the `nodes` nodes,
/// and the data proportions, the fallback of a node without neighbors.
struct Inputs {
    samples: Vec<Sample>,
    means: Vec<Vec<f64>>,
    local: Option<Vec<Vec<f64>>>,
    marg: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
fn inputs(
    data_locs: &[(f64, f64, f64)],
    data_cats: &[usize],
    data_holes: Option<&[u32]>,
    nodes: usize,
    n_categories: usize,
    variograms: &[Variogram],
    search: &Search,
    proportions: Option<(&[Vec<f64>], &[Vec<f64>])>,
) -> Result<Inputs> {
    if data_locs.len() != data_cats.len() {
        return Err(SimError::InvalidParameters("data length mismatch".into()));
    }
    if data_locs.is_empty() {
        return Err(SimError::InsufficientData("no conditioning data".into()));
    }
    if search.clamps() {
        return Err(SimError::InvalidParameters(
            "categories cannot be clamped at a high-grade threshold".into(),
        ));
    }
    if variograms.len() != n_categories {
        return Err(SimError::InvalidParameters(
            "need one variogram per category".into(),
        ));
    }
    if let Some(c) = data_cats.iter().find(|&&c| c >= n_categories) {
        return Err(SimError::InvalidParameters(format!(
            "category {c} ≥ n_categories {n_categories}"
        )));
    }
    let holes = crate::holes(data_holes, data_locs.len())?;
    let (means, local) = match proportions {
        Some((at_data, at_grid)) => {
            if at_data.len() != data_locs.len() || at_grid.len() != nodes {
                return Err(SimError::InvalidParameters(
                    "one row of proportions per datum and per node".into(),
                ));
            }
            let close = |rows: &[Vec<f64>]| {
                rows.iter()
                    .map(|r| closed(r, n_categories, "proportions"))
                    .collect::<Result<Vec<_>>>()
            };
            (close(at_data)?, Some(close(at_grid)?))
        }
        None => (vec![], None),
    };
    let mut marg = vec![0.0f64; n_categories];
    for &c in data_cats {
        marg[c] += 1.0;
    }
    for m in &mut marg {
        *m /= data_cats.len() as f64;
    }
    let samples = data_locs
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
    Ok(Inputs {
        samples,
        means,
        local,
        marg,
    })
}

enum Step {
    Datum(usize),
    Marginal(Vec<f64>),
    Probs(Vec<f64>),
}

/// Categories that share a variogram, and the index of each group's variogram.
fn variogram_groups(variograms: &[Variogram]) -> (Vec<usize>, Vec<Vec<usize>>) {
    let mut reps: Vec<usize> = Vec::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (c, vg) in variograms.iter().enumerate() {
        match reps
            .iter()
            .position(|&r| same_variogram(&variograms[r], vg))
        {
            Some(g) => groups[g].push(c),
            None => {
                reps.push(c);
                groups.push(vec![c]);
            }
        }
    }
    (reps, groups)
}

fn same_variogram(a: &Variogram, b: &Variogram) -> bool {
    a.nugget == b.nugget
        && a.structures == b.structures
        && match (&a.anisotropy, &b.anisotropy) {
            (None, None) => true,
            (Some(x), Some(y)) => x.angles == y.angles,
            _ => false,
        }
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

/// `row` of `k` probabilities (`what`) rescaled to sum 1.
pub(crate) fn closed(row: &[f64], k: usize, what: &str) -> Result<Vec<f64>> {
    let s: f64 = row.iter().sum();
    if row.len() != k || row.iter().any(|p| !p.is_finite() || *p < 0.0) || s <= 0.0 {
        return Err(SimError::InvalidParameters(format!(
            "{what} must be {k} finite, non-negative values, not all zero"
        )));
    }
    Ok(row.iter().map(|p| p / s).collect())
}

/// Draw a category index from a probability vector using `rng`.
fn draw_category(probs: &[f64], rng: &mut StdRng) -> usize {
    pick(probs, rng.r#gen::<f64>())
}

/// A uniform draw in `[0, 1)` fixed by `key`.
fn unit(key: u64) -> f64 {
    (boitata_core::rng::splitmix(key) >> 11) as f64 / (1u64 << 53) as f64
}

/// The category whose cumulative probability first reaches `u`.
fn pick(probs: &[f64], u: f64) -> usize {
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

    fn shared_run(
        reals: std::ops::Range<usize>,
        proportions: Option<(&[Vec<f64>], &[Vec<f64>])>,
    ) -> (Vec<(f64, f64, f64)>, Vec<usize>, SharedCategories) {
        let l = crate::Lattice::regular(boitata_core::Geometry {
            origin: [0.0; 3],
            size: [1.0, 1.0, 1.0],
            count: [30, 30, 1],
            rotation: [0.0; 3],
        });
        let locs: Vec<(f64, f64, f64)> = (0..30)
            .map(|i| ((i * 7 % 30) as f64 + 0.5, (i * 11 % 30) as f64 + 0.5, 0.5))
            .collect();
        let cats: Vec<usize> = (0..30)
            .map(|i| usize::from(i % 5 == 0) + usize::from(i % 7 == 0))
            .collect();
        let search = [Search {
            min_samples: 1,
            max_samples: 12,
            radius: 12.0,
            ..Default::default()
        }];
        let vgs = vec![Variogram::single(Model::Spherical, 0.2, 9.0); 3];
        let shared = crate::shared::SharedSgs {
            lattice: &l,
            search: &search,
            levels: None,
            seed: 5,
        };
        let out = sis_shared(&locs, &cats, None, 3, &vgs, proportions, &shared, reals).unwrap();
        (locs, cats, out)
    }

    #[test]
    fn shared_realizations_honor_data_on_centroids() {
        let (locs, cats, out) = shared_run(0..3, None);
        for b in 0..3 {
            let r = out.realization(b);
            for (p, &c) in locs.iter().zip(&cats) {
                let m = (p.1 - 0.5) as usize * 30 + (p.0 - 0.5) as usize;
                assert_eq!(r[m], c);
            }
        }
    }

    #[test]
    fn a_shared_realization_does_not_depend_on_its_batch() {
        let (_, _, together) = shared_run(0..4, None);
        let (_, _, alone) = shared_run(2..3, None);
        assert_eq!(together.realization(2), alone.realization(0));
        assert_ne!(together.realization(0), together.realization(1));
    }

    #[test]
    fn shared_realizations_follow_the_seed_not_the_threads() {
        let with = |t: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
                .install(|| shared_run(0..2, None).2.realization(1))
        };
        assert_eq!(with(1), with(4));
    }

    #[test]
    fn shared_proportions_follow_the_data() {
        let (_, cats, out) = shared_run(0..40, None);
        let data: Vec<f64> = (0..3)
            .map(|c| cats.iter().filter(|&&k| k == c).count() as f64 / cats.len() as f64)
            .collect();
        for c in 0..3 {
            let sim = (0..40)
                .map(|b| out.realization(b).iter().filter(|&&k| k == c).count() as f64 / 900.0)
                .sum::<f64>()
                / 40.0;
            assert!(
                (sim - data[c]).abs() < 0.06,
                "category {c}: {sim} vs {}",
                data[c]
            );
        }
    }

    #[test]
    fn shared_local_proportions_run() {
        let at_data = vec![vec![0.6, 0.3, 0.1]; 30];
        let at_nodes: Vec<Vec<f64>> = (0..900)
            .map(|m| {
                let x = (m % 30) as f64 / 29.0;
                vec![0.8 - 0.5 * x, 0.15 + 0.3 * x, 0.05 + 0.2 * x]
            })
            .collect();
        let (_, _, out) = shared_run(0..2, Some((&at_data, &at_nodes)));
        assert!(out.realization(0).iter().all(|&c| c < 3));
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
                sectors: None,
                anisotropy: None,
                high_grade: None,
                soft: None,
                calibration: None,
                ..Default::default()
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
                sectors: None,
                anisotropy: None,
                high_grade: None,
                soft: None,
                calibration: None,
                ..Default::default()
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

    #[test]
    fn realizations_follow_the_seed_not_threads_or_batches() {
        let data_locs: Vec<_> = (0..24)
            .map(|i| ((i % 6) as f64 * 4.0, (i / 6) as f64 * 4.0, 0.0))
            .collect();
        let data_cats: Vec<_> = (0..24).map(|i| i % 3).collect();
        let grid: Vec<_> = (0..16 * 16)
            .map(|i| ((i % 16) as f64, (i / 16) as f64, 0.0))
            .collect();
        let search = Search {
            min_samples: 1,
            max_samples: 8,
            radius: 20.0,
            ..Default::default()
        };
        let shared = vec![Variogram::single(Model::Spherical, 0.2, 12.0); 3];
        let distinct: Vec<_> = (0..3)
            .map(|c| Variogram::single(Model::Spherical, 0.2, 10.0 + c as f64))
            .collect();
        let at_data = vec![vec![1.0 / 3.0; 3]; data_locs.len()];
        let at_grid: Vec<_> = grid
            .iter()
            .map(|p| vec![0.5, 0.3 * p.0 / 15.0, 0.2])
            .collect();
        for (vgs, proportions) in [
            (shared, None),
            (distinct, None),
            (
                vec![Variogram::single(Model::Exponential, 0.2, 14.0); 3],
                Some((&at_data[..], &at_grid[..])),
            ),
        ] {
            let run = |threads, batch| {
                let params = SisParams {
                    search: search.clone(),
                    seed: 3,
                };
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| {
                        simulate(
                            &data_locs,
                            &data_cats,
                            None,
                            &grid,
                            3,
                            &vgs,
                            &params,
                            proportions,
                            batch,
                        )
                        .unwrap()
                        .categories
                    })
            };
            let sequential = run(1, Some(1));
            for threads in [1, 8] {
                for batch in [Some(1), Some(3), Some(64), None] {
                    assert_eq!(sequential, run(threads, batch));
                }
            }
        }
    }
}
