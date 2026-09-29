//! Sequential Gaussian Simulation (SGS).
//!
//! Produces conditional realizations of a continuous variable:
//! 1. normal-score transform the conditioning data, within each domain, and
//!    with a trend within its classes (a stepwise conditional transform),
//! 2. visit grid nodes along a random path,
//! 3. at each node, simple-krige (mean 0) from nearby data + previously simulated
//!    nodes to get a conditional mean/variance, then draw from that Gaussian;
//!    with several searches, a node uses the first that finds enough data,
//!    as a kriging pass would,
//! 4. back-transform the draw through the node's domain and add the node to
//!    the conditioning set,
//! 5. continue until every node is simulated.
//!
//! Nodes are searched and kriged in parallel batches along the path, and
//! the realization is the sequential one for any number of threads.
//!
//! The search compares grades, of data and of simulated nodes, with a
//! high-grade threshold; a clamped neighbor enters as the score of the
//! threshold. Kriging uses scores: a neighbor of the node's
//! domain its own, one of another domain (through a soft boundary) its grade
//! transformed as the node's domain transforms grades.

use crate::TrendConditioning;
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

/// A trend at the data and at the grid nodes, and its number of classes.
#[derive(Debug, Clone, Copy)]
pub struct Trend<'a> {
    pub data: &'a [f64],
    pub nodes: &'a [f64],
    pub classes: usize,
}

/// Grades to normal scores and back in one domain: the declustered normal
/// score of the grades, or with a trend of their stepwise conditional
/// scores given the trend.
#[derive(Debug, Clone)]
pub struct Transform {
    trend: Option<TrendConditioning>,
    table: NormalScoreTable,
}

impl Transform {
    /// Score of `value` at trend `trend` (ignored without one).
    pub fn forward(&self, value: f64, trend: f64) -> f64 {
        let value = match &self.trend {
            Some(c) => c.forward_one(trend, value),
            None => value,
        };
        self.table.forward(value)
    }

    /// Grade at trend `trend` (ignored without one) of `score`.
    pub fn back(&self, score: f64, trend: f64) -> f64 {
        let value = self.table.back(score);
        match &self.trend {
            Some(c) => c.back_one(trend, value),
            None => value,
        }
    }
}

/// The transform of each domain code, `None` for a code without data, and
/// the scores of the data, each in its own domain.
#[derive(Debug, Clone)]
pub struct Transforms {
    pub scores: Vec<f64>,
    pub domains: Vec<Option<Transform>>,
}

impl Transforms {
    /// One transform for all data without `domains`; `trend` is the trend at
    /// the data and its number of classes.
    pub fn fit(
        values: &[f64],
        weights: Option<&[f64]>,
        domains: Option<&[u32]>,
        trend: Option<(&[f64], usize)>,
    ) -> Result<Self> {
        let n = values.len();
        if weights.is_some_and(|w| w.len() != n) {
            return Err(SimError::InvalidParameters("one weight per datum".into()));
        }
        if domains.is_some_and(|d| d.len() != n) {
            return Err(SimError::InvalidParameters("one domain per datum".into()));
        }
        if trend.is_some_and(|t| t.0.len() != n) {
            return Err(SimError::InvalidParameters(
                "one trend value per datum".into(),
            ));
        }
        let k = domains.map_or(1, |d| d.iter().max().map_or(0, |&m| m as usize + 1));
        let mut rows = vec![vec![]; k];
        for i in 0..n {
            rows[domains.map_or(0, |d| d[i] as usize)].push(i);
        }
        let mut scores = vec![0.0; n];
        let mut fitted = Vec::with_capacity(k);
        for rows in &rows {
            if rows.is_empty() {
                fitted.push(None);
                continue;
            }
            let pick = |v: &[f64]| rows.iter().map(|&i| v[i]).collect::<Vec<_>>();
            let w = weights.map(pick);
            let trend = trend
                .map(|(t, classes)| {
                    TrendConditioning::fit(&pick(values), &pick(t), w.as_deref(), classes)
                })
                .transpose()?;
            let values = match &trend {
                Some(c) => c.scores().to_vec(),
                None => pick(values),
            };
            let ns = normal_score::transform(&values, w.as_deref())
                .map_err(|e| SimError::Transform(e.to_string()))?;
            for (&i, s) in rows.iter().zip(ns.scores) {
                scores[i] = s;
            }
            fitted.push(Some(Transform {
                trend,
                table: ns.table,
            }));
        }
        Ok(Self {
            scores,
            domains: fitted,
        })
    }
}

/// A secondary variable for collocated cosimulation: the declustered normal
/// score of its values at the data and the correlation of those scores with
/// the primary's.
#[derive(Debug, Clone)]
pub struct Secondary {
    table: NormalScoreTable,
    pub correlation: f64,
}

impl Secondary {
    /// Fitted to the secondary `values` at the data, declustered by
    /// `weights`; `correlation`, in [-1, 1], or else the weighted correlation
    /// of their scores with `scores`, the primary scores of the data
    /// ([`Transforms::scores`]).
    pub fn fit(
        values: &[f64],
        weights: Option<&[f64]>,
        scores: &[f64],
        correlation: Option<f64>,
    ) -> Result<Self> {
        if weights.is_some_and(|w| w.len() != values.len()) {
            return Err(SimError::InvalidParameters("one weight per datum".into()));
        }
        let ns = normal_score::transform(values, weights)
            .map_err(|e| SimError::Transform(e.to_string()))?;
        let correlation = match correlation {
            Some(r) => r,
            None if scores.len() != values.len() => {
                return Err(SimError::InvalidParameters(
                    "one secondary value per datum".into(),
                ));
            }
            None => pearson(scores, &ns.scores, weights),
        };
        if !(-1.0..=1.0).contains(&correlation) {
            return Err(SimError::InvalidParameters(format!(
                "correlation {correlation} outside [-1, 1]"
            )));
        }
        Ok(Self {
            table: ns.table,
            correlation,
        })
    }

    /// Score of the secondary `value`.
    pub fn forward(&self, value: f64) -> f64 {
        self.table.forward(value)
    }
}

fn pearson(a: &[f64], b: &[f64], weights: Option<&[f64]>) -> f64 {
    let w = |i: usize| weights.map_or(1.0, |w| w[i]);
    let total: f64 = (0..a.len()).map(w).sum();
    let mean = |v: &[f64]| (0..v.len()).map(|i| w(i) * v[i]).sum::<f64>() / total;
    let (ma, mb) = (mean(a), mean(b));
    let co = |x: &[f64], mx: f64, y: &[f64], my: f64| {
        (0..x.len())
            .map(|i| w(i) * (x[i] - mx) * (y[i] - my))
            .sum::<f64>()
    };
    let r = co(a, ma, b, mb) / (co(a, ma, a, ma) * co(b, mb, b, mb)).sqrt();
    if r.is_finite() { r } else { 0.0 }
}

/// As [`sgs_in`], cosimulated with a `secondary` variable known at every
/// node, `at_nodes` in its units: each node is drawn from the collocated
/// simple cokriging of its score from its neighbors and the secondary score
/// at the node, under the Markov model, the cross-covariance the correlation
/// times the primary's. A zero correlation is plain SGS.
#[allow(clippy::too_many_arguments)]
pub fn cosgs(
    data_locs: &[(f64, f64, f64)],
    data_vals: &[f64],
    data_weights: Option<&[f64]>,
    data_holes: Option<&[u32]>,
    domains: Option<Domains>,
    trend: Option<Trend>,
    grid: &[(f64, f64, f64)],
    vg_nscore: &Variogram,
    params: &SgsParams,
    local: Option<&LocalAnisotropy>,
    secondary: &Secondary,
    at_nodes: &[f64],
) -> Result<Realization> {
    if at_nodes.len() != grid.len() {
        return Err(SimError::InvalidParameters(
            "one secondary value per node".into(),
        ));
    }
    let scores: Vec<f64> = at_nodes.iter().map(|&v| secondary.forward(v)).collect();
    simulate(
        data_locs,
        data_vals,
        data_weights,
        data_holes,
        domains,
        trend,
        grid,
        vg_nscore,
        params,
        local,
        Some((&scores, secondary.correlation)),
        None,
        |_, _, _, _| {},
    )
}

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
        None,
        grid,
        vg_nscore,
        params,
        local,
    )
}

/// As [`sgs`] with `domains` and a `trend`. Each domain has its own
/// declustered transform (see [`Transforms`]), and a node is
/// back-transformed through its domain's at its trend; `vg_nscore` is then
/// the variogram of the scores. The search's soft boundaries apply to data
/// and simulated nodes alike. A node whose domain has no data is an error.
/// With simulated domains, give each realization the node domains of its
/// own domain realization.
#[allow(clippy::too_many_arguments)]
pub fn sgs_in(
    data_locs: &[(f64, f64, f64)],
    data_vals: &[f64],
    data_weights: Option<&[f64]>,
    data_holes: Option<&[u32]>,
    domains: Option<Domains>,
    trend: Option<Trend>,
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
        trend,
        grid,
        vg_nscore,
        params,
        local,
        None,
        None,
        |_, _, _, _| {},
    )
}

/// SGS calling `used(node, neighbors, samples, kriged)` for each kriged
/// node: the indices into `samples` (data, then simulated nodes, valued in
/// grades) it was simulated from, and those samples as kriged, in scores;
/// `batch` fixes the size of the batches.
#[allow(clippy::too_many_arguments)]
fn simulate(
    data_locs: &[(f64, f64, f64)],
    data_vals: &[f64],
    data_weights: Option<&[f64]>,
    data_holes: Option<&[u32]>,
    domains: Option<Domains>,
    trend: Option<Trend>,
    grid: &[(f64, f64, f64)],
    vg_nscore: &Variogram,
    params: &SgsParams,
    local: Option<&LocalAnisotropy>,
    collocated: Option<(&[f64], f64)>,
    batch: Option<usize>,
    mut used: impl FnMut(usize, &[usize], &[Sample], &[Sample]),
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
    if trend.is_some_and(|t| t.nodes.len() != grid.len()) {
        return Err(SimError::InvalidParameters(
            "one trend value per node".into(),
        ));
    }
    if local.is_some_and(|l| l.len() != grid.len()) {
        return Err(SimError::InvalidParameters(
            "one local anisotropy per grid node".into(),
        ));
    }

    // 1. Normal-score transform within each domain.
    let fitted = Transforms::fit(
        data_vals,
        data_weights,
        domains.map(|d| d.0),
        trend.map(|t| (t.data, t.classes)),
    )?;
    if grid.is_empty() {
        return Ok(Realization { values: vec![] });
    }
    let transform = |domain: Option<u32>| {
        fitted.domains[domain.unwrap_or(0) as usize]
            .as_ref()
            .expect("checked")
    };
    let node_trend = |node: usize| trend.map_or(0.0, |t| t.nodes[node]);

    let samples = data(data_locs, data_vals, holes, domains.map(|d| d.0));
    let trees: Vec<SearchTree> = params
        .search
        .iter()
        .map(|s| tree(&samples, s, vg_nscore, local))
        .collect();
    let mut known = Known {
        samples,
        scores: fitted.scores.clone(),
        trends: trend.map_or_else(|| vec![0.0; data_locs.len()], |t| t.data.to_vec()),
        trees,
    };
    let last = known.trees.len() - 1;
    let passes: Vec<usize> = match last {
        0 => vec![0; grid.len()],
        _ => first_pass(&known.trees, grid, domains.map(|d| d.1), local)
            .into_iter()
            .map(|p| p.unwrap_or(last))
            .collect(),
    };

    // 2. Random path over grid nodes.
    let mut rng = StdRng::seed_from_u64(params.seed);
    let mut path: Vec<usize> = (0..grid.len()).collect();
    path.shuffle(&mut rng);

    // 3. Kriging from neighbors (simple kriging, mean 0 in Gaussian space).
    let step = |known: &Known, node: usize, earlier: Option<&[usize]>| -> Option<Result<Step>> {
        let target = grid[node];
        let domain = domains.map(|d| d.1[node]);
        let aniso = local.map(|l| l.anisotropy(node));
        let tree = &known.trees[passes[node]];
        let found = find(tree, &target, domain, aniso.as_ref());
        if let Some(earlier) = earlier {
            // Stale when a node simulated since `known` would join the
            // search: within the farthest neighbor of a full search, or
            // within the radius of one that is not.
            let search = &params.search[passes[node]];
            let d = |p: &(f64, f64, f64)| match &aniso {
                Some(a) => a.lag(&target, p),
                None => tree.distance(&target, p),
            };
            let reach = match &found {
                Ok(idx) if idx.len() >= search.max_samples => idx
                    .iter()
                    .map(|&k| d(&known.samples[k].loc))
                    .fold(0.0, f64::max),
                _ => search.radius,
            };
            if earlier.iter().any(|&j| d(&grid[j]) <= reach * (1.0 + 1e-9)) {
                return None;
            }
        }
        if let Some(&k) = found
            .iter()
            .flatten()
            .find(|&&k| known.samples[k].loc == target)
        {
            return Some(Ok(Step::Datum(known.samples[k].value)));
        }
        let idx = match found {
            Ok(idx) if !idx.is_empty() => idx,
            _ => return Some(Ok(Step::Marginal)),
        };
        let selected: Vec<Sample> = idx
            .iter()
            .map(|&k| Sample {
                value: match (
                    tree.cap(&target, aniso.as_ref(), k),
                    known.samples[k].domain == domain,
                ) {
                    (Some(t), _) => transform(domain).forward(t, known.trends[k]),
                    (None, true) => known.scores[k],
                    (None, false) => {
                        transform(domain).forward(known.samples[k].value, known.trends[k])
                    }
                },
                ..known.samples[k].clone()
            })
            .collect();
        let vg_node = aniso.map(|a| Variogram {
            anisotropy: Some(a),
            ..vg_nscore.clone()
        });
        let vg = vg_node.as_ref().unwrap_or(vg_nscore);
        Some(
            krige(Kind::Simple { mean: 0.0 }, &target, &selected, vg)
                .map_err(|e| SimError::Estimation(e.to_string()))
                .map(|est| Step::Kriged {
                    idx,
                    selected,
                    mean: est.value,
                    variance: est.variance.max(0.0),
                }),
        )
    };

    let sill = vg_nscore.total_sill();
    let cokriged = |mean: f64, variance: f64, node: usize| match collocated {
        Some((scores, rho)) => {
            estimation::markov_collocated(mean, variance, sill, rho, scores[node])
        }
        None => (mean, variance),
    };
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut values = vec![f64::NAN; grid.len()];
    let most = params
        .search
        .iter()
        .map(|s| s.max_samples)
        .max()
        .unwrap_or(0);
    let mut start = 0;
    while start < path.len() {
        // A batch is searched and kriged in parallel against the nodes
        // simulated before it; its stale nodes are redone in path order,
        // so the realization is the sequential one for any batch size. A
        // node goes stale with probability about size * most / 2 / known,
        // one in 16 here.
        let size = batch.unwrap_or(known.samples.len() / (8 * most).max(1));
        let nodes = &path[start..(start + size.clamp(1, BATCH)).min(path.len())];
        let steps: Vec<Option<Result<Step>>> = nodes
            .par_iter()
            .enumerate()
            .map(|(t, &node)| step(&known, node, Some(&nodes[..t])))
            .collect();
        for (&node, s) in nodes.iter().zip(steps) {
            let s = s
                .or_else(|| step(&known, node, None))
                .expect("never stale alone")?;
            let score = match s {
                // A node on a datum takes its value and is not added again.
                Step::Datum(value) => {
                    values[node] = value;
                    continue;
                }
                Step::Kriged {
                    idx,
                    selected,
                    mean,
                    variance,
                } => {
                    used(node, &idx, &known.samples, &selected);
                    let (mean, variance) = cokriged(mean, variance, node);
                    mean + variance.sqrt() * normal.sample(&mut rng)
                }
                // No neighbors found: draw from the marginal (standard normal).
                Step::Marginal => match collocated {
                    Some(_) => {
                        let (mean, variance) = cokriged(0.0, sill, node);
                        mean + variance.sqrt() * normal.sample(&mut rng)
                    }
                    None => normal.sample(&mut rng),
                },
            };

            // 4. Back-transform and add the node to the conditioning set.
            let domain = domains.map(|d| d.1[node]);
            let value = transform(domain).back(score, node_trend(node));
            values[node] = value;
            let sample = Sample {
                loc: grid[node],
                value,
                hole: None,
                error_variance: 0.0,
                domain,
            };
            for tree in &mut known.trees {
                tree.add(&sample);
            }
            known.samples.push(sample);
            known.scores.push(score);
            known.trends.push(node_trend(node));
        }
        start += nodes.len();
    }
    Ok(Realization { values })
}

/// Largest batch of nodes searched and kriged in parallel.
const BATCH: usize = 512;

/// The conditioning set in grades, growing as nodes are simulated, with
/// the scores and trend alongside and a search tree per pass.
struct Known {
    samples: Vec<Sample>,
    scores: Vec<f64>,
    trends: Vec<f64>,
    trees: Vec<SearchTree>,
}

/// A node's draw: a datum's value, or the kriged mean and standard
/// deviation of its score from neighbors `idx`, kriged as `selected`.
enum Step {
    Datum(f64),
    Kriged {
        idx: Vec<usize>,
        selected: Vec<Sample>,
        mean: f64,
        variance: f64,
    },
    Marginal,
}

/// Lengths of `domains` and a datum in the domain of every node.
pub(crate) fn check(domains: Option<Domains>, data: usize, nodes: usize) -> Result<()> {
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

pub(crate) fn data(
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

pub(crate) fn tree(
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

pub(crate) fn find(
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
pub(crate) mod tests {
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
            high_grade: Some(estimation::HighGrade::new(60.0, 6.0)),
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

    pub(crate) type Point = (f64, f64, f64);

    pub(crate) struct Zoned {
        pub(crate) locs: Vec<Point>,
        pub(crate) vals: Vec<f64>,
        pub(crate) weights: Vec<f64>,
        pub(crate) holes: Vec<u32>,
        pub(crate) codes: Vec<u32>,
        pub(crate) trend: Vec<f64>,
    }

    impl Zoned {
        pub(crate) fn of(&self, code: u32) -> Vec<usize> {
            (0..self.vals.len())
                .filter(|&i| self.codes[i] == code)
                .collect()
        }
    }

    pub(crate) fn pick<T: Copy>(v: &[T], rows: &[usize]) -> Vec<T> {
        rows.iter().map(|&i| v[i]).collect()
    }

    /// Domain 0 west of x = 50, domain 1 east and richer, both richer to the
    /// north where the weights are higher and the trend `y / 100` too; holes
    /// of 3 samples down z, and holes on the contact logging both domains at
    /// every sample.
    pub(crate) fn zoned() -> Zoned {
        use rand::Rng;
        let mut rng = StdRng::seed_from_u64(8);
        let mut z = Zoned {
            locs: vec![],
            vals: vec![],
            weights: vec![],
            holes: vec![],
            codes: vec![],
            trend: vec![],
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
                z.trend.push(y / 100.0);
            }
        }
        for h in 0..5 {
            for code in 0..2 {
                let y = 10.0 + 20.0 * h as f64;
                z.locs.push((50.0, y, 1.0));
                z.vals.push(if code == 0 { 0.5 } else { 20.0 });
                z.weights.push(1.0);
                z.holes.push(100 + h);
                z.codes.push(code);
                z.trend.push(y / 100.0);
            }
        }
        z
    }

    /// Nodes on a 4 m grid at z = 1 and on the contact holes, each in both
    /// domains, with their trend.
    pub(crate) fn zoned_grid() -> (Vec<Point>, Vec<u32>, Vec<f64>) {
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
        let trend = grid.iter().map(|p| p.1 / 100.0).collect();
        (grid, codes, trend)
    }

    pub(crate) fn zoned_search(soft: Option<estimation::Soft>) -> Vec<Search> {
        let pass = |radius, min_samples| Search {
            min_samples,
            max_samples: 12,
            radius,
            max_per_hole: Some(2),
            high_grade: Some(estimation::HighGrade::new(12.0, 6.0)),
            soft: soft.clone(),
            ..Default::default()
        };
        vec![pass(12.0, 6), pass(40.0, 2)]
    }

    pub(crate) fn vg() -> Variogram {
        Variogram::single(Model::Spherical, 1.0, 25.0)
    }

    fn distance(a: &Point, b: &Point) -> f64 {
        ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2)).sqrt()
    }

    /// Grade to score in domain `code`, fitted apart from SGS.
    pub(crate) fn forward(z: &Zoned, code: u32, trended: bool) -> impl Fn(f64, f64) -> f64 {
        let rows = z.of(code);
        let (vals, w, t) = (
            pick(&z.vals, &rows),
            pick(&z.weights, &rows),
            pick(&z.trend, &rows),
        );
        let c = trended.then(|| TrendConditioning::fit(&vals, &t, Some(&w), 3).unwrap());
        let scores = c.as_ref().map_or(vals, |c| c.scores().to_vec());
        let table = normal_score::transform(&scores, Some(&w)).unwrap().table;
        move |g, t| table.forward(c.as_ref().map_or(g, |c| c.forward_one(t, g)))
    }

    /// Neighbors of the other domain used; asserts every neighbor obeys the
    /// high-grade rule in grades and is in the node's domain or strictly
    /// within `soft`, where it is kriged as its grade transformed through
    /// the node's domain.
    fn other_domain_used(soft: Option<f64>, seed: u64, trended: bool) -> usize {
        let z = zoned();
        let (grid, nodes, node_trend) = zoned_grid();
        let trend = trended.then_some(Trend {
            data: &z.trend,
            nodes: &node_trend,
            classes: 3,
        });
        let into = [forward(&z, 0, trended), forward(&z, 1, trended)];
        let params = SgsParams {
            search: zoned_search(soft.map(estimation::Soft::All)),
            seed,
        };
        let (mut other, mut high, mut simulated) = (0, 0, 0);
        simulate(
            &z.locs,
            &z.vals,
            Some(&z.weights),
            Some(&z.holes),
            Some((&z.codes, &nodes)),
            trend,
            &grid,
            &vg(),
            &params,
            None,
            None,
            None,
            |node, idx, samples, kriged| {
                for (j, &k) in idx.iter().enumerate() {
                    let (s, d) = (&samples[k], distance(&grid[node], &samples[k].loc));
                    if s.value > 12.0 {
                        assert!(d <= 6.0, "high grade {} at {d}", s.value);
                        high += 1;
                    }
                    if s.domain != Some(nodes[node]) {
                        assert!(d < soft.unwrap_or(0.0), "other domain at {d}");
                        // Data and nodes alike have trend y / 100.
                        let t = s.loc.1 / 100.0;
                        let want = into[nodes[node] as usize](s.value, t);
                        assert!((kriged[j].value - want).abs() < 1e-12);
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

    fn clamp(s: &Search) -> Search {
        Search {
            high_grade: s.high_grade.clone().map(|h| estimation::HighGrade {
                mode: estimation::HighGradeMode::Clamp,
                ..h
            }),
            ..s.clone()
        }
    }

    #[test]
    fn clamped_high_grades_enter_as_the_score_of_the_threshold() {
        let z = zoned();
        let (grid, nodes, _) = zoned_grid();
        let into = [forward(&z, 0, false), forward(&z, 1, false)];
        let params = SgsParams {
            search: zoned_search(None).iter().map(clamp).collect(),
            seed: 5,
        };
        let mut clamped = 0;
        simulate(
            &z.locs,
            &z.vals,
            Some(&z.weights),
            Some(&z.holes),
            Some((&z.codes, &nodes)),
            None,
            &grid,
            &vg(),
            &params,
            None,
            None,
            None,
            |node, idx, samples, kriged| {
                for (j, &k) in idx.iter().enumerate() {
                    let (s, d) = (&samples[k], distance(&grid[node], &samples[k].loc));
                    if s.value > 12.0 && d > 6.0 + 1e-9 {
                        let want = into[nodes[node] as usize](12.0, 0.0);
                        assert!((kriged[j].value - want).abs() < 1e-12);
                        clamped += 1;
                    }
                }
            },
        )
        .unwrap();
        assert!(clamped > 0);
    }

    #[test]
    fn hard_boundaries_use_only_the_node_domain() {
        assert_eq!(other_domain_used(None, 1, false), 0);
        assert_eq!(other_domain_used(None, 1, true), 0);
    }

    #[test]
    fn soft_samples_enter_as_grades_through_the_node_domain() {
        assert!(other_domain_used(Some(8.0), 2, false) > 0);
        assert!(other_domain_used(Some(20.0), 3, false) > 0);
        assert!(other_domain_used(Some(8.0), 4, true) > 0);
    }

    #[test]
    fn each_realization_keeps_within_its_simulated_domains() {
        let z = zoned();
        let rows: Vec<usize> = (0..z.locs.len() - 10).collect();
        let (locs, codes) = (pick(&z.locs, &rows), pick(&z.codes, &rows));
        let grid = zoned_grid().0[..625].to_vec();
        let cats: Vec<usize> = codes.iter().map(|&c| c as usize).collect();
        let search = Search {
            max_samples: 12,
            radius: 30.0,
            ..Default::default()
        };
        let facies = |seed| {
            let params = crate::SisParams {
                search: search.clone(),
                seed,
            };
            crate::sis(&locs, &cats, None, &grid, 2, &[vg(), vg()], &params, None)
                .unwrap()
                .categories
                .into_iter()
                .map(|c| c as u32)
                .collect::<Vec<_>>()
        };
        let maps: Vec<Vec<u32>> = (0..3).map(facies).collect();
        assert!(maps[0] != maps[1] && maps[1] != maps[2]);
        for (k, nodes) in maps.iter().enumerate() {
            assert!(nodes.contains(&0) && nodes.contains(&1));
            let params = SgsParams {
                search: zoned_search(None),
                seed: k as u64,
            };
            let mut used = 0;
            simulate(
                &locs,
                &pick(&z.vals, &rows),
                None,
                Some(&pick(&z.holes, &rows)),
                Some((&codes, nodes)),
                None,
                &grid,
                &vg(),
                &params,
                None,
                None,
                None,
                |node, idx, samples, _| {
                    for &i in idx {
                        assert_eq!(samples[i].domain, Some(nodes[node]));
                        used += 1;
                    }
                },
            )
            .unwrap();
            assert!(used > 0);
        }
    }

    fn run(
        z: &Zoned,
        rows: &[usize],
        grid: &[Point],
        domains: Option<Domains>,
        trend: Option<(&[f64], &[f64])>,
        search: &[Search],
        seed: u64,
    ) -> Vec<f64> {
        let t = trend.map(|(_, nodes)| (pick(&z.trend, rows), nodes));
        let params = SgsParams {
            search: search.to_vec(),
            seed,
        };
        sgs_in(
            &pick(&z.locs, rows),
            &pick(&z.vals, rows),
            Some(&pick(&z.weights, rows)),
            Some(&pick(&z.holes, rows)),
            domains,
            t.as_ref().map(|(data, nodes)| Trend {
                data,
                nodes,
                classes: 3,
            }),
            grid,
            &vg(),
            &params,
            None,
        )
        .unwrap()
        .values
    }

    #[test]
    fn one_label_everywhere_is_no_domains() {
        let z = zoned();
        let (grid, _, node_trend) = zoned_grid();
        let search = zoned_search(Some(estimation::Soft::All(5.0)));
        // Without the contact holes, which put two samples at one location.
        let rows: Vec<usize> = (0..z.locs.len() - 10).collect();
        let (one, all) = (vec![0; rows.len()], vec![0; grid.len()]);
        let domains = Some((&one[..], &all[..]));
        for trend in [None, Some((&z.trend[..], &node_trend[..]))] {
            for seed in 0..3 {
                assert_eq!(
                    run(&z, &rows, &grid, None, trend, &search, seed),
                    run(&z, &rows, &grid, domains, trend, &search, seed)
                );
            }
        }
        let passes = |domains| {
            sgs_passes(
                &pick(&z.locs, &rows),
                &pick(&z.vals, &rows),
                Some(&pick(&z.holes, &rows)),
                domains,
                &grid,
                &vg(),
                &search,
                None,
            )
            .unwrap()
        };
        assert_eq!(passes(None), passes(domains));
    }

    #[test]
    fn a_hard_domain_is_simulated_as_if_alone() {
        let z = zoned();
        let (grid, _, node_trend) = zoned_grid();
        let all: Vec<usize> = (0..z.locs.len()).collect();
        let west = z.of(0);
        let nodes = vec![0; grid.len()];
        let search = zoned_search(None);
        for trend in [None, Some((&z.trend[..], &node_trend[..]))] {
            let both = run(&z, &all, &grid, Some((&z.codes, &nodes)), trend, &search, 6);
            assert_eq!(both, run(&z, &west, &grid, None, trend, &search, 6));
        }
    }

    pub(crate) fn mean(v: &[f64], w: &[f64]) -> f64 {
        v.iter().zip(w).map(|(v, w)| v * w).sum::<f64>() / w.iter().sum::<f64>()
    }

    #[test]
    fn soft_moves_the_contact_grade_as_kriging_does() {
        // Rich domain 0 west of x = 50 and lean domain 1 east, sampled every
        // 5 m on three lines; nodes of domain 0 just west of the contact.
        let (mut locs, mut vals, mut codes) = (vec![], vec![], vec![]);
        for i in 0..60 {
            let (x, y) = ((i % 20) as f64 * 5.0 + 2.5, (i / 20) as f64 * 10.0);
            let code = u32::from(x > 50.0);
            locs.push((x, y, 0.0));
            vals.push(if code == 0 { 10.0 } else { 1.0 } * (0.3 * (i as f64).sin()).exp());
            codes.push(code);
        }
        let grid: Vec<Point> = (0..12)
            .map(|i| (46.0 + (i % 4) as f64, (i / 4) as f64 * 10.0, 0.0))
            .collect();
        let nodes = vec![0; grid.len()];
        let search = |soft| Search {
            min_samples: 1,
            max_samples: 6,
            radius: 30.0,
            soft,
            ..Default::default()
        };
        let simulated = |soft| {
            let search = vec![search(soft)];
            let params = |seed| SgsParams {
                search: search.clone(),
                seed,
            };
            let total: f64 = (0..50)
                .map(|seed| {
                    let domains = Some((&codes[..], &nodes[..]));
                    let r = sgs_in(
                        &locs,
                        &vals,
                        None,
                        None,
                        domains,
                        None,
                        &grid,
                        &vg(),
                        &params(seed),
                        None,
                    );
                    r.unwrap().values.iter().sum::<f64>()
                })
                .sum();
            total / (50 * grid.len()) as f64
        };
        let samples: Vec<Sample> = (0..locs.len())
            .map(|i| Sample {
                domain: Some(codes[i]),
                ..Sample::new(locs[i], vals[i])
            })
            .collect();
        let kriged = |soft| {
            let at = estimation::estimate_many(
                &grid,
                Some(&nodes),
                &samples,
                &search(soft),
                Some(&vg()),
                |t, n| krige(Kind::Ordinary, t, n, &vg()),
            );
            at.iter().map(|e| e.as_ref().unwrap().value).sum::<f64>() / grid.len() as f64
        };
        let soft = Some(estimation::Soft::All(10.0));
        let (sgs_shift, kriging_shift) = (
            simulated(soft.clone()) / simulated(None) - 1.0,
            kriged(soft) / kriged(None) - 1.0,
        );
        let shifts = format!("SGS {sgs_shift}, kriging {kriging_shift}");
        assert!(kriging_shift < -0.03 && sgs_shift < -0.03, "{shifts}");
    }

    /// Nodes far apart and far from the data, `per` in each domain, at trend
    /// `trend(code)`.
    pub(crate) fn far(per: usize) -> (Vec<Point>, Vec<u32>) {
        let grid = (0..2 * per)
            .map(|i| (1e4 * (1 + i / per) as f64 + 100.0 * i as f64, 1e4, 0.0))
            .collect();
        (grid, (0..2 * per).map(|i| (i / per) as u32).collect())
    }

    pub(crate) fn pooled(reals: &[Vec<f64>], range: std::ops::Range<usize>) -> Vec<f64> {
        let mut v: Vec<f64> = reals
            .iter()
            .flat_map(|r| r[range.clone()].to_vec())
            .collect();
        v.sort_by(f64::total_cmp);
        v
    }

    #[test]
    fn each_domain_reproduces_its_declustered_histogram() {
        let z = zoned();
        let per = 30;
        let (grid, nodes) = far(per);
        let all: Vec<usize> = (0..z.locs.len()).collect();
        let search = zoned_search(None);
        let reals: Vec<Vec<f64>> = (0..100)
            .map(|seed| {
                run(
                    &z,
                    &all,
                    &grid,
                    Some((&z.codes, &nodes)),
                    None,
                    &search,
                    seed,
                )
            })
            .collect();
        for code in 0..2 {
            let rows = z.of(code);
            let (data, w) = (pick(&z.vals, &rows), pick(&z.weights, &rows));
            let pooled = pooled(&reals, code as usize * per..(code as usize + 1) * per);
            let (want, naive) = (mean(&data, &w), mean(&data, &vec![1.0; data.len()]));
            let got = mean(&pooled, &vec![1.0; pooled.len()]);
            assert!((want / naive - 1.0).abs() > 0.1, "weights change nothing");
            assert!(
                (got / want - 1.0).abs() < 0.05,
                "domain {code}: {got} vs {want}"
            );
            let table = normal_score::transform(&data, Some(&w)).unwrap().table;
            for (q, s) in [
                (0.1, -1.281_551_565_545),
                (0.5, 0.0),
                (0.9, 1.281_551_565_545),
            ] {
                let (got, want) = (pooled[(q * pooled.len() as f64) as usize], table.back(s));
                assert!(
                    (got / want).ln().abs() < 0.1,
                    "domain {code} q{q}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn each_domain_reproduces_its_histogram_in_each_trend_class() {
        let z = zoned();
        let (classes, per) = (3, 20);
        // Members of each class of each domain: equal declustered
        // probability by trend, a datum in the class of its midpoint.
        let members: Vec<Vec<Vec<usize>>> = (0..2)
            .map(|code| {
                let mut rows = z.of(code);
                rows.sort_by(|&a, &b| z.trend[a].total_cmp(&z.trend[b]));
                let total: f64 = rows.iter().map(|&i| z.weights[i]).sum();
                let mut cum = 0.0;
                let mut members = vec![vec![]; classes];
                for &i in &rows {
                    let p = cum + z.weights[i] / total / 2.0;
                    cum += z.weights[i] / total;
                    members[((p * classes as f64) as usize).min(classes - 1)].push(i);
                }
                members
            })
            .collect();
        let groups = 2 * classes;
        let grid: Vec<Point> = (0..groups * per)
            .map(|i| (1e4 + 100.0 * i as f64, 1e4, 0.0))
            .collect();
        let nodes: Vec<u32> = (0..grid.len())
            .map(|i| (i / per / classes) as u32)
            .collect();
        let node_trend: Vec<f64> = (0..grid.len())
            .map(|i| {
                let m = &members[i / per / classes][i / per % classes];
                z.trend[m[m.len() / 2]]
            })
            .collect();
        let all: Vec<usize> = (0..z.locs.len()).collect();
        let search = zoned_search(None);
        let reals: Vec<Vec<f64>> = (0..100)
            .map(|seed| {
                let trend = Some((&z.trend[..], &node_trend[..]));
                run(
                    &z,
                    &all,
                    &grid,
                    Some((&z.codes, &nodes)),
                    trend,
                    &search,
                    seed,
                )
            })
            .collect();
        for g in 0..groups {
            let rows = &members[g / classes][g % classes];
            let want = mean(&pick(&z.vals, rows), &pick(&z.weights, rows));
            let got = pooled(&reals, g * per..(g + 1) * per);
            let got = mean(&got, &vec![1.0; got.len()]);
            assert!(
                (got / want - 1.0).abs() < 0.1,
                "domain {} class {}: {got} vs {want}",
                g / classes,
                g % classes
            );
        }
    }

    #[test]
    fn on_the_contact_a_node_takes_its_own_domain_datum() {
        let z = zoned();
        let (grid, nodes, node_trend) = zoned_grid();
        let all: Vec<usize> = (0..z.locs.len()).collect();
        for trend in [None, Some((&z.trend[..], &node_trend[..]))] {
            for soft in [None, Some(estimation::Soft::All(f64::INFINITY))] {
                let r = run(
                    &z,
                    &all,
                    &grid,
                    Some((&z.codes, &nodes)),
                    trend,
                    &zoned_search(soft),
                    5,
                );
                for (i, &code) in nodes.iter().enumerate().skip(625) {
                    assert_eq!(r[i], if code == 0 { 0.5 } else { 20.0 });
                }
            }
        }
    }

    #[test]
    fn realizations_follow_the_seed_not_threads_or_batches() {
        let z = zoned();
        let (grid, nodes, node_trend) = zoned_grid();
        let local = LocalAnisotropy::new(
            grid.clone(),
            (0..grid.len())
                .map(|i| [(i * 37 % 180) as f64, 0.0, 0.0])
                .collect(),
            vec![[0.5, 1.0]; grid.len()],
        )
        .unwrap();
        let trend = Trend {
            data: &z.trend,
            nodes: &node_trend,
            classes: 3,
        };
        let soft = zoned_search(Some(estimation::Soft::All(8.0)));
        let octant: Vec<Search> = soft
            .iter()
            .map(|s| Search {
                octant: true,
                ..s.clone()
            })
            .collect();
        let clamped: Vec<Search> = soft.iter().map(clamp).collect();
        for search in [soft, octant, clamped] {
            for local in [None, Some(&local)] {
                let run = |threads, batch, seed| {
                    let params = SgsParams {
                        search: search.clone(),
                        seed,
                    };
                    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads);
                    pool.build().unwrap().install(|| {
                        simulate(
                            &z.locs,
                            &z.vals,
                            Some(&z.weights),
                            Some(&z.holes),
                            Some((&z.codes, &nodes)),
                            Some(trend),
                            &grid,
                            &vg(),
                            &params,
                            local,
                            None,
                            batch,
                            |_, _, _, _| {},
                        )
                        .unwrap()
                        .values
                    })
                };
                let sequential = run(1, Some(1), 3);
                for threads in [1, 2, 8] {
                    for batch in [None, Some(16), Some(BATCH)] {
                        assert_eq!(sequential, run(threads, batch, 3));
                    }
                }
                assert_ne!(sequential, run(8, None, 4));
            }
        }
    }

    #[test]
    fn bad_domains_and_trends_are_errors() {
        let z = zoned();
        let grid = vec![(1.0, 1.0, 1.0), (2.0, 2.0, 1.0)];
        let search = zoned_search(None);
        let params = SgsParams {
            search: search.clone(),
            seed: 1,
        };
        let run = |nodes: &[u32], codes: &[u32], trend: Option<Trend>| {
            sgs_in(
                &z.locs,
                &z.vals,
                None,
                None,
                Some((codes, nodes)),
                trend,
                &grid,
                &vg(),
                &params,
                None,
            )
        };
        let trend = |data, nodes| {
            Some(Trend {
                data,
                nodes,
                classes: 3,
            })
        };
        assert!(run(&[0, 1], &z.codes, None).is_ok());
        assert!(run(&[0, 1], &z.codes, trend(&z.trend, &[0.1, 0.2])).is_ok());
        assert!(run(&[0, 2], &z.codes, None).is_err());
        assert!(run(&[0], &z.codes, None).is_err());
        assert!(run(&[0, 1], &z.codes[1..], None).is_err());
        assert!(run(&[0, 1], &z.codes, trend(&z.trend, &[0.1])).is_err());
        assert!(run(&[0, 1], &z.codes, trend(&z.trend[1..], &[0.1, 0.2])).is_err());
        let domains = Some((&z.codes[..], &[0, 2][..]));
        let passes = sgs_passes(&z.locs, &z.vals, None, domains, &grid, &vg(), &search, None);
        assert!(passes.is_err());
    }

    /// A secondary field simulated on a 30 x 30 grid of 2 m, and primary data
    /// at 60 of its nodes whose scores correlate at `rho` with its scores.
    struct Cosim {
        grid: Vec<Point>,
        secondary: Vec<f64>,
        locs: Vec<Point>,
        vals: Vec<f64>,
        at_data: Vec<f64>,
        table: Secondary,
    }

    fn cosim(rho: f64) -> Cosim {
        use rand::Rng;
        let mut rng = StdRng::seed_from_u64(11);
        let grid: Vec<Point> = (0..900)
            .map(|i| ((i % 30) as f64 * 2.0, (i / 30) as f64 * 2.0, 0.0))
            .collect();
        let seeds: Vec<Point> = (0..30)
            .map(|_| (rng.r#gen::<f64>() * 60.0, rng.r#gen::<f64>() * 60.0, 0.0))
            .collect();
        let seed_vals: Vec<f64> = (0..30)
            .map(|_| rng.sample::<f64, _>(rand_distr::StandardNormal).exp())
            .collect();
        let search = vec![Search {
            max_samples: 16,
            radius: 40.0,
            ..Default::default()
        }];
        let params = SgsParams { search, seed: 5 };
        let secondary = sgs(&seeds, &seed_vals, None, None, &grid, &vg(), &params, None)
            .unwrap()
            .values;
        let table = Secondary::fit(&secondary, None, &[], Some(rho)).unwrap();
        let nodes: Vec<usize> = (0..60).map(|i| i * 887 % 900).collect();
        let vals = nodes
            .iter()
            .map(|&k| {
                let g: f64 = rng.sample(rand_distr::StandardNormal);
                (rho * table.forward(secondary[k]) + (1.0 - rho * rho).sqrt() * g).exp()
            })
            .collect();
        Cosim {
            locs: pick(&grid, &nodes),
            at_data: pick(&secondary, &nodes),
            grid,
            secondary,
            vals,
            table,
        }
    }

    fn cosimulate(c: &Cosim, secondary: &Secondary, seed: u64) -> Vec<f64> {
        let search = vec![Search {
            max_samples: 16,
            radius: 40.0,
            ..Default::default()
        }];
        let params = SgsParams { search, seed };
        cosgs(
            &c.locs,
            &c.vals,
            None,
            None,
            None,
            None,
            &c.grid,
            &vg(),
            &params,
            None,
            secondary,
            &c.secondary,
        )
        .unwrap()
        .values
    }

    #[test]
    fn cosimulation_reproduces_the_correlation_and_honors_the_data() {
        let c = cosim(0.7);
        let primary = Transforms::fit(&c.vals, None, None, None).unwrap().domains[0]
            .clone()
            .unwrap();
        let s: Vec<f64> = c.secondary.iter().map(|&v| c.table.forward(v)).collect();
        let independent = Secondary::fit(&c.secondary, None, &[], Some(0.0)).unwrap();
        let (mut with, mut without) = (0.0, 0.0);
        for seed in 0..10 {
            let r = cosimulate(&c, &c.table, seed);
            for (k, v) in c.vals.iter().enumerate() {
                assert_eq!(r[(k * 887) % 900], *v);
            }
            let scores: Vec<f64> = r.iter().map(|&v| primary.forward(v, 0.0)).collect();
            with += pearson(&scores, &s, None) / 10.0;
            let r = cosimulate(&c, &independent, seed);
            let scores: Vec<f64> = r.iter().map(|&v| primary.forward(v, 0.0)).collect();
            without += pearson(&scores, &s, None) / 10.0;
        }
        assert!(
            (with - 0.7).abs() < 0.1,
            "correlation {with}, independent {without}"
        );
        assert!(
            without < with - 0.15,
            "independent {without}, collocated {with}"
        );
    }

    #[test]
    fn zero_correlation_is_plain_sgs() {
        let z = zoned();
        let (grid, nodes, node_trend) = zoned_grid();
        let trend = Trend {
            data: &z.trend,
            nodes: &node_trend,
            classes: 3,
        };
        let secondary: Vec<f64> = grid.iter().map(|p| p.0.sin()).collect();
        let zero = Secondary::fit(&secondary, None, &[], Some(0.0)).unwrap();
        for seed in 0..3 {
            let params = SgsParams {
                search: zoned_search(Some(estimation::Soft::All(8.0))),
                seed,
            };
            let domains = Some((&z.codes[..], &nodes[..]));
            let plain = sgs_in(
                &z.locs,
                &z.vals,
                Some(&z.weights),
                Some(&z.holes),
                domains,
                Some(trend),
                &grid,
                &vg(),
                &params,
                None,
            )
            .unwrap();
            let co = cosgs(
                &z.locs,
                &z.vals,
                Some(&z.weights),
                Some(&z.holes),
                domains,
                Some(trend),
                &grid,
                &vg(),
                &params,
                None,
                &zero,
                &secondary,
            )
            .unwrap();
            assert_eq!(plain.values, co.values);
        }
    }

    #[test]
    fn cosimulation_follows_the_seed_not_the_thread_count() {
        let c = cosim(0.6);
        let run = |threads, seed| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads);
            pool.build()
                .unwrap()
                .install(|| cosimulate(&c, &c.table, seed))
        };
        let a = run(1, 3);
        assert_eq!(a, run(8, 3));
        assert_ne!(a, run(8, 4));
    }

    #[test]
    fn secondary_fits_the_correlation_of_the_scores() {
        let c = cosim(0.8);
        let scores = Transforms::fit(&c.vals, None, None, None).unwrap().scores;
        let fitted = Secondary::fit(&c.at_data, None, &scores, None).unwrap();
        assert!(
            (fitted.correlation - 0.8).abs() < 0.15,
            "{}",
            fitted.correlation
        );
        assert!(Secondary::fit(&c.at_data, None, &scores[1..], None).is_err());
        assert!(Secondary::fit(&c.at_data, None, &scores, Some(1.2)).is_err());
        assert!(Secondary::fit(&c.at_data, Some(&[1.0]), &scores, None).is_err());
        let grid = &c.grid[..3];
        let params = SgsParams {
            search: zoned_search(None),
            seed: 1,
        };
        let short = cosgs(
            &c.locs,
            &c.vals,
            None,
            None,
            None,
            None,
            grid,
            &vg(),
            &params,
            None,
            &fitted,
            &c.secondary,
        );
        assert!(short.is_err());
    }
}
