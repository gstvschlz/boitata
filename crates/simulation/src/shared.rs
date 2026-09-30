//! SGS along one multigrid path shared by every realization. A node's
//! neighbors (data and nodes earlier on the path) and its simple-kriging
//! weights depend only on the path, so they are found once for a batch of
//! realizations and applied to all of them; only neighbor values and noise
//! differ. Noise is drawn from the realization's seed and the node, so a
//! realization does not depend on the batch, the chunking or the threads.

use std::ops::Range;

use boitata_core::rng::{gaussian, realization_seed};
use estimation::Sample;
use estimation::krige::{Kind, krige};
use estimation::search::{Search, SearchTree};
use rayon::prelude::*;
use variogram::Variogram;

use crate::error::{Result, SimError};
use crate::lattice::{Lattice, Template, default_levels, multigrid_path};
use crate::sgs::{Domains, Transforms, Trend, check, data, find, tree};

/// What is shared by the plans of one batch.
pub(crate) struct Context<'a> {
    pub lattice: &'a Lattice,
    pub samples: Vec<Sample>,
    /// Normal scores of the data, each in its own domain.
    pub scores: Vec<f64>,
    /// One search tree and one template per pass.
    pub trees: Vec<SearchTree>,
    pub templates: Vec<Template>,
    pub search: &'a [Search],
    pub vg: &'a Variogram,
    pub node_domains: Option<&'a [u32]>,
    pub rank: Vec<u32>,
    /// Variance of a draw without neighbors.
    pub marginal: f64,
}

/// A node's draw in every realization: `constant` (the weighted data
/// scores) plus the weighted scores of `nodes`, plus the square root of
/// `variance` times the node's noise.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Plan {
    pub constant: f64,
    pub nodes: Vec<(u32, f64)>,
    pub variance: f64,
}

/// Nodes whose cell centroid is a datum of their domain, with that datum;
/// the first datum of each node, sorted by node.
pub(crate) fn datum_nodes(
    lattice: &Lattice,
    locs: &[(f64, f64, f64)],
    domains: Option<Domains>,
) -> Vec<(usize, usize)> {
    let mut on: Vec<(usize, usize)> = locs
        .iter()
        .enumerate()
        .filter_map(|(k, &p)| {
            let (cell, _) = lattice.geometry().locate([p.0, p.1, p.2])?;
            let node = lattice.node_at(cell)?;
            let same = domains.is_none_or(|(of_data, of_nodes)| of_data[k] == of_nodes[node]);
            (same && lattice.location(node) == p).then_some((node, k))
        })
        .collect();
    on.sort_unstable();
    on.dedup_by_key(|d| d.0);
    on
}

/// The plan of `node`: the pass is the first whose tree finds data, or the
/// last; its neighbors the nearest `max_samples` of those data and of the
/// nodes earlier on the path within its template.
pub(crate) fn plan(ctx: &Context, node: usize) -> Result<Plan> {
    let target = ctx.lattice.location(node);
    let domain = ctx.node_domains.map(|d| d[node]);
    let last = ctx.trees.len() - 1;
    let (pass, found) = ctx
        .trees
        .iter()
        .enumerate()
        .find_map(|(p, t)| {
            find(t, &target, domain, None)
                .ok()
                .filter(|f| !f.is_empty())
                .map(|f| (p, f))
        })
        .unwrap_or((last, vec![]));
    let tree = &ctx.trees[pass];
    let max = ctx.search[pass].max_samples;
    let own = ctx.rank[node];
    // (distance, is a node, index)
    let mut near: Vec<(f64, bool, u32)> = found
        .iter()
        .map(|&k| (tree.distance(&target, &ctx.samples[k].loc), false, k as u32))
        .collect();
    let mut taken = 0;
    for &(delta, d) in ctx.templates[pass].offsets() {
        if taken == max {
            break;
        }
        let Some(m) = ctx.lattice.shifted(node, delta) else {
            continue;
        };
        if ctx.rank[m] >= own || ctx.node_domains.is_some_and(|nd| Some(nd[m]) != domain) {
            continue;
        }
        near.push((d, true, m as u32));
        taken += 1;
    }
    near.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    near.truncate(max);
    if near.is_empty() {
        return Ok(Plan {
            constant: 0.0,
            nodes: vec![],
            variance: ctx.marginal,
        });
    }
    let selected: Vec<Sample> = near
        .iter()
        .map(|&(_, is_node, i)| {
            let loc = match is_node {
                true => ctx.lattice.location(i as usize),
                false => ctx.samples[i as usize].loc,
            };
            Sample::new(loc, 0.0)
        })
        .collect();
    let est = krige(Kind::Simple { mean: 0.0 }, &target, &selected, ctx.vg)
        .map_err(|e| SimError::Estimation(e.to_string()))?;
    let mut constant = 0.0;
    let mut nodes = Vec::with_capacity(near.len());
    for (&(_, is_node, i), &w) in near.iter().zip(&est.weights) {
        match is_node {
            true => nodes.push((i, w)),
            false => constant += w * ctx.scores[i as usize],
        }
    }
    Ok(Plan {
        constant,
        nodes,
        variance: est.variance.max(0.0),
    })
}

/// Where and how a shared path runs: the lattice of the nodes, the search
/// passes, the multigrid levels (`None` sizes them from the first pass) and
/// the user seed.
pub struct SharedSgs<'a> {
    pub lattice: &'a Lattice,
    pub search: &'a [Search],
    pub levels: Option<usize>,
    pub seed: u64,
}

/// A secondary variable for collocated cosimulation: its normal scores at
/// every node, one row for all realizations or one per realization
/// (realization `k` takes row `k % rows`), and its correlation with the primary.
pub struct Collocated<'a> {
    pub scores: &'a [Vec<f64>],
    pub correlation: f64,
}

/// Nodes planned and drawn together; plans live only for their chunk.
const CHUNK: usize = 1 << 16;

/// Realizations of a batch, in scores until back-transformed one at a time.
pub struct SharedBatch<'a> {
    /// Node-major: the scores of node `m` are `scores[m * batch..][..batch]`.
    scores: Vec<f32>,
    batch: usize,
    transforms: Transforms,
    node_domains: Option<&'a [u32]>,
    trend: Option<Trend<'a>>,
    /// Nodes on a datum and the datum's grade.
    on_datum: Vec<(usize, f64)>,
}

impl SharedBatch<'_> {
    pub fn len(&self) -> usize {
        self.batch
    }

    pub fn is_empty(&self) -> bool {
        self.batch == 0
    }

    /// Realization `b` of the batch in grades, back-transformed through each
    /// node's domain at its trend; a node on a datum holds the datum's grade.
    pub fn realization(&self, b: usize) -> Vec<f64> {
        let nodes = self.scores.len() / self.batch;
        let mut values: Vec<f64> = (0..nodes)
            .into_par_iter()
            .map(|m| {
                let code = self.node_domains.map_or(0, |d| d[m]) as usize;
                let t = self.transforms.domains[code].as_ref().expect("checked");
                let trend = self.trend.map_or(0.0, |t| t.nodes[m]);
                t.back(self.scores[m * self.batch + b] as f64, trend)
            })
            .collect();
        for &(m, v) in &self.on_datum {
            values[m] = v;
        }
        values
    }
}

/// Realizations `realizations` of SGS over `shared.lattice` along one
/// multigrid path. `vg_nscore`, the data, `domains` and `trend` are as in
/// [`crate::sgs_in`]; `collocated` cosimulates with a secondary variable
/// under the Markov model, as [`crate::cosgs`] does.
#[allow(clippy::too_many_arguments)]
pub fn sgs_shared<'a>(
    data_locs: &[(f64, f64, f64)],
    data_vals: &[f64],
    data_weights: Option<&[f64]>,
    data_holes: Option<&[u32]>,
    domains: Option<Domains<'a>>,
    trend: Option<Trend<'a>>,
    vg_nscore: &Variogram,
    shared: &SharedSgs,
    realizations: Range<usize>,
    collocated: Option<&Collocated>,
) -> Result<SharedBatch<'a>> {
    if let Some(why) = shared_unsupported(shared.search, false, false) {
        return Err(SimError::InvalidParameters(format!(
            "a shared path does not support {why}"
        )));
    }
    let lattice = shared.lattice;
    let n = lattice.len();
    if data_locs.len() != data_vals.len() {
        return Err(SimError::InvalidParameters("data length mismatch".into()));
    }
    if data_locs.is_empty() {
        return Err(SimError::InsufficientData("no conditioning data".into()));
    }
    if shared.search.is_empty() {
        return Err(SimError::InvalidParameters(
            "need at least one search".into(),
        ));
    }
    if realizations.is_empty() {
        return Err(SimError::InvalidParameters(
            "need at least one realization".into(),
        ));
    }
    check(domains, data_locs.len(), n)?;
    if trend.is_some_and(|t| t.nodes.len() != n) {
        return Err(SimError::InvalidParameters(
            "one trend value per node".into(),
        ));
    }
    if collocated.is_some_and(|c| c.scores.is_empty() || c.scores.iter().any(|r| r.len() != n)) {
        return Err(SimError::InvalidParameters(
            "one secondary value per node".into(),
        ));
    }
    let holes = crate::holes(data_holes, data_locs.len())?;
    let transforms = Transforms::fit(
        data_vals,
        data_weights,
        domains.map(|d| d.0),
        trend.map(|t| (t.data, t.classes)),
    )?;
    let samples = data(data_locs, data_vals, holes, domains.map(|d| d.0));
    let trees: Vec<SearchTree> = shared
        .search
        .iter()
        .map(|s| tree(&samples, s, vg_nscore, None))
        .collect();
    let templates = trees
        .iter()
        .zip(shared.search)
        .map(|(t, s)| Template::new(lattice, t.frame(), s.radius))
        .collect::<Result<Vec<_>>>()?;
    let top = shared
        .levels
        .unwrap_or_else(|| default_levels(lattice, &templates[0]));
    let on_datum = datum_nodes(lattice, data_locs, domains);
    let mut skip = vec![false; n];
    for &(m, _) in &on_datum {
        skip[m] = true;
    }
    let (path, rank) = multigrid_path(lattice, top, shared.seed, &skip)?;
    drop(skip);

    let batch = realizations.len();
    let keys: Vec<u64> = realizations
        .clone()
        .map(|k| realization_seed(shared.seed, k as u64))
        .collect();
    let mut scores = vec![0f32; n * batch];
    for &(m, k) in &on_datum {
        scores[m * batch..(m + 1) * batch].fill(transforms.scores[k] as f32);
    }
    let sill = vg_nscore.total_sill();
    let ctx = Context {
        lattice,
        samples,
        scores: transforms.scores.clone(),
        trees,
        templates,
        search: shared.search,
        vg: vg_nscore,
        node_domains: domains.map(|d| d.1),
        rank,
        marginal: if collocated.is_some() { sill } else { 1.0 },
    };
    for chunk in path.chunks(CHUNK) {
        let plans: Vec<Plan> = chunk
            .par_iter()
            .map(|&m| plan(&ctx, m as usize))
            .collect::<Result<_>>()?;
        for wave in waves(chunk, &plans, &ctx.rank) {
            let drawn: Vec<Vec<f32>> = wave
                .par_iter()
                .map(|&t| {
                    let m = chunk[t] as usize;
                    draw(
                        &plans[t],
                        m,
                        &scores,
                        &keys,
                        realizations.start,
                        collocated,
                        sill,
                    )
                })
                .collect();
            for (&t, v) in wave.iter().zip(drawn) {
                let m = chunk[t] as usize;
                scores[m * batch..(m + 1) * batch].copy_from_slice(&v);
            }
        }
    }
    let on_datum = on_datum.iter().map(|&(m, k)| (m, data_vals[k])).collect();
    Ok(SharedBatch {
        scores,
        batch,
        transforms,
        node_domains: domains.map(|d| d.1),
        trend,
        on_datum,
    })
}

/// Positions in `chunk` grouped into waves: a node's wave is one past the
/// latest wave of its neighbors within the chunk, so the nodes of a wave
/// depend only on earlier waves and chunks and draw in parallel.
fn waves(chunk: &[u32], plans: &[Plan], rank: &[u32]) -> Vec<Vec<usize>> {
    let first = rank[chunk[0] as usize];
    let mut wave = vec![0usize; chunk.len()];
    let mut out: Vec<Vec<usize>> = vec![];
    for t in 0..chunk.len() {
        wave[t] = plans[t]
            .nodes
            .iter()
            .filter_map(|&(m, _)| rank[m as usize].checked_sub(first))
            .map(|i| wave[i as usize] + 1)
            .max()
            .unwrap_or(0);
        if out.len() <= wave[t] {
            out.resize(wave[t] + 1, vec![]);
        }
        out[wave[t]].push(t);
    }
    out
}

/// The scores of `node` in every realization of the batch.
fn draw(
    plan: &Plan,
    node: usize,
    scores: &[f32],
    keys: &[u64],
    start: usize,
    collocated: Option<&Collocated>,
    sill: f64,
) -> Vec<f32> {
    let batch = keys.len();
    keys.iter()
        .enumerate()
        .map(|(b, &key)| {
            let mean = plan.constant
                + plan
                    .nodes
                    .iter()
                    .map(|&(m, w)| w * scores[m as usize * batch + b] as f64)
                    .sum::<f64>();
            let (mean, variance) = match collocated {
                Some(c) => {
                    let row = &c.scores[(start + b) % c.scores.len()];
                    estimation::markov_collocated(
                        mean,
                        plan.variance,
                        sill,
                        c.correlation,
                        row[node],
                    )
                }
                None => (mean, plan.variance),
            };
            (mean + variance.sqrt() * gaussian(realization_seed(key, node as u64))) as f32
        })
        .collect()
}

/// Why a shared path cannot simulate these inputs, if it cannot: the first
/// of the features it leaves to the random path.
pub fn shared_unsupported(
    search: &[Search],
    local_anisotropy: bool,
    per_realization_domains: bool,
) -> Option<&'static str> {
    if local_anisotropy {
        return Some("local anisotropy");
    }
    if per_realization_domains {
        return Some("simulated domains");
    }
    if search.iter().any(|s| s.octant) {
        return Some("octant searches");
    }
    if search.iter().any(|s| s.high_grade.is_some()) {
        return Some("high-grade restrictions");
    }
    if search.iter().any(|s| s.soft.is_some()) {
        return Some("soft boundaries");
    }
    None
}

/// Realizations per batch (1 to `n`) whose shared-path simulation and
/// summary of `nodes` nodes fit `budget` bytes, with `cutoffs` cutoffs,
/// stored quantiles and `kept` kept realizations as asked. Errors, naming
/// the memory needed, when what the run holds whatever the batch (paths,
/// summaries, quantiles, kept realizations) plus one realization in flight
/// exceeds `budget`.
pub fn shared_batch(
    nodes: usize,
    n: usize,
    cutoffs: usize,
    quantiles: bool,
    kept: usize,
    budget: u64,
) -> Result<usize> {
    let (nodes, n) = (nodes as u64, n as u64);
    let fixed = nodes
        * (8 + 16 + 8 + 12 * cutoffs as u64 + if quantiles { 8 * n } else { 0 } + 8 * kept as u64);
    let each = 4 * nodes.max(1);
    if fixed + each > budget {
        return Err(SimError::InvalidParameters(format!(
            "this run needs about {:.1} GB for {nodes} nodes with one realization in flight, above the {:.1} GB budget; drop quantiles, keep fewer realizations or simulate fewer nodes",
            (fixed + each) as f64 / 1e9,
            budget as f64 / 1e9,
        )));
    }
    Ok(((budget - fixed) / each).clamp(1, n.max(1)) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use boitata_core::Geometry;
    use variogram::model::Model;

    pub(crate) fn square(n: usize) -> Lattice {
        Lattice::regular(Geometry {
            origin: [0.0; 3],
            size: [1.0, 1.0, 1.0],
            count: [n, n, 1],
            rotation: [0.0; 3],
        })
    }

    fn context<'a>(
        l: &'a Lattice,
        search: &'a [Search],
        vg: &'a Variogram,
        locs: &[(f64, f64, f64)],
    ) -> Context<'a> {
        let vals: Vec<f64> = (0..locs.len()).map(|i| i as f64).collect();
        let samples = data(locs, &vals, vec![None; locs.len()], None);
        let scores = Transforms::fit(&vals, None, None, None).unwrap().scores;
        let trees: Vec<SearchTree> = search.iter().map(|s| tree(&samples, s, vg, None)).collect();
        let templates = trees
            .iter()
            .zip(search)
            .map(|(t, s)| Template::new(l, t.frame(), s.radius).unwrap())
            .collect();
        let on = datum_nodes(l, locs, None);
        let skip: Vec<bool> = (0..l.len()).map(|m| on.iter().any(|d| d.0 == m)).collect();
        let (_, rank) = multigrid_path(l, 2, 3, &skip).unwrap();
        Context {
            lattice: l,
            samples,
            scores,
            trees,
            templates,
            search,
            vg,
            node_domains: None,
            rank,
            marginal: 1.0,
        }
    }

    #[test]
    fn a_plan_uses_data_and_earlier_nodes_only() {
        let l = square(20);
        let search = [Search {
            min_samples: 1,
            max_samples: 8,
            radius: 6.0,
            ..Default::default()
        }];
        let vg = Variogram::single(Model::Spherical, 1.0, 8.0);
        let locs = [(3.3, 4.7, 0.5), (15.2, 12.9, 0.5)];
        let ctx = context(&l, &search, &vg, &locs);
        for m in 0..l.len() {
            let p = plan(&ctx, m).unwrap();
            assert!(p.nodes.len() <= 8);
            assert!(
                p.nodes
                    .iter()
                    .all(|&(k, _)| ctx.rank[k as usize] < ctx.rank[m])
            );
            assert!(p.variance >= 0.0 && p.variance <= 1.0 + 1e-12);
        }
    }

    #[test]
    fn a_datum_on_a_centroid_leaves_the_path() {
        let l = square(10);
        let on = datum_nodes(
            &l,
            &[(2.5, 3.5, 0.5), (4.0, 4.0, 0.5), (2.5, 3.5, 0.5)],
            None,
        );
        assert_eq!(on, vec![(32, 0)]);
    }

    #[test]
    fn the_first_node_without_neighbors_is_marginal() {
        let l = square(8);
        let search = [Search {
            min_samples: 1,
            max_samples: 8,
            radius: 2.0,
            ..Default::default()
        }];
        let vg = Variogram::single(Model::Spherical, 1.0, 4.0);
        let ctx = context(&l, &search, &vg, &[(100.0, 100.0, 0.5)]);
        let first = (0..l.len()).find(|&m| ctx.rank[m] == 0).unwrap();
        let p = plan(&ctx, first).unwrap();
        assert!(p.nodes.is_empty());
        assert_eq!((p.constant, p.variance), (0.0, 1.0));
    }

    fn run(l: &Lattice, reals: Range<usize>) -> SharedBatch<'static> {
        let locs = vec![
            (2.5, 3.5, 0.5),
            (10.2, 11.7, 0.5),
            (17.5, 4.5, 0.5),
            (6.1, 16.3, 0.5),
        ];
        let vals = vec![1.0, 4.0, 2.5, 7.0];
        let search = [Search {
            min_samples: 1,
            max_samples: 12,
            radius: 10.0,
            ..Default::default()
        }];
        let vg = Variogram::single(Model::Spherical, 1.0, 8.0);
        let shared = SharedSgs {
            lattice: l,
            search: &search,
            levels: None,
            seed: 11,
        };
        sgs_shared(
            &locs, &vals, None, None, None, None, &vg, &shared, reals, None,
        )
        .unwrap()
    }

    #[test]
    fn data_on_centroids_are_honored() {
        let l = square(20);
        let b = run(&l, 0..3);
        for r in 0..3 {
            let v = b.realization(r);
            assert_eq!(v[l.node_at(3 * 20 + 2).unwrap()], 1.0);
            assert_eq!(v[l.node_at(4 * 20 + 17).unwrap()], 2.5);
        }
    }

    #[test]
    fn a_realization_does_not_depend_on_its_batch() {
        let l = square(20);
        let all = run(&l, 0..6);
        let some = run(&l, 3..5);
        let one = run(&l, 4..5);
        assert_eq!(all.realization(4), some.realization(1));
        assert_eq!(all.realization(4), one.realization(0));
        assert_ne!(all.realization(0), all.realization(1));
    }

    #[test]
    fn realizations_follow_the_seed_not_the_threads() {
        let l = square(24);
        let with = |t: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
                .install(|| run(&l, 0..2).realization(1))
        };
        assert_eq!(with(1), with(4));
    }

    #[test]
    fn masked_lattices_simulate_their_nodes_only() {
        let full = square(20);
        let cells: Vec<u64> = (0..400).filter(|c| c % 3 != 0).collect();
        let l = Lattice::masked(*full.geometry(), cells).unwrap();
        let v = run(&l, 0..1).realization(0);
        assert_eq!(v.len(), l.len());
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn unsupported_inputs_are_named() {
        let plain = Search {
            min_samples: 1,
            max_samples: 8,
            radius: 5.0,
            ..Default::default()
        };
        assert_eq!(
            shared_unsupported(std::slice::from_ref(&plain), false, false),
            None
        );
        assert_eq!(
            shared_unsupported(std::slice::from_ref(&plain), true, false),
            Some("local anisotropy")
        );
        assert_eq!(
            shared_unsupported(std::slice::from_ref(&plain), false, true),
            Some("simulated domains")
        );
        let octant = Search {
            octant: true,
            ..plain.clone()
        };
        assert_eq!(
            shared_unsupported(&[plain.clone(), octant], false, false),
            Some("octant searches")
        );
        let hg = Search {
            high_grade: Some(estimation::HighGrade::new(5.0, 10.0)),
            ..plain.clone()
        };
        assert_eq!(
            shared_unsupported(&[hg], false, false),
            Some("high-grade restrictions")
        );
        let soft = Search {
            soft: Some(estimation::Soft::All(3.0)),
            ..plain
        };
        assert_eq!(
            shared_unsupported(&[soft], false, false),
            Some("soft boundaries")
        );
    }

    #[test]
    fn sgs_shared_refuses_octant_searches() {
        let l = square(10);
        let search = [Search {
            min_samples: 1,
            octant: true,
            ..Default::default()
        }];
        let vg = Variogram::single(Model::Spherical, 1.0, 4.0);
        let shared = SharedSgs {
            lattice: &l,
            search: &search,
            levels: None,
            seed: 1,
        };
        let r = sgs_shared(
            &[(1.5, 1.5, 0.5)],
            &[1.0],
            None,
            None,
            None,
            None,
            &vg,
            &shared,
            0..1,
            None,
        );
        assert!(matches!(r, Err(SimError::InvalidParameters(_))));
    }

    #[test]
    fn hard_domains_keep_their_own_data_and_nodes() {
        let l = square(40);
        let nodes: Vec<u32> = (0..l.len()).map(|m| u32::from(m % 40 >= 20)).collect();
        let locs: Vec<_> = (0..40)
            .map(|i| {
                (
                    (i % 20) as f64 * 2.0 + 0.3,
                    (i / 20) as f64 * 20.0 + 5.3,
                    0.5,
                )
            })
            .collect();
        let of_data: Vec<u32> = locs.iter().map(|p| u32::from(p.0 >= 20.0)).collect();
        let vals: Vec<f64> = of_data
            .iter()
            .enumerate()
            .map(|(i, &d)| if d == 1 { 100.0 + i as f64 } else { i as f64 })
            .collect();
        let search = [Search {
            min_samples: 1,
            max_samples: 12,
            radius: 15.0,
            ..Default::default()
        }];
        let vg = Variogram::single(Model::Spherical, 1.0, 10.0);
        let shared = SharedSgs {
            lattice: &l,
            search: &search,
            levels: None,
            seed: 2,
        };
        let domains = Some((&of_data[..], &nodes[..]));
        let b = sgs_shared(
            &locs,
            &vals,
            None,
            None,
            domains,
            None,
            &vg,
            &shared,
            0..4,
            None,
        )
        .unwrap();
        for r in 0..4 {
            let v = b.realization(r);
            assert!((0..l.len()).all(|m| (nodes[m] == 1) == (v[m] >= 50.0)));
        }
    }

    #[test]
    fn a_node_takes_the_first_pass_that_finds_data() {
        let l = square(40);
        let locs = vec![(5.5, 5.5, 0.5)];
        let search = [
            Search {
                min_samples: 1,
                max_samples: 3,
                radius: 4.0,
                ..Default::default()
            },
            Search {
                min_samples: 1,
                max_samples: 10,
                radius: 12.0,
                ..Default::default()
            },
        ];
        let vg = Variogram::single(Model::Spherical, 1.0, 10.0);
        let ctx = context(&l, &search, &vg, &locs);
        let near = l.node_at(6 * 40 + 6).unwrap();
        let far = l.node_at(30 * 40 + 30).unwrap();
        let (n, f) = (plan(&ctx, near).unwrap(), plan(&ctx, far).unwrap());
        assert!(n.nodes.len() <= 2);
        assert!(f.nodes.len() > 3 && f.nodes.len() <= 10);
    }

    #[test]
    fn collocated_cosimulation_reproduces_the_correlation() {
        let (l, locs, vals, vg, search) = field();
        let smooth = SharedSgs {
            lattice: &l,
            search: &search,
            levels: None,
            seed: 99,
        };
        let other = sgs_shared(
            &locs,
            &vals,
            None,
            None,
            None,
            None,
            &vg,
            &smooth,
            0..1,
            None,
        )
        .unwrap();
        let secondary: Vec<f64> = (0..l.len()).map(|m| other.scores[m] as f64).collect();
        let rows = vec![secondary.clone()];
        let c = Collocated {
            scores: &rows,
            correlation: 0.8,
        };
        let shared = SharedSgs {
            lattice: &l,
            search: &search,
            levels: None,
            seed: 4,
        };
        let b = sgs_shared(
            &locs,
            &vals,
            None,
            None,
            None,
            None,
            &vg,
            &shared,
            0..4,
            Some(&c),
        )
        .unwrap();
        for r in 0..4 {
            let s: Vec<f64> = (0..l.len()).map(|m| b.scores[m * 4 + r] as f64).collect();
            let rho = crate::sgs::pearson(&s, &secondary, None);
            assert!((rho - 0.8).abs() < 0.1, "realization {r}: {rho}");
        }
    }

    #[test]
    fn a_trend_moves_the_grades_with_it() {
        let l = square(40);
        let locs: Vec<_> = (0..60)
            .map(|i| {
                (
                    (i % 10) as f64 * 4.0 + 0.3,
                    (i / 10) as f64 * 6.0 + 0.3,
                    0.5,
                )
            })
            .collect();
        let at_data: Vec<f64> = locs.iter().map(|p| p.0).collect();
        let vals: Vec<f64> = locs.iter().map(|p| p.0 + (p.1 * 0.7).sin()).collect();
        let at_nodes: Vec<f64> = (0..l.len()).map(|m| l.location(m).0).collect();
        let trend = Trend {
            data: &at_data,
            nodes: &at_nodes,
            classes: 4,
        };
        let search = [Search {
            min_samples: 1,
            max_samples: 12,
            radius: 12.0,
            ..Default::default()
        }];
        let vg = Variogram::single(Model::Spherical, 1.0, 8.0);
        let shared = SharedSgs {
            lattice: &l,
            search: &search,
            levels: None,
            seed: 6,
        };
        let b = sgs_shared(
            &locs,
            &vals,
            None,
            None,
            None,
            Some(trend),
            &vg,
            &shared,
            0..2,
            None,
        )
        .unwrap();
        let v = b.realization(0);
        let side = |east: bool| {
            let pick: Vec<f64> = (0..l.len())
                .filter(|&m| (m % 40 >= 20) == east)
                .map(|m| v[m])
                .collect();
            pick.iter().sum::<f64>() / pick.len() as f64
        };
        assert!(side(true) > side(false) + 10.0);
    }

    #[test]
    fn batches_fit_the_memory_budget() {
        // 1e6 nodes: fixed 8 + 16 + 8 = 32 B/node, 4 B/node per realization.
        let size = |n, cutoffs, quantiles, kept, budget| {
            shared_batch(1_000_000, n, cutoffs, quantiles, kept, budget)
        };
        assert_eq!(size(100, 0, false, 0, 32_000_000 + 64_000_000).unwrap(), 16);
        assert_eq!(size(5, 0, false, 0, u64::MAX).unwrap(), 5);
        assert_eq!(size(100, 0, false, 0, 36_000_000).unwrap(), 1);
        // stored quantiles and kept realizations hold 8 B/node per realization of the run.
        assert_eq!(
            size(10, 0, true, 10, 32_000_000 + 160_000_000 + 16_000_000).unwrap(),
            4
        );
        let refused = size(10, 0, true, 0, 100_000_000).unwrap_err().to_string();
        assert!(refused.contains("0.1 GB budget"), "{refused}");
    }

    type Field = (
        Lattice,
        Vec<(f64, f64, f64)>,
        Vec<f64>,
        Variogram,
        Vec<Search>,
    );

    /// 96 x 96 unit cells, 12 data in one corner, spherical range 16 cells.
    fn field() -> Field {
        let l = square(96);
        let locs: Vec<_> = (0..12)
            .map(|i| ((i % 4) as f64 * 3.0 + 0.5, (i / 4) as f64 * 3.0 + 0.5, 0.5))
            .collect();
        let vals: Vec<f64> = (0..12).map(|i| ((i * 7 % 12) as f64 + 1.0).ln()).collect();
        let vg = Variogram::single(Model::Spherical, 1.0, 16.0);
        let search = vec![Search {
            min_samples: 1,
            max_samples: 16,
            radius: 24.0,
            ..Default::default()
        }];
        (l, locs, vals, vg, search)
    }

    fn shared_scores(reals: usize) -> Vec<Vec<f64>> {
        let (l, locs, vals, vg, search) = field();
        let shared = SharedSgs {
            lattice: &l,
            search: &search,
            levels: None,
            seed: 5,
        };
        let b = sgs_shared(
            &locs,
            &vals,
            None,
            None,
            None,
            None,
            &vg,
            &shared,
            0..reals,
            None,
        )
        .unwrap();
        (0..reals)
            .map(|r| {
                (0..l.len())
                    .map(|m| b.scores[m * reals + r] as f64)
                    .collect()
            })
            .collect()
    }

    /// Mean semivariance of `values` on the 96 x 96 lattice at a lag of `h` cells along x.
    fn gamma_x(values: &[f64], h: usize) -> f64 {
        let mut sum = 0.0;
        let mut count = 0.0;
        for j in 0..96 {
            for i in 0..96 - h {
                let d = values[j * 96 + i] - values[j * 96 + i + h];
                sum += 0.5 * d * d;
                count += 1.0;
            }
        }
        sum / count
    }

    #[test]
    fn shared_scores_reproduce_the_standard_normal_away_from_the_data() {
        let reals = shared_scores(20);
        let far: Vec<f64> = reals
            .iter()
            .flat_map(|r| {
                (0..96 * 96)
                    .filter(|m| m % 96 > 48 && m / 96 > 48)
                    .map(move |m| r[m])
            })
            .collect();
        let n = far.len() as f64;
        let mean = far.iter().sum::<f64>() / n;
        let var = far.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        assert!(mean.abs() < 0.2, "mean {mean}");
        assert!((var - 1.0).abs() < 0.15, "variance {var}");
    }

    #[test]
    fn shared_scores_reproduce_the_variogram_model() {
        let shared = shared_scores(20);
        let vg = field().3;
        for h in [2, 4, 8, 16] {
            let model = 1.0 - vg.cov_points(&(0.0, 0.0, 0.0), &(h as f64, 0.0, 0.0));
            let s = shared.iter().map(|r| gamma_x(r, h)).sum::<f64>() / shared.len() as f64;
            assert!((s - model).abs() < 0.1, "lag {h}: shared {s} model {model}");
        }
    }

    #[test]
    fn the_shared_ensemble_mean_approximates_kriging() {
        let (l, locs, vals, vg, search) = field();
        let shared = SharedSgs {
            lattice: &l,
            search: &search,
            levels: None,
            seed: 8,
        };
        let b = sgs_shared(
            &locs,
            &vals,
            None,
            None,
            None,
            None,
            &vg,
            &shared,
            0..60,
            None,
        )
        .unwrap();
        let t = Transforms::fit(&vals, None, None, None).unwrap();
        let samples: Vec<Sample> = locs
            .iter()
            .zip(&t.scores)
            .map(|(&p, &s)| Sample::new(p, s))
            .collect();
        for &m in &[5 * 96 + 5, 10 * 96 + 2, 3 * 96 + 14] {
            let target = l.location(m);
            let sk = krige(Kind::Simple { mean: 0.0 }, &target, &samples, &vg).unwrap();
            let mean = (0..60).map(|r| b.scores[m * 60 + r] as f64).sum::<f64>() / 60.0;
            assert!(
                (mean - sk.value).abs() < 0.35,
                "node {m}: mean {mean} kriged {}",
                sk.value
            );
        }
    }
}
