//! SNESIM (Strebelle, 2002): single normal equation simulation of categories
//! from a training image.
//!
//! The patterns of the image are counted once into search trees, one per
//! multigrid level, and a node draws its category from the counts of the data
//! event around it, so no node scans the image. Strebelle, S. (2002).
//! Conditional simulation of complex geological structures using
//! multiple-point statistics. Mathematical Geology 34(1), 1-21.

mod tree;

use std::sync::{Arc, Mutex, PoisonError};

use ceres_core::rng::realization_seed;
use ceres_core::{Geometry, Progress};
use nalgebra::Matrix3;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::{Result, SimError};
use crate::lattice::{Lattice, Template, level, multigrid_path};
use crate::post::{CategoricalSummary, Keep, categorical};
use crate::training_image::{NO_CODE, TrainingImage};
use tree::{SearchTree, TreeScratch};

/// Share of the template of a coarse level kept next to the node, where only
/// hard data are informed; the rest reads the level's lattice.
const NEAR_SHARE: f64 = 0.5;

/// Most multigrid levels above the finest.
const MAX_LEVELS: usize = 16;

/// SNESIM parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnesimParams {
    /// Offsets every node reads: the nearest cells around it.
    pub template_size: usize,
    /// Coarse levels above the finest grid; 0 simulates on the finest only.
    pub n_levels: usize,
    /// Fewest replicates of a data event in the training image for its counts
    /// to give the probabilities; the farthest informed offsets are dropped
    /// until enough remain.
    pub min_replicates: u32,
    /// Target proportion of each category of the training image, summing to 1.
    pub target_proportions: Option<Vec<f64>>,
    /// Servosystem strength in `[0, 1)`; 0 switches it off.
    pub servo: f64,
}

impl Default for SnesimParams {
    fn default() -> Self {
        Self {
            template_size: 40,
            n_levels: 3,
            min_replicates: 10,
            target_proportions: None,
            servo: 0.5,
        }
    }
}

/// Which trees a node reads: the training image and the matrix taking grid
/// offsets to image offsets. The identity here; zones and local anisotropy
/// add their own classes, each with trees of its own.
type TemplateClass = (usize, [[f64; 3]; 3]);

const PLAIN: TemplateClass = (0, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

/// Pulls a realization towards target proportions: a node draws from
/// `P(c) + gain * (target(c) - current(c))`, negative values set to 0, where
/// `current` is the share of `c` among the nodes informed so far.
#[derive(Debug, Clone)]
struct Servo {
    targets: Vec<f64>,
    gain: f64,
}

impl Servo {
    fn correction(&self, informed: &[u64], out: &mut Vec<f64>) {
        out.clear();
        let total: u64 = informed.iter().sum();
        if total > 0 {
            out.extend(
                self.targets
                    .iter()
                    .zip(informed)
                    .map(|(t, &n)| self.gain * (t - n as f64 / total as f64)),
            );
        }
    }
}

/// SNESIM (Strebelle, 2002) over a categorical training image.
///
/// Each level of the multigrid has a template of `template_size` offsets,
/// nearest first. On a coarse level `L`, half of them stay next to the node,
/// where they see hard data off the level's lattice, and half are offsets
/// scaled by `2^L`, which see the level's nodes. Search trees are built on
/// the first `simulate` and kept.
#[derive(Debug)]
pub struct Snesim {
    ti: TrainingImage,
    params: SnesimParams,
    servo: Option<Servo>,
    /// Template offsets of each level, in cells, nearest first.
    templates: Vec<Vec<[i32; 3]>>,
    trees: Mutex<Vec<(TemplateClass, Arc<Vec<SearchTree>>)>>,
}

impl Snesim {
    pub fn new(ti: TrainingImage, params: SnesimParams) -> Result<Self> {
        let bad = |m: String| Err(SimError::InvalidParameters(m));
        if !ti.is_categorical() {
            return bad("SNESIM needs a categorical training image".into());
        }
        if params.template_size == 0 {
            return bad("template_size must be at least 1".into());
        }
        if params.n_levels > MAX_LEVELS {
            return bad(format!(
                "n_levels is {}; at most {MAX_LEVELS}",
                params.n_levels
            ));
        }
        if params.min_replicates == 0 {
            return bad("min_replicates must be at least 1".into());
        }
        if !(0.0..1.0).contains(&params.servo) {
            return bad(format!("servo is {}; it must be in [0, 1)", params.servo));
        }
        let k = ti.n_categories();
        let servo = match &params.target_proportions {
            None => None,
            Some(t) => {
                if t.len() != k {
                    return bad(format!(
                        "give {k} target proportions, one per category of the training image, got {}",
                        t.len()
                    ));
                }
                if let Some(p) = t.iter().find(|p| !(0.0..=1.0).contains(*p)) {
                    return bad(format!("target proportion {p} is outside [0, 1]"));
                }
                let sum: f64 = t.iter().sum();
                if (sum - 1.0).abs() > 1e-6 {
                    return bad(format!(
                        "target proportions sum to {sum}; they must sum to 1"
                    ));
                }
                (params.servo > 0.0).then(|| Servo {
                    targets: t.clone(),
                    gain: params.servo / (1.0 - params.servo),
                })
            }
        };
        let templates = templates(ti.dims(), params.template_size, params.n_levels)?;
        Ok(Self {
            ti,
            params,
            servo,
            templates,
            trees: Mutex::new(Vec::new()),
        })
    }

    pub fn params(&self) -> &SnesimParams {
        &self.params
    }

    pub fn training_image(&self) -> &TrainingImage {
        &self.ti
    }

    /// Summary of `n` realizations over the nodes of `lattice`, realization
    /// `i` seeded by `realization_seed(seed, i)`. `data` are hard data,
    /// locations and categories: each goes to the node whose cell holds it,
    /// several in one cell to their most frequent category (ties to the one
    /// nearest the cell center, then the lowest), and is reproduced exactly;
    /// data off the nodes are ignored. The search trees are built on the first
    /// call, once their size is bounded by `memory` bytes (0 skips the check).
    #[allow(clippy::too_many_arguments)]
    pub fn simulate(
        &self,
        lattice: &Lattice,
        data: Option<(&[(f64, f64, f64)], &[usize])>,
        n: usize,
        seed: u64,
        keep: &Keep,
        memory: u64,
        progress: Option<&Progress>,
    ) -> Result<CategoricalSummary> {
        let hard = self.snap(lattice, data)?;
        let trees = self.trees(PLAIN, memory)?;
        let mut skip = vec![false; lattice.len()];
        for &(m, _) in &hard {
            skip[m] = true;
        }
        categorical(
            n,
            self.ti.n_categories(),
            keep,
            |i| {
                self.realization(
                    lattice,
                    &hard,
                    &skip,
                    &trees,
                    realization_seed(seed, i as u64),
                )
            },
            progress,
        )
    }

    /// Hard data as `(node, category)`, one per node, sorted by node.
    fn snap(
        &self,
        lattice: &Lattice,
        data: Option<(&[(f64, f64, f64)], &[usize])>,
    ) -> Result<Vec<(usize, u8)>> {
        let Some((locs, codes)) = data else {
            return Ok(Vec::new());
        };
        if locs.len() != codes.len() {
            return Err(SimError::InvalidParameters(
                "one category per location".into(),
            ));
        }
        let k = self.ti.n_categories();
        if let Some(c) = codes.iter().find(|&&c| c >= k) {
            return Err(SimError::InvalidParameters(format!(
                "category {c} is not in the training image, whose codes are 0 to {}",
                k - 1
            )));
        }
        let mut hits: Vec<(usize, usize, f64)> = locs
            .iter()
            .zip(codes)
            .filter_map(|(p, &c)| {
                let (cell, at) = lattice.geometry().locate([p.0, p.1, p.2])?;
                let d2 = at.iter().map(|a| (a - 0.5).powi(2)).sum();
                Some((lattice.node_at(cell)?, c, d2))
            })
            .collect();
        hits.sort_by_key(|h| h.0);
        Ok(hits
            .chunk_by(|a, b| a.0 == b.0)
            .map(|group| {
                let score = |c: usize| {
                    let of = group.iter().filter(|h| h.1 == c);
                    let nearest = of.clone().map(|h| h.2).fold(f64::INFINITY, f64::min);
                    (of.count(), -nearest)
                };
                let best = group
                    .iter()
                    .map(|h| h.1)
                    .max_by(|&a, &b| {
                        let (x, y) = (score(a), score(b));
                        x.0.cmp(&y.0).then(x.1.total_cmp(&y.1)).then(b.cmp(&a))
                    })
                    .unwrap_or(0);
                (group[0].0, best as u8)
            })
            .collect())
    }

    /// The trees of `class`, one per level, built and cached on first use.
    fn trees(&self, class: TemplateClass, memory: u64) -> Result<Arc<Vec<SearchTree>>> {
        let mut cache = self.trees.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((_, trees)) = cache.iter().find(|(c, _)| *c == class) {
            return Ok(trees.clone());
        }
        let (dims, k) = (self.ti.dims(), self.ti.n_categories());
        let valid = self.ti.valid_positions().len();
        let mut bytes = 0u64;
        for template in &self.templates {
            let nodes = SearchTree::bound(dims, valid, k, template);
            if nodes > u64::from(u32::MAX) {
                return Err(self.too_large(format!(
                    "a search tree may need {nodes} nodes, more than the {} it can hold",
                    u32::MAX
                )));
            }
            bytes = bytes.saturating_add(nodes * SearchTree::node_bytes(k));
        }
        if memory > 0 && bytes > memory {
            return Err(self.too_large(format!(
                "the search trees may need {:.2} GB of memory but {:.2} GB is available",
                bytes as f64 / 1e9,
                memory as f64 / 1e9
            )));
        }
        let codes = self.ti.codes().unwrap_or_default();
        let trees: Vec<SearchTree> = self
            .templates
            .par_iter()
            .map(|t| SearchTree::build(codes, dims, k, t))
            .collect();
        let trees = Arc::new(trees);
        cache.push((class, trees.clone()));
        Ok(trees)
    }

    fn too_large(&self, what: String) -> SimError {
        SimError::InvalidParameters(format!(
            "{what}; lower template_size (now {}) or n_levels (now {}), or use a smaller training image",
            self.params.template_size, self.params.n_levels
        ))
    }

    fn realization(
        &self,
        lattice: &Lattice,
        hard: &[(usize, u8)],
        skip: &[bool],
        trees: &[SearchTree],
        seed: u64,
    ) -> Result<Vec<usize>> {
        let top = self.params.n_levels;
        let (path, _) = multigrid_path(lattice, top, seed, skip)?;
        let k = self.ti.n_categories();
        let mut grid = vec![NO_CODE; lattice.len()];
        let mut informed = vec![0u64; k];
        for &(m, c) in hard {
            grid[m] = c;
            informed[usize::from(c)] += 1;
        }
        let mut rng = StdRng::seed_from_u64(realization_seed(seed, 1));
        let (mut event, mut correction, mut weights) = (vec![], vec![], vec![]);
        let mut scratch = TreeScratch::default();
        for &m in &path {
            let m = m as usize;
            let l = level(lattice.geometry().ijk(lattice.cell(m)), top);
            event.clear();
            event.extend(
                self.templates[l]
                    .iter()
                    .map(|&d| lattice.shifted(m, d).map_or(NO_CODE, |q| grid[q])),
            );
            if let Some(servo) = &self.servo {
                servo.correction(&informed, &mut correction);
            }
            let tree = &trees[l];
            let u: f64 = rng.r#gen();
            // Longest data event first; soft probabilities will weigh `weights`
            // here and may leave none, falling back to a shorter event.
            let c = tree
                .lookup(&event, self.params.min_replicates, &mut scratch)
                .rev()
                .find_map(|(depth, counts)| {
                    probabilities(tree, depth, counts, &correction, &mut weights);
                    let total: f64 = weights.iter().sum();
                    (total > 0.0).then(|| pick(&weights, u * total))
                })
                .unwrap_or(0);
            grid[m] = c as u8;
            informed[c] += 1;
        }
        Ok(grid.into_iter().map(usize::from).collect())
    }
}

/// Template offsets of levels `0..=n_levels` over an image of `dims`, in
/// cells: level 0 the `n` nearest; a coarser level `L` the `n / 2` nearest
/// plus nearest offsets times `2^L`, up to `n`, leaving out those that cannot
/// fit in the image.
fn templates(dims: [usize; 3], n: usize, n_levels: usize) -> Result<Vec<Vec<[i32; 3]>>> {
    let unit = Lattice::regular(Geometry {
        origin: [0.0; 3],
        size: [1.0; 3],
        count: dims,
        rotation: [0.0; 3],
    });
    let frame = Matrix3::identity();
    let whole = dims.iter().sum::<usize>() as f64;
    let mut radius = 1.0;
    let pool: Vec<[i32; 3]> = loop {
        let t = Template::new(&unit, &frame, radius)?;
        if t.offsets().len() >= 2 * n || radius > whole {
            break t.offsets().iter().map(|o| o.0).collect();
        }
        radius *= 1.5;
    };
    let fits = |d: [i64; 3]| (0..3).all(|a| d[a].unsigned_abs() < dims[a] as u64);
    let squared = |d: &[i32; 3]| d.iter().map(|&x| i64::from(x).pow(2)).sum::<i64>();
    Ok((0..=n_levels)
        .map(|l| {
            if l == 0 {
                return pool[..n.min(pool.len())].to_vec();
            }
            let near = &pool[..((NEAR_SHARE * n as f64) as usize).min(pool.len())];
            let mut offsets = near.to_vec();
            for d in &pool {
                if offsets.len() >= n {
                    break;
                }
                let scaled = d.map(|x| i64::from(x) << l);
                let scaled_i32 = scaled.map(|x| x as i32);
                if fits(scaled) && !near.contains(&scaled_i32) {
                    offsets.push(scaled_i32);
                }
            }
            offsets.sort_by(|a, b| squared(a).cmp(&squared(b)).then(a.cmp(b)));
            offsets
        })
        .collect())
}

/// Fills `weights` with the share of each category in `counts` (a data event
/// at `depth`, replicates above 0), each weighted by
/// `cells_at_depth(0)[c] / cells_at_depth(depth)[c]` for the cells the
/// image's edges leave out, plus the servosystem's `correction`, negative
/// values set to 0. Where every replicate holds one category, the image
/// leaves no choice and the servosystem leaves the node alone.
fn probabilities(
    tree: &SearchTree,
    depth: usize,
    counts: &[u32],
    correction: &[f64],
    weights: &mut Vec<f64>,
) {
    let (all, reaching) = (tree.cells_at_depth(0), tree.cells_at_depth(depth));
    weights.clear();
    weights.extend(counts.iter().enumerate().map(|(c, &n)| match n {
        0 => 0.0,
        n => f64::from(n) * f64::from(all[c]) / f64::from(reaching[c]),
    }));
    let total: f64 = weights.iter().sum();
    for w in weights.iter_mut() {
        *w /= total;
    }
    if weights.contains(&1.0) {
        return;
    }
    for (w, add) in weights.iter_mut().zip(correction) {
        *w = (*w + add).max(0.0);
    }
}

/// The category `u` in `[0, total)` falls on, the categories laid end to
/// end, each as long as its weight.
fn pick(weights: &[f64], mut u: f64) -> usize {
    let mut chosen = 0;
    for (c, &w) in weights.iter().enumerate() {
        if w > 0.0 {
            chosen = c;
            if u < w {
                break;
            }
            u -= w;
        }
    }
    chosen
}

#[cfg(test)]
mod tests {
    use std::sync::Arc as StdArc;

    use arrow_array::{Float64Array, RecordBatch};
    use ceres_core::BlockModel;

    use super::*;

    fn model(count: [usize; 3], values: Vec<f64>) -> BlockModel {
        let column = StdArc::new(Float64Array::from(values));
        let batch = RecordBatch::try_from_iter([("v", column as _)]).unwrap();
        BlockModel::regular(geometry(count), batch).unwrap()
    }

    fn geometry(count: [usize; 3]) -> Geometry {
        Geometry {
            origin: [0.0; 3],
            size: [1.0; 3],
            count,
            rotation: [0.0; 3],
        }
    }

    /// Sinuous channels (code 1) 6 cells wide along x every 24 rows, a
    /// quarter of the image, in a background of code 0.
    fn channels(side: usize) -> Vec<f64> {
        (0..side * side)
            .map(|p| {
                let (x, y) = ((p % side) as f64, (p / side) as f64);
                let channel = (y / 24.0).floor();
                let wave = 5.0 * (x * std::f64::consts::TAU / 50.0 + 2.0 * channel).sin();
                let center = (channel + 0.5) * 24.0 + wave;
                f64::from(u8::from((y - center).abs() < 3.0))
            })
            .collect()
    }

    fn ti(side: usize) -> TrainingImage {
        TrainingImage::categorical(&model([side, side, 1], channels(side)), "v").unwrap()
    }

    fn params() -> SnesimParams {
        SnesimParams {
            template_size: 24,
            n_levels: 2,
            ..SnesimParams::default()
        }
    }

    fn run(
        snesim: &Snesim,
        lattice: &Lattice,
        data: Option<(&[(f64, f64, f64)], &[usize])>,
        n: usize,
    ) -> CategoricalSummary {
        snesim
            .simulate(lattice, data, n, 7, &Keep::All, 0, None)
            .unwrap()
    }

    fn grid(side: usize) -> Lattice {
        Lattice::regular(geometry([side, side, 1]))
    }

    fn share_of_1(r: &[usize]) -> f64 {
        r.iter().filter(|&&c| c == 1).count() as f64 / r.len() as f64
    }

    /// Mean length of the runs of code 1 along x in a square image.
    fn mean_run(r: &[usize], side: usize) -> f64 {
        let (mut cells, mut runs) = (0, 0);
        for row in r.chunks(side) {
            for run in row.split(|&c| c != 1).filter(|run| !run.is_empty()) {
                cells += run.len();
                runs += 1;
            }
        }
        cells as f64 / runs as f64
    }

    fn mean(s: &CategoricalSummary, f: impl Fn(&[usize]) -> f64) -> f64 {
        s.realizations.iter().map(|r| f(r)).sum::<f64>() / s.realizations.len() as f64
    }

    #[test]
    fn hard_data_are_reproduced_exactly() {
        let codes = channels(64);
        let cells: Vec<usize> = (0..300).map(|i| i * 7919 % codes.len()).collect();
        let locs: Vec<_> = cells
            .iter()
            .map(|&p| ((p % 64) as f64 + 0.5, (p / 64) as f64 + 0.5, 0.5))
            .collect();
        let cats: Vec<usize> = cells.iter().map(|&p| codes[p] as usize).collect();
        let snesim = Snesim::new(ti(64), params()).unwrap();
        let s = run(&snesim, &grid(64), Some((&locs, &cats)), 3);
        for r in &s.realizations {
            assert!(r.iter().all(|&c| c < 2));
            for &p in &cells {
                assert_eq!(r[p] as f64, codes[p], "cell {p}");
            }
        }
    }

    #[test]
    fn several_data_in_a_cell_take_their_most_frequent_category() {
        let snesim = Snesim::new(ti(32), params()).unwrap();
        let locs = [
            (0.5, 0.5, 0.5),
            (0.1, 0.1, 0.5),
            (0.9, 0.9, 0.5),
            (3.4, 0.5, 0.5),
            (3.5, 0.5, 0.5),
            (-1.0, 0.5, 0.5),
        ];
        let hard = snesim
            .snap(&grid(8), Some((&locs, &[1, 0, 1, 0, 1, 1])))
            .unwrap();
        assert_eq!(hard, [(0, 1), (3, 1)]);
        assert!(snesim.snap(&grid(8), Some((&locs, &[2; 6]))).is_err());
    }

    #[test]
    fn unconditional_realizations_reproduce_the_proportions_and_the_channels() {
        let image = channels(120);
        let ti_share = image.iter().sum::<f64>() / image.len() as f64;
        let ti_run = {
            let codes: Vec<usize> = image.iter().map(|&v| v as usize).collect();
            mean_run(&codes, 120)
        };
        let multigrid = Snesim::new(ti(120), params()).unwrap();
        let s = run(&multigrid, &grid(64), None, 6);
        let share = mean(&s, share_of_1);
        assert!((share - ti_share).abs() < 0.06, "{share} vs {ti_share}");
        let long = mean(&s, |r| mean_run(r, 64));
        let single = SnesimParams {
            n_levels: 0,
            ..params()
        };
        let short = mean(
            &run(&Snesim::new(ti(120), single).unwrap(), &grid(64), None, 6),
            |r| mean_run(r, 64),
        );
        assert!(
            long > short,
            "runs of {long} with levels, {short} without, {ti_run} in the image"
        );
    }

    #[test]
    fn too_few_replicates_fall_back_to_the_image_proportions() {
        let never = SnesimParams {
            min_replicates: u32::MAX,
            ..params()
        };
        let s = run(&Snesim::new(ti(64), never).unwrap(), &grid(48), None, 4);
        let p = ti(64).proportions()[1];
        assert!((mean(&s, share_of_1) - p).abs() < 0.03);
        // Independent draws: runs of 1 / (1 - p) cells.
        assert!((mean(&s, |r| mean_run(r, 48)) - 1.0 / (1.0 - p)).abs() < 0.1);
    }

    #[test]
    fn the_servosystem_pulls_the_proportions_to_the_target() {
        let target = |servo| SnesimParams {
            target_proportions: Some(vec![0.6, 0.4]),
            servo,
            ..params()
        };
        let share = |p| {
            mean(
                &run(&Snesim::new(ti(64), p).unwrap(), &grid(48), None, 4),
                share_of_1,
            )
        };
        let (without, with) = (share(params()), share(target(0.95)));
        assert!(
            (with - 0.4).abs() < (without - 0.4).abs(),
            "{without} -> {with}"
        );
        assert!((with - 0.4).abs() < 0.03, "{with}");
        assert_eq!(share(target(0.0)), without);
    }

    #[test]
    fn same_output_for_any_thread_count() {
        let p = SnesimParams {
            target_proportions: Some(vec![0.6, 0.4]),
            ..params()
        };
        let snesim = Snesim::new(ti(48), p).unwrap();
        let locs = [(3.5, 4.5, 0.5), (20.5, 30.5, 0.5)];
        let data = Some((&locs[..], &[1usize, 0][..]));
        let pool = |t| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
        };
        let one = pool(1).install(|| run(&snesim, &grid(40), data, 5).realizations);
        let four = pool(4).install(|| run(&snesim, &grid(40), data, 5).realizations);
        assert_eq!(one, four);
        assert_ne!(one[0], one[1]);
    }

    #[test]
    fn masked_targets_simulate_their_nodes_only() {
        let g = geometry([30, 30, 1]);
        let cells: Vec<u64> = (0..900).filter(|c| c % 30 < 20).collect();
        let lattice = Lattice::masked(g, cells).unwrap();
        let snesim = Snesim::new(ti(40), params()).unwrap();
        let locs = [(3.5, 4.5, 0.5), (25.5, 4.5, 0.5)];
        let s = run(&snesim, &lattice, Some((&locs, &[1, 1])), 2);
        let node = lattice.node_at(3 + 4 * 30).unwrap();
        for r in &s.realizations {
            assert_eq!(r.len(), 600);
            assert_eq!(r[node], 1);
        }
    }

    #[test]
    fn trees_beyond_the_memory_are_refused_before_they_are_built() {
        let snesim = Snesim::new(ti(64), params()).unwrap();
        let e = snesim
            .simulate(&grid(8), None, 1, 0, &Keep::None, 1000, None)
            .unwrap_err()
            .to_string();
        assert!(e.contains("lower template_size (now 24)"), "{e}");
        assert!(snesim.trees.lock().unwrap().is_empty());
        assert!(
            snesim
                .simulate(&grid(8), None, 1, 0, &Keep::None, 0, None)
                .is_ok()
        );
        assert_eq!(snesim.trees.lock().unwrap().len(), 1);
    }

    #[test]
    fn templates_keep_half_their_offsets_near_on_coarse_levels() {
        let t = templates([100, 100, 1], 20, 2).unwrap();
        assert_eq!(t.len(), 3);
        assert!(t.iter().all(|l| l.len() == 20));
        assert!(
            t[0].iter()
                .all(|d| d[2] == 0 && d[0].abs() <= 3 && d[1].abs() <= 3)
        );
        assert_eq!(
            t[2].iter().filter(|d| d.iter().all(|x| x % 4 == 0)).count(),
            10
        );
        assert!(t[2].iter().any(|d| d[0].abs() >= 8));
    }

    #[test]
    fn invalid_parameters_are_refused() {
        let bad = |p: SnesimParams| Snesim::new(ti(16), p).unwrap_err().to_string();
        let with = |t: Vec<f64>| SnesimParams {
            target_proportions: Some(t),
            ..params()
        };
        assert!(bad(with(vec![1.0])).contains("give 2 target proportions"));
        assert!(bad(with(vec![0.5, 0.4])).contains("sum to 0.9"));
        assert!(bad(with(vec![1.5, -0.5])).contains("1.5"));
        assert!(
            bad(SnesimParams {
                servo: 1.0,
                ..params()
            })
            .contains("servo")
        );
        assert!(
            bad(SnesimParams {
                servo: f64::NAN,
                ..params()
            })
            .contains("servo")
        );
        assert!(
            bad(SnesimParams {
                min_replicates: 0,
                ..params()
            })
            .contains("min_replicates")
        );
        assert!(
            bad(SnesimParams {
                template_size: 0,
                ..params()
            })
            .contains("template_size")
        );
        let continuous = TrainingImage::continuous(&model([4, 4, 1], vec![0.5; 16]), "v").unwrap();
        assert!(Snesim::new(continuous, params()).is_err());
    }
}
