//! SNESIM (Strebelle, 2002): single normal equation simulation of categories
//! from a training image.
//!
//! The patterns of the image are counted once into search trees, one per
//! multigrid level, and a node draws its category from the counts of the data
//! event around it, so no node scans the image. Strebelle, S. (2002).
//! Conditional simulation of complex geological structures using
//! multiple-point statistics. Mathematical Geology 34(1), 1-21.
//!
//! Zones and local anisotropy read the patterns turned and stretched: a node
//! of template class `(image, M)` compares its neighbor at grid offset `g`
//! with the cell at `round(M g)` of training image `image`, and each class
//! has trees of its own.

mod tree;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use ceres_core::rng::realization_seed;
use ceres_core::{Geometry, Progress, block_frame};
use estimation::lva::LocalAnisotropy;
use nalgebra::{Matrix3, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::{Result, SimError};
use crate::lattice::{Lattice, Template, level, multigrid_path};
use crate::post::{
    CategoricalSummary, ContinuousOptions, ContinuousSummary, Keep, categorical, continuous,
    quantile_sorted,
};
use crate::sis::closed;
use crate::training_image::{NO_CODE, TrainingImage, class_of, unify};
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
    /// Width in degrees of the classes local angles are rounded to; affinities
    /// are rounded on a logarithmic scale of the same step in radians.
    #[serde(default = "default_angle_step")]
    pub angle_step: f64,
    /// Ascending values cutting continuous training images into classes;
    /// `None` takes the quartiles of their values. Ignored for categorical
    /// images.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cutoffs: Option<Vec<f64>>,
}

fn default_angle_step() -> f64 {
    10.0
}

impl Default for SnesimParams {
    fn default() -> Self {
        Self {
            template_size: 40,
            n_levels: 3,
            min_replicates: 10,
            target_proportions: None,
            servo: 0.5,
            angle_step: default_angle_step(),
            cutoffs: None,
        }
    }
}

/// Which trees a node reads: the training image and the matrix taking grid
/// offsets to image offsets.
type TemplateClass = (usize, [[f64; 3]; 3]);

const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// What varies from node to node besides hard data and soft probabilities.
#[derive(Debug, Clone, Copy, Default)]
pub struct SnesimLocal<'a> {
    /// The training image of each node, an index into
    /// [`Snesim::training_images`]; `None` reads the first everywhere.
    pub zones: Option<&'a [usize]>,
    /// The anisotropy of each node, turning and stretching the patterns of
    /// its image (see [`Snesim::simulate`]).
    pub anisotropy: Option<&'a LocalAnisotropy>,
}

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

/// SNESIM (Strebelle, 2002) over training images, one per zone.
///
/// Continuous images are cut into classes by `cutoffs`; SNESIM simulates the
/// classes and each node draws its value from the values of its image in its
/// class.
///
/// Each level of the multigrid has a template of `template_size` offsets,
/// nearest first. On a coarse level `L`, half of them stay next to the node,
/// where they see hard data off the level's lattice, and half are offsets
/// scaled by `2^L`, which see the level's nodes. Search trees are built for
/// the template classes a `simulate` needs and kept until the next.
#[derive(Debug)]
pub struct Snesim {
    images: Vec<TrainingImage>,
    params: SnesimParams,
    servo: Option<Servo>,
    /// Template offsets of each level, in grid cells, nearest first.
    templates: Vec<Vec<[i32; 3]>>,
    trees: Mutex<Vec<(TemplateClass, Arc<Vec<SearchTree>>)>>,
    /// The continuous images the classes were cut from.
    sources: Option<Vec<TrainingImage>>,
    /// Cells `(image, position)` of each class, `[image][class]`, then those
    /// of all images.
    pools: Vec<Vec<Vec<(u32, u32)>>>,
}

/// Nearest neighbors a continuous node compares its value candidates on.
const VALUE_NEIGHBORS: usize = 8;

/// Random cells of its class a continuous node picks its value from.
const VALUE_CANDIDATES: usize = 32;

impl Snesim {
    pub fn new(ti: TrainingImage, params: SnesimParams) -> Result<Self> {
        Self::zoned(vec![ti], params)
    }

    /// SNESIM whose zone `i` draws from `images[i]`; the images share one
    /// code set (see [`unify`]).
    pub fn zoned(mut images: Vec<TrainingImage>, params: SnesimParams) -> Result<Self> {
        let bad = |m: String| Err(SimError::InvalidParameters(m));
        if images.is_empty() {
            return bad("give at least one training image".into());
        }
        unify(&mut images)?;
        let mut params = params;
        let (mut sources, mut pools) = (None, Vec::new());
        if !images[0].is_categorical() {
            let cutoffs = params.cutoffs.take().unwrap_or_else(|| quartiles(&images));
            if cutoffs.iter().any(|c| !c.is_finite()) || cutoffs.windows(2).any(|w| w[0] >= w[1]) {
                return bad(format!(
                    "cutoffs {cutoffs:?} must be finite and strictly ascending"
                ));
            }
            let classes = images
                .iter()
                .map(|t| t.classes(&cutoffs))
                .collect::<Result<Vec<_>>>()?;
            pools = images
                .iter()
                .enumerate()
                .map(|(i, t)| pool(i as u32, t, &cutoffs))
                .collect();
            let all: Vec<Vec<(u32, u32)>> = (0..=cutoffs.len())
                .map(|c| pools.iter().flat_map(|p| p[c].iter().copied()).collect())
                .collect();
            if let Some(c) = all.iter().position(Vec::is_empty) {
                return bad(format!(
                    "class {c} of cutoffs {cutoffs:?} holds no value of the training images"
                ));
            }
            pools.push(all);
            params.cutoffs = Some(cutoffs);
            sources = Some(std::mem::replace(&mut images, classes));
            unify(&mut images)?;
        }
        if !(params.angle_step > 0.0 && params.angle_step.is_finite()) {
            return bad(format!(
                "angle_step is {}; it must be a finite number of degrees above 0",
                params.angle_step
            ));
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
        let k = images[0].n_categories();
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
        let mut dims = [1; 3];
        for image in &images {
            dims = std::array::from_fn(|a| dims[a].max(image.dims()[a]));
        }
        let templates = templates(dims, params.template_size, params.n_levels)?;
        Ok(Self {
            images,
            params,
            servo,
            templates,
            trees: Mutex::new(Vec::new()),
            sources,
            pools,
        })
    }

    pub fn params(&self) -> &SnesimParams {
        &self.params
    }

    /// Whether the training images are continuous, cut into classes.
    pub fn is_continuous(&self) -> bool {
        self.sources.is_some()
    }

    /// The continuous training images the classes were cut from.
    pub fn continuous_images(&self) -> Option<&[TrainingImage]> {
        self.sources.as_deref()
    }

    /// The training images, one per zone, sharing one code set: the classes
    /// of continuous images.
    pub fn training_images(&self) -> &[TrainingImage] {
        &self.images
    }

    /// Categories of the training images.
    pub fn n_categories(&self) -> usize {
        self.images[0].n_categories()
    }

    /// Template classes whose trees the last `simulate` read.
    pub fn n_classes(&self) -> usize {
        self.trees
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Summary of `n` realizations over the nodes of `lattice`, realization
    /// `i` seeded by `realization_seed(seed, i)`. `data` are hard data,
    /// locations and categories: each goes to the node whose cell holds it,
    /// several in one cell to their most frequent category (ties to the one
    /// nearest the cell center, then the lowest), and is reproduced exactly;
    /// data off the nodes are ignored. `soft` holds one row per node: `None`
    /// where the node has no soft information, else a probability per
    /// category, closed to sum 1 (see [`Snesim::prior`] for how they combine
    /// with the image); hard data override them.
    ///
    /// `local.zones` picks the training image of each node. With
    /// `local.anisotropy`, one per node, the patterns of a node's image are
    /// turned by its angles (world azimuth, dip and rake; azimuth only on a
    /// grid one cell thick) and stretched by its affinity: at angles 0 the
    /// image's y axis runs north along the major axis, x east against the
    /// semi-major axis and z up along the minor axis, and the image's
    /// structures grow by `scale` along y, `scale * semi_ratio` along x and
    /// `scale * minor_ratio` along z. The image's cells are taken to have the
    /// size of the grid's. Angles are rounded to multiples of `angle_step`
    /// and the logarithm of each affinity to multiples of `angle_step` in
    /// radians, so that either moves an offset at distance `r` by about
    /// `r * angle_step`. Each distinct (image, rounded transform) is a
    /// template class with trees of its own, and a node of class `(image, M)`
    /// reads its neighbor at grid offset `g`, on every level, from image
    /// offset `round(M g)`.
    ///
    /// The trees of the classes are built once their summed size is bounded
    /// by `memory` bytes (0 skips the check), and kept until the next call.
    #[allow(clippy::too_many_arguments)]
    pub fn simulate(
        &self,
        lattice: &Lattice,
        data: Option<(&[(f64, f64, f64)], &[usize])>,
        soft: Option<&[Option<Vec<f64>>]>,
        local: SnesimLocal,
        n: usize,
        seed: u64,
        keep: &Keep,
        memory: u64,
        progress: Option<&Progress>,
    ) -> Result<CategoricalSummary> {
        if self.is_continuous() {
            return Err(SimError::InvalidParameters(
                "the training images are continuous; simulate values instead".into(),
            ));
        }
        let hard = self.snap(lattice, data)?;
        let soft = soft.map(|rows| self.soft(lattice, rows)).transpose()?;
        let (classes, skip) = self.prepare(lattice, local, memory, &hard)?;
        categorical(
            n,
            self.n_categories(),
            keep,
            |i| {
                self.realization(
                    lattice,
                    &hard,
                    &skip,
                    soft.as_deref(),
                    &classes,
                    realization_seed(seed, i as u64),
                )
            },
            progress,
        )
    }

    /// Summary of `n` realizations of continuous training images, as
    /// [`Snesim::simulate`] without soft probabilities: each datum goes to
    /// its class, and a node with data keeps the mean of those in the class
    /// it took. Every other node takes the value of a cell of its image in
    /// the class simulated there, the one of `VALUE_CANDIDATES` random cells
    /// whose neighbors best match the values already around the node.
    #[allow(clippy::too_many_arguments)]
    pub fn simulate_values(
        &self,
        lattice: &Lattice,
        data: Option<(&[(f64, f64, f64)], &[f64])>,
        local: SnesimLocal,
        n: usize,
        seed: u64,
        keep: &Keep,
        memory: u64,
        progress: Option<&Progress>,
    ) -> Result<ContinuousSummary> {
        let cutoffs = match (&self.params.cutoffs, self.is_continuous()) {
            (Some(c), true) => c,
            _ => {
                return Err(SimError::InvalidParameters(
                    "the training images are categorical; simulate categories instead".into(),
                ));
            }
        };
        let (locs, values) = data.unwrap_or((&[], &[]));
        if locs.len() != values.len() {
            return Err(SimError::InvalidParameters("one value per location".into()));
        }
        if let Some(v) = values.iter().find(|v| !v.is_finite()) {
            return Err(SimError::InvalidParameters(format!(
                "data value {v} is not finite"
            )));
        }
        let codes: Vec<usize> = values.iter().map(|&v| class_of(cutoffs, v)).collect();
        let hard = self.snap(lattice, Some((locs, &codes)))?;
        let mut class = vec![NO_CODE; lattice.len()];
        for &(m, c) in &hard {
            class[m] = c;
        }
        let mut sums = vec![(0.0, 0usize); lattice.len()];
        for ((p, &v), &c) in locs.iter().zip(values).zip(&codes) {
            let node = lattice
                .geometry()
                .locate([p.0, p.1, p.2])
                .and_then(|(cell, _)| lattice.node_at(cell));
            if let Some(m) = node.filter(|&m| usize::from(class[m]) == c) {
                sums[m] = (sums[m].0 + v, sums[m].1 + 1);
            }
        }
        let (classes, skip) = self.prepare(lattice, local, memory, &hard)?;
        let options = ContinuousOptions {
            keep: keep.clone(),
            ..ContinuousOptions::default()
        };
        let mut grid = vec![f64::NAN; lattice.len()];
        for (g, (sum, k)) in grid.iter_mut().zip(&sums) {
            if *k > 0 {
                *g = sum / *k as f64;
            }
        }
        continuous(
            n,
            &options,
            |i| {
                let seed = realization_seed(seed, i as u64);
                let cats = self.realization(lattice, &hard, &skip, None, &classes, seed)?;
                self.values(lattice, &cats, grid.clone(), &skip, &classes, seed)
            },
            progress,
        )
    }

    /// Values for the simulated classes `cats`, filling the nodes of `grid`
    /// that `skip` leaves free in the order of the realization's path: each
    /// takes the value of the cell, among `VALUE_CANDIDATES` drawn from its
    /// image in its class, whose surroundings differ least from the values
    /// already at the node's `VALUE_NEIGHBORS` nearest offsets.
    fn values(
        &self,
        lattice: &Lattice,
        cats: &[usize],
        mut grid: Vec<f64>,
        skip: &[bool],
        classes: &Classes,
        seed: u64,
    ) -> Result<Vec<f64>> {
        let (path, _) = multigrid_path(lattice, self.params.n_levels, seed, skip)?;
        let sources = self.sources.as_deref().unwrap_or_default();
        let everywhere = &self.pools[self.pools.len() - 1];
        let near = &self.templates[0][..VALUE_NEIGHBORS.min(self.templates[0].len())];
        let mut rng = StdRng::seed_from_u64(realization_seed(seed, 2));
        let mut informed: Vec<([i32; 3], f64)> = Vec::with_capacity(near.len());
        for &m in &path {
            let (m, c) = (m as usize, cats[m as usize]);
            let class = classes.of_node[m] as usize;
            let pool = match &self.pools[classes.images[class]][c] {
                p if p.is_empty() => &everywhere[c],
                p => p,
            };
            informed.clear();
            let offsets = image_offsets(near, &classes.matrices[class]);
            informed.extend(near.iter().zip(offsets).filter_map(|(&d, o)| {
                let v = grid[lattice.shifted(m, d)?];
                (!v.is_nan()).then_some((o, v))
            }));
            let mut best = (f64::INFINITY, pool[0]);
            for _ in 0..VALUE_CANDIDATES.min(pool.len()) {
                let cell = pool[rng.gen_range(0..pool.len())];
                let image = &sources[cell.0 as usize];
                let cost: f64 = informed
                    .iter()
                    .map(|&(o, v)| match f64::from(value_at(image, cell.1, o)) {
                        w if w.is_nan() => image.value_range().powi(2),
                        w => (w - v).powi(2),
                    })
                    .sum();
                if cost < best.0 {
                    best = (cost, cell);
                }
                if informed.is_empty() {
                    break;
                }
            }
            grid[m] = f64::from(value_at(&sources[best.1.0 as usize], best.1.1, [0; 3]));
        }
        Ok(grid)
    }

    /// The template classes of the nodes of `lattice` with their trees, and
    /// the nodes `hard` fixes.
    fn prepare(
        &self,
        lattice: &Lattice,
        local: SnesimLocal,
        memory: u64,
        hard: &[(usize, u8)],
    ) -> Result<(Classes, Vec<bool>)> {
        let (of_node, classes) = self.classes(lattice, local)?;
        let trees = self.trees(&classes, local.anisotropy.is_some(), memory)?;
        let classes = Classes {
            of_node,
            images: classes.iter().map(|c| c.0).collect(),
            matrices: classes.iter().map(|c| c.1).collect(),
            trees,
        };
        let mut skip = vec![false; lattice.len()];
        for &(m, _) in hard {
            skip[m] = true;
        }
        Ok((classes, skip))
    }

    /// The template class of each node, an index into the classes, which are
    /// in order of their first node.
    fn classes(
        &self,
        lattice: &Lattice,
        local: SnesimLocal,
    ) -> Result<(Vec<u32>, Vec<TemplateClass>)> {
        let bad = |m: String| Err(SimError::InvalidParameters(m));
        let nodes = lattice.len();
        if let Some(zones) = local.zones {
            if zones.len() != nodes {
                return bad(format!(
                    "give one zone per target, {nodes}, got {}",
                    zones.len()
                ));
            }
            if let Some(z) = zones.iter().find(|&&z| z >= self.images.len()) {
                return bad(format!(
                    "zone {z} has no training image; there are {} (zones 0 to {})",
                    self.images.len(),
                    self.images.len() - 1
                ));
            }
        }
        let image = |m: usize| local.zones.map_or(0, |z| z[m]);
        let Some(field) = local.anisotropy else {
            let mut of_image: HashMap<usize, u32> = HashMap::new();
            let mut classes = Vec::new();
            let of_node = (0..nodes)
                .map(|m| {
                    *of_image.entry(image(m)).or_insert_with(|| {
                        classes.push((image(m), IDENTITY));
                        classes.len() as u32 - 1
                    })
                })
                .collect();
            return Ok((of_node, classes));
        };
        if field.len() != nodes {
            return bad(format!(
                "give one local anisotropy per target, {nodes}, got {}",
                field.len()
            ));
        }
        let flat = lattice.geometry().count[2] == 1;
        let keys: Vec<Result<[u64; 6]>> = (0..nodes)
            .into_par_iter()
            .map(|m| self.binned(field, m, flat).map(|t| t.map(f64::to_bits)))
            .collect();
        let mut of_key: HashMap<(usize, [u64; 6]), u32> = HashMap::new();
        let mut classes = Vec::new();
        let mut of_node = Vec::with_capacity(nodes);
        for (m, key) in keys.into_iter().enumerate() {
            let key = (image(m), key?);
            let class = *of_key.entry(key).or_insert_with(|| {
                let t = key.1.map(f64::from_bits);
                let matrix = lag_matrix(lattice.geometry(), [t[0], t[1], t[2]], [t[3], t[4], t[5]]);
                classes.push((key.0, matrix));
                classes.len() as u32 - 1
            });
            of_node.push(class);
        }
        Ok((of_node, classes))
    }

    /// Angles and affinity along the image's x, y and z at node `m` of
    /// `field`, rounded by `angle_step`; `flat` keeps the azimuth only.
    fn binned(&self, field: &LocalAnisotropy, m: usize, flat: bool) -> Result<[f64; 6]> {
        let [azimuth, dip, rake] = field.angles[m];
        let [semi, minor] = field.ratios[m];
        let scale = field.scales[m];
        let finite = [azimuth, dip, rake, semi, minor, scale];
        if finite.iter().any(|x| !x.is_finite()) || [semi, minor, scale].iter().any(|&x| x <= 0.0) {
            return Err(SimError::InvalidParameters(format!(
                "the anisotropy of target {m} is angles {:?}, ratios {:?}, scale {scale}; give finite angles and ratios and scales above 0",
                field.angles[m], field.ratios[m]
            )));
        }
        let step = self.params.angle_step;
        let angle = |a: f64| ((a / step).round() * step).rem_euclid(360.0) + 0.0;
        let log_step = step.to_radians();
        let affinity = |a: f64| ((a.ln() / log_step).round() * log_step).exp();
        Ok(if flat {
            [
                angle(azimuth),
                0.0,
                0.0,
                affinity(scale * semi),
                affinity(scale),
                1.0,
            ]
        } else {
            [
                angle(azimuth),
                (dip / step).round() * step + 0.0,
                angle(rake),
                affinity(scale * semi),
                affinity(scale),
                affinity(scale * minor),
            ]
        })
    }

    /// `rows` checked against `lattice`, each closed to sum 1.
    fn soft(&self, lattice: &Lattice, rows: &[Option<Vec<f64>>]) -> Result<Vec<Option<Vec<f64>>>> {
        let k = self.n_categories();
        if rows.len() != lattice.len() {
            return Err(SimError::InvalidParameters(format!(
                "give one row of soft probabilities per target, {}, got {}",
                lattice.len(),
                rows.len()
            )));
        }
        rows.iter()
            .map(|r| {
                r.as_deref()
                    .map(|r| closed(r, k, "soft probabilities"))
                    .transpose()
            })
            .collect()
    }

    /// `P0`, what soft probabilities are measured against: permanence of
    /// ratios (Journel, 2002) with tau = 1 draws from
    /// `P(c) ∝ P_tree(c) · P_soft(c) / P0(c)`. `P0` is the proportions of
    /// `image`, the node's training image, or the target proportions while
    /// the servosystem pulls the tree probabilities towards them, so that
    /// soft probabilities equal to `P0` change nothing. Journel, A. G. (2002). Combining knowledge from
    /// diverse sources: an alternative to traditional data independence
    /// hypotheses. Mathematical Geology 34(5), 573-596.
    fn prior(&self, image: usize) -> &[f64] {
        self.servo
            .as_ref()
            .map_or(self.images[image].proportions(), |s| &s.targets)
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
        let k = self.n_categories();
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

    /// The trees of each of `classes`, one per level, those not cached
    /// built once the bound of all of them fits in `memory` bytes (0 skips
    /// the check); the cache then holds `classes` only. `turned` says the
    /// classes come from local anisotropy, which `angle_step` rounds.
    fn trees(
        &self,
        classes: &[TemplateClass],
        turned: bool,
        memory: u64,
    ) -> Result<Vec<Arc<Vec<SearchTree>>>> {
        let mut cache = self.trees.lock().unwrap_or_else(PoisonError::into_inner);
        let k = self.n_categories();
        let too_large = |what: String| {
            let step = match turned {
                true => format!(
                    "widen angle_step (now {}) for fewer template classes, ",
                    self.params.angle_step
                ),
                false => String::new(),
            };
            SimError::InvalidParameters(format!(
                "{what}; {step}lower template_size (now {}) or n_levels (now {}), or use a smaller training image",
                self.params.template_size, self.params.n_levels
            ))
        };
        let mut bytes = 0u64;
        for &(image, matrix) in classes {
            let ti = &self.images[image];
            for template in &self.templates {
                let offsets = image_offsets(template, &matrix);
                let nodes = SearchTree::bound(ti.dims(), ti.valid_positions().len(), k, &offsets);
                if nodes > u64::from(u32::MAX) {
                    return Err(too_large(format!(
                        "a search tree may need {nodes} nodes, more than the {} it can hold",
                        u32::MAX
                    )));
                }
                bytes = bytes.saturating_add(nodes * SearchTree::node_bytes(k));
            }
        }
        if memory > 0 && bytes > memory {
            return Err(too_large(format!(
                "the search trees of {} template classes may need {:.2} GB of memory but {:.2} GB is available",
                classes.len(),
                bytes as f64 / 1e9,
                memory as f64 / 1e9
            )));
        }
        let cached = |class: &TemplateClass| {
            cache
                .iter()
                .find(|(c, _)| c == class)
                .map(|(_, trees)| trees.clone())
        };
        let missing: Vec<(TemplateClass, usize)> = classes
            .iter()
            .filter(|c| cached(c).is_none())
            .flat_map(|&c| (0..self.templates.len()).map(move |l| (c, l)))
            .collect();
        let built: Vec<SearchTree> = missing
            .par_iter()
            .map(|&((image, matrix), l)| {
                let ti = &self.images[image];
                let offsets = image_offsets(&self.templates[l], &matrix);
                SearchTree::build(ti.codes().unwrap_or_default(), ti.dims(), k, &offsets)
            })
            .collect();
        let mut built = built.into_iter();
        let trees: Vec<Arc<Vec<SearchTree>>> = classes
            .iter()
            .map(|c| {
                cached(c).unwrap_or_else(|| {
                    Arc::new(built.by_ref().take(self.templates.len()).collect())
                })
            })
            .collect();
        *cache = classes.iter().copied().zip(trees.iter().cloned()).collect();
        Ok(trees)
    }

    fn realization(
        &self,
        lattice: &Lattice,
        hard: &[(usize, u8)],
        skip: &[bool],
        soft: Option<&[Option<Vec<f64>>]>,
        classes: &Classes,
        seed: u64,
    ) -> Result<Vec<usize>> {
        let top = self.params.n_levels;
        let (path, _) = multigrid_path(lattice, top, seed, skip)?;
        let k = self.n_categories();
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
            let class = classes.of_node[m] as usize;
            let (tree, prior) = (&classes.trees[class][l], self.prior(classes.images[class]));
            let u: f64 = rng.r#gen();
            let soft = soft.and_then(|rows| rows[m].as_deref());
            // Longest data event first; where the soft probabilities and the
            // image share no category, the next shorter one.
            let c = tree
                .lookup(&event, self.params.min_replicates, &mut scratch)
                .rev()
                .find_map(|(depth, counts)| {
                    probabilities(tree, depth, counts, &correction, &mut weights);
                    if let Some(soft) = soft {
                        weigh_by_soft(&mut weights, soft, prior);
                    }
                    let total: f64 = weights.iter().sum();
                    (total > 0.0).then(|| pick(&weights, u * total))
                });
            // Soft probabilities only on categories of `P0` 0: the image's
            // proportions decide, as without them.
            let c = c.unwrap_or_else(|| {
                let (depth, counts) = tree.lookup(&[], 0, &mut scratch).next().unwrap_or_default();
                probabilities(tree, depth, counts, &correction, &mut weights);
                pick(&weights, u * weights.iter().sum::<f64>())
            });
            grid[m] = c as u8;
            informed[c] += 1;
        }
        Ok(grid.into_iter().map(usize::from).collect())
    }
}

/// The template classes of a run.
struct Classes {
    /// The class of each node.
    of_node: Vec<u32>,
    /// Grid offsets to image offsets of each class.
    matrices: Vec<[[f64; 3]; 3]>,
    /// The training image of each class.
    images: Vec<usize>,
    /// The trees of each class, one per level.
    trees: Vec<Arc<Vec<SearchTree>>>,
}

/// Cutoffs at the quartiles of the values of continuous `images`, above the
/// smallest: few classes keep the data events frequent enough to count.
fn quartiles(images: &[TrainingImage]) -> Vec<f64> {
    let mut all: Vec<f64> = images
        .iter()
        .flat_map(|t| {
            let values = t.continuous_values().unwrap_or_default();
            t.valid_positions()
                .iter()
                .map(|&p| f64::from(values[p as usize]))
        })
        .collect();
    all.sort_by(f64::total_cmp);
    let mut cutoffs: Vec<f64> = (1..4)
        .map(|q| quantile_sorted(&all, f64::from(q) / 4.0))
        .filter(|&c| c > all[0])
        .collect();
    cutoffs.dedup();
    cutoffs
}

/// The valid cells `(index, position)` of continuous `image` in each class
/// of `cutoffs`.
fn pool(index: u32, image: &TrainingImage, cutoffs: &[f64]) -> Vec<Vec<(u32, u32)>> {
    let values = image.continuous_values().unwrap_or_default();
    let mut pools = vec![Vec::new(); cutoffs.len() + 1];
    for &p in image.valid_positions() {
        let v = f64::from(values[p as usize]);
        pools[class_of(cutoffs, v)].push((index, p));
    }
    pools
}

/// The value of `image` at `offset` from `position`; NaN off the image or
/// its data.
fn value_at(image: &TrainingImage, position: u32, offset: [i32; 3]) -> f32 {
    let dims = image.dims();
    let p = position as usize;
    let ijk = [p % dims[0], p / dims[0] % dims[1], p / (dims[0] * dims[1])];
    let mut at = 0;
    for a in (0..3).rev() {
        let c = ijk[a] as i64 + i64::from(offset[a]);
        if !(0..dims[a] as i64).contains(&c) {
            return f32::NAN;
        }
        at = at * dims[a] + c as usize;
    }
    image.continuous_values().map_or(f32::NAN, |v| v[at])
}

/// Grid offsets, in cells of `geometry`, to training-image offsets, for
/// patterns turned by `angles` (world azimuth, dip and rake) and stretched by
/// `affinity` along the image's x, y and z: at angles 0 the image's axes are
/// east, north and up, and its cells have the size of the grid's. Entries
/// below 1e-9 in size become 0, so that `cos 90°` rounds no offset of half a
/// cell.
fn lag_matrix(geometry: &Geometry, angles: [f64; 3], affinity: [f64; 3]) -> [[f64; 3]; 3] {
    let size = Vector3::from(geometry.size);
    let to_image = Matrix3::from_diagonal(&Vector3::from(affinity).map(|a| 1.0 / a))
        * block_frame(angles)
        * block_frame(geometry.rotation).transpose();
    let m =
        Matrix3::from_diagonal(&size.map(|s| 1.0 / s)) * to_image * Matrix3::from_diagonal(&size);
    std::array::from_fn(|r| {
        std::array::from_fn(|c| {
            if m[(r, c)].abs() < 1e-9 {
                0.0
            } else {
                m[(r, c)]
            }
        })
    })
}

/// Where each offset of `template` lands in the training image read through
/// `matrix`: `round(matrix · offset)`, halves up.
fn image_offsets(template: &[[i32; 3]], matrix: &[[f64; 3]; 3]) -> Vec<[i32; 3]> {
    if *matrix == IDENTITY {
        return template.to_vec();
    }
    template
        .iter()
        .map(|d| {
            matrix.map(|row| {
                let x: f64 = (0..3).map(|a| row[a] * f64::from(d[a])).sum();
                (x + 0.5).floor() as i32
            })
        })
        .collect()
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

/// Multiplies each category's tree probability by `soft[c] / prior[c]`
/// (see [`Snesim::prior`]); a category of prior 0 gets 0.
fn weigh_by_soft(weights: &mut [f64], soft: &[f64], prior: &[f64]) {
    for ((w, s), p) in weights.iter_mut().zip(soft).zip(prior) {
        *w = if *p > 0.0 { *w * s / p } else { 0.0 };
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
            .simulate(
                lattice,
                data,
                None,
                SnesimLocal::default(),
                n,
                7,
                &Keep::All,
                0,
                None,
            )
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
            .simulate(
                &grid(8),
                None,
                None,
                SnesimLocal::default(),
                1,
                0,
                &Keep::None,
                1000,
                None,
            )
            .unwrap_err()
            .to_string();
        assert!(e.contains("lower template_size (now 24)"), "{e}");
        assert!(snesim.trees.lock().unwrap().is_empty());
        assert!(
            snesim
                .simulate(
                    &grid(8),
                    None,
                    None,
                    SnesimLocal::default(),
                    1,
                    0,
                    &Keep::None,
                    0,
                    None
                )
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

    fn run_soft(
        snesim: &Snesim,
        side: usize,
        soft: &[Option<Vec<f64>>],
        n: usize,
    ) -> CategoricalSummary {
        snesim
            .simulate(
                &grid(side),
                None,
                Some(soft),
                SnesimLocal::default(),
                n,
                7,
                &Keep::All,
                0,
                None,
            )
            .unwrap()
    }

    /// Share of code 1 in each column of a square image, over realizations.
    fn column_shares(s: &CategoricalSummary, side: usize) -> Vec<f64> {
        let mut shares = vec![0.0; side];
        for r in &s.realizations {
            for (p, &c) in r.iter().enumerate() {
                shares[p % side] += (c == 1) as u8 as f64;
            }
        }
        let cells = (side * s.realizations.len()) as f64;
        shares.iter().map(|x| x / cells).collect()
    }

    #[test]
    fn realizations_follow_a_soft_trend() {
        let side = 48;
        let soft: Vec<_> = (0..side * side)
            .map(|p| {
                let east = 0.05 + 0.9 * (p % side) as f64 / (side - 1) as f64;
                Some(vec![1.0 - east, east])
            })
            .collect();
        let s = run_soft(&Snesim::new(ti(64), params()).unwrap(), side, &soft, 8);
        let shares = column_shares(&s, side);
        let band = |a: usize| shares[a..a + 12].iter().sum::<f64>() / 12.0;
        let bands: Vec<f64> = (0..4).map(|b| band(12 * b)).collect();
        assert!(bands.windows(2).all(|w| w[1] > w[0] + 0.05), "{bands:?}");
    }

    #[test]
    fn soft_probabilities_at_the_prior_change_nothing() {
        let snesim = Snesim::new(ti(64), params()).unwrap();
        let p = snesim.training_images()[0].proportions().to_vec();
        let plain = run(&snesim, &grid(40), None, 4);
        let soft = run_soft(&snesim, 40, &vec![Some(p); 1600], 4);
        let (a, b) = (mean(&plain, share_of_1), mean(&soft, share_of_1));
        assert!((a - b).abs() < 0.01, "{a} vs {b}");
        let same = plain
            .realizations
            .iter()
            .flatten()
            .zip(soft.realizations.iter().flatten())
            .filter(|(x, y)| x == y)
            .count();
        assert!(same as f64 > 0.99 * 4.0 * 1600.0, "{same}");
    }

    #[test]
    fn certain_soft_probabilities_are_reproduced_and_hard_data_override_them() {
        let snesim = Snesim::new(ti(64), params()).unwrap();
        let soft: Vec<_> = (0..1600)
            .map(|p| match p % 5 {
                0 => Some(vec![0.0, 1.0]),
                1 => Some(vec![3.0, 0.0]),
                _ => None,
            })
            .collect();
        let locs = [(0.5, 0.5, 0.5)];
        let s = snesim
            .simulate(
                &grid(40),
                Some((&locs, &[0])),
                Some(&soft),
                SnesimLocal::default(),
                3,
                7,
                &Keep::All,
                0,
                None,
            )
            .unwrap();
        for r in &s.realizations {
            assert_eq!(r[0], 0);
            for p in 1..1600 {
                match p % 5 {
                    0 => assert_eq!(r[p], 1, "node {p}"),
                    1 => assert_eq!(r[p], 0, "node {p}"),
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn soft_probabilities_on_categories_of_prior_zero_fall_back_to_the_image() {
        let p = SnesimParams {
            target_proportions: Some(vec![1.0, 0.0]),
            ..params()
        };
        let snesim = Snesim::new(ti(64), p).unwrap();
        let s = run_soft(&snesim, 24, &vec![Some(vec![0.0, 1.0]); 576], 2);
        assert!(s.realizations.iter().flatten().all(|&c| c < 2));
    }

    #[test]
    fn soft_output_is_the_same_for_any_thread_count() {
        let snesim = Snesim::new(ti(48), params()).unwrap();
        let soft: Vec<_> = (0..1600)
            .map(|p| (p % 3 != 0).then(|| vec![0.3, 0.7 * (p % 40) as f64 / 40.0]))
            .collect();
        let pool = |t| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
        };
        let one = pool(1).install(|| run_soft(&snesim, 40, &soft, 4).realizations);
        let four = pool(4).install(|| run_soft(&snesim, 40, &soft, 4).realizations);
        assert_eq!(one, four);
    }

    #[test]
    fn bad_soft_probabilities_are_refused() {
        let snesim = Snesim::new(ti(32), params()).unwrap();
        let bad = |soft: Vec<Option<Vec<f64>>>| {
            snesim
                .simulate(
                    &grid(4),
                    None,
                    Some(&soft),
                    SnesimLocal::default(),
                    1,
                    0,
                    &Keep::None,
                    0,
                    None,
                )
                .unwrap_err()
                .to_string()
        };
        assert!(bad(vec![None; 3]).contains("one row of soft probabilities per target, 16"));
        let mut rows = vec![None; 16];
        rows[5] = Some(vec![0.5, -0.1]);
        assert!(bad(rows.clone()).contains("soft probabilities must be 2 finite"));
        rows[5] = Some(vec![1.0]);
        assert!(bad(rows.clone()).contains("soft probabilities"));
        rows[5] = Some(vec![0.0, 0.0]);
        assert!(bad(rows).contains("not all zero"));
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
        assert!(
            bad(SnesimParams {
                angle_step: 0.0,
                ..params()
            })
            .contains("angle_step")
        );
    }

    /// A value rising across each channel of [`channels`] from 1 at its edges
    /// to 3 at its center, over a background from 0 to 0.5 along x.
    fn grades(side: usize) -> TrainingImage {
        let values = channels(side)
            .iter()
            .enumerate()
            .map(|(p, &c)| {
                let (x, y) = ((p % side) as f64, (p / side) as f64);
                match c {
                    0.0 => 0.5 * x / side as f64,
                    _ => 1.0 + 2.0 * (y * 0.7).sin().abs(),
                }
            })
            .collect();
        TrainingImage::continuous(&model([side, side, 1], values), "v").unwrap()
    }

    fn sorted(mut v: Vec<f64>) -> Vec<f64> {
        v.sort_by(f64::total_cmp);
        v
    }

    fn run_values(
        snesim: &Snesim,
        side: usize,
        data: Option<(&[(f64, f64, f64)], &[f64])>,
        n: usize,
    ) -> ContinuousSummary {
        snesim
            .simulate_values(
                &grid(side),
                data,
                SnesimLocal::default(),
                n,
                7,
                &Keep::All,
                0,
                None,
            )
            .unwrap()
    }

    #[test]
    fn continuous_realizations_reproduce_the_histogram() {
        let image = grades(96);
        let values = sorted(
            image
                .continuous_values()
                .unwrap()
                .iter()
                .map(|&v| f64::from(v))
                .collect(),
        );
        let snesim = Snesim::new(image, params()).unwrap();
        assert_eq!(snesim.params().cutoffs.as_ref().unwrap().len(), 3);
        let s = run_values(&snesim, 64, None, 6);
        let simulated = sorted(s.realizations.concat());
        let below = |v: &[f64], t: f64| v.partition_point(|&x| x < t) as f64 / v.len() as f64;
        for t in [0.1, 0.25, 0.4, 1.5, 2.0, 2.5] {
            let (a, b) = (below(&values, t), below(&simulated, t));
            assert!(
                (a - b).abs() < 0.06,
                "share below {t}: {a} in the image, {b} simulated"
            );
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        assert!((mean(&values) - mean(&simulated)).abs() < 0.1);
        assert!(
            snesim
                .simulate(
                    &grid(8),
                    None,
                    None,
                    SnesimLocal::default(),
                    1,
                    0,
                    &Keep::None,
                    0,
                    None
                )
                .is_err()
        );
        assert!(
            Snesim::new(ti(16), params())
                .unwrap()
                .simulate_values(
                    &grid(8),
                    None,
                    SnesimLocal::default(),
                    1,
                    0,
                    &Keep::None,
                    0,
                    None
                )
                .is_err()
        );
    }

    #[test]
    fn continuous_hard_data_are_reproduced_and_output_is_thread_independent() {
        let snesim = Snesim::new(grades(64), params()).unwrap();
        let locs: Vec<_> = (0..200)
            .map(|i| ((i * 13 % 48) as f64 + 0.5, (i * 7 % 48) as f64 + 0.5, 0.5))
            .collect();
        let values: Vec<f64> = (0..200).map(|i| 0.01 * i as f64).collect();
        let data = Some((&locs[..], &values[..]));
        let pool = |t| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
        };
        let one = pool(1).install(|| run_values(&snesim, 48, data, 3).realizations);
        let four = pool(4).install(|| run_values(&snesim, 48, data, 3).realizations);
        assert_eq!(one, four);
        let mut expected: HashMap<usize, (f64, usize)> = HashMap::new();
        for (p, &v) in locs.iter().zip(&values) {
            let e = expected
                .entry(p.0 as usize + 48 * p.1 as usize)
                .or_default();
            *e = (v, e.1 + 1);
        }
        for r in &one {
            for (&node, &(v, k)) in &expected {
                if k == 1 {
                    assert_eq!(r[node], v, "node {node}");
                }
            }
        }
    }

    #[test]
    fn bad_cutoffs_are_refused() {
        let with = |c: Vec<f64>| {
            let p = SnesimParams {
                cutoffs: Some(c),
                ..params()
            };
            Snesim::new(grades(16), p).unwrap_err().to_string()
        };
        assert!(with(vec![1.0, 0.5]).contains("strictly ascending"));
        assert!(with(vec![f64::NAN]).contains("finite"));
        assert!(with(vec![-1.0]).contains("class 0"));
        assert!(with(vec![1.0; 300]).contains("ascending"));
    }

    /// `azimuth`, ratios `semi` and 1 and `scale` at each of `n` nodes.
    fn field(n: usize, at: impl Fn(usize) -> (f64, f64, f64)) -> LocalAnisotropy {
        let (angles, (ratios, scales)): (Vec<_>, (Vec<_>, Vec<_>)) = (0..n)
            .map(|m| {
                let (azimuth, semi, scale) = at(m);
                ([azimuth, 0.0, 0.0], ([semi, 1.0], scale))
            })
            .unzip();
        LocalAnisotropy::new(vec![(0.0, 0.0, 0.0); n], angles, ratios)
            .unwrap()
            .with_scales(scales)
            .unwrap()
    }

    fn turned(snesim: &Snesim, side: usize, local: SnesimLocal, n: usize) -> CategoricalSummary {
        snesim
            .simulate(&grid(side), None, None, local, n, 7, &Keep::All, 0, None)
            .unwrap()
    }

    fn anisotropic(field: &LocalAnisotropy) -> SnesimLocal<'_> {
        SnesimLocal {
            anisotropy: Some(field),
            ..SnesimLocal::default()
        }
    }

    /// Mean length of the runs of code 1 along x and along y, in the columns
    /// `x0..x1` of a square image.
    fn runs(r: &[usize], side: usize, x0: usize, x1: usize) -> [f64; 2] {
        let mean = |lines: Vec<Vec<usize>>| {
            let (mut cells, mut n) = (0, 0);
            for line in &lines {
                for run in line.split(|&c| c != 1).filter(|run| !run.is_empty()) {
                    cells += run.len();
                    n += 1;
                }
            }
            cells as f64 / n.max(1) as f64
        };
        let along_x = (0..side).map(|y| (x0..x1).map(|x| r[x + side * y]).collect());
        let along_y = (x0..x1).map(|x| (0..side).map(|y| r[x + side * y]).collect());
        [mean(along_x.collect()), mean(along_y.collect())]
    }

    #[test]
    fn a_quarter_turn_reads_grid_east_as_image_north() {
        let unit = geometry([8, 8, 8]);
        let m = lag_matrix(&unit, [90.0, 0.0, 0.0], [1.0; 3]);
        assert_eq!(
            image_offsets(&[[3, 0, 0], [0, -3, 0], [0, 0, 4]], &m),
            [[0, 3, 0], [3, 0, 0], [0, 0, 4]]
        );
        let m = lag_matrix(&unit, [0.0, 90.0, 0.0], [1.0; 3]);
        assert_eq!(image_offsets(&[[0, 0, -5]], &m), [[0, 5, 0]]);
        let m = lag_matrix(&unit, [0.0; 3], [2.0, 1.0, 1.0]);
        assert_eq!(
            image_offsets(&[[6, 5, 0], [1, 0, 0], [-1, 0, 0]], &m),
            [[3, 5, 0], [1, 0, 0], [0, 0, 0]]
        );
        assert_eq!(lag_matrix(&unit, [0.0; 3], [1.0; 3]), IDENTITY);
    }

    #[test]
    fn a_turned_image_simulates_as_the_image_turned_on_every_level() {
        // Azimuth 90 reads grid offset (x, y) at image offset (-y, x): the
        // same counts as the image turned a quarter, exactly.
        let side = 64;
        let codes = channels(side);
        let quarter: Vec<f64> = (0..side * side)
            .map(|p| codes[(side - 1 - p / side) + side * (p % side)])
            .collect();
        let quarter = TrainingImage::categorical(&model([side, side, 1], quarter), "v").unwrap();
        let expected = run(&Snesim::new(quarter, params()).unwrap(), &grid(40), None, 3);
        let f = field(1600, |_| (90.0, 1.0, 1.0));
        let s = turned(
            &Snesim::new(ti(side), params()).unwrap(),
            40,
            anisotropic(&f),
            3,
        );
        assert_eq!(s.realizations, expected.realizations);
    }

    #[test]
    fn zero_anisotropy_changes_nothing() {
        let snesim = Snesim::new(ti(64), params()).unwrap();
        let f = field(1600, |_| (1.0, 0.99, 1.01));
        let s = turned(&snesim, 40, anisotropic(&f), 3);
        assert_eq!(snesim.n_classes(), 1);
        assert_eq!(
            s.realizations,
            run(&snesim, &grid(40), None, 3).realizations
        );
    }

    #[test]
    fn channels_follow_a_rotation_field() {
        let side = 64;
        let snesim = Snesim::new(ti(120), params()).unwrap();
        let f = field(side * side, |m| {
            (if m % side < side / 2 { 90.0 } else { 0.0 }, 1.0, 1.0)
        });
        let s = turned(&snesim, side, anisotropic(&f), 6);
        let [west, east] = [(0, side / 2), (side / 2, side)].map(|(a, b)| {
            let r = s.realizations.iter().map(|r| runs(r, side, a, b));
            let [x, y] = r.fold([0.0; 2], |t, v| [t[0] + v[0], t[1] + v[1]]);
            x / y
        });
        // Azimuth 0 keeps the image's channels along x; 90 turns them north.
        assert!(
            east > 1.4 && west < 0.6,
            "x/y runs {west} west, {east} east"
        );
    }

    #[test]
    fn affinity_widens_the_channels() {
        let side = 64;
        let snesim = Snesim::new(ti(120), params()).unwrap();
        let width = |scale: f64| {
            let f = field(side * side, |_| (0.0, 1.0 / scale, scale));
            let s = turned(&snesim, side, anisotropic(&f), 6);
            mean(&s, |r| runs(r, side, 0, side)[1])
        };
        let (plain, wide) = (width(1.0), width(2.0));
        assert!(
            (1.5..2.6).contains(&(wide / plain)),
            "widths {plain} and {wide}"
        );
        // A semi-major ratio of 0.5 halves the image along its x axis, and
        // with it the channels' length.
        let f = field(side * side, |_| (0.0, 0.5, 1.0));
        let short = mean(&turned(&snesim, side, anisotropic(&f), 6), |r| {
            runs(r, side, 0, side)[0]
        });
        let long = mean(&run(&snesim, &grid(side), None, 6), |r| mean_run(r, side));
        assert!(short < 0.75 * long, "runs along x {short} vs {long}");
    }

    #[test]
    fn each_zone_draws_from_its_image() {
        let side = 64;
        let second: Vec<f64> = channels(side).iter().map(|&c| 2.0 * c).collect();
        let second = TrainingImage::categorical(&model([side, side, 1], second), "v").unwrap();
        let snesim = Snesim::zoned(vec![ti(side), second], params()).unwrap();
        assert_eq!(snesim.n_categories(), 3);
        let zones: Vec<usize> = (0..side * side)
            .map(|m| usize::from(m % side >= side / 2))
            .collect();
        let local = SnesimLocal {
            zones: Some(&zones),
            ..SnesimLocal::default()
        };
        let s = turned(&snesim, side, local, 8);
        let p = snesim.training_images()[0].proportions()[1];
        let share = |zone: usize, c: usize| {
            let cells = s.realizations.iter().flat_map(|r| r.iter().zip(&zones));
            let of_zone: Vec<usize> = cells
                .filter(|(_, z)| **z == zone)
                .map(|(c, _)| *c)
                .collect();
            assert!(of_zone.iter().all(|&x| x == 0 || x == c));
            of_zone.iter().filter(|&&x| x == c).count() as f64 / of_zone.len() as f64
        };
        let (west, east) = (share(0, 1), share(1, 2));
        assert!(
            (west - p).abs() < 0.06 && (east - p).abs() < 0.06,
            "{west} and {east} vs {p}"
        );
        let zones: Vec<usize> = (0..1600).map(|m| usize::from(m / 40 >= 20)).collect();
        let local = SnesimLocal {
            zones: Some(&zones),
            ..SnesimLocal::default()
        };
        let f = field(1600, |m| ((m % 4) as f64 * 90.0, 1.0, 1.0));
        let both = SnesimLocal {
            anisotropy: Some(&f),
            ..local
        };
        turned(&snesim, 40, both, 1);
        assert_eq!(snesim.n_classes(), 8);
        let bad = |zones: &[usize]| {
            let local = SnesimLocal {
                zones: Some(zones),
                ..SnesimLocal::default()
            };
            let e = snesim.simulate(&grid(4), None, None, local, 1, 0, &Keep::None, 0, None);
            e.unwrap_err().to_string()
        };
        assert!(bad(&[0; 3]).contains("one zone per target, 16"));
        assert!(bad(&[2; 16]).contains("zone 2 has no training image"));
    }

    #[test]
    fn angle_step_bounds_the_template_classes_and_the_memory() {
        let f = field(1600, |m| (m as f64 * 0.3, 1.0, 1.0));
        let classes = |step: f64, memory: u64| {
            let p = SnesimParams {
                angle_step: step,
                ..params()
            };
            let snesim = Snesim::new(ti(48), p).unwrap();
            let run = |memory| {
                let local = anisotropic(&f);
                snesim.simulate(
                    &grid(40),
                    None,
                    None,
                    local,
                    1,
                    0,
                    &Keep::None,
                    memory,
                    None,
                )
            };
            run(memory).map(|_| snesim.n_classes())
        };
        assert_eq!(classes(10.0, 0).unwrap(), 36);
        assert_eq!(classes(90.0, 0).unwrap(), 4);
        let e = classes(10.0, 20_000_000).unwrap_err().to_string();
        assert!(
            e.contains("36 template classes") && e.contains("widen angle_step (now 10)"),
            "{e}"
        );
        assert_eq!(classes(90.0, 20_000_000).unwrap(), 4);
    }

    #[test]
    fn anisotropic_zoned_output_is_the_same_for_any_thread_count() {
        let second = TrainingImage::categorical(&model([40, 40, 1], channels(40)), "v").unwrap();
        let snesim = Snesim::zoned(vec![ti(48), second], params()).unwrap();
        let f = field(1600, |m| {
            ((m % 40) as f64 * 4.0, 0.7, 1.0 + (m / 40) as f64 / 40.0)
        });
        let zones: Vec<usize> = (0..1600).map(|m| m % 3 / 2).collect();
        let local = SnesimLocal {
            zones: Some(&zones),
            anisotropy: Some(&f),
        };
        let locs = [(3.5, 4.5, 0.5), (20.5, 30.5, 0.5)];
        let data = Some((&locs[..], &[1usize, 0][..]));
        let go = |t| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap();
            pool.install(|| {
                snesim
                    .simulate(&grid(40), data, None, local, 4, 7, &Keep::All, 0, None)
                    .unwrap()
                    .realizations
            })
        };
        assert_eq!(go(1), go(4));
    }
}
