//! Image quilting: simulation patch by patch (Efros and Freeman, 2001;
//! Mariethoz and Lefebvre, 2014).
//!
//! The grid is covered by patches that overlap their neighbors, visited along
//! a raster path from the grid's first cell: x fastest, then y, then z. For
//! each patch, [`Quilting`]:
//!
//! 1. reads what the grid already holds under the patch: the cells of earlier
//!    patches and the hard data;
//! 2. scores every position of the training image against them ([`cost_map`]);
//! 3. picks one of the `n_best` cheapest positions at random;
//! 4. finds the seam through the overlap where the new patch and the old cells
//!    agree best ([`crate::seam`]);
//! 5. pastes the patch on the new side of the seam, never over a hard datum.
//!
//! A realization draws from one random stream in patch order, and the cost
//! map sums every position in a fixed order, so a realization does not depend
//! on the number of threads.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::error::{Result, SimError};
use crate::lattice::Lattice;
use crate::seam::{Side, path_cut, surface_cut};
use crate::training_image::{NO_CODE, TrainingImage, TrainingValues};

/// `[x, y, z]` of cell `index` of a box of `dims`, x fastest.
fn ijk_of(index: usize, dims: [usize; 3]) -> [usize; 3] {
    [
        index % dims[0],
        index / dims[0] % dims[1],
        index / (dims[0] * dims[1]),
    ]
}

/// One term of a patch's cost: a template compared with an image at every
/// position. Categorical values are codes, exact in `f32`; NaN is no data.
#[derive(Debug, Clone, Copy)]
pub struct Term<'a> {
    /// The image, x fastest.
    pub image: &'a [f32],
    /// True when values are codes, compared for equality.
    pub categorical: bool,
    /// `1 / range` of a continuous image, 0 for a constant one.
    pub inv_range: f64,
    /// What the grid holds under the patch, x fastest over the patch box;
    /// NaN where there is nothing to compare.
    pub template: &'a [f32],
    /// The weight of each cell of the template.
    pub weights: &'a [f64],
}

impl Term<'_> {
    /// Mismatch in `[0, 1]`: the squared difference over the squared range,
    /// or 0 if equal and 1 otherwise for codes. Image no data is a full
    /// mismatch.
    fn mismatch(&self, image: f32, template: f32) -> f64 {
        if image.is_nan() {
            1.0
        } else if self.categorical {
            f64::from(u8::from(image != template))
        } else {
            let d = (f64::from(image) - f64::from(template)) * self.inv_range;
            d * d
        }
    }
}

/// The cost of every patch position of an image of `image_dims`:
///
/// `cost(t) = Σ_terms Σ_c weights[c] · mismatch(image[t + c], template[c])`
///
/// over the cells `c` of the patch box with a template value and a positive
/// weight. Position `t = (tx, ty, tz)`, where the patch's first cell lies in
/// the image, is at `tx + mx * (ty + my * tz)` of the result, with
/// `m = image_dims - patch + 1` positions along each axis.
///
/// A direct scan, in parallel over rows of positions; each cost is summed in
/// a fixed order.
pub fn cost_map(image_dims: [usize; 3], patch: [usize; 3], terms: &[Term<'_>]) -> Vec<f64> {
    let [nx, ny, _] = image_dims;
    let positions: [usize; 3] = std::array::from_fn(|a| image_dims[a] - patch[a] + 1);
    let compared: Vec<Vec<(usize, f32, f64)>> = terms
        .iter()
        .map(|term| compared_cells(term, image_dims, patch))
        .collect();
    let mut costs = vec![0.0; positions.iter().product()];
    costs
        .par_chunks_mut(positions[0])
        .enumerate()
        .for_each(|(row, costs)| {
            let (ty, tz) = (row % positions[1], row / positions[1]);
            let row_start = nx * (ty + ny * tz);
            for (tx, cost) in costs.iter_mut().enumerate() {
                for (term, cells) in terms.iter().zip(&compared) {
                    for &(offset, value, weight) in cells {
                        let image = term.image[row_start + tx + offset];
                        *cost += weight * term.mismatch(image, value);
                    }
                }
            }
        });
    costs
}

/// The cells of `term` that [`cost_map`] compares, as (offset in the image
/// from a position's first cell, template value, weight).
fn compared_cells(
    term: &Term<'_>,
    image_dims: [usize; 3],
    patch: [usize; 3],
) -> Vec<(usize, f32, f64)> {
    let [nx, ny, _] = image_dims;
    term.template
        .iter()
        .zip(term.weights)
        .enumerate()
        .filter(|(_, (value, weight))| !value.is_nan() && **weight > 0.0)
        .map(|(cell, (&value, &weight))| {
            let [u, v, w] = ijk_of(cell, patch);
            (u + nx * (v + ny * w), value, weight)
        })
        .collect()
}

/// Image quilting parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuiltingParams {
    /// Cells of a patch along x, y and z, each at least 1; clamped to the
    /// training image and to the grid.
    pub patch_size: [usize; 3],
    /// Cells a patch shares with the patch before it along each axis, at most
    /// half the patch; `None` is a sixth of the patch, at least 1 cell.
    pub overlap: Option<[usize; 3]>,
    /// How many of the cheapest positions to pick from, at least 1.
    pub n_best: usize,
    /// Weight of the hard data against the overlap in a position's cost; 0
    /// leaves them out of the choice.
    pub data_weight: f64,
}

impl Default for QuiltingParams {
    fn default() -> Self {
        Self {
            patch_size: [40; 3],
            overlap: None,
            n_best: 10,
            data_weight: 5.0,
        }
    }
}

impl QuiltingParams {
    pub fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(SimError::InvalidParameters(m));
        if self.patch_size.contains(&0) {
            return bad(format!(
                "patch_size is {:?}; it must be at least 1 cell along every axis",
                self.patch_size
            ));
        }
        if let Some(overlap) = self.overlap
            && (0..3).any(|a| self.patch_size[a] > 1 && overlap[a] > self.patch_size[a] / 2)
        {
            return bad(format!(
                "overlap is {overlap:?}, more than half of patch_size {:?} along an axis",
                self.patch_size
            ));
        }
        if self.n_best == 0 {
            return bad("n_best must be at least 1".into());
        }
        if !(self.data_weight.is_finite() && self.data_weight >= 0.0) {
            return bad(format!(
                "data_weight is {}; it must be finite and at least 0",
                self.data_weight
            ));
        }
        Ok(())
    }
}

/// The patches that cover a grid: their size, their overlap, and how many
/// there are along each axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatchGrid {
    pub patch: [usize; 3],
    pub overlap: [usize; 3],
    /// Patches along each axis, at least 1.
    pub counts: [usize; 3],
}

impl PatchGrid {
    /// Patches of `patch_size` cells, each axis clamped to the training image
    /// and the grid, starting every `patch - overlap` cells; the last along an
    /// axis may reach past the grid. `overlap` defaults to a sixth of the
    /// clamped patch, at least 1, and is at most half of it.
    pub fn new(
        patch_size: [usize; 3],
        overlap: Option<[usize; 3]>,
        ti_dims: [usize; 3],
        grid_dims: [usize; 3],
    ) -> Self {
        let patch: [usize; 3] =
            std::array::from_fn(|a| patch_size[a].min(ti_dims[a]).min(grid_dims[a]));
        let overlap: [usize; 3] = std::array::from_fn(|a| {
            overlap
                .map_or((patch[a] / 6).max(1), |o| o[a])
                .min(patch[a] / 2)
        });
        let counts = std::array::from_fn(|a| match grid_dims[a] <= patch[a] {
            true => 1,
            false => (grid_dims[a] - overlap[a]).div_ceil(patch[a] - overlap[a]),
        });
        Self {
            patch,
            overlap,
            counts,
        }
    }

    pub fn len(&self) -> usize {
        self.counts.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Patch `k` of the raster path: its index along each axis.
    fn index(&self, k: usize) -> [usize; 3] {
        ijk_of(k, self.counts)
    }

    /// The grid cell where patch `k` starts.
    fn origin(&self, k: usize) -> [i64; 3] {
        let index = self.index(k);
        std::array::from_fn(|a| (index[a] * (self.patch[a] - self.overlap[a])) as i64)
    }
}

/// Positions of an image of `dims` whose patch holds no no-data cell, in
/// [`cost_map`] order.
fn valid_positions(image: &[f32], dims: [usize; 3], patch: [usize; 3]) -> Vec<bool> {
    let positions: [usize; 3] = std::array::from_fn(|a| dims[a] - patch[a] + 1);
    let mut valid = vec![true; positions.iter().product()];
    for (cell, value) in image.iter().enumerate() {
        if !value.is_nan() {
            continue;
        }
        let ijk = ijk_of(cell, dims);
        let [xs, ys, zs] = std::array::from_fn(|a| {
            (ijk[a] + 1).saturating_sub(patch[a])..(ijk[a] + 1).min(positions[a])
        });
        for tz in zs {
            for ty in ys.clone() {
                let row = positions[0] * (ty + positions[1] * tz);
                valid[row + xs.start..row + xs.end].fill(false);
            }
        }
    }
    valid
}

/// Box of `shape` cells of `values` (a grid of `dims`) from `origin`, x
/// fastest; `outside` beyond the grid.
fn cells_in_box<T: Copy>(
    values: &[T],
    dims: [usize; 3],
    origin: [i64; 3],
    shape: [usize; 3],
    outside: T,
) -> Vec<T> {
    (0..shape.iter().product())
        .map(|cell| {
            let p = ijk_of(cell, shape);
            let at: [i64; 3] = std::array::from_fn(|a| origin[a] + p[a] as i64);
            match (0..3).all(|a| (0..dims[a] as i64).contains(&at[a])) {
                true => {
                    values[at[0] as usize + dims[0] * (at[1] as usize + dims[1] * at[2] as usize)]
                }
                false => outside,
            }
        })
        .collect()
}

/// Not a node (or outside the grid), in [`Quilting`]'s roles.
const INACTIVE: u8 = 0;
/// A node to simulate.
const SIMULATE: u8 = 1;
/// A node holding a hard datum.
const HARD: u8 = 2;

/// Largest mismatch between a hard datum and the patch around it for the two
/// to agree: the same code, or a value within 5 % of the image's range.
const AGREEMENT_TOLERANCE: f64 = 0.05;

/// One realization.
#[derive(Debug, Clone)]
pub struct Quilt {
    /// The value, or code, at each node.
    pub values: Vec<f64>,
    /// Hard data the patch pasted around them disagreed with.
    pub disagreed: usize,
}

/// A hard datum: its grid cell, node and value.
#[derive(Debug, Clone, Copy)]
struct Hard {
    cell: usize,
    node: usize,
    value: f64,
}

/// Everything the realizations of an image-quilting run share.
#[derive(Debug, Clone)]
pub struct Quilting {
    lattice: Lattice,
    /// The training image as [`cost_map`] reads it; NaN where it has no data.
    image: Vec<f32>,
    ti_dims: [usize; 3],
    categorical: bool,
    inv_range: f64,
    patches: PatchGrid,
    n_best: usize,
    data_weight: f64,
    /// Positions that may be pasted, at least one.
    valid: Vec<bool>,
    /// Role of every grid cell.
    roles: Vec<u8>,
    /// Sorted by cell.
    hard: Vec<Hard>,
}

impl Quilting {
    /// Prepares the simulation of `lattice` from `ti`, conditioned on hard
    /// `data` given as (location, value). A datum outside the nodes is
    /// ignored; of several in one node the first is kept. Categorical data
    /// must be codes of the image.
    pub fn new(
        ti: &TrainingImage,
        lattice: &Lattice,
        data: &[([f64; 3], f64)],
        params: &QuiltingParams,
    ) -> Result<Self> {
        params.validate()?;
        let geometry = lattice.geometry();
        let grid = geometry.count;
        let cells = usize::try_from(geometry.cells())
            .map_err(|_| SimError::InvalidParameters("the grid has too many cells".into()))?;
        let ti_dims = ti.dims();
        let image: Vec<f32> = match ti.values() {
            TrainingValues::Categorical(codes) => codes
                .iter()
                .map(|&c| if c == NO_CODE { f32::NAN } else { f32::from(c) })
                .collect(),
            TrainingValues::Continuous(values) => values.clone(),
        };
        let patches = PatchGrid::new(params.patch_size, params.overlap, ti_dims, grid);
        let valid = valid_positions(&image, ti_dims, patches.patch);
        if !valid.contains(&true) {
            return Err(SimError::InsufficientData(format!(
                "no place in the training image holds a patch of {:?} cells without a null; \
                 use a smaller patch_size",
                patches.patch
            )));
        }
        let mut roles = vec![INACTIVE; cells];
        for node in 0..lattice.len() {
            roles[lattice.cell(node) as usize] = SIMULATE;
        }
        let categorical = ti.is_categorical();
        let k = ti.n_categories();
        let mut hard = Vec::new();
        for &(p, value) in data {
            let ok = match categorical {
                true => value.fract() == 0.0 && (0.0..k as f64).contains(&value),
                false => value.is_finite(),
            };
            if !ok {
                return Err(SimError::InvalidParameters(match categorical {
                    true => format!(
                        "hard datum {value} is not a code of the training image (0 to {})",
                        k.saturating_sub(1)
                    ),
                    false => format!("hard datum {value} is not finite"),
                }));
            }
            let Some((cell, _)) = geometry.locate(p) else {
                continue;
            };
            let Some(node) = lattice.node_at(cell) else {
                continue;
            };
            let cell = cell as usize;
            if roles[cell] == SIMULATE {
                roles[cell] = HARD;
                hard.push(Hard { cell, node, value });
            }
        }
        hard.sort_by_key(|h| h.cell);
        let range = ti.value_range();
        Ok(Self {
            lattice: lattice.clone(),
            image,
            ti_dims,
            categorical,
            inv_range: if range > 0.0 { 1.0 / range } else { 0.0 },
            patches,
            n_best: params.n_best,
            data_weight: params.data_weight,
            valid,
            roles,
            hard,
        })
    }

    /// The patches covering the grid, after clamping.
    pub fn patches(&self) -> &PatchGrid {
        &self.patches
    }

    /// Number of nodes holding a hard datum.
    pub fn n_hard(&self) -> usize {
        self.hard.len()
    }

    /// One realization, fixed by `seed`.
    pub fn simulate(&self, seed: u64) -> Quilt {
        let mut store = vec![f32::NAN; self.roles.len()];
        for h in &self.hard {
            store[h.cell] = h.value as f32;
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut disagrees = vec![false; self.hard.len()];
        for k in 0..self.patches.len() {
            self.paste(&mut store, &mut rng, k, &mut disagrees);
        }
        let mut values: Vec<f64> = (0..self.lattice.len())
            .map(|n| f64::from(store[self.lattice.cell(n) as usize]))
            .collect();
        for h in &self.hard {
            values[h.node] = h.value;
        }
        Quilt {
            values,
            disagreed: disagrees.iter().filter(|&&d| d).count(),
        }
    }

    /// Mismatch between two values, in `[0, 1]`.
    fn mismatch(&self, a: f32, b: f32) -> f64 {
        match self.categorical {
            true => f64::from(u8::from(a != b)),
            false => f64::from((a - b).abs()) * self.inv_range,
        }
    }

    /// Chooses patch `k` of the raster path and pastes it into `store`.
    fn paste(&self, store: &mut [f32], rng: &mut StdRng, k: usize, disagrees: &mut [bool]) {
        let patch = self.patches.patch;
        let grid = self.lattice.geometry().count;
        let origin = self.patches.origin(k);
        let old = cells_in_box(store, grid, origin, patch, f32::NAN);
        let roles = cells_in_box(&self.roles, grid, origin, patch, INACTIVE);
        if !(0..old.len()).any(|c| roles[c] == SIMULATE && old[c].is_nan()) {
            return;
        }

        // 1. The cost of every position: the mean mismatch over the cells of
        //    earlier patches plus `data_weight` times that over the hard data.
        let n_data = roles.iter().filter(|&&r| r == HARD).count();
        let n_old = old.iter().filter(|v| !v.is_nan()).count() - n_data;
        let weights: Vec<f64> = (0..old.len())
            .map(|c| match roles[c] {
                HARD => self.data_weight / n_data as f64,
                SIMULATE if !old[c].is_nan() => 1.0 / n_old as f64,
                _ => 0.0,
            })
            .collect();
        let terms = [Term {
            image: &self.image,
            categorical: self.categorical,
            inv_range: self.inv_range,
            template: &old,
            weights: &weights,
        }];
        let costs = cost_map(self.ti_dims, patch, &terms);

        // 2. One of the `n_best` cheapest positions.
        let position = self.choose(&costs, rng);
        let positions: [usize; 3] = std::array::from_fn(|a| self.ti_dims[a] - patch[a] + 1);
        let t = ijk_of(position, positions);
        let [nx, ny, _] = self.ti_dims;
        let new: Vec<f32> = (0..old.len())
            .map(|c| {
                let [u, v, w] = ijk_of(c, patch);
                self.image[(t[0] + u) + nx * ((t[1] + v) + ny * (t[2] + w))]
            })
            .collect();

        // 3. The seam: which cells take the new patch.
        let errors: Vec<f64> = (0..old.len())
            .map(|c| match old[c].is_nan() {
                true => 0.0,
                false => self.mismatch(old[c], new[c]).powi(2),
            })
            .collect();
        let takes_new = if n_old + n_data == 0 {
            vec![true; old.len()]
        } else if let Some(flat) = patch.iter().position(|&n| n == 1) {
            let [a, b] = match flat {
                0 => [1, 2],
                1 => [0, 2],
                _ => [0, 1],
            };
            let index = self.patches.index(k);
            let overlap = |axis: usize| match index[axis] {
                0 => 0,
                _ => self.patches.overlap[axis],
            };
            path_cut(&errors, [patch[a], patch[b]], [overlap(a), overlap(b)])
        } else {
            self.surface(store, origin, &old, &roles, &errors)
        };

        // 4. Paste every cell to simulate that is empty or on the new side; a
        //    hard datum keeps its value.
        for c in 0..old.len() {
            let p = ijk_of(c, patch);
            let [i, j, l] = std::array::from_fn(|a| origin[a] as usize + p[a]);
            let cell = i + grid[0] * (j + grid[1] * l);
            match roles[c] {
                HARD if takes_new[c] => {
                    if let Ok(d) = self.hard.binary_search_by_key(&cell, |h| h.cell) {
                        disagrees[d] = self.mismatch(old[c], new[c]) > AGREEMENT_TOLERANCE;
                    }
                }
                SIMULATE if old[c].is_nan() || takes_new[c] => store[cell] = new[c],
                _ => {}
            }
        }
    }

    /// One of the `n_best` cheapest valid positions. The scan starts at a
    /// random position and wraps around, and of equal costs the first found
    /// wins, so ties are taken from anywhere in the image.
    fn choose(&self, costs: &[f64], rng: &mut StdRng) -> usize {
        let n = costs.len();
        let start = rng.gen_range(0..n);
        let mut best: Vec<(f64, usize)> = Vec::with_capacity(self.n_best + 1);
        for step in 0..n {
            let position = (start + step) % n;
            let cost = costs[position];
            let full = best.len() == self.n_best;
            if !self.valid[position] || (full && cost >= best[self.n_best - 1].0) {
                continue;
            }
            let at = best.partition_point(|&(other, _)| other <= cost);
            best.insert(at, (cost, position));
            best.truncate(self.n_best);
        }
        best[rng.gen_range(0..best.len())].1
    }

    /// Which cells of a 3D patch at `origin` take the new patch
    /// ([`surface_cut`]); `old`, `roles` and `errors` cover the patch box.
    fn surface(
        &self,
        store: &[f32],
        origin: [i64; 3],
        old: &[f32],
        roles: &[u8],
        errors: &[f64],
    ) -> Vec<bool> {
        let patch = self.patches.patch;
        let grid = self.lattice.geometry().count;
        // The box and the cells around it: an old cell next to the patch
        // joins it too.
        let shape = patch.map(|n| n + 2);
        let around = cells_in_box(store, grid, origin.map(|c| c - 1), shape, f32::NAN);
        let mut sides: Vec<Side> = around
            .iter()
            .map(|v| match v.is_nan() {
                true => Side::Nothing,
                false => Side::Old,
            })
            .collect();
        let mut errors_around = vec![0.0; sides.len()];
        let inner = |c: usize| {
            let [u, v, w] = ijk_of(c, patch);
            (u + 1) + shape[0] * ((v + 1) + shape[1] * (w + 1))
        };
        for c in 0..old.len() {
            sides[inner(c)] = match (old[c].is_nan(), roles[c]) {
                (false, _) => Side::Overlap,
                (true, SIMULATE) => Side::New,
                (true, _) => Side::Nothing,
            };
            errors_around[inner(c)] = errors[c];
        }
        let new_side = surface_cut(&sides, &errors_around, shape);
        (0..old.len()).map(|c| new_side[inner(c)]).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Float64Array, RecordBatch};
    use ceres_core::{BlockModel, Geometry};

    use super::*;

    fn geometry(count: [usize; 3]) -> Geometry {
        Geometry {
            origin: [0.0; 3],
            size: [1.0; 3],
            count,
            rotation: [0.0; 3],
        }
    }

    fn model(count: [usize; 3], values: Vec<Option<f64>>) -> BlockModel {
        let column = Arc::new(Float64Array::from(values));
        let batch = RecordBatch::try_from_iter([("v", column as _)]).unwrap();
        BlockModel::regular(geometry(count), batch).unwrap()
    }

    fn image(count: [usize; 3], f: impl Fn([usize; 3]) -> f64, categorical: bool) -> TrainingImage {
        let n = count.iter().product();
        let values = (0..n).map(|i| Some(f(ijk_of(i, count)))).collect();
        let m = model(count, values);
        match categorical {
            true => TrainingImage::categorical(&m, "v").unwrap(),
            false => TrainingImage::continuous(&m, "v").unwrap(),
        }
    }

    /// Meandering channels (code 1) in a background (code 0).
    fn channels(count: [usize; 3]) -> TrainingImage {
        image(
            count,
            |[x, y, _]| {
                let centre = 6.0 * (x as f64 / 7.0).sin();
                f64::from(u8::from((y as f64 - centre).rem_euclid(16.0) < 5.0))
            },
            true,
        )
    }

    fn field(count: [usize; 3]) -> TrainingImage {
        image(
            count,
            |[x, y, _]| (x as f64 / 6.0).sin() * (y as f64 / 9.0).cos() + x as f64 / 80.0,
            false,
        )
    }

    fn lattice(count: [usize; 3]) -> Lattice {
        Lattice::regular(geometry(count))
    }

    fn params(patch: usize) -> QuiltingParams {
        QuiltingParams {
            patch_size: [patch; 3],
            ..Default::default()
        }
    }

    #[test]
    fn the_cost_map_is_the_weighted_mismatch_at_every_position() {
        #[rustfmt::skip]
        let image = [
            0.0, 1.0, 2.0, 3.0,
            4.0, 5.0, 6.0, f32::NAN,
            8.0, 9.0, 10.0, 7.0,
        ];
        let template = [1.0, 3.0, f32::NAN, f32::NAN];
        let term = Term {
            image: &image,
            categorical: false,
            inv_range: 0.1,
            template: &template,
            weights: &[1.0, 2.0, 5.0, 0.0],
        };
        let costs = cost_map([4, 3, 1], [2, 2, 1], &[term]);
        // At (0, 0): (0 - 1)² + 2 (1 - 3)² = 9, over the range squared; the
        // last position meets the no-data cell, a full mismatch.
        let expected = [0.09, 0.02, 0.01, 0.17, 0.34, 0.25 + 2.0];
        assert_eq!(costs.len(), 6);
        for (cost, expected) in costs.iter().zip(expected) {
            assert!((cost - expected).abs() < 1e-12, "{costs:?}");
        }
        let categorical = Term {
            categorical: true,
            ..term
        };
        let costs = cost_map([4, 3, 1], [2, 2, 1], &[categorical, categorical]);
        assert_eq!(costs, [6.0, 4.0, 2.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn patches_cover_the_grid_and_clamp_to_it_and_to_the_image() {
        let patches = PatchGrid::new([6, 6, 6], Some([2, 2, 2]), [50, 50, 1], [10, 11, 1]);
        assert_eq!(patches.patch, [6, 6, 1]);
        assert_eq!(patches.overlap, [2, 2, 0]);
        assert_eq!(patches.counts, [2, 3, 1]);
        assert_eq!(patches.len(), 6);
        assert_eq!(patches.origin(1), [4, 0, 0]);
        assert_eq!(patches.origin(5), [4, 8, 0]);
        let patches = PatchGrid::new([30, 4, 2], None, [50, 50, 50], [100, 100, 100]);
        assert_eq!(patches.overlap, [5, 1, 1]);
        let patches = PatchGrid::new([40, 40, 40], Some([6, 6, 6]), [20, 100, 100], [100, 8, 1]);
        assert_eq!(patches.patch, [20, 8, 1]);
        assert_eq!(patches.overlap, [6, 4, 0]);
        assert_eq!(patches.counts, [7, 1, 1]);
    }

    #[test]
    fn a_position_is_valid_when_its_patch_holds_no_null() {
        let mut values = vec![0.0f32; 12];
        values[2 + 4] = f32::NAN;
        let valid = valid_positions(&values, [4, 3, 1], [2, 2, 1]);
        assert_eq!(valid, [true, false, false, true, false, false]);
    }

    /// A grid quilted from a continuous image of which only position 0 may be
    /// pasted: every patch is the image's first `patch` cells.
    fn quilted_from_position_0(
        values: &[f64],
        ti_dims: [usize; 3],
        patch: [usize; 3],
        overlap: [usize; 3],
        grid: [usize; 3],
    ) -> Vec<f64> {
        let ti = image(
            ti_dims,
            |[x, y, z]| values[x + ti_dims[0] * (y + ti_dims[1] * z)],
            false,
        );
        let params = QuiltingParams {
            patch_size: patch,
            overlap: Some(overlap),
            n_best: 1,
            data_weight: 1.0,
        };
        let mut quilting = Quilting::new(&ti, &lattice(grid), &[], &params).unwrap();
        quilting.valid.fill(false);
        quilting.valid[0] = true;
        quilting.simulate(0).values
    }

    #[test]
    fn a_flat_patch_is_pasted_on_the_new_side_of_its_paths() {
        #[rustfmt::skip]
        let values = [
            1.0, 9.0, 1.0, 2.0,  0.0, 0.0, 0.0, 0.0,
            7.0, 3.0, 0.0, 3.0,  0.0, 0.0, 0.0, 0.0,
            8.0, 4.0, 2.0, 4.0,  0.0, 0.0, 0.0, 0.0,
        ];
        let quilted = quilted_from_position_0(&values, [8, 3, 1], [4, 3, 1], [2, 0, 0], [6, 3, 1]);
        // Row 0 agrees in column 2, so the path runs there; rows 1 and 2 agree
        // in column 3, so column 2 stays old.
        #[rustfmt::skip]
        let expected = [
            1.0, 9.0, 1.0, 9.0, 1.0, 2.0,
            7.0, 3.0, 0.0, 3.0, 0.0, 3.0,
            8.0, 4.0, 2.0, 4.0, 2.0, 4.0,
        ];
        assert_eq!(quilted, expected);
    }

    #[test]
    fn a_3d_patch_is_pasted_on_the_new_side_of_its_surface() {
        #[rustfmt::skip]
        let values = [
            1.0, 5.0, 2.0, 3.0, 5.0, 5.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 5.0, 2.0, 5.0, 5.0, 8.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            2.0, 6.0, 3.0, 4.0, 6.0, 6.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 4.0, 1.0, 4.0, 4.0, 7.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let quilted = quilted_from_position_0(&values, [12, 2, 2], [6, 2, 2], [3, 0, 0], [9, 2, 2]);
        // Every row agrees in column 4 and differs least in column 3: the
        // surface runs between them.
        #[rustfmt::skip]
        let expected = [
            1.0, 5.0, 2.0, 3.0, 5.0, 2.0, 3.0, 5.0, 5.0,
            1.0, 5.0, 2.0, 5.0, 5.0, 2.0, 5.0, 5.0, 8.0,
            2.0, 6.0, 3.0, 4.0, 6.0, 3.0, 4.0, 6.0, 6.0,
            0.0, 4.0, 1.0, 4.0, 4.0, 1.0, 4.0, 4.0, 7.0,
        ];
        assert_eq!(quilted, expected);
    }

    #[test]
    fn category_proportions_match_the_training_image() {
        let ti = channels([80, 80, 1]);
        let q = Quilting::new(&ti, &lattice([64, 64, 1]), &[], &params(16)).unwrap();
        let n = 12;
        let mut ones = 0.0;
        let mut runs = vec![];
        for seed in 0..n {
            let quilt = q.simulate(seed);
            assert!(quilt.values.iter().all(|&v| v == 0.0 || v == 1.0));
            ones += quilt.values.iter().sum::<f64>() / (64.0 * 64.0 * n as f64);
            runs.push(quilt.values);
        }
        let want = ti.proportions()[1];
        assert!((ones - want).abs() < 0.04, "{ones} vs {want}");
        assert_ne!(runs[0], runs[1]);
        assert_eq!(runs[0], q.simulate(0).values);
    }

    #[test]
    fn a_continuous_histogram_matches_the_training_image() {
        let ti = field([80, 80, 1]);
        let q = Quilting::new(&ti, &lattice([60, 60, 1]), &[], &params(14)).unwrap();
        let mut simulated: Vec<f64> = (0..10).flat_map(|s| q.simulate(s).values).collect();
        let mut reference: Vec<f64> = ti
            .continuous_values()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        simulated.sort_by(f64::total_cmp);
        reference.sort_by(f64::total_cmp);
        let quantile = |v: &[f64], p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
        for p in [0.05, 0.25, 0.5, 0.75, 0.95] {
            let (got, want) = (quantile(&simulated, p), quantile(&reference, p));
            assert!(
                (got - want).abs() < 0.08 * ti.value_range(),
                "q{p}: {got} vs {want}"
            );
        }
        let [lo, hi] = ti.range();
        assert!(simulated.iter().all(|v| (lo..=hi).contains(v)));
    }

    /// Every `every`-th cell of the image as hard data at the same cell.
    fn hard_from(ti: &TrainingImage, count: [usize; 3], every: usize) -> Vec<([f64; 3], f64)> {
        let codes = ti.codes().unwrap();
        (0..count[0] * count[1])
            .step_by(every)
            .map(|i| {
                let [x, y, _] = ijk_of(i, count);
                ([x as f64 + 0.5, y as f64 + 0.5, 0.5], f64::from(codes[i]))
            })
            .collect()
    }

    #[test]
    fn hard_data_are_never_overwritten_and_steer_the_patches() {
        let count = [60, 60, 1];
        let ti = channels(count);
        let data = hard_from(&ti, count, 37);
        let grid = lattice(count);
        let run = |weight: f64| {
            let p = QuiltingParams {
                data_weight: weight,
                ..params(15)
            };
            let q = Quilting::new(&ti, &grid, &data, &p).unwrap();
            assert_eq!(q.n_hard(), data.len());
            let mut disagreed = 0;
            for seed in 0..4 {
                let quilt = q.simulate(seed);
                for (p, v) in &data {
                    let node = grid.node_at(grid.geometry().locate(*p).unwrap().0).unwrap();
                    assert_eq!(quilt.values[node], *v);
                }
                disagreed += quilt.disagreed;
            }
            disagreed
        };
        let (free, steered) = (run(0.0), run(50.0));
        // Of 4 x 98 data, left free about 40 % disagree with the patch
        // around them; steered, next to none.
        let n = 4 * data.len();
        assert!(
            5 * free > n && 50 * steered < n,
            "{steered} and {free} of {n}"
        );
    }

    #[test]
    fn a_masked_grid_is_filled_and_nothing_else() {
        let ti = channels([50, 50, 1]);
        let cells: Vec<u64> = (0..40 * 30)
            .filter(|c| (c % 40 + c / 40) % 3 != 0)
            .collect();
        let n = cells.len();
        let grid = Lattice::masked(geometry([40, 30, 1]), cells).unwrap();
        let data = [
            ([0.5, 0.5, 0.5], 1.0),
            ([1e3, 0.5, 0.5], 1.0),
            ([1.5, 0.5, 0.5], 0.0),
        ];
        let q = Quilting::new(&ti, &grid, &data, &params(12)).unwrap();
        // Cell 0 is outside the mask and the second datum outside the grid.
        assert_eq!(q.n_hard(), 1);
        let quilt = q.simulate(3);
        assert_eq!(quilt.values.len(), n);
        assert!(quilt.values.iter().all(|&v| v == 0.0 || v == 1.0));
        assert_eq!(quilt.values[0], 0.0);
    }

    #[test]
    fn a_3d_run_fills_the_grid() {
        let ti = image([16, 16, 8], |[x, _, z]| ((x + 2 * z) / 4 % 2) as f64, true);
        let q = Quilting::new(
            &ti,
            &lattice([12, 11, 8]),
            &[],
            &QuiltingParams {
                patch_size: [6, 6, 4],
                ..Default::default()
            },
        )
        .unwrap();
        let quilt = q.simulate(1);
        assert_eq!(quilt.values.len(), 12 * 11 * 8);
        assert!(quilt.values.iter().all(|&v| v == 0.0 || v == 1.0));
    }

    #[test]
    fn realizations_do_not_depend_on_the_thread_count() {
        let ti = field([50, 50, 1]);
        let q = Quilting::new(&ti, &lattice([45, 40, 1]), &[], &params(12)).unwrap();
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| (0..3).map(|s| q.simulate(s).values).collect::<Vec<_>>())
        };
        assert_eq!(run(1), run(4));
    }

    #[test]
    fn bad_parameters_and_data_are_rejected() {
        let ti = channels([30, 30, 1]);
        let grid = lattice([20, 20, 1]);
        let fails = |p: QuiltingParams| Quilting::new(&ti, &grid, &[], &p).is_err();
        assert!(fails(QuiltingParams {
            patch_size: [12, 0, 1],
            ..Default::default()
        }));
        assert!(fails(QuiltingParams {
            patch_size: [12, 12, 1],
            overlap: Some([7, 2, 0]),
            ..Default::default()
        }));
        assert!(fails(QuiltingParams {
            n_best: 0,
            ..Default::default()
        }));
        assert!(fails(QuiltingParams {
            data_weight: f64::NAN,
            ..Default::default()
        }));
        let at = [0.5, 0.5, 0.5];
        assert!(Quilting::new(&ti, &grid, &[(at, 2.0)], &params(8)).is_err());
        assert!(Quilting::new(&ti, &grid, &[(at, 0.5)], &params(8)).is_err());
        let holes = model(
            [4, 4, 1],
            (0..16).map(|i| (i % 3 != 0).then_some(1.0)).collect(),
        );
        let ti = TrainingImage::categorical(&holes, "v").unwrap();
        assert!(Quilting::new(&ti, &grid, &[], &params(3)).is_err());
    }
}
