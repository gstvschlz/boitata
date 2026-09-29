//! Simulation on a lattice: the cells of a regular or masked block model,
//! one node each, a multigrid path over them and templates of cell offsets.

use ceres_core::{BlockModel, Geometry, Layout, block_frame};
use nalgebra::{Matrix3, Vector3};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;

use crate::error::{Result, SimError};

/// The cells of a regular or masked block model, one node each, in cell
/// index order (x fastest, then y, then z).
#[derive(Debug, Clone)]
pub struct Lattice {
    geometry: Geometry,
    /// The cell of each node, strictly increasing; `None` when every cell is a node.
    cells: Option<Vec<u64>>,
}

impl Lattice {
    pub fn regular(geometry: Geometry) -> Self {
        Self {
            geometry,
            cells: None,
        }
    }

    /// Nodes at `cells`, which must be strictly increasing cell indices.
    pub fn masked(geometry: Geometry, cells: Vec<u64>) -> Result<Self> {
        if cells.windows(2).any(|w| w[0] >= w[1]) {
            return Err(SimError::InvalidParameters(
                "cells must be strictly increasing".into(),
            ));
        }
        if cells.last().is_some_and(|&c| c >= geometry.cells()) {
            return Err(SimError::InvalidParameters("cell outside the grid".into()));
        }
        Ok(Self {
            geometry,
            cells: Some(cells),
        })
    }

    /// The lattice of a regular or masked block model; `None` when sub-blocked.
    pub fn from_model(model: &BlockModel) -> Option<Self> {
        match model.layout() {
            Layout::Regular => Some(Self::regular(*model.geometry())),
            Layout::Masked(cells) => Some(Self {
                geometry: *model.geometry(),
                cells: Some(cells.clone()),
            }),
            Layout::SubBlocked { .. } => None,
        }
    }

    pub fn len(&self) -> usize {
        self.cells
            .as_ref()
            .map_or(self.geometry.cells() as usize, Vec::len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    pub fn cell(&self, node: usize) -> u64 {
        self.cells.as_ref().map_or(node as u64, |c| c[node])
    }

    pub fn node_at(&self, cell: u64) -> Option<usize> {
        match &self.cells {
            None => (cell < self.geometry.cells()).then_some(cell as usize),
            Some(c) => c.binary_search(&cell).ok(),
        }
    }

    /// Centroid of the cell of `node`.
    pub fn location(&self, node: usize) -> (f64, f64, f64) {
        let [x, y, z] = self.geometry.centroid(self.cell(node));
        (x, y, z)
    }

    /// The node `delta` cells from `node`, when that cell is in the grid and a node.
    pub fn shifted(&self, node: usize, delta: [i32; 3]) -> Option<usize> {
        let ijk = self.geometry.ijk(self.cell(node));
        let mut to = [0usize; 3];
        for a in 0..3 {
            let c = ijk[a] as i64 + delta[a] as i64;
            if !(0..self.geometry.count[a] as i64).contains(&c) {
                return None;
            }
            to[a] = c as usize;
        }
        self.node_at(self.geometry.index(to))
    }

    /// World vector spanned by `delta` cells.
    pub fn span(&self, delta: [i32; 3]) -> Vector3<f64> {
        let local = Vector3::from_fn(|a, _| delta[a] as f64 * self.geometry.size[a]);
        block_frame(self.geometry.rotation).transpose() * local
    }
}

/// Level of cell `ijk`: the largest `l <= top` with every coordinate a
/// multiple of `2^l`.
fn level(ijk: [usize; 3], top: usize) -> usize {
    (0..=top)
        .rev()
        .find(|&l| ijk.iter().all(|&c| c % (1 << l) == 0))
        .unwrap_or(0)
}

/// The simulation path: nodes of level `top` first, then each finer level,
/// in a random order within a level fixed by `seed`, leaving out the nodes
/// `skip` marks; and the rank of each node on it, `u32::MAX` if left out.
pub fn multigrid_path(
    lattice: &Lattice,
    top: usize,
    seed: u64,
    skip: &[bool],
) -> Result<(Vec<u32>, Vec<u32>)> {
    let n = lattice.len();
    if n >= u32::MAX as usize {
        return Err(SimError::InvalidParameters(format!(
            "a shared path holds fewer than {} nodes, got {n}",
            u32::MAX
        )));
    }
    if skip.len() != n {
        return Err(SimError::InvalidParameters("one skip flag per node".into()));
    }
    let levels: Vec<u8> = (0..n)
        .into_par_iter()
        .map(|m| level(lattice.geometry().ijk(lattice.cell(m)), top) as u8)
        .collect();
    let mut rng = StdRng::seed_from_u64(ceres_core::rng::splitmix(seed));
    let mut path = Vec::with_capacity(n);
    for l in (0..=top).rev() {
        let start = path.len();
        path.extend(
            (0..n as u32).filter(|&m| levels[m as usize] as usize == l && !skip[m as usize]),
        );
        path[start..].shuffle(&mut rng);
    }
    let mut rank = vec![u32::MAX; n];
    for (r, &m) in path.iter().enumerate() {
        rank[m as usize] = r as u32;
    }
    Ok((path, rank))
}

/// Largest bounding box, in cells, a template enumerates.
pub const MAX_BOX: u64 = 1 << 24;

/// Cell offsets within a search radius, nearest first under the search
/// metric, ties broken by offset.
#[derive(Debug, Clone)]
pub struct Template {
    offsets: Vec<([i32; 3], f64)>,
    reach: [usize; 3],
}

impl Template {
    /// Offsets within `radius` of a cell, distances measured by `frame`
    /// (world vector to isotropic search space). An endless radius covers
    /// the grid; the radius shrinks until the offsets' bounding box holds
    /// at most [`MAX_BOX`] cells.
    pub fn new(lattice: &Lattice, frame: &Matrix3<f64>, radius: f64) -> Result<Self> {
        let unit = |a: usize| std::array::from_fn(|b| i32::from(a == b));
        let g = Matrix3::from_columns(&[0, 1, 2].map(|a| frame * lattice.span(unit(a))));
        let inv = g
            .try_inverse()
            .ok_or_else(|| SimError::InvalidParameters("degenerate search frame".into()))?;
        let count = lattice.geometry().count;
        let half = |r: f64| -> [usize; 3] {
            std::array::from_fn(|a| match count[a] {
                1 => 0,
                n => ((r * inv.row(a).norm()).ceil() as usize).min(n - 1),
            })
        };
        let mut r = if radius.is_finite() {
            radius
        } else {
            (0..3).map(|a| count[a] as f64 * g.column(a).norm()).sum()
        };
        let mut reach = half(r);
        while reach.iter().map(|&h| 2 * h as u64 + 1).product::<u64>() > MAX_BOX {
            r *= 0.8;
            reach = half(r);
        }
        let span = |h: usize| -(h as i32)..=h as i32;
        let mut offsets = vec![];
        for k in span(reach[2]) {
            for j in span(reach[1]) {
                for i in span(reach[0]) {
                    if (i, j, k) == (0, 0, 0) {
                        continue;
                    }
                    let d = (g * Vector3::new(i as f64, j as f64, k as f64)).norm();
                    if d <= r {
                        offsets.push(([i, j, k], d));
                    }
                }
            }
        }
        offsets.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        Ok(Self { offsets, reach })
    }

    pub fn offsets(&self) -> &[([i32; 3], f64)] {
        &self.offsets
    }

    pub fn reach(&self) -> [usize; 3] {
        self.reach
    }
}

/// Multigrid levels for `template`: the coarsest spacing stays within half
/// its reach along every axis with more than one cell; at most 6.
pub fn default_levels(lattice: &Lattice, template: &Template) -> usize {
    let count = lattice.geometry().count;
    let shortest = (0..3)
        .filter(|&a| count[a] > 1)
        .map(|a| template.reach()[a])
        .min()
        .unwrap_or(0);
    ((shortest / 2).max(1) as f64)
        .log2()
        .floor()
        .clamp(0.0, 6.0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(rotation: [f64; 3]) -> Geometry {
        Geometry {
            origin: [100.0, 200.0, 0.0],
            size: [5.0, 4.0, 2.0],
            count: [6, 5, 3],
            rotation,
        }
    }

    #[test]
    fn regular_nodes_are_cells() {
        let l = Lattice::regular(geometry([0.0; 3]));
        assert_eq!(l.len(), 90);
        assert_eq!(l.cell(17), 17);
        assert_eq!(l.node_at(17), Some(17));
        assert_eq!(l.location(0), (102.5, 202.0, 1.0));
        assert_eq!(l.shifted(0, [1, 1, 1]), Some(1 + 6 + 30));
        assert_eq!(l.shifted(0, [-1, 0, 0]), None);
        assert_eq!(l.shifted(5, [1, 0, 0]), None);
    }

    #[test]
    fn masked_nodes_skip_missing_cells() {
        let l = Lattice::masked(geometry([0.0; 3]), vec![0, 1, 7, 37]).unwrap();
        assert_eq!(l.len(), 4);
        assert_eq!(l.node_at(7), Some(2));
        assert_eq!(l.node_at(2), None);
        assert_eq!(l.shifted(0, [1, 1, 1]), Some(3));
        assert_eq!(l.shifted(0, [0, 1, 0]), None);
        assert!(Lattice::masked(geometry([0.0; 3]), vec![3, 3]).is_err());
        assert!(Lattice::masked(geometry([0.0; 3]), vec![90]).is_err());
    }

    #[test]
    fn span_follows_the_rotation() {
        let l = Lattice::regular(geometry([30.0, 10.0, 5.0]));
        let d = l.span([2, -1, 1]);
        let from = l.geometry().centroid(l.geometry().index([1, 1, 0]));
        let to = l.geometry().centroid(l.geometry().index([3, 0, 1]));
        for axis in 0..3 {
            assert!((to[axis] - from[axis] - d[axis]).abs() < 1e-9);
        }
    }

    fn unit_grid(count: [usize; 3]) -> Lattice {
        Lattice::regular(Geometry {
            origin: [0.0; 3],
            size: [1.0; 3],
            count,
            rotation: [0.0; 3],
        })
    }

    #[test]
    fn path_visits_every_node_once_coarse_levels_first() {
        let l = unit_grid([16, 8, 1]);
        let skip = vec![false; l.len()];
        let (path, rank) = multigrid_path(&l, 3, 5, &skip).unwrap();
        let mut seen = path.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..l.len() as u32).collect::<Vec<_>>());
        for (r, &n) in path.iter().enumerate() {
            assert_eq!(rank[n as usize], r as u32);
        }
        let lvl = |n: u32| level(l.geometry().ijk(n as u64), 3);
        assert!(path.windows(2).all(|w| lvl(w[0]) >= lvl(w[1])));
        assert_eq!(lvl(path[0]), 3);
    }

    #[test]
    fn path_is_fixed_by_the_seed_and_skips_nodes() {
        let l = unit_grid([10, 10, 1]);
        let mut skip = vec![false; 100];
        skip[42] = true;
        let (a, rank) = multigrid_path(&l, 2, 9, &skip).unwrap();
        let (b, _) = multigrid_path(&l, 2, 9, &skip).unwrap();
        let (c, _) = multigrid_path(&l, 2, 10, &skip).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 99);
        assert_eq!(rank[42], u32::MAX);
    }

    fn flat(count: [usize; 3]) -> Lattice {
        Lattice::regular(Geometry {
            origin: [0.0; 3],
            size: [2.0, 2.0, 1.0],
            count,
            rotation: [0.0; 3],
        })
    }

    #[test]
    fn template_holds_the_cells_within_the_radius_nearest_first() {
        let l = flat([50, 50, 1]);
        let t = Template::new(&l, &Matrix3::identity(), 7.0).unwrap();
        let brute: Vec<[i32; 3]> = (-4..=4)
            .flat_map(|i| (-4..=4).map(move |j| [i, j, 0]))
            .filter(|&d| d != [0, 0, 0] && l.span(d).norm() <= 7.0)
            .collect();
        assert_eq!(t.offsets().len(), brute.len());
        assert!(t.offsets().windows(2).all(|w| w[0].1 <= w[1].1));
        assert_eq!(t.offsets()[0].1, 2.0);
        assert_eq!(t.reach(), [4, 4, 0]);
    }

    #[test]
    fn template_follows_an_anisotropic_frame() {
        let l = flat([50, 50, 1]);
        let frame = Matrix3::from_diagonal(&Vector3::new(1.0, 0.25, 1.0));
        let t = Template::new(&l, &frame, 4.0).unwrap();
        assert!(
            t.offsets()
                .iter()
                .all(|(d, _)| d[0].abs() <= 2 && d[1].abs() <= 8)
        );
        assert!(t.offsets().iter().any(|(d, _)| d[1] == 8));
    }

    #[test]
    fn an_endless_radius_is_bounded_by_the_grid_and_the_box() {
        let l = flat([4000, 4000, 1]);
        let t = Template::new(&l, &Matrix3::identity(), f64::INFINITY).unwrap();
        let [a, b, c] = t.reach().map(|h| 2 * h as u64 + 1);
        assert!(a * b * c <= MAX_BOX);
    }

    #[test]
    fn levels_stay_within_half_the_reach() {
        let l = flat([200, 200, 1]);
        let t = Template::new(&l, &Matrix3::identity(), 64.0).unwrap();
        assert_eq!(t.reach(), [32, 32, 0]);
        assert_eq!(default_levels(&l, &t), 4);
        let small = Template::new(&l, &Matrix3::identity(), 3.0).unwrap();
        assert_eq!(default_levels(&l, &small), 0);
    }
}
