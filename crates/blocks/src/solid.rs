//! Block-and-sample geometry against a closed triangle solid: the layer between
//! an imported wireframe and the data you want to flag with it (#259).
//!
//! [`is_inside`](crate::is_inside) answers one point at a time and
//! checks closure on every call. Flagging a block model asks the same
//! question millions of times, so [`SolidTester`] pays the setup once —
//! validation, vertex lookup, per-triangle bounds — and adds the two rejections
//! that make the volume tractable:
//!
//! 1. A block outside the solid's bounding box is outside, for free.
//! 2. A block no triangle passes through is *uniformly* inside or outside, so
//!    one centroid test settles it exactly. Only blocks the surface actually
//!    cuts pay for sub-cell sampling.
//!
//! The second rejection is why partial-block proportions stay affordable: the
//! blocks that need the expensive path are the ones on the solid's skin, which
//! grow with its surface area rather than its volume.
//!
//! Both queries look up triangles binned in plan. A point counts the signed
//! crossings of a ray up z through the few triangles under it, which is the
//! winding number of a closed mesh; within a tolerance of an edge or crossing,
//! or on a mesh whose edges do not cancel, it takes the full winding sum, so
//! answers never change.

use crate::error::Result;
use crate::{is_inside_winding, require_closed, signed_solid_angle};
use ceres_core::Mesh;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::RangeInclusive;

/// Axis-aligned bounds, `[min, max]` per axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Aabb {
    pub(crate) fn of_points(points: impl Iterator<Item = [f64; 3]>) -> Option<Self> {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        let mut any = false;
        for p in points {
            any = true;
            for axis in 0..3 {
                min[axis] = min[axis].min(p[axis]);
                max[axis] = max[axis].max(p[axis]);
            }
        }
        any.then_some(Aabb { min, max })
    }

    /// True when the two boxes share any volume (touching faces count).
    pub fn overlaps(&self, other: &Aabb) -> bool {
        (0..3).all(|axis| self.min[axis] <= other.max[axis] && other.min[axis] <= self.max[axis])
    }
}

/// How a partially-filled block is assigned to the solid's domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BlockDomainRule {
    /// Inside when the block's center point is inside. The cheapest rule and
    /// the one most packages default to; ignores how much of the block is in.
    Centroid,
    /// Inside when more than half the block's volume is inside.
    Majority,
    /// Inside when any part of the block is inside. Over-reports volume, but
    /// never drops a block that clips the solid.
    Any,
}

impl BlockDomainRule {
    /// Whether this rule counts the block as belonging to the solid.
    pub fn selects(&self, block: &BlockSolid) -> bool {
        match self {
            BlockDomainRule::Centroid => block.centroid_inside,
            BlockDomainRule::Majority => block.proportion > 0.5,
            BlockDomainRule::Any => block.proportion > 0.0,
        }
    }
}

/// One block measured against the solid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockSolid {
    /// Fraction of the block's volume inside the solid, in `[0, 1]`.
    pub proportion: f64,
    /// Whether the block's center point itself is inside.
    pub centroid_inside: bool,
}

/// A closed triangle mesh prepared for repeated inside/outside queries.
#[derive(Clone)]
pub struct SolidTester {
    /// Triangle vertices, resolved once so queries don't chase indices.
    triangles: Vec<[Vector3<f64>; 3]>,
    /// Per-triangle bounds, for the "does the surface cut this block" test.
    triangle_bounds: Vec<Aabb>,
    bounds: Aabb,
    /// Triangles binned in plan; `None` when the mesh's edges do not cancel,
    /// so only the full winding sum is exact.
    bins: Option<Bins>,
}

/// Plan grid over the solid's bounds, each cell listing the triangles whose
/// bounds, grown by `tol`, reach it.
#[derive(Clone)]
struct Bins {
    lo: [f64; 2],
    step: [f64; 2],
    side: usize,
    cells: Vec<Vec<u32>>,
    /// Distance under which a point counts as on the surface and takes the
    /// winding sum instead of the ray.
    tol: f64,
    scale: f64,
}

impl Bins {
    fn new(triangle_bounds: &[Aabb], bounds: &Aabb) -> Self {
        let scale = bounds
            .min
            .iter()
            .chain(&bounds.max)
            .fold(1.0f64, |s, v| s.max(v.abs()));
        let side = ((triangle_bounds.len() as f64).sqrt().ceil() as usize).clamp(1, 1024);
        let lo = [bounds.min[0], bounds.min[1]];
        let step = [0, 1].map(|a| (bounds.max[a] - bounds.min[a]) / side as f64);
        let mut bins = Bins {
            lo,
            step,
            side,
            cells: vec![vec![]; side * side],
            tol: 1e-10 * scale,
            scale,
        };
        for (t, b) in triangle_bounds.iter().enumerate() {
            let (x, y) = bins.span(b.min, b.max, bins.tol);
            for j in y {
                for i in x.clone() {
                    bins.cells[j * side + i].push(t as u32);
                }
            }
        }
        bins
    }

    fn cell(&self, v: f64, axis: usize) -> usize {
        if self.step[axis] > 0.0 {
            (((v - self.lo[axis]) / self.step[axis]) as usize).min(self.side - 1)
        } else {
            0
        }
    }

    fn span(
        &self,
        min: [f64; 3],
        max: [f64; 3],
        grow: f64,
    ) -> (RangeInclusive<usize>, RangeInclusive<usize>) {
        let range = |a| self.cell(min[a] - grow, a)..=self.cell(max[a] + grow, a);
        (range(0), range(1))
    }

    /// Signed count of the triangles a ray from `p` up world z crosses, the
    /// winding number of a closed mesh; `None` when `p` lies within `tol` of
    /// an edge the ray passes or of a crossing.
    fn crossings(
        &self,
        triangles: &[[Vector3<f64>; 3]],
        bounds: &[Aabb],
        p: [f64; 3],
    ) -> Option<i64> {
        let tol = self.tol;
        let mut winding = 0;
        for &t in &self.cells[self.cell(p[1], 1) * self.side + self.cell(p[0], 0)] {
            let b = &bounds[t as usize];
            if (0..2).any(|a| p[a] < b.min[a] - tol || p[a] > b.max[a] + tol) {
                continue;
            }
            let v = &triangles[t as usize];
            let area =
                (v[1].x - v[0].x) * (v[2].y - v[0].y) - (v[1].y - v[0].y) * (v[2].x - v[0].x);
            let sign = if area < 0.0 { -1.0 } else { 1.0 };
            // Edge i runs between the other two corners; its edge function
            // weighs corner i.
            let edges = [0, 1, 2].map(|i| {
                let (a, c) = (&v[(i + 1) % 3], &v[(i + 2) % 3]);
                let (dx, dy) = (c.x - a.x, c.y - a.y);
                let e = dx * (p[1] - a.y) - dy * (p[0] - a.x);
                let length = dx.hypot(dy);
                let distance = if length > 0.0 { sign * e / length } else { 0.0 };
                (e, length, distance)
            });
            if edges.iter().any(|&(_, _, d)| d < -tol) {
                continue;
            }
            if edges.iter().any(|&(_, _, d)| d <= tol) {
                return None;
            }
            let weight = edges.map(|(e, _, _)| e);
            let perimeter: f64 = edges.iter().map(|&(_, l, _)| l).sum();
            let total: f64 = weight.iter().sum();
            let dz: [f64; 3] = [0, 1, 2].map(|i| v[i].z - p[2]);
            let height = (0..3).map(|i| weight[i] * dz[i]).sum::<f64>() / total;
            let error = 64.0
                * f64::EPSILON
                * self.scale
                * perimeter
                * dz.iter().map(|d| d.abs()).sum::<f64>()
                / total.abs();
            if height.abs() <= tol + error {
                return None;
            }
            if height > 0.0 {
                winding += sign as i64;
            }
        }
        Some(winding)
    }
}

/// Whether every edge is walked as often one way as the other, so the
/// winding number is an integer off the surface.
fn edges_cancel(mesh: &Mesh) -> bool {
    let mut edges: HashMap<(u32, u32), i64> = HashMap::new();
    for &[a, b, c] in mesh.triangles() {
        for (i, j) in [(a, b), (b, c), (c, a)] {
            if i != j {
                *edges.entry((i.min(j), i.max(j))).or_insert(0) += if i < j { 1 } else { -1 };
            }
        }
    }
    edges.values().all(|&n| n == 0)
}

impl SolidTester {
    /// Validates the mesh and precomputes the query structures.
    pub fn new(mesh: &Mesh) -> Result<Self> {
        require_closed(mesh)?;
        let triangles: Vec<[Vector3<f64>; 3]> = (0..mesh.triangles().len())
            .map(|t| mesh.corners(t).map(Vector3::from))
            .collect();
        let triangle_bounds: Vec<Aabb> = triangles
            .iter()
            .map(|t| {
                Aabb::of_points(t.iter().map(|v| [v.x, v.y, v.z]))
                    .expect("a triangle always has three vertices")
            })
            .collect();
        let bounds =
            Aabb::of_points(mesh.vertices().iter().copied()).expect("a closed mesh has vertices");
        let bins = edges_cancel(mesh).then(|| Bins::new(&triangle_bounds, &bounds));
        Ok(SolidTester {
            triangles,
            triangle_bounds,
            bounds,
            bins,
        })
    }

    /// The same tester answering every query with the full winding sum.
    #[cfg(test)]
    pub(crate) fn unbinned(mut self) -> Self {
        self.bins = None;
        self
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// The solid's axis-aligned bounds.
    pub fn bounds(&self) -> Aabb {
        self.bounds
    }

    /// Generalized winding number at a point: ~±1 inside a closed solid, ~0
    /// outside, fractional where the mesh has holes.
    pub fn winding_number(&self, point: [f64; 3]) -> f64 {
        let p = Vector3::new(point[0], point[1], point[2]);
        let total: f64 = self
            .triangles
            .iter()
            .map(|[v0, v1, v2]| signed_solid_angle(&(v0 - p), &(v1 - p), &(v2 - p)))
            .sum();
        total / (4.0 * std::f64::consts::PI)
    }

    /// Whether a point is inside the solid.
    pub fn contains(&self, point: [f64; 3]) -> bool {
        if !self.bounds.overlaps(&Aabb {
            min: point,
            max: point,
        }) {
            return false;
        }
        let ray = self
            .bins
            .as_ref()
            .and_then(|b| b.crossings(&self.triangles, &self.triangle_bounds, point));
        match ray {
            Some(winding) => winding != 0,
            None => {
                let p = Vector3::new(point[0], point[1], point[2]);
                let winding: f64 = self
                    .triangles
                    .iter()
                    .map(|[v0, v1, v2]| signed_solid_angle(&(v0 - p), &(v1 - p), &(v2 - p)))
                    .sum();
                is_inside_winding(winding)
            }
        }
    }

    /// True when at least one triangle's bounds reach into the box — i.e. the
    /// surface may pass through it. Conservative: a false positive only costs
    /// the slower sampling path, never a wrong answer.
    pub fn surface_may_cut(&self, box_bounds: &Aabb) -> bool {
        let Some(bins) = &self.bins else {
            return self.triangle_bounds.iter().any(|t| t.overlaps(box_bounds));
        };
        if !self.bounds.overlaps(box_bounds) {
            return false;
        }
        let (x, y) = bins.span(box_bounds.min, box_bounds.max, 0.0);
        y.flat_map(|j| x.clone().map(move |i| j * bins.side + i))
            .flat_map(|c| &bins.cells[c])
            .any(|&t| self.triangle_bounds[t as usize].overlaps(box_bounds))
    }

    /// Measures one axis-aligned block against the solid.
    ///
    /// `discretization` is the sub-cell count along each axis used where the
    /// surface cuts the block, with sample points at sub-cell centers, so the
    /// proportion is a midpoint-rule estimate converging as O(1/d). Blocks the
    /// surface misses entirely are resolved exactly, without sampling.
    pub fn evaluate_block(
        &self,
        center: [f64; 3],
        size: [f64; 3],
        discretization: [usize; 3],
    ) -> BlockSolid {
        let half = [
            size[0].abs() / 2.0,
            size[1].abs() / 2.0,
            size[2].abs() / 2.0,
        ];
        let block_bounds = Aabb {
            min: [
                center[0] - half[0],
                center[1] - half[1],
                center[2] - half[2],
            ],
            max: [
                center[0] + half[0],
                center[1] + half[1],
                center[2] + half[2],
            ],
        };

        if !self.bounds.overlaps(&block_bounds) {
            return BlockSolid {
                proportion: 0.0,
                centroid_inside: false,
            };
        }

        // No triangle reaches into the block, so it lies wholly inside or
        // wholly outside and the centroid speaks for all of it.
        if !self.surface_may_cut(&block_bounds) {
            let inside = self.contains(center);
            return BlockSolid {
                proportion: if inside { 1.0 } else { 0.0 },
                centroid_inside: inside,
            };
        }

        let d = discretization.map(|n| n.max(1));
        let step = [0, 1, 2].map(|a| size[a] / d[a] as f64);
        let mut inside_count = 0usize;
        for i in 0..d[0] {
            for j in 0..d[1] {
                for k in 0..d[2] {
                    let sample = [
                        block_bounds.min[0] + (i as f64 + 0.5) * step[0],
                        block_bounds.min[1] + (j as f64 + 0.5) * step[1],
                        block_bounds.min[2] + (k as f64 + 0.5) * step[2],
                    ];
                    if self.contains(sample) {
                        inside_count += 1;
                    }
                }
            }
        }

        BlockSolid {
            proportion: inside_count as f64 / d.iter().product::<usize>() as f64,
            centroid_inside: self.contains(center),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Axis-aligned cube spanning `[lo, hi]` on every axis, triangles wound
    /// counter-clockwise seen from outside.
    pub(crate) fn cube(lo: f64, hi: f64) -> Mesh {
        Mesh::new(
            vec![
                [lo, lo, lo],
                [hi, lo, lo],
                [hi, hi, lo],
                [lo, hi, lo],
                [lo, lo, hi],
                [hi, lo, hi],
                [hi, hi, hi],
                [lo, hi, hi],
            ],
            CUBE.to_vec(),
        )
        .unwrap()
    }

    const CUBE: [[u32; 3]; 12] = [
        [0, 2, 1],
        [0, 3, 2], // bottom, -Z
        [4, 5, 6],
        [4, 6, 7], // top, +Z
        [0, 1, 5],
        [0, 5, 4], // front, -Y
        [2, 3, 7],
        [2, 7, 6], // back, +Y
        [0, 4, 7],
        [0, 7, 3], // left, -X
        [1, 2, 6],
        [1, 6, 5], // right, +X
    ];

    /// Same cube with every triangle reversed, so it winds inward.
    fn inverted_cube(lo: f64, hi: f64) -> Mesh {
        let cube = cube(lo, hi);
        Mesh::new(
            cube.vertices().to_vec(),
            CUBE.iter().map(|&[a, b, c]| [c, b, a]).collect(),
        )
        .unwrap()
    }

    fn merged(meshes: &[Mesh]) -> Mesh {
        let (mut vertices, mut triangles) = (vec![], vec![]);
        for m in meshes {
            let offset = vertices.len() as u32;
            vertices.extend_from_slice(m.vertices());
            triangles.extend(m.triangles().iter().map(|t| t.map(|v| v + offset)));
        }
        Mesh::new(vertices, triangles).unwrap()
    }

    fn shifted(mesh: &Mesh, by: [f64; 3]) -> Mesh {
        let vertices = mesh
            .vertices()
            .iter()
            .map(|v| [0, 1, 2].map(|a| v[a] + by[a]))
            .collect();
        Mesh::new(vertices, mesh.triangles().to_vec()).unwrap()
    }

    /// The binned ray answers exactly as the winding sum, on lattice points
    /// that sit on faces, edges and corners, inside nested shells, on inward
    /// wound meshes and far from the origin.
    #[test]
    fn binned_queries_match_the_winding_sum() {
        let utm = [500_000.0, 7_000_000.0, 1_000.0];
        let meshes = [
            cube(0.0, 10.0),
            inverted_cube(0.0, 10.0),
            merged(&[cube(0.0, 10.0), cube(2.0, 8.0)]),
            crate::subblock::tests::sphere(6.0),
            shifted(&crate::subblock::tests::sphere(6.0), utm),
            shifted(&cube(0.0, 10.0), utm),
        ];
        for mesh in &meshes {
            let fast = SolidTester::new(mesh).unwrap();
            assert!(fast.bins.is_some());
            let exact = fast.clone().unbinned();
            let o = fast.bounds().min;
            for i in -2..=30 {
                for j in -2..=30 {
                    for k in -2..=30 {
                        let p = [i, j, k].map(|v| v as f64 * 0.5);
                        let p = [0, 1, 2].map(|a| o[a].floor() + p[a]);
                        assert_eq!(fast.contains(p), exact.contains(p), "{p:?}");
                        let b = Aabb {
                            min: p,
                            max: p.map(|v| v + 0.6),
                        };
                        assert_eq!(fast.surface_may_cut(&b), exact.surface_may_cut(&b));
                    }
                }
            }
        }
    }

    #[test]
    fn contains_separates_inside_from_outside() {
        let solid = SolidTester::new(&cube(0.0, 10.0)).expect("valid mesh");

        assert!(solid.contains([5.0, 5.0, 5.0]), "center");
        assert!(solid.contains([0.5, 0.5, 0.5]), "inside near a corner");
        assert!(solid.contains([5.0, 5.0, 9.5]), "inside just under a face");

        assert!(!solid.contains([5.0, 5.0, 10.5]), "just outside a face");
        assert!(!solid.contains([-0.5, 5.0, 5.0]), "just outside -X");
        assert!(!solid.contains([5.0, 5.0, 50.0]), "far above");
        assert!(!solid.contains([100.0, 100.0, 100.0]), "far away");
    }

    /// Regression: summing *unsigned* solid angles makes the total approach 2π
    /// just outside any face, so points in a shell around the solid read as
    /// inside. Walk out from a face and check the flip happens at the face.
    #[test]
    fn no_false_inside_halo_around_the_surface() {
        let solid = SolidTester::new(&cube(0.0, 10.0)).expect("valid mesh");
        for step in 1..=20 {
            let z = 10.0 + step as f64 * 0.25;
            assert!(
                !solid.contains([5.0, 5.0, z]),
                "point {z} above the top face must be outside"
            );
        }
    }

    #[test]
    fn winding_number_is_one_inside_and_zero_outside() {
        let solid = SolidTester::new(&cube(0.0, 10.0)).expect("valid mesh");
        assert!((solid.winding_number([5.0, 5.0, 5.0]).abs() - 1.0).abs() < 1e-9);
        assert!(solid.winding_number([5.0, 5.0, 25.0]).abs() < 1e-9);
    }

    /// Imported solids are often wound inward; that mesh describes the same
    /// volume and must classify the same way.
    #[test]
    fn inward_wound_mesh_classifies_the_same() {
        let solid = SolidTester::new(&inverted_cube(0.0, 10.0)).expect("valid mesh");
        assert!(solid.contains([5.0, 5.0, 5.0]));
        assert!(!solid.contains([5.0, 5.0, 10.5]));
    }

    #[test]
    fn block_fully_inside_is_exactly_one() {
        let solid = SolidTester::new(&cube(0.0, 100.0)).expect("valid mesh");
        let block = solid.evaluate_block([50.0, 50.0, 50.0], [10.0, 10.0, 10.0], [4; 3]);
        assert_eq!(block.proportion, 1.0);
        assert!(block.centroid_inside);
    }

    #[test]
    fn block_clear_of_the_solid_is_exactly_zero() {
        let solid = SolidTester::new(&cube(0.0, 100.0)).expect("valid mesh");
        let block = solid.evaluate_block([500.0, 500.0, 500.0], [10.0, 10.0, 10.0], [4; 3]);
        assert_eq!(block.proportion, 0.0);
        assert!(!block.centroid_inside);
    }

    /// Cube-in-cube: a block straddling a face of an axis-aligned solid has an
    /// exact answer, and midpoint sampling on an aligned split hits it.
    #[test]
    fn block_straddling_a_face_is_half() {
        let solid = SolidTester::new(&cube(0.0, 100.0)).expect("valid mesh");
        // Block centered on the z = 100 face: exactly half its volume is inside.
        let block = solid.evaluate_block([50.0, 50.0, 100.0], [10.0, 10.0, 10.0], [4; 3]);
        assert!(
            (block.proportion - 0.5).abs() < 1e-9,
            "expected 0.5, got {}",
            block.proportion
        );
    }

    #[test]
    fn block_over_a_corner_is_an_eighth() {
        let solid = SolidTester::new(&cube(0.0, 100.0)).expect("valid mesh");
        // Centered on the (100, 100, 100) corner: one octant of the block is in.
        let block = solid.evaluate_block([100.0, 100.0, 100.0], [10.0, 10.0, 10.0], [4; 3]);
        assert!(
            (block.proportion - 0.125).abs() < 1e-9,
            "expected 0.125, got {}",
            block.proportion
        );
    }

    /// A quarter-covered block: the solid's face cuts the block at 25% along Z.
    #[test]
    fn partial_coverage_tracks_the_cut_position() {
        let solid = SolidTester::new(&cube(0.0, 100.0)).expect("valid mesh");
        // Block from z = 97.5 to 107.5 — 2.5 of its 10 units are inside.
        let block = solid.evaluate_block([50.0, 50.0, 102.5], [10.0, 10.0, 10.0], [8; 3]);
        assert!(
            (block.proportion - 0.25).abs() < 1e-9,
            "expected 0.25, got {}",
            block.proportion
        );
    }

    /// A solid smaller than the block it sits in still registers: the cheap
    /// "no triangle reaches this block" rejection must not fire here, and the
    /// centroid test is exact regardless of sampling resolution.
    #[test]
    fn solid_entirely_within_one_block_is_not_missed() {
        let solid = SolidTester::new(&cube(45.0, 55.0)).expect("valid mesh");
        let block = solid.evaluate_block([50.0, 50.0, 50.0], [100.0, 100.0, 100.0], [50; 3]);
        assert!(block.centroid_inside);
        // 10³ inside a 100³ block = 0.1%. Midpoint sampling lands near it once
        // the sub-cells (2 units here) are small enough to resolve the solid.
        assert!(
            (block.proportion - 0.001).abs() < 0.001,
            "expected ≈0.001, got {}",
            block.proportion
        );
    }

    /// The floor of the method, pinned deliberately: midpoint sampling cannot
    /// see a feature smaller than one sub-cell. Here 10³ sub-cells across a
    /// 100-unit block are exactly as wide as the solid, every sample misses it,
    /// and the proportion collapses to zero even though the centroid is inside.
    /// Callers that care must raise the discretization — the command layer
    /// warns when sub-cells are coarse against the solid (see geometry_ops).
    #[test]
    fn sub_cell_sampling_cannot_resolve_a_solid_finer_than_one_sub_cell() {
        let solid = SolidTester::new(&cube(45.0, 55.0)).expect("valid mesh");
        let coarse = solid.evaluate_block([50.0, 50.0, 50.0], [100.0, 100.0, 100.0], [10; 3]);
        assert_eq!(coarse.proportion, 0.0);
        assert!(coarse.centroid_inside, "the centroid test still sees it");

        let fine = solid.evaluate_block([50.0, 50.0, 50.0], [100.0, 100.0, 100.0], [50; 3]);
        assert!(fine.proportion > 0.0, "finer sampling recovers the solid");
    }

    #[test]
    fn domain_rules_disagree_on_a_partial_block() {
        let barely = BlockSolid {
            proportion: 0.2,
            centroid_inside: false,
        };
        assert!(!BlockDomainRule::Centroid.selects(&barely));
        assert!(!BlockDomainRule::Majority.selects(&barely));
        assert!(BlockDomainRule::Any.selects(&barely));

        let mostly = BlockSolid {
            proportion: 0.8,
            centroid_inside: true,
        };
        assert!(BlockDomainRule::Centroid.selects(&mostly));
        assert!(BlockDomainRule::Majority.selects(&mostly));
        assert!(BlockDomainRule::Any.selects(&mostly));

        let empty = BlockSolid {
            proportion: 0.0,
            centroid_inside: false,
        };
        assert!(!BlockDomainRule::Any.selects(&empty));
    }

    #[test]
    fn discretization_of_zero_is_treated_as_one() {
        let solid = SolidTester::new(&cube(0.0, 100.0)).expect("valid mesh");
        let block = solid.evaluate_block([50.0, 50.0, 100.0], [10.0, 10.0, 10.0], [0; 3]);
        // A single sample at the block center, which sits on the face.
        assert!(block.proportion == 0.0 || block.proportion == 1.0);
    }
}
