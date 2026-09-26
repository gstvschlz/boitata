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

use crate::error::Result;
use crate::{is_inside_winding, require_closed, signed_solid_angle};
use ceres_core::Mesh;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

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
}

impl SolidTester {
    /// Validates the mesh and precomputes the query structures.
    pub fn new(mesh: &Mesh) -> Result<Self> {
        require_closed(mesh)?;
        let triangles: Vec<[Vector3<f64>; 3]> = (0..mesh.triangles().len())
            .map(|t| mesh.corners(t).map(Vector3::from))
            .collect();
        let triangle_bounds = triangles
            .iter()
            .map(|t| {
                Aabb::of_points(t.iter().map(|v| [v.x, v.y, v.z]))
                    .expect("a triangle always has three vertices")
            })
            .collect();
        let bounds =
            Aabb::of_points(mesh.vertices().iter().copied()).expect("a closed mesh has vertices");

        Ok(SolidTester {
            triangles,
            triangle_bounds,
            bounds,
        })
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
        let p = Vector3::new(point[0], point[1], point[2]);
        let winding: f64 = self
            .triangles
            .iter()
            .map(|[v0, v1, v2]| signed_solid_angle(&(v0 - p), &(v1 - p), &(v2 - p)))
            .sum();
        is_inside_winding(winding)
    }

    /// True when at least one triangle's bounds reach into the box — i.e. the
    /// surface may pass through it. Conservative: a false positive only costs
    /// the slower sampling path, never a wrong answer.
    pub fn surface_may_cut(&self, box_bounds: &Aabb) -> bool {
        self.triangle_bounds.iter().any(|t| t.overlaps(box_bounds))
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
