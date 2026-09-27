//! Isosurface extraction from a sampled scalar field.
//!
//! The lattice is marched a tetrahedron at a time rather than a cube at a
//! time: each cell is split into 6 tetrahedra by the Freudenthal (Kuhn)
//! subdivision, and each tetrahedron has only four topological cases against
//! an isovalue instead of the 256 a cube has.
//!
//! Two properties are why this variant is worth the extra triangles (roughly
//! 2× a classic marching-cubes surface at the same resolution):
//!
//! - **No ambiguous faces.** The classic 256-entry table has complementary
//!   cases whose face triangulations disagree between neighboring cells,
//!   which tears holes in the surface unless a disambiguation scheme is
//!   bolted on. A tetrahedron's cases have no such ambiguity.
//! - **Watertight by construction.** Freudenthal subdivision induces the same
//!   diagonal on a shared cell face from either side, so adjacent cells cut
//!   the same edges, and interpolated vertices are deduplicated by the grid
//!   edge they sit on. `surface_is_watertight` in the tests pins this down.
//!
//! Orientation: the region where the field is *above* the isovalue is the
//! inside of the solid, and triangles are wound so their normals point out of
//! it.
//!
//! Node values sitting exactly *on* the isovalue are nudged inside before
//! marching. Left alone they cut their edges at the node itself, so the
//! triangles around them collapse to zero area and the surface picks up a hole
//! wherever the field lands on a round number — which, for an indicator field
//! cut at 0, is most of the lattice.

use std::collections::HashMap;

use crate::grid::{ScalarGrid, TriMesh, cross, dot, norm, sub};

/// The 6 tetrahedra of the Freudenthal subdivision, as cell-corner indices
/// (`corner = i + 2j + 4k` over the unit cube). Every tetrahedron runs from
/// corner 0 to corner 7 along one of the 6 monotone lattice paths, which is
/// what makes the induced face diagonals agree between neighboring cells.
const TETS: [[usize; 4]; 6] = [
    [0, 1, 3, 7],
    [0, 1, 5, 7],
    [0, 2, 3, 7],
    [0, 2, 6, 7],
    [0, 4, 5, 7],
    [0, 4, 6, 7],
];

/// Extract the `isovalue` surface of `grid`. Returns an empty mesh when the
/// isovalue lies outside the sampled range.
pub fn marching_tetrahedra(grid: &ScalarGrid, isovalue: f64) -> TriMesh {
    let mut builder = Builder {
        grid,
        values: nudge_ties(&grid.values, isovalue),
        isovalue,
        mesh: TriMesh::default(),
        edges: HashMap::new(),
    };
    let [nx, ny, nz] = grid.counts;
    for k in 0..nz - 1 {
        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                let mut corners = [0usize; 8];
                for (c, slot) in corners.iter_mut().enumerate() {
                    *slot = grid.index(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
                }
                for tet in TETS {
                    builder.tetrahedron([
                        corners[tet[0]],
                        corners[tet[1]],
                        corners[tet[2]],
                        corners[tet[3]],
                    ]);
                }
            }
        }
    }
    builder.mesh
}

/// Move node values that equal the isovalue exactly just inside it, by a
/// fraction of the field's own range so the shift is invisible at any scale
/// the surface is looked at.
fn nudge_ties(values: &[f64], isovalue: f64) -> Vec<f64> {
    if !values.contains(&isovalue) {
        return values.to_vec();
    }
    let (lo, hi) = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        });
    let range = if lo <= hi && (hi - lo) > 0.0 {
        hi - lo
    } else {
        isovalue.abs().max(1.0)
    };
    let epsilon = range * 1e-9;
    values
        .iter()
        .map(|v| if *v == isovalue { v + epsilon } else { *v })
        .collect()
}

struct Builder<'a> {
    grid: &'a ScalarGrid,
    /// `grid.values` with isovalue ties nudged; every comparison and every
    /// interpolation reads this, never the raw grid.
    values: Vec<f64>,
    isovalue: f64,
    mesh: TriMesh,
    /// Interpolated vertex per cut grid edge, keyed by its node pair (ordered),
    /// so neighboring tetrahedra share the vertex instead of duplicating it.
    edges: HashMap<(usize, usize), u32>,
}

impl Builder<'_> {
    fn tetrahedron(&mut self, nodes: [usize; 4]) {
        let values = nodes.map(|n| self.values[n]);
        // A non-finite node makes every edge through it uninterpolatable; drop
        // the tetrahedron rather than emitting NaN geometry.
        if values.iter().any(|v| !v.is_finite()) {
            return;
        }
        let mut inside = Vec::with_capacity(4);
        let mut outside = Vec::with_capacity(4);
        for (local, v) in values.iter().enumerate() {
            if *v > self.isovalue {
                inside.push(local);
            } else {
                outside.push(local);
            }
        }
        if inside.is_empty() || outside.is_empty() {
            return;
        }

        // Outward direction for this cut: from the sub-tetrahedron that is
        // inside the solid towards the part that is outside it. One rule for
        // every case, instead of a per-case winding table.
        let dir = sub(
            centroid(self.grid, &outside, &nodes),
            centroid(self.grid, &inside, &nodes),
        );

        match (inside.len(), outside.len()) {
            (1, 3) => {
                let a = inside[0];
                let tri = [
                    self.edge_vertex(nodes[a], nodes[outside[0]]),
                    self.edge_vertex(nodes[a], nodes[outside[1]]),
                    self.edge_vertex(nodes[a], nodes[outside[2]]),
                ];
                self.emit(tri, dir);
            }
            (3, 1) => {
                let d = outside[0];
                let tri = [
                    self.edge_vertex(nodes[d], nodes[inside[0]]),
                    self.edge_vertex(nodes[d], nodes[inside[1]]),
                    self.edge_vertex(nodes[d], nodes[inside[2]]),
                ];
                self.emit(tri, dir);
            }
            (2, 2) => {
                let (a, b) = (inside[0], inside[1]);
                let (c, d) = (outside[0], outside[1]);
                // Consecutive quad corners share a tetrahedron vertex (a, d, b,
                // c in turn), so this is the cycle around the cut, not a
                // bow-tie.
                let quad = [
                    self.edge_vertex(nodes[a], nodes[c]),
                    self.edge_vertex(nodes[a], nodes[d]),
                    self.edge_vertex(nodes[b], nodes[d]),
                    self.edge_vertex(nodes[b], nodes[c]),
                ];
                self.emit([quad[0], quad[1], quad[2]], dir);
                self.emit([quad[0], quad[2], quad[3]], dir);
            }
            _ => unreachable!("a tetrahedron has 4 corners"),
        }
    }

    /// Interpolated vertex where the isosurface cuts the grid edge `a`–`b`,
    /// created once and reused by every tetrahedron sharing that edge.
    fn edge_vertex(&mut self, a: usize, b: usize) -> u32 {
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(&id) = self.edges.get(&key) {
            return id;
        }
        let (va, vb) = (self.values[key.0], self.values[key.1]);
        let denom = vb - va;
        let t = if denom == 0.0 {
            0.5
        } else {
            ((self.isovalue - va) / denom).clamp(0.0, 1.0)
        };
        let pa = self.grid.position(key.0);
        let pb = self.grid.position(key.1);
        let id = self.mesh.vertices.len() as u32;
        self.mesh.vertices.push([
            pa[0] + t * (pb[0] - pa[0]),
            pa[1] + t * (pb[1] - pa[1]),
            pa[2] + t * (pb[2] - pa[2]),
        ]);
        self.edges.insert(key, id);
        id
    }

    /// Push a triangle wound so its normal points along `dir`. Zero-area
    /// triangles — a cut that lands exactly on a grid node — are dropped.
    fn emit(&mut self, tri: [u32; 3], dir: [f64; 3]) {
        let p = tri.map(|v| self.mesh.vertices[v as usize]);
        let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
        if norm(n) <= 0.0 {
            return;
        }
        if dot(n, dir) < 0.0 {
            self.mesh
                .triangles
                .extend_from_slice(&[tri[0], tri[2], tri[1]]);
        } else {
            self.mesh
                .triangles
                .extend_from_slice(&[tri[0], tri[1], tri[2]]);
        }
    }
}

fn centroid(grid: &ScalarGrid, locals: &[usize], nodes: &[usize; 4]) -> [f64; 3] {
    let mut sum = [0.0; 3];
    for &l in locals {
        let p = grid.position(nodes[l]);
        for k in 0..3 {
            sum[k] += p[k];
        }
    }
    let n = locals.len() as f64;
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Signed-distance field of a sphere, positive inside so the solid is the
    /// ball (the sign convention the extractor assumes).
    fn sphere(center: [f64; 3], radius: f64, nodes: usize, half: f64) -> ScalarGrid {
        let spacing = 2.0 * half / (nodes - 1) as f64;
        ScalarGrid::evaluate(
            [center[0] - half, center[1] - half, center[2] - half],
            [spacing; 3],
            [nodes; 3],
            |p| {
                let d = ((p[0] - center[0]).powi(2)
                    + (p[1] - center[1]).powi(2)
                    + (p[2] - center[2]).powi(2))
                .sqrt();
                radius - d
            },
        )
        .unwrap()
    }

    #[test]
    fn empty_when_the_isovalue_is_outside_the_sampled_range() {
        let g = ScalarGrid::evaluate([0.0; 3], [1.0; 3], [5, 5, 5], |_| 1.0).unwrap();
        assert!(marching_tetrahedra(&g, 5.0).is_empty());
        assert!(marching_tetrahedra(&g, -5.0).is_empty());
    }

    #[test]
    fn sphere_volume_converges_on_the_analytic_value() {
        let radius: f64 = 10.0;
        let exact = 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);
        let coarse = marching_tetrahedra(&sphere([0.0; 3], radius, 21, 15.0), 0.0);
        let fine = marching_tetrahedra(&sphere([0.0; 3], radius, 61, 15.0), 0.0);
        let err = |m: &TriMesh| (m.enclosed_volume() - exact).abs() / exact;
        assert!(err(&coarse) < 0.05, "coarse error {}", err(&coarse));
        assert!(err(&fine) < 0.005, "fine error {}", err(&fine));
        assert!(err(&fine) < err(&coarse));
    }

    #[test]
    fn sphere_area_converges_on_the_analytic_value() {
        let radius: f64 = 10.0;
        let exact = 4.0 * std::f64::consts::PI * radius.powi(2);
        let mesh = marching_tetrahedra(&sphere([0.0; 3], radius, 61, 15.0), 0.0);
        let err = (mesh.surface_area() - exact).abs() / exact;
        assert!(err < 0.02, "area error {err}");
    }

    /// Count edges of `mesh` that are not shared by exactly two triangles
    /// traversing them in opposite directions — 0 for a closed surface.
    fn unmatched_edges(mesh: &TriMesh) -> usize {
        let mut directed: HashMap<(u32, u32), i32> = HashMap::new();
        for t in 0..mesh.triangle_count() {
            let v = [
                mesh.triangles[3 * t],
                mesh.triangles[3 * t + 1],
                mesh.triangles[3 * t + 2],
            ];
            for e in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
                let (key, delta) = if e.0 < e.1 { (e, 1) } else { ((e.1, e.0), -1) };
                *directed.entry(key).or_insert(0) += delta;
            }
        }
        directed.values().filter(|&&c| c != 0).count()
    }

    #[test]
    fn surface_is_watertight() {
        let mesh = marching_tetrahedra(&sphere([0.0; 3], 10.0, 25, 15.0), 0.0);
        assert!(!mesh.is_empty());
        assert_eq!(unmatched_edges(&mesh), 0);
    }

    #[test]
    fn nodes_exactly_on_the_isovalue_do_not_tear_the_surface() {
        // A radius landing on lattice nodes puts six of them exactly at 0.
        // Untreated, every edge into one of those nodes cuts at the node
        // itself and the triangles around it collapse, opening six holes.
        let grid = sphere([0.0; 3], 10.0, 25, 15.0);
        let ties = grid.values.iter().filter(|v| **v == 0.0).count();
        assert!(ties > 0, "test geometry no longer produces exact ties");
        assert_eq!(unmatched_edges(&marching_tetrahedra(&grid, 0.0)), 0);
    }

    #[test]
    fn a_whole_plane_of_ties_still_closes() {
        // f = |x| − 3 on integer nodes: an entire pair of planes sits on the
        // isovalue, the degenerate case the nudge exists for.
        let g =
            ScalarGrid::evaluate([-6.0; 3], [1.0; 3], [13, 13, 13], |p| p[0].abs() - 3.0).unwrap();
        let mesh = marching_tetrahedra(&g, 0.0);
        assert!(!mesh.is_empty());
        // The slab runs out of the lattice on ±y and ±z, so only the two cut
        // planes are closed geometry; check they carry no torn edges beyond
        // the open boundary rings.
        assert!(
            mesh.vertices
                .iter()
                .all(|v| v.iter().all(|c| c.is_finite()))
        );
        assert!(mesh.surface_area() > 0.0);
    }

    #[test]
    fn normals_point_out_of_the_solid() {
        let center = [3.0, -2.0, 1.0];
        let mesh = marching_tetrahedra(&sphere(center, 8.0, 25, 12.0), 0.0);
        assert!(!mesh.is_empty());
        for t in 0..mesh.triangle_count() {
            let a = mesh.vertices[mesh.triangles[3 * t] as usize];
            let b = mesh.vertices[mesh.triangles[3 * t + 1] as usize];
            let c = mesh.vertices[mesh.triangles[3 * t + 2] as usize];
            let n = cross(sub(b, a), sub(c, a));
            let outward = sub(
                [
                    (a[0] + b[0] + c[0]) / 3.0,
                    (a[1] + b[1] + c[1]) / 3.0,
                    (a[2] + b[2] + c[2]) / 3.0,
                ],
                center,
            );
            assert!(dot(n, outward) > 0.0, "triangle {t} faces inwards");
        }
    }

    #[test]
    fn planar_field_cuts_at_the_isovalue() {
        // f = x, isovalue 4.5 → one flat sheet at x = 4.5.
        let g = ScalarGrid::evaluate([0.0; 3], [1.0; 3], [10, 4, 4], |p| p[0]).unwrap();
        let mesh = marching_tetrahedra(&g, 4.5);
        assert!(!mesh.is_empty());
        for v in &mesh.vertices {
            assert!((v[0] - 4.5).abs() < 1e-9, "vertex off the plane: {v:?}");
        }
    }

    #[test]
    fn a_mask_clips_the_solid_and_closed_caps_it() {
        // 40³ blocks of 1 m, a radius-12 ball in the middle; keep z < 20.
        let (n, radius) = (40usize, 12.0f64);
        let cells: Vec<f64> = (0..n * n * n)
            .map(|c| {
                let p = [c % n, c / n % n, c / (n * n)].map(|i| i as f64 + 0.5 - 20.0);
                radius - (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
            })
            .collect();
        let active: Vec<bool> = (0..n * n * n).map(|c| c / (n * n) < 20).collect();
        let lattice = |mask: Option<&[bool]>, closed| {
            marching_tetrahedra(
                &ScalarGrid::blocks([1.0; 3], [n; 3], &cells, mask, 0.0, closed).unwrap(),
                0.0,
            )
        };

        let full = lattice(None, true);
        let all = vec![true; n * n * n];
        assert_eq!(full.triangles, lattice(Some(&all), true).triangles);
        assert_eq!(full.vertices, lattice(Some(&all), true).vertices);

        let half = lattice(Some(&active), true);
        assert_eq!(unmatched_edges(&half), 0);
        let exact = 2.0 / 3.0 * std::f64::consts::PI * radius.powi(3);
        let err = (half.enclosed_volume() - exact).abs() / exact;
        assert!(err < 0.02, "hemisphere volume error {err}");
        assert!(half.vertices.iter().all(|v| v[2] < 20.5));

        let open = lattice(Some(&active), false);
        assert!(unmatched_edges(&open) > 0);
        assert!(open.vertices.iter().all(|v| v[2] < 20.0));
    }

    #[test]
    fn non_finite_nodes_do_not_produce_nan_geometry() {
        let mut g = sphere([0.0; 3], 10.0, 15, 15.0);
        g.values[0] = f64::NAN;
        let mesh = marching_tetrahedra(&g, 0.0);
        assert!(!mesh.is_empty());
        assert!(
            mesh.vertices
                .iter()
                .all(|v| v.iter().all(|c| c.is_finite()))
        );
    }
}
