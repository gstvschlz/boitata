//! The sampling lattice an implicit field is evaluated on, and the triangle
//! mesh an isosurface comes back as.

use crate::error::{ModelError, Result};

/// A scalar field sampled on a regular lattice of *nodes* (not cells): a
/// `counts` of `[nx, ny, nz]` means `nx·ny·nz` samples and
/// `(nx−1)·(ny−1)·(nz−1)` cells. Values are stored x-fastest.
#[derive(Debug, Clone)]
pub struct ScalarGrid {
    pub origin: [f64; 3],
    pub spacing: [f64; 3],
    pub counts: [usize; 3],
    pub values: Vec<f64>,
}

impl ScalarGrid {
    pub fn new(
        origin: [f64; 3],
        spacing: [f64; 3],
        counts: [usize; 3],
        values: Vec<f64>,
    ) -> Result<Self> {
        if counts.iter().any(|&c| c < 2) {
            return Err(ModelError::InvalidParameter(
                "grid needs at least 2 nodes on every axis".into(),
            ));
        }
        if spacing.iter().any(|&s| !s.is_finite() || s <= 0.0) {
            return Err(ModelError::InvalidParameter(
                "grid spacing must be positive".into(),
            ));
        }
        let expected = counts[0] * counts[1] * counts[2];
        if values.len() != expected {
            return Err(ModelError::InvalidParameter(format!(
                "grid holds {} values, expected {expected}",
                values.len()
            )));
        }
        Ok(Self {
            origin,
            spacing,
            counts,
            values,
        })
    }

    pub fn node_count(&self) -> usize {
        self.counts[0] * self.counts[1] * self.counts[2]
    }

    pub fn cell_count(&self) -> usize {
        (self.counts[0] - 1) * (self.counts[1] - 1) * (self.counts[2] - 1)
    }

    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.counts[0] * (j + self.counts[1] * k)
    }

    /// World position of the node at a flat index.
    pub fn position(&self, index: usize) -> [f64; 3] {
        let nx = self.counts[0];
        let ny = self.counts[1];
        let k = index / (nx * ny);
        let rem = index % (nx * ny);
        let j = rem / nx;
        let i = rem % nx;
        [
            self.origin[0] + i as f64 * self.spacing[0],
            self.origin[1] + j as f64 * self.spacing[1],
            self.origin[2] + k as f64 * self.spacing[2],
        ]
    }

    /// Fill the lattice by evaluating `f` at every node.
    pub fn evaluate(
        origin: [f64; 3],
        spacing: [f64; 3],
        counts: [usize; 3],
        f: impl Fn(&[f64; 3]) -> f64,
    ) -> Result<Self> {
        let total = counts[0] * counts[1] * counts[2];
        let mut values = Vec::with_capacity(total);
        for k in 0..counts[2] {
            for j in 0..counts[1] {
                for i in 0..counts[0] {
                    values.push(f(&[
                        origin[0] + i as f64 * spacing[0],
                        origin[1] + j as f64 * spacing[1],
                        origin[2] + k as f64 * spacing[2],
                    ]));
                }
            }
        }
        Self::new(origin, spacing, counts, values)
    }

    /// Lattice on the centroids of a `count` grid of `size` blocks, in the
    /// grid's local frame. `cells` holds one field value per block, x-fastest;
    /// blocks where `active` is false fall outside the solid. With `closed`,
    /// a ring of nodes is added around the grid and every node outside the
    /// solid is mirrored below `isovalue`, so the surface caps at the grid's
    /// faces and at the edge of the active blocks. Without it, inactive
    /// blocks are NaN and the surface stops open at them.
    pub fn blocks(
        size: [f64; 3],
        count: [usize; 3],
        cells: &[f64],
        active: Option<&[bool]>,
        isovalue: f64,
        closed: bool,
    ) -> Result<Self> {
        let total = count[0] * count[1] * count[2];
        if cells.len() != total || active.is_some_and(|a| a.len() != total) {
            return Err(ModelError::InvalidParameter(format!(
                "block grid needs {total} cell values"
            )));
        }
        let pad = closed as usize;
        let counts = count.map(|n| n + 2 * pad);
        let values = (0..counts.iter().product::<usize>())
            .map(|n| {
                let ijk = [
                    n % counts[0],
                    n / counts[0] % counts[1],
                    n / (counts[0] * counts[1]),
                ];
                let inner = [0, 1, 2].map(|a| ijk[a].clamp(pad, count[a] + pad - 1) - pad);
                let cell = inner[0] + count[0] * (inner[1] + count[1] * inner[2]);
                let v = cells[cell];
                let on = active.is_none_or(|a| a[cell]) && inner.map(|i| i + pad) == ijk;
                match (on, closed) {
                    (true, _) => v,
                    (false, true) => isovalue - (v - isovalue).abs(),
                    (false, false) => f64::NAN,
                }
            })
            .collect();
        Self::new(size.map(|s| s * (0.5 - pad as f64)), size, counts, values)
    }

    /// Smallest and largest sampled value, or `None` for an empty grid.
    pub fn value_range(&self) -> Option<(f64, f64)> {
        let mut iter = self.values.iter().copied().filter(|v| v.is_finite());
        let first = iter.next()?;
        Some(iter.fold((first, first), |(lo, hi), v| (lo.min(v), hi.max(v))))
    }
}

/// A triangulated surface. Same vertex/index layout as `miningio`'s `Mesh`,
/// which this crate deliberately does not depend on — isosurfacing is pure
/// geometry and the ingestion crate is not in its way.
#[derive(Debug, Clone, Default)]
pub struct TriMesh {
    pub vertices: Vec<[f64; 3]>,
    /// Flattened triangle indices, 3 per triangle.
    pub triangles: Vec<u32>,
}

impl TriMesh {
    pub fn triangle_count(&self) -> usize {
        self.triangles.len() / 3
    }

    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    fn triangle(&self, t: usize) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let a = self.vertices[self.triangles[3 * t] as usize];
        let b = self.vertices[self.triangles[3 * t + 1] as usize];
        let c = self.vertices[self.triangles[3 * t + 2] as usize];
        (a, b, c)
    }

    /// Total area of the triangles (m²).
    pub fn surface_area(&self) -> f64 {
        (0..self.triangle_count())
            .map(|t| {
                let (a, b, c) = self.triangle(t);
                let u = sub(b, a);
                let v = sub(c, a);
                norm(cross(u, v)) * 0.5
            })
            .sum()
    }

    /// Volume enclosed by the surface (m³), by the divergence theorem. Only
    /// meaningful for a closed, outward-oriented mesh — which is what
    /// [`crate::isosurface::marching_tetrahedra`] produces except where the
    /// solid runs out through the edge of the grid.
    pub fn enclosed_volume(&self) -> f64 {
        let sixth: f64 = (0..self.triangle_count())
            .map(|t| {
                let (a, b, c) = self.triangle(t);
                dot(a, cross(b, c))
            })
            .sum();
        (sixth / 6.0).abs()
    }

    pub fn bounds(&self) -> Option<([f64; 3], [f64; 3])> {
        let first = *self.vertices.first()?;
        Some(self.vertices.iter().fold((first, first), |(lo, hi), p| {
            (
                [lo[0].min(p[0]), lo[1].min(p[1]), lo[2].min(p[2])],
                [hi[0].max(p[0]), hi[1].max(p[1]), hi[2].max(p[2])],
            )
        }))
    }
}

pub(crate) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_round_trips_through_the_flat_index() {
        let g =
            ScalarGrid::evaluate([10.0, 20.0, 30.0], [2.0, 3.0, 4.0], [4, 5, 6], |_| 0.0).unwrap();
        assert_eq!(g.node_count(), 120);
        assert_eq!(g.cell_count(), 3 * 4 * 5);
        let idx = g.index(2, 3, 4);
        assert_eq!(g.position(idx), [14.0, 29.0, 46.0]);
    }

    #[test]
    fn evaluate_fills_x_fastest() {
        let g = ScalarGrid::evaluate([0.0; 3], [1.0; 3], [2, 2, 2], |p| p[0]).unwrap();
        assert_eq!(g.values, vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn rejects_a_degenerate_lattice() {
        assert!(ScalarGrid::new([0.0; 3], [1.0; 3], [1, 4, 4], vec![0.0; 16]).is_err());
        assert!(ScalarGrid::new([0.0; 3], [0.0; 3], [2, 2, 2], vec![0.0; 8]).is_err());
        assert!(ScalarGrid::new([0.0; 3], [1.0; 3], [2, 2, 2], vec![0.0; 7]).is_err());
    }

    #[test]
    fn area_and_volume_of_a_unit_tetrahedron() {
        // Outward-oriented tetrahedron on the origin corner: volume 1/6.
        let mesh = TriMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            triangles: vec![0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
        };
        assert!((mesh.enclosed_volume() - 1.0 / 6.0).abs() < 1e-12);
        let want = 3.0 * 0.5 + (3.0f64).sqrt() / 2.0;
        assert!((mesh.surface_area() - want).abs() < 1e-12);
    }
}
