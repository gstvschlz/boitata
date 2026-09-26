use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions, UInt32Array};
use arrow_schema::Schema;
use arrow_select::take::take_record_batch;

use crate::{Error, Result};

/// Triangulated surface or solid with per-vertex and per-face attributes.
#[derive(Debug, Clone)]
pub struct Mesh {
    vertices: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
    vertex_attributes: RecordBatch,
    face_attributes: RecordBatch,
    analysis: MeshAnalysis,
    pub crs: Option<String>,
}

/// Edge topology of a mesh. Degenerate triangles (repeated vertex or zero
/// area) are left out of the edge counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshAnalysis {
    pub degenerate_triangles: usize,
    /// Edges of exactly one triangle.
    pub boundary_edges: usize,
    /// Edges of three or more triangles.
    pub non_manifold_edges: usize,
    /// At least one valid triangle and every edge shared by exactly two.
    pub is_closed: bool,
}

const AREA_EPS: f64 = 1e-10;

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn empty(rows: usize) -> RecordBatch {
    RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(rows)),
    )
    .expect("empty batch")
}

fn degenerate(vertices: &[[f64; 3]], [a, b, c]: [u32; 3]) -> bool {
    let [p, q, r] = [a, b, c].map(|i| vertices[i as usize]);
    let (e1, e2) = (sub(q, p), sub(r, p));
    a == b || b == c || a == c || norm(cross(e1, e2)) <= AREA_EPS * norm(e1) * norm(e2)
}

/// Six times the signed volume the triangles enclose with `o`.
fn volume6(vertices: &[[f64; 3]], o: [f64; 3], triangles: impl Iterator<Item = [u32; 3]>) -> f64 {
    triangles
        .map(|t| {
            let [a, b, c] = t.map(|i| sub(vertices[i as usize], o));
            let n = cross(b, c);
            a[0] * n[0] + a[1] * n[1] + a[2] * n[2]
        })
        .sum()
}

fn take_rows(batch: &RecordBatch, keep: Vec<u32>) -> Result<RecordBatch> {
    if batch.num_columns() == 0 {
        return Ok(empty(keep.len()));
    }
    take_record_batch(batch, &UInt32Array::from(keep)).map_err(|e| Error::Geometry(e.to_string()))
}

fn analyze(vertices: &[[f64; 3]], triangles: &[[u32; 3]]) -> MeshAnalysis {
    let mut degenerate_triangles = 0;
    let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
    for &[a, b, c] in triangles {
        if degenerate(vertices, [a, b, c]) {
            degenerate_triangles += 1;
            continue;
        }
        for (i, j) in [(a, b), (b, c), (c, a)] {
            *edges.entry((i.min(j), i.max(j))).or_insert(0) += 1;
        }
    }
    let boundary_edges = edges.values().filter(|&&n| n == 1).count();
    let non_manifold_edges = edges.values().filter(|&&n| n >= 3).count();
    MeshAnalysis {
        degenerate_triangles,
        boundary_edges,
        non_manifold_edges,
        is_closed: triangles.len() > degenerate_triangles
            && boundary_edges == 0
            && non_manifold_edges == 0,
    }
}

impl Mesh {
    /// Checks that vertices are finite and triangles index them.
    pub fn new(vertices: Vec<[f64; 3]>, triangles: Vec<[u32; 3]>) -> Result<Self> {
        if vertices.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Error::Geometry("mesh vertices must be finite".into()));
        }
        if triangles
            .iter()
            .flatten()
            .any(|&i| i as usize >= vertices.len())
        {
            return Err(Error::Geometry(
                "triangle index outside the vertices".into(),
            ));
        }
        Ok(Self {
            analysis: analyze(&vertices, &triangles),
            vertex_attributes: empty(vertices.len()),
            face_attributes: empty(triangles.len()),
            vertices,
            triangles,
            crs: None,
        })
    }

    pub fn vertices(&self) -> &[[f64; 3]] {
        &self.vertices
    }

    pub fn triangles(&self) -> &[[u32; 3]] {
        &self.triangles
    }

    pub fn corners(&self, triangle: usize) -> [[f64; 3]; 3] {
        self.triangles[triangle].map(|i| self.vertices[i as usize])
    }

    pub fn vertex_attributes(&self) -> &RecordBatch {
        &self.vertex_attributes
    }

    pub fn face_attributes(&self) -> &RecordBatch {
        &self.face_attributes
    }

    /// Adds or replaces the per-vertex attribute `name`.
    pub fn with_vertex_column(&self, name: &str, column: ArrayRef) -> Result<Self> {
        Ok(Self {
            vertex_attributes: crate::set_column(&self.vertex_attributes, name, column)?,
            ..self.clone()
        })
    }

    /// Adds or replaces the per-triangle attribute `name`.
    pub fn with_face_column(&self, name: &str, column: ArrayRef) -> Result<Self> {
        Ok(Self {
            face_attributes: crate::set_column(&self.face_attributes, name, column)?,
            ..self.clone()
        })
    }

    pub fn analysis(&self) -> MeshAnalysis {
        self.analysis
    }

    pub fn is_closed(&self) -> bool {
        self.analysis.is_closed
    }

    pub fn area(&self) -> f64 {
        (0..self.triangles.len())
            .map(|t| {
                let [a, b, c] = self.corners(t);
                norm(cross(sub(b, a), sub(c, a))) / 2.0
            })
            .sum()
    }

    /// Enclosed volume, positive when triangles wind counter-clockwise seen
    /// from outside.
    pub fn volume(&self) -> Result<f64> {
        if !self.is_closed() {
            return Err(Error::Geometry("volume needs a closed mesh".into()));
        }
        let o = self.vertices[0];
        Ok(volume6(&self.vertices, o, self.triangles.iter().copied()) / 6.0)
    }

    /// Copy with vertices within `tolerance` of an earlier one welded to it,
    /// degenerate and duplicate triangles and unused vertices dropped, and
    /// each connected piece wound consistently, outward where it is closed.
    /// Kept vertices and triangles keep their order and attributes.
    pub fn repair(&self, tolerance: f64) -> Result<Self> {
        if !(tolerance.is_finite() && tolerance >= 0.0) {
            return Err(Error::Geometry("tolerance must be finite and >= 0".into()));
        }
        let key = |p: [f64; 3]| {
            p.map(|v| match tolerance > 0.0 {
                true => (v / tolerance).floor() as i64,
                false => (v + 0.0).to_bits() as i64,
            })
        };
        let reach = i64::from(tolerance > 0.0);
        let mut cells: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
        let mut first = Vec::with_capacity(self.vertices.len());
        for (i, &p) in self.vertices.iter().enumerate() {
            let k = key(p);
            let mut found: Option<u32> = None;
            for dx in -reach..=reach {
                for dy in -reach..=reach {
                    for dz in -reach..=reach {
                        let cell = [
                            k[0].wrapping_add(dx),
                            k[1].wrapping_add(dy),
                            k[2].wrapping_add(dz),
                        ];
                        for &j in cells.get(&cell).into_iter().flatten() {
                            if norm(sub(self.vertices[j as usize], p)) <= tolerance {
                                found = Some(found.map_or(j, |f| f.min(j)));
                            }
                        }
                    }
                }
            }
            first.push(found.unwrap_or_else(|| {
                cells.entry(k).or_default().push(i as u32);
                i as u32
            }));
        }

        let mut seen = HashSet::new();
        let (mut faces, mut triangles) = (vec![], vec![]);
        for (f, t) in self.triangles.iter().enumerate() {
            let t = t.map(|i| first[i as usize]);
            let mut sorted = t;
            sorted.sort_unstable();
            if !degenerate(&self.vertices, t) && seen.insert(sorted) {
                faces.push(f as u32);
                triangles.push(t);
            }
        }

        let mut edges: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        for (f, &[a, b, c]) in triangles.iter().enumerate() {
            for (i, j) in [(a, b), (b, c), (c, a)] {
                edges.entry((i.min(j), i.max(j))).or_default().push(f);
            }
        }
        let directed = |t: [u32; 3]| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])];
        let mut flip: Vec<Option<bool>> = vec![None; triangles.len()];
        for seed in 0..triangles.len() {
            if flip[seed].is_some() {
                continue;
            }
            flip[seed] = Some(false);
            let (mut piece, mut next, mut closed) = (vec![seed], 0, true);
            while let Some(&f) = piece.get(next) {
                next += 1;
                let mut t = triangles[f];
                if flip[f] == Some(true) {
                    t.swap(1, 2);
                }
                for (a, b) in directed(t) {
                    let shared = &edges[&(a.min(b), a.max(b))];
                    closed &= shared.len() == 2;
                    if shared.len() != 2 {
                        continue;
                    }
                    let g = if shared[0] == f { shared[1] } else { shared[0] };
                    if flip[g].is_none() {
                        flip[g] = Some(directed(triangles[g]).contains(&(a, b)));
                        piece.push(g);
                    }
                }
            }
            let oriented = |&f: &usize| {
                let mut t = triangles[f];
                if flip[f] == Some(true) {
                    t.swap(1, 2);
                }
                t
            };
            let o = self.vertices[triangles[seed][0] as usize];
            if closed && volume6(&self.vertices, o, piece.iter().map(oriented)) < 0.0 {
                for f in piece {
                    flip[f] = flip[f].map(|x| !x);
                }
            }
        }
        for (t, flip) in triangles.iter_mut().zip(&flip) {
            if *flip == Some(true) {
                t.swap(1, 2);
            }
        }

        let mut used: Vec<u32> = triangles.iter().flatten().copied().collect();
        used.sort_unstable();
        used.dedup();
        let triangles = triangles
            .iter()
            .map(|t| t.map(|i| used.binary_search(&i).expect("used") as u32))
            .collect();
        let vertices = used.iter().map(|&i| self.vertices[i as usize]).collect();
        let mut mesh = Self::new(vertices, triangles)?;
        mesh.vertex_attributes = take_rows(&self.vertex_attributes, used)?;
        mesh.face_attributes = take_rows(&self.face_attributes, faces)?;
        mesh.crs = self.crs.clone();
        Ok(mesh)
    }

    /// `(min, max)` corners, `None` without vertices.
    pub fn bounds(&self) -> Option<([f64; 3], [f64; 3])> {
        let first = *self.vertices.first()?;
        Some(self.vertices.iter().fold((first, first), |(lo, hi), p| {
            (
                [0, 1, 2].map(|a| lo[a].min(p[a])),
                [0, 1, 2].map(|a| hi[a].max(p[a])),
            )
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Float64Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Float64Type;

    fn mesh(vertices: &[[f64; 3]], triangles: &[[u32; 3]]) -> Mesh {
        Mesh::new(vertices.to_vec(), triangles.to_vec()).unwrap()
    }

    fn cube() -> Mesh {
        let v: Vec<[f64; 3]> = (0..8)
            .map(|i| [(i & 1) as f64, (i >> 1 & 1) as f64, (i >> 2) as f64])
            .collect();
        mesh(
            &v,
            &[
                [0, 2, 1],
                [1, 2, 3],
                [4, 5, 6],
                [5, 7, 6],
                [0, 1, 4],
                [1, 5, 4],
                [2, 6, 3],
                [3, 6, 7],
                [0, 4, 2],
                [2, 4, 6],
                [1, 3, 5],
                [3, 7, 5],
            ],
        )
    }

    #[test]
    fn unit_cube_is_closed_with_unit_volume() {
        let c = cube();
        assert!(c.is_closed());
        assert!((c.volume().unwrap() - 1.0).abs() < 1e-12);
        assert!((c.area() - 6.0).abs() < 1e-12);
        assert_eq!(c.bounds(), Some(([0.0; 3], [1.0; 3])));
    }

    #[test]
    fn open_square_has_four_boundary_edges() {
        let m = mesh(
            &[[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]],
            &[[0, 1, 2], [0, 2, 3]],
        );
        let a = m.analysis();
        assert_eq!((a.boundary_edges, a.non_manifold_edges), (4, 0));
        assert!(!a.is_closed && m.volume().is_err());
    }

    #[test]
    fn fin_is_non_manifold() {
        let m = mesh(
            &[
                [0., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., -1., 0.],
                [0., 0., 1.],
            ],
            &[[0, 1, 2], [0, 1, 3], [0, 1, 4]],
        );
        assert_eq!(m.analysis().non_manifold_edges, 1);
        assert!(!m.is_closed());
    }

    #[test]
    fn repeated_and_collinear_triangles_are_degenerate() {
        let m = mesh(
            &[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]],
            &[[0, 1, 1], [0, 1, 2]],
        );
        let a = m.analysis();
        assert_eq!((a.degenerate_triangles, a.boundary_edges), (2, 0));
        assert!(!a.is_closed);
    }

    #[test]
    fn empty_mesh_is_open_and_bad_input_is_rejected() {
        assert!(!mesh(&[], &[]).is_closed());
        assert!(Mesh::new(vec![[0.0; 3]], vec![[0, 0, 9]]).is_err());
        assert!(Mesh::new(vec![[f64::NAN, 0.0, 0.0]], vec![]).is_err());
    }

    /// Theory check: a cube as a triangle soup (each triangle its own nearly
    /// equal corners), half its faces flipped, with a sliver and a repeated
    /// face, repairs to the closed, outward unit cube.
    #[test]
    fn repair_rebuilds_a_broken_cube() {
        let c = cube();
        let mut vertices = vec![];
        let mut triangles = vec![];
        for (t, tri) in c.triangles().iter().enumerate() {
            let n = vertices.len() as u32;
            vertices.extend(tri.map(|i| c.vertices()[i as usize].map(|v| v + 1e-7 * t as f64)));
            triangles.push(if t % 2 == 0 {
                [n, n + 1, n + 2]
            } else {
                [n, n + 2, n + 1]
            });
        }
        triangles.push(triangles[3]);
        triangles.push([0, 1, 1]);
        let faces: ArrayRef = Arc::new(Float64Array::from_iter_values((0..14).map(f64::from)));
        let broken = mesh(&vertices, &triangles)
            .with_face_column("face", faces)
            .unwrap();
        assert!(!broken.is_closed());
        assert!(broken.repair(1e-9).unwrap().triangles().len() == 12);
        assert!(!broken.repair(1e-9).unwrap().is_closed());

        let fixed = broken.repair(1e-5).unwrap();
        assert_eq!((fixed.vertices().len(), fixed.triangles().len()), (8, 12));
        assert!(fixed.is_closed());
        assert!((fixed.volume().unwrap() - 1.0).abs() < 1e-5);
        let face = fixed
            .face_attributes()
            .column(0)
            .as_primitive::<Float64Type>();
        assert_eq!(
            face.values().to_vec(),
            (0..12).map(f64::from).collect::<Vec<_>>()
        );

        let again = fixed.repair(1e-5).unwrap();
        assert_eq!(again.vertices(), fixed.vertices());
        assert_eq!(again.triangles(), fixed.triangles());
        assert_eq!(broken.repair(1e-5).unwrap().triangles(), fixed.triangles());
    }

    #[test]
    fn repair_turns_an_inward_cube_outward_and_rejects_bad_tolerance() {
        let c = cube();
        let inward: Vec<[u32; 3]> = c.triangles().iter().map(|&[a, b, c]| [a, c, b]).collect();
        let fixed = mesh(c.vertices(), &inward).repair(0.0).unwrap();
        assert!((fixed.volume().unwrap() - 1.0).abs() < 1e-12);
        assert!(c.repair(-1.0).is_err() && c.repair(f64::NAN).is_err());
    }

    #[test]
    fn columns_are_length_checked() {
        let c = cube();
        let per_vertex: ArrayRef = Arc::new(Float64Array::from(vec![1.0; 8]));
        let c = c.with_vertex_column("v", per_vertex.clone()).unwrap();
        assert_eq!(c.vertex_attributes().num_columns(), 1);
        assert!(c.with_face_column("f", per_vertex).is_err());
        let per_face: ArrayRef = Arc::new(Float64Array::from(vec![1.0; 12]));
        assert!(c.with_face_column("f", per_face).is_ok());
    }
}
