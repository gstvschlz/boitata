use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions, UInt32Array};
use arrow_schema::Schema;
use arrow_select::take::take_record_batch;
use nalgebra::Vector3;

use crate::{Error, Result};

/// Triangulated surface or solid with per-vertex and per-face attributes.
#[derive(Debug, Clone)]
pub struct Mesh {
    vertices: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
    vertex_attributes: RecordBatch,
    face_attributes: RecordBatch,
    boundary_edges: usize,
    closed: bool,
    pub crs: Option<String>,
    /// Unit of the coordinates, such as `m` or `ft`; None when not declared.
    pub length_unit: Option<String>,
}

/// What [`Mesh::validate`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MeshProblemKind {
    /// Repeated corner or zero area.
    DegenerateFace,
    /// Same corners as an earlier face, in any order.
    DuplicateFace,
    /// Within the tolerance of an earlier vertex.
    DuplicateVertex,
    /// Edge of exactly one face.
    BoundaryEdge,
    /// Edge of three or more faces.
    NonManifoldEdge,
    /// Its faces form more than one fan, e.g. two cones touching at the apex.
    NonManifoldVertex,
    /// Edge whose two faces run it the same way.
    InconsistentWinding,
    /// Closed shell wound against its nesting: inward outside cavities, or
    /// outward as a cavity.
    InwardShell,
    /// Face crossing an earlier face it shares no edge with. Coplanar
    /// overlaps and faces touching at a point or along an edge do not count.
    SelfIntersection,
}

impl MeshProblemKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::DegenerateFace => "degenerate_face",
            Self::DuplicateFace => "duplicate_face",
            Self::DuplicateVertex => "duplicate_vertex",
            Self::BoundaryEdge => "boundary_edge",
            Self::NonManifoldEdge => "non_manifold_edge",
            Self::NonManifoldVertex => "non_manifold_vertex",
            Self::InconsistentWinding => "inconsistent_winding",
            Self::InwardShell => "inward_shell",
            Self::SelfIntersection => "self_intersection",
        }
    }
}

/// One defect. Edge problems give a `face` holding the edge, its local
/// `edge` (corner `edge` to corner `(edge + 1) % 3`) and its first `vertex`.
/// `other` is the earlier face or vertex a duplicate repeats, or the other
/// face of an inconsistent edge, or the earlier face of a crossing pair. An
/// inward shell gives its first face.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshProblem {
    pub kind: MeshProblemKind,
    pub face: Option<u32>,
    pub vertex: Option<u32>,
    pub edge: Option<u8>,
    pub other: Option<u32>,
}

/// Problem counts of [`Mesh::validate`]. Degenerate faces are left out of
/// the edge, vertex and shell checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MeshSummary {
    pub degenerate_faces: usize,
    pub duplicate_faces: usize,
    pub duplicate_vertices: usize,
    pub boundary_edges: usize,
    pub non_manifold_edges: usize,
    pub non_manifold_vertices: usize,
    pub inconsistent_edges: usize,
    /// Pieces connected through shared edges.
    pub shells: usize,
    pub inward_shells: usize,
    /// Pairs of crossing faces.
    pub self_intersections: usize,
    /// At least one valid face and every edge shared by exactly two.
    pub is_closed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshReport {
    pub summary: MeshSummary,
    /// Ordered by kind, then face or vertex.
    pub problems: Vec<MeshProblem>,
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

/// Signed solid angle subtended by a triangle at the origin (van Oosterom &
/// Strackee, IEEE Trans. Biomed. Eng.). Summed over a closed mesh and divided
/// by 4π it is the winding number: ±1 inside, 0 outside. `atan2` keeps angles
/// past π in the right quadrant; `atan2(0, 0) = 0` covers degenerate triangles
/// and points on a vertex.
pub fn signed_solid_angle(a: &Vector3<f64>, b: &Vector3<f64>, c: &Vector3<f64>) -> f64 {
    let num = a.dot(&b.cross(c));
    let denom = a.norm() * b.norm() * c.norm()
        + a.dot(b) * c.norm()
        + b.dot(c) * a.norm()
        + c.dot(a) * b.norm();
    2.0 * num.atan2(denom)
}

fn encloses(vertices: &[[f64; 3]], triangles: &[[u32; 3]], p: [f64; 3]) -> bool {
    let p = Vector3::from(p);
    let w: f64 = triangles
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|i| Vector3::from(vertices[i as usize]) - p);
            signed_solid_angle(&a, &b, &c)
        })
        .sum();
    (w / (4.0 * std::f64::consts::PI)).abs() > 0.5
}

pub(crate) fn take_rows(batch: &RecordBatch, keep: Vec<u32>) -> Result<RecordBatch> {
    if batch.num_columns() == 0 {
        return Ok(empty(keep.len()));
    }
    take_record_batch(batch, &UInt32Array::from(keep)).map_err(|e| Error::Geometry(e.to_string()))
}

fn edge_counts(vertices: &[[f64; 3]], triangles: &[[u32; 3]]) -> (usize, bool) {
    let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
    let mut valid = 0;
    for &t in triangles {
        if degenerate(vertices, t) {
            continue;
        }
        valid += 1;
        for (i, j) in directed(t) {
            *edges.entry((i.min(j), i.max(j))).or_insert(0) += 1;
        }
    }
    let boundary = edges.values().filter(|&&n| n == 1).count();
    (boundary, valid > 0 && edges.values().all(|&n| n == 2))
}

fn directed(t: [u32; 3]) -> [(u32, u32); 3] {
    [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]
}

fn check_tolerance(tolerance: f64) -> Result<()> {
    if tolerance.is_finite() && tolerance >= 0.0 {
        Ok(())
    } else {
        Err(Error::Geometry("tolerance must be finite and >= 0".into()))
    }
}

/// For each vertex, the earliest vertex within `tolerance` of it (itself if
/// none), by grid hashing.
fn weld(vertices: &[[f64; 3]], tolerance: f64) -> Vec<u32> {
    let key = |p: [f64; 3]| {
        p.map(|v| match tolerance > 0.0 {
            true => (v / tolerance).floor() as i64,
            false => (v + 0.0).to_bits() as i64,
        })
    };
    let reach = i64::from(tolerance > 0.0);
    let mut cells: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
    let mut first = Vec::with_capacity(vertices.len());
    for (i, &p) in vertices.iter().enumerate() {
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
                        if norm(sub(vertices[j as usize], p)) <= tolerance {
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
    first
}

/// Which closed, consistently wound shells wind against their nesting: a
/// shell inside an even number of others should wind outward, inside an odd
/// number inward.
fn misoriented(vertices: &[[f64; 3]], shells: &[Vec<[u32; 3]>]) -> Vec<bool> {
    let mut owners: HashMap<u32, usize> = HashMap::new();
    for s in shells {
        let mut v: Vec<u32> = s.iter().flatten().copied().collect();
        v.sort_unstable();
        v.dedup();
        for i in v {
            *owners.entry(i).or_default() += 1;
        }
    }
    shells
        .iter()
        .enumerate()
        .map(|(i, own)| {
            let v = own.iter().flatten().copied().find(|v| owners[v] == 1);
            let p = vertices[v.unwrap_or(own[0][0]) as usize];
            let depth = shells
                .iter()
                .enumerate()
                .filter(|&(j, t)| j != i && encloses(vertices, t, p))
                .count();
            let outward = volume6(vertices, p, own.iter().copied()) > 0.0;
            outward != (depth % 2 == 0)
        })
        .collect()
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Whether segment `p`-`q` passes through the interior of triangle `t`,
/// excluding its ends, the triangle's edges and segments in its plane.
fn pierces(p: [f64; 3], q: [f64; 3], [a, b, c]: [[f64; 3]; 3]) -> bool {
    const EPS: f64 = 1e-9;
    let n = cross(sub(b, a), sub(c, a));
    let (dp, dq) = (dot(n, sub(p, a)), dot(n, sub(q, a)));
    if dp * dq >= 0.0 {
        return false;
    }
    let s = dp / (dp - dq);
    if s <= EPS || s >= 1.0 - EPS {
        return false;
    }
    let x = [0, 1, 2].map(|k| p[k] + s * (q[k] - p[k]));
    let nn = dot(n, n);
    [(a, b), (b, c), (c, a)]
        .iter()
        .all(|&(u, v)| dot(cross(sub(v, u), sub(x, u)), n) > EPS * nn)
}

/// Pairs `(earlier, later)` of valid faces sharing at most one vertex whose
/// interiors cross: two triangles cross when an edge of one pierces the
/// other. Candidates come from a sweep over x-sorted bounding boxes.
fn crossings(vertices: &[[f64; 3]], triangles: &[[u32; 3]], valid: &[bool]) -> Vec<(usize, usize)> {
    let corners = |f: usize| triangles[f].map(|i| vertices[i as usize]);
    let boxes: Vec<[[f64; 3]; 2]> = (0..triangles.len())
        .map(|f| {
            let c = corners(f);
            let lo = [0, 1, 2].map(|k| c[0][k].min(c[1][k]).min(c[2][k]));
            let hi = [0, 1, 2].map(|k| c[0][k].max(c[1][k]).max(c[2][k]));
            [lo, hi]
        })
        .collect();
    let mut order: Vec<usize> = (0..triangles.len()).filter(|&f| valid[f]).collect();
    order.sort_by(|&f, &g| boxes[f][0][0].total_cmp(&boxes[g][0][0]).then(f.cmp(&g)));
    let through = |e: [[f64; 3]; 3], t| (0..3).any(|k| pierces(e[k], e[(k + 1) % 3], t));
    let mut pairs = vec![];
    for (n, &f) in order.iter().enumerate() {
        for &g in &order[n + 1..] {
            if boxes[g][0][0] > boxes[f][1][0] {
                break;
            }
            let apart =
                (1..3).any(|k| boxes[g][0][k] > boxes[f][1][k] || boxes[f][0][k] > boxes[g][1][k]);
            let shared = triangles[f]
                .iter()
                .filter(|i| triangles[g].contains(i))
                .count();
            if apart || shared > 1 {
                continue;
            }
            let (cf, cg) = (corners(f), corners(g));
            if through(cf, cg) || through(cg, cf) {
                pairs.push((f.min(g), f.max(g)));
            }
        }
    }
    pairs
}

fn root(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

fn join(parent: &mut [usize], a: usize, b: usize) {
    let (a, b) = (root(parent, a), root(parent, b));
    parent[a.max(b)] = a.min(b);
}

fn validate(vertices: &[[f64; 3]], triangles: &[[u32; 3]], tolerance: f64) -> MeshReport {
    use MeshProblemKind::*;
    let problem = |kind, face: Option<usize>, vertex: Option<u32>| MeshProblem {
        kind,
        face: face.map(|f| f as u32),
        vertex,
        edge: None,
        other: None,
    };
    let mut problems = vec![];
    let valid: Vec<bool> = triangles
        .iter()
        .map(|&t| !degenerate(vertices, t))
        .collect();
    let faces = || (0..triangles.len()).filter(|&f| valid[f]);
    for f in (0..triangles.len()).filter(|&f| !valid[f]) {
        problems.push(problem(DegenerateFace, Some(f), None));
    }
    let mut seen: HashMap<[u32; 3], usize> = HashMap::new();
    for f in faces() {
        let mut sorted = triangles[f];
        sorted.sort_unstable();
        let first = *seen.entry(sorted).or_insert(f);
        if first != f {
            problems.push(MeshProblem {
                other: Some(first as u32),
                ..problem(DuplicateFace, Some(f), None)
            });
        }
    }
    for (v, &w) in weld(vertices, tolerance).iter().enumerate() {
        if w != v as u32 {
            problems.push(MeshProblem {
                other: Some(w),
                ..problem(DuplicateVertex, None, Some(v as u32))
            });
        }
    }

    let mut edges: BTreeMap<(u32, u32), Vec<(usize, u8)>> = BTreeMap::new();
    for f in faces() {
        for (k, (i, j)) in directed(triangles[f]).into_iter().enumerate() {
            edges
                .entry((i.min(j), i.max(j)))
                .or_default()
                .push((f, k as u8));
        }
    }
    let edge_problem = |kind, (f, k): (usize, u8)| MeshProblem {
        edge: Some(k),
        ..problem(kind, Some(f), Some(triangles[f][k as usize]))
    };
    let mut shell_of: Vec<usize> = (0..triangles.len()).collect();
    let mut inconsistent = vec![false; triangles.len()];
    for shared in edges.values() {
        for &(g, _) in &shared[1..] {
            join(&mut shell_of, shared[0].0, g);
        }
        match shared[..] {
            [one] => problems.push(edge_problem(BoundaryEdge, one)),
            [a, b] if triangles[a.0][a.1 as usize] == triangles[b.0][b.1 as usize] => {
                inconsistent[a.0] = true;
                inconsistent[b.0] = true;
                problems.push(MeshProblem {
                    other: Some(a.0 as u32),
                    ..edge_problem(InconsistentWinding, b)
                });
            }
            [_, _] => {}
            _ => problems.push(edge_problem(NonManifoldEdge, shared[0])),
        }
    }

    let mut around: Vec<Vec<usize>> = vec![vec![]; vertices.len()];
    for f in faces() {
        for v in triangles[f] {
            around[v as usize].push(f);
        }
    }
    for (v, fan) in around.iter().enumerate() {
        let v = v as u32;
        let mut parent: Vec<usize> = (0..fan.len()).collect();
        for (i, &f) in fan.iter().enumerate() {
            for &w in triangles[f].iter().filter(|&&w| w != v) {
                for &(g, _) in &edges[&(v.min(w), v.max(w))] {
                    let j = fan.iter().position(|&h| h == g).expect("g holds v");
                    join(&mut parent, i, j);
                }
            }
        }
        if (0..fan.len())
            .filter(|&i| root(&mut parent, i) == i)
            .count()
            > 1
        {
            problems.push(problem(NonManifoldVertex, None, Some(v)));
        }
    }

    let mut shells: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for f in faces() {
        shells.entry(root(&mut shell_of, f)).or_default().push(f);
    }
    let shared = |(i, j): (u32, u32)| edges[&(i.min(j), i.max(j))].len();
    let closed: Vec<&Vec<usize>> = shells
        .values()
        .filter(|s| {
            s.iter().all(|&f| {
                !inconsistent[f] && directed(triangles[f]).into_iter().all(|e| shared(e) == 2)
            })
        })
        .collect();
    let closed_triangles: Vec<Vec<[u32; 3]>> = closed
        .iter()
        .map(|s| s.iter().map(|&f| triangles[f]).collect())
        .collect();
    for (s, wrong) in closed.iter().zip(misoriented(vertices, &closed_triangles)) {
        if wrong {
            problems.push(problem(InwardShell, Some(s[0]), None));
        }
    }
    for (f, g) in crossings(vertices, triangles, &valid) {
        problems.push(MeshProblem {
            other: Some(f as u32),
            ..problem(SelfIntersection, Some(g), None)
        });
    }
    problems.sort_by_key(|p| (p.kind, p.face, p.vertex, p.edge, p.other));

    let count = |kind| problems.iter().filter(|p| p.kind == kind).count();
    let (boundary_edges, non_manifold_edges) = (count(BoundaryEdge), count(NonManifoldEdge));
    let summary = MeshSummary {
        degenerate_faces: count(DegenerateFace),
        duplicate_faces: count(DuplicateFace),
        duplicate_vertices: count(DuplicateVertex),
        boundary_edges,
        non_manifold_edges,
        non_manifold_vertices: count(NonManifoldVertex),
        inconsistent_edges: count(InconsistentWinding),
        shells: shells.len(),
        inward_shells: count(InwardShell),
        self_intersections: count(SelfIntersection),
        is_closed: valid.contains(&true) && boundary_edges == 0 && non_manifold_edges == 0,
    };
    MeshReport { summary, problems }
}

impl Mesh {
    /// The mesh with coordinates converted to `unit`, which becomes the length
    /// unit; `crs` names their CRS, required when they had one.
    pub fn to_length_unit(&self, unit: &str, crs: Option<String>) -> Result<Self> {
        let (k, crs) =
            crate::units::rescale(self.length_unit.as_deref(), self.crs.as_deref(), unit, crs)?;
        let mut out = self.clone();
        out.vertices.iter_mut().flatten().for_each(|v| *v *= k);
        out.crs = crs;
        out.length_unit = Some(unit.into());
        Ok(out)
    }

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
        let (boundary_edges, closed) = edge_counts(&vertices, &triangles);
        Ok(Self {
            boundary_edges,
            closed,
            vertex_attributes: empty(vertices.len()),
            face_attributes: empty(triangles.len()),
            vertices,
            triangles,
            crs: None,
            length_unit: None,
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

    /// Degenerate, duplicate, edge, vertex, orientation and crossing problems.
    /// `tolerance` is the distance within which a vertex duplicates an
    /// earlier one.
    pub fn validate(&self, tolerance: f64) -> Result<MeshReport> {
        check_tolerance(tolerance)?;
        Ok(validate(&self.vertices, &self.triangles, tolerance))
    }

    /// Every edge shared by exactly two non-degenerate triangles.
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Edges of exactly one non-degenerate triangle.
    pub fn boundary_edges(&self) -> usize {
        self.boundary_edges
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
    /// each connected piece wound consistently. A closed piece inside an even
    /// number of other closed pieces winds outward; inside an odd number it is
    /// a cavity and winds inward. Kept vertices and triangles keep their order
    /// and attributes. Crossing faces are left as they are.
    pub fn repair(&self, tolerance: f64) -> Result<Self> {
        check_tolerance(tolerance)?;
        let first = weld(&self.vertices, tolerance);
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
        let mut flip: Vec<Option<bool>> = vec![None; triangles.len()];
        let mut shells = vec![];
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
            if closed {
                shells.push(piece);
            }
        }
        for (t, flip) in triangles.iter_mut().zip(&flip) {
            if *flip == Some(true) {
                t.swap(1, 2);
            }
        }
        let shell_triangles: Vec<Vec<[u32; 3]>> = shells
            .iter()
            .map(|s| s.iter().map(|&f| triangles[f]).collect())
            .collect();
        for (shell, wrong) in shells
            .iter()
            .zip(misoriented(&self.vertices, &shell_triangles))
        {
            if wrong {
                for &f in shell {
                    triangles[f].swap(1, 2);
                }
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
        mesh.length_unit = self.length_unit.clone();
        Ok(mesh)
    }

    /// Copy with each hole closed by a fan of triangles from its first
    /// vertex, wound like the faces around it. A hole is a loop of boundary
    /// edges; a loop through a vertex that starts two boundary edges stays
    /// open. New faces come last with null attributes. A fan suits small,
    /// nearly planar, convex holes; others can come out folded.
    pub fn fill_holes(&self) -> Result<Self> {
        let valid: Vec<[u32; 3]> = self
            .triangles
            .iter()
            .copied()
            .filter(|&t| !degenerate(&self.vertices, t))
            .collect();
        let mut count: HashMap<(u32, u32), usize> = HashMap::new();
        for (i, j) in valid.iter().flat_map(|&t| directed(t)) {
            *count.entry((i.min(j), i.max(j))).or_default() += 1;
        }
        let boundary: Vec<(u32, u32)> = valid
            .iter()
            .flat_map(|&t| directed(t))
            .filter(|&(i, j)| count[&(i.min(j), i.max(j))] == 1)
            .collect();
        let mut next: HashMap<u32, Vec<u32>> = HashMap::new();
        for &(i, j) in &boundary {
            next.entry(i).or_default().push(j);
        }
        let mut seen = HashSet::new();
        let mut triangles = self.triangles.clone();
        for &(start, _) in &boundary {
            if !seen.insert(start) {
                continue;
            }
            let mut ring = vec![start];
            let closed = loop {
                let &[w] = &next[ring.last().expect("ring")][..] else {
                    break false;
                };
                if w == start {
                    break true;
                }
                if !seen.insert(w) || !next.contains_key(&w) {
                    break false;
                }
                ring.push(w);
            };
            if closed && ring.len() >= 3 {
                triangles.extend(ring.windows(2).skip(1).map(|w| [start, w[1], w[0]]));
            }
        }
        let n = self.triangles.len() as u32;
        let rows: Vec<Option<u32>> = (0..triangles.len() as u32)
            .map(|f| (f < n).then_some(f))
            .collect();
        let face_attributes = match self.face_attributes.num_columns() {
            0 => empty(rows.len()),
            _ => take_record_batch(&self.face_attributes, &UInt32Array::from(rows))
                .map_err(|e| Error::Geometry(e.to_string()))?,
        };
        let mut mesh = Self::new(self.vertices.clone(), triangles)?;
        mesh.vertex_attributes = self.vertex_attributes.clone();
        mesh.face_attributes = face_attributes;
        mesh.crs = self.crs.clone();
        mesh.length_unit = self.length_unit.clone();
        Ok(mesh)
    }

    /// `(min, max)` corners, `None` without vertices.
    pub fn bounds(&self) -> Option<([f64; 3], [f64; 3])> {
        crate::bounds(self.vertices.iter().copied())
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
        let a = m.validate(0.0).unwrap().summary;
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
        assert_eq!(m.validate(0.0).unwrap().summary.non_manifold_edges, 1);
        assert!(!m.is_closed());
    }

    #[test]
    fn repeated_and_collinear_triangles_are_degenerate() {
        let m = mesh(
            &[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]],
            &[[0, 1, 1], [0, 1, 2]],
        );
        let a = m.validate(0.0).unwrap().summary;
        assert_eq!((a.degenerate_faces, a.boundary_edges), (2, 0));
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

    /// Theory check: a 3-cube with a unit-cube cavity repairs to volume 27 - 1
    /// with the cavity wound inward, from any starting windings.
    #[test]
    fn repair_keeps_a_cavity_wound_inward() {
        let c = cube();
        let mut vertices: Vec<[f64; 3]> = c.vertices().iter().map(|p| p.map(|v| 3.0 * v)).collect();
        vertices.extend(c.vertices().iter().map(|p| p.map(|v| v + 1.0)));
        let outer = c.triangles().to_vec();
        let inner: Vec<[u32; 3]> = outer.iter().map(|t| t.map(|i| i + 8)).collect();
        let flipped = |ts: &[[u32; 3]]| ts.iter().map(|&[a, b, c]| [a, c, b]).collect::<Vec<_>>();
        let correct = [outer.clone(), flipped(&inner)].concat();
        let fixed = mesh(&vertices, &correct).repair(0.0).unwrap();
        assert_eq!(fixed.triangles(), &correct[..]);
        assert!((fixed.volume().unwrap() - 26.0).abs() < 1e-12);
        for start in [
            [outer.clone(), inner.clone()],
            [flipped(&outer), inner.clone()],
            [flipped(&outer), flipped(&inner)],
        ] {
            let m = mesh(&vertices, &start.concat()).repair(0.0).unwrap();
            assert_eq!(m.triangles(), &correct[..]);
        }
        let inside = |p| encloses(fixed.vertices(), fixed.triangles(), p);
        assert!(inside([0.5; 3]) && !inside([1.5; 3]) && !inside([4.0; 3]));
    }

    fn kinds(m: &Mesh, tolerance: f64) -> Vec<(MeshProblemKind, Option<u32>, Option<u32>)> {
        let r = m.validate(tolerance).unwrap();
        r.problems
            .iter()
            .map(|p| (p.kind, p.face, p.vertex))
            .collect()
    }

    /// Theory check: V - E + F = 2 for every closed shell of genus 0, and a
    /// clean cube reports one shell and no problems.
    #[test]
    fn clean_cube_validates_with_euler_characteristic_two() {
        let c = cube();
        let r = c.validate(1e-9).unwrap();
        assert!(r.problems.is_empty());
        assert_eq!(r.summary.shells, 1);
        assert!(r.summary.is_closed);
        let mut edges: Vec<(u32, u32)> = c
            .triangles()
            .iter()
            .flat_map(|&t| directed(t).map(|(i, j)| (i.min(j), i.max(j))))
            .collect();
        edges.sort_unstable();
        edges.dedup();
        let euler = c.vertices().len() as i64 - edges.len() as i64 + c.triangles().len() as i64;
        assert_eq!(euler, 2);
    }

    #[test]
    fn validate_finds_each_kind_of_problem() {
        use MeshProblemKind::*;
        let c = cube();
        let mut t = c.triangles().to_vec();
        t.pop();
        let open = mesh(c.vertices(), &t);
        let r = open.validate(0.0).unwrap();
        assert_eq!(r.summary.boundary_edges, 3);
        assert!(!r.summary.is_closed && r.summary.inward_shells == 0);
        let p = r.problems[0];
        let face = t[p.face.unwrap() as usize];
        assert_eq!(face[p.edge.unwrap() as usize], p.vertex.unwrap());

        let mut t = c.triangles().to_vec();
        t[0].swap(1, 2);
        let r = mesh(c.vertices(), &t).validate(0.0).unwrap();
        assert_eq!(r.summary.inconsistent_edges, 3);
        assert_eq!(r.summary.inward_shells, 0);

        let inward: Vec<[u32; 3]> = c.triangles().iter().map(|&[a, b, c]| [a, c, b]).collect();
        let m = mesh(c.vertices(), &inward);
        assert_eq!(kinds(&m, 0.0), vec![(InwardShell, Some(0), None)]);

        let mut v = c.vertices().to_vec();
        v.push([1.0 + 1e-7, 1.0, 1.0]);
        let mut t = c.triangles().to_vec();
        t.push(t[5]);
        t.push([0, 0, 1]);
        let m = mesh(&v, &t);
        assert_eq!(
            kinds(&m, 1e-6),
            vec![
                (DegenerateFace, Some(13), None),
                (DuplicateFace, Some(12), None),
                (DuplicateVertex, None, Some(8)),
                (NonManifoldEdge, Some(2), Some(4)),
                (NonManifoldEdge, Some(4), Some(1)),
                (NonManifoldEdge, Some(5), Some(1)),
            ]
        );
        assert!(kinds(&m, 0.0).iter().all(|k| k.0 != DuplicateVertex));
        assert!(m.validate(-1.0).is_err());
    }

    /// Two tetrahedra sharing only a vertex: each is a closed shell, the
    /// shared vertex has two fans.
    #[test]
    fn touching_tetrahedra_have_a_non_manifold_vertex() {
        let tet = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]];
        let mut v = vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
        v.extend([[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]);
        let second = tet.map(|t: [u32; 3]| t.map(|i| if i == 0 { 0 } else { i + 3 }));
        let second = second.map(|[a, b, c]| [a, c, b]);
        let m = mesh(&v, &[tet, second].concat());
        let r = m.validate(0.0).unwrap();
        assert_eq!(
            r.problems
                .iter()
                .map(|p| (p.kind, p.vertex))
                .collect::<Vec<_>>(),
            vec![(MeshProblemKind::NonManifoldVertex, Some(0))]
        );
        assert_eq!(r.summary.shells, 2);
        assert!(r.summary.is_closed);
    }

    #[test]
    fn cavity_wound_inward_is_not_a_problem() {
        let c = cube();
        let mut vertices: Vec<[f64; 3]> = c.vertices().iter().map(|p| p.map(|v| 3.0 * v)).collect();
        vertices.extend(c.vertices().iter().map(|p| p.map(|v| v + 1.0)));
        let outer = c.triangles().to_vec();
        let inner: Vec<[u32; 3]> = outer
            .iter()
            .map(|&[a, b, c]| [a + 8, c + 8, b + 8])
            .collect();
        let good = mesh(&vertices, &[outer.clone(), inner.clone()].concat());
        assert!(good.validate(0.0).unwrap().problems.is_empty());
        let flipped: Vec<[u32; 3]> = inner.iter().map(|&[a, b, c]| [a, c, b]).collect();
        let bad = mesh(&vertices, &[outer, flipped].concat());
        assert_eq!(
            kinds(&bad, 0.0),
            vec![(MeshProblemKind::InwardShell, Some(12), None)]
        );
    }

    /// Theory check: a cube missing one square side has the 4 edges of that
    /// square open; a flipped triangle gives its 3 edges inconsistent winding
    /// and repair winds it back.
    #[test]
    fn missing_side_opens_four_edges_and_flipped_face_repairs() {
        let c = cube();
        let r = mesh(c.vertices(), &c.triangles()[2..])
            .validate(0.0)
            .unwrap();
        assert_eq!((r.summary.boundary_edges, r.summary.is_closed), (4, false));
        let faces: ArrayRef = Arc::new(Float64Array::from_iter_values((2..12).map(f64::from)));
        let open = mesh(c.vertices(), &c.triangles()[2..])
            .with_face_column("face", faces)
            .unwrap();
        let filled = open.fill_holes().unwrap();
        assert_eq!(filled.triangles().len(), 12);
        assert!(filled.validate(0.0).unwrap().problems.is_empty());
        assert!((filled.volume().unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(filled.face_attributes().column(0).null_count(), 2);
        assert_eq!(filled.fill_holes().unwrap().triangles(), filled.triangles());
        let mut t = c.triangles().to_vec();
        t[7].swap(0, 1);
        let flipped = mesh(c.vertices(), &t);
        assert_eq!(flipped.validate(0.0).unwrap().summary.inconsistent_edges, 3);
        let fixed = flipped.repair(0.0).unwrap();
        assert!(fixed.validate(0.0).unwrap().problems.is_empty());
        assert_eq!(fixed.triangles()[7], [6, 7, 3]);
        assert!((fixed.volume().unwrap() - 1.0).abs() < 1e-12);
    }

    fn two_cubes(shift: [f64; 3]) -> Mesh {
        let c = cube();
        let mut v = c.vertices().to_vec();
        v.extend(
            c.vertices()
                .iter()
                .map(|p| [0, 1, 2].map(|k| p[k] + shift[k])),
        );
        let mut t = c.triangles().to_vec();
        t.extend(c.triangles().iter().map(|f| f.map(|i| i + 8)));
        mesh(&v, &t)
    }

    #[test]
    fn crossing_triangles_are_one_self_intersection() {
        let m = mesh(
            &[
                [0., 0., 0.],
                [2., 0., 0.],
                [0., 2., 0.],
                [0.5, 0.5, -1.],
                [0.5, 0.5, 1.],
                [3., 3., 0.5],
            ],
            &[[0, 1, 2], [3, 4, 5]],
        );
        let r = m.validate(0.0).unwrap();
        assert_eq!(r.summary.self_intersections, 1);
        assert_eq!(
            r.problems.last(),
            Some(&MeshProblem {
                kind: MeshProblemKind::SelfIntersection,
                face: Some(1),
                vertex: None,
                edge: None,
                other: Some(0),
            })
        );
        let touching = mesh(m.vertices(), &[[0, 1, 2], [0, 4, 5]]);
        assert!(
            touching
                .validate(0.0)
                .unwrap()
                .problems
                .iter()
                .all(|p| p.kind != MeshProblemKind::SelfIntersection)
        );
    }

    #[test]
    fn overlapping_cubes_cross_and_repair_keeps_the_crossing() {
        let m = two_cubes([0.5, 0.25, 0.25]);
        let s = m.validate(0.0).unwrap().summary;
        assert!(s.self_intersections > 0 && s.is_closed);
        let again = m.repair(0.0).unwrap().validate(0.0).unwrap().summary;
        assert_eq!(again.self_intersections, s.self_intersections);
        assert_eq!(
            two_cubes([2.0, 0.0, 0.0])
                .validate(0.0)
                .unwrap()
                .summary
                .self_intersections,
            0
        );
    }

    proptest::proptest! {
        /// A rotated, shifted cube stays clean; two cubes cross only while
        /// they overlap.
        #[test]
        fn crossings_follow_overlap(
            angles in proptest::array::uniform3(-3.0..3.0f64),
            offset in proptest::array::uniform3(-1e3..1e3f64),
            shift in 0.1..1.9f64,
        ) {
            let r = nalgebra::Rotation3::from_euler_angles(angles[0], angles[1], angles[2]);
            let pair = two_cubes([shift, 0.3, 0.2]);
            let moved: Vec<[f64; 3]> = pair
                .vertices()
                .iter()
                .map(|&p| (r * Vector3::from(p)).into())
                .map(|p: [f64; 3]| [0, 1, 2].map(|k| p[k] + offset[k]))
                .collect();
            let s = mesh(&moved, pair.triangles()).validate(0.0).unwrap().summary;
            proptest::prop_assert_eq!(s.self_intersections > 0, shift < 1.0);
            let one = mesh(&moved, &pair.triangles()[..12]).validate(0.0).unwrap();
            proptest::prop_assert!(one.problems.is_empty());
        }
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
