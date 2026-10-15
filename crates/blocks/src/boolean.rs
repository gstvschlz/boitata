//! Booleans of closed meshes, and plane cuts.
//!
//! The booleans corefine the two surfaces. Every pair of triangles whose
//! boxes meet is tested with exact orientation predicates (Shewchuk 1997,
//! "Adaptive precision floating-point arithmetic and fast robust geometric
//! predicates", through `robust`); a pair that crosses contributes the
//! segment where the two triangles' intervals on the line of their planes
//! overlap (Möller 1997, "A fast triangle-triangle intersection test").
//! Coplanar pairs exchange their edges clipped to each other. Points closer
//! than 1e-9 of the size of both meshes weld into one; where that leaves the
//! result open, as when faces lie within rounding of each other, a second
//! pass welds at 1e-6. Each triangle a
//! segment touches, or with points on its edges, is split by a constrained
//! Delaunay triangulation (Chew 1989, "Constrained Delaunay triangulations",
//! through `spade`) of its corners, those points and its segments, so both
//! surfaces share every edge along the curve where they meet. Each piece is
//! kept or dropped by whether its centroid lies inside the other mesh, by
//! winding number ([`SolidTester`]). A piece on a face of the other mesh
//! goes by whether the two faces point the same way: kept once where they
//! do and the operation keeps that boundary, dropped where they face each
//! other.
//!
//! A plane cut splits the triangles that cross the plane where their edges
//! cross it and closes a closed mesh with a cap triangulated the same way,
//! inside by even-odd counting of the cut's loops, so holes stay open.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, UInt32Array};
use arrow_schema::{Field, Schema};
use arrow_select::take::take;
use arrow_select::zip::zip;
use boitata_core::Mesh;
use rayon::prelude::*;
use robust::{Coord, Coord3D, orient2d, orient3d};
use spade::handles::FixedVertexHandle;
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};

use crate::error::{BlockModelError, Result};
use crate::{SolidTester, require_closed};

/// Which part of two solids a boolean keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boolean {
    /// Inside either.
    Merge,
    /// Inside both.
    Overlap,
    /// Inside the first and outside the second.
    Subtract,
}

type P = [f64; 3];
type Cdt = ConstrainedDelaunayTriangulation<Point2<f64>>;

fn sub(a: P, b: P) -> P {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: P, b: P) -> P {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: P, b: P) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn lerp(a: P, b: P, w: f64) -> P {
    [0, 1, 2].map(|k| a[k] + w * (b[k] - a[k]))
}

fn normal(t: [P; 3]) -> P {
    cross(sub(t[1], t[0]), sub(t[2], t[0]))
}

fn key(i: u32, j: u32) -> (u32, u32) {
    (i.min(j), i.max(j))
}

fn c3(p: P) -> Coord3D<f64> {
    Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

fn c2(p: [f64; 2]) -> Coord<f64> {
    Coord { x: p[0], y: p[1] }
}

/// Axis of the largest normal component; dropping it projects a triangle
/// without folding it.
fn axis(n: P) -> usize {
    (0..3)
        .max_by(|&i, &j| n[i].abs().total_cmp(&n[j].abs()))
        .expect("three axes")
}

/// The two other coordinates in cyclic order, so a triangle keeps its turn
/// where the normal's `k` component is positive.
fn flat(p: P, k: usize) -> [f64; 2] {
    [p[(k + 1) % 3], p[(k + 2) % 3]]
}

/// Points welded within `tol`, by grid hashing; each new point takes the
/// earliest one in reach.
struct Pool {
    points: Vec<P>,
    cells: HashMap<[i64; 3], Vec<u32>>,
    tol: f64,
}

impl Pool {
    fn add(&mut self, p: P) -> u32 {
        let k = p.map(|v| (v / self.tol).floor() as i64);
        let mut found: Option<u32> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let cell = [k[0] + dx, k[1] + dy, k[2] + dz];
                    for &j in self.cells.get(&cell).into_iter().flatten() {
                        let d = sub(self.points[j as usize], p);
                        if dot(d, d) <= self.tol * self.tol {
                            found = Some(found.map_or(j, |f| f.min(j)));
                        }
                    }
                }
            }
        }
        found.unwrap_or_else(|| {
            let i = self.points.len() as u32;
            self.points.push(p);
            self.cells.entry(k).or_default().push(i);
            i
        })
    }

    fn at(&self, i: u32) -> P {
        self.points[i as usize]
    }
}

enum Contact {
    Segment(P, P),
    Coplanar,
}

fn one_side(s: &[f64; 3]) -> bool {
    s.iter().all(|&v| v > 0.0) || s.iter().all(|&v| v < 0.0)
}

/// Where triangles `t` and `u` meet. The sides of each triangle's corners
/// against the other's plane are exact; the segment's ends are rounded.
fn contact(t: [P; 3], u: [P; 3]) -> Option<Contact> {
    let side = |plane: [P; 3], p: P| orient3d(c3(plane[0]), c3(plane[1]), c3(plane[2]), c3(p));
    let su = u.map(|q| side(t, q));
    if one_side(&su) {
        return None;
    }
    if su.iter().all(|&s| s == 0.0) {
        return Some(Contact::Coplanar);
    }
    let st = t.map(|p| side(u, p));
    if one_side(&st) {
        return None;
    }
    let dir = cross(normal(t), normal(u));
    let along = |p: P| dot(p, dir);
    let span = |tri: [P; 3], s: [f64; 3]| {
        let mut ends = vec![];
        for i in 0..3 {
            let j = (i + 1) % 3;
            if s[i] == 0.0 {
                ends.push(tri[i]);
            } else if s[j] != 0.0 && (s[i] > 0.0) != (s[j] > 0.0) {
                ends.push(lerp(tri[i], tri[j], (s[i] / (s[i] - s[j])).clamp(0.0, 1.0)));
            }
        }
        let (p, q) = (ends[0], ends[ends.len() - 1]);
        if along(p) <= along(q) { (p, q) } else { (q, p) }
    };
    let ((t0, t1), (u0, u1)) = (span(t, st), span(u, su));
    let start = if along(t0) >= along(u0) { t0 } else { u0 };
    let end = if along(t1) <= along(u1) { t1 } else { u1 };
    (along(start) <= along(end)).then_some(Contact::Segment(start, end))
}

/// The part of segment `a`-`b` inside triangle `t`, which shares its plane.
fn clip(a: P, b: P, t: [P; 3]) -> Option<(P, P)> {
    let k = axis(normal(t));
    let t2 = t.map(|p| c2(flat(p, k)));
    let turn = orient2d(t2[0], t2[1], t2[2]).signum();
    let (a2, b2) = (c2(flat(a, k)), c2(flat(b, k)));
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for i in 0..3 {
        let (p, q) = (t2[i], t2[(i + 1) % 3]);
        let (da, db) = (orient2d(p, q, a2) * turn, orient2d(p, q, b2) * turn);
        if da < 0.0 && db < 0.0 {
            return None;
        }
        if da < 0.0 {
            lo = lo.max(da / (da - db));
        } else if db < 0.0 {
            hi = hi.min(da / (da - db));
        }
    }
    (lo < hi).then(|| (lerp(a, b, lo), lerp(a, b, hi)))
}

/// Whether `p` lies strictly inside triangle `t` seen along its normal.
fn within(p: P, t: [P; 3]) -> bool {
    let k = axis(normal(t));
    let t2 = t.map(|q| c2(flat(q, k)));
    let turn = orient2d(t2[0], t2[1], t2[2]).signum();
    let p2 = c2(flat(p, k));
    (0..3).all(|i| orient2d(t2[i], t2[(i + 1) % 3], p2) * turn > 0.0)
}

/// Distance from `p` to the inside of edge `a`-`b`, when it projects there.
fn edge_distance(p: P, a: P, b: P) -> Option<f64> {
    let ab = sub(b, a);
    let w = dot(sub(p, a), ab) / dot(ab, ab);
    let d = sub(p, lerp(a, b, w));
    (w > 0.0 && w < 1.0).then(|| dot(d, d).sqrt())
}

/// One mesh's triangles as welded points, with the segments and coplanar
/// faces of the other mesh found on each.
struct Side {
    ids: Vec<u32>,
    faces: Vec<[u32; 3]>,
    segments: Vec<Vec<(u32, u32)>>,
    coplanar: Vec<Vec<u32>>,
}

impl Side {
    fn new(mesh: &Mesh, pool: &mut Pool) -> Self {
        let ids: Vec<u32> = mesh.vertices().iter().map(|&p| pool.add(p)).collect();
        let faces: Vec<[u32; 3]> = mesh
            .triangles()
            .iter()
            .map(|t| t.map(|i| ids[i as usize]))
            .collect();
        let n = faces.len();
        Self {
            ids,
            faces,
            segments: vec![vec![]; n],
            coplanar: vec![vec![]; n],
        }
    }

    fn corners(&self, pool: &Pool, f: usize) -> [P; 3] {
        self.faces[f].map(|i| pool.at(i))
    }
}

/// Pairs `(a, b)` of triangles whose boxes, grown by `tol`, meet; candidates
/// come from a grid over the triangles of `b`.
fn candidates(a: &[[P; 3]], b: &[[P; 3]], tol: f64) -> Vec<(usize, usize)> {
    let boxed = |t: &[P; 3]| -> (P, P) {
        let lo = [0, 1, 2].map(|k| t[0][k].min(t[1][k]).min(t[2][k]) - tol);
        let hi = [0, 1, 2].map(|k| t[0][k].max(t[1][k]).max(t[2][k]) + tol);
        (lo, hi)
    };
    let (ba, bb): (Vec<_>, Vec<_>) = (a.iter().map(boxed).collect(), b.iter().map(boxed).collect());
    let (lo, hi) = bb
        .iter()
        .fold(([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]), |(l, h), b| {
            (
                [0, 1, 2].map(|k| l[k].min(b.0[k])),
                [0, 1, 2].map(|k| h[k].max(b.1[k])),
            )
        });
    let diag = dot(sub(hi, lo), sub(hi, lo)).sqrt();
    let mean = bb
        .iter()
        .map(|b| (0..3).map(|k| b.1[k] - b.0[k]).fold(0.0, f64::max))
        .sum::<f64>()
        / bb.len().max(1) as f64;
    let h = mean.max(diag / 100.0).max(tol);
    let cell = |p: P| p.map(|v| (v / h).floor() as i64);
    let span = |b: &(P, P)| {
        let (l, u) = (
            cell([0, 1, 2].map(|k| b.0[k].max(lo[k]))),
            cell([0, 1, 2].map(|k| b.1[k].min(hi[k]))),
        );
        (l[0]..=u[0]).flat_map(move |x| {
            (l[1]..=u[1]).flat_map(move |y| (l[2]..=u[2]).map(move |z| [x, y, z]))
        })
    };
    let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for (g, b) in bb.iter().enumerate() {
        for c in span(b) {
            grid.entry(c).or_default().push(g);
        }
    }
    let overlap = |p: &(P, P), q: &(P, P)| (0..3).all(|k| p.0[k] <= q.1[k] && q.0[k] <= p.1[k]);
    let mut out = vec![];
    for (f, b) in ba.iter().enumerate() {
        if !overlap(b, &(lo, hi)) {
            continue;
        }
        let mut near: Vec<usize> = span(b)
            .flat_map(|c| grid.get(&c).into_iter().flatten().copied())
            .collect();
        near.sort_unstable();
        near.dedup();
        out.extend(
            near.into_iter()
                .filter(|&g| overlap(b, &bb[g]))
                .map(|g| (f, g)),
        );
    }
    out
}

/// Triangulates the plane through `o` spanned by `frame`'s first two axes
/// with `edges` as constraints and `inner` as extra points. Keeps what the
/// constraints enclose, or with `parity` what an odd number of them
/// enclose. Pieces wind counter-clockwise seen along `frame`'s normal.
/// Points the triangulation adds where constraints cross join the pool.
fn triangulate(
    o: P,
    frame: [P; 3],
    edges: &[(u32, u32)],
    inner: &[u32],
    pool: &mut Pool,
    parity: bool,
) -> Result<Vec<[u32; 3]>> {
    let [u, v, _] = frame;
    let flat2 = |p: P| Point2::new(dot(sub(p, o), u), dot(sub(p, o), v));
    let mut cdt = Cdt::new();
    let mut ids: Vec<u32> = vec![];
    let mut handle: HashMap<u32, FixedVertexHandle> = HashMap::new();
    let points = edges
        .iter()
        .flat_map(|&(p, q)| [p, q])
        .chain(inner.iter().copied());
    for p in points {
        if handle.contains_key(&p) {
            continue;
        }
        let h = cdt
            .insert(flat2(pool.at(p)))
            .map_err(|e| BlockModelError::InvalidMesh(format!("triangulation failed: {e:?}")))?;
        if h.index() == ids.len() {
            ids.push(p);
        }
        handle.insert(p, h);
    }
    for &(p, q) in edges {
        if handle[&p] != handle[&q] {
            cdt.add_constraint_and_split(handle[&p], handle[&q], |x| x);
        }
    }
    for i in ids.len()..cdt.num_vertices() {
        let q = cdt.vertex(FixedVertexHandle::from_index(i)).position();
        ids.push(pool.add([0, 1, 2].map(|k| o[k] + q.x * u[k] + q.y * v[k])));
    }
    let mut side: Vec<Option<bool>> = vec![None; cdt.num_all_faces()];
    let mut stack = vec![];
    for f in cdt.inner_faces() {
        for e in f.adjacent_edges() {
            let crossed = cdt.is_constraint_edge(e.fix().as_undirected());
            if e.rev().face().is_outer() && (parity || !crossed) {
                stack.push((f.fix(), crossed));
            }
        }
    }
    while let Some((f, inside)) = stack.pop() {
        if side[f.index()].is_some() {
            continue;
        }
        side[f.index()] = Some(inside);
        let face = cdt.face(f);
        for e in face.adjacent_edges() {
            let Some(g) = e.rev().face().as_inner() else {
                continue;
            };
            let crossed = cdt.is_constraint_edge(e.fix().as_undirected());
            if side[g.fix().index()].is_none() && (parity || !crossed) {
                stack.push((g.fix(), inside != crossed));
            }
        }
    }
    Ok(cdt
        .inner_faces()
        .filter(|f| match side[f.fix().index()] {
            None => !parity,
            Some(inside) => parity && inside,
        })
        .map(|f| f.vertices().map(|v| ids[v.fix().index()]))
        .filter(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2])
        .collect())
}

/// Orthonormal `[u, v, n]` with `u × v = n`.
fn frame(n: P) -> [P; 3] {
    let n = n.map(|x| x / dot(n, n).sqrt());
    let k = axis(n);
    let mut seed = [0.0; 3];
    seed[(k + 1) % 3] = 1.0;
    let u = cross(seed, n);
    let u = u.map(|x| x / dot(u, u).sqrt());
    [u, cross(n, u), n]
}

/// Each face as itself, or split along its segments and edge points; with
/// the face it came from.
fn refine(side: &Side, pool: &mut Pool) -> Result<Vec<(u32, [u32; 3])>> {
    let mut on_edges: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    let mut inner: Vec<Vec<u32>> = vec![vec![]; side.faces.len()];
    for (f, &t) in side.faces.iter().enumerate() {
        let c = side.corners(pool, f);
        for &p in side.segments[f].iter().flat_map(|(p, q)| [p, q]) {
            if t.contains(&p) {
                continue;
            }
            let near = (0..3)
                .filter_map(|e| Some((edge_distance(pool.at(p), c[e], c[(e + 1) % 3])?, e)))
                .filter(|&(d, _)| d <= pool.tol)
                .min_by(|a, b| a.0.total_cmp(&b.0));
            match near {
                Some((_, e)) => on_edges
                    .entry(key(t[e], t[(e + 1) % 3]))
                    .or_default()
                    .push(p),
                None => inner[f].push(p),
            }
        }
    }
    for list in on_edges.values_mut().chain(&mut inner) {
        list.sort_unstable();
        list.dedup();
    }
    let mut out = vec![];
    for (f, &t) in side.faces.iter().enumerate() {
        if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
            continue;
        }
        let edge = |e: usize| on_edges.get(&key(t[e], t[(e + 1) % 3]));
        if side.segments[f].is_empty() && (0..3).all(|e| edge(e).is_none()) {
            out.push((f as u32, t));
            continue;
        }
        let mut corners = vec![];
        for e in 0..3 {
            let (a, b) = (pool.at(t[e]), pool.at(t[(e + 1) % 3]));
            let mut along: Vec<u32> = edge(e).cloned().unwrap_or_default();
            along.sort_by(|&p, &q| {
                let w = |i: u32| dot(sub(pool.at(i), a), sub(b, a));
                w(p).total_cmp(&w(q))
            });
            corners.push(t[e]);
            corners.extend(along);
        }
        let mut edges: Vec<(u32, u32)> = (0..corners.len())
            .map(|i| (corners[i], corners[(i + 1) % corners.len()]))
            .collect();
        edges.extend(&side.segments[f]);
        let c = side.corners(pool, f);
        let pieces = triangulate(c[0], frame(normal(c)), &edges, &inner[f], pool, false)?;
        out.extend(pieces.into_iter().map(|piece| (f as u32, piece)));
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Place {
    Inside,
    Outside,
    /// On a face of the other mesh pointing the same way.
    Same,
    /// On a face of the other mesh pointing the other way.
    Opposite,
}

fn places(
    pieces: &[(u32, [u32; 3])],
    side: &Side,
    other: &Side,
    pool: &Pool,
    tester: &SolidTester,
) -> Vec<Place> {
    pieces
        .par_iter()
        .map(|&(f, t)| {
            let c = t.map(|i| pool.at(i));
            let centroid = [0, 1, 2].map(|k| (c[0][k] + c[1][k] + c[2][k]) / 3.0);
            let own = normal(side.corners(pool, f as usize));
            for &g in &side.coplanar[f as usize] {
                let u = other.corners(pool, g as usize);
                if within(centroid, u) {
                    return if dot(own, normal(u)) > 0.0 {
                        Place::Same
                    } else {
                        Place::Opposite
                    };
                }
            }
            if tester.contains(centroid) {
                Place::Inside
            } else {
                Place::Outside
            }
        })
        .collect()
}

/// One table from rows of `tables`: row `i` of the result is row `r` of
/// table `s` for `rows[i] = Some((s, r))`, else null. Columns are the union
/// by name, in order of first appearance.
fn combine(tables: &[&RecordBatch], rows: &[Option<(usize, u32)>]) -> Result<RecordBatch> {
    let mut names: Vec<String> = vec![];
    for t in tables {
        for f in t.schema().fields() {
            if !names.contains(f.name()) {
                names.push(f.name().clone());
            }
        }
    }
    let arrow = |e| BlockModelError::Core(boitata_core::Error::Arrow(e));
    let (mut fields, mut columns) = (vec![], vec![]);
    for name in &names {
        let mut found: Option<(Field, ArrayRef)> = None;
        for (s, t) in tables.iter().enumerate() {
            let Ok(i) = t.schema().index_of(name) else {
                continue;
            };
            let picks: UInt32Array = rows
                .iter()
                .map(|r| r.filter(|r| r.0 == s).map(|r| r.1))
                .collect();
            let column = take(t.column(i), &picks, None).map_err(arrow)?;
            let field = t.schema().field(i).clone().with_nullable(true);
            found = Some(match found {
                None => (field, column),
                Some((first, _)) if first.data_type() != field.data_type() => {
                    return Err(BlockModelError::InvalidMesh(format!(
                        "column `{name}` is {} in one mesh and {} in the other",
                        first.data_type(),
                        field.data_type()
                    )));
                }
                Some((first, before)) => {
                    let mask: BooleanArray = rows
                        .iter()
                        .map(|r| Some(r.is_some_and(|r| r.0 == s)))
                        .collect();
                    (first, zip(&mask, &column, &before).map_err(arrow)?)
                }
            });
        }
        let (field, column) = found.expect("named by a table");
        fields.push(field);
        columns.push(column);
    }
    let options = RecordBatchOptions::new().with_row_count(Some(rows.len()));
    RecordBatch::try_new_with_options(Arc::new(Schema::new(fields)), columns, &options)
        .map_err(arrow)
}

/// The mesh of `triangles` over the `points` they use, with attributes
/// picked from `sources` by `vertex_rows` (per point) and `face_rows`.
fn assemble(
    points: &[P],
    triangles: Vec<[u32; 3]>,
    sources: &[&Mesh],
    vertex_rows: &[Option<(usize, u32)>],
    face_rows: &[Option<(usize, u32)>],
) -> Result<Mesh> {
    let mut used: Vec<u32> = triangles.iter().flatten().copied().collect();
    used.sort_unstable();
    used.dedup();
    let mut index = vec![0u32; points.len()];
    for (n, &i) in used.iter().enumerate() {
        index[i as usize] = n as u32;
    }
    let triangles = triangles
        .into_iter()
        .map(|t| t.map(|i| index[i as usize]))
        .collect();
    let mesh = Mesh::new(
        used.iter().map(|&i| points[i as usize]).collect(),
        triangles,
    )?;
    let vertex_rows: Vec<_> = used
        .iter()
        .map(|&i| vertex_rows.get(i as usize).copied().flatten())
        .collect();
    let vertices: Vec<&RecordBatch> = sources.iter().map(|m| m.vertex_attributes()).collect();
    let faces: Vec<&RecordBatch> = sources.iter().map(|m| m.face_attributes()).collect();
    Ok(mesh.with_attributes(
        combine(&vertices, &vertex_rows)?,
        combine(&faces, face_rows)?,
    )?)
}

fn check_closed(mesh: &Mesh, what: &str) -> Result<()> {
    if mesh.triangles().is_empty() || mesh.is_closed() {
        return Ok(());
    }
    let s = mesh.validate(0.0)?.summary;
    Err(BlockModelError::InvalidMesh(if s.boundary_edges == 0 {
        format!("the {what} meets itself along an edge, so it is not a solid")
    } else {
        format!(
            "the {what} came out open, which happens when an input crosses itself or the meshes \
             meet within rounding of each other; check them with validate"
        )
    }))
}

/// The part of closed meshes `a` and `b` that `op` keeps, as one closed
/// mesh wound outward like the inputs. Faces keep their attributes, the
/// union of both meshes' columns with nulls where a mesh lacks one; so do
/// the vertices the inputs had, while vertices made where the meshes cross
/// get nulls. Faces of `b` kept by [`Boolean::Subtract`] are turned over.
/// The meshes must agree on CRS and length unit where both declare one;
/// the result takes the one declared.
pub fn boolean(a: &Mesh, b: &Mesh, op: Boolean) -> Result<Mesh> {
    require_closed(a)?;
    require_closed(b)?;
    boitata_core::crs::same_crs(
        ("the first mesh", a.crs.as_deref()),
        ("the second", b.crs.as_deref()),
    )?;
    boitata_core::units::same_length_unit(
        ("the first mesh", a.length_unit.as_deref()),
        ("the second", b.length_unit.as_deref()),
    )?;
    let (lo, hi) = [a, b]
        .iter()
        .filter_map(|m| m.bounds())
        .reduce(|(l, h), (m, n)| {
            (
                [0, 1, 2].map(|k| l[k].min(m[k])),
                [0, 1, 2].map(|k| h[k].max(n[k])),
            )
        })
        .expect("closed meshes have vertices");
    let size = dot(sub(hi, lo), sub(hi, lo)).sqrt();
    let reach = lo.iter().chain(&hi).fold(0.0f64, |m, v| m.max(v.abs()));
    let tol = (1e-9 * size).max(64.0 * f64::EPSILON * reach);
    // Faces within a few rounding errors of each other defeat the first
    // weld; snapping them together at a coarser one settles most.
    corefine(a, b, op, tol).or_else(|e| corefine(a, b, op, 1e3 * tol).map_err(|_| e))
}

fn corefine(a: &Mesh, b: &Mesh, op: Boolean, tol: f64) -> Result<Mesh> {
    let mut pool = Pool {
        points: vec![],
        cells: HashMap::new(),
        tol,
    };
    let mut first = Side::new(a, &mut pool);
    let mut second = Side::new(b, &mut pool);
    let mut vertex_rows: Vec<Option<(usize, u32)>> = vec![None; pool.points.len()];
    for (s, side) in [&first, &second].into_iter().enumerate() {
        for (v, &id) in side.ids.iter().enumerate() {
            vertex_rows[id as usize].get_or_insert((s, v as u32));
        }
    }

    let corners = |side: &Side, pool: &Pool| -> Vec<[P; 3]> {
        (0..side.faces.len())
            .map(|f| side.corners(pool, f))
            .collect()
    };
    let (ca, cb) = (corners(&first, &pool), corners(&second, &pool));
    let degenerate = |t: [P; 3]| normal(t) == [0.0; 3];
    for (f, g) in candidates(&ca, &cb, tol) {
        let (t, u) = (ca[f], cb[g]);
        if degenerate(t) || degenerate(u) {
            continue;
        }
        let mut found = vec![];
        match contact(t, u) {
            None => continue,
            Some(Contact::Segment(p, q)) => found.push((p, q)),
            Some(Contact::Coplanar) => {
                first.coplanar[f].push(g as u32);
                second.coplanar[g].push(f as u32);
                for i in 0..3 {
                    found.extend(clip(u[i], u[(i + 1) % 3], t));
                    found.extend(clip(t[i], t[(i + 1) % 3], u));
                }
            }
        }
        for (p, q) in found {
            let (p, q) = (pool.add(p), pool.add(q));
            if p != q {
                first.segments[f].push((p, q));
                second.segments[g].push((p, q));
            }
        }
    }

    let pieces_a = refine(&first, &mut pool)?;
    let pieces_b = refine(&second, &mut pool)?;
    let (ta, tb) = (SolidTester::new(a)?, SolidTester::new(b)?);
    let places_a = places(&pieces_a, &first, &second, &pool, &tb);
    let places_b = places(&pieces_b, &second, &first, &pool, &ta);
    use Place::*;
    let keep_a = |p: Place| match op {
        Boolean::Merge => matches!(p, Outside | Same),
        Boolean::Overlap => matches!(p, Inside | Same),
        Boolean::Subtract => matches!(p, Outside | Opposite),
    };
    let keep_b = |p: Place| match op {
        Boolean::Merge => p == Outside,
        Boolean::Overlap | Boolean::Subtract => p == Inside,
    };
    let (mut triangles, mut face_rows) = (vec![], vec![]);
    for (&(f, t), &p) in pieces_a.iter().zip(&places_a) {
        if keep_a(p) {
            triangles.push(t);
            face_rows.push(Some((0, f)));
        }
    }
    for (&(g, [i, j, k]), &p) in pieces_b.iter().zip(&places_b) {
        if keep_b(p) {
            triangles.push(if op == Boolean::Subtract {
                [i, k, j]
            } else {
                [i, j, k]
            });
            face_rows.push(Some((1, g)));
        }
    }
    let mut mesh = assemble(&pool.points, triangles, &[a, b], &vertex_rows, &face_rows)?;
    mesh.crs = a.crs.clone().or_else(|| b.crs.clone());
    mesh.length_unit = a.length_unit.clone().or_else(|| b.length_unit.clone());
    check_closed(&mesh, "result")?;
    Ok(mesh)
}

/// The part of `mesh` on the side of the plane through `origin` that
/// `normal` points away from, so `normal` is the outward normal of the cut
/// face. A vertex on the plane counts as cut away, as in
/// [`Mesh::section`]. A closed mesh gets a cap over the cut, so it stays
/// closed; an open surface is just trimmed. Kept faces and the vertices the
/// mesh had keep their attributes; cap faces and vertices made on the plane
/// get nulls.
pub fn cut(mesh: &Mesh, origin: P, normal: P) -> Result<Mesh> {
    let length = dot(normal, normal).sqrt();
    if !(length.is_finite() && length > 0.0) || origin.iter().any(|v| !v.is_finite()) {
        return Err(BlockModelError::InvalidMesh(
            "origin must be finite and normal finite and non-zero".into(),
        ));
    }
    let n = normal.map(|v| v / length);
    let d: Vec<f64> = mesh
        .vertices()
        .iter()
        .map(|&p| dot(sub(p, origin), n))
        .collect();
    let kept = |v: u32| d[v as usize] < 0.0;
    let mut points = mesh.vertices().to_vec();
    let mut crossings: HashMap<(u32, u32), u32> = HashMap::new();
    let mut cross_at = |i: u32, j: u32| -> u32 {
        if d[j as usize] == 0.0 {
            return j;
        }
        *crossings.entry(key(i, j)).or_insert_with(|| {
            let (di, dj) = (d[i as usize], d[j as usize]);
            let (p, q) = (mesh.vertices()[i as usize], mesh.vertices()[j as usize]);
            points.push(lerp(p, q, di / (di - dj)));
            points.len() as u32 - 1
        })
    };
    let (mut triangles, mut face_rows) = (vec![], vec![]);
    for (f, &t) in mesh.triangles().iter().enumerate() {
        let mut polygon = vec![];
        for e in 0..3 {
            let (i, j) = (t[e], t[(e + 1) % 3]);
            if kept(i) {
                polygon.push(i);
            }
            match (kept(i), kept(j)) {
                (true, false) => polygon.push(cross_at(i, j)),
                (false, true) => polygon.push(cross_at(j, i)),
                _ => {}
            }
        }
        polygon.dedup();
        if polygon.len() > 1 && polygon[0] == polygon[polygon.len() - 1] {
            polygon.pop();
        }
        for w in 1..polygon.len().saturating_sub(1) {
            let piece = [polygon[0], polygon[w], polygon[w + 1]];
            if piece[0] != piece[1] && piece[1] != piece[2] && piece[0] != piece[2] {
                triangles.push(piece);
                face_rows.push(Some((0, f as u32)));
            }
        }
    }
    let original = mesh.vertices().len();
    if mesh.is_closed() {
        let mut directed: HashMap<(u32, u32), usize> = HashMap::new();
        for t in &triangles {
            for e in 0..3 {
                *directed.entry((t[e], t[(e + 1) % 3])).or_default() += 1;
            }
        }
        let mut open: Vec<(u32, u32)> = directed
            .keys()
            .filter(|&&(i, j)| !directed.contains_key(&(j, i)))
            .copied()
            .collect();
        open.sort_unstable();
        if !open.is_empty() {
            let size = mesh
                .bounds()
                .map_or(1.0, |(lo, hi)| dot(sub(hi, lo), sub(hi, lo)).sqrt());
            let mut pool = Pool {
                points,
                cells: HashMap::new(),
                tol: 1e-9 * size,
            };
            let cap = triangulate(origin, frame(n), &open, &[], &mut pool, true)?;
            let along = cap
                .iter()
                .flat_map(|t| (0..3).map(move |e| (t[e], t[(e + 1) % 3])))
                .filter(|e| open.binary_search(e).is_ok())
                .count();
            for &[i, j, k] in &cap {
                triangles.push(if 2 * along > open.len() {
                    [i, k, j]
                } else {
                    [i, j, k]
                });
                face_rows.push(None);
            }
            points = pool.points;
        }
    }
    let vertex_rows: Vec<_> = (0..points.len())
        .map(|i| (i < original).then_some((0, i as u32)))
        .collect();
    let mut out = assemble(&points, triangles, &[mesh], &vertex_rows, &face_rows)?;
    out.crs = mesh.crs.clone();
    out.length_unit = mesh.length_unit.clone();
    if mesh.is_closed() {
        check_closed(&out, "cut")?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solid::tests::cube;
    use crate::subblock::tests::sphere;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Float64Type;
    use arrow_array::{Array, Float64Array, StringArray};
    use boitata_core::rng::splitmix;
    use nalgebra::{Rotation3, Vector3};

    fn moved(mesh: &Mesh, rotation: Rotation3<f64>, shift: P) -> Mesh {
        let v = mesh
            .vertices()
            .iter()
            .map(|&p| {
                let q = rotation * Vector3::from(p);
                [q.x + shift[0], q.y + shift[1], q.z + shift[2]]
            })
            .collect();
        Mesh::new(v, mesh.triangles().to_vec()).unwrap()
    }

    fn volume(mesh: &Mesh) -> f64 {
        if mesh.triangles().is_empty() {
            0.0
        } else {
            mesh.volume().unwrap()
        }
    }

    /// Closed, manifold and consistently wound outward, or empty.
    fn clean(mesh: &Mesh) {
        if mesh.triangles().is_empty() {
            return;
        }
        let s = mesh.validate(0.0).unwrap().summary;
        assert!(s.is_closed && s.boundary_edges == 0, "{s:?}");
        assert_eq!(
            (s.non_manifold_edges, s.inconsistent_edges, s.inward_shells),
            (0, 0, 0),
            "{s:?}"
        );
        assert!(mesh.volume().unwrap() > 0.0);
    }

    /// The three booleans of `a` and `b`, checked clean and against
    /// inclusion-exclusion; returns their volumes.
    fn all(a: &Mesh, b: &Mesh) -> [f64; 3] {
        let [m, o, s] = [Boolean::Merge, Boolean::Overlap, Boolean::Subtract]
            .map(|op| boolean(a, b, op).unwrap_or_else(|e| panic!("{op:?}: {e}")));
        [&m, &o, &s].iter().for_each(|m| clean(m));
        let [vm, vo, vs] = [&m, &o, &s].map(volume);
        let (va, vb) = (volume(a), volume(b));
        let scale = va.max(vb);
        assert!(
            (vm - (va + vb - vo)).abs() < 1e-9 * scale,
            "{vm} {va} {vb} {vo}"
        );
        assert!((vs - (va - vo)).abs() < 1e-9 * scale, "{vs} {va} {vo}");
        [vm, vo, vs]
    }

    /// Theory check: on seeded random boxes and spheres, placed and turned
    /// to overlap, every boolean is a clean closed solid and
    /// vol(A ∪ B) = vol(A) + vol(B) - vol(A ∩ B), vol(A - B) = vol(A) - vol(A ∩ B).
    #[test]
    fn booleans_add_up_on_random_solids() {
        for seed in 0..24u64 {
            let mut k = 0;
            let mut u = || {
                k += 1;
                splitmix(seed * 1000 + k) as f64 / u64::MAX as f64
            };
            let solid = |u: &mut dyn FnMut() -> f64| {
                let base = if u() < 0.5 {
                    cube(-1.0, 1.0)
                } else {
                    sphere(1.2)
                };
                let r = Rotation3::from_euler_angles(6.0 * u(), 6.0 * u(), 6.0 * u());
                moved(&base, r, [u() - 0.5, u() - 0.5, u() - 0.5])
            };
            let (a, b) = (solid(&mut u), solid(&mut u));
            let [_, overlap, _] = all(&a, &b);
            assert!(overlap > 0.0, "seed {seed}");
        }
    }

    /// Boxes on a half-unit lattice, so faces are often coplanar, edges
    /// collinear and corners shared.
    #[test]
    fn booleans_add_up_on_lattice_boxes() {
        for seed in 0..40u64 {
            let mut k = 0;
            let mut u = || {
                k += 1;
                (splitmix(seed * 1000 + k) % 5) as f64 * 0.5
            };
            let mut corner = || {
                let lo = [u(), u(), u()];
                let hi = [0, 1, 2].map(|i| lo[i] + 0.5 + u());
                (lo, hi)
            };
            let boxed = |(lo, hi): (P, P)| {
                let c = cube(0.0, 1.0);
                let v = c
                    .vertices()
                    .iter()
                    .map(|p| [0, 1, 2].map(|i| lo[i] + p[i] * (hi[i] - lo[i])))
                    .collect();
                Mesh::new(v, c.triangles().to_vec()).unwrap()
            };
            let (a, b) = (boxed(corner()), boxed(corner()));
            match boolean(&a, &b, Boolean::Merge) {
                Err(e) if e.to_string().contains("along an edge") => {
                    let rest = boolean(&a, &b, Boolean::Subtract).unwrap();
                    assert!((volume(&rest) - volume(&a)).abs() < 1e-12, "seed {seed}");
                }
                _ => {
                    all(&a, &b);
                }
            }
        }
    }

    /// A solid against itself turned and moved by amounts from well above
    /// to well below rounding: every boolean still closes and adds up.
    #[test]
    fn nearly_identical_solids() {
        let s = sphere(1.0);
        for d in [1e-3, 1e-5, 3e-7, 1e-7, 2e-8, 1e-8, 1e-10, 1e-14, 0.0] {
            let t = moved(&s, Rotation3::from_euler_angles(d, 0.0, 0.0), [d, 0.0, 0.0]);
            all(&s, &t);
        }
    }

    #[test]
    fn far_meshes_and_nested_meshes() {
        let (a, far) = (cube(0.0, 1.0), cube(5.0, 6.0));
        assert_eq!(all(&a, &far), [2.0, 0.0, 1.0]);
        let inner = cube(0.25, 0.75);
        let [m, o, s] = all(&a, &inner);
        assert!((m - 1.0).abs() < 1e-12 && (o - 0.125).abs() < 1e-12 && (s - 0.875).abs() < 1e-12);
        let hollow = boolean(&a, &inner, Boolean::Subtract).unwrap();
        assert_eq!(hollow.validate(0.0).unwrap().summary.shells, 2);
    }

    /// Faces in one plane: cubes sharing a face, overlapping along shared
    /// planes, and a cube with itself.
    #[test]
    fn coplanar_and_touching_faces() {
        let a = cube(0.0, 1.0);
        let next = moved(&a, Rotation3::identity(), [1.0, 0.0, 0.0]);
        let [m, o, s] = all(&a, &next);
        assert!((m - 2.0).abs() < 1e-12 && o == 0.0 && (s - 1.0).abs() < 1e-12);
        let merged = boolean(&a, &next, Boolean::Merge).unwrap();
        assert_eq!(merged.validate(0.0).unwrap().summary.shells, 1);

        let half = moved(&a, Rotation3::identity(), [0.5, 0.0, 0.0]);
        let [m, o, s] = all(&a, &half);
        assert!((m - 1.5).abs() < 1e-12 && (o - 0.5).abs() < 1e-12 && (s - 0.5).abs() < 1e-12);

        let corner = moved(&a, Rotation3::identity(), [0.5, 0.5, 0.5]);
        let [m, o, _] = all(&a, &corner);
        assert!((m - 1.875).abs() < 1e-12 && (o - 0.125).abs() < 1e-12);

        let [m, o, s] = all(&a, &a);
        assert!((m - 1.0).abs() < 1e-12 && (o - 1.0).abs() < 1e-12 && s == 0.0);
        let turned = moved(&a, Rotation3::from_euler_angles(0.0, 0.0, 0.3), [0.0; 3]);
        all(&a, &turned);
        let edge = moved(&a, Rotation3::identity(), [1.0, 1.0, 0.0]);
        let error = boolean(&a, &edge, Boolean::Merge).unwrap_err().to_string();
        assert!(error.contains("along an edge"), "{error}");
        assert!(
            boolean(&a, &edge, Boolean::Overlap)
                .unwrap()
                .triangles()
                .is_empty()
        );
        let rest = boolean(&a, &edge, Boolean::Subtract).unwrap();
        assert!((rest.volume().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn attributes_frames_and_open_input() {
        let column = |name: &str, v: f64| -> (String, ArrayRef) {
            (name.into(), Arc::new(Float64Array::from(vec![v; 12])))
        };
        let (n, x) = column("a", 1.0);
        let mut a = cube(0.0, 1.0).with_face_column(&n, x).unwrap();
        let rock: ArrayRef = Arc::new(StringArray::from(vec!["ore"; 8]));
        a = a.with_vertex_column("rock", rock).unwrap();
        a.crs = Some("EPSG:31982".into());
        let (n, x) = column("b", 2.0);
        let b = moved(&cube(0.0, 1.0), Rotation3::identity(), [0.5, 0.5, 0.5])
            .with_face_column(&n, x)
            .unwrap();
        let m = boolean(&a, &b, Boolean::Merge).unwrap();
        assert_eq!(m.crs, a.crs);
        let faces = m.face_attributes();
        let (fa, fb) = (
            faces["a"].as_primitive::<Float64Type>(),
            faces["b"].as_primitive::<Float64Type>(),
        );
        for f in 0..m.triangles().len() {
            assert!(fa.is_null(f) != fb.is_null(f));
        }
        assert!(fa.null_count() > 0 && fb.null_count() > 0);
        let rock = m.vertex_attributes()["rock"].as_string::<i32>();
        assert!(rock.iter().any(|r| r == Some("ore")) && rock.null_count() > 0);

        let (n, x) = column("a", 1.0);
        let text: ArrayRef = Arc::new(StringArray::from(vec!["x"; 12]));
        let clash = b.with_face_column(&n, text).unwrap();
        assert!(boolean(&a.with_face_column(&n, x).unwrap(), &clash, Boolean::Merge).is_err());
        let mut other = b.clone();
        other.crs = Some("EPSG:4326".into());
        assert!(boolean(&a, &other, Boolean::Merge).is_err());
        let open = Mesh::new(a.vertices().to_vec(), a.triangles()[1..].to_vec()).unwrap();
        assert!(boolean(&open, &b, Boolean::Overlap).is_err());
    }

    /// Theory check: a cut keeps the volume below the plane and stays closed,
    /// with a cap whose holes follow cavities.
    #[test]
    fn cuts_are_capped() {
        let c = cube(0.0, 1.0);
        let half = cut(&c, [0.0, 0.0, 0.25], [0.0, 0.0, 1.0]).unwrap();
        clean(&half);
        assert!((half.volume().unwrap() - 0.25).abs() < 1e-12);
        let slanted = cut(&c, [0.5; 3], [1.0, 1.0, 1.0]).unwrap();
        clean(&slanted);
        assert!((slanted.volume().unwrap() - 0.5).abs() < 1e-12);
        for normal in [[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]] {
            let wedge = cut(&c, [0.0; 3], normal).unwrap();
            clean(&wedge);
            assert!((wedge.volume().unwrap() - 0.5).abs() < 1e-12);
        }
        let whole = cut(&c, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]).unwrap();
        clean(&whole);
        assert!((whole.volume().unwrap() - 1.0).abs() < 1e-12);
        assert!(
            cut(&c, [0.0; 3], [0.0, 0.0, 1.0])
                .unwrap()
                .triangles()
                .is_empty()
        );

        let s = sphere(2.0);
        let r = Rotation3::from_euler_angles(0.3, 0.2, 0.1);
        let lower = cut(&s, [0.0; 3], (r * Vector3::z()).into()).unwrap();
        let upper = cut(&s, [0.0; 3], (r * -Vector3::z()).into()).unwrap();
        clean(&lower);
        clean(&upper);
        let v = s.volume().unwrap();
        assert!((volume(&lower) + volume(&upper) - v).abs() < 1e-9 * v);

        let hollow = boolean(&cube(0.0, 3.0), &cube(1.0, 2.0), Boolean::Subtract).unwrap();
        let ring = cut(&hollow, [0.0, 0.0, 1.5], [0.0, 0.0, 1.0]).unwrap();
        clean(&ring);
        assert!((ring.volume().unwrap() - 13.0).abs() < 1e-12);
        assert_eq!(ring.validate(0.0).unwrap().summary.shells, 1);

        let square = Mesh::new(
            vec![[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]],
            vec![[0, 1, 2], [0, 2, 3]],
        )
        .unwrap();
        let trimmed = cut(&square, [0.25, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap();
        assert!((trimmed.area() - 0.25).abs() < 1e-12 && !trimmed.is_closed());
        assert!(cut(&c, [0.0; 3], [0.0; 3]).is_err());
    }
}
