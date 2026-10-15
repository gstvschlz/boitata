//! Plane sections of a mesh as polylines.
//!
//! A vertex at signed distance `d >= 0` from the plane counts as on the side
//! the normal points to, so a vertex on the plane sits on that side and every
//! edge crosses the plane at most once, at a point keyed by the edge. Faces
//! sharing an edge then share its crossing and the pieces join into loops
//! without any tolerance.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array};
use arrow_ord::ord::{DynComparator, make_comparator};
use arrow_schema::SortOptions;

use crate::mesh::take_rows;
use crate::{Error, Mesh, Polylines, Result};

type Edge = (u32, u32);

/// One piece of a section line: from the crossing on edge `from` to the
/// crossing on edge `to`, across `face`.
#[derive(Clone, Copy)]
struct Piece {
    from: Edge,
    to: Edge,
    face: u32,
}

fn key(i: u32, j: u32) -> Edge {
    (i.min(j), i.max(j))
}

/// The pieces of the section at signed distances `d`, in face order. Each
/// runs counter-clockwise around the inside seen from the side `d >= 0` for
/// a mesh wound outward.
fn pieces(triangles: &[[u32; 3]], d: &[f64]) -> Vec<Piece> {
    let above = |v: u32| d[v as usize] >= 0.0;
    let mut out = vec![];
    for (f, &[a, b, c]) in triangles.iter().enumerate() {
        let (mut from, mut to) = (None, None);
        for (i, j) in [(a, b), (b, c), (c, a)] {
            match (above(i), above(j)) {
                (true, false) => from = Some(key(i, j)),
                (false, true) => to = Some(key(i, j)),
                _ => {}
            }
        }
        if let (Some(from), Some(to)) = (from, to) {
            out.push(Piece {
                from,
                to,
                face: f as u32,
            });
        }
    }
    out
}

/// Chains of pieces, each joined end to start, and whether it closes.
fn chains(pieces: &[Piece]) -> Vec<(Vec<Piece>, bool)> {
    let mut starting: HashMap<Edge, Vec<usize>> = HashMap::new();
    let mut ending: HashMap<Edge, Vec<usize>> = HashMap::new();
    for (k, p) in pieces.iter().enumerate() {
        starting.entry(p.from).or_default().push(k);
        ending.entry(p.to).or_default().push(k);
    }
    let mut used = vec![false; pieces.len()];
    let unused = |list: Option<&Vec<usize>>, used: &[bool]| {
        list.into_iter().flatten().copied().find(|&k| !used[k])
    };
    let mut out = vec![];
    for seed in 0..pieces.len() {
        if used[seed] {
            continue;
        }
        used[seed] = true;
        let mut chain = vec![seed];
        let first = pieces[seed].from;
        let mut closed = false;
        while let Some(k) = unused(
            starting.get(&pieces[*chain.last().expect("seed")].to),
            &used,
        ) {
            used[k] = true;
            chain.push(k);
        }
        if pieces[*chain.last().expect("seed")].to == first {
            closed = true;
        } else {
            let mut back = vec![];
            let mut at = first;
            while let Some(k) = unused(ending.get(&at), &used) {
                used[k] = true;
                back.push(k);
                at = pieces[k].from;
            }
            back.reverse();
            back.extend(chain);
            chain = back;
        }
        out.push((chain.into_iter().map(|k| pieces[k]).collect(), closed));
    }
    out
}

/// Whether faces `f` and `g` hold the same value in every column.
fn same_row(columns: &[DynComparator], f: u32, g: u32) -> bool {
    columns
        .iter()
        .all(|cmp| cmp(f as usize, g as usize).is_eq())
}

/// Runs of consecutive pieces from faces with equal attributes. A closed
/// chain whose faces all agree stays one closed run.
fn runs(mut chain: Vec<Piece>, closed: bool, columns: &[DynComparator]) -> Vec<(Vec<Piece>, bool)> {
    let n = chain.len();
    let mut breaks: Vec<usize> = (1..n)
        .filter(|&k| !same_row(columns, chain[k - 1].face, chain[k].face))
        .collect();
    if closed && breaks.is_empty() {
        return vec![(chain, true)];
    }
    if closed && same_row(columns, chain[n - 1].face, chain[0].face) {
        let shift = breaks.remove(0);
        chain.rotate_left(shift);
        breaks = breaks.iter().map(|b| b - shift).collect();
    }
    let mut out = vec![];
    let mut start = 0;
    for b in breaks.into_iter().chain([n]) {
        out.push((chain[start..b].to_vec(), false));
        start = b;
    }
    out
}

impl Mesh {
    /// Lines where planes cut the mesh: the plane through `origin`
    /// perpendicular to `normal`, moved each of `offsets` along the unit
    /// normal. One feature per line, with the `offset` it came from and the
    /// attributes of its faces. A line through faces with different
    /// attributes splits into open lines where they change; otherwise a
    /// closed mesh gives closed loops and an open surface open lines where
    /// the cut leaves it. Loops of a mesh wound outward run
    /// counter-clockwise around its inside seen from the side `normal`
    /// points to, so loops round cavities run clockwise. A vertex on a plane
    /// counts as on the side `normal` points to, so a face lying in the
    /// plane adds nothing and a cube cut at its top face gives that face's
    /// outline, at its bottom face nothing.
    pub fn section(
        &self,
        origin: [f64; 3],
        normal: [f64; 3],
        offsets: &[f64],
    ) -> Result<Polylines> {
        let length = normal.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !(length.is_finite() && length > 0.0) || origin.iter().any(|v| !v.is_finite()) {
            return Err(Error::Geometry(
                "origin must be finite and normal finite and non-zero".into(),
            ));
        }
        if offsets.iter().any(|v| !v.is_finite()) {
            return Err(Error::Geometry("offsets must be finite".into()));
        }
        let faces = self.face_attributes();
        if faces.column_by_name("offset").is_some() {
            return Err(Error::Geometry(
                "face attribute `offset` is reserved for the section offset".into(),
            ));
        }
        let unit = normal.map(|v| v / length);
        let columns = faces
            .columns()
            .iter()
            .map(|c| make_comparator(c, c, SortOptions::default()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let (mut vertices, mut parts, mut closed) = (vec![], vec![0u32], vec![]);
        let (mut rows, mut offset) = (vec![], vec![]);
        for &o in offsets {
            let d: Vec<f64> = self
                .vertices()
                .iter()
                .map(|v| (0..3).map(|k| (v[k] - origin[k]) * unit[k]).sum::<f64>() - o)
                .collect();
            let at = |(i, j): Edge| {
                let (i, j) = if d[i as usize] >= 0.0 { (i, j) } else { (j, i) };
                let (p, q) = (self.vertices()[i as usize], self.vertices()[j as usize]);
                let t = d[i as usize] / (d[i as usize] - d[j as usize]);
                [0, 1, 2].map(|k| p[k] + t * (q[k] - p[k]))
            };
            for (chain, ring) in chains(&pieces(self.triangles(), &d)) {
                for (run, ring) in runs(chain, ring, &columns) {
                    let mut line: Vec<[f64; 3]> = vec![at(run[0].from)];
                    line.extend(run.iter().map(|p| at(p.to)));
                    if ring {
                        line.pop();
                    }
                    line.dedup();
                    if ring && line.len() > 1 && line[0] == line[line.len() - 1] {
                        line.pop();
                    }
                    if line.len() < if ring { 3 } else { 2 } {
                        continue;
                    }
                    vertices.extend(line);
                    parts.push(vertices.len() as u32);
                    closed.push(ring);
                    rows.push(run[0].face);
                    offset.push(o);
                }
            }
        }
        let features = (0..parts.len() as u32).collect();
        let attributes = crate::set_column(
            &take_rows(faces, rows)?,
            "offset",
            Arc::new(Float64Array::from(offset)) as ArrayRef,
        )?;
        let mut lines = Polylines::new(vertices, parts, features, closed, attributes)?;
        lines.crs.clone_from(&self.crs);
        lines.length_unit.clone_from(&self.length_unit);
        Ok(lines)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Float64Type;
    use arrow_array::{Int64Array, StringArray};

    /// Icosahedron subdivided `levels` times onto the sphere of `radius`,
    /// wound outward.
    pub(crate) fn icosphere(radius: f64, levels: usize) -> Mesh {
        let g = (1.0 + 5f64.sqrt()) / 2.0;
        let mut v: Vec<[f64; 3]> = vec![
            [-1., g, 0.],
            [1., g, 0.],
            [-1., -g, 0.],
            [1., -g, 0.],
            [0., -1., g],
            [0., 1., g],
            [0., -1., -g],
            [0., 1., -g],
            [g, 0., -1.],
            [g, 0., 1.],
            [-g, 0., -1.],
            [-g, 0., 1.],
        ];
        let mut t: Vec<[u32; 3]> = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        for _ in 0..levels {
            let mut mid: HashMap<Edge, u32> = HashMap::new();
            let mut half = |i: u32, j: u32, v: &mut Vec<[f64; 3]>| {
                *mid.entry(key(i, j)).or_insert_with(|| {
                    let (p, q) = (v[i as usize], v[j as usize]);
                    v.push([0, 1, 2].map(|k| (p[k] + q[k]) / 2.0));
                    v.len() as u32 - 1
                })
            };
            t = t
                .iter()
                .flat_map(|&[a, b, c]| {
                    let (ab, bc, ca) = (half(a, b, &mut v), half(b, c, &mut v), half(c, a, &mut v));
                    [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
                })
                .collect();
        }
        let v = v
            .iter()
            .map(|p| {
                let r = p.iter().map(|x| x * x).sum::<f64>().sqrt();
                p.map(|x| radius * x / r)
            })
            .collect();
        Mesh::new(v, t).unwrap()
    }

    fn cube() -> Mesh {
        let v: Vec<[f64; 3]> = (0..8)
            .map(|i| [(i & 1) as f64, (i >> 1 & 1) as f64, (i >> 2) as f64])
            .collect();
        let t = vec![
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
        ];
        Mesh::new(v, t).unwrap()
    }

    fn signed_area(ring: &[[f64; 3]]) -> f64 {
        let n = ring.len();
        (0..n)
            .map(|i| {
                let (a, b) = (ring[i], ring[(i + 1) % n]);
                a[0] * b[1] - b[0] * a[1]
            })
            .sum::<f64>()
            / 2.0
    }

    /// Theory check: a plane through the centre of a fine sphere meets it in
    /// one closed loop of length 2πr, short of it by the sag of the chords
    /// only, running counter-clockwise seen from above.
    #[test]
    fn sphere_section_is_a_great_circle() {
        let r = 10.0;
        let sphere = icosphere(r, 5);
        assert!(sphere.volume().unwrap() > 0.0);
        let lines = sphere
            .section([0.1, -0.2, 0.0], [0.0, 0.0, 2.0], &[0.0])
            .unwrap();
        assert_eq!((lines.len(), lines.closed()), (1, &[true][..]));
        let length = lines.length()[0];
        let circle = 2.0 * std::f64::consts::PI * r;
        assert!(
            length < circle && (circle - length) / circle < 1e-3,
            "{length}"
        );
        assert!(signed_area(lines.part(0)) > 0.0);
        assert!(lines.vertices().iter().all(|p| p[2].abs() < 1e-12));

        let tilted = sphere.section([0.0; 3], [1.0, 2.0, 3.0], &[0.0]).unwrap();
        assert_eq!(tilted.len(), 1);
        assert!((circle - tilted.length()[0]) / circle < 1e-3);
    }

    #[test]
    fn offsets_give_parallel_sections_and_on_plane_vertices_count_above() {
        let mut c = cube();
        c.crs = Some("EPSG:31982".into());
        let lines = c
            .section([0.0; 3], [0.0, 0.0, 1.0], &[0.0, 0.25, 1.0, 1.5])
            .unwrap();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines.crs, c.crs);
        let offset = lines.attributes()["offset"].as_primitive::<Float64Type>();
        assert_eq!(offset.values().to_vec(), [0.25, 1.0]);
        assert!(lines.closed().iter().all(|&c| c));
        assert_eq!(lines.length(), [4.0, 4.0]);
        assert_eq!(lines.area(), [1.0, 1.0]);
        assert!(lines.part(1).iter().all(|p| p[2] == 1.0));
        assert_eq!(lines.part(1).len(), 4);
        let below = c.section([0.0; 3], [0.0, 0.0, -1.0], &[0.0]).unwrap();
        assert_eq!((below.len(), below.part(0).len()), (1, 4));
        assert!(signed_area(below.part(0)) < 0.0);
    }

    /// Octahedron cut at its equator, where four vertices lie on the plane:
    /// the loop runs through them, without repeated points.
    #[test]
    fn a_plane_through_vertices_runs_through_them() {
        let v = vec![
            [1., 0., 0.],
            [0., 1., 0.],
            [-1., 0., 0.],
            [0., -1., 0.],
            [0., 0., 1.],
            [0., 0., -1.],
        ];
        let t = vec![
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            [1, 0, 5],
            [2, 1, 5],
            [3, 2, 5],
            [0, 3, 5],
        ];
        let m = Mesh::new(v, t).unwrap();
        let lines = m.section([0.0; 3], [0.0, 0.0, 1.0], &[0.0]).unwrap();
        assert_eq!((lines.len(), lines.part(0).len()), (1, 4));
        assert!((lines.length()[0] - 4.0 * 2f64.sqrt()).abs() < 1e-12);
        let top = m.section([0.0; 3], [0.0, 0.0, 1.0], &[1.0]).unwrap();
        assert!(top.is_empty());
    }

    #[test]
    fn an_open_surface_gives_open_lines() {
        let square = Mesh::new(
            vec![[0., 0., 0.], [2., 0., 0.], [2., 2., 1.], [0., 2., 1.]],
            vec![[0, 1, 2], [0, 2, 3]],
        )
        .unwrap();
        let lines = square
            .section([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], &[0.0])
            .unwrap();
        assert_eq!((lines.len(), lines.closed()), (1, &[false][..]));
        assert_eq!(lines.part(0).len(), 3);
        assert!((lines.length()[0] - 2.0).abs() < 1e-12);
        assert!(lines.part(0).iter().all(|p| p[1] == 1.0 && p[2] == 0.5));
    }

    /// Each line carries the attributes of its faces, splitting where they
    /// change and staying closed where they do not.
    #[test]
    fn face_attributes_carry_over() {
        let side: ArrayRef = Arc::new(StringArray::from(
            [
                "z0", "z0", "z1", "z1", "y0", "y0", "y1", "y1", "x0", "x0", "x1", "x1",
            ]
            .to_vec(),
        ));
        let one: ArrayRef = Arc::new(Int64Array::from(vec![7; 12]));
        let c = cube().with_face_column("rock", one).unwrap();
        let ring = c.section([0.0; 3], [0.0, 0.0, 1.0], &[0.5]).unwrap();
        assert_eq!((ring.len(), ring.closed()), (1, &[true][..]));
        let schema = ring.attributes().schema();
        let names: Vec<_> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, ["rock", "offset"]);

        let c = c.with_face_column("side", side).unwrap();
        let lines = c.section([0.0; 3], [0.0, 0.0, 1.0], &[0.5]).unwrap();
        assert_eq!(lines.len(), 4);
        assert!(lines.closed().iter().all(|&c| !c));
        assert!(lines.length().iter().all(|&l| l == 1.0));
        let sides = lines.attributes()["side"].as_string::<i32>();
        let mut names: Vec<_> = sides.iter().flatten().collect();
        names.sort_unstable();
        assert_eq!(names, ["x0", "x1", "y0", "y1"]);
        for f in 0..4 {
            let [p, q] = [lines.part(f)[0], lines.part(f)[1]];
            let on = match sides.value(f) {
                "x0" => p[0] == 0.0 && q[0] == 0.0,
                "x1" => p[0] == 1.0 && q[0] == 1.0,
                "y0" => p[1] == 0.0 && q[1] == 0.0,
                _ => p[1] == 1.0 && q[1] == 1.0,
            };
            assert!(on);
        }
    }

    #[test]
    fn bad_planes_and_reserved_names_are_rejected() {
        let c = cube();
        assert!(c.section([0.0; 3], [0.0; 3], &[0.0]).is_err());
        let nan = [f64::NAN, 0.0, 0.0];
        assert!(c.section(nan, [0.0, 0.0, 1.0], &[0.0]).is_err());
        assert!(
            c.section([0.0; 3], [0.0, 0.0, 1.0], &[f64::INFINITY])
                .is_err()
        );
        let o: ArrayRef = Arc::new(Float64Array::from(vec![0.0; 12]));
        let c = c.with_face_column("offset", o).unwrap();
        assert!(c.section([0.0; 3], [0.0, 0.0, 1.0], &[0.0]).is_err());
        let none = cube().section([0.0; 3], [0.0, 0.0, 1.0], &[]).unwrap();
        assert!(none.is_empty());
    }
}
