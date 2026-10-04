//! Outer faces of a block model: the faces a viewer can see, between a block
//! and an empty cell or the edge of the grid.

use boitata_core::{BlockModel, Layout, block_frame};
use rayon::prelude::*;

use crate::error::{BlockModelError, Result};

/// Faces of a block model, quads sharing the corners of their block.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OuterFaces {
    pub points: Vec<[f64; 3]>,
    /// Indices into `points`, counter-clockwise seen from outside the block.
    pub quads: Vec<[i64; 4]>,
    /// Row of the block each quad bounds.
    pub rows: Vec<i64>,
}

/// Per axis, the steps a parent cell splits into and their fractions of it:
/// every sub-block corner on that axis is one of the fractions.
fn steps(model: &BlockModel) -> [Vec<f64>; 3] {
    match model.layout() {
        Layout::SubBlocked {
            grid: Some(n),
            extent,
            ..
        } if fits(extent, *n) => n.map(|n| (0..=n).map(|k| f64::from(k) / f64::from(n)).collect()),
        Layout::SubBlocked { extent, .. } => [0, 1, 2].map(|a| {
            let mut u: Vec<f64> = extent.iter().flat_map(|e| [e[a], e[a + 3]]).collect();
            u.extend([0.0, 1.0]);
            u.par_sort_unstable_by(f64::total_cmp);
            u.dedup();
            u
        }),
        _ => [0, 1, 2].map(|_| vec![0.0, 1.0]),
    }
}

fn fits(extent: &[[f64; 6]], n: [u32; 3]) -> bool {
    extent.iter().all(|e| {
        (0..6).all(|i| {
            let k = (e[i] * f64::from(n[i % 3])).round();
            k / f64::from(n[i % 3]) == e[i]
        })
    })
}

/// Lower and upper corner of a row on the lattice of `steps`.
fn span(model: &BlockModel, steps: &[Vec<f64>; 3], row: usize) -> [[u64; 3]; 2] {
    let ijk = model.geometry().ijk(model.parent_index(row));
    let e = match model.layout() {
        Layout::SubBlocked { extent, .. } => extent[row],
        _ => [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    };
    let at = |a: usize, f: f64| {
        let s = steps[a].len() - 1;
        let k = steps[a].partition_point(|&u| u < f);
        (ijk[a] * s + k) as u64
    };
    [
        [0, 1, 2].map(|a| at(a, e[a])),
        [0, 1, 2].map(|a| at(a, e[a + 3])),
    ]
}

/// Corners of the face of `axis` on its upper or lower side, as bits of the
/// block's corners (bit `a` set: upper on axis `a`), wound outward.
fn quad(axis: usize, upper: bool) -> [usize; 4] {
    let (b, c) = ((axis + 1) % 3, (axis + 2) % 3);
    let side = usize::from(upper) << axis;
    let q = [(0, 0), (1, 0), (1, 1), (0, 1)].map(|(i, j)| side | i << b | j << c);
    if upper { q } else { [q[3], q[2], q[1], q[0]] }
}

/// Outer faces of the rows where `keep` is true (every row when `None`).
///
/// A face is left out when one kept block across it covers exactly the same
/// rectangle; a face partly covered stays, inside the solid. Lattice positions
/// of sub-block corners match by exact fraction, or on `grid` when every
/// corner lies on it.
pub fn outer_faces(model: &BlockModel, keep: Option<&[bool]>) -> Result<OuterFaces> {
    if keep.is_some_and(|k| k.len() != model.len()) {
        return Err(BlockModelError::InvalidGridParams(
            "keep needs one value per row".into(),
        ));
    }
    let steps = steps(model);
    let rows: Vec<usize> = (0..model.len())
        .filter(|&r| keep.is_none_or(|k| k[r]))
        .collect();
    let spans: Vec<[[u64; 3]; 2]> = rows.par_iter().map(|&r| span(model, &steps, r)).collect();
    let mut open = vec![0b11_1111u8; rows.len()];
    for a in 0..3 {
        let key = |mut p: [u64; 3], at: u64| {
            p[a] = at;
            [p[2], p[1], p[0]]
        };
        let mut lower: Vec<([u64; 3], usize)> = spans
            .par_iter()
            .enumerate()
            .map(|(i, [lo, _])| (key(*lo, lo[a]), i))
            .collect();
        lower.par_sort_unstable();
        let (b, c) = ((a + 1) % 3, (a + 2) % 3);
        let shared: Vec<(usize, usize)> = spans
            .par_iter()
            .enumerate()
            .filter_map(|(i, [lo, hi])| {
                let k = lower
                    .binary_search_by(|(x, _)| x.cmp(&key(*lo, hi[a])))
                    .ok()?;
                let j = lower[k].1;
                let other = spans[j][1];
                (other[b] == hi[b] && other[c] == hi[c]).then_some((i, j))
            })
            .collect();
        for (i, j) in shared {
            open[i] &= !(2 << (2 * a));
            open[j] &= !(1 << (2 * a));
        }
    }
    let faces = |mask: u8| {
        (0..6)
            .filter(move |f| mask >> f & 1 == 1)
            .map(|f| quad(f / 2, f % 2 == 1))
    };
    let used: Vec<u8> = open
        .par_iter()
        .map(|&m| faces(m).flatten().fold(0u8, |u, c| u | 1 << c))
        .collect();
    let mut first = Vec::with_capacity(used.len());
    let mut total = 0i64;
    for u in &used {
        first.push(total);
        total += i64::from(u.count_ones());
    }

    let (spans, open, used, first) = (&spans, &open, &used, &first);
    let g = model.geometry();
    let frame = block_frame(g.rotation).transpose();
    let nodes: [Vec<f64>; 3] = [0, 1, 2].map(|a| {
        let s = steps[a].len() - 1;
        (0..=g.count[a] * s)
            .map(|c| ((c / s) as f64 + steps[a][c % s]) * g.size[a])
            .collect()
    });
    let nodes = &nodes;
    let points = (0..rows.len())
        .into_par_iter()
        .flat_map_iter(|i| {
            let [lo, hi] = spans[i];
            (0..8).filter(move |c| used[i] >> c & 1 == 1).map(move |c| {
                let local = nalgebra::Vector3::from_fn(|a, _| {
                    nodes[a][(if c >> a & 1 == 1 { hi[a] } else { lo[a] }) as usize]
                });
                let w = frame * local;
                [0, 1, 2].map(|a| g.origin[a] + w[a])
            })
        })
        .collect();
    let quads = (0..rows.len())
        .into_par_iter()
        .flat_map_iter(|i| {
            let rank =
                move |c: usize| first[i] + i64::from((used[i] & ((1u8 << c) - 1)).count_ones());
            faces(open[i]).map(move |q| q.map(rank))
        })
        .collect();
    let rows = (0..rows.len())
        .into_par_iter()
        .flat_map_iter(|i| std::iter::repeat_n(rows[i] as i64, open[i].count_ones() as usize))
        .collect();
    Ok(OuterFaces {
        points,
        quads,
        rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, Float64Array, RecordBatch};
    use boitata_core::Geometry;
    use std::sync::Arc;

    fn geometry(n: usize) -> Geometry {
        Geometry {
            origin: [10.0, 20.0, 30.0],
            size: [1.0, 2.0, 3.0],
            count: [n; 3],
            rotation: [30.0, 20.0, 10.0],
        }
    }

    fn batch(n: usize) -> RecordBatch {
        let column: ArrayRef = Arc::new(Float64Array::from(vec![0.0; n]));
        RecordBatch::try_from_iter([("a", column)]).unwrap()
    }

    fn masked(n: usize, index: Vec<u64>) -> BlockModel {
        let rows = index.len();
        BlockModel::masked(geometry(n), index, batch(rows)).unwrap()
    }

    fn area(f: &OuterFaces) -> f64 {
        let p = |i: i64| nalgebra::Vector3::from(f.points[i as usize]);
        f.quads
            .iter()
            .map(|q| (p(q[1]) - p(q[0])).cross(&(p(q[3]) - p(q[0]))).norm())
            .sum()
    }

    #[test]
    fn a_solid_cube_draws_its_outer_faces_once() {
        let n = 4;
        let model = masked(n, (0..(n * n * n) as u64).collect());
        let faces = outer_faces(&model, None).unwrap();
        assert_eq!(faces.quads.len(), 6 * n * n);
        assert!((area(&faces) - 22.0 * (n * n) as f64).abs() < 1e-9);
        let mut keep = vec![true; n * n * n];
        keep[1 + n + n * n] = false;
        let hollow = outer_faces(&model, Some(&keep)).unwrap();
        assert_eq!(hollow.quads.len(), 6 * n * n + 6);
        assert!(!hollow.rows.contains(&((1 + n + n * n) as i64)));
    }

    #[test]
    fn two_blocks_sharing_a_face_draw_ten_faces_wound_outward() {
        let faces = outer_faces(&masked(3, vec![0, 1]), None).unwrap();
        assert_eq!(faces.quads.len(), 10);
        assert_eq!(faces.points.len(), 16);
        let p = |i: i64| nalgebra::Vector3::from(faces.points[i as usize]);
        let center = faces
            .points
            .iter()
            .map(|&x| nalgebra::Vector3::from(x))
            .sum::<nalgebra::Vector3<f64>>()
            / 16.0;
        for q in &faces.quads {
            let normal = (p(q[1]) - p(q[0])).cross(&(p(q[3]) - p(q[0])));
            let middle = (p(q[0]) + p(q[2])) / 2.0;
            assert!(normal.dot(&(middle - center)) > 0.0);
        }
    }

    #[test]
    fn sub_blocks_hide_faces_matched_whole_only() {
        let extent = vec![
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.3, 1.0],
            [0.0, 0.3, 0.0, 1.0, 1.0, 1.0],
        ];
        let model =
            BlockModel::subblocked(geometry(2), vec![0, 1, 1], extent, None, batch(3)).unwrap();
        let faces = outer_faces(&model, None).unwrap();
        assert_eq!(faces.quads.len(), 16);
        assert!((area(&faces) - 32.0 - 12.0).abs() < 1e-9);
        let halves = vec![
            [0.0, 0.0, 0.0, 0.5, 1.0, 1.0],
            [0.5, 0.0, 0.0, 1.0, 1.0, 1.0],
        ];
        let pair =
            BlockModel::subblocked(geometry(2), vec![0, 0], halves, Some([2, 1, 1]), batch(2))
                .unwrap();
        assert_eq!(outer_faces(&pair, None).unwrap().quads.len(), 10);
    }

    #[test]
    fn keep_needs_one_value_per_row() {
        assert!(outer_faces(&masked(2, vec![0]), Some(&[true, false])).is_err());
    }
}
