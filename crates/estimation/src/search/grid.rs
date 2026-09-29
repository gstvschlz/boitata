//! Cells in the search ellipsoid's frame, visited nearest first.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::OnceLock;

use super::work;

/// Cell offsets are tabled up to this many cells on each axis; farther
/// cells are visited ring by ring.
const REACH: i64 = 12;

/// Candidates by (squared distance bits, index): distances are never
/// negative, so their bits order them.
type Heap = BinaryHeap<Reverse<(u64, u32)>>;

#[derive(Clone, Copy)]
struct Entry {
    p: [f64; 3],
    i: u32,
}

/// Points in cubic cells of side `h` over the box `lo..=hi`: those of the
/// last build packed by cell, those added since in a list per cell.
pub(super) struct Grid {
    lo: [f64; 3],
    hi: [f64; 3],
    h: f64,
    dims: [i64; 3],
    start: Vec<u32>,
    entries: Vec<Entry>,
    added: Vec<Vec<Entry>>,
    built: usize,
}

/// Offsets within `REACH` sorted by `m = Σ max(0, |o| - 1)²`, the squared
/// lower bound, in cell sides, of distances between points of a cell and of
/// the cell at that offset; and the end of each run of one `m`.
fn table() -> &'static (Vec<[i64; 3]>, Vec<(u64, usize)>) {
    static TABLE: OnceLock<(Vec<[i64; 3]>, Vec<(u64, usize)>)> = OnceLock::new();
    TABLE.get_or_init(|| {
        let axis = || -REACH..=REACH;
        let mut keyed: Vec<(u64, [i64; 3])> = axis()
            .flat_map(|x| axis().flat_map(move |y| axis().map(move |z| [x, y, z])))
            .map(|o| {
                (
                    o.iter()
                        .map(|&v| ((v.abs() - 1).max(0) as u64).pow(2))
                        .sum(),
                    o,
                )
            })
            .collect();
        keyed.sort_unstable();
        let mut shells: Vec<(u64, usize)> = vec![];
        for (k, &(m, _)) in keyed.iter().enumerate() {
            match shells.last_mut() {
                Some(s) if s.0 == m => s.1 = k + 1,
                _ => shells.push((m, k + 1)),
            }
        }
        (keyed.into_iter().map(|k| k.1).collect(), shells)
    })
}

fn bounds(points: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in points.iter().filter(|p| p.iter().all(|v| v.is_finite())) {
        for d in 0..3 {
            lo[d] = lo[d].min(p[d]);
            hi[d] = hi[d].max(p[d]);
        }
    }
    match lo[0] <= hi[0] {
        true => (lo, hi),
        false => ([0.0; 3], [0.0; 3]),
    }
}

/// Cell side for about one of `n` points per cell over the axes the box
/// spans, doubled until there are at most about four cells per point.
fn side(lo: &[f64; 3], hi: &[f64; 3], n: usize) -> f64 {
    let extent: [f64; 3] = std::array::from_fn(|d| hi[d] - lo[d]);
    let longest = extent.iter().copied().fold(0.0, f64::max);
    if longest <= 0.0 {
        return 1.0;
    }
    let spanned: Vec<f64> = extent
        .iter()
        .copied()
        .filter(|&e| e > longest * 1e-9)
        .collect();
    let mut h =
        (spanned.iter().product::<f64>() / n.max(1) as f64).powf(1.0 / spanned.len() as f64);
    let cells = |h: f64| {
        extent
            .iter()
            .map(|e| (e / h).floor() + 1.0)
            .product::<f64>()
    };
    while cells(h) > 4.0 * n as f64 + 64.0 {
        h *= 2.0;
    }
    h
}

/// Hands the heap's candidates closer than `bound` to `visit`, nearest
/// first; true when `visit` asks to stop.
fn release(heap: &mut Heap, bound: f64, visit: &mut impl FnMut(f64, usize) -> bool) -> bool {
    while let Some(&Reverse((bits, i))) = heap.peek() {
        let d2 = f64::from_bits(bits);
        if d2 >= bound {
            return false;
        }
        heap.pop();
        if visit(d2, i as usize) {
            return true;
        }
    }
    false
}

impl Grid {
    pub(super) fn new(points: &[[f64; 3]]) -> Self {
        let (lo, hi) = bounds(points);
        Self::build(points, lo, hi)
    }

    fn build(points: &[[f64; 3]], lo: [f64; 3], hi: [f64; 3]) -> Self {
        work(points.len());
        let h = side(&lo, &hi, points.len());
        let dims = std::array::from_fn(|d| ((hi[d] - lo[d]) / h).floor() as i64 + 1);
        let mut grid = Self {
            lo,
            hi,
            h,
            dims,
            start: vec![],
            entries: vec![],
            added: vec![],
            built: points.len(),
        };
        let cells = (dims[0] * dims[1] * dims[2]) as usize;
        let ids: Vec<usize> = points.iter().map(|p| grid.id(grid.cell(p))).collect();
        let mut start = vec![0u32; cells + 1];
        for &c in &ids {
            start[c + 1] += 1;
        }
        for c in 0..cells {
            start[c + 1] += start[c];
        }
        let mut fill = start.clone();
        let mut entries = vec![Entry { p: [0.0; 3], i: 0 }; points.len()];
        for (i, (p, &c)) in points.iter().zip(&ids).enumerate() {
            entries[fill[c] as usize] = Entry { p: *p, i: i as u32 };
            fill[c] += 1;
        }
        grid.start = start;
        grid.entries = entries;
        grid.added = vec![vec![]; cells];
        grid
    }

    /// Indexes the last of `points`. A finite point outside the box rebuilds
    /// the grid over a box padded by a quarter of its extent, so points
    /// marching outward rebuild it a logarithmic number of times; so does
    /// doubling the number of points, which keeps about one point per cell.
    pub(super) fn add(&mut self, points: &[[f64; 3]]) {
        let i = points.len() - 1;
        let p = points[i];
        let finite = p.iter().all(|v| v.is_finite());
        let outside = finite && (0..3).any(|d| p[d] < self.lo[d] || p[d] > self.hi[d]);
        if outside || points.len() >= 2 * self.built.max(32) {
            let (lo, hi) = bounds(points);
            let pad: [f64; 3] = std::array::from_fn(|d| match outside {
                true => 0.25 * (hi[d] - lo[d]),
                false => 0.0,
            });
            *self = Self::build(
                points,
                std::array::from_fn(|d| lo[d] - pad[d]),
                std::array::from_fn(|d| hi[d] + pad[d]),
            );
            return;
        }
        let id = self.id(self.cell(&p));
        self.added[id].push(Entry { p, i: i as u32 });
    }

    fn cell(&self, p: &[f64; 3]) -> [i64; 3] {
        std::array::from_fn(|d| {
            (((p[d] - self.lo[d]) / self.h).floor() as i64).clamp(0, self.dims[d] - 1)
        })
    }

    fn id(&self, c: [i64; 3]) -> usize {
        (c[0] + self.dims[0] * (c[1] + self.dims[1] * c[2])) as usize
    }

    /// Calls `visit(d2, i)` for every point within `radius2` of `q`, in
    /// increasing (d2, i), until it returns true. A query outside the box
    /// starts from the nearest cell, which keeps every bound a lower bound.
    pub(super) fn nearest(
        &self,
        q: &[f64; 3],
        radius2: f64,
        mut visit: impl FnMut(f64, usize) -> bool,
    ) {
        let (offsets, shells) = table();
        let c = self.cell(q);
        let h2 = self.h * self.h;
        // Cells beyond the table may be as close as REACH - 1 sides.
        let beyond = h2 * ((REACH - 1) * (REACH - 1)) as f64;
        let mut heap = Heap::new();
        let mut begin = 0;
        for &(m, end) in shells {
            let bound = (h2 * m as f64).min(beyond) * (1.0 - 1e-9);
            if bound > radius2 {
                break;
            }
            if release(&mut heap, bound, &mut visit) {
                return;
            }
            for o in &offsets[begin..end] {
                self.scan(
                    [c[0] + o[0], c[1] + o[1], c[2] + o[2]],
                    q,
                    radius2,
                    &mut heap,
                );
            }
            begin = end;
        }
        let far = (0..3)
            .map(|d| c[d].max(self.dims[d] - 1 - c[d]))
            .max()
            .unwrap_or(0);
        for r in REACH + 1..=far {
            let bound = ((r - 1) as f64 * self.h).powi(2) * (1.0 - 1e-9);
            if bound > radius2 {
                break;
            }
            if release(&mut heap, bound, &mut visit) {
                return;
            }
            self.ring(c, r, q, radius2, &mut heap);
        }
        release(&mut heap, f64::INFINITY, &mut visit);
    }

    /// Scans the cells exactly `r` cells from `c` on some axis.
    fn ring(&self, c: [i64; 3], r: i64, q: &[f64; 3], radius2: f64, heap: &mut Heap) {
        let lo = |d: usize| (-r).max(-c[d]);
        let hi = |d: usize| r.min(self.dims[d] - 1 - c[d]);
        for x in lo(0)..=hi(0) {
            for y in lo(1)..=hi(1) {
                let zs: Vec<i64> = match x.abs() == r || y.abs() == r {
                    true => (lo(2)..=hi(2)).collect(),
                    false => [-r, r]
                        .into_iter()
                        .filter(|z| (lo(2)..=hi(2)).contains(z))
                        .collect(),
                };
                for z in zs {
                    self.scan([c[0] + x, c[1] + y, c[2] + z], q, radius2, heap);
                }
            }
        }
    }

    fn scan(&self, c: [i64; 3], q: &[f64; 3], radius2: f64, heap: &mut Heap) {
        if (0..3).any(|d| c[d] < 0 || c[d] >= self.dims[d]) {
            return;
        }
        let id = self.id(c);
        let packed = &self.entries[self.start[id] as usize..self.start[id + 1] as usize];
        work(packed.len() + self.added[id].len());
        for e in packed.iter().chain(&self.added[id]) {
            let d2 = (e.p[0] - q[0]).powi(2) + (e.p[1] - q[1]).powi(2) + (e.p[2] - q[2]).powi(2);
            if d2 <= radius2 {
                heap.push(Reverse((d2.to_bits(), e.i)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud(n: usize, seed: u64, scale: [f64; 3]) -> Vec<[f64; 3]> {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| std::array::from_fn(|d| next() * scale[d]))
            .collect()
    }

    fn d2(p: &[f64; 3], q: &[f64; 3]) -> f64 {
        (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)
    }

    fn brute(points: &[[f64; 3]], q: &[f64; 3], radius2: f64) -> Vec<(u64, usize)> {
        let mut all: Vec<(u64, usize)> = points
            .iter()
            .enumerate()
            .map(|(i, p)| (d2(p, q), i))
            .filter(|(d, _)| *d <= radius2)
            .map(|(d, i)| (d.to_bits(), i))
            .collect();
        all.sort_unstable();
        all
    }

    fn visited(grid: &Grid, q: &[f64; 3], radius2: f64) -> Vec<(u64, usize)> {
        let mut out = vec![];
        grid.nearest(q, radius2, |d, i| {
            out.push((d.to_bits(), i));
            false
        });
        out
    }

    #[test]
    fn visits_every_point_within_the_radius_nearest_first() {
        let shapes = [
            ([1000.0, 1000.0, 100.0], 3000),
            ([1000.0, 1000.0, 0.0], 2000),
            ([0.0, 0.0, 500.0], 500),
            ([3000.0, 20.0, 5.0], 1500),
        ];
        for (scale, n) in shapes {
            let points = cloud(n, 7, scale);
            let grid = Grid::new(&points);
            let targets = cloud(200, 11, scale.map(|s| s * 1.6 + 1.0));
            for t in &targets {
                let q = std::array::from_fn(|d| t[d] - 0.3 * scale[d] - 0.5);
                for radius2 in [0.0, 2500.0, 40_000.0, f64::INFINITY] {
                    assert_eq!(
                        visited(&grid, &q, radius2),
                        brute(&points, &q, radius2),
                        "{scale:?} {q:?} {radius2}"
                    );
                }
            }
        }
    }

    #[test]
    fn duplicates_come_in_index_order_and_nan_never() {
        let mut points = vec![[5.0, 5.0, 5.0]; 100];
        points.push([f64::NAN, 0.0, 0.0]);
        points.extend(cloud(50, 3, [10.0, 10.0, 10.0]));
        let grid = Grid::new(&points);
        let q = [5.0, 5.0, 5.0];
        let got = visited(&grid, &q, f64::INFINITY);
        assert_eq!(got, brute(&points, &q, f64::INFINITY));
        assert!(got.iter().all(|&(_, i)| i != 100));
        assert_eq!(
            got[..100].iter().map(|g| g.1).collect::<Vec<_>>(),
            (0..100).collect::<Vec<_>>()
        );
    }

    #[test]
    fn stops_when_the_visit_says_so() {
        let points = cloud(1000, 5, [100.0, 100.0, 100.0]);
        let grid = Grid::new(&points);
        let mut seen = 0;
        grid.nearest(&[50.0, 50.0, 50.0], f64::INFINITY, |_, _| {
            seen += 1;
            seen == 5
        });
        assert_eq!(seen, 5);
    }

    #[test]
    fn added_points_are_found_and_the_box_grows() {
        let mut points = cloud(300, 9, [100.0, 100.0, 10.0]);
        let mut grid = Grid::new(&points);
        let extra = cloud(3000, 13, [400.0, 400.0, 60.0]);
        for (k, p) in extra.iter().enumerate() {
            points.push(std::array::from_fn(|d| p[d] - [150.0, 150.0, 25.0][d]));
            grid.add(&points);
            if k % 97 == 0 {
                for q in [[0.0, 0.0, 0.0], [240.0, -140.0, 30.0], [50.0, 50.0, 5.0]] {
                    for radius2 in [400.0, 10_000.0, f64::INFINITY] {
                        assert_eq!(visited(&grid, &q, radius2), brute(&points, &q, radius2));
                    }
                }
            }
        }
    }
}
