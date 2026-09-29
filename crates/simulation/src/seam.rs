//! Seams of image quilting: where a new patch takes over from the cells
//! already simulated.
//!
//! A new patch overlaps its neighbors. In the overlap each cell has an error,
//! how much the new patch differs from the value already there, and the seam
//! crosses the overlap where the two agree best, so the join shows least.
//!
//! - A flat (2D) patch takes the minimum-error boundary cut of Efros and
//!   Freeman (2001): [`path_cut`] finds, by dynamic programming, the cheapest
//!   path along each overlap.
//! - A 3D patch needs a surface, which dynamic programming cannot find.
//!   [`surface_cut`] finds the cheapest one as the minimum cut of a graph
//!   (Kwatra et al., 2003), by a maximum flow ([`CutGraph`]).

/// The cheapest path along a strip of `length` rows of `width` cells,
/// `errors[row * width + column]`, `width` at least 1: one cell per row,
/// moving by at most one column from row to row, costing the sum of its
/// errors. Returns the column of the path in each row.
pub fn path_seam(errors: &[f64], width: usize, length: usize) -> Vec<usize> {
    debug_assert_eq!(errors.len(), width * length);
    let reach = |column: usize| column.saturating_sub(1)..=(column + 1).min(width - 1);
    // Cost of the cheapest path from row 0 to each cell.
    let mut cheapest = errors.to_vec();
    for row in 1..length {
        for column in 0..width {
            let before = reach(column)
                .map(|c| cheapest[(row - 1) * width + c])
                .fold(f64::INFINITY, f64::min);
            cheapest[row * width + column] += before;
        }
    }
    // Walk back from the cheapest end; of equal costs the lowest column wins.
    let cheapest_of = |row: usize, columns: std::ops::RangeInclusive<usize>| {
        columns
            .min_by(|&a, &b| cheapest[row * width + a].total_cmp(&cheapest[row * width + b]))
            .unwrap_or(0)
    };
    let mut path = vec![0; length];
    for row in (0..length).rev() {
        path[row] = if row + 1 == length {
            cheapest_of(row, 0..=width - 1)
        } else {
            cheapest_of(row, reach(path[row + 1]))
        };
    }
    path
}

/// Which cells of a flat patch of `pu × pv` cells take the new patch, by the
/// minimum-error boundary cut of Efros and Freeman (2001).
///
/// `errors[u + pu * v]` is the error of each cell and `[ou, ov]` the overlap
/// along each axis at its low side, 0 without a neighbor there. The overlap
/// along `u` gets a path running along `v` ([`path_seam`]), the overlap along
/// `v` one running along `u`; a cell takes the new patch when it lies on the
/// new side of both paths, the paths included.
pub fn path_cut(errors: &[f64], [pu, pv]: [usize; 2], [ou, ov]: [usize; 2]) -> Vec<bool> {
    debug_assert_eq!(errors.len(), pu * pv);
    let along_v = if ou == 0 {
        vec![0; pv]
    } else {
        let strip: Vec<f64> = (0..pv)
            .flat_map(|v| errors[pu * v..pu * v + ou].iter().copied())
            .collect();
        path_seam(&strip, ou, pv)
    };
    let along_u = if ov == 0 {
        vec![0; pu]
    } else {
        let strip: Vec<f64> = (0..pu)
            .flat_map(|u| (0..ov).map(move |v| errors[u + pu * v]))
            .collect();
        path_seam(&strip, ov, pu)
    };
    (0..pu * pv)
        .map(|cell| {
            let (u, v) = (cell % pu, cell / pu);
            u >= along_v[v] && v >= along_u[u]
        })
        .collect()
}

/// What a cell is to [`surface_cut`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Simulated and outside the new patch: it keeps its value.
    Old,
    /// Simulated and inside the new patch: the cut decides.
    Overlap,
    /// Not simulated yet and inside the new patch: it takes the new patch.
    New,
    /// No cell to join: outside the grid, not a node, or not simulated and
    /// outside the patch.
    Nothing,
}

/// Which cells of a box of `shape` take the new patch, by the cheapest
/// surface through the overlap.
///
/// `sides[x + nx * (y + ny * z)]` says what each cell is and `errors` holds
/// the error of each [`Side::Overlap`] cell. Wherever two neighbors (along x,
/// y or z) end on different sides the join costs the errors of both; an old
/// or new cell, whose error is unknown, counts as much as the overlap cell
/// next to it, so the cut is not drawn to the rim of the overlap. The cut of
/// least total cost is a minimum cut, found by a maximum flow ([`CutGraph`]).
///
/// Returns true for the cells that take the new patch.
pub fn surface_cut(sides: &[Side], errors: &[f64], shape: [usize; 3]) -> Vec<bool> {
    let [nx, ny, nz] = shape;
    debug_assert_eq!(sides.len(), nx * ny * nz);
    let mut node_of = vec![usize::MAX; sides.len()];
    let mut n_nodes = 0;
    for (cell, side) in sides.iter().enumerate() {
        if *side == Side::Overlap {
            node_of[cell] = n_nodes;
            n_nodes += 1;
        }
    }
    let mut graph = CutGraph::new(n_nodes);
    let strides = [1, nx, nx * ny];
    for cell in 0..sides.len() {
        let ijk = [cell % nx, cell / nx % ny, cell / (nx * ny)];
        for axis in 0..3 {
            if ijk[axis] + 1 == shape[axis] {
                continue;
            }
            let next = cell + strides[axis];
            match (sides[cell], sides[next]) {
                (Side::Overlap, Side::Overlap) => {
                    graph.link(node_of[cell], node_of[next], errors[cell] + errors[next]);
                }
                (Side::Overlap, Side::Old) => graph.link_old(node_of[cell], 2.0 * errors[cell]),
                (Side::Old, Side::Overlap) => graph.link_old(node_of[next], 2.0 * errors[next]),
                (Side::Overlap, Side::New) => graph.link_new(node_of[cell], 2.0 * errors[cell]),
                (Side::New, Side::Overlap) => graph.link_new(node_of[next], 2.0 * errors[next]),
                _ => {}
            }
        }
    }
    let new_side = graph.cut();
    sides
        .iter()
        .zip(&node_of)
        .map(|(side, &node)| match side {
            Side::Overlap => new_side[node],
            Side::New => true,
            Side::Old | Side::Nothing => false,
        })
        .collect()
}

const NO_ARC: usize = usize::MAX;

/// A graph between an "old" and a "new" terminal, for the minimum cut that
/// separates them.
///
/// The minimum cut costs as much as the maximum flow between the terminals
/// (Ford and Fulkerson, 1956), found here by Dinic's algorithm (1970), which
/// pushes flow along shortest paths a level at a time, without recursion.
///
/// Each link is two arcs, `2 k` and `2 k + 1`: the reverse of arc `a` is
/// `a ^ 1`, and `residual[a]` is the flow arc `a` still takes.
#[derive(Debug, Clone)]
pub struct CutGraph {
    /// The first arc out of each node, then the next, until `NO_ARC`.
    first: Vec<usize>,
    next: Vec<usize>,
    to: Vec<usize>,
    residual: Vec<f64>,
}

impl CutGraph {
    /// A graph of `n_cells` nodes without links; the old terminal is node
    /// `n_cells`, the new one `n_cells + 1`.
    pub fn new(n_cells: usize) -> Self {
        Self {
            first: vec![NO_ARC; n_cells + 2],
            next: Vec::new(),
            to: Vec::new(),
            residual: Vec::new(),
        }
    }

    fn old(&self) -> usize {
        self.first.len() - 2
    }

    fn new_terminal(&self) -> usize {
        self.first.len() - 1
    }

    fn arcs(&mut self, a: usize, b: usize, capacity: f64, reverse_capacity: f64) {
        for (from, to, capacity) in [(a, b, capacity), (b, a, reverse_capacity)] {
            self.next.push(self.first[from]);
            self.first[from] = self.to.len();
            self.to.push(to);
            self.residual.push(capacity);
        }
    }

    /// Cells `a` and `b` are neighbors: separating them costs `cost`.
    pub fn link(&mut self, a: usize, b: usize, cost: f64) {
        self.arcs(a, b, cost, cost);
    }

    /// `cell` is next to an old cell: giving it to the new side costs `cost`.
    pub fn link_old(&mut self, cell: usize, cost: f64) {
        self.arcs(self.old(), cell, cost, 0.0);
    }

    /// `cell` is next to a new cell: keeping it old costs `cost`.
    pub fn link_new(&mut self, cell: usize, cost: f64) {
        self.arcs(cell, self.new_terminal(), cost, 0.0);
    }

    /// The minimum cut: true for the cells on the new side. A cell that could
    /// go either way at the same cost goes to the new side.
    pub fn cut(mut self) -> Vec<bool> {
        let (old, new) = (self.old(), self.new_terminal());
        loop {
            let level = self.levels();
            if level[new] == usize::MAX {
                // The cells still reached from the old terminal are its side.
                return level[..old].iter().map(|&l| l == usize::MAX).collect();
            }
            // Each node's next arc to try at this level.
            let mut arc = self.first.clone();
            let mut path: Vec<usize> = Vec::new();
            let mut node = old;
            loop {
                if node == new {
                    let pushed = path
                        .iter()
                        .map(|&a| self.residual[a])
                        .fold(f64::INFINITY, f64::min);
                    for &a in &path {
                        self.residual[a] -= pushed;
                        self.residual[a ^ 1] += pushed;
                    }
                    path.clear();
                    node = old;
                    continue;
                }
                let a = arc[node];
                if a == NO_ARC {
                    // A dead end: step back and drop the arc that led here.
                    let Some(last) = path.pop() else { break };
                    node = self.to[last ^ 1];
                    arc[node] = self.next[last];
                } else if self.residual[a] > 0.0 && level[self.to[a]] == level[node] + 1 {
                    path.push(a);
                    node = self.to[a];
                } else {
                    arc[node] = self.next[a];
                }
            }
        }
    }

    /// Fewest arcs with room for flow from the old terminal to each node;
    /// `usize::MAX` where none reaches.
    fn levels(&self) -> Vec<usize> {
        let mut level = vec![usize::MAX; self.first.len()];
        let old = self.old();
        level[old] = 0;
        let mut queue = std::collections::VecDeque::from([old]);
        while let Some(node) = queue.pop_front() {
            let mut a = self.first[node];
            while a != NO_ARC {
                let to = self.to[a];
                if self.residual[a] > 0.0 && level[to] == usize::MAX {
                    level[to] = level[node] + 1;
                    queue.push_back(to);
                }
                a = self.next[a];
            }
        }
        level
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;

    /// Squared differences of a continuous image, or (`whole`) the 0 and 1
    /// of a categorical one, where equal costs are common.
    fn random_errors(rng: &mut StdRng, n: usize, whole: bool) -> Vec<f64> {
        (0..n)
            .map(|_| match whole {
                true => f64::from(rng.gen_range(0..2u8)),
                false => rng.r#gen::<f64>().powi(2),
            })
            .collect()
    }

    fn cheapest_path(errors: &[f64], width: usize, row: usize, column: usize) -> f64 {
        let here = errors[row * width + column];
        if (row + 1) * width == errors.len() {
            return here;
        }
        let next = (column.saturating_sub(1)..=(column + 1).min(width - 1))
            .map(|c| cheapest_path(errors, width, row + 1, c))
            .fold(f64::INFINITY, f64::min);
        here + next
    }

    #[test]
    fn the_path_seam_costs_the_minimum_over_all_paths() {
        let mut rng = StdRng::seed_from_u64(1);
        for case in 0..200 {
            let width = rng.gen_range(1..5);
            let length = rng.gen_range(1..7);
            let errors = random_errors(&mut rng, width * length, case % 2 == 0);
            let path = path_seam(&errors, width, length);
            assert_eq!(path.len(), length);
            assert!(path.iter().all(|&column| column < width));
            assert!(path.windows(2).all(|p| p[0].abs_diff(p[1]) <= 1));
            let cost: f64 = (0..length).map(|row| errors[row * width + path[row]]).sum();
            let brute = (0..width)
                .map(|column| cheapest_path(&errors, width, 0, column))
                .fold(f64::INFINITY, f64::min);
            assert!((cost - brute).abs() < 1e-12, "{cost} vs {brute}");
        }
    }

    #[test]
    fn a_path_cut_keeps_the_old_cells_before_both_paths() {
        // Zero errors in column 1 of a 4 x 3 patch overlapping by 2 along u:
        // the path follows it and only column 0 stays old.
        let mut errors = vec![1.0; 12];
        for v in 0..3 {
            errors[1 + 4 * v] = 0.0;
        }
        let new = path_cut(&errors, [4, 3], [2, 0]);
        let expected: Vec<bool> = (0..12).map(|cell| cell % 4 >= 1).collect();
        assert_eq!(new, expected);
        assert!(path_cut(&errors, [4, 3], [0, 1]).iter().all(|&new| new));
        assert!(path_cut(&errors, [4, 3], [0, 0]).iter().all(|&new| new));
    }

    /// Cost of sharing the overlap out as `new_side` says.
    fn cut_cost(sides: &[Side], errors: &[f64], shape: [usize; 3], new_side: &[bool]) -> f64 {
        let [nx, ny, _] = shape;
        let strides = [1, nx, nx * ny];
        let error = |cell: usize, other: usize| match (sides[cell], sides[other]) {
            (Side::Overlap, _) => errors[cell],
            (_, Side::Overlap) => errors[other],
            _ => 0.0,
        };
        let mut cost = 0.0;
        for cell in 0..sides.len() {
            let ijk = [cell % nx, cell / nx % ny, cell / (nx * ny)];
            for axis in 0..3 {
                let next = cell + strides[axis];
                if ijk[axis] + 1 == shape[axis]
                    || sides[cell] == Side::Nothing
                    || sides[next] == Side::Nothing
                {
                    continue;
                }
                if new_side[cell] != new_side[next] {
                    cost += error(cell, next) + error(next, cell);
                }
            }
        }
        cost
    }

    #[test]
    fn the_surface_cut_costs_the_minimum_over_all_cuts() {
        let mut rng = StdRng::seed_from_u64(2);
        for case in 0..60 {
            let shape = [
                rng.gen_range(2..5),
                rng.gen_range(2..4),
                rng.gen_range(1..4),
            ];
            let n = shape.iter().product();
            let mut n_overlap = 0;
            let sides: Vec<Side> = (0..n)
                .map(|_| match rng.gen_range(0..8) {
                    0 | 1 => Side::Old,
                    2 | 3 => Side::New,
                    4 => Side::Nothing,
                    _ if n_overlap < 12 => {
                        n_overlap += 1;
                        Side::Overlap
                    }
                    _ => Side::New,
                })
                .collect();
            let errors = random_errors(&mut rng, n, case % 2 == 0);
            let new_side = surface_cut(&sides, &errors, shape);
            for (side, &new) in sides.iter().zip(&new_side) {
                match side {
                    Side::New => assert!(new),
                    Side::Old | Side::Nothing => assert!(!new),
                    Side::Overlap => {}
                }
            }
            let cost = cut_cost(&sides, &errors, shape, &new_side);
            let brute = (0..1u32 << n_overlap)
                .map(|cut| {
                    let mut k = 0;
                    let tried: Vec<bool> = sides
                        .iter()
                        .map(|side| match side {
                            Side::Overlap => {
                                k += 1;
                                cut >> (k - 1) & 1 == 1
                            }
                            Side::New => true,
                            Side::Old | Side::Nothing => false,
                        })
                        .collect();
                    cut_cost(&sides, &errors, shape, &tried)
                })
                .fold(f64::INFINITY, f64::min);
            assert!(
                (cost - brute).abs() < 1e-9,
                "case {case}: {cost} vs {brute}"
            );
        }
    }
}
