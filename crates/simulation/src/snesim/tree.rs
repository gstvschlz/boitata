//! Search trees: the training-image patterns SNESIM draws from (Strebelle, 2002).
//!
//! A template is a list of offsets around a node, nearest first. A data event gives a
//! category to some of them. A [`SearchTree`], built by one scan of a training image,
//! answers without reading the image again: how many of its cells have this data event
//! around them, and which category do those cells hold?
//!
//! A node at depth `d` stands for one combination of categories at the first `d` offsets
//! and holds `k` counts, one per category of the cell itself. A cell counts at depth `d`
//! only if its first `d` offsets fall inside the image on cells with data. That leaves a
//! band along the edges, as wide as the template reaches, out of the deep nodes; when the
//! band differs from the rest, so do the counts. The tree therefore keeps how many cells
//! of each category reach each depth ([`SearchTree::cells_at_depth`]), and the counts of
//! a data event at depth `d` are weighted by `cells_at_depth(0)[c] / cells_at_depth(d)[c]`:
//! the patterns around a category come from the cells that have them, its share from the
//! whole image.

use crate::training_image::NO_CODE;

/// The training-image patterns of one template, as replicate counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SearchTree {
    k: usize,
    /// `counts[node * k + c]`: cells of category `c` with the node's data event around
    /// them. Node 0 is the root, the empty data event.
    counts: Vec<u32>,
    /// `children[node * k + c]`: the node adding category `c` at the next offset, 0 when
    /// the image lacks that pattern.
    children: Vec<u32>,
    /// `cells_at_depth[d * k + c]`: cells of category `c` counted at depth `d`.
    cells_at_depth: Vec<u32>,
}

/// Working memory of [`SearchTree::lookup`]; one per thread.
#[derive(Debug, Clone, Default)]
pub(crate) struct TreeScratch {
    frontier: Vec<u32>,
    next: Vec<u32>,
    /// `k` counts per data event kept.
    counts: Vec<u32>,
    /// Depth of each data event kept: one more than the index of its last offset.
    depths: Vec<usize>,
}

impl SearchTree {
    /// The tree of `codes` (x fastest, [`NO_CODE`] without data) of size `dims`, with `k`
    /// categories, for `offsets`. The caller bounds the node count with
    /// [`SearchTree::bound`] first.
    pub fn build(codes: &[u8], dims: [usize; 3], k: usize, offsets: &[[i32; 3]]) -> Self {
        let mut tree = Self {
            k,
            counts: vec![0; k],
            children: vec![0; k],
            cells_at_depth: vec![0; k * (offsets.len() + 1)],
        };
        let [nx, ny, nz] = dims.map(|d| d as i64);
        let position = |x: i64, y: i64, z: i64| (x + nx * (y + ny * z)) as usize;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let centre = codes[position(x, y, z)];
                    if centre == NO_CODE {
                        continue;
                    }
                    let c = usize::from(centre);
                    let mut node = 0;
                    tree.counts[c] += 1;
                    tree.cells_at_depth[c] += 1;
                    for (depth, offset) in (1..).zip(offsets) {
                        let [qx, qy, qz] = [
                            x + i64::from(offset[0]),
                            y + i64::from(offset[1]),
                            z + i64::from(offset[2]),
                        ];
                        let inside =
                            (0..nx).contains(&qx) && (0..ny).contains(&qy) && (0..nz).contains(&qz);
                        if !inside {
                            break;
                        }
                        let value = codes[position(qx, qy, qz)];
                        if value == NO_CODE {
                            break;
                        }
                        node = tree.child_or_new(node, value);
                        tree.counts[node * k + c] += 1;
                        tree.cells_at_depth[depth * k + c] += 1;
                    }
                }
            }
        }
        tree.counts.shrink_to_fit();
        tree.children.shrink_to_fit();
        tree
    }

    fn child_or_new(&mut self, node: usize, value: u8) -> usize {
        let slot = node * self.k + usize::from(value);
        if self.children[slot] == 0 {
            self.children[slot] = (self.counts.len() / self.k) as u32;
            self.counts.resize(self.counts.len() + self.k, 0);
            self.children.resize(self.children.len() + self.k, 0);
        }
        self.children[slot] as usize
    }

    /// Most nodes the tree of an image of `dims` with `n_valid` cells with data and `k`
    /// categories can have for `offsets`: over the depths, the smaller of `k^d` and the
    /// cells whose first `d` offsets fall inside the image.
    pub fn bound(dims: [usize; 3], n_valid: usize, k: usize, offsets: &[[i32; 3]]) -> u64 {
        let (mut low, mut high) = ([0i64; 3], [0i64; 3]);
        let mut combinations = 1u64;
        let mut nodes = 1u64;
        for offset in offsets {
            for a in 0..3 {
                low[a] = low[a].min(i64::from(offset[a]));
                high[a] = high[a].max(i64::from(offset[a]));
            }
            let cells: u64 = (0..3)
                .map(|a| (dims[a] as i64 - (high[a] - low[a])).max(0) as u64)
                .product();
            combinations = combinations.saturating_mul(k as u64);
            nodes = nodes.saturating_add(combinations.min(cells).min(n_valid as u64));
        }
        nodes
    }

    /// Bytes of a node with `k` categories: a `u32` count and a `u32` child per category.
    pub fn node_bytes(k: usize) -> u64 {
        8 * k as u64
    }

    /// Cells of each category counted at `depth`; depth 0 holds every cell with data.
    pub fn cells_at_depth(&self, depth: usize) -> &[u32] {
        &self.cells_at_depth[depth * self.k..][..self.k]
    }

    #[cfg(test)]
    pub fn n_nodes(&self) -> u64 {
        (self.counts.len() / self.k) as u64
    }

    /// The replicate counts of `event` (a category per template offset, [`NO_CODE`] where
    /// uninformed): `k` counts for the empty data event, then for the first informed
    /// offset, the first two, and so on while at least `min_replicates` replicates
    /// remain. The last item is `event` with its farthest informed offsets dropped until
    /// enough replicates remain; each comes with its depth, for
    /// [`SearchTree::cells_at_depth`].
    pub fn lookup<'s>(
        &self,
        event: &[u8],
        min_replicates: u32,
        scratch: &'s mut TreeScratch,
    ) -> impl DoubleEndedIterator<Item = (usize, &'s [u32])> {
        let k = self.k;
        scratch.counts.clear();
        scratch.counts.extend_from_slice(&self.counts[..k]);
        scratch.depths.clear();
        scratch.depths.push(0);
        scratch.frontier.clear();
        scratch.frontier.push(0);
        let informed = event.iter().rposition(|&v| v != NO_CODE);
        for (depth, &value) in (1..).zip(&event[..informed.map_or(0, |last| last + 1)]) {
            scratch.next.clear();
            if value == NO_CODE {
                // An uninformed offset agrees with every category.
                for &node in &scratch.frontier {
                    let children = &self.children[node as usize * k..][..k];
                    scratch.next.extend(children.iter().filter(|&&c| c != 0));
                }
                std::mem::swap(&mut scratch.frontier, &mut scratch.next);
                continue;
            }
            for &node in &scratch.frontier {
                let child = self.children[node as usize * k + usize::from(value)];
                if child != 0 {
                    scratch.next.push(child);
                }
            }
            std::mem::swap(&mut scratch.frontier, &mut scratch.next);
            let start = scratch.counts.len();
            scratch.counts.resize(start + k, 0);
            for &node in &scratch.frontier {
                let counts = &self.counts[node as usize * k..][..k];
                for (total, count) in scratch.counts[start..].iter_mut().zip(counts) {
                    *total += count;
                }
            }
            let replicates: u64 = scratch.counts[start..].iter().map(|&c| u64::from(c)).sum();
            if replicates < u64::from(min_replicates) || replicates == 0 {
                scratch.counts.truncate(start);
                break;
            }
            scratch.depths.push(depth);
        }
        scratch
            .depths
            .iter()
            .copied()
            .zip(scratch.counts.chunks_exact(k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEFT_RIGHT: [[i32; 3]; 2] = [[-1, 0, 0], [1, 0, 0]];
    const X: u8 = NO_CODE;

    fn lookup(tree: &SearchTree, event: &[u8], min_replicates: u32) -> Vec<Vec<u32>> {
        let mut scratch = TreeScratch::default();
        tree.lookup(event, min_replicates, &mut scratch)
            .map(|(_, counts)| counts.to_vec())
            .collect()
    }

    #[test]
    fn counts_the_replicates_of_each_data_event_by_central_category() {
        let tree = SearchTree::build(&[0, 1, 0, 1, 1, 0], [6, 1, 1], 2, &LEFT_RIGHT);
        assert_eq!(lookup(&tree, &[X, X], 1), [[3, 3]]);
        assert_eq!(lookup(&tree, &[0], 1), [vec![3, 3], vec![0, 2]]);
        assert_eq!(lookup(&tree, &[1, X], 1), [vec![3, 3], vec![2, 1]]);
        assert_eq!(
            lookup(&tree, &[1, 0], 1),
            [vec![3, 3], vec![2, 1], vec![0, 1]]
        );
        assert_eq!(lookup(&tree, &[X, 1], 1), [vec![3, 3], vec![1, 1]]);
    }

    #[test]
    fn each_depth_knows_the_cells_that_reach_it() {
        let tree = SearchTree::build(&[0, 1, 0, 1, 1, 0], [6, 1, 1], 2, &LEFT_RIGHT);
        assert_eq!(tree.cells_at_depth(0), [3, 3]);
        assert_eq!(tree.cells_at_depth(1), [2, 3]);
        assert_eq!(tree.cells_at_depth(2), [1, 3]);
        let mut scratch = TreeScratch::default();
        let depths: Vec<usize> = tree
            .lookup(&[X, 1], 1, &mut scratch)
            .map(|(d, _)| d)
            .collect();
        assert_eq!(depths, [0, 2]);
    }

    #[test]
    fn drops_the_farthest_informed_offsets_until_enough_replicates_remain() {
        let tree = SearchTree::build(&[0, 1, 0, 1, 1, 0], [6, 1, 1], 2, &LEFT_RIGHT);
        assert_eq!(lookup(&tree, &[1, 0], 2), [vec![3, 3], vec![2, 1]]);
        assert_eq!(lookup(&tree, &[1, 0], 4), [[3, 3]]);
        assert_eq!(lookup(&tree, &[1, 0], 100), [[3, 3]]);
        let tree = SearchTree::build(&[0, 0, 0, 1], [4, 1, 1], 2, &LEFT_RIGHT);
        assert_eq!(lookup(&tree, &[1, 0], 0), [[3, 1]]);
    }

    #[test]
    fn cells_without_data_are_neither_counted_nor_matched() {
        let tree = SearchTree::build(&[0, 1, X, 1, 0], [5, 1, 1], 2, &LEFT_RIGHT);
        assert_eq!(lookup(&tree, &[], 1), [[2, 2]]);
        assert_eq!(lookup(&tree, &[0], 1), [vec![2, 2], vec![0, 1]]);
        assert_eq!(lookup(&tree, &[1], 1), [vec![2, 2], vec![1, 0]]);
        assert_eq!(lookup(&tree, &[X, 1], 0), [[2, 2]]);
    }

    #[test]
    fn lookups_agree_with_counting_in_the_image() {
        let codes: Vec<u8> = (0..12 * 9u32)
            .map(|p| ((p % 12) * (p / 12 + 2) / 5 % 3) as u8)
            .collect();
        let offsets = [[0, -1, 0], [-1, 0, 0], [1, 0, 0], [0, 1, 0], [-2, 1, 0]];
        let tree = SearchTree::build(&codes, [12, 9, 1], 3, &offsets);
        let at = |x: i32, y: i32| {
            ((0..12).contains(&x) && (0..9).contains(&y)).then(|| codes[(x + 12 * y) as usize])
        };
        for key in 0..4u32.pow(5) {
            let event: Vec<u8> = (0..5)
                .map(|d| [0, 1, 2, X][(key / 4u32.pow(d) % 4) as usize])
                .collect();
            let Some(last) = event.iter().rposition(|&v| v != X) else {
                continue;
            };
            let mut expected = vec![0u32; 3];
            for y in 0..9 {
                for x in 0..12 {
                    let agrees = (0..=last).all(|d| {
                        let [dx, dy, _] = offsets[d];
                        at(x + dx, y + dy).is_some_and(|v| event[d] == X || event[d] == v)
                    });
                    if agrees {
                        expected[usize::from(at(x, y).unwrap())] += 1;
                    }
                }
            }
            let found = lookup(&tree, &event, 1);
            if expected.iter().sum::<u32>() > 0 {
                assert_eq!(found.last().unwrap(), &expected, "event {event:?}");
            }
        }
        let bound = SearchTree::bound([12, 9, 1], codes.len(), 3, &offsets);
        assert_eq!(bound, 1 + 3 + 9 + 27 + 70 + 63);
        assert!(tree.n_nodes() <= bound);
        assert_eq!(SearchTree::bound([6, 1, 1], 6, 2, &LEFT_RIGHT), 7);
    }
}
