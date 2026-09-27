use ceres_core::{BlockModel, Layout};
use rayon::prelude::*;

use crate::error::{BlockModelError, Result};

/// Majority filter of `classes` (one per row of `model`) over a `window` of
/// parent cells centered on each block's parent, applied `iterations` times.
/// Each block votes with its volume, so sub-blocks count by the fraction of the
/// parent cell they fill.
/// Blocks keep their class on a tie, and absent cells do not vote, so isolated
/// blocks take the class around them while contacts stay put. With `domains`
/// (one per row), only blocks of the same domain vote.
pub fn smooth_classes(
    model: &BlockModel,
    classes: &[u32],
    window: [usize; 3],
    iterations: usize,
    domains: Option<&[u32]>,
) -> Result<Vec<u32>> {
    let invalid = |m: &str| Err(BlockModelError::InvalidGridParams(m.into()));
    if classes.len() != model.len() {
        return invalid("one class per block");
    }
    if domains.is_some_and(|d| d.len() != model.len()) {
        return invalid("one domain per block");
    }
    if window.iter().any(|&w| w % 2 == 0) {
        return invalid("window sizes must be odd");
    }
    let rows = |p: u64| match model.layout() {
        Layout::Regular => p as usize..p as usize + 1,
        Layout::Masked(index) => index.binary_search(&p).map_or(0..0, |r| r..r + 1),
        Layout::SubBlocked { parent, .. } => {
            parent.partition_point(|&c| c < p)..parent.partition_point(|&c| c <= p)
        }
    };
    let geometry = model.geometry();
    let fraction: Vec<f64> = model
        .volumes()
        .iter()
        .map(|v| v / geometry.cell_volume())
        .collect();
    let half = window.map(|w| (w / 2) as isize);
    let count = geometry.count.map(|n| n as isize);
    let mut current = classes.to_vec();
    for _ in 0..iterations {
        let next = (0..current.len())
            .into_par_iter()
            .map(|r| {
                let [i, j, k] = geometry.ijk(model.parent_index(r)).map(|v| v as isize);
                let mut votes: Vec<(u32, f64)> = vec![];
                for dk in -half[2]..=half[2] {
                    for dj in -half[1]..=half[1] {
                        for di in -half[0]..=half[0] {
                            let (a, b, c) = (i + di, j + dj, k + dk);
                            if a < 0
                                || b < 0
                                || c < 0
                                || a >= count[0]
                                || b >= count[1]
                                || c >= count[2]
                            {
                                continue;
                            }
                            for n in rows(geometry.index([a as usize, b as usize, c as usize])) {
                                if domains.is_some_and(|d| d[n] != d[r]) {
                                    continue;
                                }
                                match votes.iter_mut().find(|v| v.0 == current[n]) {
                                    Some(v) => v.1 += fraction[n],
                                    None => votes.push((current[n], fraction[n])),
                                }
                            }
                        }
                    }
                }
                let most = votes.iter().map(|v| v.1).fold(0.0, f64::max);
                let own = votes
                    .iter()
                    .find(|v| v.0 == current[r])
                    .map_or(0.0, |v| v.1);
                if own == most {
                    current[r]
                } else {
                    votes
                        .iter()
                        .filter(|v| v.1 == most)
                        .map(|v| v.0)
                        .min()
                        .unwrap_or(current[r])
                }
            })
            .collect();
        current = next;
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, Float64Array, RecordBatch};
    use ceres_core::Geometry;
    use std::sync::Arc;

    fn geometry(count: [usize; 3]) -> Geometry {
        Geometry {
            origin: [0.0; 3],
            size: [1.0; 3],
            count,
            rotation: [0.0; 3],
        }
    }

    fn batch(n: usize) -> RecordBatch {
        let column: ArrayRef = Arc::new(Float64Array::from(vec![0.0; n]));
        RecordBatch::try_from_iter([("a", column)]).unwrap()
    }

    fn model(count: [usize; 3]) -> BlockModel {
        BlockModel::regular(geometry(count), batch(count.iter().product())).unwrap()
    }

    #[test]
    fn an_isolated_block_joins_its_neighbors_and_contacts_stay() {
        let m = model([5, 5, 1]);
        let mut classes: Vec<u32> = (0..25).map(|i| u32::from(i % 5 >= 3)).collect();
        classes[6] = 1;
        let out = smooth_classes(&m, &classes, [3, 3, 1], 1, None).unwrap();
        assert_eq!(out[6], 0);
        let expected: Vec<u32> = (0..25).map(|i| u32::from(i % 5 >= 3)).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn masked_cells_do_not_vote() {
        let m =
            BlockModel::masked(geometry([3, 3, 1]), vec![0, 2, 3, 5, 6, 7, 8], batch(7)).unwrap();
        let out = smooth_classes(&m, &[1, 1, 0, 0, 0, 0, 0], [3, 3, 1], 1, None).unwrap();
        assert_eq!(out.len(), 7);
        assert_eq!(out[..2], [1, 1]);
        assert!(smooth_classes(&m, &[0; 7], [2, 3, 1], 1, None).is_err());
    }

    #[test]
    fn sub_blocks_vote_with_their_volume() {
        // Parent 4 (center of 3x3) holds a quarter sub-block of class 1 and three
        // quarters of class 0; parent 0 is split in halves of class 1; the rest are 1.
        let quarter = |u: f64, v: f64| [u, v, 0.0, u + 0.5, v + 0.5, 1.0];
        let whole = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut parent = vec![0, 0];
        let mut extent = vec![
            [0.0, 0.0, 0.0, 0.5, 1.0, 1.0],
            [0.5, 0.0, 0.0, 1.0, 1.0, 1.0],
        ];
        let mut classes = vec![1, 1];
        for p in 1..9 {
            if p == 4 {
                for (u, v) in [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)] {
                    parent.push(4);
                    extent.push(quarter(u, v));
                    classes.push(u32::from(u == 0.0 && v == 0.0));
                }
            } else {
                parent.push(p);
                extent.push(whole);
                classes.push(1);
            }
        }
        let n = parent.len();
        let m = BlockModel::subblocked(
            geometry([3, 3, 1]),
            parent,
            extent,
            Some([2, 2, 1]),
            batch(n),
        )
        .unwrap();
        let out = smooth_classes(&m, &classes, [3, 3, 1], 1, None).unwrap();
        assert!(out.iter().all(|&c| c == 1));

        // One parent: 0.6 of it class 0, two sub-blocks of 0.2 class 1. By count the
        // two would win; by volume class 0 does.
        let slab = |u0: f64, u1: f64| [u0, 0.0, 0.0, u1, 1.0, 1.0];
        let extent = vec![slab(0.0, 0.6), slab(0.6, 0.8), slab(0.8, 1.0)];
        let m = BlockModel::subblocked(geometry([1, 1, 1]), vec![0; 3], extent, None, batch(3))
            .unwrap();
        let out = smooth_classes(&m, &[0, 1, 1], [1, 1, 1], 1, None).unwrap();
        assert_eq!(out, [0, 0, 0]);
    }

    #[test]
    fn only_blocks_of_the_same_domain_vote() {
        let m = model([5, 5, 1]);
        let domains: Vec<u32> = (0..25).map(|i| u32::from(i % 5 == 1)).collect();
        let classes = domains.clone();
        let mixed = smooth_classes(&m, &classes, [3, 3, 1], 1, None).unwrap();
        assert!(mixed.iter().all(|&c| c == 0));
        let apart = smooth_classes(&m, &classes, [3, 3, 1], 1, Some(&domains)).unwrap();
        assert_eq!(apart, classes);
        assert!(smooth_classes(&m, &classes, [3, 3, 1], 1, Some(&domains[..3])).is_err());
    }
}
