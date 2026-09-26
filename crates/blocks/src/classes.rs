use ceres_core::{BlockModel, Layout};
use rayon::prelude::*;

use crate::error::{BlockModelError, Result};

/// Majority filter of `classes` (one per row of a regular or masked `model`)
/// over a `window` of cells centered on each block, applied `iterations` times.
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
    let index: Option<&[u64]> = match model.layout() {
        Layout::Regular => None,
        Layout::Masked(index) => Some(index),
        Layout::SubBlocked { .. } => return invalid("sub-blocked models are not supported"),
    };
    let geometry = model.geometry();
    let row = |parent: u64| match index {
        None => Some(parent as usize),
        Some(index) => index.binary_search(&parent).ok(),
    };
    let half = window.map(|w| (w / 2) as isize);
    let count = geometry.count.map(|n| n as isize);
    let mut current = classes.to_vec();
    for _ in 0..iterations {
        let next = (0..current.len())
            .into_par_iter()
            .map(|r| {
                let [i, j, k] = geometry.ijk(model.parent_index(r)).map(|v| v as isize);
                let mut votes: Vec<(u32, usize)> = vec![];
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
                            let Some(n) = row(geometry.index([a as usize, b as usize, c as usize]))
                            else {
                                continue;
                            };
                            if domains.is_some_and(|d| d[n] != d[r]) {
                                continue;
                            }
                            match votes.iter_mut().find(|v| v.0 == current[n]) {
                                Some(v) => v.1 += 1,
                                None => votes.push((current[n], 1)),
                            }
                        }
                    }
                }
                let most = votes.iter().map(|v| v.1).max().unwrap_or(0);
                let own = votes.iter().find(|v| v.0 == current[r]).map_or(0, |v| v.1);
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
