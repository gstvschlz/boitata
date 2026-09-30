use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZero;
use std::ops::Range;

use boitata_core::{BlockModel, Layout};
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use rayon::prelude::*;

use crate::error::{BlockModelError, Result};

fn invalid<T>(message: &str) -> Result<T> {
    Err(BlockModelError::InvalidGridParams(message.into()))
}

fn check(model: &BlockModel, classes: &[u32], domains: Option<&[u32]>) -> Result<()> {
    if classes.len() != model.len() {
        return invalid("one class per block");
    }
    if domains.is_some_and(|d| d.len() != model.len()) {
        return invalid("one domain per block");
    }
    Ok(())
}

/// Rows of `model` in parent cell `p`.
fn rows(model: &BlockModel, p: u64) -> Range<usize> {
    match model.layout() {
        Layout::Regular => p as usize..p as usize + 1,
        Layout::Masked(index) => index.binary_search(&p).map_or(0..0, |r| r..r + 1),
        Layout::SubBlocked { parent, .. } => {
            parent.partition_point(|&c| c < p)..parent.partition_point(|&c| c <= p)
        }
    }
}

/// Rows other than `r` in its parent cell or in the parent cells next to it:
/// sharing a face for `connectivity` 6, a face, edge or corner for 26.
fn neighbors(model: &BlockModel, r: usize, connectivity: usize) -> impl Iterator<Item = usize> {
    let geometry = model.geometry();
    let count = geometry.count.map(|n| n as isize);
    let [i, j, k] = geometry.ijk(model.parent_index(r)).map(|v| v as isize);
    (-1..=1isize)
        .flat_map(|dk| (-1..=1isize).flat_map(move |dj| (-1..=1isize).map(move |di| [di, dj, dk])))
        .filter(move |d| connectivity == 26 || d.iter().map(|v| v.abs()).sum::<isize>() <= 1)
        .map(move |[di, dj, dk]| [i + di, j + dj, k + dk])
        .filter(move |c| (0..3).all(|a| c[a] >= 0 && c[a] < count[a]))
        .flat_map(move |c| rows(model, geometry.index(c.map(|v| v as usize))))
        .filter(move |&s| s != r)
}

/// Connected units: a label per row, the lowest row of its unit, joining
/// neighboring rows (see [`neighbors`]) of the same class and domain.
fn units(
    model: &BlockModel,
    classes: &[u32],
    connectivity: usize,
    domains: Option<&[u32]>,
) -> Vec<usize> {
    let same =
        |r: usize, s: usize| classes[r] == classes[s] && domains.is_none_or(|d| d[r] == d[s]);
    let mut label = vec![usize::MAX; model.len()];
    for start in 0..model.len() {
        if label[start] != usize::MAX {
            continue;
        }
        label[start] = start;
        let mut stack = vec![start];
        while let Some(r) = stack.pop() {
            for s in neighbors(model, r, connectivity) {
                if label[s] == usize::MAX && same(r, s) {
                    label[s] = start;
                    stack.push(s);
                }
            }
        }
    }
    label
}

/// Smallest unit size kept by [`remove_small_units`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MinSize {
    Volume(f64),
    Blocks(usize),
}

/// Reclassify connected units of `classes` (one per row of `model`) smaller than
/// `min` into the class of their neighbors.
///
/// A unit joins rows of the same class (and of the same domain, with `domains`)
/// whose parent cells share a face (`connectivity` 6) or a face, edge or corner
/// (26); sub-blocks of one parent cell are neighbors. Units are taken smallest
/// first, ties by lowest row, and each takes the class with the most volume
/// among the blocks next to it, ties to the lowest class, merging with the
/// units of that class it touches; a merged unit still below `min` goes back
/// in line. Every unit left below `min` has no neighbor to join: it is
/// surrounded by absent cells or by other domains.
pub fn remove_small_units(
    model: &BlockModel,
    classes: &[u32],
    min: MinSize,
    connectivity: usize,
    domains: Option<&[u32]>,
) -> Result<Vec<u32>> {
    check(model, classes, domains)?;
    if connectivity != 6 && connectivity != 26 {
        return invalid("connectivity must be 6 or 26");
    }
    let volumes = model.volumes();
    let (size, threshold) = match min {
        MinSize::Volume(v) if v.is_finite() && v >= 0.0 => (volumes.clone(), v),
        MinSize::Blocks(b) => (vec![1.0; model.len()], b as f64),
        MinSize::Volume(_) => return invalid("min volume must be finite and not negative"),
    };
    let label = units(model, classes, connectivity, domains);
    let mut root: Vec<usize> = (0..model.len()).collect();
    let mut members: Vec<Vec<usize>> = vec![vec![]; model.len()];
    let mut total = vec![0.0; model.len()];
    for (r, &l) in label.iter().enumerate() {
        members[l].push(r);
        total[l] += size[r];
    }
    let find = |root: &mut Vec<usize>, mut u: usize| {
        while root[u] != u {
            root[u] = root[root[u]];
            u = root[u];
        }
        u
    };
    let mut line: BTreeSet<(u64, usize)> = (0..model.len())
        .filter(|&u| !members[u].is_empty() && total[u] < threshold)
        .map(|u| (total[u].to_bits(), u))
        .collect();
    let mut current = classes.to_vec();
    while let Some((_, u)) = line.pop_first() {
        let mut seen = BTreeSet::new();
        let mut weight: BTreeMap<u32, f64> = BTreeMap::new();
        for &r in &members[u] {
            for s in neighbors(model, r, connectivity) {
                if domains.is_none_or(|d| d[r] == d[s])
                    && current[s] != current[r]
                    && seen.insert(s)
                {
                    *weight.entry(current[s]).or_default() += volumes[s];
                }
            }
        }
        let Some(class) = weight
            .iter()
            .fold(None, |best: Option<(u32, f64)>, (&c, &w)| match best {
                Some((_, b)) if b >= w => best,
                _ => Some((c, w)),
            })
            .map(|b| b.0)
        else {
            continue;
        };
        let mut joined: BTreeSet<usize> = seen
            .into_iter()
            .filter(|&s| current[s] == class)
            .map(|s| find(&mut root, label[s]))
            .collect();
        for &r in &members[u] {
            current[r] = class;
        }
        joined.insert(u);
        let into = *joined
            .iter()
            .max_by_key(|&&v| (members[v].len(), std::cmp::Reverse(v)))
            .expect("u is in");
        for &v in &joined {
            line.remove(&(total[v].to_bits(), v));
        }
        for &v in &joined {
            if v != into {
                root[v] = into;
                let moved = std::mem::take(&mut members[v]);
                members[into].extend(moved);
                total[into] += total[v];
            }
        }
        if total[into] < threshold {
            line.insert((total[into].to_bits(), into));
        }
    }
    Ok(current)
}

/// Distance from each block's centroid to the nearest centroid of another class,
/// or with `target`, of the `target` class; blocks of `target` get the distance
/// to the nearest block outside it, negative when `signed`. NaN when no such
/// block exists. Distances are exact, between centroids, in any layout.
pub fn contact_distance(
    model: &BlockModel,
    classes: &[u32],
    target: Option<u32>,
    signed: bool,
) -> Result<Vec<f64>> {
    check(model, classes, None)?;
    let centroids = model.centroids();
    let tree = |keep: &dyn Fn(u32) -> bool| {
        let points: Vec<[f64; 3]> = centroids
            .iter()
            .zip(classes)
            .filter(|(_, c)| keep(**c))
            .map(|(p, _)| *p)
            .collect();
        (!points.is_empty())
            .then(|| ImmutableKdTree::<f64, 3>::new_from_slice(&points))
            .transpose()
            .map_err(|e| BlockModelError::InvalidGridParams(format!("{e:?}")))
    };
    let nearest = |tree: &Option<ImmutableKdTree<f64, 3>>, p: &[f64; 3]| {
        tree.as_ref().map_or(f64::NAN, |t| {
            t.query(p)
                .nearest_n::<SquaredEuclidean<f64>>(NonZero::<usize>::MIN)
                .execute()
                .first()
                .map_or(f64::NAN, |f| f.distance.sqrt())
        })
    };
    let distances = match target {
        Some(t) => {
            let inside = tree(&|c| c == t)?;
            let outside = tree(&|c| c != t)?;
            let sign = if signed { -1.0 } else { 1.0 };
            centroids
                .par_iter()
                .zip(classes)
                .map(|(p, &c)| match c == t {
                    true => sign * nearest(&outside, p),
                    false => nearest(&inside, p),
                })
                .collect()
        }
        None => {
            let names: BTreeSet<u32> = classes.iter().copied().collect();
            let trees = names
                .iter()
                .map(|&n| Ok((n, tree(&|c| c == n)?)))
                .collect::<Result<Vec<_>>>()?;
            centroids
                .par_iter()
                .zip(classes)
                .map(|(p, &c)| {
                    trees
                        .iter()
                        .filter(|(n, _)| *n != c)
                        .map(|(_, t)| nearest(t, p))
                        .fold(f64::NAN, f64::min)
                })
                .collect()
        }
    };
    Ok(distances)
}

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
    check(model, classes, domains)?;
    if window.iter().any(|&w| w % 2 == 0) {
        return invalid("window sizes must be odd");
    }
    let rows = |p: u64| rows(model, p);
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
    use boitata_core::Geometry;
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

    fn speckled(n: usize, classes: u32, seed: u64) -> Vec<u32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) % u64::from(classes)) as u32
            })
            .collect()
    }

    #[test]
    fn units_follow_the_connectivity() {
        // j = 0: 0 0 1 0
        // j = 1: 1 0 0 1
        // j = 2: 1 1 0 0
        let m = model([4, 3, 1]);
        let classes = [0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0];
        let faces = units(&m, &classes, 6, None);
        assert_eq!(faces, [0, 0, 2, 3, 4, 0, 0, 7, 4, 4, 0, 0]);
        let corners = units(&m, &classes, 26, None);
        assert_eq!(corners, [0, 0, 2, 0, 4, 0, 0, 2, 4, 4, 0, 0]);
        let domains = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1];
        let split = units(&m, &classes, 6, Some(&domains));
        assert_eq!(split, [0, 0, 2, 3, 4, 0, 0, 7, 8, 8, 10, 10]);
    }

    fn sizes(m: &BlockModel, classes: &[u32], connectivity: usize) -> BTreeMap<usize, f64> {
        let volumes = m.volumes();
        let mut out = BTreeMap::new();
        for (r, l) in units(m, classes, connectivity, None)
            .into_iter()
            .enumerate()
        {
            *out.entry(l).or_default() += volumes[r];
        }
        out
    }

    #[test]
    fn no_unit_is_left_below_the_minimum_and_volume_is_kept() {
        let geometry = Geometry {
            size: [2.0, 2.0, 1.0],
            ..geometry([30, 20, 3])
        };
        let m = BlockModel::regular(geometry, batch(1800)).unwrap();
        let classes = speckled(1800, 3, 7);
        for connectivity in [6, 26] {
            let before = sizes(&m, &classes, connectivity);
            let out = remove_small_units(&m, &classes, MinSize::Volume(40.0), connectivity, None)
                .unwrap();
            let after = sizes(&m, &out, connectivity);
            assert!(before.values().any(|&v| v < 40.0));
            assert!(after.values().all(|&v| v >= 40.0));
            assert_eq!(before.values().sum::<f64>(), after.values().sum::<f64>());
            let blocks =
                remove_small_units(&m, &classes, MinSize::Blocks(10), connectivity, None).unwrap();
            assert_eq!(blocks, out);
            let threads = |n: usize| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .unwrap()
                    .install(|| {
                        remove_small_units(&m, &classes, MinSize::Volume(40.0), connectivity, None)
                    })
                    .unwrap()
            };
            assert_eq!(threads(1), out);
            assert_eq!(threads(8), out);
        }
        assert!(remove_small_units(&m, &classes, MinSize::Blocks(2), 18, None).is_err());
        assert!(remove_small_units(&m, &classes, MinSize::Volume(-1.0), 6, None).is_err());
    }

    #[test]
    fn small_units_join_the_largest_neighbor_or_stay_alone() {
        // A speck touching as much of class 1 as of class 2 goes to 1; one
        // touching more of class 2 goes to 2.
        let m = model([3, 3, 1]);
        let tie = [1, 1, 1, 1, 3, 2, 2, 2, 2];
        let out = remove_small_units(&m, &tie, MinSize::Blocks(2), 6, None).unwrap();
        assert_eq!(out[4], 1);
        let more = [1, 1, 1, 2, 3, 2, 2, 2, 2];
        let out = remove_small_units(&m, &more, MinSize::Blocks(2), 6, None).unwrap();
        assert_eq!(out[4], 2);
        // An island of absent cells and a domain of its own keep their block.
        let masked = BlockModel::masked(geometry([4, 1, 1]), vec![0, 1, 3], batch(3)).unwrap();
        let out = remove_small_units(&masked, &[0, 0, 1], MinSize::Blocks(2), 26, None).unwrap();
        assert_eq!(out, [0, 0, 1]);
        let line = model([3, 1, 1]);
        let out =
            remove_small_units(&line, &[0, 0, 1], MinSize::Blocks(2), 6, Some(&[0, 0, 1])).unwrap();
        assert_eq!(out, [0, 0, 1]);
    }

    #[test]
    fn sub_blocks_count_by_volume() {
        // Parent 0 holds a 0.2 sub-block of class 1 next to 0.8 of class 0;
        // parent 1 is class 1. By volume the class-1 unit (1.2) survives min 1.1.
        let slab = |u0: f64, u1: f64| [u0, 0.0, 0.0, u1, 1.0, 1.0];
        let extent = vec![slab(0.0, 0.8), slab(0.8, 1.0), slab(0.0, 1.0)];
        let m = BlockModel::subblocked(geometry([2, 1, 1]), vec![0, 0, 1], extent, None, batch(3))
            .unwrap();
        let out = remove_small_units(&m, &[0, 1, 1], MinSize::Volume(1.1), 6, None).unwrap();
        assert_eq!(out, [1, 1, 1]);
        let out = remove_small_units(&m, &[0, 1, 1], MinSize::Blocks(2), 6, None).unwrap();
        assert_eq!(out, [1, 1, 1]);
    }

    #[test]
    fn contact_distance_matches_brute_force() {
        let g = Geometry {
            origin: [10.0, -5.0, 3.0],
            size: [2.0, 1.5, 1.0],
            count: [7, 6, 4],
            rotation: [30.0, 10.0, 0.0],
        };
        let index: Vec<u64> = (0..g.cells()).filter(|p| p % 5 != 2).collect();
        let n = index.len();
        let m = BlockModel::masked(g, index, batch(n)).unwrap();
        let classes = speckled(n, 3, 11);
        let c = m.centroids();
        let brute = |r: usize, other: &dyn Fn(u32) -> bool| {
            (0..n)
                .filter(|&s| other(classes[s]))
                .map(|s| {
                    (0..3)
                        .map(|a| (c[r][a] - c[s][a]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .fold(f64::NAN, f64::min)
        };
        let any = contact_distance(&m, &classes, None, true).unwrap();
        let to_one = contact_distance(&m, &classes, Some(1), true).unwrap();
        for r in 0..n {
            assert!((any[r] - brute(r, &|k| k != classes[r])).abs() < 1e-9);
            let expected = match classes[r] {
                1 => -brute(r, &|k| k != 1),
                _ => brute(r, &|k| k == 1),
            };
            assert!((to_one[r] - expected).abs() < 1e-9);
        }
        let unsigned = contact_distance(&m, &classes, Some(1), false).unwrap();
        assert!(unsigned.iter().zip(&to_one).all(|(u, s)| *u == s.abs()));
        let single = contact_distance(&m, &vec![0; n], None, true).unwrap();
        assert!(single.iter().all(|d| d.is_nan()));
    }

    #[test]
    fn contact_buffers_are_symmetric() {
        let m = model([10, 4, 3]);
        let classes: Vec<u32> = (0..120).map(|r| u32::from(r % 10 >= 5)).collect();
        for target in [None, Some(0), Some(1)] {
            let d = contact_distance(&m, &classes, target, true).unwrap();
            let band = |class: u32| {
                (0..120)
                    .filter(|&r| classes[r] == class && d[r].abs() <= 2.5)
                    .count()
            };
            assert_eq!(band(0), 24);
            assert_eq!(band(1), 24);
        }
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
