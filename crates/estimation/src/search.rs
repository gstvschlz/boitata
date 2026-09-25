//! Search neighborhoods for local estimation.
//!
//! Selects the samples used to estimate a target location, honoring an
//! anisotropic search ellipsoid, min/max sample counts, a per-hole cap, and
//! optional octant balancing. [`SearchTree`] answers the same query as
//! [`neighbors`] from a k-d tree, in logarithmic rather than linear time.

use std::collections::HashMap;
use std::num::NonZero;

use kiddo::{MutableKdTree, SquaredEuclidean};
use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};
use variogram::aniso::euclidean;
use variogram::{Anisotropy, Variogram};

use crate::Sample;
use crate::error::{EstimError, Result};

type Point = (f64, f64, f64);

/// Search-neighborhood parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Search {
    pub min_samples: usize,
    pub max_samples: usize,
    /// Search radius in the metric of the ellipsoid: metres along the major
    /// axis when the anisotropy ranges are ratios (major = 1).
    pub radius: f64,
    /// Maximum samples taken from any single drill hole (requires `Sample::hole`).
    pub max_per_hole: Option<usize>,
    /// Balance samples across 8 octants around the target.
    pub octant: bool,
    /// Search ellipsoid; without it the variogram's anisotropy is used.
    #[serde(default)]
    pub anisotropy: Option<Anisotropy>,
}

impl Default for Search {
    fn default() -> Self {
        Self {
            min_samples: 4,
            max_samples: 16,
            radius: f64::INFINITY,
            max_per_hole: None,
            octant: false,
            anisotropy: None,
        }
    }
}

fn metric<'a>(params: &'a Search, vg: Option<&'a Variogram>) -> Option<&'a Anisotropy> {
    params
        .anisotropy
        .as_ref()
        .or_else(|| vg.and_then(|v| v.anisotropy.as_ref()))
}

/// Octant index (0..8) of `sample` relative to `target`.
fn octant_of(target: &Point, s: &Point) -> usize {
    let bx = (s.0 >= target.0) as usize;
    let by = (s.1 >= target.1) as usize;
    let bz = (s.2 >= target.2) as usize;
    (bx << 2) | (by << 1) | bz
}

/// Greedy selection over candidates ordered by increasing distance.
fn select(
    target: &Point,
    ordered: impl IntoIterator<Item = usize>,
    loc: impl Fn(usize) -> Point,
    hole: impl Fn(usize) -> Option<u32>,
    params: &Search,
) -> Vec<usize> {
    let mut chosen = Vec::with_capacity(params.max_samples);
    let mut per_hole: HashMap<u32, usize> = HashMap::new();
    let mut per_octant = [0usize; 8];
    let octant_cap = if params.octant {
        params.max_samples.div_ceil(8)
    } else {
        usize::MAX
    };
    for idx in ordered {
        if chosen.len() >= params.max_samples {
            break;
        }
        if let (Some(cap), Some(h)) = (params.max_per_hole, hole(idx))
            && per_hole.get(&h).copied().unwrap_or(0) >= cap
        {
            continue;
        }
        if params.octant {
            let o = octant_of(target, &loc(idx));
            if per_octant[o] >= octant_cap {
                continue;
            }
            per_octant[o] += 1;
        }
        if let Some(h) = hole(idx) {
            *per_hole.entry(h).or_insert(0) += 1;
        }
        chosen.push(idx);
    }
    chosen
}

fn enough(chosen: Vec<usize>, params: &Search) -> Result<Vec<usize>> {
    if chosen.len() < params.min_samples {
        return Err(EstimError::SearchFailed(format!(
            "found {} samples, need at least {}",
            chosen.len(),
            params.min_samples
        )));
    }
    Ok(chosen)
}

/// Select neighbor sample indices for a target location by scanning every
/// sample; the reference for [`SearchTree`].
pub fn neighbors(
    target: &Point,
    samples: &[Sample],
    params: &Search,
    vg: Option<&Variogram>,
) -> Result<Vec<usize>> {
    let aniso = metric(params, vg);
    let mut cand: Vec<(usize, f64)> = samples
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let d = match aniso {
                Some(a) => a.lag(target, &s.loc),
                None => euclidean(target, &s.loc),
            };
            (i, d)
        })
        .filter(|(_, d)| *d <= params.radius)
        .collect();
    cand.sort_by(|a, b| a.1.total_cmp(&b.1));
    let chosen = select(
        target,
        cand.into_iter().map(|c| c.0),
        |i| samples[i].loc,
        |i| samples[i].hole,
        params,
    );
    enough(chosen, params)
}

/// Samples indexed in a k-d tree for repeated neighbourhood queries. Points
/// are stored in the search ellipsoid's frame, so Euclidean distance in the
/// tree is the anisotropic distance. Axes on which every point shares one
/// value (2D data, a single level) are left out of the tree and added back
/// exactly at query time. Samples can be added after construction.
pub struct SearchTree {
    index: Index,
    frame: Matrix3<f64>,
    points: Vec<[f64; 3]>,
    locs: Vec<Point>,
    holes: Vec<Option<u32>>,
    params: Search,
}

enum Index {
    Scan,
    One(MutableKdTree<f64, 1>, Flat),
    Two(MutableKdTree<f64, 2>, Flat),
    Three(MutableKdTree<f64, 3>),
}

/// Axes kept in the tree and the shared value of the others.
#[derive(Clone, Copy)]
struct Flat {
    active: [usize; 3],
    constant: [Option<f64>; 3],
}

impl Flat {
    fn of(points: &[[f64; 3]]) -> Self {
        let mut constant = [None; 3];
        for (d, c) in constant.iter_mut().enumerate() {
            let first = points[0][d];
            if points.iter().all(|p| p[d] == first) {
                *c = Some(first);
            }
        }
        let mut active = [0; 3];
        let mut n = 0;
        for d in 0..3 {
            if constant[d].is_none() {
                active[n] = d;
                n += 1;
            }
        }
        Self { active, constant }
    }

    fn keep<const K: usize>(&self, p: &[f64; 3]) -> [f64; K] {
        std::array::from_fn(|i| p[self.active[i]])
    }

    fn holds(&self, p: &[f64; 3]) -> bool {
        (0..3).all(|d| self.constant[d].is_none_or(|c| p[d] == c))
    }

    /// Squared distance from `q` to the plane or line of the points.
    fn offset(&self, q: &[f64; 3]) -> f64 {
        (0..3)
            .filter_map(|d| self.constant[d].map(|c| (q[d] - c).powi(2)))
            .sum()
    }
}

fn build<const K: usize>(points: &[[f64; 3]], flat: &Flat) -> Option<MutableKdTree<f64, K>> {
    let kept: Vec<[f64; K]> = points.iter().map(|p| flat.keep(p)).collect();
    MutableKdTree::new_from_slice(&kept).ok()
}

/// The `k` nearest within the radius, plus every point tied with the k-th so
/// that ties resolve by index exactly as in the scan.
fn nearest<const K: usize>(
    tree: &MutableKdTree<f64, K>,
    query: [f64; K],
    k: usize,
    radius2: f64,
) -> Vec<(f64, usize)> {
    let found = tree
        .query(&query)
        .nearest_n::<SquaredEuclidean<f64>>(NonZero::new(k).expect("k > 0"))
        .within(radius2)
        .execute();
    let found = match found.last() {
        Some(last) if found.len() == k => tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(last.distance)
            .execute(),
        _ => found,
    };
    found
        .iter()
        .map(|r| (r.distance, r.item as usize))
        .collect()
}

impl Index {
    fn build(points: &[[f64; 3]]) -> Self {
        if points.len() < 2 {
            return Index::Scan;
        }
        let flat = Flat::of(points);
        let built = match flat.constant.iter().filter(|c| c.is_none()).count() {
            1 => build::<1>(points, &flat).map(|t| Index::One(t, flat)),
            2 => build::<2>(points, &flat).map(|t| Index::Two(t, flat)),
            3 => build::<3>(points, &flat).map(Index::Three),
            _ => None,
        };
        built.unwrap_or(Index::Scan)
    }

    /// Adds point `item`; `false` when the index must be rebuilt.
    fn add(&mut self, point: &[f64; 3], item: usize) -> bool {
        let item = item as u32;
        match self {
            Index::Scan => false,
            Index::One(tree, flat) => {
                flat.holds(point) && tree.add(&flat.keep(point), item).is_ok()
            }
            Index::Two(tree, flat) => {
                flat.holds(point) && tree.add(&flat.keep(point), item).is_ok()
            }
            Index::Three(tree) => tree.add(point, item).is_ok(),
        }
    }
}

impl SearchTree {
    pub fn new(samples: &[Sample], params: &Search, vg: Option<&Variogram>) -> Self {
        let frame = metric(params, vg).map_or_else(Matrix3::identity, Anisotropy::matrix);
        let mut tree = Self {
            index: Index::Scan,
            frame,
            points: vec![],
            locs: samples.iter().map(|s| s.loc).collect(),
            holes: samples.iter().map(|s| s.hole).collect(),
            params: params.clone(),
        };
        tree.points = tree.locs.iter().map(|p| tree.project(p)).collect();
        tree.index = Index::build(&tree.points);
        tree
    }

    fn project(&self, p: &Point) -> [f64; 3] {
        let q = self.frame * Vector3::new(p.0, p.1, p.2);
        [q.x, q.y, q.z]
    }

    /// Adds a sample; its index is the number of samples before it.
    pub fn add(&mut self, sample: &Sample) {
        let point = self.project(&sample.loc);
        let item = self.points.len();
        self.points.push(point);
        self.locs.push(sample.loc);
        self.holes.push(sample.hole);
        if !self.index.add(&point, item) {
            self.index = Index::build(&self.points);
        }
    }

    pub fn len(&self) -> usize {
        self.locs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.locs.is_empty()
    }

    fn candidates(&self, query: &[f64; 3], k: usize, radius2: f64) -> Vec<(f64, usize)> {
        let reduced = |flat: &Flat| radius2 - flat.offset(query);
        match &self.index {
            Index::Scan => {
                let mut all: Vec<(f64, usize)> = self
                    .points
                    .iter()
                    .enumerate()
                    .map(|(i, p)| ((0..3).map(|d| (p[d] - query[d]).powi(2)).sum(), i))
                    .filter(|(d, _)| *d <= radius2)
                    .collect();
                all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                if let Some(&(last, _)) = all.get(k.saturating_sub(1)) {
                    all.retain(|a| a.0 <= last);
                }
                all
            }
            Index::One(tree, flat) if reduced(flat) >= 0.0 => {
                nearest(tree, flat.keep(query), k, reduced(flat))
            }
            Index::Two(tree, flat) if reduced(flat) >= 0.0 => {
                nearest(tree, flat.keep(query), k, reduced(flat))
            }
            Index::Three(tree) => nearest(tree, *query, k, radius2),
            _ => vec![],
        }
    }

    /// Same selection as [`neighbors`] over the indexed samples.
    pub fn neighbors(&self, target: &Point) -> Result<Vec<usize>> {
        let params = &self.params;
        if self.is_empty() || params.max_samples == 0 {
            return enough(vec![], params);
        }
        let capped = params.octant || params.max_per_hole.is_some();
        let query = self.project(target);
        let radius2 = params.radius * params.radius;
        let mut k = if capped {
            params.max_samples * 4
        } else {
            params.max_samples
        };
        loop {
            k = k.min(self.len());
            let mut found = self.candidates(&query, k, radius2);
            found.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let exhausted = found.len() < k || k == self.len();
            let chosen = select(
                target,
                found.into_iter().map(|f| f.1),
                |i| self.locs[i],
                |i| self.holes[i],
                params,
            );
            if chosen.len() >= params.max_samples || exhausted {
                return enough(chosen, params);
            }
            k *= 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use variogram::Angles;

    fn s(x: f64, y: f64, z: f64, v: f64, hole: Option<u32>) -> Sample {
        Sample {
            loc: (x, y, z),
            value: v,
            hole,
        }
    }

    fn params(min_samples: usize, max_samples: usize, radius: f64) -> Search {
        Search {
            min_samples,
            max_samples,
            radius,
            ..Default::default()
        }
    }

    #[test]
    fn selects_nearest_within_radius() {
        let samples = vec![
            s(1.0, 0.0, 0.0, 1.0, None),
            s(2.0, 0.0, 0.0, 2.0, None),
            s(100.0, 0.0, 0.0, 3.0, None),
        ];
        let p = params(1, 2, 10.0);
        let got = neighbors(&(0.0, 0.0, 0.0), &samples, &p, None).unwrap();
        assert_eq!(got, vec![0, 1]);
        let tree = SearchTree::new(&samples, &p, None);
        assert_eq!(tree.neighbors(&(0.0, 0.0, 0.0)).unwrap(), vec![0, 1]);
    }

    #[test]
    fn respects_max_per_hole() {
        let samples = vec![
            s(1.0, 0.0, 0.0, 1.0, Some(1)),
            s(2.0, 0.0, 0.0, 2.0, Some(1)),
            s(3.0, 0.0, 0.0, 3.0, Some(1)),
            s(4.0, 0.0, 0.0, 4.0, Some(2)),
        ];
        let p = Search {
            max_per_hole: Some(2),
            ..params(1, 10, 100.0)
        };
        let got = neighbors(&(0.0, 0.0, 0.0), &samples, &p, None).unwrap();
        assert_eq!(got, vec![0, 1, 3]);
    }

    #[test]
    fn errors_when_too_few() {
        let samples = vec![s(1.0, 0.0, 0.0, 1.0, None)];
        let p = params(5, 10, 100.0);
        assert!(neighbors(&(0.0, 0.0, 0.0), &samples, &p, None).is_err());
        assert!(
            SearchTree::new(&samples, &p, None)
                .neighbors(&(0.0, 0.0, 0.0))
                .is_err()
        );
    }

    #[test]
    fn tree_matches_the_scan() {
        let mut state = 12345u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let samples: Vec<Sample> = (0..3000)
            .map(|i| {
                let loc = (next() * 1000.0, next() * 1000.0, next() * 100.0);
                Sample {
                    loc,
                    value: 0.0,
                    hole: Some((i / 20) as u32),
                }
            })
            .collect();
        let aniso = Anisotropy::new(Angles {
            azimuth: 35.0,
            dip: 10.0,
            pitch: 0.0,
            major: 1.0,
            semi: 0.4,
            minor: 0.2,
        })
        .unwrap();
        let cases = [
            params(1, 16, 120.0),
            Search {
                octant: true,
                ..params(1, 24, 200.0)
            },
            Search {
                max_per_hole: Some(3),
                anisotropy: Some(aniso),
                ..params(1, 12, 150.0)
            },
        ];
        for p in &cases {
            let tree = SearchTree::new(&samples, p, None);
            for _ in 0..200 {
                let t = (next() * 1000.0, next() * 1000.0, next() * 100.0);
                assert_eq!(
                    tree.neighbors(&t).ok(),
                    neighbors(&t, &samples, p, None).ok()
                );
            }
        }
    }

    #[test]
    fn flat_and_duplicate_data_are_searched_exactly() {
        let mut samples: Vec<Sample> = (0..500)
            .map(|i| s((i % 25) as f64 * 3.0, (i / 25) as f64 * 3.0, 0.0, 0.0, None))
            .collect();
        samples.extend((0..100).map(|_| s(10.0, 10.0, 0.0, 0.0, None)));
        let p = params(1, 12, 20.0);
        let tree = SearchTree::new(&samples, &p, None);
        for t in [(5.0, 7.0, 0.0), (30.0, 40.0, 4.0), (10.0, 10.0, 0.0)] {
            let mut a = tree.neighbors(&t).unwrap();
            let mut b = neighbors(&t, &samples, &p, None).unwrap();
            a.sort();
            b.sort();
            assert_eq!(a.len(), b.len());
        }
        let line: Vec<Sample> = (0..200).map(|i| s(0.0, 0.0, i as f64, 0.0, None)).collect();
        let mut tree = SearchTree::new(&line, &p, None);
        assert_eq!(tree.neighbors(&(3.0, 4.0, 50.0)).unwrap()[0], 50);
        tree.add(&s(3.0, 4.0, 50.0, 0.0, None));
        assert_eq!(tree.neighbors(&(3.0, 4.0, 50.0)).unwrap()[0], 200);
    }

    #[test]
    fn tree_grows() {
        let p = params(1, 2, 10.0);
        let mut tree = SearchTree::new(&[s(5.0, 0.0, 0.0, 1.0, None)], &p, None);
        tree.add(&s(1.0, 0.0, 0.0, 2.0, None));
        assert_eq!(tree.neighbors(&(0.0, 0.0, 0.0)).unwrap(), vec![1, 0]);
    }
}
