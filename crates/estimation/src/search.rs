//! Search neighborhoods for local estimation.
//!
//! Selects the samples used to estimate a target location, honoring an
//! anisotropic search ellipsoid, min/max sample counts, a per-hole cap, and
//! optional octant balancing. [`SearchTree`] answers the same query as
//! [`neighbors`] from a k-d tree, in logarithmic rather than linear time.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::num::NonZero;

use kiddo::{ImmutableKdTree, SquaredEuclidean};
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
    #[serde(with = "ceres_core::nonfinite")]
    pub radius: f64,
    /// Maximum samples taken from any single drill hole (requires `Sample::hole`).
    pub max_per_hole: Option<usize>,
    /// Balance samples across 8 octants around the target.
    pub octant: bool,
    /// Search ellipsoid; without it the variogram's anisotropy is used.
    #[serde(default)]
    pub anisotropy: Option<Anisotropy>,
    /// Samples above a threshold only inform targets within a smaller radius.
    #[serde(default)]
    pub high_grade: Option<HighGrade>,
    /// Samples of another domain than the target's only inform it within a
    /// distance; without it domain boundaries are hard.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub soft: Option<Soft>,
}

/// Soft domain boundaries: a sample of another domain informs a target only
/// when strictly closer than the distance of their pair, measured in the
/// same ellipsoid as [`Search::radius`], so 0 is a hard boundary and
/// infinity pools the domains. Pairs not listed are hard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Soft<L = u32> {
    /// One distance for every pair of domains, both ways.
    All(#[serde(with = "ceres_core::nonfinite")] f64),
    /// Distances from a target's domain to a sample's domain, one way.
    Pairs(Vec<SoftPair<L>>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoftPair<L = u32> {
    pub target: L,
    pub sample: L,
    #[serde(with = "ceres_core::nonfinite")]
    pub distance: f64,
}

impl<L: PartialEq> Soft<L> {
    /// Distance within which samples of domain `sample` inform targets of
    /// domain `target`; 0 when the pair is hard.
    pub fn distance(&self, target: &L, sample: &L) -> f64 {
        match self {
            Soft::All(d) => *d,
            Soft::Pairs(pairs) => pairs
                .iter()
                .find(|p| p.target == *target && p.sample == *sample)
                .map_or(0.0, |p| p.distance),
        }
    }
}

/// Samples valued above `threshold` are used only within `radius`, measured in
/// the same ellipsoid as [`Search::radius`]. `threshold` is in data units;
/// simulators working on normal scores convert it through their transform.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HighGrade {
    #[serde(with = "ceres_core::nonfinite")]
    pub threshold: f64,
    #[serde(with = "ceres_core::nonfinite")]
    pub radius: f64,
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
            high_grade: None,
            soft: None,
        }
    }
}

impl Search {
    /// Whether a sample of `value` and `domain` at `distance` may inform a
    /// target of domain `target` under the high-grade and soft-boundary rules.
    fn admits(&self, target: Option<u32>, value: f64, domain: Option<u32>, distance: f64) -> bool {
        let high = self
            .high_grade
            .is_some_and(|h| value > h.threshold && distance > h.radius);
        let reached = match (target, domain) {
            (Some(t), Some(s)) if t != s => {
                distance < self.soft.as_ref().map_or(0.0, |r| r.distance(&t, &s))
            }
            _ => true,
        };
        !high && reached
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

/// Ordered candidates with samples of different domains at one location
/// reduced to one, which takes the place of the first: the one of the
/// target's `domain` if any, otherwise the first. `kept` is passed through
/// untouched. Nothing changes without a target domain.
fn one_per_location(
    ordered: impl IntoIterator<Item = usize>,
    loc: impl Fn(usize) -> Point,
    of: impl Fn(usize) -> Option<u32>,
    domain: Option<u32>,
    kept: Option<usize>,
) -> Vec<usize> {
    let ordered = ordered.into_iter();
    if domain.is_none() {
        return ordered.collect();
    }
    let key = |i: usize| {
        let (x, y, z) = loc(i);
        [x + 0.0, y + 0.0, z + 0.0].map(f64::to_bits)
    };
    let mut at: HashMap<[u64; 3], usize> = HashMap::new();
    let mut out = Vec::new();
    for i in ordered {
        if kept == Some(i) {
            out.push(i);
            continue;
        }
        match at.entry(key(i)) {
            Entry::Vacant(e) => {
                e.insert(out.len());
                out.push(i);
            }
            Entry::Occupied(e) => {
                let first = &mut out[*e.get()];
                if of(*first) != domain && of(i) == domain {
                    *first = i;
                }
            }
        }
    }
    out
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
    neighbors_in(target, None, samples, params, vg)
}

/// As [`neighbors`] for a target of `domain`: samples of other domains are
/// kept only within [`Search::soft`].
pub fn neighbors_in(
    target: &Point,
    domain: Option<u32>,
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
        .filter(|&(i, d)| {
            let s = &samples[i];
            d <= params.radius && params.admits(domain, s.value, s.domain, d)
        })
        .collect();
    cand.sort_by(|a, b| a.1.total_cmp(&b.1));
    let ordered = one_per_location(
        cand.into_iter().map(|c| c.0),
        |i| samples[i].loc,
        |i| samples[i].domain,
        domain,
        None,
    );
    let chosen = select(
        target,
        ordered,
        |i| samples[i].loc,
        |i| samples[i].hole,
        params,
    );
    enough(chosen, params)
}

/// Samples indexed in k-d trees for repeated neighbourhood queries. Points
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
    values: Vec<f64>,
    domains: Vec<Option<u32>>,
    params: Search,
}

fn project(frame: &Matrix3<f64>, p: &Point) -> [f64; 3] {
    let q = frame * Vector3::new(p.0, p.1, p.2);
    [q.x, q.y, q.z]
}

/// Added points are scanned until this many, then indexed in a block.
const SCANNED: usize = 64;

/// Consecutive points in blocks of decreasing size, one tree each, and the
/// points after the last block scanned. A new block absorbs the blocks before
/// it that are no larger, so each point is re-indexed O(log n) times.
struct Index {
    blocks: Vec<Block>,
    flat: Flat,
}

struct Block {
    start: usize,
    tree: Tree,
}

enum Tree {
    One(ImmutableKdTree<f64, 1>),
    Two(ImmutableKdTree<f64, 2>),
    Three(ImmutableKdTree<f64, 3>),
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
        if let Some(first) = points.first() {
            for (d, c) in constant.iter_mut().enumerate() {
                if points.iter().all(|p| p[d] == first[d]) {
                    *c = Some(first[d]);
                }
            }
        }
        if constant.iter().all(Option::is_some) {
            constant = [None; 3];
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

    fn dims(&self) -> usize {
        self.constant.iter().filter(|c| c.is_none()).count()
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

    /// Squared distance along the kept axes.
    fn distance(&self, p: &[f64; 3], q: &[f64; 3]) -> f64 {
        self.active[..self.dims()]
            .iter()
            .map(|&d| (p[d] - q[d]).powi(2))
            .sum()
    }
}

fn build<const K: usize>(points: &[[f64; 3]], flat: &Flat) -> Option<ImmutableKdTree<f64, K>> {
    let kept: Vec<[f64; K]> = points.iter().map(|p| flat.keep(p)).collect();
    ImmutableKdTree::new_from_slice(&kept).ok()
}

/// The `k` nearest within the radius, plus every point tied with the k-th so
/// that ties resolve by index exactly as in the scan; every point within the
/// radius when `k` is 0.
fn nearest<const K: usize>(
    tree: &ImmutableKdTree<f64, K>,
    query: [f64; K],
    k: usize,
    radius2: f64,
) -> Vec<(f64, usize)> {
    if k == 0 || k >= tree.size() {
        return tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(radius2)
            .unsorted()
            .execute()
            .iter()
            .map(|r| (r.distance, r.item as usize))
            .collect();
    }
    let found = tree
        .query(&query)
        .nearest_n::<SquaredEuclidean<f64>>(NonZero::new(k).expect("k > 0"))
        .within(radius2)
        .execute();
    let found = match found.last() {
        Some(last) if found.len() == k => tree
            .query(&query)
            .within::<SquaredEuclidean<f64>>(last.distance)
            .unsorted()
            .execute(),
        _ => found,
    };
    found
        .iter()
        .map(|r| (r.distance, r.item as usize))
        .collect()
}

impl Tree {
    fn build(points: &[[f64; 3]], flat: &Flat) -> Option<Self> {
        match flat.dims() {
            1 => build::<1>(points, flat).map(Tree::One),
            2 => build::<2>(points, flat).map(Tree::Two),
            _ => build::<3>(points, flat).map(Tree::Three),
        }
    }

    fn size(&self) -> usize {
        match self {
            Tree::One(t) => t.size(),
            Tree::Two(t) => t.size(),
            Tree::Three(t) => t.size(),
        }
    }

    fn nearest(&self, flat: &Flat, query: &[f64; 3], k: usize, radius2: f64) -> Vec<(f64, usize)> {
        match self {
            Tree::One(t) => nearest(t, flat.keep(query), k, radius2),
            Tree::Two(t) => nearest(t, flat.keep(query), k, radius2),
            Tree::Three(t) => nearest(t, flat.keep(query), k, radius2),
        }
    }
}

impl Index {
    fn new(points: &[[f64; 3]]) -> Self {
        let mut index = Self {
            blocks: vec![],
            flat: Flat::of(points),
        };
        index.push(points, 0);
        index
    }

    fn push(&mut self, points: &[[f64; 3]], start: usize) {
        if start == points.len() {
            return;
        }
        if let Some(tree) = Tree::build(&points[start..], &self.flat) {
            self.blocks
                .truncate(self.blocks.partition_point(|b| b.start < start));
            self.blocks.push(Block { start, tree });
        }
    }

    fn scanned(&self) -> usize {
        self.blocks.last().map_or(0, |b| b.start + b.tree.size())
    }

    /// Indexes the last of `points`; `false` when it leaves the plane or
    /// line of the others and the index must be rebuilt.
    fn add(&mut self, points: &[[f64; 3]]) -> bool {
        if !self.flat.holds(&points[points.len() - 1]) {
            return false;
        }
        let mut start = self.scanned();
        if points.len() - start >= SCANNED {
            for b in self.blocks.iter().rev() {
                if b.tree.size() > points.len() - start {
                    break;
                }
                start = b.start;
            }
            self.push(points, start);
        }
        true
    }

    fn candidates(
        &self,
        points: &[[f64; 3]],
        query: &[f64; 3],
        k: usize,
        radius2: f64,
    ) -> Vec<(f64, usize)> {
        let flat = &self.flat;
        let mut bound = radius2 - flat.offset(query);
        if bound < 0.0 {
            return vec![];
        }
        let mut found: Vec<(f64, usize)> = vec![];
        for b in &self.blocks {
            let enough = found.len() >= k;
            let near = b
                .tree
                .nearest(flat, query, if enough { 0 } else { k }, bound);
            if !enough && near.len() >= k {
                bound = near.iter().map(|n| n.0).fold(0.0, f64::max);
            }
            found.extend(near.into_iter().map(|(d, i)| (d, i + b.start)));
        }
        let scanned = self.scanned();
        if self.blocks.len() > 1 || points.len() > scanned {
            found.extend(
                points[scanned..]
                    .iter()
                    .zip(scanned..)
                    .map(|(p, i)| (flat.distance(p, query), i))
                    .filter(|(d, _)| *d <= bound),
            );
            found.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            if let Some(&(last, _)) = found.get(k.saturating_sub(1)) {
                found.retain(|a| a.0 <= last);
            }
        }
        found
    }
}

impl SearchTree {
    pub fn new(samples: &[Sample], params: &Search, vg: Option<&Variogram>) -> Self {
        let frame = metric(params, vg).map_or_else(Matrix3::identity, Anisotropy::matrix);
        let locs: Vec<Point> = samples.iter().map(|s| s.loc).collect();
        let points: Vec<[f64; 3]> = locs.iter().map(|p| project(&frame, p)).collect();
        Self {
            index: Index::new(&points),
            frame,
            points,
            locs,
            holes: samples.iter().map(|s| s.hole).collect(),
            values: samples.iter().map(|s| s.value).collect(),
            domains: samples.iter().map(|s| s.domain).collect(),
            params: params.clone(),
        }
    }

    fn project(&self, p: &Point) -> [f64; 3] {
        project(&self.frame, p)
    }

    /// Adds a sample; its index is the number of samples before it.
    pub fn add(&mut self, sample: &Sample) {
        let point = self.project(&sample.loc);
        self.points.push(point);
        self.locs.push(sample.loc);
        self.holes.push(sample.hole);
        self.values.push(sample.value);
        self.domains.push(sample.domain);
        if !self.index.add(&self.points) {
            self.index = Index::new(&self.points);
        }
    }

    pub fn len(&self) -> usize {
        self.locs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.locs.is_empty()
    }

    fn candidates(&self, query: &[f64; 3], k: usize, radius2: f64) -> Vec<(f64, usize)> {
        self.index.candidates(&self.points, query, k, radius2)
    }

    /// Selection by a per-query ellipsoid `local` (major = 1, other ratios ≤ 1)
    /// for a tree built without anisotropy: every sample in the sphere of
    /// `radius` is re-ranked by its local distance. `domain` is the target's,
    /// as in [`neighbors_in`].
    pub fn neighbors_within(
        &self,
        target: &Point,
        domain: Option<u32>,
        local: &Anisotropy,
    ) -> Result<Vec<usize>> {
        let params = &self.params;
        let query = self.project(target);
        let radius2 = params.radius * params.radius;
        let mut found: Vec<(f64, usize)> = self
            .candidates(&query, self.len(), radius2)
            .into_iter()
            .map(|(_, i)| (local.lag(target, &self.locs[i]), i))
            .filter(|&(d, i)| d <= params.radius && self.admits(domain, i, d))
            .collect();
        found.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let ordered = self.one_per_location(found.into_iter().map(|f| f.1), domain, None);
        let chosen = select(target, ordered, |i| self.locs[i], |i| self.holes[i], params);
        enough(chosen, params)
    }

    fn admits(&self, domain: Option<u32>, i: usize, distance: f64) -> bool {
        self.params
            .admits(domain, self.values[i], self.domains[i], distance)
    }

    fn one_per_location(
        &self,
        ordered: impl IntoIterator<Item = usize>,
        domain: Option<u32>,
        kept: Option<usize>,
    ) -> Vec<usize> {
        one_per_location(ordered, |i| self.locs[i], |i| self.domains[i], domain, kept)
    }

    /// Same selection as [`neighbors`] over the indexed samples.
    pub fn neighbors(&self, target: &Point) -> Result<Vec<usize>> {
        self.neighbors_in(target, None)
    }

    /// Same selection as [`neighbors_in`] over the indexed samples.
    pub fn neighbors_in(&self, target: &Point, domain: Option<u32>) -> Result<Vec<usize>> {
        self.neighbors_around(target, domain, None)
    }

    /// As [`SearchTree::neighbors_in`] for the target at sample `kept`, which
    /// is selected as usual but leaves its location to samples of other
    /// domains there; for leave-one-out.
    pub fn neighbors_around(
        &self,
        target: &Point,
        domain: Option<u32>,
        kept: Option<usize>,
    ) -> Result<Vec<usize>> {
        let params = &self.params;
        if self.is_empty() || params.max_samples == 0 {
            return enough(vec![], params);
        }
        let capped = params.octant
            || params.max_per_hole.is_some()
            || params.high_grade.is_some()
            || domain.is_some();
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
            let admitted = found
                .into_iter()
                .filter(|&(_, i)| {
                    let d2: f64 = (0..3).map(|d| (self.points[i][d] - query[d]).powi(2)).sum();
                    self.admits(domain, i, d2.sqrt())
                })
                .map(|f| f.1);
            let chosen = select(
                target,
                self.one_per_location(admitted, domain, kept),
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
            error_variance: 0.0,
            domain: None,
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
                    value: (i % 10) as f64,
                    hole: Some((i / 20) as u32),
                    error_variance: 0.0,
                    domain: Some(zone(&loc)),
                }
            })
            .collect();
        let aniso = Anisotropy::new(Angles {
            azimuth: 35.0,
            dip: 10.0,
            rake: 0.0,
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
                high_grade: Some(HighGrade {
                    threshold: 7.0,
                    radius: 40.0,
                }),
                anisotropy: Some(aniso.clone()),
                ..params(1, 12, 150.0)
            },
            Search {
                soft: Some(Soft::All(60.0)),
                anisotropy: Some(aniso),
                ..params(1, 16, 150.0)
            },
            Search {
                soft: Some(Soft::Pairs(vec![SoftPair {
                    target: 1,
                    sample: 2,
                    distance: 80.0,
                }])),
                octant: true,
                high_grade: Some(HighGrade {
                    threshold: 7.0,
                    radius: 40.0,
                }),
                ..params(1, 24, 200.0)
            },
        ];
        for p in &cases {
            let tree = SearchTree::new(&samples, p, None);
            for _ in 0..200 {
                let t = (next() * 1000.0, next() * 1000.0, next() * 100.0);
                for d in [None, Some(zone(&t))] {
                    assert_eq!(
                        tree.neighbors_in(&t, d).ok(),
                        neighbors_in(&t, d, &samples, p, None).ok()
                    );
                }
            }
        }
    }

    fn zone(p: &Point) -> u32 {
        ((p.0 + p.1) / 400.0) as u32 % 3
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
    fn a_level_added_among_3d_data_is_searched_exactly() {
        let mut samples: Vec<Sample> = (0..200)
            .map(|i| {
                s(
                    (i * 37 % 101) as f64,
                    (i * 53 % 97) as f64,
                    (i % 9) as f64,
                    0.0,
                    None,
                )
            })
            .collect();
        let p = Search {
            octant: true,
            ..params(1, 16, 30.0)
        };
        let mut tree = SearchTree::new(&samples, &p, None);
        for i in 0..3000 {
            let t = ((i * 7 % 60) as f64, (i / 60) as f64 * 2.0, 4.0);
            if i % 50 == 0 {
                assert_eq!(
                    tree.neighbors(&t).ok(),
                    neighbors(&t, &samples, &p, None).ok()
                );
            }
            let node = s(t.0, t.1, t.2, 0.0, None);
            tree.add(&node);
            samples.push(node);
        }
    }

    #[test]
    fn tree_grows() {
        let p = params(1, 2, 10.0);
        let mut tree = SearchTree::new(&[s(5.0, 0.0, 0.0, 1.0, None)], &p, None);
        tree.add(&s(1.0, 0.0, 0.0, 2.0, None));
        assert_eq!(tree.neighbors(&(0.0, 0.0, 0.0)).unwrap(), vec![1, 0]);
    }
}
