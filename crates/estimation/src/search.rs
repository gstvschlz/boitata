//! Search neighborhoods for local estimation.
//!
//! Selects the samples used to estimate a target location, honoring an
//! anisotropic search ellipsoid, min/max sample counts, a per-hole cap, and
//! optional octant balancing. [`SearchTree`] answers the same query as
//! [`neighbors`] from a grid of cells visited nearest first.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use ceres_core::block_frame;
use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};
use variogram::aniso::euclidean;
use variogram::{Angles, Anisotropy, Variogram};

use crate::Sample;
use crate::error::{EstimError, Result};

mod grid;
use grid::Grid;

type Point = (f64, f64, f64);

/// Search-neighborhood parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Search {
    pub min_samples: usize,
    pub max_samples: usize,
    /// Search radius in the metric of the ellipsoid: meters along the major
    /// axis when the anisotropy ranges are ratios (major = 1).
    #[serde(with = "ceres_core::nonfinite")]
    pub radius: f64,
    /// Maximum samples taken from any single drill hole (requires `Sample::hole`).
    pub max_per_hole: Option<usize>,
    /// Balance samples across the octants of the search ellipsoid around the
    /// target, or its quadrants when the data are 2D (one elevation).
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

/// Samples valued above `threshold` are restricted beyond `radius`, measured
/// in the same ellipsoid as [`Search::radius`], or in `anisotropy` when
/// given (ratios, major = 1, so `radius` is along its major axis).
/// `threshold` is in data units, also in simulators that krige normal
/// scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighGrade {
    #[serde(with = "ceres_core::nonfinite")]
    pub threshold: f64,
    #[serde(with = "ceres_core::nonfinite")]
    pub radius: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anisotropy: Option<Anisotropy>,
    #[serde(default)]
    pub mode: HighGradeMode,
}

/// What happens to a high-grade sample beyond the restricted distance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HighGradeMode {
    /// Left out of the search.
    #[default]
    Drop,
    /// Kept, with its value capped at the threshold.
    Clamp,
}

impl HighGrade {
    pub fn new(threshold: f64, radius: f64) -> Self {
        Self {
            threshold,
            radius,
            anisotropy: None,
            mode: HighGradeMode::Drop,
        }
    }

    /// Whether a sample of `value` at `loc`, at `distance` from `target` in
    /// the search ellipsoid, lies beyond the restriction.
    fn beyond(&self, target: &Point, loc: &Point, value: f64, distance: f64) -> bool {
        value > self.threshold
            && match &self.anisotropy {
                Some(a) => a.lag(target, loc),
                None => distance,
            } > self.radius
    }
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
    /// Whether a sample of `value` and `domain` at `loc`, `distance` from a
    /// target of domain `target_domain`, may inform it under the high-grade
    /// and soft-boundary rules.
    fn admits(
        &self,
        target: &Point,
        target_domain: Option<u32>,
        loc: &Point,
        value: f64,
        domain: Option<u32>,
        distance: f64,
    ) -> bool {
        let high = self.high_grade.as_ref().is_some_and(|h| {
            h.mode == HighGradeMode::Drop && h.beyond(target, loc, value, distance)
        });
        let reached = match (target_domain, domain) {
            (Some(t), Some(s)) if t != s => {
                distance < self.soft.as_ref().map_or(0.0, |r| r.distance(&t, &s))
            }
            _ => true,
        };
        !high && reached
    }

    /// The threshold when a sample of `value` at `loc`, `distance` from
    /// `target`, is capped by a clamping high-grade restriction.
    pub fn cap(&self, target: &Point, loc: &Point, value: f64, distance: f64) -> Option<f64> {
        self.high_grade
            .as_ref()
            .filter(|h| h.mode == HighGradeMode::Clamp && h.beyond(target, loc, value, distance))
            .map(|h| h.threshold)
    }

    /// Whether high-grade samples may be capped rather than left out.
    pub fn clamps(&self) -> bool {
        self.high_grade
            .as_ref()
            .is_some_and(|h| h.mode == HighGradeMode::Clamp)
    }
}

/// An error when a search clamps high grades for an estimator that
/// cannot, named `what`.
pub fn unclamped(searches: &[Search], what: &str) -> Result<()> {
    match searches.iter().any(Search::clamps) {
        true => Err(EstimError::InvalidParameters(format!(
            "{what} does not clamp high grades; use mode drop"
        ))),
        false => Ok(()),
    }
}

/// The samples at `chosen` for `target`, as selected by [`neighbors`] with
/// the same arguments, their values capped by a clamping restriction.
pub fn take(
    target: &Point,
    chosen: &[usize],
    samples: &[Sample],
    params: &Search,
    vg: Option<&Variogram>,
) -> Vec<Sample> {
    let aniso = metric(params, vg);
    chosen
        .iter()
        .map(|&i| {
            let mut s = samples[i].clone();
            if params.clamps() {
                let d = aniso.map_or_else(|| euclidean(target, &s.loc), |a| a.lag(target, &s.loc));
                if let Some(t) = params.cap(target, &s.loc, s.value, d) {
                    s.value = t;
                }
            }
            s
        })
        .collect()
}

fn metric<'a>(params: &'a Search, vg: Option<&'a Variogram>) -> Option<&'a Anisotropy> {
    params
        .anisotropy
        .as_ref()
        .or_else(|| vg.and_then(|v| v.anisotropy.as_ref()))
}

/// Sectors around a target: octants split by the axes of the search
/// ellipsoid, or quadrants split by its horizontal axes when the data are 2D.
#[derive(Clone, Copy)]
struct Sectors {
    axes: Matrix3<f64>,
    planar: bool,
}

impl Sectors {
    /// Axes of `aniso` as (x, y, z) at zero rotation, so an unrotated
    /// ellipsoid splits along the coordinate axes.
    fn new(aniso: Option<&Anisotropy>, planar: bool) -> Self {
        let axes = aniso.map_or_else(Matrix3::identity, |a| {
            let Angles {
                azimuth, dip, rake, ..
            } = a.angles;
            block_frame([azimuth, dip, rake])
        });
        Self { axes, planar }
    }

    fn count(&self) -> usize {
        if self.planar { 4 } else { 8 }
    }

    /// Sector (0..count) of `s` around `target`.
    fn of(&self, target: &Point, s: &Point) -> usize {
        let d = self.axes * Vector3::new(s.0 - target.0, s.1 - target.1, s.2 - target.2);
        let [bx, by, bz] = [d.x, d.y, d.z].map(|v| (v >= 0.0) as usize);
        if self.planar {
            (bx << 1) | by
        } else {
            (bx << 2) | (by << 1) | bz
        }
    }
}

/// Whether every point shares one elevation.
fn planar<'a>(mut locs: impl Iterator<Item = &'a Point>) -> bool {
    let z = locs.next().map(|p| p.2);
    locs.all(|p| Some(p.2) == z)
}

/// Greedy selection over candidates offered by increasing distance.
struct Selector<'a> {
    target: &'a Point,
    params: &'a Search,
    sectors: &'a Sectors,
    chosen: Vec<usize>,
    per_hole: Vec<(u32, usize)>,
    per_octant: [usize; 8],
    octant_cap: usize,
}

impl<'a> Selector<'a> {
    fn new(target: &'a Point, params: &'a Search, sectors: &'a Sectors) -> Self {
        Self {
            target,
            params,
            sectors,
            chosen: Vec::with_capacity(params.max_samples),
            per_hole: vec![],
            per_octant: [0; 8],
            octant_cap: if params.octant {
                params.max_samples.div_ceil(sectors.count())
            } else {
                usize::MAX
            },
        }
    }

    fn full(&self) -> bool {
        self.chosen.len() >= self.params.max_samples
    }

    /// Offers sample `idx`; true once the selection is full.
    fn offer(&mut self, idx: usize, loc: &Point, hole: Option<u32>) -> bool {
        if self.full() {
            return true;
        }
        let seen = hole.and_then(|h| self.per_hole.iter().position(|p| p.0 == h));
        if let (Some(cap), Some(k)) = (self.params.max_per_hole, seen)
            && self.per_hole[k].1 >= cap
        {
            return false;
        }
        if self.params.octant {
            let o = self.sectors.of(self.target, loc);
            if self.per_octant[o] >= self.octant_cap {
                return false;
            }
            self.per_octant[o] += 1;
        }
        match (hole, seen) {
            (Some(_), Some(k)) => self.per_hole[k].1 += 1,
            (Some(h), None) => self.per_hole.push((h, 1)),
            _ => {}
        }
        self.chosen.push(idx);
        self.full()
    }
}

/// Greedy selection over candidates ordered by increasing distance.
fn select(
    target: &Point,
    ordered: impl IntoIterator<Item = usize>,
    loc: impl Fn(usize) -> Point,
    hole: impl Fn(usize) -> Option<u32>,
    params: &Search,
    sectors: &Sectors,
) -> Vec<usize> {
    let mut selector = Selector::new(target, params, sectors);
    for idx in ordered {
        if selector.offer(idx, &loc(idx), hole(idx)) {
            break;
        }
    }
    selector.chosen
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
            d <= params.radius && params.admits(target, domain, &s.loc, s.value, s.domain, d)
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
    let sectors = Sectors::new(
        aniso,
        params.octant && planar(samples.iter().map(|s| &s.loc)),
    );
    let chosen = select(
        target,
        ordered,
        |i| samples[i].loc,
        |i| samples[i].hole,
        params,
        &sectors,
    );
    enough(chosen, params)
}

/// Samples indexed for repeated neighborhood queries. Points are stored in
/// the search ellipsoid's frame, where the ellipsoid is a sphere, in a grid
/// of cells visited nearest first. Samples can be added after construction.
pub struct SearchTree {
    index: Grid,
    frame: Matrix3<f64>,
    points: Vec<[f64; 3]>,
    locs: Vec<Point>,
    holes: Vec<Option<u32>>,
    values: Vec<f64>,
    domains: Vec<Option<u32>>,
    params: Search,
    sectors: Sectors,
}

fn project(frame: &Matrix3<f64>, p: &Point) -> [f64; 3] {
    let q = frame * Vector3::new(p.0, p.1, p.2);
    [q.x, q.y, q.z]
}

#[cfg(test)]
thread_local! {
    /// Points indexed or scanned on this thread.
    static WORK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn work(_n: usize) {
    #[cfg(test)]
    WORK.set(WORK.get() + _n);
}

impl SearchTree {
    pub fn new(samples: &[Sample], params: &Search, vg: Option<&Variogram>) -> Self {
        let aniso = metric(params, vg);
        let frame = aniso.map_or_else(Matrix3::identity, Anisotropy::matrix);
        let locs: Vec<Point> = samples.iter().map(|s| s.loc).collect();
        let sectors = Sectors::new(aniso, planar(locs.iter()));
        let points: Vec<[f64; 3]> = locs.iter().map(|p| project(&frame, p)).collect();
        Self {
            index: Grid::new(&points),
            frame,
            points,
            locs,
            holes: samples.iter().map(|s| s.hole).collect(),
            values: samples.iter().map(|s| s.value).collect(),
            domains: samples.iter().map(|s| s.domain).collect(),
            params: params.clone(),
            sectors,
        }
    }

    fn project(&self, p: &Point) -> [f64; 3] {
        project(&self.frame, p)
    }

    /// Adds a sample; its index is the number of samples before it.
    pub fn add(&mut self, sample: &Sample) {
        if let Some(first) = self.locs.first() {
            self.sectors.planar &= first.2 == sample.loc.2;
        }
        let point = self.project(&sample.loc);
        self.points.push(point);
        self.locs.push(sample.loc);
        self.holes.push(sample.hole);
        self.values.push(sample.value);
        self.domains.push(sample.domain);
        self.index.add(&self.points);
    }

    /// Distance from `a` to `b` in the search ellipsoid, up to rounding.
    pub fn distance(&self, a: &Point, b: &Point) -> f64 {
        let (p, q) = (self.project(a), self.project(b));
        (0..3).map(|d| (p[d] - q[d]).powi(2)).sum::<f64>().sqrt()
    }

    pub fn len(&self) -> usize {
        self.locs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.locs.is_empty()
    }

    /// Selection by a per-query ellipsoid `local` for a tree built without
    /// anisotropy: every sample in the sphere of `radius` times the longest
    /// axis of `local` is re-ranked by its local distance. `domain` is the target's,
    /// as in [`neighbors_in`].
    pub fn neighbors_within(
        &self,
        target: &Point,
        domain: Option<u32>,
        local: &Anisotropy,
    ) -> Result<Vec<usize>> {
        let params = &self.params;
        let query = self.project(target);
        let axes = &local.angles;
        let reach = params.radius * axes.major.max(axes.semi).max(axes.minor);
        let radius2 = reach * reach;
        let mut near = vec![];
        self.index.nearest(&query, radius2, |_, i| {
            near.push(i);
            false
        });
        let mut found: Vec<(f64, usize)> = near
            .into_iter()
            .map(|i| (local.lag(target, &self.locs[i]), i))
            .filter(|&(d, i)| d <= params.radius && self.admits(target, domain, i, d))
            .collect();
        found.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let ordered = self.one_per_location(found.into_iter().map(|f| f.1), domain, None);
        let sectors = Sectors::new(Some(local), self.sectors.planar);
        let chosen = select(
            target,
            ordered,
            |i| self.locs[i],
            |i| self.holes[i],
            params,
            &sectors,
        );
        enough(chosen, params)
    }

    fn admits(&self, target: &Point, domain: Option<u32>, i: usize, distance: f64) -> bool {
        self.params.admits(
            target,
            domain,
            &self.locs[i],
            self.values[i],
            self.domains[i],
            distance,
        )
    }

    /// The threshold when sample `i` is capped for `target` by a clamping
    /// restriction; `local` is the ellipsoid of
    /// [`SearchTree::neighbors_within`], if searched so.
    pub fn cap(&self, target: &Point, local: Option<&Anisotropy>, i: usize) -> Option<f64> {
        if !self.params.clamps() {
            return None;
        }
        let loc = &self.locs[i];
        let d = local.map_or_else(|| self.distance(target, loc), |a| a.lag(target, loc));
        self.params.cap(target, loc, self.values[i], d)
    }

    /// The samples at `chosen` for `target`, their values capped by a
    /// clamping restriction; `samples` are those the tree was built from.
    pub fn take(
        &self,
        target: &Point,
        local: Option<&Anisotropy>,
        chosen: &[usize],
        samples: &[Sample],
    ) -> Vec<Sample> {
        chosen
            .iter()
            .map(|&i| {
                let mut s = samples[i].clone();
                if let Some(t) = self.cap(target, local, i) {
                    s.value = t;
                }
                s
            })
            .collect()
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
        let query = self.project(target);
        let mut selector = Selector::new(target, params, &self.sectors);
        let mut group: Vec<usize> = vec![];
        let mut at = f64::NAN;
        let mut full = false;
        self.index
            .nearest(&query, params.radius * params.radius, |d2, i| {
                if d2 != at && !group.is_empty() {
                    full = self.offer(&mut group, at, target, domain, kept, &mut selector);
                    if full {
                        return true;
                    }
                }
                at = d2;
                group.push(i);
                false
            });
        if !full {
            self.offer(&mut group, at, target, domain, kept, &mut selector);
        }
        enough(selector.chosen, params)
    }

    /// Offers the samples of `group`, all at squared distance `d2`, that the
    /// search admits, one per location; true once the selection is full.
    fn offer(
        &self,
        group: &mut Vec<usize>,
        d2: f64,
        target: &Point,
        domain: Option<u32>,
        kept: Option<usize>,
        selector: &mut Selector,
    ) -> bool {
        let admitted: Vec<usize> = group
            .drain(..)
            .filter(|&i| self.admits(target, domain, i, d2.sqrt()))
            .collect();
        for i in self.one_per_location(admitted, domain, kept) {
            if selector.offer(i, &self.locs[i], self.holes[i]) {
                return true;
            }
        }
        false
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
                high_grade: Some(HighGrade::new(7.0, 40.0)),
                anisotropy: Some(aniso.clone()),
                ..params(1, 12, 150.0)
            },
            Search {
                high_grade: Some(HighGrade {
                    anisotropy: Some(aniso.clone()),
                    ..HighGrade::new(7.0, 60.0)
                }),
                octant: true,
                ..params(1, 16, 150.0)
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
                high_grade: Some(HighGrade::new(7.0, 40.0)),
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

    fn ellipse(azimuth: f64) -> Anisotropy {
        Anisotropy::new(Angles {
            azimuth,
            dip: 0.0,
            rake: 0.0,
            major: 1.0,
            semi: 0.5,
            minor: 0.5,
        })
        .unwrap()
    }

    #[test]
    fn octants_follow_the_ellipse() {
        let (sa, ca) = 30f64.to_radians().sin_cos();
        let at = |along: f64, across: f64, z: f64| {
            (along * sa + across * ca, along * ca - across * sa, z)
        };
        let o = (0.0, 0.0, 0.0);
        for planar in [false, true] {
            let sectors = Sectors::new(Some(&ellipse(30.0)), planar);
            let z = if planar { 0.0 } else { 1.0 };
            let quadrants = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)];
            let got =
                quadrants.map(|(along, across)| sectors.of(&o, &at(5.0 * along, 0.5 * across, z)));
            let expected = if planar { [0, 2, 1, 3] } else { [1, 5, 3, 7] };
            assert_eq!(got, expected);
        }
        let samples: Vec<Sample> = [-0.5, 0.5, -0.4, 0.4]
            .iter()
            .zip([4.0, 5.0, 6.0, 7.0])
            .map(|(&across, along)| {
                let (x, y, z) = at(along, across, 0.0);
                s(x, y, z, 0.0, None)
            })
            .chain([s(-20.0, -20.0, 0.0, 0.0, None)])
            .collect();
        let p = Search {
            octant: true,
            anisotropy: Some(ellipse(30.0)),
            ..params(1, 4, 100.0)
        };
        let mut got = neighbors(&o, &samples, &p, None).unwrap();
        got.sort();
        assert_eq!(got, vec![0, 1, 4]);
    }

    #[test]
    fn planar_octants_fill_max_samples() {
        let samples: Vec<Sample> = (0..400)
            .map(|i| s((i % 20) as f64, (i / 20) as f64, 0.0, 0.0, None))
            .collect();
        for anisotropy in [None, Some(ellipse(35.0))] {
            let p = Search {
                octant: true,
                anisotropy,
                ..params(1, 24, 100.0)
            };
            let tree = SearchTree::new(&samples, &p, None);
            let t = (9.3, 10.6, 0.0);
            assert_eq!(neighbors(&t, &samples, &p, None).unwrap().len(), 24);
            assert_eq!(tree.neighbors(&t).unwrap().len(), 24);
        }
    }

    #[test]
    fn unrotated_octants_split_along_the_axes() {
        let old = |t: &Point, s: &Point| {
            (((s.0 >= t.0) as usize) << 2) | (((s.1 >= t.1) as usize) << 1) | (s.2 >= t.2) as usize
        };
        let zero = Anisotropy::new(Angles {
            major: 1.0,
            semi: 0.3,
            minor: 0.1,
            ..ellipse(0.0).angles
        })
        .unwrap();
        let t = (0.0, 0.0, 0.0);
        for sectors in [Sectors::new(None, false), Sectors::new(Some(&zero), false)] {
            for i in 0..125 {
                let p = (
                    (i % 5 - 2) as f64,
                    (i / 5 % 5 - 2) as f64,
                    (i / 25 - 2) as f64,
                );
                assert_eq!(sectors.of(&t, &p), old(&t, &p));
            }
        }
    }

    #[test]
    fn octant_selection_is_deterministic() {
        let samples: Vec<Sample> = (0..2000)
            .map(|i| s((i * 37 % 211) as f64, (i * 53 % 197) as f64, 0.0, 0.0, None))
            .collect();
        let p = Search {
            octant: true,
            anisotropy: Some(ellipse(60.0)),
            ..params(1, 16, 60.0)
        };
        let tree = SearchTree::new(&samples, &p, None);
        let run = |threads| {
            use rayon::prelude::*;
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                (0..500)
                    .into_par_iter()
                    .map(|i| {
                        tree.neighbors(&((i % 25) as f64 * 8.1, (i / 25) as f64 * 9.7, 0.0))
                            .ok()
                    })
                    .collect::<Vec<_>>()
            })
        };
        let one = run(1);
        assert_eq!(one, run(4));
    }

    #[test]
    fn tree_grows() {
        let p = params(1, 2, 10.0);
        let mut tree = SearchTree::new(&[s(5.0, 0.0, 0.0, 1.0, None)], &p, None);
        tree.add(&s(1.0, 0.0, 0.0, 2.0, None));
        assert_eq!(tree.neighbors(&(0.0, 0.0, 0.0)).unwrap(), vec![1, 0]);
    }

    /// Nodes of one level among 3D data used to rebuild the whole index on
    /// every addition and scan it, quadratic in the number of nodes.
    #[test]
    fn a_level_grown_among_3d_data_is_indexed_in_n_log_n() {
        let data: Vec<Sample> = (0..300)
            .map(|i| {
                let (x, y, z) = (i * 37 % 101, i * 53 % 97, i % 7);
                s(x as f64, y as f64, z as f64, 0.0, None)
            })
            .collect();
        let n = 10_000;
        WORK.set(0);
        let mut tree = SearchTree::new(&data, &params(1, 16, 40.0), None);
        for i in 0..n {
            let loc = ((i % 100) as f64, (i / 100) as f64, 3.5);
            tree.neighbors(&loc).unwrap();
            tree.add(&s(loc.0, loc.1, loc.2, 0.0, None));
        }
        let work = WORK.get();
        assert!(work < 400 * n, "{work} points indexed or scanned");
    }
}
