//! Search neighborhoods for local estimation.
//!
//! Selects the samples used to estimate a target location, honoring an
//! anisotropic search ellipsoid, min/max sample counts, a per-hole cap, and
//! optional balancing over octants or angular sectors. [`SearchTree`] answers the same query as
//! [`neighbors`] from a grid of cells visited nearest first.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use boitata_core::block_frame;
use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};
use variogram::{Angles, Anisotropy, Variogram};

use crate::Sample;
use crate::error::{EstimError, Result};

mod grid;
use grid::{Grid, d2};

type Point = (f64, f64, f64);

/// Search-neighborhood parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Search {
    pub min_samples: usize,
    pub max_samples: usize,
    /// Search radius in the metric of the ellipsoid: meters along the major
    /// axis when the anisotropy ranges are ratios (major = 1).
    #[serde(with = "boitata_core::nonfinite")]
    pub radius: f64,
    /// Maximum samples taken from any single drill hole (requires `Sample::hole`).
    pub max_per_hole: Option<usize>,
    /// Balance samples across the octants of the search ellipsoid around the
    /// target, or its quadrants when the data are 2D (one elevation).
    pub octant: bool,
    /// Balance samples across equal angular sectors in the plane of the
    /// search ellipsoid's major and semi-major axes, the first starting at
    /// the major axis; exclusive with `octant`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sectors: Option<PlaneSectors>,
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
    /// Per-target sample count: the fewest samples from `min_samples` whose
    /// kriging meets the calibration, `max_samples` when none does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<Calibration>,
}

/// Angular sectors in the plane of the search ellipsoid's major and
/// semi-major axes, each holding at most `max_per_sector` samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlaneSectors {
    pub count: usize,
    pub max_per_sector: usize,
}

/// A kriging quality a search is calibrated to reach per target.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Calibration {
    /// Slope of regression of true on estimated values at least this.
    Slope(#[serde(with = "boitata_core::nonfinite")] f64),
    /// Kriging efficiency at least this.
    Efficiency(#[serde(with = "boitata_core::nonfinite")] f64),
}

/// Soft domain boundaries: a sample of another domain informs a target only
/// when strictly closer than the distance of their pair, measured in the
/// same ellipsoid as [`Search::radius`], so 0 is a hard boundary and
/// infinity pools the domains. Pairs not listed are hard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Soft<L = u32> {
    /// One distance for every pair of domains, both ways.
    All(#[serde(with = "boitata_core::nonfinite")] f64),
    /// Distances from a target's domain to a sample's domain, one way.
    Pairs(Vec<SoftPair<L>>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoftPair<L = u32> {
    pub target: L,
    pub sample: L,
    #[serde(with = "boitata_core::nonfinite")]
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
    #[serde(with = "boitata_core::nonfinite")]
    pub threshold: f64,
    #[serde(with = "boitata_core::nonfinite")]
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
            sectors: None,
            anisotropy: None,
            high_grade: None,
            soft: None,
            calibration: None,
        }
    }
}

impl Search {
    /// Whether the search balances samples over octants or sectors.
    pub fn balances(&self) -> bool {
        self.octant || self.sectors.is_some()
    }

    /// An error unless the sector settings are consistent.
    pub fn check_sectors(&self) -> Result<()> {
        match self.sectors {
            Some(_) if self.octant => Err(EstimError::InvalidParameters(
                "give octant or sectors, not both".into(),
            )),
            Some(PlaneSectors {
                count,
                max_per_sector,
            }) if count < 2 || max_per_sector == 0 => Err(EstimError::InvalidParameters(
                "sectors need count >= 2 and max_per_sector >= 1".into(),
            )),
            _ => Ok(()),
        }
    }

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
    let space = Space::new(metric(params, vg), samples.iter().map(|s| &s.loc));
    chosen
        .iter()
        .map(|&i| {
            let mut s = samples[i].clone();
            if params.clamps() {
                let d = space.distance(target, &s.loc);
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

/// Whether `a` with variogram `va` and `b` with `vb` select the same
/// neighbors from the same sample locations, whatever the values: same
/// parameters and ellipsoid, and no high-grade restriction, which depends
/// on values.
pub fn same_neighborhood(
    a: &Search,
    va: Option<&Variogram>,
    b: &Search,
    vb: Option<&Variogram>,
) -> bool {
    a.high_grade.is_none()
        && b.high_grade.is_none()
        && a.min_samples == b.min_samples
        && a.max_samples == b.max_samples
        && a.radius == b.radius
        && a.max_per_hole == b.max_per_hole
        && a.octant == b.octant
        && a.sectors == b.sectors
        && a.soft == b.soft
        && metric(a, va).map(|m| &m.angles) == metric(b, vb).map(|m| &m.angles)
}

/// Indices of `searches` grouped by [`same_neighborhood`]: each group in
/// index order, groups in the order of their first index.
pub fn neighborhood_groups(searches: &[(&Search, Option<&Variogram>)]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = vec![];
    for (i, &(s, v)) in searches.iter().enumerate() {
        match groups.iter_mut().find(|g| {
            let (t, w) = searches[g[0]];
            same_neighborhood(s, v, t, w)
        }) {
            Some(g) => g.push(i),
            None => groups.push(vec![i]),
        }
    }
    groups
}

/// Sectors around a target: octants split by the axes of the search
/// ellipsoid, or quadrants split by its horizontal axes when the data are 2D,
/// or `plane` equal angular sectors in its major / semi-major plane.
#[derive(Clone, Copy)]
struct Sectors {
    axes: Matrix3<f64>,
    planar: bool,
    plane: Option<usize>,
}

impl Sectors {
    /// Axes of `aniso` as (x, y, z) at zero rotation, so an unrotated
    /// ellipsoid splits along the coordinate axes.
    fn new(aniso: Option<&Anisotropy>, planar: bool, params: &Search) -> Self {
        let axes = aniso.map_or_else(Matrix3::identity, |a| {
            let Angles {
                azimuth, dip, rake, ..
            } = a.angles;
            block_frame([azimuth, dip, rake])
        });
        Self {
            axes,
            planar,
            plane: params.sectors.map(|p| p.count),
        }
    }

    fn count(&self) -> usize {
        match self.plane {
            Some(n) => n,
            None if self.planar => 4,
            None => 8,
        }
    }

    /// Sector (0..count) of `s` around `target`.
    fn of(&self, target: &Point, s: &Point) -> usize {
        let d = self.axes * Vector3::new(s.0 - target.0, s.1 - target.1, s.2 - target.2);
        if let Some(n) = self.plane {
            // block_frame puts the major axis on y and the semi-major on -x
            let turn = (-d.x).atan2(d.y).rem_euclid(std::f64::consts::TAU);
            return ((turn / std::f64::consts::TAU * n as f64) as usize).min(n - 1);
        }
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
    per_sector: Vec<usize>,
    sector_cap: usize,
}

impl<'a> Selector<'a> {
    fn new(target: &'a Point, params: &'a Search, sectors: &'a Sectors) -> Self {
        Self {
            target,
            params,
            sectors,
            chosen: Vec::with_capacity(params.max_samples),
            per_hole: vec![],
            per_sector: vec![
                0;
                if params.balances() {
                    sectors.count()
                } else {
                    0
                }
            ],
            sector_cap: match (params.sectors, params.octant) {
                (Some(p), _) => p.max_per_sector,
                (None, true) => params.max_samples.div_ceil(sectors.count()),
                (None, false) => usize::MAX,
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
        if self.params.balances() {
            let o = self.sectors.of(self.target, loc);
            if self.per_sector[o] >= self.sector_cap {
                return false;
            }
            self.per_sector[o] += 1;
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
    let space = Space::new(aniso, samples.iter().map(|s| &s.loc));
    let query = space.project(target);
    let radius2 = params.radius * params.radius;
    let mut cand: Vec<(usize, f64)> = samples
        .iter()
        .enumerate()
        .map(|(i, s)| (i, d2(&space.project(&s.loc), &query)))
        .filter(|&(i, d2)| {
            let s = &samples[i];
            d2 <= radius2 && params.admits(target, domain, &s.loc, s.value, s.domain, d2.sqrt())
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
        params,
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
    space: Space,
    points: Vec<[f64; 3]>,
    locs: Vec<Point>,
    holes: Vec<Option<u32>>,
    values: Vec<f64>,
    domains: Vec<Option<u32>>,
    params: Search,
    sectors: Sectors,
}

/// The search ellipsoid's frame, applied to offsets from `origin`, the first
/// finite sample location, so large coordinates cancel exactly before the
/// rotation. The scan and the tree measure every distance here.
#[derive(Clone, Copy)]
struct Space {
    frame: Matrix3<f64>,
    origin: Option<Point>,
}

impl Space {
    fn new<'a>(aniso: Option<&Anisotropy>, locs: impl IntoIterator<Item = &'a Point>) -> Self {
        let mut space = Self {
            frame: aniso.map_or_else(Matrix3::identity, Anisotropy::matrix),
            origin: None,
        };
        for p in locs {
            if space.anchor(p) {
                break;
            }
        }
        space
    }

    /// Takes `p` as the origin if there is none and it is finite; true once
    /// there is an origin.
    fn anchor(&mut self, p: &Point) -> bool {
        if self.origin.is_none() && [p.0, p.1, p.2].iter().all(|v| v.is_finite()) {
            self.origin = Some(*p);
        }
        self.origin.is_some()
    }

    fn project(&self, p: &Point) -> [f64; 3] {
        let o = self.origin.unwrap_or_default();
        let q = self.frame * Vector3::new(p.0 - o.0, p.1 - o.1, p.2 - o.2);
        [q.x, q.y, q.z]
    }

    fn distance(&self, a: &Point, b: &Point) -> f64 {
        d2(&self.project(a), &self.project(b)).sqrt()
    }
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
        let locs: Vec<Point> = samples.iter().map(|s| s.loc).collect();
        let space = Space::new(aniso, &locs);
        let sectors = Sectors::new(aniso, planar(locs.iter()), params);
        let points: Vec<[f64; 3]> = locs.iter().map(|p| space.project(p)).collect();
        Self {
            index: Grid::new(&points),
            space,
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
        self.space.project(p)
    }

    /// Adds a sample; its index is the number of samples before it.
    pub fn add(&mut self, sample: &Sample) {
        if let Some(first) = self.locs.first() {
            self.sectors.planar &= first.2 == sample.loc.2;
        }
        self.space.anchor(&sample.loc);
        let point = self.project(&sample.loc);
        self.points.push(point);
        self.locs.push(sample.loc);
        self.holes.push(sample.hole);
        self.values.push(sample.value);
        self.domains.push(sample.domain);
        self.index.add(&self.points);
    }

    /// Linear map to the isotropic search space: the search distance
    /// between two points is the norm of their difference mapped by it.
    pub fn frame(&self) -> &Matrix3<f64> {
        &self.space.frame
    }

    /// Distance from `a` to `b` in the search ellipsoid, as the search
    /// measures it.
    pub fn distance(&self, a: &Point, b: &Point) -> f64 {
        self.space.distance(a, b)
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
        let sectors = Sectors::new(Some(local), self.sectors.planar, params);
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

    /// `chosen` round-robin over the sectors around `target`, each sector in
    /// selection order, so every leading part stays balanced; unchanged
    /// unless the search is calibrated and balances octants or sectors. `local` is the
    /// ellipsoid of [`SearchTree::neighbors_within`], if searched so.
    pub fn balanced(
        &self,
        target: &Point,
        local: Option<&Anisotropy>,
        chosen: Vec<usize>,
    ) -> Vec<usize> {
        if !self.params.balances() || self.params.calibration.is_none() {
            return chosen;
        }
        let sectors = local.map_or(self.sectors, |a| {
            Sectors::new(Some(a), self.sectors.planar, &self.params)
        });
        let mut seen = vec![0usize; sectors.count()];
        let mut ranked: Vec<(usize, usize)> = chosen
            .into_iter()
            .map(|i| {
                let o = sectors.of(target, &self.locs[i]);
                seen[o] += 1;
                (seen[o], i)
            })
            .collect();
        ranked.sort_by_key(|r| r.0);
        ranked.into_iter().map(|r| r.1).collect()
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
    #[test]
    fn frame_gives_the_search_distance() {
        let samples = vec![Sample::new((0.0, 0.0, 0.0), 1.0)];
        let anisotropy = Anisotropy::new(Angles {
            azimuth: 40.0,
            dip: 10.0,
            rake: 0.0,
            major: 1.0,
            semi: 0.5,
            minor: 0.25,
        })
        .unwrap();
        let search = Search {
            min_samples: 1,
            max_samples: 4,
            radius: 100.0,
            anisotropy: Some(anisotropy),
            ..Default::default()
        };
        let tree = SearchTree::new(&samples, &search, None);
        let (a, b) = ((1.0, 2.0, 3.0), (20.0, -5.0, 9.0));
        let v = Vector3::new(b.0 - a.0, b.1 - a.1, b.2 - a.2);
        assert!(((tree.frame() * v).norm() - tree.distance(&a, &b)).abs() < 1e-9);
    }

    use super::*;
    use variogram::Angles;

    fn tilted(azimuth: f64) -> Variogram {
        Variogram::single(variogram::Model::Spherical, 1.0, 100.0).with_anisotropy(
            variogram::Anisotropy::new(Angles {
                azimuth,
                dip: 0.0,
                rake: 0.0,
                major: 1.0,
                semi: 0.5,
                minor: 0.5,
            })
            .unwrap(),
        )
    }

    #[test]
    fn factors_share_a_search_only_with_the_same_ellipsoid_and_parameters() {
        let plain = Search {
            max_samples: 24,
            radius: 150.0,
            ..Default::default()
        };
        let (north, east) = (tilted(0.0), tilted(90.0));
        let per_hole = Search {
            max_per_hole: Some(4),
            ..plain.clone()
        };
        let groups = neighborhood_groups(&[
            (&plain, Some(&north)),
            (&plain, Some(&east)),
            (&plain, Some(&north)),
            (&per_hole, Some(&north)),
        ]);
        assert_eq!(groups, vec![vec![0, 2], vec![1], vec![3]]);
    }

    #[test]
    fn an_explicit_search_ellipsoid_overrides_the_variograms() {
        let fixed = Search {
            anisotropy: tilted(45.0).anisotropy,
            ..Default::default()
        };
        assert!(same_neighborhood(
            &fixed,
            Some(&tilted(0.0)),
            &fixed,
            Some(&tilted(90.0))
        ));
    }

    #[test]
    fn a_high_grade_restriction_never_shares() {
        let restricted = Search {
            high_grade: Some(HighGrade::new(5.0, 20.0)),
            ..Default::default()
        };
        let vg = tilted(0.0);
        assert!(!same_neighborhood(
            &restricted,
            Some(&vg),
            &restricted,
            Some(&vg)
        ));
    }

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

    #[test]
    fn tree_matches_the_scan_among_ties_far_from_the_origin() {
        let (x0, y0, z0) = (612_345.0, 7_654_321.0, 850.0);
        let samples: Vec<Sample> = (0..9 * 9 * 12)
            .map(|i| {
                let (a, b, c) = (i % 9, i / 9 % 9, i / 81);
                let loc = (
                    x0 + a as f64 * 10.0,
                    y0 + b as f64 * 10.0,
                    z0 - c as f64 * 2.5,
                );
                s(loc.0, loc.1, loc.2, 0.0, Some((i % 81) as u32))
            })
            .collect();
        let ellipsoid = |azimuth, dip| {
            Anisotropy::new(Angles {
                azimuth,
                dip,
                rake: 0.0,
                major: 1.0,
                semi: 0.5,
                minor: 0.25,
            })
            .unwrap()
        };
        let ellipsoids = [
            None,
            Some(ellipsoid(0.0, 0.0)),
            Some(ellipsoid(90.0, 0.0)),
            Some(ellipsoid(45.0, 30.0)),
        ];
        for anisotropy in ellipsoids {
            for (octant, max_per_hole) in [
                (false, None),
                (true, None),
                (false, Some(2)),
                (true, Some(3)),
            ] {
                for radius in [10.0, 20.0, 25.0, 40.0] {
                    let p = Search {
                        octant,
                        max_per_hole,
                        anisotropy: anisotropy.clone(),
                        ..params(1, 16, radius)
                    };
                    let tree = SearchTree::new(&samples, &p, None);
                    for k in 0..80 {
                        let t = (
                            x0 + (k % 9) as f64 * 5.0,
                            y0 + (k / 9) as f64 * 10.0,
                            z0 - (k % 5) as f64 * 5.0,
                        );
                        assert_eq!(
                            tree.neighbors(&t).ok(),
                            neighbors(&t, &samples, &p, None).ok(),
                            "{t:?} {p:?}"
                        );
                    }
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
            let sectors = Sectors::new(Some(&ellipse(30.0)), planar, &Search::default());
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
        let p = Search::default();
        for sectors in [
            Sectors::new(None, false, &p),
            Sectors::new(Some(&zero), false, &p),
        ] {
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

    fn sectored(count: usize, max_per_sector: usize) -> Search {
        Search {
            sectors: Some(PlaneSectors {
                count,
                max_per_sector,
            }),
            ..params(1, 100, 100.0)
        }
    }

    #[test]
    fn plane_sectors_turn_from_the_major_axis_and_ignore_the_minor() {
        let (sa, ca) = 30f64.to_radians().sin_cos();
        let at = |along: f64, across: f64, z: f64| {
            (along * sa + across * ca, along * ca - across * sa, z)
        };
        let sectors = Sectors::new(Some(&ellipse(30.0)), false, &sectored(6, 1));
        let o = (0.0, 0.0, 0.0);
        for z in [-3.0, 0.0, 3.0] {
            let got: Vec<usize> = (0..6)
                .map(|k| {
                    let (s, c) = (30.0 + 60.0 * k as f64).to_radians().sin_cos();
                    sectors.of(&o, &at(c, s, z))
                })
                .collect();
            // sector centres, turning one way or the other from the major axis
            let ahead: Vec<usize> = (0..6).collect();
            let back: Vec<usize> = (0..6).rev().collect();
            assert!(got == ahead || got == back, "{got:?}");
        }
    }

    #[test]
    fn plane_sectors_cap_each_sector_with_the_nearest() {
        let samples: Vec<Sample> = (0..4)
            .flat_map(|q| {
                let (s, c) = (45.0 + 90.0 * q as f64).to_radians().sin_cos();
                (1..=3).map(move |r| (r as f64 * s, r as f64 * c))
            })
            .map(|(x, y)| s(x, y, 0.0, 0.0, None))
            .collect();
        let p = sectored(4, 2);
        let o = (0.1, -0.05, 0.0);
        let mut got = neighbors(&o, &samples, &p, None).unwrap();
        got.sort();
        assert_eq!(got, vec![0, 1, 3, 4, 6, 7, 9, 10]);
        let mut tree = SearchTree::new(&samples, &p, None).neighbors(&o).unwrap();
        tree.sort();
        assert_eq!(tree, got);
    }

    #[test]
    fn plane_sectors_exclude_octants_and_need_two() {
        assert!(sectored(4, 2).check_sectors().is_ok());
        assert!(sectored(1, 2).check_sectors().is_err());
        assert!(sectored(4, 0).check_sectors().is_err());
        let both = Search {
            octant: true,
            ..sectored(4, 2)
        };
        assert!(both.check_sectors().is_err());
    }

    #[test]
    fn plane_sector_tree_matches_the_scan() {
        let samples: Vec<Sample> = (0..3000)
            .map(|i| {
                s(
                    (i * 37 % 211) as f64,
                    (i * 53 % 197) as f64,
                    (i * 11 % 23) as f64,
                    0.0,
                    Some((i % 300) as u32),
                )
            })
            .collect();
        for (count, cap) in [(4, 3), (6, 2), (8, 4), (12, 1)] {
            let p = Search {
                anisotropy: Some(ellipse(40.0)),
                max_per_hole: Some(2),
                ..sectored(count, cap)
            };
            let tree = SearchTree::new(&samples, &p, None);
            for i in 0..200 {
                let t = ((i % 20) as f64 * 9.3, (i / 20) as f64 * 17.1, 7.5);
                let scan = neighbors(&t, &samples, &p, None).ok();
                assert_eq!(tree.neighbors(&t).ok(), scan);
                assert!(scan.is_some_and(|c| c.len() <= count * cap));
            }
        }
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
