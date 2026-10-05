//! Drilling plans: the kriging metrics of target blocks with candidate
//! holes added to the data, and an objective summed over the blocks.
//!
//! Kriging variance, slope and efficiency depend on sample locations only,
//! so sample values never enter. A candidate's composites can only enter the
//! search of the blocks within the search radius of them, its reach, so
//! adding or removing a hole re-kriges those blocks alone, and a cached gain
//! stays valid until a change touches its reach. With domains, a target of
//! a domain without samples stays unestimated, so a hole bringing a domain
//! its first samples reaches every target of that domain.

use std::collections::BTreeSet;

use rayon::prelude::*;
use variogram::Variogram;

use crate::Sample;
use crate::block::{Discretization, block_covariances, block_krige_points};
use crate::error::{EstimError, Result};
use crate::krige::{Estimate, Kind, krige};
use crate::search::{Search, SearchTree};

type Point = (f64, f64, f64);

/// Kriging the plan evaluates: `kind` at points, or ordinary kriging of
/// blocks of `block` (size, discretization) centered on the targets.
#[derive(Debug, Clone)]
pub struct Kriging {
    pub kind: Kind,
    pub block: Option<(Point, Discretization)>,
    pub variogram: Variogram,
    /// Search passes: a target the first leaves unestimated goes to the next.
    pub passes: Vec<Search>,
}

impl Kriging {
    fn krige(&self, t: &Point, s: &[Sample]) -> Result<Estimate> {
        match &self.block {
            None => krige(self.kind, t, s, &self.variogram),
            Some((size, disc)) => {
                block_krige_points(Kind::Ordinary, &disc.points(t, size), s, &self.variogram)
            }
        }
    }

    /// C(v, v) of the target's support.
    fn support(&self, t: &Point) -> f64 {
        match &self.block {
            None => self.variogram.total_sill(),
            Some((size, disc)) => block_covariances(&disc.points(t, size), &[], &self.variogram).1,
        }
    }

    /// The ellipsoid of the data spacing: the first pass's radius and own
    /// anisotropy, as `data_spacing` reads a Search.
    fn spacing(&self) -> Search {
        let first = &self.passes[0];
        Search {
            radius: first.radius,
            anisotropy: first.anisotropy.clone(),
            ..Default::default()
        }
    }
}

/// Names of the [`Metrics`] fields, in [`Metrics::get`] order.
pub const METRICS: [&str; 6] = [
    "variance",
    "slope",
    "efficiency",
    "n_samples",
    "n_holes",
    "data_spacing",
];

/// Metrics of one target: kriging metrics as `predict(..., diagnostics=True)`
/// reports them, NaN when unestimated, and the data spacing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    pub variance: f64,
    pub slope: f64,
    pub efficiency: f64,
    pub n_samples: f64,
    /// Distinct holes among the samples used; untagged samples count one each.
    pub n_holes: f64,
    /// Equivalent spacing sqrt(V / (c n)) of Cabral Pinto and Deutsch
    /// (2017): V the volume of the first search pass's ellipsoid (radius
    /// and the Search's own anisotropy), n the samples inside it around the
    /// target, of any domain, and c the composite length; NaN when n is 0.
    pub data_spacing: f64,
}

impl Metrics {
    pub const UNESTIMATED: Self = Self {
        variance: f64::NAN,
        slope: f64::NAN,
        efficiency: f64::NAN,
        n_samples: f64::NAN,
        n_holes: f64::NAN,
        data_spacing: f64::NAN,
    };

    fn of(e: &Estimate, used: &[Sample]) -> Self {
        let mut holes = BTreeSet::new();
        let n_holes = used
            .iter()
            .filter(|s| s.hole.is_none_or(|h| holes.insert(h)))
            .count();
        Self {
            variance: e.variance,
            slope: e.slope(),
            efficiency: e.efficiency(),
            n_samples: e.n_used as f64,
            n_holes: n_holes as f64,
            data_spacing: f64::NAN,
        }
    }

    pub fn get(&self, k: usize) -> f64 {
        [
            self.variance,
            self.slope,
            self.efficiency,
            self.n_samples,
            self.n_holes,
            self.data_spacing,
        ][k]
    }

    /// As the built-in objectives read it: an unestimated target has the
    /// variance of its support, `support`, and zero slope, efficiency and
    /// counts; a target without samples in its ellipsoid an infinite spacing.
    fn filled(&self, support: f64) -> Self {
        let mut m = *self;
        if m.variance.is_nan() {
            (m.variance, m.slope, m.efficiency) = (support, 0.0, 0.0);
            (m.n_samples, m.n_holes) = (0.0, 0.0);
        }
        if m.data_spacing.is_nan() {
            m.data_spacing = f64::INFINITY;
        }
        m
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

impl Op {
    fn holds(self, value: f64, threshold: f64) -> bool {
        match self {
            Op::Less => value < threshold,
            Op::LessEqual => value <= threshold,
            Op::Greater => value > threshold,
            Op::GreaterEqual => value >= threshold,
        }
    }
}

/// `metric` (an index into [`METRICS`]) `op` `threshold`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Condition {
    pub metric: usize,
    pub op: Op,
    pub threshold: f64,
}

/// Classes in priority order, best first, each the conditions a target
/// must all meet; targets meeting none are below the last.
pub type Rules = Vec<Vec<Condition>>;

/// Per-target score g(metrics), summed over the targets; both built-ins are
/// 0 with the data alone and weighted per target.
#[derive(Debug, Clone, PartialEq)]
pub enum Objective {
    /// w (σ²₀ − σ²) / C(v, v), σ²₀ with the data alone.
    Variance,
    /// Targets classed by `rules[of[b]]`, all by `rules[0]` without `of`;
    /// a target whose `of` is None scores 0. A target scores w × its
    /// progress towards the class above its class with the data alone, and
    /// 0 when that class is the best. Condition j gives p_j = 1 when met, 0
    /// when met with the data alone but no longer, and otherwise
    /// clip((m − m₀) / (t − m₀), 0, 1) for metric m, m₀ with the data alone
    /// and threshold t, 0 where that is undefined. With P the mean p_j,
    /// progress is (P − P₀) / (1 − P₀): 1 exactly when the target reaches
    /// the class.
    Classification {
        rules: Vec<Rules>,
        of: Option<Vec<Option<usize>>>,
    },
    /// Scores from the caller, as [`Scorer`].
    Custom,
}

/// Scores of the rows of metrics, each of the target it names; for
/// [`Objective::Custom`].
pub type Scorer<'a> = &'a mut dyn FnMut(&[u32], &[Metrics]) -> Result<Vec<f64>>;

/// A candidate hole.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub composites: Vec<Point>,
    /// Domain code of each composite, or empty without domains.
    pub domains: Vec<u32>,
    pub collar: Point,
    pub cost: f64,
    /// Left out by an exclusion zone.
    pub excluded: bool,
}

/// Where to krige: `weights` 1 by default; `domains`, one code per target,
/// confine each to the samples of its domain and of others within the
/// search's soft distance.
#[derive(Debug, Clone, Default)]
pub struct Targets {
    pub points: Vec<Point>,
    pub weights: Option<Vec<f64>>,
    pub domains: Option<Vec<u32>>,
}

/// Limits on a plan; holes closer than `min_spacing` (horizontal distance
/// between collars) cannot both be drilled.
#[derive(Debug, Clone, Default)]
pub struct Constraints {
    pub n_holes: Option<usize>,
    pub budget: Option<f64>,
    pub min_spacing: f64,
}

/// Candidate holes over targets, with the metrics of the current selection
/// and the gains of adding each candidate cached.
pub struct Plan {
    kriging: Kriging,
    composite_length: f64,
    data: Vec<Sample>,
    holes: Vec<Vec<Sample>>,
    candidates: Vec<Candidate>,
    constraints: Constraints,
    targets: Vec<Point>,
    domains: Option<Vec<u32>>,
    /// Targets of each domain code.
    members: Vec<Vec<u32>>,
    /// Domain codes of each candidate's composites, distinct.
    brings: Vec<Vec<u32>>,
    weights: Vec<f64>,
    support: Vec<f64>,
    objective: Objective,
    baseline: Vec<Metrics>,
    /// Class of each target with the data alone, for classification.
    class: Vec<Option<usize>>,
    /// Targets whose score can change: weighted and, for classification,
    /// classed below the best class.
    active: Vec<bool>,
    reach: Vec<Vec<u32>>,
    state: State,
}

/// A selection, sorted, its samples and search trees; for the current
/// selection also what it caches.
struct State {
    selected: Vec<usize>,
    samples: Vec<Sample>,
    trees: Vec<SearchTree>,
    spacing: SearchTree,
    /// Samples of each domain code.
    counts: Vec<usize>,
    current: Vec<Metrics>,
    /// Score of each target with the selection; None before scoring.
    scores: Option<Vec<f64>>,
    /// Targets whose metrics changed since they were scored.
    stale: Vec<bool>,
    gains: Vec<Option<f64>>,
    /// Summed scores with the data alone.
    origin: Option<f64>,
}

fn invalid<T>(message: impl Into<String>) -> Result<T> {
    Err(EstimError::InvalidParameters(message.into()))
}

fn finite(p: &Point) -> bool {
    p.0.is_finite() && p.1.is_finite() && p.2.is_finite()
}

fn planar<'a>(mut locs: impl Iterator<Item = &'a Point>) -> bool {
    let z = locs.next().map(|p| p.2);
    locs.all(|p| Some(p.2) == z)
}

fn merged(mut a: Vec<u32>, b: impl IntoIterator<Item = u32>) -> Vec<u32> {
    a.extend(b);
    a.sort_unstable();
    a.dedup();
    a
}

impl Plan {
    /// A plan over `targets` from `data`, whose values are ignored; the
    /// candidates' composites are `composite_length` long.
    pub fn new(
        kriging: Kriging,
        data: Vec<Sample>,
        candidates: Vec<Candidate>,
        targets: Targets,
        objective: Objective,
        constraints: Constraints,
        composite_length: f64,
    ) -> Result<Self> {
        let passes = &kriging.passes;
        if passes.is_empty() {
            return invalid("the estimator needs a search");
        }
        if passes.iter().any(|s| s.high_grade.is_some()) {
            return invalid("a high-grade search depends on sample values; drop it for planning");
        }
        if passes.iter().any(|s| s.calibration.is_some()) {
            return invalid("calibrated searches are not supported for planning");
        }
        if !(composite_length.is_finite() && composite_length > 0.0) {
            return invalid("composite_length must be finite and > 0");
        }
        let Targets {
            points: targets,
            weights,
            domains,
        } = targets;
        let composites = || candidates.iter().flat_map(|c| &c.composites);
        let points = data.iter().map(|s| &s.loc).chain(composites());
        if !points.chain(&targets).all(finite) || !candidates.iter().all(|c| finite(&c.collar)) {
            return invalid("coordinates must be finite");
        }
        let flat = planar(data.iter().map(|s| &s.loc).chain(composites()));
        let mixed = match data.is_empty() {
            false => planar(data.iter().map(|s| &s.loc)) != flat,
            true => {
                let one = |c: &Candidate| !c.composites.is_empty() && planar(c.composites.iter());
                !flat && candidates.iter().any(one)
            }
        };
        if passes.iter().any(|s| s.octant) && mixed {
            return invalid(
                "an octant search on data at one elevation turns to quadrants; give data in 3D",
            );
        }
        if candidates
            .iter()
            .any(|c| !(c.cost.is_finite() && c.cost >= 0.0))
        {
            return invalid("costs must be finite and >= 0");
        }
        let Constraints {
            budget,
            min_spacing,
            ..
        } = constraints;
        if !(min_spacing.is_finite() && min_spacing >= 0.0) {
            return invalid("min_spacing must be finite and >= 0");
        }
        if budget.is_some_and(|b| !(b.is_finite() && b >= 0.0)) {
            return invalid("budget must be finite and >= 0");
        }
        let n = targets.len();
        let weights = weights.unwrap_or_else(|| vec![1.0; n]);
        if weights.len() != n {
            return invalid(format!("{} weights for {n} targets", weights.len()));
        }
        if weights.iter().any(|w| !(w.is_finite() && *w >= 0.0)) {
            return invalid("weights must be finite and >= 0");
        }
        let tagged = data.iter().all(|s| s.domain.is_some())
            && candidates
                .iter()
                .all(|c| c.domains.len() == c.composites.len());
        match &domains {
            Some(d) if d.len() != n || !tagged => {
                return invalid("domains need one code per target, sample and composite");
            }
            None if data.iter().any(|s| s.domain.is_some())
                || candidates.iter().any(|c| !c.domains.is_empty()) =>
            {
                return invalid("samples carry domains; the targets need them too");
            }
            _ => {}
        }
        if let Objective::Classification { rules, of } = &objective {
            let bad = |c: &Condition| c.metric >= METRICS.len() || !c.threshold.is_finite();
            let sets = rules.len();
            let unknown = of
                .as_ref()
                .is_some_and(|of| of.len() != n || of.iter().flatten().any(|&k| k >= sets));
            if sets == 0 || rules.iter().any(Vec::is_empty) || unknown {
                return invalid("classification needs rules for the targets");
            }
            if rules.iter().flatten().flatten().any(bad) {
                return invalid("rules need known metrics and finite thresholds");
            }
        }
        let codes = domains.iter().flatten().copied();
        let codes = codes
            .chain(data.iter().filter_map(|s| s.domain))
            .chain(candidates.iter().flat_map(|c| c.domains.iter().copied()));
        let n_domains = codes.max().map_or(0, |d| d as usize + 1);
        let mut members = vec![vec![]; n_domains];
        for (b, &d) in domains.iter().flatten().enumerate() {
            members[d as usize].push(b as u32);
        }
        let tags = data
            .iter()
            .filter_map(|s| s.hole)
            .max()
            .map_or(0, |h| h + 1);
        let holes: Vec<Vec<Sample>> = candidates
            .iter()
            .enumerate()
            .map(|(c, cand)| {
                let hole = tags + c as u32;
                let at = |(k, &p)| Sample {
                    domain: cand.domains.get(k).copied(),
                    ..Sample::with_hole(p, 0.0, hole)
                };
                cand.composites.iter().enumerate().map(at).collect()
            })
            .collect();
        let brings = candidates
            .iter()
            .map(|c| merged(c.domains.clone(), []))
            .collect();
        let data: Vec<Sample> = data
            .into_iter()
            .map(|s| Sample { value: 0.0, ..s })
            .collect();
        let support: Vec<f64> = targets.par_iter().map(|t| kriging.support(t)).collect();
        let mut plan = Self {
            data,
            holes,
            candidates,
            constraints,
            domains,
            members,
            brings,
            weights,
            support,
            objective,
            composite_length,
            baseline: vec![],
            class: vec![],
            active: vec![],
            reach: vec![],
            state: State {
                selected: vec![],
                samples: vec![],
                trees: vec![],
                spacing: SearchTree::new(&[], &Search::default(), None),
                counts: vec![],
                current: vec![],
                scores: None,
                stale: vec![false; n],
                gains: vec![],
                origin: None,
            },
            targets,
            kriging,
        };
        let state = plan.state_of(vec![]);
        plan.state = State {
            stale: vec![false; n],
            gains: vec![None; plan.candidates.len()],
            ..state
        };
        let all: Vec<u32> = (0..n as u32).collect();
        plan.baseline = plan.evaluate(&plan.state, &all);
        plan.state.current = plan.baseline.clone();
        plan.class = plan.classes();
        plan.active = plan.actives();
        plan.reach = plan.reaches();
        Ok(plan)
    }

    /// Samples and search trees of `selected`, sorted, with nothing cached.
    fn state_of(&self, selected: Vec<usize>) -> State {
        let samples = self.samples(&selected);
        let vg = Some(&self.kriging.variogram);
        let trees = self.kriging.passes.iter();
        let trees = trees.map(|s| SearchTree::new(&samples, s, vg)).collect();
        let mut counts = vec![0; self.members.len()];
        samples
            .iter()
            .filter_map(|s| s.domain)
            .for_each(|d| counts[d as usize] += 1);
        State {
            selected,
            trees,
            spacing: SearchTree::new(&samples, &self.kriging.spacing(), None),
            samples,
            counts,
            current: vec![],
            scores: None,
            stale: vec![],
            gains: vec![],
            origin: None,
        }
    }

    fn classes(&self) -> Vec<Option<usize>> {
        let Objective::Classification { rules, of } = &self.objective else {
            return vec![];
        };
        (0..self.targets.len())
            .map(|b| {
                let set = &rules[of.as_ref().map_or(Some(0), |of| of[b])?];
                let m = self.baseline[b].filled(self.support[b]);
                let meets =
                    |r: &Vec<Condition>| r.iter().all(|c| c.op.holds(m.get(c.metric), c.threshold));
                Some(set.iter().position(meets).unwrap_or(set.len()))
            })
            .collect()
    }

    fn actives(&self) -> Vec<bool> {
        (0..self.targets.len())
            .map(|b| match &self.objective {
                Objective::Custom => true,
                Objective::Variance => self.weights[b] != 0.0 && self.support[b] > 0.0,
                Objective::Classification { .. } => {
                    self.weights[b] != 0.0 && self.class[b].is_some_and(|k| k > 0)
                }
            })
            .collect()
    }

    /// Targets each candidate's composites reach in any search pass or in
    /// the ellipsoid of the data spacing.
    fn reaches(&self) -> Vec<Vec<u32>> {
        let vg = Some(&self.kriging.variogram);
        let nodes: Vec<Sample> = self.targets.iter().map(|&t| Sample::new(t, 0.0)).collect();
        let spacing = self.kriging.spacing();
        let searches = self.kriging.passes.iter().map(|s| (s, vg));
        let trees: Vec<(SearchTree, f64)> = searches
            .chain([(&spacing, None)])
            .map(|(s, vg)| (SearchTree::new(&nodes, s, vg), s.radius * (1.0 + 1e-9)))
            .collect();
        self.holes
            .par_iter()
            .map(|hole| {
                let reach = trees
                    .iter()
                    .flat_map(|(tree, r)| hole.iter().flat_map(|s| tree.within(&s.loc, *r)))
                    .map(|b| b as u32);
                merged(vec![], reach)
            })
            .collect()
    }

    /// Targets of the domains `codes` that have no samples in `state`.
    fn opened(&self, state: &State, codes: &[u32]) -> Vec<u32> {
        codes
            .iter()
            .filter(|&&d| state.counts[d as usize] == 0)
            .flat_map(|&d| self.members[d as usize].iter().copied())
            .collect()
    }

    /// Targets candidate `c` changes when added to, or removed from, `state`.
    fn reach_in(&self, state: &State, c: usize) -> Vec<u32> {
        merged(self.reach[c].clone(), self.opened(state, &self.brings[c]))
    }

    /// Metrics of `blocks` with the samples of `state`.
    fn evaluate(&self, state: &State, blocks: &[u32]) -> Vec<Metrics> {
        blocks
            .par_iter()
            .map(|&b| self.metrics_at(state, b as usize, &[], 0))
            .collect()
    }

    /// Index in the samples of `state` at which the composites of candidate
    /// `c` would sit once added.
    fn slot(&self, state: &State, c: usize) -> usize {
        let before = state.selected.iter().take_while(|&&s| s < c);
        self.data.len() + before.map(|&s| self.holes[s].len()).sum::<usize>()
    }

    /// Metrics of target `b` with the samples of `state` and `extra` inserted
    /// before sample `at`, as [`SearchTree::neighbors_plus`] ranks them.
    fn metrics_at(&self, state: &State, b: usize, extra: &[Sample], at: usize) -> Metrics {
        let t = &self.targets[b];
        let data_spacing = self.spacing_at(state, t, extra);
        let domain = self.domains.as_ref().map(|d| d[b]);
        let present = domain.is_none_or(|d| {
            state.counts[d as usize] > 0 || extra.iter().any(|s| s.domain == Some(d))
        });
        let n = state.samples.len();
        for tree in state.trees.iter().filter(|_| present) {
            let Ok(chosen) = tree.neighbors_plus(t, domain, extra, at) else {
                continue;
            };
            let used: Vec<Sample> = chosen
                .iter()
                .map(|&i| match i < n {
                    true => state.samples[i].clone(),
                    false => extra[i - n].clone(),
                })
                .collect();
            if let Ok(e) = self.kriging.krige(t, &used) {
                return Metrics {
                    data_spacing,
                    ..Metrics::of(&e, &used)
                };
            }
        }
        Metrics {
            data_spacing,
            ..Metrics::UNESTIMATED
        }
    }

    fn spacing_at(&self, state: &State, t: &Point, extra: &[Sample]) -> f64 {
        let search = &self.kriging.passes[0];
        let r = search.radius;
        if !(r.is_finite() && r > 0.0) {
            return f64::NAN;
        }
        let count = state.spacing.count_plus(t, r, extra);
        let axes = search.anisotropy.as_ref();
        let axes = axes.map_or(1.0, |a| a.angles.major * a.angles.semi * a.angles.minor);
        let volume = 4.0 / 3.0 * std::f64::consts::PI * r.powi(3) * axes;
        match count {
            0 => f64::NAN,
            n => (volume / (self.composite_length * n as f64)).sqrt(),
        }
    }

    /// Built-in score of target `b` with metrics `m`.
    fn builtin(&self, b: usize, m: &Metrics) -> f64 {
        if !self.active[b] {
            return 0.0;
        }
        let (w, cvv) = (self.weights[b], self.support[b]);
        let (m, m0) = (m.filled(cvv), self.baseline[b].filled(cvv));
        match &self.objective {
            Objective::Variance => w * (m0.variance - m.variance) / cvv,
            Objective::Classification { rules, of } => {
                let set = &rules[of.as_ref().map_or(0, |of| of[b].expect("active"))];
                let class = self.class[b].expect("active");
                w * progress(&set[class - 1], &m0, &m)
            }
            Objective::Custom => 0.0,
        }
    }

    fn scores(
        &self,
        blocks: &[u32],
        metrics: &[Metrics],
        custom: Option<Scorer>,
    ) -> Result<Vec<f64>> {
        if self.objective != Objective::Custom {
            let at = |(&b, m)| self.builtin(b as usize, m);
            return Ok(blocks.iter().zip(metrics).map(at).collect());
        }
        let Some(score) = custom else {
            return invalid("a custom objective needs its scorer");
        };
        if blocks.is_empty() {
            return Ok(vec![]);
        }
        let scores = score(blocks, metrics)?;
        if scores.len() != blocks.len() || scores.iter().any(|s| !s.is_finite()) {
            return invalid(format!(
                "the objective must give {} finite scores",
                blocks.len()
            ));
        }
        Ok(scores)
    }

    fn sorted(&self, selected: &[usize]) -> Result<Vec<usize>> {
        let mut s = selected.to_vec();
        s.sort_unstable();
        if s.windows(2).any(|w| w[0] == w[1]) {
            return invalid("selected holes repeat");
        }
        if s.last().is_some_and(|&i| i >= self.candidates.len()) {
            return invalid(format!(
                "selected holes must be below {}",
                self.candidates.len()
            ));
        }
        Ok(s)
    }

    /// Samples of the data and then the composites of `selected`.
    fn samples(&self, selected: &[usize]) -> Vec<Sample> {
        let holes = selected.iter().flat_map(|&c| &self.holes[c]);
        self.data.iter().chain(holes).cloned().collect()
    }

    /// Moves the state to `selected`, re-kriging the targets the change
    /// reaches and dropping the gains that read them.
    fn update(&mut self, selected: &[usize]) -> Result<()> {
        let selected = self.sorted(selected)?;
        if selected == self.state.selected {
            return Ok(());
        }
        let old: BTreeSet<usize> = self.state.selected.iter().copied().collect();
        let new: BTreeSet<usize> = selected.iter().copied().collect();
        let next = self.state_of(selected);
        let flipped: Vec<u32> = (0..self.members.len() as u32)
            .filter(|&d| (self.state.counts[d as usize] == 0) != (next.counts[d as usize] == 0))
            .collect();
        let mut touched = vec![false; self.targets.len()];
        let changed = old.symmetric_difference(&new).copied();
        let opened = flipped.iter().flat_map(|&d| &self.members[d as usize]);
        for &b in changed.clone().flat_map(|c| &self.reach[c]).chain(opened) {
            touched[b as usize] = true;
        }
        let blocks: Vec<u32> = (0..self.targets.len() as u32)
            .filter(|&b| touched[b as usize])
            .collect();
        let fresh = self.evaluate(&next, &blocks);
        let state = &mut self.state;
        (state.selected, state.samples, state.trees) = (next.selected, next.samples, next.trees);
        (state.spacing, state.counts) = (next.spacing, next.counts);
        for c in changed {
            state.gains[c] = None;
        }
        for (&b, m) in blocks.iter().zip(fresh) {
            state.current[b as usize] = m;
            state.stale[b as usize] = true;
        }
        let (reach, active, brings) = (&self.reach, &self.active, &self.brings);
        let hit = |b: &u32| touched[*b as usize] && active[*b as usize];
        // A hole opening an empty domain reaches all its targets.
        let empty: Vec<bool> = (0..self.members.len())
            .map(|d| state.counts[d] == 0 && self.members[d].iter().any(hit))
            .collect();
        let stale: Vec<bool> = reach
            .par_iter()
            .zip(brings)
            .map(|(r, d)| {
                r.iter().any(hit) || d.iter().any(|d| flipped.contains(d) || empty[*d as usize])
            })
            .collect();
        for (gain, stale) in state.gains.iter_mut().zip(stale) {
            if stale {
                *gain = None;
            }
        }
        Ok(())
    }

    /// Targets to score so the state's scores are current: all of them
    /// before the first scoring.
    fn unscored(&self) -> Vec<u32> {
        let all = 0..self.targets.len() as u32;
        match self.state.scores {
            Some(_) => all.filter(|&b| self.state.stale[b as usize]).collect(),
            None => all.collect(),
        }
    }

    fn store(&mut self, blocks: &[u32], scores: &[f64]) {
        let n = self.targets.len();
        let all = self.state.scores.get_or_insert_with(|| vec![0.0; n]);
        for (&b, &s) in blocks.iter().zip(scores) {
            all[b as usize] = s;
            self.state.stale[b as usize] = false;
        }
    }

    /// Metrics of every target with the composites of `selected`.
    pub fn metrics(&mut self, selected: &[usize]) -> Result<Vec<Metrics>> {
        self.update(selected)?;
        Ok(self.state.current.clone())
    }

    /// Objective with `selected`: Σ g(metrics) minus its value with the
    /// data alone.
    pub fn score(&mut self, selected: &[usize], custom: Option<Scorer>) -> Result<f64> {
        self.update(selected)?;
        let blocks = self.unscored();
        let all: Vec<u32> = (0..self.targets.len() as u32).collect();
        let origin = self.state.origin.is_none() && self.objective == Objective::Custom;
        let mut rows = blocks.clone();
        let mut metrics: Vec<Metrics> = blocks
            .iter()
            .map(|&b| self.state.current[b as usize])
            .collect();
        if origin {
            rows.extend(&all);
            metrics.extend(&self.baseline);
        }
        let scores = self.scores(&rows, &metrics, custom)?;
        self.store(&blocks, &scores[..blocks.len()]);
        if origin {
            self.state.origin = Some(scores[blocks.len()..].iter().sum());
        }
        let total: f64 = self.state.scores.as_ref().map_or(0.0, |s| s.iter().sum());
        Ok(total - self.state.origin.unwrap_or(0.0))
    }

    /// Whether each candidate could join `selected`: not selected, not
    /// excluded, within the hole count and budget, and at least
    /// `min_spacing` from every selected collar.
    pub fn feasible(&self, selected: &[usize]) -> Result<Vec<bool>> {
        let selected = self.sorted(selected)?;
        let Constraints {
            n_holes,
            budget,
            min_spacing,
        } = self.constraints;
        let spent: f64 = selected.iter().map(|&c| self.candidates[c].cost).sum();
        let full = n_holes.is_some_and(|n| selected.len() >= n);
        let spaced = |a: &Point| {
            selected.iter().all(|&s| {
                let b = &self.candidates[s].collar;
                (a.0 - b.0).hypot(a.1 - b.1) >= min_spacing
            })
        };
        Ok(self
            .candidates
            .iter()
            .enumerate()
            .map(|(c, cand)| {
                !full
                    && !cand.excluded
                    && selected.binary_search(&c).is_err()
                    && budget.is_none_or(|b| spent + cand.cost <= b)
                    && spaced(&cand.collar)
            })
            .collect())
    }

    /// Gain in the objective from adding each candidate to `selected`;
    /// −∞ where [`Plan::feasible`] is false.
    pub fn gains(&mut self, selected: &[usize], custom: Option<Scorer>) -> Result<Vec<f64>> {
        let feasible = self.feasible(selected)?;
        self.update(selected)?;
        let blocks = self.unscored();
        let wanted: Vec<usize> = (0..self.candidates.len())
            .filter(|&c| feasible[c] && self.state.gains[c].is_none())
            .collect();
        let (fresh, now) = self.gain_scores(&blocks, &wanted, custom)?;
        self.store(&blocks, &now);
        let scores = self.state.scores.as_ref().expect("scored");
        for (c, reach, s) in fresh {
            let before = reach.iter().map(|&b| scores[b as usize]);
            let gain = s.iter().zip(before).map(|(a, b)| a - b).sum();
            self.state.gains[c] = Some(gain);
        }
        Ok((0..self.candidates.len())
            .map(|c| match feasible[c] {
                true => self.state.gains[c].expect("computed"),
                false => f64::NEG_INFINITY,
            })
            .collect())
    }

    /// Scores of the active targets each of `wanted` reaches with it added,
    /// and the current scores of `blocks`, from one call of the scorer.
    #[allow(clippy::type_complexity)]
    fn gain_scores(
        &self,
        blocks: &[u32],
        wanted: &[usize],
        custom: Option<Scorer>,
    ) -> Result<(Vec<(usize, Vec<u32>, Vec<f64>)>, Vec<f64>)> {
        let state = &self.state;
        let current: Vec<Metrics> = blocks.iter().map(|&b| state.current[b as usize]).collect();
        let reach = |c: usize| -> Vec<u32> {
            let all = self.reach_in(state, c).into_iter();
            all.filter(|&b| self.active[b as usize]).collect()
        };
        let with = |c: usize, b: u32| {
            self.metrics_at(state, b as usize, &self.holes[c], self.slot(state, c))
        };
        if self.objective != Objective::Custom {
            let fresh = wanted
                .par_iter()
                .map(|&c| {
                    let r = reach(c);
                    let s = r
                        .iter()
                        .map(|&b| self.builtin(b as usize, &with(c, b)))
                        .collect();
                    (c, r, s)
                })
                .collect();
            return Ok((fresh, self.scores(blocks, &current, None)?));
        }
        let evaluated: Vec<(Vec<u32>, Vec<Metrics>)> = wanted
            .par_iter()
            .map(|&c| {
                let r = reach(c);
                let m = r.iter().map(|&b| with(c, b)).collect();
                (r, m)
            })
            .collect();
        let mut rows = blocks.to_vec();
        let mut metrics = current;
        for (r, m) in &evaluated {
            rows.extend(r);
            metrics.extend(m);
        }
        let scores = self.scores(&rows, &metrics, custom)?;
        let (now, mut rest) = scores.split_at(blocks.len());
        let mut fresh = Vec::with_capacity(wanted.len());
        for (&c, (r, _)) in wanted.iter().zip(evaluated) {
            let (mine, tail) = rest.split_at(r.len());
            fresh.push((c, r, mine.to_vec()));
            rest = tail;
        }
        Ok((fresh, now.to_vec()))
    }

    /// Loss in the objective from removing each of `selected`, in its order.
    pub fn loss(&mut self, selected: &[usize], custom: Option<Scorer>) -> Result<Vec<f64>> {
        self.update(selected)?;
        let blocks = self.unscored();
        let kept = &self.state.selected;
        let without: Vec<(Vec<u32>, Vec<Metrics>)> = selected
            .par_iter()
            .map(|&s| {
                let others: Vec<usize> = kept.iter().copied().filter(|&k| k != s).collect();
                let state = self.state_of(others);
                let r = self.reach_in(&state, s);
                let m = r
                    .iter()
                    .map(|&b| self.metrics_at(&state, b as usize, &[], 0))
                    .collect();
                (r, m)
            })
            .collect();
        let mut rows = blocks.clone();
        let mut metrics: Vec<Metrics> = blocks
            .iter()
            .map(|&b| self.state.current[b as usize])
            .collect();
        for (r, m) in &without {
            rows.extend(r);
            metrics.extend(m);
        }
        let scores = self.scores(&rows, &metrics, custom)?;
        let (now, mut rest) = scores.split_at(blocks.len());
        self.store(&blocks, now);
        let current = self.state.scores.as_ref().expect("scored");
        let mut out = Vec::with_capacity(selected.len());
        for (r, _) in &without {
            let (mine, tail) = rest.split_at(r.len());
            rest = tail;
            let loss = r
                .iter()
                .zip(mine)
                .map(|(&b, s)| current[b as usize] - s)
                .sum();
            out.push(loss);
        }
        Ok(out)
    }

    /// Candidates other than `i` whose collars lie within `radius`
    /// horizontally of its collar, in index order.
    pub fn neighbors(&self, i: usize, radius: f64) -> Result<Vec<usize>> {
        let Some(a) = self.candidates.get(i).map(|c| c.collar) else {
            return invalid(format!("candidate {i} out of range"));
        };
        if radius.is_nan() || radius < 0.0 {
            return invalid("radius must be >= 0");
        }
        Ok((0..self.candidates.len())
            .filter(|&c| {
                let b = &self.candidates[c].collar;
                c != i && (a.0 - b.0).hypot(a.1 - b.1) <= radius
            })
            .collect())
    }

    /// Indices of the targets each candidate's composites can reach.
    pub fn reach(&self, c: usize) -> &[u32] {
        &self.reach[c]
    }

    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// C(v, v) of each target's support.
    pub fn support(&self) -> &[f64] {
        &self.support
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
}

/// Progress of metrics `m` from `m0` towards meeting `rule`; see
/// [`Objective::Classification`].
fn progress(rule: &[Condition], m0: &Metrics, m: &Metrics) -> f64 {
    let p = |m: &Metrics| {
        let sum: f64 = rule
            .iter()
            .map(|c| {
                let (v, v0, t) = (m.get(c.metric), m0.get(c.metric), c.threshold);
                if c.op.holds(v, t) {
                    1.0
                } else if c.op.holds(v0, t) {
                    0.0
                } else {
                    let r = (v - v0) / (t - v0);
                    if r.is_nan() { 0.0 } else { r.clamp(0.0, 1.0) }
                }
            })
            .sum();
        sum / rule.len() as f64
    };
    let start = p(m0);
    (p(m) - start) / (1.0 - start)
}

#[cfg(test)]
mod tests {
    use super::*;
    use variogram::Model;

    fn lcg(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn kriging(block: bool, passes: Vec<Search>) -> Kriging {
        Kriging {
            kind: Kind::Ordinary,
            block: block.then_some((
                (10.0, 10.0, 10.0),
                Discretization {
                    nx: 2,
                    ny: 2,
                    nz: 2,
                },
            )),
            variogram: Variogram {
                nugget: 0.1,
                ..Variogram::single(Model::Spherical, 1.0, 80.0)
            },
            passes,
        }
    }

    fn search(radius: f64, max_samples: usize) -> Search {
        Search {
            min_samples: 3,
            max_samples,
            radius,
            ..Default::default()
        }
    }

    fn hole(x: f64, y: f64) -> Vec<Point> {
        (0..6).map(|k| (x, y, -2.5 - 5.0 * k as f64)).collect()
    }

    /// Domain of a location in the two-domain case: west and east of x =
    /// 100, and a north-east corner without data.
    fn domain_at(p: &Point) -> u32 {
        match (p.0 >= 150.0 && p.1 >= 150.0, p.0 < 100.0) {
            (true, _) => 2,
            (_, true) => 0,
            _ => 1,
        }
    }

    fn grid() -> Vec<Point> {
        (0..20 * 20 * 3)
            .map(|i| {
                let (x, y, z) = (i % 20, (i / 20) % 20, i / 400);
                (
                    5.0 + 10.0 * x as f64,
                    5.0 + 10.0 * y as f64,
                    -5.0 - 10.0 * z as f64,
                )
            })
            .collect()
    }

    /// Scattered data in holes, a grid of candidates and targets.
    fn case(block: bool, passes: Vec<Search>, objective: Objective) -> Plan {
        build(block, passes, objective, false)
    }

    fn build(block: bool, passes: Vec<Search>, objective: Objective, domains: bool) -> Plan {
        let mut u = lcg(7);
        let tag = |p: &Point| domains.then(|| domain_at(p));
        let data: Vec<Sample> = (0..25)
            .flat_map(|h| {
                let (x, y) = (u() * 200.0, u() * 200.0);
                let v = u();
                hole(x, y).into_iter().map(move |p| Sample {
                    domain: tag(&p),
                    ..Sample::with_hole(p, v, h)
                })
            })
            .filter(|s| !(domains && s.domain == Some(2)))
            .collect();
        let candidates = (0..64)
            .map(|i| {
                let (x, y) = (12.5 + 25.0 * (i % 8) as f64, 12.5 + 25.0 * (i / 8) as f64);
                let composites = hole(x, y);
                Candidate {
                    domains: composites.iter().filter_map(tag).collect(),
                    composites,
                    collar: (x, y, 0.0),
                    cost: 30.0,
                    excluded: i == 9,
                }
            })
            .collect();
        let points = grid();
        let targets = Targets {
            domains: domains.then(|| points.iter().map(domain_at).collect()),
            weights: Some((0..1200).map(|i| (i % 3) as f64).collect()),
            points,
        };
        Plan::new(
            kriging(block, passes),
            data,
            candidates,
            targets,
            objective,
            Constraints {
                n_holes: Some(5),
                budget: Some(120.0),
                min_spacing: 30.0,
            },
            5.0,
        )
        .unwrap()
    }

    fn rules() -> Objective {
        let c = |metric, op, threshold| Condition {
            metric,
            op,
            threshold,
        };
        Objective::Classification {
            rules: vec![vec![
                vec![c(1, Op::GreaterEqual, 0.8), c(4, Op::GreaterEqual, 3.0)],
                vec![c(1, Op::GreaterEqual, 0.5), c(5, Op::LessEqual, 30.0)],
                vec![c(0, Op::LessEqual, 0.9)],
            ]],
            of: None,
        }
    }

    /// Metrics of a fresh kriging run, as `predict` does it, with the
    /// composites of `selected` appended to the data in sorted order;
    /// targets of a domain without samples are left out, and the data
    /// spacing counts samples by brute force.
    fn rerun(plan: &Plan, selected: &[usize]) -> Vec<Metrics> {
        let mut sorted = selected.to_vec();
        sorted.sort_unstable();
        let samples = plan.samples(&sorted);
        let k = &plan.kriging;
        let known: Vec<usize> = (0..plan.targets.len())
            .filter(|&b| {
                plan.domains
                    .as_ref()
                    .is_none_or(|d| samples.iter().any(|s| s.domain == Some(d[b])))
            })
            .collect();
        let found = crate::by_pass(known.len(), &k.passes, |search, remaining| {
            let rows: Vec<usize> = remaining.iter().map(|&i| known[i]).collect();
            let at: Vec<Point> = rows.iter().map(|&i| plan.targets[i]).collect();
            let codes: Option<Vec<u32>> = plan
                .domains
                .as_ref()
                .map(|d| rows.iter().map(|&i| d[i]).collect());
            Ok(crate::estimate_many(
                &at,
                codes.as_deref(),
                &samples,
                search,
                Some(&k.variogram),
                |t, s| k.krige(t, s).map(|e| Metrics::of(&e, s)),
            ))
        })
        .unwrap();
        let mut out = vec![Metrics::UNESTIMATED; plan.targets.len()];
        for (&b, r) in known.iter().zip(found) {
            out[b] = r.map_or(Metrics::UNESTIMATED, |r| r.1);
        }
        let r = k.passes[0].radius;
        let volume = 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
        for (m, t) in out.iter_mut().zip(&plan.targets) {
            let d = |p: &Point| {
                ((p.0 - t.0).powi(2) + (p.1 - t.1).powi(2) + (p.2 - t.2).powi(2)).sqrt()
            };
            let n = samples.iter().filter(|s| d(&s.loc) <= r).count();
            m.data_spacing = if n == 0 {
                f64::NAN
            } else {
                (volume / (5.0 * n as f64)).sqrt()
            };
        }
        out
    }

    fn same(a: &[Metrics], b: &[Metrics]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            for k in 0..METRICS.len() {
                let (x, y) = (x.get(k), y.get(k));
                assert!(x == y || (x.is_nan() && y.is_nan()) || (x - y).abs() < 1e-10);
            }
        }
    }

    fn greedy(plan: &mut Plan, n: usize) -> Vec<usize> {
        let mut chosen = vec![];
        for _ in 0..n {
            let g = plan.gains(&chosen, None).unwrap();
            let best = (0..g.len()).max_by(|&a, &b| g[a].total_cmp(&g[b]).then(b.cmp(&a)));
            match best {
                Some(b) if g[b].is_finite() => chosen.push(b),
                _ => break,
            }
        }
        chosen
    }

    #[test]
    fn incremental_metrics_equal_a_full_rerun() {
        for block in [false, true] {
            let passes = vec![search(25.0, 12), search(45.0, 16)];
            let mut plan = case(block, passes, Objective::Variance);
            let unestimated = plan.baseline.iter().filter(|m| m.variance.is_nan()).count();
            assert!(unestimated > 0 && unestimated < 1200, "{unestimated}");
            let mut selected = vec![];
            for step in [
                vec![20],
                vec![20, 45],
                vec![45, 3],
                vec![3, 45, 62, 0],
                vec![],
            ] {
                plan.gains(&selected, None).unwrap();
                selected = step;
                let m = plan.metrics(&selected).unwrap();
                same(&m, &rerun(&plan, &selected));
            }
            let greedy = greedy(&mut plan, 3);
            same(&plan.metrics(&greedy).unwrap(), &rerun(&plan, &greedy));
        }
    }

    #[test]
    fn domains_and_soft_boundaries_match_a_full_rerun() {
        let c = |metric, op, threshold| Condition {
            metric,
            op,
            threshold,
        };
        let by_domain = Objective::Classification {
            rules: vec![
                vec![vec![c(1, Op::GreaterEqual, 0.7)]],
                vec![
                    vec![c(5, Op::LessEqual, 25.0)],
                    vec![c(0, Op::LessEqual, 0.9)],
                ],
            ],
            of: Some(
                grid()
                    .iter()
                    .map(|p| [Some(0), Some(1), None][domain_at(p) as usize])
                    .collect(),
            ),
        };
        for (soft, objective) in [(None, rules()), (Some(crate::Soft::All(12.0)), by_domain)] {
            let passes = [search(25.0, 12), search(45.0, 16)]
                .map(|s| Search {
                    soft: soft.clone(),
                    ..s
                })
                .to_vec();
            let mut plan = build(true, passes, objective, true);
            let corner: Vec<usize> = (0..1200)
                .filter(|&b| plan.domains.as_ref().unwrap()[b] == 2)
                .collect();
            assert!(!corner.is_empty());
            assert!(corner.iter().all(|&b| plan.baseline[b].variance.is_nan()));
            let mut selected = vec![];
            for step in [vec![63], vec![63, 20], vec![20], vec![20, 7, 54]] {
                plan.gains(&selected, None).unwrap();
                selected = step;
                same(&plan.metrics(&selected).unwrap(), &rerun(&plan, &selected));
            }
            let m = plan.metrics(&[63]).unwrap();
            assert!(corner.iter().any(|&b| !m[b].variance.is_nan()));
            let gains = plan.gains(&[20], None).unwrap();
            let mut fresh = case_like(&plan);
            let base = fresh.score(&[20], None).unwrap();
            for c in 0..64 {
                if gains[c].is_finite() {
                    let score = fresh.score(&[20, c], None).unwrap();
                    assert!((gains[c] - (score - base)).abs() < 1e-9, "{c}");
                }
            }
            assert!(gains[63] > 0.0 || gains[54] > 0.0);
            let loss = plan.loss(&[63, 20], None).unwrap();
            let both = fresh.score(&[63, 20], None).unwrap();
            assert!((loss[0] - (both - base)).abs() < 1e-9);
        }
    }

    /// A fresh plan with the set-up of `plan`.
    fn case_like(plan: &Plan) -> Plan {
        build(
            plan.kriging.block.is_some(),
            plan.kriging.passes.clone(),
            plan.objective.clone(),
            plan.domains.is_some(),
        )
    }

    #[test]
    fn neighbors_plus_matches_indexing_the_extra_samples() {
        let plan = case(false, vec![search(60.0, 10)], Objective::Variance);
        let extra = &plan.holes[27];
        let all: Vec<Sample> = plan.data.iter().chain(extra).cloned().collect();
        let tree = SearchTree::new(&plan.data, &plan.kriging.passes[0], None);
        let full = SearchTree::new(&all, &plan.kriging.passes[0], None);
        for t in &plan.targets {
            assert_eq!(
                tree.neighbors_plus(t, None, extra, plan.data.len()).ok(),
                full.neighbors(t).ok()
            );
        }
        let plan = tied(Objective::Variance, false, 6, None);
        let (d, extra) = (plan.data.len(), &plan.holes[0]);
        let ordered = |a: usize, b: usize| -> Vec<Sample> {
            let holes = plan.holes[a].iter().chain(&plan.holes[b]);
            plan.data.iter().chain(holes).cloned().collect()
        };
        let base: Vec<Sample> = plan.data.iter().chain(&plan.holes[1]).cloned().collect();
        for search in &plan.kriging.passes {
            let tree = SearchTree::new(&base, search, None);
            let full = SearchTree::new(&ordered(0, 1), search, None);
            let n = base.len();
            let renumber = |i: usize| match i {
                i if i < d => i,
                i if i < n => i + extra.len(),
                i => i - n + d,
            };
            for t in &plan.targets {
                let plus = tree.neighbors_plus(t, None, extra, d).ok();
                let plus = plus.map(|c| c.into_iter().map(renumber).collect::<Vec<_>>());
                assert_eq!(plus, full.neighbors(t).ok());
            }
        }
    }

    #[test]
    fn plan_ignores_data_values() {
        let mut a = case(true, vec![search(70.0, 16)], rules());
        let mut b = case(true, vec![search(70.0, 16)], rules());
        let mut u = lcg(3);
        b.data.iter_mut().for_each(|s| s.value = u() * 100.0);
        assert_eq!(greedy(&mut a, 4), greedy(&mut b, 4));
        assert_eq!(a.gains(&[5], None).unwrap(), b.gains(&[5], None).unwrap());
    }

    #[test]
    fn best_single_hole_gain_is_the_brute_force_best() {
        for objective in [Objective::Variance, rules()] {
            let mut plan = case(false, vec![search(70.0, 16)], objective);
            let gains = plan.gains(&[], None).unwrap();
            let feasible = plan.feasible(&[]).unwrap();
            let mut best = (f64::NEG_INFINITY, 0);
            for c in 0..plan.len() {
                if !feasible[c] {
                    assert_eq!(gains[c], f64::NEG_INFINITY);
                    continue;
                }
                let score = plan.score(&[c], None).unwrap();
                assert!((gains[c] - score).abs() < 1e-9, "{} {score}", gains[c]);
                if score > best.0 {
                    best = (score, c);
                }
            }
            assert!(best.0 > 0.0);
            assert_eq!(greedy(&mut plan, 1), vec![best.1]);
        }
    }

    #[test]
    fn cached_gains_equal_fresh_scores_after_moves() {
        let mut moved = case(true, vec![search(70.0, 16)], rules());
        for s in [vec![], vec![20], vec![20, 45], vec![45, 3]] {
            moved.gains(&s, None).unwrap();
        }
        moved.metrics(&[3, 21]).unwrap();
        let gains = moved.gains(&[3, 21], None).unwrap();
        let mut fresh = case(true, vec![search(70.0, 16)], rules());
        let base = fresh.score(&[3, 21], None).unwrap();
        for c in (0..64).filter(|c| gains[*c].is_finite()) {
            let score = fresh.score(&[3, 21, c], None).unwrap();
            assert!((gains[c] - (score - base)).abs() < 1e-9, "{c}");
        }
    }

    #[test]
    fn loss_is_the_score_without_the_hole() {
        let mut plan = case(true, vec![search(70.0, 16)], rules());
        let selected = [40, 12, 30];
        let loss = plan.loss(&selected, None).unwrap();
        let total = plan.score(&selected, None).unwrap();
        for (k, &s) in selected.iter().enumerate() {
            let others: Vec<usize> = selected.iter().copied().filter(|&o| o != s).collect();
            let without = plan.score(&others, None).unwrap();
            assert!((loss[k] - (total - without)).abs() < 1e-9);
        }
    }

    #[test]
    fn gains_do_not_depend_on_the_thread_count() {
        let run = |threads| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut plan = case(true, vec![search(70.0, 16)], rules());
                let g = greedy(&mut plan, 3);
                (g.clone(), plan.gains(&g[..2], None).unwrap())
            })
        };
        let (a, b) = (run(1), run(4));
        assert_eq!(a.0, b.0);
        assert!(
            a.1.iter()
                .zip(&b.1)
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
    }

    #[test]
    fn custom_scores_reproduce_the_built_in_variance() {
        let mut builtin = case(false, vec![search(70.0, 16)], Objective::Variance);
        let mut custom = case(false, vec![search(70.0, 16)], Objective::Custom);
        let (weights, support) = (custom.weights.clone(), custom.support.clone());
        let mut calls = 0;
        let mut score = |blocks: &[u32], m: &[Metrics]| -> Result<Vec<f64>> {
            calls += 1;
            Ok(blocks
                .iter()
                .zip(m)
                .map(|(&b, m)| {
                    let b = b as usize;
                    let v = if m.variance.is_nan() {
                        support[b]
                    } else {
                        m.variance
                    };
                    -weights[b] * v / support[b]
                })
                .collect())
        };
        for selected in [vec![], vec![10, 33], vec![33]] {
            let a = builtin.gains(&selected, None).unwrap();
            let b = custom.gains(&selected, Some(&mut score)).unwrap();
            assert!(
                a.iter()
                    .zip(&b)
                    .all(|(x, y)| x == y || (x - y).abs() < 1e-9)
            );
            let (sa, sb) = (
                builtin.score(&selected, None).unwrap(),
                custom.score(&selected, Some(&mut score)).unwrap(),
            );
            assert!((sa - sb).abs() < 1e-9, "{sa} {sb}");
        }
        assert!(calls <= 6, "one call per evaluation at most");
    }

    #[test]
    fn feasibility_spaces_holes_and_keeps_to_count_and_budget() {
        let plan = case(false, vec![search(70.0, 16)], Objective::Variance);
        let f = plan.feasible(&[0]).unwrap();
        assert!(!f[0] && !f[1] && !f[8] && !f[9] && f[2] && f[16]);
        assert_eq!(plan.neighbors(0, 30.0).unwrap(), vec![1, 8]);
        let f = plan.feasible(&[0, 2, 4, 6]).unwrap();
        assert!(f.iter().all(|f| !f), "budget of 4 holes spent");
        assert!(plan.feasible(&[0, 0]).is_err());
    }

    #[test]
    fn classification_progress_is_one_on_reaching_the_class() {
        let rule = [
            Condition {
                metric: 1,
                op: Op::GreaterEqual,
                threshold: 0.8,
            },
            Condition {
                metric: 4,
                op: Op::GreaterEqual,
                threshold: 3.0,
            },
        ];
        let m = |slope, n_holes| Metrics {
            slope,
            n_holes,
            ..Metrics::UNESTIMATED
        };
        let start = m(0.4, 3.0);
        assert_eq!(progress(&rule, &start, &start), 0.0);
        assert!((progress(&rule, &start, &m(0.6, 3.0)) - 0.5).abs() < 1e-12);
        assert_eq!(progress(&rule, &start, &m(0.9, 4.0)), 1.0);
        assert_eq!(progress(&rule, &start, &m(0.9, 2.0)), 0.0);
        assert!((progress(&rule, &m(0.4, 1.0), &m(0.6, 2.0)) - 0.5).abs() < 1e-12);
    }

    /// Candidates and targets on grids offset so that many composites tie in
    /// distance, an octant search capped per hole, then a wider pass.
    fn tied(
        objective: Objective,
        domains: bool,
        holes: usize,
        passes: Option<Vec<Search>>,
    ) -> Plan {
        let mut u = lcg(11);
        let tag = |p: &Point| domains.then(|| domain_at(p));
        let data: Vec<Sample> = (0..holes as u32)
            .flat_map(|h| {
                let (x, y) = (u() * 200.0, u() * 200.0);
                hole(x, y).into_iter().map(move |p| Sample {
                    domain: tag(&p),
                    ..Sample::with_hole(p, 0.0, h)
                })
            })
            .filter(|s| !(domains && s.domain == Some(2)))
            .collect();
        let candidates = (0..64)
            .map(|i| {
                let (x, y) = (12.5 + 25.0 * (i % 8) as f64, 12.5 + 25.0 * (i / 8) as f64);
                let composites = hole(x, y);
                Candidate {
                    domains: composites.iter().filter_map(tag).collect(),
                    composites,
                    collar: (x, y, 0.0),
                    cost: 30.0,
                    excluded: false,
                }
            })
            .collect();
        let points: Vec<Point> = (0..200)
            .map(|i| {
                let (x, y, z) = (i % 10, (i / 10) % 10, i / 100);
                (
                    5.0 + 20.0 * x as f64,
                    5.0 + 20.0 * y as f64,
                    -5.0 - 10.0 * z as f64,
                )
            })
            .collect();
        let soft = domains.then_some(crate::Soft::All(30.0));
        let passes = passes.unwrap_or_else(|| {
            vec![
                Search {
                    min_samples: 3,
                    max_samples: 6,
                    radius: 30.0,
                    octant: true,
                    max_per_hole: Some(2),
                    soft: soft.clone(),
                    ..Default::default()
                },
                Search {
                    min_samples: 2,
                    max_samples: 10,
                    radius: 55.0,
                    soft,
                    ..Default::default()
                },
            ]
        });
        let targets = Targets {
            domains: domains.then(|| points.iter().map(domain_at).collect()),
            weights: Some((0..200).map(|i| 1.0 + (i % 2) as f64).collect()),
            points,
        };
        Plan::new(
            kriging(false, passes),
            data,
            candidates,
            targets,
            objective,
            Constraints::default(),
            5.0,
        )
        .unwrap()
    }

    /// Objective with `selected` from a full re-run, without any cache.
    fn brute(plan: &Plan, selected: &[usize]) -> f64 {
        let m = rerun(plan, selected);
        (0..m.len()).map(|b| plan.builtin(b, &m[b])).sum()
    }

    #[test]
    fn cached_gains_equal_brute_force_over_random_moves() {
        for (objective, domains, holes) in [
            (Objective::Variance, false, 6),
            (rules(), true, 6),
            (rules(), false, 0),
        ] {
            let mut plan = tied(objective, domains, holes, None);
            let mut u = lcg(5 + holes as u64);
            let mut pick = |n: usize| ((u() * n as f64) as usize).min(n - 1);
            let mut selected: Vec<usize> = vec![];
            for step in 0..14 {
                if !selected.is_empty() && (step % 3 == 2 || selected.len() > 5) {
                    selected.remove(pick(selected.len()));
                } else {
                    let free: Vec<usize> = (0..64).filter(|c| !selected.contains(c)).collect();
                    selected.push(free[pick(free.len())]);
                }
                let gains = plan.gains(&selected, None).unwrap();
                let base = brute(&plan, &selected);
                let score = plan.score(&selected, None).unwrap();
                assert!((score - base).abs() < 1e-9, "{step}: {score} {base}");
                let corner = [54, 55, 62, 63];
                let checked: Vec<usize> = (0..8).map(|_| pick(64)).collect();
                for c in checked.into_iter().chain(corner) {
                    if selected.contains(&c) {
                        assert_eq!(gains[c], f64::NEG_INFINITY);
                        continue;
                    }
                    let with: Vec<usize> = selected.iter().copied().chain([c]).collect();
                    let gain = brute(&plan, &with) - base;
                    assert!(
                        (gains[c] - gain).abs() < 1e-9,
                        "{step} {c}: {} {gain}",
                        gains[c]
                    );
                }
            }
        }
    }

    /// Hole 61 is soft data for a corner target that hole 55 opens but
    /// does not reach, so adding 61 changes the gain of 55.
    #[test]
    fn opening_gains_drop_when_their_domain_changes_beyond_reach() {
        let pass = Search {
            min_samples: 1,
            max_samples: 6,
            radius: 30.0,
            soft: Some(crate::Soft::All(30.0)),
            ..Default::default()
        };
        let mut plan = tied(rules(), true, 6, Some(vec![pass]));
        let b = plan.targets.iter().position(|t| *t == (165.0, 185.0, -5.0));
        let b = b.unwrap() as u32;
        assert!(!plan.reach[55].contains(&b) && plan.reach[61].contains(&b));
        plan.gains(&[], None).unwrap();
        let gains = plan.gains(&[61], None).unwrap();
        let gain = brute(&plan, &[61, 55]) - brute(&plan, &[61]);
        assert!((gains[55] - gain).abs() < 1e-9, "{} {gain}", gains[55]);
    }
}
