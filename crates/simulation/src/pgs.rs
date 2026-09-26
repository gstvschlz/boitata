//! Plurigaussian simulation (PGS) of categorical facies.
//!
//! Simulates facies by thresholding independent latent Gaussian random fields
//! through a **truncation rule**: a partition of the (`Y₁`, …, `Yₙ`) space
//! into boxes, each mapped to a facies. One field reproduces ordered
//! (sequential) facies; more fields reproduce richer contact relationships.
//! A [`Hierarchy`] builds the rule from facies proportions by splitting one
//! field at a time, so facies only touch where the tree lets them.
//!
//! Conditioning is the standard PGS pipeline:
//! 1. each datum's observed facies constrains every field to an interval
//!    (the facies' box on that axis);
//! 2. a Gibbs sweep ([`crate::gibbs`]) draws consistent Gaussian values at the
//!    data honoring those intervals + the spatial correlation;
//! 3. each field is conditioned to those values by turning bands
//!    ([`crate::turning_bands::conditional_gaussian_field`]);
//! 4. the truncation rule classifies each grid node.

use crate::error::{Result, SimError};
use crate::gibbs::{GibbsParams, gibbs};
use crate::turning_bands::{TurningBandsParams, conditional_gaussian_field};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use transforms::normal::{phi, probit};
use variogram::Variogram;

const ANY: (f64, f64) = (f64::NEG_INFINITY, f64::INFINITY);

/// A box of the latent Gaussian space mapped to a facies: `bounds[k]` is the
/// `(low, high]` interval of field `k`; fields past `bounds` are unbounded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Region {
    #[serde(with = "ceres_core::nonfinite")]
    pub bounds: Vec<(f64, f64)>,
    pub facies: usize,
}

impl Region {
    fn bound(&self, field: usize) -> (f64, f64) {
        self.bounds.get(field).copied().unwrap_or(ANY)
    }

    fn contains(&self, y: &[f64]) -> bool {
        self.bounds
            .iter()
            .zip(y)
            .all(|(&(lo, hi), &v)| v > lo && v <= hi)
    }
}

/// A truncation rule: regions that partition the Gaussian space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TruncationRule {
    pub regions: Vec<Region>,
}

impl TruncationRule {
    /// Ordered single-field rule from facies **proportions** (summing to 1): the
    /// `Y₁` axis is cut at the cumulative Gaussian quantiles.
    pub fn from_proportions(proportions: &[f64]) -> Self {
        let order = Hierarchy::Split {
            field: 0,
            children: (0..proportions.len()).map(Hierarchy::Facies).collect(),
        };
        order.rule(proportions)
    }

    /// Number of latent fields the rule thresholds.
    pub fn fields(&self) -> usize {
        self.regions
            .iter()
            .map(|r| r.bounds.len())
            .max()
            .unwrap_or(0)
    }

    /// Facies at a point of the Gaussian space; falls back to the first
    /// region if uncovered.
    pub fn classify(&self, y: &[f64]) -> usize {
        self.regions
            .iter()
            .find(|r| r.contains(y))
            .or(self.regions.first())
            .map_or(0, |r| r.facies)
    }

    /// Per-field intervals a facies imposes: the bounding box over all regions
    /// mapping to that facies; unconstrained for an unknown facies.
    fn facies_intervals(&self, facies: usize, fields: usize) -> Vec<(f64, f64)> {
        let mut out = vec![(f64::INFINITY, f64::NEG_INFINITY); fields];
        let mut found = false;
        for r in self.regions.iter().filter(|r| r.facies == facies) {
            found = true;
            for (k, b) in out.iter_mut().enumerate() {
                let (lo, hi) = r.bound(k);
                *b = (b.0.min(lo), b.1.max(hi));
            }
        }
        if found { out } else { vec![ANY; fields] }
    }
}

/// A hierarchical truncation rule: each split cuts one latent field into
/// ordered slices, one per child, sized by the facies proportions below it.
/// Facies under different children of a split touch only across that cut.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Hierarchy {
    Facies(usize),
    Split {
        field: usize,
        children: Vec<Hierarchy>,
    },
}

impl Hierarchy {
    /// Checks the tree holds every facies `0..k` exactly once, splits only
    /// fields below `fields`, and every split has children.
    pub fn validate(&self, k: usize, fields: usize) -> Result<()> {
        let mut seen = vec![false; k];
        self.check(&mut seen, fields)?;
        if seen.iter().all(|&s| s) {
            Ok(())
        } else {
            Err(invalid("the rule must hold every facies once"))
        }
    }

    fn check(&self, seen: &mut [bool], fields: usize) -> Result<()> {
        match self {
            Self::Facies(f) => match seen.get_mut(*f) {
                Some(s) if !*s => {
                    *s = true;
                    Ok(())
                }
                Some(_) => Err(invalid(format!("facies {f} appears twice in the rule"))),
                None => Err(invalid(format!("facies {f} has no proportion"))),
            },
            Self::Split { field, children } => {
                if *field >= fields {
                    return Err(invalid(format!(
                        "the rule splits field {field} of {fields}"
                    )));
                }
                if children.is_empty() {
                    return Err(invalid("a split needs children"));
                }
                children.iter().try_for_each(|c| c.check(seen, fields))
            }
        }
    }

    fn mass(&self, proportions: &[f64]) -> f64 {
        match self {
            Self::Facies(f) => proportions.get(*f).copied().unwrap_or(0.0),
            Self::Split { children, .. } => children.iter().map(|c| c.mass(proportions)).sum(),
        }
    }

    /// Latent fields the tree splits.
    pub fn fields(&self) -> usize {
        match self {
            Self::Facies(_) => 0,
            Self::Split { field, children } => children
                .iter()
                .map(Self::fields)
                .fold(field + 1, usize::max),
        }
    }

    /// Box rule reproducing `proportions` with independent standard fields:
    /// each child takes its share of the parent's probability along the split
    /// field.
    pub fn rule(&self, proportions: &[f64]) -> TruncationRule {
        let mut regions = vec![];
        let mut cell = vec![(0.0, 1.0); self.fields()];
        self.boxes(proportions, &mut cell, &mut regions);
        TruncationRule { regions }
    }

    fn boxes(&self, proportions: &[f64], cell: &mut [(f64, f64)], out: &mut Vec<Region>) {
        match self {
            Self::Facies(f) => out.push(Region {
                bounds: cell
                    .iter()
                    .map(|&(a, b)| (quantile(a), quantile(b)))
                    .collect(),
                facies: *f,
            }),
            Self::Split { field, children } => {
                let whole = cell[*field];
                for (child, slice) in children.iter().zip(slices(children, proportions, whole)) {
                    cell[*field] = slice;
                    child.boxes(proportions, cell, out);
                }
                cell[*field] = whole;
            }
        }
    }

    /// Facies at a point `y` of the Gaussian space under `proportions`; the
    /// same as `self.rule(proportions).classify(y)` without building the rule.
    pub fn classify(&self, y: &[f64], proportions: &[f64]) -> usize {
        let mut cell = vec![(0.0, 1.0); y.len()];
        let mut node = self;
        loop {
            match node {
                Self::Facies(f) => return *f,
                Self::Split { field, children } => {
                    let u = phi(y[*field]);
                    let parts = slices(children, proportions, cell[*field]);
                    let last = children.len() - 1;
                    let i = parts.iter().position(|s| u <= s.1).unwrap_or(last);
                    cell[*field] = parts[i];
                    node = &children[i];
                }
            }
        }
    }
}

/// The probability interval `(a, b)` of a split field cut into one slice per
/// child, sized by the child's mass (equal slices when all are empty).
fn slices(children: &[Hierarchy], proportions: &[f64], (a, b): (f64, f64)) -> Vec<(f64, f64)> {
    let masses: Vec<f64> = children.iter().map(|c| c.mass(proportions)).collect();
    let total: f64 = masses.iter().sum();
    let n = children.len();
    let mut cum = 0.0;
    let mut out = Vec::with_capacity(n);
    for (i, m) in masses.iter().enumerate() {
        let share = |c: f64| if total > 0.0 { c / total } else { c / n as f64 };
        let lo = a + (b - a) * share(cum);
        cum += if total > 0.0 { *m } else { 1.0 };
        let hi = if i + 1 == n {
            b
        } else {
            a + (b - a) * share(cum)
        };
        out.push((lo, hi));
    }
    out
}

fn quantile(p: f64) -> f64 {
    match p {
        p if p <= 0.0 => f64::NEG_INFINITY,
        p if p >= 1.0 => f64::INFINITY,
        p => probit(p),
    }
}

fn invalid(msg: impl Into<String>) -> SimError {
    SimError::InvalidParameters(msg.into())
}

/// PGS parameters.
#[derive(Debug, Clone)]
pub struct PgsParams {
    pub bands: TurningBandsParams,
    pub gibbs: GibbsParams,
    pub seed: u64,
}

impl Default for PgsParams {
    fn default() -> Self {
        Self {
            bands: TurningBandsParams::default(),
            gibbs: GibbsParams::default(),
            seed: 1,
        }
    }
}

/// Conditional plurigaussian simulation: facies index per grid node.
///
/// `data_facies` are the observed facies at `data_locs`. `variograms` are the
/// (unit-sill) Gaussian variograms of the independent latent fields, one per
/// field the rule thresholds.
pub fn plurigaussian(
    data_locs: &[(f64, f64, f64)],
    data_facies: &[usize],
    grid: &[(f64, f64, f64)],
    variograms: &[Variogram],
    rule: &TruncationRule,
    params: &PgsParams,
) -> Result<Vec<usize>> {
    check(data_locs, data_facies, variograms, rule.fields())?;
    let fields = variograms.len();
    let intervals: Vec<Vec<(f64, f64)>> = data_facies
        .iter()
        .map(|&f| rule.facies_intervals(f, fields))
        .collect();
    let values = latent(data_locs, &intervals, grid, variograms, params)?;
    Ok(values.iter().map(|y| rule.classify(y)).collect())
}

/// Plurigaussian simulation with locally varying proportions: the thresholds
/// of `hierarchy` follow `data_proportions` at each datum and
/// `grid_proportions` at each node, one row of facies proportions each.
#[allow(clippy::too_many_arguments)]
pub fn plurigaussian_local(
    data_locs: &[(f64, f64, f64)],
    data_facies: &[usize],
    grid: &[(f64, f64, f64)],
    variograms: &[Variogram],
    hierarchy: &Hierarchy,
    data_proportions: &[Vec<f64>],
    grid_proportions: &[Vec<f64>],
    params: &PgsParams,
) -> Result<Vec<usize>> {
    check(data_locs, data_facies, variograms, hierarchy.fields())?;
    if data_proportions.len() != data_locs.len() || grid_proportions.len() != grid.len() {
        return Err(invalid("one row of proportions per datum and per node"));
    }
    let fields = variograms.len();
    let intervals: Vec<Vec<(f64, f64)>> = data_facies
        .iter()
        .zip(data_proportions)
        .map(|(&f, p)| hierarchy.rule(p).facies_intervals(f, fields))
        .collect();
    let values = latent(data_locs, &intervals, grid, variograms, params)?;
    Ok(values
        .iter()
        .zip(grid_proportions)
        .map(|(y, p)| hierarchy.classify(y, p))
        .collect())
}

fn check(
    data_locs: &[(f64, f64, f64)],
    data_facies: &[usize],
    variograms: &[Variogram],
    fields: usize,
) -> Result<()> {
    if data_locs.len() != data_facies.len() {
        return Err(invalid("data length mismatch"));
    }
    if data_locs.is_empty() {
        return Err(SimError::InsufficientData("no conditioning data".into()));
    }
    if variograms.is_empty() || fields > variograms.len() {
        return Err(invalid(format!(
            "the rule thresholds {fields} fields but {} variograms were given",
            variograms.len()
        )));
    }
    Ok(())
}

/// Latent Gaussian values at every node, one per field, conditioned by Gibbs
/// draws at the data within their facies `intervals`.
fn latent(
    data_locs: &[(f64, f64, f64)],
    intervals: &[Vec<(f64, f64)>],
    grid: &[(f64, f64, f64)],
    variograms: &[Variogram],
    params: &PgsParams,
) -> Result<Vec<Vec<f64>>> {
    if grid.is_empty() {
        return Ok(vec![]);
    }
    let mut rng = StdRng::seed_from_u64(params.seed);
    let mut values = vec![Vec::with_capacity(variograms.len()); grid.len()];
    for (k, vg) in variograms.iter().enumerate() {
        let k = k as u64;
        let bounds: Vec<(f64, f64)> = intervals.iter().map(|b| b[k as usize]).collect();
        let g = GibbsParams {
            seed: params.seed.wrapping_add(101 * k),
            ..params.gibbs.clone()
        };
        let at_data = gibbs(data_locs, &bounds, vg, &g)?;
        let b = TurningBandsParams {
            seed: params.seed.wrapping_add(11 + 200 * k),
            ..params.bands.clone()
        };
        let field = conditional_gaussian_field(data_locs, &at_data, grid, vg, &b, &mut rng)?;
        for (v, y) in values.iter_mut().zip(field) {
            v.push(y);
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use variogram::model::Model;

    fn split(field: usize, children: Vec<Hierarchy>) -> Hierarchy {
        Hierarchy::Split { field, children }
    }

    fn leaf(f: usize) -> Hierarchy {
        Hierarchy::Facies(f)
    }

    #[test]
    fn proportions_rule_partitions_and_thresholds() {
        let rule = TruncationRule::from_proportions(&[0.5, 0.5]);
        assert_eq!(rule.classify(&[-1.0]), 0);
        assert_eq!(rule.classify(&[1.0]), 1);
        let i0 = rule.facies_intervals(0, 2);
        assert!(i0[0].0.is_infinite() && i0[0].1.abs() < 1e-9, "{i0:?}");
        assert_eq!(i0[1], ANY);
    }

    #[test]
    fn hierarchy_boxes_carry_the_proportions() {
        // Y₁ splits facies 0 from the rest; Y₂ orders 1 and 2; Y₃ splits 2 from 3.
        let tree = split(
            0,
            vec![
                leaf(0),
                split(1, vec![leaf(1), split(2, vec![leaf(2), leaf(3)])]),
            ],
        );
        let p = [0.4, 0.3, 0.2, 0.1];
        tree.validate(4, 3).unwrap();
        let rule = tree.rule(&p);
        for r in &rule.regions {
            let mass: f64 = r
                .bounds
                .iter()
                .map(|&(lo, hi)| transforms::normal::phi(hi) - transforms::normal::phi(lo))
                .product();
            assert!((mass - p[r.facies]).abs() < 1e-8, "{r:?} mass {mass}");
        }
        assert!(tree.validate(4, 2).is_err());
        assert!(split(0, vec![leaf(0), leaf(0)]).validate(1, 1).is_err());
        assert!(split(0, vec![leaf(0)]).validate(2, 1).is_err());
    }

    fn plane(n: usize) -> Vec<(f64, f64, f64)> {
        (0..n * n)
            .map(|i| ((i % n) as f64, (i / n) as f64, 0.0))
            .collect()
    }

    fn quiet_params(seed: u64) -> PgsParams {
        PgsParams {
            bands: TurningBandsParams {
                n_bands: 400,
                seed,
                ..Default::default()
            },
            gibbs: GibbsParams {
                seed,
                ..Default::default()
            },
            seed,
        }
    }

    #[test]
    fn three_field_hierarchy_reproduces_proportions_and_forbids_contacts() {
        // Facies 0 | 1 | {2, 3} along Y₁: 0 never touches 2 or 3.
        let tree = split(
            0,
            vec![
                leaf(0),
                leaf(1),
                split(1, vec![leaf(2), split(2, vec![leaf(3), leaf(4)])]),
            ],
        );
        let p = [0.3, 0.25, 0.2, 0.15, 0.1];
        let rule = tree.rule(&p);
        let n = 80;
        let grid = plane(n);
        let data = vec![(40.0, 40.0, 0.0)];
        let vgs = vec![Variogram::single(Model::Gaussian, 1.0, 20.0); 3];
        let mut share = [0.0; 5];
        let reals = 16;
        for seed in 0..reals {
            let f = plurigaussian(&data, &[1], &grid, &vgs, &rule, &quiet_params(seed)).unwrap();
            for &c in &f {
                share[c] += 1.0 / (grid.len() * reals as usize) as f64;
            }
            for i in 0..grid.len() {
                for j in [i + 1, i + n] {
                    if j < grid.len() && (j != i + 1 || (i + 1) % n != 0) {
                        let pair = (f[i].min(f[j]), f[i].max(f[j]));
                        assert!(
                            !matches!(pair, (0, 2) | (0, 3) | (0, 4)),
                            "forbidden contact {pair:?} at seed {seed}"
                        );
                    }
                }
            }
        }
        for (s, q) in share.iter().zip(p) {
            assert!((s - q).abs() < 0.03, "shares {share:?} vs {p:?}");
        }
    }

    #[test]
    fn walking_the_tree_matches_its_boxes() {
        let tree = split(0, vec![leaf(2), split(1, vec![leaf(0), leaf(3)]), leaf(1)]);
        let p = [0.1, 0.4, 0.2, 0.3];
        let rule = tree.rule(&p);
        for i in 0..400 {
            let y = [(i % 20) as f64 / 5.0 - 2.0, (i / 20) as f64 / 5.0 - 2.0];
            assert_eq!(tree.classify(&y, &p), rule.classify(&y), "at {y:?}");
        }
    }

    #[test]
    fn local_proportions_follow_their_trend() {
        // Facies 0 falls from 80 % in the west to 20 % in the east.
        let tree = split(0, vec![leaf(0), split(1, vec![leaf(1), leaf(2)])]);
        let n = 80;
        let grid = plane(n);
        let at = |x: f64| {
            let p0 = 0.8 - 0.6 * x / (n - 1) as f64;
            vec![p0, (1.0 - p0) / 2.0, (1.0 - p0) / 2.0]
        };
        let props: Vec<Vec<f64>> = grid.iter().map(|g| at(g.0)).collect();
        let data = vec![(40.0, 40.0, 0.0)];
        let vgs = vec![Variogram::single(Model::Spherical, 1.0, 15.0); 2];
        let (mut west, mut east) = (0.0, 0.0);
        let reals = 10;
        for seed in 0..reals {
            let f = plurigaussian_local(
                &data,
                &[1],
                &grid,
                &vgs,
                &tree,
                &[at(40.0)],
                &props,
                &quiet_params(seed),
            )
            .unwrap();
            for (g, &c) in grid.iter().zip(&f) {
                let share = (c == 0) as u8 as f64 / (reals as f64 * (n * n / 4) as f64);
                if g.0 < 20.0 {
                    west += share;
                } else if g.0 >= 60.0 {
                    east += share;
                }
            }
        }
        let expect = |x0: usize| (x0..x0 + 20).map(|x| at(x as f64)[0]).sum::<f64>() / 20.0;
        assert!(
            (west - expect(0)).abs() < 0.05,
            "west {west} vs {}",
            expect(0)
        );
        assert!(
            (east - expect(60)).abs() < 0.05,
            "east {east} vs {}",
            expect(60)
        );
        assert!(
            plurigaussian_local(
                &data,
                &[1],
                &grid,
                &vgs,
                &tree,
                &[],
                &props,
                &quiet_params(0)
            )
            .is_err()
        );
    }

    #[test]
    fn realizations_depend_on_the_seed_not_the_thread_count() {
        let rule = split(0, vec![leaf(0), split(1, vec![leaf(1), leaf(2)])]).rule(&[0.4, 0.3, 0.3]);
        let grid = plane(30);
        let data = vec![(3.0, 4.0, 0.0), (20.0, 25.0, 0.0), (12.0, 8.0, 0.0)];
        let vgs = vec![Variogram::single(Model::Spherical, 1.0, 10.0); 2];
        let run = |threads: usize, seed: u64| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    plurigaussian(&data, &[0, 1, 2], &grid, &vgs, &rule, &quiet_params(seed))
                        .unwrap()
                })
        };
        assert_eq!(run(1, 5), run(4, 5));
        assert_ne!(run(4, 5), run(4, 6));
    }

    #[test]
    fn conditional_facies_honor_data() {
        let data_locs = vec![
            (0.0, 0.0, 0.0),
            (100.0, 0.0, 0.0),
            (0.0, 100.0, 0.0),
            (100.0, 100.0, 0.0),
        ];
        let data_facies = vec![0usize, 1, 2, 0];
        let mut grid = data_locs.clone();
        grid.push((50.0, 50.0, 0.0));
        let vg = Variogram::single(Model::Spherical, 1.0, 250.0);
        let rule = split(0, vec![leaf(0), split(1, vec![leaf(1), leaf(2)])]).rule(&[0.4, 0.3, 0.3]);
        let params = PgsParams {
            bands: TurningBandsParams {
                n_bands: 200,
                step: Some(5.0),
                seed: 3,
                ..Default::default()
            },
            gibbs: GibbsParams {
                iterations: 200,
                burn_in: 50,
                seed: 3,
            },
            seed: 3,
        };
        let vgs = [vg.clone(), vg];
        let facies = plurigaussian(&data_locs, &data_facies, &grid, &vgs, &rule, &params).unwrap();
        assert_eq!(&facies[..4], &data_facies[..]);
        assert!(plurigaussian(&data_locs, &data_facies, &grid, &vgs[..1], &rule, &params).is_err());
    }

    #[test]
    fn explicit_regions_classify() {
        let rule = TruncationRule {
            regions: vec![
                Region {
                    bounds: vec![(f64::NEG_INFINITY, 0.0)],
                    facies: 0,
                },
                Region {
                    bounds: vec![(0.0, f64::INFINITY), (f64::NEG_INFINITY, 0.0)],
                    facies: 1,
                },
                Region {
                    bounds: vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
                    facies: 2,
                },
            ],
        };
        assert_eq!(rule.fields(), 2);
        assert_eq!(rule.classify(&[-1.0, 5.0]), 0);
        assert_eq!(rule.classify(&[1.0, -1.0]), 1);
        assert_eq!(rule.classify(&[1.0, 1.0]), 2);
    }
}
