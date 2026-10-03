//! Uncertainty summaries accumulated realization by realization, so an ensemble
//! is never held or returned in full unless asked for.
//!
//! Realizations are simulated in parallel batches and folded in realization
//! order, so every summary is identical for any number of threads.

use boitata_core::{BlockModel, Progress};
use rayon::prelude::*;

use crate::error::{Result, SimError};

/// Runs realizations `0..n` in parallel batches and hands them to `add` in
/// order; at most one batch (one realization per thread) is held at a time.
/// `progress`, when given, counts finished realizations.
fn run<R: Send>(
    n: usize,
    simulate: impl Fn(usize) -> Result<R> + Sync,
    mut add: impl FnMut(R) -> Result<()>,
    progress: Option<&Progress>,
) -> Result<()> {
    if n == 0 {
        return Err(SimError::InvalidParameters(
            "need at least one realization".into(),
        ));
    }
    let batch = rayon::current_num_threads().max(1);
    for start in (0..n).step_by(batch) {
        let done: Vec<R> = (start..(start + batch).min(n))
            .into_par_iter()
            .map(&simulate)
            .collect::<Result<_>>()?;
        let count = done.len() as u64;
        done.into_iter().try_for_each(&mut add)?;
        tick(progress, count);
    }
    Ok(())
}

fn tick(progress: Option<&Progress>, n: u64) {
    if let Some(p) = progress {
        p.inc_by(n);
    }
}

/// Which realizations a summary returns beside its statistics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Keep {
    /// None: each realization updates the summary, then is freed.
    #[default]
    None,
    /// Every realization.
    All,
    /// These 0-based realizations, returned in realization order.
    Indices(Vec<usize>),
}

impl Keep {
    pub fn keeps(&self, k: usize) -> bool {
        match self {
            Keep::None => false,
            Keep::All => true,
            Keep::Indices(indices) => indices.contains(&k),
        }
    }

    /// The kept realizations of `0..n`, ascending.
    pub fn kept(&self, n: usize) -> Vec<usize> {
        (0..n).filter(|&k| self.keeps(k)).collect()
    }

    /// Every index is below `n` and listed once.
    pub fn validate(&self, n: usize) -> Result<()> {
        let Keep::Indices(indices) = self else {
            return Ok(());
        };
        for (position, &k) in indices.iter().enumerate() {
            if k >= n {
                return Err(SimError::InvalidParameters(format!(
                    "keep lists realization {k}, but there are {n}; indices start at 0"
                )));
            }
            if indices[..position].contains(&k) {
                return Err(SimError::InvalidParameters(format!(
                    "keep lists realization {k} twice"
                )));
            }
        }
        Ok(())
    }
}

/// What to accumulate for a continuous variable.
#[derive(Debug, Clone, Default)]
pub struct ContinuousOptions {
    pub cutoffs: Vec<f64>,
    /// Probabilities in `[0, 1]`; exact, but they hold every value.
    pub quantiles: Vec<f64>,
    /// Realizations to return beside the statistics.
    pub keep: Keep,
    /// Grade–tonnage curves to accumulate per realization.
    pub tonnage: Option<TonnageOptions>,
}

/// Grade–tonnage curves of each realization: tonnes, metal and mean grade
/// of the targets at or above each cutoff, as `grade_tonnage` counts them.
#[derive(Debug, Clone, Default)]
pub struct TonnageOptions {
    pub cutoffs: Vec<f64>,
    /// Tonnes of each target.
    pub tonnes: Vec<f64>,
    /// Category code of each target, or `None` outside every category; a
    /// curve per category besides the curve over all targets.
    pub categories: Option<Vec<Option<u32>>>,
    /// Name of each category code.
    pub names: Vec<String>,
}

/// Per-realization grade–tonnage curves, `[group][cutoff][realization]`;
/// groups are the categories ascending, then all targets (`None`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RealizedTonnage {
    pub cutoffs: Vec<f64>,
    pub groups: Vec<Option<u32>>,
    /// Name of each category code.
    pub names: Vec<String>,
    pub tonnage: Vec<Vec<Vec<f64>>>,
    pub metal: Vec<Vec<Vec<f64>>>,
}

/// One row of [`RealizedTonnage::quantiles`].
#[derive(Debug, Clone, PartialEq)]
pub struct TonnageQuantile {
    pub group: Option<u32>,
    pub cutoff: f64,
    pub probability: f64,
    pub tonnage: f64,
    pub metal: f64,
    pub mean_grade: f64,
}

impl RealizedTonnage {
    /// Each quantity's quantile across realizations at each group, cutoff
    /// and probability, taken independently: the P50 tonnage and the P50
    /// grade may come from different realizations.
    pub fn quantiles(&self, probabilities: &[f64]) -> Result<Vec<TonnageQuantile>> {
        if probabilities.iter().any(|p| !(0.0..=1.0).contains(p)) {
            return Err(SimError::InvalidParameters(
                "probabilities must be in [0, 1]".into(),
            ));
        }
        let at = |values: &[f64], p: f64| {
            let mut v: Vec<f64> = values.iter().copied().filter(|x| !x.is_nan()).collect();
            v.sort_by(f64::total_cmp);
            if v.is_empty() {
                f64::NAN
            } else {
                quantile_sorted(&v, p)
            }
        };
        let mut rows = vec![];
        for (g, &group) in self.groups.iter().enumerate() {
            for (c, &cutoff) in self.cutoffs.iter().enumerate() {
                let (t, m) = (&self.tonnage[g][c], &self.metal[g][c]);
                let grade: Vec<f64> = t
                    .iter()
                    .zip(m)
                    .map(|(t, m)| if *t > 0.0 { m / t } else { f64::NAN })
                    .collect();
                for &probability in probabilities {
                    rows.push(TonnageQuantile {
                        group,
                        cutoff,
                        probability,
                        tonnage: at(t, probability),
                        metal: at(m, probability),
                        mean_grade: at(&grade, probability),
                    });
                }
            }
        }
        Ok(rows)
    }
}

/// Per-target and per-realization summary of a continuous ensemble. Per-cutoff
/// and per-quantile fields are indexed `[cutoff or quantile][target or realization]`.
#[derive(Debug, Clone)]
pub struct ContinuousSummary {
    pub n: usize,
    pub mean: Vec<f64>,
    /// Population variance across realizations.
    pub variance: Vec<f64>,
    pub cutoffs: Vec<f64>,
    /// Fraction of realizations above each cutoff.
    pub probability_above: Vec<Vec<f64>>,
    /// Mean of the values above each cutoff; NaN where none is.
    pub mean_above: Vec<Vec<f64>>,
    pub quantiles: Vec<f64>,
    pub quantile_values: Vec<Vec<f64>>,
    /// Mean of each realization over all targets.
    pub realization_mean: Vec<f64>,
    /// Fraction of targets above each cutoff in each realization.
    pub realization_above: Vec<Vec<f64>>,
    /// Indices of the kept realizations, ascending.
    pub kept: Vec<usize>,
    /// The kept realizations, one row per entry of `kept`.
    pub realizations: Vec<Vec<f64>>,
    /// Grade–tonnage curves per realization, when requested.
    pub grade_tonnage: Option<RealizedTonnage>,
}

/// Center of [`ContinuousSummary::relative_error`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Center {
    Mean,
    Median,
}

impl ContinuousSummary {
    /// Coefficient of variation across realizations, the standard deviation
    /// over the mean; NaN where the mean is 0.
    pub fn cv(&self) -> Vec<f64> {
        self.mean
            .iter()
            .zip(&self.variance)
            .map(|(m, v)| if *m == 0.0 { f64::NAN } else { v.sqrt() / m })
            .collect()
    }

    /// Half the central `confidence` interval over the mean or median,
    /// `(q_hi - q_lo) / (2 × center)`, from the stored quantiles; NaN where
    /// the center is 0. Errors naming the quantiles to request when absent.
    pub fn relative_error(&self, confidence: f64, center: Center) -> Result<Vec<f64>> {
        if !(confidence > 0.0 && confidence < 1.0) {
            return Err(SimError::InvalidParameters(
                "confidence must be in (0, 1)".into(),
            ));
        }
        let (lo, hi) = ((1.0 - confidence) / 2.0, (1.0 + confidence) / 2.0);
        let mut needed = vec![lo, hi];
        if center == Center::Median {
            needed.push(0.5);
        }
        let find = |q: f64| self.quantiles.iter().position(|x| (x - q).abs() < 1e-9);
        let missing: Vec<String> = needed
            .iter()
            .filter(|q| find(**q).is_none())
            .map(|q| format!("{}", (q * 1e9).round() / 1e9))
            .collect();
        if !missing.is_empty() {
            return Err(SimError::InvalidParameters(format!(
                "relative error at confidence {confidence} needs quantiles {}; add them to quantiles=",
                missing.join(", ")
            )));
        }
        let (lo, hi) = (
            &self.quantile_values[find(lo).unwrap()],
            &self.quantile_values[find(hi).unwrap()],
        );
        let middle = match center {
            Center::Mean => &self.mean,
            Center::Median => &self.quantile_values[find(0.5).unwrap()],
        };
        Ok((0..middle.len())
            .map(|i| {
                if middle[i] == 0.0 {
                    f64::NAN
                } else {
                    (hi[i] - lo[i]) / (2.0 * middle[i])
                }
            })
            .collect())
    }
}

/// Summarizes `n` realizations of a continuous variable; `simulate(k)` returns
/// realization `k` over the same targets every time.
pub fn continuous(
    n: usize,
    options: &ContinuousOptions,
    simulate: impl Fn(usize) -> Result<Vec<f64>> + Sync,
    progress: Option<&Progress>,
) -> Result<ContinuousSummary> {
    let mut acc = Accumulator::new(n, options)?;
    run(n, simulate, |values| acc.add(values), progress)?;
    Ok(acc.finish())
}

/// As [`continuous`], simulating `batch` realizations at a time:
/// `simulate(ks)` returns realizations `ks`, in order. The summary does not
/// depend on `batch`.
pub fn continuous_batched(
    n: usize,
    options: &ContinuousOptions,
    batch: usize,
    mut simulate: impl FnMut(std::ops::Range<usize>) -> Result<Vec<Vec<f64>>>,
    progress: Option<&Progress>,
) -> Result<ContinuousSummary> {
    if n == 0 {
        return Err(SimError::InvalidParameters(
            "need at least one realization".into(),
        ));
    }
    let mut acc = Accumulator::new(n, options)?;
    let batch = batch.max(1);
    for start in (0..n).step_by(batch) {
        let ks = start..(start + batch).min(n);
        let done = simulate(ks.clone())?;
        if done.len() != ks.len() {
            return Err(SimError::InvalidParameters(format!(
                "{} realizations for {} indices",
                done.len(),
                ks.len()
            )));
        }
        done.into_iter().try_for_each(|v| acc.add(v))?;
        tick(progress, ks.len() as u64);
    }
    Ok(acc.finish())
}

/// As [`continuous`], simulating realizations `batch` at a time:
/// `simulate(range)` prepares a batch and `realization(&batch, i)` returns
/// its `i`-th realization; realizations reach the summary in order, one at a
/// time, so a batch never holds them all in grades. The summary does not
/// depend on `batch`.
pub fn continuous_in_batches<B>(
    n: usize,
    options: &ContinuousOptions,
    batch: usize,
    mut simulate: impl FnMut(std::ops::Range<usize>) -> Result<B>,
    realization: impl Fn(&B, usize) -> Result<Vec<f64>>,
    progress: Option<&Progress>,
) -> Result<ContinuousSummary> {
    if n == 0 {
        return Err(SimError::InvalidParameters(
            "need at least one realization".into(),
        ));
    }
    let mut acc = Accumulator::new(n, options)?;
    let batch = batch.clamp(1, n);
    for start in (0..n).step_by(batch) {
        let ks = start..(start + batch).min(n);
        let prepared = simulate(ks.clone())?;
        for i in 0..ks.len() {
            acc.add(realization(&prepared, i)?)?;
        }
        tick(progress, ks.len() as u64);
    }
    Ok(acc.finish())
}

/// Summarizes `n` realizations of `variables` continuous variables at once;
/// `simulate(k)` returns realization `k` of every variable over the same
/// targets every time.
pub fn continuous_many(
    n: usize,
    variables: usize,
    options: &ContinuousOptions,
    simulate: impl Fn(usize) -> Result<Vec<Vec<f64>>> + Sync,
    progress: Option<&Progress>,
) -> Result<Vec<ContinuousSummary>> {
    let mut accs = (0..variables)
        .map(|_| Accumulator::new(n, options))
        .collect::<Result<Vec<_>>>()?;
    run(
        n,
        simulate,
        |values: Vec<Vec<f64>>| {
            if values.len() != variables {
                return Err(SimError::InvalidParameters(format!(
                    "one realization per variable ({variables}) needed"
                )));
            }
            accs.iter_mut().zip(values).try_for_each(|(a, v)| a.add(v))
        },
        progress,
    )?;
    Ok(accs.into_iter().map(Accumulator::finish).collect())
}

/// As [`continuous_many`], `batch` realizations at a time: `simulate(ks)`
/// returns, for each realization of `ks` in order, every variable.
pub fn continuous_many_batched(
    n: usize,
    variables: usize,
    options: &ContinuousOptions,
    batch: usize,
    mut simulate: impl FnMut(std::ops::Range<usize>) -> Result<Vec<Vec<Vec<f64>>>>,
    progress: Option<&Progress>,
) -> Result<Vec<ContinuousSummary>> {
    if n == 0 {
        return Err(SimError::InvalidParameters(
            "need at least one realization".into(),
        ));
    }
    let mut accs = (0..variables)
        .map(|_| Accumulator::new(n, options))
        .collect::<Result<Vec<_>>>()?;
    let batch = batch.max(1);
    for start in (0..n).step_by(batch) {
        let ks = start..(start + batch).min(n);
        let done = simulate(ks.clone())?;
        if done.len() != ks.len() {
            return Err(SimError::InvalidParameters(format!(
                "{} realizations for {} indices",
                done.len(),
                ks.len()
            )));
        }
        for values in done {
            if values.len() != variables {
                return Err(SimError::InvalidParameters(format!(
                    "one realization per variable ({variables}) needed"
                )));
            }
            accs.iter_mut()
                .zip(values)
                .try_for_each(|(a, v)| a.add(v))?;
        }
        tick(progress, ks.len() as u64);
    }
    Ok(accs.into_iter().map(Accumulator::finish).collect())
}

struct Accumulator<'a> {
    options: &'a ContinuousOptions,
    targets: Option<usize>,
    k: f64,
    mean: Vec<f64>,
    m2: Vec<f64>,
    above: Vec<Vec<u32>>,
    sum_above: Vec<Vec<f64>>,
    stored: Vec<f64>,
    out: ContinuousSummary,
}

impl<'a> Accumulator<'a> {
    fn new(n: usize, options: &'a ContinuousOptions) -> Result<Self> {
        options.keep.validate(n)?;
        if options.quantiles.iter().any(|q| !(0.0..=1.0).contains(q)) {
            return Err(SimError::InvalidParameters(
                "quantiles must be in [0, 1]".into(),
            ));
        }
        let nc = options.cutoffs.len();
        let grade_tonnage = options.tonnage.as_ref().map(|t| {
            let mut groups: Vec<Option<u32>> = t
                .categories
                .iter()
                .flatten()
                .flatten()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .map(Some)
                .collect();
            groups.push(None);
            let empty = vec![vec![Vec::with_capacity(n); t.cutoffs.len()]; groups.len()];
            RealizedTonnage {
                cutoffs: t.cutoffs.clone(),
                groups,
                names: t.names.clone(),
                tonnage: empty.clone(),
                metal: empty,
            }
        });
        if let Some(t) = &options.tonnage {
            if t.cutoffs.iter().any(|c| c.is_nan())
                || t.tonnes.iter().any(|w| w.is_nan() || *w < 0.0)
            {
                return Err(SimError::InvalidParameters(
                    "cutoffs must not be NaN and tonnes must be non-negative".into(),
                ));
            }
            if t.categories
                .as_ref()
                .is_some_and(|c| c.len() != t.tonnes.len())
            {
                return Err(SimError::InvalidParameters(
                    "one category per target".into(),
                ));
            }
        }
        Ok(Self {
            options,
            targets: None,
            k: 0.0,
            mean: vec![],
            m2: vec![],
            above: vec![vec![]; nc],
            sum_above: vec![vec![]; nc],
            stored: vec![],
            out: ContinuousSummary {
                n,
                mean: vec![],
                variance: vec![],
                cutoffs: options.cutoffs.clone(),
                probability_above: vec![],
                mean_above: vec![],
                quantiles: options.quantiles.clone(),
                quantile_values: vec![],
                realization_mean: Vec::with_capacity(n),
                realization_above: vec![Vec::with_capacity(n); nc],
                kept: options.keep.kept(n),
                realizations: vec![],
                grade_tonnage,
            },
        })
    }

    fn add(&mut self, values: Vec<f64>) -> Result<()> {
        let m = *self.targets.get_or_insert(values.len());
        if values.len() != m {
            return Err(SimError::InvalidParameters(
                "realizations differ in length".into(),
            ));
        }
        let index = self.out.realization_mean.len();
        let nc = self.options.cutoffs.len();
        if self.k == 0.0 {
            (self.mean, self.m2) = (vec![0.0; m], vec![0.0; m]);
            self.above = vec![vec![0u32; m]; nc];
            self.sum_above = vec![vec![0.0; m]; nc];
        }
        self.k += 1.0;
        for (i, &v) in values.iter().enumerate() {
            let delta = v - self.mean[i];
            self.mean[i] += delta / self.k;
            self.m2[i] += delta * (v - self.mean[i]);
        }
        for (c, &cut) in self.options.cutoffs.iter().enumerate() {
            let mut count = 0usize;
            for (i, &v) in values.iter().enumerate() {
                if v > cut {
                    self.above[c][i] += 1;
                    self.sum_above[c][i] += v;
                    count += 1;
                }
            }
            self.out.realization_above[c].push(count as f64 / m.max(1) as f64);
        }
        if let (Some(t), Some(out)) = (&self.options.tonnage, &mut self.out.grade_tonnage) {
            if t.tonnes.len() != m {
                return Err(SimError::InvalidParameters(format!(
                    "{} tonnes for {m} targets",
                    t.tonnes.len()
                )));
            }
            let nc = t.cutoffs.len();
            let all = out.groups.len() - 1;
            let mut sums = vec![vec![(0.0, 0.0); nc]; out.groups.len()];
            for (i, &v) in values.iter().enumerate() {
                let group = t.categories.as_ref().and_then(|c| c[i]).map(|code| {
                    out.groups[..all]
                        .binary_search(&Some(code))
                        .expect("a known category")
                });
                for (c, _) in t.cutoffs.iter().enumerate().filter(|(_, c)| v >= **c) {
                    for g in [Some(all), group].into_iter().flatten() {
                        sums[g][c].0 += t.tonnes[i];
                        sums[g][c].1 += t.tonnes[i] * v;
                    }
                }
            }
            for (g, row) in sums.iter().enumerate() {
                for (c, &(tonnes, metal)) in row.iter().enumerate() {
                    out.tonnage[g][c].push(tonnes);
                    out.metal[g][c].push(metal);
                }
            }
        }
        self.out
            .realization_mean
            .push(values.iter().sum::<f64>() / m.max(1) as f64);
        if !self.options.quantiles.is_empty() {
            self.stored.extend_from_slice(&values);
        }
        if self.options.keep.keeps(index) {
            self.out.realizations.push(values);
        }
        Ok(())
    }

    fn finish(self) -> ContinuousSummary {
        let Self {
            options,
            targets,
            k,
            mean,
            m2,
            above,
            sum_above,
            stored,
            mut out,
        } = self;
        let (m, n) = (targets.unwrap_or(0), out.n);
        out.variance = m2.iter().map(|s| s / k).collect();
        out.mean = mean;
        out.probability_above = above
            .iter()
            .map(|a| a.iter().map(|&c| c as f64 / k).collect())
            .collect();
        out.mean_above = above
            .iter()
            .zip(&sum_above)
            .map(|(a, s)| {
                a.iter()
                    .zip(s)
                    .map(|(&c, &s)| if c == 0 { f64::NAN } else { s / c as f64 })
                    .collect()
            })
            .collect();
        if !options.quantiles.is_empty() {
            let columns: Vec<Vec<f64>> = (0..m)
                .into_par_iter()
                .map(|i| {
                    let mut col: Vec<f64> = (0..n).map(|r| stored[r * m + i]).collect();
                    col.sort_by(f64::total_cmp);
                    options
                        .quantiles
                        .iter()
                        .map(|&q| quantile_sorted(&col, q))
                        .collect()
                })
                .collect();
            out.quantile_values = (0..options.quantiles.len())
                .map(|q| columns.iter().map(|c| c[q]).collect())
                .collect();
        }
        out
    }
}

/// Per-target and per-realization summary of a categorical ensemble.
#[derive(Debug, Clone)]
pub struct CategoricalSummary {
    pub n: usize,
    /// Fraction of realizations in each category, `[category][target]`.
    pub probabilities: Vec<Vec<f64>>,
    /// Most probable category per target; ties go to the lowest.
    pub most_likely: Vec<usize>,
    /// Shannon entropy of the probabilities divided by `ln k`: 0 when every
    /// realization agrees, 1 when all categories are equally likely.
    pub entropy: Vec<f64>,
    /// Share of targets in each category, `[realization][category]`.
    pub proportions: Vec<Vec<f64>>,
    /// Indices of the kept realizations, ascending.
    pub kept: Vec<usize>,
    /// The kept realizations, one row per entry of `kept`.
    pub realizations: Vec<Vec<usize>>,
}

impl CategoricalSummary {
    /// Least probable category per target among those some realization
    /// took; ties go to the lowest. None where one category takes them all.
    pub fn least_likely(&self) -> Vec<Option<usize>> {
        let k = self.probabilities.len();
        (0..self.most_likely.len())
            .map(|i| {
                let seen: Vec<usize> = (0..k).filter(|&c| self.probabilities[c][i] > 0.0).collect();
                if seen.len() < 2 {
                    return None;
                }
                seen.into_iter()
                    .min_by(|&a, &b| self.probabilities[a][i].total_cmp(&self.probabilities[b][i]))
            })
            .collect()
    }
}

/// Summarizes `n` realizations of categories `0..k`.
pub fn categorical(
    n: usize,
    k: usize,
    keep: &Keep,
    simulate: impl Fn(usize) -> Result<Vec<usize>> + Sync,
    progress: Option<&Progress>,
) -> Result<CategoricalSummary> {
    if k == 0 {
        return Err(SimError::InvalidParameters(
            "need at least one category".into(),
        ));
    }
    keep.validate(n)?;
    let mut counts: Vec<Vec<u32>> = vec![];
    let mut proportions = Vec::with_capacity(n);
    let mut realizations = vec![];
    run(
        n,
        simulate,
        |cats: Vec<usize>| {
            if counts.is_empty() {
                counts = vec![vec![0; cats.len()]; k];
            }
            if cats.len() != counts[0].len() {
                return Err(SimError::InvalidParameters(
                    "realizations differ in length".into(),
                ));
            }
            let mut share = vec![0.0; k];
            for (i, &c) in cats.iter().enumerate() {
                if c >= k {
                    return Err(SimError::InvalidParameters(format!(
                        "category {c} outside 0..{k}"
                    )));
                }
                counts[c][i] += 1;
                share[c] += 1.0;
            }
            let m = cats.len().max(1) as f64;
            let index = proportions.len();
            proportions.push(share.into_iter().map(|s| s / m).collect());
            if keep.keeps(index) {
                realizations.push(cats);
            }
            Ok(())
        },
        progress,
    )?;
    let probabilities: Vec<Vec<f64>> = counts
        .iter()
        .map(|c| c.iter().map(|&c| c as f64 / n as f64).collect())
        .collect();
    let m = counts[0].len();
    let most_likely = (0..m)
        .map(|i| (0..k).rev().max_by_key(|&c| counts[c][i]).unwrap_or(0))
        .collect();
    let scale = if k > 1 { (k as f64).ln() } else { 1.0 };
    let entropy = (0..m)
        .map(|i| {
            -probabilities
                .iter()
                .map(|p| p[i])
                .filter(|&p| p > 0.0)
                .map(|p| p * p.ln())
                .sum::<f64>()
                / scale
        })
        .collect();
    Ok(CategoricalSummary {
        n,
        probabilities,
        most_likely,
        entropy,
        proportions,
        kept: keep.kept(n),
        realizations,
    })
}

/// Averages realizations from simulation nodes to the rows of a coarser block
/// model: each node counts in the block holding it, weighted by its volume;
/// nodes outside every block are ignored.
#[derive(Debug, Clone)]
pub struct BlockSupport {
    block: Vec<Option<usize>>,
    volume: Vec<f64>,
    blocks: usize,
}

impl BlockSupport {
    /// `volumes` of the nodes, equal when `None`. Every block must hold a node.
    pub fn new(
        nodes: &[(f64, f64, f64)],
        volumes: Option<&[f64]>,
        blocks: &BlockModel,
    ) -> Result<Self> {
        let volume = volumes.map_or_else(|| vec![1.0; nodes.len()], <[f64]>::to_vec);
        if volume.len() != nodes.len() {
            return Err(SimError::InvalidParameters("one volume per node".into()));
        }
        if volume.iter().any(|v| !(v.is_finite() && *v > 0.0)) {
            return Err(SimError::InvalidParameters(
                "node volumes must be positive".into(),
            ));
        }
        let block: Vec<Option<usize>> = nodes
            .par_iter()
            .map(|&(x, y, z)| blocks.row_at([x, y, z]))
            .collect();
        let mut held = vec![false; blocks.len()];
        block.iter().flatten().for_each(|&b| held[b] = true);
        if let Some(empty) = held.iter().position(|h| !h) {
            return Err(SimError::InvalidParameters(format!(
                "block row {empty} holds no node"
            )));
        }
        Ok(Self {
            block,
            volume,
            blocks: blocks.len(),
        })
    }

    fn check(&self, n: usize) -> Result<()> {
        if n == self.block.len() {
            Ok(())
        } else {
            Err(SimError::InvalidParameters(
                "one value per node needed".into(),
            ))
        }
    }

    /// Volume-weighted mean of `values` in each block.
    pub fn mean(&self, values: &[f64]) -> Result<Vec<f64>> {
        self.check(values.len())?;
        let mut sum = vec![(0.0, 0.0); self.blocks];
        for ((b, v), w) in self.block.iter().zip(values).zip(&self.volume) {
            if let Some(b) = *b {
                sum[b].0 += w * v;
                sum[b].1 += w;
            }
        }
        Ok(sum.into_iter().map(|(s, w)| s / w).collect())
    }

    /// Category `0..k` filling the most volume of each block; ties go to the
    /// smallest, as in `BlockModel::regularize`.
    pub fn majority(&self, categories: &[usize], k: usize) -> Result<Vec<usize>> {
        self.check(categories.len())?;
        let mut share = vec![0.0; self.blocks * k];
        for ((b, &c), w) in self.block.iter().zip(categories).zip(&self.volume) {
            if c >= k {
                return Err(SimError::InvalidParameters(format!(
                    "category {c} outside 0..{k}"
                )));
            }
            if let Some(b) = *b {
                share[b * k + c] += w;
            }
        }
        Ok(share
            .chunks(k.max(1))
            .map(|s| {
                (0..k)
                    .rev()
                    .max_by(|&a, &b| s[a].total_cmp(&s[b]))
                    .unwrap_or(0)
            })
            .collect())
    }
}

/// Localized grade of every block nested in `panels`, from realizations at
/// block support, `realizations[realization][block]`.
///
/// Each panel pools its blocks' values over the `n` realizations, sorts them,
/// and gives its block ranked `i` by `ranking` (ties by row order) the mean of
/// the `i`-th chunk of `n` sorted values. The blocks average to the pooled
/// mean, and the top `k` hold the top `k n` pooled values. Blocks outside every
/// panel stay `None`; a `None` rank inside a panel is an error.
pub fn localize(
    panels: &BlockModel,
    blocks: &BlockModel,
    ranking: &[Option<f64>],
    realizations: &[Vec<f64>],
) -> Result<Vec<Option<f64>>> {
    let n = realizations.len();
    if n == 0 {
        return Err(SimError::InvalidParameters(
            "need at least one realization".into(),
        ));
    }
    if realizations.iter().any(|r| r.len() != blocks.len()) {
        return Err(SimError::InvalidParameters(
            "every realization needs one value per selective block".into(),
        ));
    }
    if realizations.iter().flatten().any(|v| !v.is_finite()) {
        return Err(SimError::InvalidParameters(
            "realizations must be finite".into(),
        ));
    }
    let transform = |e: transforms::TransformError| SimError::Transform(e.to_string());
    let mut members = vec![vec![]; panels.len()];
    let nest = transforms::localize::nest(panels, blocks).map_err(transform)?;
    for (row, panel) in nest.into_iter().enumerate() {
        if let Some(p) = panel {
            members[p].push(row);
        }
    }
    transforms::localize::localize(panels, blocks, ranking, |p, _| {
        let mut pooled: Vec<f64> = realizations
            .iter()
            .flat_map(|r| members[p].iter().map(|&b| r[b]))
            .collect();
        pooled.sort_by(f64::total_cmp);
        let means = pooled.chunks(n).map(|c| c.iter().sum::<f64>() / n as f64);
        Ok(Some(means.collect()))
    })
    .map_err(transform)
}

/// Empirical quantile of `sorted` (ascending) using linear interpolation.
pub fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake(k: usize) -> Result<Vec<f64>> {
        Ok((0..5).map(|i| ((k * 7 + i * 3) % 11) as f64).collect())
    }

    #[test]
    fn progress_counts_each_realization_once_and_changes_nothing() {
        let options = ContinuousOptions::default();
        let bar = Progress::new(Some(23));
        let with = continuous(23, &options, fake, Some(&bar)).unwrap();
        assert_eq!(bar.snapshot(), (23, Some(23)));
        assert_eq!(
            with.mean,
            continuous(23, &options, fake, None).unwrap().mean
        );
        let bar = Progress::new(Some(23));
        continuous_batched(23, &options, 5, |ks| ks.map(fake).collect(), Some(&bar)).unwrap();
        assert_eq!(bar.snapshot().0, 23);
        let bar = Progress::new(Some(4));
        categorical(4, 2, &Keep::None, |k| Ok(vec![k % 2]), Some(&bar)).unwrap();
        assert_eq!(bar.snapshot().0, 4);
    }

    fn with_threads<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(f)
    }

    #[test]
    fn keep_selects_realizations_in_order() {
        let simulate = |k: usize| Ok(vec![k as f64, 10.0 * k as f64]);
        let options = |keep| ContinuousOptions {
            keep,
            ..Default::default()
        };
        let all = continuous(5, &options(Keep::All), simulate, None).unwrap();
        assert_eq!(all.kept, vec![0, 1, 2, 3, 4]);
        assert_eq!(all.realizations.len(), 5);
        let some = continuous(5, &options(Keep::Indices(vec![3, 1])), simulate, None).unwrap();
        assert_eq!(some.kept, vec![1, 3]);
        assert_eq!(some.realizations, vec![vec![1.0, 10.0], vec![3.0, 30.0]]);
        assert_eq!(some.mean, all.mean);
        let none = continuous(5, &options(Keep::None), simulate, None).unwrap();
        assert!(none.kept.is_empty() && none.realizations.is_empty());
        let cats = categorical(4, 2, &Keep::Indices(vec![2]), |k| Ok(vec![k % 2]), None).unwrap();
        assert_eq!((cats.kept, cats.realizations), (vec![2], vec![vec![0]]));
    }

    #[test]
    fn keep_rejects_bad_indices() {
        let options = |keep| ContinuousOptions {
            keep,
            ..Default::default()
        };
        let simulate = |_| Ok(vec![0.0]);
        assert!(continuous(3, &options(Keep::Indices(vec![3])), simulate, None).is_err());
        assert!(continuous(3, &options(Keep::Indices(vec![1, 1])), simulate, None).is_err());
        assert!(categorical(3, 2, &Keep::Indices(vec![5]), |_| Ok(vec![0]), None).is_err());
    }

    #[test]
    fn realizations_folded_from_batches_match_one_by_one() {
        let options = ContinuousOptions {
            cutoffs: vec![5.0],
            quantiles: vec![0.5],
            keep: Keep::All,
            tonnage: None,
        };
        let one = continuous(7, &options, fake, None).unwrap();
        let batched = continuous_in_batches(
            7,
            &options,
            3,
            |range| Ok(range.map(|k| fake(k).unwrap()).collect::<Vec<_>>()),
            |b: &Vec<Vec<f64>>, i| Ok(b[i].clone()),
            None,
        )
        .unwrap();
        assert_eq!(one.mean, batched.mean);
        assert_eq!(one.quantile_values, batched.quantile_values);
        assert_eq!(one.realizations, batched.realizations);
    }

    #[test]
    fn streamed_statistics_match_the_stored_ensemble() {
        let options = ContinuousOptions {
            cutoffs: vec![4.5],
            quantiles: vec![0.1, 0.5, 0.9],
            keep: Keep::All,
            tonnage: None,
        };
        let s = continuous(23, &options, fake, None).unwrap();
        let reals = s.realizations;
        for i in 0..5 {
            let mut col: Vec<f64> = reals.iter().map(|r| r[i]).collect();
            let mean = col.iter().sum::<f64>() / 23.0;
            let var = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 23.0;
            assert!((s.mean[i] - mean).abs() < 1e-12);
            assert!((s.variance[i] - var).abs() < 1e-9);
            let high: Vec<f64> = col.iter().copied().filter(|&v| v > 4.5).collect();
            assert_eq!(s.probability_above[0][i], high.len() as f64 / 23.0);
            let mean_high = high.iter().sum::<f64>() / high.len() as f64;
            assert!((s.mean_above[0][i] - mean_high).abs() < 1e-12);
            col.sort_by(f64::total_cmp);
            assert_eq!(s.quantile_values[1][i], quantile_sorted(&col, 0.5));
        }
        let first = reals[0].iter().sum::<f64>() / 5.0;
        assert!((s.realization_mean[0] - first).abs() < 1e-12);
    }

    #[test]
    fn summaries_do_not_depend_on_thread_count() {
        let options = ContinuousOptions {
            cutoffs: vec![3.0, 7.0],
            quantiles: vec![0.5],
            keep: Keep::None,
            tonnage: None,
        };
        let one = with_threads(1, || continuous(37, &options, fake, None).unwrap());
        let many = with_threads(6, || continuous(37, &options, fake, None).unwrap());
        assert_eq!(one.mean, many.mean);
        assert_eq!(one.variance, many.variance);
        assert_eq!(one.realization_above, many.realization_above);
        assert!(one.realizations.is_empty());
    }

    #[test]
    fn category_probabilities_sum_to_one_and_entropy_is_bounded() {
        let s = categorical(
            10,
            3,
            &Keep::None,
            |k| Ok(vec![0, k % 3, (k / 4) % 2]),
            None,
        )
        .unwrap();
        for i in 0..3 {
            let total: f64 = s.probabilities.iter().map(|p| p[i]).sum();
            assert!((total - 1.0).abs() < 1e-12);
        }
        assert_eq!(s.most_likely[0], 0);
        assert_eq!(s.entropy[0], 0.0);
        assert!(s.entropy[1] > 0.9 && s.entropy[1] <= 1.0);
        assert_eq!(s.proportions.len(), 10);
    }

    #[test]
    fn derived_statistics() {
        // Target 0 takes 1..=20, target 1 is constant 0.
        let options = ContinuousOptions {
            quantiles: vec![0.05, 0.5, 0.95],
            ..Default::default()
        };
        let s = continuous(20, &options, |k| Ok(vec![k as f64 + 1.0, 0.0]), None).unwrap();
        let cv = s.cv();
        assert!((cv[0] - s.variance[0].sqrt() / 10.5).abs() < 1e-12 && cv[1].is_nan());
        let q = |j: usize| s.quantile_values[j][0];
        let mean = s.relative_error(0.9, Center::Mean).unwrap();
        assert!((mean[0] - (q(2) - q(0)) / 21.0).abs() < 1e-12 && mean[1].is_nan());
        let median = s.relative_error(0.9, Center::Median).unwrap();
        assert!((median[0] - (q(2) - q(0)) / (2.0 * q(1))).abs() < 1e-12);
        let err = s.relative_error(0.8, Center::Mean).unwrap_err().to_string();
        assert!(err.contains("0.1, 0.9"), "{err}");
    }

    #[test]
    fn grade_tonnage_per_realization_and_category() {
        // Realization k scales the grades [1, 2, 3, 4] by k + 1.
        let options = ContinuousOptions {
            tonnage: Some(TonnageOptions {
                cutoffs: vec![0.0, 2.5],
                tonnes: vec![10.0, 20.0, 30.0, 40.0],
                categories: Some(vec![Some(7), Some(3), Some(7), None]),
                names: vec![],
            }),
            ..Default::default()
        };
        let grades = |k: usize| (1..=4).map(|g| (g * (k + 1)) as f64).collect::<Vec<_>>();
        let s = continuous(3, &options, |k| Ok(grades(k)), None).unwrap();
        let t = s.grade_tonnage.as_ref().unwrap();
        assert_eq!(t.groups, vec![Some(3), Some(7), None]);
        // All targets, cutoff 2.5: realization 0 keeps grades 3 and 4.
        assert_eq!(t.tonnage[2][1], vec![70.0, 90.0, 100.0]);
        assert_eq!(t.metal[2][1], vec![250.0, 580.0, 900.0]);
        assert_eq!(t.tonnage[1][0], vec![40.0; 3]);
        let rows = t.quantiles(&[0.5]).unwrap();
        let all = rows
            .iter()
            .find(|r| r.group.is_none() && r.cutoff == 2.5)
            .unwrap();
        assert_eq!((all.tonnage, all.metal), (90.0, t.metal[2][1][1]));
        assert!((all.mean_grade - t.metal[2][1][1] / 90.0).abs() < 1e-12);
        let bad = TonnageOptions {
            tonnes: vec![1.0],
            ..options.tonnage.clone().unwrap()
        };
        let bad = ContinuousOptions {
            tonnage: Some(bad),
            ..Default::default()
        };
        assert!(continuous(1, &bad, |k| Ok(grades(k)), None).is_err());
    }

    #[test]
    fn least_likely_skips_categories_never_drawn() {
        // Target 0: always 0; target 1: 0, 1, 1, 2; target 2: 1 and 2 tied.
        let draws = [[0, 0, 1], [0, 1, 2], [0, 1, 1], [0, 2, 2]];
        let s = categorical(4, 3, &Keep::None, |k| Ok(draws[k].to_vec()), None).unwrap();
        assert_eq!(s.least_likely(), vec![None, Some(0), Some(1)]);
    }

    fn grid(size: f64, count: usize) -> BlockModel {
        let geometry = boitata_core::Geometry {
            origin: [0.0; 3],
            size: [size, size, 1.0],
            count: [count, count, 1],
            rotation: [0.0; 3],
        };
        let rows = arrow_array::RecordBatchOptions::new().with_row_count(Some(count * count));
        let empty = boitata_core::RecordBatch::try_new_with_options(
            std::sync::Arc::new(arrow_schema::Schema::empty()),
            vec![],
            &rows,
        )
        .unwrap();
        BlockModel::regular(geometry, empty).unwrap()
    }

    fn nodes(model: &BlockModel) -> Vec<(f64, f64, f64)> {
        model
            .centroids()
            .into_iter()
            .map(|[x, y, z]| (x, y, z))
            .collect()
    }

    #[test]
    fn block_mean_of_realizations_is_the_average_of_node_means() {
        let fine = nodes(&grid(1.0, 8));
        let volumes: Vec<f64> = (0..64).map(|i| 1.0 + (i % 3) as f64).collect();
        let support = BlockSupport::new(&fine, Some(&volumes), &grid(4.0, 2)).unwrap();
        let field = |k: usize| Ok((0..64).map(|i| ((k * 7 + i * 3) % 11) as f64).collect());
        let options = ContinuousOptions::default();
        let node = continuous(19, &options, field, None).unwrap();
        let block = continuous(19, &options, |k| support.mean(&field(k)?), None).unwrap();
        assert_eq!(support.mean(&node.mean).unwrap().len(), 4);
        for (a, b) in support.mean(&node.mean).unwrap().iter().zip(&block.mean) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn block_variance_is_below_node_variance() {
        use crate::sgs::{SgsParams, sgs};
        let data = vec![(0.5, 0.5, 0.5), (7.5, 7.5, 0.5), (0.5, 7.5, 0.5)];
        let values = vec![1.0, 4.0, 9.0];
        let fine = nodes(&grid(1.0, 8));
        let support = BlockSupport::new(&fine, None, &grid(4.0, 2)).unwrap();
        let vg = variogram::Variogram::single(variogram::model::Model::Spherical, 1.0, 5.0);
        let realization = |k: usize| {
            let params = SgsParams {
                search: vec![estimation::search::Search {
                    min_samples: 1,
                    max_samples: 16,
                    radius: f64::INFINITY,
                    ..Default::default()
                }],
                seed: 7 + k as u64,
            };
            sgs(&data, &values, None, None, &fine, &vg, &params, None).map(|r| r.values)
        };
        let options = ContinuousOptions::default();
        let node = continuous(40, &options, realization, None).unwrap();
        let block = continuous(40, &options, |k| support.mean(&realization(k)?), None).unwrap();
        let within = support.mean(&node.variance).unwrap();
        for (b, n) in block.variance.iter().zip(&within) {
            assert!(b <= &(n + 1e-12), "block {b} above nodes {n}");
        }
        assert!(block.variance.iter().sum::<f64>() < 0.8 * within.iter().sum::<f64>());
    }

    #[test]
    fn blocks_the_size_of_the_nodes_change_nothing() {
        let model = grid(2.5, 5);
        let support = BlockSupport::new(&nodes(&model), None, &model).unwrap();
        let values: Vec<f64> = (0..25).map(|i| (i * i % 7) as f64 - 0.3).collect();
        assert_eq!(support.mean(&values).unwrap(), values);
        let cats: Vec<usize> = (0..25).map(|i| i % 3).collect();
        assert_eq!(support.majority(&cats, 3).unwrap(), cats);
    }

    #[test]
    fn majority_is_deterministic_on_ties() {
        let fine = nodes(&grid(1.0, 2));
        let support = BlockSupport::new(&fine, None, &grid(2.0, 1)).unwrap();
        assert_eq!(support.majority(&[2, 1, 1, 2], 3).unwrap(), [1]);
        assert_eq!(support.majority(&[1, 2, 2, 1], 3).unwrap(), [1]);
        let heavier = BlockSupport::new(&fine, Some(&[1.0, 1.0, 1.0, 4.0]), &grid(2.0, 1)).unwrap();
        assert_eq!(heavier.majority(&[1, 1, 1, 2], 3).unwrap(), [2]);
        let s = categorical(
            5,
            3,
            &Keep::None,
            |_| support.majority(&[2, 0, 0, 2], 3),
            None,
        )
        .unwrap();
        assert_eq!(s.most_likely, [0]);
    }

    #[test]
    fn empty_blocks_and_bad_volumes_are_errors() {
        let fine = nodes(&grid(1.0, 2));
        assert!(BlockSupport::new(&fine, None, &grid(1.0, 3)).is_err());
        assert!(BlockSupport::new(&fine, Some(&[1.0, 0.0, 1.0, 1.0]), &grid(2.0, 1)).is_err());
        let support = BlockSupport::new(&fine, None, &grid(2.0, 1)).unwrap();
        assert!(support.mean(&[1.0]).is_err());
        assert!(support.majority(&[0, 0, 0, 5], 3).is_err());
    }

    #[test]
    fn bad_input_is_an_error() {
        assert!(continuous(0, &ContinuousOptions::default(), fake, None).is_err());
        assert!(categorical(3, 2, &Keep::None, |_| Ok(vec![2]), None).is_err());
    }

    /// Panels of 20 m holding 4 × 4 blocks of 5 m; the blocks overhang the
    /// panels by one row and column.
    type Nested = (BlockModel, BlockModel, Vec<Option<f64>>, Vec<Vec<f64>>);

    fn nested(n: usize) -> Nested {
        let (panels, blocks) = (grid(20.0, 3), grid(5.0, 13));
        let ranking = (0..blocks.len())
            .map(|i| Some(((i * 29) % 17) as f64))
            .collect();
        let reals = (0..n)
            .map(|r| {
                (0..blocks.len())
                    .map(|i| ((i * 37 + r * 11) % 23) as f64 + 0.1 * r as f64)
                    .collect()
            })
            .collect();
        (panels, blocks, ranking, reals)
    }

    fn by_panel(panels: &BlockModel, blocks: &BlockModel) -> Vec<Vec<usize>> {
        let mut members = vec![vec![]; panels.len()];
        let nest = transforms::localize::nest(panels, blocks).unwrap();
        for (row, p) in nest.into_iter().enumerate() {
            if let Some(p) = p {
                members[p].push(row);
            }
        }
        members
    }

    fn ranked(rows: &[usize], ranking: &[Option<f64>]) -> Vec<usize> {
        let mut rows = rows.to_vec();
        rows.sort_by(|&i, &j| ranking[i].unwrap().total_cmp(&ranking[j].unwrap()));
        rows
    }

    #[test]
    fn localized_panels_reproduce_their_pooled_realizations() {
        let (panels, blocks, ranking, reals) = nested(7);
        let out = localize(&panels, &blocks, &ranking, &reals).unwrap();
        let members = by_panel(&panels, &blocks);
        let inside: usize = members.iter().map(Vec::len).sum();
        assert_eq!(inside, 144);
        assert_eq!(out.iter().flatten().count(), inside);
        for rows in &members {
            let mut pooled: Vec<f64> = reals
                .iter()
                .flat_map(|r| rows.iter().map(|&b| r[b]))
                .collect();
            pooled.sort_by(|a, b| b.total_cmp(a));
            let grades: Vec<f64> = ranked(rows, &ranking)
                .iter()
                .map(|&b| out[b].unwrap())
                .collect();
            assert!(grades.windows(2).all(|w| w[0] <= w[1]), "not monotone");
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            assert!((mean(&grades) - mean(&pooled)).abs() < 1e-9);
            let n = rows.len();
            for k in 1..=n {
                let metal = grades[n - k..].iter().sum::<f64>() / n as f64;
                let pooled_metal = pooled[..k * 7].iter().sum::<f64>() / pooled.len() as f64;
                assert!((metal - pooled_metal).abs() < 1e-9, "tonnage {k}/{n}");
            }
        }
    }

    #[test]
    fn one_realization_is_sorted_onto_the_ranks() {
        let (panels, blocks, ranking, reals) = nested(1);
        let own: Vec<Option<f64>> = reals[0].iter().map(|&v| Some(v)).collect();
        let out = localize(&panels, &blocks, &own, &reals).unwrap();
        let members = by_panel(&panels, &blocks);
        for &b in members.iter().flatten() {
            assert_eq!(out[b], Some(reals[0][b]));
        }
        let out = localize(&panels, &blocks, &ranking, &reals).unwrap();
        for rows in &members {
            let mut values: Vec<f64> = rows.iter().map(|&b| reals[0][b]).collect();
            values.sort_by(f64::total_cmp);
            let got: Vec<f64> = ranked(rows, &ranking)
                .iter()
                .map(|&b| out[b].unwrap())
                .collect();
            assert_eq!(got, values);
        }
    }

    #[test]
    fn localization_does_not_depend_on_thread_count() {
        let (panels, blocks, ranking, reals) = nested(9);
        let one = with_threads(1, || localize(&panels, &blocks, &ranking, &reals).unwrap());
        let many = with_threads(5, || localize(&panels, &blocks, &ranking, &reals).unwrap());
        assert_eq!(one, many);
    }

    #[test]
    fn bad_realizations_are_errors() {
        let (panels, blocks, ranking, mut reals) = nested(2);
        assert!(localize(&panels, &blocks, &ranking, &[]).is_err());
        assert!(localize(&panels, &blocks, &ranking[1..], &reals).is_err());
        reals[1][3] = f64::NAN;
        assert!(localize(&panels, &blocks, &ranking, &reals).is_err());
        reals[1].pop();
        assert!(localize(&panels, &blocks, &ranking, &reals).is_err());
    }

    #[test]
    fn batched_summary_does_not_depend_on_the_batch() {
        let options = ContinuousOptions {
            cutoffs: vec![0.2, 0.7],
            quantiles: vec![0.1, 0.5, 0.9],
            keep: Keep::All,
            tonnage: None,
        };
        let whole = continuous(23, &options, fake, None).unwrap();
        for batch in [1, 4, 23, 100] {
            let s =
                continuous_batched(23, &options, batch, |ks| ks.map(fake).collect(), None).unwrap();
            assert_eq!(s.mean, whole.mean);
            assert_eq!(s.variance, whole.variance);
            assert_eq!(s.probability_above, whole.probability_above);
            assert_eq!(s.quantile_values, whole.quantile_values);
            assert_eq!(s.realization_mean, whole.realization_mean);
            assert_eq!(s.realizations, whole.realizations);
        }
        assert!(continuous_batched(3, &options, 4, |_| Ok(vec![]), None).is_err());
    }

    #[test]
    fn quantiles_are_exact_in_f64() {
        let options = ContinuousOptions {
            quantiles: vec![0.25, 0.5, 0.9],
            ..Default::default()
        };
        let value = |k: usize, i: usize| 1.0 + (k * 7 + i) as f64 * 1e-9 + 1.0 / 3.0;
        let s = continuous(
            9,
            &options,
            |k| Ok((0..5).map(|i| value(k, i)).collect()),
            None,
        )
        .unwrap();
        for i in 0..5 {
            let mut col: Vec<f64> = (0..9).map(|k| value(k, i)).collect();
            col.sort_by(f64::total_cmp);
            for (q, &p) in options.quantiles.iter().enumerate() {
                assert_eq!(s.quantile_values[q][i], quantile_sorted(&col, p));
            }
        }
    }
}
