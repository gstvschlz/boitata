//! Uncertainty summaries accumulated realization by realization, so an ensemble
//! is never held or returned in full unless asked for.
//!
//! Realizations are simulated in parallel batches and folded in realization
//! order, so every summary is identical for any number of threads.

use rayon::prelude::*;

use crate::error::{Result, SimError};

/// Runs realizations `0..n` in parallel batches and hands them to `add` in
/// order; at most one batch (one realization per thread) is held at a time.
fn run<R: Send>(
    n: usize,
    simulate: impl Fn(usize) -> Result<R> + Sync,
    mut add: impl FnMut(R) -> Result<()>,
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
        done.into_iter().try_for_each(&mut add)?;
    }
    Ok(())
}

/// What to accumulate for a continuous variable.
#[derive(Debug, Clone, Default)]
pub struct ContinuousOptions {
    pub cutoffs: Vec<f64>,
    /// Probabilities in `[0, 1]`; exact, but they hold every value as `f32`.
    pub quantiles: Vec<f64>,
    /// Also return every realization.
    pub keep: bool,
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
    pub realizations: Option<Vec<Vec<f64>>>,
}

/// Summarizes `n` realizations of a continuous variable; `simulate(k)` returns
/// realization `k` over the same targets every time.
pub fn continuous(
    n: usize,
    options: &ContinuousOptions,
    simulate: impl Fn(usize) -> Result<Vec<f64>> + Sync,
) -> Result<ContinuousSummary> {
    if options.quantiles.iter().any(|q| !(0.0..=1.0).contains(q)) {
        return Err(SimError::InvalidParameters(
            "quantiles must be in [0, 1]".into(),
        ));
    }
    let nc = options.cutoffs.len();
    let mut targets = None;
    let (mut mean, mut m2) = (vec![], vec![]);
    let (mut above, mut sum_above) = (vec![vec![]; nc], vec![vec![]; nc]);
    let mut stored: Vec<f32> = vec![];
    let mut out = ContinuousSummary {
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
        realizations: options.keep.then(Vec::new),
    };
    let mut k = 0.0;
    run(n, simulate, |values: Vec<f64>| {
        let m = *targets.get_or_insert(values.len());
        if values.len() != m {
            return Err(SimError::InvalidParameters(
                "realizations differ in length".into(),
            ));
        }
        if k == 0.0 {
            (mean, m2) = (vec![0.0; m], vec![0.0; m]);
            above = vec![vec![0u32; m]; nc];
            sum_above = vec![vec![0.0; m]; nc];
        }
        k += 1.0;
        for (i, &v) in values.iter().enumerate() {
            let delta = v - mean[i];
            mean[i] += delta / k;
            m2[i] += delta * (v - mean[i]);
        }
        for (c, &cut) in options.cutoffs.iter().enumerate() {
            let mut count = 0usize;
            for (i, &v) in values.iter().enumerate() {
                if v > cut {
                    above[c][i] += 1;
                    sum_above[c][i] += v;
                    count += 1;
                }
            }
            out.realization_above[c].push(count as f64 / m.max(1) as f64);
        }
        out.realization_mean
            .push(values.iter().sum::<f64>() / m.max(1) as f64);
        if !options.quantiles.is_empty() {
            stored.extend(values.iter().map(|&v| v as f32));
        }
        if let Some(r) = out.realizations.as_mut() {
            r.push(values);
        }
        Ok(())
    })?;
    let m = targets.unwrap_or(0);
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
                let mut col: Vec<f64> = (0..n).map(|r| stored[r * m + i] as f64).collect();
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
    Ok(out)
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
    pub realizations: Option<Vec<Vec<usize>>>,
}

/// Summarizes `n` realizations of categories `0..k`.
pub fn categorical(
    n: usize,
    k: usize,
    keep: bool,
    simulate: impl Fn(usize) -> Result<Vec<usize>> + Sync,
) -> Result<CategoricalSummary> {
    if k == 0 {
        return Err(SimError::InvalidParameters(
            "need at least one category".into(),
        ));
    }
    let mut counts: Vec<Vec<u32>> = vec![];
    let mut proportions = Vec::with_capacity(n);
    let mut realizations = keep.then(Vec::new);
    run(n, simulate, |cats: Vec<usize>| {
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
        proportions.push(share.into_iter().map(|s| s / m).collect());
        if let Some(r) = realizations.as_mut() {
            r.push(cats);
        }
        Ok(())
    })?;
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
        realizations,
    })
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

    fn with_threads<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(f)
    }

    #[test]
    fn streamed_statistics_match_the_stored_ensemble() {
        let options = ContinuousOptions {
            cutoffs: vec![4.5],
            quantiles: vec![0.1, 0.5, 0.9],
            keep: true,
        };
        let s = continuous(23, &options, fake).unwrap();
        let reals = s.realizations.as_ref().unwrap();
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
            keep: false,
        };
        let one = with_threads(1, || continuous(37, &options, fake).unwrap());
        let many = with_threads(6, || continuous(37, &options, fake).unwrap());
        assert_eq!(one.mean, many.mean);
        assert_eq!(one.variance, many.variance);
        assert_eq!(one.realization_above, many.realization_above);
        assert!(one.realizations.is_none());
    }

    #[test]
    fn category_probabilities_sum_to_one_and_entropy_is_bounded() {
        let s = categorical(10, 3, false, |k| Ok(vec![0, k % 3, (k / 4) % 2])).unwrap();
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
    fn bad_input_is_an_error() {
        assert!(continuous(0, &ContinuousOptions::default(), fake).is_err());
        assert!(categorical(3, 2, false, |_| Ok(vec![2])).is_err());
    }
}
