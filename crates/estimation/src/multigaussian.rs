//! Multigaussian kriging: the data take a normal-score transform, the scores
//! are simple-kriged about 0, and at each target the score is Gaussian with
//! the kriged mean and variance. That distribution is back-transformed into
//! data units: probabilities above cutoffs and quantiles through the monotone
//! back-transform; the mean, variance and means above cutoffs by averaging the
//! back-transform over equal-probability cells of the Gaussian, each cell
//! represented by its conditional mean.
//!
//! A block takes the average of the distributions at its discretization
//! points: the distribution of the point values within it.

use std::sync::OnceLock;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use transforms::{NormalScoreTable, normal_score, phi, probit};
use variogram::Variogram;

use crate::Sample;
use crate::batch::{by_pass, k_fold_at, leave_one_out_at};
use crate::error::{EstimError, Result};
use crate::indicator::{Conditional, IndicatorSummary, diagnostics, indicator_targets, summary};
use crate::krige::{Kind, krige};
use crate::neighborhood::neighborhood_stats;
use crate::search::Search;

type Point = (f64, f64, f64);

/// Equal-probability cells of the conditional Gaussian.
const CELLS: usize = 200;

/// Parameters of multigaussian kriging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Multigaussian {
    /// Variogram of the normal scores.
    pub variogram: Variogram,
    /// Bounds of the back-transform, widened to cover the data; the data
    /// minimum and maximum when `None`.
    pub tails: Option<(f64, f64)>,
}

fn invalid(message: &str) -> EstimError {
    EstimError::InvalidParameters(message.into())
}

fn density(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Conditional means of a standard Gaussian within `CELLS` cells of equal
/// probability splitting its lowest `q` of probability.
fn lower_cells(q: f64) -> Vec<f64> {
    let width = q / CELLS as f64;
    let edge = |k: usize| match k {
        0 => 0.0,
        k if k == CELLS && q >= 1.0 => 0.0,
        k => density(probit(width * k as f64)),
    };
    (1..=CELLS)
        .map(|k| (edge(k - 1) - edge(k)) / width)
        .collect()
}

/// `P(Y > y)` for `Y` Gaussian with mean `m` and standard deviation `s`.
fn above(y: f64, (m, s): (f64, f64)) -> f64 {
    if s > 0.0 {
        phi((m - y) / s)
    } else {
        f64::from(u8::from(m > y))
    }
}

/// The average of the Gaussian score distributions `gaussians`, each a
/// kriged mean and standard deviation, back-transformed through `table` and
/// summarized.
pub fn conditional(
    table: &NormalScoreTable,
    gaussians: &[(f64, f64)],
    cutoffs: &[f64],
    quantiles: &[f64],
) -> Conditional {
    static FULL: OnceLock<Vec<f64>> = OnceLock::new();
    let full = FULL.get_or_init(|| lower_cells(1.0));
    let n = gaussians.len() as f64;
    let over = |nodes: &[f64], (m, s): (f64, f64), f: &dyn Fn(f64) -> f64| -> f64 {
        nodes.iter().map(|y| f(table.back(m + s * y))).sum::<f64>() / CELLS as f64
    };
    let mean = gaussians
        .iter()
        .map(|&g| over(full, g, &|z| z))
        .sum::<f64>()
        / n;
    let square = gaussians
        .iter()
        .map(|&g| over(full, g, &|z| z * z))
        .sum::<f64>()
        / n;
    let mut probability_above = vec![];
    let mut mean_above = vec![];
    for &c in cutoffs {
        let yc = table.forward(c);
        let (mut p, mut partial) = (0.0, 0.0);
        for &g in gaussians {
            let q = above(yc, g);
            p += q;
            partial += q * if q > 1e-12 {
                let tail: Vec<f64> = lower_cells(q).iter().map(|y| -y).collect();
                over(&tail, g, &|z| z)
            } else {
                c
            };
        }
        probability_above.push(p / n);
        mean_above.push(if p > 0.0 { partial / p } else { f64::NAN });
    }
    let (lo, hi) = gaussians
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), &(m, s)| {
            (lo.min(m - 9.0 * s), hi.max(m + 9.0 * s))
        });
    let quantile_values = quantiles
        .iter()
        .map(|&u| {
            let (mut lo, mut hi) = (lo - 1.0, hi + 1.0);
            for _ in 0..100 {
                let mid = 0.5 * (lo + hi);
                if gaussians.iter().map(|&g| 1.0 - above(mid, g)).sum::<f64>() / n >= u {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            table.back(hi)
        })
        .collect();
    Conditional {
        cdf: vec![],
        correction: 0.0,
        violations: 0,
        mean,
        variance: (square - mean * mean).max(0.0),
        probability_above,
        mean_above,
        quantile_values,
    }
}

impl Multigaussian {
    pub fn validate(&self) -> Result<()> {
        match self.tails {
            Some((lo, hi)) if !(lo.is_finite() && hi.is_finite() && lo <= hi) => {
                Err(invalid("tails must be finite with lower <= upper"))
            }
            _ => Ok(()),
        }
    }

    /// Normal-score table of the sample values, declustered by `weights`.
    pub fn table(&self, samples: &[Sample], weights: Option<&[f64]>) -> Result<NormalScoreTable> {
        let values: Vec<f64> = samples.iter().map(|s| s.value).collect();
        let table = normal_score(&values, weights)
            .map_err(|e| EstimError::InvalidParameters(e.to_string()))?
            .table;
        Ok(match self.tails {
            Some((lo, hi)) => table.with_tails(lo, hi),
            None => table,
        })
    }

    /// The samples with their normal scores as values; tied values share
    /// their mean score.
    fn scores(table: &NormalScoreTable, samples: &[Sample]) -> Vec<Sample> {
        samples
            .iter()
            .map(|s| Sample {
                value: table.forward(s.value),
                ..s.clone()
            })
            .collect()
    }

    /// Simple-kriged mean and standard deviation of the score at `target`,
    /// or at each point of `block`, offsets from `target`.
    pub fn kriged(
        &self,
        target: &Point,
        scores: &[Sample],
        block: Option<&[Point]>,
    ) -> Result<Vec<(f64, f64)>> {
        let at = |p: &Point| {
            krige(Kind::Simple { mean: 0.0 }, p, scores, &self.variogram)
                .map(|e| (e.value, e.variance.max(0.0).sqrt()))
        };
        match block {
            None => Ok(vec![at(target)?]),
            Some(b) => b
                .iter()
                .map(|o| at(&(target.0 + o.0, target.1 + o.1, target.2 + o.2)))
                .collect(),
        }
    }

    /// Conditional distributions at `targets`, each search pass filling the
    /// targets the previous left unestimated. `block`, discretization points
    /// relative to a target (see [`crate::Discretization::offsets`]), gives
    /// each target the distribution of the points within its block.
    /// Identical for any number of threads.
    #[allow(clippy::too_many_arguments)]
    pub fn predict(
        &self,
        samples: &[Sample],
        weights: Option<&[f64]>,
        targets: &[Point],
        block: Option<&[Point]>,
        searches: &[Search],
        cutoffs: &[f64],
        quantiles: &[f64],
    ) -> Result<IndicatorSummary> {
        self.validate()?;
        if cutoffs.iter().any(|c| !c.is_finite()) {
            return Err(invalid("cutoffs must be finite"));
        }
        if quantiles.iter().any(|q| !(0.0..=1.0).contains(q)) {
            return Err(invalid("quantiles must be in [0, 1]"));
        }
        let table = self.table(samples, weights)?;
        let scores = Self::scores(&table, samples);
        let results = indicator_targets(
            targets,
            None,
            &scores,
            searches,
            None,
            &self.variogram,
            |t, s, _| {
                let near = neighborhood_stats(t, s, s.len(), f64::INFINITY, None);
                let gaussians = self.kriged(t, s, block)?;
                Ok((conditional(&table, &gaussians, cutoffs, quantiles), near))
            },
        )?;
        let diagnostics = diagnostics(&results, searches, |_| 0);
        let conditionals = results.into_iter().map(|r| r.map(|(_, (c, _))| c));
        Ok(IndicatorSummary {
            diagnostics: Some(diagnostics),
            ..summary(&[], targets.len(), conditionals, cutoffs, quantiles)
        })
    }

    /// Re-estimates every sample's distribution through the search passes
    /// without the sample (`folds` None) or without its fold, holes kept
    /// whole (see [`k_fold_at`]); the transform comes from all samples.
    /// Returns the summary at the samples and the PIT, the probability of
    /// each sample's distribution not exceeding its own value.
    pub fn cross_validate(
        &self,
        samples: &[Sample],
        weights: Option<&[f64]>,
        searches: &[Search],
        folds: Option<usize>,
    ) -> Result<(IndicatorSummary, Vec<f64>)> {
        self.validate()?;
        let table = self.table(samples, weights)?;
        let scores = Self::scores(&table, samples);
        let vg = Some(&self.variogram);
        let kriged = |t: &Point, s: &[Sample]| self.kriged(t, s, None);
        let found = by_pass(scores.len(), searches, |search, remaining| match folds {
            None => Ok(leave_one_out_at(remaining, &scores, search, vg, kriged)),
            Some(k) => k_fold_at(k, remaining, &scores, search, vg, kriged),
        })?;
        let pit = found
            .iter()
            .zip(&scores)
            .map(|(r, s)| {
                r.as_ref()
                    .map_or(f64::NAN, |(_, g)| 1.0 - above(s.value, g[0]))
            })
            .collect();
        let results: Vec<Option<Conditional>> = found
            .par_iter()
            .map(|r| r.as_ref().map(|(_, g)| conditional(&table, g, &[], &[])))
            .collect();
        Ok((
            summary(&[], samples.len(), results.into_iter(), &[], &[]),
            pit,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use variogram::Model;

    /// Lognormal values of a smooth field plus seeded noise on a regular grid.
    fn data(side: usize, seed: u64) -> Vec<Sample> {
        let mut state = seed;
        let mut noise = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        (0..side * side)
            .map(|i| {
                let (x, y) = ((i % side) as f64 * 10.0, (i / side) as f64 * 10.0);
                let z = (x / 25.0).sin() + (y / 30.0).cos() + noise();
                Sample::new((x, y, 0.0), z.exp())
            })
            .collect()
    }

    fn model() -> Multigaussian {
        Multigaussian {
            variogram: Variogram::single(Model::Spherical, 1.0, 60.0),
            tails: None,
        }
    }

    fn search() -> Search {
        Search {
            min_samples: 1,
            max_samples: 16,
            ..Search::default()
        }
    }

    fn grid(side: usize, step: f64) -> Vec<Point> {
        (0..side * side)
            .map(|i| ((i % side) as f64 * step, (i / side) as f64 * step, 0.0))
            .collect()
    }

    #[test]
    fn identity_transform_gives_the_simple_kriging_estimate() {
        let scores: Vec<f64> = (0..=4000).map(|i| -20.0 + i as f64 * 0.01).collect();
        let identity = NormalScoreTable {
            values: scores.clone(),
            scores,
            tails: (-20.0, 20.0),
        };
        let samples: Vec<Sample> = data(8, 3)
            .into_iter()
            .map(|s| Sample::new(s.loc, s.value.ln()))
            .collect();
        let m = model();
        for target in [(15.0, 25.0, 0.0), (43.0, 7.0, 0.0), (120.0, 120.0, 0.0)] {
            let sk = krige(Kind::Simple { mean: 0.0 }, &target, &samples, &m.variogram).unwrap();
            let g = m.kriged(&target, &samples, None).unwrap();
            let c = conditional(&identity, &g, &[sk.value], &[0.5]);
            assert!((c.mean - sk.value).abs() < 1e-9, "{} {}", c.mean, sk.value);
            assert!((c.variance / sk.variance - 1.0).abs() < 0.005);
            assert!((c.probability_above[0] - 0.5).abs() < 1e-12);
            assert!((c.quantile_values[0] - sk.value).abs() < 1e-9);
        }
    }

    #[test]
    fn probabilities_fall_with_the_cutoff() {
        let samples = data(10, 7);
        let cutoffs: Vec<f64> = (0..30).map(|i| 0.1 * i as f64).collect();
        let blocks = [(-2.5, -2.5, 0.0), (2.5, -2.5, 0.0), (-2.5, 2.5, 0.0)];
        for block in [None, Some(&blocks[..])] {
            let s = model()
                .predict(
                    &samples,
                    None,
                    &grid(12, 8.0),
                    block,
                    &[search()],
                    &cutoffs,
                    &[],
                )
                .unwrap();
            for t in 0..s.mean.len() {
                for k in 1..cutoffs.len() {
                    assert!(s.probability_above[k][t] <= s.probability_above[k - 1][t]);
                    let m = s.mean_above[k][t];
                    assert!(m.is_nan() || m >= cutoffs[k] - 1e-9);
                }
            }
        }
    }

    #[test]
    fn exact_at_the_data() {
        let samples = data(8, 11);
        let at: Vec<Point> = samples.iter().map(|s| s.loc).collect();
        let s = model()
            .predict(&samples, None, &at, None, &[search()], &[], &[0.1, 0.9])
            .unwrap();
        for (i, sample) in samples.iter().enumerate() {
            assert!((s.mean[i] - sample.value).abs() < 1e-9 * sample.value);
            assert!(s.variance[i] < 1e-12);
            assert!((s.quantile_values[1][i] - sample.value).abs() < 1e-6);
        }
    }

    #[test]
    fn dense_mean_matches_the_declustered_mean() {
        let samples = data(12, 5);
        let mut clustered = samples.clone();
        let mut weights = vec![1.0; samples.len()];
        for s in samples.iter().filter(|s| s.value > 3.0) {
            clustered.push(Sample::new((s.loc.0 + 1.0, s.loc.1 + 1.0, 0.0), s.value));
            weights.push(1e-6);
        }
        let declustered = samples.iter().map(|s| s.value).sum::<f64>() / samples.len() as f64;
        let area: Vec<Point> = grid(120, 1.0)
            .into_iter()
            .map(|(x, y, z)| (x - 4.5, y - 4.5, z))
            .collect();
        let s = model()
            .predict(
                &clustered,
                Some(&weights),
                &area,
                None,
                &[search()],
                &[],
                &[],
            )
            .unwrap();
        let dense = s.mean.iter().sum::<f64>() / s.mean.len() as f64;
        let naive = clustered.iter().map(|s| s.value).sum::<f64>() / clustered.len() as f64;
        assert!(
            (dense / declustered - 1.0).abs() < 0.05,
            "{dense} {declustered}"
        );
        assert!(naive / declustered - 1.0 > 0.2, "{naive} {declustered}");
    }

    #[test]
    fn output_does_not_depend_on_thread_count() {
        let samples = data(10, 13);
        let passes = [
            Search {
                radius: 12.0,
                min_samples: 3,
                ..search()
            },
            search(),
        ];
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let m = model();
                    let s = m
                        .predict(
                            &samples,
                            None,
                            &grid(20, 5.0),
                            None,
                            &passes,
                            &[1.0],
                            &[0.5],
                        )
                        .unwrap();
                    let (cv, pit) = m.cross_validate(&samples, None, &passes, None).unwrap();
                    (s.mean, s.probability_above, s.quantile_values, cv.mean, pit)
                })
        };
        assert_eq!(run(1), run(8));
    }
}
