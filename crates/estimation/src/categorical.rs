//! Indicator kriging of categories: the probability of each category kriged
//! from its indicator, corrected to a distribution, then summarized.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use variogram::Variogram;

use crate::Sample;
use crate::batch::{by_pass, k_fold_at, leave_one_out_at};
use crate::error::{EstimError, Result};
use crate::indicator::{IndicatorDiagnostics, diagnostics, indicator_targets, krige_indicators};
use crate::lva::LocalAnisotropy;
use crate::neighborhood::neighborhood_stats;
use crate::search::Search;

type Point = (f64, f64, f64);

/// Parameters of categorical indicator kriging; a sample's value is its
/// category code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalIndicator {
    /// Categories are coded `0..categories`.
    pub categories: usize,
    /// One per category, or one shared by all.
    pub variograms: Vec<Variogram>,
    /// Simple kriging with the declustered global proportions as means.
    pub simple: bool,
}

/// Category probabilities per target, indexed `[category][target]`, NaN
/// where unestimated.
#[derive(Debug, Clone, Default)]
pub struct CategoricalIndicatorSummary {
    pub probabilities: Vec<Vec<f64>>,
    /// Sum over the categories of |corrected − kriged| probability.
    pub correction: Vec<f64>,
    /// Declustered proportion of each category among the samples.
    pub proportions: Vec<f64>,
    /// `n_order_violations` counts the categories kriged outside [0, 1].
    pub diagnostics: Option<IndicatorDiagnostics>,
}

fn invalid(message: &str) -> EstimError {
    EstimError::InvalidParameters(message.into())
}

/// Clips to [0, 1] and rescales to sum 1, or `fallback` where every
/// probability clips to 0; also returns Σ|corrected − raw|.
pub fn correct_probabilities(raw: &[f64], fallback: &[f64]) -> (Vec<f64>, f64) {
    let clipped: Vec<f64> = raw.iter().map(|p| p.clamp(0.0, 1.0)).collect();
    let total: f64 = clipped.iter().sum();
    let corrected = if total > 0.0 {
        clipped.iter().map(|p| p / total).collect()
    } else {
        fallback.to_vec()
    };
    let correction = corrected.iter().zip(raw).map(|(c, r)| (c - r).abs()).sum();
    (corrected, correction)
}

struct Corrected {
    probabilities: Vec<f64>,
    correction: f64,
    violations: usize,
}

impl CategoricalIndicator {
    pub fn validate(&self) -> Result<()> {
        if self.categories == 0 {
            return Err(invalid("need at least one category"));
        }
        if self.variograms.len() != 1 && self.variograms.len() != self.categories {
            return Err(invalid("need one variogram, or one per category"));
        }
        Ok(())
    }

    /// Variogram whose anisotropy orients the search: the middle category's.
    pub fn search_variogram(&self) -> &Variogram {
        &self.variograms[self.variograms.len() / 2]
    }

    /// Declustered proportion of each category; checks the codes.
    pub fn proportions(&self, samples: &[Sample], weights: Option<&[f64]>) -> Result<Vec<f64>> {
        self.validate()?;
        if samples.is_empty() {
            return Err(EstimError::InsufficientData("no samples".into()));
        }
        if let Some(w) = weights {
            if w.len() != samples.len() {
                return Err(invalid("need one weight per sample"));
            }
            if w.iter().any(|w| !w.is_finite() || *w < 0.0) || w.iter().sum::<f64>() <= 0.0 {
                return Err(invalid("weights must be finite, >= 0 and not all 0"));
            }
        }
        let mut shares = vec![0.0; self.categories];
        for (i, s) in samples.iter().enumerate() {
            let c = s.value;
            if !(c >= 0.0 && c.fract() == 0.0 && (c as usize) < self.categories) {
                return Err(invalid("category codes must be integers in 0..categories"));
            }
            shares[c as usize] += weights.map_or(1.0, |w| w[i]);
        }
        let total: f64 = shares.iter().sum();
        Ok(shares.into_iter().map(|s| s / total).collect())
    }

    /// Kriged probability of each category at `target`, before correction.
    pub fn kriged(
        &self,
        target: &Point,
        samples: &[Sample],
        proportions: &[f64],
        oriented: Option<&Variogram>,
    ) -> Result<Vec<f64>> {
        krige_indicators(
            &self.variograms,
            self.simple,
            target,
            samples,
            oriented,
            None,
            proportions,
            |c, s| s.value == c as f64,
        )
    }

    fn corrected(&self, raw: &[f64], proportions: &[f64]) -> Corrected {
        let (probabilities, correction) = correct_probabilities(raw, proportions);
        let violations = raw.iter().filter(|p| !(0.0..=1.0).contains(*p)).count();
        Corrected {
            probabilities,
            correction,
            violations,
        }
    }

    /// Category probabilities at `targets`, each search pass filling the
    /// targets the previous left unestimated. `domains`, one per target,
    /// restricts each to the samples of its domain and those the search's
    /// soft boundaries admit; a target without a domain stays unestimated.
    /// `local` (one entry per target) orients every variogram and the
    /// search. Identical for any number of threads.
    #[allow(clippy::too_many_arguments)]
    pub fn predict(
        &self,
        samples: &[Sample],
        weights: Option<&[f64]>,
        targets: &[Point],
        domains: Option<&[Option<u32>]>,
        searches: &[Search],
        local: Option<&LocalAnisotropy>,
    ) -> Result<CategoricalIndicatorSummary> {
        self.predict_with_progress(samples, weights, targets, domains, searches, local, None)
    }

    /// As [`Self::predict`], ticking `progress` for each target estimated.
    #[allow(clippy::too_many_arguments)]
    pub fn predict_with_progress(
        &self,
        samples: &[Sample],
        weights: Option<&[f64]>,
        targets: &[Point],
        domains: Option<&[Option<u32>]>,
        searches: &[Search],
        local: Option<&LocalAnisotropy>,
        progress: Option<&ceres_core::Progress>,
    ) -> Result<CategoricalIndicatorSummary> {
        crate::search::unclamped(searches, "categorical kriging")?;
        let proportions = self.proportions(samples, weights)?;
        if domains.is_some_and(|d| d.len() != targets.len()) {
            return Err(invalid("need one domain per target"));
        }
        let results = indicator_targets(
            targets,
            domains,
            samples,
            searches,
            local,
            self.search_variogram(),
            |t, s, o| {
                let raw = self.kriged(t, s, &proportions, o)?;
                let near = neighborhood_stats(t, s, s.len(), f64::INFINITY, None);
                Ok((self.corrected(&raw, &proportions), near))
            },
            progress,
        )?;
        let diagnostics = diagnostics(&results, searches, |c: &Corrected| c.violations);
        let corrected = results.into_iter().map(|r| r.map(|(_, (c, _))| c));
        Ok(CategoricalIndicatorSummary {
            diagnostics: Some(diagnostics),
            ..self.summary(targets.len(), proportions, corrected)
        })
    }

    /// Re-estimates every sample's probabilities through the search passes
    /// without the sample (`folds` None) or without its fold, holes kept
    /// whole (see [`k_fold_at`]); the proportions come from all samples.
    /// Identical for any number of threads.
    pub fn cross_validate(
        &self,
        samples: &[Sample],
        weights: Option<&[f64]>,
        searches: &[Search],
        folds: Option<usize>,
    ) -> Result<CategoricalIndicatorSummary> {
        crate::search::unclamped(searches, "categorical kriging")?;
        let proportions = self.proportions(samples, weights)?;
        let vg = Some(self.search_variogram());
        let kriged = |t: &Point, s: &[Sample]| self.kriged(t, s, &proportions, None);
        let raw = by_pass(samples.len(), searches, |search, remaining| match folds {
            None => Ok(leave_one_out_at(remaining, samples, search, vg, kriged)),
            Some(k) => k_fold_at(k, remaining, samples, search, vg, kriged),
        })?;
        let corrected: Vec<Option<Corrected>> = raw
            .par_iter()
            .map(|r| r.as_ref().map(|(_, raw)| self.corrected(raw, &proportions)))
            .collect();
        Ok(self.summary(samples.len(), proportions, corrected.into_iter()))
    }

    fn summary(
        &self,
        n: usize,
        proportions: Vec<f64>,
        results: impl Iterator<Item = Option<Corrected>>,
    ) -> CategoricalIndicatorSummary {
        let mut out = CategoricalIndicatorSummary {
            probabilities: vec![Vec::with_capacity(n); self.categories],
            proportions,
            ..Default::default()
        };
        for r in results {
            out.correction
                .push(r.as_ref().map_or(f64::NAN, |r| r.correction));
            for (c, row) in out.probabilities.iter_mut().enumerate() {
                row.push(r.as_ref().map_or(f64::NAN, |r| r.probabilities[c]));
            }
        }
        out
    }
}

impl CategoricalIndicatorSummary {
    fn targets(&self) -> usize {
        self.correction.len()
    }

    /// Most probable category code per target, ties to the lowest; NaN
    /// where unestimated.
    pub fn most_likely(&self) -> Vec<f64> {
        (0..self.targets())
            .map(|i| {
                let mut best = (f64::NAN, f64::NEG_INFINITY);
                for (c, p) in self.probabilities.iter().enumerate() {
                    if p[i] > best.1 {
                        best = (c as f64, p[i]);
                    }
                }
                best.0
            })
            .collect()
    }

    /// Shannon entropy of the probabilities over `ln k`: 0 where one
    /// category is certain, 1 where all are equally likely; NaN where
    /// unestimated.
    pub fn entropy(&self) -> Vec<f64> {
        let k = self.probabilities.len();
        let scale = if k > 1 { (k as f64).ln() } else { 1.0 };
        (0..self.targets())
            .map(|i| {
                let p = || self.probabilities.iter().map(|p| p[i]);
                if p().any(f64::is_nan) {
                    return f64::NAN;
                }
                -p().filter(|&p| p > 0.0).map(|p| p * p.ln()).sum::<f64>() / scale
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use variogram::Model;

    fn uniform(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Three categories in bands across x, with noise.
    fn data(n: usize, seed: u64) -> Vec<Sample> {
        let mut next = uniform(seed);
        (0..n)
            .map(|_| {
                let (x, y) = (next() * 100.0, next() * 100.0);
                let c = ((x + 25.0 * (next() - 0.5)) / 34.0).floor().clamp(0.0, 2.0);
                Sample::new((x, y, 0.0), c)
            })
            .collect()
    }

    fn search(radius: f64) -> Search {
        Search {
            min_samples: 1,
            max_samples: 12,
            radius,
            max_per_hole: None,
            octant: false,
            anisotropy: None,
            high_grade: None,
            soft: None,
        }
    }

    fn grid() -> Vec<Point> {
        (0..400)
            .map(|i| {
                (
                    (i % 20) as f64 * 5.0 + 2.5,
                    (i / 20) as f64 * 5.0 + 2.5,
                    0.0,
                )
            })
            .collect()
    }

    fn model(simple: bool, ranges: &[f64]) -> CategoricalIndicator {
        CategoricalIndicator {
            categories: 3,
            variograms: ranges
                .iter()
                .map(|&r| Variogram::single(Model::Gaussian, 0.2, r))
                .collect(),
            simple,
        }
    }

    #[test]
    fn corrected_probabilities_are_a_distribution() {
        let samples = data(150, 7);
        for simple in [false, true] {
            let s = model(simple, &[10.0, 60.0, 20.0])
                .predict(&samples, None, &grid(), None, &[search(40.0)], None)
                .unwrap();
            let d = s.diagnostics.as_ref().unwrap();
            for i in 0..grid().len() {
                let p: Vec<f64> = s.probabilities.iter().map(|p| p[i]).collect();
                assert!(p.iter().all(|p| (0.0..=1.0).contains(p)), "{p:?}");
                assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12, "{p:?}");
                assert!(d.n_order_violations[i] == 0.0 || s.correction[i] > 0.0);
            }
            assert!(d.n_order_violations.iter().any(|&v| v > 0.0));
            let (h, m) = (s.entropy(), s.most_likely());
            assert!(h.iter().all(|h| (0.0..=1.0 + 1e-12).contains(h)));
            assert!(m.iter().all(|m| [0.0, 1.0, 2.0].contains(m)));
        }
    }

    #[test]
    fn exact_at_the_data() {
        let samples = data(80, 3);
        let at: Vec<Point> = samples.iter().map(|s| s.loc).collect();
        for simple in [false, true] {
            let s = model(simple, &[30.0])
                .predict(&samples, None, &at, None, &[search(30.0)], None)
                .unwrap();
            for (i, sample) in samples.iter().enumerate() {
                let c = sample.value as usize;
                assert!((s.probabilities[c][i] - 1.0).abs() < 1e-9);
                assert_eq!(s.most_likely()[i], sample.value);
                assert!(s.entropy()[i] < 1e-9);
            }
        }
    }

    #[test]
    fn one_category_is_certain() {
        let samples: Vec<Sample> = data(60, 5)
            .into_iter()
            .map(|s| Sample { value: 0.0, ..s })
            .collect();
        for simple in [false, true] {
            let m = CategoricalIndicator {
                categories: 1,
                ..model(simple, &[25.0])
            };
            let s = m
                .predict(&samples, None, &grid(), None, &[search(200.0)], None)
                .unwrap();
            assert!(s.probabilities[0].iter().all(|&p| p == 1.0));
            assert!(s.correction.iter().all(|&c| c < 1e-12));
        }
    }

    #[test]
    fn global_proportions_follow_the_declustered_data() {
        let mut samples = data(200, 11);
        let mut weights = vec![1.0; samples.len()];
        let mut next = uniform(29);
        for _ in 0..200 {
            let loc = (90.0 + next() * 10.0, next() * 100.0, 0.0);
            samples.push(Sample::new(loc, 2.0));
            weights.push(0.0);
        }
        let m = model(true, &[25.0]);
        let proportions = m.proportions(&samples, Some(&weights)).unwrap();
        let s = m
            .predict(
                &samples,
                Some(&weights),
                &grid(),
                None,
                &[search(30.0)],
                None,
            )
            .unwrap();
        for (p, f) in s.probabilities.iter().zip(&proportions) {
            let mean = p.iter().sum::<f64>() / p.len() as f64;
            assert!((mean - f).abs() < 0.05, "{mean} vs {f}");
        }
        let far = m
            .predict(
                &samples,
                Some(&weights),
                &[(1e4, 1e4, 0.0)],
                None,
                &[search(1e6)],
                None,
            )
            .unwrap();
        for (p, f) in far.probabilities.iter().zip(&proportions) {
            assert!((p[0] - f).abs() < 1e-9, "{} vs {f}", p[0]);
        }
    }

    #[test]
    fn domains_split_the_samples() {
        let samples: Vec<Sample> = data(120, 13)
            .into_iter()
            .map(|s| Sample {
                domain: Some(u32::from(s.loc.0 >= 50.0)),
                ..s
            })
            .collect();
        let targets = grid();
        let domains: Vec<Option<u32>> = targets
            .iter()
            .map(|t| (t.1 < 90.0).then_some(u32::from(t.0 >= 50.0)))
            .collect();
        let s = model(false, &[30.0])
            .predict(
                &samples,
                None,
                &targets,
                Some(&domains),
                &[search(200.0)],
                None,
            )
            .unwrap();
        for (i, d) in domains.iter().enumerate() {
            let p = s.probabilities[2][i];
            match d {
                None => assert!(p.is_nan()),
                Some(0) => assert!(p < 0.5, "{p}"),
                Some(_) => assert!(s.probabilities[0][i] < 0.5),
            }
        }
    }

    #[test]
    fn cross_validation_leaves_each_sample_out() {
        let samples = data(100, 17);
        let m = model(false, &[30.0]);
        let s = m
            .cross_validate(&samples, None, &[search(40.0)], None)
            .unwrap();
        let folds = m
            .cross_validate(&samples, None, &[search(40.0)], Some(samples.len()))
            .unwrap();
        assert_eq!(s.probabilities, folds.probabilities);
        let hits = s
            .most_likely()
            .iter()
            .zip(&samples)
            .filter(|(m, s)| **m == s.value)
            .count();
        assert!(hits > 70 && hits < 100, "{hits}");
    }

    #[test]
    fn output_does_not_depend_on_thread_count() {
        let samples = data(200, 5);
        let m = model(true, &[15.0, 40.0, 25.0]);
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    m.predict(
                        &samples,
                        None,
                        &grid(),
                        None,
                        &[search(10.0), search(80.0)],
                        None,
                    )
                    .unwrap()
                })
        };
        let (one, many) = (run(1), run(8));
        assert_eq!(one.probabilities, many.probabilities);
        assert_eq!(one.correction, many.correction);
    }

    #[test]
    fn correction_clips_and_rescales() {
        let (p, c) = correct_probabilities(&[-0.2, 0.6, 0.8], &[0.3, 0.3, 0.4]);
        let expected = [0.0, 0.6 / 1.4, 0.8 / 1.4];
        assert!(p.iter().zip(expected).all(|(a, b)| (a - b).abs() < 1e-12));
        assert!((c - (0.2 + (0.6 - 0.6 / 1.4) + (0.8 - 0.8 / 1.4))).abs() < 1e-12);
        assert_eq!(
            correct_probabilities(&[-0.1, 0.0], &[0.5, 0.5]).0,
            vec![0.5, 0.5]
        );
    }
}
