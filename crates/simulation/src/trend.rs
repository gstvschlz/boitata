//! Simulation with a trend.
//!
//! The stepwise conditional transform of `[trend, value]` normal-scores the
//! values within equal-probability classes of the trend, which leaves scores
//! independent of it. The scores are simulated like any Gaussian variable, by
//! [`crate::sgs`] or turning bands, and each node is back-transformed with the
//! table of its own trend class.

use transforms::StepwiseConditional;

use crate::error::{Result, SimError};

#[derive(Debug, Clone)]
pub struct TrendConditioning {
    transform: StepwiseConditional,
    scores: Vec<f64>,
}

impl TrendConditioning {
    /// `trend` at the data; `weights` decluster every normal score.
    pub fn fit(
        values: &[f64],
        trend: &[f64],
        weights: Option<&[f64]>,
        classes: usize,
    ) -> Result<Self> {
        if values.len() != trend.len() {
            return Err(SimError::InvalidParameters(
                "one trend value per datum".into(),
            ));
        }
        let rows: Vec<Vec<f64>> = trend
            .iter()
            .zip(values)
            .map(|(&t, &v)| vec![t, v])
            .collect();
        let transform = StepwiseConditional::fit(&rows, weights, classes)
            .map_err(|e| SimError::Transform(e.to_string()))?;
        let scores = transform.forward(&rows).into_iter().map(|r| r[1]).collect();
        Ok(Self { transform, scores })
    }

    /// Scores of the data, independent of the trend: the values to simulate.
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// Values at nodes of trend `trend` from their simulated `scores`.
    pub fn back(&self, trend: &[f64], scores: &[f64]) -> Result<Vec<f64>> {
        if trend.len() != scores.len() {
            return Err(SimError::InvalidParameters(
                "one trend value per node".into(),
            ));
        }
        Ok(trend
            .iter()
            .zip(scores)
            .map(|(&t, &y)| {
                let g = self.transform.forward_one(&[], t);
                self.transform.back_one(&[g], y)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::post::{ContinuousOptions, ContinuousSummary, continuous};
    use crate::sgs::{SgsParams, sgs};
    use estimation::Search;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rand_distr::StandardNormal;
    use variogram::Variogram;
    use variogram::model::Model;

    type Point = (f64, f64, f64);

    fn trend_at(p: &Point) -> f64 {
        p.0 / 100.0 + 0.5 * p.1 / 100.0
    }

    /// Lognormal values rising with a planar trend, denser in one corner.
    fn samples() -> (Vec<Point>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng = StdRng::seed_from_u64(21);
        let locs: Vec<Point> = (0..600)
            .map(|i| {
                let side = if i < 450 { 100.0 } else { 25.0 };
                (rng.r#gen::<f64>() * side, rng.r#gen::<f64>() * side, 0.0)
            })
            .collect();
        let trend: Vec<f64> = locs.iter().map(trend_at).collect();
        let values = trend
            .iter()
            .map(|t| {
                let g: f64 = rng.sample(StandardNormal);
                (1.5 * t + 0.5 * g).exp()
            })
            .collect();
        let weights = transforms::cell_weights(&locs, &trend, 20.0, (0.0, 0.0, 0.0))
            .unwrap()
            .weights;
        (locs, values, trend, weights)
    }

    fn params(seed: u64) -> SgsParams {
        SgsParams {
            search: Search {
                max_samples: 12,
                radius: 30.0,
                ..Default::default()
            },
            seed,
        }
    }

    fn vg() -> Variogram {
        Variogram::single(Model::Spherical, 1.0, 10.0)
    }

    fn simulate(
        locs: &[Point],
        conditioning: &TrendConditioning,
        weights: &[f64],
        nodes: &[Point],
        trend: &[f64],
        n: usize,
    ) -> ContinuousSummary {
        let options = ContinuousOptions {
            keep: true,
            ..Default::default()
        };
        continuous(n, &options, |k| {
            let r = sgs(
                locs,
                conditioning.scores(),
                Some(weights),
                None,
                nodes,
                &vg(),
                &params(k as u64),
                None,
            )?;
            conditioning.back(trend, &r.values)
        })
        .unwrap()
    }

    fn grid() -> Vec<Point> {
        (0..30 * 30)
            .map(|i| {
                (
                    (i % 30) as f64 * 3.3 + 1.0,
                    (i / 30) as f64 * 3.3 + 1.0,
                    0.0,
                )
            })
            .collect()
    }

    fn corr(a: &[f64], b: &[f64], w: &[f64]) -> f64 {
        let t: f64 = w.iter().sum();
        let ma = a.iter().zip(w).map(|(x, w)| x * w).sum::<f64>() / t;
        let mb = b.iter().zip(w).map(|(x, w)| x * w).sum::<f64>() / t;
        let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
        for ((x, y), w) in a.iter().zip(b).zip(w) {
            c += w * (x - ma) * (y - mb);
            va += w * (x - ma).powi(2);
            vb += w * (y - mb).powi(2);
        }
        c / (va * vb).sqrt()
    }

    fn weighted_quantile(values: &[f64], weights: &[f64], q: f64) -> f64 {
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        let total: f64 = weights.iter().sum();
        let mut cum = 0.0;
        for &i in &order {
            cum += weights[i] / total;
            if cum >= q {
                return values[i];
            }
        }
        values[order[order.len() - 1]]
    }

    #[test]
    fn a_constant_trend_is_plain_sgs() {
        let (locs, values, trend, weights) = samples();
        let flat = vec![1.0; trend.len()];
        let conditioning = TrendConditioning::fit(&values, &flat, Some(&weights), 10).unwrap();
        let nodes = grid();
        for seed in 0..3 {
            let r = sgs(
                &locs,
                conditioning.scores(),
                Some(&weights),
                None,
                &nodes,
                &vg(),
                &params(seed),
                None,
            )
            .unwrap();
            let with = conditioning
                .back(&vec![1.0; nodes.len()], &r.values)
                .unwrap();
            let plain = sgs(
                &locs,
                &values,
                Some(&weights),
                None,
                &nodes,
                &vg(),
                &params(seed),
                None,
            )
            .unwrap()
            .values;
            for (a, b) in with.iter().zip(&plain) {
                assert!((a / b - 1.0).abs() < 1e-6, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn far_from_data_each_trend_class_reproduces_its_histogram() {
        let (locs, values, trend, weights) = samples();
        let classes = 5;
        let conditioning =
            TrendConditioning::fit(&values, &trend, Some(&weights), classes).unwrap();
        let mut order: Vec<usize> = (0..trend.len()).collect();
        order.sort_by(|&a, &b| trend[a].total_cmp(&trend[b]));
        let total: f64 = weights.iter().sum();
        let mut cum = 0.0;
        let mut members = vec![vec![]; classes];
        for &i in &order {
            let p = cum + weights[i] / total / 2.0;
            cum += weights[i] / total;
            members[((p * classes as f64) as usize).min(classes - 1)].push(i);
        }
        let per_class = 40;
        let nodes: Vec<Point> = (0..classes * per_class)
            .map(|i| (1e5 + i as f64 * 50.0, 1e5, 0.0))
            .collect();
        let node_trend: Vec<f64> = (0..nodes.len())
            .map(|i| {
                let m = &members[i / per_class];
                trend[m[m.len() / 2]]
            })
            .collect();
        let s = simulate(&locs, &conditioning, &weights, &nodes, &node_trend, 50);
        let reals = s.realizations.unwrap();
        for (c, m) in members.iter().enumerate() {
            let data: Vec<f64> = m.iter().map(|&i| values[i]).collect();
            let w: Vec<f64> = m.iter().map(|&i| weights[i]).collect();
            let want = data.iter().zip(&w).map(|(v, w)| v * w).sum::<f64>() / w.iter().sum::<f64>();
            let range = c * per_class..(c + 1) * per_class;
            let mean = s.mean[range.clone()].iter().sum::<f64>() / per_class as f64;
            assert!(
                (mean / want - 1.0).abs() < 0.1,
                "class {c}: {mean} vs {want}"
            );
            let pooled: Vec<f64> = reals
                .iter()
                .flat_map(|r| r[range.clone()].to_vec())
                .collect();
            let ones = vec![1.0; pooled.len()];
            for q in [0.1, 0.5, 0.9] {
                let (want, got) = (
                    weighted_quantile(&data, &w, q),
                    weighted_quantile(&pooled, &ones, q),
                );
                assert!(
                    (got / want).ln().abs() < 0.15,
                    "class {c} q{q}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn realizations_keep_the_correlation_with_the_trend() {
        let (locs, values, trend, weights) = samples();
        let conditioning = TrendConditioning::fit(&values, &trend, Some(&weights), 10).unwrap();
        let nodes = grid();
        let node_trend: Vec<f64> = nodes.iter().map(trend_at).collect();
        let s = simulate(&locs, &conditioning, &weights, &nodes, &node_trend, 8);
        let logs: Vec<f64> = values.iter().map(|v| v.ln()).collect();
        let want = corr(&logs, &trend, &weights);
        let ones = vec![1.0; nodes.len()];
        for r in s.realizations.unwrap() {
            let logs: Vec<f64> = r.iter().map(|v| v.ln()).collect();
            let got = corr(&logs, &node_trend, &ones);
            assert!((got - want).abs() < 0.1, "correlation {got} vs {want}");
        }
    }

    #[test]
    fn honours_data_and_ignores_thread_count() {
        let (locs, values, trend, weights) = samples();
        let conditioning = TrendConditioning::fit(&values, &trend, Some(&weights), 10).unwrap();
        let nodes = [&locs[..30], &grid()[..100]].concat();
        let node_trend: Vec<f64> = nodes.iter().map(trend_at).collect();
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| simulate(&locs, &conditioning, &weights, &nodes, &node_trend, 6))
        };
        let (one, four) = (run(1), run(4));
        let reals = one.realizations.unwrap();
        assert_eq!(reals, four.realizations.unwrap());
        for r in &reals {
            for (got, want) in r.iter().zip(&values[..30]) {
                assert!((got / want - 1.0).abs() < 1e-9, "{got} vs {want}");
            }
        }
        assert!(TrendConditioning::fit(&values, &trend[1..], None, 10).is_err());
        let c = TrendConditioning::fit(&values, &trend, None, 10).unwrap();
        assert!(c.back(&trend[1..], &values).is_err());
    }
}
