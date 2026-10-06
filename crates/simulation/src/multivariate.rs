//! Multivariate simulation through independent factors: the variables are
//! decorrelated, each factor is simulated on its own, and every realization is
//! back-transformed at the nodes before any averaging to blocks.

use transforms::{Maf, Pca, Ppmt, StepwiseConditional};

use boitata_core::rng::realization_seed;

use crate::error::Result;
use crate::post::{
    BlockSupport, ContinuousOptions, ContinuousSummary, continuous_many, continuous_many_batched,
};

/// A fitted transform from correlated variables to independent factors.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decorrelation {
    Pca(Pca),
    Maf(Maf),
    Stepwise(StepwiseConditional),
    Ppmt(Ppmt),
}

impl Decorrelation {
    /// Rows of variables to rows of factors.
    pub fn forward(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        match self {
            Self::Pca(t) => t.forward(data),
            Self::Maf(t) => t.forward(data),
            Self::Stepwise(t) => t.forward(data),
            Self::Ppmt(t) => t.forward(data),
        }
    }

    /// Rows of factors to rows of variables.
    pub fn back(&self, factors: &[Vec<f64>]) -> Vec<Vec<f64>> {
        match self {
            Self::Pca(t) => t.back(factors),
            Self::Maf(t) => t.back(factors),
            Self::Stepwise(t) => t.back(factors),
            Self::Ppmt(t) => t.back(factors),
        }
    }
}

/// Seed of factor `j` in realization `k`, so that no two factors share a
/// random stream.
pub fn factor_seed(seed: u64, k: usize, j: usize) -> u64 {
    realization_seed(realization_seed(seed, k as u64), j as u64)
}

/// Summaries of `n` realizations of every variable. `simulate(k, j, seed)`
/// returns factor `j` of realization `k` at the nodes, simulated with `seed`;
/// the factors of a realization are back-transformed together, then averaged
/// to `support`.
#[allow(clippy::too_many_arguments)]
pub fn multivariate(
    n: usize,
    seed: u64,
    factors: usize,
    transform: &Decorrelation,
    options: &ContinuousOptions,
    support: Option<&BlockSupport>,
    simulate: impl Fn(usize, usize, u64) -> Result<Vec<f64>> + Sync,
) -> Result<Vec<ContinuousSummary>> {
    continuous_many(n, factors, options, |k| {
        let columns = (0..factors)
            .map(|j| simulate(k, j, factor_seed(seed, k, j)))
            .collect::<Result<Vec<_>>>()?;
        let nodes = columns.first().map_or(0, Vec::len);
        let rows: Vec<Vec<f64>> = (0..nodes)
            .map(|i| columns.iter().map(|c| c[i]).collect())
            .collect();
        let back = transform.back(&rows);
        (0..factors)
            .map(|v| {
                let values: Vec<f64> = back.iter().map(|r| r[v]).collect();
                match support {
                    Some(s) => s.mean(&values),
                    None => Ok(values),
                }
            })
            .collect()
    })
}

/// As [`multivariate`], `batch` realizations at a time: `simulate(j, ks)`
/// returns realizations `ks` of factor `j` at the nodes.
#[allow(clippy::too_many_arguments)]
pub fn multivariate_batched(
    n: usize,
    factors: usize,
    transform: &Decorrelation,
    options: &ContinuousOptions,
    support: Option<&BlockSupport>,
    batch: usize,
    mut simulate: impl FnMut(usize, std::ops::Range<usize>) -> Result<Vec<Vec<f64>>>,
) -> Result<Vec<ContinuousSummary>> {
    continuous_many_batched(n, factors, options, batch, |ks| {
        let columns = (0..factors)
            .map(|j| simulate(j, ks.clone()))
            .collect::<Result<Vec<_>>>()?;
        (0..ks.len())
            .map(|kk| {
                let nodes = columns.first().map_or(0, |c| c[kk].len());
                let rows: Vec<Vec<f64>> = (0..nodes)
                    .map(|i| columns.iter().map(|c| c[kk][i]).collect())
                    .collect();
                let back = transform.back(&rows);
                (0..factors)
                    .map(|v| {
                        let values: Vec<f64> = back.iter().map(|r| r[v]).collect();
                        match support {
                            Some(s) => s.mean(&values),
                            None => Ok(values),
                        }
                    })
                    .collect()
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::post::Keep;
    use crate::sgs::{SgsParams, sgs};
    use estimation::Search;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rand_distr::StandardNormal;
    use transforms::PpmtParams;
    use variogram::Variogram;
    use variogram::model::Model;

    type Point = (f64, f64, f64);

    /// Correlated lognormal pair, with a high-grade cluster in one corner.
    fn samples() -> (Vec<Point>, Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = StdRng::seed_from_u64(11);
        let mut locs = vec![];
        let mut data = vec![];
        for i in 0..300 {
            let (side, lift) = if i < 200 { (100.0, 0.0) } else { (20.0, 1.0) };
            locs.push((rng.r#gen::<f64>() * side, rng.r#gen::<f64>() * side, 0.0));
            let g1: f64 = rng.sample(StandardNormal);
            let g2: f64 = rng.sample(StandardNormal);
            data.push(vec![(g1 + lift).exp(), (0.8 * g1 + 0.6 * g2 + lift).exp()]);
        }
        let first: Vec<f64> = data.iter().map(|r| r[0]).collect();
        let weights = transforms::cell_weights(&locs, &first, 20.0, (0.0, 0.0, 0.0))
            .unwrap()
            .weights;
        (locs, data, weights)
    }

    fn simulate(
        locs: &[Point],
        data: &[Vec<f64>],
        weights: &[f64],
        grid: &[Point],
        n: usize,
    ) -> Vec<ContinuousSummary> {
        let transform =
            Decorrelation::Ppmt(Ppmt::fit(data, Some(weights), &PpmtParams::default()).unwrap());
        let factors = transform.forward(data);
        let vg = Variogram::single(Model::Spherical, 1.0, 15.0);
        let options = ContinuousOptions {
            keep: Keep::All,
            ..Default::default()
        };
        multivariate(n, 3, 2, &transform, &options, None, |_, j, seed| {
            let column: Vec<f64> = factors.iter().map(|r| r[j]).collect();
            let params = SgsParams {
                search: vec![Search {
                    max_samples: 12,
                    ..Default::default()
                }],
                seed,
            };
            Ok(sgs(locs, &column, Some(weights), None, grid, &vg, &params, None)?.values)
        })
        .unwrap()
    }

    fn grid() -> Vec<Point> {
        (0..40 * 40)
            .map(|i| {
                (
                    (i % 40) as f64 * 2.5 + 1.25,
                    (i / 40) as f64 * 2.5 + 1.25,
                    0.0,
                )
            })
            .collect()
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

    fn corr(a: &[f64], b: &[f64], w: Option<&[f64]>) -> f64 {
        let w = w.map_or_else(|| vec![1.0; a.len()], <[f64]>::to_vec);
        let t: f64 = w.iter().sum();
        let ma = a.iter().zip(&w).map(|(x, w)| x * w).sum::<f64>() / t;
        let mb = b.iter().zip(&w).map(|(x, w)| x * w).sum::<f64>() / t;
        let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
        for ((x, y), w) in a.iter().zip(b).zip(&w) {
            c += w * (x - ma) * (y - mb);
            va += w * (x - ma).powi(2);
            vb += w * (y - mb).powi(2);
        }
        c / (va * vb).sqrt()
    }

    #[test]
    fn decorrelation_json_round_trip_is_bit_identical() {
        let (_, data, weights) = samples();
        let transforms = [
            Decorrelation::Pca(Pca::fit(&data, Some(&weights), true).unwrap()),
            Decorrelation::Ppmt(Ppmt::fit(&data, Some(&weights), &PpmtParams::default()).unwrap()),
        ];
        let bits = |rows: Vec<Vec<f64>>| {
            rows.concat()
                .into_iter()
                .map(f64::to_bits)
                .collect::<Vec<_>>()
        };
        for t in transforms {
            let back: Decorrelation =
                serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
            assert_eq!(bits(back.forward(&data)), bits(t.forward(&data)));
            let factors = t.forward(&data);
            assert_eq!(bits(back.back(&factors)), bits(t.back(&factors)));
        }
    }

    #[test]
    fn reproduces_declustered_histograms_and_correlation() {
        let (locs, data, weights) = samples();
        let out = simulate(&locs, &data, &weights, &grid(), 8);
        let reals: Vec<&Vec<Vec<f64>>> = out.iter().map(|s| &s.realizations).collect();
        for (v, r) in reals.iter().enumerate() {
            let column: Vec<f64> = data.iter().map(|row| row[v]).collect();
            let pooled: Vec<f64> = r.iter().flatten().copied().collect();
            let ones = vec![1.0; pooled.len()];
            for q in [0.1, 0.25, 0.5, 0.75, 0.9] {
                let (want, got) = (
                    weighted_quantile(&column, &weights, q),
                    weighted_quantile(&pooled, &ones, q),
                );
                assert!(
                    (got / want).ln().abs() < 0.2,
                    "variable {v} q{q}: {got} vs {want}"
                );
            }
        }
        let (a, b): (Vec<f64>, Vec<f64>) = data.iter().map(|r| (r[0].ln(), r[1].ln())).unzip();
        let want = corr(&a, &b, Some(&weights));
        let mean = (0..8)
            .map(|k| {
                let a: Vec<f64> = reals[0][k].iter().map(|v| v.ln()).collect();
                let b: Vec<f64> = reals[1][k].iter().map(|v| v.ln()).collect();
                corr(&a, &b, None)
            })
            .sum::<f64>()
            / 8.0;
        assert!((mean - want).abs() < 0.1, "correlation {mean} vs {want}");
    }

    #[test]
    fn honors_data_and_ignores_thread_count() {
        let (locs, data, weights) = samples();
        let targets = locs[..20].to_vec();
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| simulate(&locs, &data, &weights, &targets, 5))
        };
        let (one, four) = (run(1), run(4));
        for v in 0..2 {
            let reals = &one[v].realizations;
            assert_eq!(reals, &four[v].realizations);
            for r in reals {
                for (i, value) in r.iter().enumerate() {
                    assert!(
                        (value / data[i][v] - 1.0).abs() < 1e-6,
                        "variable {v} datum {i}: {value} vs {}",
                        data[i][v]
                    );
                }
            }
        }
    }

    #[test]
    fn batched_turning_bands_factors_equal_one_realization_at_a_time() {
        use crate::turning_bands::{
            TurningBandsEnsemble, TurningBandsParams, bounds, turning_bands,
        };
        let (locs, data, weights) = samples();
        let targets = grid();
        let transform =
            Decorrelation::Ppmt(Ppmt::fit(&data, Some(&weights), &PpmtParams::default()).unwrap());
        let factors = transform.forward(&data);
        let column = |j: usize| -> Vec<f64> { factors.iter().map(|r| r[j]).collect() };
        let vg = Variogram::single(Model::Spherical, 1.0, 15.0);
        let params = TurningBandsParams::default();
        let options = ContinuousOptions::default();
        let (n, seed) = (4, 3);
        let want = multivariate(n, seed, 2, &transform, &options, None, |_, j, seed| {
            let params = TurningBandsParams {
                seed,
                ..params.clone()
            };
            Ok(turning_bands(
                &locs,
                &column(j),
                Some(&weights),
                None,
                &targets,
                &vg,
                &params,
            )?
            .values)
        })
        .unwrap();
        let (lo, hi) = bounds(&targets);
        for batch in [1, 3, n] {
            let got = multivariate_batched(n, 2, &transform, &options, None, batch, |j, ks| {
                let seeds: Vec<u64> = ks.clone().map(|k| factor_seed(seed, k, j)).collect();
                TurningBandsEnsemble::with_realizations(
                    &locs,
                    &[column(j)],
                    Some(&weights),
                    None,
                    lo,
                    hi,
                    &vg,
                    &params,
                    &seeds,
                )?
                .realizations(0..ks.len(), &targets, |_| None, None)
            })
            .unwrap();
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.mean, w.mean);
                assert_eq!(g.variance, w.variance);
                assert_eq!(g.realization_mean, w.realization_mean);
            }
        }
    }

    #[test]
    fn factor_seeds_differ() {
        let seeds: std::collections::HashSet<u64> = (0..50)
            .flat_map(|k| (0..4).map(move |j| factor_seed(7, k, j)))
            .collect();
        assert_eq!(seeds.len(), 200);
    }
}
