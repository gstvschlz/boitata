//! Multivariate simulation through independent factors: the variables are
//! decorrelated, each factor is simulated on its own, and every realization is
//! back-transformed at the nodes before any averaging to blocks.
//!
//! A trend, one per variable, is removed first with the stepwise conditional
//! transform [`TrendConditioning`] SGS uses, and put back at each node after
//! the joint back-transform, still before averaging to blocks.

use transforms::{Maf, Pca, Ppmt, StepwiseConditional};

use crate::trend::TrendConditioning;

use boitata_core::rng::realization_seed;
use rayon::prelude::*;

use crate::error::Result;
use crate::post::{
    BlockSupport, ContinuousOptions, ContinuousSummary, continuous_many_batched,
    continuous_many_bounded,
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

/// Per-variable trend removed before the joint transform.
///
/// `trends` has one entry per variable; `None` leaves that variable as it is.
/// The returned rows are what the joint transform is fitted to and what the
/// factors are simulated from. `back` of each conditioning, with the trend at
/// a node, restores that variable.
pub fn condition_variables(
    data: &[Vec<f64>],
    trends: &[Option<&[f64]>],
    weights: Option<&[f64]>,
    classes: usize,
) -> Result<(Vec<Vec<f64>>, Vec<Option<TrendConditioning>>)> {
    let n = data.len();
    let p = data.first().map_or(0, Vec::len);
    if trends.len() != p {
        return Err(crate::error::SimError::InvalidParameters(
            "one trend column per variable".into(),
        ));
    }
    if data.iter().any(|row| row.len() != p) {
        return Err(crate::error::SimError::InvalidParameters(
            "one value per variable".into(),
        ));
    }
    if let Some(w) = weights
        && w.len() != n
    {
        return Err(crate::error::SimError::InvalidParameters(
            "one weight per datum".into(),
        ));
    }
    let mut columns: Vec<Vec<f64>> = (0..p)
        .map(|j| data.iter().map(|row| row[j]).collect())
        .collect();
    let mut fitted = Vec::with_capacity(p);
    for (j, trend) in trends.iter().enumerate() {
        let Some(trend) = trend else {
            fitted.push(None);
            continue;
        };
        if trend.len() != n {
            return Err(crate::error::SimError::InvalidParameters(
                "one trend value per datum".into(),
            ));
        }
        let seen: Vec<usize> = columns[j]
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(i, _)| i)
            .collect();
        let values: Vec<f64> = seen.iter().map(|&i| columns[j][i]).collect();
        let at: Vec<f64> = seen.iter().map(|&i| trend[i]).collect();
        let local = weights.map(|w| seen.iter().map(|&i| w[i]).collect::<Vec<_>>());
        let conditioning = TrendConditioning::fit(&values, &at, local.as_deref(), classes)?;
        for (k, &i) in seen.iter().enumerate() {
            columns[j][i] = conditioning.scores()[k];
        }
        fitted.push(Some(conditioning));
    }
    let rows = (0..n)
        .map(|i| (0..p).map(|j| columns[j][i]).collect())
        .collect();
    Ok((rows, fitted))
}

/// Trend conditioning of each variable and the trend at the nodes being
/// simulated. `None` leaves that variable as the joint back-transform left it.
pub struct NodeTrends<'a> {
    pub conditioning: &'a [Option<TrendConditioning>],
    pub at_nodes: &'a [Option<Vec<f64>>],
}

impl NodeTrends<'_> {
    fn restore(&self, variable: usize, values: Vec<f64>) -> Result<Vec<f64>> {
        let conditioning = self.conditioning.get(variable).and_then(Option::as_ref);
        let trend = self.at_nodes.get(variable).and_then(Option::as_deref);
        match (conditioning, trend) {
            (Some(c), Some(trend)) => c.back(trend, &values),
            (None, None) => Ok(values),
            (Some(_), None) => Err(crate::error::SimError::InvalidParameters(format!(
                "variable {variable} was fitted with a trend; give its trend at the targets"
            ))),
            (None, Some(_)) => Err(crate::error::SimError::InvalidParameters(format!(
                "a trend for variable {variable} at the targets needs one at fit"
            ))),
        }
    }
}

fn with_trend(values: Vec<f64>, variable: usize, trends: Option<&NodeTrends>) -> Result<Vec<f64>> {
    match trends {
        Some(t) => t.restore(variable, values),
        None => Ok(values),
    }
}

/// Summaries of `n` realizations of every variable. `simulate(k, j, seed)`
/// returns factor `j` of realization `k` at the nodes, simulated with `seed`;
/// the factors of a realization are back-transformed together, each variable's
/// trend is put back at the nodes, then the values are averaged to `support`.
#[allow(clippy::too_many_arguments)]
pub fn multivariate(
    n: usize,
    seed: u64,
    factors: usize,
    transform: &Decorrelation,
    options: &ContinuousOptions,
    support: Option<&BlockSupport>,
    trends: Option<&NodeTrends>,
    simulate: impl Fn(usize, usize, u64) -> Result<Vec<f64>> + Sync,
) -> Result<Vec<ContinuousSummary>> {
    multivariate_bounded(
        n,
        seed,
        factors,
        transform,
        options,
        support,
        usize::MAX,
        trends,
        simulate,
    )
}

/// As [`multivariate`] with at most `in_flight` realizations simulated at
/// once (see [`crate::post::in_flight`]). The summaries do not depend on
/// `in_flight`.
#[allow(clippy::too_many_arguments)]
pub fn multivariate_bounded(
    n: usize,
    seed: u64,
    factors: usize,
    transform: &Decorrelation,
    options: &ContinuousOptions,
    support: Option<&BlockSupport>,
    in_flight: usize,
    trends: Option<&NodeTrends>,
    simulate: impl Fn(usize, usize, u64) -> Result<Vec<f64>> + Sync,
) -> Result<Vec<ContinuousSummary>> {
    continuous_many_bounded(n, factors, options, in_flight, |k| {
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
                let values = with_trend(values, v, trends)?;
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
    trends: Option<&NodeTrends>,
    mut simulate: impl FnMut(usize, std::ops::Range<usize>) -> Result<Vec<Vec<f64>>>,
) -> Result<Vec<ContinuousSummary>> {
    continuous_many_batched(n, factors, options, batch, |ks| {
        let columns = (0..factors)
            .map(|j| simulate(j, ks.clone()))
            .collect::<Result<Vec<_>>>()?;
        (0..ks.len())
            .into_par_iter()
            .map(|kk| {
                let nodes = columns.first().map_or(0, |c| c[kk].len());
                let rows: Vec<Vec<f64>> = (0..nodes)
                    .map(|i| columns.iter().map(|c| c[kk][i]).collect())
                    .collect();
                let back = transform.back(&rows);
                (0..factors)
                    .map(|v| {
                        let values: Vec<f64> = back.iter().map(|r| r[v]).collect();
                        let values = with_trend(values, v, trends)?;
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
        simulate_bounded(locs, data, weights, grid, n, usize::MAX)
    }

    fn simulate_bounded(
        locs: &[Point],
        data: &[Vec<f64>],
        weights: &[f64],
        grid: &[Point],
        n: usize,
        in_flight: usize,
    ) -> Vec<ContinuousSummary> {
        let transform =
            Decorrelation::Ppmt(Ppmt::fit(data, Some(weights), &PpmtParams::default()).unwrap());
        let factors = transform.forward(data);
        let vg = Variogram::single(Model::Spherical, 1.0, 15.0);
        let options = ContinuousOptions {
            keep: Keep::All,
            ..Default::default()
        };
        let simulate = |_, j: usize, seed| {
            let column: Vec<f64> = factors.iter().map(|r| r[j]).collect();
            let params = SgsParams {
                search: vec![Search {
                    max_samples: 12,
                    ..Default::default()
                }],
                seed,
            };
            Ok(sgs(locs, &column, Some(weights), None, grid, &vg, &params, None)?.values)
        };
        multivariate_bounded(
            n, 3, 2, &transform, &options, None, in_flight, None, simulate,
        )
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
    fn realizations_do_not_depend_on_realizations_in_flight() {
        let (locs, data, weights) = samples();
        let targets = locs[..20].to_vec();
        let run = |threads, in_flight| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| simulate_bounded(&locs, &data, &weights, &targets, 5, in_flight))
        };
        let want = run(1, usize::MAX);
        for (threads, in_flight) in [(3, 1), (3, 2), (3, usize::MAX), (1, 1)] {
            let got = run(threads, in_flight);
            for v in 0..2 {
                assert_eq!(
                    got[v].realizations, want[v].realizations,
                    "{threads} threads, {in_flight} in flight"
                );
            }
        }
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
        let want = multivariate(
            n,
            seed,
            2,
            &transform,
            &options,
            None,
            None,
            |_, j, seed| {
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
            },
        )
        .unwrap();
        let (lo, hi) = bounds(&targets);
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        for (batch, threads) in [(1, None), (3, None), (n, None), (n, Some(&one))] {
            let run = || {
                multivariate_batched(n, 2, &transform, &options, None, batch, None, |j, ks| {
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
            };
            let got = match threads {
                Some(pool) => pool.install(run),
                None => run(),
            }
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

    fn draw_factors(_: usize, j: usize, _: u64) -> Result<Vec<f64>> {
        Ok(vec![0.2 + j as f64, -0.1 * j as f64])
    }

    #[test]
    fn no_trend_matches_leaving_every_variable_untrended() {
        let data = vec![
            vec![1.0, 4.0],
            vec![1.4, 3.0],
            vec![0.6, 5.0],
            vec![2.0, 2.5],
        ];
        let transform = Decorrelation::Pca(Pca::fit(&data, None, false).unwrap());
        let options = ContinuousOptions {
            keep: Keep::All,
            ..Default::default()
        };
        let plain = multivariate(1, 0, 2, &transform, &options, None, None, draw_factors).unwrap();
        let none = vec![None, None];
        let at = vec![None, None];
        let trends = NodeTrends {
            conditioning: &none,
            at_nodes: &at,
        };
        let tagged = multivariate(
            1,
            0,
            2,
            &transform,
            &options,
            None,
            Some(&trends),
            draw_factors,
        )
        .unwrap();
        for v in 0..2 {
            assert_eq!(plain[v].realizations, tagged[v].realizations);
        }
    }

    /// Two grades, each a plane plus a shared Gaussian residual.
    fn trended_pair() -> (Vec<Point>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut rng = StdRng::seed_from_u64(21);
        let mut locs = vec![];
        let mut data = vec![];
        let mut trends = vec![vec![], vec![]];
        for _ in 0..480 {
            let p = (rng.r#gen::<f64>() * 100.0, rng.r#gen::<f64>() * 100.0, 0.0);
            let (t0, t1) = (2.0 + p.0 / 100.0, 2.0 + 0.7 * p.1 / 100.0);
            let g1: f64 = rng.sample(StandardNormal);
            let g2: f64 = rng.sample(StandardNormal);
            locs.push(p);
            trends[0].push(t0);
            trends[1].push(t1);
            data.push(vec![t0 + 0.2 * g1, t1 + 0.2 * (0.8 * g1 + 0.6 * g2)]);
        }
        (locs, data, trends)
    }

    fn far_nodes(n: usize) -> Vec<Point> {
        (0..n)
            .map(|i| {
                (
                    1.0e5 + (i % 20) as f64 * 40.0,
                    1.0e5 + (i / 20) as f64 * 40.0,
                    0.0,
                )
            })
            .collect()
    }

    fn trended(
        locs: &[Point],
        data: &[Vec<f64>],
        trends: &[Vec<f64>],
        nodes: &[Point],
        node_trends: &[Vec<f64>],
        n: usize,
        in_flight: usize,
    ) -> Vec<ContinuousSummary> {
        let refs: Vec<Option<&[f64]>> = trends.iter().map(|t| Some(t.as_slice())).collect();
        let (scores, mut conditioning) = condition_variables(data, &refs, None, 6).unwrap();
        conditioning = conditioning
            .iter()
            .map(|c| {
                c.as_ref()
                    .map(|c| serde_json::from_str(&serde_json::to_string(c).unwrap()).unwrap())
            })
            .collect();
        let transform =
            Decorrelation::Ppmt(Ppmt::fit(&scores, None, &PpmtParams::default()).unwrap());
        let factors = transform.forward(&scores);
        let vg = Variogram::single(Model::Spherical, 1.0, 15.0);
        let options = ContinuousOptions {
            keep: Keep::All,
            ..Default::default()
        };
        let at: Vec<Option<Vec<f64>>> = node_trends.iter().cloned().map(Some).collect();
        let node_trends = NodeTrends {
            conditioning: &conditioning,
            at_nodes: &at,
        };
        let simulate = |_: usize, j: usize, seed| {
            let column: Vec<f64> = factors.iter().map(|row| row[j]).collect();
            let params = SgsParams {
                search: vec![Search {
                    max_samples: 12,
                    radius: 30.0,
                    ..Default::default()
                }],
                seed,
            };
            Ok(sgs(locs, &column, None, None, nodes, &vg, &params, None)?.values)
        };
        multivariate_bounded(
            n,
            5,
            2,
            &transform,
            &options,
            None,
            in_flight,
            Some(&node_trends),
            simulate,
        )
        .unwrap()
    }

    fn mad(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / a.len() as f64
    }

    #[test]
    fn away_from_data_the_mean_follows_each_trend_and_the_correlation_holds() {
        let (locs, data, trends) = trended_pair();
        let nodes = far_nodes(160);
        let node_trends: Vec<Vec<f64>> = (0..2)
            .map(|j| (0..nodes.len()).map(|i| trends[j][i]).collect())
            .collect();
        let out = trended(&locs, &data, &trends, &nodes, &node_trends, 30, usize::MAX);
        for v in 0..2 {
            let along = mad(&out[v].mean, &node_trends[v]);
            let flat = mad(
                &out[v].mean,
                &vec![out[v].mean.iter().sum::<f64>() / nodes.len() as f64; nodes.len()],
            );
            assert!(
                along < 0.12 && along < flat * 0.6,
                "variable {v}: {along} from its trend, {flat} from a flat mean"
            );
        }
        let columns: Vec<Vec<f64>> = (0..2)
            .map(|v| data.iter().take(nodes.len()).map(|row| row[v]).collect())
            .collect();
        let want = corr(&columns[0], &columns[1], None);
        let got = (0..30)
            .map(|k| corr(&out[0].realizations[k], &out[1].realizations[k], None))
            .sum::<f64>()
            / 30.0;
        assert!((got - want).abs() < 0.12, "correlation {got} vs {want}");
    }

    #[test]
    fn a_trend_honors_data_and_ignores_thread_count_and_batching() {
        let (locs, data, trends) = trended_pair();
        let n_data = 16;
        let nodes = locs[..n_data].to_vec();
        let node_trends: Vec<Vec<f64>> = (0..2).map(|j| trends[j][..n_data].to_vec()).collect();
        let run = |threads, in_flight| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| trended(&locs, &data, &trends, &nodes, &node_trends, 4, in_flight))
        };
        let one = run(1, usize::MAX);
        for (threads, in_flight) in [(4, usize::MAX), (1, 1), (3, 2)] {
            let got = run(threads, in_flight);
            for v in 0..2 {
                assert_eq!(
                    got[v].realizations, one[v].realizations,
                    "{threads} threads, {in_flight} in flight"
                );
            }
        }
        for v in 0..2 {
            for real in &one[v].realizations {
                for (got, want) in real.iter().zip(data.iter().take(n_data).map(|row| row[v])) {
                    assert!((got / want - 1.0).abs() < 1e-6, "{got} vs {want}");
                }
            }
        }
    }

    #[test]
    fn one_variable_may_have_no_trend() {
        let (locs, data, trends) = trended_pair();
        let refs = [Some(trends[0].as_slice()), None];
        let (scores, conditioning) = condition_variables(&data, &refs, None, 6).unwrap();
        assert!(conditioning[0].is_some() && conditioning[1].is_none());
        let nodes = far_nodes(80);
        let node_trend = (0..nodes.len()).map(|i| trends[0][i]).collect::<Vec<_>>();
        let at = vec![Some(node_trend.clone()), None];
        let transform = Decorrelation::Pca(Pca::fit(&scores, None, true).unwrap());
        let factors = transform.forward(&scores);
        let vg = Variogram::single(Model::Spherical, 1.0, 15.0);
        let options = ContinuousOptions {
            keep: Keep::All,
            ..Default::default()
        };
        let nodes_trend = NodeTrends {
            conditioning: &conditioning,
            at_nodes: &at,
        };
        let out = multivariate(
            24,
            9,
            2,
            &transform,
            &options,
            None,
            Some(&nodes_trend),
            |_, j, seed| {
                let column: Vec<f64> = factors.iter().map(|row| row[j]).collect();
                let params = SgsParams {
                    search: vec![Search {
                        max_samples: 12,
                        radius: 30.0,
                        ..Default::default()
                    }],
                    seed,
                };
                Ok(sgs(&locs, &column, None, None, &nodes, &vg, &params, None)?.values)
            },
        )
        .unwrap();
        let along = mad(&out[0].mean, &node_trend);
        assert!(along < 0.12, "trended variable {along} from its trend");
        let mean1 = out[1].mean.iter().sum::<f64>() / nodes.len() as f64;
        let flat = mad(&out[1].mean, &vec![mean1; nodes.len()]);
        assert!(flat < 0.08, "untrended variable wanders by {flat}");
        let at_data = locs[..12].to_vec();
        let data_trend = vec![Some(trends[0][..12].to_vec()), None];
        let at_data_trend = NodeTrends {
            conditioning: &conditioning,
            at_nodes: &data_trend,
        };
        let honored = multivariate(
            2,
            3,
            2,
            &transform,
            &options,
            None,
            Some(&at_data_trend),
            |_, j, seed| {
                let column: Vec<f64> = factors.iter().map(|row| row[j]).collect();
                let params = SgsParams {
                    search: vec![Search {
                        max_samples: 12,
                        radius: 30.0,
                        ..Default::default()
                    }],
                    seed,
                };
                Ok(sgs(&locs, &column, None, None, &at_data, &vg, &params, None)?.values)
            },
        )
        .unwrap();
        for v in 0..2 {
            for real in &honored[v].realizations {
                for (got, want) in real.iter().zip(data.iter().take(12).map(|row| row[v])) {
                    assert!(
                        (got / want - 1.0).abs() < 1e-6,
                        "variable {v}: {got} vs {want}"
                    );
                }
            }
        }
        assert!(
            NodeTrends {
                conditioning: &conditioning,
                at_nodes: &[None, None],
            }
            .restore(0, vec![0.0])
            .is_err()
        );
        assert!(condition_variables(&data, &[Some(&trends[0][1..])], None, 6).is_err());
    }
}
