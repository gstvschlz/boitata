//! Direct sequential simulation (DSS).
//!
//! The sequential loop of [`crate::sgs`] in data units: each node is simple
//! kriged from raw neighbors with its domain's declustered mean and the
//! variogram rescaled to the domain's declustered variance, then drawn from
//! the domain's declustered histogram through a Gaussian `m + s·z`, whose
//! back-transform has the kriged mean and variance (a lookup table over
//! `(m, s)`). A kriged pair no draw can reach takes the nearest reachable
//! one, and is counted.

use crate::error::{Result, SimError};
use crate::sgs::{Domains, SgsParams, Space, Transform, Transforms, inputs, sequential};
use estimation::lva::LocalAnisotropy;
use std::sync::atomic::{AtomicUsize, Ordering};
use transforms::weighted_mean_variance;
use variogram::{Structure, Variogram};

/// One DSS realization over the simulation grid.
#[derive(Debug, Clone)]
pub struct DssRealization {
    /// Simulated values, one per grid node (same order as `grid`).
    pub values: Vec<f64>,
    /// Nodes whose kriged mean and variance no draw reaches, drawn from the
    /// nearest reachable pair.
    pub clamped: usize,
}

const MEANS: usize = 161;
const SPREADS: usize = 41;

fn gaussian_mean(i: usize) -> f64 {
    -4.0 + 8.0 * i as f64 / (MEANS - 1) as f64
}

fn spread(j: usize) -> f64 {
    2.0 * j as f64 / (SPREADS - 1) as f64
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Scores at which the back-transform is tabulated for the lookup: from
/// `-REACH` to `REACH` by `STEP`.
const STEP: f64 = 0.01;
const REACH: f64 = 16.0;

/// Mean and variance of `back(m + s·Z)`, `Z` standard normal, over a grid of
/// `m` in [-4, 4] and `s` in [0, 2], row by row of `s`. The expectations
/// are Gaussian-weighted sums of the back-transform tabulated every `STEP`
/// over ±8 `s`: a quadrature over few nodes misses the kinks of the
/// piecewise-linear back-transform by a few percent.
struct Lookup {
    means: Vec<f64>,
    variances: Vec<f64>,
}

impl Lookup {
    fn new(transform: &Transform) -> Self {
        let last = (2.0 * REACH / STEP).round() as isize;
        let back: Vec<f64> = (0..=last)
            .map(|k| transform.back(-REACH + k as f64 * STEP, 0.0))
            .collect();
        let mut means = Vec::with_capacity(MEANS * SPREADS);
        let mut variances = Vec::with_capacity(MEANS * SPREADS);
        for j in 0..SPREADS {
            let s = spread(j);
            let reach = (8.0 * s / STEP).ceil() as isize;
            let kernel: Vec<f64> = (-reach..=reach)
                .map(|d| (-0.5 * (d as f64 * STEP / s).powi(2)).exp())
                .collect();
            for i in 0..MEANS {
                let c = ((gaussian_mean(i) + REACH) / STEP).round() as isize;
                let center = back[c as usize];
                if s == 0.0 {
                    means.push(center);
                    variances.push(0.0);
                    continue;
                }
                let (mut total, mut first, mut second) = (0.0, 0.0, 0.0);
                for k in (c - reach).max(0)..=(c + reach).min(last) {
                    let (w, d) = (kernel[(k - c + reach) as usize], back[k as usize] - center);
                    total += w;
                    first += w * d;
                    second += w * d * d;
                }
                let (first, second) = (first / total, second / total);
                means.push(center + first);
                variances.push((second - first * first).max(0.0));
            }
        }
        Self { means, variances }
    }

    /// The `m` of row `j` whose mean is `mean`, and the variance there.
    fn hit(&self, j: usize, mean: f64) -> Option<(f64, f64)> {
        let row = j * MEANS..(j + 1) * MEANS;
        let (means, variances) = (&self.means[row.clone()], &self.variances[row]);
        if !(means[0]..=means[MEANS - 1]).contains(&mean) {
            return None;
        }
        let i = means.partition_point(|&x| x < mean).clamp(1, MEANS - 1);
        let width = means[i] - means[i - 1];
        let t = if width > 0.0 {
            ((mean - means[i - 1]) / width).clamp(0.0, 1.0)
        } else {
            0.0
        };
        Some((
            lerp(gaussian_mean(i - 1), gaussian_mean(i), t),
            lerp(variances[i - 1], variances[i], t),
        ))
    }

    /// `(m, s, clamped)` such that `back(m + s·Z)` has `mean` and `variance`.
    fn query(&self, mean: f64, variance: f64) -> (f64, f64, bool) {
        let Some((m, v)) = self.hit(0, mean) else {
            let m = if mean < self.means[0] { -4.0 } else { 4.0 };
            return (m, 0.0, true);
        };
        let mut previous = (m, 0.0, v);
        for j in 1..SPREADS {
            let Some((m, v)) = self.hit(j, mean) else {
                break;
            };
            if v >= variance {
                let (m0, s0, v0) = previous;
                if variance <= v0 {
                    return (m0, s0, false);
                }
                // On the standard deviation, about linear in `s`.
                let t = ((variance.sqrt() - v0.sqrt()) / (v.sqrt() - v0.sqrt())).clamp(0.0, 1.0);
                return (lerp(m0, m, t), lerp(s0, spread(j), t), false);
            }
            previous = (m, spread(j), v);
        }
        (previous.0, previous.1, true)
    }
}

struct Domain {
    transform: Transform,
    lookup: Lookup,
    vg: Variogram,
    mean: f64,
}

/// Raw values kriged with each domain's mean and rescaled variogram,
/// cokriged with a collocated secondary, drawn through its lookup.
struct DssSpace<'a> {
    domains: &'a [Option<Domain>],
    /// Standardized secondary at each node and its correlation.
    collocated: Option<(&'a [f64], f64)>,
    clamped: AtomicUsize,
}

impl DssSpace<'_> {
    fn domain(&self, domain: Option<u32>) -> &Domain {
        self.domains[domain.unwrap_or(0) as usize]
            .as_ref()
            .expect("checked")
    }

    /// Kriged `mean` and `variance` cokriged with the secondary at `node`,
    /// in the domain's standardized units.
    fn cokriged(&self, d: &Domain, mean: f64, variance: f64, node: usize) -> (f64, f64) {
        let Some((secondary, rho)) = self.collocated else {
            return (mean, variance);
        };
        let sill = d.vg.total_sill();
        let sd = sill.sqrt();
        let (m, v) = estimation::markov_collocated(
            (mean - d.mean) / sd,
            variance / sill,
            1.0,
            rho,
            secondary[node],
        );
        (d.mean + sd * m, sill * v)
    }

    fn drawn(&self, d: &Domain, mean: f64, variance: f64, z: f64) -> (f64, f64) {
        let (m, s, clamped) = d.lookup.query(mean, variance);
        if clamped {
            self.clamped.fetch_add(1, Ordering::Relaxed);
        }
        let value = d.transform.back(m + s * z, 0.0);
        (value, value)
    }
}

impl Space for DssSpace<'_> {
    fn neighbor(&self, value: f64, _: f64, _: Option<u32>) -> f64 {
        value
    }

    fn variogram(&self, domain: Option<u32>) -> &Variogram {
        &self.domain(domain).vg
    }

    fn mean(&self, domain: Option<u32>) -> f64 {
        self.domain(domain).mean
    }

    fn draw(
        &self,
        mean: f64,
        variance: f64,
        node: usize,
        domain: Option<u32>,
        _: f64,
        z: f64,
    ) -> (f64, f64) {
        let d = self.domain(domain);
        let (mean, variance) = self.cokriged(d, mean, variance, node);
        self.drawn(d, mean, variance, z)
    }

    fn marginal(&self, node: usize, domain: Option<u32>, _: f64, z: f64) -> (f64, f64) {
        let d = self.domain(domain);
        if self.collocated.is_none() {
            let value = d.transform.back(z, 0.0);
            return (value, value);
        }
        let (mean, variance) = self.cokriged(d, d.mean, d.vg.total_sill(), node);
        self.drawn(d, mean, variance, z)
    }
}

/// A secondary variable for collocated co-DSS: its declustered mean and
/// standard deviation at the data, which standardize it, and its
/// correlation with the primary.
#[derive(Debug, Clone)]
pub struct DssSecondary {
    mean: f64,
    sd: f64,
    pub correlation: f64,
}

impl DssSecondary {
    /// The secondary `value` in standardized units.
    pub fn standardize(&self, value: f64) -> f64 {
        (value - self.mean) / self.sd
    }
}

/// DSS fitted to its data: each domain's declustered histogram, lookup
/// table, mean and rescaled variogram, built once for any number of
/// realizations.
pub struct Dss<'a> {
    data_locs: &'a [(f64, f64, f64)],
    data_vals: &'a [f64],
    data_weights: Option<&'a [f64]>,
    data_holes: Option<&'a [u32]>,
    data_domains: Option<&'a [u32]>,
    vg: &'a Variogram,
    /// By domain code; `None` for a domain without data or whose values do
    /// not vary, an error only once a node falls in it.
    domains: Vec<Option<Domain>>,
}

impl<'a> Dss<'a> {
    /// Fitted to the data, with `data_domains` the domain code of each
    /// datum.
    ///
    /// `vg` is the variogram of the values; within each domain its nugget
    /// and sills are rescaled so the total sill is the domain's declustered
    /// variance (by `data_weights`), and nodes are simple kriged with the
    /// domain's declustered mean.
    pub fn new(
        data_locs: &'a [(f64, f64, f64)],
        data_vals: &'a [f64],
        data_weights: Option<&'a [f64]>,
        data_holes: Option<&'a [u32]>,
        data_domains: Option<&'a [u32]>,
        vg: &'a Variogram,
    ) -> Result<Self> {
        let sill = vg.total_sill();
        if !(sill.is_finite() && sill > 0.0 && vg.is_stationary()) {
            return Err(SimError::InvalidParameters(
                "the variogram needs a finite positive sill".into(),
            ));
        }
        if let Some(v) = data_vals.iter().find(|v| !v.is_finite()) {
            return Err(SimError::InvalidParameters(format!(
                "data value {v} is not finite"
            )));
        }
        let fitted = Transforms::fit(data_vals, data_weights, data_domains, None)?;
        let domains = fitted
            .domains
            .into_iter()
            .enumerate()
            .map(|(k, transform)| {
                let Some(transform) = transform else {
                    return Ok(None);
                };
                let rows: Vec<usize> = (0..data_vals.len())
                    .filter(|&i| data_domains.map_or(0, |c| c[i] as usize) == k)
                    .collect();
                let pick = |v: &[f64]| rows.iter().map(|&i| v[i]).collect::<Vec<_>>();
                let (mean, variance) =
                    weighted_mean_variance(&pick(data_vals), data_weights.map(pick).as_deref())
                        .map_err(|e| SimError::Transform(e.to_string()))?;
                if variance <= 0.0 {
                    return Ok(None);
                }
                let f = variance / sill;
                let vg = Variogram {
                    nugget: vg.nugget * f,
                    structures: vg
                        .structures
                        .iter()
                        .map(|s| Structure {
                            sill: s.sill * f,
                            ..*s
                        })
                        .collect(),
                    anisotropy: vg.anisotropy.clone(),
                };
                Ok(Some(Domain {
                    lookup: Lookup::new(&transform),
                    transform,
                    vg,
                    mean,
                }))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            data_locs,
            data_vals,
            data_weights,
            data_holes,
            data_domains,
            vg,
            domains,
        })
    }

    /// One realization over `grid`, with `node_domains` the domain code of
    /// each node when fitted with domains (see [`crate::sgs_in`]).
    pub fn simulate(
        &self,
        node_domains: Option<&[u32]>,
        grid: &[(f64, f64, f64)],
        params: &SgsParams,
        local: Option<&LocalAnisotropy>,
    ) -> Result<DssRealization> {
        self.realization(node_domains, grid, params, local, None)
    }

    /// The secondary variable `at_data`, one value per datum, standardized
    /// by its mean and standard deviation declustered by the data weights;
    /// `correlation`, in [-1, 1], or else the weighted correlation of the
    /// secondary with the primary standardized within each domain.
    pub fn secondary(&self, at_data: &[f64], correlation: Option<f64>) -> Result<DssSecondary> {
        if at_data.len() != self.data_vals.len() {
            return Err(SimError::InvalidParameters(
                "one secondary value per datum".into(),
            ));
        }
        if at_data.iter().any(|v| !v.is_finite()) {
            return Err(SimError::InvalidParameters(
                "the secondary must be finite at every datum".into(),
            ));
        }
        let (mean, variance) = weighted_mean_variance(at_data, self.data_weights)
            .map_err(|e| SimError::Transform(e.to_string()))?;
        if variance <= 0.0 {
            return Err(SimError::InsufficientData(
                "the secondary does not vary at the data".into(),
            ));
        }
        let correlation = correlation.unwrap_or_else(|| {
            let (mut primary, mut secondary, mut weights) = (vec![], vec![], vec![]);
            for (i, (&v, &s)) in self.data_vals.iter().zip(at_data).enumerate() {
                let code = self.data_domains.map_or(0, |c| c[i] as usize);
                if let Some(d) = &self.domains[code] {
                    primary.push((v - d.mean) / d.vg.total_sill().sqrt());
                    secondary.push(s);
                    weights.push(self.data_weights.map_or(1.0, |w| w[i]));
                }
            }
            crate::sgs::pearson(&primary, &secondary, Some(&weights))
        });
        if !(-1.0..=1.0).contains(&correlation) {
            return Err(SimError::InvalidParameters(format!(
                "correlation {correlation} outside [-1, 1]"
            )));
        }
        Ok(DssSecondary {
            mean,
            sd: variance.sqrt(),
            correlation,
        })
    }

    /// As [`Dss::simulate`], cosimulated with a `secondary` known at every
    /// node, `at_nodes` in its units: each node's kriged mean and variance
    /// are those of the collocated simple cokriging, in standardized units,
    /// of its neighbors and the secondary at the node under the Markov
    /// model, the cross-covariance the correlation times the primary's.
    /// A zero correlation is plain DSS.
    pub fn cosimulate(
        &self,
        node_domains: Option<&[u32]>,
        grid: &[(f64, f64, f64)],
        params: &SgsParams,
        local: Option<&LocalAnisotropy>,
        secondary: &DssSecondary,
        at_nodes: &[f64],
    ) -> Result<DssRealization> {
        if at_nodes.len() != grid.len() {
            return Err(SimError::InvalidParameters(
                "one secondary value per node".into(),
            ));
        }
        if at_nodes.iter().any(|v| !v.is_finite()) {
            return Err(SimError::InvalidParameters(
                "the secondary must be finite at every node".into(),
            ));
        }
        if secondary.correlation == 0.0 {
            return self.simulate(node_domains, grid, params, local);
        }
        let scores: Vec<f64> = at_nodes.iter().map(|&v| secondary.standardize(v)).collect();
        self.realization(
            node_domains,
            grid,
            params,
            local,
            Some((&scores, secondary.correlation)),
        )
    }

    fn realization(
        &self,
        node_domains: Option<&[u32]>,
        grid: &[(f64, f64, f64)],
        params: &SgsParams,
        local: Option<&LocalAnisotropy>,
        collocated: Option<(&[f64], f64)>,
    ) -> Result<DssRealization> {
        let domains = match (self.data_domains, node_domains) {
            (Some(d), Some(n)) => Some((d, n)),
            (None, None) => None,
            _ => {
                return Err(SimError::InvalidParameters(
                    "domains at both the data and the nodes, or neither".into(),
                ));
            }
        };
        let holes = inputs(
            self.data_locs,
            self.data_vals,
            self.data_holes,
            domains,
            None,
            grid,
            params,
            local,
        )?;
        let flat = match node_domains {
            None => (!grid.is_empty() && self.domains[0].is_none()).then_some(0),
            Some(n) => n
                .iter()
                .copied()
                .find(|&c| self.domains[c as usize].is_none()),
        };
        if let Some(k) = flat {
            return Err(SimError::InsufficientData(format!(
                "the values of domain {k} do not vary"
            )));
        }
        let space = DssSpace {
            domains: &self.domains,
            collocated,
            clamped: AtomicUsize::new(0),
        };
        let realization = sequential(
            &space,
            self.data_vals.to_vec(),
            self.data_locs,
            self.data_vals,
            holes,
            domains,
            None,
            grid,
            self.vg,
            params,
            local,
            None,
            |_, _, _, _| {},
        )?;
        Ok(DssRealization {
            values: realization.values,
            clamped: space.clamped.into_inner(),
        })
    }
}

/// One DSS realization, with `domains` as in [`crate::sgs_in`]: [`Dss::new`]
/// then [`Dss::simulate`]. Several realizations share one [`Dss`].
#[allow(clippy::too_many_arguments)]
pub fn dss_in(
    data_locs: &[(f64, f64, f64)],
    data_vals: &[f64],
    data_weights: Option<&[f64]>,
    data_holes: Option<&[u32]>,
    domains: Option<Domains>,
    grid: &[(f64, f64, f64)],
    vg: &Variogram,
    params: &SgsParams,
    local: Option<&LocalAnisotropy>,
) -> Result<DssRealization> {
    Dss::new(
        data_locs,
        data_vals,
        data_weights,
        data_holes,
        domains.map(|d| d.0),
        vg,
    )?
    .simulate(domains.map(|d| d.1), grid, params, local)
}

#[cfg(test)]
mod tests {
    use super::*;
    use estimation::search::Search;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rand_distr::{Distribution, Normal};
    use transforms::probit;
    use variogram::Model;

    fn moments(v: &[f64]) -> (f64, f64) {
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        (mean, v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n)
    }

    #[test]
    fn lookup_draws_have_the_queried_mean_and_variance() {
        let mut rng = StdRng::seed_from_u64(5);
        let normal = Normal::new(0.0_f64, 0.8).unwrap();
        let values: Vec<f64> = (0..500).map(|_| normal.sample(&mut rng).exp()).collect();
        let transform = Transforms::fit(&values, None, None, None)
            .unwrap()
            .domains
            .remove(0)
            .unwrap();
        let lookup = Lookup::new(&transform);
        let (mean, variance) = moments(&values);
        // Stratified standard normal draws, so the check is the lookup's
        // error and not the sampling error of a skewed variance.
        let n = 20_000;
        let z: Vec<f64> = (0..n)
            .map(|k| probit((k as f64 + 0.5) / n as f64))
            .collect();
        for (mu, var) in [
            (mean, 0.5 * variance),
            (0.6 * mean, 0.2 * variance),
            (1.5 * mean, variance),
            (mean, 0.02 * variance),
        ] {
            let (m, s, clamped) = lookup.query(mu, var);
            assert!(!clamped, "({mu}, {var})");
            let draws: Vec<f64> = z.iter().map(|z| transform.back(m + s * z, 0.0)).collect();
            let (got_mean, got_var) = moments(&draws);
            assert!((got_mean / mu - 1.0).abs() < 0.02, "{got_mean} vs {mu}");
            assert!((got_var / var - 1.0).abs() < 0.02, "{got_var} vs {var}");
        }
        let (lo, hi) = values
            .iter()
            .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        for (mu, var) in [
            (2.0 * hi, variance),
            (mean, 100.0 * variance),
            (lo, variance),
        ] {
            let (m, s, clamped) = lookup.query(mu, var);
            assert!(clamped, "({mu}, {var})");
            assert!(
                z.iter()
                    .map(|z| transform.back(m + s * z, 0.0))
                    .all(|v| (lo..=hi).contains(&v))
            );
        }
    }

    struct Field {
        grid: Vec<(f64, f64, f64)>,
        data: Vec<usize>,
        values: Vec<f64>,
    }

    fn grid() -> Vec<(f64, f64, f64)> {
        (0..2500)
            .map(|i| ((i % 50) as f64 * 2.0, (i / 50) as f64 * 2.0, 0.0))
            .collect()
    }

    /// A standard Gaussian field of Gaussian covariance (scale 10) at the
    /// nodes of [`grid`], a sum of random waves.
    fn spectral(seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let frequency = Normal::new(0.0_f64, 1.0 / 10.0).unwrap();
        let waves: Vec<(f64, f64, f64)> = (0..300)
            .map(|_| {
                (
                    frequency.sample(&mut rng),
                    frequency.sample(&mut rng),
                    rng.gen_range(0.0..std::f64::consts::TAU),
                )
            })
            .collect();
        grid()
            .iter()
            .map(|p| {
                (2.0 / waves.len() as f64).sqrt()
                    * waves
                        .iter()
                        .map(|(u, v, phase)| (u * p.0 + v * p.1 + phase).cos())
                        .sum::<f64>()
            })
            .collect()
    }

    /// A lognormal field, `exp(0.5·Y)` with `Y` a [`spectral`] field, on a
    /// 50 × 50 grid of spacing 2, sampled every fifth node.
    fn field() -> Field {
        let gaussian = spectral(11);
        let data: Vec<usize> = (0..2500)
            .filter(|i| i % 5 == 2 && i / 50 % 5 == 2)
            .collect();
        let values = data.iter().map(|&i| (0.5 * gaussian[i]).exp()).collect();
        Field {
            grid: grid(),
            data,
            values,
        }
    }

    fn search() -> SgsParams {
        SgsParams {
            search: vec![Search {
                min_samples: 1,
                max_samples: 16,
                radius: 60.0,
                ..Default::default()
            }],
            seed: 0,
        }
    }

    /// The covariance of the field: Gaussian of practical range √6 · 10.
    fn vg() -> Variogram {
        Variogram {
            nugget: 0.02,
            ..Variogram::single(Model::Gaussian, 0.98, 24.5)
        }
    }

    #[test]
    fn dss_honors_data_histogram_and_variogram() {
        let f = field();
        let locs: Vec<_> = f.data.iter().map(|&i| f.grid[i]).collect();
        let transform = Transforms::fit(&f.values, None, None, None)
            .unwrap()
            .domains
            .remove(0)
            .unwrap();
        let (mean, variance) = moments(&f.values);
        let model = |h: f64| vg().gamma(h) * variance / vg().total_sill();
        let probabilities = [0.1, 0.5, 0.9];
        let lags = [2, 5, 10];
        let n = 50;
        let (mut quantiles, mut gammas, mut simulated) = ([0.0; 3], [0.0; 3], (0.0, 0.0));
        for seed in 0..n {
            let params = SgsParams { seed, ..search() };
            let r = dss_in(
                &locs,
                &f.values,
                None,
                None,
                None,
                &f.grid,
                &vg(),
                &params,
                None,
            )
            .unwrap();
            for (&i, &v) in f.data.iter().zip(&f.values) {
                assert_eq!(r.values[i], v);
            }
            let (m, v) = moments(&r.values);
            simulated = (simulated.0 + m / n as f64, simulated.1 + v / n as f64);
            let mut sorted = r.values.clone();
            sorted.sort_by(f64::total_cmp);
            for (q, p) in quantiles.iter_mut().zip(probabilities) {
                *q += sorted[(p * sorted.len() as f64) as usize] / n as f64;
            }
            for (g, lag) in gammas.iter_mut().zip(lags) {
                let (mut sum, mut pairs) = (0.0, 0);
                for y in 0..50 {
                    for x in 0..50 - lag {
                        for (a, b) in [
                            (y * 50 + x, y * 50 + x + lag),
                            (x * 50 + y, (x + lag) * 50 + y),
                        ] {
                            sum += 0.5 * (r.values[a] - r.values[b]).powi(2);
                            pairs += 1;
                        }
                    }
                }
                *g += sum / pairs as f64 / n as f64;
            }
        }
        assert!((simulated.0 / mean - 1.0).abs() < 0.02, "{simulated:?}");
        assert!((simulated.1 / variance - 1.0).abs() < 0.05, "{simulated:?}");
        // DSS reproduces the mean and variance, the histogram only
        // approximately: the kriged means, averages of skewed data, are less
        // skewed than the data, and pull the median toward the mean (+7 %
        // here, P10 -6 %, P90 -2 %).
        for (q, p) in quantiles.iter().zip(probabilities) {
            let target = transform.back(probit(p), 0.0);
            assert!((q / target - 1.0).abs() < 0.1, "P{p}: {q} vs {target}");
        }
        for (g, lag) in gammas.iter().zip(lags) {
            let target = model(2.0 * lag as f64);
            assert!((g / target - 1.0).abs() < 0.1, "lag {lag}: {g} vs {target}");
        }
    }

    #[test]
    fn tied_data_rare_domains_and_nan_are_handled() {
        let mut values = vec![1.0; 60];
        values.extend((0..40).map(|i| 2.0 + i as f64));
        let transform = Transforms::fit(&values, None, None, None)
            .unwrap()
            .domains
            .remove(0)
            .unwrap();
        let (m, s, _) = Lookup::new(&transform).query(1.0, 0.0);
        assert!(m.is_finite() && s.is_finite());

        let f = field();
        let mut locs: Vec<_> = f.data.iter().map(|&i| f.grid[i]).collect();
        let mut vals = f.values.clone();
        locs.push((1.0, 1.0, 0.0));
        vals.push(3.0);
        let codes: Vec<u32> = (0..vals.len())
            .map(|i| u32::from(i + 1 == vals.len()))
            .collect();
        let nodes = vec![0; f.grid.len()];
        let run = |vals: &[f64]| {
            dss_in(
                &locs,
                vals,
                None,
                None,
                Some((&codes, &nodes)),
                &f.grid,
                &vg(),
                &search(),
                None,
            )
        };
        assert!(run(&vals).is_ok());
        vals[0] = f64::NAN;
        assert!(run(&vals).is_err());
    }

    #[test]
    fn realizations_follow_the_seed_not_threads() {
        let f = field();
        let locs: Vec<_> = f.data.iter().map(|&i| f.grid[i]).collect();
        let codes: Vec<u32> = locs.iter().map(|p| u32::from(p.0 > 50.0)).collect();
        let nodes: Vec<u32> = f.grid.iter().map(|p| u32::from(p.0 > 50.0)).collect();
        let weights: Vec<f64> = (0..locs.len()).map(|i| 1.0 + (i % 3) as f64).collect();
        let run = |threads| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads);
            pool.build().unwrap().install(|| {
                let r = dss_in(
                    &locs,
                    &f.values,
                    Some(&weights),
                    None,
                    Some((&codes, &nodes)),
                    &f.grid,
                    &vg(),
                    &search(),
                    None,
                )
                .unwrap();
                (r.values, r.clamped)
            })
        };
        assert_eq!(run(1), run(4));
    }

    #[test]
    fn realizations_match_their_snapshot() {
        use crate::sgs::tests::{vg, zoned, zoned_grid, zoned_search};
        let z = zoned();
        let (grid, nodes, _) = zoned_grid();
        let params = SgsParams {
            search: zoned_search(Some(estimation::Soft::All(8.0))),
            seed: 17,
        };
        let r = dss_in(
            &z.locs,
            &z.vals,
            Some(&z.weights),
            Some(&z.holes),
            Some((&z.codes, &nodes)),
            &grid,
            &vg(),
            &params,
            None,
        )
        .unwrap();
        // Sum and index-weighted sum: robust to the platform's last bits.
        let got = r
            .values
            .iter()
            .enumerate()
            .fold([0.0, 0.0], |[s, m], (i, x)| [s + x, m + (i + 1) as f64 * x]);
        let want = [6148.831983024331, 2472068.620560976];
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() <= 1e-9 * w.abs(), "{got:?}");
        }
        assert_eq!(r.clamped, 74);
    }

    fn correlation(a: &[f64], b: &[f64]) -> f64 {
        crate::sgs::pearson(a, b, None)
    }

    #[test]
    fn codss_follows_the_secondary_and_zero_correlation_is_dss() {
        let f = field();
        let locs: Vec<_> = f.data.iter().map(|&i| f.grid[i]).collect();
        // A secondary of the same covariance, correlated 0.8 with `Y`.
        let (y, other) = (spectral(11), spectral(12));
        let secondary: Vec<f64> = y
            .iter()
            .zip(&other)
            .map(|(a, b)| 10.0 + 2.0 * (0.8 * a + 0.6 * b))
            .collect();
        let at_data: Vec<f64> = f.data.iter().map(|&i| secondary[i]).collect();
        let model = vg();
        let dss = Dss::new(&locs, &f.values, None, None, None, &model).unwrap();
        let fitted = dss.secondary(&at_data, None).unwrap();
        let target = correlation(&f.values, &at_data);
        assert!((fitted.correlation - target).abs() < 1e-12);
        let n = 20;
        let mut simulated = 0.0;
        for seed in 0..n {
            let params = SgsParams { seed, ..search() };
            let r = dss
                .cosimulate(None, &f.grid, &params, None, &fitted, &secondary)
                .unwrap();
            for (&i, &v) in f.data.iter().zip(&f.values) {
                assert_eq!(r.values[i], v);
            }
            simulated += correlation(&r.values, &secondary) / n as f64;
        }
        assert!((simulated - target).abs() < 0.08, "{simulated} vs {target}");

        let zero = dss.secondary(&at_data, Some(0.0)).unwrap();
        let params = SgsParams {
            seed: 3,
            ..search()
        };
        let plain = dss.simulate(None, &f.grid, &params, None).unwrap();
        let co = dss
            .cosimulate(None, &f.grid, &params, None, &zero, &secondary)
            .unwrap();
        assert_eq!(plain.values, co.values);

        assert!(dss.secondary(&at_data[1..], None).is_err());
        assert!(dss.secondary(&at_data, Some(1.2)).is_err());
        assert!(dss.secondary(&vec![1.0; at_data.len()], None).is_err());
        let short = &secondary[1..];
        assert!(
            dss.cosimulate(None, &f.grid, &params, None, &fitted, short)
                .is_err()
        );
    }
}
