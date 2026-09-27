//! Gaussian mixture fitted by expectation-maximization, weighted, with
//! missing (NaN) entries allowed.
//!
//! The components start from a seeded k-means++ clustering, so a seed gives
//! one fit. Missing entries are integrated out: each row weighs the
//! components by the density of its present entries, and its missing entries
//! enter the moments by their conditional mean and covariance.

use crate::error::{Result, TransformError};
use crate::normal::phi;
use crate::normal_score::{Reference, invert};
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const LN_2PI: f64 = 1.837_877_066_409_345_3;

/// Fitted mixture of Gaussian components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianMixture {
    pub proportions: Vec<f64>,
    pub means: Vec<DVector<f64>>,
    pub covariances: Vec<DMatrix<f64>>,
    /// Weighted log-likelihood, weights scaled to sum to the row count.
    pub log_likelihood: f64,
    /// Bayesian information criterion, `-2 log L + parameters ln n`.
    pub bic: f64,
}

/// Conditional mean of the whole row (present entries unchanged) and
/// covariance of its missing entries, whose indices are returned too.
pub(crate) fn conditional(
    mean: &DVector<f64>,
    cov: &DMatrix<f64>,
    z: &[f64],
) -> Result<(DVector<f64>, DMatrix<f64>, Vec<usize>)> {
    let present: Vec<bool> = z.iter().map(|v| v.is_finite()).collect();
    let view = View::new(cov, &present)?;
    Ok((view.fill(mean, z), view.cov, view.m))
}

fn singular() -> TransformError {
    TransformError::FittingFailed("a component covariance is singular".into())
}

/// One component seen through one pattern of present entries.
struct View {
    o: Vec<usize>,
    m: Vec<usize>,
    /// Lower Cholesky factor of the present block.
    l: DMatrix<f64>,
    log_det: f64,
    /// Regression of the missing entries on the present ones.
    gain: DMatrix<f64>,
    /// Covariance of the missing entries given the present ones.
    cov: DMatrix<f64>,
}

impl View {
    fn new(cov: &DMatrix<f64>, present: &[bool]) -> Result<Self> {
        let (o, m): (Vec<usize>, Vec<usize>) = (0..present.len()).partition(|&v| present[v]);
        if o.is_empty() {
            let cov = cov.clone();
            let (l, gain) = (DMatrix::zeros(0, 0), DMatrix::zeros(m.len(), 0));
            return Ok(Self {
                o,
                m,
                l,
                log_det: 0.0,
                gain,
                cov,
            });
        }
        let chol = cov
            .select_rows(&o)
            .select_columns(&o)
            .cholesky()
            .ok_or_else(singular)?;
        let s_om = cov.select_rows(&o).select_columns(&m);
        let gain = chol.solve(&s_om).transpose();
        let cond = cov.select_rows(&m).select_columns(&m) - &gain * s_om;
        let l = chol.l();
        let log_det = 2.0 * l.diagonal().iter().map(|d| d.ln()).sum::<f64>();
        Ok(Self {
            o,
            m,
            l,
            log_det,
            gain,
            cov: cond,
        })
    }

    fn residual(&self, mean: &DVector<f64>, z: &[f64]) -> DVector<f64> {
        DVector::from_iterator(self.o.len(), self.o.iter().map(|&v| z[v] - mean[v]))
    }

    /// Log-density of the present entries.
    fn log_density(&self, mean: &DVector<f64>, z: &[f64]) -> f64 {
        if self.o.is_empty() {
            return 0.0;
        }
        let y = self
            .l
            .solve_lower_triangular(&self.residual(mean, z))
            .unwrap_or_else(|| DVector::from_element(self.o.len(), f64::INFINITY));
        -0.5 * (self.o.len() as f64 * LN_2PI + self.log_det + y.norm_squared())
    }

    /// The row with its missing entries at their conditional mean.
    fn fill(&self, mean: &DVector<f64>, z: &[f64]) -> DVector<f64> {
        let mut x = DVector::from_row_slice(z);
        let mu = &self.gain * self.residual(mean, z);
        for (k, &v) in self.m.iter().enumerate() {
            x[v] = mean[v] + mu[k];
        }
        x
    }
}

fn log_sum_exp(x: &[f64]) -> f64 {
    let m = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    m + x.iter().map(|v| (v - m).exp()).sum::<f64>().ln()
}

/// Seeded weighted k-means++ then Lloyd iterations on standardized rows;
/// the cluster of each row.
fn kmeans(rows: &[Vec<f64>], w: &[f64], k: usize, rng: &mut StdRng) -> Vec<usize> {
    let dist = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>();
    let pick = |p: &[f64], rng: &mut StdRng| {
        let total: f64 = p.iter().sum();
        let u = rng.gen_range(0.0..1.0) * total;
        let mut cum = 0.0;
        p.iter()
            .position(|x| {
                cum += x;
                cum > u
            })
            .unwrap_or(p.len() - 1)
    };
    let mut centers = vec![rows[pick(w, rng)].clone()];
    while centers.len() < k {
        let p: Vec<f64> = rows
            .iter()
            .zip(w)
            .map(|(r, wi)| wi * centers.iter().map(|c| dist(r, c)).fold(f64::MAX, f64::min))
            .collect();
        centers.push(rows[pick(&p, rng)].clone());
    }
    let nearest = |r: &Vec<f64>, centers: &[Vec<f64>]| {
        (0..k)
            .min_by(|&a, &b| dist(r, &centers[a]).total_cmp(&dist(r, &centers[b])))
            .unwrap()
    };
    let mut labels: Vec<usize> = rows.iter().map(|r| nearest(r, &centers)).collect();
    for _ in 0..20 {
        for (c, center) in centers.iter_mut().enumerate() {
            let mut sum = vec![0.0; rows[0].len()];
            let mut total = 0.0;
            for ((r, &l), wi) in rows.iter().zip(&labels).zip(w) {
                if l == c {
                    total += wi;
                    sum.iter_mut().zip(r).for_each(|(s, x)| *s += wi * x);
                }
            }
            if total > 0.0 {
                *center = sum.into_iter().map(|s| s / total).collect();
            }
        }
        let next: Vec<usize> = rows.iter().map(|r| nearest(r, &centers)).collect();
        if next == labels {
            break;
        }
        labels = next;
    }
    labels
}

/// One row's share of each component: log-likelihood, and per component the
/// responsibility and the row filled with its conditional mean.
type RowShare = (f64, Vec<(f64, DVector<f64>)>);

impl GaussianMixture {
    /// Fits `k` components to rows of `d` variables, NaN where missing;
    /// `weights` (e.g. declustering) weigh the rows.
    pub fn fit(data: &[Vec<f64>], weights: Option<&[f64]>, k: usize, seed: u64) -> Result<Self> {
        let dim = data.first().map_or(0, Vec::len);
        if dim == 0 || data.iter().any(|r| r.len() != dim) {
            return Err(TransformError::InvalidParameters(
                "ragged/empty rows".into(),
            ));
        }
        if data.iter().flatten().any(|v| v.is_infinite()) {
            return Err(TransformError::InvalidParameters("infinite value".into()));
        }
        if k == 0 {
            return Err(TransformError::InvalidParameters(
                "components must be ≥ 1".into(),
            ));
        }
        if let Some(w) = weights
            && (w.len() != data.len()
                || w.iter().any(|&x| !x.is_finite() || x < 0.0)
                || w.iter().sum::<f64>() <= 0.0)
        {
            return Err(TransformError::InvalidParameters(
                "weights must match rows, be ≥ 0 and sum > 0".into(),
            ));
        }
        let n = data.len() as f64;
        let total = weights.map_or(n, |w| w.iter().sum());
        let w: Vec<f64> = (0..data.len())
            .map(|i| weights.map_or(1.0, |w| w[i]) * n / total)
            .collect();
        let (mut mean, mut var) = (vec![0.0; dim], vec![0.0; dim]);
        for v in 0..dim {
            let seen: Vec<(f64, f64)> = data
                .iter()
                .zip(&w)
                .filter(|(r, _)| !r[v].is_nan())
                .map(|(r, &wi)| (r[v], wi))
                .collect();
            let sw: f64 = seen.iter().map(|s| s.1).sum();
            if seen.len() < 2 || sw <= 0.0 {
                return Err(TransformError::InsufficientData(format!(
                    "variable {v} has < 2 values"
                )));
            }
            mean[v] = seen.iter().map(|(x, wi)| x * wi).sum::<f64>() / sw;
            var[v] = seen
                .iter()
                .map(|(x, wi)| wi * (x - mean[v]).powi(2))
                .sum::<f64>()
                / sw;
            if var[v] <= 0.0 {
                return Err(TransformError::InvalidParameters(format!(
                    "variable {v} is constant"
                )));
            }
        }
        if (data.len() as f64) < (k * (dim + 1)) as f64 {
            return Err(TransformError::InsufficientData(format!(
                "{} rows are too few for {k} components",
                data.len()
            )));
        }
        let ridge =
            DMatrix::from_diagonal(&DVector::from_iterator(dim, var.iter().map(|v| 1e-6 * v)));

        // Initial moments from a hard clustering of standardized, filled rows.
        let standard: Vec<Vec<f64>> = data
            .iter()
            .map(|r| {
                (0..dim)
                    .map(|v| match r[v] {
                        x if x.is_nan() => 0.0,
                        x => (x - mean[v]) / var[v].sqrt(),
                    })
                    .collect()
            })
            .collect();
        let mut rng = StdRng::seed_from_u64(seed);
        let labels = match k {
            1 => vec![0; data.len()],
            _ => kmeans(&standard, &w, k, &mut rng),
        };
        let global = DMatrix::from_diagonal(&DVector::from_row_slice(&var));
        let mut patterns: Vec<Vec<bool>> = Vec::new();
        let pattern: Vec<usize> = data
            .iter()
            .map(|r| {
                let p: Vec<bool> = r.iter().map(|v| v.is_finite()).collect();
                patterns.iter().position(|q| *q == p).unwrap_or_else(|| {
                    patterns.push(p);
                    patterns.len() - 1
                })
            })
            .collect();
        let mut shares: Vec<RowShare> = data
            .iter()
            .zip(&labels)
            .map(|(r, &l)| {
                let filled = DVector::from_iterator(
                    dim,
                    (0..dim).map(|v| if r[v].is_nan() { mean[v] } else { r[v] }),
                );
                let mut parts = vec![(0.0, filled); k];
                parts[l].0 = 1.0;
                (0.0, parts)
            })
            .collect();
        let mut gm = Self {
            proportions: vec![1.0 / k as f64; k],
            means: vec![DVector::from_row_slice(&mean); k],
            covariances: vec![global.clone(); k],
            log_likelihood: f64::NEG_INFINITY,
            bic: f64::INFINITY,
        };
        gm.maximize(&shares, &w, None, &pattern, &ridge, &global);

        let mut previous = f64::NEG_INFINITY;
        for _ in 0..1000 {
            let views = gm.views(&patterns)?;
            shares = data
                .par_iter()
                .zip(&pattern)
                .with_min_len(256)
                .map(|(z, &p)| gm.share(z, |c| &views[c][p]))
                .collect();
            let ll: f64 = shares.iter().zip(&w).map(|(s, wi)| wi * s.0).sum();
            gm.log_likelihood = ll;
            if ll - previous <= 1e-6 * n {
                break;
            }
            previous = ll;
            gm.maximize(&shares, &w, Some(&views), &pattern, &ridge, &global);
        }
        let params = (k - 1) + k * dim + k * dim * (dim + 1) / 2;
        gm.bic = -2.0 * gm.log_likelihood + params as f64 * n.ln();
        Ok(gm)
    }

    /// Fits 1 to `max_components` components and keeps the lowest BIC; the
    /// BIC of each count is returned too.
    pub fn select(
        data: &[Vec<f64>],
        weights: Option<&[f64]>,
        max_components: usize,
        seed: u64,
    ) -> Result<(Self, Vec<f64>)> {
        let mut best: Option<Self> = None;
        let mut bics = Vec::with_capacity(max_components);
        for k in 1..=max_components {
            let gm = match Self::fit(data, weights, k, seed) {
                Ok(gm) => gm,
                Err(TransformError::InsufficientData(_) | TransformError::FittingFailed(_))
                    if k > 1 =>
                {
                    break;
                }
                Err(e) => return Err(e),
            };
            bics.push(gm.bic);
            if best.as_ref().is_none_or(|b| gm.bic < b.bic) {
                best = Some(gm);
            }
        }
        best.map(|b| (b, bics))
            .ok_or_else(|| TransformError::InvalidParameters("max_components must be ≥ 1".into()))
    }

    /// Each component through each pattern of present entries.
    fn views(&self, patterns: &[Vec<bool>]) -> Result<Vec<Vec<View>>> {
        self.covariances
            .iter()
            .map(|cov| patterns.iter().map(|p| View::new(cov, p)).collect())
            .collect()
    }

    fn share<'a>(&self, z: &[f64], view: impl Fn(usize) -> &'a View) -> RowShare {
        let logs: Vec<f64> = (0..self.proportions.len())
            .map(|c| self.proportions[c].ln() + view(c).log_density(&self.means[c], z))
            .collect();
        let ll = log_sum_exp(&logs);
        let parts = (0..logs.len())
            .map(|c| ((logs[c] - ll).exp(), view(c).fill(&self.means[c], z)))
            .collect();
        (ll, parts)
    }

    /// The share of the single row `z`.
    fn share_of(&self, z: &[f64]) -> Result<RowShare> {
        let present: Vec<bool> = z.iter().map(|v| v.is_finite()).collect();
        let views = self.views(&[present])?;
        Ok(self.share(z, |c| &views[c][0]))
    }

    fn maximize(
        &mut self,
        shares: &[RowShare],
        w: &[f64],
        views: Option<&[Vec<View>]>,
        pattern: &[usize],
        ridge: &DMatrix<f64>,
        global: &DMatrix<f64>,
    ) {
        let n: f64 = w.iter().sum();
        let dim = ridge.nrows();
        let count = pattern.iter().max().map_or(0, |p| p + 1);
        for c in 0..self.proportions.len() {
            let (mut total, mut m1, mut m2) =
                (0.0, DVector::zeros(dim), DMatrix::<f64>::zeros(dim, dim));
            let mut per_pattern = vec![0.0; count];
            for (((_, parts), wi), &p) in shares.iter().zip(w).zip(pattern) {
                let (r, x) = &parts[c];
                let f = wi * r;
                total += f;
                per_pattern[p] += f;
                m1.axpy(f, x, 1.0);
                m2.ger(f, x, x, 1.0);
            }
            if let Some(views) = views {
                for (view, f) in views[c].iter().zip(&per_pattern) {
                    for (a, &i) in view.m.iter().enumerate() {
                        for (b, &j) in view.m.iter().enumerate() {
                            m2[(i, j)] += f * view.cov[(a, b)];
                        }
                    }
                }
            }
            self.proportions[c] = total / n;
            if total <= 1e-9 * n {
                continue;
            }
            let mean = m1 / total;
            let cov = m2 / total - &mean * mean.transpose();
            self.covariances[c] = if total < (dim + 1) as f64 {
                global.clone()
            } else {
                cov + ridge
            };
            self.means[c] = mean;
        }
    }

    /// Number of variables.
    pub fn dim(&self) -> usize {
        self.means[0].len()
    }

    /// Probability of each component given the present entries of `z`.
    pub fn posterior(&self, z: &[f64]) -> Result<Vec<f64>> {
        if z.len() != self.dim() {
            return Err(TransformError::InvalidParameters(format!(
                "rows must have {} values",
                self.dim()
            )));
        }
        Ok(self.share_of(z)?.1.into_iter().map(|p| p.0).collect())
    }

    /// Mixture density at a complete row.
    pub fn pdf(&self, x: &[f64]) -> Result<f64> {
        if x.len() != self.dim() || x.iter().any(|v| !v.is_finite()) {
            return Err(TransformError::InvalidParameters(format!(
                "rows must have {} finite values",
                self.dim()
            )));
        }
        Ok(self.share_of(x)?.0.exp())
    }

    /// Mean and covariance of the whole mixture.
    pub fn moments(&self) -> (DVector<f64>, DMatrix<f64>) {
        let mut mean = DVector::zeros(self.dim());
        let mut second = DMatrix::zeros(self.dim(), self.dim());
        for ((p, m), c) in self
            .proportions
            .iter()
            .zip(&self.means)
            .zip(&self.covariances)
        {
            mean += m * *p;
            second += (c + m * m.transpose()) * *p;
        }
        let cov = second - &mean * mean.transpose();
        (mean, cov)
    }

    /// Lower Cholesky factor of each component covariance.
    pub(crate) fn factors(&self) -> Vec<DMatrix<f64>> {
        self.covariances
            .iter()
            .map(|c| {
                c.clone()
                    .cholesky()
                    .map_or_else(|| DMatrix::zeros(c.nrows(), c.nrows()), |l| l.l())
            })
            .collect()
    }

    /// A component drawn by `probabilities`.
    pub(crate) fn draw_component(probabilities: &[f64], rng: &mut StdRng) -> usize {
        let u: f64 = rng.gen_range(0.0..1.0) * probabilities.iter().sum::<f64>();
        let mut cum = 0.0;
        probabilities
            .iter()
            .position(|p| {
                cum += p;
                cum > u
            })
            .unwrap_or(probabilities.len() - 1)
    }

    /// `n` rows drawn from the mixture; the same `seed` gives the same rows.
    pub fn sample(&self, n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let factors = self.factors();
        (0..n)
            .map(|_| {
                let c = Self::draw_component(&self.proportions, &mut rng);
                let e = DVector::from_fn(self.dim(), |_, _| StandardNormal.sample(&mut rng));
                (&self.means[c] + &factors[c] * e).iter().copied().collect()
            })
            .collect()
    }

    fn cdf1(&self, x: f64) -> f64 {
        self.proportions
            .iter()
            .zip(&self.means)
            .zip(&self.covariances)
            .map(|((p, m), c)| p * phi((x - m[0]) / c[(0, 0)].sqrt()))
            .sum()
    }

    /// Probability of a value at most `x`, for one variable.
    pub fn cdf(&self, x: f64) -> Result<f64> {
        match self.dim() {
            1 => Ok(self.cdf1(x)),
            _ => Err(TransformError::InvalidParameters(
                "the CDF needs a one-variable mixture".into(),
            )),
        }
    }
}

impl Reference for GaussianMixture {
    fn quantile(&self, p: f64) -> f64 {
        let spread = self
            .covariances
            .iter()
            .map(|c| 12.0 * c[(0, 0)].sqrt())
            .fold(0.0, f64::max);
        let (lo, hi) = self.means.iter().fold((f64::MAX, f64::MIN), |(lo, hi), m| {
            (lo.min(m[0]), hi.max(m[0]))
        });
        invert(|x| self.cdf1(x), lo - spread, hi + spread, p)
    }

    fn support(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two well-separated correlated clusters, 30% and 70%.
    fn clusters(n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|i| {
                let a: f64 = StandardNormal.sample(&mut rng);
                let b: f64 = StandardNormal.sample(&mut rng);
                match i % 10 < 3 {
                    true => vec![-4.0 + a, 2.0 + 0.6 * a + 0.8 * b],
                    false => vec![3.0 + 0.5 * a, -1.0 + 0.5 * b],
                }
            })
            .collect()
    }

    #[test]
    fn recovers_known_components() {
        let data = clusters(3000, 1);
        let gm = GaussianMixture::fit(&data, None, 2, 7).unwrap();
        let first = if gm.means[0][0] < 0.0 { 0 } else { 1 };
        let (a, b) = (first, 1 - first);
        assert!((gm.proportions[a] - 0.3).abs() < 0.02);
        assert!((gm.means[a][0] + 4.0).abs() < 0.1 && (gm.means[a][1] - 2.0).abs() < 0.1);
        assert!((gm.means[b][0] - 3.0).abs() < 0.05 && (gm.means[b][1] + 1.0).abs() < 0.05);
        assert!((gm.covariances[a][(0, 1)] - 0.6).abs() < 0.1);
        assert!((gm.covariances[b][(1, 1)] - 0.25).abs() < 0.03);
        let again = GaussianMixture::fit(&data, None, 2, 7).unwrap();
        assert_eq!(gm.means, again.means);
        assert_eq!(gm.sample(100, 3), again.sample(100, 3));
    }

    #[test]
    fn weights_shift_the_proportions() {
        let data = clusters(2000, 2);
        let w: Vec<f64> = (0..2000)
            .map(|i| if i % 10 < 3 { 7.0 / 3.0 } else { 1.0 })
            .collect();
        let gm = GaussianMixture::fit(&data, Some(&w), 2, 1).unwrap();
        assert!(
            gm.proportions.iter().all(|p| (p - 0.5).abs() < 0.02),
            "{:?}",
            gm.proportions
        );
    }

    #[test]
    fn bic_picks_the_true_count() {
        let (gm, bics) = GaussianMixture::select(&clusters(1500, 3), None, 5, 0).unwrap();
        assert_eq!(gm.proportions.len(), 2, "{bics:?}");
        let mut rng = StdRng::seed_from_u64(4);
        let one: Vec<Vec<f64>> = (0..1000)
            .map(|_| vec![StandardNormal.sample(&mut rng)])
            .collect();
        assert_eq!(
            GaussianMixture::select(&one, None, 4, 0)
                .unwrap()
                .0
                .proportions
                .len(),
            1
        );
        let three: Vec<Vec<f64>> = (0..1500)
            .map(|i| {
                let e: f64 = StandardNormal.sample(&mut rng);
                vec![[-6.0, 0.0, 6.0][i % 3] + e]
            })
            .collect();
        assert_eq!(
            GaussianMixture::select(&three, None, 5, 0)
                .unwrap()
                .0
                .proportions
                .len(),
            3
        );
    }

    #[test]
    fn missing_entries_are_integrated_out() {
        let full = clusters(3000, 5);
        let holed: Vec<Vec<f64>> = full
            .iter()
            .enumerate()
            .map(|(i, r)| {
                if i % 4 == 0 {
                    vec![r[0], f64::NAN]
                } else {
                    r.clone()
                }
            })
            .collect();
        let gm = GaussianMixture::fit(&holed, None, 2, 7).unwrap();
        let a = if gm.means[0][0] < 0.0 { 0 } else { 1 };
        assert!((gm.proportions[a] - 0.3).abs() < 0.02);
        assert!((gm.means[a][1] - 2.0).abs() < 0.1);
        let p = gm.posterior(&[-4.0, f64::NAN]).unwrap();
        assert!(p[a] > 0.999);
    }

    #[test]
    fn one_variable_reference() {
        let data: Vec<Vec<f64>> = clusters(2000, 6).into_iter().map(|r| vec![r[0]]).collect();
        let gm = GaussianMixture::fit(&data, None, 2, 0).unwrap();
        for p in [0.001, 0.3, 0.5, 0.97] {
            assert!((gm.cdf(gm.quantile(p)).unwrap() - p).abs() < 1e-9);
        }
        let values: Vec<f64> = data.iter().map(|r| r[0]).collect();
        let ns = crate::normal_score::from_reference(&values, &gm);
        let mean = ns.scores.iter().sum::<f64>() / ns.scores.len() as f64;
        assert!(mean.abs() < 0.1, "{mean}");
    }
}
