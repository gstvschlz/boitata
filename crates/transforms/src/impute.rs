//! Imputation of missing variables from the present ones.
//!
//! Each variable is normal-scored on its observed values; a Gaussian, or a
//! mixture of Gaussians, is fitted to the scores by expectation-maximization
//! over all rows, missing entries included. A missing score is drawn from its
//! distribution conditional on the scores present in its row, then
//! back-transformed, so imputed values keep the histograms and the
//! correlation of the data.
//!
//! With a spatial model the draws also condition on nearby samples. The
//! scores follow an intrinsic model: every variable and pair of variables
//! shares one correlogram `ρ(h)`, scaled by the score covariance (that of
//! the component drawn for the row, with a mixture). Rows are visited along
//! a random path and their missing scores drawn by simple cokriging from the
//! scores present in the row and in the nearest samples, imputed ones
//! included, so each draw conditions the later ones.

use crate::error::{Result, TransformError};
use crate::mixture::{GaussianMixture, conditional};
use crate::normal_score::{NormalScoreTable, transform as normal_score};
use boitata_core::rng::realization_seed;
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use nalgebra::{DMatrix, DVector, Vector3};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use variogram::Variogram;

type Point = (f64, f64, f64);

/// Fitted Gaussian imputation of missing (NaN) variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianImputer {
    tables: Vec<NormalScoreTable>,
    mixture: GaussianMixture,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    spatial: Option<Spatial>,
}

/// Correlogram of the scores and the number of neighbors to condition on.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Spatial {
    variogram: Variogram,
    neighbors: usize,
}

impl Spatial {
    /// Correlation of the scores at two distinct samples.
    fn rho(&self, p: &Point, q: &Point) -> f64 {
        let v = &self.variogram;
        let h = v.lag(p, q);
        let c = if h > 0.0 {
            v.cov(h)
        } else {
            v.total_sill() - v.nugget
        };
        c / v.total_sill()
    }

    /// For each sample, its nearest `neighbors` other samples in the
    /// variogram's metric, with their correlation; uncorrelated ones are left
    /// out.
    fn near(&self, coords: &[Point]) -> Result<Vec<Vec<(usize, f64)>>> {
        let m = self
            .variogram
            .anisotropy
            .as_ref()
            .map_or_else(nalgebra::Matrix3::identity, |a| a.matrix());
        let points: Vec<[f64; 3]> = coords
            .iter()
            .map(|p| (m * Vector3::new(p.0, p.1, p.2)).into())
            .collect();
        let tree = ImmutableKdTree::<f64, 3>::new_from_slice(&points)
            .map_err(|e| TransformError::InvalidParameters(format!("{e:?}")))?;
        let k = std::num::NonZero::<usize>::MIN.saturating_add(self.neighbors);
        Ok((0..coords.len())
            .into_par_iter()
            .map(|i| {
                let mut found: Vec<(u64, usize)> = tree
                    .query(&points[i])
                    .nearest_n::<SquaredEuclidean<f64>>(k)
                    .execute()
                    .iter()
                    .map(|f| (f.distance.to_bits(), f.item as usize))
                    .filter(|&(_, j)| j != i)
                    .collect();
                found.sort_unstable();
                found
                    .into_iter()
                    .take(self.neighbors)
                    .map(|(_, j)| (j, self.rho(&coords[i], &coords[j])))
                    .filter(|&(_, r)| r > 0.0)
                    .collect()
            })
            .collect())
    }
}

impl GaussianImputer {
    /// Fits on `n` rows of `d` variables, NaN where missing; `weights`
    /// (e.g. declustering) shape the scores and the covariance.
    pub fn fit(data: &[Vec<f64>], weights: Option<&[f64]>) -> Result<Self> {
        Self::fit_mixture(data, weights, Some(1), 0)
    }

    /// As [`GaussianImputer::fit`], with `components` Gaussians in the
    /// scores, chosen by BIC up to 6 when `None`; `seed` starts the mixture.
    pub fn fit_mixture(
        data: &[Vec<f64>],
        weights: Option<&[f64]>,
        components: Option<usize>,
        seed: u64,
    ) -> Result<Self> {
        let dim = data.first().map_or(0, Vec::len);
        if dim == 0 || data.iter().any(|r| r.len() != dim) {
            return Err(TransformError::InvalidParameters(
                "ragged/empty rows".into(),
            ));
        }
        if data.iter().flatten().any(|v| v.is_infinite()) {
            return Err(TransformError::InvalidParameters("infinite value".into()));
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
        let w = |i: usize| weights.map_or(1.0, |w| w[i]);
        let mut scores = vec![vec![f64::NAN; dim]; data.len()];
        let mut tables = Vec::with_capacity(dim);
        for v in 0..dim {
            let seen: Vec<usize> = (0..data.len()).filter(|&i| !data[i][v].is_nan()).collect();
            if seen.len() < 2 {
                return Err(TransformError::InsufficientData(format!(
                    "variable {v} has < 2 values"
                )));
            }
            let values: Vec<f64> = seen.iter().map(|&i| data[i][v]).collect();
            let ws: Vec<f64> = seen.iter().map(|&i| w(i)).collect();
            let table = normal_score(&values, Some(&ws))?.table;
            for &i in &seen {
                scores[i][v] = table.forward(data[i][v]);
            }
            tables.push(table);
        }
        let mixture = match components {
            Some(k) => GaussianMixture::fit(&scores, weights, k, seed)?,
            None => GaussianMixture::select(&scores, weights, 6, seed)?.0,
        };
        Ok(Self {
            tables,
            mixture,
            spatial: None,
        })
    }

    /// Draws also condition on the `neighbors` nearest samples, whose scores
    /// correlate by `variogram` scaled to a unit sill.
    pub fn with_spatial(mut self, variogram: Variogram, neighbors: usize) -> Result<Self> {
        let sill = variogram.total_sill();
        if !variogram.is_stationary() || !sill.is_finite() || sill <= 0.0 {
            return Err(TransformError::InvalidParameters(
                "the spatial variogram needs a finite, positive sill".into(),
            ));
        }
        self.spatial = Some(Spatial {
            variogram,
            neighbors,
        });
        Ok(self)
    }

    /// Number of variables.
    pub fn dim(&self) -> usize {
        self.tables.len()
    }

    /// The mixture fitted to the normal scores.
    pub fn mixture(&self) -> &GaussianMixture {
        &self.mixture
    }

    /// Correlation matrix of the normal scores.
    pub fn correlation(&self) -> DMatrix<f64> {
        let cov = self.mixture.moments().1;
        let s = cov.diagonal().map(f64::sqrt);
        cov.component_div(&(&s * s.transpose()))
    }

    /// Whether draws condition on nearby samples.
    pub fn is_spatial(&self) -> bool {
        self.spatial.is_some()
    }

    /// `data` with every NaN replaced by a draw conditional on the values
    /// present in its row and, when spatial, on the nearest samples at
    /// `coords`; the same `seed` gives the same draws. A spatial model with no
    /// correlation between samples gives the draws of the non-spatial one.
    pub fn impute(
        &self,
        data: &[Vec<f64>],
        coords: Option<&[Point]>,
        seed: u64,
    ) -> Result<Vec<Vec<f64>>> {
        let dim = self.dim();
        if data.iter().any(|r| r.len() != dim) {
            return Err(TransformError::InvalidParameters(format!(
                "rows must have {dim} values"
            )));
        }
        let near = match (&self.spatial, coords) {
            (None, _) => vec![vec![]; data.len()],
            (Some(_), None) => {
                return Err(TransformError::InvalidParameters(
                    "a spatial imputer needs coords".into(),
                ));
            }
            (Some(spatial), Some(coords)) => {
                if coords.len() != data.len() {
                    return Err(TransformError::InvalidParameters(
                        "coords must have one row per data row".into(),
                    ));
                }
                if coords
                    .iter()
                    .any(|p| !(p.0.is_finite() && p.1.is_finite() && p.2.is_finite()))
                {
                    return Err(TransformError::InvalidParameters(
                        "coords must be finite".into(),
                    ));
                }
                spatial.near(coords)?
            }
        };
        let mut z: Vec<Vec<f64>> = data
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&self.tables)
                    .map(|(&x, t)| if x.is_nan() { x } else { t.forward(x) })
                    .collect()
            })
            .collect();
        let gm = &self.mixture;
        let mut rng = StdRng::seed_from_u64(realization_seed(seed, 0));
        let mut draws = vec![None; z.len()];
        for (i, r) in z.iter().enumerate() {
            let m = r.iter().filter(|v| v.is_nan()).count();
            if m == 0 {
                continue;
            }
            let c = match gm.proportions.len() {
                1 => 0,
                _ => GaussianMixture::draw_component(&gm.posterior(r)?, &mut rng),
            };
            let e: Vec<f64> = (0..m).map(|_| StandardNormal.sample(&mut rng)).collect();
            draws[i] = Some((c, e));
        }
        let mut path: Vec<usize> = (0..z.len()).filter(|&i| draws[i].is_some()).collect();
        path.shuffle(&mut StdRng::seed_from_u64(realization_seed(seed, 1)));
        for i in path {
            let Some((c, e)) = &draws[i] else { continue };
            let (mean, cov) = (&gm.means[*c], &gm.covariances[*c]);
            let (mu, c, missing) = match (&self.spatial, coords) {
                (Some(spatial), Some(coords)) if !near[i].is_empty() => {
                    cokriging(mean, cov, spatial, coords, &z, &near[i], i)?
                }
                _ => conditional(mean, cov, &z[i])?,
            };
            let jitter = DMatrix::identity(missing.len(), missing.len()) * 1e-12;
            let l = (c + jitter)
                .cholesky()
                .map_or_else(|| DMatrix::zeros(missing.len(), missing.len()), |ch| ch.l());
            let draw = l * DVector::from_row_slice(e);
            for (k, &v) in missing.iter().enumerate() {
                z[i][v] = mu[v] + draw[k];
            }
        }
        Ok(data
            .iter()
            .zip(&z)
            .map(|(row, s)| {
                row.iter()
                    .zip(s)
                    .zip(&self.tables)
                    .map(|((&x, &y), t)| if x.is_nan() { t.back(y) } else { x })
                    .collect()
            })
            .collect())
    }
}

/// Conditional mean of row `i` and covariance of its missing scores, given
/// its present scores and those of its neighbors `(row, ρ)`, under the
/// intrinsic model of `mean` and `cov`.
fn cokriging(
    mean: &DVector<f64>,
    cov: &DMatrix<f64>,
    spatial: &Spatial,
    coords: &[Point],
    z: &[Vec<f64>],
    near: &[(usize, f64)],
    i: usize,
) -> Result<(DVector<f64>, DMatrix<f64>, Vec<usize>)> {
    let dim = mean.len();
    let mut entries: Vec<(Option<usize>, usize)> = (0..dim).map(|v| (None, v)).collect();
    for (k, &(j, _)) in near.iter().enumerate() {
        entries.extend(
            (0..dim)
                .filter(|&v| !z[j][v].is_nan())
                .map(|v| (Some(k), v)),
        );
    }
    let rho = |a: Option<usize>, b: Option<usize>| match (a, b) {
        (None, Some(k)) | (Some(k), None) => near[k].1,
        (Some(a), Some(b)) if a != b => spatial.rho(&coords[near[a].0], &coords[near[b].0]),
        _ => 1.0,
    };
    let n = entries.len();
    let mut big = DMatrix::from_fn(n, n, |a, b| {
        let ((ka, p), (kb, q)) = (entries[a], entries[b]);
        cov[(p, q)] * rho(ka, kb)
    });
    for a in dim..n {
        big[(a, a)] *= 1.0 + 1e-10;
    }
    let means = DVector::from_iterator(n, entries.iter().map(|&(_, v)| mean[v]));
    let values: Vec<f64> = entries
        .iter()
        .map(|&(k, v)| z[k.map_or(i, |k| near[k].0)][v])
        .collect();
    let (mu, c, missing) = conditional(&means, &big, &values)?;
    Ok((mu.rows(0, dim).into_owned(), c, missing))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corr(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
        for (x, y) in a.iter().zip(b) {
            c += (x - ma) * (y - mb);
            va += (x - ma).powi(2);
            vb += (y - mb).powi(2);
        }
        c / (va * vb).sqrt()
    }

    /// Lognormal pair with log correlation 0.8; the second variable is
    /// missing in 40% of rows, the first in 10% of the others.
    fn samples() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut rng = StdRng::seed_from_u64(5);
        let full: Vec<Vec<f64>> = (0..3000)
            .map(|_| {
                let a: f64 = StandardNormal.sample(&mut rng);
                let b: f64 = StandardNormal.sample(&mut rng);
                vec![a.exp(), (0.8 * a + 0.6 * b).exp()]
            })
            .collect();
        let holed = full
            .iter()
            .enumerate()
            .map(|(i, r)| match i % 10 {
                0..=3 => vec![r[0], f64::NAN],
                4 => vec![f64::NAN, r[1]],
                _ => r.clone(),
            })
            .collect();
        (full, holed)
    }

    #[test]
    fn keeps_observed_values_and_reproduces_correlation() {
        let (_, holed) = samples();
        let imputer = GaussianImputer::fit(&holed, None).unwrap();
        assert!((imputer.correlation()[(0, 1)] - 0.8).abs() < 0.03);
        let out = imputer.impute(&holed, None, 3).unwrap();
        assert_eq!(out, imputer.impute(&holed, None, 3).unwrap());
        for (r, h) in out.iter().zip(&holed) {
            for (x, y) in r.iter().zip(h) {
                assert!(x.is_finite());
                if !y.is_nan() {
                    assert_eq!(x.to_bits(), y.to_bits());
                }
            }
        }
        let imputed: Vec<&Vec<f64>> = out
            .iter()
            .zip(&holed)
            .filter(|(_, h)| h.iter().any(|v| v.is_nan()))
            .map(|(r, _)| r)
            .collect();
        for rows in [out.iter().collect::<Vec<_>>(), imputed] {
            let a: Vec<f64> = rows.iter().map(|r| r[0].ln()).collect();
            let b: Vec<f64> = rows.iter().map(|r| r[1].ln()).collect();
            let r = corr(&a, &b);
            assert!((r - 0.8).abs() < 0.05, "correlation {r}");
            let n = b.len() as f64;
            let var =
                b.iter().map(|x| x * x).sum::<f64>() / n - (b.iter().sum::<f64>() / n).powi(2);
            assert!((var - 1.0).abs() < 0.15, "variance {var}");
        }
    }

    #[test]
    fn ties_add_no_correlation() {
        let mut rng = StdRng::seed_from_u64(9);
        let mut detect = || {
            let x: f64 = StandardNormal.sample(&mut rng);
            x.max(0.0)
        };
        let data: Vec<Vec<f64>> = (0..2000).map(|_| vec![detect(), detect()]).collect();
        let r = GaussianImputer::fit(&data, None).unwrap().correlation()[(0, 1)];
        assert!(r.abs() < 0.05, "correlation {r}");
    }

    #[test]
    fn rejects_bad_input() {
        assert!(GaussianImputer::fit(&[vec![1.0, f64::NAN], vec![2.0, f64::NAN]], None).is_err());
        assert!(GaussianImputer::fit(&[vec![1.0, f64::INFINITY]], None).is_err());
        let (_, holed) = samples();
        let imputer = GaussianImputer::fit(&holed, None).unwrap();
        assert!(imputer.impute(&[vec![1.0]], None, 0).is_err());
    }

    /// Two groups in an L: one high in the first variable only, one high in
    /// the second only; a single Gaussian fills the empty corner.
    #[test]
    fn a_mixture_keeps_imputed_rows_on_the_l() {
        let mut rng = StdRng::seed_from_u64(8);
        let mut normal = || -> f64 { StandardNormal.sample(&mut rng) };
        let full: Vec<Vec<f64>> = (0..3000)
            .map(|i| match i % 2 {
                0 => vec![(2.0 + 0.8 * normal()).exp(), (-1.0 + 0.3 * normal()).exp()],
                _ => vec![(-1.0 + 0.3 * normal()).exp(), (2.0 + 0.8 * normal()).exp()],
            })
            .collect();
        let holed: Vec<Vec<f64>> = full
            .iter()
            .enumerate()
            .map(|(i, r)| {
                if i % 3 == 0 {
                    vec![r[0], f64::NAN]
                } else {
                    r.clone()
                }
            })
            .collect();
        let corner = |rows: &[Vec<f64>]| {
            let hidden = rows.iter().step_by(3);
            hidden.filter(|r| r[0] > 1.5 && r[1] > 1.5).count() as f64 / 1000.0
        };
        let truth = corner(&full);
        let one = GaussianImputer::fit(&holed, None).unwrap();
        let two = GaussianImputer::fit_mixture(&holed, None, None, 0).unwrap();
        assert!(two.mixture().proportions.len() >= 2);
        let (one, two) = (
            corner(&one.impute(&holed, None, 1).unwrap()),
            corner(&two.impute(&holed, None, 1).unwrap()),
        );
        assert!(
            truth < 0.01 && two < 0.4 * one && one > 0.1,
            "{truth} {two} {one}"
        );
    }

    /// Two Gaussian fields with correlation 0.7 over 600 scattered points,
    /// both spherical with range 25; the second is hidden in 60% of rows.
    fn field() -> (Vec<Point>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        use rand::Rng;
        let mut rng = StdRng::seed_from_u64(11);
        let coords: Vec<Point> = (0..600)
            .map(|_| (rng.gen_range(0.0..100.0), rng.gen_range(0.0..100.0), 0.0))
            .collect();
        let v = Variogram::single(variogram::Model::Spherical, 1.0, 25.0);
        let n = coords.len();
        let c = DMatrix::from_fn(n, n, |a, b| {
            v.cov_points(&coords[a], &coords[b]) + if a == b { 1e-9 } else { 0.0 }
        });
        let l = c.cholesky().unwrap().l();
        let mut normal = || DVector::from_fn(n, |_, _| StandardNormal.sample(&mut rng));
        let (y1, y2) = (&l * normal(), &l * normal());
        let full: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![y1[i], 0.7 * y1[i] + 0.51f64.sqrt() * y2[i]])
            .collect();
        let holed = full
            .iter()
            .enumerate()
            .map(|(i, r)| {
                if i % 5 < 3 {
                    vec![r[0], f64::NAN]
                } else {
                    r.clone()
                }
            })
            .collect();
        (coords, full, holed)
    }

    fn spherical() -> Variogram {
        Variogram::single(variogram::Model::Spherical, 1.0, 25.0)
    }

    #[test]
    fn pure_nugget_draws_match_the_non_spatial_imputer() {
        let (_, holed) = samples();
        let coords: Vec<Point> = (0..holed.len())
            .map(|i| ((i % 60) as f64, (i / 60) as f64, 0.0))
            .collect();
        let plain = GaussianImputer::fit(&holed, None).unwrap();
        let nugget = Variogram {
            nugget: 1.0,
            structures: vec![],
            anisotropy: None,
        };
        let spatial = plain.clone().with_spatial(nugget, 16).unwrap();
        let a = plain.impute(&holed, None, 4).unwrap();
        let b = spatial.impute(&holed, Some(&coords), 4).unwrap();
        assert_eq!(a, b);
        let two = GaussianImputer::fit_mixture(&holed, None, Some(2), 0).unwrap();
        let a = two.impute(&holed, None, 4).unwrap();
        let spatial = two
            .with_spatial(spatial.spatial.unwrap().variogram, 16)
            .unwrap();
        assert_eq!(a, spatial.impute(&holed, Some(&coords), 4).unwrap());
    }

    #[test]
    fn neighbors_cut_the_error_and_keep_the_correlation() {
        let (coords, full, holed) = field();
        let plain = GaussianImputer::fit(&holed, None).unwrap();
        let spatial = plain.clone().with_spatial(spherical(), 16).unwrap();
        let hidden: Vec<usize> = (0..holed.len()).filter(|&i| holed[i][1].is_nan()).collect();
        let (mut err_plain, mut err_spatial, mut r) = (0.0, 0.0, 0.0);
        for seed in 0..10 {
            let p = plain.impute(&holed, None, seed).unwrap();
            let s = spatial.impute(&holed, Some(&coords), seed).unwrap();
            for (out, sum) in [(&p, &mut err_plain), (&s, &mut err_spatial)] {
                *sum += hidden
                    .iter()
                    .map(|&i| (out[i][1] - full[i][1]).powi(2))
                    .sum::<f64>();
            }
            for (x, h) in s.iter().zip(&holed) {
                for (a, b) in x.iter().zip(h) {
                    if !b.is_nan() {
                        assert_eq!(a.to_bits(), b.to_bits());
                    }
                }
            }
            let a: Vec<f64> = hidden.iter().map(|&i| s[i][0]).collect();
            let b: Vec<f64> = hidden.iter().map(|&i| s[i][1]).collect();
            r += corr(&a, &b) / 10.0;
        }
        assert!(
            err_spatial < 0.6 * err_plain,
            "{err_spatial} vs {err_plain}"
        );
        let target = spatial.correlation()[(0, 1)];
        assert!((r - target).abs() < 0.05, "correlation {r} vs {target}");
    }

    #[test]
    fn spatial_draws_ignore_the_thread_count() {
        let (coords, _, holed) = field();
        let imputer = GaussianImputer::fit(&holed, None)
            .unwrap()
            .with_spatial(spherical(), 12)
            .unwrap();
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| imputer.impute(&holed, Some(&coords), 7).unwrap())
        };
        assert_eq!(run(1), run(4));
        assert!(imputer.impute(&holed, None, 7).is_err());
        assert!(imputer.impute(&holed, Some(&coords[1..]), 7).is_err());
        let power = Variogram::single(variogram::Model::Power { exponent: 1.0 }, 1.0, 1.0);
        assert!(imputer.clone().with_spatial(power, 8).is_err());
    }
}
