//! Drift-aware kriging algebra: SK / OK / UK / KED, plus dual, Bayesian and
//! factorial variants.
//!
//! This is the reusable linear-algebra core the monolithic [`crate::krige`]
//! never had. Instead of hard-coding a single `1`s unbiasedness row, the system
//! is assembled from a **drift matrix** `X`:
//!
//! ```text
//! ┌         ┐ ┌   ┐   ┌    ┐
//! │  C   X  │ │ λ │ = │ c₀ │        estimate  ẑ = Σ λᵢ zᵢ   (+ known mean for SK)
//! │  Xᵀ  0  │ │ μ │   │ x₀ │        variance  σ² = C(0) − Σ λᵢ c₀ᵢ − Σ μₗ x₀ₗ
//! └         ┘ └   ┘   └    ┘
//! ```
//!
//! The number and content of the drift columns select the estimator:
//! - **SK** — no drift columns (`p = 0`), a known mean is subtracted;
//! - **OK** — one constant column;
//! - **UK** — polynomial monomials of the coordinates;
//! - **KED** — polynomial columns plus external-drift variables.
//!
//! All variants share one assembly + one dense LU solve, so adding a new drift
//! is a data change, not a new code path.

use crate::Sample;
use crate::error::{EstimError, Result};
use crate::krige::Estimate;
use nalgebra::{DMatrix, DVector};
use variogram::Variogram;

/// Drift design for a single estimate: the basis evaluated at every datum
/// (`data`, `n × p`) and at the target (`target`, length `p`).
#[derive(Debug, Clone)]
pub struct DriftSpec {
    pub data: Vec<Vec<f64>>,
    pub target: Vec<f64>,
}

impl DriftSpec {
    /// No drift columns → simple kriging (a known mean is used).
    pub fn simple(n: usize) -> Self {
        Self {
            data: vec![vec![]; n],
            target: vec![],
        }
    }

    /// One constant column → ordinary kriging (`Σλ = 1`).
    pub fn ordinary(n: usize) -> Self {
        Self {
            data: vec![vec![1.0]; n],
            target: vec![1.0],
        }
    }

    /// Polynomial drift (universal kriging) of total `degree`.
    ///
    /// Monomials are built in coordinates centered on the data centroid and
    /// scaled per axis — the span of the basis is translation/scale invariant,
    /// so this leaves the estimator unchanged while keeping the saddle-point
    /// system well conditioned even for far targets.
    ///
    /// `degree` 0 = constant (≡ OK), 1 = linear, 2 = quadratic.
    pub fn polynomial(coords: &[(f64, f64, f64)], target: &(f64, f64, f64), degree: usize) -> Self {
        let basis = PolyBasis::new(coords, degree);
        let data = coords.iter().map(|c| basis.eval(c)).collect();
        let tgt = basis.eval(target);
        Self { data, target: tgt }
    }

    /// Append external-drift columns (KED): one extra column per external
    /// variable, its value at each datum and at the target.
    pub fn with_external(mut self, data_ext: &[Vec<f64>], target_ext: &[f64]) -> Self {
        for (row, ext) in self.data.iter_mut().zip(data_ext) {
            row.extend_from_slice(ext);
        }
        self.target.extend_from_slice(target_ext);
        self
    }

    fn p(&self) -> usize {
        self.target.len()
    }
}

/// A polynomial monomial basis in coordinates centered on `origin` (the data
/// centroid) and scaled per axis, so values stay `O(1)`. Axes with (near) zero
/// spread are inactive and contribute no monomials — this keeps the drift matrix
/// full-rank for 2-D data (`z` constant) or single lines.
#[derive(Debug, Clone)]
struct PolyBasis {
    origin: (f64, f64, f64),
    scale: (f64, f64, f64),
    active: [bool; 3],
    degree: usize,
}

impl PolyBasis {
    fn new(coords: &[(f64, f64, f64)], degree: usize) -> Self {
        let n = coords.len().max(1) as f64;
        let origin = (
            coords.iter().map(|c| c.0).sum::<f64>() / n,
            coords.iter().map(|c| c.1).sum::<f64>() / n,
            coords.iter().map(|c| c.2).sum::<f64>() / n,
        );
        let raw = |sel: fn(&(f64, f64, f64)) -> f64, o: f64| {
            (coords.iter().map(|c| (sel(c) - o).powi(2)).sum::<f64>() / n).sqrt()
        };
        let sx = raw(|c| c.0, origin.0);
        let sy = raw(|c| c.1, origin.1);
        let sz = raw(|c| c.2, origin.2);
        let active = [sx > 1e-9, sy > 1e-9, sz > 1e-9];
        let scale = (
            if active[0] { sx } else { 1.0 },
            if active[1] { sy } else { 1.0 },
            if active[2] { sz } else { 1.0 },
        );
        Self {
            origin,
            scale,
            active,
            degree,
        }
    }

    fn eval(&self, p: &(f64, f64, f64)) -> Vec<f64> {
        let c = [
            (p.0 - self.origin.0) / self.scale.0,
            (p.1 - self.origin.1) / self.scale.1,
            (p.2 - self.origin.2) / self.scale.2,
        ];
        let mut m = vec![1.0];
        let axes: Vec<usize> = (0..3).filter(|&a| self.active[a]).collect();
        if self.degree >= 1 {
            for &a in &axes {
                m.push(c[a]);
            }
        }
        if self.degree >= 2 {
            for (i, &a) in axes.iter().enumerate() {
                for &b in &axes[i..] {
                    m.push(c[a] * c[b]);
                }
            }
        }
        m
    }
}

/// Universal / external-drift kriging (also SK when `drift` has no columns).
///
/// `mean` is required and used only in the simple-kriging case (`p = 0`).
pub fn krige_universal(
    target: &(f64, f64, f64),
    samples: &[Sample],
    drift: &DriftSpec,
    vg: &Variogram,
    mean: Option<f64>,
) -> Result<Estimate> {
    let n = samples.len();
    if n == 0 {
        return Err(EstimError::InsufficientData("no samples".into()));
    }
    if drift.data.len() != n {
        return Err(EstimError::InvalidParameters(
            "drift/sample length mismatch".into(),
        ));
    }
    let p = drift.p();
    let c0 = vg.total_sill();

    if p == 0 {
        let m = mean.ok_or_else(|| {
            EstimError::InvalidParameters("simple kriging needs a known mean".into())
        })?;
        let mut a = DMatrix::<f64>::zeros(n, n);
        let mut b = DVector::<f64>::zeros(n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = vg.cov_points(&samples[i].loc, &samples[j].loc);
            }
            a[(i, i)] += samples[i].error_variance;
            b[i] = vg.cov_points(&samples[i].loc, target);
        }
        let x = a
            .lu()
            .solve(&b)
            .ok_or_else(|| EstimError::Singular("SK matrix not invertible".into()))?;
        let weights: Vec<f64> = x.as_slice().to_vec();
        let value = m
            + (0..n)
                .map(|i| weights[i] * (samples[i].value - m))
                .sum::<f64>();
        let sum_wc: f64 = (0..n).map(|i| weights[i] * b[i]).sum();
        return Ok(Estimate {
            value,
            variance: (c0 - sum_wc).max(0.0),
            n_used: n,
            weights,
            lagrange: 0.0,
            support_variance: c0,
        });
    }

    let dim = n + p;
    let mut a = DMatrix::<f64>::zeros(dim, dim);
    let mut b = DVector::<f64>::zeros(dim);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = vg.cov_points(&samples[i].loc, &samples[j].loc);
        }
        a[(i, i)] += samples[i].error_variance;
        for l in 0..p {
            a[(i, n + l)] = drift.data[i][l];
            a[(n + l, i)] = drift.data[i][l];
        }
        b[i] = vg.cov_points(&samples[i].loc, target);
    }
    for l in 0..p {
        b[n + l] = drift.target[l];
    }

    let x = a
        .lu()
        .solve(&b)
        .ok_or_else(|| EstimError::Singular("UK/KED matrix not invertible".into()))?;
    let weights: Vec<f64> = x.as_slice()[..n].to_vec();
    let value: f64 = (0..n).map(|i| weights[i] * samples[i].value).sum();
    let sum_wc: f64 = (0..n).map(|i| weights[i] * b[i]).sum();
    let sum_mu: f64 = (0..p).map(|l| x[n + l] * b[n + l]).sum();
    let variance = (c0 - sum_wc - sum_mu).max(0.0);
    Ok(Estimate {
        value,
        variance,
        n_used: n,
        weights,
        lagrange: sum_mu,
        support_variance: c0,
    })
}

/// Ordinary kriging via the drift algebra (`Σλ = 1`).
pub fn krige_ordinary(
    target: &(f64, f64, f64),
    samples: &[Sample],
    vg: &Variogram,
) -> Result<Estimate> {
    krige_universal(
        target,
        samples,
        &DriftSpec::ordinary(samples.len()),
        vg,
        None,
    )
}

/// Dual kriging: factorize the (drift-aware) system once, then evaluate at many
/// targets in `O(n)` per target. Numerically identical to per-target kriging.
///
/// Build with [`DualKriging::new`], then call [`DualKriging::estimate`].
pub struct DualKriging<'a> {
    samples: &'a [Sample],
    vg: &'a Variogram,
    /// Dual data coefficients `d` (length `n`).
    d: Vec<f64>,
    /// Dual drift coefficients `e` (length `p`).
    e: Vec<f64>,
    basis: PolyBasis,
}

impl<'a> DualKriging<'a> {
    /// Build a dual system with polynomial drift of `degree` (0 = OK).
    pub fn new(samples: &'a [Sample], vg: &'a Variogram, degree: usize) -> Result<Self> {
        let n = samples.len();
        if n == 0 {
            return Err(EstimError::InsufficientData("no samples".into()));
        }
        let coords: Vec<(f64, f64, f64)> = samples.iter().map(|s| s.loc).collect();
        let basis = PolyBasis::new(&coords, degree);
        let fdata: Vec<Vec<f64>> = coords.iter().map(|c| basis.eval(c)).collect();
        let p = fdata[0].len();
        let dim = n + p;
        let mut a = DMatrix::<f64>::zeros(dim, dim);
        let mut rhs = DVector::<f64>::zeros(dim);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = vg.cov_points(&samples[i].loc, &samples[j].loc);
            }
            a[(i, i)] += samples[i].error_variance;
            for l in 0..p {
                a[(i, n + l)] = fdata[i][l];
                a[(n + l, i)] = fdata[i][l];
            }
            rhs[i] = samples[i].value;
        }
        let x = a
            .lu()
            .solve(&rhs)
            .ok_or_else(|| EstimError::Singular("dual system not invertible".into()))?;
        Ok(Self {
            samples,
            vg,
            d: x.as_slice()[..n].to_vec(),
            e: x.as_slice()[n..].to_vec(),
            basis,
        })
    }

    /// Estimate at `target`: `ẑ = Σ dᵢ C(xᵢ, target) + Σ eₗ fₗ(target)`.
    pub fn estimate(&self, target: &(f64, f64, f64)) -> f64 {
        let cov: f64 = self
            .samples
            .iter()
            .zip(&self.d)
            .map(|(s, d)| d * self.vg.cov_points(&s.loc, target))
            .sum();
        let f = self.basis.eval(target);
        let drift: f64 = f.iter().zip(&self.e).map(|(f, e)| f * e).sum();
        cov + drift
    }
}

/// Bayesian kriging with a Gaussian prior `β ~ N(β₀, S₀)` on the drift
/// coefficients. As `S₀ → ∞` this tends to UK; as `S₀ → 0` it tends to SK with
/// the fixed drift `β₀`.
///
/// The predictor blends the SK-with-prior-drift solution and the data via the
/// posterior of `β`. `drift` supplies the basis; `prior_mean`/`prior_var` give a
/// diagonal prior (length `p`).
pub fn krige_bayesian(
    target: &(f64, f64, f64),
    samples: &[Sample],
    drift: &DriftSpec,
    vg: &Variogram,
    prior_mean: &[f64],
    prior_var: &[f64],
) -> Result<Estimate> {
    let n = samples.len();
    let p = drift.p();
    if n == 0 {
        return Err(EstimError::InsufficientData("no samples".into()));
    }
    if p == 0 || prior_mean.len() != p || prior_var.len() != p {
        return Err(EstimError::InvalidParameters(
            "Bayesian kriging needs a prior of the drift dimension".into(),
        ));
    }

    // Covariance C (n×n) and drift F (n×p).
    let mut c = DMatrix::<f64>::zeros(n, n);
    let mut cvec = DVector::<f64>::zeros(n);
    let f = DMatrix::from_fn(n, p, |i, l| drift.data[i][l]);
    let z = DVector::from_fn(n, |i, _| samples[i].value);
    for i in 0..n {
        for j in 0..n {
            c[(i, j)] = vg.cov_points(&samples[i].loc, &samples[j].loc);
        }
        c[(i, i)] += samples[i].error_variance;
        cvec[i] = vg.cov_points(&samples[i].loc, target);
    }
    let f0 = DVector::from_row_slice(&drift.target);
    let b0 = DVector::from_row_slice(prior_mean);
    let s0_inv =
        DMatrix::from_diagonal(&DVector::from_fn(p, |l, _| 1.0 / prior_var[l].max(1e-300)));

    let c_lu = c.clone().lu();
    let c_inv = c_lu
        .try_inverse()
        .ok_or_else(|| EstimError::Singular("Bayesian C not invertible".into()))?;

    // Posterior of β: precision = S₀⁻¹ + Fᵀ C⁻¹ F ; mean solves
    // precision·β̂ = S₀⁻¹ β₀ + Fᵀ C⁻¹ z.
    let ftci = f.transpose() * &c_inv;
    let precision = &s0_inv + &ftci * &f;
    let rhs = &s0_inv * &b0 + &ftci * &z;
    let prec_lu = precision.clone().lu();
    let beta = prec_lu
        .solve(&rhs)
        .ok_or_else(|| EstimError::Singular("Bayesian posterior singular".into()))?;

    // Estimate: drift mean at target + SK of residuals z − Fβ̂.
    let resid = &z - &f * &beta;
    let lambda = c_lu
        .solve(&cvec)
        .ok_or_else(|| EstimError::Singular("Bayesian residual solve failed".into()))?;
    let drift_mean: f64 = f0.dot(&beta);
    let value = drift_mean + lambda.dot(&resid);

    // Variance: SK variance + drift-uncertainty term (f₀ − Fᵀλ)ᵀ Σ_β (f₀ − Fᵀλ).
    let sig_beta = precision
        .lu()
        .try_inverse()
        .ok_or_else(|| EstimError::Singular("posterior covariance singular".into()))?;
    let sk_var = vg.total_sill() - cvec.dot(&lambda);
    let g = &f0 - f.transpose() * &lambda;
    let drift_var = (g.transpose() * &sig_beta * &g)[(0, 0)];
    let variance = (sk_var + drift_var).max(0.0);

    Ok(Estimate {
        value,
        variance,
        n_used: n,
        weights: lambda.as_slice().to_vec(),
        lagrange: f64::NAN,
        support_variance: f64::NAN,
    })
}

/// Factorial kriging (structure filtering): estimate only the contribution of a
/// chosen subset of variogram structures.
///
/// The left-hand covariance uses the *full* variogram, but the right-hand side
/// (target coupling) keeps only the selected structures — so the estimate is the
/// filtered component (e.g. drop `keep`ing the nugget to denoise, or keep a
/// single long-range structure to extract regional trend).
///
/// `keep_nugget` and `keep_structures` (indices into `vg.structures`) select the
/// component. Ordinary form (one unbiasedness constraint).
pub fn krige_factorial(
    target: &(f64, f64, f64),
    samples: &[Sample],
    vg: &Variogram,
    keep_nugget: bool,
    keep_structures: &[usize],
) -> Result<Estimate> {
    let n = samples.len();
    if n == 0 {
        return Err(EstimError::InsufficientData("no samples".into()));
    }
    let dim = n + 1;
    let mut a = DMatrix::<f64>::zeros(dim, dim);
    let mut b = DVector::<f64>::zeros(dim);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = vg.cov_points(&samples[i].loc, &samples[j].loc);
        }
        a[(i, i)] += samples[i].error_variance;
        a[(i, n)] = 1.0;
        a[(n, i)] = 1.0;
        // Filtered RHS: covariance of only the kept structures.
        b[i] = partial_cov(vg, &samples[i].loc, target, keep_nugget, keep_structures);
    }
    // Filtering estimates a zero-mean component → no unbiasedness on the mean.
    b[n] = 0.0;
    let x = a
        .lu()
        .solve(&b)
        .ok_or_else(|| EstimError::Singular("factorial matrix not invertible".into()))?;
    let weights: Vec<f64> = x.as_slice()[..n].to_vec();
    let value: f64 = (0..n).map(|i| weights[i] * samples[i].value).sum();
    let sum_wc: f64 = (0..n).map(|i| weights[i] * b[i]).sum();
    let c0_partial = partial_sill(vg, keep_nugget, keep_structures);
    let variance = (c0_partial - sum_wc - x[n]).max(0.0);
    Ok(Estimate {
        value,
        variance,
        n_used: n,
        weights,
        lagrange: f64::NAN,
        support_variance: f64::NAN,
    })
}

/// Covariance contribution of a chosen subset of structures (+ optional nugget).
fn partial_cov(
    vg: &Variogram,
    p: &(f64, f64, f64),
    q: &(f64, f64, f64),
    keep_nugget: bool,
    keep: &[usize],
) -> f64 {
    let h = vg.lag(p, q);
    let mut cov = 0.0;
    if keep_nugget && h <= 0.0 {
        cov += vg.nugget;
    }
    for &s in keep {
        if let Some(st) = vg.structures.get(s) {
            // C_s(h) = sill·(1 − shape(h)).
            cov += st.sill * (1.0 - variogram::model::shape(st.model, h, st.range));
        }
    }
    cov
}

fn partial_sill(vg: &Variogram, keep_nugget: bool, keep: &[usize]) -> f64 {
    let mut s = if keep_nugget { vg.nugget } else { 0.0 };
    for &i in keep {
        if let Some(st) = vg.structures.get(i) {
            s += st.sill;
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::krige::{Kind, krige};
    use variogram::model::{Model, Structure};

    fn samp(x: f64, y: f64, v: f64) -> Sample {
        Sample {
            loc: (x, y, 0.0),
            value: v,
            hole: None,
            error_variance: 0.0,
        }
    }

    fn data() -> Vec<Sample> {
        vec![
            samp(0.0, 0.0, 1.0),
            samp(50.0, 0.0, 2.0),
            samp(0.0, 50.0, 3.0),
            samp(50.0, 50.0, 2.5),
            samp(25.0, 25.0, 2.2),
        ]
    }

    #[test]
    fn ordinary_matches_monolithic_krige() {
        let vg = Variogram::single(Model::Spherical, 1.0, 100.0);
        let d = data();
        let t = (20.0, 30.0, 0.0);
        let a = krige(Kind::Ordinary, &t, &d, &vg).unwrap();
        let b = krige_ordinary(&t, &d, &vg).unwrap();
        assert!(
            (a.value - b.value).abs() < 1e-9,
            "{} vs {}",
            a.value,
            b.value
        );
        assert!((a.variance - b.variance).abs() < 1e-9);
    }

    #[test]
    fn simple_matches_monolithic_krige() {
        let vg = Variogram::single(Model::Exponential, 1.0, 120.0);
        let d = data();
        let t = (20.0, 30.0, 0.0);
        let a = krige(Kind::Simple { mean: 2.0 }, &t, &d, &vg).unwrap();
        let b = krige_universal(&t, &d, &DriftSpec::simple(d.len()), &vg, Some(2.0)).unwrap();
        assert!(
            (a.value - b.value).abs() < 1e-9,
            "{} vs {}",
            a.value,
            b.value
        );
        assert!((a.variance - b.variance).abs() < 1e-9);
    }

    #[test]
    fn universal_reproduces_linear_trend_exactly() {
        // Data on an exact linear trend z = 2x + 3y + 1 → UK(degree 1) must be exact.
        let f = |x: f64, y: f64| 2.0 * x + 3.0 * y + 1.0;
        let coords = [
            (0.0, 0.0),
            (100.0, 0.0),
            (0.0, 100.0),
            (100.0, 100.0),
            (40.0, 60.0),
            (80.0, 20.0),
        ];
        let samples: Vec<Sample> = coords.iter().map(|&(x, y)| samp(x, y, f(x, y))).collect();
        let vg = Variogram::single(Model::Spherical, 1.0, 30.0); // short range → far target
        let t = (300.0, 300.0, 0.0);
        let cvec: Vec<(f64, f64, f64)> = coords.iter().map(|&(x, y)| (x, y, 0.0)).collect();
        let drift = DriftSpec::polynomial(&cvec, &t, 1);
        let est = krige_universal(&t, &samples, &drift, &vg, None).unwrap();
        assert!(
            (est.value - f(300.0, 300.0)).abs() < 1e-6,
            "UK trend {} vs {}",
            est.value,
            f(300.0, 300.0)
        );
    }

    #[test]
    fn dual_matches_ordinary() {
        let vg = Variogram::single(Model::Spherical, 1.0, 100.0);
        let d = data();
        let dual = DualKriging::new(&d, &vg, 0).unwrap();
        for t in [(20.0, 30.0, 0.0), (10.0, 10.0, 0.0), (45.0, 5.0, 0.0)] {
            let ok = krige_ordinary(&t, &d, &vg).unwrap();
            let du = dual.estimate(&t);
            assert!((ok.value - du).abs() < 1e-7, "dual {du} vs ok {}", ok.value);
        }
    }

    #[test]
    fn bayesian_limits_to_uk_and_sk() {
        let vg = Variogram::single(Model::Spherical, 1.0, 100.0);
        let d = data();
        let t = (20.0, 30.0, 0.0);
        let coords: Vec<(f64, f64, f64)> = d.iter().map(|s| s.loc).collect();
        let drift = DriftSpec::polynomial(&coords, &t, 0); // constant drift (mean)

        // Large prior variance → approaches OK.
        let ok = krige_ordinary(&t, &d, &vg).unwrap();
        let bayes_flat = krige_bayesian(&t, &d, &drift, &vg, &[0.0], &[1e12]).unwrap();
        assert!(
            (bayes_flat.value - ok.value).abs() < 1e-3,
            "flat {} vs ok {}",
            bayes_flat.value,
            ok.value
        );

        // Tiny prior variance around mean m → approaches SK with mean m.
        let m = 2.0;
        let sk = krige(Kind::Simple { mean: m }, &t, &d, &vg).unwrap();
        let bayes_tight = krige_bayesian(&t, &d, &drift, &vg, &[m], &[1e-12]).unwrap();
        assert!(
            (bayes_tight.value - sk.value).abs() < 1e-3,
            "tight {} vs sk {}",
            bayes_tight.value,
            sk.value
        );
    }

    #[test]
    fn factorial_filtering_denoises() {
        // Nugget + structure. Filtering out the nugget should pull the estimate
        // toward the smooth component and reduce it relative to keeping all.
        let vg = Variogram {
            nugget: 0.5,
            structures: vec![Structure::new(Model::Spherical, 0.5, 100.0)],
            anisotropy: None,
        };
        let d = data();
        let t = (25.0, 25.0, 0.0);
        // Factorial kriging filters the mean (Σλ = 0) → estimates a zero-mean
        // component, so it must not reproduce the OK estimate but the OK
        // estimate minus the (kriged) mean.
        let filtered = krige_factorial(&t, &d, &vg, false, &[0]).unwrap();
        assert!(filtered.value.is_finite());
        assert!(filtered.variance >= 0.0);
        assert!(
            filtered.weights.iter().sum::<f64>().abs() < 1e-9,
            "weights must sum to 0"
        );

        // Keeping every component recovers OK value minus its kriged mean.
        let full = krige_factorial(&t, &d, &vg, true, &[0]).unwrap();
        let ok = krige_ordinary(&t, &d, &vg).unwrap();
        let ok_mean = ok
            .weights
            .iter()
            .zip(&d)
            .map(|(w, s)| w * s.value)
            .sum::<f64>();
        // full ≈ ok.value − mean_component; both finite and full smaller in magnitude.
        assert!(full.value.is_finite() && (ok_mean - ok.value).abs() < 1e-9);
        assert!(
            full.value.abs() < ok.value.abs(),
            "component {} not below total {}",
            full.value,
            ok.value
        );
    }
}
