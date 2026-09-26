//! Cokriging over a Linear Model of Coregionalization.
//!
//! Estimates one (primary) variable from several correlated variables using the
//! full cross-covariance structure ([`variogram::Coregionalization`]). Supports
//! heterotopic sampling — each datum is a [`CoSample`] tagged with its variable,
//! so variables need not be co-located.
//!
//! - **Simple cokriging** — known mean per variable.
//! - **Ordinary cokriging** — one unbiasedness constraint per variable
//!   (`Σλ = 1` for the primary, `0` for the others).
//! - **Collocated cokriging** — a convenience that appends a secondary datum at
//!   the target location before solving.

use crate::error::{EstimError, Result};
use crate::krige::Estimate;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use variogram::Coregionalization;

/// A located datum tagged with the variable it measures.
#[derive(Debug, Clone, PartialEq)]
pub struct CoSample {
    pub loc: (f64, f64, f64),
    /// Variable index into the coregionalization model.
    pub var: usize,
    pub value: f64,
}

impl CoSample {
    pub fn new(loc: (f64, f64, f64), var: usize, value: f64) -> Self {
        Self { loc, var, value }
    }
}

/// Cokriging variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoKind {
    /// Simple cokriging with a known mean per variable.
    Simple { means: Vec<f64> },
    /// Ordinary cokriging (unbiasedness constraint per variable).
    Ordinary,
}

/// Cokrige `target_var` at `target` from multivariable `samples`.
pub fn cokrige(
    target: &(f64, f64, f64),
    target_var: usize,
    samples: &[CoSample],
    model: &Coregionalization,
    kind: &CoKind,
) -> Result<Estimate> {
    let n = samples.len();
    if n == 0 {
        return Err(EstimError::InsufficientData("no samples".into()));
    }
    if target_var >= model.nvar {
        return Err(EstimError::InvalidParameters(
            "target variable out of range".into(),
        ));
    }
    if samples.iter().any(|s| s.var >= model.nvar) {
        return Err(EstimError::InvalidParameters(
            "sample variable out of range".into(),
        ));
    }

    let c_tt = model.sill(target_var, target_var);

    // Data-data cross-covariance and data-target RHS.
    let cov_dd = |i: usize, j: usize| {
        model.cross_cov(
            samples[i].var,
            samples[j].var,
            &samples[i].loc,
            &samples[j].loc,
        )
    };
    let cov_dt = |i: usize| model.cross_cov(samples[i].var, target_var, &samples[i].loc, target);

    match kind {
        CoKind::Simple { means } => {
            if means.len() != model.nvar {
                return Err(EstimError::InvalidParameters("means length ≠ nvar".into()));
            }
            let mut a = DMatrix::<f64>::zeros(n, n);
            let mut b = DVector::<f64>::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    a[(i, j)] = cov_dd(i, j);
                }
                b[i] = cov_dt(i);
            }
            let x = a
                .lu()
                .solve(&b)
                .ok_or_else(|| EstimError::Singular("simple cokriging singular".into()))?;
            let w: Vec<f64> = x.as_slice().to_vec();
            let value = means[target_var]
                + (0..n)
                    .map(|i| w[i] * (samples[i].value - means[samples[i].var]))
                    .sum::<f64>();
            let sum_wc: f64 = (0..n).map(|i| w[i] * b[i]).sum();
            Ok(Estimate {
                value,
                variance: (c_tt - sum_wc).max(0.0),
                n_used: n,
                weights: w,
                lagrange: f64::NAN,
                support_variance: f64::NAN,
            })
        }
        CoKind::Ordinary => {
            // One unbiasedness constraint per variable present in the neighbourhood.
            let mut present: Vec<usize> = samples.iter().map(|s| s.var).collect();
            present.sort_unstable();
            present.dedup();
            if !present.contains(&target_var) {
                return Err(EstimError::InsufficientData(
                    "no sample of the target variable".into(),
                ));
            }
            let slot = |var: usize| present.binary_search(&var).expect("present");
            let nv = present.len();
            let dim = n + nv;
            let mut a = DMatrix::<f64>::zeros(dim, dim);
            let mut b = DVector::<f64>::zeros(dim);
            for i in 0..n {
                for j in 0..n {
                    a[(i, j)] = cov_dd(i, j);
                }
                let v = slot(samples[i].var);
                a[(i, n + v)] = 1.0;
                a[(n + v, i)] = 1.0;
                b[i] = cov_dt(i);
            }
            b[n + slot(target_var)] = 1.0;
            let x = a
                .lu()
                .solve(&b)
                .ok_or_else(|| EstimError::Singular("ordinary cokriging singular".into()))?;
            let w: Vec<f64> = x.as_slice()[..n].to_vec();
            let value: f64 = (0..n).map(|i| w[i] * samples[i].value).sum();
            let sum_wc: f64 = (0..n).map(|i| w[i] * b[i]).sum();
            let sum_mu: f64 = (0..nv).map(|v| x[n + v] * b[n + v]).sum();
            Ok(Estimate {
                value,
                variance: (c_tt - sum_wc - sum_mu).max(0.0),
                n_used: n,
                weights: w,
                lagrange: f64::NAN,
                support_variance: f64::NAN,
            })
        }
    }
}

/// Collocated cokriging: the secondary variable is known at the target. Appends
/// the collocated datum(s) `(var, value)` at the target location, then cokriges.
pub fn collocated_cokrige(
    target: &(f64, f64, f64),
    target_var: usize,
    samples: &[CoSample],
    collocated: &[(usize, f64)],
    model: &Coregionalization,
    kind: &CoKind,
) -> Result<Estimate> {
    let mut all = samples.to_vec();
    for &(var, value) in collocated {
        all.push(CoSample::new(*target, var, value));
    }
    cokrige(target, target_var, &all, model, kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_cokriging_without_secondary_data_is_ordinary_kriging() {
        use variogram::{CoregStructure, Model, Variogram};
        let model = Coregionalization::new(
            vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            vec![CoregStructure {
                model: Model::Spherical,
                range: 30.0,
                sills: vec![vec![1.0, 0.6], vec![0.6, 1.0]],
            }],
        );
        let locs = [
            (0.0, 0.0, 0.0),
            (10.0, 5.0, 0.0),
            (3.0, 12.0, 0.0),
            (15.0, 15.0, 0.0),
        ];
        let vals = [1.0, 2.5, 0.5, 3.0];
        let co: Vec<CoSample> = locs
            .iter()
            .zip(vals)
            .map(|(&l, v)| CoSample::new(l, 0, v))
            .collect();
        let plain: Vec<crate::Sample> = locs
            .iter()
            .zip(vals)
            .map(|(&l, v)| crate::Sample::new(l, v))
            .collect();
        let target = (7.0, 7.0, 0.0);
        let ck = cokrige(&target, 0, &co, &model, &CoKind::Ordinary).unwrap();
        let vg = Variogram::single(Model::Spherical, 1.0, 30.0);
        let ok = crate::krige(crate::Kind::Ordinary, &target, &plain, &vg).unwrap();
        assert!((ck.value - ok.value).abs() < 1e-10);
        assert!((ck.variance - ok.variance).abs() < 1e-10);
        assert!(cokrige(&target, 1, &co, &model, &CoKind::Ordinary).is_err());
    }
    use variogram::model::Model;
    use variogram::{CoregStructure, Variogram};

    fn biv_model(cross: f64) -> Coregionalization {
        Coregionalization::new(
            vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            vec![CoregStructure {
                model: Model::Spherical,
                range: 100.0,
                sills: vec![vec![1.0, cross], vec![cross, 1.0]],
            }],
        )
    }

    #[test]
    fn zero_cross_reduces_to_univariate_ordinary() {
        // With no cross-correlation, cokriging of var 0 ignores var-1 data and
        // matches ordinary kriging on the var-0 data alone.
        let model = biv_model(0.0);
        let samples = vec![
            CoSample::new((0.0, 0.0, 0.0), 0, 1.0),
            CoSample::new((50.0, 0.0, 0.0), 0, 2.0),
            CoSample::new((0.0, 50.0, 0.0), 0, 3.0),
            CoSample::new((25.0, 25.0, 0.0), 1, 9.0), // secondary, uncorrelated
        ];
        let t = (20.0, 20.0, 0.0);
        let co = cokrige(&t, 0, &samples, &model, &CoKind::Ordinary).unwrap();

        let vg = Variogram::single(Model::Spherical, 1.0, 100.0);
        let prim: Vec<crate::Sample> = samples
            .iter()
            .filter(|s| s.var == 0)
            .map(|s| crate::Sample {
                loc: s.loc,
                value: s.value,
                hole: None,
                error_variance: 0.0,
            })
            .collect();
        let ok = crate::krige_ordinary(&t, &prim, &vg).unwrap();
        assert!(
            (co.value - ok.value).abs() < 1e-6,
            "cok {} vs ok {}",
            co.value,
            ok.value
        );
    }

    #[test]
    fn cross_correlation_shifts_estimate() {
        // A strongly correlated, high secondary datum near the target should pull
        // the primary estimate up relative to the uncorrelated case.
        let t = (20.0, 20.0, 0.0);
        let samples = vec![
            CoSample::new((0.0, 0.0, 0.0), 0, 1.0),
            CoSample::new((60.0, 0.0, 0.0), 0, 1.0),
            CoSample::new((0.0, 60.0, 0.0), 0, 1.0),
            CoSample::new((20.0, 20.0, 0.0), 1, 5.0), // high secondary at target
        ];
        let base = cokrige(
            &t,
            0,
            &samples,
            &biv_model(0.0),
            &CoKind::Simple {
                means: vec![1.0, 1.0],
            },
        )
        .unwrap();
        let corr = cokrige(
            &t,
            0,
            &samples,
            &biv_model(0.8),
            &CoKind::Simple {
                means: vec![1.0, 1.0],
            },
        )
        .unwrap();
        assert!(
            corr.value > base.value + 1e-6,
            "corr {} base {}",
            corr.value,
            base.value
        );
    }

    #[test]
    fn collocated_uses_target_secondary() {
        let model = biv_model(0.7);
        let samples = vec![
            CoSample::new((0.0, 0.0, 0.0), 0, 1.0),
            CoSample::new((60.0, 0.0, 0.0), 0, 2.0),
            CoSample::new((0.0, 60.0, 0.0), 0, 1.5),
        ];
        let t = (20.0, 20.0, 0.0);
        let means = CoKind::Simple {
            means: vec![1.5, 1.5],
        };
        let plain = cokrige(&t, 0, &samples, &model, &means).unwrap();
        let colo = collocated_cokrige(&t, 0, &samples, &[(1, 4.0)], &model, &means).unwrap();
        // The high collocated secondary should raise the estimate.
        assert!(
            colo.value > plain.value + 1e-6,
            "colo {} plain {}",
            colo.value,
            plain.value
        );
        assert_eq!(colo.n_used, samples.len() + 1);
    }
}
