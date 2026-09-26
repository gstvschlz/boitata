//! Automatic variogram-model fitting by weighted least squares.

use crate::composite::Variogram;
use crate::empirical::Experimental;
use crate::error::{Result, VarioError};
use crate::model::{Model, Structure};
use serde::{Deserialize, Serialize};

/// Weighting scheme for the least-squares objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Weighting {
    /// Uniform weights.
    Uniform,
    /// Weight by pair count `N(h)` (more pairs ⇒ more reliable lag).
    ByCount,
    /// `N(h) / γ_model(h)²` (emphasizes short lags).
    ByCountOverGamma,
}

/// Result of a fit: the model and the achieved weighted SSE.
#[derive(Debug, Clone)]
pub struct FitResult {
    pub variogram: Variogram,
    pub wsse: f64,
}

/// Fit a single-structure model (with nugget) to an experimental variogram by WLS.
///
/// Uses a coarse grid over `(nugget, sill, range)` followed by coordinate-descent
/// refinement. The `Power` and `SineHole` models are supported but fit only their
/// scaling/range (no bounded sill for `Power`).
pub fn fit(exp: &Experimental, model: Model, weighting: Weighting) -> Result<FitResult> {
    if exp.lags.is_empty() {
        return Err(VarioError::FittingFailed(
            "empty experimental variogram".into(),
        ));
    }

    let max_gamma = exp
        .gammas
        .iter()
        .cloned()
        .fold(f64::MIN, f64::max)
        .max(f64::MIN_POSITIVE);
    let max_lag = exp.lags.iter().cloned().fold(f64::MIN, f64::max);

    let sill_hi = max_gamma * 1.5;
    let range_hi = max_lag * 1.5;

    let mut best = FitResult {
        variogram: Variogram {
            nugget: 0.0,
            structures: vec![Structure::new(model, max_gamma, max_lag)],
            anisotropy: None,
        },
        wsse: f64::INFINITY,
    };

    let steps = 12;
    for ni in 0..=6 {
        let nugget = max_gamma * (ni as f64) / 12.0; // nugget up to half the sill
        for si in 1..=steps {
            let sill = sill_hi * (si as f64) / steps as f64;
            for ri in 1..=steps {
                let range = range_hi * (ri as f64) / steps as f64;
                let cand = Variogram {
                    nugget,
                    structures: vec![Structure::new(model, sill, range)],
                    anisotropy: None,
                };
                let e = wsse(&cand, exp, weighting);
                if e < best.wsse {
                    best = FitResult {
                        variogram: cand,
                        wsse: e,
                    };
                }
            }
        }
    }

    refine(&mut best, exp, model, weighting);

    Ok(best)
}

fn refine(best: &mut FitResult, exp: &Experimental, model: Model, weighting: Weighting) {
    let mut nugget = best.variogram.nugget;
    let mut sill = best.variogram.structures[0].sill;
    let mut range = best.variogram.structures[0].range;

    let mut scale = 0.5;
    for _ in 0..40 {
        let mut improved = false;
        for &(dn, ds, dr) in &[
            (scale, 0.0, 0.0),
            (-scale, 0.0, 0.0),
            (0.0, scale, 0.0),
            (0.0, -scale, 0.0),
            (0.0, 0.0, scale),
            (0.0, 0.0, -scale),
        ] {
            let cn = (nugget * (1.0 + dn)).max(0.0);
            let cs = (sill * (1.0 + ds)).max(1e-9);
            let cr = (range * (1.0 + dr)).max(1e-6);
            let cand = Variogram {
                nugget: cn,
                structures: vec![Structure::new(model, cs, cr)],
                anisotropy: None,
            };
            let e = wsse(&cand, exp, weighting);
            if e < best.wsse {
                best.wsse = e;
                best.variogram = cand;
                nugget = cn;
                sill = cs;
                range = cr;
                improved = true;
            }
        }
        if !improved {
            scale *= 0.5;
            if scale < 1e-4 {
                break;
            }
        }
    }
}

fn wsse(v: &Variogram, exp: &Experimental, weighting: Weighting) -> f64 {
    let mut acc = 0.0;
    for i in 0..exp.lags.len() {
        let h = exp.lags[i];
        let model_g = v.gamma(h);
        let resid = exp.gammas[i] - model_g;
        let w = match weighting {
            Weighting::Uniform => 1.0,
            Weighting::ByCount => exp.counts[i] as f64,
            Weighting::ByCountOverGamma => {
                let denom = model_g.max(1e-6);
                exp.counts[i] as f64 / (denom * denom)
            }
        };
        acc += w * resid * resid;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::empirical::{Estimator, LagBins, experimental};

    #[test]
    fn recovers_spherical_parameters() {
        // Generate synthetic samples from a known spherical model via a moving average,
        // then check the fit lands in a sensible neighborhood.
        // Here we build an experimental variogram directly from a known model.
        let truth = Variogram::single(Model::Spherical, 2.0, 120.0);
        let lags: Vec<f64> = (1..20).map(|i| i as f64 * 10.0).collect();
        let gammas: Vec<f64> = lags.iter().map(|&h| truth.gamma(h)).collect();
        let counts: Vec<usize> = lags.iter().map(|_| 100).collect();
        let exp = Experimental {
            lags,
            gammas,
            counts,
            covariances: None,
        };

        let fit = fit(&exp, Model::Spherical, Weighting::ByCount).unwrap();
        let s = &fit.variogram.structures[0];
        // Sill+nugget should approach 2.0; range near 120.
        assert!(
            (fit.variogram.total_sill() - 2.0).abs() < 0.3,
            "sill {}",
            fit.variogram.total_sill()
        );
        assert!((s.range - 120.0).abs() < 40.0, "range {}", s.range);
    }

    #[test]
    fn fits_experimental_from_data() {
        let locs: Vec<(f64, f64, f64)> = (0..30).map(|i| (i as f64 * 5.0, 0.0, 0.0)).collect();
        let vals: Vec<f64> = (0..30).map(|i| ((i as f64) * 0.3).sin()).collect();
        let bins = LagBins {
            max_lag: 100.0,
            lag_width: 10.0,
        };
        let exp = experimental(&locs, &vals, &bins, Estimator::Matheron, None, false).unwrap();
        let fit = fit(&exp, Model::Exponential, Weighting::ByCountOverGamma).unwrap();
        assert!(fit.wsse.is_finite());
        assert!(fit.variogram.total_sill() > 0.0);
    }
}
