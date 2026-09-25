//! Point→SMU support corrections of a grade histogram.
//!
//! Both corrections take a variance reduction factor
//! `f = Var(Z_v) / Var(Z) ∈ [0, 1]` — obtained from the average variogram over
//! the SMU (`f = 1 − γ̄(v,v)/sill`, i.e. [`crate::dgm::block_average_correlation`])
//! or from known dispersion variances — and rescale the point-support values so
//! their histogram has block-support selectivity:
//!
//! - **Affine**: `z_v = √f·(z − m) + m` — shape-preserving shrink about the
//!   mean; variance becomes exactly `f·σ²`.
//! - **Indirect lognormal**: `z_v = a·z^b` with the exponents implied by two
//!   lognormal distributions sharing the mean (`b` from the CV ratio, per
//!   Isaaks & Srivastava), then rescaled by `m/m′` so the mean is
//!   preserved exactly; unlike affine it reduces skewness and cannot go
//!   negative.
//!
//! The distribution-correcting alternative is the discrete Gaussian model
//! ([`crate::dgm`]).

use crate::error::{Result, TransformError};

/// Weighted mean and (population) variance. `weights` optional → uniform.
pub fn weighted_mean_variance(values: &[f64], weights: Option<&[f64]>) -> Result<(f64, f64)> {
    if values.is_empty() {
        return Err(TransformError::InsufficientData("no values".into()));
    }
    if let Some(w) = weights {
        if w.len() != values.len() {
            return Err(TransformError::InvalidParameters(
                "weights length mismatch".into(),
            ));
        }
        if w.iter().any(|&x| x < 0.0) {
            return Err(TransformError::InvalidParameters(
                "weights must be ≥ 0".into(),
            ));
        }
    }
    let sw: f64 = match weights {
        Some(w) => w.iter().sum(),
        None => values.len() as f64,
    };
    if sw <= 0.0 {
        return Err(TransformError::InvalidParameters("weights sum ≤ 0".into()));
    }
    let wi = |i: usize| weights.map_or(1.0, |w| w[i]);
    let mean = values
        .iter()
        .enumerate()
        .map(|(i, &z)| wi(i) * z)
        .sum::<f64>()
        / sw;
    let var = values
        .iter()
        .enumerate()
        .map(|(i, &z)| wi(i) * (z - mean) * (z - mean))
        .sum::<f64>()
        / sw;
    Ok((mean, var))
}

fn check_factor(f: f64) -> Result<()> {
    if !(0.0..=1.0).contains(&f) {
        return Err(TransformError::InvalidParameters(format!(
            "variance reduction factor must be in [0, 1], got {f}"
        )));
    }
    Ok(())
}

/// Affine correction to SMU support: `z_v = √f·(z − m) + m`.
///
/// Preserves the (weighted) mean exactly and scales the variance by exactly
/// `f`; the histogram shape (skewness, CV ordering) is unchanged.
pub fn affine_correction(values: &[f64], weights: Option<&[f64]>, f: f64) -> Result<Vec<f64>> {
    check_factor(f)?;
    let (mean, _) = weighted_mean_variance(values, weights)?;
    let s = f.sqrt();
    Ok(values.iter().map(|&z| s * (z - mean) + mean).collect())
}

/// Indirect lognormal correction to SMU support: `z_v = a·z^b`, then a final
/// `× m/m′` rescale so the mean is preserved exactly (the variance lands near — not exactly on — `f·σ²`).
///
/// Requires non-negative values with a positive mean (a lognormal-shaped
/// grade histogram).
pub fn indirect_lognormal_correction(
    values: &[f64],
    weights: Option<&[f64]>,
    f: f64,
) -> Result<Vec<f64>> {
    check_factor(f)?;
    let (mean, var) = weighted_mean_variance(values, weights)?;
    if mean <= 0.0 || values.iter().any(|&z| z < 0.0) {
        return Err(TransformError::InvalidParameters(
            "indirect lognormal correction requires non-negative values with a positive mean"
                .into(),
        ));
    }
    let cv2 = var / (mean * mean);
    if cv2 <= 0.0 {
        // Constant data: nothing to correct.
        return Ok(values.to_vec());
    }
    // b from the CV ratio of two lognormals sharing the mean; a matches means.
    let b = ((f * cv2 + 1.0).ln() / (cv2 + 1.0).ln()).max(0.0).sqrt();
    let root = (cv2 + 1.0).sqrt();
    let a = (mean / root) * (root / mean).powf(b);
    let corrected: Vec<f64> = values.iter().map(|&z| a * z.powf(b)).collect();
    let (m_new, _) = weighted_mean_variance(&corrected, weights)?;
    if m_new <= 0.0 {
        return Err(TransformError::InvalidParameters(
            "indirect lognormal correction collapsed the mean".into(),
        ));
    }
    let scale = mean / m_new;
    Ok(corrected.into_iter().map(|z| z * scale).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lognormal_values() -> Vec<f64> {
        (1..=400).map(|i| ((i as f64) / 80.0).exp()).collect()
    }

    #[test]
    fn affine_preserves_mean_and_scales_variance_exactly() {
        let vals = lognormal_values();
        let (m, v) = weighted_mean_variance(&vals, None).unwrap();
        let f = 0.6;
        let out = affine_correction(&vals, None, f).unwrap();
        let (m2, v2) = weighted_mean_variance(&out, None).unwrap();
        assert!((m2 - m).abs() < 1e-9 * m.abs().max(1.0), "mean {m2} vs {m}");
        assert!((v2 - f * v).abs() < 1e-9 * v, "var {v2} vs {}", f * v);
    }

    #[test]
    fn affine_with_f_one_is_identity() {
        let vals = lognormal_values();
        let out = affine_correction(&vals, None, 1.0).unwrap();
        for (a, b) in vals.iter().zip(&out) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn affine_respects_weights() {
        let vals = vec![1.0, 2.0, 10.0];
        let w = vec![1.0, 1.0, 0.0];
        // Weighted mean ignores the 10 → 1.5; corrected values shrink toward it.
        let out = affine_correction(&vals, Some(&w), 0.25).unwrap();
        assert!((out[0] - (0.5 * (1.0 - 1.5) + 1.5)).abs() < 1e-12);
        assert!((out[1] - (0.5 * (2.0 - 1.5) + 1.5)).abs() < 1e-12);
    }

    #[test]
    fn indirect_lognormal_preserves_mean_and_reduces_variance() {
        let vals = lognormal_values();
        let (m, v) = weighted_mean_variance(&vals, None).unwrap();
        let f = 0.5;
        let out = indirect_lognormal_correction(&vals, None, f).unwrap();
        let (m2, v2) = weighted_mean_variance(&out, None).unwrap();
        assert!((m2 - m).abs() < 1e-9 * m, "mean {m2} vs {m}");
        assert!(v2 < v, "variance not reduced: {v2} vs {v}");
        // Lands near f·σ² (the power correction is approximate, unlike affine).
        assert!(v2 > 0.3 * v && v2 < 0.8 * v, "v2/v = {}", v2 / v);
    }

    #[test]
    fn indirect_lognormal_with_f_one_is_identity() {
        let vals = lognormal_values();
        let out = indirect_lognormal_correction(&vals, None, 1.0).unwrap();
        for (a, b) in vals.iter().zip(&out) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
    }

    #[test]
    fn indirect_lognormal_reduces_skewness_stays_positive() {
        let vals = lognormal_values();
        let out = indirect_lognormal_correction(&vals, None, 0.4).unwrap();
        assert!(out.iter().all(|&z| z >= 0.0));
        let skew = |xs: &[f64]| {
            let (m, v) = weighted_mean_variance(xs, None).unwrap();
            xs.iter().map(|&z| (z - m).powi(3)).sum::<f64>() / xs.len() as f64 / v.powf(1.5)
        };
        assert!(skew(&out) < skew(&vals), "skewness not reduced");
    }

    #[test]
    fn negative_values_are_rejected_by_indirect_lognormal() {
        assert!(indirect_lognormal_correction(&[-1.0, 2.0, 3.0], None, 0.5).is_err());
    }

    #[test]
    fn out_of_range_factor_is_rejected() {
        let vals = lognormal_values();
        assert!(affine_correction(&vals, None, 1.5).is_err());
        assert!(affine_correction(&vals, None, -0.1).is_err());
        assert!(indirect_lognormal_correction(&vals, None, 1.5).is_err());
    }
}
