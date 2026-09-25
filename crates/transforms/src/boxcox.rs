//! Box–Cox power transform and its inverse.
//!
//! `y = (xᵏ − 1)/λ` for λ ≠ 0, `y = ln x` for λ = 0; requires strictly positive
//! input. [`optimal_lambda`] picks the λ (over a candidate grid) that minimizes
//! the magnitude of the transformed data's skewness — a simple normalizing rule.

use crate::error::{Result, TransformError};

/// Forward Box–Cox transform of strictly-positive `values`.
pub fn box_cox(values: &[f64], lambda: f64) -> Result<Vec<f64>> {
    if values.is_empty() {
        return Err(TransformError::InsufficientData("no values".into()));
    }
    if values.iter().any(|&x| x <= 0.0) {
        return Err(TransformError::InvalidParameters(
            "Box–Cox requires strictly positive values".into(),
        ));
    }
    Ok(values
        .iter()
        .map(|&x| {
            if lambda.abs() < 1e-9 {
                x.ln()
            } else {
                (x.powf(lambda) - 1.0) / lambda
            }
        })
        .collect())
}

/// Inverse Box–Cox: map a transformed value back to data space.
pub fn box_cox_inverse(y: f64, lambda: f64) -> f64 {
    if lambda.abs() < 1e-9 {
        y.exp()
    } else {
        (lambda * y + 1.0).max(0.0).powf(1.0 / lambda)
    }
}

/// Sample skewness (Fisher–Pearson) of a slice.
fn skewness(xs: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = xs.iter().sum::<f64>() / n;
    let m2 = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let m3 = xs.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
    let sd = m2.sqrt();
    if sd <= 0.0 { 0.0 } else { m3 / sd.powi(3) }
}

/// Skewness of the Box–Cox transform at `lambda` (0 when the transform fails).
pub fn skewness_at(values: &[f64], lambda: f64) -> f64 {
    box_cox(values, lambda).map(|y| skewness(&y)).unwrap_or(0.0)
}

/// Choose the λ (from `candidates`) whose transform minimizes |skewness|.
pub fn optimal_lambda(values: &[f64], candidates: &[f64]) -> Result<f64> {
    if candidates.is_empty() {
        return Err(TransformError::InvalidParameters(
            "no candidate lambdas".into(),
        ));
    }
    let mut best = candidates[0];
    let mut best_skew = f64::INFINITY;
    for &lam in candidates {
        let s = box_cox(values, lam)?;
        let skew = skewness(&s).abs();
        if skew < best_skew {
            best_skew = skew;
            best = lam;
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lambda_one_is_shift_by_one() {
        let y = box_cox(&[2.0, 3.0, 4.0], 1.0).unwrap();
        assert_eq!(y, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn lambda_zero_is_log() {
        let y = box_cox(&[1.0, std::f64::consts::E], 0.0).unwrap();
        assert!((y[0] - 0.0).abs() < 1e-12);
        assert!((y[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn inverse_round_trips() {
        for &lam in &[0.0, 0.5, 1.0, -0.3, 2.0] {
            let y = box_cox(&[1.5, 4.2, 9.9], lam).unwrap();
            for (i, &yi) in y.iter().enumerate() {
                let back = box_cox_inverse(yi, lam);
                let orig = [1.5, 4.2, 9.9][i];
                assert!((back - orig).abs() < 1e-6, "lam {lam} {back} != {orig}");
            }
        }
    }

    #[test]
    fn rejects_nonpositive() {
        assert!(box_cox(&[1.0, 0.0, 2.0], 0.5).is_err());
        assert!(box_cox(&[-1.0], 1.0).is_err());
    }

    #[test]
    fn optimal_lambda_reduces_right_skew() {
        // A strongly right-skewed sample: a log-ish λ should beat the identity.
        let data: Vec<f64> = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 5.0, 8.0, 21.0, 55.0].to_vec();
        let cands: Vec<f64> = (-4..=8).map(|i| i as f64 * 0.25).collect();
        let lam = optimal_lambda(&data, &cands).unwrap();
        assert!(skewness_at(&data, lam).abs() < skewness_at(&data, 1.0).abs());
    }
}
