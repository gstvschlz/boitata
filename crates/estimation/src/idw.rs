//! Inverse-distance weighting (IDW) and nearest-neighbor (NN) estimators.
//!
//! These need no variogram, making them useful for quick passes, sanity checks,
//! and cross-validation baselines.

use crate::Sample;
use crate::error::{EstimError, Result};
use variogram::aniso::euclidean;

/// Inverse-distance-weighted estimate with exponent `power` (commonly 2.0).
///
/// If the target coincides with a sample (distance < `eps`), that sample's value
/// is returned exactly.
pub fn idw(target: &(f64, f64, f64), samples: &[Sample], power: f64) -> Result<f64> {
    if samples.is_empty() {
        return Err(EstimError::InsufficientData("no samples".into()));
    }
    let eps = 1e-9;
    let mut num = 0.0;
    let mut den = 0.0;
    for s in samples {
        let d = euclidean(target, &s.loc);
        if d < eps {
            return Ok(s.value);
        }
        let w = 1.0 / d.powf(power);
        num += w * s.value;
        den += w;
    }
    if den == 0.0 {
        return Err(EstimError::InvalidParameters("zero weight sum".into()));
    }
    Ok(num / den)
}

/// Nearest-neighbor estimate (value of the closest sample).
pub fn nearest(target: &(f64, f64, f64), samples: &[Sample]) -> Result<f64> {
    samples
        .iter()
        .map(|s| (euclidean(target, &s.loc), s.value))
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .map(|(_, v)| v)
        .ok_or_else(|| EstimError::InsufficientData("no samples".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(x: f64, v: f64) -> Sample {
        Sample {
            loc: (x, 0.0, 0.0),
            value: v,
            hole: None,
            error_variance: 0.0,
        }
    }

    #[test]
    fn idw_between_two_points() {
        let samples = vec![s(0.0, 0.0), s(10.0, 10.0)];
        // Midpoint with power 1 → average.
        let v = idw(&(5.0, 0.0, 0.0), &samples, 1.0).unwrap();
        assert!((v - 5.0).abs() < 1e-9);
    }

    #[test]
    fn idw_exact_at_sample() {
        let samples = vec![s(0.0, 7.0), s(10.0, 2.0)];
        let v = idw(&(0.0, 0.0, 0.0), &samples, 2.0).unwrap();
        assert!((v - 7.0).abs() < 1e-9);
    }

    #[test]
    fn nearest_picks_closest() {
        let samples = vec![s(0.0, 7.0), s(10.0, 2.0)];
        assert_eq!(nearest(&(9.0, 0.0, 0.0), &samples).unwrap(), 2.0);
        assert_eq!(nearest(&(1.0, 0.0, 0.0), &samples).unwrap(), 7.0);
    }
}
