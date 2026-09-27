//! Cross-validation for estimators, with RMSP-style diagnostics.
//!
//! Provides leave-one-out and k-fold cross-validation over kriging, returning per-sample
//! actual/estimated pairs plus summary error statistics (ME, MAE, RMSE, and the mean
//! standardized squared error, whose target value is ≈ 1 for a well-calibrated model).

use crate::Sample;
use crate::error::{EstimError, Result};
use crate::krige::{Kind, krige};
use crate::search::{Search, neighbors, take};
use serde::{Deserialize, Serialize};
use variogram::Variogram;

/// Per-sample cross-validation record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CvRecord {
    pub actual: f64,
    pub estimate: f64,
    pub variance: f64,
    /// Standardized error `(actual − estimate) / √variance` (NaN if variance = 0).
    pub std_error: f64,
}

/// Cross-validation summary statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CvSummary {
    pub n: usize,
    /// Mean error (bias); ideally ≈ 0.
    pub mean_error: f64,
    pub mean_abs_error: f64,
    pub rmse: f64,
    /// Mean standardized squared error; ideally ≈ 1 (kriging variance well-calibrated).
    pub mean_std_sq_error: f64,
    /// Pearson correlation between actual and estimate.
    pub correlation: f64,
    pub records: Vec<CvRecord>,
}

impl CvSummary {
    fn from_records(records: Vec<CvRecord>) -> Self {
        let n = records.len();
        if n == 0 {
            return Self {
                n: 0,
                mean_error: 0.0,
                mean_abs_error: 0.0,
                rmse: 0.0,
                mean_std_sq_error: 0.0,
                correlation: 0.0,
                records,
            };
        }
        let nf = n as f64;
        let mut me = 0.0;
        let mut mae = 0.0;
        let mut mse = 0.0;
        let mut msse = 0.0;
        let mut msse_count = 0.0;
        for r in &records {
            let err = r.actual - r.estimate;
            me += err;
            mae += err.abs();
            mse += err * err;
            if r.std_error.is_finite() {
                msse += r.std_error * r.std_error;
                msse_count += 1.0;
            }
        }
        let mean_a = records.iter().map(|r| r.actual).sum::<f64>() / nf;
        let mean_e = records.iter().map(|r| r.estimate).sum::<f64>() / nf;
        let mut cov = 0.0;
        let mut va = 0.0;
        let mut ve = 0.0;
        for r in &records {
            let da = r.actual - mean_a;
            let de = r.estimate - mean_e;
            cov += da * de;
            va += da * da;
            ve += de * de;
        }
        let correlation = if va > 0.0 && ve > 0.0 {
            cov / (va.sqrt() * ve.sqrt())
        } else {
            0.0
        };

        Self {
            n,
            mean_error: me / nf,
            mean_abs_error: mae / nf,
            rmse: (mse / nf).sqrt(),
            mean_std_sq_error: if msse_count > 0.0 {
                msse / msse_count
            } else {
                0.0
            },
            correlation,
            records,
        }
    }
}

/// Leave-one-out cross-validation of ordinary kriging.
///
/// Each sample is removed in turn and re-estimated from its neighbors.
pub fn leave_one_out(samples: &[Sample], vg: &Variogram, search: &Search) -> Result<CvSummary> {
    if samples.len() < 2 {
        return Err(EstimError::InsufficientData("need ≥ 2 samples".into()));
    }

    let mut records = Vec::with_capacity(samples.len());
    for i in 0..samples.len() {
        let target = samples[i].loc;
        let rest: Vec<Sample> = samples
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, s)| s.clone())
            .collect();

        let idx = match neighbors(&target, &rest, search, Some(vg)) {
            Ok(v) => v,
            Err(_) => continue, // not enough neighbors; skip this sample
        };
        let selected = take(&target, &idx, &rest, search, Some(vg));
        let est = krige(Kind::Ordinary, &target, &selected, vg)?;
        let std_error = if est.variance > 0.0 {
            (samples[i].value - est.value) / est.variance.sqrt()
        } else {
            f64::NAN
        };
        records.push(CvRecord {
            actual: samples[i].value,
            estimate: est.value,
            variance: est.variance,
            std_error,
        });
    }

    Ok(CvSummary::from_records(records))
}

/// K-fold cross-validation of ordinary kriging.
///
/// Samples are partitioned into `k` folds deterministically (round-robin by index).
/// Each fold is held out and estimated from the remaining samples.
pub fn k_fold(samples: &[Sample], vg: &Variogram, search: &Search, k: usize) -> Result<CvSummary> {
    if k < 2 {
        return Err(EstimError::InvalidParameters("k must be ≥ 2".into()));
    }
    if samples.len() < k {
        return Err(EstimError::InsufficientData(
            "fewer samples than folds".into(),
        ));
    }

    let mut records = Vec::with_capacity(samples.len());
    for fold in 0..k {
        // Training = samples not in this fold; test = samples in this fold.
        let train: Vec<Sample> = samples
            .iter()
            .enumerate()
            .filter(|(i, _)| i % k != fold)
            .map(|(_, s)| s.clone())
            .collect();
        let test: Vec<(usize, &Sample)> = samples
            .iter()
            .enumerate()
            .filter(|(i, _)| i % k == fold)
            .collect();

        for (_, s) in test {
            let idx = match neighbors(&s.loc, &train, search, Some(vg)) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let selected = take(&s.loc, &idx, &train, search, Some(vg));
            let est = krige(Kind::Ordinary, &s.loc, &selected, vg)?;
            let std_error = if est.variance > 0.0 {
                (s.value - est.value) / est.variance.sqrt()
            } else {
                f64::NAN
            };
            records.push(CvRecord {
                actual: s.value,
                estimate: est.value,
                variance: est.variance,
                std_error,
            });
        }
    }

    Ok(CvSummary::from_records(records))
}

#[cfg(test)]
mod tests {
    use super::*;
    use variogram::model::Model;

    fn grid_samples() -> Vec<Sample> {
        // Smooth linear field on a 6×6 grid → kriging should reproduce it well.
        let mut v = Vec::new();
        for i in 0..6 {
            for j in 0..6 {
                v.push(Sample {
                    loc: (i as f64 * 20.0, j as f64 * 20.0, 0.0),
                    value: i as f64 + j as f64,
                    hole: None,
                    error_variance: 0.0,
                    domain: None,
                });
            }
        }
        v
    }

    #[test]
    fn loo_reasonable_for_smooth_field() {
        let vg = Variogram::single(Model::Spherical, 2.0, 120.0);
        let search = Search {
            min_samples: 3,
            max_samples: 12,
            radius: f64::INFINITY,
            max_per_hole: None,
            octant: false,
            anisotropy: None,
            high_grade: None,
            soft: None,
        };
        let cv = leave_one_out(&grid_samples(), &vg, &search).unwrap();
        assert!(cv.n > 0);
        assert!(cv.correlation > 0.8, "correlation {}", cv.correlation);
        assert!(cv.mean_error.abs() < 1.0, "bias {}", cv.mean_error);
    }

    #[test]
    fn kfold_runs_and_summarizes() {
        let vg = Variogram::single(Model::Spherical, 2.0, 120.0);
        let search = Search {
            min_samples: 3,
            max_samples: 12,
            radius: f64::INFINITY,
            max_per_hole: None,
            octant: false,
            anisotropy: None,
            high_grade: None,
            soft: None,
        };
        let cv = k_fold(&grid_samples(), &vg, &search, 5).unwrap();
        assert!(cv.n > 0);
        assert!(cv.rmse.is_finite());
    }
}
