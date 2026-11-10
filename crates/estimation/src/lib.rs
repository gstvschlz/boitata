//! Estimation.
//!
//! Modules:
//! - [`krige`]   — ordinary/simple/indicator kriging (point support)
//! - [`block`]   — block kriging via discretization
//! - [`idw`]     — inverse-distance weighting and nearest-neighbor
//! - [`indicator`] — multiple indicator kriging of conditional distributions
//! - [`categorical`] — indicator kriging of category probabilities
//! - [`multigaussian`] — conditional distributions from kriged normal scores
//! - [`search`]  — search-neighborhood selection (anisotropic, octant, per-hole caps)
//! - [`validate`] — leave-one-out and k-fold cross-validation with diagnostics
//!
//! All estimators operate on [`Sample`]s and (for kriging) a [`variogram::Variogram`].

pub mod batch;
pub mod block;
pub mod categorical;
mod coincident;
pub mod cokrige;
pub mod disjunctive;
pub mod error;
pub mod idw;
pub mod indicator;
mod kernel;
pub mod krige;
pub mod kriging_algebra;
pub mod lva;
pub mod multigaussian;
pub mod neighborhood;
pub mod search;
pub mod simple_interp;
pub mod validate;

pub use batch::{
    by_pass, estimate_many, estimate_many_ext, k_fold_at, leave_one_out_at, leave_one_out_many,
    weight_declustering, weights_many,
};
pub use block::{Discretization, block_covariances, block_krige, block_krige_points};
pub use categorical::{CategoricalIndicator, CategoricalIndicatorSummary, correct_probabilities};
pub use cokrige::{CoKind, CoSample, cokrige, collocated_cokrige, markov_collocated};
pub use disjunctive::{DisjunctiveKriging, GaussianSample};
pub use error::{EstimError, Result};
pub use idw::{idw, nearest};
pub use indicator::{
    Conditional, Global, IndicatorDiagnostics, IndicatorSummary, Interpolation, MultipleIndicator,
    UpperTail, correct_order_relations, variance_factor,
};
pub use krige::{Estimate, Kind, krige};
pub use kriging_algebra::{
    DriftSpec, DualKriging, Quality, calibrated_count, check_bayesian_prior, krige_bayesian,
    krige_bayesian_polynomial, krige_calibrated, krige_factorial, krige_ordinary, krige_universal,
    prefix_quality,
};
pub use multigaussian::Multigaussian;
pub use neighborhood::{NeighborhoodStats, hole_distance, neighborhood_stats};
pub use search::{
    Calibration, HighGrade, HighGradeMode, PlaneSectors, Search, Soft, SoftPair,
    neighborhood_groups, neighbors, neighbors_in, same_neighborhood, take,
};
pub use simple_interp::{
    InterpEstimate, InterpOptions, inverse_distance, inverse_distance_weights, local_least_squares,
    moving_average, moving_median, nearest_index,
};
pub use validate::{CvRecord, CvSummary, k_fold, leave_one_out};

use serde::{Deserialize, Serialize};

/// A located sample value, optionally tagged with its drill hole (for per-hole caps).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    pub loc: (f64, f64, f64),
    pub value: f64,
    pub hole: Option<u32>,
    /// Variance of the measurement error, added to this sample's diagonal
    /// entry in kriging systems: the estimate no longer honors the value.
    #[serde(default)]
    pub error_variance: f64,
    /// Domain code: a target of another domain only sees this sample
    /// within the search's soft distance.
    #[serde(default)]
    pub domain: Option<u32>,
}

impl Sample {
    pub fn new(loc: (f64, f64, f64), value: f64) -> Self {
        Self {
            loc,
            value,
            hole: None,
            error_variance: 0.0,
            domain: None,
        }
    }

    pub fn with_hole(loc: (f64, f64, f64), value: f64, hole: u32) -> Self {
        Self {
            loc,
            value,
            hole: Some(hole),
            error_variance: 0.0,
            domain: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use variogram::{Model, Variogram};

    #[test]
    fn twins_krige_as_one_sample() {
        let locs = [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (5.0, 5.0, -0.0),
            (5.0, 5.0, 0.0),
        ];
        let vg = Variogram::single(Model::Spherical, 1.0, 50.0);
        let samples: Vec<Sample> = locs
            .iter()
            .zip([1.0, 2.0, 3.0, 4.0, 6.0])
            .map(|(&l, v)| Sample::new(l, v))
            .collect();
        let target = (2.0, 2.0, 0.0);
        let est = krige(Kind::Ordinary, &target, &samples, &vg).unwrap();
        let kept: Vec<Sample> = [(0, 2.0), (1, 2.0), (3, 5.0)]
            .iter()
            .map(|&(i, v)| Sample::new(locs[i], v))
            .collect();
        let one = krige(Kind::Ordinary, &target, &kept, &vg).unwrap();
        assert!((est.value - one.value).abs() < 1e-12);
        assert!((est.variance - one.variance).abs() < 1e-12);
        assert_eq!(est.weights[0], est.weights[2]);
        assert_eq!(est.weights[3], est.weights[4]);
    }
}
