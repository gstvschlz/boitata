//! Estimation.
//!
//! Modules:
//! - [`krige`]   — ordinary/simple/indicator kriging (point support)
//! - [`block`]   — block kriging via discretization
//! - [`idw`]     — inverse-distance weighting and nearest-neighbor
//! - [`search`]  — search-neighborhood selection (anisotropic, octant, per-hole caps)
//! - [`validate`] — leave-one-out and k-fold cross-validation with diagnostics
//!
//! All estimators operate on [`Sample`]s and (for kriging) a [`variogram::Variogram`].

pub mod block;
pub mod cokrige;
pub mod disjunctive;
pub mod error;
pub mod idw;
pub mod krige;
pub mod kriging_algebra;
pub mod neighborhood;
pub mod search;
pub mod simple_interp;
pub mod validate;

pub use block::{Discretization, block_krige};
pub use cokrige::{CoKind, CoSample, cokrige, collocated_cokrige};
pub use disjunctive::{DisjunctiveKriging, GaussianSample};
pub use error::{EstimError, Result};
pub use idw::{idw, nearest};
pub use krige::{Estimate, Kind, krige};
pub use kriging_algebra::{
    DriftSpec, DualKriging, krige_bayesian, krige_factorial, krige_ordinary, krige_universal,
};
pub use neighborhood::{NeighborhoodStats, neighborhood_stats};
pub use search::{Search, neighbors};
pub use simple_interp::{
    InterpEstimate, InterpOptions, inverse_distance, local_least_squares, moving_average,
    moving_median,
};
pub use validate::{CvRecord, CvSummary, k_fold, leave_one_out};

use serde::{Deserialize, Serialize};

/// A located sample value, optionally tagged with its drill hole (for per-hole caps).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    pub loc: (f64, f64, f64),
    pub value: f64,
    pub hole: Option<u32>,
}

impl Sample {
    pub fn new(loc: (f64, f64, f64), value: f64) -> Self {
        Self {
            loc,
            value,
            hole: None,
        }
    }

    pub fn with_hole(loc: (f64, f64, f64), value: f64, hole: u32) -> Self {
        Self {
            loc,
            value,
            hole: Some(hole),
        }
    }
}
