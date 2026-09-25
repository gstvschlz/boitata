//! Geostatistical simulation.
//!
//! Modules:
//! - [`sgs`]  — Sequential Gaussian Simulation (continuous variables)
//! - [`sis`]  — Sequential Indicator Simulation (categorical facies)
//! - [`post`] — ensemble post-processing (mean, quantiles P10/P50/P90, prob. above cutoff)
//!
//! Realizations are conditional and reproducible given a seed.

pub mod error;
pub mod gibbs;
pub mod pgs;
pub mod post;
pub mod sgs;
pub mod sis;
pub mod turning_bands;

pub use error::{Result, SimError};
pub use gibbs::{GibbsParams, gibbs};
pub use pgs::{PgsParams, Region, TruncationRule, plurigaussian};
pub use post::{NodeStats, node_stats, probability_above, quantile_sorted};
pub use sgs::{Realization, SgsParams, sgs, sgs_ensemble};
pub use sis::{CategoricalRealization, SisParams, sis};
pub use turning_bands::{
    TurningBandsParams, conditional_gaussian_field, turning_bands, turning_bands_ensemble,
};
