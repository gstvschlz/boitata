//! Geostatistical simulation.
//!
//! Modules:
//! - [`sgs`]  — Sequential Gaussian Simulation (continuous variables)
//! - [`sis`]  — Sequential Indicator Simulation (categorical facies)
//! - [`post`] — uncertainty summaries streamed over realizations
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
pub use post::{
    CategoricalSummary, ContinuousOptions, ContinuousSummary, categorical, continuous,
    quantile_sorted,
};
pub use sgs::{Realization, SgsParams, sgs};
pub use sis::{CategoricalRealization, SisParams, sis};
pub use turning_bands::{
    Bands, GlobalSummary, TurningBandsEnsemble, TurningBandsParams, bounds,
    conditional_gaussian_field, turning_bands, turning_bands_to_parquet,
};
