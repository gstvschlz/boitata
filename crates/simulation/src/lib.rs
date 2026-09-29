//! Geostatistical simulation.
//!
//! Modules:
//! - [`sgs`]  — Sequential Gaussian Simulation (continuous variables)
//! - [`sis`]  — Sequential Indicator Simulation (categorical facies)
//! - [`multivariate`] — several correlated variables through independent factors
//! - [`trend`] — a continuous variable whose distribution follows a trend
//! - [`correct`] — correction of realizations to a target distribution
//! - [`post`] — uncertainty summaries streamed over realizations, at node or
//!   block support, and localization of block realizations within panels
//!
//! Realizations are conditional and reproducible given a seed.

pub mod correct;
pub mod error;
pub mod gibbs;
pub mod lattice;
pub mod multivariate;
pub mod pgs;
pub mod post;
pub mod sgs;
pub mod sis;
pub mod trend;
pub mod turning_bands;

pub use correct::{Empirical, correct_distribution};
pub use error::{Result, SimError};
pub use gibbs::{GibbsParams, gibbs};
pub use lattice::{Lattice, Template, default_levels, multigrid_path};
pub use multivariate::{Decorrelation, factor_seed, multivariate};
pub use pgs::{
    Hierarchy, PgsParams, Region, TruncationRule, fit_latent, plurigaussian, plurigaussian_local,
};
pub use post::{
    BlockSupport, CategoricalSummary, ContinuousOptions, ContinuousSummary, Keep, categorical,
    continuous, continuous_many, localize, quantile_sorted,
};
pub use sgs::{
    Domains, Realization, Secondary, SgsParams, Transform, Transforms, Trend, cosgs, sgs, sgs_in,
    sgs_passes,
};
pub use sis::{CategoricalRealization, SisParams, sis};
pub use trend::TrendConditioning;
pub use turning_bands::{
    Bands, GlobalSummary, TurningBandsEnsemble, TurningBandsParams, bounds,
    conditional_gaussian_field, turning_bands, turning_bands_in, turning_bands_to_parquet,
};

/// Hole of each of `n` data, for `Search::max_per_hole`; all `None` without
/// `holes`.
pub(crate) fn holes(holes: Option<&[u32]>, n: usize) -> Result<Vec<Option<u32>>> {
    match holes {
        None => Ok(vec![None; n]),
        Some(h) if h.len() == n => Ok(h.iter().copied().map(Some).collect()),
        Some(_) => Err(SimError::InvalidParameters("one hole per datum".into())),
    }
}
