//! Geostatistical simulation.
//!
//! Modules:
//! - [`sgs`]  — Sequential Gaussian Simulation (continuous variables)
//! - [`sis`]  — Sequential Indicator Simulation (categorical facies)
//! - [`training_image`] — block model columns as training images
//! - [`objects`] — object-based training images: channels and ellipsoids
//! - [`multivariate`] — several correlated variables through independent factors
//! - [`trend`] — a continuous variable whose distribution follows a trend
//! - [`correct`] — correction of realizations to a target distribution
//! - [`post`] — uncertainty summaries streamed over realizations, at node or
//!   block support, and localization of block realizations within panels
//!
//! Realizations are conditional and reproducible given a seed.

mod conditioning;
pub mod correct;
pub mod error;
pub mod gibbs;
pub mod lattice;
pub mod multivariate;
pub mod objects;
pub mod pgs;
pub mod post;
pub mod sgs;
pub mod shared;
pub mod sis;
pub mod training_image;
pub mod trend;
pub mod turning_bands;

pub use correct::{Empirical, correct_distribution};
pub use error::{Result, SimError};
pub use gibbs::{GibbsParams, gibbs};
pub use lattice::{Lattice, Template, default_levels, multigrid_path};
pub use multivariate::{Decorrelation, factor_seed, multivariate, multivariate_batched};
pub use objects::{ObjectSet, Param, Shape, object_training_image};
pub use pgs::{
    Hierarchy, PgsParams, Region, TruncationRule, fit_latent, plurigaussian, plurigaussian_local,
};
pub use post::{
    BlockSupport, CategoricalSummary, ContinuousOptions, ContinuousSummary, Keep, categorical,
    continuous, continuous_batched, continuous_in_batches, continuous_many,
    continuous_many_batched, localize, quantile_sorted,
};
pub use sgs::{
    Domains, Realization, Secondary, SgsParams, Transform, Transforms, Trend, cosgs, sgs, sgs_in,
    sgs_passes,
};
pub use shared::{
    Collocated, SharedBatch, SharedSgs, sgs_shared, shared_batch, shared_unsupported,
};
pub use sis::{CategoricalRealization, SisParams, sis};
pub use training_image::{NO_CODE, TrainingImage, TrainingValues, unify};
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
