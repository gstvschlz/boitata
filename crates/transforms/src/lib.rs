//! Geospatial transforms.
//!
//! Modules:
//! - [`decluster`] — cell declustering (weights, optimal cell size)
//! - [`nscore`]   — normal-score transform and back-transform
//! - [`detrend`]  — polynomial and kernel trends
//! - [`scale`]    — up/downscaling between supports
//! - [`normal`]   — standard-normal CDF/quantile helpers (shared)

pub mod anamorphosis;
pub mod boxcox;
pub mod decluster;
pub mod detrend;
pub mod dgm;
pub mod error;
pub mod hermite;
pub mod impute;
pub mod localize;
pub mod normal;
pub mod normal_score;
pub mod pca;
pub mod ppmt;
pub mod scale;
pub mod selectivity;
pub mod stepwise;
pub mod support;
pub mod uc;

pub use anamorphosis::HermiteAnamorphosis;
pub use boxcox::{box_cox, box_cox_inverse, optimal_lambda, skewness_at};
pub use decluster::{
    Weights, cell_weights, decluster_mean_over_offsets, optimal_cell_size, polygon_weights,
};
pub use detrend::{KernelTrend, Trend, detrend};
pub use dgm::{BlockDiscretization, change_of_support};
pub use error::{Result, TransformError};
pub use impute::GaussianImputer;
pub use normal::{phi, probit};
pub use normal_score::{NormalScore, NormalScoreTable, transform as normal_score};
pub use pca::{Maf, Pca};
pub use ppmt::{Ppmt, PpmtParams};
pub use scale::{CoarseBlock, downscale, upscale};
pub use selectivity::{Recovery, grade_tonnage, recovery};
pub use stepwise::StepwiseConditional;
pub use support::{affine_correction, indirect_lognormal_correction, weighted_mean_variance};
pub use uc::UniformConditioning;
