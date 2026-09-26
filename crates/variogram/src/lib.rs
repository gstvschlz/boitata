//! Variography.
//!
//! Modules:
//! - [`model`]  — theoretical variogram shapes (spherical, exponential, gaussian,
//!   cubic, pentaspherical, circular, sine-hole, Matérn, power)
//! - [`composite`] — nugget + nested structures with optional anisotropy ([`Variogram`])
//! - [`aniso`]  — anisotropy (rotation + range scaling)
//! - [`empirical`] — experimental variograms (classical, robust, covariance, correlogram,
//!   pairwise relative; directional) and cross-variograms
//! - [`fit`]    — automatic model fitting by weighted least squares
//! - [`surface`] — γ on a cut plane (variogram maps)
//! - [`transio`] — transiograms for categorical / facies variables

pub mod aniso;
pub mod composite;
pub mod coreg;
pub mod empirical;
pub mod error;
pub mod fit;
pub mod model;
pub mod surface;
pub mod transio;

pub use aniso::{Angles, Anisotropy};
pub use composite::Variogram;
pub use coreg::{CoregStructure, Coregionalization};
pub use empirical::{
    Direction, Estimator, Experimental, LagBins, cross_experimental, experimental,
};
pub use error::{Result, VarioError};
pub use fit::{
    AnisotropySpec, Bounds, CoregFit, FitResult, NestedSpec, StructureSpec, Weighting, fit,
    fit_coregionalization, fit_directional, fit_nested,
};
pub use model::{Model, Structure, is_differentiable, shape, shape_d1, shape_d2};
pub use surface::{PlaneMap, PlaneMapParams, azimuth_dip, plane_map, unit_vector};
pub use transio::{EmpiricalTransiogram, Transiogram, empirical_transiogram};
