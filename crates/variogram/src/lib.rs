//! Variography.
//!
//! Modules:
//! - [`model`]  — theoretical variogram shapes (spherical, exponential, gaussian,
//!   cubic, pentaspherical, circular, sine-hole, Matérn, power)
//! - [`composite`] — nugget + nested structures with optional anisotropy ([`Variogram`])
//! - [`aniso`]  — anisotropy (rotation + range scaling)
//! - [`empirical`] — experimental variograms (Matheron / Cressie–Hawkins, directional)
//! - [`fit`]    — automatic model fitting by weighted least squares
//! - [`surface`] — γ over a hemisphere of directions, and γ on a cut plane
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
pub use empirical::{Direction, Estimator, Experimental, LagBins, experimental};
pub use error::{Result, VarioError};
pub use fit::{FitResult, Weighting, fit};
pub use model::{Model, Structure, is_differentiable, shape, shape_d1, shape_d2};
pub use surface::{
    PlaneMap, PlaneMapParams, SurfaceDirection, SurfaceParams, VariogramSurface, azimuth_dip,
    plane_map, unit_vector, variogram_surface,
};
pub use transio::{EmpiricalTransiogram, Transiogram, empirical_transiogram};
