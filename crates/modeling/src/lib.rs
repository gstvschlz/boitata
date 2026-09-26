//! Implicit geological modeling.
//!
//! Builds surfaces and solids from data — contacts, indicator coding, grade —
//! as isosurfaces of an interpolated scalar field, the "implicit" alternative
//! to digitizing sections by hand.
//!
//! Modules:
//! - [`rbf`]        — radial basis function interpolation (biharmonic,
//!   triharmonic, thin-plate) with polynomial drift and optional smoothing
//! - [`implicit`]   — the three engines (RBF, dual kriging, sparse variational
//!   GP), the data drivers, grid sizing, and the fit → sample → extract
//!   pipeline
//! - [`hermite`]    — dual kriging with derivative data (the potential-field
//!   formulation), the kriging face of structural constraints
//! - [`svgp`]       — sparse variational Gaussian process: `O(nm²)` rather than
//!   `O(n³)`, so the whole drillhole database is fitted rather than a sample of
//!   it, and the posterior variance comes out of the model
//! - [`isosurface`] — marching tetrahedra over a sampled field
//! - [`grid`]       — the sampling lattice and the triangle mesh it produces
//! - [`constraint`] — what a fit is asked to honor: samples, boundary picks
//!   on the contact, and structural readings as derivative rows
//! - [`orientation`] — dip/dip-direction and plunge/trend as unit vectors
//! - [`aniso`]      — anisotropy as a coordinate transform
//!
//! The kriging engine reuses `variogram` + `estimation` rather than
//! reimplementing them; this crate owns the RBF and the isosurfacing.

pub mod aniso;
pub mod constraint;
pub mod error;
pub mod grid;
pub mod hermite;
pub mod implicit;
pub mod isosurface;
pub mod orientation;
pub mod rbf;
pub mod svgp;

pub use aniso::AnisoTransform;
pub use constraint::{
    ConstraintSet, DerivativeConstraint, PlaneEncoding, ValueConstraint, ValueKind,
};
pub use error::{ModelError, Result};
pub use grid::{ScalarGrid, TriMesh};
pub use hermite::{HermiteKriging, HermiteSpec};
pub use implicit::{
    Driver, Engine, GridSpec, ImplicitSurface, build_surface, build_surface_constrained, decimate,
};
pub use isosurface::marching_tetrahedra;
pub use orientation::{Lineation, Plane};
pub use rbf::{Kernel, Rbf, RbfSpec};
pub use svgp::{Convergence, FitReport, Svgp, SvgpSpec};
pub use variogram::Model;
