use numpy::PyArray2;
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::blocks::Mesh;
use crate::containers::{PyBlockModel, PyPointSet, PyPolylines, coords_arg, coords_array};
use crate::drillholes::Drillholes;
use crate::invalid;

/// Similarity transform from a local mine grid to projected coordinates:
/// ``projected = shift + scale * R(rotation) @ local`` in xy and
/// ``z + z_shift``.
///
/// Parameters
/// ----------
/// scale : float, default 1.0
///     Ratio of projected to local horizontal distances.
/// rotation : float, default 0.0
///     Azimuth of local grid north in projected coordinates, degrees
///     clockwise.
/// shift : (float, float), default (0.0, 0.0)
///     Projected xy of the local origin.
/// z_shift : float, default 0.0
///     Added to local elevations.
/// crs : str, optional
///     CRS of the projected coordinates, set by `apply`.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "LocalGrid", frozen)]
pub struct LocalGrid(boitata_core::LocalGrid);

impl LocalGrid {
    fn convert<'py>(
        &self,
        object: &Bound<'py, PyAny>,
        forward: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = object.py();
        let g = &self.0;
        if let Ok(p) = object.cast::<PyPointSet>() {
            return PyPointSet(g.points(&p.get().0, forward).map_err(invalid)?)
                .into_bound_py_any(py);
        }
        if let Ok(p) = object.cast::<PyPolylines>() {
            return PyPolylines(g.polylines(&p.get().0, forward).map_err(invalid)?)
                .into_bound_py_any(py);
        }
        if let Ok(m) = object.cast::<PyBlockModel>() {
            return PyBlockModel(g.block_model(&m.get().0, forward).map_err(invalid)?)
                .into_bound_py_any(py);
        }
        if let Ok(m) = object.cast::<Mesh>() {
            return Mesh::from_core(g.mesh(&m.get().mesh, forward).map_err(invalid)?)
                .into_bound_py_any(py);
        }
        if let Ok(d) = object.cast::<Drillholes>() {
            let d = d.get();
            g.check(d.crs_ref(), forward).map_err(invalid)?;
            let crs = if forward { g.crs.clone() } else { None };
            return d
                .map_coords(|q| g.map(q, forward), crs)
                .into_bound_py_any(py);
        }
        let coords: Vec<_> = coords_arg(object)?
            .into_iter()
            .map(|q| g.map(q, forward))
            .collect();
        coords_array(py, &coords).into_bound_py_any(py)
    }
}

#[pymethods]
impl LocalGrid {
    #[new]
    #[pyo3(signature = (*, scale=1.0, rotation=0.0, shift=(0.0, 0.0), z_shift=0.0, crs=None))]
    fn new(
        scale: f64,
        rotation: f64,
        shift: (f64, f64),
        z_shift: f64,
        crs: Option<String>,
    ) -> PyResult<Self> {
        if !(scale.is_finite() && scale > 0.0) {
            return Err(invalid("scale must be positive"));
        }
        if ![rotation, shift.0, shift.1, z_shift]
            .iter()
            .all(|v| v.is_finite())
        {
            return Err(invalid("rotation, shift and z_shift must be finite"));
        }
        Ok(Self(boitata_core::LocalGrid {
            scale,
            rotation,
            shift: [shift.0, shift.1],
            z_shift,
            crs,
        }))
    }

    /// Least-squares similarity from control point pairs.
    ///
    /// Parameters
    /// ----------
    /// local, projected : array_like, shape (n, 2) or (n, 3)
    ///     The same control points in local and projected coordinates; at
    ///     least two distinct local positions.
    /// z : bool, default True
    ///     Fit `z_shift` as the mean elevation difference; otherwise 0.
    /// crs : str, optional
    ///     CRS of the projected coordinates.
    ///
    /// Returns
    /// -------
    /// LocalGrid
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     With unequal lengths, fewer than two distinct points or
    ///     non-finite coordinates.
    #[staticmethod]
    #[pyo3(signature = (local, projected, *, z=true, crs=None))]
    fn fit(
        local: &Bound<PyAny>,
        projected: &Bound<PyAny>,
        z: bool,
        crs: Option<String>,
    ) -> PyResult<Self> {
        let (local, projected) = (coords_arg(local)?, coords_arg(projected)?);
        Ok(Self(
            boitata_core::LocalGrid::fit(&local, &projected, z, crs).map_err(invalid)?,
        ))
    }

    /// ``projected - apply(local)`` per control point, shape (n, 3).
    fn residuals<'py>(
        &self,
        py: Python<'py>,
        local: &Bound<PyAny>,
        projected: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (local, projected) = (coords_arg(local)?, coords_arg(projected)?);
        if local.len() != projected.len() {
            return Err(invalid(
                "local and projected need the same number of points",
            ));
        }
        Ok(coords_array(py, &self.0.residuals(&local, &projected)))
    }

    /// Local to projected coordinates.
    ///
    /// Parameters
    /// ----------
    /// object : PointSet, Drillholes, Polylines, Mesh, BlockModel or array_like
    ///     Containers come back in the grid's `crs`; a block model's origin
    ///     moves, its azimuth turns by `rotation` and its horizontal block
    ///     sizes scale by `scale`. An (n, 2) or (n, 3) array comes back as
    ///     an (n, 3) array.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When the container is already in the grid's `crs`, or for a
    ///     dipping or raked block model with a scale other than 1.
    fn apply<'py>(&self, object: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.convert(object, true)
    }

    /// Projected to local coordinates, the exact inverse of `apply`;
    /// containers come back without a CRS.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When the container declares a CRS other than the grid's.
    fn inverse<'py>(&self, object: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.convert(object, false)
    }

    #[getter]
    fn scale(&self) -> f64 {
        self.0.scale
    }

    #[getter]
    fn rotation(&self) -> f64 {
        self.0.rotation
    }

    #[getter]
    fn shift(&self) -> (f64, f64) {
        (self.0.shift[0], self.0.shift[1])
    }

    #[getter]
    fn z_shift(&self) -> f64 {
        self.0.z_shift
    }

    #[getter]
    fn crs(&self) -> Option<String> {
        self.0.crs.clone()
    }

    fn to_json(&self) -> PyResult<String> {
        crate::persist::to_json(self)
    }

    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        crate::persist::from_json(text)
    }

    fn __repr__(&self) -> String {
        let g = &self.0;
        format!(
            "LocalGrid(scale={}, rotation={}, shift=({}, {}), z_shift={}, crs={})",
            g.scale,
            g.rotation,
            g.shift[0],
            g.shift[1],
            g.z_shift,
            g.crs.as_deref().unwrap_or("none")
        )
    }
}
