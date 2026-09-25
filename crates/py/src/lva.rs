use estimation::lva::{LocalAnisotropy as Core, MeshMajor};
use pyo3::prelude::*;

use crate::args::{Point, array2, points, points_array, rows};
use crate::blocks::Mesh;
use crate::containers::PyBlockModel;
use crate::estimation::targets;
use crate::invalid;

fn err(e: estimation::EstimError) -> PyErr {
    invalid(e)
}

/// Orientation (azimuth, dip, rake in degrees) and range ratios (semi/major,
/// minor/major) at a set of locations.
#[pyclass(module = "ceres", name = "LocalAnisotropy", frozen)]
pub struct LocalAnisotropy(pub Core);

impl LocalAnisotropy {
    /// This field transferred to `targets` by nearest neighbour.
    pub fn at_targets(&self, targets: &[Point]) -> Core {
        self.0.at(targets)
    }
}

fn pairs(obj: &Bound<PyAny>, what: &str) -> PyResult<Vec<[f64; 2]>> {
    rows(obj, what)?
        .into_iter()
        .map(|r| match r[..] {
            [a, b] => Ok([a, b]),
            _ => Err(invalid(format!("{what} must have shape (n, 2)"))),
        })
        .collect()
}

#[pymethods]
impl LocalAnisotropy {
    #[new]
    fn new(coords: &Bound<PyAny>, angles: &Bound<PyAny>, ratios: &Bound<PyAny>) -> PyResult<Self> {
        let angles = rows(angles, "angles")?
            .into_iter()
            .map(|r| match r[..] {
                [a, d, k] => Ok([a, d, k]),
                _ => Err(invalid("angles must have shape (n, 3)")),
            })
            .collect::<PyResult<_>>()?;
        Ok(Self(
            Core::new(points(coords)?, angles, pairs(ratios, "ratios")?).map_err(err)?,
        ))
    }

    /// From the gradient of a block-model attribute: the structure tensor summed
    /// over `window` cells each side; the least-change direction is the major
    /// axis. Without `ratios`, ratios follow the tensor's eigenvalues.
    #[staticmethod]
    #[pyo3(signature = (model, column, window=2, ratios=None))]
    fn from_grid(
        model: PyRef<PyBlockModel>,
        column: &str,
        window: usize,
        ratios: Option<[f64; 2]>,
    ) -> PyResult<Self> {
        let regular = model.0.to_regular().map_err(invalid)?;
        let values = Python::attach(|py| {
            crate::args::floats(
                &crate::table::column(py, regular.attributes(), column)?,
                "column",
            )
        })?;
        Ok(Self(
            Core::from_grid(regular.geometry(), &values, window, ratios).map_err(err)?,
        ))
    }

    /// From a point cloud: principal axes of each point's `k` nearest points.
    #[staticmethod]
    #[pyo3(signature = (coords, k=20, ratios=None))]
    fn from_points(coords: &Bound<PyAny>, k: usize, ratios: Option<[f64; 2]>) -> PyResult<Self> {
        Ok(Self(
            Core::from_points(&points(coords)?, k, ratios).map_err(err)?,
        ))
    }

    /// At `targets`, from the nearest triangle of `mesh`: the normal is the
    /// minor axis and `major` is `"dip"` or `"strike"`.
    #[staticmethod]
    #[pyo3(signature = (mesh, targets, major="dip", ratios=(1.0, 0.2)))]
    fn from_mesh(
        mesh: PyRef<Mesh>,
        targets: &Bound<PyAny>,
        major: &str,
        ratios: (f64, f64),
    ) -> PyResult<Self> {
        let major = match major {
            "dip" => MeshMajor::Dip,
            "strike" => MeshMajor::Strike,
            other => {
                return Err(invalid(format!(
                    "major must be dip or strike, not {other:?}"
                )));
            }
        };
        let (vertices, triangles) = mesh.parts();
        Ok(Self(
            Core::from_mesh(
                vertices,
                triangles,
                &self::targets(targets)?,
                major,
                [ratios.0, ratios.1],
            )
            .map_err(err)?,
        ))
    }

    /// Averages orientation tensors within `radius` of each location.
    fn smooth(&self, radius: f64) -> PyResult<Self> {
        if radius.is_nan() || radius <= 0.0 {
            return Err(invalid("radius must be positive"));
        }
        Ok(Self(self.0.smooth(radius)))
    }

    /// The anisotropy of the nearest location, at `targets`.
    fn at(&self, targets: &Bound<PyAny>) -> PyResult<Self> {
        Ok(Self(self.0.at(&self::targets(targets)?)))
    }

    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        points_array(py, &self.0.coords).into_any()
    }

    #[getter]
    fn angles<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        let rows: Vec<Vec<f64>> = self.0.angles.iter().map(|a| a.to_vec()).collect();
        array2(py, &rows).into_any()
    }

    #[getter]
    fn ratios<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        let rows: Vec<Vec<f64>> = self.0.ratios.iter().map(|r| r.to_vec()).collect();
        array2(py, &rows).into_any()
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __repr__(&self) -> String {
        format!("LocalAnisotropy({} locations)", self.0.len())
    }
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<LocalAnisotropy>()?;
    Ok(())
}
