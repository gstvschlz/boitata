use estimation::lva::{LocalAnisotropy as Core, MeshMajor};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::args::{Point, array2, points, points_array, rows};
use crate::blocks::Mesh;
use crate::containers::PyBlockModel;
use crate::estimation::targets;
use crate::invalid;
use crate::persist::{self, Columns, Found, Tabular};

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

/// No parameters: everything is in the columns.
#[derive(Serialize, Deserialize)]
struct Empty {}

impl Serialize for LocalAnisotropy {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        Empty {}.serialize(s)
    }
}

impl<'de> Deserialize<'de> for LocalAnisotropy {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Empty::deserialize(d)?;
        Ok(Self(Core {
            coords: Vec::new(),
            angles: Vec::new(),
            ratios: Vec::new(),
        }))
    }
}

impl Tabular for LocalAnisotropy {
    /// `x`, `y`, `z`, `azimuth`, `dip`, `rake`, `semi_ratio`, `minor_ratio`.
    fn columns(&self) -> Option<Columns> {
        let mut columns = persist::point_columns(self.0.coords.iter().copied());
        for (i, name) in ["azimuth", "dip", "rake"].into_iter().enumerate() {
            columns.push(persist::column(name, self.0.angles.iter().map(|a| a[i])));
        }
        for (i, name) in ["semi_ratio", "minor_ratio"].into_iter().enumerate() {
            columns.push(persist::column(name, self.0.ratios.iter().map(|r| r[i])));
        }
        Some(columns)
    }

    fn restore(&mut self, columns: Found) -> PyResult<()> {
        let [azimuth, dip, rake, semi, minor] =
            ["azimuth", "dip", "rake", "semi_ratio", "minor_ratio"].map(|n| columns.values(n));
        let (azimuth, dip, rake, semi, minor) = (azimuth?, dip?, rake?, semi?, minor?);
        let rows = 0..azimuth.len();
        let angles = rows
            .clone()
            .map(|i| [azimuth[i], dip[i], rake[i]])
            .collect();
        let ratios = rows.map(|i| [semi[i], minor[i]]).collect();
        self.0 = Core::new(columns.points()?, angles, ratios).map_err(err)?;
        Ok(())
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
        let vertices: Vec<Point> = mesh
            .mesh
            .vertices()
            .iter()
            .map(|v| (v[0], v[1], v[2]))
            .collect();
        let triangles: Vec<(usize, usize, usize)> = mesh
            .mesh
            .triangles()
            .iter()
            .map(|t| (t[0] as usize, t[1] as usize, t[2] as usize))
            .collect();
        Ok(Self(
            Core::from_mesh(
                &vertices,
                &triangles,
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

    /// Writes locations, angles and ratios as Parquet columns; `from_parquet`
    /// reads them back.
    fn to_parquet(&self, path: std::path::PathBuf) -> PyResult<()> {
        persist::to_parquet(<Self as pyo3::PyClass>::NAME, self, &path)
    }

    /// Reads `to_parquet` output; raises InvalidInput on another class's file
    /// or a newer format.
    #[staticmethod]
    fn from_parquet(path: std::path::PathBuf) -> PyResult<Self> {
        persist::from_parquet(<Self as pyo3::PyClass>::NAME, &path)
    }

    fn _state(&self) -> PyResult<(String, Option<Columns>)> {
        persist::state(<Self as pyo3::PyClass>::NAME, self)
    }

    #[staticmethod]
    fn _from_state(meta: &str, columns: Option<Columns>) -> PyResult<Self> {
        persist::from_state(<Self as pyo3::PyClass>::NAME, meta, columns)
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
