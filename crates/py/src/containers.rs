use arrow_array::BooleanArray;
use ceres_core::{BlockModel, Geometry, Layout, PointSet};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use pyo3_arrow::error::PyArrowResult;

use crate::invalid;
use crate::table::{Table, arrow_c_stream, column, describe, empty, to_batch};

/// `(n, 2)` or `(n, 3)` array-like to xyz rows; 2D gets z = 0.
pub fn coords_arg(coords: &Bound<PyAny>) -> PyResult<Vec<[f64; 3]>> {
    let array = coords
        .py()
        .import("numpy")?
        .call_method1("asarray", (coords, "float64"))?;
    let array: PyReadonlyArray2<f64> = array
        .extract()
        .map_err(|_| invalid("coordinates must be a 2-D array of shape (n, 2) or (n, 3)"))?;
    let array = array.as_array();
    let dims = array.ncols();
    if !(dims == 2 || dims == 3) {
        return Err(invalid(format!(
            "coordinates need 2 or 3 columns, got {dims}"
        )));
    }
    if array.iter().any(|v| !v.is_finite()) {
        return Err(invalid("coordinates must be finite"));
    }
    Ok(array
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], if dims == 3 { r[2] } else { 0.0 }])
        .collect())
}

pub fn coords_array<'py>(py: Python<'py>, coords: &[[f64; 3]]) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_vec((coords.len(), 3), coords.concat())
        .expect("n x 3")
        .into_pyarray(py)
}

fn triple<T: Copy>(values: Vec<T>, fill: T, what: &str) -> PyResult<[T; 3]> {
    match values[..] {
        [x, y] => Ok([x, y, fill]),
        [x, y, z] => Ok([x, y, z]),
        _ => Err(invalid(format!("{what} needs 2 or 3 values"))),
    }
}

fn core_error(e: ceres_core::Error) -> PyErr {
    invalid(e)
}

/// Scattered samples: coordinates plus an attribute table.
#[pyclass(module = "ceres", name = "PointSet", frozen)]
pub struct PyPointSet(pub PointSet);

#[pymethods]
impl PyPointSet {
    #[new]
    #[pyo3(signature = (coords, attributes=None, crs=None))]
    fn new(
        coords: &Bound<PyAny>,
        attributes: Option<&Bound<PyAny>>,
        crs: Option<String>,
    ) -> PyResult<Self> {
        let coords = coords_arg(coords)?;
        let attributes = match attributes {
            Some(a) => to_batch(a)?,
            None => empty(coords.len()),
        };
        let mut points = PointSet::new(coords, attributes).map_err(core_error)?;
        points.crs = crs;
        Ok(Self(points))
    }

    /// Splits coordinate columns out of an Arrow-compatible table.
    #[staticmethod]
    #[pyo3(signature = (table, x="X", y="Y", z=None, crs=None))]
    fn from_table(
        table: &Bound<PyAny>,
        x: &str,
        y: &str,
        z: Option<&str>,
        crs: Option<String>,
    ) -> PyResult<Self> {
        let mut points = PointSet::from_table(&to_batch(table)?, x, y, z).map_err(core_error)?;
        points.crs = crs;
        Ok(Self(points))
    }

    /// `(n, 3)` coordinates.
    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        coords_array(py, self.0.coords())
    }

    #[getter]
    fn attributes(&self) -> Table {
        Table(self.0.attributes().clone())
    }

    #[getter]
    fn crs(&self) -> Option<String> {
        self.0.crs.clone()
    }

    /// New point set with the attribute `name` added or replaced (NaN is null).
    fn with_column(&self, name: &str, values: &Bound<PyAny>) -> PyResult<Self> {
        let column = float_column(values, self.0.len())?;
        Ok(Self(self.0.with_column(name, column).map_err(core_error)?))
    }

    /// Attributes preceded by `x`, `y`, `z`.
    fn to_table(&self) -> PyResult<Table> {
        Ok(Table(self.0.to_table().map_err(core_error)?))
    }

    fn __getitem__<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        column(py, self.0.attributes(), name)
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __repr__(&self) -> String {
        let crs = self.0.crs.as_deref().unwrap_or("none");
        format!(
            "PointSet({} points, crs: {crs}){}",
            self.0.len(),
            describe(self.0.attributes())
        )
    }

    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyArrowResult<Bound<'py, PyCapsule>> {
        let table = self.0.to_table().map_err(core_error)?;
        arrow_c_stream(py, &table, requested_schema)
    }
}

/// Regular or masked grid of blocks, 2D or 3D, optionally rotated.
#[pyclass(module = "ceres", name = "BlockModel", frozen)]
pub struct PyBlockModel(pub BlockModel);

#[pymethods]
impl PyBlockModel {
    /// `origin` is the corner of the first cell; `rotation` is azimuth, dip,
    /// rake in degrees. Pass `index` (sorted cell indices) for a masked model.
    #[new]
    #[pyo3(signature = (origin, size, count, rotation=(0.0, 0.0, 0.0), attributes=None, index=None, crs=None))]
    fn new(
        origin: Vec<f64>,
        size: Vec<f64>,
        count: Vec<usize>,
        rotation: (f64, f64, f64),
        attributes: Option<&Bound<PyAny>>,
        index: Option<PyReadonlyArray1<u64>>,
        crs: Option<String>,
    ) -> PyResult<Self> {
        let geometry = Geometry {
            origin: triple(origin, 0.0, "origin")?,
            size: triple(size, 1.0, "size")?,
            count: triple(count, 1, "count")?,
            rotation: [rotation.0, rotation.1, rotation.2],
        };
        geometry.validate().map_err(core_error)?;
        let index = index.map(|i| i.as_array().to_vec());
        let rows = match &index {
            Some(i) => i.len(),
            None => geometry.cells() as usize,
        };
        let attributes = match attributes {
            Some(a) => to_batch(a)?,
            None => empty(rows),
        };
        let mut model = match index {
            Some(i) => BlockModel::masked(geometry, i, attributes),
            None => BlockModel::regular(geometry, attributes),
        }
        .map_err(core_error)?;
        model.crs = crs;
        Ok(Self(model))
    }

    #[getter]
    fn origin(&self) -> [f64; 3] {
        self.0.geometry().origin
    }

    #[getter]
    fn size(&self) -> [f64; 3] {
        self.0.geometry().size
    }

    #[getter]
    fn count(&self) -> [usize; 3] {
        self.0.geometry().count
    }

    #[getter]
    fn rotation(&self) -> [f64; 3] {
        self.0.geometry().rotation
    }

    #[getter]
    fn crs(&self) -> Option<String> {
        self.0.crs.clone()
    }

    /// Parent cell index of each row, or `None` for a regular model.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<u64>>> {
        match self.0.layout() {
            Layout::Regular => None,
            Layout::Masked(i) => Some(PyArray1::from_slice(py, i)),
        }
    }

    /// `(n, 3)` cell centres in world coordinates.
    #[getter]
    fn centroids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        coords_array(py, &self.0.centroids())
    }

    #[getter]
    fn attributes(&self) -> Table {
        Table(self.0.attributes().clone())
    }

    /// Keeps rows where `keep` is true; the result is masked.
    fn mask(&self, keep: PyReadonlyArray1<bool>) -> PyResult<Self> {
        let keep = BooleanArray::from(keep.as_array().to_vec());
        Ok(Self(self.0.mask(&keep).map_err(core_error)?))
    }

    /// New model with the attribute `name` added or replaced (NaN is null).
    fn with_column(&self, name: &str, values: &Bound<PyAny>) -> PyResult<Self> {
        let column = float_column(values, self.0.len())?;
        Ok(Self(self.0.with_column(name, column).map_err(core_error)?))
    }

    /// Every parent cell, absent cells null.
    fn to_regular(&self) -> PyResult<Self> {
        Ok(Self(self.0.to_regular().map_err(core_error)?))
    }

    /// Attributes preceded by centroid `x`, `y`, `z`.
    fn to_table(&self) -> PyResult<Table> {
        let points =
            PointSet::new(self.0.centroids(), self.0.attributes().clone()).map_err(core_error)?;
        Ok(Table(points.to_table().map_err(core_error)?))
    }

    fn __getitem__<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        column(py, self.0.attributes(), name)
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __repr__(&self) -> String {
        let g = self.0.geometry();
        let layout = match self.0.layout() {
            Layout::Regular => "regular",
            Layout::Masked(_) => "masked",
        };
        format!(
            "BlockModel({layout}, {} of {} cells, count {:?}, size {:?}, rotation {:?}){}",
            self.0.len(),
            g.cells(),
            g.count,
            g.size,
            g.rotation,
            describe(self.0.attributes())
        )
    }

    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyArrowResult<Bound<'py, PyCapsule>> {
        arrow_c_stream(py, &self.to_table()?.0, requested_schema)
    }
}

fn float_column(values: &Bound<PyAny>, rows: usize) -> PyResult<arrow_array::ArrayRef> {
    let values = crate::args::floats(values, "values")?;
    crate::args::same_length(rows, values.len(), "values")?;
    let array: arrow_array::Float64Array = values
        .into_iter()
        .map(|v| (!v.is_nan()).then_some(v))
        .collect();
    Ok(std::sync::Arc::new(array))
}
