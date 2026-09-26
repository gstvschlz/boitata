use arrow_array::BooleanArray;
use ceres_core::{BlockModel, Geometry, Layout, PointSet, Polylines};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use pyo3_arrow::error::PyArrowResult;

use crate::invalid;
use crate::table::{Table, arrow_c_stream, column, describe, empty, to_batch};

/// `(n, 2)` or `(n, 3)` array-like to xyz rows, 2D with z = 0; the coords
/// of a PointSet or the centroids of a BlockModel.
pub fn coords_arg(coords: &Bound<PyAny>) -> PyResult<Vec<[f64; 3]>> {
    if let Ok(points) = coords.cast::<PyPointSet>() {
        return Ok(points.get().0.coords().to_vec());
    }
    if let Ok(model) = coords.cast::<PyBlockModel>() {
        return Ok(model.get().0.centroids());
    }
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

    /// New point set with the columns of `data`, a dict or table, added or replaced.
    fn with_columns(&self, data: &Bound<PyAny>) -> PyResult<Self> {
        Ok(Self(with_columns(
            self.0.clone(),
            data,
            PointSet::with_column,
        )?))
    }

    /// Points where `mask` is true.
    fn filter(&self, mask: PyReadonlyArray1<bool>) -> PyResult<Self> {
        let mask = mask.as_array().to_vec();
        crate::args::same_length(self.0.len(), mask.len(), "mask")?;
        let coords = crate::args::pick(
            self.0.coords(),
            &(0..mask.len()).filter(|&i| mask[i]).collect::<Vec<_>>(),
        );
        let attributes =
            arrow_select::filter::filter_record_batch(self.0.attributes(), &mask.into())
                .map_err(invalid)?;
        let mut points = PointSet::new(coords, attributes).map_err(core_error)?;
        points.crs = self.0.crs.clone();
        Ok(Self(points))
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

#[derive(FromPyObject)]
enum Closed {
    All(bool),
    Each(Vec<bool>),
}

/// Lines and polygons with one attribute row per feature.
///
/// Parameters
/// ----------
/// parts : sequence of array_like
///     ``(m, 2)`` or ``(m, 3)`` vertices per part; 2D gets z = 0.
/// closed : bool or sequence of bool, default False
///     Whether each part is a ring. Rings do not repeat their first vertex
///     and need 3 vertices; open parts need 2.
/// features : sequence of int, optional
///     Feature of each part, non-decreasing from 0. A feature with several
///     parts is multipart; its rings bound an area by even-odd counting in
///     plan, so a ring inside another is a hole. Defaults to one feature per
///     part.
/// attributes : table-like or dict of arrays, optional
///     One row per feature.
/// crs : str, optional
#[pyclass(module = "ceres", name = "Polylines", frozen)]
pub struct PyPolylines(pub Polylines);

#[pymethods]
impl PyPolylines {
    #[new]
    #[pyo3(signature = (parts, closed=Closed::All(false), features=None, attributes=None, crs=None))]
    fn new(
        parts: Vec<Bound<PyAny>>,
        closed: Closed,
        features: Option<Vec<i64>>,
        attributes: Option<&Bound<PyAny>>,
        crs: Option<String>,
    ) -> PyResult<Self> {
        let mut vertices = vec![];
        let mut offsets = vec![0u32];
        for part in &parts {
            vertices.extend(coords_arg(part)?);
            offsets.push(u32::try_from(vertices.len()).map_err(|_| invalid("too many vertices"))?);
        }
        let closed = match closed {
            Closed::All(c) => vec![c; parts.len()],
            Closed::Each(c) => c,
        };
        let features = features.unwrap_or_else(|| (0..parts.len() as i64).collect());
        crate::args::same_length(parts.len(), features.len(), "features")?;
        if features.first().is_some_and(|&f| f != 0) || features.windows(2).any(|w| w[1] < w[0]) {
            return Err(invalid("features must be non-decreasing from 0"));
        }
        let attributes = match attributes {
            Some(a) => to_batch(a)?,
            None => empty(features.last().map_or(0, |&f| f as usize + 1)),
        };
        let count = attributes.num_rows() as i64;
        if features.last().is_some_and(|&f| f >= count) {
            return Err(invalid(format!(
                "features must be below {count}, the attribute rows"
            )));
        }
        let features = (0..=count)
            .map(|f| features.partition_point(|&x| x < f) as u32)
            .collect();
        let mut lines =
            Polylines::new(vertices, offsets, features, closed, attributes).map_err(core_error)?;
        lines.crs = crs;
        Ok(Self(lines))
    }

    /// ``(n, 3)`` vertices of all parts, one part after the other.
    #[getter]
    fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        coords_array(py, self.0.vertices())
    }

    /// ``(m, 3)`` vertices per part.
    #[getter]
    fn parts<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
        (0..self.0.num_parts())
            .map(|i| coords_array(py, self.0.part(i)))
            .collect()
    }

    /// Whether each part is a ring.
    #[getter]
    fn closed<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_slice(py, self.0.closed())
    }

    /// Feature of each part.
    #[getter]
    fn feature<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let ids = (0..self.0.len()).flat_map(|f| self.0.feature_parts(f).map(move |_| f as i64));
        PyArray1::from_iter(py, ids)
    }

    #[getter]
    fn attributes(&self) -> Table {
        Table(self.0.attributes().clone())
    }

    #[getter]
    fn crs(&self) -> Option<String> {
        self.0.crs.clone()
    }

    /// New polylines with the per-feature attribute `name` added or replaced.
    fn with_column(&self, name: &str, values: &Bound<PyAny>) -> PyResult<Self> {
        let column = crate::blocks::attribute(values, self.0.len())?;
        Ok(Self(self.0.with_column(name, column).map_err(core_error)?))
    }

    /// One point per vertex with ``feature`` and ``part`` indices and the
    /// feature's attributes.
    fn to_points(&self) -> PyResult<PyPointSet> {
        Ok(PyPointSet(self.0.to_points().map_err(core_error)?))
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
            "Polylines({} features, {} parts, crs: {crs}){}",
            self.0.len(),
            self.0.num_parts(),
            describe(self.0.attributes())
        )
    }

    /// Builds polylines from a long table with one row per vertex.
    ///
    /// Parameters
    /// ----------
    /// table : table-like
    ///     Any Arrow-compatible table (polars, pandas, pyarrow).
    /// feature : str, default "ID"
    ///     Column identifying the feature of each vertex. Features are ordered
    ///     by first appearance; a feature without rows cannot be expressed.
    /// x, y : str, default "X", "Y"
    ///     Coordinate columns.
    /// z : str, optional
    ///     Elevation column; z = 0 without it.
    /// part : str, optional
    ///     Column identifying the part within a feature, ordered by first
    ///     appearance; without it each feature is one part.
    /// closed : bool, default False
    ///     Whether every part is a ring.
    /// crs : str, optional
    ///
    /// Returns
    /// -------
    /// Polylines
    ///     Vertices keep their row order within a part. Attributes, the
    ///     ``feature`` column included, come from each feature's first row;
    ///     the coordinate and ``part`` columns are dropped.
    #[staticmethod]
    #[pyo3(signature = (table, feature="ID", x="X", y="Y", z=None, part=None, closed=false, crs=None))]
    #[allow(clippy::too_many_arguments)]
    fn from_table(
        table: &Bound<PyAny>,
        feature: &str,
        x: &str,
        y: &str,
        z: Option<&str>,
        part: Option<&str>,
        closed: bool,
        crs: Option<String>,
    ) -> PyResult<Self> {
        let table = to_batch(table)?;
        let mut lines =
            Polylines::from_table(&table, feature, x, y, z, part, closed).map_err(core_error)?;
        lines.crs = crs;
        Ok(Self(lines))
    }

    /// One row per feature: the attributes, ``geometry`` as a list of parts,
    /// each a list of ``{x, y, z}`` structs, and ``closed`` as a list of flags.
    fn to_table(&self) -> PyResult<Table> {
        Ok(Table(self.0.to_table().map_err(core_error)?))
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

    /// Parent cell index of each row (masked or sub-blocked), or `None` when regular.
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<u64>>> {
        match self.0.layout() {
            Layout::Regular => None,
            Layout::Masked(i) => Some(PyArray1::from_slice(py, i)),
            Layout::SubBlocked { parent, .. } => Some(PyArray1::from_slice(py, parent)),
        }
    }

    /// `(n, 6)` sub-block extents as fractions of the parent cell
    /// `[u0, v0, w0, u1, v1, w1]`, or `None` when not sub-blocked.
    #[getter]
    fn extents<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        match self.0.layout() {
            Layout::SubBlocked { extent, .. } => {
                let rows: Vec<Vec<f64>> = extent.iter().map(|e| e.to_vec()).collect();
                Some(crate::args::array2(py, &rows).into_any())
            }
            _ => None,
        }
    }

    /// Volume of each row (area with unit height in 2D).
    #[getter]
    fn volumes<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        crate::args::array1(py, self.0.volumes()).into_any()
    }

    /// Sub-blocked model: `parent` cell of each sub-block (sorted) and its
    /// `(n, 6)` extent as fractions of that cell; `subgrid` (e.g. `(4, 4, 8)`)
    /// requires corners on that subdivision.
    #[staticmethod]
    #[pyo3(signature = (origin, size, count, parent, extents, rotation=(0.0, 0.0, 0.0), subgrid=None, attributes=None, crs=None))]
    #[allow(clippy::too_many_arguments)]
    fn subblocked(
        origin: Vec<f64>,
        size: Vec<f64>,
        count: Vec<usize>,
        parent: PyReadonlyArray1<u64>,
        extents: &Bound<PyAny>,
        rotation: (f64, f64, f64),
        subgrid: Option<[u32; 3]>,
        attributes: Option<&Bound<PyAny>>,
        crs: Option<String>,
    ) -> PyResult<Self> {
        let geometry = Geometry {
            origin: triple(origin, 0.0, "origin")?,
            size: triple(size, 1.0, "size")?,
            count: triple(count, 1, "count")?,
            rotation: [rotation.0, rotation.1, rotation.2],
        };
        let parent = parent.as_array().to_vec();
        let extent = crate::args::rows(extents, "extents")?
            .into_iter()
            .map(|r| {
                <[f64; 6]>::try_from(r.as_slice())
                    .map_err(|_| invalid("extents must have shape (n, 6)"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let attributes = match attributes {
            Some(a) => to_batch(a)?,
            None => empty(parent.len()),
        };
        let mut model = BlockModel::subblocked(geometry, parent, extent, subgrid, attributes)
            .map_err(core_error)?;
        model.crs = crs;
        Ok(Self(model))
    }

    /// `(n, 3)` cell centers in world coordinates.
    #[getter]
    fn centroids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        coords_array(py, &self.0.centroids())
    }

    /// Row holding each point.
    ///
    /// Parameters
    /// ----------
    /// points : array_like
    ///     ``(n, 3)`` or ``(n, 2)`` world coordinates.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     ``int64`` row of each point, -1 outside the model or in a missing block.
    fn row_at<'py>(
        &self,
        py: Python<'py>,
        points: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let rows: Vec<i64> = coords_arg(points)?
            .into_iter()
            .map(|p| self.0.row_at(p).map_or(-1, |r| r as i64))
            .collect();
        Ok(rows.into_pyarray(py))
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

    /// New model with the columns of `data`, a dict or table, added or replaced.
    fn with_columns(&self, data: &Bound<PyAny>) -> PyResult<Self> {
        Ok(Self(with_columns(
            self.0.clone(),
            data,
            BlockModel::with_column,
        )?))
    }

    /// Nodes discretizing every block, e.g. targets for ``simulate(blocks=)``.
    ///
    /// Parameters
    /// ----------
    /// n : int or sequence of int
    ///     Nodes per axis inside each block; an int applies to x, y and z,
    ///     or to x and y only in a 2D model.
    ///
    /// Returns
    /// -------
    /// BlockModel
    ///     A masked grid ``n`` times finer, or for a sub-blocked model each
    ///     sub-block split into ``n`` per axis. Its one column, ``block``, is
    ///     the row in this model of each node.
    fn discretize(&self, n: &Bound<PyAny>) -> PyResult<Self> {
        let n = self.per_axis(n)?;
        Ok(Self(self.0.discretize(n).map_err(core_error)?))
    }

    /// Every parent cell, absent cells null. Sub-blocks merge into their
    /// parent as in `regularize`, without the ``fraction`` column: floats as
    /// volume-weighted means, other columns by volume majority.
    fn to_regular(&self) -> PyResult<Self> {
        Ok(Self(self.0.to_regular().map_err(core_error)?))
    }

    /// Columns of this model averaged onto the blocks of ``target``.
    ///
    /// Each row of this model counts in a target block by the volume they
    /// share, from exact box overlaps, so it works both ways between regular,
    /// masked and sub-blocked models: to a coarser grid, sub-blocks to their
    /// parents, or a regular estimate onto sub-blocks.
    ///
    /// Parameters
    /// ----------
    /// target : BlockModel
    ///     Blocks to average onto, with the same rotation (the origin and
    ///     sizes may differ). Its own columns are kept.
    /// min_fraction : float, default 0.0
    ///     Blocks less covered than this are null.
    ///
    /// Returns
    /// -------
    /// BlockModel
    ///     ``target`` with every column of this model: floats as
    ///     volume-weighted means of the non-null rows, other columns (text,
    ///     integers) as the value filling the most volume, ties to the
    ///     smallest, nulls not voting. ``fraction`` is the share of each block
    ///     covered by this model, so ``volumes * fraction * grade`` sums to
    ///     this model's ``volumes * grade``.
    #[pyo3(signature = (target, min_fraction=0.0))]
    fn regularize(&self, py: Python, target: PyRef<Self>, min_fraction: f64) -> PyResult<Self> {
        let (source, target) = (&self.0, &target.0);
        let model = py
            .detach(|| source.regularize(target, min_fraction))
            .map_err(core_error)?;
        Ok(Self(model))
    }

    /// Sub-blocks every block by prioritized meshes.
    ///
    /// Each block is split on a regular ``subgrid``; each sub-cell takes the
    /// label of the first domain holding its center. Blocks of one label stay
    /// whole, others keep runs of sub-cells merged along x, then y.
    ///
    /// Parameters
    /// ----------
    /// domains : sequence of (Mesh, str, str)
    ///     ``(mesh, rule, label)`` in priority order, the first match
    ///     winning. ``rule`` is ``"inside"`` a closed mesh, or ``"below"`` or
    ///     ``"above"`` a surface such as topography, along world z; off the
    ///     surface's footprint neither matches.
    /// subgrid : int or sequence of int
    ///     Sub-cells per axis, which sets the smallest sub-block; an int
    ///     applies to x, y and z, or to x and y only in a 2D model.
    /// column : str, default "domain"
    ///     Name of the label column.
    /// fill : str, optional
    ///     Label of the sub-cells no domain holds; without it they are
    ///     dropped.
    ///
    /// Returns
    /// -------
    /// BlockModel
    ///     Sub-blocked model on ``subgrid`` with the label column and the
    ///     columns of each sub-block's parent block.
    #[pyo3(signature = (domains, subgrid, column="domain", fill=None))]
    fn subblock(
        &self,
        py: Python,
        domains: Vec<(PyRef<crate::blocks::Mesh>, String, String)>,
        subgrid: &Bound<PyAny>,
        column: &str,
        fill: Option<&str>,
    ) -> PyResult<Self> {
        let subgrid = self
            .per_axis(subgrid)?
            .map(|n| u32::try_from(n).unwrap_or(u32::MAX));
        let domains = domains
            .into_iter()
            .map(|(mesh, rule, label)| mesh.domain(&rule, label))
            .collect::<PyResult<Vec<_>>>()?;
        let model = py
            .detach(|| blocks::subblock(&self.0, &domains, subgrid, column, fill))
            .map_err(crate::invalid)?;
        Ok(Self(model))
    }

    /// Sub-blocked model of the grid ``origin``, ``size``, ``count`` and
    /// ``rotation`` from prioritized meshes; see `subblock`.
    #[staticmethod]
    #[pyo3(signature = (origin, size, count, domains, subgrid, rotation=(0.0, 0.0, 0.0), column="domain", fill=None, crs=None))]
    #[allow(clippy::too_many_arguments)]
    fn from_meshes(
        py: Python,
        origin: Vec<f64>,
        size: Vec<f64>,
        count: Vec<usize>,
        domains: Vec<(PyRef<crate::blocks::Mesh>, String, String)>,
        subgrid: &Bound<PyAny>,
        rotation: (f64, f64, f64),
        column: &str,
        fill: Option<&str>,
        crs: Option<String>,
    ) -> PyResult<Self> {
        let grid = Self::new(origin, size, count, rotation, None, None, crs)?;
        grid.subblock(py, domains, subgrid, column, fill)
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
            Layout::SubBlocked { .. } => "sub-blocked",
        };
        let rows = match self.0.layout() {
            Layout::SubBlocked { .. } => {
                format!("{} sub-blocks in {} cells", self.0.len(), g.cells())
            }
            _ => format!("{} of {} cells", self.0.len(), g.cells()),
        };
        format!(
            "BlockModel({layout}, {rows}, count {:?}, size {:?}, rotation {:?}){}",
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

impl PyBlockModel {
    /// An int for every axis (x and y only in 2D), or 2 or 3 values.
    fn per_axis(&self, n: &Bound<PyAny>) -> PyResult<[usize; 3]> {
        match n.extract::<usize>() {
            Ok(n) if self.0.geometry().count[2] == 1 => Ok([n, n, 1]),
            Ok(n) => Ok([n; 3]),
            Err(_) => triple(n.extract()?, 1, "n"),
        }
    }
}

fn with_columns<T>(
    mut target: T,
    data: &Bound<PyAny>,
    set: fn(&T, &str, arrow_array::ArrayRef) -> ceres_core::Result<T>,
) -> PyResult<T> {
    let batch = to_batch(data)?;
    for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
        target = set(&target, field.name(), column.clone()).map_err(core_error)?;
    }
    Ok(target)
}

pub fn float_column(values: &Bound<PyAny>, rows: usize) -> PyResult<arrow_array::ArrayRef> {
    let values = crate::args::floats(values, "values")?;
    crate::args::same_length(rows, values.len(), "values")?;
    let array: arrow_array::Float64Array = values
        .into_iter()
        .map(|v| (!v.is_nan()).then_some(v))
        .collect();
    Ok(std::sync::Arc::new(array))
}
