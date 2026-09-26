use std::path::PathBuf;

use ceres_io::{CsvOptions, NODATA};
use pyo3::prelude::*;

use crate::blocks::Mesh;
use crate::containers::{PyBlockModel, PyPointSet};
use crate::table::{Table, to_batch};
use crate::{error, invalid};

pub(crate) fn io_error(e: ceres_io::Error) -> PyErr {
    match e {
        ceres_io::Error::Io(e) => error("FileError", e),
        e => invalid(e),
    }
}

fn nodata(values: Option<Vec<String>>) -> Vec<String> {
    values.unwrap_or_else(|| NODATA.iter().map(|s| s.to_string()).collect())
}

/// Reads a headed CSV; numbers become float64, `nodata` tokens become null.
#[pyfunction]
#[pyo3(signature = (path, nodata=None, delimiter=","))]
fn read_csv(path: PathBuf, nodata: Option<Vec<String>>, delimiter: &str) -> PyResult<Table> {
    let &[delimiter] = delimiter.as_bytes() else {
        return Err(invalid("delimiter must be a single byte"));
    };
    let options = CsvOptions {
        delimiter,
        nodata: self::nodata(nodata),
    };
    Ok(Table(ceres_io::read_csv(path, &options).map_err(io_error)?))
}

#[pyfunction]
fn write_csv(path: PathBuf, table: &Bound<PyAny>) -> PyResult<()> {
    ceres_io::write_csv(path, &to_batch(table)?).map_err(io_error)
}

/// Reads a GSLIB file; the title is kept in the schema metadata.
#[pyfunction]
#[pyo3(signature = (path, nodata=None))]
fn read_gslib(path: PathBuf, nodata: Option<Vec<String>>) -> PyResult<Table> {
    let batch = ceres_io::read_gslib(path, &self::nodata(nodata)).map_err(io_error)?;
    Ok(Table(batch))
}

#[pyfunction]
#[pyo3(signature = (path, table, missing=-999.0))]
fn write_gslib(path: PathBuf, table: &Bound<PyAny>, missing: f64) -> PyResult<()> {
    ceres_io::write_gslib(path, &to_batch(table)?, missing).map_err(io_error)
}

/// Writes a PointSet, a BlockModel or any table to Parquet; containers keep
/// their geometry, layout and CRS in the file metadata.
#[pyfunction]
fn write_parquet(path: PathBuf, data: &Bound<PyAny>) -> PyResult<()> {
    if let Ok(points) = data.cast::<PyPointSet>() {
        return ceres_io::write_points(path, &points.get().0).map_err(io_error);
    }
    if let Ok(model) = data.cast::<PyBlockModel>() {
        return ceres_io::write_block_model(path, &model.get().0).map_err(io_error);
    }
    ceres_io::write_parquet(path, &to_batch(data)?).map_err(io_error)
}

/// Reads Parquet as the PointSet or BlockModel it was written from, or a Table.
#[pyfunction]
fn read_parquet(py: Python, path: PathBuf) -> PyResult<Py<PyAny>> {
    Ok(match ceres_io::read_parquet(path).map_err(io_error)? {
        ceres_io::Stored::Points(p) => PyPointSet(p).into_pyobject(py)?.into_any().unbind(),
        ceres_io::Stored::Blocks(b) => PyBlockModel(b).into_pyobject(py)?.into_any().unbind(),
        ceres_io::Stored::Table(t) => Table(t).into_pyobject(py)?.into_any().unbind(),
    })
}

/// Reads a `.obj`, `.stl` or `.dxf` mesh; DXF faces carry a `layer` column.
#[pyfunction]
fn read_mesh(path: PathBuf) -> PyResult<Mesh> {
    Ok(Mesh::from_core(
        ceres_io::read_mesh(path).map_err(io_error)?,
    ))
}

/// Writes a `.obj`, `.stl` (binary unless `ascii`) or `.dxf` mesh.
#[pyfunction]
#[pyo3(signature = (path, mesh, ascii=false))]
fn write_mesh(path: PathBuf, mesh: PyRef<Mesh>, ascii: bool) -> PyResult<()> {
    ceres_io::write_mesh(path, &mesh.mesh, ascii).map_err(io_error)
}

/// A block model file read in chunks, for models larger than memory.
#[pyclass(module = "ceres", name = "BlockModelFile", frozen)]
pub struct BlockModelFile(ceres_io::BlockModelReader);

#[pymethods]
impl BlockModelFile {
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        Ok(Self(
            ceres_io::BlockModelReader::open(path).map_err(io_error)?,
        ))
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
        self.0.crs().map(str::to_string)
    }

    #[getter]
    fn column_names(&self) -> Vec<String> {
        self.0.column_names().to_vec()
    }

    /// BlockModel pieces of at most `rows` blocks with the chosen `columns`
    /// (all by default); pieces of a regular model are masked to their cells.
    #[pyo3(signature = (rows=1_000_000, columns=None))]
    fn chunks(&self, rows: usize, columns: Option<Vec<String>>) -> PyResult<BlockChunkIterator> {
        let names: Option<Vec<&str>> = columns
            .as_ref()
            .map(|c| c.iter().map(String::as_str).collect());
        Ok(BlockChunkIterator(
            self.0.chunks(rows, names.as_deref()).map_err(io_error)?,
        ))
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "BlockModelFile({} blocks, count {:?}, columns {:?})",
            self.0.len(),
            self.0.geometry().count,
            self.0.column_names()
        )
    }
}

#[pyclass(module = "ceres", unsendable)]
pub struct BlockChunkIterator(ceres_io::BlockChunks);

#[pymethods]
impl BlockChunkIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<PyBlockModel>> {
        self.0
            .next()
            .transpose()
            .map(|chunk| chunk.map(PyBlockModel))
            .map_err(io_error)
    }
}

struct StreamError(PyErr);

impl From<ceres_io::Error> for StreamError {
    fn from(e: ceres_io::Error) -> Self {
        Self(io_error(e))
    }
}

/// Streams the block model file `path` to `out` in chunks of `rows` blocks:
/// `func(chunk)` returns a dict of new columns, written with the chunk's
/// layout and, when `keep`, its columns. Memory stays bounded by `rows`.
#[pyfunction]
#[pyo3(signature = (path, out, func, rows=1_000_000, keep=true))]
fn map_blocks(
    path: PathBuf,
    out: PathBuf,
    func: &Bound<PyAny>,
    rows: usize,
    keep: bool,
) -> PyResult<()> {
    ceres_io::stream_map::<StreamError>(path, out, rows, keep, |chunk| {
        let columns = func
            .call1((PyBlockModel(chunk.clone()),))
            .and_then(|result| to_batch(&result))
            .map_err(StreamError)?;
        if columns.num_rows() != chunk.len() {
            return Err(StreamError(invalid(format!(
                "func returned {} rows for a chunk of {}",
                columns.num_rows(),
                chunk.len()
            ))));
        }
        let schema = columns.schema();
        Ok(schema
            .fields()
            .iter()
            .zip(columns.columns())
            .map(|(f, c)| (f.name().clone(), c.clone()))
            .collect())
    })
    .map_err(|e| e.0)
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<BlockModelFile>()?;
    m.add_function(wrap_pyfunction!(map_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(write_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(read_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(read_csv, m)?)?;
    m.add_function(wrap_pyfunction!(write_csv, m)?)?;
    m.add_function(wrap_pyfunction!(read_gslib, m)?)?;
    m.add_function(wrap_pyfunction!(write_gslib, m)?)?;
    m.add_function(wrap_pyfunction!(read_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(write_mesh, m)?)?;
    Ok(())
}
