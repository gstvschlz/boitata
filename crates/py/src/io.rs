use std::path::PathBuf;

use ceres_io::{CsvOptions, NODATA};
use pyo3::prelude::*;

use crate::containers::{PyBlockModel, PyPointSet};
use crate::table::{Table, to_batch};
use crate::{error, invalid};

fn io_error(e: ceres_io::Error) -> PyErr {
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

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(write_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(read_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(read_csv, m)?)?;
    m.add_function(wrap_pyfunction!(write_csv, m)?)?;
    m.add_function(wrap_pyfunction!(read_gslib, m)?)?;
    m.add_function(wrap_pyfunction!(write_gslib, m)?)?;
    Ok(())
}
