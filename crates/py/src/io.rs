use std::path::PathBuf;

use ceres_io::{CsvOptions, NODATA};
use pyo3::prelude::*;

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

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_csv, m)?)?;
    m.add_function(wrap_pyfunction!(write_csv, m)?)?;
    m.add_function(wrap_pyfunction!(read_gslib, m)?)?;
    m.add_function(wrap_pyfunction!(write_gslib, m)?)?;
    Ok(())
}
