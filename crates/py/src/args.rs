use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::containers::coords_arg;
use crate::invalid;

pub type Point = (f64, f64, f64);

fn asarray<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    obj.py()
        .import("numpy")?
        .call_method1("asarray", (obj, "float64"))
}

/// 1-D array-like of floats.
pub fn floats(obj: &Bound<PyAny>, what: &str) -> PyResult<Vec<f64>> {
    let array: PyReadonlyArray1<f64> = asarray(obj)?
        .extract()
        .map_err(|_| invalid(format!("{what} must be a 1-D numeric array")))?;
    Ok(array.as_array().to_vec())
}

/// 1-D finite floats, e.g. sample values.
pub fn finite(obj: &Bound<PyAny>, what: &str) -> PyResult<Vec<f64>> {
    let values = floats(obj, what)?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid(format!(
            "{what} must be finite; drop missing values first"
        )));
    }
    Ok(values)
}

pub fn optional_finite(obj: Option<&Bound<PyAny>>, what: &str) -> PyResult<Option<Vec<f64>>> {
    obj.map(|o| finite(o, what)).transpose()
}

/// `(n, 2)` or `(n, 3)` coordinates as tuples.
pub fn points(obj: &Bound<PyAny>) -> PyResult<Vec<Point>> {
    Ok(coords_arg(obj)?
        .into_iter()
        .map(|[x, y, z]| (x, y, z))
        .collect())
}

pub fn same_length(n: usize, m: usize, what: &str) -> PyResult<()> {
    if n == m {
        Ok(())
    } else {
        Err(invalid(format!("{what}: expected {n} values, got {m}")))
    }
}

/// `(n, d)` array-like as rows.
pub fn rows(obj: &Bound<PyAny>, what: &str) -> PyResult<Vec<Vec<f64>>> {
    let array: PyReadonlyArray2<f64> = asarray(obj)?
        .extract()
        .map_err(|_| invalid(format!("{what} must be a 2-D numeric array")))?;
    Ok(array
        .as_array()
        .rows()
        .into_iter()
        .map(|r| r.to_vec())
        .collect())
}

pub fn array1<'py>(py: Python<'py>, values: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, values)
}

pub fn array2<'py>(py: Python<'py>, rows: &[Vec<f64>]) -> Bound<'py, PyArray2<f64>> {
    let cols = rows.first().map_or(0, Vec::len);
    Array2::from_shape_vec((rows.len(), cols), rows.concat())
        .expect("rectangular rows")
        .into_pyarray(py)
}

pub fn points_array<'py>(py: Python<'py>, points: &[Point]) -> Bound<'py, PyArray2<f64>> {
    let rows: Vec<Vec<f64>> = points.iter().map(|p| vec![p.0, p.1, p.2]).collect();
    array2(py, &rows)
}

pub fn triple(values: Vec<f64>, fill: f64, what: &str) -> PyResult<Point> {
    match values[..] {
        [x, y] => Ok((x, y, fill)),
        [x, y, z] => Ok((x, y, z)),
        _ => Err(invalid(format!("{what} needs 2 or 3 values"))),
    }
}
