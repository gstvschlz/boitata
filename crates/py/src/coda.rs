use pyo3::prelude::*;

use crate::args::{array1, array2, rows, same_length};
use crate::invalid;

fn by_row<'py>(
    py: Python<'py>,
    data: &Bound<PyAny>,
    f: impl Fn(&[f64]) -> coda::Result<Vec<f64>>,
) -> PyResult<Bound<'py, PyAny>> {
    let out = rows(data, "compositions")?
        .iter()
        .map(|r| f(r))
        .collect::<Result<Vec<_>, _>>()
        .map_err(invalid)?;
    Ok(array2(py, &out).into_any())
}

/// Rescales each row of parts to sum to `total`.
#[pyfunction]
#[pyo3(signature = (parts, total=1.0))]
fn closure<'py>(py: Python<'py>, parts: &Bound<PyAny>, total: f64) -> PyResult<Bound<'py, PyAny>> {
    by_row(py, parts, |r| coda::closure(r, total))
}

/// Centerd log-ratio of each row.
#[pyfunction]
fn clr<'py>(py: Python<'py>, parts: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
    by_row(py, parts, coda::clr)
}

#[pyfunction]
fn clr_inverse<'py>(py: Python<'py>, coords: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
    by_row(py, coords, coda::clr_inv)
}

/// Additive log-ratio against the last part.
#[pyfunction]
fn alr<'py>(py: Python<'py>, parts: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
    by_row(py, parts, coda::alr)
}

#[pyfunction]
fn alr_inverse<'py>(py: Python<'py>, coords: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
    by_row(py, coords, coda::alr_inv)
}

/// Isometric log-ratio; `D` parts give `D - 1` coordinates.
#[pyfunction]
fn ilr<'py>(py: Python<'py>, parts: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
    by_row(py, parts, coda::ilr)
}

#[pyfunction]
fn ilr_inverse<'py>(py: Python<'py>, coords: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
    by_row(py, coords, coda::ilr_inv)
}

/// Aitchison distance between paired rows of `a` and `b`.
#[pyfunction]
fn aitchison_distance<'py>(
    py: Python<'py>,
    a: &Bound<PyAny>,
    b: &Bound<PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let (a, b) = (rows(a, "a")?, rows(b, "b")?);
    same_length(a.len(), b.len(), "b")?;
    let d = a
        .iter()
        .zip(&b)
        .map(|(a, b)| coda::aitchison_distance(a, b))
        .collect::<Result<Vec<_>, _>>()
        .map_err(invalid)?;
    Ok(array1(py, d).into_any())
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(closure, m)?)?;
    m.add_function(wrap_pyfunction!(clr, m)?)?;
    m.add_function(wrap_pyfunction!(clr_inverse, m)?)?;
    m.add_function(wrap_pyfunction!(alr, m)?)?;
    m.add_function(wrap_pyfunction!(alr_inverse, m)?)?;
    m.add_function(wrap_pyfunction!(ilr, m)?)?;
    m.add_function(wrap_pyfunction!(ilr_inverse, m)?)?;
    m.add_function(wrap_pyfunction!(aitchison_distance, m)?)?;
    Ok(())
}
