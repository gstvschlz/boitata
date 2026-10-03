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
#[pyo3(signature = (parts, *, total=1.0))]
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

/// Additive log-ratio against part `reference` (default the last).
#[pyfunction]
#[pyo3(signature = (parts, *, reference=None))]
fn alr<'py>(
    py: Python<'py>,
    parts: &Bound<PyAny>,
    reference: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    match reference {
        Some(r) => by_row(py, parts, |x| coda::alr_with(x, r)),
        None => by_row(py, parts, coda::alr),
    }
}

#[pyfunction]
#[pyo3(signature = (coords, *, reference=None))]
fn alr_inverse<'py>(
    py: Python<'py>,
    coords: &Bound<PyAny>,
    reference: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    match reference {
        Some(r) => by_row(py, coords, |y| coda::alr_with_inv(y, r)),
        None => by_row(py, coords, coda::alr_inv),
    }
}

/// Isometric log-ratio; `D` parts give `D - 1` coordinates, on `basis` (from
/// `partition_basis`) or the default pivot balances.
#[pyfunction]
#[pyo3(signature = (parts, *, basis=None))]
fn ilr<'py>(
    py: Python<'py>,
    parts: &Bound<PyAny>,
    basis: Option<&Bound<PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    match basis {
        Some(b) => {
            let b = rows(b, "basis")?;
            by_row(py, parts, |x| coda::ilr_with(x, &b))
        }
        None => by_row(py, parts, coda::ilr),
    }
}

#[pyfunction]
#[pyo3(signature = (coords, *, basis=None))]
fn ilr_inverse<'py>(
    py: Python<'py>,
    coords: &Bound<PyAny>,
    basis: Option<&Bound<PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    match basis {
        Some(b) => {
            let b = rows(b, "basis")?;
            by_row(py, coords, |y| coda::ilr_with_inv(y, &b))
        }
        None => by_row(py, coords, coda::ilr_inv),
    }
}

/// Orthonormal balances of a sequential binary partition, as the `basis` of
/// `ilr`.
///
/// Parameters
/// ----------
/// signs : array_like of int
///     ``(D - 1, D)``: each row a balance, 1 for the parts in its numerator,
///     -1 in its denominator, 0 outside it. Each row splits one group of the
///     rows before it in two (the first row all parts).
///
/// Returns
/// -------
/// ndarray
///     ``(D - 1, D)`` clr weights, one orthonormal row per balance.
///
/// Raises
/// ------
/// InvalidInput
///     If the rows do not form a sequential binary partition.
#[pyfunction]
fn partition_basis<'py>(py: Python<'py>, signs: Vec<Vec<i8>>) -> PyResult<Bound<'py, PyAny>> {
    Ok(array2(py, &coda::partition_basis(&signs).map_err(invalid)?).into_any())
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
    m.add_function(wrap_pyfunction!(partition_basis, m)?)?;
    Ok(())
}
