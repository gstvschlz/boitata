use eda::{Along, Direction, Method, Profile};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::args::{array1, array2, floats, holes, rows};
use crate::containers::coords_arg;
use crate::invalid;

fn optional_floats(obj: Option<&Bound<PyAny>>, what: &str) -> PyResult<Option<Vec<f64>>> {
    obj.map(|o| floats(o, what)).transpose()
}

fn profile<'py>(py: Python<'py>, p: Profile, key: &str) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item(key, array1(py, p.centres))?;
    d.set_item("mean", array1(py, p.mean))?;
    d.set_item("count", p.count)?;
    Ok(d)
}

/// Weighted (declustered) statistics of `values`; NaN is skipped.
///
/// Returns
/// -------
/// dict
///     ``n``, ``mean``, ``variance``, ``std``, ``cv``, ``min``, ``max`` and
///     ``quantiles`` at the given probabilities (mid-point cumulative weights).
#[pyfunction]
#[pyo3(signature = (values, weights=None, quantiles=vec![0.1, 0.25, 0.5, 0.75, 0.9]))]
fn describe<'py>(
    py: Python<'py>,
    values: &Bound<PyAny>,
    weights: Option<&Bound<PyAny>>,
    quantiles: Vec<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let w = optional_floats(weights, "weights")?;
    let s = eda::describe(&floats(values, "values")?, w.as_deref(), &quantiles).map_err(invalid)?;
    let d = PyDict::new(py);
    d.set_item("n", s.n)?;
    for (k, v) in [
        ("mean", s.mean),
        ("variance", s.variance),
        ("std", s.std),
        ("cv", s.cv),
        ("min", s.min),
        ("max", s.max),
    ] {
        d.set_item(k, v)?;
    }
    d.set_item("quantiles", array1(py, s.quantiles))?;
    Ok(d)
}

/// Mean of `values` per slice of `width` along `azimuth` (degrees from north)
/// or `axis` ("x", "y", "z"); slices start at 0 so swaths of samples and blocks
/// line up.
///
/// Returns
/// -------
/// dict
///     ``centres``, ``mean`` and ``count`` of the non-empty slices.
#[pyfunction]
#[pyo3(signature = (coords, values, width, azimuth=None, axis=None, weights=None))]
fn swath<'py>(
    py: Python<'py>,
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    width: f64,
    azimuth: Option<f64>,
    axis: Option<&str>,
    weights: Option<&Bound<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let along = match (azimuth, axis) {
        (Some(a), None) => Along::Azimuth(a),
        (None, Some(a)) => Along::Axis(
            ["x", "y", "z"]
                .iter()
                .position(|x| *x == a)
                .ok_or_else(|| invalid("axis must be 'x', 'y' or 'z'"))?,
        ),
        _ => return Err(invalid("give one of azimuth or axis")),
    };
    let w = optional_floats(weights, "weights")?;
    let p = eda::swath(
        &coords_arg(coords)?,
        &floats(values, "values")?,
        w.as_deref(),
        width,
        along,
    )
    .map_err(invalid)?;
    profile(py, p, "centres")
}

/// Mean of `values` against signed distance to the `inside`/`outside` contact
/// along each hole, negative inside.
///
/// Returns
/// -------
/// dict
///     ``distance`` (bin centres), ``mean`` and ``count``.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn contact<'py>(
    py: Python<'py>,
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    domains: &Bound<PyAny>,
    holes: &Bound<PyAny>,
    inside: &Bound<PyAny>,
    outside: &Bound<PyAny>,
    max_distance: f64,
    bin: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let coords = coords_arg(coords)?;
    let n = coords.len();
    let (_, hole_ids) = self::holes(Some(holes), n)?.expect("given");
    let (labels, ids) = self::holes(Some(domains), n)
        .map_err(|_| invalid("domains must be a 1-D sequence of labels"))?
        .expect("given");
    let code = |d: &Bound<PyAny>| -> PyResult<u32> {
        let label = d.str()?.to_string();
        labels
            .iter()
            .position(|l| *l == label)
            .map(|i| ids[i])
            .ok_or_else(|| invalid(format!("no sample in domain {label}")))
    };
    let p = eda::contact(
        &coords,
        &floats(values, "values")?,
        &ids,
        &hole_ids,
        code(inside)?,
        code(outside)?,
        max_distance,
        bin,
    )
    .map_err(invalid)?;
    profile(py, p, "distance")
}

/// Metal removed and statistics after capping at each of `caps` (default:
/// weighted quantiles 0.9, 0.95, 0.975, 0.99, 0.995, 0.999).
///
/// Returns
/// -------
/// dict
///     ``cap``, ``fraction`` of weight above it, ``metal_removed``
///     (1 - capped mean / mean), capped ``mean`` and ``cv``.
#[pyfunction]
#[pyo3(signature = (values, weights=None, caps=None))]
fn capping<'py>(
    py: Python<'py>,
    values: &Bound<PyAny>,
    weights: Option<&Bound<PyAny>>,
    caps: Option<&Bound<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let w = optional_floats(weights, "weights")?;
    let caps = optional_floats(caps, "caps")?;
    let r =
        eda::capping(&floats(values, "values")?, w.as_deref(), caps.as_deref()).map_err(invalid)?;
    let d = PyDict::new(py);
    let col = |f: fn(&eda::Cap) -> f64| array1(py, r.iter().map(f).collect());
    d.set_item("cap", col(|c| c.cap))?;
    d.set_item("fraction", col(|c| c.fraction))?;
    d.set_item("metal_removed", col(|c| c.metal_removed))?;
    d.set_item("mean", col(|c| c.mean))?;
    d.set_item("cv", col(|c| c.cv))?;
    Ok(d)
}

/// Head and tail values of pairs `lag ± tolerance` apart, and their correlation.
///
/// With `azimuth`, pairs are oriented within `angle_tolerance` of (`azimuth`,
/// `dip`); without, omnidirectional. Tails come from `other` when given.
#[pyfunction]
#[pyo3(signature = (coords, values, lag, tolerance, azimuth=None, dip=0.0, angle_tolerance=22.5, other=None))]
#[allow(clippy::too_many_arguments)]
fn h_scatter<'py>(
    py: Python<'py>,
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    lag: f64,
    tolerance: f64,
    azimuth: Option<f64>,
    dip: f64,
    angle_tolerance: f64,
    other: Option<&Bound<PyAny>>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, f64)> {
    let direction = azimuth.map(|azimuth| Direction {
        azimuth,
        dip,
        tolerance: angle_tolerance,
        bandwidth: None,
    });
    let other = optional_floats(other, "other")?;
    let (head, tail, r) = eda::h_scatter(
        &coords_arg(coords)?,
        &floats(values, "values")?,
        other.as_deref(),
        lag,
        tolerance,
        direction.as_ref(),
    )
    .map_err(invalid)?;
    Ok((array1(py, head).into_any(), array1(py, tail).into_any(), r))
}

/// Correlation matrix of the columns of `data`, each pair over rows without NaN.
#[pyfunction]
#[pyo3(signature = (data, weights=None, method="pearson"))]
fn correlation<'py>(
    py: Python<'py>,
    data: &Bound<PyAny>,
    weights: Option<&Bound<PyAny>>,
    method: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let method = match method {
        "pearson" => Method::Pearson,
        "spearman" => Method::Spearman,
        _ => return Err(invalid("method must be 'pearson' or 'spearman'")),
    };
    let rows = rows(data, "data")?;
    let d = rows.first().map_or(0, Vec::len);
    let columns: Vec<Vec<f64>> = (0..d)
        .map(|j| rows.iter().map(|r| r[j]).collect())
        .collect();
    let w = optional_floats(weights, "weights")?;
    let r = eda::correlation(&columns, w.as_deref(), method).map_err(invalid)?;
    Ok(array2(py, &r).into_any())
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(describe, m)?)?;
    m.add_function(wrap_pyfunction!(swath, m)?)?;
    m.add_function(wrap_pyfunction!(contact, m)?)?;
    m.add_function(wrap_pyfunction!(capping, m)?)?;
    m.add_function(wrap_pyfunction!(h_scatter, m)?)?;
    m.add_function(wrap_pyfunction!(correlation, m)?)?;
    Ok(())
}
