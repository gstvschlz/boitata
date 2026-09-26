use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray, UInt64Array};
use eda::{Along, Direction, Method, Profile};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::args::{array1, array2, floats, holes, rows};
use crate::containers::coords_arg;
use crate::invalid;
use crate::table::Table;

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

/// Sorted category labels and each value's index into them.
fn categories(obj: &Bound<PyAny>, n: usize) -> PyResult<(Vec<String>, Vec<u32>)> {
    let (labels, _) = holes(Some(obj), n)
        .map_err(|_| invalid("categories must be a 1-D sequence of labels"))?
        .expect("given");
    let mut names = labels.clone();
    names.sort();
    names.dedup();
    let codes = labels
        .iter()
        .map(|l| names.binary_search(l).expect("listed") as u32)
        .collect();
    Ok((names, codes))
}

fn nullable(values: impl IntoIterator<Item = f64>) -> ArrayRef {
    Arc::new(
        values
            .into_iter()
            .map(|v| v.is_finite().then_some(v))
            .collect::<Float64Array>(),
    )
}

/// First column the category labels, "all" for the all-data row.
fn table<T>(
    label: &str,
    names: &[String],
    rows: &[(Option<u32>, T)],
    columns: Vec<(String, ArrayRef)>,
) -> PyResult<Table> {
    let labels: Vec<&str> = rows
        .iter()
        .map(|(c, _)| c.map_or("all", |c| names[c as usize].as_str()))
        .collect();
    let mut all: Vec<(String, ArrayRef)> =
        vec![(label.into(), Arc::new(StringArray::from(labels)))];
    all.extend(columns);
    Ok(Table(RecordBatch::try_from_iter(all).map_err(invalid)?))
}

/// Weighted statistics per category, as `describe`, then over all values.
///
/// Parameters
/// ----------
/// values : array_like
///     Values; NaN is skipped.
/// categories : array_like
///     Category (e.g. domain) of each value, int or str.
/// weights : array_like, optional
///     Declustering weights.
/// quantiles : sequence of float
///     Probabilities of the quantile columns.
///
/// Returns
/// -------
/// Table
///     ``category`` (sorted, then ``"all"``), ``n``, ``mean``, ``variance``,
///     ``std``, ``cv``, ``min``, ``max`` and one column per quantile named
///     ``P10``, ``P97.5``, ...
#[pyfunction]
#[pyo3(signature = (values, categories, weights=None, quantiles=vec![0.1, 0.25, 0.5, 0.75, 0.9]))]
fn describe_by(
    values: &Bound<PyAny>,
    categories: &Bound<PyAny>,
    weights: Option<&Bound<PyAny>>,
    quantiles: Vec<f64>,
) -> PyResult<Table> {
    let values = floats(values, "values")?;
    let (names, codes) = self::categories(categories, values.len())?;
    let w = optional_floats(weights, "weights")?;
    let rows = eda::describe_by(&values, &codes, w.as_deref(), &quantiles).map_err(invalid)?;
    let col = |f: &dyn Fn(&eda::Summary) -> f64| nullable(rows.iter().map(|(_, s)| f(s)));
    let mut columns: Vec<(String, ArrayRef)> = vec![(
        "n".into(),
        Arc::new(UInt64Array::from_iter_values(
            rows.iter().map(|(_, s)| s.n as u64),
        )),
    )];
    for (k, f) in [
        ("mean", (|s| s.mean) as fn(&eda::Summary) -> f64),
        ("variance", |s| s.variance),
        ("std", |s| s.std),
        ("cv", |s| s.cv),
        ("min", |s| s.min),
        ("max", |s| s.max),
    ] {
        columns.push((k.into(), col(&f)));
    }
    for (j, p) in quantiles.iter().enumerate() {
        let name = format!("{:.4}", 100.0 * p);
        let name = name.trim_end_matches('0').trim_end_matches('.');
        columns.push((format!("P{name}"), col(&|s| s.quantiles[j])));
    }
    table("category", &names, &rows, columns)
}

/// Grade-tonnage curve straight from the data: tonnage, mean grade and metal at
/// or above each cutoff.
///
/// Each value stands for ``weights × density`` tonnes (1 when both are
/// omitted), e.g. block volumes and densities, or composite lengths.
///
/// Parameters
/// ----------
/// values : array_like
///     Grades; NaN is skipped.
/// cutoffs : array_like
///     Cutoff grades; ``-inf`` keeps everything.
/// weights : array_like, optional
///     Volume or declustering weight of each value.
/// density : array_like, optional
///     Density of each value.
///
/// Returns
/// -------
/// dict
///     ``cutoff``, ``tonnage``, ``mean_grade`` (NaN when nothing is above) and
///     ``metal`` (tonnage × mean grade).
#[pyfunction]
#[pyo3(signature = (values, cutoffs, weights=None, density=None))]
fn grade_tonnage<'py>(
    py: Python<'py>,
    values: &Bound<PyAny>,
    cutoffs: &Bound<PyAny>,
    weights: Option<&Bound<PyAny>>,
    density: Option<&Bound<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let w = optional_floats(weights, "weights")?;
    let density = optional_floats(density, "density")?;
    let r = eda::grade_tonnage(
        &floats(values, "values")?,
        w.as_deref(),
        density.as_deref(),
        &floats(cutoffs, "cutoffs")?,
    )
    .map_err(invalid)?;
    let d = PyDict::new(py);
    let col = |f: fn(&eda::Tonnage) -> f64| array1(py, r.iter().map(f).collect());
    d.set_item("cutoff", col(|t| t.cutoff))?;
    d.set_item("tonnage", col(|t| t.tonnage))?;
    d.set_item("mean_grade", col(|t| t.mean_grade))?;
    d.set_item("metal", col(|t| t.metal))?;
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

/// Weighted statistics per domain before and after capping, and the metal removed.
///
/// Parameters
/// ----------
/// values : array_like
///     Values; NaN is skipped.
/// domains : array_like
///     Domain of each value, int or str.
/// caps : dict
///     Cap per domain; domains left out are not capped.
/// weights : array_like, optional
///     Declustering weights.
///
/// Returns
/// -------
/// Table
///     ``domain`` (sorted, then ``"all"``), ``cap`` (null when not capped),
///     ``n``, ``n_capped`` (values above the cap), ``mean``, ``std``, ``cv``,
///     ``max``, the same after capping as ``mean_capped``, ``std_capped``,
///     ``cv_capped``, ``max_capped``, and ``metal_removed``, the sum of
///     ``(value - cap) × weight`` above the cap; ``1 - mean_capped / mean`` is
///     the fraction removed.
#[pyfunction]
#[pyo3(signature = (values, domains, caps, weights=None))]
fn capping_report(
    values: &Bound<PyAny>,
    domains: &Bound<PyAny>,
    caps: &Bound<PyDict>,
    weights: Option<&Bound<PyAny>>,
) -> PyResult<Table> {
    let values = floats(values, "values")?;
    let (names, codes) = categories(domains, values.len())?;
    let mut by_domain = vec![f64::INFINITY; names.len()];
    for (k, cap) in caps.iter() {
        let k = k.str()?.to_string();
        let i = names
            .binary_search(&k)
            .map_err(|_| invalid(format!("no value in domain {k}")))?;
        by_domain[i] = cap.extract()?;
    }
    let w = optional_floats(weights, "weights")?;
    let rows = eda::capping_report(&values, &codes, w.as_deref(), &by_domain).map_err(invalid)?;
    let count = |f: fn(&eda::CapReport) -> usize| -> ArrayRef {
        Arc::new(UInt64Array::from_iter_values(
            rows.iter().map(|(_, r)| f(r) as u64),
        ))
    };
    let col = |f: fn(&eda::CapReport) -> f64| nullable(rows.iter().map(|(_, r)| f(r)));
    let columns: Vec<(String, ArrayRef)> = vec![
        ("cap".into(), col(|r| r.cap)),
        ("n".into(), count(|r| r.before.n)),
        ("n_capped".into(), count(|r| r.capped)),
        ("mean".into(), col(|r| r.before.mean)),
        ("std".into(), col(|r| r.before.std)),
        ("cv".into(), col(|r| r.before.cv)),
        ("max".into(), col(|r| r.before.max)),
        ("mean_capped".into(), col(|r| r.after.mean)),
        ("std_capped".into(), col(|r| r.after.std)),
        ("cv_capped".into(), col(|r| r.after.cv)),
        ("max_capped".into(), col(|r| r.after.max)),
        ("metal_removed".into(), col(|r| r.metal_removed)),
    ];
    table("domain", &names, &rows, columns)
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
    m.add_function(wrap_pyfunction!(describe_by, m)?)?;
    m.add_function(wrap_pyfunction!(grade_tonnage, m)?)?;
    m.add_function(wrap_pyfunction!(capping_report, m)?)?;
    m.add_function(wrap_pyfunction!(swath, m)?)?;
    m.add_function(wrap_pyfunction!(contact, m)?)?;
    m.add_function(wrap_pyfunction!(capping, m)?)?;
    m.add_function(wrap_pyfunction!(h_scatter, m)?)?;
    m.add_function(wrap_pyfunction!(correlation, m)?)?;
    Ok(())
}
