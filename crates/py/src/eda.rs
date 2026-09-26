use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::DataType;
use ceres_core::PointSet;
use eda::{Along, Direction, Merge, Method, Profile};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::args::{array1, array2, floats, holes, rows};
use crate::containers::{PyPointSet, coords_arg};
use crate::invalid;
use crate::table::Table;

fn optional_floats(obj: Option<&Bound<PyAny>>, what: &str) -> PyResult<Option<Vec<f64>>> {
    obj.map(|o| floats(o, what)).transpose()
}

fn profile<'py>(py: Python<'py>, p: Profile, key: &str) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item(key, array1(py, p.centers))?;
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
/// categories : array_like, optional
///     Category (e.g. domain) of each value, int or str; adds a curve per
///     category.
///
/// Returns
/// -------
/// Table
///     ``category`` (with `categories`: sorted, then ``"all"``), ``cutoff``,
///     ``tonnage``, ``mean_grade`` (null when nothing is above) and ``metal``
///     (tonnage × mean grade).
#[pyfunction]
#[pyo3(signature = (values, cutoffs, weights=None, density=None, categories=None))]
fn grade_tonnage(
    values: &Bound<PyAny>,
    cutoffs: &Bound<PyAny>,
    weights: Option<&Bound<PyAny>>,
    density: Option<&Bound<PyAny>>,
    categories: Option<&Bound<PyAny>>,
) -> PyResult<Table> {
    let values = floats(values, "values")?;
    let w = optional_floats(weights, "weights")?;
    let density = optional_floats(density, "density")?;
    let cutoffs = floats(cutoffs, "cutoffs")?;
    let (names, rows) = match categories {
        None => {
            let r = eda::grade_tonnage(&values, w.as_deref(), density.as_deref(), &cutoffs);
            (vec![], vec![(None, r.map_err(invalid)?)])
        }
        Some(c) => {
            let (names, codes) = self::categories(c, values.len())?;
            let rows =
                eda::grade_tonnage_by(&values, &codes, w.as_deref(), density.as_deref(), &cutoffs)
                    .map_err(invalid)?;
            (names, rows)
        }
    };
    let flat: Vec<(Option<u32>, &eda::Tonnage)> = rows
        .iter()
        .flat_map(|(c, r)| r.iter().map(move |t| (*c, t)))
        .collect();
    let columns = tonnage_columns(flat.iter().map(|r| r.1));
    if categories.is_none() {
        return Ok(Table(RecordBatch::try_from_iter(columns).map_err(invalid)?));
    }
    table("category", &names, &flat, columns)
}

fn tonnage_columns<'a>(
    rows: impl Iterator<Item = &'a eda::Tonnage> + Clone,
) -> Vec<(String, ArrayRef)> {
    let col = |f: fn(&eda::Tonnage) -> f64| rows.clone().map(f);
    vec![
        (
            "cutoff".into(),
            Arc::new(Float64Array::from_iter_values(col(|t| t.cutoff))) as ArrayRef,
        ),
        ("tonnage".into(), nullable(col(|t| t.tonnage))),
        ("mean_grade".into(), nullable(col(|t| t.mean_grade))),
        ("metal".into(), nullable(col(|t| t.metal))),
    ]
}

/// Grade-tonnage of several models of the same blocks, by cutoff and category,
/// against a reference model.
///
/// Each block stands for ``volume × density`` tonnes (1 each by default), as in
/// `grade_tonnage`.
///
/// Parameters
/// ----------
/// models : dict of str to array_like
///     Block grades of each model, all on the same blocks; NaN is skipped.
/// cutoffs : array_like
///     Cutoff grades; ``-inf`` keeps everything.
/// categories : array_like, optional
///     Category (e.g. class or domain) of each block, int or str.
/// reference : str, optional
///     Model the others are compared with; the first by default.
/// volume, density : float or array_like, optional
///     Block volumes and densities.
///
/// Returns
/// -------
/// Table
///     One row per category (sorted, then ``"all"``), cutoff and model:
///     ``category``, ``cutoff``, ``model``, ``tonnage``, ``mean_grade`` and
///     ``metal`` at or above the cutoff, and ``tonnage_diff``, ``grade_diff``
///     and ``metal_diff``, each ``value / reference value - 1``.
#[pyfunction]
#[pyo3(signature = (models, cutoffs, categories=None, reference=None, volume=None, density=None))]
fn compare_models(
    models: &Bound<PyAny>,
    cutoffs: &Bound<PyAny>,
    categories: Option<&Bound<PyAny>>,
    reference: Option<&str>,
    volume: Option<&Bound<PyAny>>,
    density: Option<&Bound<PyAny>>,
) -> PyResult<Table> {
    let (labels, grades) = models
        .call_method0("items")
        .map_err(|_| invalid("models must be a dict of name to grades"))?
        .try_iter()?
        .map(|item| {
            let (name, values): (String, Bound<PyAny>) = item?.extract()?;
            Ok((name.clone(), floats(&values, &name)?))
        })
        .collect::<PyResult<(Vec<String>, Vec<Vec<f64>>)>>()?;
    let reference = match reference {
        None => 0,
        Some(r) => labels
            .iter()
            .position(|l| l == r)
            .ok_or_else(|| invalid(format!("reference {r:?} is not one of the models")))?,
    };
    let n = grades.first().map_or(0, Vec::len);
    let (names, codes) = categories
        .map(|c| self::categories(c, n))
        .transpose()?
        .unwrap_or_default();
    let tonnes = block_tonnes(volume, density, n)?;
    let slices: Vec<&[f64]> = grades.iter().map(Vec::as_slice).collect();
    let rows = eda::compare_models(
        &slices,
        categories.map(|_| &codes[..]),
        tonnes.as_deref(),
        &floats(cutoffs, "cutoffs")?,
        reference,
    )
    .map_err(invalid)?;
    let mut columns = tonnage_columns(rows.iter().map(|r| &r.tonnage));
    let model = rows.iter().map(|r| labels[r.model].as_str());
    columns.insert(
        1,
        (
            "model".into(),
            Arc::new(StringArray::from_iter_values(model)),
        ),
    );
    let col = |f: fn(&eda::Comparison) -> f64| nullable(rows.iter().map(f));
    columns.push(("tonnage_diff".into(), col(|r| r.tonnage_diff)));
    columns.push(("grade_diff".into(), col(|r| r.grade_diff)));
    columns.push(("metal_diff".into(), col(|r| r.metal_diff)));
    let keyed: Vec<(Option<u32>, ())> = rows.iter().map(|r| (r.category, ())).collect();
    table("category", &names, &keyed, columns)
}

/// `volume × density` per block, or `None` when both are omitted.
fn block_tonnes(
    volume: Option<&Bound<PyAny>>,
    density: Option<&Bound<PyAny>>,
    n: usize,
) -> PyResult<Option<Vec<f64>>> {
    if volume.is_none() && density.is_none() {
        return Ok(None);
    }
    let one =
        |o: Option<&Bound<PyAny>>, what| o.map_or(Ok(vec![1.0; n]), |o| per_value(o, n, what));
    let (v, d) = (one(volume, "volume")?, one(density, "density")?);
    Ok(Some(v.iter().zip(&d).map(|(v, d)| v * d).collect()))
}

/// Mean of `values` per slice of `width` along `azimuth` (degrees from north)
/// or `axis` ("x", "y", "z"); slices start at 0 so swaths of samples and blocks
/// line up.
///
/// Each value stands for ``weights × density`` tonnes, as in `grade_tonnage`,
/// so slice tonnages and metals add up to the totals at cutoff ``-inf``.
///
/// Returns
/// -------
/// dict
///     ``center``, ``mean`` (weighted by `weights`), ``count``, ``tonnage``
///     and ``metal`` of the non-empty slices.
#[pyfunction]
#[pyo3(signature = (coords, values, width, azimuth=None, axis=None, weights=None, density=None))]
#[allow(clippy::too_many_arguments)]
fn swath<'py>(
    py: Python<'py>,
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    width: f64,
    azimuth: Option<f64>,
    axis: Option<&str>,
    weights: Option<&Bound<PyAny>>,
    density: Option<&Bound<PyAny>>,
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
    let values = floats(values, "values")?;
    let w = optional_floats(weights, "weights")?;
    let density = density
        .map(|d| per_value(d, values.len(), "density"))
        .transpose()?;
    let p = eda::swath(
        &coords_arg(coords)?,
        &values,
        w.as_deref(),
        density.as_deref(),
        width,
        along,
    )
    .map_err(invalid)?;
    let (tonnage, metal) = (p.tonnage.clone(), p.metal.clone());
    let d = profile(py, p, "center")?;
    d.set_item("tonnage", array1(py, tonnage))?;
    d.set_item("metal", array1(py, metal))?;
    Ok(d)
}

/// A constant or one value each.
fn per_value(obj: &Bound<PyAny>, n: usize, what: &str) -> PyResult<Vec<f64>> {
    match obj.extract::<f64>() {
        Ok(x) => Ok(vec![x; n]),
        Err(_) => floats(obj, what),
    }
}

/// Statistics of a block model against the data it was estimated from, per
/// domain and over all.
///
/// Parameters
/// ----------
/// model : array_like
///     Block grades; NaN is skipped.
/// data : array_like
///     Sample (composite) grades; NaN is skipped.
/// weights : array_like, optional
///     Declustering weights of `data`; adds a ``declustered`` row that the
///     model is compared with.
/// domains : tuple of array_like, optional
///     Domain labels of the blocks and of the data.
/// volume, density : float or array_like, optional
///     Block volumes and densities; blocks are weighted by their tonnage,
///     ``volume × density`` (1 each by default).
/// reference : array_like, optional
///     Other grades on the same blocks, e.g. the truth or a previous model.
///
/// Returns
/// -------
/// Table
///     One row per domain (sorted, then ``"all"``) and source (``naive`` and
///     ``declustered`` data, ``model``, ``reference``): ``n``, ``tonnage``
///     (null for data), ``mean``, ``variance``, ``cv``, ``P10``, ``P50``,
///     ``P90``, ``mean_diff`` (``mean / data mean - 1``) and ``variance_ratio``
///     (``variance / data variance``), against the declustered data when
///     `weights` are given.
#[pyfunction]
#[pyo3(signature = (model, data, weights=None, domains=None, volume=None, density=None, reference=None))]
fn validate_model(
    model: &Bound<PyAny>,
    data: &Bound<PyAny>,
    weights: Option<&Bound<PyAny>>,
    domains: Option<&Bound<PyAny>>,
    volume: Option<&Bound<PyAny>>,
    density: Option<&Bound<PyAny>>,
    reference: Option<&Bound<PyAny>>,
) -> PyResult<Table> {
    let (model, data) = (floats(model, "model")?, floats(data, "data")?);
    let w = optional_floats(weights, "weights")?;
    let reference = optional_floats(reference, "reference")?;
    let n = model.len();
    let tonnes = block_tonnes(volume, density, n)?;
    let (names, codes) = match domains {
        None => (vec![], None),
        Some(obj) => {
            let (om, od) = two(obj, "domains")?;
            let (lm, _) = holes(Some(&om), n)?.expect("given");
            let (ld, _) = holes(Some(&od), data.len())?.expect("given");
            let mut names: Vec<String> = lm.iter().chain(&ld).cloned().collect();
            names.sort();
            names.dedup();
            let code = |l: &String| names.binary_search(l).expect("listed") as u32;
            let codes: (Vec<u32>, Vec<u32>) =
                (lm.iter().map(code).collect(), ld.iter().map(code).collect());
            (names, Some(codes))
        }
    };
    let rows = eda::validate_model(
        &model,
        &data,
        w.as_deref(),
        codes.as_ref().map(|(m, d)| (&m[..], &d[..])),
        tonnes.as_deref(),
        reference.as_deref(),
        &[0.1, 0.5, 0.9],
    )
    .map_err(invalid)?;
    let source = rows.iter().map(|r| match r.source {
        eda::Source::Naive => "naive",
        eda::Source::Declustered => "declustered",
        eda::Source::Model => "model",
        eda::Source::Reference => "reference",
    });
    let col = |f: &dyn Fn(&eda::Validation) -> f64| nullable(rows.iter().map(f));
    let mut columns: Vec<(String, ArrayRef)> = vec![
        (
            "source".into(),
            Arc::new(StringArray::from_iter_values(source)),
        ),
        (
            "n".into(),
            Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|r| r.summary.n as u64),
            )),
        ),
        ("tonnage".into(), col(&|r| r.tonnage)),
        ("mean".into(), col(&|r| r.summary.mean)),
        ("variance".into(), col(&|r| r.summary.variance)),
        ("cv".into(), col(&|r| r.summary.cv)),
    ];
    for (j, name) in ["P10", "P50", "P90"].into_iter().enumerate() {
        columns.push((name.into(), col(&|r| r.summary.quantiles[j])));
    }
    columns.push(("mean_diff".into(), col(&|r| r.mean_diff)));
    columns.push(("variance_ratio".into(), col(&|r| r.variance_ratio)));
    let keyed: Vec<(Option<u32>, ())> = rows.iter().map(|r| (r.domain, ())).collect();
    table("domain", &names, &keyed, columns)
}

/// Mean of `values` against signed distance to the `inside`/`outside` contact
/// along each hole, negative inside.
///
/// Returns
/// -------
/// dict
///     ``distance`` (bin centers), ``mean`` and ``count``.
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
///
/// Parameters
/// ----------
/// data : array_like
///     ``(n, d)`` values.
/// weights : array_like, optional
///     Declustering weights.
/// method : {"pearson", "spearman", "covariance"}
///     Pearson correlation, rank correlation, or the covariance (over the sum
///     of weights, as the variance of `describe`), variances on the diagonal.
///
/// Returns
/// -------
/// ndarray
///     ``(d, d)`` symmetric matrix.
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
        "covariance" => Method::Covariance,
        _ => {
            return Err(invalid(
                "method must be 'pearson', 'spearman' or 'covariance'",
            ));
        }
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

/// Samples at most `tolerance` apart, grouped, and optionally merged.
///
/// Grouping is transitive: samples further apart share a group when a chain of
/// close samples links them. Groups are numbered by their first row.
///
/// Parameters
/// ----------
/// points : PointSet or array_like
///     Samples, or their ``(n, 2)`` or ``(n, 3)`` coordinates.
/// tolerance : float
///     Largest distance between duplicates; 0 groups samples at exactly the
///     same location.
/// merge : {"mean", "first", "max"}, optional
///     Merge each group into one sample at its first sample's location; needs
///     a PointSet. Numeric columns become float: ``mean`` is weighted by
///     `weights` (by count without), ``max`` is each column's own maximum,
///     both skipping nulls; ``first`` is the first sample's value. Other
///     columns take the first sample's value.
/// weights : array_like, optional
///     Weights of ``merge="mean"``, e.g. composite lengths.
///
/// Returns
/// -------
/// report : Table
///     Without `merge`: one row per group, ``group``, ``n`` samples, ``x``,
///     ``y``, ``z`` of the first sample and ``spread``, the largest distance
///     from it.
/// group : ndarray of int64
///     Without `merge`: the group of each sample, -1 for samples alone.
/// merged : PointSet
///     With `merge`, instead: the samples alone and one per group, in row
///     order, with an ``n`` column counting the samples merged into each.
#[pyfunction]
#[pyo3(signature = (points, tolerance=0.0, merge=None, weights=None))]
fn duplicates<'py>(
    py: Python<'py>,
    points: &Bound<'py, PyAny>,
    tolerance: f64,
    merge: Option<&str>,
    weights: Option<&Bound<PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let set = points.cast::<PyPointSet>().ok().map(|p| p.get().0.clone());
    let coords = match &set {
        Some(p) => p.coords().to_vec(),
        None => coords_arg(points)?,
    };
    let n = coords.len();
    let groups = eda::duplicates(&coords, tolerance).map_err(invalid)?;
    let Some(merge) = merge else {
        if weights.is_some() {
            return Err(invalid("weights are only used with merge='mean'"));
        }
        let mut group = vec![-1i64; n];
        for (k, g) in groups.iter().enumerate() {
            g.iter().for_each(|&i| group[i] = k as i64);
        }
        let first = |c: usize| -> ArrayRef {
            Arc::new(Float64Array::from_iter_values(
                groups.iter().map(|g| coords[g[0]][c]),
            ))
        };
        let spread = groups.iter().map(|g| {
            g.iter()
                .map(|&i| {
                    (0..3)
                        .map(|c| (coords[i][c] - coords[g[0]][c]).powi(2))
                        .sum::<f64>()
                })
                .fold(0.0, f64::max)
                .sqrt()
        });
        let report = RecordBatch::try_from_iter([
            (
                "group",
                Arc::new(UInt64Array::from_iter_values(0..groups.len() as u64)) as ArrayRef,
            ),
            (
                "n",
                Arc::new(UInt64Array::from_iter_values(
                    groups.iter().map(|g| g.len() as u64),
                )),
            ),
            ("x", first(0)),
            ("y", first(1)),
            ("z", first(2)),
            ("spread", Arc::new(Float64Array::from_iter_values(spread))),
        ])
        .map_err(invalid)?;
        let group = numpy::PyArray1::from_vec(py, group);
        return (Table(report), group).into_bound_py_any(py);
    };
    let rule = match merge {
        "mean" => Merge::Mean,
        "first" => Merge::First,
        "max" => Merge::Max,
        _ => return Err(invalid("merge must be 'mean', 'first' or 'max'")),
    };
    let set = set.ok_or_else(|| invalid("merge needs a PointSet"))?;
    let w = optional_floats(weights, "weights")?;
    if w.is_some() && rule != Merge::Mean {
        return Err(invalid("weights are only used with merge='mean'"));
    }
    let batch = set.attributes();
    if batch.column_by_name("n").is_some() {
        return Err(invalid("points already have a column 'n'"));
    }
    let mut dropped = vec![false; n];
    let mut count = vec![1u64; n];
    for g in &groups {
        g[1..].iter().for_each(|&i| dropped[i] = true);
        count[g[0]] = g.len() as u64;
    }
    let keep: Vec<u64> = (0..n as u64).filter(|&i| !dropped[i as usize]).collect();
    let take = UInt64Array::from(keep.clone());
    let mut columns: Vec<(String, ArrayRef)> = Vec::new();
    for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
        let merged: ArrayRef = if field.data_type().is_numeric() {
            let column = arrow_cast::cast(column, &DataType::Float64).map_err(invalid)?;
            let values: Vec<f64> = column
                .as_primitive::<Float64Type>()
                .iter()
                .map(|v| v.unwrap_or(f64::NAN))
                .collect();
            let merged = eda::merge_duplicates(&values, &groups, rule, w.as_deref())
                .map_err(|e| invalid(format!("column {}: {e}", field.name())))?;
            nullable(merged)
        } else {
            arrow_select::take::take(column, &take, None).map_err(invalid)?
        };
        columns.push((field.name().clone(), merged));
    }
    columns.push((
        "n".into(),
        Arc::new(UInt64Array::from_iter_values(
            keep.iter().map(|&i| count[i as usize]),
        )),
    ));
    let attributes = RecordBatch::try_from_iter(columns).map_err(invalid)?;
    let coords = keep.iter().map(|&i| coords[i as usize]).collect();
    let mut merged = PointSet::new(coords, attributes).map_err(invalid)?;
    merged.crs = set.crs.clone();
    PyPointSet(merged).into_bound_py_any(py)
}

fn two<'py>(
    obj: &Bound<'py, PyAny>,
    what: &str,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    obj.extract()
        .map_err(|_| invalid(format!("{what} must be a pair: one for a, one for b")))
}

fn set_coords(obj: &Bound<PyAny>) -> PyResult<Vec<[f64; 3]>> {
    match obj.cast::<PyPointSet>() {
        Ok(p) => Ok(p.get().0.coords().to_vec()),
        Err(_) => coords_arg(obj),
    }
}

/// Nearest pairs between two sets of samples within `max_distance`, e.g. twin
/// holes or two drilling types.
///
/// Parameters
/// ----------
/// a, b : PointSet or array_like
///     Samples, or their ``(n, 2)`` or ``(n, 3)`` coordinates.
/// max_distance : float
///     Largest pairing distance.
/// values : tuple of array_like, optional
///     Values of `a` and of `b`; samples with NaN are not paired.
/// unique : bool
///     Pair each sample at most once, the closest pairs first; else each
///     sample of `a` takes its nearest of `b`, which may repeat.
/// holes : tuple of array_like, optional
///     Hole ids of `a` and of `b`; samples of one hole are not paired.
///
/// Returns
/// -------
/// Table
///     One row per pair, by row of `a`: ``a`` and ``b`` rows, ``distance``
///     and, with `values`, ``value_a`` and ``value_b``.
#[pyfunction]
#[pyo3(signature = (a, b, max_distance, values=None, unique=true, holes=None))]
fn pairs(
    a: &Bound<PyAny>,
    b: &Bound<PyAny>,
    max_distance: f64,
    values: Option<&Bound<PyAny>>,
    unique: bool,
    holes: Option<&Bound<PyAny>>,
) -> PyResult<Table> {
    let (a, b) = (set_coords(a)?, set_coords(b)?);
    let values = values
        .map(|v| -> PyResult<_> {
            let (va, vb) = two(v, "values")?;
            Ok((floats(&va, "values")?, floats(&vb, "values")?))
        })
        .transpose()?;
    let codes = holes
        .map(|h| -> PyResult<_> {
            let (ha, hb) = two(h, "holes")?;
            let (la, _) = self::holes(Some(&ha), a.len())?.expect("given");
            let (lb, _) = self::holes(Some(&hb), b.len())?.expect("given");
            let mut ids = std::collections::HashMap::new();
            let mut code = |l: String| {
                let next = ids.len() as u32;
                *ids.entry(l).or_insert(next)
            };
            let ca: Vec<u32> = la.into_iter().map(&mut code).collect();
            let cb: Vec<u32> = lb.into_iter().map(&mut code).collect();
            Ok((ca, cb))
        })
        .transpose()?;
    let found = eda::pairs(
        &a,
        &b,
        max_distance,
        values.as_ref().map(|(va, vb)| (&va[..], &vb[..])),
        codes.as_ref().map(|(ca, cb)| (&ca[..], &cb[..])),
        unique,
    )
    .map_err(invalid)?;
    let index = |f: fn(&(usize, usize, f64)) -> usize| -> ArrayRef {
        Arc::new(UInt64Array::from_iter_values(
            found.iter().map(|p| f(p) as u64),
        ))
    };
    let mut columns: Vec<(&str, ArrayRef)> = vec![
        ("a", index(|p| p.0)),
        ("b", index(|p| p.1)),
        (
            "distance",
            Arc::new(Float64Array::from_iter_values(found.iter().map(|p| p.2))),
        ),
    ];
    if let Some((va, vb)) = &values {
        columns.push(("value_a", nullable(found.iter().map(|p| va[p.0]))));
        columns.push(("value_b", nullable(found.iter().map(|p| vb[p.1]))));
    }
    Ok(Table(RecordBatch::try_from_iter(columns).map_err(invalid)?))
}

/// Data spacing: the distance to the `n`th nearest sample.
///
/// Parameters
/// ----------
/// coords : PointSet or array_like
///     Samples, or their ``(n, 2)`` or ``(n, 3)`` coordinates.
/// n : int
///     Rank of the neighbor; on a square grid of spacing ``s`` seen in plan,
///     ``n=4`` gives ``s``.
/// targets : array_like, PointSet or BlockModel, optional
///     Locations to measure from, e.g. block centroids; default each sample,
///     not counting itself.
/// horizontal : bool
///     Measure in plan, ignoring elevation.
///
/// Returns
/// -------
/// ndarray
///     One distance per target, ``inf`` when there are fewer than `n` samples.
#[pyfunction]
#[pyo3(signature = (coords, n=1, targets=None, horizontal=false))]
fn data_spacing<'py>(
    py: Python<'py>,
    coords: &Bound<PyAny>,
    n: usize,
    targets: Option<&Bound<PyAny>>,
    horizontal: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let coords = set_coords(coords)?;
    let targets = targets
        .map(|t| -> PyResult<Vec<[f64; 3]>> {
            Ok(crate::estimation::targets(t)?
                .into_iter()
                .map(|(x, y, z)| [x, y, z])
                .collect())
        })
        .transpose()?;
    let d = py
        .detach(|| eda::spacing(&coords, targets.as_deref(), n, horizontal))
        .map_err(invalid)?;
    Ok(array1(py, d).into_any())
}

/// Mean of paired values and their relative bias per bin of pairing distance.
///
/// Parameters
/// ----------
/// pairs : Table
///     Result of `pairs` with `values`, or any table with ``distance``,
///     ``value_a`` and ``value_b`` columns.
/// bins : int or array_like
///     Number of equal bins from 0 to the largest distance, or bin edges.
///
/// Returns
/// -------
/// Table
///     One row per bin: ``from``, ``to``, ``n`` pairs, ``mean_a``, ``mean_b``
///     and ``bias``, ``mean_b / mean_a - 1``; null when the bin is empty.
#[pyfunction]
fn paired_bias(pairs: &Bound<PyAny>, bins: &Bound<PyAny>) -> PyResult<Table> {
    let column = |name: &str| -> PyResult<Vec<f64>> {
        let c = pairs
            .get_item(name)
            .map_err(|_| invalid(format!("pairs need a '{name}' column; give pairs values")))?;
        floats(&c, name)
    };
    let distance = column("distance")?;
    let edges = match bins.extract::<usize>() {
        Ok(0) => return Err(invalid("bins must be positive")),
        Ok(k) => {
            let top = distance.iter().copied().fold(0.0, f64::max);
            let top = if top > 0.0 { top } else { 1.0 };
            (0..=k).map(|i| top * i as f64 / k as f64).collect()
        }
        Err(_) => floats(bins, "bins")?,
    };
    let rows = eda::paired_bias(&distance, &column("value_a")?, &column("value_b")?, &edges)
        .map_err(invalid)?;
    let col = |f: fn(&eda::Bias) -> f64| nullable(rows.iter().map(f));
    let batch = RecordBatch::try_from_iter([
        ("from", col(|r| r.from)),
        ("to", col(|r| r.to)),
        (
            "n",
            Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|r| r.n as u64),
            )) as ArrayRef,
        ),
        ("mean_a", col(|r| r.mean_a)),
        ("mean_b", col(|r| r.mean_b)),
        ("bias", col(|r| r.bias)),
    ])
    .map_err(invalid)?;
    Ok(Table(batch))
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pairs, m)?)?;
    m.add_function(wrap_pyfunction!(data_spacing, m)?)?;
    m.add_function(wrap_pyfunction!(paired_bias, m)?)?;
    m.add_function(wrap_pyfunction!(validate_model, m)?)?;
    m.add_function(wrap_pyfunction!(duplicates, m)?)?;
    m.add_function(wrap_pyfunction!(describe, m)?)?;
    m.add_function(wrap_pyfunction!(describe_by, m)?)?;
    m.add_function(wrap_pyfunction!(grade_tonnage, m)?)?;
    m.add_function(wrap_pyfunction!(compare_models, m)?)?;
    m.add_function(wrap_pyfunction!(capping_report, m)?)?;
    m.add_function(wrap_pyfunction!(swath, m)?)?;
    m.add_function(wrap_pyfunction!(contact, m)?)?;
    m.add_function(wrap_pyfunction!(capping, m)?)?;
    m.add_function(wrap_pyfunction!(h_scatter, m)?)?;
    m.add_function(wrap_pyfunction!(correlation, m)?)?;
    Ok(())
}
