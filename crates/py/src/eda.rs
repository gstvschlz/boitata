use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::DataType;
use boitata_core::PointSet;
use eda::{Along, Direction, Merge, Method};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use serde::{Deserialize, Serialize};

use crate::args::{
    Label, array1, array2, column, column_names, domain_codes, floats, holes, named, pair, per_row,
    rows, same_length,
};
use crate::categories::Categories;
use crate::containers::{PyBlockModel, PyPointSet, coords_arg};
use crate::estimation::py_label;
use crate::invalid;
use crate::table::Table;

fn optional_floats(obj: Option<&Bound<PyAny>>, what: &str) -> PyResult<Option<Vec<f64>>> {
    obj.map(|o| floats(o, what)).transpose()
}

/// Name of the quantile at probability `p`: ``P10``, ``P97.5``, ...
pub(crate) fn quantile_name(p: f64) -> String {
    let name = format!("{:.4}", 100.0 * p);
    format!("P{}", name.trim_end_matches('0').trim_end_matches('.'))
}

/// Weighted (declustered) statistics of `values`; NaN is skipped.
///
/// Parameters
/// ----------
/// values : array_like or str
///     Values, or their column in `data`.
/// weights : array_like or str, optional
///     Declustering weights; default the volumes of a BlockModel `data`, else 1.
/// quantiles : sequence of float
///     Probabilities of the quantile keys.
/// data : PointSet, BlockModel, Table or dict, optional
///     Where column names are looked up.
///
/// Returns
/// -------
/// dict
///     ``n``, ``mean``, ``variance``, ``std``, ``cv``, ``min``, ``max`` and one
///     key per quantile named ``P10``, ``P97.5``, ... (mid-point cumulative
///     weights).
#[pyfunction]
#[pyo3(signature = (values, *, weights=None, quantiles=vec![0.1, 0.25, 0.5, 0.75, 0.9], data=None))]
fn describe<'py>(
    py: Python<'py>,
    values: &Bound<PyAny>,
    weights: Option<&Bound<PyAny>>,
    quantiles: Vec<f64>,
    data: Option<&Bound<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let values = floats(&column(data, values, "values")?, "values")?;
    let w = weights_or_volumes(data, weights, values.len())?;
    let s = eda::describe(&values, w.as_deref(), &quantiles).map_err(invalid)?;
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
    for (p, q) in quantiles.iter().zip(s.quantiles) {
        d.set_item(quantile_name(*p), q)?;
    }
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
/// values : array_like or str
///     Values, or their column in `data`; NaN is skipped.
/// categories : array_like or str
///     Category (e.g. domain) of each value, int or str, or its column.
/// weights : array_like or str, optional
///     Declustering weights; default the volumes of a BlockModel `data`, else 1.
/// quantiles : sequence of float
///     Probabilities of the quantile columns.
/// data : PointSet, BlockModel, Table or dict, optional
///     Where column names are looked up.
///
/// Returns
/// -------
/// Table
///     ``category`` (sorted, then ``"all"``), ``n``, ``mean``, ``variance``,
///     ``std``, ``cv``, ``min``, ``max`` and one column per quantile named
///     ``P10``, ``P97.5``, ...
#[pyfunction]
#[pyo3(signature = (values, categories, *, weights=None, quantiles=vec![0.1, 0.25, 0.5, 0.75, 0.9], data=None))]
fn describe_by(
    values: &Bound<PyAny>,
    categories: &Bound<PyAny>,
    weights: Option<&Bound<PyAny>>,
    quantiles: Vec<f64>,
    data: Option<&Bound<PyAny>>,
) -> PyResult<Table> {
    let values = floats(&column(data, values, "values")?, "values")?;
    let n = values.len();
    let (names, codes) = self::categories(&column(data, categories, "categories")?, n)?;
    let w = weights_or_volumes(data, weights, n)?;
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
        columns.push((quantile_name(*p), col(&|s| s.quantiles[j])));
    }
    table("category", &names, &rows, columns)
}

/// Grade-tonnage curve straight from the data: tonnage, mean grade and metal at
/// or above each cutoff.
///
/// Each value stands for ``weights × density`` tonnes, e.g. block volumes and
/// densities, or composite lengths.
///
/// Parameters
/// ----------
/// values : array_like or str
///     Grades, or their column in `data`; NaN is skipped.
/// cutoffs : array_like
///     Cutoff grades; ``-inf`` keeps everything.
/// weights : array_like or str, optional
///     Volume or declustering weight of each value; default the volumes of a
///     BlockModel `data`, else 1.
/// density : float, array_like or str, optional
///     Density of each value; default 1.
/// categories : array_like or str, optional
///     Category (e.g. domain) of each value, int or str; adds a curve per
///     category.
/// data : PointSet, BlockModel, Table or dict, optional
///     Where column names are looked up.
///
/// Returns
/// -------
/// Table
///     ``category`` (with `categories`: sorted, then ``"all"``), ``cutoff``,
///     ``tonnage``, ``mean_grade`` (null when nothing is above) and ``metal``
///     (tonnage × mean grade).
#[pyfunction]
#[pyo3(signature = (values, cutoffs, *, weights=None, density=None, categories=None, data=None))]
fn grade_tonnage(
    values: &Bound<PyAny>,
    cutoffs: &Bound<PyAny>,
    weights: Option<&Bound<PyAny>>,
    density: Option<&Bound<PyAny>>,
    categories: Option<&Bound<PyAny>>,
    data: Option<&Bound<PyAny>>,
) -> PyResult<Table> {
    let values = floats(&column(data, values, "values")?, "values")?;
    let n = values.len();
    let w = weights_or_volumes(data, weights, n)?;
    let density = optional_per_row(data, density, n, "density")?;
    let cutoffs = floats(cutoffs, "cutoffs")?;
    let (names, rows) = match categories {
        None => {
            let r = eda::grade_tonnage(&values, w.as_deref(), density.as_deref(), &cutoffs);
            (vec![], vec![(None, r.map_err(invalid)?)])
        }
        Some(c) => {
            let (names, codes) = self::categories(&column(data, c, "categories")?, n)?;
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

/// Codes of `classes` as a scheme reads them: numbers are its codes (NaN for
/// none), labels are encoded by it.
fn scheme_codes(
    classes: &Bound<PyAny>,
    scheme: &Categories,
    n: usize,
) -> PyResult<Vec<Option<u32>>> {
    let codes = match floats(classes, "classes") {
        Ok(c) => {
            let k = scheme.0.len() as f64;
            c.into_iter()
                .map(|c| match c {
                    c if c.is_nan() => Ok(None),
                    c if c >= 0.0 && c < k && c.fract() == 0.0 => Ok(Some(c as u32)),
                    c => Err(invalid(format!("{c} is not a code of the scheme"))),
                })
                .collect::<PyResult<Vec<_>>>()?
        }
        Err(_) => crate::categories::coded(&crate::categories::labels(classes)?, Some(scheme))?.1,
    };
    same_length(n, codes.len(), "classes")?;
    Ok(codes)
}

/// Tonnage, and with `grades` metal, moving between the classes of two
/// categorical models of the same blocks, such as an old and a new domain
/// model, or the most likely category and a simulated realization.
///
/// Parameters
/// ----------
/// before, after : array_like or str
///     Class of each block in each model, or their columns in `data`; null
///     or NaN leaves the block out. With `scheme`, numbers are its codes and
///     labels are encoded by it; without, labels of either model, in the
///     order of `Categories.from_values`.
/// weights : array_like or str, optional
///     Volume of each block; default the volumes of a BlockModel `data`,
///     else 1.
/// density : float, array_like or str, optional
///     Density of each block; default 1.
/// grades : array_like or str, optional
///     Grade of each block; adds ``mean_grade`` and ``metal``. NaN grades add
///     tonnage but no metal.
/// scheme : Categories, optional
///     Names and order of the classes.
/// data : PointSet, BlockModel, Table or dict, optional
///     Where column names are looked up.
///
/// Returns
/// -------
/// Table
///     One row per pair of classes, ``from`` then ``to`` in class order:
///     ``from``, ``to``, ``tonnage`` (weights × density), and with `grades`
///     ``mean_grade`` (null without tonnage) and ``metal``. Summing
///     ``tonnage`` over ``to`` gives each class of `before`, over ``from``
///     each class of `after`; rows with ``from == to`` hold what is unchanged.
#[pyfunction]
#[pyo3(signature = (before, after, *, weights=None, density=None, grades=None, scheme=None, data=None))]
#[allow(clippy::too_many_arguments)]
fn domain_change(
    before: &Bound<PyAny>,
    after: &Bound<PyAny>,
    weights: Option<&Bound<PyAny>>,
    density: Option<&Bound<PyAny>>,
    grades: Option<&Bound<PyAny>>,
    scheme: Option<PyRef<Categories>>,
    data: Option<&Bound<PyAny>>,
) -> PyResult<Table> {
    let before = column(data, before, "before")?;
    let after = column(data, after, "after")?;
    let (names, a, b) = match scheme.as_deref() {
        Some(s) => {
            let a = scheme_codes(&before, s, before.len()?)?;
            let b = scheme_codes(&after, s, a.len())?;
            (s.0.names().to_vec(), a, b)
        }
        None => {
            let mut labels = crate::categories::labels(&before)?;
            let n = labels.len();
            labels.extend(crate::categories::labels(&after)?);
            same_length(n, labels.len() - n, "after")?;
            let (names, mut codes) = crate::categories::coded(&labels, None)?;
            let b = codes.split_off(n);
            (names, codes, b)
        }
    };
    let n = a.len();
    let w = weights_or_volumes(data, weights, n)?;
    let d = optional_per_row(data, density, n, "density")?;
    let tonnes = match (w, d) {
        (Some(w), Some(d)) => Some(w.iter().zip(&d).map(|(w, d)| w * d).collect()),
        (w, d) => w.or(d),
    };
    let g = optional_per_row(data, grades, n, "grades")?;
    let cells = eda::domain_change(&a, &b, names.len(), tonnes.as_deref(), g.as_deref())
        .map_err(invalid)?;
    let label = |f: fn(&eda::Change) -> usize| -> ArrayRef {
        Arc::new(StringArray::from_iter_values(
            cells.iter().map(|c| names[f(c)].as_str()),
        ))
    };
    let mut columns: Vec<(&str, ArrayRef)> = vec![
        ("from", label(|c| c.from)),
        ("to", label(|c| c.to)),
        (
            "tonnage",
            Arc::new(Float64Array::from_iter_values(
                cells.iter().map(|c| c.tonnage),
            )),
        ),
    ];
    if g.is_some() {
        columns.push(("mean_grade", nullable(cells.iter().map(|c| c.mean_grade))));
        columns.push((
            "metal",
            Arc::new(Float64Array::from_iter_values(
                cells.iter().map(|c| c.metal),
            )),
        ));
    }
    Ok(Table(RecordBatch::try_from_iter(columns).map_err(invalid)?))
}

/// Along-hole category transition matrix: for every sample, the classes of
/// samples `lag ± tolerance` deeper in the same hole.
///
/// Parameters
/// ----------
/// depth : array_like or str
///     Depth down the hole of each sample, or its column in `data`.
/// categories : array_like or str
///     Class of each sample, int or str, or its column; null leaves the
///     sample out. With `scheme`, numbers are its codes and labels are
///     encoded by it; without, labels in the order of `Categories.from_values`.
/// holes : array_like or str
///     Hole id of each sample, or its column.
/// lag : float
///     Along-hole distance searched for a deeper sample.
/// tolerance : float
///     Half-width of the lag window.
/// scheme : Categories, optional
///     Names and order of the classes.
/// data : PointSet, BlockModel, Table or dict, optional
///     Where column names are looked up.
///
/// Returns
/// -------
/// Table
///     One row per pair of classes, ``from`` (shallower) then ``to`` (deeper)
///     in class order: ``from``, ``to``, ``count`` and ``frequency``
///     (``count`` over its row's total, null when the row has none).
#[pyfunction]
#[pyo3(signature = (depth, categories, holes, *, lag, tolerance=0.0, scheme=None, data=None))]
#[allow(clippy::too_many_arguments)]
fn transition_matrix(
    depth: &Bound<PyAny>,
    categories: &Bound<PyAny>,
    holes: &Bound<PyAny>,
    lag: f64,
    tolerance: f64,
    scheme: Option<PyRef<Categories>>,
    data: Option<&Bound<PyAny>>,
) -> PyResult<Table> {
    let categories = column(data, categories, "categories")?;
    let (names, codes) = match scheme.as_deref() {
        Some(s) => {
            let n = categories.len()?;
            (s.0.names().to_vec(), scheme_codes(&categories, s, n)?)
        }
        None => {
            let labels = crate::categories::labels(&categories)?;
            crate::categories::coded(&labels, None)?
        }
    };
    let n = codes.len();
    let depth = floats(&column(data, depth, "depth")?, "depth")?;
    same_length(n, depth.len(), "depth")?;
    let (_, hole_ids) = self::holes(Some(&column(data, holes, "holes")?), n)?.expect("given");
    let keep: Vec<usize> = (0..n).filter(|&i| codes[i].is_some()).collect();
    let d: Vec<f64> = keep.iter().map(|&i| depth[i]).collect();
    let c: Vec<u32> = keep.iter().map(|&i| codes[i].expect("kept")).collect();
    let h: Vec<u32> = keep.iter().map(|&i| hole_ids[i]).collect();
    let k = names.len();
    let cells = eda::transition_matrix(&d, &c, &h, k, lag, tolerance).map_err(invalid)?;
    let mut totals = vec![0u64; k];
    for t in &cells {
        totals[t.from as usize] += t.count;
    }
    let label = |f: fn(&eda::Transition) -> u32| -> ArrayRef {
        Arc::new(StringArray::from_iter_values(
            cells.iter().map(|t| names[f(t) as usize].as_str()),
        ))
    };
    let frequency = cells.iter().map(|t| {
        let total = totals[t.from as usize];
        if total > 0 {
            t.count as f64 / total as f64
        } else {
            f64::NAN
        }
    });
    let columns: Vec<(&str, ArrayRef)> = vec![
        ("from", label(|t| t.from)),
        ("to", label(|t| t.to)),
        (
            "count",
            Arc::new(UInt64Array::from_iter_values(cells.iter().map(|t| t.count))),
        ),
        ("frequency", nullable(frequency)),
    ];
    Ok(Table(RecordBatch::try_from_iter(columns).map_err(invalid)?))
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

/// The block volumes of a BlockModel `data`.
fn volumes(data: Option<&Bound<PyAny>>) -> Option<Vec<f64>> {
    data?
        .cast::<PyBlockModel>()
        .ok()
        .map(|m| m.get().0.volumes())
}

fn block_volumes(model: &Bound<PyAny>) -> PyResult<Vec<f64>> {
    volumes(Some(model)).ok_or_else(|| invalid("model must be a BlockModel"))
}

fn optional_per_row(
    data: Option<&Bound<PyAny>>,
    arg: Option<&Bound<PyAny>>,
    n: usize,
    what: &str,
) -> PyResult<Option<Vec<f64>>> {
    arg.map(|a| per_row(data, a, n, what)).transpose()
}

/// `weights`, or the block volumes when `data` is a BlockModel.
fn weights_or_volumes(
    data: Option<&Bound<PyAny>>,
    weights: Option<&Bound<PyAny>>,
    n: usize,
) -> PyResult<Option<Vec<f64>>> {
    match weights {
        Some(_) => optional_per_row(data, weights, n, "weights"),
        None => Ok(volumes(data)),
    }
}

/// `volume × density` per block of `model`.
fn block_tonnes(
    model: &Bound<PyAny>,
    density: Option<&Bound<PyAny>>,
) -> PyResult<(usize, Vec<f64>)> {
    let volumes = block_volumes(model)?;
    let n = volumes.len();
    let density = optional_per_row(Some(model), density, n, "density")?;
    let tonnes = match density {
        Some(d) => volumes.iter().zip(&d).map(|(v, d)| v * d).collect(),
        None => volumes,
    };
    Ok((n, tonnes))
}

/// Grade-tonnage of several models of the same blocks, by cutoff and category,
/// against a reference model.
///
/// Each block stands for ``volume × density`` tonnes, as in `grade_tonnage`.
///
/// Parameters
/// ----------
/// model : BlockModel
///     The blocks, with their volumes.
/// columns : sequence of str or dict of str to array_like or str
///     Columns of `model` holding the grades of each model, or model names to
///     columns or grades; NaN is skipped.
/// cutoffs : array_like
///     Cutoff grades; ``-inf`` keeps everything.
/// categories : array_like or str, optional
///     Category (e.g. class or domain) of each block, int or str, or its column.
/// reference : str, optional
///     Model the others are compared with; the first by default.
/// density : float, array_like or str, optional
///     Block densities, or their column; default 1.
///
/// Returns
/// -------
/// Table
///     One row per category (sorted, then ``"all"``), cutoff and model:
///     ``category``, ``cutoff``, ``model``, ``tonnage``, ``mean_grade`` and
///     ``metal`` at or above the cutoff, and ``tonnage_diff``, ``grade_diff``
///     and ``metal_diff``, each ``value / reference value - 1``.
#[pyfunction]
#[pyo3(signature = (model, columns, cutoffs, *, categories=None, reference=None, density=None))]
fn compare_models(
    model: &Bound<PyAny>,
    columns: &Bound<PyAny>,
    cutoffs: &Bound<PyAny>,
    categories: Option<&Bound<PyAny>>,
    reference: Option<&str>,
    density: Option<&Bound<PyAny>>,
) -> PyResult<Table> {
    let (n, tonnes) = block_tonnes(model, density)?;
    let items = match columns.hasattr("items")? {
        true => columns.call_method0("items")?,
        false => columns.clone(),
    };
    let (labels, grades) = items
        .try_iter()
        .map_err(|_| invalid("columns must be a sequence of names or a dict"))?
        .map(|item| {
            let item = item?;
            let (name, grades): (String, Bound<PyAny>) = match item.extract::<String>() {
                Ok(name) => (name, item),
                Err(_) => item.extract()?,
            };
            let grades = per_row(Some(model), &grades, n, &name)?;
            Ok((name, grades))
        })
        .collect::<PyResult<(Vec<String>, Vec<Vec<f64>>)>>()?;
    let reference = match reference {
        None => 0,
        Some(r) => labels
            .iter()
            .position(|l| l == r)
            .ok_or_else(|| invalid(format!("reference {r:?} is not one of the models")))?,
    };
    let (names, codes) = categories
        .map(|c| self::categories(&column(Some(model), c, "categories")?, n))
        .transpose()?
        .unwrap_or_default();
    let slices: Vec<&[f64]> = grades.iter().map(Vec::as_slice).collect();
    let rows = eda::compare_models(
        &slices,
        categories.map(|_| &codes[..]),
        Some(&tonnes),
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

/// Mean of `values` per slice of `width` along `azimuth` (degrees from north)
/// or `axis` ("x", "y", "z"); slices start at 0 so swaths of samples and blocks
/// line up.
///
/// Each value stands for ``weights × density`` tonnes, as in `grade_tonnage`,
/// so slice tonnages and metals add up to the totals at cutoff ``-inf``.
///
/// Parameters
/// ----------
/// coords : PointSet, BlockModel or array_like
///     Samples or blocks, or their ``(n, 2)`` or ``(n, 3)`` coordinates.
/// values : array_like or str
///     Values, or their column in `coords`.
/// width : float
///     Slice width.
/// azimuth : float, optional
/// axis : {"x", "y", "z"}, optional
///     Direction of the slices; give one.
/// weights : array_like or str, optional
///     Declustering weights or volumes; default the block volumes of a
///     BlockModel, else 1.
/// density : float, array_like or str, optional
///     Density of each value; default 1.
///
/// Returns
/// -------
/// Table
///     ``center``, ``n``, ``mean`` (weighted by `weights`), ``tonnage`` and
///     ``metal`` of the non-empty slices.
#[pyfunction]
#[pyo3(signature = (coords, values, width, *, azimuth=None, axis=None, weights=None, density=None))]
fn swath(
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    width: f64,
    azimuth: Option<f64>,
    axis: Option<&str>,
    weights: Option<&Bound<PyAny>>,
    density: Option<&Bound<PyAny>>,
) -> PyResult<Table> {
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
    let values = floats(&column(Some(coords), values, "values")?, "values")?;
    let n = values.len();
    let w = weights_or_volumes(Some(coords), weights, n)?;
    let density = optional_per_row(Some(coords), density, n, "density")?;
    let p = eda::swath(
        &coords_arg(coords)?,
        &values,
        w.as_deref(),
        density.as_deref(),
        width,
        along,
    )
    .map_err(invalid)?;
    let f = |v: Vec<f64>| Arc::new(Float64Array::from(v)) as ArrayRef;
    let count = p.count.iter().map(|&c| c as u64);
    let columns = vec![
        ("center", f(p.centers)),
        (
            "n",
            Arc::new(UInt64Array::from_iter_values(count)) as ArrayRef,
        ),
        ("mean", f(p.mean)),
        ("tonnage", f(p.tonnage)),
        ("metal", f(p.metal)),
    ];
    Ok(Table(RecordBatch::try_from_iter(columns).map_err(invalid)?))
}

/// Statistics of a block model against the data it was estimated from, per
/// domain and over all.
///
/// Parameters
/// ----------
/// model : BlockModel
///     The blocks; weighted by their tonnage, ``volume × density``.
/// grade : str or array_like
///     Column of `model` holding the block grades, or the grades; NaN is
///     skipped.
/// data : PointSet, Table or dict
///     The samples (composites).
/// values : str or array_like
///     Column of `data` holding the sample grades, or the grades; NaN is
///     skipped.
/// weights : array_like or str, optional
///     Declustering weights of `data`; adds a ``declustered`` row that the
///     model is compared with.
/// domain_column : str or tuple of str, optional
///     Domain column of both `model` and `data`, or one name for each.
/// density : float, array_like or str, optional
///     Block densities, or their column; default 1.
/// reference : str or array_like, optional
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
#[pyo3(signature = (model, grade, data, values, *, weights=None, domain_column=None, density=None, reference=None))]
#[allow(clippy::too_many_arguments)]
fn validate_model(
    model: &Bound<PyAny>,
    grade: &Bound<PyAny>,
    data: &Bound<PyAny>,
    values: &Bound<PyAny>,
    weights: Option<&Bound<PyAny>>,
    domain_column: Option<&Bound<PyAny>>,
    density: Option<&Bound<PyAny>>,
    reference: Option<&Bound<PyAny>>,
) -> PyResult<Table> {
    let (n, tonnes) = block_tonnes(model, density)?;
    let grades = per_row(Some(model), grade, n, "grade")?;
    let values = floats(&column(Some(data), values, "values")?, "values")?;
    let w = optional_per_row(Some(data), weights, values.len(), "weights")?;
    let reference = optional_per_row(Some(model), reference, n, "reference")?;
    let (names, codes) = match domain_column {
        None => (vec![], None),
        Some(c) => {
            let (om, od) = pair(model, data, c, "domain_column")?;
            let (lm, _) = holes(Some(&om), n)?.expect("given");
            let (ld, _) = holes(Some(&od), values.len())?.expect("given");
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
        &grades,
        &values,
        w.as_deref(),
        codes.as_ref().map(|(m, d)| (&m[..], &d[..])),
        Some(&tonnes),
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
/// Parameters
/// ----------
/// coords : PointSet or array_like
///     Samples, or their ``(n, 2)`` or ``(n, 3)`` coordinates.
/// values : array_like or str
///     Values, or their column in `coords`.
/// domains : array_like, optional
///     Domain of each sample, int or str.
/// domain_column : str, optional
///     Column of `coords` holding the domains; give it or `domains`.
/// holes : array_like or str
///     Hole id of each sample, or its column.
/// inside, outside : int or str
///     Domains on either side of the contact.
/// max_distance : float
///     Largest distance to the contact; the outermost bins end there.
/// bin : float
///     Width of the distance bins.
///
/// Returns
/// -------
/// Table
///     ``distance`` (bin centers), ``mean`` and ``n`` samples per bin.
#[pyfunction]
#[pyo3(signature = (coords, values, *, domains=None, domain_column=None, holes, inside, outside, max_distance, bin))]
#[allow(clippy::too_many_arguments)]
fn contact(
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    domains: Option<&Bound<PyAny>>,
    domain_column: Option<&str>,
    holes: &Bound<PyAny>,
    inside: &Bound<PyAny>,
    outside: &Bound<PyAny>,
    max_distance: f64,
    bin: f64,
) -> PyResult<Table> {
    let data = coords;
    let values = floats(&column(Some(data), values, "values")?, "values")?;
    let coords = coords_arg(coords)?;
    let n = coords.len();
    let (_, hole_ids) = self::holes(Some(&column(Some(data), holes, "holes")?), n)?.expect("given");
    let domains = domain_labels(Some(data), domains, domain_column)?;
    let (labels, ids) = self::holes(Some(&domains), n)
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
        &values,
        &ids,
        &hole_ids,
        code(inside)?,
        code(outside)?,
        max_distance,
        bin,
    )
    .map_err(invalid)?;
    let f = |v: Vec<f64>| Arc::new(Float64Array::from(v)) as ArrayRef;
    let count = p.count.iter().map(|&c| c as u64);
    let columns = vec![
        ("distance", f(p.centers)),
        ("mean", f(p.mean)),
        (
            "n",
            Arc::new(UInt64Array::from_iter_values(count)) as ArrayRef,
        ),
    ];
    Ok(Table(RecordBatch::try_from_iter(columns).map_err(invalid)?))
}

/// `domains`, or the `domain_column` of `data`; exactly one of the two.
fn domain_labels<'py>(
    data: Option<&Bound<'py, PyAny>>,
    domains: Option<&Bound<'py, PyAny>>,
    domain_column: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    match (domains, domain_column) {
        (Some(d), None) => Ok(d.clone()),
        (None, Some(c)) => named(data, c, "domain_column"),
        _ => Err(invalid("give one of domains or domain_column")),
    }
}

/// Hard (`target` domain alone) vs soft statistics: also folding in samples of
/// other domains within `buffer`, as a search opened with `Search(...,
/// soft=buffer)` would.
///
/// Parameters
/// ----------
/// coords : PointSet or array_like
///     Samples, or their ``(n, 2)`` or ``(n, 3)`` coordinates.
/// values : array_like or str
///     Values, or their column in `coords`; NaN is skipped.
/// domains : array_like, optional
///     Domain of each sample, int or str.
/// domain_column : str, optional
///     Column of `coords` holding the domains; give it or `domains`.
/// target : int or str
///     The domain whose statistics are computed.
/// buffer : float
///     Distance within which a sample of another domain is folded into the
///     soft row.
/// weights : array_like or str, optional
///     Declustering weights; default 1.
/// quantiles : sequence of float
///     Probabilities of the quantile columns.
///
/// Returns
/// -------
/// added : ndarray of bool
///     One per sample of the other domains, in their original order: whether
///     it is within `buffer` of `target` and folded into the soft row.
/// table : Table
///     ``kind`` (``"hard"``, ``"soft"``), ``n``, ``mean``, ``variance``,
///     ``std``, ``cv``, ``min``, ``max`` and one column per quantile named
///     ``P10``, ``P97.5``, ...
#[pyfunction]
#[pyo3(signature = (coords, values, *, domains=None, domain_column=None, target, buffer, weights=None, quantiles=vec![0.1, 0.25, 0.5, 0.75, 0.9]))]
#[allow(clippy::too_many_arguments)]
fn soft_boundary<'py>(
    py: Python<'py>,
    coords: &Bound<PyAny>,
    values: &Bound<PyAny>,
    domains: Option<&Bound<PyAny>>,
    domain_column: Option<&str>,
    target: &Bound<PyAny>,
    buffer: f64,
    weights: Option<&Bound<PyAny>>,
    quantiles: Vec<f64>,
) -> PyResult<(Bound<'py, PyAny>, Table)> {
    let data = coords;
    let values = floats(&column(Some(data), values, "values")?, "values")?;
    let coords = coords_arg(coords)?;
    let n = coords.len();
    let domains = domain_labels(Some(data), domains, domain_column)?;
    let (labels, ids) = self::holes(Some(&domains), n)
        .map_err(|_| invalid("domains must be a 1-D sequence of labels"))?
        .expect("given");
    let label = target.str()?.to_string();
    let target = labels
        .iter()
        .position(|l| *l == label)
        .map(|i| ids[i])
        .ok_or_else(|| invalid(format!("no sample in domain {label}")))?;
    let w = weights_or_volumes(Some(data), weights, n)?;
    let r = eda::soft_boundary(
        &coords,
        &values,
        &ids,
        w.as_deref(),
        target,
        buffer,
        &quantiles,
    )
    .map_err(invalid)?;
    let added = numpy::PyArray1::from_vec(py, r.added);
    let names = vec!["hard".to_string(), "soft".to_string()];
    let rows: Vec<(Option<u32>, eda::Summary)> = r
        .rows
        .into_iter()
        .map(|(b, s)| {
            let code = match b {
                eda::Boundary::Hard => 0,
                eda::Boundary::Soft => 1,
            };
            (Some(code), s)
        })
        .collect();
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
        columns.push((quantile_name(*p), col(&|s| s.quantiles[j])));
    }
    let table = table("kind", &names, &rows, columns)?;
    Ok((added.into_any(), table))
}

/// Metal removed and statistics after capping at each of `caps`.
///
/// Parameters
/// ----------
/// values : array_like or str
///     Values, or their column in `data`; NaN is skipped.
/// weights : array_like or str, optional
///     Declustering weights; default the volumes of a BlockModel `data`, else 1.
/// caps : array_like, optional
///     Caps; default the weighted quantiles 0.9, 0.95, 0.975, 0.99, 0.995 and
///     0.999.
/// data : PointSet, BlockModel, Table or dict, optional
///     Where column names are looked up.
///
/// Returns
/// -------
/// Table
///     ``cap``, ``fraction`` of weight above it, ``metal_removed``
///     (1 - capped mean / mean), capped ``mean`` and ``cv``.
#[pyfunction]
#[pyo3(signature = (values, *, weights=None, caps=None, data=None))]
fn capping(
    values: &Bound<PyAny>,
    weights: Option<&Bound<PyAny>>,
    caps: Option<&Bound<PyAny>>,
    data: Option<&Bound<PyAny>>,
) -> PyResult<Table> {
    let values = floats(&column(data, values, "values")?, "values")?;
    let w = weights_or_volumes(data, weights, values.len())?;
    let caps = optional_floats(caps, "caps")?;
    let r = eda::capping(&values, w.as_deref(), caps.as_deref()).map_err(invalid)?;
    let col = |f: fn(&eda::Cap) -> f64| nullable(r.iter().map(f));
    let columns = vec![
        ("cap", col(|c| c.cap)),
        ("fraction", col(|c| c.fraction)),
        ("metal_removed", col(|c| c.metal_removed)),
        ("mean", col(|c| c.mean)),
        ("cv", col(|c| c.cv)),
    ];
    Ok(Table(RecordBatch::try_from_iter(columns).map_err(invalid)?))
}

/// Weighted statistics per domain before and after capping, and the metal removed.
///
/// Parameters
/// ----------
/// values : array_like or str
///     Values, or their column in `data`; NaN is skipped.
/// caps : dict
///     Cap per domain; domains left out are not capped.
/// domains : array_like, optional
///     Domain of each value, int or str.
/// domain_column : str, optional
///     Column of `data` holding the domains; give it or `domains`.
/// weights : array_like or str, optional
///     Declustering weights; default the volumes of a BlockModel `data`, else 1.
/// data : PointSet, BlockModel, Table or dict, optional
///     Where column names are looked up.
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
#[pyo3(signature = (values, caps, *, domains=None, domain_column=None, weights=None, data=None))]
fn capping_report(
    values: &Bound<PyAny>,
    caps: &Bound<PyDict>,
    domains: Option<&Bound<PyAny>>,
    domain_column: Option<&str>,
    weights: Option<&Bound<PyAny>>,
    data: Option<&Bound<PyAny>>,
) -> PyResult<Table> {
    let values = floats(&column(data, values, "values")?, "values")?;
    let domains = domain_labels(data, domains, domain_column)?;
    let (names, codes) = categories(&domains, values.len())?;
    let mut by_domain = vec![f64::INFINITY; names.len()];
    for (k, cap) in caps.iter() {
        let k = k.str()?.to_string();
        let i = names
            .binary_search(&k)
            .map_err(|_| invalid(format!("no value in domain {k}")))?;
        by_domain[i] = cap.extract()?;
    }
    let w = weights_or_volumes(data, weights, values.len())?;
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

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum Given {
    One(f64),
    PerDomain(Vec<(Label, f64)>),
}

#[derive(Serialize, Deserialize)]
struct FittedCaps {
    domains: Option<Vec<Label>>,
    /// None: not capped.
    caps: Vec<Option<f64>>,
    removed: Vec<Option<f64>>,
}

/// Top-cut transform: clips values to a cap per domain.
///
/// Give exactly one of `cap`, `quantile`, `metal_removed` or `cv`. The rules
/// choose each domain's cap from its own weighted values when fitted.
///
/// Parameters
/// ----------
/// cap : float or dict, optional
///     The cap, or a cap per domain; domains left out are not capped.
/// quantile : float, optional
///     Cap at the weighted quantile of this probability, as in `describe`.
/// metal_removed : float, optional
///     Cap removing this fraction of the metal, ``1 - capped mean / mean``,
///     in [0, 1).
/// cv : float, optional
///     The largest cap whose capped coefficient of variation is at most this;
///     the largest value when the data already are.
///
/// Attributes
/// ----------
/// caps_ : float or dict
///     The cap, or the cap per domain when fitted with domains (inf when not
///     capped).
/// metal_removed_ : float or dict
///     Fraction of the metal removed, ``1 - mean_capped / mean`` of
///     `capping_report`, likewise.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "Capping")]
pub struct Capping {
    cap: Option<Given>,
    quantile: Option<f64>,
    metal_removed: Option<f64>,
    cv: Option<f64>,
    fitted: Option<FittedCaps>,
}

impl Capping {
    fn fitted(&self) -> PyResult<&FittedCaps> {
        self.fitted
            .as_ref()
            .ok_or_else(|| invalid("Capping is not fitted; call fit first"))
    }

    fn caps(&self) -> PyResult<Vec<f64>> {
        Ok(self
            .fitted()?
            .caps
            .iter()
            .map(|c| c.unwrap_or(f64::INFINITY))
            .collect())
    }

    /// Fits and returns the values and their domain codes.
    fn fit_codes(
        &mut self,
        values: &Bound<PyAny>,
        domains: Option<&Bound<PyAny>>,
        domain_column: Option<&str>,
        weights: Option<&Bound<PyAny>>,
        data: Option<&Bound<PyAny>>,
    ) -> PyResult<(Vec<f64>, Vec<u32>)> {
        let values = floats(&column(data, values, "values")?, "values")?;
        let n = values.len();
        let (labels, codes) = match (domains, domain_column) {
            (None, None) => (None, vec![0; n]),
            _ => {
                let (l, c) = domain_codes(data, domains, domain_column, n)?;
                (Some(l), c)
            }
        };
        let w = weights_or_volumes(data, weights, n)?;
        let k = labels.as_ref().map_or(1, Vec::len);
        let rule = |r: fn(f64) -> eda::CapRule, x: f64| {
            eda::fit_caps(&values, &codes, w.as_deref(), r(x)).map_err(invalid)
        };
        let caps = match (&self.cap, self.quantile, self.metal_removed, self.cv) {
            (Some(Given::One(c)), ..) => vec![*c; k],
            (Some(Given::PerDomain(given)), ..) => {
                let labels = labels
                    .as_ref()
                    .ok_or_else(|| invalid("caps per domain need domains or domain_column"))?;
                let mut caps = vec![f64::INFINITY; k];
                for (l, c) in given {
                    let i = labels
                        .iter()
                        .position(|x| x == l)
                        .ok_or_else(|| invalid(format!("no value in domain {l}")))?;
                    caps[i] = *c;
                }
                caps
            }
            (_, Some(p), ..) => rule(eda::CapRule::Quantile, p)?,
            (_, _, Some(t), _) => rule(eda::CapRule::MetalRemoved, t)?,
            (.., Some(t)) => rule(eda::CapRule::Cv, t)?,
            _ => return Err(invalid("give one of cap, quantile, metal_removed or cv")),
        };
        let rows = eda::capping_report(&values, &codes, w.as_deref(), &caps).map_err(invalid)?;
        let mut removed = vec![None; k];
        for (c, r) in rows {
            if let Some(c) = c {
                removed[c as usize] = Some(1.0 - r.after.mean / r.before.mean);
            }
        }
        self.fitted = Some(FittedCaps {
            domains: labels,
            caps: caps.iter().map(|c| c.is_finite().then_some(*c)).collect(),
            removed,
        });
        Ok((values, codes))
    }

    /// `per_domain` as a dict by domain, or its only value without domains.
    fn by_domain<'py>(
        &self,
        py: Python<'py>,
        per_domain: impl Iterator<Item = f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut per_domain = per_domain.peekable();
        let Some(labels) = &self.fitted()?.domains else {
            return per_domain.next().unwrap_or(f64::NAN).into_bound_py_any(py);
        };
        let d = PyDict::new(py);
        for (l, v) in labels.iter().zip(per_domain) {
            d.set_item(py_label(py, l)?, v)?;
        }
        Ok(d.into_any())
    }
}

#[pymethods]
impl Capping {
    /// JSON of the parameters and, once fitted, the fitted state.
    fn to_json(&self) -> PyResult<String> {
        crate::persist::to_json(self)
    }

    /// Reads `to_json` output; raises InvalidInput on another class's JSON
    /// or a newer format.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        crate::persist::from_json(text)
    }

    #[new]
    #[pyo3(signature = (*, cap=None, quantile=None, metal_removed=None, cv=None))]
    fn new(
        cap: Option<&Bound<PyAny>>,
        quantile: Option<f64>,
        metal_removed: Option<f64>,
        cv: Option<f64>,
    ) -> PyResult<Self> {
        let given = [
            cap.is_some(),
            quantile.is_some(),
            metal_removed.is_some(),
            cv.is_some(),
        ];
        if given.iter().filter(|g| **g).count() != 1 {
            return Err(invalid("give one of cap, quantile, metal_removed or cv"));
        }
        let cap = match cap {
            None => None,
            Some(c) => match c.cast::<PyDict>() {
                Ok(d) => Some(Given::PerDomain(
                    d.iter()
                        .map(|(k, v)| {
                            let l = crate::args::label(&k)?
                                .ok_or_else(|| invalid(format!("invalid domain {k}")))?;
                            Ok((l, v.extract()?))
                        })
                        .collect::<PyResult<_>>()?,
                )),
                Err(_) => Some(Given::One(c.extract()?)),
            },
        };
        let finite = |c: &f64| c.is_finite();
        let ok = match &cap {
            Some(Given::One(c)) => finite(c),
            Some(Given::PerDomain(given)) => given.iter().all(|(_, c)| finite(c)),
            None => true,
        };
        if !ok {
            return Err(invalid("caps must be finite"));
        }
        Ok(Self {
            cap,
            quantile,
            metal_removed,
            cv,
            fitted: None,
        })
    }

    /// Fits the caps to `values`, per domain when given.
    ///
    /// Parameters
    /// ----------
    /// values : array_like or str
    ///     Values, or their column in `data`; NaN is skipped.
    /// domains : array_like, optional
    ///     Domain of each value, int or str.
    /// domain_column : str, optional
    ///     Column of `data` holding the domains; give it or `domains`.
    /// weights : array_like or str, optional
    ///     Declustering weights; default the volumes of a BlockModel `data`,
    ///     else 1.
    /// data : PointSet, BlockModel, Table or dict, optional
    ///     Where column names are looked up.
    #[pyo3(signature = (values, *, domains=None, domain_column=None, weights=None, data=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        values: &Bound<PyAny>,
        domains: Option<&Bound<PyAny>>,
        domain_column: Option<&str>,
        weights: Option<&Bound<PyAny>>,
        data: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.fit_codes(values, domains, domain_column, weights, data)?;
        Ok(slf)
    }

    /// `fit`, then the fitted values capped.
    #[pyo3(signature = (values, *, domains=None, domain_column=None, weights=None, data=None))]
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        values: &Bound<PyAny>,
        domains: Option<&Bound<PyAny>>,
        domain_column: Option<&str>,
        weights: Option<&Bound<PyAny>>,
        data: Option<&Bound<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (values, codes) = self.fit_codes(values, domains, domain_column, weights, data)?;
        Ok(array1(py, eda::cap_values(&values, &codes, &self.caps()?)).into_any())
    }

    /// `values` clipped to the cap of their domain; values of a domain not
    /// fitted, and NaN, are unchanged.
    #[pyo3(signature = (values, *, domains=None, domain_column=None, data=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        values: &Bound<PyAny>,
        domains: Option<&Bound<PyAny>>,
        domain_column: Option<&str>,
        data: Option<&Bound<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let values = floats(&column(data, values, "values")?, "values")?;
        let n = values.len();
        let mut caps = self.caps()?;
        let obj = match (domains, domain_column) {
            (None, None) => None,
            _ => Some(domain_labels(data, domains, domain_column)?),
        };
        let fitted = self.fitted()?.domains.as_deref();
        let codes = match crate::estimation::codes(fitted, obj.as_ref(), n, "transform")? {
            None => vec![0; n],
            Some(codes) => {
                let unknown = caps.len() as u32;
                caps.push(f64::INFINITY);
                codes.into_iter().map(|c| c.unwrap_or(unknown)).collect()
            }
        };
        Ok(array1(py, eda::cap_values(&values, &codes, &caps)).into_any())
    }

    #[getter]
    fn caps_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.by_domain(py, self.caps()?.into_iter())
    }

    #[getter]
    fn metal_removed_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let removed = &self.fitted()?.removed;
        self.by_domain(py, removed.iter().map(|r| r.unwrap_or(f64::NAN)))
    }
}

/// Head and tail values of pairs `lag ± tolerance` apart, and their correlation.
///
/// With `azimuth`, pairs are oriented within `angle_tolerance` of (`azimuth`,
/// `dip`); without, omnidirectional. Tails come from `other` when given.
/// `values` and `other` may name columns of a PointSet `coords`.
///
/// Returns
/// -------
/// head, tail : ndarray
/// r : float
#[pyfunction]
#[pyo3(signature = (coords, values, lag, tolerance, *, azimuth=None, dip=0.0, angle_tolerance=22.5, other=None))]
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
    let other = other
        .map(|o| floats(&column(Some(coords), o, "other")?, "other"))
        .transpose()?;
    let (head, tail, r) = eda::h_scatter(
        &coords_arg(coords)?,
        &floats(&column(Some(coords), values, "values")?, "values")?,
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
/// data : array_like, PointSet, BlockModel, Table or dict
///     ``(n, d)`` values, or named columns.
/// columns : sequence of str, optional
///     Columns of `data` to correlate, in order; default all of them.
/// weights : array_like or str, optional
///     Declustering weights, or their column in `data`.
/// method : {"pearson", "spearman", "covariance"}
///     Pearson correlation, rank correlation, or the covariance (over the sum
///     of weights, as the variance of `describe`), variances on the diagonal.
///
/// Returns
/// -------
/// ndarray
///     ``(d, d)`` symmetric matrix.
#[pyfunction]
#[pyo3(signature = (data, *, columns=None, weights=None, method="pearson"))]
fn correlation<'py>(
    py: Python<'py>,
    data: &Bound<PyAny>,
    columns: Option<Vec<String>>,
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
    let by_name = columns.is_some() || column_names(data).is_ok();
    let columns: Vec<Vec<f64>> = if by_name {
        columns
            .map_or_else(|| column_names(data), Ok)?
            .iter()
            .map(|c| floats(&named(Some(data), c, "columns")?, c))
            .collect::<PyResult<_>>()?
    } else {
        let rows = rows(data, "data")?;
        let d = rows.first().map_or(0, Vec::len);
        (0..d)
            .map(|j| rows.iter().map(|r| r[j]).collect())
            .collect()
    };
    let w = weights
        .map(|w| floats(&column(Some(data), w, "weights")?, "weights"))
        .transpose()?;
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
/// coords : PointSet or array_like
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
/// weights : array_like or str, optional
///     Weights of ``merge="mean"``, e.g. composite lengths, or their column.
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
#[pyo3(signature = (coords, *, tolerance=0.0, merge=None, weights=None))]
fn duplicates<'py>(
    py: Python<'py>,
    coords: &Bound<'py, PyAny>,
    tolerance: f64,
    merge: Option<&str>,
    weights: Option<&Bound<PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let points = coords;
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
    let w = optional_per_row(Some(points), weights, n, "weights")?;
    if w.is_some() && rule != Merge::Mean {
        return Err(invalid("weights are only used with merge='mean'"));
    }
    let batch = set.attributes();
    if batch.column_by_name("n").is_some() {
        return Err(invalid("coords already have a column 'n'"));
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

/// Nearest pairs between two sets of samples within `max_distance`, e.g. twin
/// holes or two drilling types.
///
/// Parameters
/// ----------
/// a, b : PointSet or array_like
///     Samples, or their ``(n, 2)`` or ``(n, 3)`` coordinates.
/// max_distance : float
///     Largest pairing distance.
/// values : str or tuple of array_like or str, optional
///     Column of both `a` and `b`, or the values (or column) of each;
///     samples with NaN are not paired.
/// unique : bool
///     Pair each sample at most once, the closest pairs first; else each
///     sample of `a` takes its nearest of `b`, which may repeat.
/// holes : str or tuple of array_like or str, optional
///     Hole ids, as `values`; samples of one hole are not paired.
///
/// Returns
/// -------
/// Table
///     One row per pair, by row of `a`: ``a`` and ``b`` rows, ``distance``
///     and, with `values`, ``value_a`` and ``value_b``.
#[pyfunction]
#[pyo3(signature = (a, b, max_distance, *, values=None, unique=true, holes=None))]
fn pairs(
    a: &Bound<PyAny>,
    b: &Bound<PyAny>,
    max_distance: f64,
    values: Option<&Bound<PyAny>>,
    unique: bool,
    holes: Option<&Bound<PyAny>>,
) -> PyResult<Table> {
    let (sa, sb) = (a, b);
    let (a, b) = (coords_arg(a)?, coords_arg(b)?);
    let values = values
        .map(|v| -> PyResult<_> {
            let (va, vb) = pair(sa, sb, v, "values")?;
            Ok((floats(&va, "values")?, floats(&vb, "values")?))
        })
        .transpose()?;
    let codes = holes
        .map(|h| -> PyResult<_> {
            let (ha, hb) = pair(sa, sb, h, "holes")?;
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
#[pyo3(signature = (coords, *, n=1, targets=None, horizontal=false))]
fn data_spacing<'py>(
    py: Python<'py>,
    coords: &Bound<PyAny>,
    n: usize,
    targets: Option<&Bound<PyAny>>,
    horizontal: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let coords = coords_arg(coords)?;
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

/// Uncertainty quantiles and the share within a threshold per bin of data spacing.
///
/// Parameters
/// ----------
/// spacing : array_like or str
///     Data spacing per row (e.g. `data_spacing`), or its column in `data`.
/// uncertainty : array_like or str
///     Uncertainty per row (e.g. relative error), or its column in `data`.
/// bins : int or array_like, optional
///     Number of equal bins over the spacing range, or bin edges; default
///     Freedman–Diaconis widths.
/// quantiles : sequence of float
///     Probabilities of the uncertainty quantiles.
/// threshold : float
///     Uncertainty counted in ``share``.
/// min_count : int
///     Bins with fewer rows are dropped.
/// data : PointSet, BlockModel, Table or dict, optional
///     Where column names are looked up.
///
/// Returns
/// -------
/// Table
///     One row per bin: ``spacing`` (mean in the bin), ``n``, one column per
///     quantile named ``q0.5``, ``q0.9``, ..., non-decreasing in spacing, and
///     ``share``, the fraction of rows with uncertainty at or below
///     `threshold`. Rows with a null or NaN are skipped.
#[pyfunction]
#[pyo3(signature = (spacing, uncertainty, *, bins=None, quantiles=vec![0.5, 0.9], threshold=0.15, min_count=8, data=None))]
fn uncertainty_curve(
    spacing: &Bound<PyAny>,
    uncertainty: &Bound<PyAny>,
    bins: Option<&Bound<PyAny>>,
    quantiles: Vec<f64>,
    threshold: f64,
    min_count: usize,
    data: Option<&Bound<PyAny>>,
) -> PyResult<Table> {
    let s = floats(&column(data, spacing, "spacing")?, "spacing")?;
    let u = floats(&column(data, uncertainty, "uncertainty")?, "uncertainty")?;
    let bins = match bins {
        None => eda::Bins::Auto,
        Some(b) => match b.extract::<usize>() {
            Ok(k) => eda::Bins::Count(k),
            Err(_) => eda::Bins::Edges(floats(b, "bins")?),
        },
    };
    let rows =
        eda::uncertainty_curve(&s, &u, &bins, &quantiles, threshold, min_count).map_err(invalid)?;
    let mut columns: Vec<(String, ArrayRef)> = vec![
        ("spacing".into(), nullable(rows.iter().map(|r| r.spacing))),
        (
            "n".into(),
            Arc::new(UInt64Array::from_iter_values(
                rows.iter().map(|r| r.n as u64),
            )),
        ),
    ];
    for (k, p) in quantiles.iter().enumerate() {
        columns.push((
            format!("q{p}"),
            nullable(rows.iter().map(|r| r.quantiles[k])),
        ));
    }
    columns.push(("share".into(), nullable(rows.iter().map(|r| r.share))));
    Ok(Table(RecordBatch::try_from_iter(columns).map_err(invalid)?))
}

/// Spacing where an uncertainty curve first rises above a threshold.
///
/// Parameters
/// ----------
/// curve : Table
///     Result of `uncertainty_curve`, or any table with ``spacing`` and
///     `column`, sorted by spacing.
/// column : str
///     Uncertainty column of `curve`.
/// threshold : float
///     Acceptable uncertainty.
///
/// Returns
/// -------
/// float
///     Spacing interpolated linearly between the last row at or below
///     `threshold` and the first above it; the first spacing when the curve
///     starts above, NaN when it never rises above.
#[pyfunction]
#[pyo3(signature = (curve, *, column="q0.9", threshold=0.15))]
fn required_spacing(curve: &Bound<PyAny>, column: &str, threshold: f64) -> PyResult<f64> {
    let get = |name: &str| -> PyResult<Vec<f64>> {
        let c = curve
            .get_item(name)
            .map_err(|_| invalid(format!("curve needs a '{name}' column")))?;
        floats(&c, name)
    };
    Ok(eda::required_spacing(
        &get("spacing")?,
        &get(column)?,
        threshold,
    ))
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(uncertainty_curve, m)?)?;
    m.add_function(wrap_pyfunction!(required_spacing, m)?)?;
    m.add_function(wrap_pyfunction!(pairs, m)?)?;
    m.add_function(wrap_pyfunction!(data_spacing, m)?)?;
    m.add_function(wrap_pyfunction!(paired_bias, m)?)?;
    m.add_function(wrap_pyfunction!(validate_model, m)?)?;
    m.add_function(wrap_pyfunction!(duplicates, m)?)?;
    m.add_function(wrap_pyfunction!(describe, m)?)?;
    m.add_function(wrap_pyfunction!(domain_change, m)?)?;
    m.add_function(wrap_pyfunction!(transition_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(describe_by, m)?)?;
    m.add_function(wrap_pyfunction!(grade_tonnage, m)?)?;
    m.add_function(wrap_pyfunction!(compare_models, m)?)?;
    m.add_function(wrap_pyfunction!(capping_report, m)?)?;
    m.add_function(wrap_pyfunction!(swath, m)?)?;
    m.add_function(wrap_pyfunction!(contact, m)?)?;
    m.add_function(wrap_pyfunction!(soft_boundary, m)?)?;
    m.add_function(wrap_pyfunction!(capping, m)?)?;
    m.add_class::<Capping>()?;
    m.add_function(wrap_pyfunction!(h_scatter, m)?)?;
    m.add_function(wrap_pyfunction!(correlation, m)?)?;
    Ok(())
}
