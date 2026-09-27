use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch};
use ceres_core::Categories as Core;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use transforms::VerticalCurve;

use crate::args::{
    array1, column, floats, optional_finite, per_row, pick, points, rows, same_length, text, texts,
};
use crate::invalid;
use crate::table::Table;

/// Named categories with integer codes: a code is the index of its name.
///
/// Parameters
/// ----------
/// names : sequence of str
///     Category names in code order.
/// colors : sequence of str, optional
///     One matplotlib color per name; None picks them at plot time.
/// mapping : dict, optional
///     Raw label to name, many to one.
/// other : str, optional
///     Name that unlisted labels fall into; appended when not listed, and
///     always the last code.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
#[pyclass(module = "ceres", name = "Categories", frozen, eq, from_py_object)]
pub struct Categories(pub Core);

fn name(obj: Bound<PyAny>) -> PyResult<String> {
    text(&obj)?.ok_or_else(|| invalid("names and mapping entries must not be null"))
}

fn mapping(obj: Option<&Bound<PyDict>>) -> PyResult<BTreeMap<String, String>> {
    obj.map_or(Ok(BTreeMap::new()), |d| {
        d.iter().map(|(k, v)| Ok((name(k)?, name(v)?))).collect()
    })
}

pub(crate) fn labels(obj: &Bound<PyAny>) -> PyResult<Vec<Option<String>>> {
    texts(obj, "values")
}

pub(crate) fn refs(labels: &[Option<String>]) -> Vec<Option<&str>> {
    labels.iter().map(Option::as_deref).collect()
}

fn codes(obj: &Bound<PyAny>) -> PyResult<Vec<Option<u32>>> {
    floats(obj, "codes")?
        .into_iter()
        .map(|c| match c {
            c if c.is_nan() => Ok(None),
            c if c >= 0.0 && c.fract() == 0.0 && c < u32::MAX as f64 => Ok(Some(c as u32)),
            c => Err(invalid(format!("{c} is not a code"))),
        })
        .collect()
}

#[pymethods]
impl Categories {
    #[new]
    #[pyo3(signature = (names, *, colors=None, mapping=None, other=None))]
    fn new(
        names: &Bound<PyAny>,
        colors: Option<Vec<String>>,
        mapping: Option<&Bound<PyDict>>,
        other: Option<&Bound<PyAny>>,
    ) -> PyResult<Self> {
        let names = names
            .try_iter()?
            .map(|n| name(n?))
            .collect::<PyResult<_>>()?;
        let other = other.map(|o| name(o.clone())).transpose()?;
        Core::new(names, colors, self::mapping(mapping)?, other)
            .map(Self)
            .map_err(invalid)
    }

    /// Categories of observed values.
    ///
    /// `mapping` is applied first; the distinct names are then sorted,
    /// numerically when all are integers. Names whose weighted share is
    /// below `min_share` are lumped into `other`, the last code, and
    /// recorded in `mapping`; `other` exists only when something was
    /// lumped. Integral floats are integer labels (1.0 is "1"); None and
    /// NaN are skipped. The result does not depend on row order.
    #[staticmethod]
    #[pyo3(signature = (values, *, weights=None, min_share=0.0, mapping=None, other="other", colors=None))]
    fn from_values(
        values: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
        min_share: f64,
        mapping: Option<&Bound<PyDict>>,
        other: &str,
        colors: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let labels = labels(values)?;
        let weights = optional_finite(weights, "weights")?;
        let c = Core::from_labels(
            &refs(&labels),
            weights.as_deref(),
            min_share,
            self::mapping(mapping)?,
            other,
        )
        .map_err(invalid)?;
        let other = c.other().map(str::to_string);
        Core::new(c.names().to_vec(), colors, c.mapping().clone(), other)
            .map(Self)
            .map_err(invalid)
    }

    /// Code of each value as float64, NaN for null. 1, 1.0 and "1" are one
    /// label. An unlisted label goes to `other`, else raises InvalidInput
    /// naming it and its row count.
    fn encode<'py>(&self, py: Python<'py>, values: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let codes = self.0.encode(&refs(&labels(values)?)).map_err(invalid)?;
        let codes = codes.into_iter().map(|c| c.map_or(f64::NAN, f64::from));
        Ok(array1(py, codes.collect()).into_any())
    }

    /// Name of each code; None for NaN.
    fn decode(&self, codes: &Bound<PyAny>) -> PyResult<Vec<Option<String>>> {
        let names = self.0.decode(&self::codes(codes)?).map_err(invalid)?;
        Ok(names.into_iter().map(|n| n.map(str::to_string)).collect())
    }

    /// New categories with `names` merged into `into`: a listed name, else
    /// `other` when there is none, else a new name before `other`. The
    /// merged names become mapping entries.
    #[pyo3(signature = (names, *, into="other"))]
    fn lump(&self, names: Vec<String>, into: &str) -> PyResult<Self> {
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        self.0.lump(&names, into).map(Self).map_err(invalid)
    }

    /// Weighted proportion of each code, nulls skipped.
    #[pyo3(signature = (codes, *, weights=None))]
    fn shares<'py>(
        &self,
        py: Python<'py>,
        codes: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let weights = optional_finite(weights, "weights")?;
        let shares = self
            .0
            .shares(&self::codes(codes)?, weights.as_deref())
            .map_err(invalid)?;
        Ok(array1(py, shares).into_any())
    }

    #[getter]
    fn names(&self) -> Vec<String> {
        self.0.names().to_vec()
    }

    #[getter]
    fn colors(&self) -> Option<Vec<String>> {
        self.0.colors().map(<[String]>::to_vec)
    }

    #[getter]
    fn other(&self) -> Option<&str> {
        self.0.other()
    }

    #[getter]
    fn mapping(&self) -> BTreeMap<String, String> {
        self.0.mapping().clone()
    }

    fn to_json(&self) -> PyResult<String> {
        crate::persist::to_json(self)
    }

    /// Reads `to_json` output; raises InvalidInput on another class's JSON
    /// or a newer format.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        crate::persist::from_json(text)
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __repr__(&self) -> String {
        let other = self
            .0
            .other()
            .map_or(String::new(), |o| format!(", other={o:?}"));
        format!("Categories({:?}{other})", self.0.names())
    }
}

/// Names and per-row codes, None for null, of `labels`: encoded by `scheme`,
/// else in the order of `Categories.from_values`.
pub(crate) fn coded(
    labels: &[Option<String>],
    scheme: Option<&Categories>,
) -> PyResult<(Vec<String>, Vec<Option<u32>>)> {
    let refs = refs(labels);
    let scheme = match scheme {
        Some(s) => s.0.clone(),
        None => Core::from_labels(&refs, None, 0.0, BTreeMap::new(), "other").map_err(invalid)?,
    };
    let codes = scheme.encode(&refs).map_err(invalid)?;
    Ok((scheme.names().to_vec(), codes))
}

fn float_table(columns: Vec<(String, Vec<f64>)>) -> PyResult<Table> {
    let columns = columns
        .into_iter()
        .map(|(name, v)| (name, Arc::new(Float64Array::from(v)) as ArrayRef));
    RecordBatch::try_from_iter(columns)
        .map(Table)
        .map_err(invalid)
}

/// `elevation` of each row of `coords`, or their z.
fn heights(
    coords: &Bound<PyAny>,
    elevation: Option<&Bound<PyAny>>,
    n: usize,
) -> PyResult<Vec<f64>> {
    match elevation {
        Some(e) => per_row(Some(coords), e, n, "elevation"),
        None => Ok(points(coords)?.into_iter().map(|p| p.2).collect()),
    }
}

/// Vertical proportion curve: the weighted proportion of each category in
/// slices of height.
///
/// Parameters
/// ----------
/// coords : array_like, PointSet or BlockModel
///     Sample locations, shape (n, 2) or (n, 3), or a container.
/// categories : array_like or str
///     Category label of each sample, or the column of `coords` holding them;
///     null labels are skipped.
/// size : float
///     Slice thickness; slices start at multiples of it.
/// elevation : array_like or str, optional
///     Height of each sample, e.g. stratigraphic height above a base
///     surface, or the column of `coords` holding it; default the z
///     coordinate.
/// weights : array_like or str, optional
///     Declustering weights or lengths, or the column of `coords` holding them.
/// scheme : Categories, optional
///     Order and names of the categories, which encode the labels; default
///     the order of `Categories.from_values`.
///
/// Returns
/// -------
/// Table
///     One row per slice holding data, bottom up: ``elevation``, the slice
///     center, ``weight``, the weight in it, and the proportion of each
///     category, one column per name, summing to 1.
#[pyfunction]
#[pyo3(signature = (coords, categories, *, size, elevation=None, weights=None, scheme=None))]
fn vertical_proportions(
    coords: &Bound<PyAny>,
    categories: &Bound<PyAny>,
    size: f64,
    elevation: Option<&Bound<PyAny>>,
    weights: Option<&Bound<PyAny>>,
    scheme: Option<PyRef<Categories>>,
) -> PyResult<Table> {
    let labels = labels(&column(Some(coords), categories, "categories")?)?;
    let n = labels.len();
    let heights = heights(coords, elevation, n)?;
    same_length(n, heights.len(), "categories")?;
    let weights = weights
        .map(|w| per_row(Some(coords), w, n, "weights"))
        .transpose()?;
    let (names, codes) = coded(&labels, scheme.as_deref())?;
    let keep: Vec<usize> = (0..n).filter(|&i| codes[i].is_some()).collect();
    let cats: Vec<usize> = keep
        .iter()
        .map(|&i| codes[i].unwrap_or(0) as usize)
        .collect();
    let curve = VerticalCurve::fit(
        &pick(&heights, &keep),
        &cats,
        names.len(),
        weights.map(|w| pick(&w, &keep)).as_deref(),
        size,
    )
    .map_err(invalid)?;
    let mut columns = vec![
        ("elevation".to_string(), curve.heights),
        ("weight".to_string(), curve.weights),
    ];
    for (c, name) in names.into_iter().enumerate() {
        columns.push((name, curve.proportions.iter().map(|r| r[c]).collect()));
    }
    float_table(columns)
}

/// Category proportions in 3D from a vertical proportion curve and areal
/// proportion maps: at each target, the curve at its height times the areal
/// proportions over the global ones, rescaled to sum 1. Areal proportions
/// equal to the global ones everywhere give back the curve.
///
/// Parameters
/// ----------
/// coords : array_like, PointSet or BlockModel
///     Target locations, e.g. BlockModel cells.
/// vertical : Table
///     A curve from `vertical_proportions`: ``elevation``, ``weight`` and one
///     column per category. The global proportions are its weighted mean.
///     It is linear between slice centers and constant beyond the ends.
/// areal : Table or array_like
///     Areal proportions at the targets, e.g. `Trend.predict` of a
///     categorical trend at their x and y: a Table with the curve's
///     category columns, or an (n, k) array in the curve's order. A row
///     with a NaN keeps the curve alone.
/// elevation : array_like or str, optional
///     Height of each target, as in `vertical_proportions`; default the z
///     coordinate.
///
/// Returns
/// -------
/// Table
///     One column per category, each in [0, 1], summing to 1 in every row;
///     `proportions=` of `SIS` and `Plurigaussian`.
#[pyfunction]
#[pyo3(signature = (coords, vertical, areal, *, elevation=None))]
fn combine_proportions(
    py: Python,
    coords: &Bound<PyAny>,
    vertical: &Bound<PyAny>,
    areal: &Bound<PyAny>,
    elevation: Option<&Bound<PyAny>>,
) -> PyResult<Table> {
    use crate::table::floats;
    let curve = crate::table::to_batch(vertical)?;
    let names: Vec<String> = crate::table::names(&curve)
        .into_iter()
        .filter(|n| n != "elevation" && n != "weight")
        .collect();
    let columns: Vec<Vec<f64>> = names
        .iter()
        .map(|n| floats(&curve, n))
        .collect::<PyResult<_>>()?;
    let curve = VerticalCurve::new(
        floats(&curve, "elevation")?,
        floats(&curve, "weight")?,
        (0..curve.num_rows())
            .map(|i| columns.iter().map(|c| c[i]).collect())
            .collect(),
    )
    .map_err(invalid)?;
    let rows = match areal.cast::<Table>() {
        Ok(t) => {
            let columns: Vec<Vec<f64>> = names
                .iter()
                .map(|n| floats(&t.get().0, n))
                .collect::<PyResult<_>>()?;
            (0..t.get().0.num_rows())
                .map(|i| columns.iter().map(|c| c[i]).collect())
                .collect()
        }
        Err(_) => rows(areal, "areal")?,
    };
    let heights = heights(coords, elevation, rows.len())?;
    same_length(heights.len(), rows.len(), "areal")?;
    let areal: Vec<Option<Vec<f64>>> = rows
        .into_iter()
        .map(|r| (!r.iter().any(|p| p.is_nan())).then_some(r))
        .collect();
    let combined = py
        .detach(|| curve.combine(&heights, &areal))
        .map_err(invalid)?;
    let columns = names
        .into_iter()
        .enumerate()
        .map(|(c, name)| (name, combined.iter().map(|r| r[c]).collect()))
        .collect();
    float_table(columns)
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Categories>()?;
    m.add_function(wrap_pyfunction!(vertical_proportions, m)?)?;
    m.add_function(wrap_pyfunction!(combine_proportions, m)?)
}
