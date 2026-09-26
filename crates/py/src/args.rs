use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::containers::coords_arg;
use crate::invalid;

pub type Point = (f64, f64, f64);

/// A label: a string, number or boolean.
pub type Label = serde_json::Value;

/// `obj` as a label; integral floats become integers, as `1.0 == 1` in
/// Python. None for None and NaN.
pub fn label(obj: &Bound<PyAny>) -> PyResult<Option<Label>> {
    if obj.is_none() {
        return Ok(None);
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(Some(Label::String(s.to_str()?.into())));
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Some(Label::Bool(b)));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Some(Label::from(i)));
    }
    match obj.extract::<f64>() {
        Ok(f) if f.fract() == 0.0 && f.abs() < 9e15 => Ok(Some(Label::from(f as i64))),
        Ok(f) if f.is_finite() => Ok(Some(Label::from(f))),
        Ok(f) if f.is_nan() => Ok(None),
        _ => Err(invalid(format!(
            "labels must be strings, finite numbers or booleans, not {obj}"
        ))),
    }
}

/// `label` as text: integers without a decimal point.
pub fn text(obj: &Bound<PyAny>) -> PyResult<Option<String>> {
    Ok(label(obj)?.map(|l| match l {
        Label::String(s) => s,
        l => l.to_string(),
    }))
}

/// Labels of a 1-D sequence as text; nulls, including Arrow's, are None.
pub fn texts(obj: &Bound<PyAny>, what: &str) -> PyResult<Vec<Option<String>>> {
    let obj = match obj.hasattr("to_pylist")? {
        true => obj.call_method0("to_pylist")?,
        false => obj.clone(),
    };
    if obj.is_instance_of::<PyString>() {
        return Err(invalid(format!("{what} must be a sequence of labels")));
    }
    obj.try_iter()
        .map_err(|_| invalid(format!("{what} must be a sequence of labels")))?
        .map(|item| text(&item?))
        .collect()
}

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

/// Hole labels as text plus integer codes for `Sample::hole`; ints or strings.
pub fn holes(obj: Option<&Bound<PyAny>>, n: usize) -> PyResult<Option<(Vec<String>, Vec<u32>)>> {
    let Some(obj) = obj else { return Ok(None) };
    let np = obj.py().import("numpy")?;
    let labels: Vec<String> = np
        .call_method1("asarray", (obj,))?
        .call_method1("astype", ("str",))?
        .call_method0("tolist")?
        .extract()
        .map_err(|_| invalid("holes must be a 1-D sequence of ids"))?;
    same_length(n, labels.len(), "holes")?;
    let mut codes = std::collections::HashMap::new();
    let ids = labels
        .iter()
        .map(|l| {
            let next = codes.len() as u32;
            *codes.entry(l.clone()).or_insert(next)
        })
        .collect();
    Ok(Some((labels, ids)))
}

/// Rows to keep so that no two samples share a location: the first of each
/// group. Dropped groups are reported in a `UserWarning` naming their holes
/// (or rows when `holes` is `None`).
pub fn distinct(py: Python, locs: &[Point], holes: Option<&[String]>) -> PyResult<Vec<usize>> {
    distinct_in(py, locs, holes, None)
}

/// As [`distinct`], with locations shared only within each of the `domains`
/// codes, one per row.
pub fn distinct_in(
    py: Python,
    locs: &[Point],
    holes: Option<&[String]>,
    domains: Option<&[u32]>,
) -> PyResult<Vec<usize>> {
    let shared = |rows: &[usize]| -> PyResult<Vec<Vec<usize>>> {
        let coords: Vec<[f64; 3]> = rows.iter().map(|&i| locs[i].into()).collect();
        let groups = eda::duplicates(&coords, 0.0).map_err(invalid)?;
        Ok(groups.into_iter().map(|g| pick(rows, &g)).collect())
    };
    let mut rows: std::collections::BTreeMap<u32, Vec<usize>> = Default::default();
    for i in 0..locs.len() {
        rows.entry(domains.map_or(0, |d| d[i])).or_default().push(i);
    }
    let mut groups = vec![];
    for rows in rows.values() {
        groups.extend(shared(rows)?);
    }
    groups.sort_by_key(|g| g[0]);
    if groups.is_empty() {
        return Ok((0..locs.len()).collect());
    }
    let mut dropped = vec![false; locs.len()];
    let mut lines = Vec::new();
    for g in &groups {
        g[1..].iter().for_each(|&i| dropped[i] = true);
        let who = match holes {
            Some(h) => format!(
                "holes {}",
                g.iter()
                    .map(|&i| h[i].as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            None => format!(
                "rows {}",
                g.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        let (x, y, z) = locs[g[0]];
        lines.push(format!("{who} at ({x}, {y}, {z})"));
    }
    let more = lines.len().saturating_sub(20);
    lines.truncate(20);
    if more > 0 {
        lines.push(format!("and {more} more"));
    }
    let message = format!(
        "{} locations hold several samples; kept the first of each: {}",
        groups.len(),
        lines.join("; ")
    );
    let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
    PyErr::warn(py, &category, &std::ffi::CString::new(message)?, 1)?;
    Ok((0..locs.len()).filter(|&i| !dropped[i]).collect())
}

pub fn pick<T: Clone>(items: &[T], rows: &[usize]) -> Vec<T> {
    rows.iter().map(|&i| items[i].clone()).collect()
}
