use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{ArrayRef, Float64Array, RecordBatch, RecordBatchOptions};
use arrow_schema::{DataType, Field, Schema};
use pyo3::PyClass;
use pyo3::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use crate::args::Point;
use crate::invalid;
use crate::io::io_error;

const FORMAT: u64 = 1;

/// `{"type": name, "format": 1, <extra>, <fields of value>...}`.
fn envelope<T: Serialize>(name: &str, value: &T, extra: &str) -> PyResult<String> {
    let fields = serde_json::to_string(value).map_err(invalid)?;
    let fields = match fields.strip_prefix('{') {
        Some("}") => "}".into(),
        Some(rest) => format!(",{rest}"),
        None => return Err(invalid(format!("{name} does not serialize to an object"))),
    };
    Ok(format!(
        r#"{{"type":"{name}","format":{FORMAT}{extra}{fields}"#
    ))
}

/// The fields of an envelope, once its type and format are checked.
fn open(name: &str, text: &str) -> PyResult<Map<String, Value>> {
    let Value::Object(mut map) = serde_json::from_str(text).map_err(invalid)? else {
        return Err(invalid("expected a JSON object"));
    };
    let found = map.remove("type").unwrap_or_default();
    if found.as_str() != Some(name) {
        return Err(invalid(format!("expected a {name}, found {found}")));
    }
    match map.remove("format").and_then(|f| f.as_u64()) {
        Some(f) if f <= FORMAT => Ok(map),
        f => Err(invalid(format!(
            "format {f:?} is not readable by this boitata (up to {FORMAT})"
        ))),
    }
}

fn parse<T: DeserializeOwned>(map: Map<String, Value>) -> PyResult<T> {
    serde_json::from_value(Value::Object(map)).map_err(invalid)
}

pub fn to_json<T: PyClass + Serialize>(value: &T) -> PyResult<String> {
    envelope(<T as PyClass>::NAME, value, "")
}

pub fn from_json<T: PyClass + DeserializeOwned>(text: &str) -> PyResult<T> {
    parse(open(<T as PyClass>::NAME, text)?)
}

/// Equal-length named columns; `None` is null.
pub type Columns = Vec<(String, Vec<Option<f64>>)>;

/// An object holding sample or node arrays: its parameters are JSON, its
/// arrays columns. The arrays are skipped by its serde derive.
pub trait Tabular: Serialize + DeserializeOwned {
    /// None before `fit`.
    fn columns(&self) -> Option<Columns>;
    fn restore(&mut self, columns: Found) -> PyResult<()>;
}

/// Parameters and arrays of `value`, the state its pickle carries.
pub fn state<T: Tabular>(name: &str, value: &T) -> PyResult<(String, Option<Columns>)> {
    let columns = value.columns();
    let fitted = format!(r#","fitted":{}"#, columns.is_some());
    Ok((envelope(name, value, &fitted)?, columns))
}

pub fn from_state<T: Tabular>(name: &str, meta: &str, columns: Option<Columns>) -> PyResult<T> {
    if let Some(c) = &columns
        && c.iter().any(|(_, v)| v.len() != c[0].1.len())
    {
        return Err(invalid("columns differ in length"));
    }
    let mut map = open(name, meta)?;
    let fitted = map.remove("fitted").and_then(|f| f.as_bool()) == Some(true);
    let mut value: T = parse(map)?;
    if let Some(columns) = columns.filter(|_| fitted) {
        value.restore(Found(columns.into_iter().collect()))?;
    }
    Ok(value)
}

/// Arrays as Parquet columns, parameters as JSON under the `boitata` key.
pub fn to_parquet<T: Tabular>(name: &str, value: &T, path: &Path) -> PyResult<()> {
    let (meta, columns) = state(name, value)?;
    let columns = columns.unwrap_or_default();
    let rows = columns.first().map_or(0, |c| c.1.len());
    let fields: Vec<_> = columns
        .iter()
        .map(|(n, _)| Field::new(n, DataType::Float64, true))
        .collect();
    let arrays = columns
        .into_iter()
        .map(|(_, v)| Arc::new(Float64Array::from(v)) as ArrayRef)
        .collect();
    let table = RecordBatch::try_new_with_options(
        Arc::new(Schema::new(fields)),
        arrays,
        &RecordBatchOptions::new().with_row_count(Some(rows)),
    )
    .map_err(invalid)?;
    boitata_io::write_model(path, &table, meta, None).map_err(io_error)
}

pub fn from_parquet<T: Tabular>(name: &str, path: &Path) -> PyResult<T> {
    let (table, meta) = boitata_io::read_model(path, None).map_err(io_error)?;
    let columns = table
        .schema()
        .fields()
        .iter()
        .zip(table.columns())
        .map(|(f, c)| {
            let c = c
                .as_primitive_opt::<Float64Type>()
                .ok_or_else(|| invalid(format!("column {} is not float64", f.name())))?;
            Ok((f.name().clone(), c.iter().collect()))
        })
        .collect::<PyResult<_>>()?;
    from_state(name, &meta, Some(columns))
}

/// Columns read back, taken by name.
pub struct Found(HashMap<String, Vec<Option<f64>>>);

impl Found {
    pub fn optional(&self, name: &str) -> PyResult<Vec<Option<f64>>> {
        self.0
            .get(name)
            .cloned()
            .ok_or_else(|| invalid(format!("missing column {name}")))
    }

    pub fn values(&self, name: &str) -> PyResult<Vec<f64>> {
        self.optional(name)?
            .into_iter()
            .map(|v| v.ok_or_else(|| invalid(format!("null in column {name}"))))
            .collect()
    }

    pub fn indices(&self, name: &str) -> PyResult<Vec<usize>> {
        self.values(name)?.into_iter().map(index).collect()
    }

    pub fn points(&self) -> PyResult<Vec<Point>> {
        let (x, y, z) = (self.values("x")?, self.values("y")?, self.values("z")?);
        if x.len() != y.len() || x.len() != z.len() {
            return Err(invalid("x, y and z differ in length"));
        }
        Ok((0..x.len()).map(|i| (x[i], y[i], z[i])).collect())
    }
}

pub fn index(v: f64) -> PyResult<usize> {
    if v >= 0.0 && v.fract() == 0.0 && v < u32::MAX as f64 {
        Ok(v as usize)
    } else {
        Err(invalid(format!("{v} is not an index")))
    }
}

/// `x`, `y`, `z` columns.
pub fn point_columns(points: impl Iterator<Item = Point> + Clone) -> Columns {
    let axis = |name: &str, f: fn(Point) -> f64| {
        (
            name.to_string(),
            points.clone().map(|p| Some(f(p))).collect(),
        )
    };
    vec![axis("x", |p| p.0), axis("y", |p| p.1), axis("z", |p| p.2)]
}

pub fn column(name: &str, values: impl Iterator<Item = f64>) -> (String, Vec<Option<f64>>) {
    (name.to_string(), values.map(Some).collect())
}
