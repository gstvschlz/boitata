//! Shared loading and output helpers for the example binaries.

use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{ArrayRef, Float64Array, RecordBatch};
use ceres_core::PointSet;
use ceres_io::{CsvOptions, read_csv, write_csv};

pub type Point = (f64, f64, f64);

fn dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(name)
}

/// Loads `examples/data/<file>` with `X`, `Y` as coordinates.
pub fn load(file: &str) -> PointSet {
    let table = read_csv(dir("data").join(file), &CsvOptions::default())
        .unwrap_or_else(|e| panic!("{file}: {e}; run `mise run examples:data` first"));
    PointSet::from_table(&table, "X", "Y", None).expect("X and Y columns")
}

pub fn locations(points: &PointSet) -> Vec<Point> {
    points.coords().iter().map(|c| (c[0], c[1], c[2])).collect()
}

/// Non-null values of `name`, with the matching locations.
pub fn variable(points: &PointSet, name: &str) -> (Vec<Point>, Vec<f64>) {
    let column = points
        .attributes()
        .column_by_name(name)
        .unwrap_or_else(|| panic!("no column {name}"))
        .as_primitive::<Float64Type>();
    locations(points)
        .into_iter()
        .zip(column.iter())
        .filter_map(|(p, v)| v.map(|v| (p, v)))
        .unzip()
}

/// Writes equally long columns to `examples/out/<file>`.
pub fn write(file: &str, columns: &[(&str, &[f64])]) {
    let arrays = columns.iter().map(|(name, values)| {
        (
            *name,
            Arc::new(Float64Array::from(values.to_vec())) as ArrayRef,
        )
    });
    let table = RecordBatch::try_from_iter(arrays).expect("equal column lengths");
    std::fs::create_dir_all(dir("out")).expect("create examples/out");
    write_csv(dir("out").join(file), &table).expect("write output");
}

pub fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}
