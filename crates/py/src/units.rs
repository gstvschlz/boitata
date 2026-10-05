//! Units carried by numpy outputs, as `boitata._units.UnitArray`.

use arrow_array::RecordBatch;
use boitata_core::units::{with_unit, with_unit_unchecked};
use pyo3::prelude::*;

use crate::invalid;

/// The unit of `values`: its `unit` attribute, or the unit of the column of
/// `data` it names.
pub fn of(values: &Bound<PyAny>, data: Option<&Bound<PyAny>>) -> PyResult<Option<String>> {
    if let Ok(name) = values.extract::<String>() {
        let Some(units) = data.and_then(|d| d.getattr("units").ok()) else {
            return Ok(None);
        };
        return Ok(units.call_method1("get", (name,))?.extract().ok().flatten());
    }
    Ok(values
        .getattr_opt("unit")?
        .and_then(|u| u.extract::<String>().ok()))
}

/// `array` tagged with `unit`, or `array` itself without one.
pub fn tag<'py>(array: Bound<'py, PyAny>, unit: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    match unit {
        None => Ok(array),
        Some(u) => array
            .py()
            .import("boitata._units")?
            .getattr("UnitArray")?
            .call1((array, u)),
    }
}

/// The unit of a variance of values in `unit`.
pub fn squared(unit: Option<&str>) -> Option<String> {
    unit.map(|u| format!("({u})^2"))
}

/// `batch` with column `name` in `unit` when given, else in the unit
/// `values` carries; an inherited unit is kept even when not understood.
pub fn set(
    batch: &RecordBatch,
    name: &str,
    values: &Bound<PyAny>,
    unit: Option<&str>,
) -> PyResult<RecordBatch> {
    match (unit, of(values, None)?) {
        (Some(u), _) => with_unit(batch, name, Some(u)).map_err(invalid),
        (None, Some(u)) => with_unit_unchecked(batch, name, Some(&u)).map_err(invalid),
        (None, None) => Ok(batch.clone()),
    }
}

/// `batch` with the units of `columns`, column name to unit, as inherited.
pub fn label(batch: RecordBatch, columns: &[(&str, Option<&str>)]) -> PyResult<RecordBatch> {
    let mut batch = batch;
    for (name, unit) in columns {
        if unit.is_some() {
            batch = with_unit_unchecked(&batch, name, *unit).map_err(invalid)?;
        }
    }
    Ok(batch)
}
