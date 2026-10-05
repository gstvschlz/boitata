//! Units carried by numpy outputs, as `boitata._units.UnitArray`.

use arrow_array::RecordBatch;
use boitata_core::units::{with_unit, with_unit_unchecked};
use pyo3::prelude::*;

use crate::invalid;

/// The unit of `values`: its `unit` attribute, the unit of a quantity such
/// as "2.7 t/m3", or the unit of the column of `data` it names.
pub fn of(values: &Bound<PyAny>, data: Option<&Bound<PyAny>>) -> PyResult<Option<String>> {
    if let Ok(name) = values.extract::<String>() {
        if let Ok((_, unit)) = boitata_core::units::quantity(&name) {
            return Ok(Some(unit));
        }
        let Some(units) = data.and_then(|d| d.getattr("units").ok()) else {
            return Ok(None);
        };
        return Ok(units.call_method1("get", (name,))?.extract().ok().flatten());
    }
    Ok(values
        .getattr_opt("unit")?
        .and_then(|u| u.extract::<String>().ok()))
}

/// The length unit of the coordinates `obj` holds, if a container declares one.
pub fn length_of(obj: &Bound<PyAny>) -> PyResult<Option<String>> {
    Ok(obj
        .getattr_opt("length_unit")?
        .and_then(|u| u.extract::<String>().ok()))
}

/// The one length unit among `units`, None when none declares one; errors
/// when two differ.
pub fn common_length(units: impl IntoIterator<Item = Option<String>>) -> PyResult<Option<String>> {
    let mut found: Option<String> = None;
    for unit in units.into_iter().flatten() {
        match &found {
            Some(f) if *f != unit => {
                return Err(invalid(format!(
                    "the variogram and search lengths are in {f} and {unit}; give them one length unit"
                )));
            }
            _ => found = Some(unit),
        }
    }
    Ok(found)
}

/// Errors when `other`, a container, declares another length unit than
/// `own`; `names` say what each holds.
pub fn check_against(own: Option<&str>, other: &Bound<PyAny>, names: (&str, &str)) -> PyResult<()> {
    boitata_core::units::same_length_unit((names.0, own), (names.1, length_of(other)?.as_deref()))
        .map_err(invalid)
}

/// A number in the unit of the data, or text such as "150 ft" or "0.5 g/t".
#[derive(FromPyObject)]
pub enum Quantity {
    Number(f64),
    Text(String),
}

impl Quantity {
    /// The number, and the unit text gives it.
    pub fn split(&self, what: &str) -> PyResult<(f64, Option<String>)> {
        match self {
            Self::Number(v) => Ok((*v, None)),
            Self::Text(t) => boitata_core::units::quantity(t)
                .map(|(v, u)| (v, Some(u)))
                .map_err(|e| invalid(format!("{what}: {e}"))),
        }
    }

    /// The number in `unit`; text needs `unit` known.
    pub fn to(&self, unit: Option<&str>, what: &str) -> PyResult<f64> {
        match (self.split(what)?, unit) {
            ((v, None), _) => Ok(v),
            ((v, Some(from)), Some(to)) => boitata_core::units::conversion(&from, to)
                .map(|k| v * k)
                .map_err(|e| invalid(format!("{what}: {e}"))),
            ((_, Some(from)), None) => Err(invalid(format!(
                "{what} is in {from}, but the data have no unit to convert it to; declare theirs"
            ))),
        }
    }
}

/// Whether `obj` is text holding a quantity such as "2.7 t/m3", rather than
/// a column name.
pub fn is_quantity(obj: &Bound<PyAny>) -> bool {
    obj.extract::<String>()
        .is_ok_and(|t| boitata_core::units::quantity(&t).is_ok())
}

/// `obj`, a number or text such as "150 ft", and the unit the text gives;
/// for parameters whose unit is their own.
pub fn given(obj: &Bound<PyAny>, what: &str) -> PyResult<(f64, Option<String>)> {
    if let Ok(v) = obj.extract::<f64>() {
        return Ok((v, None));
    }
    let text: String = obj
        .extract()
        .map_err(|_| invalid(format!("{what} must be a number or text such as '150 ft'")))?;
    let (v, u) =
        boitata_core::units::quantity(&text).map_err(|e| invalid(format!("{what}: {e}")))?;
    Ok((v, Some(u)))
}

/// `obj`, a number in `unit` or text such as "150 ft", as a number in
/// `unit`; text needs `unit` known.
pub fn value(obj: &Bound<PyAny>, unit: Option<&str>, what: &str) -> PyResult<f64> {
    match (given(obj, what)?, unit) {
        ((v, None), _) => Ok(v),
        ((v, Some(from)), Some(to)) => boitata_core::units::conversion(&from, to)
            .map(|k| v * k)
            .map_err(|e| invalid(format!("{what}: {e}"))),
        ((_, Some(from)), None) => Err(invalid(format!(
            "{what} is in {from}, but the data have no unit to convert it to; declare theirs"
        ))),
    }
}

/// As [`value`], for each item of `obj`.
pub fn values(obj: &Bound<PyAny>, unit: Option<&str>, what: &str) -> PyResult<Vec<f64>> {
    if let Ok(v) = obj.extract::<Vec<f64>>() {
        return Ok(v);
    }
    if obj.extract::<String>().is_ok() {
        return Ok(vec![value(obj, unit, what)?]);
    }
    obj.try_iter()
        .map_err(|_| invalid(format!("{what} must be numbers or texts such as '0.5 g/t'")))?
        .map(|item| value(&item?, unit, what))
        .collect()
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

/// `out` of an estimator, estimates or `(estimates, variances)`, tagged
/// with `unit` and its square.
pub fn tag_estimates<'py>(
    out: Bound<'py, PyAny>,
    unit: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    if unit.is_none() {
        return Ok(out);
    }
    if let Ok(pair) = out.cast::<pyo3::types::PyTuple>()
        && pair.len() == 2
    {
        let estimates = tag(pair.get_item(0)?, unit)?;
        let variances = tag(pair.get_item(1)?, squared(unit).as_deref())?;
        return Ok(pyo3::types::PyTuple::new(out.py(), [estimates, variances])?.into_any());
    }
    tag(out, unit)
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

/// `batch`, a grade–tonnage table, with units: `cutoff` and `mean_grade` in
/// `grade`, `tonnage` (computed in `tonnes`) rescaled to t, kt or Mt and
/// `metal` to the metal units of `grade`, each to read best; the `ratios`
/// columns in `ratio`. A unit not known, or not a mass for `tonnes` or a
/// grade for `grade`, leaves its columns as computed.
pub fn grade_tonnage(
    batch: RecordBatch,
    tonnes: Option<&str>,
    grade: Option<&str>,
    ratios: &[&str],
) -> PyResult<RecordBatch> {
    use boitata_core::units::{TONNAGE, metal_units, parse};
    let kind = |unit: &str, like: &str| matches!((parse(unit), parse(like)), (Ok(a), Ok(b)) if a.dimension() == b.dimension());
    let grade = grade.filter(|g| kind(g, "%"));
    let tonnes = tonnes.filter(|t| kind(t, "t"));
    let mut named: Vec<(&str, Option<&str>)> = ratios.iter().map(|r| (*r, Some("ratio"))).collect();
    named.extend([("cutoff", grade), ("mean_grade", grade)]);
    let present: Vec<(&str, Option<&str>)> = named
        .into_iter()
        .filter(|(n, _)| batch.schema().column_with_name(n).is_some())
        .collect();
    let mut batch = label(batch, &present)?;
    if let Some(t) = tonnes {
        batch = rescale(batch, "tonnage", t, &TONNAGE)?;
        if let Some(g) = grade {
            let raw = format!("({t})*({g})");
            batch = rescale(batch, "metal", &raw, &metal_units(g))?;
            if let (Some(to), Some(_)) = (
                boitata_core::units::unit(&batch, "metal").ok().flatten(),
                batch.column_by_name("benefit"),
            ) {
                let tagged = with_unit_unchecked(&batch, "benefit", Some(&raw)).map_err(invalid)?;
                batch =
                    boitata_core::units::convert_units(&tagged, "benefit", &to).map_err(invalid)?;
            }
        }
    }
    Ok(batch)
}

/// `batch` with column `name`, in `unit`, converted to the unit of `family`
/// it reads best in.
fn rescale(batch: RecordBatch, name: &str, unit: &str, family: &[&str]) -> PyResult<RecordBatch> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::Float64Type;
    let Some(column) = batch.column_by_name(name) else {
        return Ok(batch);
    };
    let values = arrow_cast::cast(column, &arrow_schema::DataType::Float64).map_err(invalid)?;
    let values: Vec<f64> = values
        .as_primitive::<Float64Type>()
        .iter()
        .flatten()
        .collect();
    let (to, _) = boitata_core::units::autoscale(&values, unit, family).map_err(invalid)?;
    let tagged = with_unit_unchecked(&batch, name, Some(unit)).map_err(invalid)?;
    boitata_core::units::convert_units(&tagged, name, &to).map_err(invalid)
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
