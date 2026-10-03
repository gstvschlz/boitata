//! Column units, kept in Arrow field metadata under `unit`.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_cast::cast;
use arrow_schema::{DataType, Field, Schema};

use crate::{Error, Result};

const KEY: &str = "unit";

/// Size of one `unit` in the base unit of its kind (mass fraction, metre or
/// t/m3), and the kind.
fn factor(unit: &str) -> Option<(&'static str, f64)> {
    Some(match unit {
        "ppb" => ("grade", 1e-9),
        "ppm" | "g/t" => ("grade", 1e-6),
        "%" => ("grade", 1e-2),
        // troy ounce per short ton
        "oz/t" => ("grade", 31.103_476_8 / 907_184.74),
        "m" => ("length", 1.0),
        "ft" => ("length", 0.3048),
        "t/m3" | "g/cm3" => ("density", 1.0),
        _ => return None,
    })
}

/// The unit of column `name`, if it has one.
pub fn unit(table: &RecordBatch, name: &str) -> Result<Option<String>> {
    let schema = table.schema();
    let field = schema
        .field_with_name(name)
        .map_err(|_| Error::MissingColumn(name.into()))?;
    Ok(field.metadata().get(KEY).cloned())
}

/// The columns that have a unit, in column order.
pub fn units(table: &RecordBatch) -> Vec<(String, String)> {
    table
        .schema()
        .fields()
        .iter()
        .filter_map(|f| Some((f.name().clone(), f.metadata().get(KEY)?.clone())))
        .collect()
}

/// `table` with the unit of column `name` set, or removed for `None`.
pub fn with_unit(table: &RecordBatch, name: &str, unit: Option<&str>) -> Result<RecordBatch> {
    let schema = table.schema();
    let i = schema
        .index_of(name)
        .map_err(|_| Error::MissingColumn(name.into()))?;
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    let mut metadata: HashMap<String, String> = fields[i].metadata().clone();
    match unit {
        Some(u) => metadata.insert(KEY.into(), u.into()),
        None => metadata.remove(KEY),
    };
    fields[i] = fields[i].clone().with_metadata(metadata);
    let schema = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
    Ok(RecordBatch::try_new_with_options(
        schema,
        table.columns().to_vec(),
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(table.num_rows())),
    )?)
}

/// `table` with column `name` converted from its unit to `to`: grades (ppb,
/// ppm, g/t, %, oz/t as troy ounces per short ton), lengths (m, ft) or
/// densities (t/m3, g/cm3).
pub fn convert_units(table: &RecordBatch, name: &str, to: &str) -> Result<RecordBatch> {
    let from =
        unit(table, name)?.ok_or_else(|| Error::Units(format!("column `{name}` has no unit")))?;
    let known = |u: &str| {
        factor(u).ok_or_else(|| {
            Error::Units(format!(
                "unknown unit `{u}`; known: ppb, ppm, g/t, %, oz/t, m, ft, t/m3, g/cm3"
            ))
        })
    };
    let ((kind, a), (to_kind, b)) = (known(&from)?, known(to)?);
    if kind != to_kind {
        return Err(Error::Units(format!(
            "cannot convert {kind} ({from}) to {to_kind} ({to})"
        )));
    }
    let column = table.column(table.schema().index_of(name)?);
    let values = cast(column, &DataType::Float64)
        .map_err(|_| Error::Units(format!("column `{name}` is not numeric")))?;
    let scaled: ArrayRef = Arc::new(
        values
            .as_primitive::<Float64Type>()
            .unary::<_, Float64Type>(|v| v * a / b),
    );
    let replaced = crate::set_column(table, name, scaled)?;
    with_unit(&replaced, name, Some(to))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Float64Array};

    fn grades() -> RecordBatch {
        let au: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.0), None, Some(34.285_714)]));
        let batch = RecordBatch::try_from_iter([("au", au)]).unwrap();
        with_unit(&batch, "au", Some("g/t")).unwrap()
    }

    #[test]
    fn conversions_round_trip_and_keep_nulls() {
        let t = grades();
        assert_eq!(unit(&t, "au").unwrap().as_deref(), Some("g/t"));
        let oz = convert_units(&t, "au", "oz/t").unwrap();
        let v = oz.column(0).as_primitive::<Float64Type>();
        assert!((v.value(2) - 1.0).abs() < 1e-6 && v.is_null(1));
        let back = convert_units(&convert_units(&t, "au", "ppb").unwrap(), "au", "g/t").unwrap();
        assert!((back.column(0).as_primitive::<Float64Type>().value(0) - 1.0).abs() < 1e-12);
        assert_eq!(units(&back), vec![("au".into(), "g/t".into())]);
    }

    #[test]
    fn refuses_other_kinds_and_missing_units() {
        let t = grades();
        assert!(matches!(convert_units(&t, "au", "m"), Err(Error::Units(_))));
        assert!(matches!(
            convert_units(&t, "au", "furlong"),
            Err(Error::Units(_))
        ));
        let none = with_unit(&t, "au", None).unwrap();
        assert!(units(&none).is_empty());
        assert!(matches!(
            convert_units(&none, "au", "ppm"),
            Err(Error::Units(_))
        ));
    }
}
