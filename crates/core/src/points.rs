use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow_cast::cast;
use arrow_schema::{DataType, Field, Schema};

use crate::{Error, Result, check_rows};

/// Scattered samples: one coordinate triple and one attribute row per point.
#[derive(Debug, Clone)]
pub struct PointSet {
    coords: Vec<[f64; 3]>,
    attributes: RecordBatch,
    pub crs: Option<String>,
}

impl PointSet {
    pub fn new(coords: Vec<[f64; 3]>, attributes: RecordBatch) -> Result<Self> {
        check_rows(coords.len(), &attributes)?;
        Ok(Self {
            coords,
            attributes,
            crs: None,
        })
    }

    /// Splits the coordinate columns out of `table`; without `z`, z is 0.
    pub fn from_table(table: &RecordBatch, x: &str, y: &str, z: Option<&str>) -> Result<Self> {
        let xs = coordinate(table, x)?;
        let ys = coordinate(table, y)?;
        let zs = z.map(|z| coordinate(table, z)).transpose()?;
        let coords = (0..table.num_rows())
            .map(|i| {
                [
                    xs.value(i),
                    ys.value(i),
                    zs.as_ref().map_or(0.0, |z| z.value(i)),
                ]
            })
            .collect();
        let taken = [Some(x), Some(y), z];
        let keep: Vec<usize> = (0..table.num_columns())
            .filter(|&i| !taken.contains(&Some(table.schema().field(i).name().as_str())))
            .collect();
        Self::new(coords, table.project(&keep)?)
    }

    /// Attributes preceded by `x`, `y`, `z` columns.
    pub fn to_table(&self) -> Result<RecordBatch> {
        let mut fields = Vec::with_capacity(self.attributes.num_columns() + 3);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(fields.capacity());
        for (axis, name) in ["x", "y", "z"].into_iter().enumerate() {
            fields.push(Field::new(name, DataType::Float64, false));
            columns.push(Arc::new(Float64Array::from_iter_values(
                self.coords.iter().map(|c| c[axis]),
            )));
        }
        let schema = self.attributes.schema();
        fields.extend(schema.fields().iter().map(|f| f.as_ref().clone()));
        columns.extend(self.attributes.columns().iter().cloned());
        Ok(RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            columns,
        )?)
    }

    pub fn len(&self) -> usize {
        self.coords.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    pub fn coords(&self) -> &[[f64; 3]] {
        &self.coords
    }

    pub fn attributes(&self) -> &RecordBatch {
        &self.attributes
    }
}

fn coordinate(table: &RecordBatch, name: &str) -> Result<Float64Array> {
    let column = table
        .column_by_name(name)
        .ok_or_else(|| Error::MissingColumn(name.into()))?;
    let column =
        cast(column, &DataType::Float64).map_err(|_| Error::InvalidCoordinates(name.into()))?;
    if column.null_count() > 0 {
        return Err(Error::InvalidCoordinates(name.into()));
    }
    Ok(column.as_primitive::<Float64Type>().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StringArray;

    fn table() -> RecordBatch {
        RecordBatch::try_from_iter([
            (
                "east",
                Arc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef,
            ),
            ("north", Arc::new(Float64Array::from(vec![3.0, 4.0]))),
            ("au", Arc::new(Float64Array::from(vec![Some(0.5), None]))),
            ("rock", Arc::new(StringArray::from(vec!["ox", "fr"]))),
        ])
        .unwrap()
    }

    #[test]
    fn from_table_splits_coordinates_and_defaults_z() {
        let p = PointSet::from_table(&table(), "east", "north", None).unwrap();
        assert_eq!(p.coords(), &[[1.0, 3.0, 0.0], [2.0, 4.0, 0.0]]);
        let names: Vec<_> = p
            .attributes()
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, ["au", "rock"]);
        assert_eq!(p.to_table().unwrap().num_columns(), 5);
    }

    #[test]
    fn null_or_text_coordinates_are_rejected() {
        assert!(PointSet::from_table(&table(), "east", "au", None).is_err());
        assert!(PointSet::from_table(&table(), "east", "rock", None).is_err());
        assert!(PointSet::from_table(&table(), "east", "missing", None).is_err());
    }
}
