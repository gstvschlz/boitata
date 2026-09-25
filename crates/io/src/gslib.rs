use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{Array, ArrayRef, Float64Array, RecordBatch, RecordBatchOptions};
use arrow_cast::cast;
use arrow_schema::{DataType, Field, Schema};

use crate::{Error, Result, is_nodata};

/// Reads a GSLIB file: title, column count, one name per line, then
/// whitespace-separated rows. Extra tokens on a row are ignored. The title is
/// kept in the schema metadata under `title`.
pub fn read_gslib(path: impl AsRef<Path>, nodata: &[String]) -> Result<RecordBatch> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text
        .lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l.trim()))
        .filter(|(_, l)| !l.is_empty());
    let bad = |line, message: &str| Error::Format {
        line,
        message: message.into(),
    };

    let (_, title) = lines.next().ok_or_else(|| bad(1, "missing title"))?;
    let (line, count) = lines.next().ok_or_else(|| bad(2, "missing column count"))?;
    let count: usize = count
        .split_whitespace()
        .next()
        .and_then(|n| n.parse().ok())
        .ok_or_else(|| bad(line, "column count is not an integer"))?;
    let names: Vec<&str> = lines.by_ref().take(count).map(|(_, l)| l).collect();
    if names.len() < count {
        return Err(bad(line, "fewer column names than declared"));
    }

    let mut columns = vec![Vec::new(); count];
    for (line, row) in lines {
        let tokens: Vec<&str> = row.split_whitespace().collect();
        if tokens.len() < count {
            return Err(bad(
                line,
                &format!("expected {count} values, found {}", tokens.len()),
            ));
        }
        for (column, token) in columns.iter_mut().zip(tokens) {
            let value = if is_nodata(token, nodata) {
                None
            } else {
                Some(
                    token
                        .parse::<f64>()
                        .map_err(|_| bad(line, &format!("`{token}` is not a number")))?,
                )
            };
            column.push(value);
        }
    }

    let rows = columns.first().map_or(0, Vec::len);
    let fields: Vec<Field> = names
        .iter()
        .map(|n| Field::new(*n, DataType::Float64, true))
        .collect();
    let metadata = HashMap::from([("title".to_string(), title.to_string())]);
    let columns: Vec<ArrayRef> = columns
        .into_iter()
        .map(|c| Arc::new(Float64Array::from(c)) as ArrayRef)
        .collect();
    Ok(RecordBatch::try_new_with_options(
        Arc::new(Schema::new_with_metadata(fields, metadata)),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(rows)),
    )?)
}

/// Writes numeric columns as GSLIB; nulls are written as `missing`.
pub fn write_gslib(path: impl AsRef<Path>, table: &RecordBatch, missing: f64) -> Result<()> {
    let schema = table.schema();
    if let Some(f) = schema.fields().iter().find(|f| !f.data_type().is_numeric()) {
        return Err(Error::NotNumeric(f.name().clone()));
    }
    let columns = table
        .columns()
        .iter()
        .map(|c| cast(c, &DataType::Float64))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let columns: Vec<&Float64Array> = columns
        .iter()
        .map(|c| c.as_primitive::<Float64Type>())
        .collect();

    let mut out = BufWriter::new(File::create(path)?);
    let title = schema
        .metadata()
        .get("title")
        .map_or("ceres", String::as_str);
    writeln!(out, "{title}\n{}", columns.len())?;
    for field in schema.fields() {
        writeln!(out, "{}", field.name())?;
    }
    for row in 0..table.num_rows() {
        let values: Vec<String> = columns
            .iter()
            .map(|c| {
                if c.is_valid(row) {
                    c.value(row)
                } else {
                    missing
                }
                .to_string()
            })
            .collect();
        writeln!(out, "{}", values.join(" "))?;
    }
    Ok(out.flush()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_nodata;

    fn temp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("ceres-io-{}-{name}", std::process::id()))
    }

    #[test]
    fn reads_header_nulls_and_ignores_extra_tokens() {
        let path = temp("a.dat");
        std::fs::write(&path, "walker\n3\nx\ny\nv\n1 2 3.5\n\n4 5 -999 99\n").unwrap();
        let t = read_gslib(&path, &default_nodata()).unwrap();
        assert_eq!(t.num_rows(), 2);
        assert_eq!(t.schema().metadata()["title"], "walker");
        let v = t.column(2).as_primitive::<Float64Type>();
        assert_eq!(v.iter().collect::<Vec<_>>(), [Some(3.5), None]);
    }

    #[test]
    fn short_or_text_rows_report_the_line() {
        let path = temp("b.dat");
        std::fs::write(&path, "t\n2\na\nb\n1 2\n3\n").unwrap();
        assert!(matches!(
            read_gslib(&path, &[]),
            Err(Error::Format { line: 6, .. })
        ));
        std::fs::write(&path, "t\n1\na\nabc\n").unwrap();
        assert!(matches!(
            read_gslib(&path, &[]),
            Err(Error::Format { line: 4, .. })
        ));
    }

    #[test]
    fn write_then_read_round_trips() {
        let path = temp("c.dat");
        std::fs::write(&path, "grid\n2\nau\ncu\n1 -999\n0.25 3\n").unwrap();
        let t = read_gslib(&path, &default_nodata()).unwrap();
        let out = temp("c-out.dat");
        write_gslib(&out, &t, -999.0).unwrap();
        let back = read_gslib(&out, &default_nodata()).unwrap();
        assert_eq!(back, t);
        assert_eq!(back.column(1).null_count(), 1);
    }

    #[test]
    fn text_columns_are_not_written() {
        let t = RecordBatch::try_from_iter([(
            "rock",
            Arc::new(arrow_array::StringArray::from(vec!["ox"])) as ArrayRef,
        )])
        .unwrap();
        assert!(matches!(
            write_gslib(temp("d.dat"), &t, -999.0),
            Err(Error::NotNumeric(_))
        ));
    }
}
