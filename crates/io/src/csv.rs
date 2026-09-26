use std::fs::File;
use std::io::{BufReader, BufWriter, Seek};
use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow_csv::reader::Format;
use arrow_csv::{ReaderBuilder, WriterBuilder};
use arrow_schema::{DataType, Field, Schema};
use arrow_select::concat::concat_batches;
use regex::Regex;

use crate::{Nodata, Result, default_nodata, is_nodata};

#[derive(Debug, Clone)]
pub struct CsvOptions {
    pub delimiter: u8,
    /// Values read as null; empty cells are always null.
    pub nodata: Vec<Nodata>,
}

impl Default for CsvOptions {
    fn default() -> Self {
        Self {
            delimiter: b',',
            nodata: default_nodata(),
        }
    }
}

/// Reads a headed CSV. Integer and all-null columns become `Float64`.
pub fn read_csv(path: impl AsRef<Path>, options: &CsvOptions) -> Result<RecordBatch> {
    let format = Format::default()
        .with_header(true)
        .with_delimiter(options.delimiter)
        .with_null_regex(nodata_regex(&options.nodata)?);
    let mut file = BufReader::new(File::open(path)?);
    let (inferred, _) = format.infer_schema(&mut file, None)?;
    file.rewind()?;

    let fields: Vec<Field> = inferred
        .fields()
        .iter()
        .map(|f| match f.data_type() {
            DataType::Int64 | DataType::Null => Field::new(f.name(), DataType::Float64, true),
            _ => f.as_ref().clone().with_nullable(true),
        })
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let batches = ReaderBuilder::new(schema.clone())
        .with_format(format)
        .build(file)?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let table = concat_batches(&schema, &batches)?;
    let columns = table
        .columns()
        .iter()
        .map(|c| -> ArrayRef {
            match c.data_type() {
                DataType::Float64 => Arc::new(Float64Array::from_iter(
                    c.as_primitive::<Float64Type>()
                        .iter()
                        .map(|v| v.filter(|v| !options.nodata.contains(&Nodata::Number(*v)))),
                )),
                DataType::Utf8 => Arc::new(StringArray::from_iter(
                    c.as_string::<i32>()
                        .iter()
                        .map(|s| s.filter(|s| !is_nodata(s, &options.nodata))),
                )),
                _ => c.clone(),
            }
        })
        .collect();
    Ok(RecordBatch::try_new(schema, columns)?)
}

/// Writes a headed CSV; nulls are written as empty cells.
pub fn write_csv(path: impl AsRef<Path>, table: &RecordBatch) -> Result<()> {
    let mut writer = WriterBuilder::new()
        .with_header(true)
        .build(BufWriter::new(File::create(path)?));
    writer.write(table)?;
    Ok(())
}

fn nodata_regex(nodata: &[Nodata]) -> Result<Regex> {
    let tokens: Vec<String> = nodata
        .iter()
        .filter_map(|n| match n {
            Nodata::Text(s) => Some(regex::escape(s)),
            Nodata::Number(_) => None,
        })
        .collect();
    Ok(Regex::new(&format!("^(?i:{}|)$", tokens.join("|")))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;

    fn temp(name: &str, contents: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("ceres-io-{}-{name}", std::process::id()));
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn nodata_and_integers_are_read_as_nullable_floats() {
        let path = temp(
            "a.csv",
            "id,x,au,rock,empty\n1,10.5,-999,ox,\n2,11,0.3,na,\n3,12,N/A,fr,\n",
        );
        let t = read_csv(&path, &CsvOptions::default()).unwrap();
        assert_eq!(t.num_rows(), 3);
        for name in ["id", "x", "au", "empty"] {
            assert_eq!(
                t.schema().field_with_name(name).unwrap().data_type(),
                &DataType::Float64
            );
        }
        let au = t
            .column_by_name("au")
            .unwrap()
            .as_primitive::<Float64Type>();
        assert_eq!(au.iter().collect::<Vec<_>>(), [None, Some(0.3), None]);
        assert_eq!(t.column_by_name("rock").unwrap().null_count(), 1);
    }

    #[test]
    fn exponent_sentinels_are_null() {
        let path = temp("e.csv", "v\n1e21\n1E+21\n2\n");
        let t = read_csv(&path, &CsvOptions::default()).unwrap();
        let v = t.column_by_name("v").unwrap().as_primitive::<Float64Type>();
        assert_eq!(v.iter().collect::<Vec<_>>(), [None, None, Some(2.0)]);
    }

    #[test]
    fn custom_nodata_replaces_defaults() {
        let path = temp("b.csv", "v\n-999\n-1\n");
        let options = CsvOptions {
            nodata: vec![Nodata::Number(-1.0)],
            ..Default::default()
        };
        let t = read_csv(&path, &options).unwrap();
        let v = t.column(0).as_primitive::<Float64Type>();
        assert_eq!(v.iter().collect::<Vec<_>>(), [Some(-999.0), None]);
    }

    #[test]
    fn numbers_match_numerically_and_text_as_tokens() {
        let path = temp("n.csv", "v,rock\n-999.0,-999.0\n-999,ox\n");
        let read = |n: Nodata| {
            let options = CsvOptions {
                nodata: vec![n],
                ..Default::default()
            };
            let t = read_csv(&path, &options).unwrap();
            (t.column(0).null_count(), t.column(1).null_count())
        };
        assert_eq!(read(Nodata::Number(-999.0)), (2, 1));
        assert_eq!(read(Nodata::Text("-999".into())), (1, 0));
    }

    #[test]
    fn write_then_read_round_trips() {
        let path = temp("c.csv", "a,b\n1.5,x\n,y\n");
        let t = read_csv(&path, &CsvOptions::default()).unwrap();
        let out = std::env::temp_dir().join(format!("ceres-io-{}-c-out.csv", std::process::id()));
        write_csv(&out, &t).unwrap();
        assert_eq!(read_csv(&out, &CsvOptions::default()).unwrap(), t);
    }
}
