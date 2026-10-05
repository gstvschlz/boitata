use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek};
use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow_csv::reader::Format;
use arrow_csv::{ReaderBuilder, WriterBuilder};
use arrow_schema::{DataType, Field, Schema};
use arrow_select::concat::concat_batches;
use boitata_core::{Progress, units};
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

const CHUNK: usize = 1 << 16;

/// Reads a headed CSV. Integer and all-null columns become `Float64`.
/// `progress` is ticked per row; its total is the file's line count less the header.
pub fn read_csv(
    path: impl AsRef<Path>,
    options: &CsvOptions,
    progress: Option<&Progress>,
) -> Result<RecordBatch> {
    let format = Format::default()
        .with_header(true)
        .with_delimiter(options.delimiter)
        .with_null_regex(nodata_regex(&options.nodata)?);
    if let Some(p) = progress {
        p.set_total(line_count(File::open(&path)?)?.saturating_sub(1));
    }
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
    let mut batches = Vec::new();
    for batch in ReaderBuilder::new(schema.clone())
        .with_format(format)
        .build(file)?
    {
        let batch = batch?;
        if let Some(p) = progress {
            p.inc_by(batch.num_rows() as u64);
        }
        batches.push(batch);
    }
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
    header_units(RecordBatch::try_new(schema, columns)?)
}

/// Splits headers such as `au [g/t]` or `au (g/t)` into column `au` with unit
/// `g/t`; a bracket that is not a unit stays in the name.
fn header_units(table: RecordBatch) -> Result<RecordBatch> {
    let header = Regex::new(r"^(.+?)\s*(?:\[([^\[\]]+)\]|\(([^()]+)\))\s*$")?;
    let mut found = Vec::new();
    let fields: Vec<Field> = table
        .schema()
        .fields()
        .iter()
        .map(|f| {
            let split = header.captures(f.name()).and_then(|c| {
                let unit = c.get(2).or(c.get(3))?.as_str().trim().to_string();
                units::parse(&unit).ok()?;
                Some((c[1].to_string(), unit))
            });
            match split {
                Some((name, unit)) => {
                    found.push((name.clone(), unit));
                    f.as_ref().clone().with_name(name)
                }
                None => f.as_ref().clone(),
            }
        })
        .collect();
    let mut names: Vec<&String> = fields.iter().map(Field::name).collect();
    names.sort();
    if let Some(w) = names.windows(2).find(|w| w[0] == w[1]) {
        return Err(boitata_core::Error::Units(format!(
            "two columns read as `{}` once their units are split off",
            w[0]
        ))
        .into());
    }
    let mut table = RecordBatch::try_new(Arc::new(Schema::new(fields)), table.columns().to_vec())?;
    for (name, unit) in found {
        table = units::with_unit(&table, &name, Some(&unit))?;
    }
    Ok(table)
}

fn line_count(mut file: File) -> Result<u64> {
    let (mut lines, mut last, mut buf) = (0, b'\n', vec![0; 1 << 16]);
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            return Ok(lines + u64::from(last != b'\n'));
        }
        lines += buf[..n].iter().filter(|&&b| b == b'\n').count() as u64;
        last = buf[n - 1];
    }
}

/// Writes a headed CSV; nulls are written as empty cells. `progress` is ticked
/// by the rows of each chunk written.
pub fn write_csv(
    path: impl AsRef<Path>,
    table: &RecordBatch,
    progress: Option<&Progress>,
) -> Result<()> {
    let mut writer = WriterBuilder::new()
        .with_header(true)
        .build(BufWriter::new(File::create(path)?));
    let rows = table.num_rows();
    let mut offset = 0;
    loop {
        let len = CHUNK.min(rows - offset);
        writer.write(&table.slice(offset, len))?;
        offset += len;
        if let Some(p) = progress {
            p.inc_by(len as u64);
        }
        if offset >= rows {
            return Ok(());
        }
    }
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
        let path = std::env::temp_dir().join(format!("boitata-io-{}-{name}", std::process::id()));
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn nodata_and_integers_are_read_as_nullable_floats() {
        let path = temp(
            "a.csv",
            "id,x,au,rock,empty\n1,10.5,-999,ox,\n2,11,0.3,na,\n3,12,N/A,fr,\n",
        );
        let t = read_csv(&path, &CsvOptions::default(), None).unwrap();
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
    fn header_units_are_split_off_when_they_parse() {
        let path = temp(
            "u.csv",
            "hole (DDH),au [g/t],cu (%),dens (t/m3),note (field)\nA,1,2,2.7,x\n",
        );
        let t = read_csv(&path, &CsvOptions::default(), None).unwrap();
        let names: Vec<String> = t
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, ["hole (DDH)", "au", "cu", "dens", "note (field)"]);
        assert_eq!(
            units::units(&t),
            [("au", "g/t"), ("cu", "%"), ("dens", "t/m3")].map(|(a, b)| (a.into(), b.into()))
        );
        let path = temp("d.csv", "au [g/t],au\n1,2\n");
        assert!(read_csv(&path, &CsvOptions::default(), None).is_err());
    }

    #[test]
    fn exponent_sentinels_are_null() {
        let path = temp("e.csv", "v\n1e21\n1E+21\n2\n");
        let t = read_csv(&path, &CsvOptions::default(), None).unwrap();
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
        let t = read_csv(&path, &options, None).unwrap();
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
            let t = read_csv(&path, &options, None).unwrap();
            (t.column(0).null_count(), t.column(1).null_count())
        };
        assert_eq!(read(Nodata::Number(-999.0)), (2, 1));
        assert_eq!(read(Nodata::Text("-999".into())), (1, 0));
    }

    #[test]
    fn write_then_read_round_trips() {
        let path = temp("c.csv", "a,b\n1.5,x\n,y\n");
        let t = read_csv(&path, &CsvOptions::default(), None).unwrap();
        let out = std::env::temp_dir().join(format!("boitata-io-{}-c-out.csv", std::process::id()));
        let progress = Progress::new(Some(t.num_rows() as u64));
        write_csv(&out, &t, Some(&progress)).unwrap();
        assert_eq!(progress.snapshot().0, t.num_rows() as u64);
        assert_eq!(read_csv(&out, &CsvOptions::default(), None).unwrap(), t);
    }
}
