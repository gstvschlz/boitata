//! Parquet storage of tables and containers. Container geometry, layout and
//! CRS are stored as JSON under the `ceres` key of the file's metadata, so
//! any Parquet reader still sees a plain table.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, UInt64Type};
use arrow_array::{ArrayRef, Float64Array, RecordBatch, RecordBatchOptions, UInt64Array};
use arrow_schema::{Field, Schema};
use arrow_select::concat::concat_batches;
use ceres_core::{BlockModel, Geometry, Layout, PointSet};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use serde_json::{Value, json};

use crate::{Error, Result};

const KEY: &str = "ceres";
const INDEX: &str = "__ceres_index";
const EXTENT: [&str; 6] = [
    "__ceres_u0",
    "__ceres_v0",
    "__ceres_w0",
    "__ceres_u1",
    "__ceres_v1",
    "__ceres_w1",
];
const ROW_GROUP: usize = 1 << 20;

/// What a Parquet file holds.
#[derive(Debug)]
pub enum Stored {
    Table(RecordBatch),
    Points(PointSet),
    Blocks(BlockModel),
}

fn write(path: &Path, table: &RecordBatch, meta: Option<Value>) -> Result<()> {
    let mut metadata = table.schema().metadata().clone();
    if let Some(meta) = meta {
        metadata.insert(KEY.into(), meta.to_string());
    }
    let schema = Arc::new(Schema::new_with_metadata(
        table.schema().fields().clone(),
        metadata,
    ));
    let batch = RecordBatch::try_new_with_options(
        schema.clone(),
        table.columns().to_vec(),
        &RecordBatchOptions::new().with_row_count(Some(table.num_rows())),
    )?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .set_max_row_group_row_count(Some(ROW_GROUP))
        .build();
    let mut writer = ArrowWriter::try_new(File::create(path)?, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Writes a plain table.
pub fn write_parquet(path: impl AsRef<Path>, table: &RecordBatch) -> Result<()> {
    write(path.as_ref(), table, None)
}

/// Writes points as `x`, `y`, `z` and their attributes.
pub fn write_points(path: impl AsRef<Path>, points: &PointSet) -> Result<()> {
    let meta = json!({ "kind": "points", "crs": points.crs });
    write(path.as_ref(), &points.to_table()?, Some(meta))
}

/// Writes a block model's attributes, with its cell index when masked and its
/// parent index and extents when sub-blocked.
pub fn write_block_model(path: impl AsRef<Path>, model: &BlockModel) -> Result<()> {
    let g = model.geometry();
    let mut meta = json!({
        "kind": "block_model",
        "crs": model.crs,
        "origin": g.origin,
        "size": g.size,
        "count": g.count,
        "rotation": g.rotation,
        "layout": "regular",
    });
    let mut extra: Vec<(String, ArrayRef)> = vec![];
    match model.layout() {
        Layout::Regular => {}
        Layout::Masked(index) => {
            meta["layout"] = json!("masked");
            extra.push((INDEX.into(), Arc::new(UInt64Array::from(index.clone()))));
        }
        Layout::SubBlocked {
            parent,
            extent,
            grid,
        } => {
            meta["layout"] = json!("subblocked");
            meta["grid"] = json!(grid);
            extra.push((INDEX.into(), Arc::new(UInt64Array::from(parent.clone()))));
            for (i, name) in EXTENT.iter().enumerate() {
                let column = Float64Array::from_iter_values(extent.iter().map(|e| e[i]));
                extra.push((name.to_string(), Arc::new(column)));
            }
        }
    }
    let table = model.attributes();
    let mut fields: Vec<_> = table.schema().fields().iter().cloned().collect();
    let mut columns = table.columns().to_vec();
    for (name, column) in extra {
        fields.push(Arc::new(Field::new(
            name,
            column.data_type().clone(),
            false,
        )));
        columns.push(column);
    }
    let table = RecordBatch::try_new_with_options(
        Arc::new(Schema::new_with_metadata(
            fields,
            table.schema().metadata().clone(),
        )),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(table.num_rows())),
    )?;
    write(path.as_ref(), &table, Some(meta))
}

fn bad(message: impl Into<String>) -> Error {
    Error::Metadata(message.into())
}

fn triple<T: serde::de::DeserializeOwned>(meta: &Value, key: &str) -> Result<[T; 3]> {
    serde_json::from_value(meta[key].clone())
        .map_err(|_| bad(format!("`{key}` must hold 3 values")))
}

/// Reads a file written by any Parquet tool; files written from a container
/// come back as that container.
pub fn read_parquet(path: impl AsRef<Path>) -> Result<Stored> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
    let schema = builder.schema().clone();
    let batches = builder
        .with_batch_size(ROW_GROUP)
        .build()?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut table = concat_batches(&schema, &batches)?;
    let Some(meta) = schema.metadata().get(KEY) else {
        return Ok(Stored::Table(table));
    };
    let meta: Value = serde_json::from_str(meta).map_err(|e| bad(e.to_string()))?;
    let crs = meta["crs"].as_str().map(str::to_string);
    let mut metadata: HashMap<String, String> = schema.metadata().clone();
    metadata.remove(KEY);
    table = RecordBatch::try_new_with_options(
        Arc::new(Schema::new_with_metadata(
            table.schema().fields().clone(),
            metadata,
        )),
        table.columns().to_vec(),
        &RecordBatchOptions::new().with_row_count(Some(table.num_rows())),
    )?;
    match meta["kind"].as_str() {
        Some("points") => {
            let mut points = PointSet::from_table(&table, "x", "y", Some("z"))?;
            points.crs = crs;
            Ok(Stored::Points(points))
        }
        Some("block_model") => {
            let geometry = Geometry {
                origin: triple(&meta, "origin")?,
                size: triple(&meta, "size")?,
                count: triple(&meta, "count")?,
                rotation: triple(&meta, "rotation")?,
            };
            let column = |name: &str| {
                table
                    .schema()
                    .index_of(name)
                    .map_err(|_| bad(format!("missing column {name}")))
            };
            let layout = meta["layout"].as_str();
            let hidden: Vec<usize> = match layout {
                Some("masked") => vec![column(INDEX)?],
                Some("subblocked") => std::iter::once(INDEX)
                    .chain(EXTENT)
                    .map(column)
                    .collect::<Result<_>>()?,
                _ => vec![],
            };
            let keep: Vec<usize> = (0..table.num_columns())
                .filter(|i| !hidden.contains(i))
                .collect();
            let attributes = table.project(&keep)?;
            let index = |t: &RecordBatch| -> Result<Vec<u64>> {
                Ok(t.column(column(INDEX)?)
                    .as_primitive::<UInt64Type>()
                    .values()
                    .to_vec())
            };
            let mut model = match layout {
                Some("masked") => BlockModel::masked(geometry, index(&table)?, attributes)?,
                Some("subblocked") => {
                    let parts = EXTENT
                        .iter()
                        .map(|n| {
                            Ok(table
                                .column(column(n)?)
                                .as_primitive::<Float64Type>()
                                .values()
                                .to_vec())
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let extent = (0..table.num_rows())
                        .map(|r| std::array::from_fn(|i| parts[i][r]))
                        .collect();
                    let grid: Option<[u32; 3]> = serde_json::from_value(meta["grid"].clone())
                        .map_err(|_| bad("sub-grid must hold 3 counts"))?;
                    BlockModel::subblocked(geometry, index(&table)?, extent, grid, attributes)?
                }
                _ => BlockModel::regular(geometry, table)?,
            };
            model.crs = crs;
            Ok(Stored::Blocks(model))
        }
        other => Err(bad(format!("unknown kind {other:?}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StringArray;

    fn temp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("ceres-pq-{}-{name}", std::process::id()))
    }

    fn attributes(n: usize) -> RecordBatch {
        RecordBatch::try_from_iter([
            (
                "au",
                Arc::new(Float64Array::from_iter(
                    (0..n).map(|i| (i % 3 != 0).then_some(i as f64)),
                )) as ArrayRef,
            ),
            (
                "rock",
                Arc::new(StringArray::from_iter_values(
                    (0..n).map(|i| format!("r{}", i % 2)),
                )),
            ),
        ])
        .unwrap()
    }

    #[test]
    fn points_round_trip() {
        let mut points = PointSet::new(
            vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            attributes(3),
        )
        .unwrap();
        points.crs = Some("EPSG:32611".into());
        write_points(temp("p.parquet"), &points).unwrap();
        let Stored::Points(back) = read_parquet(temp("p.parquet")).unwrap() else {
            panic!("expected points")
        };
        assert_eq!(back.coords(), points.coords());
        assert_eq!(back.attributes(), points.attributes());
        assert_eq!(back.crs, points.crs);
    }

    #[test]
    fn masked_block_model_round_trips() {
        let geometry = Geometry {
            origin: [10.0, 20.0, 0.0],
            size: [5.0, 5.0, 2.0],
            count: [4, 3, 2],
            rotation: [30.0, 0.0, 0.0],
        };
        let model = BlockModel::masked(geometry, vec![1, 5, 22], attributes(3)).unwrap();
        write_block_model(temp("b.parquet"), &model).unwrap();
        let Stored::Blocks(back) = read_parquet(temp("b.parquet")).unwrap() else {
            panic!("expected a block model")
        };
        assert_eq!(back.geometry(), model.geometry());
        assert_eq!(back.layout(), model.layout());
        assert_eq!(back.attributes(), model.attributes());
    }

    #[test]
    fn subblocked_model_round_trips() {
        let geometry = Geometry {
            origin: [0.0; 3],
            size: [10.0, 10.0, 5.0],
            count: [2, 2, 1],
            rotation: [0.0; 3],
        };
        let extent = vec![
            [0.0, 0.0, 0.0, 0.5, 1.0, 1.0],
            [0.5, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ];
        let model = BlockModel::subblocked(
            geometry,
            vec![0, 0, 3],
            extent,
            Some([2, 1, 1]),
            attributes(3),
        )
        .unwrap();
        write_block_model(temp("s.parquet"), &model).unwrap();
        let Stored::Blocks(back) = read_parquet(temp("s.parquet")).unwrap() else {
            panic!("expected a block model")
        };
        assert_eq!(back.layout(), model.layout());
        assert_eq!(back.attributes(), model.attributes());
    }

    #[test]
    fn plain_tables_stay_tables() {
        write_parquet(temp("t.parquet"), &attributes(5)).unwrap();
        assert!(
            matches!(read_parquet(temp("t.parquet")).unwrap(), Stored::Table(t) if t.num_rows() == 5)
        );
    }
}
