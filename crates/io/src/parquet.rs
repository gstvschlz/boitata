//! Parquet storage of tables and containers. Container geometry, layout and
//! CRS are stored as JSON under the `ceres` key of the file's metadata, so
//! any Parquet reader still sees a plain table.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, UInt64Type};
use arrow_array::{ArrayRef, Float64Array, RecordBatch, RecordBatchOptions, UInt64Array};
use arrow_schema::{Field, Schema};
use arrow_select::concat::concat_batches;
use ceres_core::{BlockModel, Geometry, Layout, PointSet};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use parquet::arrow::{ArrowWriter, ProjectionMask};
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

fn properties() -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .set_max_row_group_row_count(Some(ROW_GROUP))
        .build()
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
    let mut writer = ArrowWriter::try_new(File::create(path)?, schema, Some(properties()))?;
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

/// How a block model file stores its rows; the same for every chunk.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FileLayout {
    /// Every cell, in index order, without an index column.
    Regular,
    /// Listed cells with their index.
    Masked,
    /// Sub-blocks with their parent index and extent.
    SubBlocked { grid: Option<[u32; 3]> },
}

impl FileLayout {
    fn of(model: &BlockModel) -> Self {
        match model.layout() {
            Layout::Regular => FileLayout::Regular,
            Layout::Masked(_) => FileLayout::Masked,
            Layout::SubBlocked { grid, .. } => FileLayout::SubBlocked { grid: *grid },
        }
    }
}

fn hidden(layout: FileLayout) -> Vec<&'static str> {
    match layout {
        FileLayout::Regular => vec![],
        FileLayout::Masked => vec![INDEX],
        FileLayout::SubBlocked { .. } => std::iter::once(INDEX).chain(EXTENT).collect(),
    }
}

/// Writes a block model file chunk by chunk; `write_block_model` is one chunk.
pub struct BlockModelWriter {
    file: Option<File>,
    writer: Option<ArrowWriter<File>>,
    meta: Value,
    geometry: Geometry,
    layout: FileLayout,
    rows: u64,
}

impl BlockModelWriter {
    /// Starts a file for blocks of `geometry` stored as `layout`.
    pub fn create(
        path: impl AsRef<Path>,
        geometry: Geometry,
        layout: FileLayout,
        crs: Option<&str>,
    ) -> Result<Self> {
        let mut meta = json!({
            "kind": "block_model",
            "crs": crs,
            "origin": geometry.origin,
            "size": geometry.size,
            "count": geometry.count,
            "rotation": geometry.rotation,
            "layout": "regular",
        });
        match layout {
            FileLayout::Regular => {}
            FileLayout::Masked => meta["layout"] = json!("masked"),
            FileLayout::SubBlocked { grid } => {
                meta["layout"] = json!("subblocked");
                meta["grid"] = json!(grid);
            }
        }
        Ok(Self {
            file: Some(File::create(path)?),
            writer: None,
            meta,
            geometry,
            layout,
            rows: 0,
        })
    }

    /// Appends the rows of `chunk`, which must share the file's geometry; a
    /// regular file takes its cells in index order.
    pub fn write(&mut self, chunk: &BlockModel) -> Result<()> {
        if chunk.geometry() != &self.geometry {
            return Err(bad("chunk geometry differs from the file's"));
        }
        let n = chunk.len();
        let mut extra: Vec<(String, ArrayRef)> = vec![];
        match self.layout {
            FileLayout::Regular => {
                if (0..n).any(|r| chunk.parent_index(r) != self.rows + r as u64)
                    || matches!(chunk.layout(), Layout::SubBlocked { .. })
                {
                    return Err(bad("a regular file takes every cell once, in index order"));
                }
            }
            FileLayout::Masked => {
                if matches!(chunk.layout(), Layout::SubBlocked { .. }) {
                    return Err(bad("sub-blocks cannot go to a masked file"));
                }
                let index = UInt64Array::from_iter_values((0..n).map(|r| chunk.parent_index(r)));
                extra.push((INDEX.into(), Arc::new(index)));
            }
            FileLayout::SubBlocked { .. } => {
                let Layout::SubBlocked { parent, extent, .. } = chunk.layout() else {
                    return Err(bad("a sub-blocked file takes sub-blocked chunks"));
                };
                extra.push((INDEX.into(), Arc::new(UInt64Array::from(parent.clone()))));
                for (i, name) in EXTENT.iter().enumerate() {
                    let column = Float64Array::from_iter_values(extent.iter().map(|e| e[i]));
                    extra.push((name.to_string(), Arc::new(column)));
                }
            }
        }
        let table = chunk.attributes();
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
        let mut metadata = table.schema().metadata().clone();
        metadata.insert(KEY.into(), self.meta.to_string());
        let schema = Arc::new(Schema::new_with_metadata(fields, metadata));
        let batch = RecordBatch::try_new_with_options(
            schema.clone(),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(n)),
        )?;
        if self.writer.is_none() {
            let file = self.file.take().expect("a file until the first chunk");
            self.writer = Some(ArrowWriter::try_new(file, schema, Some(properties()))?);
        }
        self.writer.as_mut().expect("created above").write(&batch)?;
        self.rows += n as u64;
        Ok(())
    }

    /// Closes the file.
    pub fn finish(self) -> Result<()> {
        if self.layout == FileLayout::Regular && self.rows != self.geometry.cells() {
            return Err(bad(format!(
                "a regular file needs all {} cells, got {}",
                self.geometry.cells(),
                self.rows
            )));
        }
        match self.writer {
            Some(writer) => {
                writer.close()?;
                Ok(())
            }
            None => Err(bad("no blocks were written")),
        }
    }
}

/// Writes a block model's attributes, with its cell index when masked and its
/// parent index and extents when sub-blocked.
pub fn write_block_model(path: impl AsRef<Path>, model: &BlockModel) -> Result<()> {
    let mut writer = BlockModelWriter::create(
        path,
        *model.geometry(),
        FileLayout::of(model),
        model.crs.as_deref(),
    )?;
    writer.write(model)?;
    writer.finish()
}

fn bad(message: impl Into<String>) -> Error {
    Error::Metadata(message.into())
}

fn triple<T: serde::de::DeserializeOwned>(meta: &Value, key: &str) -> Result<[T; 3]> {
    serde_json::from_value(meta[key].clone())
        .map_err(|_| bad(format!("`{key}` must hold 3 values")))
}

fn geometry(meta: &Value) -> Result<Geometry> {
    Ok(Geometry {
        origin: triple(meta, "origin")?,
        size: triple(meta, "size")?,
        count: triple(meta, "count")?,
        rotation: triple(meta, "rotation")?,
    })
}

fn file_layout(meta: &Value) -> Result<FileLayout> {
    Ok(match meta["layout"].as_str() {
        Some("masked") => FileLayout::Masked,
        Some("subblocked") => FileLayout::SubBlocked {
            grid: serde_json::from_value(meta["grid"].clone())
                .map_err(|_| bad("sub-grid must hold 3 counts"))?,
        },
        _ => FileLayout::Regular,
    })
}

/// Rows `offset..` of a block model file as a model: regular when `whole`,
/// else the regular cells come back masked to the rows read.
fn decode(
    geometry: Geometry,
    layout: FileLayout,
    crs: Option<String>,
    table: &RecordBatch,
    offset: usize,
    whole: bool,
) -> Result<BlockModel> {
    let column = |name: &str| {
        table
            .schema()
            .index_of(name)
            .map_err(|_| bad(format!("missing column {name}")))
    };
    let hidden = hidden(layout)
        .into_iter()
        .map(column)
        .collect::<Result<Vec<_>>>()?;
    let keep: Vec<usize> = (0..table.num_columns())
        .filter(|i| !hidden.contains(i))
        .collect();
    let mut attributes = table.project(&keep)?;
    let mut metadata = attributes.schema().metadata().clone();
    metadata.remove(KEY);
    attributes = RecordBatch::try_new_with_options(
        Arc::new(Schema::new_with_metadata(
            attributes.schema().fields().clone(),
            metadata,
        )),
        attributes.columns().to_vec(),
        &RecordBatchOptions::new().with_row_count(Some(table.num_rows())),
    )?;
    let index = || -> Result<Vec<u64>> {
        Ok(table
            .column(column(INDEX)?)
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec())
    };
    let mut model = match layout {
        FileLayout::Regular if whole => BlockModel::regular(geometry, attributes)?,
        FileLayout::Regular => {
            let rows = offset as u64..(offset + table.num_rows()) as u64;
            BlockModel::masked(geometry, rows.collect(), attributes)?
        }
        FileLayout::Masked => BlockModel::masked(geometry, index()?, attributes)?,
        FileLayout::SubBlocked { grid } => {
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
            BlockModel::subblocked(geometry, index()?, extent, grid, attributes)?
        }
    };
    model.crs = crs;
    Ok(model)
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
    let table = concat_batches(&schema, &batches)?;
    let Some(meta) = schema.metadata().get(KEY) else {
        return Ok(Stored::Table(table));
    };
    let meta: Value = serde_json::from_str(meta).map_err(|e| bad(e.to_string()))?;
    let crs = meta["crs"].as_str().map(str::to_string);
    match meta["kind"].as_str() {
        Some("points") => {
            let mut metadata: HashMap<String, String> = schema.metadata().clone();
            metadata.remove(KEY);
            let table = RecordBatch::try_new_with_options(
                Arc::new(Schema::new_with_metadata(
                    table.schema().fields().clone(),
                    metadata,
                )),
                table.columns().to_vec(),
                &RecordBatchOptions::new().with_row_count(Some(table.num_rows())),
            )?;
            let mut points = PointSet::from_table(&table, "x", "y", Some("z"))?;
            points.crs = crs;
            Ok(Stored::Points(points))
        }
        Some("block_model") => Ok(Stored::Blocks(decode(
            geometry(&meta)?,
            file_layout(&meta)?,
            crs,
            &table,
            0,
            true,
        )?)),
        other => Err(bad(format!("unknown kind {other:?}"))),
    }
}

/// A block model file read chunk by chunk, so it never has to fit in memory.
#[derive(Debug, Clone)]
pub struct BlockModelReader {
    path: PathBuf,
    geometry: Geometry,
    layout: FileLayout,
    crs: Option<String>,
    rows: usize,
    columns: Vec<String>,
}

impl BlockModelReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(&path)?)?;
        let meta: Value = builder
            .schema()
            .metadata()
            .get(KEY)
            .map(|m| serde_json::from_str(m).map_err(|e| bad(e.to_string())))
            .transpose()?
            .filter(|m: &Value| m["kind"] == "block_model")
            .ok_or_else(|| bad("not a block model file"))?;
        let layout = file_layout(&meta)?;
        let hidden = hidden(layout);
        Ok(Self {
            geometry: geometry(&meta)?,
            layout,
            crs: meta["crs"].as_str().map(str::to_string),
            rows: builder.metadata().file_metadata().num_rows() as usize,
            columns: builder
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .filter(|n| !hidden.contains(&n.as_str()))
                .collect(),
            path,
        })
    }

    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    pub fn layout(&self) -> FileLayout {
        self.layout
    }

    pub fn crs(&self) -> Option<&str> {
        self.crs.as_deref()
    }

    pub fn len(&self) -> usize {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    pub fn column_names(&self) -> &[String] {
        &self.columns
    }

    /// Chunks of at most `rows` blocks with the chosen `columns` (all when
    /// `None`); regular files come back as masked chunks of consecutive cells.
    pub fn chunks(&self, rows: usize, columns: Option<&[&str]>) -> Result<BlockChunks> {
        if rows == 0 {
            return Err(bad("chunks need at least one row"));
        }
        let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(&self.path)?)?;
        let names: Vec<&str> = match columns {
            Some(c) => c.to_vec(),
            None => self.columns.iter().map(String::as_str).collect(),
        };
        let schema = builder.schema().clone();
        let roots = names
            .iter()
            .copied()
            .chain(hidden(self.layout))
            .map(|n| {
                schema
                    .index_of(n)
                    .map_err(|_| bad(format!("no column {n}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let mask = ProjectionMask::roots(builder.parquet_schema(), roots);
        Ok(BlockChunks {
            inner: builder
                .with_projection(mask)
                .with_batch_size(rows)
                .build()?,
            geometry: self.geometry,
            layout: self.layout,
            crs: self.crs.clone(),
            offset: 0,
        })
    }
}

/// Iterator over the chunks of a [`BlockModelReader`].
pub struct BlockChunks {
    inner: ParquetRecordBatchReader,
    geometry: Geometry,
    layout: FileLayout,
    crs: Option<String>,
    offset: usize,
}

impl Iterator for BlockChunks {
    type Item = Result<BlockModel>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = match self.inner.next()? {
            Ok(batch) => batch,
            Err(e) => return Some(Err(e.into())),
        };
        let offset = self.offset;
        self.offset += batch.num_rows();
        Some(decode(
            self.geometry,
            self.layout,
            self.crs.clone(),
            &batch,
            offset,
            false,
        ))
    }
}

/// Streams `input` to `output` chunk by chunk: `f` returns new columns for
/// each chunk, written with the chunk's layout and, when `keep`, its columns.
pub fn stream_map<E: From<Error>>(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    rows: usize,
    keep: bool,
    mut f: impl FnMut(&BlockModel) -> std::result::Result<Vec<(String, ArrayRef)>, E>,
) -> std::result::Result<(), E> {
    let reader = BlockModelReader::open(input)?;
    let mut writer =
        BlockModelWriter::create(output, *reader.geometry(), reader.layout(), reader.crs())?;
    for chunk in reader.chunks(rows, None)? {
        let chunk = chunk?;
        let mut out = if keep {
            chunk.clone()
        } else {
            let none = chunk.attributes().project(&[]).map_err(Error::from)?;
            chunk.with_attributes(none).map_err(Error::from)?
        };
        for (name, column) in f(&chunk)? {
            out = out.with_column(&name, column).map_err(Error::from)?;
        }
        writer.write(&out)?;
    }
    writer.finish()?;
    Ok(())
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

    fn models() -> Vec<BlockModel> {
        let geometry = Geometry {
            origin: [10.0, 20.0, 0.0],
            size: [5.0, 5.0, 2.0],
            count: [4, 3, 2],
            rotation: [30.0, 0.0, 0.0],
        };
        let extent = vec![[0.0, 0.0, 0.0, 0.5, 1.0, 1.0]; 5];
        vec![
            BlockModel::regular(geometry, attributes(24)).unwrap(),
            BlockModel::masked(geometry, vec![1, 5, 7, 8, 22], attributes(5)).unwrap(),
            BlockModel::subblocked(geometry, vec![0, 0, 3, 4, 9], extent, None, attributes(5))
                .unwrap(),
        ]
    }

    #[test]
    fn chunks_cover_the_file_in_order() {
        for (k, model) in models().into_iter().enumerate() {
            let path = temp(&format!("c{k}.parquet"));
            write_block_model(&path, &model).unwrap();
            let reader = BlockModelReader::open(&path).unwrap();
            assert_eq!(reader.len(), model.len());
            assert_eq!(reader.column_names(), ["au", "rock"]);
            let chunks: Vec<BlockModel> = reader
                .chunks(2, None)
                .unwrap()
                .map(Result::unwrap)
                .collect();
            assert!(chunks.iter().all(|c| c.len() <= 2));
            let rows: Vec<u64> = chunks
                .iter()
                .flat_map(|c| (0..c.len()).map(|r| c.parent_index(r)).collect::<Vec<_>>())
                .collect();
            assert_eq!(
                rows,
                (0..model.len())
                    .map(|r| model.parent_index(r))
                    .collect::<Vec<_>>()
            );
            let centroids: Vec<[f64; 3]> = chunks.iter().flat_map(BlockModel::centroids).collect();
            assert_eq!(centroids, model.centroids());
            let only_au = reader
                .chunks(10, Some(&["au"]))
                .unwrap()
                .next()
                .unwrap()
                .unwrap();
            assert_eq!(only_au.attributes().num_columns(), 1);
        }
    }

    #[test]
    fn stream_map_rewrites_every_chunk() {
        for (k, model) in models().into_iter().enumerate() {
            let (input, output) = (
                temp(&format!("m{k}.parquet")),
                temp(&format!("o{k}.parquet")),
            );
            write_block_model(&input, &model).unwrap();
            stream_map::<Error>(&input, &output, 3, false, |chunk| {
                let x = Float64Array::from_iter_values(chunk.centroids().iter().map(|c| c[0]));
                Ok(vec![("x".into(), Arc::new(x) as ArrayRef)])
            })
            .unwrap();
            let Stored::Blocks(back) = read_parquet(&output).unwrap() else {
                panic!("expected a block model")
            };
            assert_eq!(back.layout(), model.layout());
            let x = back
                .attributes()
                .column(0)
                .as_primitive::<Float64Type>()
                .values()
                .to_vec();
            assert_eq!(
                x,
                model.centroids().iter().map(|c| c[0]).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn regular_files_take_every_cell_in_order() {
        let model = &models()[0];
        let mut writer = BlockModelWriter::create(
            temp("r.parquet"),
            *model.geometry(),
            FileLayout::Regular,
            None,
        )
        .unwrap();
        let skipped = model
            .mask(&(0..24).map(|i| Some(i != 3)).collect())
            .unwrap();
        assert!(writer.write(&skipped).is_err());
        let first = model
            .mask(&(0..24).map(|i| Some(i < 12)).collect())
            .unwrap();
        writer.write(&first).unwrap();
        assert!(writer.finish().is_err());
    }
}
