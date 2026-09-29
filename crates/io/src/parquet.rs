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
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use arrow_select::concat::concat_batches;
use ceres_core::{BlockModel, Geometry, Layout, PointSet, Polylines, Progress};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use parquet::arrow::arrow_writer::{
    ArrowColumnChunk, ArrowColumnWriter, ArrowLeafColumn, ArrowRowGroupWriterFactory,
    compute_leaves,
};
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::basic::{Compression, Encoding, ZstdLevel};
use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::ColumnPath;
use rayon::prelude::*;
use serde_json::{Value, json};

use crate::{Error, Result};

const KEY: &str = "ceres";
const HIDDEN: &str = "__ceres_";
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
    Polylines(Polylines),
}

fn properties(schema: &Schema) -> WriterProperties {
    let mut builder = WriterProperties::builder()
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .set_dictionary_enabled(false)
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_max_row_group_row_count(Some(ROW_GROUP));
    for field in schema.fields() {
        let path = ColumnPath::from(field.name().as_str());
        builder = match field.data_type() {
            DataType::Float32 | DataType::Float64 => {
                builder.set_column_encoding(path, Encoding::BYTE_STREAM_SPLIT)
            }
            _ if field.name() == INDEX => {
                builder.set_column_encoding(path, Encoding::DELTA_BINARY_PACKED)
            }
            _ => builder.set_column_dictionary_enabled(path, true),
        };
    }
    builder.build()
}

/// A Parquet file whose row groups are encoded column by column in
/// parallel and appended in order, so its bytes do not depend on the
/// number of threads.
struct Encoder {
    file: SerializedFileWriter<File>,
    factory: ArrowRowGroupWriterFactory,
    schema: SchemaRef,
    group: usize,
}

impl Encoder {
    /// A file of row groups of at most `group` rows.
    fn create(file: File, schema: SchemaRef, group: usize) -> Result<Self> {
        let props = properties(&schema);
        let (file, factory) =
            ArrowWriter::try_new(file, schema.clone(), Some(props))?.into_serialized_writer()?;
        Ok(Self {
            file,
            factory,
            schema,
            group,
        })
    }

    /// Appends `batch` as row groups, encoding up to one row group per
    /// thread at a time.
    fn write(&mut self, batch: &RecordBatch, progress: Option<&Progress>) -> Result<()> {
        let rows = batch.num_rows();
        let starts: Vec<usize> = (0..rows).step_by(self.group.max(1)).collect();
        for wave in starts.chunks(rayon::current_num_threads().max(1)) {
            let first = self.file.flushed_row_groups().len();
            let mut jobs: Vec<Vec<(ArrowColumnWriter, ArrowLeafColumn)>> = vec![];
            for (g, &start) in wave.iter().enumerate() {
                let slice = batch.slice(start, self.group.min(rows - start));
                let mut writers = self.factory.create_column_writers(first + g)?.into_iter();
                let mut job = vec![];
                for (field, column) in self.schema.fields().iter().zip(slice.columns()) {
                    for leaf in compute_leaves(field, column)? {
                        job.push((writers.next().expect("one writer per leaf"), leaf));
                    }
                }
                jobs.push(job);
            }
            let groups: Vec<Vec<ArrowColumnChunk>> = jobs
                .into_par_iter()
                .map(|job| {
                    job.into_par_iter()
                        .map(|(mut writer, leaf)| {
                            writer.write(&leaf)?;
                            writer.close()
                        })
                        .collect::<parquet::errors::Result<Vec<_>>>()
                })
                .collect::<parquet::errors::Result<_>>()?;
            for chunks in groups {
                let mut group = self.file.next_row_group()?;
                for chunk in chunks {
                    chunk.append_to_row_group(&mut group)?;
                }
                group.close()?;
            }
            if let Some(p) = progress {
                let done: usize = wave.iter().map(|&s| self.group.min(rows - s)).sum();
                p.inc_by(done as u64);
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<()> {
        self.file.close()?;
        Ok(())
    }
}

fn write(
    path: &Path,
    table: &RecordBatch,
    meta: Option<String>,
    progress: Option<&Progress>,
) -> Result<()> {
    let mut metadata = table.schema().metadata().clone();
    if let Some(meta) = meta {
        metadata.insert(KEY.into(), meta);
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
    let mut encoder = Encoder::create(File::create(path)?, schema, ROW_GROUP)?;
    encoder.write(&batch, progress)?;
    encoder.finish()
}

/// Writes a plain table.
pub fn write_parquet(
    path: impl AsRef<Path>,
    table: &RecordBatch,
    progress: Option<&Progress>,
) -> Result<()> {
    write(path.as_ref(), table, None, progress)
}

/// Writes a plain table given as batches of one schema.
pub fn write_parquet_batches(
    path: impl AsRef<Path>,
    schema: SchemaRef,
    batches: &[RecordBatch],
    progress: Option<&Progress>,
) -> Result<()> {
    let mut encoder = Encoder::create(File::create(path)?, schema, ROW_GROUP)?;
    for batch in batches {
        encoder.write(batch, progress)?;
    }
    encoder.finish()
}

/// Writes points as `x`, `y`, `z` and their attributes.
pub fn write_points(
    path: impl AsRef<Path>,
    points: &PointSet,
    progress: Option<&Progress>,
) -> Result<()> {
    let meta = json!({ "kind": "points", "crs": points.crs });
    write(
        path.as_ref(),
        &points.to_table()?,
        Some(meta.to_string()),
        progress,
    )
}

/// Writes polylines one row per feature, as [`Polylines::to_table`].
pub fn write_polylines(
    path: impl AsRef<Path>,
    lines: &Polylines,
    progress: Option<&Progress>,
) -> Result<()> {
    let meta = json!({ "kind": "polylines", "crs": lines.crs });
    write(
        path.as_ref(),
        &lines.to_table()?,
        Some(meta.to_string()),
        progress,
    )
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

/// Writes a block model file chunk by chunk; `write_block_model` is one chunk.
pub struct BlockModelWriter {
    file: Option<File>,
    writer: Option<Encoder>,
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
        self.write_ticking(chunk, None)
    }

    fn write_ticking(&mut self, chunk: &BlockModel, progress: Option<&Progress>) -> Result<()> {
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
                if chunk.attributes().num_columns() == 0 {
                    let index =
                        UInt64Array::from_iter_values((0..n).map(|r| chunk.parent_index(r)));
                    extra.push((INDEX.into(), Arc::new(index)));
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
            self.writer = Some(Encoder::create(file, schema, ROW_GROUP)?);
        }
        self.writer
            .as_mut()
            .expect("created above")
            .write(&batch, progress)?;
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
            Some(writer) => writer.finish(),
            None => Err(bad("no blocks were written")),
        }
    }
}

/// Writes a block model's attributes, with its cell index when masked and its
/// parent index and extents when sub-blocked. `progress` is ticked per row
/// group written.
pub fn write_block_model(
    path: impl AsRef<Path>,
    model: &BlockModel,
    progress: Option<&Progress>,
) -> Result<()> {
    let mut writer = BlockModelWriter::create(
        path,
        *model.geometry(),
        FileLayout::of(model),
        model.crs.as_deref(),
    )?;
    writer.write_ticking(model, progress)?;
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
    let hidden: Vec<usize> = (0..table.num_columns())
        .filter(|&i| table.schema().field(i).name().starts_with(HIDDEN))
        .collect();
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

fn read_table(path: &Path, progress: Option<&Progress>) -> Result<RecordBatch> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
    let schema = builder.schema().clone();
    let mut batches = Vec::new();
    for batch in builder.with_batch_size(ROW_GROUP).build()? {
        let batch = batch?;
        if let Some(p) = progress {
            p.inc_by(batch.num_rows() as u64);
        }
        batches.push(batch);
    }
    Ok(concat_batches(&schema, &batches)?)
}

/// Writes a fitted object: its arrays as `table`, its parameters as the JSON
/// `meta` under the `ceres` key.
pub fn write_model(
    path: impl AsRef<Path>,
    table: &RecordBatch,
    meta: String,
    progress: Option<&Progress>,
) -> Result<()> {
    write(path.as_ref(), table, Some(meta), progress)
}

/// Table and `ceres` metadata of a file written by [`write_model`].
pub fn read_model(
    path: impl AsRef<Path>,
    progress: Option<&Progress>,
) -> Result<(RecordBatch, String)> {
    let table = read_table(path.as_ref(), progress)?;
    let meta = table.schema().metadata().get(KEY).cloned();
    Ok((table, meta.ok_or_else(|| bad("no ceres metadata"))?))
}

/// Reads a file written by any Parquet tool; files written from a container
/// come back as that container, others (e.g. fitted models) as a table.
pub fn read_parquet(path: impl AsRef<Path>, progress: Option<&Progress>) -> Result<Stored> {
    let table = read_table(path.as_ref(), progress)?;
    let schema = table.schema();
    let Some(meta) = schema.metadata().get(KEY) else {
        return Ok(Stored::Table(table));
    };
    let meta: Value = serde_json::from_str(meta).map_err(|e| bad(e.to_string()))?;
    let crs = meta["crs"].as_str().map(str::to_string);
    let mut metadata: HashMap<String, String> = schema.metadata().clone();
    metadata.remove(KEY);
    let plain = || {
        RecordBatch::try_new_with_options(
            Arc::new(Schema::new_with_metadata(
                table.schema().fields().clone(),
                metadata.clone(),
            )),
            table.columns().to_vec(),
            &RecordBatchOptions::new().with_row_count(Some(table.num_rows())),
        )
    };
    match meta["kind"].as_str() {
        Some("points") => {
            let mut points = PointSet::from_table(&plain()?, "x", "y", Some("z"))?;
            points.crs = crs;
            Ok(Stored::Points(points))
        }
        Some("polylines") => {
            let mut lines = Polylines::from_nested(&plain()?)?;
            lines.crs = crs;
            Ok(Stored::Polylines(lines))
        }
        Some("block_model") => Ok(Stored::Blocks(decode(
            geometry(&meta)?,
            file_layout(&meta)?,
            crs,
            &table,
            0,
            true,
        )?)),
        None => Ok(Stored::Table(table)),
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
                .filter(|n| !n.starts_with(HIDDEN))
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
            .chain(
                schema
                    .fields()
                    .iter()
                    .map(|f| f.name().as_str())
                    .filter(|n| n.starts_with(HIDDEN)),
            )
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
    fn floats_are_split_by_byte_stream() {
        use parquet::basic::Encoding;
        use parquet::file::reader::{FileReader, SerializedFileReader};
        let path = temp("encodings.parquet");
        write_parquet(&path, &attributes(64), None).unwrap();
        let reader = SerializedFileReader::new(File::open(&path).unwrap()).unwrap();
        let group = reader.metadata().row_group(0);
        let encodings = |i: usize| group.column(i).encodings().collect::<Vec<Encoding>>();
        assert!(
            encodings(0).contains(&Encoding::BYTE_STREAM_SPLIT),
            "{:?}",
            encodings(0)
        );
        assert!(
            encodings(1).contains(&Encoding::RLE_DICTIONARY),
            "{:?}",
            encodings(1)
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn files_do_not_depend_on_thread_count() {
        let batch = attributes(4500);
        let bytes = |threads: usize| {
            let path = temp(&format!("threads-{threads}.parquet"));
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut encoder =
                    Encoder::create(File::create(&path).unwrap(), batch.schema(), 1000).unwrap();
                encoder.write(&batch.slice(0, 2500), None).unwrap();
                encoder.write(&batch.slice(2500, 2000), None).unwrap();
                encoder.finish().unwrap();
            });
            let bytes = std::fs::read(&path).unwrap();
            std::fs::remove_file(path).unwrap();
            bytes
        };
        let one = bytes(1);
        assert_eq!(one, bytes(8));
        let path = temp("threads-back.parquet");
        std::fs::write(&path, &one).unwrap();
        let Stored::Table(back) = read_parquet(&path, None).unwrap() else {
            panic!("expected a table")
        };
        assert_eq!(back.columns(), batch.columns());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn model_round_trip() {
        let path = temp("model.parquet");
        let meta = r#"{"type":"X","format":1}"#.to_string();
        write_model(&path, &attributes(4), meta.clone(), None).unwrap();
        let (table, back) = read_model(&path, None).unwrap();
        assert_eq!(back, meta);
        assert_eq!(table.columns(), attributes(4).columns());
        assert!(matches!(
            read_parquet(&path, None).unwrap(),
            Stored::Table(_)
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn points_round_trip() {
        let mut points = PointSet::new(
            vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            attributes(3),
        )
        .unwrap();
        points.crs = Some("EPSG:32611".into());
        write_points(temp("p.parquet"), &points, None).unwrap();
        let Stored::Points(back) = read_parquet(temp("p.parquet"), None).unwrap() else {
            panic!("expected points")
        };
        assert_eq!(back.coords(), points.coords());
        assert_eq!(back.attributes(), points.attributes());
        assert_eq!(back.crs, points.crs);
    }

    #[test]
    fn polylines_round_trip() {
        let square = |o: f64, s: f64| {
            [
                [o, o, 1.],
                [o + s, o, 1.],
                [o + s, o + s, 1.],
                [o, o + s, 1.],
            ]
        };
        let mut v = [square(0., 10.), square(4., 2.)].concat();
        v.extend([[20., 0., 5.], [30., 0., 5.], [40., 5., 5.], [50., 5., 5.]]);
        let mut lines = Polylines::new(
            v,
            vec![0, 4, 8, 10, 12],
            vec![0, 2, 2, 4],
            vec![true, true, false, false],
            attributes(3),
        )
        .unwrap();
        lines.crs = Some("EPSG:31982".into());
        write_polylines(temp("l.parquet"), &lines, None).unwrap();
        let Stored::Polylines(back) = read_parquet(temp("l.parquet"), None).unwrap() else {
            panic!("expected polylines")
        };
        assert_eq!(back.to_table().unwrap(), lines.to_table().unwrap());
        assert_eq!(back.crs, lines.crs);
        assert!(back.feature_parts(1).is_empty());
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
        write_block_model(temp("b.parquet"), &model, None).unwrap();
        let Stored::Blocks(back) = read_parquet(temp("b.parquet"), None).unwrap() else {
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
        write_block_model(temp("s.parquet"), &model, None).unwrap();
        let Stored::Blocks(back) = read_parquet(temp("s.parquet"), None).unwrap() else {
            panic!("expected a block model")
        };
        assert_eq!(back.layout(), model.layout());
        assert_eq!(back.attributes(), model.attributes());
    }

    #[test]
    fn regular_models_without_columns_keep_their_cells() {
        let geometry = Geometry {
            origin: [0.0; 3],
            size: [1.0; 3],
            count: [3, 2, 1],
            rotation: [0.0; 3],
        };
        let model = BlockModel::regular(geometry, attributes(6).project(&[]).unwrap()).unwrap();
        write_block_model(temp("e.parquet"), &model, None).unwrap();
        let Stored::Blocks(back) = read_parquet(temp("e.parquet"), None).unwrap() else {
            panic!("expected a block model")
        };
        assert_eq!((back.len(), back.attributes().num_columns()), (6, 0));
        assert_eq!(back.layout(), &Layout::Regular);
    }

    #[test]
    fn plain_tables_stay_tables() {
        write_parquet(temp("t.parquet"), &attributes(5), None).unwrap();
        assert!(
            matches!(read_parquet(temp("t.parquet"), None).unwrap(), Stored::Table(t) if t.num_rows() == 5)
        );
    }

    #[test]
    fn writes_and_reads_tick_every_row() {
        let write = Progress::new(Some(5));
        write_parquet(temp("g.parquet"), &attributes(5), Some(&write)).unwrap();
        assert_eq!(write.snapshot().0, 5);
        let read = Progress::new(None);
        read_parquet(temp("g.parquet"), Some(&read)).unwrap();
        assert_eq!(read.snapshot().0, 5);
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
            write_block_model(&path, &model, None).unwrap();
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
            write_block_model(&input, &model, None).unwrap();
            stream_map::<Error>(&input, &output, 3, false, |chunk| {
                let x = Float64Array::from_iter_values(chunk.centroids().iter().map(|c| c[0]));
                Ok(vec![("x".into(), Arc::new(x) as ArrayRef)])
            })
            .unwrap();
            let Stored::Blocks(back) = read_parquet(&output, None).unwrap() else {
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
