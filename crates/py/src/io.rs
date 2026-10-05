use std::collections::HashMap;
use std::path::PathBuf;

use arrow_array::RecordBatch;

use boitata_io::{CsvOptions, Nodata, Shapes};
use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::blocks::Mesh;
use crate::containers::{PyBlockModel, PyPointSet, PyPolylines, coords_array};
use crate::progress::with_progress;
use crate::table::{Table, fill_units, to_batch, to_batches};
use crate::{error, invalid};

pub(crate) fn io_error(e: boitata_io::Error) -> PyErr {
    match e {
        boitata_io::Error::Io(e) => error("FileError", e),
        e => invalid(e),
    }
}

fn nodata(values: Option<Vec<Bound<PyAny>>>) -> PyResult<Vec<Nodata>> {
    let Some(values) = values else {
        return Ok(boitata_io::default_nodata());
    };
    values
        .iter()
        .map(|v| match v.extract::<String>() {
            Ok(s) => Ok(Nodata::Text(s)),
            Err(_) => v
                .extract::<f64>()
                .map(Nodata::Number)
                .map_err(|_| invalid("nodata must hold numbers or strings")),
        })
        .collect()
}

/// Reads a headed CSV; numbers become float64, `nodata` values become null.
///
/// Parameters
/// ----------
/// path : str or Path
///     The file.
/// nodata : sequence of float or str, optional
///     Values read as null. Numbers match numerically (-999 matches
///     ``-999.0``), strings match tokens case-insensitively. Default -99,
///     -999, 1e21 and text such as ``NA``, ``N/A`` or ``NULL``. Empty cells
///     are always null.
/// delimiter : str, default ","
///     Single-byte field separator.
/// units : dict of str to str, optional
///     Column name to unit for columns that have none; the defaults of
///     `set_units` fill the rest.
/// progress : bool, default True
///     Show a `tqdm` progress bar.
///
/// Notes
/// -----
/// A header such as ``au [g/t]`` or ``au (g/t)`` reads as column ``au`` with
/// unit ``g/t`` when the bracket holds a unit.
#[pyfunction]
#[pyo3(signature = (path, *, nodata=None, delimiter=",", units=None, progress=true))]
fn read_csv(
    py: Python,
    path: PathBuf,
    nodata: Option<Vec<Bound<PyAny>>>,
    delimiter: &str,
    units: Option<HashMap<String, String>>,
    progress: bool,
) -> PyResult<Table> {
    let &[delimiter] = delimiter.as_bytes() else {
        return Err(invalid("delimiter must be a single byte"));
    };
    let options = CsvOptions {
        delimiter,
        nodata: self::nodata(nodata)?,
    };
    let batch = with_progress(py, None, progress, |counter| {
        boitata_io::read_csv(path, &options, counter)
    })?
    .map_err(io_error)?;
    Ok(Table(fill_units(batch, units.as_ref())?))
}

/// Writes a headed CSV. `progress` shows a `tqdm` bar.
#[pyfunction]
#[pyo3(signature = (path, table, *, progress=true))]
fn write_csv(py: Python, path: PathBuf, table: &Bound<PyAny>, progress: bool) -> PyResult<()> {
    let batch = to_batch(table)?;
    let total = Some(batch.num_rows() as u64);
    with_progress(py, total, progress, |counter| {
        boitata_io::write_csv(path, &batch, counter)
    })?
    .map_err(io_error)
}

/// Reads a GSLIB file; the title is kept in the schema metadata and
/// `nodata` values, as in `read_csv`, become null. `units` gives columns their
/// units as in `read_csv`. `progress` shows a `tqdm` bar.
#[pyfunction]
#[pyo3(signature = (path, *, nodata=None, units=None, progress=true))]
fn read_gslib(
    py: Python,
    path: PathBuf,
    nodata: Option<Vec<Bound<PyAny>>>,
    units: Option<HashMap<String, String>>,
    progress: bool,
) -> PyResult<Table> {
    let nodata = self::nodata(nodata)?;
    let batch = with_progress(py, None, progress, |counter| {
        boitata_io::read_gslib(path, &nodata, counter)
    })?
    .map_err(io_error)?;
    Ok(Table(fill_units(batch, units.as_ref())?))
}

/// Writes numeric columns as GSLIB; nulls are written as `nodata`. `progress`
/// shows a `tqdm` bar.
#[pyfunction]
#[pyo3(signature = (path, table, *, nodata=-999.0, progress=true))]
fn write_gslib(
    py: Python,
    path: PathBuf,
    table: &Bound<PyAny>,
    nodata: f64,
    progress: bool,
) -> PyResult<()> {
    let batch = to_batch(table)?;
    let total = Some(batch.num_rows() as u64);
    with_progress(py, total, progress, |counter| {
        boitata_io::write_gslib(path, &batch, nodata, counter)
    })?
    .map_err(io_error)
}

/// Writes a PointSet, a BlockModel, Polylines or any table to Parquet;
/// containers keep their geometry, layout and CRS in the file metadata.
/// Polylines are stored one row per feature, as ``Polylines.to_table``.
/// `progress` shows a `tqdm` bar.
#[pyfunction]
#[pyo3(signature = (path, data, *, progress=true))]
fn write_parquet(py: Python, path: PathBuf, data: &Bound<PyAny>, progress: bool) -> PyResult<()> {
    if let Ok(points) = data.cast::<PyPointSet>() {
        let points = &points.get().0;
        let total = Some(points.len() as u64);
        return with_progress(py, total, progress, |counter| {
            boitata_io::write_points(path, points, counter)
        })?
        .map_err(io_error);
    }
    if let Ok(lines) = data.cast::<PyPolylines>() {
        let lines = &lines.get().0;
        let total = Some(lines.len() as u64);
        return with_progress(py, total, progress, |counter| {
            boitata_io::write_polylines(path, lines, counter)
        })?
        .map_err(io_error);
    }
    if let Ok(model) = data.cast::<PyBlockModel>() {
        let model = &model.get().0;
        let total = Some(model.len() as u64);
        return with_progress(py, total, progress, |counter| {
            boitata_io::write_block_model(path, model, counter)
        })?
        .map_err(io_error);
    }
    let (schema, batches) = to_batches(data)?;
    let total = Some(batches.iter().map(|b| b.num_rows() as u64).sum());
    with_progress(py, total, progress, |counter| {
        boitata_io::write_parquet_batches(path, schema, &batches, counter)
    })?
    .map_err(io_error)
}

/// Reads Parquet as the PointSet, BlockModel or Polylines it was written
/// from, or a Table. `units` gives columns their units as in `read_csv`;
/// units stored in the file are kept as written, even when not understood.
/// `progress` shows a `tqdm` bar.
#[pyfunction]
#[pyo3(signature = (path, *, units=None, progress=true))]
fn read_parquet(
    py: Python,
    path: PathBuf,
    units: Option<HashMap<String, String>>,
    progress: bool,
) -> PyResult<Py<PyAny>> {
    let fill = |batch: &RecordBatch| fill_units(batch.clone(), units.as_ref());
    let stored = with_progress(py, None, progress, |counter| {
        boitata_io::read_parquet(path, counter)
    })?
    .map_err(io_error)?;
    Ok(match stored {
        boitata_io::Stored::Polylines(l) => {
            let l = l.with_attributes(fill(l.attributes())?).map_err(invalid)?;
            Py::new(py, PyPolylines(l))?.into_any()
        }
        boitata_io::Stored::Points(p) => {
            let p = p.with_attributes(fill(p.attributes())?).map_err(invalid)?;
            PyPointSet(p).into_pyobject(py)?.into_any().unbind()
        }
        boitata_io::Stored::Blocks(b) => {
            let b = b.with_attributes(fill(b.attributes())?).map_err(invalid)?;
            PyBlockModel(b).into_pyobject(py)?.into_any().unbind()
        }
        boitata_io::Stored::Table(t) => Table(fill(&t)?).into_pyobject(py)?.into_any().unbind(),
    })
}

/// Reads a `.obj`, `.stl` or `.dxf` mesh.
///
/// Parameters
/// ----------
/// path : str or Path
///     The file. OBJ `g` groups become a face column `group` when the file
///     has any. DXF reads `3DFACE`, polyface `POLYLINE` and `MESH` (level-0
///     cage, ASCII files only) entities as triangles with a face column
///     `layer`; `MESH` faces come last.
/// progress : bool, default True
///     Show a `tqdm` bar.
///
/// Returns
/// -------
/// Mesh
#[pyfunction]
#[pyo3(signature = (path, *, progress=true))]
fn read_mesh(py: Python, path: PathBuf, progress: bool) -> PyResult<Mesh> {
    let mesh = with_progress(py, None, progress, |counter| {
        boitata_io::read_mesh(path, counter)
    })?
    .map_err(io_error)?;
    Ok(Mesh::from_core(mesh))
}

/// Writes a `.obj`, `.stl` or ASCII R2000 `.dxf` mesh.
///
/// Parameters
/// ----------
/// path : str or Path
///     The file; its extension picks the format.
/// mesh : Mesh
///     The mesh. OBJ writes the face column `group` as `g` lines; DXF puts
///     faces on the layer named by the face column `layer`, else `0`.
/// ascii : bool, default False
///     Write ASCII instead of binary STL.
/// dxf_entity : {"3dface", "polyface"}, default "3dface"
///     One `3DFACE` per triangle, or one polyface `POLYLINE` with shared
///     vertices per layer, split past 32767 vertices or faces.
/// progress : bool, default True
///     Show a `tqdm` bar.
#[pyfunction]
#[pyo3(signature = (path, mesh, *, ascii=false, dxf_entity="3dface", progress=true))]
fn write_mesh(
    py: Python,
    path: PathBuf,
    mesh: PyRef<Mesh>,
    ascii: bool,
    dxf_entity: &str,
    progress: bool,
) -> PyResult<()> {
    let dxf_entity = match dxf_entity {
        "3dface" => boitata_io::DxfEntity::Face3D,
        "polyface" => boitata_io::DxfEntity::Polyface,
        other => {
            return Err(invalid(format!(
                "dxf_entity must be '3dface' or 'polyface', got '{other}'"
            )));
        }
    };
    let mesh = &mesh.mesh;
    let total = Some(mesh.triangles().len() as u64);
    with_progress(py, total, progress, |counter| {
        boitata_io::write_mesh(path, mesh, ascii, dxf_entity, counter)
    })?
    .map_err(io_error)
}

/// Reads a shapefile as a PointSet or Polylines.
///
/// Parameters
/// ----------
/// path : str or Path
///     The `.shp` file; the `.dbf` beside it holds the attributes and an
///     optional `.prj` the CRS.
/// nodata : sequence of float or str, optional
///     Values read as null, as in `read_csv`; blank fields are always null.
///
/// Returns
/// -------
/// PointSet or Polylines
///     Point files give one point per shape, and one per member of a
///     multipoint, which repeats its row. Line files give open parts, polygon
///     files closed parts without the repeated closing vertex; a null shape is
///     a feature without parts. 2D shapes get z = 0, M values are dropped.
///     Numeric fields are float64, logical fields bool, the rest text. The
///     `.prj` text is the CRS.
#[pyfunction]
#[pyo3(signature = (path, *, nodata=None))]
fn read_shapefile(
    py: Python,
    path: PathBuf,
    nodata: Option<Vec<Bound<PyAny>>>,
) -> PyResult<Py<PyAny>> {
    Ok(
        match boitata_io::read_shapefile(path, &self::nodata(nodata)?).map_err(io_error)? {
            Shapes::Points(p) => Py::new(py, PyPointSet(p))?.into_any(),
            Shapes::Polylines(l) => Py::new(py, PyPolylines(l))?.into_any(),
        },
    )
}

/// Writes a PointSet or Polylines as a 3D shapefile.
///
/// Parameters
/// ----------
/// path : str or Path
///     The `.shp` file; `.shx`, `.dbf`, `.cpg` and, with a CRS, `.prj` are
///     written beside it.
/// data : PointSet or Polylines
///     Points are written as PointZ. Polylines with all parts open are written
///     as PolyLineZ, with all parts closed as PolygonZ, rings wound outer
///     clockwise and holes counter-clockwise; split mixed ones by ``closed``.
///     Attribute names must be ASCII of at most 10 characters; numeric, bool
///     and text columns are written, nulls as blanks. The CRS is written
///     verbatim to the `.prj`, which GIS software expects as WKT.
#[pyfunction]
fn write_shapefile(path: PathBuf, data: &Bound<PyAny>) -> PyResult<()> {
    if let Ok(lines) = data.cast::<PyPolylines>() {
        return boitata_io::write_polylines_shapefile(path, &lines.get().0).map_err(io_error);
    }
    let points = data
        .cast::<PyPointSet>()
        .map_err(|_| invalid("data must be a PointSet or Polylines"))?;
    boitata_io::write_shapefile(path, &points.get().0).map_err(io_error)
}

/// Decodes GeoPackage geometry blobs (None for null) into the WKB kind of each
/// row (0 for null), the vertices, part offsets, and the row and closed flag
/// of each part.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn _decode_geometries<'py>(
    py: Python<'py>,
    blobs: Vec<Option<Bound<'py, PyBytes>>>,
) -> PyResult<(
    Vec<u32>,
    Bound<'py, PyArray2<f64>>,
    Vec<usize>,
    Vec<usize>,
    Vec<bool>,
)> {
    let (mut kinds, mut vertices, mut offsets, mut rows, mut closed) =
        (vec![], vec![], vec![0], vec![], vec![]);
    for (row, blob) in blobs.iter().enumerate() {
        let Some(blob) = blob else {
            kinds.push(0);
            continue;
        };
        let g = boitata_io::decode_geometry(blob.as_bytes()).map_err(io_error)?;
        kinds.push(g.kind);
        for (part, c) in g.parts {
            vertices.extend(part);
            offsets.push(vertices.len());
            rows.push(row);
            closed.push(c);
        }
    }
    Ok((kinds, coords_array(py, &vertices), offsets, rows, closed))
}

/// GeoPackage geometry type name and one blob per point or feature (None for
/// a feature without parts) of a PointSet or Polylines.
#[pyfunction]
fn _encode_geometries<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    srs_id: i32,
) -> PyResult<(&'static str, Vec<Option<Bound<'py, PyBytes>>>)> {
    if let Ok(points) = data.cast::<PyPointSet>() {
        let blobs = points.get().0.coords().iter();
        let blobs = blobs.map(|&p| Some(PyBytes::new(py, &boitata_io::encode_point(p, srs_id))));
        return Ok(("POINT", blobs.collect()));
    }
    let lines = &data
        .cast::<PyPolylines>()
        .map_err(|_| invalid("data must be a PointSet or Polylines"))?
        .get()
        .0;
    let polygon = lines.closed().iter().any(|&c| c);
    if polygon && !lines.closed().iter().all(|&c| c) {
        return Err(invalid("parts mix open and closed; split by `closed`"));
    }
    let blobs = (0..lines.len()).map(|f| {
        let parts: Vec<&[[f64; 3]]> = lines.feature_parts(f).map(|p| lines.part(p)).collect();
        (!parts.is_empty())
            .then(|| PyBytes::new(py, &boitata_io::encode_parts(&parts, polygon, srs_id)))
    });
    let name = if polygon {
        "MULTIPOLYGON"
    } else {
        "MULTILINESTRING"
    };
    Ok((name, blobs.collect()))
}

/// Reads a GeoTIFF raster as a 2D BlockModel.
///
/// Parameters
/// ----------
/// path : str or Path
///     A stripped or tiled GeoTIFF, uncompressed or compressed with deflate,
///     LZW or PackBits, with integer or float samples.
/// nodata : float, optional
///     Pixel value read as null, replacing the file's `GDAL_NODATA`. NaN
///     pixels are always null.
///
/// Returns
/// -------
/// BlockModel
///     A regular grid with nz = 1 and one column per band, named from the band
///     descriptions or `band_1`, `band_2`, ... float32 bands stay float32, the
///     rest become float64. The geometry comes from the pixel scale and tie
///     point or from the model transformation, rotation included; cell
///     centers fall on the tie points of pixel-is-point rasters. An EPSG code
///     in the GeoKeys becomes the CRS `"EPSG:<code>"`, otherwise the citation.
#[pyfunction]
#[pyo3(signature = (path, *, nodata=None))]
fn read_geotiff(path: PathBuf, nodata: Option<f64>) -> PyResult<PyBlockModel> {
    Ok(PyBlockModel(
        boitata_io::read_geotiff(path, nodata).map_err(io_error)?,
    ))
}

/// Writes a 2D BlockModel as a deflate-compressed GeoTIFF.
///
/// Parameters
/// ----------
/// path : str or Path
///     The `.tif` file.
/// model : BlockModel
///     A regular or masked grid with nz = 1, rotated by azimuth only. Each
///     numeric column becomes a band named after it: float32 when every column
///     is float32, float64 otherwise. Absent cells of a masked model are
///     nodata. An `"EPSG:<code>"` CRS is written as GeoKeys, any other as the
///     citation.
/// nodata : float, default -9999.0
///     Value written for nulls and stored in `GDAL_NODATA`; a column holding
///     it raises `InvalidInput`.
///
/// Raises
/// ------
/// InvalidInput
///     For sub-blocked or 3D (nz > 1) models, dip or rake rotations and text
///     columns.
#[pyfunction]
#[pyo3(signature = (path, model, *, nodata=-9999.0))]
fn write_geotiff(path: PathBuf, model: PyRef<PyBlockModel>, nodata: f64) -> PyResult<()> {
    boitata_io::write_geotiff(path, &model.0, nodata).map_err(io_error)
}

/// Reads a post-stack SEG-Y cube as a 3D BlockModel.
///
/// Parameters
/// ----------
/// path : str or Path
///     A SEG-Y file with one trace per inline and crossline, in sample format
///     1 (IBM float), 2, 3, 5, 6, 8, 10, 11 or 16 and either byte order.
/// column : str, default "amplitude"
///     Name of the float32 column holding the samples.
/// inline_byte, crossline_byte : int, default 189 and 193
///     1-based trace-header bytes of the 4-byte inline and crossline numbers.
/// x_byte, y_byte : int, default 181 and 185
///     1-based trace-header bytes of the 4-byte CDP x and y, scaled by the
///     coordinate scalar at byte 71 (negative divides).
/// nodata : float, optional
///     Sample value read as null. NaN samples are always null.
///
/// Returns
/// -------
/// BlockModel
///     A regular grid with x along the inlines and y along the crosslines,
///     each from the smallest to the largest line number in steps of the gcd
///     of their gaps; positions without a trace are null. Cell sizes, origin
///     and azimuth (that of the crossline axis) are a least-squares fit of the
///     CDP coordinates; a left-handed survey has its inlines reversed. z is
///     minus the sample time (ms) or depth, the sample interval divided by
///     1000, so the first sample is the top cell.
///
/// Raises
/// ------
/// InvalidInput
///     For truncated or non-SEG-Y files, unsupported sample formats, two
///     traces at one position (prestack gathers) and skewed surveys.
#[pyfunction]
#[pyo3(signature = (
    path, *, column="amplitude", inline_byte=189, crossline_byte=193, x_byte=181,
    y_byte=185, nodata=None
))]
fn read_segy(
    path: PathBuf,
    column: &str,
    inline_byte: usize,
    crossline_byte: usize,
    x_byte: usize,
    y_byte: usize,
    nodata: Option<f64>,
) -> PyResult<PyBlockModel> {
    let options = boitata_io::SegyOptions {
        column: column.into(),
        inline_byte,
        crossline_byte,
        x_byte,
        y_byte,
        nodata,
    };
    Ok(PyBlockModel(
        boitata_io::read_segy(path, &options).map_err(io_error)?,
    ))
}

/// Writes one column of a BlockModel as SEG-Y revision 1.
///
/// Parameters
/// ----------
/// path : str or Path
///     The `.sgy` file: big-endian IEEE floats (format 5), an EBCDIC textual
///     header and one trace per (x, y) column.
/// model : BlockModel
///     A regular or masked grid rotated by azimuth only. Inline `i + 1` and
///     crossline `j + 1` go to trace bytes 189 and 193 and the cell-center
///     CDP x and y to bytes 181 and 185. The cell height times 1000 is the
///     sample interval and minus the top cell-center z the delay; both must
///     be stored exactly.
/// column : str
///     Numeric column to write, as float32, top cell first.
/// nodata : float, default 0.0
///     Value written for nulls and absent cells of a masked model.
///
/// Raises
/// ------
/// InvalidInput
///     For sub-blocked models, dip or rake rotations, missing or text
///     columns, and cell heights or tops that SEG-Y cannot hold exactly.
#[pyfunction]
#[pyo3(signature = (path, model, column, *, nodata=0.0))]
fn write_segy(
    path: PathBuf,
    model: PyRef<PyBlockModel>,
    column: &str,
    nodata: f64,
) -> PyResult<()> {
    boitata_io::write_segy(path, &model.0, column, nodata).map_err(io_error)
}

/// A block model file read in chunks, for models larger than memory.
#[pyclass(module = "boitata", name = "BlockModelFile", frozen)]
pub struct BlockModelFile(boitata_io::BlockModelReader);

#[pymethods]
impl BlockModelFile {
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        Ok(Self(
            boitata_io::BlockModelReader::open(path).map_err(io_error)?,
        ))
    }

    #[getter]
    fn origin(&self) -> [f64; 3] {
        self.0.geometry().origin
    }

    #[getter]
    fn size(&self) -> [f64; 3] {
        self.0.geometry().size
    }

    #[getter]
    fn count(&self) -> [usize; 3] {
        self.0.geometry().count
    }

    #[getter]
    fn rotation(&self) -> [f64; 3] {
        self.0.geometry().rotation
    }

    #[getter]
    fn crs(&self) -> Option<String> {
        self.0.crs().map(str::to_string)
    }

    #[getter]
    fn column_names(&self) -> Vec<String> {
        self.0.column_names().to_vec()
    }

    /// BlockModel pieces of at most `rows` blocks with the chosen `columns`
    /// (all by default); pieces of a regular model are masked to their cells.
    #[pyo3(signature = (*, rows=1_000_000, columns=None))]
    fn chunks(&self, rows: usize, columns: Option<Vec<String>>) -> PyResult<BlockChunkIterator> {
        let names: Option<Vec<&str>> = columns
            .as_ref()
            .map(|c| c.iter().map(String::as_str).collect());
        Ok(BlockChunkIterator(
            self.0.chunks(rows, names.as_deref()).map_err(io_error)?,
        ))
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "BlockModelFile({} blocks, count {:?}, columns {:?})",
            self.0.len(),
            self.0.geometry().count,
            self.0.column_names()
        )
    }
}

#[pyclass(module = "boitata", unsendable)]
pub struct BlockChunkIterator(boitata_io::BlockChunks);

#[pymethods]
impl BlockChunkIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<PyBlockModel>> {
        self.0
            .next()
            .transpose()
            .map(|chunk| chunk.map(PyBlockModel))
            .map_err(io_error)
    }
}

struct StreamError(PyErr);

impl From<boitata_io::Error> for StreamError {
    fn from(e: boitata_io::Error) -> Self {
        Self(io_error(e))
    }
}

/// Streams the block model file `path` to `out` in chunks of `rows` blocks:
/// `func(chunk)` returns a dict of new columns, written with the chunk's
/// layout and, when `keep`, its columns. Memory stays bounded by `rows`.
#[pyfunction]
#[pyo3(signature = (path, out, func, *, rows=1_000_000, keep=true))]
fn map_blocks(
    path: PathBuf,
    out: PathBuf,
    func: &Bound<PyAny>,
    rows: usize,
    keep: bool,
) -> PyResult<()> {
    boitata_io::stream_map::<StreamError>(path, out, rows, keep, |chunk| {
        let columns = func
            .call1((PyBlockModel(chunk.clone()),))
            .and_then(|result| to_batch(&result))
            .map_err(StreamError)?;
        if columns.num_rows() != chunk.len() {
            return Err(StreamError(invalid(format!(
                "func returned {} rows for a chunk of {}",
                columns.num_rows(),
                chunk.len()
            ))));
        }
        let schema = columns.schema();
        Ok(schema
            .fields()
            .iter()
            .zip(columns.columns())
            .map(|(f, c)| (f.name().clone(), c.clone()))
            .collect())
    })
    .map_err(|e| e.0)
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<BlockModelFile>()?;
    m.add_function(wrap_pyfunction!(map_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(write_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(read_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(read_csv, m)?)?;
    m.add_function(wrap_pyfunction!(write_csv, m)?)?;
    m.add_function(wrap_pyfunction!(read_gslib, m)?)?;
    m.add_function(wrap_pyfunction!(write_gslib, m)?)?;
    m.add_function(wrap_pyfunction!(read_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(write_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(read_shapefile, m)?)?;
    m.add_function(wrap_pyfunction!(write_shapefile, m)?)?;
    m.add_function(wrap_pyfunction!(_decode_geometries, m)?)?;
    m.add_function(wrap_pyfunction!(_encode_geometries, m)?)?;
    m.add_function(wrap_pyfunction!(read_geotiff, m)?)?;
    m.add_function(wrap_pyfunction!(write_geotiff, m)?)?;
    m.add_function(wrap_pyfunction!(read_segy, m)?)?;
    m.add_function(wrap_pyfunction!(write_segy, m)?)?;
    Ok(())
}
