use std::path::PathBuf;

use ceres_io::{CsvOptions, Nodata, Shapes};
use pyo3::prelude::*;

use crate::blocks::Mesh;
use crate::containers::{PyBlockModel, PyPointSet, PyPolylines};
use crate::table::{Table, to_batch};
use crate::{error, invalid};

pub(crate) fn io_error(e: ceres_io::Error) -> PyErr {
    match e {
        ceres_io::Error::Io(e) => error("FileError", e),
        e => invalid(e),
    }
}

fn nodata(values: Option<Vec<Bound<PyAny>>>) -> PyResult<Vec<Nodata>> {
    let Some(values) = values else {
        return Ok(ceres_io::default_nodata());
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
#[pyfunction]
#[pyo3(signature = (path, *, nodata=None, delimiter=","))]
fn read_csv(path: PathBuf, nodata: Option<Vec<Bound<PyAny>>>, delimiter: &str) -> PyResult<Table> {
    let &[delimiter] = delimiter.as_bytes() else {
        return Err(invalid("delimiter must be a single byte"));
    };
    let options = CsvOptions {
        delimiter,
        nodata: self::nodata(nodata)?,
    };
    Ok(Table(ceres_io::read_csv(path, &options).map_err(io_error)?))
}

#[pyfunction]
fn write_csv(path: PathBuf, table: &Bound<PyAny>) -> PyResult<()> {
    ceres_io::write_csv(path, &to_batch(table)?).map_err(io_error)
}

/// Reads a GSLIB file; the title is kept in the schema metadata and
/// `nodata` values, as in `read_csv`, become null.
#[pyfunction]
#[pyo3(signature = (path, *, nodata=None))]
fn read_gslib(path: PathBuf, nodata: Option<Vec<Bound<PyAny>>>) -> PyResult<Table> {
    let batch = ceres_io::read_gslib(path, &self::nodata(nodata)?).map_err(io_error)?;
    Ok(Table(batch))
}

/// Writes numeric columns as GSLIB; nulls are written as `nodata`.
#[pyfunction]
#[pyo3(signature = (path, table, *, nodata=-999.0))]
fn write_gslib(path: PathBuf, table: &Bound<PyAny>, nodata: f64) -> PyResult<()> {
    ceres_io::write_gslib(path, &to_batch(table)?, nodata).map_err(io_error)
}

/// Writes a PointSet, a BlockModel, Polylines or any table to Parquet;
/// containers keep their geometry, layout and CRS in the file metadata.
/// Polylines are stored one row per feature, as ``Polylines.to_table``.
#[pyfunction]
fn write_parquet(path: PathBuf, data: &Bound<PyAny>) -> PyResult<()> {
    if let Ok(points) = data.cast::<PyPointSet>() {
        return ceres_io::write_points(path, &points.get().0).map_err(io_error);
    }
    if let Ok(lines) = data.cast::<PyPolylines>() {
        return ceres_io::write_polylines(path, &lines.get().0).map_err(io_error);
    }
    if let Ok(model) = data.cast::<PyBlockModel>() {
        return ceres_io::write_block_model(path, &model.get().0).map_err(io_error);
    }
    ceres_io::write_parquet(path, &to_batch(data)?).map_err(io_error)
}

/// Reads Parquet as the PointSet, BlockModel or Polylines it was written
/// from, or a Table.
#[pyfunction]
fn read_parquet(py: Python, path: PathBuf) -> PyResult<Py<PyAny>> {
    Ok(match ceres_io::read_parquet(path).map_err(io_error)? {
        ceres_io::Stored::Polylines(l) => Py::new(py, PyPolylines(l))?.into_any(),
        ceres_io::Stored::Points(p) => PyPointSet(p).into_pyobject(py)?.into_any().unbind(),
        ceres_io::Stored::Blocks(b) => PyBlockModel(b).into_pyobject(py)?.into_any().unbind(),
        ceres_io::Stored::Table(t) => Table(t).into_pyobject(py)?.into_any().unbind(),
    })
}

/// Reads a `.obj`, `.stl` or `.dxf` mesh; DXF faces carry a `layer` column.
#[pyfunction]
fn read_mesh(path: PathBuf) -> PyResult<Mesh> {
    Ok(Mesh::from_core(
        ceres_io::read_mesh(path).map_err(io_error)?,
    ))
}

/// Writes a `.obj`, `.stl` (binary unless `ascii`) or `.dxf` mesh.
#[pyfunction]
#[pyo3(signature = (path, mesh, *, ascii=false))]
fn write_mesh(path: PathBuf, mesh: PyRef<Mesh>, ascii: bool) -> PyResult<()> {
    ceres_io::write_mesh(path, &mesh.mesh, ascii).map_err(io_error)
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
        match ceres_io::read_shapefile(path, &self::nodata(nodata)?).map_err(io_error)? {
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
        return ceres_io::write_polylines_shapefile(path, &lines.get().0).map_err(io_error);
    }
    let points = data
        .cast::<PyPointSet>()
        .map_err(|_| invalid("data must be a PointSet or Polylines"))?;
    ceres_io::write_shapefile(path, &points.get().0).map_err(io_error)
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
        ceres_io::read_geotiff(path, nodata).map_err(io_error)?,
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
    ceres_io::write_geotiff(path, &model.0, nodata).map_err(io_error)
}

/// A block model file read in chunks, for models larger than memory.
#[pyclass(module = "ceres", name = "BlockModelFile", frozen)]
pub struct BlockModelFile(ceres_io::BlockModelReader);

#[pymethods]
impl BlockModelFile {
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        Ok(Self(
            ceres_io::BlockModelReader::open(path).map_err(io_error)?,
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

#[pyclass(module = "ceres", unsendable)]
pub struct BlockChunkIterator(ceres_io::BlockChunks);

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

impl From<ceres_io::Error> for StreamError {
    fn from(e: ceres_io::Error) -> Self {
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
    ceres_io::stream_map::<StreamError>(path, out, rows, keep, |chunk| {
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
    m.add_function(wrap_pyfunction!(read_geotiff, m)?)?;
    m.add_function(wrap_pyfunction!(write_geotiff, m)?)?;
    Ok(())
}
