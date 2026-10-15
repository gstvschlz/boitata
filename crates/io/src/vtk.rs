use std::borrow::Cow;
use std::fmt::Write as _;
use std::io::Read as _;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, RecordBatch, RecordBatchOptions,
    UInt32Array,
};
use arrow_schema::{DataType, Field, Schema};
use arrow_select::take::take_record_batch;
use boitata_core::{
    BlockModel, Geometry, Layout, Mesh, PointSet, Polylines, angles_from_axes, block_frame,
};
use regex::Regex;

use crate::{Error, Nodata, Result};

/// Contents of a VTK XML file.
#[derive(Debug, Clone)]
pub enum Vtk {
    Points(PointSet),
    Polylines(Polylines),
    Mesh(Mesh),
    Blocks(BlockModel),
}

fn bad(message: impl Into<String>) -> Error {
    Error::Vtk(message.into())
}

/// Reads a `.vtp`, `.vtu`, `.vti`, `.vtr` or `.vts` file in ASCII, base64
/// binary or appended (raw or base64) encoding, zlib-compressed or not, with
/// UInt32 or UInt64 block headers.
///
/// With `point_data`, any file gives a PointSet of its points (grid nodes)
/// with the point data. Otherwise, in `.vtp` and `.vtu` files, surface cells
/// (triangles, strips, quads, polygons) give a Mesh with point data as vertex
/// columns and cell data as face columns; else line cells give Polylines, one
/// part per cell, closed when it ends where it starts, cells sharing a
/// `feature` value forming one feature, with cell data as feature columns;
/// else cells that are all hexahedra or voxels give blocks; else a PointSet of
/// the points with the point data. Grid files and hexahedra give a BlockModel
/// of the cell data when the cells are equal boxes on a lattice (a masked one
/// when some are missing, or hidden by `vtkGhostType`), else a PointSet of the
/// cell centers. A grid file with point data and no cell data gives the
/// PointSet of its nodes: point values are never taken as cell values. A flat
/// axis (one node layer) makes cells one spacing thick (1 in `.vtr` and
/// `.vts`) centered on the nodes.
///
/// Arrays with several components become `name_0`, `name_1`, ...; Float32
/// arrays stay Float32, the rest become Float64. NaN and `nodata` numbers
/// read as null, and so does every value whose `<name>_valid` array is 0.
pub fn read_vtk(path: impl AsRef<Path>, point_data: bool, nodata: &[Nodata]) -> Result<Vtk> {
    let path = path.as_ref();
    let kind = extension(path)?;
    let bytes = std::fs::read(path)?;
    let mut file = File::parse(&bytes)?;
    file.nodata = nodata
        .iter()
        .filter_map(|n| match n {
            Nodata::Number(v) => Some(*v),
            Nodata::Text(_) => None,
        })
        .collect();
    let root = file.root()?;
    let tag = match kind {
        "vtp" => "PolyData",
        "vtu" => "UnstructuredGrid",
        "vti" => "ImageData",
        "vtr" => "RectilinearGrid",
        _ => "StructuredGrid",
    };
    if root.attr("type") != Some(tag) {
        return Err(bad(format!("{} is not a {tag} file", path.display())));
    }
    let dataset = root.first(tag)?.ok_or_else(|| bad(format!("no {tag}")))?;
    let pieces = dataset.all("Piece")?;
    let [piece] = &pieces[..] else {
        return Err(bad("files must hold exactly one Piece"));
    };
    if matches!(kind, "vti" | "vtr" | "vts") {
        return grid(&file, kind, &dataset, piece, point_data);
    }
    let n = piece.count("NumberOfPoints")?;
    let points = match piece.first("Points")? {
        Some(p) => file.coordinates(&p, n)?,
        None if n == 0 => Vec::new(),
        None => return Err(bad("no Points")),
    };
    let cells = if kind == "vtp" {
        let mut cells = Cells::default();
        for (tag, kind) in [("Verts", 2), ("Lines", 4), ("Strips", 6), ("Polys", 7)] {
            if let Some(e) = piece.first(tag)? {
                cells.extend(&file, &e, None, kind)?;
            }
        }
        cells
    } else {
        let mut cells = Cells::default();
        if let Some(e) = piece.first("Cells")? {
            let types = file.values(&named(&e, "types")?)?;
            cells.extend(&file, &e, Some(types), 0)?;
        }
        cells
    };
    if cells.ids.iter().any(|&i| i >= points.len()) {
        return Err(bad("cell point index outside the points"));
    }
    let values = file.columns(piece.first("PointData")?.as_ref(), points.len())?;
    if point_data {
        return Ok(Vtk::Points(PointSet::new(points, values)?));
    }
    let cell_data = file.columns(piece.first("CellData")?.as_ref(), cells.len())?;
    build(points, values, cells, cell_data)
}

/// Writes a PointSet, Polylines, Mesh or BlockModel by extension. `.vtp` and
/// `.vtu` take vertex cells with point data, one polyline cell per part with
/// the feature columns and a `feature` index as cell data (closed parts repeat
/// their first point), or triangles with vertex and face columns; `.vtu` also
/// takes any BlockModel as one hexahedron per row with the columns as cell
/// data. `.vti` takes a regular or masked BlockModel as image data, its
/// rotation as `Direction`, absent cells hidden by `vtkGhostType`. Arrays are
/// appended raw with UInt64 headers unless `ascii`. Numeric and boolean
/// columns are written, Float32 as Float32 and the rest as Float64; a column
/// with nulls writes them as NaN plus a UInt8 `<name>_valid` array.
pub fn write_vtk(path: impl AsRef<Path>, data: &Vtk, ascii: bool) -> Result<()> {
    let path = path.as_ref();
    let kind = extension(path)?;
    let mut w = Writer::new(ascii, 8);
    match (kind, data) {
        ("vti", Vtk::Blocks(model)) => return write_vti(path, model, w),
        ("vti", _) => return Err(bad(".vti files hold block models")),
        ("vtr" | "vts", _) => return Err(bad(".vtr and .vts files are read only")),
        ("vtp", Vtk::Blocks(_)) => return Err(bad("block models go to .vti or .vtu")),
        _ => {}
    }
    let (points, cells, point_data, cell_data) = cells_of(data)?;
    let tag = if kind == "vtp" {
        "PolyData"
    } else {
        "UnstructuredGrid"
    };
    w.open(tag, "");
    let count = cells.len();
    if kind == "vtp" {
        let section = |t| matches!(cells.types.first(), Some(&k) if k == t) as usize * count;
        writeln!(
            w.xml,
            r#"<Piece NumberOfPoints="{}" NumberOfVerts="{}" NumberOfLines="{}" NumberOfStrips="0" NumberOfPolys="{}">"#,
            points.len(),
            section(1),
            section(4),
            section(5)
        )
        .expect("string");
    } else {
        writeln!(
            w.xml,
            r#"<Piece NumberOfPoints="{}" NumberOfCells="{count}">"#,
            points.len()
        )
        .expect("string");
    }
    w.data("PointData", &point_data)?;
    w.data("CellData", &cell_data)?;
    w.points(&points);
    let section = match (kind, cells.types.first()) {
        ("vtu", _) => "Cells",
        (_, Some(1)) => "Verts",
        (_, Some(4)) => "Lines",
        _ => "Polys",
    };
    writeln!(w.xml, "<{section}>").expect("string");
    w.cells(&cells, kind == "vtu");
    writeln!(w.xml, "</{section}>\n</Piece>").expect("string");
    Ok(std::fs::write(path, w.finish(tag))?)
}

fn extension(path: &Path) -> Result<&'static str> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    ["vtp", "vtu", "vti", "vtr", "vts"]
        .into_iter()
        .find(|f| f.eq_ignore_ascii_case(ext))
        .ok_or_else(|| {
            bad(format!(
                "{}: files must be .vtp, .vtu, .vti, .vtr or .vts",
                path.display()
            ))
        })
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| a[i] - b[i])
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Grid of boxes with edges `v` (one per axis) whose corner `(0, 0, 0)` is
/// `origin`, if the edges are orthogonal, right-handed and non-zero.
fn frame(origin: [f64; 3], v: [[f64; 3]; 3], count: [usize; 3]) -> Option<Geometry> {
    let size = v.map(|e| dot(e, e).sqrt());
    let u: [[f64; 3]; 3] = [0, 1, 2].map(|a| v[a].map(|x| x / size[a]));
    let square = [(0, 1), (1, 2), (0, 2)]
        .iter()
        .all(|&(a, b)| dot(u[a], u[b]).abs() < 1e-9);
    let geometry = Geometry {
        origin,
        size,
        count,
        rotation: angles_from_axes(u[1], u[0].map(|x| -x)).map(|a| a + 0.0),
    };
    (square && dot(cross(u[0], u[1]), u[2]) > 0.0 && geometry.validate().is_ok())
        .then_some(geometry)
}

/// Whether `p` is within a tolerance of `q`, relative to the coordinates.
fn near(p: [f64; 3], q: [f64; 3], scale: f64) -> bool {
    sub(p, q).iter().all(|d| d.abs() <= 1e-7 * scale)
}

/// Nodes of a grid file, x fastest, with the node counts and the thickness of
/// cells along a flat axis.
fn grid(
    file: &File,
    kind: &str,
    dataset: &Element,
    piece: &Element,
    point_data: bool,
) -> Result<Vtk> {
    let numbers = |text: Option<&str>, n: usize, what: &str| -> Result<Vec<f64>> {
        let v: Vec<f64> = text
            .unwrap_or("")
            .split_ascii_whitespace()
            .map(|t| {
                t.parse()
                    .map_err(|_| bad(format!("`{what}` must hold numbers")))
            })
            .collect::<Result<_>>()?;
        (v.len() == n)
            .then_some(v)
            .ok_or_else(|| bad(format!("`{what}` must hold {n} numbers")))
    };
    let extent = numbers(
        piece.attr("Extent").or(dataset.attr("WholeExtent")),
        6,
        "Extent",
    )?;
    let n = [0, 1, 2].map(|a| (extent[2 * a + 1] - extent[2 * a] + 1.0).max(0.0) as usize);
    let nodes_count = n[0] * n[1] * n[2];
    let mut thickness = [1.0; 3];
    let nodes: Vec<[f64; 3]> = match kind {
        "vti" => {
            let origin = numbers(dataset.attr("Origin").or(Some("0 0 0")), 3, "Origin")?;
            let spacing = numbers(dataset.attr("Spacing").or(Some("1 1 1")), 3, "Spacing")?;
            let d = numbers(
                dataset.attr("Direction").or(Some("1 0 0 0 1 0 0 0 1")),
                9,
                "Direction",
            )?;
            thickness = [0, 1, 2].map(|a| if spacing[a] > 0.0 { spacing[a] } else { 1.0 });
            let mut nodes = Vec::with_capacity(nodes_count);
            for k in 0..n[2] {
                for j in 0..n[1] {
                    for i in 0..n[0] {
                        let ijk = [i, j, k];
                        let local = [0, 1, 2].map(|a| (extent[2 * a] + ijk[a] as f64) * spacing[a]);
                        nodes.push([0, 1, 2].map(|r| {
                            origin[r] + (0..3).map(|c| d[3 * r + c] * local[c]).sum::<f64>()
                        }));
                    }
                }
            }
            nodes
        }
        "vtr" => {
            let coordinates = piece
                .first("Coordinates")?
                .ok_or_else(|| bad("no Coordinates"))?
                .all("DataArray")?;
            let [x, y, z] = &coordinates[..] else {
                return Err(bad("Coordinates must hold three arrays"));
            };
            let [x, y, z] = [x, y, z].map(|a| file.values(a));
            let (x, y, z) = (x?, y?, z?);
            if [x.len(), y.len(), z.len()] != n {
                return Err(bad("coordinate arrays do not match the extent"));
            }
            let mut nodes = Vec::with_capacity(nodes_count);
            for &z in &z {
                for &y in &y {
                    nodes.extend(x.iter().map(|&x| [x, y, z]));
                }
            }
            nodes
        }
        _ => {
            let points = piece.first("Points")?.ok_or_else(|| bad("no Points"))?;
            file.coordinates(&points, nodes_count)?
        }
    };
    let values = file.columns(piece.first("PointData")?.as_ref(), nodes.len())?;
    let count = n.map(|n| n.saturating_sub(1).max(1));
    let cells = count.iter().product::<usize>() * (nodes_count > 0) as usize;
    let mut cell_data = file.columns(piece.first("CellData")?.as_ref(), cells)?;
    if point_data || (cell_data.num_columns() == 0 && values.num_columns() > 0) {
        return Ok(Vtk::Points(PointSet::new(nodes, values)?));
    }
    let mut keep: Vec<u32> = (0..cells as u32).collect();
    if let Ok(i) = cell_data.schema().index_of("vtkGhostType") {
        let ghost = arrow_cast::cast(cell_data.column(i), &DataType::Float64)?;
        let ghost = ghost.as_primitive::<Float64Type>();
        keep.retain(|&c| ghost.is_null(c as usize) || (ghost.value(c as usize) as u8 & 32) == 0);
        cell_data.remove_column(i);
    }
    let cell_data = take_rows(&cell_data, keep.clone())?;
    let node = |ijk: [usize; 3]| nodes[ijk[0] + n[0] * (ijk[1] + n[1] * ijk[2])];
    if nodes_count > 0 {
        let o = node([0; 3]);
        let mut v = [[0.0; 3]; 3];
        for a in (0..3).filter(|&a| n[a] > 1) {
            let mut ijk = [0; 3];
            ijk[a] = 1;
            v[a] = sub(node(ijk), o);
        }
        let mut origin = o;
        for a in (0..3).filter(|&a| n[a] == 1) {
            let w = cross(v[(a + 1) % 3], v[(a + 2) % 3]);
            let len = dot(w, w).sqrt();
            v[a] = w.map(|x| x * thickness[a] / len);
            origin = sub(origin, v[a].map(|x| x / 2.0));
        }
        let scale = nodes
            .iter()
            .flatten()
            .fold(0.0f64, |m, x| m.max(x.abs()))
            .max(1.0);
        let on_lattice = || {
            (0..nodes_count).all(|p| {
                let ijk = [p % n[0], p / n[0] % n[1], p / (n[0] * n[1])];
                let q =
                    [0, 1, 2].map(|r| o[r] + (0..3).map(|a| ijk[a] as f64 * v[a][r]).sum::<f64>());
                near(nodes[p], q, scale)
            })
        };
        if let Some(geometry) = frame(origin, v, count).filter(|_| on_lattice()) {
            let model = if keep.len() == cells {
                BlockModel::regular(geometry, cell_data)?
            } else {
                BlockModel::masked(
                    geometry,
                    keep.iter().map(|&c| c as u64).collect(),
                    cell_data,
                )?
            };
            return Ok(Vtk::Blocks(model));
        }
    }
    let centers = keep
        .iter()
        .map(|&c| {
            let c = c as usize;
            let ijk = [
                c % count[0],
                c / count[0] % count[1],
                c / (count[0] * count[1]),
            ];
            let corners: Vec<[f64; 3]> = (0..8)
                .filter(|b| (0..3).all(|a| n[a] > 1 || (b >> a) & 1 == 0))
                .map(|b| node([0, 1, 2].map(|a| ijk[a] + ((b >> a) & 1))))
                .collect();
            let k = corners.len() as f64;
            [0, 1, 2].map(|a| corners.iter().map(|p| p[a]).sum::<f64>() / k)
        })
        .collect();
    Ok(Vtk::Points(PointSet::new(centers, cell_data)?))
}

/// Hexahedra or voxels as a BlockModel when they are equal boxes on one
/// lattice, else as a PointSet of their centers.
fn blocks(points: &[[f64; 3]], cells: &Cells, cell_data: RecordBatch) -> Result<Vtk> {
    let corners = |c: usize| {
        let ids = cells.cell(c);
        let p = |k: usize| points[ids[k]];
        let b = if cells.types[c] == 11 { 2 } else { 3 };
        (p(0), [sub(p(1), p(0)), sub(p(b), p(0)), sub(p(4), p(0))])
    };
    let scale = points
        .iter()
        .flatten()
        .fold(0.0f64, |m, x| m.max(x.abs()))
        .max(1.0);
    let fit = || -> Option<BlockModel> {
        let (o, v) = corners(0);
        let geometry = frame(o, v, [1; 3])?;
        let mut ijk = Vec::with_capacity(cells.len());
        for c in 0..cells.len() {
            let (p, e) = corners(c);
            if (0..3).any(|a| !near(e[a], v[a], scale)) {
                return None;
            }
            let d = sub(p, o);
            let f = [0, 1, 2].map(|a| dot(d, v[a]) / dot(v[a], v[a]));
            let i = f.map(|x| x.round());
            let q = [0, 1, 2].map(|r| o[r] + (0..3).map(|a| i[a] * v[a][r]).sum::<f64>());
            if !near(p, q, scale) {
                return None;
            }
            ijk.push(i.map(|x| x as i64));
        }
        let low = [0, 1, 2].map(|a| ijk.iter().map(|x| x[a]).min().unwrap_or(0));
        let high = [0, 1, 2].map(|a| ijk.iter().map(|x| x[a]).max().unwrap_or(0));
        let count = [0, 1, 2].map(|a| (high[a] - low[a] + 1) as usize);
        let origin = [0, 1, 2].map(|r| o[r] + (0..3).map(|a| low[a] as f64 * v[a][r]).sum::<f64>());
        let geometry = Geometry {
            origin,
            count,
            ..geometry
        };
        let mut rows: Vec<(u64, u32)> = ijk
            .iter()
            .enumerate()
            .map(|(row, x)| {
                let at = [0, 1, 2].map(|a| (x[a] - low[a]) as usize);
                (geometry.index(at), row as u32)
            })
            .collect();
        rows.sort_unstable();
        if rows.windows(2).any(|w| w[0].0 == w[1].0) {
            return None;
        }
        let (index, rows): (Vec<u64>, Vec<u32>) = rows.into_iter().unzip();
        let attributes = take_rows(&cell_data, rows).ok()?;
        if index.len() as u64 == geometry.cells() {
            BlockModel::regular(geometry, attributes).ok()
        } else {
            BlockModel::masked(geometry, index, attributes).ok()
        }
    };
    if let Some(model) = fit() {
        return Ok(Vtk::Blocks(model));
    }
    let centers = (0..cells.len())
        .map(|c| {
            let (p, e) = corners(c);
            [0, 1, 2].map(|r| p[r] + (e[0][r] + e[1][r] + e[2][r]) / 2.0)
        })
        .collect();
    Ok(Vtk::Points(PointSet::new(centers, cell_data)?))
}

fn write_vti(path: &Path, model: &BlockModel, mut w: Writer) -> Result<()> {
    let g = model.geometry();
    let (rows, ghost): (Vec<Option<u32>>, Vec<f64>) = match model.layout() {
        Layout::Regular => ((0..model.len() as u32).map(Some).collect(), Vec::new()),
        Layout::Masked(index) => {
            let mut rows = vec![None; g.cells() as usize];
            for (row, &i) in index.iter().enumerate() {
                rows[i as usize] = Some(row as u32);
            }
            let ghost = rows
                .iter()
                .map(|r| if r.is_some() { 0.0 } else { 32.0 })
                .collect();
            (rows, ghost)
        }
        Layout::SubBlocked { .. } => return Err(bad("sub-blocked models go to .vtu")),
    };
    let attributes = model.attributes();
    let mut attributes = if attributes.num_columns() == 0 {
        empty(rows.len())
    } else {
        take_record_batch(attributes, &UInt32Array::from(rows))?
    };
    if !ghost.is_empty() {
        let ghost: ArrayRef = Arc::new(Float64Array::from(ghost));
        attributes = boitata_core::set_column(&attributes, "vtkGhostType", ghost)?;
    }
    let frame = block_frame(g.rotation);
    let direction: Vec<String> = (0..3)
        .flat_map(|r| (0..3).map(move |c| frame[(c, r)].to_string()))
        .collect();
    let [nx, ny, nz] = g.count;
    let extent = format!("0 {nx} 0 {ny} 0 {nz}");
    let join = |v: [f64; 3]| v.map(|x| x.to_string()).join(" ");
    let attrs = format!(
        r#" WholeExtent="{extent}" Origin="{}" Spacing="{}" Direction="{}""#,
        join(g.origin),
        join(g.size),
        direction.join(" ")
    );
    w.open("ImageData", &attrs);
    writeln!(w.xml, r#"<Piece Extent="{extent}">"#).expect("string");
    w.data("PointData", &empty(0))?;
    w.data("CellData", &attributes)?;
    w.xml.push_str("</Piece>\n");
    Ok(std::fs::write(path, w.finish("ImageData"))?)
}

/// Cells as VTK cell types and point ids, `offsets` starting at 0.
#[derive(Default)]
struct Cells {
    types: Vec<u8>,
    offsets: Vec<usize>,
    ids: Vec<usize>,
}

impl Cells {
    fn len(&self) -> usize {
        self.types.len()
    }

    fn cell(&self, c: usize) -> &[usize] {
        &self.ids[self.offsets[c]..self.offsets[c + 1]]
    }

    fn push(&mut self, kind: u8, ids: impl IntoIterator<Item = usize>) {
        if self.offsets.is_empty() {
            self.offsets.push(0);
        }
        self.ids.extend(ids);
        self.types.push(kind);
        self.offsets.push(self.ids.len());
    }

    /// Appends the cells of a `connectivity` and `offsets` pair, of `types`
    /// or all of type `kind`.
    fn extend(
        &mut self,
        file: &File,
        e: &Element,
        types: Option<Vec<f64>>,
        kind: u8,
    ) -> Result<()> {
        let ids = file.values(&named(e, "connectivity")?)?;
        let ends = file.values(&named(e, "offsets")?)?;
        if types.as_ref().is_some_and(|t| t.len() != ends.len()) {
            return Err(bad("one cell type per cell expected"));
        }
        let mut start = 0;
        for (c, &end) in ends.iter().enumerate() {
            let end = end as usize;
            let cell = ids
                .get(start..end)
                .ok_or_else(|| bad("cell offsets outside the connectivity"))?;
            let kind = types.as_ref().map_or(kind, |t| t[c] as u8);
            self.push(kind, cell.iter().map(|&i| i as usize));
            start = end;
        }
        Ok(())
    }
}

fn build(
    points: Vec<[f64; 3]>,
    point_data: RecordBatch,
    cells: Cells,
    cell_data: RecordBatch,
) -> Result<Vtk> {
    let of = |f: fn(u8) -> bool| {
        (0..cells.len())
            .filter(|&c| f(cells.types[c]))
            .collect::<Vec<_>>()
    };
    let surface = of(|t| matches!(t, 5..=9));
    if !surface.is_empty() {
        let (mut triangles, mut rows) = (Vec::new(), Vec::new());
        for &c in &surface {
            let ids = cells.cell(c);
            let ids: Vec<u32> = match cells.types[c] {
                8 if ids.len() == 4 => vec![ids[0], ids[1], ids[3], ids[2]],
                _ => ids.to_vec(),
            }
            .into_iter()
            .map(|i| i as u32)
            .collect();
            let n = ids.len().saturating_sub(2);
            if cells.types[c] == 6 {
                triangles.extend((0..n).map(|k| match k % 2 {
                    0 => [ids[k], ids[k + 1], ids[k + 2]],
                    _ => [ids[k + 1], ids[k], ids[k + 2]],
                }));
            } else {
                triangles.extend((1..=n).map(|k| [ids[0], ids[k], ids[k + 1]]));
            }
            rows.extend(std::iter::repeat_n(c as u32, n));
        }
        let mut mesh = Mesh::new(points, triangles)?;
        for (field, column) in point_data
            .schema()
            .fields()
            .iter()
            .zip(point_data.columns())
        {
            mesh = mesh.with_vertex_column(field.name(), column.clone())?;
        }
        let faces = take_rows(&cell_data, rows)?;
        for (field, column) in faces.schema().fields().iter().zip(faces.columns()) {
            mesh = mesh.with_face_column(field.name(), column.clone())?;
        }
        return Ok(Vtk::Mesh(mesh));
    }
    let lines = of(|t| matches!(t, 3 | 4));
    if lines.is_empty() {
        let boxes =
            (0..cells.len()).all(|c| matches!(cells.types[c], 11 | 12) && cells.cell(c).len() == 8);
        if boxes && cells.len() > 0 {
            return blocks(&points, &cells, cell_data);
        }
        return Ok(Vtk::Points(PointSet::new(points, point_data)?));
    }
    let feature = cell_data
        .column_by_name("feature")
        .map(|c| arrow_cast::cast(c, &DataType::Float64))
        .transpose()?;
    let feature = feature.as_ref().map(|c| c.as_primitive::<Float64Type>());
    let (mut vertices, mut parts, mut features, mut closed, mut rows) =
        (Vec::new(), vec![0u32], vec![0u32], Vec::new(), Vec::new());
    for (k, &c) in lines.iter().enumerate() {
        let mut ids = cells.cell(c);
        let ring = ids.len() > 2 && ids.first() == ids.last();
        if ring {
            ids = &ids[..ids.len() - 1];
        }
        vertices.extend(ids.iter().map(|&i| points[i]));
        parts.push(u32::try_from(vertices.len()).map_err(|_| bad("too many points"))?);
        closed.push(ring);
        let same = k > 0
            && feature.is_some_and(|f| {
                f.is_valid(c) && f.is_valid(lines[k - 1]) && f.value(c) == f.value(lines[k - 1])
            });
        if same {
            *features.last_mut().expect("feature") += 1;
        } else {
            features.push(features.last().expect("feature") + 1);
            rows.push(c as u32);
        }
    }
    let mut attributes = take_rows(&cell_data, rows)?;
    if let Ok(i) = attributes.schema().index_of("feature") {
        attributes.remove_column(i);
    }
    Ok(Vtk::Polylines(Polylines::new(
        vertices, parts, features, closed, attributes,
    )?))
}

fn empty(rows: usize) -> RecordBatch {
    RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(rows)),
    )
    .expect("empty batch")
}

fn take_rows(batch: &RecordBatch, rows: Vec<u32>) -> Result<RecordBatch> {
    if batch.num_columns() == 0 {
        return Ok(empty(rows.len()));
    }
    Ok(take_record_batch(batch, &UInt32Array::from(rows))?)
}

/// Points, cells, point data and cell data to write for `data`.
fn cells_of(data: &Vtk) -> Result<(Cow<'_, [[f64; 3]]>, Cells, RecordBatch, RecordBatch)> {
    let mut cells = Cells::default();
    Ok(match data {
        Vtk::Points(p) => {
            (0..p.len()).for_each(|i| cells.push(1, [i]));
            (
                p.coords().into(),
                cells,
                p.attributes().clone(),
                empty(p.len()),
            )
        }
        Vtk::Blocks(model) => {
            let corners = model.corners();
            for row in 0..corners.len() {
                cells.push(12, [0, 1, 3, 2, 4, 5, 7, 6].map(|k| 8 * row + k));
            }
            let points: Vec<[f64; 3]> = corners.into_iter().flatten().collect();
            let n = points.len();
            (points.into(), cells, empty(n), model.attributes().clone())
        }
        Vtk::Polylines(l) => {
            let (mut feature, mut first) = (Vec::new(), 0);
            for f in 0..l.len() {
                for p in l.feature_parts(f) {
                    let n = l.part(p).len();
                    let ring = l.closed()[p] && n > 1;
                    cells.push(4, (first..first + n).chain(ring.then_some(first)));
                    feature.push(f as u32);
                    first += n;
                }
            }
            let cell_data = take_rows(l.attributes(), feature.clone())?;
            let column: ArrayRef = Arc::new(UInt32Array::from(feature));
            let cell_data = boitata_core::set_column(&cell_data, "feature", column)?;
            (
                l.vertices().into(),
                cells,
                empty(l.vertices().len()),
                cell_data,
            )
        }
        Vtk::Mesh(m) => {
            for t in m.triangles() {
                cells.push(5, t.map(|i| i as usize));
            }
            let (v, f) = (m.vertex_attributes(), m.face_attributes());
            (m.vertices().into(), cells, v.clone(), f.clone())
        }
    })
}

struct Element<'a> {
    attrs: Vec<(String, String)>,
    body: &'a str,
}

impl<'a> Element<'a> {
    fn attr(&self, name: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    fn count(&self, name: &str) -> Result<usize> {
        self.attr(name)
            .unwrap_or("0")
            .trim()
            .parse()
            .map_err(|_| bad(format!("`{name}` must be a count")))
    }

    fn all(&self, tag: &str) -> Result<Vec<Element<'a>>> {
        elements(self.body, tag)
    }

    fn first(&self, tag: &str) -> Result<Option<Element<'a>>> {
        Ok(self.all(tag)?.into_iter().next())
    }
}

fn named<'a>(e: &Element<'a>, name: &str) -> Result<Element<'a>> {
    e.all("DataArray")?
        .into_iter()
        .find(|a| a.attr("Name") == Some(name))
        .ok_or_else(|| bad(format!("no `{name}` array")))
}

/// Elements `<tag ...>...</tag>` or `<tag .../>` in `text`, not nested in
/// one another.
fn elements<'a>(text: &'a str, tag: &str) -> Result<Vec<Element<'a>>> {
    let (open, close) = (format!("<{tag}"), format!("</{tag}>"));
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find(&open) {
        let after = &rest[at + open.len()..];
        if !after.starts_with(|c: char| c.is_whitespace() || c == '>' || c == '/') {
            rest = after;
            continue;
        }
        let end = after
            .find('>')
            .ok_or_else(|| bad(format!("unterminated <{tag}>")))?;
        let (head, tail) = (&after[..end], &after[end + 1..]);
        let attrs = attributes(head);
        let (body, next) = if head.ends_with('/') {
            ("", tail)
        } else {
            let e = tail
                .find(&close)
                .ok_or_else(|| bad(format!("no {close}")))?;
            (&tail[..e], &tail[e + close.len()..])
        };
        out.push(Element { attrs, body });
        rest = next;
    }
    Ok(out)
}

fn attributes(head: &str) -> Vec<(String, String)> {
    static ATTR: OnceLock<Regex> = OnceLock::new();
    let attr = ATTR
        .get_or_init(|| Regex::new(r#"([\w:.-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')"#).expect("regex"));
    attr.captures_iter(head)
        .map(|c| {
            let value = c.get(2).or(c.get(3)).map_or("", |m| m.as_str());
            (c[1].to_string(), unescape(value))
        })
        .collect()
}

fn unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// The XML of a file with how its binary arrays are stored.
struct File<'a> {
    xml: &'a str,
    big: bool,
    header: usize,
    zlib: bool,
    appended: &'a [u8],
    base64: bool,
    /// Sorted offsets of appended arrays and the end of the appended data.
    ends: Vec<usize>,
    /// Data values read as null.
    nodata: Vec<f64>,
}

impl<'a> File<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self> {
        let at = find(bytes, b"<AppendedData").unwrap_or(bytes.len());
        let xml = std::str::from_utf8(&bytes[..at]).map_err(|_| bad("XML is not UTF-8"))?;
        let mut file = File {
            xml,
            big: false,
            header: 4,
            zlib: false,
            appended: &[],
            base64: false,
            ends: Vec::new(),
            nodata: Vec::new(),
        };
        let root = file.root()?;
        file.big = root.attr("byte_order") == Some("BigEndian");
        file.header = match root.attr("header_type") {
            None | Some("UInt32") => 4,
            Some("UInt64") => 8,
            Some(t) => return Err(bad(format!("header_type {t} is not supported"))),
        };
        file.zlib = match root.attr("compressor") {
            None | Some("") => false,
            Some("vtkZLibDataCompressor") => true,
            Some(c) => return Err(bad(format!("compressor {c} is not supported"))),
        };
        if at < bytes.len() {
            let rest = &bytes[at..];
            let head_end = find(rest, b">").ok_or_else(|| bad("unterminated <AppendedData>"))?;
            let head =
                std::str::from_utf8(&rest[..head_end]).map_err(|_| bad("XML is not UTF-8"))?;
            file.base64 = head.contains("base64");
            let start = head_end
                + find(&rest[head_end..], b"_").ok_or_else(|| bad("no `_` in AppendedData"))?
                + 1;
            let data = &rest[start..];
            let end = data
                .windows(15)
                .rposition(|w| w == b"</AppendedData>")
                .unwrap_or(data.len());
            file.appended = data;
            let mut ends = vec![end];
            for tag in elements(xml, "DataArray")? {
                if tag.attr("format") == Some("appended") {
                    ends.push(tag.count("offset")?);
                }
            }
            ends.sort_unstable();
            file.ends = ends;
        }
        Ok(file)
    }

    fn root(&self) -> Result<Element<'a>> {
        let missing = || bad("no VTKFile element");
        let rest = &self.xml[self.xml.find("<VTKFile").ok_or_else(missing)?..];
        let end = rest.find('>').ok_or_else(missing)?;
        Ok(Element {
            attrs: attributes(&rest[..end]),
            body: &rest[end + 1..],
        })
    }

    fn uint(&self, b: &[u8], i: usize) -> Result<usize> {
        let h = self.header;
        let s = b
            .get(i * h..(i + 1) * h)
            .ok_or_else(|| bad("data block is truncated"))?;
        let mut v = 0u64;
        for (k, &byte) in s.iter().enumerate() {
            v |= (byte as u64) << (8 * if self.big { h - 1 - k } else { k });
        }
        usize::try_from(v).map_err(|_| bad("data block is too large"))
    }

    /// Bytes of a block: its header, then raw or zlib-compressed data.
    fn unpack(&self, b: &[u8]) -> Result<Vec<u8>> {
        let h = self.header;
        let truncated = || bad("data block is truncated");
        if !self.zlib {
            let n = self.uint(b, 0)?;
            return b.get(h..h + n).map(<[u8]>::to_vec).ok_or_else(truncated);
        }
        let blocks = self.uint(b, 0)?;
        let mut at = h * (3 + blocks);
        let mut out = Vec::new();
        for k in 0..blocks {
            let size = self.uint(b, 3 + k)?;
            let chunk = b.get(at..at + size).ok_or_else(truncated)?;
            flate2::read::ZlibDecoder::new(chunk)
                .read_to_end(&mut out)
                .map_err(|e| bad(format!("zlib: {e}")))?;
            at += size;
        }
        Ok(out)
    }

    fn values(&self, a: &Element) -> Result<Vec<f64>> {
        let ty = a.attr("type").unwrap_or("");
        let size = width(ty).ok_or_else(|| bad(format!("array type {ty} is not supported")))?;
        let text = a.body.split('<').next().unwrap_or("");
        let bytes = match a.attr("format").unwrap_or("ascii") {
            "ascii" => {
                return text
                    .split_ascii_whitespace()
                    .map(|t| {
                        t.parse::<f64>()
                            .map_err(|_| bad(format!("`{t}` is not a number")))
                    })
                    .collect();
            }
            "binary" => self.unpack(&base64(text.as_bytes())?)?,
            "appended" => {
                let offset = a.count("offset")?;
                let end = self.ends.iter().find(|&&e| e > offset).copied();
                let chunk = end
                    .and_then(|end| self.appended.get(offset..end))
                    .ok_or_else(|| bad("appended offset outside the data"))?;
                if self.base64 {
                    self.unpack(&base64(chunk)?)?
                } else {
                    self.unpack(chunk)?
                }
            }
            f => return Err(bad(format!("format {f} is not supported"))),
        };
        Ok(bytes
            .chunks_exact(size)
            .map(|c| {
                let mut b = [0u8; 8];
                b[..size].copy_from_slice(c);
                if self.big {
                    b[..size].reverse();
                }
                let [b0, b1, b2, b3, ..] = b;
                match ty {
                    "Int8" => b0 as i8 as f64,
                    "UInt8" => b0 as f64,
                    "Int16" => i16::from_le_bytes([b0, b1]) as f64,
                    "UInt16" => u16::from_le_bytes([b0, b1]) as f64,
                    "Int32" => i32::from_le_bytes([b0, b1, b2, b3]) as f64,
                    "UInt32" => u32::from_le_bytes([b0, b1, b2, b3]) as f64,
                    "Float32" => f32::from_le_bytes([b0, b1, b2, b3]) as f64,
                    "Int64" => i64::from_le_bytes(b) as f64,
                    "UInt64" => u64::from_le_bytes(b) as f64,
                    _ => f64::from_le_bytes(b),
                }
            })
            .collect())
    }

    fn coordinates(&self, points: &Element, n: usize) -> Result<Vec<[f64; 3]>> {
        let array = points
            .first("DataArray")?
            .ok_or_else(|| bad("no Points array"))?;
        let v = self.values(&array)?;
        if v.len() != 3 * n {
            return Err(bad(format!(
                "{} point coordinates, expected {}",
                v.len(),
                3 * n
            )));
        }
        Ok(v.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect())
    }

    /// One column per array component of `data`, numeric arrays only, with
    /// `<name>_valid` arrays applied as nulls.
    fn columns(&self, data: Option<&Element>, rows: usize) -> Result<RecordBatch> {
        let mut columns: Vec<(String, ArrayRef)> = Vec::new();
        for a in data
            .map(|d| d.all("DataArray"))
            .transpose()?
            .unwrap_or_default()
        {
            let ty = a.attr("type").unwrap_or("");
            if width(ty).is_none() {
                continue;
            }
            let name = a.attr("Name").unwrap_or("array");
            let n = a.count("NumberOfComponents")?.max(1);
            let v = self.values(&a)?;
            if v.len() != rows * n {
                return Err(bad(format!(
                    "array `{name}` has {} values, expected {}",
                    v.len(),
                    rows * n
                )));
            }
            for c in 0..n {
                let it = (0..rows).map(|r| v[r * n + c]).map(|x| {
                    let missing = x.is_nan()
                        || self
                            .nodata
                            .iter()
                            .any(|&m| x == m || (ty == "Float32" && x == m as f32 as f64));
                    (!missing).then_some(x)
                });
                let column: ArrayRef = if ty == "Float32" {
                    Arc::new(it.map(|x| x.map(|x| x as f32)).collect::<Float32Array>())
                } else {
                    Arc::new(it.collect::<Float64Array>())
                };
                let name = if n == 1 {
                    name.to_string()
                } else {
                    format!("{name}_{c}")
                };
                columns.push((name, column));
            }
        }
        let masks: Vec<(usize, usize)> = (0..columns.len())
            .filter_map(|m| {
                let base = columns[m].0.strip_suffix("_valid")?;
                Some((columns.iter().position(|(n, _)| n == base)?, m))
            })
            .collect();
        for &(c, m) in &masks {
            let mask = arrow_cast::cast(&columns[m].1, &DataType::Float64)?;
            let invalid: BooleanArray = mask
                .as_primitive::<Float64Type>()
                .iter()
                .map(|v| Some(v != Some(1.0)))
                .collect();
            columns[c].1 = arrow_select::nullif::nullif(&columns[c].1, &invalid)?;
        }
        let drop: Vec<usize> = masks.iter().map(|&(_, m)| m).collect();
        let (fields, arrays): (Vec<Field>, Vec<ArrayRef>) = columns
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !drop.contains(i))
            .map(|(_, (name, a))| (Field::new(name, a.data_type().clone(), true), a))
            .unzip();
        Ok(RecordBatch::try_new_with_options(
            Arc::new(Schema::new(fields)),
            arrays,
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )?)
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn width(ty: &str) -> Option<usize> {
    match ty {
        "Int8" | "UInt8" => Some(1),
        "Int16" | "UInt16" => Some(2),
        "Int32" | "UInt32" | "Float32" => Some(4),
        "Int64" | "UInt64" | "Float64" => Some(8),
        _ => None,
    }
}

/// Decodes base64 four characters at a time, so blocks encoded one after
/// another, each with its own padding, decode to their concatenation.
fn base64(text: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let (mut quad, mut n, mut pad) = ([0u32; 4], 0, 0);
    for &c in text {
        quad[n] = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => {
                pad += 1;
                0
            }
            c if c.is_ascii_whitespace() => continue,
            _ => return Err(bad("invalid base64")),
        } as u32;
        n += 1;
        if n == 4 {
            let w = quad[0] << 18 | quad[1] << 12 | quad[2] << 6 | quad[3];
            out.extend(&w.to_be_bytes()[1..4 - pad.min(2)]);
            (n, pad) = (0, 0);
        }
    }
    Ok(out)
}

struct Writer {
    xml: String,
    appended: Vec<u8>,
    ascii: bool,
    header: usize,
}

impl Writer {
    fn new(ascii: bool, header: usize) -> Self {
        Self {
            xml: String::new(),
            appended: Vec::new(),
            ascii,
            header,
        }
    }

    fn open(&mut self, tag: &str, attrs: &str) {
        let header = if self.header == 8 { "UInt64" } else { "UInt32" };
        writeln!(
            self.xml,
            "<?xml version=\"1.0\"?>\n<VTKFile type=\"{tag}\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"{header}\">\n<{tag}{attrs}>"
        )
        .expect("string");
    }

    fn finish(mut self, tag: &str) -> Vec<u8> {
        writeln!(self.xml, "</{tag}>").expect("string");
        let mut out = self.xml.into_bytes();
        if !self.ascii {
            out.extend(b"<AppendedData encoding=\"raw\">\n_");
            out.extend(self.appended);
            out.extend(b"\n</AppendedData>\n");
        }
        out.extend(b"</VTKFile>\n");
        out
    }

    fn array(&mut self, ty: &str, name: &str, components: usize, values: &[f64]) {
        let name = escape(name);
        let head =
            format!(r#"<DataArray type="{ty}" Name="{name}" NumberOfComponents="{components}""#);
        if self.ascii {
            writeln!(self.xml, "{head} format=\"ascii\">").expect("string");
            for v in values {
                match ty {
                    "Float32" => write!(self.xml, "{} ", *v as f32),
                    _ => write!(self.xml, "{v} "),
                }
                .expect("string");
            }
            self.xml.push_str("\n</DataArray>\n");
            return;
        }
        writeln!(
            self.xml,
            "{head} format=\"appended\" offset=\"{}\"/>",
            self.appended.len()
        )
        .expect("string");
        let size = width(ty).expect("type") * values.len();
        self.appended
            .extend(&(size as u64).to_le_bytes()[..self.header]);
        for &v in values {
            match ty {
                "Float32" => self.appended.extend((v as f32).to_le_bytes()),
                "Float64" => self.appended.extend(v.to_le_bytes()),
                "Int64" => self.appended.extend((v as i64).to_le_bytes()),
                _ => self.appended.push(v as u8),
            }
        }
    }

    fn data(&mut self, tag: &str, batch: &RecordBatch) -> Result<()> {
        writeln!(self.xml, "<{tag}>").expect("string");
        for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
            let dt = field.data_type();
            if !(dt.is_numeric() || dt == &DataType::Boolean) {
                return Err(Error::NotNumeric(field.name().clone()));
            }
            let ty = if field.name() == "vtkGhostType" {
                "UInt8"
            } else if dt == &DataType::Float32 {
                "Float32"
            } else {
                "Float64"
            };
            let values = arrow_cast::cast(column, &DataType::Float64)?;
            let values = values.as_primitive::<Float64Type>();
            let v: Vec<f64> = values.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
            self.array(ty, field.name(), 1, &v);
            if values.null_count() > 0 {
                let valid: Vec<f64> = (0..v.len())
                    .map(|i| values.is_valid(i) as u8 as f64)
                    .collect();
                self.array("UInt8", &format!("{}_valid", field.name()), 1, &valid);
            }
        }
        writeln!(self.xml, "</{tag}>").expect("string");
        Ok(())
    }

    fn points(&mut self, points: &[[f64; 3]]) {
        self.xml.push_str("<Points>\n");
        self.array("Float64", "Points", 3, points.as_flattened());
        self.xml.push_str("</Points>\n");
    }

    fn cells(&mut self, cells: &Cells, types: bool) {
        let ids: Vec<f64> = cells.ids.iter().map(|&i| i as f64).collect();
        let ends: Vec<f64> = cells.offsets.iter().skip(1).map(|&i| i as f64).collect();
        self.array("Int64", "connectivity", 1, &ids);
        self.array("Int64", "offsets", 1, &ends);
        if types {
            let types: Vec<f64> = cells.types.iter().map(|&t| t as f64).collect();
            self.array("UInt8", "types", 1, &types);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StringArray;
    use std::io::Write as _;

    fn dir() -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("boitata-vtk-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn mesh() -> Mesh {
        let m = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        )
        .unwrap();
        let v: ArrayRef = Arc::new(Float32Array::from(vec![
            Some(1.5),
            None,
            Some(3.0),
            Some(4.0),
        ]));
        let f: ArrayRef = Arc::new(Float64Array::from(vec![0.1, 0.2, 0.3, 0.4]));
        m.with_vertex_column("v", v)
            .unwrap()
            .with_face_column("f", f)
            .unwrap()
    }

    fn round_trip(name: &str, data: &Vtk, ascii: bool) -> Vtk {
        let path = dir().join(name);
        write_vtk(&path, data, ascii).unwrap();
        read_vtk(&path, false, &[]).unwrap()
    }

    #[test]
    fn mesh_round_trips_with_nulls() {
        let m = mesh();
        for (name, ascii) in [
            ("m.vtp", false),
            ("m.vtu", false),
            ("a.vtp", true),
            ("a.vtu", true),
        ] {
            let Vtk::Mesh(back) = round_trip(name, &Vtk::Mesh(m.clone()), ascii) else {
                panic!("not a mesh");
            };
            assert_eq!(back.vertices(), m.vertices());
            assert_eq!(back.triangles(), m.triangles());
            assert_eq!(back.vertex_attributes(), m.vertex_attributes());
            assert_eq!(back.face_attributes(), m.face_attributes());
        }
    }

    #[test]
    fn points_and_polylines_round_trip() {
        let a: ArrayRef = Arc::new(Float64Array::from(vec![Some(1.0), None]));
        let points = PointSet::new(
            vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            RecordBatch::try_from_iter([("a", a.clone())]).unwrap(),
        )
        .unwrap();
        let Vtk::Points(back) = round_trip("p.vtu", &Vtk::Points(points.clone()), false) else {
            panic!()
        };
        assert_eq!(back.coords(), points.coords());
        assert_eq!(back.attributes(), points.attributes());
        let lines = Polylines::new(
            (0..7).map(|i| [i as f64, (i * i) as f64, 0.0]).collect(),
            vec![0, 2, 5, 7],
            vec![0, 2, 3],
            vec![false, true, false],
            RecordBatch::try_from_iter([("a", a)]).unwrap(),
        )
        .unwrap();
        for name in ["l.vtp", "l.vtu"] {
            let Vtk::Polylines(back) = round_trip(name, &Vtk::Polylines(lines.clone()), false)
            else {
                panic!()
            };
            assert_eq!(back.vertices(), lines.vertices());
            assert_eq!(back.closed(), lines.closed());
            assert_eq!(
                (0..2).map(|f| back.feature_parts(f)).collect::<Vec<_>>(),
                [0..2, 2..3]
            );
            assert_eq!(back.attributes(), lines.attributes());
        }
    }

    #[test]
    fn text_columns_are_refused() {
        let text: ArrayRef = Arc::new(StringArray::from(vec!["a"; 4]));
        let m = Vtk::Mesh(mesh().with_face_column("t", text).unwrap());
        assert!(matches!(
            write_vtk(dir().join("t.vtp"), &m, false),
            Err(Error::NotNumeric(_))
        ));
    }

    fn encode(bytes: &[u8]) -> String {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let mut b = [0u8; 3];
            b[..chunk.len()].copy_from_slice(chunk);
            let w = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
            for k in 0..4 {
                out.push(if k <= chunk.len() {
                    ALPHABET[(w >> (18 - 6 * k) & 63) as usize] as char
                } else {
                    '='
                });
            }
        }
        out
    }

    fn block(data: &[u8], header: usize, zlib: bool, big: bool) -> Vec<u8> {
        let int = |v: usize| {
            let b = (v as u64).to_le_bytes()[..header].to_vec();
            if big {
                b.into_iter().rev().collect()
            } else {
                b
            }
        };
        if !zlib {
            return [int(data.len()), data.to_vec()].concat();
        }
        let halves = [&data[..data.len() / 2], &data[data.len() / 2..]];
        let packed: Vec<Vec<u8>> = halves
            .iter()
            .map(|h| {
                let mut e =
                    flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
                e.write_all(h).unwrap();
                e.finish().unwrap()
            })
            .collect();
        let mut out = [int(2), int(halves[0].len()), int(halves[1].len())].concat();
        packed.iter().for_each(|p| out.extend(int(p.len())));
        packed.iter().for_each(|p| out.extend(p));
        out
    }

    /// A vtp triangle in binary or appended raw encoding, every header type,
    /// byte order and compression.
    #[test]
    fn reads_binary_appended_compressed_and_both_header_types() {
        for header in [4, 8] {
            for zlib in [false, true] {
                for big in [false, true] {
                    for appended in [false, true] {
                        let le = |v: Vec<u8>| {
                            if big {
                                v.chunks(8)
                                    .flat_map(|c| c.iter().rev().copied().collect::<Vec<_>>())
                                    .collect()
                            } else {
                                v
                            }
                        };
                        let coords: Vec<u8> = le([0.0f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0]
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect());
                        let ids: Vec<u8> =
                            le([0i64, 1, 2].iter().flat_map(|v| v.to_le_bytes()).collect());
                        let ends: Vec<u8> = le(3i64.to_le_bytes().to_vec());
                        let blocks = [coords, ids, ends].map(|d| block(&d, header, zlib, big));
                        let mut tail = Vec::new();
                        let arrays: Vec<String> = blocks
                            .iter()
                            .map(|b| {
                                if appended {
                                    let at = tail.len();
                                    tail.extend(b);
                                    format!(r#"format="appended" offset="{at}"/>"#)
                                } else if zlib {
                                    let n =
                                        3 + b[..header].iter().map(|&x| x as usize).sum::<usize>();
                                    format!(
                                        r#"format="binary">{} {}</DataArray>"#,
                                        encode(&b[..header * n]),
                                        encode(&b[header * n..])
                                    )
                                } else {
                                    format!(
                                        r#"format="binary">{}{}</DataArray>"#,
                                        encode(&b[..header]),
                                        encode(&b[header..])
                                    )
                                }
                            })
                            .collect();
                        let text = format!(
                            r#"<VTKFile type="PolyData" byte_order="{}" header_type="UInt{}" {}><PolyData><Piece NumberOfPoints="3" NumberOfPolys="1">
<Points><DataArray type="Float64" NumberOfComponents="3" {}</Points>
<Polys><DataArray type="Int64" Name="connectivity" {}<DataArray type="Int64" Name="offsets" {}</Polys></Piece></PolyData>"#,
                            if big { "BigEndian" } else { "LittleEndian" },
                            header * 8,
                            if zlib {
                                r#"compressor="vtkZLibDataCompressor""#
                            } else {
                                ""
                            },
                            arrays[0],
                            arrays[1],
                            arrays[2]
                        );
                        let mut bytes = text.into_bytes();
                        if appended {
                            bytes.extend(b"<AppendedData encoding=\"raw\">_");
                            bytes.extend(&tail);
                            bytes.extend(b"</AppendedData>");
                        }
                        bytes.extend(b"</VTKFile>");
                        let path = dir().join(format!("b{header}{zlib}{big}{appended}.vtp"));
                        std::fs::write(&path, bytes).unwrap();
                        let Vtk::Mesh(m) = read_vtk(&path, false, &[]).unwrap() else {
                            panic!()
                        };
                        assert_eq!(m.triangles(), &[[0, 1, 2]]);
                        assert_eq!(
                            m.area(),
                            1.0,
                            "header {header} zlib {zlib} big {big} appended {appended}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn base64_decodes_separately_padded_blocks() {
        assert_eq!(
            base64(format!("{}{}", encode(b"ab"), encode(b"cde")).as_bytes()).unwrap(),
            b"abcde"
        );
        assert!(base64(b"a*bc").is_err());
    }

    fn model(rotation: [f64; 3], count: [usize; 3]) -> BlockModel {
        let n = count.iter().product::<usize>();
        let au: ArrayRef = Arc::new(Float64Array::from_iter(
            (0..n).map(|i| (i % 4 != 1).then_some(i as f64)),
        ));
        let geometry = Geometry {
            origin: [500.0, 7000.0, -40.0],
            size: [10.0, 5.0, 2.5],
            count,
            rotation,
        };
        BlockModel::regular(geometry, RecordBatch::try_from_iter([("au", au)]).unwrap()).unwrap()
    }

    fn same(back: &Vtk, model: &BlockModel) {
        let Vtk::Blocks(back) = back else {
            panic!("not blocks: {back:?}")
        };
        let (g, h) = (back.geometry(), model.geometry());
        assert_eq!(g.count, h.count);
        for a in 0..3 {
            assert!((g.origin[a] - h.origin[a]).abs() < 1e-9);
            assert!((g.size[a] - h.size[a]).abs() < 1e-9);
        }
        let (c, d) = (back.centroids(), model.centroids());
        assert!(
            c.iter()
                .flatten()
                .zip(d.iter().flatten())
                .all(|(a, b)| (a - b).abs() < 1e-9)
        );
        assert_eq!(back.layout(), model.layout());
        assert_eq!(back.attributes().columns(), model.attributes().columns());
    }

    #[test]
    fn block_models_round_trip_through_vti_and_vtu() {
        for (rotation, count) in [
            ([0.0; 3], [3, 2, 2]),
            ([30.0, 20.0, 10.0], [3, 2, 2]),
            ([300.0, 0.0, 0.0], [4, 3, 1]),
        ] {
            let m = model(rotation, count);
            let keep = BooleanArray::from_iter((0..m.len()).map(|i| Some(i % 5 != 1)));
            for m in [m.clone(), m.mask(&keep).unwrap()] {
                for (name, ascii) in [("b.vti", false), ("a.vti", true), ("b.vtu", false)] {
                    same(&round_trip(name, &Vtk::Blocks(m.clone()), ascii), &m);
                }
            }
        }
    }

    fn grid_file(name: &str, kind: &str, attrs: &str, piece: &str) -> std::path::PathBuf {
        let path = dir().join(name);
        let text = format!(
            r#"<?xml version="1.0"?><VTKFile type="{kind}" version="1.0"><{kind} {attrs}><Piece {piece}</Piece></{kind}></VTKFile>"#
        );
        std::fs::write(&path, text).unwrap();
        path
    }

    fn array(name: &str, components: usize, values: &[f64]) -> String {
        let v: Vec<String> = values.iter().map(f64::to_string).collect();
        format!(
            r#"<DataArray type="Float64" Name="{name}" NumberOfComponents="{components}" format="ascii">{}</DataArray>"#,
            v.join(" ")
        )
    }

    #[test]
    fn rectilinear_grids_give_blocks_when_regular_else_cell_centers() {
        let coordinates = |x: &[f64]| {
            format!(
                r#"Extent="0 {} 0 1 0 0"><Coordinates>{}{}{}</Coordinates><CellData>{}</CellData>"#,
                x.len() - 1,
                array("x", 1, x),
                array("y", 1, &[0.0, 5.0]),
                array("z", 1, &[2.0]),
                array("au", 1, &[1.0, -999.0])
            )
        };
        let path = grid_file(
            "r.vtr",
            "RectilinearGrid",
            "",
            &coordinates(&[10.0, 12.0, 14.0]),
        );
        let Vtk::Blocks(m) = read_vtk(&path, false, &crate::default_nodata()).unwrap() else {
            panic!()
        };
        assert_eq!(m.geometry().origin, [10.0, 0.0, 1.5]);
        assert_eq!(
            (m.geometry().size, m.geometry().count),
            ([2.0, 5.0, 1.0], [2, 1, 1])
        );
        assert_eq!(m.geometry().rotation, [0.0; 3]);
        assert_eq!(m.attributes().column(0).null_count(), 1);
        let path = grid_file(
            "i.vtr",
            "RectilinearGrid",
            "",
            &coordinates(&[10.0, 12.0, 15.0]),
        );
        let Vtk::Points(p) = read_vtk(&path, false, &[]).unwrap() else {
            panic!()
        };
        assert_eq!(p.coords(), &[[11.0, 2.5, 2.0], [13.5, 2.5, 2.0]]);
        assert_eq!(p.attributes().column(0).null_count(), 0);
    }

    #[test]
    fn structured_grids_recover_a_rotated_lattice() {
        let m = model([25.0, 0.0, 0.0], [2, 3, 1]);
        let g = *m.geometry();
        let mut nodes = Vec::new();
        for k in 0..2 {
            for j in 0..4 {
                for i in 0..3 {
                    nodes.extend(g.point(0, [i as f64, j as f64, k as f64]));
                }
            }
        }
        let au: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let piece = format!(
            r#"Extent="0 2 0 3 0 1"><Points>{}</Points><CellData>{}</CellData>"#,
            array("Points", 3, &nodes),
            array("au", 1, &au)
        );
        let path = grid_file("s.vts", "StructuredGrid", "", &piece);
        let back = read_vtk(&path, false, &[]).unwrap();
        let au: ArrayRef = Arc::new(Float64Array::from(au));
        let m = m
            .with_attributes(RecordBatch::try_from_iter([("au", au)]).unwrap())
            .unwrap();
        same(&back, &m);
    }

    #[test]
    fn point_data_stays_on_the_nodes() {
        let attrs = r#"WholeExtent="0 2 0 1 0 0" Origin="0 0 0" Spacing="10 10 1""#;
        let only = format!(
            r#"Extent="0 2 0 1 0 0"><PointData>{}</PointData>"#,
            array("au", 1, &[1.0, 2.0, 3.0, -999.0, 5.0, 6.0])
        );
        let path = grid_file("p.vti", "ImageData", attrs, &only);
        let Vtk::Points(p) = read_vtk(&path, false, &crate::default_nodata()).unwrap() else {
            panic!()
        };
        assert_eq!(p.len(), 6);
        assert_eq!(p.coords()[4], [10.0, 10.0, 0.0]);
        assert_eq!(p.attributes().column(0).null_count(), 1);
        let both = format!(
            r#"Extent="0 2 0 1 0 0"><PointData>{}</PointData><CellData>{}</CellData>"#,
            array("node", 1, &[0.0; 6]),
            array("au", 1, &[1.0, 2.0])
        );
        let path = grid_file("c.vti", "ImageData", attrs, &both);
        let Vtk::Blocks(m) = read_vtk(&path, false, &[]).unwrap() else {
            panic!()
        };
        assert_eq!(m.centroids(), [[5.0, 5.0, 0.0], [15.0, 5.0, 0.0]]);
        assert_eq!(m.attributes().schema().field(0).name(), "au");
        let Vtk::Points(p) = read_vtk(&path, true, &[]).unwrap() else {
            panic!()
        };
        assert_eq!(
            (p.len(), p.attributes().schema().field(0).name().as_str()),
            (6, "node")
        );
    }

    /// Unstructured hexahedra of a rotated, sparse grid with an `index` cell
    /// array read back as a masked model.
    #[test]
    fn sparse_hexahedra_with_an_index_read_as_a_masked_model() {
        let m = model([40.0, 10.0, 0.0], [3, 2, 2]);
        let keep = BooleanArray::from_iter((0..m.len()).map(|i| Some(i % 2 == 0)));
        let m = m.mask(&keep).unwrap();
        let Layout::Masked(index) = m.layout().clone() else {
            panic!()
        };
        let column: ArrayRef = Arc::new(Float64Array::from_iter_values(
            index.iter().map(|&i| i as f64),
        ));
        let m = m.with_column("index", column).unwrap();
        let path = dir().join("sparse.vtu");
        write_vtk(&path, &Vtk::Blocks(m.clone()), false).unwrap();
        same(&read_vtk(&path, false, &[]).unwrap(), &m);
    }
}
