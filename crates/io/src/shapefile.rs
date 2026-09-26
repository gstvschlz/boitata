use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, RecordBatch, RecordBatchOptions, StringArray,
    UInt32Array,
};
use arrow_cast::cast;
use arrow_schema::{DataType, Field, Schema};
use arrow_select::take::take_record_batch;
use ceres_core::{PointSet, Polylines};

use crate::{Error, Nodata, Result, is_nodata};

fn bad(message: impl Into<String>) -> Error {
    Error::Shapefile(message.into())
}

fn bytes<const N: usize>(b: &[u8], at: usize) -> Result<[u8; N]> {
    b.get(at..at + N)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| bad("file is truncated"))
}

fn le_i32(b: &[u8], at: usize) -> Result<i32> {
    bytes(b, at).map(i32::from_le_bytes)
}

fn be_i32(b: &[u8], at: usize) -> Result<i32> {
    bytes(b, at).map(i32::from_be_bytes)
}

fn le_f64(b: &[u8], at: usize) -> Result<f64> {
    bytes(b, at).map(f64::from_le_bytes)
}

fn count(n: i32) -> Result<usize> {
    usize::try_from(n).map_err(|_| bad("negative length"))
}

/// `n` points from byte `at` of a record, z from the block after them.
fn xyz(c: &[u8], at: usize, n: usize, z: bool) -> Result<Vec<[f64; 3]>> {
    let z0 = at + 16 * n + 16;
    (0..n)
        .map(|i| {
            let z = if z { le_f64(c, z0 + 8 * i)? } else { 0.0 };
            Ok([le_f64(c, at + 16 * i)?, le_f64(c, at + 8 + 16 * i)?, z])
        })
        .collect()
}

/// Parts of one record; a point is one part of one vertex.
fn shape(c: &[u8]) -> Result<Vec<Vec<[f64; 3]>>> {
    Ok(match le_i32(c, 0)? {
        0 => vec![],
        1 | 21 => vec![xyz(c, 4, 1, false)?],
        11 => vec![vec![[le_f64(c, 4)?, le_f64(c, 12)?, le_f64(c, 20)?]]],
        kind @ (8 | 18 | 28) => vec![xyz(c, 40, count(le_i32(c, 36)?)?, kind == 18)?],
        kind @ (3 | 13 | 23 | 5 | 15 | 25) => {
            let (parts, n) = (count(le_i32(c, 36)?)?, count(le_i32(c, 40)?)?);
            let v = xyz(c, 44 + 4 * parts, n, matches!(kind, 13 | 15))?;
            let starts = (0..parts)
                .map(|i| count(le_i32(c, 44 + 4 * i)?))
                .chain([Ok(n)])
                .collect::<Result<Vec<_>>>()?;
            starts
                .windows(2)
                .map(|w| v.get(w[0]..w[1]).map(<[_]>::to_vec))
                .collect::<Option<_>>()
                .ok_or_else(|| bad("part offsets outside the points"))?
        }
        31 => return Err(bad("MultiPatch shapes are not supported")),
        kind => return Err(bad(format!("shape type {kind} is not supported"))),
    })
}

/// Contents of a shapefile: points or lines and polygons.
#[derive(Debug, Clone)]
pub enum Shapes {
    Points(PointSet),
    Polylines(Polylines),
}

/// Reads a shapefile (`.shp` with its `.dbf` and optional `.prj`). Point and
/// multipoint files give a PointSet, a multipoint repeating its row per point.
/// Line files give Polylines with open parts, polygon files closed parts
/// without the repeated closing vertex; a null shape is a feature without
/// parts. 2D shapes get z = 0, M values are dropped. Numeric fields are
/// `Float64`, logical `Boolean`, the rest text; blanks and `nodata` values are
/// null. The `.prj` text is the CRS.
pub fn read_shapefile(path: impl AsRef<Path>, nodata: &[Nodata]) -> Result<Shapes> {
    let path = path.as_ref();
    let shp = std::fs::read(path.with_extension("shp"))?;
    if be_i32(&shp, 0)? != 9994 {
        return Err(bad("not a shapefile"));
    }
    let kind = le_i32(&shp, 32)?;
    let mut shapes = Vec::new();
    let mut at = 100;
    while at < shp.len() {
        let length = count(be_i32(&shp, at + 4)?)? * 2;
        let c = shp
            .get(at + 8..at + 8 + length)
            .ok_or_else(|| bad("file is truncated"))?;
        shapes.push(shape(c)?);
        at += 8 + length;
    }

    let table = read_dbf(&std::fs::read(path.with_extension("dbf"))?, nodata)?;
    if table.num_rows() != shapes.len() {
        return Err(bad(format!(
            "{} shapes but {} attribute rows",
            shapes.len(),
            table.num_rows()
        )));
    }
    let crs = match std::fs::read_to_string(path.with_extension("prj")) {
        Ok(text) => Some(text.trim().to_string()).filter(|t| !t.is_empty()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e.into()),
    };

    if matches!(kind, 3 | 13 | 23 | 5 | 15 | 25) {
        let closed = matches!(kind, 5 | 15 | 25);
        let (mut vertices, mut parts, mut features) = (vec![], vec![0u32], vec![0u32]);
        for shape in shapes {
            for mut part in shape {
                if closed && part.len() > 1 && part.first() == part.last() {
                    part.pop();
                }
                vertices.extend(part);
                parts.push(u32::try_from(vertices.len()).map_err(|_| bad("too many points"))?);
            }
            features.push(parts.len() as u32 - 1);
        }
        let closed = vec![closed; parts.len() - 1];
        let mut lines = Polylines::new(vertices, parts, features, closed, table)?;
        lines.crs = crs;
        return Ok(Shapes::Polylines(lines));
    }
    let rows: Vec<u32> = (0..shapes.len() as u32)
        .flat_map(|r| std::iter::repeat_n(r, shapes[r as usize].iter().map(Vec::len).sum()))
        .collect();
    let attributes = if table.num_columns() == 0 {
        RecordBatch::try_new_with_options(
            table.schema(),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(rows.len())),
        )?
    } else {
        take_record_batch(&table, &UInt32Array::from(rows))?
    };
    let mut points = PointSet::new(shapes.into_iter().flatten().flatten().collect(), attributes)?;
    points.crs = crs;
    Ok(Shapes::Points(points))
}

fn read_dbf(b: &[u8], nodata: &[Nodata]) -> Result<RecordBatch> {
    let records = u32::from_le_bytes(bytes(b, 4)?) as usize;
    let header = u16::from_le_bytes(bytes(b, 8)?) as usize;
    let width = u16::from_le_bytes(bytes(b, 10)?) as usize;
    if b.len() < header {
        return Err(bad("attribute file is truncated"));
    }
    let mut fields = Vec::new();
    let (mut at, mut offset) = (32, 1);
    while at + 32 <= header && b[at] != 0x0D {
        let name = &b[at..at + 11];
        let name =
            String::from_utf8_lossy(&name[..name.iter().position(|&c| c == 0).unwrap_or(11)])
                .trim()
                .to_string();
        let (kind, len) = (b[at + 11], b[at + 16] as usize);
        if matches!(kind, b'I' | b'O' | b'B' | b'+' | b'@' | b'Y' | b'T') {
            return Err(bad(format!("binary field type `{}`", kind as char)));
        }
        fields.push((name, kind, offset, len));
        offset += len;
        at += 32;
    }
    if b.len() < header + records * width || offset > width {
        return Err(bad("attribute file is truncated"));
    }
    let tokens = |offset: usize, len: usize| {
        (0..records).map(move |r| {
            let start = header + r * width + offset;
            let token = String::from_utf8_lossy(&b[start..start + len]);
            let token = token.trim();
            (!is_nodata(token, nodata)).then(|| token.to_string())
        })
    };
    let (schema, columns): (Vec<Field>, Vec<ArrayRef>) = fields
        .into_iter()
        .map(|(name, kind, offset, len)| {
            let values = tokens(offset, len);
            let (dtype, column): (DataType, ArrayRef) = match kind {
                b'N' | b'F' => (
                    DataType::Float64,
                    Arc::new(Float64Array::from_iter(
                        values.map(|t| t.and_then(|t| t.parse().ok())),
                    )),
                ),
                b'L' => (
                    DataType::Boolean,
                    Arc::new(BooleanArray::from_iter(values.map(
                        |t| match t?.as_bytes().first()? {
                            b'T' | b't' | b'Y' | b'y' => Some(true),
                            b'F' | b'f' | b'N' | b'n' => Some(false),
                            _ => None,
                        },
                    ))),
                ),
                _ => (DataType::Utf8, Arc::new(StringArray::from_iter(values))),
            };
            (Field::new(name, dtype, true), column)
        })
        .unzip();
    Ok(RecordBatch::try_new_with_options(
        Arc::new(Schema::new(schema)),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(records)),
    )?)
}

/// Writes a PointSet as a PointZ shapefile: `.shp`, `.shx`, `.dbf` (UTF-8, as
/// the `.cpg` says) and, with a CRS, `.prj`. Attribute names are at most 10
/// bytes; numbers, booleans and text are supported, nulls are blank.
pub fn write_shapefile(path: impl AsRef<Path>, points: &PointSet) -> Result<()> {
    let records = points.coords().iter().map(|c| {
        let mut r = 11i32.to_le_bytes().to_vec();
        r.extend([c[0], c[1], c[2], 0.0].iter().flat_map(|v| v.to_le_bytes()));
        r
    });
    write_shapes(
        path.as_ref(),
        11,
        points.coords(),
        records,
        points.attributes(),
        &points.crs,
    )
}

/// Writes Polylines as a PolyLineZ shapefile when all parts are open, or a
/// PolygonZ one when all are closed; mixed parts are rejected. Polygon rings
/// repeat their first vertex and wind by nesting depth within their feature:
/// outer rings clockwise, holes counter-clockwise, islands in holes clockwise.
/// A feature without parts is a null shape. Other files as `write_shapefile`.
pub fn write_polylines_shapefile(path: impl AsRef<Path>, lines: &Polylines) -> Result<()> {
    let polygon = lines.closed().iter().any(|&c| c);
    if polygon && !lines.closed().iter().all(|&c| c) {
        return Err(bad("parts mix open and closed; split by `closed`"));
    }
    let kind: i32 = if polygon { 15 } else { 13 };
    let records = (0..lines.len()).map(|f| {
        let parts: Vec<Vec<[f64; 3]>> = lines
            .feature_parts(f)
            .map(|p| {
                let mut v = lines.part(p).to_vec();
                if polygon {
                    let depth = lines
                        .feature_parts(f)
                        .filter(|&q| q != p && inside(lines.part(q), v[0]))
                        .count();
                    if (area2(&v) > 0.0) == (depth % 2 == 0) {
                        v.reverse();
                    }
                    v.push(v[0]);
                }
                v
            })
            .collect();
        if parts.is_empty() {
            return 0i32.to_le_bytes().to_vec();
        }
        let v: Vec<[f64; 3]> = parts.concat();
        let (lo, hi) = bounds(&v);
        let mut r = kind.to_le_bytes().to_vec();
        r.extend(
            [lo[0], lo[1], hi[0], hi[1]]
                .iter()
                .flat_map(|x| x.to_le_bytes()),
        );
        r.extend((parts.len() as i32).to_le_bytes());
        r.extend((v.len() as i32).to_le_bytes());
        let mut start = 0;
        for p in &parts {
            r.extend((start as i32).to_le_bytes());
            start += p.len();
        }
        v.iter()
            .for_each(|p| r.extend([p[0], p[1]].iter().flat_map(|x| x.to_le_bytes())));
        r.extend(lo[2].to_le_bytes());
        r.extend(hi[2].to_le_bytes());
        v.iter().for_each(|p| r.extend(p[2].to_le_bytes()));
        r.extend(vec![0; 16 + 8 * v.len()]);
        r
    });
    write_shapes(
        path.as_ref(),
        kind,
        lines.vertices(),
        records,
        lines.attributes(),
        &lines.crs,
    )
}

/// Twice the signed xy area, positive counter-clockwise.
fn area2(ring: &[[f64; 3]]) -> f64 {
    let next = ring.iter().cycle().skip(1);
    ring.iter()
        .zip(next)
        .map(|(a, b)| a[0] * b[1] - b[0] * a[1])
        .sum()
}

/// Whether `p` is inside `ring` in xy by even-odd crossing.
fn inside(ring: &[[f64; 3]], p: [f64; 3]) -> bool {
    let next = ring.iter().cycle().skip(1);
    ring.iter()
        .zip(next)
        .filter(|(a, b)| {
            (a[1] > p[1]) != (b[1] > p[1])
                && p[0] < a[0] + (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1])
        })
        .count()
        % 2
        == 1
}

fn bounds(v: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    let Some(&first) = v.first() else {
        return ([0.0; 3], [0.0; 3]);
    };
    v.iter().fold((first, first), |(lo, hi), p| {
        (
            [0, 1, 2].map(|a| lo[a].min(p[a])),
            [0, 1, 2].map(|a| hi[a].max(p[a])),
        )
    })
}

fn write_shapes(
    path: &Path,
    kind: i32,
    vertices: &[[f64; 3]],
    records: impl Iterator<Item = Vec<u8>>,
    attributes: &RecordBatch,
    crs: &Option<String>,
) -> Result<()> {
    let dbf = dbf_bytes(attributes)?;
    let words =
        |bytes: usize| i32::try_from(bytes / 2).map_err(|_| bad("too large for a shapefile"));
    let (mut shp, mut shx) = (vec![0; 100], vec![0; 100]);
    for (i, r) in records.enumerate() {
        shx.extend(words(shp.len())?.to_be_bytes());
        shx.extend(words(r.len())?.to_be_bytes());
        shp.extend((i as i32 + 1).to_be_bytes());
        shp.extend(words(r.len())?.to_be_bytes());
        shp.extend(r);
    }
    let (lo, hi) = bounds(vertices);
    for file in [&mut shp, &mut shx] {
        let mut h = 9994i32.to_be_bytes().to_vec();
        h.extend([0; 20]);
        h.extend(words(file.len())?.to_be_bytes());
        h.extend(1000i32.to_le_bytes());
        h.extend(kind.to_le_bytes());
        h.extend(
            [lo[0], lo[1], hi[0], hi[1], lo[2], hi[2], 0.0, 0.0]
                .iter()
                .flat_map(|v| v.to_le_bytes()),
        );
        file[..100].copy_from_slice(&h);
    }

    std::fs::write(path.with_extension("shp"), shp)?;
    std::fs::write(path.with_extension("shx"), shx)?;
    std::fs::write(path.with_extension("dbf"), dbf)?;
    std::fs::write(path.with_extension("cpg"), "UTF-8")?;
    let prj = path.with_extension("prj");
    match crs {
        Some(crs) => std::fs::write(prj, crs)?,
        None => match std::fs::remove_file(prj) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.into()),
            _ => {}
        },
    }
    Ok(())
}

fn dbf_bytes(table: &RecordBatch) -> Result<Vec<u8>> {
    let rows = table.num_rows();
    let mut descriptors = Vec::new();
    let mut cells: Vec<Vec<Option<String>>> = Vec::new();
    for (field, column) in table.schema().fields().iter().zip(table.columns()) {
        let name = field.name();
        if name.len() > 10 || !name.is_ascii() {
            return Err(bad(format!(
                "column name `{name}` must be ASCII of at most 10 bytes"
            )));
        }
        let (kind, values): (u8, Vec<Option<String>>) = match column.data_type() {
            DataType::Boolean => (
                b'L',
                column
                    .as_boolean()
                    .iter()
                    .map(|v| v.map(|v| (if v { "T" } else { "F" }).into()))
                    .collect(),
            ),
            t if t.is_numeric() => (
                b'N',
                cast(column, &DataType::Float64)?
                    .as_primitive::<Float64Type>()
                    .iter()
                    .map(|v| v.filter(|v| v.is_finite()).map(|v| format!("{v:?}")))
                    .collect(),
            ),
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => (
                b'C',
                cast(column, &DataType::Utf8)?
                    .as_string::<i32>()
                    .iter()
                    .map(|v| v.map(str::to_string))
                    .collect(),
            ),
            t => return Err(bad(format!("column `{name}` has unsupported type {t}"))),
        };
        let width = values.iter().flatten().map(String::len).max().unwrap_or(0);
        if width > 254 {
            return Err(bad(format!("column `{name}` has values over 254 bytes")));
        }
        let decimals = values
            .iter()
            .flatten()
            .filter_map(|v| v.split('e').next()?.split_once('.'))
            .map(|(_, fraction)| fraction.len())
            .max()
            .unwrap_or(0)
            .min(15);
        let mut d = [0u8; 32];
        d[..name.len()].copy_from_slice(name.as_bytes());
        d[11] = kind;
        d[16] = width.max(1) as u8;
        d[17] = if kind == b'N' { decimals as u8 } else { 0 };
        descriptors.push(d);
        cells.push(values);
    }

    let width = 1 + descriptors.iter().map(|d| d[16] as usize).sum::<usize>();
    let header = 32 + 32 * descriptors.len() + 1;
    let (Ok(records), Ok(header16), Ok(width16)) = (
        u32::try_from(rows),
        u16::try_from(header),
        u16::try_from(width),
    ) else {
        return Err(bad("attribute table is too large for dBASE"));
    };
    let mut out = Vec::with_capacity(header + rows * width + 1);
    out.extend([0x03, 70, 1, 1]);
    out.extend(records.to_le_bytes());
    out.extend(header16.to_le_bytes());
    out.extend(width16.to_le_bytes());
    out.extend([0; 20]);
    descriptors.iter().for_each(|d| out.extend(d));
    out.push(0x0D);
    for r in 0..rows {
        out.push(b' ');
        for (d, values) in descriptors.iter().zip(&cells) {
            let len = d[16] as usize;
            let value = values[r].as_deref().unwrap_or("").as_bytes();
            let pad = std::iter::repeat_n(b' ', len - value.len());
            if d[11] == b'N' {
                out.extend(pad);
                out.extend(value);
            } else {
                out.extend(value);
                out.extend(pad);
            }
        }
    }
    out.push(0x1A);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> PointSet {
        let attributes = RecordBatch::try_from_iter([
            (
                "au",
                Arc::new(Float64Array::from(vec![Some(0.1 + 0.2), None, Some(-1e-7)])) as ArrayRef,
            ),
            (
                "rock",
                Arc::new(StringArray::from(vec![Some("óxido"), Some("fresh"), None])),
            ),
            (
                "ok",
                Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)])),
            ),
        ])
        .unwrap();
        let mut points = PointSet::new(
            vec![[1.5, 2.25, -3.0], [1e6, 7e6, 350.125], [0.0, 0.0, 0.0]],
            attributes,
        )
        .unwrap();
        points.crs = Some("PROJCS[\"SIRGAS 2000 / UTM zone 22S\"]".into());
        points
    }

    fn read_points(path: &Path) -> PointSet {
        match read_shapefile(path, &[]).unwrap() {
            Shapes::Points(p) => p,
            Shapes::Polylines(_) => panic!("expected points"),
        }
    }

    fn read_lines(path: &Path) -> Polylines {
        match read_shapefile(path, &[]).unwrap() {
            Shapes::Polylines(l) => l,
            Shapes::Points(_) => panic!("expected lines"),
        }
    }

    fn lines(v: Vec<[f64; 3]>, parts: Vec<u32>, features: Vec<u32>, closed: bool) -> Polylines {
        let names: Vec<String> = (1..features.len()).map(|f| format!("f{f}")).collect();
        let names =
            RecordBatch::try_from_iter([("name", Arc::new(StringArray::from(names)) as ArrayRef)])
                .unwrap();
        let closed = vec![closed; parts.len() - 1];
        let mut l = Polylines::new(v, parts, features, closed, names).unwrap();
        l.crs = Some("EPSG:31982".into());
        l
    }

    fn same(a: &Polylines, b: &Polylines) {
        assert_eq!(a.vertices(), b.vertices());
        assert_eq!(a.closed(), b.closed());
        let parts = |l: &Polylines| {
            (0..l.num_parts())
                .map(|p| l.part(p).len())
                .collect::<Vec<_>>()
        };
        assert_eq!(parts(a), parts(b));
        let features = |l: &Polylines| (0..l.len()).map(|f| l.feature_parts(f)).collect::<Vec<_>>();
        assert_eq!(features(a), features(b));
        let columns = |l: &Polylines| l.attributes().columns().to_vec();
        assert_eq!((columns(a), &a.crs), (columns(b), &b.crs));
    }

    #[test]
    fn round_trip_keeps_geometry_attributes_and_crs() {
        let dir = std::env::temp_dir().join("ceres-shapefile-round-trip");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("points.shp");
        let points = sample();
        write_shapefile(&path, &points).unwrap();
        let back = read_points(&path);
        assert_eq!(back.coords(), points.coords());
        assert_eq!(back.attributes(), points.attributes());
        assert_eq!(back.crs, points.crs);

        let mut plain = points.clone();
        plain.crs = None;
        write_shapefile(&path, &plain).unwrap();
        assert_eq!(read_points(&path).crs, None);

        let empty = RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(3)),
        )
        .unwrap();
        let bare = PointSet::new(points.coords().to_vec(), empty).unwrap();
        write_shapefile(&path, &bare).unwrap();
        let back = read_points(&path);
        assert_eq!(back.coords(), points.coords());
        assert_eq!(back.attributes().num_columns(), 0);
    }

    /// A clockwise pit with a counter-clockwise hole, a null shape and a
    /// two-part line come back unchanged; rings wound the other way are
    /// turned to that convention.
    #[test]
    fn polygons_and_lines_round_trip() {
        let dir = std::env::temp_dir().join("ceres-shapefile-lines");
        std::fs::create_dir_all(&dir).unwrap();
        let (pit, hole) = ([0., 10.], [4., 6.]);
        let square = |[a, b]: [f64; 2], z: f64| [[a, a, z], [a, b, z], [b, b, z], [b, a, z]];
        let mut v = square(pit, 1.).to_vec();
        v.extend(square(hole, 2.).iter().rev());
        let polygons = lines(v.clone(), vec![0, 4, 8], vec![0, 2, 2], true);
        write_polylines_shapefile(dir.join("pit.shp"), &polygons).unwrap();
        same(&read_lines(&dir.join("pit.shp")), &polygons);

        v.reverse();
        let flipped = lines(v, vec![0, 4, 8], vec![0, 2], true);
        write_polylines_shapefile(dir.join("flip.shp"), &flipped).unwrap();
        let back = read_lines(&dir.join("flip.shp"));
        assert!(area2(back.part(0)) > 0.0 && area2(back.part(1)) < 0.0);

        let v = vec![
            [0., 0., 5.],
            [1., 0., 5.],
            [2., 2., 6.],
            [3., 2., 6.],
            [4., 3., 6.],
        ];
        let line = lines(v, vec![0, 2, 5], vec![0, 2], false);
        write_polylines_shapefile(dir.join("line.shp"), &line).unwrap();
        same(&read_lines(&dir.join("line.shp")), &line);

        let mixed = Polylines::new(
            polygons.vertices().to_vec(),
            vec![0, 4, 8],
            vec![0, 2],
            vec![true, false],
            polygons.attributes().slice(0, 1),
        )
        .unwrap();
        assert!(write_polylines_shapefile(dir.join("mixed.shp"), &mixed).is_err());
    }

    #[test]
    fn rejects_long_names_and_multipatches() {
        let dir = std::env::temp_dir().join("ceres-shapefile-errors");
        std::fs::create_dir_all(&dir).unwrap();
        let long = RecordBatch::try_from_iter([(
            "a_very_long_name",
            Arc::new(Float64Array::from(vec![1.0])) as ArrayRef,
        )])
        .unwrap();
        let points = PointSet::new(vec![[0.0; 3]], long).unwrap();
        assert!(write_shapefile(dir.join("long.shp"), &points).is_err());

        write_shapefile(dir.join("patch.shp"), &sample()).unwrap();
        let mut shp = std::fs::read(dir.join("patch.shp")).unwrap();
        shp[32..36].copy_from_slice(&31i32.to_le_bytes());
        shp[108..112].copy_from_slice(&31i32.to_le_bytes());
        std::fs::write(dir.join("patch.shp"), shp).unwrap();
        let e = read_shapefile(dir.join("patch.shp"), &[]).unwrap_err();
        assert!(e.to_string().contains("MultiPatch"));
    }
}
