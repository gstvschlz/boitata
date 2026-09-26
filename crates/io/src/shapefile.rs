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
use ceres_core::PointSet;

use crate::{Error, Result, is_nodata};

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

/// Reads a point shapefile (`.shp` with its `.dbf` and optional `.prj`) as a
/// PointSet. Multipoint records give one point each, repeating their row; 2D
/// shapes get z = 0. Numeric fields are `Float64`, logical `Boolean`, the rest
/// text; blanks and `nodata` tokens are null. The `.prj` text is the CRS.
pub fn read_shapefile(path: impl AsRef<Path>, nodata: &[String]) -> Result<PointSet> {
    let path = path.as_ref();
    let shp = std::fs::read(path.with_extension("shp"))?;
    if be_i32(&shp, 0)? != 9994 {
        return Err(bad("not a shapefile"));
    }
    let mut coords = Vec::new();
    let mut rows = Vec::new();
    let (mut at, mut record) = (100, 0u32);
    while at < shp.len() {
        let length = count(be_i32(&shp, at + 4)?)? * 2;
        let c = shp
            .get(at + 8..at + 8 + length)
            .ok_or_else(|| bad("file is truncated"))?;
        let points = match le_i32(c, 0)? {
            0 => vec![],
            1 | 21 => vec![[le_f64(c, 4)?, le_f64(c, 12)?, 0.0]],
            11 => vec![[le_f64(c, 4)?, le_f64(c, 12)?, le_f64(c, 20)?]],
            kind @ (8 | 18 | 28) => {
                let n = count(le_i32(c, 36)?)?;
                let z0 = 40 + 16 * n + 16;
                (0..n)
                    .map(|i| {
                        let z = if kind == 18 {
                            le_f64(c, z0 + 8 * i)?
                        } else {
                            0.0
                        };
                        Ok([le_f64(c, 40 + 16 * i)?, le_f64(c, 48 + 16 * i)?, z])
                    })
                    .collect::<Result<_>>()?
            }
            kind => {
                return Err(bad(format!(
                    "shape type {kind} is not supported; only points are read"
                )));
            }
        };
        rows.extend(std::iter::repeat_n(record, points.len()));
        coords.extend(points);
        record += 1;
        at += 8 + length;
    }

    let table = read_dbf(&std::fs::read(path.with_extension("dbf"))?, nodata)?;
    if table.num_rows() != record as usize {
        return Err(bad(format!(
            "{record} shapes but {} attribute rows",
            table.num_rows()
        )));
    }
    let attributes = if table.num_columns() == 0 {
        RecordBatch::try_new_with_options(
            table.schema(),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(rows.len())),
        )?
    } else {
        take_record_batch(&table, &UInt32Array::from(rows))?
    };
    let mut points = PointSet::new(coords, attributes)?;
    points.crs = match std::fs::read_to_string(path.with_extension("prj")) {
        Ok(text) => Some(text.trim().to_string()).filter(|t| !t.is_empty()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e.into()),
    };
    Ok(points)
}

fn read_dbf(b: &[u8], nodata: &[String]) -> Result<RecordBatch> {
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
    let path = path.as_ref();
    let dbf = dbf_bytes(points.attributes())?;
    let coords = points.coords();
    let n = coords.len();

    let header = |words: usize| -> Result<Vec<u8>> {
        let words = i32::try_from(words).map_err(|_| bad("too many points for a shapefile"))?;
        let mut h = Vec::with_capacity(100);
        h.extend(9994i32.to_be_bytes());
        h.extend([0; 20]);
        h.extend(words.to_be_bytes());
        h.extend(1000i32.to_le_bytes());
        h.extend(11i32.to_le_bytes());
        let (mut lo, mut hi) = ([0.0; 3], [0.0; 3]);
        if n > 0 {
            (lo, hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for c in coords {
                for k in 0..3 {
                    lo[k] = lo[k].min(c[k]);
                    hi[k] = hi[k].max(c[k]);
                }
            }
        }
        for v in [lo[0], lo[1], hi[0], hi[1], lo[2], hi[2], 0.0, 0.0] {
            h.extend(v.to_le_bytes());
        }
        Ok(h)
    };

    let mut shp = header(50 + 22 * n)?;
    let mut shx = header(50 + 4 * n)?;
    for (i, c) in coords.iter().enumerate() {
        shx.extend((50 + 22 * i as i32).to_be_bytes());
        shx.extend(18i32.to_be_bytes());
        shp.extend((i as i32 + 1).to_be_bytes());
        shp.extend(18i32.to_be_bytes());
        shp.extend(11i32.to_le_bytes());
        for v in [c[0], c[1], c[2], 0.0] {
            shp.extend(v.to_le_bytes());
        }
    }

    std::fs::write(path.with_extension("shp"), shp)?;
    std::fs::write(path.with_extension("shx"), shx)?;
    std::fs::write(path.with_extension("dbf"), dbf)?;
    std::fs::write(path.with_extension("cpg"), "UTF-8")?;
    let prj = path.with_extension("prj");
    match &points.crs {
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

    #[test]
    fn round_trip_keeps_geometry_attributes_and_crs() {
        let dir = std::env::temp_dir().join("ceres-shapefile-round-trip");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("points.shp");
        let points = sample();
        write_shapefile(&path, &points).unwrap();
        let back = read_shapefile(&path, &[]).unwrap();
        assert_eq!(back.coords(), points.coords());
        assert_eq!(back.attributes(), points.attributes());
        assert_eq!(back.crs, points.crs);

        let mut plain = points.clone();
        plain.crs = None;
        write_shapefile(&path, &plain).unwrap();
        assert_eq!(read_shapefile(&path, &[]).unwrap().crs, None);

        let empty = RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(3)),
        )
        .unwrap();
        let bare = PointSet::new(points.coords().to_vec(), empty).unwrap();
        write_shapefile(&path, &bare).unwrap();
        let back = read_shapefile(&path, &[]).unwrap();
        assert_eq!(back.coords(), points.coords());
        assert_eq!(back.attributes().num_columns(), 0);
    }

    #[test]
    fn rejects_long_names_and_polygons() {
        let dir = std::env::temp_dir().join("ceres-shapefile-errors");
        std::fs::create_dir_all(&dir).unwrap();
        let long = RecordBatch::try_from_iter([(
            "a_very_long_name",
            Arc::new(Float64Array::from(vec![1.0])) as ArrayRef,
        )])
        .unwrap();
        let points = PointSet::new(vec![[0.0; 3]], long).unwrap();
        assert!(write_shapefile(dir.join("long.shp"), &points).is_err());

        write_shapefile(dir.join("poly.shp"), &sample()).unwrap();
        let mut shp = std::fs::read(dir.join("poly.shp")).unwrap();
        shp[108..112].copy_from_slice(&5i32.to_le_bytes());
        std::fs::write(dir.join("poly.shp"), shp).unwrap();
        assert!(read_shapefile(dir.join("poly.shp"), &[]).is_err());
    }
}
