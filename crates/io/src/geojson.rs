//! GeoJSON (RFC 7946) feature collections as a PointSet or Polylines.

use std::fmt;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, RecordBatch, RecordBatchOptions, StringArray, UInt32Array,
};
use arrow_cast::cast;
use arrow_schema::{DataType, Field, Schema};
use arrow_select::take::take_record_batch;
use boitata_core::{PointSet, Polylines};
use serde::de::{Deserializer, IgnoredAny, MapAccess, Visitor};
use serde_json::{Value, json};

use crate::geopackage::polygons;
use crate::{Error, Nodata, Result, Shapes, is_nodata};

fn bad(message: impl Into<String>) -> Error {
    Error::GeoJson(message.into())
}

/// Property names in file order; serde_json maps sort their keys.
struct Keys(Vec<String>);

impl<'de> serde::Deserialize<'de> for Keys {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct Names;
        impl<'de> Visitor<'de> for Names {
            type Value = Keys;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an object")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Keys, A::Error> {
                let mut keys = vec![];
                while let Some((key, IgnoredAny)) = map.next_entry()? {
                    keys.push(key);
                }
                Ok(Keys(keys))
            }
        }
        d.deserialize_map(Names)
    }
}

#[derive(serde::Deserialize)]
struct Order {
    #[serde(default)]
    features: Vec<Order>,
    properties: Option<Keys>,
}

fn position(v: &Value) -> Result<[f64; 3]> {
    let a = v
        .as_array()
        .filter(|a| a.len() >= 2)
        .ok_or_else(|| bad("a position needs two or three numbers"))?;
    let n = |i: usize| {
        a.get(i)
            .map_or(Some(0.0), Value::as_f64)
            .ok_or_else(|| bad("coordinates must be numbers"))
    };
    Ok([n(0)?, n(1)?, n(2)?])
}

fn array(v: &Value) -> Result<&Vec<Value>> {
    v.as_array()
        .ok_or_else(|| bad("coordinates must be arrays"))
}

fn positions(v: &Value) -> Result<Vec<[f64; 3]>> {
    array(v)?.iter().map(position).collect()
}

/// Rings of a polygon as closed parts without the repeated closing vertex.
fn rings(v: &Value) -> Result<Vec<(Vec<[f64; 3]>, bool)>> {
    array(v)?
        .iter()
        .map(|r| {
            let mut ring = positions(r)?;
            if ring.len() > 1 && ring.first() == ring.last() {
                ring.pop();
            }
            Ok((ring, true))
        })
        .collect()
}

enum Shape {
    Points(Vec<[f64; 3]>),
    Parts(Vec<(Vec<[f64; 3]>, bool)>),
}

fn shape(g: &Value) -> Result<Option<Shape>> {
    if g.is_null() {
        return Ok(None);
    }
    let c = &g["coordinates"];
    let lines = |c: &Value| -> Result<Vec<_>> {
        array(c)?
            .iter()
            .map(|l| Ok((positions(l)?, false)))
            .collect()
    };
    Ok(Some(match g["type"].as_str() {
        Some("Point") => Shape::Points(vec![position(c)?]),
        Some("MultiPoint") => Shape::Points(positions(c)?),
        Some("LineString") => Shape::Parts(vec![(positions(c)?, false)]),
        Some("MultiLineString") => Shape::Parts(lines(c)?),
        Some("Polygon") => Shape::Parts(rings(c)?),
        Some("MultiPolygon") => Shape::Parts(
            array(c)?
                .iter()
                .map(rings)
                .collect::<Result<Vec<_>>>()?
                .concat(),
        ),
        Some(kind) => return Err(bad(format!("{kind} geometries are not supported"))),
        None => return Err(bad("a geometry needs a type")),
    }))
}

/// The CRS a `crs` member names, as `EPSG:n` when it can; EPSG:4326 without one.
fn crs(doc: &Value) -> String {
    let name = doc["crs"]["properties"]["name"].as_str().unwrap_or("CRS84");
    let code = name.rsplit(':').next().filter(|c| c.parse::<u32>().is_ok());
    match code {
        _ if name.ends_with("CRS84") => "EPSG:4326".into(),
        Some(code) if name.to_ascii_uppercase().contains("EPSG") => format!("EPSG:{code}"),
        _ => name.to_string(),
    }
}

fn column(values: &[&Value], nodata: &[Nodata]) -> (DataType, ArrayRef) {
    let values: Vec<&Value> = values
        .iter()
        .map(|&v| match v {
            Value::Number(n) if is_nodata(&n.to_string(), nodata) => &Value::Null,
            Value::String(s) if is_nodata(s.trim(), nodata) => &Value::Null,
            v => v,
        })
        .collect();
    let present = || values.iter().copied().filter(|v| !v.is_null());
    if present().all(Value::is_number) {
        let column = Float64Array::from_iter(values.iter().map(|v| v.as_f64()));
        return (DataType::Float64, Arc::new(column));
    }
    if present().all(Value::is_boolean) {
        let column = BooleanArray::from_iter(values.iter().map(|v| v.as_bool()));
        return (DataType::Boolean, Arc::new(column));
    }
    let text = values.iter().map(|v| match v {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        v => Some(v.to_string()),
    });
    (DataType::Utf8, Arc::new(StringArray::from_iter(text)))
}

/// Reads a GeoJSON feature collection, feature or geometry. Points and
/// multipoints give a PointSet, a multipoint repeating its row per point;
/// lines and polygons give Polylines, polygon rings as closed parts without
/// the repeated closing vertex, a null geometry as a feature without parts.
/// Positions without z get z = 0. Properties become columns in file order:
/// numbers `Float64`, booleans `Boolean`, the rest text (objects and arrays
/// as JSON); missing properties, JSON null and `nodata` values are null. The
/// CRS is the one the `crs` member names, else EPSG:4326 as RFC 7946 says.
pub fn read_geojson(path: impl AsRef<Path>, nodata: &[Nodata]) -> Result<Shapes> {
    let text = std::fs::read_to_string(path)?;
    let doc: Value = serde_json::from_str(&text).map_err(|e| bad(e.to_string()))?;
    let order: Order = serde_json::from_str(&text).map_err(|e| bad(e.to_string()))?;
    let (features, order): (Vec<(&Value, &Value)>, Vec<Order>) = match doc["type"].as_str() {
        Some("FeatureCollection") => (
            array(&doc["features"])?
                .iter()
                .map(|f| (&f["geometry"], &f["properties"]))
                .collect(),
            order.features,
        ),
        Some("Feature") => (vec![(&doc["geometry"], &doc["properties"])], vec![order]),
        _ => (vec![(&doc, &Value::Null)], vec![]),
    };
    let mut names: Vec<String> = vec![];
    for name in order
        .into_iter()
        .filter_map(|o| o.properties)
        .flat_map(|k| k.0)
    {
        if !names.contains(&name) {
            names.push(name);
        }
    }
    let shapes = features
        .iter()
        .map(|(g, _)| shape(g))
        .collect::<Result<Vec<_>>>()?;
    let (fields, columns): (Vec<Field>, Vec<ArrayRef>) = names
        .iter()
        .map(|name| {
            let values: Vec<&Value> = features
                .iter()
                .map(|(_, p)| p.get(name).unwrap_or(&Value::Null))
                .collect();
            let (dtype, column) = column(&values, nodata);
            (Field::new(name, dtype, true), column)
        })
        .unzip();
    let table = RecordBatch::try_new_with_options(
        Arc::new(Schema::new(fields)),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(features.len())),
    )?;
    let crs = Some(crs(&doc));

    if shapes
        .iter()
        .flatten()
        .any(|s| matches!(s, Shape::Parts(_)))
    {
        let (mut vertices, mut parts, mut feature_parts, mut closed) =
            (vec![], vec![0u32], vec![0u32], vec![]);
        for s in shapes {
            match s {
                Some(Shape::Points(_)) => {
                    return Err(bad("points mixed with lines or polygons; split the file"));
                }
                Some(Shape::Parts(p)) => {
                    for (part, c) in p {
                        vertices.extend(part);
                        parts.push(
                            u32::try_from(vertices.len()).map_err(|_| bad("too many points"))?,
                        );
                        closed.push(c);
                    }
                }
                None => {}
            }
            feature_parts.push(parts.len() as u32 - 1);
        }
        let mut lines = Polylines::new(vertices, parts, feature_parts, closed, table)?;
        lines.crs = crs;
        return Ok(Shapes::Polylines(lines));
    }
    let points = shapes
        .into_iter()
        .enumerate()
        .map(|(i, s)| match s {
            Some(Shape::Points(p)) => Ok(p),
            _ => Err(bad(format!("feature {i} has no geometry"))),
        })
        .collect::<Result<Vec<_>>>()?;
    let rows: Vec<u32> = (0..points.len() as u32)
        .flat_map(|r| std::iter::repeat_n(r, points[r as usize].len()))
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
    let mut out = PointSet::new(points.concat(), attributes)?;
    out.crs = crs;
    Ok(Shapes::Points(out))
}

/// Each row's properties as JSON object text, keys in column order.
fn properties(table: &RecordBatch) -> Result<Vec<String>> {
    let mut rows = vec![String::new(); table.num_rows()];
    for (field, column) in table.schema().fields().iter().zip(table.columns()) {
        let values: Vec<Value> = match column.data_type() {
            DataType::Boolean => column.as_boolean().iter().map(|v| json!(v)).collect(),
            t if t.is_numeric() => cast(column, &DataType::Float64)?
                .as_primitive::<Float64Type>()
                .iter()
                .map(|v| json!(v.filter(|v| v.is_finite())))
                .collect(),
            _ => cast(column, &DataType::Utf8)?
                .as_string::<i32>()
                .iter()
                .map(|v| json!(v))
                .collect(),
        };
        let key = json!(field.name()).to_string();
        for (row, value) in rows.iter_mut().zip(values) {
            let comma = if row.is_empty() { "" } else { "," };
            row.push_str(&format!("{comma}{key}:{value}"));
        }
    }
    Ok(rows)
}

fn write(path: &Path, geometries: Vec<Value>, attributes: &RecordBatch) -> Result<()> {
    let mut file = BufWriter::new(std::fs::File::create(path)?);
    file.write_all(br#"{"type":"FeatureCollection","features":["#)?;
    for (i, (g, p)) in geometries
        .into_iter()
        .zip(properties(attributes)?)
        .enumerate()
    {
        let comma = if i == 0 { "" } else { "," };
        write!(
            file,
            r#"{comma}{{"type":"Feature","geometry":{g},"properties":{{{p}}}}}"#
        )?;
    }
    file.write_all(b"]}")?;
    Ok(file.flush()?)
}

/// Writes a PointSet as a GeoJSON feature collection of 3D points, each with
/// its attributes as properties: numbers (NaN as null), booleans, and other
/// columns as text. Coordinates with a CRS are converted to longitude and
/// latitude on WGS 84 as RFC 7946 requires; without one they are written as
/// they are.
pub fn write_geojson(path: impl AsRef<Path>, points: &PointSet) -> Result<()> {
    let points = if points.crs.is_none() {
        points.clone()
    } else {
        points.to_crs("EPSG:4326")?
    };
    let geometries = points
        .coords()
        .iter()
        .map(|p| json!({"type": "Point", "coordinates": p}))
        .collect();
    write(path.as_ref(), geometries, points.attributes())
}

/// Writes Polylines as a GeoJSON feature collection: features of open parts
/// as LineString or MultiLineString, of closed parts as Polygon or
/// MultiPolygon, with exteriors counter-clockwise, holes clockwise and the
/// closing vertex repeated; a feature without parts has a null geometry.
/// Features mixing open and closed parts are rejected. Properties and CRS as
/// [`write_geojson`].
pub fn write_polylines_geojson(path: impl AsRef<Path>, lines: &Polylines) -> Result<()> {
    let lines = if lines.crs.is_none() {
        lines.clone()
    } else {
        lines.to_crs("EPSG:4326")?
    };
    let geometries = (0..lines.len())
        .map(|f| {
            let parts: Vec<&[[f64; 3]]> = lines.feature_parts(f).map(|p| lines.part(p)).collect();
            let closed: Vec<bool> = lines.feature_parts(f).map(|p| lines.closed()[p]).collect();
            Ok(match parts.len() {
                0 => Value::Null,
                _ if closed.iter().all(|&c| !c) => match parts.as_slice() {
                    [line] => json!({"type": "LineString", "coordinates": line}),
                    _ => json!({"type": "MultiLineString", "coordinates": parts}),
                },
                _ if closed.iter().all(|&c| c) => match polygons(&parts).as_slice() {
                    [polygon] => json!({"type": "Polygon", "coordinates": polygon}),
                    many => json!({"type": "MultiPolygon", "coordinates": many}),
                },
                _ => return Err(bad(format!("feature {f} mixes open and closed parts"))),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    write(path.as_ref(), geometries, lines.attributes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x: f64, s: f64) -> Vec<[f64; 3]> {
        vec![
            [x, x, 1.0],
            [x + s, x, 1.0],
            [x + s, x + s, 2.0],
            [x, x + s, 2.0],
        ]
    }

    #[test]
    fn reads_features_properties_and_crs() {
        let dir = std::env::temp_dir().join("boitata_geojson_read");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.geojson");
        std::fs::write(
            &path,
            r#"{"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:EPSG::31982"}},
            "features":[
              {"type":"Feature","geometry":{"type":"Point","coordinates":[1,2]},"properties":{"zn":1.5,"rock":"ox","ok":true}},
              {"type":"Feature","geometry":{"type":"MultiPoint","coordinates":[[3,4,5],[6,7,8]]},"properties":{"zn":-999,"rock":null}}]}"#,
        )
        .unwrap();
        let Shapes::Points(p) = read_geojson(&path, &crate::default_nodata()).unwrap() else {
            panic!("expected points")
        };
        assert_eq!(p.crs.as_deref(), Some("EPSG:31982"));
        assert_eq!(
            p.coords(),
            &[[1.0, 2.0, 0.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
        );
        let names: Vec<_> = p
            .attributes()
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, ["zn", "rock", "ok"]);
        assert_eq!(p.attributes().column(0).null_count(), 2);
        assert_eq!(p.attributes().column(2).data_type(), &DataType::Boolean);
    }

    #[test]
    fn polylines_round_trip_with_holes_nulls_and_3d() {
        let dir = std::env::temp_dir().join("boitata_geojson_lines");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lines.geojson");
        let mut vertices = square(0.0, 10.0);
        vertices.extend(square(2.0, 2.0));
        vertices.extend(square(20.0, 1.0));
        vertices.extend([[0.0, 30.0, 0.0], [5.0, 31.0, 1.0]]);
        let attributes = RecordBatch::try_from_iter([(
            "v",
            Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)])) as ArrayRef,
        )])
        .unwrap();
        let lines = Polylines::new(
            vertices,
            vec![0, 4, 8, 12, 14],
            vec![0, 3, 3, 4],
            vec![true, true, true, false],
            attributes,
        )
        .unwrap();
        write_polylines_geojson(&path, &lines).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("MultiPolygon") && text.contains("\"geometry\":null"));
        let Shapes::Polylines(back) = read_geojson(&path, &[]).unwrap() else {
            panic!("expected lines")
        };
        assert_eq!(back.crs.as_deref(), Some("EPSG:4326"));
        assert_eq!(back.len(), 3);
        assert_eq!(back.closed(), lines.closed());
        for (a, b) in back.area().iter().zip(lines.area()) {
            assert!((a - b).abs() < 1e-9 || a.is_nan() && b.is_nan(), "{a} {b}");
        }
        assert_eq!(back.part(3), lines.part(3));
        assert_eq!(back.attributes().column(0).null_count(), 1);
    }

    #[test]
    fn writes_longitude_latitude() {
        let dir = std::env::temp_dir().join("boitata_geojson_crs");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p.geojson");
        let mut points = PointSet::new(
            vec![[500_000.0, 7_400_000.0, 812.5]],
            RecordBatch::try_new_with_options(
                Arc::new(Schema::empty()),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(1)),
            )
            .unwrap(),
        )
        .unwrap();
        points.crs = Some("EPSG:31983".into());
        write_geojson(&path, &points).unwrap();
        let Shapes::Points(back) = read_geojson(&path, &[]).unwrap() else {
            panic!("expected points")
        };
        let [lon, lat, z] = back.coords()[0];
        assert!((lon + 45.0).abs() < 1e-9 && (-24.0..-23.0).contains(&lat) && z == 812.5);
        let again = back.to_crs("EPSG:31983").unwrap().coords()[0];
        assert!((again[0] - 500_000.0).abs() < 1e-6 && (again[1] - 7_400_000.0).abs() < 1e-6);
    }
}
