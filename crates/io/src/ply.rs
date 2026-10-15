use std::fmt::Write as _;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{Array, ArrayRef, Float64Array, RecordBatch};
use arrow_schema::DataType;
use boitata_core::Mesh;

use crate::{Error, Result};

fn bad(message: impl Into<String>) -> Error {
    Error::Mesh(message.into())
}

#[derive(Clone, Copy)]
enum Scalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Scalar {
    fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "char" | "int8" => Self::I8,
            "uchar" | "uint8" => Self::U8,
            "short" | "int16" => Self::I16,
            "ushort" | "uint16" => Self::U16,
            "int" | "int32" => Self::I32,
            "uint" | "uint32" => Self::U32,
            "float" | "float32" => Self::F32,
            "double" | "float64" => Self::F64,
            _ => return Err(bad(format!("unknown PLY type `{name}`"))),
        })
    }

    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

/// A PLY property; `list` is the count type of a list property.
struct Property {
    name: String,
    kind: Scalar,
    list: Option<Scalar>,
}

struct Element {
    name: String,
    count: usize,
    properties: Vec<Property>,
}

enum Body<'a> {
    Ascii(std::str::SplitAsciiWhitespace<'a>),
    Binary { bytes: &'a [u8], big: bool },
}

impl Body<'_> {
    fn read(&mut self, kind: Scalar) -> Result<f64> {
        let (bytes, big) = match self {
            Body::Ascii(tokens) => {
                return tokens
                    .next()
                    .and_then(|t| t.parse().ok())
                    .ok_or_else(|| bad("PLY data ends early or holds a non-number"));
            }
            Body::Binary { bytes, big } => (bytes, *big),
        };
        let n = kind.size();
        if bytes.len() < n {
            return Err(bad("PLY data ends early"));
        }
        let mut b = [0u8; 8];
        b[..n].copy_from_slice(&bytes[..n]);
        *bytes = &bytes[n..];
        if big {
            b[..n].reverse();
        }
        let four = || b[..4].try_into().expect("4 bytes");
        Ok(match kind {
            Scalar::I8 => b[0] as i8 as f64,
            Scalar::U8 => b[0] as f64,
            Scalar::I16 => i16::from_le_bytes([b[0], b[1]]) as f64,
            Scalar::U16 => u16::from_le_bytes([b[0], b[1]]) as f64,
            Scalar::I32 => i32::from_le_bytes(four()) as f64,
            Scalar::U32 => u32::from_le_bytes(four()) as f64,
            Scalar::F32 => f32::from_le_bytes(four()) as f64,
            Scalar::F64 => f64::from_le_bytes(b),
        })
    }
}

fn float_column(values: Vec<f64>) -> ArrayRef {
    Arc::new(
        values
            .into_iter()
            .map(|v| (!v.is_nan()).then_some(v))
            .collect::<Float64Array>(),
    )
}

/// Fan triangles of a polygon of vertex indices.
fn fan(polygon: &[u32]) -> impl Iterator<Item = [u32; 3]> + '_ {
    (1..polygon.len().saturating_sub(1)).map(|k| [polygon[0], polygon[k], polygon[k + 1]])
}

pub(crate) fn parse_ply(bytes: &[u8]) -> Result<Mesh> {
    let end = bytes
        .windows(10)
        .position(|w| w == b"end_header")
        .ok_or_else(|| bad("no PLY header"))?;
    let start = bytes[end..]
        .iter()
        .position(|&b| b == b'\n')
        .map_or(bytes.len(), |p| end + p + 1);
    let header = std::str::from_utf8(&bytes[..end]).map_err(|_| bad("PLY header is not text"))?;
    let mut lines = header.lines();
    if lines.next().map(str::trim) != Some("ply") {
        return Err(bad("not a PLY file"));
    }
    let mut format = "";
    let mut elements: Vec<Element> = Vec::new();
    for line in lines {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let property = |elements: &mut Vec<Element>, p: Property| {
            elements
                .last_mut()
                .map(|e| e.properties.push(p))
                .ok_or_else(|| bad("PLY property before any element"))
        };
        match tokens[..] {
            ["format", f, _] => format = f,
            ["element", name, count] => elements.push(Element {
                name: name.into(),
                count: count.parse().map_err(|_| bad("bad PLY element count"))?,
                properties: Vec::new(),
            }),
            ["property", "list", count, kind, name] => {
                let p = Property {
                    name: name.into(),
                    kind: Scalar::parse(kind)?,
                    list: Some(Scalar::parse(count)?),
                };
                property(&mut elements, p)?;
            }
            ["property", kind, name] => {
                let p = Property {
                    name: name.into(),
                    kind: Scalar::parse(kind)?,
                    list: None,
                };
                property(&mut elements, p)?;
            }
            _ => {}
        }
    }
    let rest = &bytes[start..];
    let mut body = match format {
        "ascii" => Body::Ascii(
            std::str::from_utf8(rest)
                .map_err(|_| bad("ASCII PLY data is not text"))?
                .split_ascii_whitespace(),
        ),
        "binary_little_endian" => Body::Binary {
            bytes: rest,
            big: false,
        },
        "binary_big_endian" => Body::Binary {
            bytes: rest,
            big: true,
        },
        _ => return Err(bad(format!("unknown PLY format `{format}`"))),
    };
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    let mut vertex_columns = Vec::new();
    let mut face_columns = Vec::new();
    for e in &elements {
        let role = e.name.as_str();
        let is_xyz = |p: &Property| role == "vertex" && matches!(p.name.as_str(), "x" | "y" | "z");
        let mut columns: Vec<(&str, Vec<f64>)> = e
            .properties
            .iter()
            .filter(|p| p.list.is_none() && !is_xyz(p))
            .map(|p| (p.name.as_str(), Vec::new()))
            .collect();
        for _ in 0..e.count {
            let mut xyz = [f64::NAN; 3];
            let mut polygon = Vec::new();
            let mut row = Vec::with_capacity(columns.len());
            for p in &e.properties {
                if let Some(count) = p.list {
                    let n = body.read(count)? as usize;
                    let items = (0..n)
                        .map(|_| body.read(p.kind).map(|i| i as u32))
                        .collect::<Result<Vec<_>>>()?;
                    if role == "face"
                        && matches!(p.name.as_str(), "vertex_indices" | "vertex_index")
                    {
                        polygon = items;
                    }
                    continue;
                }
                let v = body.read(p.kind)?;
                match (is_xyz(p), p.name.as_str()) {
                    (true, "x") => xyz[0] = v,
                    (true, "y") => xyz[1] = v,
                    (true, _) => xyz[2] = v,
                    _ => row.push(v),
                }
            }
            let repeat = match role {
                "vertex" => {
                    vertices.push(xyz);
                    1
                }
                "face" => {
                    let before = triangles.len();
                    triangles.extend(fan(&polygon));
                    triangles.len() - before
                }
                _ => 0,
            };
            for ((_, column), v) in columns.iter_mut().zip(row) {
                column.extend(std::iter::repeat_n(v, repeat));
            }
        }
        match role {
            "vertex" => vertex_columns = columns,
            "face" => face_columns = columns,
            _ => {}
        }
    }
    let mut mesh = Mesh::new(vertices, triangles)?;
    for (name, values) in vertex_columns {
        mesh = mesh.with_vertex_column(name, float_column(values))?;
    }
    for (name, values) in face_columns {
        mesh = mesh.with_face_column(name, float_column(values))?;
    }
    Ok(mesh)
}

/// Numeric columns of `table` other than `reserved`, as PLY name, type and values.
fn properties(
    table: &RecordBatch,
    reserved: &[&str],
) -> Result<Vec<(String, &'static str, Float64Array)>> {
    let mut out = Vec::new();
    for (field, column) in table.schema().fields().iter().zip(table.columns()) {
        let name = field
            .name()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join("_");
        if !field.data_type().is_numeric() || reserved.contains(&name.as_str()) {
            continue;
        }
        let kind = match field.data_type() {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::UInt8
            | DataType::UInt16
                if column.null_count() == 0 =>
            {
                "int"
            }
            DataType::Float32 => "float",
            _ => "double",
        };
        let values = arrow_cast::cast(column, &DataType::Float64)?;
        out.push((name, kind, values.as_primitive::<Float64Type>().clone()));
    }
    Ok(out)
}

fn put(out: &mut Vec<u8>, ascii: bool, kind: &str, v: f64) {
    if ascii {
        out.extend(format!("{v} ").bytes());
        return;
    }
    match kind {
        "uchar" => out.push(v as u8),
        "int" => out.extend((v as i32).to_le_bytes()),
        "float" => out.extend((v as f32).to_le_bytes()),
        _ => out.extend(v.to_le_bytes()),
    }
}

fn put_row(out: &mut Vec<u8>, ascii: bool, columns: &[(String, &str, Float64Array)], i: usize) {
    for (_, kind, values) in columns {
        let v = if values.is_valid(i) {
            values.value(i)
        } else {
            f64::NAN
        };
        put(out, ascii, kind, v);
    }
    if ascii {
        out.pop();
        out.push(b'\n');
    }
}

/// PLY with double coordinates and numeric vertex and face columns as
/// properties; nulls become NaN.
pub(crate) fn ply_bytes(mesh: &Mesh, ascii: bool) -> Result<Vec<u8>> {
    let vertex = properties(mesh.vertex_attributes(), &["x", "y", "z"])?;
    let face = properties(mesh.face_attributes(), &["vertex_indices"])?;
    let format = if ascii {
        "ascii"
    } else {
        "binary_little_endian"
    };
    let mut header = format!("ply\nformat {format} 1.0\ncomment boitata\n");
    writeln!(header, "element vertex {}", mesh.vertices().len()).expect("string");
    for name in ["x", "y", "z"] {
        writeln!(header, "property double {name}").expect("string");
    }
    for (name, kind, _) in &vertex {
        writeln!(header, "property {kind} {name}").expect("string");
    }
    writeln!(header, "element face {}", mesh.triangles().len()).expect("string");
    header.push_str("property list uchar int vertex_indices\n");
    for (name, kind, _) in &face {
        writeln!(header, "property {kind} {name}").expect("string");
    }
    header.push_str("end_header\n");
    let mut out = header.into_bytes();
    for (i, p) in mesh.vertices().iter().enumerate() {
        p.iter().for_each(|&x| put(&mut out, ascii, "double", x));
        put_row(&mut out, ascii, &vertex, i);
    }
    for (t, triangle) in mesh.triangles().iter().enumerate() {
        put(&mut out, ascii, "uchar", 3.0);
        triangle
            .iter()
            .for_each(|&v| put(&mut out, ascii, "int", v as f64));
        put_row(&mut out, ascii, &face, t);
    }
    Ok(out)
}

pub(crate) fn parse_off(text: &str) -> Result<Mesh> {
    let malformed = || bad("malformed OFF file");
    let mut lines = text
        .lines()
        .map(|l| {
            l.split('#')
                .next()
                .unwrap_or("")
                .split_whitespace()
                .collect::<Vec<_>>()
        })
        .filter(|t| !t.is_empty());
    let mut first = lines.next().ok_or_else(malformed)?;
    if !first[0].ends_with("OFF") {
        return Err(bad("not an OFF file"));
    }
    first.remove(0);
    let counts = if first.is_empty() {
        lines.next().ok_or_else(malformed)?
    } else {
        first
    };
    let count = |k: usize| {
        counts
            .get(k)
            .and_then(|t| t.parse::<usize>().ok())
            .ok_or_else(malformed)
    };
    let (nv, nf) = (count(0)?, count(1)?);
    let mut vertices = Vec::new();
    for _ in 0..nv {
        let t = lines.next().ok_or_else(malformed)?;
        let p = (0..3)
            .map(|a| t.get(a).and_then(|x| x.parse().ok()))
            .collect::<Option<Vec<f64>>>()
            .ok_or_else(malformed)?;
        vertices.push([p[0], p[1], p[2]]);
    }
    let mut triangles = Vec::new();
    for _ in 0..nf {
        let t = lines.next().ok_or_else(malformed)?;
        let n: usize = t[0].parse().map_err(|_| malformed())?;
        let polygon = (1..=n)
            .map(|k| t.get(k).and_then(|i| i.parse().ok()))
            .collect::<Option<Vec<u32>>>()
            .ok_or_else(malformed)?;
        triangles.extend(fan(&polygon));
    }
    Ok(Mesh::new(vertices, triangles)?)
}

pub(crate) fn off_string(mesh: &Mesh) -> String {
    let mut out = format!(
        "OFF\n{} {} 0\n",
        mesh.vertices().len(),
        mesh.triangles().len()
    );
    for [x, y, z] in mesh.vertices() {
        writeln!(out, "{x} {y} {z}").expect("string");
    }
    for [a, b, c] in mesh.triangles() {
        writeln!(out, "3 {a} {b} {c}").expect("string");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float32Array, Int32Array, StringArray};

    fn tetra() -> Mesh {
        Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.0, 1.0, 1e-7],
                [0.0, 0.0, 1.0],
            ],
            vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        )
        .unwrap()
    }

    fn values(table: &RecordBatch, name: &str) -> Vec<Option<f64>> {
        let column = table.column_by_name(name).unwrap();
        column.as_primitive::<Float64Type>().iter().collect()
    }

    #[test]
    fn ply_round_trip_keeps_numeric_columns() {
        let m = tetra()
            .with_vertex_column(
                "grade",
                Arc::new(Float64Array::from(vec![
                    Some(1.25),
                    None,
                    Some(3.0),
                    Some(-4.0),
                ])),
            )
            .unwrap()
            .with_vertex_column("name", Arc::new(StringArray::from(vec!["a"; 4])))
            .unwrap()
            .with_face_column("domain", Arc::new(Int32Array::from(vec![1, 2, 2, 7])))
            .unwrap()
            .with_face_column("w", Arc::new(Float32Array::from(vec![0.5; 4])))
            .unwrap();
        for ascii in [true, false] {
            let back = parse_ply(&ply_bytes(&m, ascii).unwrap()).unwrap();
            assert_eq!(back.vertices(), m.vertices());
            assert_eq!(back.triangles(), m.triangles());
            assert_eq!(
                values(back.vertex_attributes(), "grade"),
                [Some(1.25), None, Some(3.0), Some(-4.0)]
            );
            assert!(back.vertex_attributes().column_by_name("name").is_none());
            assert_eq!(
                values(back.face_attributes(), "domain"),
                [1.0, 2.0, 2.0, 7.0].map(Some)
            );
            assert_eq!(values(back.face_attributes(), "w"), [Some(0.5); 4]);
        }
    }

    #[test]
    fn big_endian_ply_with_quads_and_extra_elements() {
        let mut bytes = b"ply\nformat binary_big_endian 1.0\nelement vertex 4\nproperty float x\n\
            property float y\nproperty float z\nproperty uchar red\nelement face 1\n\
            property list uchar uint vertex_index\nproperty short id\nelement edge 1\n\
            property int a\nproperty int b\nend_header\n"
            .to_vec();
        for (x, y) in [(0.0f32, 0.0f32), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
            [x, y, 0.0]
                .iter()
                .for_each(|v| bytes.extend(v.to_be_bytes()));
            bytes.push(200);
        }
        bytes.push(4);
        (0u32..4).for_each(|i| bytes.extend(i.to_be_bytes()));
        bytes.extend((-3i16).to_be_bytes());
        bytes.extend([0, 0, 0, 0, 0, 0, 0, 1]);
        let m = parse_ply(&bytes).unwrap();
        assert_eq!(m.triangles(), &[[0, 1, 2], [0, 2, 3]]);
        assert_eq!(m.area(), 1.0);
        assert_eq!(values(m.vertex_attributes(), "red"), [Some(200.0); 4]);
        assert_eq!(values(m.face_attributes(), "id"), [Some(-3.0); 2]);
        assert!(parse_ply(&bytes[..bytes.len() - 12]).is_err());
        assert!(parse_ply(b"garbage").is_err());
    }

    #[test]
    fn off_round_trip_and_polygons() {
        let back = parse_off(&off_string(&tetra())).unwrap();
        assert_eq!(back.vertices(), tetra().vertices());
        assert_eq!(back.triangles(), tetra().triangles());
        let text = "OFF # square\n4 1 0\n0 0 0\n1 0 0\n1 1 0\n\n0 1 0\n4 0 1 2 3 255 0 0\n";
        let m = parse_off(text).unwrap();
        assert_eq!(m.triangles(), &[[0, 1, 2], [0, 2, 3]]);
        assert!(parse_off("OFF\n4 1 0\n0 0 0\n").is_err());
        assert!(parse_off("PLY\n").is_err());
    }
}
