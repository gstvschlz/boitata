use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, StringArray};
use arrow_schema::DataType;
use boitata_core::Mesh;
use dxf::entities::{Entity, EntityType, Face3D, Vertex};
use dxf::enums::AcadVersion;
use dxf::tables::Layer;
use dxf::{Drawing, DxfError, Handle};

use crate::{Error, Result};

/// Most vertices or faces one polyface mesh holds; its indices are 16-bit.
const POLYFACE_LIMIT: usize = 32767;

/// DXF entity that `write_mesh` stores triangles as.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DxfEntity {
    /// One `3DFACE` per triangle.
    #[default]
    Face3D,
    /// One `POLYLINE` polyface mesh with shared vertices per layer, split
    /// when it would exceed 32767 vertices or faces.
    Polyface,
}

impl From<DxfError> for Error {
    fn from(e: DxfError) -> Self {
        match e {
            DxfError::IoError(e) => Error::Io(e),
            e => Error::Mesh(e.to_string()),
        }
    }
}

/// Reads an OBJ, STL (binary or ASCII) or DXF mesh by extension. OBJ `g`
/// groups become the face column `group` when the file has any. DXF reads
/// `3DFACE`, polyface `POLYLINE` and `MESH` (level-0 cage, ASCII files only)
/// entities into triangles with a face column `layer`; `MESH` faces come last.
pub fn read_mesh(path: impl AsRef<Path>) -> Result<Mesh> {
    let path = path.as_ref();
    match extension(path)? {
        "obj" => parse_obj(&std::fs::read_to_string(path)?),
        "stl" => parse_stl(&std::fs::read(path)?),
        _ => parse_dxf(&std::fs::read(path)?),
    }
}

/// Writes OBJ, STL (binary unless `ascii`) or ASCII R2000 DXF by extension.
/// OBJ writes the face column `group` as `g` lines. DXF writes `dxf_entity`
/// on the layer named by the face column `layer`, else layer `0`.
pub fn write_mesh(
    path: impl AsRef<Path>,
    mesh: &Mesh,
    ascii: bool,
    dxf_entity: DxfEntity,
) -> Result<()> {
    let path = path.as_ref();
    match extension(path)? {
        "obj" => Ok(std::fs::write(path, obj_string(mesh)?)?),
        "stl" => Ok(std::fs::write(path, stl_bytes(mesh, ascii))?),
        _ => Ok(std::fs::write(path, dxf_bytes(mesh, dxf_entity)?)?),
    }
}

fn text_column(mesh: &Mesh, name: &str) -> Result<Option<StringArray>> {
    Ok(mesh
        .face_attributes()
        .column_by_name(name)
        .map(|c| arrow_cast::cast(c, &DataType::Utf8))
        .transpose()?
        .map(|c| c.as_string::<i32>().clone()))
}

fn text<'a>(column: Option<&'a StringArray>, t: usize, default: &'a str) -> &'a str {
    column
        .filter(|c| c.is_valid(t))
        .map_or(default, |c| c.value(t))
}

fn extension(path: &Path) -> Result<&'static str> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    ["obj", "stl", "dxf"]
        .into_iter()
        .find(|f| f.eq_ignore_ascii_case(ext))
        .ok_or_else(|| {
            Error::Mesh(format!(
                "{}: mesh files must be .obj, .stl or .dxf",
                path.display()
            ))
        })
}

/// Mesh from triangle corners, merging vertices with identical coordinate bits.
fn weld(corners: &[[f64; 3]]) -> Result<Mesh> {
    let mut vertices = Vec::new();
    let mut index: HashMap<[u64; 3], u32> = HashMap::new();
    let mut add = |p: [f64; 3]| {
        *index.entry(p.map(f64::to_bits)).or_insert_with(|| {
            vertices.push(p);
            (vertices.len() - 1) as u32
        })
    };
    let triangles = corners
        .chunks_exact(3)
        .map(|c| [add(c[0]), add(c[1]), add(c[2])])
        .collect();
    Ok(Mesh::new(vertices, triangles)?)
}

fn parse_obj(text: &str) -> Result<Mesh> {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    let mut groups = Vec::new();
    let mut group: Option<String> = None;
    for (n, line) in text.lines().enumerate() {
        let bad = |message: &str| Error::Format {
            line: n + 1,
            message: message.into(),
        };
        let mut tokens = line.split_whitespace();
        match tokens.next() {
            Some("v") => {
                let xyz: Vec<f64> = tokens
                    .take(3)
                    .map(str::parse)
                    .collect::<std::result::Result<_, _>>()
                    .map_err(|_| bad("vertex needs three numbers"))?;
                let [x, y, z] = xyz[..] else {
                    return Err(bad("vertex needs three numbers"));
                };
                vertices.push([x, y, z]);
            }
            Some("f") => {
                let count = vertices.len() as i64;
                let ids = tokens
                    .map(|t| {
                        let i: i64 = t.split('/').next()?.parse().ok()?;
                        u32::try_from(if i < 0 { count + i } else { i - 1 }).ok()
                    })
                    .collect::<Option<Vec<u32>>>()
                    .filter(|ids| ids.len() >= 3)
                    .ok_or_else(|| bad("face needs three or more vertex indices"))?;
                triangles.extend((1..ids.len() - 1).map(|k| [ids[0], ids[k], ids[k + 1]]));
                groups.resize(triangles.len(), group.clone());
            }
            Some("g") => {
                let names = tokens.collect::<Vec<_>>().join(" ");
                group = Some(if names.is_empty() {
                    "default".into()
                } else {
                    names
                });
            }
            _ => {}
        }
    }
    let mesh = Mesh::new(vertices, triangles)?;
    if group.is_none() {
        return Ok(mesh);
    }
    let groups = groups
        .into_iter()
        .map(|g| g.unwrap_or_else(|| "default".into()));
    Ok(mesh.with_face_column("group", Arc::new(StringArray::from_iter_values(groups)))?)
}

fn obj_string(mesh: &Mesh) -> Result<String> {
    let groups = text_column(mesh, "group")?;
    let mut out = String::new();
    for [x, y, z] in mesh.vertices() {
        writeln!(out, "v {x} {y} {z}").expect("string");
    }
    let mut current = None;
    for (t, [a, b, c]) in mesh.triangles().iter().enumerate() {
        if groups.is_some() {
            let group = text(groups.as_ref(), t, "default");
            if current != Some(group) {
                writeln!(out, "g {group}").expect("string");
                current = Some(group);
            }
        }
        writeln!(out, "f {} {} {}", a + 1, b + 1, c + 1).expect("string");
    }
    Ok(out)
}

fn normal(c: [[f64; 3]; 3]) -> [f64; 3] {
    let (u, v) = (
        [0, 1, 2].map(|a| c[1][a] - c[0][a]),
        [0, 1, 2].map(|a| c[2][a] - c[0][a]),
    );
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let len = n.iter().map(|x| x * x).sum::<f64>().sqrt();
    if len > 0.0 { n.map(|x| x / len) } else { n }
}

/// STL stores single precision; coordinates are rounded to f32 on write.
fn stl_bytes(mesh: &Mesh, ascii: bool) -> Vec<u8> {
    let n = mesh.triangles().len();
    if ascii {
        let mut out = String::from("solid boitata\n");
        for t in 0..n {
            let c = mesh.corners(t);
            let [i, j, k] = normal(c).map(|x| x as f32);
            writeln!(out, "facet normal {i} {j} {k}\nouter loop").expect("string");
            for [x, y, z] in c.map(|p| p.map(|x| x as f32)) {
                writeln!(out, "vertex {x} {y} {z}").expect("string");
            }
            out.push_str("endloop\nendfacet\n");
        }
        out.push_str("endsolid boitata\n");
        return out.into_bytes();
    }
    let mut out = Vec::with_capacity(84 + 50 * n);
    out.extend(b"boitata");
    out.resize(80, 0);
    out.extend((n as u32).to_le_bytes());
    for t in 0..n {
        let c = mesh.corners(t);
        for x in normal(c).into_iter().chain(c.into_iter().flatten()) {
            out.extend((x as f32).to_le_bytes());
        }
        out.extend([0, 0]);
    }
    out
}

fn parse_stl(bytes: &[u8]) -> Result<Mesh> {
    let mut corners = Vec::new();
    let count = bytes
        .get(80..84)
        .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")) as usize);
    if count.is_some_and(|n| bytes.len() == 84 + 50 * n) {
        for record in bytes[84..].chunks_exact(50) {
            let f = |i: usize| f32::from_le_bytes(record[i..i + 4].try_into().expect("4")) as f64;
            corners.extend((0..3).map(|v| [0, 1, 2].map(|a| f(12 + 12 * v + 4 * a))));
        }
    } else {
        let bad = || Error::Mesh("not a binary or ASCII STL file".into());
        let text = std::str::from_utf8(bytes).map_err(|_| bad())?;
        if !text.trim_start().starts_with("solid") {
            return Err(bad());
        }
        let mut tokens = text.split_whitespace();
        while let Some(token) = tokens.next() {
            if token == "vertex" {
                let mut p = [0.0; 3];
                for x in &mut p {
                    *x = tokens
                        .next()
                        .and_then(|t| t.parse::<f32>().ok())
                        .ok_or_else(bad)? as f64;
                }
                corners.push(p);
            }
        }
        if corners.len() % 3 != 0 {
            return Err(bad());
        }
    }
    weld(&corners)
}

type DxfMesh = (String, Vec<[f64; 3]>, Vec<Vec<usize>>);

fn parse_dxf(bytes: &[u8]) -> Result<Mesh> {
    let drawing = Drawing::load(&mut &bytes[..])?;
    let meshes = if bytes.starts_with(b"AutoCAD Binary DXF") {
        Vec::new()
    } else {
        mesh_entities(&String::from_utf8_lossy(bytes))?
    };
    let mut corners = Vec::new();
    let mut layers = Vec::new();
    let mut add = |polygon: &[[f64; 3]], layer: &str| {
        for k in 1..polygon.len().saturating_sub(1) {
            corners.extend([polygon[0], polygon[k], polygon[k + 1]]);
            layers.push(layer.to_string());
        }
    };
    for entity in drawing.entities() {
        let layer = &entity.common.layer;
        match &entity.specific {
            EntityType::Face3D(f) => {
                let p = [
                    &f.first_corner,
                    &f.second_corner,
                    &f.third_corner,
                    &f.fourth_corner,
                ]
                .map(|p| [p.x, p.y, p.z]);
                add(if p[3] == p[2] { &p[..3] } else { &p }, layer);
            }
            EntityType::Polyline(poly) if poly.is_polyface_mesh() => {
                let (faces, points): (Vec<&Vertex>, Vec<&Vertex>) = poly
                    .vertices()
                    .partition(|v| record(v).iter().any(|&i| i != 0));
                for face in faces {
                    let mut ids: Vec<i32> = record(face).into_iter().filter(|&i| i != 0).collect();
                    ids.dedup_by_key(|i| i.abs());
                    if ids.len() > 1 && ids[0].abs() == ids[ids.len() - 1].abs() {
                        ids.pop();
                    }
                    let polygon = ids
                        .iter()
                        .map(|i| {
                            let p = &points.get(i.unsigned_abs() as usize - 1)?.location;
                            Some([p.x, p.y, p.z])
                        })
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| Error::Mesh("polyface face index out of range".into()))?;
                    add(&polygon, layer);
                }
            }
            _ => {}
        }
    }
    for (layer, vertices, faces) in &meshes {
        for face in faces {
            add(
                &face.iter().map(|&i| vertices[i]).collect::<Vec<_>>(),
                layer,
            );
        }
    }
    Ok(weld(&corners)?.with_face_column("layer", Arc::new(StringArray::from(layers)))?)
}

fn record(v: &Vertex) -> [i32; 4] {
    [
        v.polyface_mesh_vertex_index1,
        v.polyface_mesh_vertex_index2,
        v.polyface_mesh_vertex_index3,
        v.polyface_mesh_vertex_index4,
    ]
}

/// `MESH` entities of the ENTITIES section of an ASCII DXF, which the `dxf`
/// crate skips, as their layer, level-0 vertices and faces.
fn mesh_entities(text: &str) -> Result<Vec<DxfMesh>> {
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    let mut meshes = Vec::new();
    let mut section = "";
    let mut entity: Option<Vec<(i32, &str)>> = None;
    for pair in lines.chunks_exact(2) {
        let (code, value) = (pair[0].parse().unwrap_or(-1), pair[1]);
        match (code, value) {
            (0, _) => {
                if let Some(pairs) = entity.take() {
                    meshes.push(mesh_entity(&pairs)?);
                }
                match value {
                    "SECTION" => section = "?",
                    "ENDSEC" => section = "",
                    "MESH" if section == "ENTITIES" => entity = Some(Vec::new()),
                    _ => {}
                }
            }
            (2, _) if section == "?" => section = value,
            _ => entity.iter_mut().for_each(|e| e.push((code, value))),
        }
    }
    if let Some(pairs) = entity {
        meshes.push(mesh_entity(&pairs)?);
    }
    Ok(meshes)
}

fn mesh_entity(pairs: &[(i32, &str)]) -> Result<DxfMesh> {
    let bad = || Error::Mesh("malformed MESH entity".into());
    let layer = pairs.iter().find(|p| p.0 == 8).map_or("0", |p| p.1);
    let start = pairs
        .iter()
        .position(|&p| p == (100, "AcDbSubDMesh"))
        .unwrap_or(0);
    let mut pairs = pairs[start..].iter();
    let count = |pairs: &mut std::slice::Iter<(i32, &str)>, code: i32| {
        pairs
            .find(|p| p.0 == code)
            .and_then(|p| p.1.parse::<usize>().ok())
            .ok_or_else(bad)
    };
    let n = count(&mut pairs, 92)?;
    let coordinates = (0..3 * n)
        .map(|k| match pairs.next() {
            Some(&(c, v)) if c == [10, 20, 30][k % 3] => v.parse::<f64>().map_err(|_| bad()),
            _ => Err(bad()),
        })
        .collect::<Result<Vec<_>>>()?;
    let vertices = coordinates
        .chunks_exact(3)
        .map(|p| [p[0], p[1], p[2]])
        .collect();
    let size = count(&mut pairs, 93)?;
    let list = (0..size)
        .map(|_| match pairs.next() {
            Some(&(90, v)) => v.parse::<usize>().map_err(|_| bad()),
            _ => Err(bad()),
        })
        .collect::<Result<Vec<_>>>()?;
    let mut faces = Vec::new();
    let mut rest = &list[..];
    while let Some((&k, tail)) = rest.split_first() {
        let face = tail
            .get(..k)
            .filter(|f| f.iter().all(|&i| i < n))
            .ok_or_else(bad)?;
        faces.push(face.to_vec());
        rest = &tail[k..];
    }
    Ok((layer.to_string(), vertices, faces))
}

fn dxf_bytes(mesh: &Mesh, entity: DxfEntity) -> Result<Vec<u8>> {
    let layers = text_column(mesh, "layer")?;
    let layer = |t: usize| text(layers.as_ref(), t, "0");
    let mut drawing = Drawing::new();
    drawing.header.version = AcadVersion::R2000;
    let mut out = Vec::new();
    if entity == DxfEntity::Face3D {
        for t in 0..mesh.triangles().len() {
            let [a, b, c] = mesh.corners(t).map(|p| dxf::Point::new(p[0], p[1], p[2]));
            let mut entity = Entity::new(EntityType::Face3D(Face3D::new(a, b, c.clone(), c)));
            entity.common.layer = layer(t).to_string();
            drawing.add_entity(entity);
        }
        drawing.save(&mut out)?;
        return Ok(out);
    }
    let mut groups: Vec<(&str, Vec<usize>)> = Vec::new();
    let mut index = HashMap::new();
    for t in 0..mesh.triangles().len() {
        let name = layer(t);
        let g = *index.entry(name).or_insert_with(|| {
            groups.push((name, Vec::new()));
            groups.len() - 1
        });
        groups[g].1.push(t);
    }
    for (name, _) in &groups {
        if !drawing.layers().any(|l| l.name == *name) {
            drawing.add_layer(Layer {
                name: name.to_string(),
                ..Default::default()
            });
        }
    }
    let mut handle = drawing.header.next_available_handle.0;
    let mut entities = String::new();
    for (name, triangles) in &groups {
        let mut local = HashMap::new();
        let mut points = Vec::new();
        let mut faces = Vec::new();
        for &t in triangles {
            let triangle = mesh.triangles()[t];
            let new = triangle.iter().filter(|v| !local.contains_key(*v)).count();
            if points.len() + new > POLYFACE_LIMIT || faces.len() == POLYFACE_LIMIT {
                polyface(&mut entities, &mut handle, name, mesh, &points, &faces);
                local.clear();
                points.clear();
                faces.clear();
            }
            faces.push(triangle.map(|v| {
                *local.entry(v).or_insert_with(|| {
                    points.push(v);
                    points.len()
                })
            }));
        }
        polyface(&mut entities, &mut handle, name, mesh, &points, &faces);
    }
    drawing.header.next_available_handle = Handle(handle);
    drawing.save(&mut out)?;
    let at = out
        .windows(11)
        .position(|w| w == b"\nENTITIES\r\n")
        .ok_or_else(|| Error::Mesh("no ENTITIES section written".into()))?;
    out.splice(at + 11..at + 11, entities.into_bytes());
    Ok(out)
}

/// Appends a polyface `POLYLINE` of `faces` over 1-based `points`, written by
/// hand because the `dxf` crate misflags polyface vertices.
fn polyface(
    out: &mut String,
    handle: &mut u64,
    layer: &str,
    mesh: &Mesh,
    points: &[u32],
    faces: &[[usize; 3]],
) {
    let mut entity = |out: &mut String, kind: &str, subclasses: &[&str]| {
        write!(
            out,
            "  0\r\n{kind}\r\n  5\r\n{handle:X}\r\n100\r\nAcDbEntity\r\n  8\r\n{layer}\r\n"
        )
        .expect("string");
        *handle += 1;
        for s in subclasses {
            write!(out, "100\r\n{s}\r\n").expect("string");
        }
    };
    let pairs = |out: &mut String, pairs: &[(i32, &dyn std::fmt::Display)]| {
        for (code, value) in pairs {
            write!(out, "{code:>3}\r\n{value}\r\n").expect("string");
        }
    };
    let origin: [(i32, &dyn std::fmt::Display); 3] = [(10, &0.0), (20, &0.0), (30, &0.0)];
    entity(out, "POLYLINE", &["AcDbPolyFaceMesh"]);
    pairs(out, &[(66, &1)]);
    pairs(out, &origin);
    pairs(out, &[(70, &64), (71, &points.len()), (72, &faces.len())]);
    for &v in points {
        let [x, y, z] = mesh.vertices()[v as usize];
        entity(out, "VERTEX", &["AcDbVertex", "AcDbPolyFaceMeshVertex"]);
        pairs(out, &[(10, &x), (20, &y), (30, &z), (70, &192)]);
    }
    for [a, b, c] in faces {
        entity(out, "VERTEX", &["AcDbFaceRecord"]);
        pairs(out, &origin);
        pairs(out, &[(70, &128), (71, a), (72, b), (73, c)]);
    }
    entity(out, "SEQEND", &[]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tetra() -> Mesh {
        Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        )
        .unwrap()
    }

    fn corners(m: &Mesh) -> Vec<[[f64; 3]; 3]> {
        (0..m.triangles().len()).map(|t| m.corners(t)).collect()
    }

    fn layers(m: &Mesh) -> Vec<&str> {
        let column = m.face_attributes().column_by_name("layer").unwrap();
        column
            .as_string::<i32>()
            .iter()
            .map(Option::unwrap)
            .collect()
    }

    #[test]
    fn obj_round_trip() {
        let m = parse_obj(&obj_string(&tetra()).unwrap()).unwrap();
        assert_eq!(m.vertices(), tetra().vertices());
        assert_eq!(m.triangles(), tetra().triangles());
        assert!(m.face_attributes().column_by_name("group").is_none());
    }

    #[test]
    fn obj_groups_round_trip() {
        let text = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3\ng top\nf 1 2 3 4\ng\nf 1 3 4\n";
        let m = parse_obj(text).unwrap();
        let group = |m: &Mesh| {
            let column = m.face_attributes().column_by_name("group").unwrap().clone();
            let column = column.as_string::<i32>();
            column
                .iter()
                .map(|g| g.unwrap().to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(group(&m), ["default", "top", "top", "default"]);
        let back = parse_obj(&obj_string(&m).unwrap()).unwrap();
        assert_eq!(back.triangles(), m.triangles());
        assert_eq!(group(&back), group(&m));
    }

    #[test]
    fn obj_quads_and_negative_indices() {
        let text =
            "# c\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvn 0 0 1\nf 1/1 2//1 3/3/1 4\nf -4 -3 -2\n";
        let m = parse_obj(text).unwrap();
        assert_eq!(m.triangles(), &[[0, 1, 2], [0, 2, 3], [0, 1, 2]]);
        assert!(parse_obj("v 0 0 0\nf 1 2 0\n").is_err());
    }

    #[test]
    fn binary_and_ascii_stl_read_the_same() {
        let m = tetra();
        let binary = parse_stl(&stl_bytes(&m, false)).unwrap();
        let ascii = parse_stl(&stl_bytes(&m, true)).unwrap();
        assert_eq!(corners(&binary), corners(&m));
        assert_eq!(binary.vertices(), ascii.vertices());
        assert_eq!(binary.triangles(), ascii.triangles());
        assert_eq!(binary.vertices().len(), 4);
        assert!(parse_stl(b"garbage").is_err());
    }

    #[test]
    fn dxf_round_trip_keeps_layers() {
        let layers_in = Arc::new(StringArray::from(vec!["a", "a", "b", "b"]));
        let m = tetra().with_face_column("layer", layers_in).unwrap();
        for entity in [DxfEntity::Face3D, DxfEntity::Polyface] {
            let bytes = dxf_bytes(&m, entity).unwrap();
            let read = parse_dxf(&bytes).unwrap();
            assert_eq!(corners(&read), corners(&m));
            assert_eq!(layers(&read), ["a", "a", "b", "b"]);
        }
    }

    #[test]
    fn polyface_shares_vertices_and_splits_past_the_index_limit() {
        let bytes = dxf_bytes(&tetra(), DxfEntity::Polyface).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.contains("AC1015") && text.contains("AcDbPolyFaceMesh"));
        assert_eq!(text.matches("\r\nVERTEX\r\n").count(), 4 + 4);
        let n = 182;
        let vertices = (0..n * n)
            .map(|k| [(k % n) as f64, (k / n) as f64, 0.0])
            .collect();
        let triangles = (0..n - 1)
            .flat_map(|j| (0..n - 1).map(move |i| j * n + i))
            .flat_map(|k| [[k, k + 1, k + n + 1], [k, k + n + 1, k + n]])
            .collect();
        let grid = Mesh::new(vertices, triangles).unwrap();
        let bytes = dxf_bytes(&grid, DxfEntity::Polyface).unwrap();
        assert!(
            String::from_utf8_lossy(&bytes)
                .matches("\r\nPOLYLINE\r\n")
                .count()
                > 1
        );
        let read = parse_dxf(&bytes).unwrap();
        assert_eq!(corners(&read), corners(&grid));
        assert_eq!(read.vertices().len(), grid.vertices().len());
    }

    const POLYFACE_AND_MESH: &str = "0\nSECTION\n2\nENTITIES\n\
        0\nPOLYLINE\n8\npf\n66\n1\n70\n64\n71\n4\n72\n1\n\
        0\nVERTEX\n8\npf\n10\n0\n20\n0\n30\n0\n70\n192\n\
        0\nVERTEX\n8\npf\n10\n1\n20\n0\n30\n0\n70\n192\n\
        0\nVERTEX\n8\npf\n10\n1\n20\n1\n30\n0\n70\n192\n\
        0\nVERTEX\n8\npf\n10\n0\n20\n1\n30\n0\n70\n192\n\
        0\nVERTEX\n8\npf\n10\n0\n20\n0\n30\n0\n70\n128\n71\n1\n72\n-2\n73\n3\n74\n-4\n\
        0\nSEQEND\n8\npf\n\
        0\nLINE\n8\nx\n10\n0\n20\n0\n30\n0\n11\n1\n21\n1\n31\n1\n\
        0\nMESH\n8\nm\n100\nAcDbEntity\n100\nAcDbSubDMesh\n71\n2\n72\n0\n91\n1\n92\n5\n\
        10\n0\n20\n0\n30\n1\n10\n1\n20\n0\n30\n1\n10\n1\n20\n1\n30\n1\n10\n0\n20\n1\n30\n1\n\
        10\n0.5\n20\n-1\n30\n1\n\
        93\n9\n90\n4\n90\n0\n90\n1\n90\n2\n90\n3\n90\n3\n90\n0\n90\n4\n90\n1\n\
        94\n1\n90\n0\n90\n1\n95\n0\n90\n0\n\
        0\nENDSEC\n0\nEOF\n";

    #[test]
    fn dxf_reads_polyface_and_mesh_entities() {
        let m = parse_dxf(POLYFACE_AND_MESH.as_bytes()).unwrap();
        assert_eq!(layers(&m), ["pf", "pf", "m", "m", "m"]);
        assert_eq!(m.triangles()[..2], [[0, 1, 2], [0, 2, 3]]);
        assert_eq!(m.vertices().len(), 9);
        assert_eq!(m.area(), 1.0 + 1.0 + 0.5);
        let bad = POLYFACE_AND_MESH.replace("90\n4\n90\n1\n94", "90\n4\n90\n7\n94");
        assert!(parse_dxf(bad.as_bytes()).is_err());
        let bad = POLYFACE_AND_MESH.replace("74\n-4", "74\n-9");
        assert!(parse_dxf(bad.as_bytes()).is_err());
    }

    #[test]
    fn dxf_quad_face_is_two_triangles() {
        let mut drawing = Drawing::new();
        let p = |x, y| dxf::Point::new(x, y, 0.0);
        let face = Face3D::new(p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0));
        drawing.add_entity(Entity::new(EntityType::Face3D(face)));
        let mut bytes = Vec::new();
        drawing.save(&mut bytes).unwrap();
        let m = parse_dxf(&bytes).unwrap();
        assert_eq!(m.triangles(), &[[0, 1, 2], [0, 2, 3]]);
        assert_eq!(m.area(), 1.0);
    }
}
