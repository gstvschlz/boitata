use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, StringArray};
use arrow_schema::DataType;
use boitata_core::{Mesh, Progress};
use dxf::entities::{Entity, EntityType, Face3D};
use dxf::{Drawing, DxfError};

use crate::{Error, Result};

impl From<DxfError> for Error {
    fn from(e: DxfError) -> Self {
        match e {
            DxfError::IoError(e) => Error::Io(e),
            e => Error::Mesh(e.to_string()),
        }
    }
}

/// Reads an OBJ, STL (binary or ASCII) or DXF (`3DFACE`) mesh by extension.
pub fn read_mesh(path: impl AsRef<Path>, progress: Option<&Progress>) -> Result<Mesh> {
    let path = path.as_ref();
    match extension(path)? {
        "obj" => parse_obj(&std::fs::read_to_string(path)?, progress),
        "stl" => parse_stl(&std::fs::read(path)?, progress),
        _ => from_drawing(&Drawing::load_file(path)?, progress),
    }
}

/// Writes OBJ, STL (binary unless `ascii`) or DXF by extension. DXF faces go
/// on the layer named by the face column `layer`, else layer `0`.
pub fn write_mesh(
    path: impl AsRef<Path>,
    mesh: &Mesh,
    ascii: bool,
    progress: Option<&Progress>,
) -> Result<()> {
    let path = path.as_ref();
    match extension(path)? {
        "obj" => Ok(std::fs::write(path, obj_string(mesh, progress))?),
        "stl" => Ok(std::fs::write(path, stl_bytes(mesh, ascii, progress))?),
        _ => Ok(to_drawing(mesh, progress)?.save_file(path)?),
    }
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

fn parse_obj(text: &str, progress: Option<&Progress>) -> Result<Mesh> {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    if let Some(p) = progress {
        p.set_total(text.lines().count() as u64);
    }
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
            }
            _ => {}
        }
        if let Some(p) = progress {
            p.inc();
        }
    }
    Ok(Mesh::new(vertices, triangles)?)
}

fn obj_string(mesh: &Mesh, progress: Option<&Progress>) -> String {
    let mut out = String::new();
    for [x, y, z] in mesh.vertices() {
        writeln!(out, "v {x} {y} {z}").expect("string");
    }
    for [a, b, c] in mesh.triangles() {
        writeln!(out, "f {} {} {}", a + 1, b + 1, c + 1).expect("string");
        if let Some(p) = progress {
            p.inc();
        }
    }
    out
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
fn stl_bytes(mesh: &Mesh, ascii: bool, progress: Option<&Progress>) -> Vec<u8> {
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
            if let Some(p) = progress {
                p.inc();
            }
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
        if let Some(p) = progress {
            p.inc();
        }
    }
    out
}

fn parse_stl(bytes: &[u8], progress: Option<&Progress>) -> Result<Mesh> {
    let mut corners = Vec::new();
    let count = bytes
        .get(80..84)
        .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")) as usize);
    if let Some(p) = progress {
        p.set_total(count.filter(|n| bytes.len() == 84 + 50 * n).unwrap_or(1) as u64);
    }
    if count.is_some_and(|n| bytes.len() == 84 + 50 * n) {
        for record in bytes[84..].chunks_exact(50) {
            let f = |i: usize| f32::from_le_bytes(record[i..i + 4].try_into().expect("4")) as f64;
            corners.extend((0..3).map(|v| [0, 1, 2].map(|a| f(12 + 12 * v + 4 * a))));
            if let Some(p) = progress {
                p.inc();
            }
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

fn from_drawing(drawing: &Drawing, progress: Option<&Progress>) -> Result<Mesh> {
    let mut corners = Vec::new();
    let mut layers = Vec::new();
    if let Some(p) = progress {
        p.set_total(drawing.entities().count() as u64);
    }
    for entity in drawing.entities() {
        if let EntityType::Face3D(f) = &entity.specific {
            let p = [
                &f.first_corner,
                &f.second_corner,
                &f.third_corner,
                &f.fourth_corner,
            ]
            .map(|p| [p.x, p.y, p.z]);
            corners.extend([p[0], p[1], p[2]]);
            layers.push(entity.common.layer.clone());
            if p[3] != p[2] {
                corners.extend([p[0], p[2], p[3]]);
                layers.push(entity.common.layer.clone());
            }
        }
        if let Some(p) = progress {
            p.inc();
        }
    }
    Ok(weld(&corners)?.with_face_column("layer", Arc::new(StringArray::from(layers)))?)
}

fn to_drawing(mesh: &Mesh, progress: Option<&Progress>) -> Result<Drawing> {
    let layers = mesh
        .face_attributes()
        .column_by_name("layer")
        .map(|c| arrow_cast::cast(c, &DataType::Utf8))
        .transpose()?;
    let mut drawing = Drawing::new();
    for t in 0..mesh.triangles().len() {
        let [a, b, c] = mesh.corners(t).map(|p| dxf::Point::new(p[0], p[1], p[2]));
        let mut entity = Entity::new(EntityType::Face3D(Face3D::new(a, b, c.clone(), c)));
        entity.common.layer = layers
            .as_ref()
            .map(|l| l.as_string::<i32>())
            .filter(|l| l.is_valid(t))
            .map_or("0", |l| l.value(t))
            .to_string();
        drawing.add_entity(entity);
        if let Some(p) = progress {
            p.inc();
        }
    }
    Ok(drawing)
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

    #[test]
    fn obj_round_trip() {
        let m = parse_obj(&obj_string(&tetra(), None), None).unwrap();
        assert_eq!(m.vertices(), tetra().vertices());
        assert_eq!(m.triangles(), tetra().triangles());
    }

    #[test]
    fn obj_quads_and_negative_indices() {
        let text =
            "# c\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvn 0 0 1\nf 1/1 2//1 3/3/1 4\nf -4 -3 -2\n";
        let m = parse_obj(text, None).unwrap();
        assert_eq!(m.triangles(), &[[0, 1, 2], [0, 2, 3], [0, 1, 2]]);
        assert!(parse_obj("v 0 0 0\nf 1 2 0\n", None).is_err());
    }

    #[test]
    fn binary_and_ascii_stl_read_the_same() {
        let m = tetra();
        let binary = parse_stl(&stl_bytes(&m, false, None), None).unwrap();
        let ascii = parse_stl(&stl_bytes(&m, true, None), None).unwrap();
        assert_eq!(corners(&binary), corners(&m));
        assert_eq!(binary.vertices(), ascii.vertices());
        assert_eq!(binary.triangles(), ascii.triangles());
        assert_eq!(binary.vertices().len(), 4);
        assert!(parse_stl(b"garbage", None).is_err());
    }

    #[test]
    fn dxf_round_trip_keeps_layers() {
        let layers = Arc::new(StringArray::from(vec!["a", "a", "b", "b"]));
        let m = tetra().with_face_column("layer", layers.clone()).unwrap();
        let mut buffer = Vec::new();
        to_drawing(&m, None).unwrap().save(&mut buffer).unwrap();
        let read = from_drawing(&Drawing::load(&mut buffer.as_slice()).unwrap(), None).unwrap();
        assert_eq!(corners(&read), corners(&m));
        let got = read.face_attributes().column_by_name("layer").unwrap();
        assert_eq!(got.as_string::<i32>(), layers.as_ref());
    }

    #[test]
    fn writers_tick_once_per_triangle() {
        let m = tetra();
        for (name, ascii) in [
            ("t.obj", false),
            ("t.stl", false),
            ("t.stl", true),
            ("t.dxf", false),
        ] {
            let path =
                std::env::temp_dir().join(format!("boitata-io-{}-{name}", std::process::id()));
            let progress = Progress::new(Some(4));
            write_mesh(&path, &m, ascii, Some(&progress)).unwrap();
            assert_eq!(progress.snapshot().0, 4);
        }
    }

    #[test]
    fn dxf_quad_face_is_two_triangles() {
        let mut drawing = Drawing::new();
        let p = |x, y| dxf::Point::new(x, y, 0.0);
        let face = Face3D::new(p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0));
        drawing.add_entity(Entity::new(EntityType::Face3D(face)));
        let m = from_drawing(&drawing, None).unwrap();
        assert_eq!(m.triangles(), &[[0, 1, 2], [0, 2, 3]]);
        assert_eq!(m.area(), 1.0);
    }
}
