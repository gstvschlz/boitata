use std::collections::HashMap;
use std::fmt::Display;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{Array, ArrayRef, Float64Array, StringArray};
use arrow_schema::DataType;
use boitata_core::Mesh;

use crate::{Error, Result};

fn bad(message: impl Into<String>) -> Error {
    Error::Mesh(format!("MSH: {}", message.into()))
}

/// Nodes of a Gmsh element type, for the types up to fourth order.
fn node_count(kind: i64) -> Result<usize> {
    const NODES: [usize; 32] = [
        0, 2, 3, 4, 4, 8, 6, 5, 3, 6, 9, 10, 27, 18, 14, 1, 8, 20, 15, 13, 9, 10, 12, 15, 15, 21,
        4, 5, 6, 20, 35, 56,
    ];
    usize::try_from(kind)
        .ok()
        .and_then(|k| NODES.get(k))
        .filter(|&&n| n > 0)
        .copied()
        .ok_or_else(|| bad(format!("unsupported element type {kind}")))
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
    binary: bool,
}

impl<'a> Cursor<'a> {
    fn skip_space(&mut self) {
        while self.bytes.get(self.at).is_some_and(u8::is_ascii_whitespace) {
            self.at += 1;
        }
    }

    fn line(&mut self) -> Option<&'a [u8]> {
        self.skip_space();
        if self.at == self.bytes.len() {
            return None;
        }
        let start = self.at;
        while self.bytes.get(self.at).is_some_and(|&b| b != b'\n') {
            self.at += 1;
        }
        let line = self.bytes[start..self.at].trim_ascii();
        self.at = (self.at + 1).min(self.bytes.len());
        Some(line)
    }

    fn text(&mut self) -> Result<&'a str> {
        let line = self.line().ok_or_else(|| bad("file ends early"))?;
        std::str::from_utf8(line).map_err(|_| bad("text line is not UTF-8"))
    }

    fn rest_of_line(&mut self) {
        while self.bytes.get(self.at).is_some_and(|&b| b != b'\n') {
            self.at += 1;
        }
        self.at = (self.at + 1).min(self.bytes.len());
    }

    fn token<T: std::str::FromStr>(&mut self) -> Result<T> {
        self.skip_space();
        let start = self.at;
        while self
            .bytes
            .get(self.at)
            .is_some_and(|b| !b.is_ascii_whitespace())
        {
            self.at += 1;
        }
        std::str::from_utf8(&self.bytes[start..self.at])
            .ok()
            .and_then(|t| t.parse().ok())
            .ok_or_else(|| bad("bad or missing number"))
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let b = self
            .bytes
            .get(self.at..self.at + N)
            .ok_or_else(|| bad("file ends early"))?;
        self.at += N;
        Ok(b.try_into().expect("N bytes"))
    }

    fn size(&mut self) -> Result<u64> {
        if self.binary {
            Ok(u64::from_le_bytes(self.take()?))
        } else {
            self.token()
        }
    }

    fn int(&mut self) -> Result<i64> {
        if self.binary {
            Ok(i32::from_le_bytes(self.take()?) as i64)
        } else {
            self.token()
        }
    }

    fn double(&mut self) -> Result<f64> {
        if self.binary {
            Ok(f64::from_le_bytes(self.take()?))
        } else {
            self.token()
        }
    }

    fn end(&mut self, section: &str) -> Result<()> {
        let marker = format!("$End{section}");
        while let Some(line) = self.line() {
            if line == marker.as_bytes() {
                return Ok(());
            }
        }
        Err(bad(format!("no {marker}")))
    }
}

type Columns = Vec<(String, Vec<Option<f64>>)>;

/// Reads `$NodeData` or `$ElementData` into `columns`, one per component,
/// placing the value of tag `k` at the rows `rows(k)`.
fn read_data(
    c: &mut Cursor,
    columns: &mut Columns,
    len: usize,
    rows: impl Fn(u64) -> Result<Vec<usize>>,
) -> Result<()> {
    let strings: usize = c.token()?;
    let mut name = "data".to_string();
    for k in 0..strings {
        let text = c.text()?.trim_matches('"').to_string();
        if k == 0 {
            name = text;
        }
    }
    let reals: usize = c.token()?;
    for _ in 0..reals {
        c.token::<f64>()?;
    }
    let ints: usize = c.token()?;
    let ints = (0..ints)
        .map(|_| c.token())
        .collect::<Result<Vec<usize>>>()?;
    let (components, count) = (
        ints.get(1).copied().unwrap_or(1),
        ints.get(2).copied().unwrap_or(0),
    );
    if c.binary {
        c.rest_of_line();
    }
    let names: Vec<String> = match components {
        1 => vec![name],
        n => (0..n).map(|i| format!("{name}_{i}")).collect(),
    };
    let mut values = vec![vec![None; len]; components];
    for _ in 0..count {
        let tag = c.int()?;
        let at = rows(tag as u64)?;
        for column in &mut values {
            let v = c.double()?;
            at.iter().for_each(|&i| column[i] = Some(v));
        }
    }
    for (name, column) in names.into_iter().zip(values) {
        match columns.iter_mut().find(|(n, _)| *n == name) {
            Some(existing) => existing.1 = column,
            None => columns.push((name, column)),
        }
    }
    Ok(())
}

/// Triangles of an element with nodes `nodes`, for 3- and 6-node triangles
/// and 4-, 8- and 9-node quadrangles.
fn element_triangles(kind: i64, nodes: &[u32]) -> Vec<[u32; 3]> {
    match kind {
        2 | 9 => vec![[nodes[0], nodes[1], nodes[2]]],
        3 | 10 | 16 => vec![
            [nodes[0], nodes[1], nodes[2]],
            [nodes[0], nodes[2], nodes[3]],
        ],
        _ => Vec::new(),
    }
}

/// Block count of a 4.1 `$Nodes` or `$Elements` header, skipping its totals.
fn header(c: &mut Cursor) -> Result<u64> {
    let blocks = c.size()?;
    for _ in 0..3 {
        c.size()?;
    }
    Ok(blocks)
}

fn column(values: Vec<Option<f64>>) -> ArrayRef {
    Arc::new(Float64Array::from(values))
}

pub(crate) fn parse_msh(bytes: &[u8]) -> Result<Mesh> {
    let mut c = Cursor {
        bytes,
        at: 0,
        binary: false,
    };
    let mut v4 = None;
    let mut names: HashMap<(i64, i64), String> = HashMap::new();
    let mut entity_physical: HashMap<(i64, i64), i64> = HashMap::new();
    let mut vertices = Vec::new();
    let mut node_index: HashMap<u64, u32> = HashMap::new();
    let mut triangles = Vec::new();
    let mut physical: Vec<Option<(i64, i64)>> = Vec::new();
    let mut element_rows: HashMap<u64, Vec<usize>> = HashMap::new();
    let (mut node_columns, mut face_columns) = (Columns::new(), Columns::new());
    while let Some(marker) = c.line() {
        let Some(section) = marker.strip_prefix(b"$") else {
            return Err(bad("expected a $section"));
        };
        let section = String::from_utf8_lossy(section).into_owned();
        match section.as_str() {
            "MeshFormat" => {
                let version: String = c.token()?;
                let binary: u8 = c.token()?;
                let size: u8 = c.token()?;
                v4 = Some(match version.as_str() {
                    "4.1" => true,
                    v if v.starts_with("2.") && binary == 0 => false,
                    v => {
                        return Err(bad(format!(
                            "version {v} {} is not read",
                            ["ASCII", "binary"][binary.min(1) as usize]
                        )));
                    }
                });
                if binary == 1 {
                    c.rest_of_line();
                    if size != 8 || i32::from_le_bytes(c.take()?) != 1 {
                        return Err(bad("only little-endian binary with 8-byte sizes is read"));
                    }
                    c.binary = true;
                }
            }
            "PhysicalNames" => {
                for _ in 0..c.token::<usize>()? {
                    let (dim, tag): (i64, i64) = (c.token()?, c.token()?);
                    names.insert((dim, tag), c.text()?.trim_matches('"').to_string());
                }
            }
            "Entities" => {
                let counts = [c.size()?, c.size()?, c.size()?, c.size()?];
                for (dim, &n) in counts.iter().enumerate() {
                    for _ in 0..n {
                        let tag = c.int()?;
                        for _ in 0..if dim == 0 { 3 } else { 6 } {
                            c.double()?;
                        }
                        let tags = (0..c.size()?)
                            .map(|_| c.int())
                            .collect::<Result<Vec<_>>>()?;
                        if let Some(&p) = tags.first() {
                            entity_physical.insert((dim as i64, tag), p.abs());
                        }
                        if dim > 0 {
                            for _ in 0..c.size()? {
                                c.int()?;
                            }
                        }
                    }
                }
            }
            "Nodes" if v4 == Some(true) => {
                for _ in 0..header(&mut c)? {
                    let (dim, _, parametric) = (c.int()?, c.int()?, c.int()?);
                    let n = c.size()?;
                    let tags = (0..n).map(|_| c.size()).collect::<Result<Vec<_>>>()?;
                    for tag in tags {
                        let p = [c.double()?, c.double()?, c.double()?];
                        for _ in 0..if parametric == 1 { dim } else { 0 } {
                            c.double()?;
                        }
                        node_index.insert(tag, vertices.len() as u32);
                        vertices.push(p);
                    }
                }
            }
            "Nodes" => {
                for _ in 0..c.token::<usize>()? {
                    let tag = c.token()?;
                    node_index.insert(tag, vertices.len() as u32);
                    vertices.push([c.token()?, c.token()?, c.token()?]);
                }
            }
            "Elements" => {
                let v4 = v4 == Some(true);
                let blocks = if v4 { header(&mut c)? } else { 1 };
                for _ in 0..blocks {
                    let (entity, mut kind, n) = if v4 {
                        let entity = (c.int()?, c.int()?);
                        (entity, c.int()?, c.size()?)
                    } else {
                        ((0, 0), 0, c.token()?)
                    };
                    let block_group = entity_physical.get(&entity).map(|&p| (entity.0, p));
                    for _ in 0..n {
                        let tag = c.size()?;
                        let mut group = block_group;
                        if !v4 {
                            kind = c.token()?;
                            let tags = (0..c.token::<usize>()?)
                                .map(|_| c.token())
                                .collect::<Result<Vec<i64>>>()?;
                            group = tags.first().filter(|&&p| p != 0).map(|&p| (2, p));
                        }
                        let nodes = (0..node_count(kind)?)
                            .map(|_| {
                                let t = c.size()?;
                                node_index.get(&t).copied().ok_or_else(|| {
                                    bad(format!("element {tag} has unknown node {t}"))
                                })
                            })
                            .collect::<Result<Vec<_>>>()?;
                        for triangle in element_triangles(kind, &nodes) {
                            element_rows.entry(tag).or_default().push(triangles.len());
                            triangles.push(triangle);
                            physical.push(group);
                        }
                    }
                }
            }
            "NodeData" => {
                let len = vertices.len();
                read_data(&mut c, &mut node_columns, len, |t| {
                    node_index
                        .get(&t)
                        .map(|&i| vec![i as usize])
                        .ok_or_else(|| bad(format!("data for unknown node {t}")))
                })?;
            }
            "ElementData" => {
                let len = triangles.len();
                read_data(&mut c, &mut face_columns, len, |t| {
                    Ok(element_rows.get(&t).cloned().unwrap_or_default())
                })?;
            }
            _ => {}
        }
        c.end(&section)?;
    }
    if v4.is_none() {
        return Err(bad("no $MeshFormat"));
    }
    let mut mesh = Mesh::new(vertices, triangles)?;
    if physical.iter().any(Option::is_some) {
        let groups: StringArray = physical
            .iter()
            .map(|g| g.map(|g| names.get(&g).cloned().unwrap_or_else(|| g.1.to_string())))
            .collect();
        mesh = mesh.with_face_column("physical", Arc::new(groups))?;
    }
    for (name, values) in node_columns {
        mesh = mesh.with_vertex_column(&name, column(values))?;
    }
    for (name, values) in face_columns {
        mesh = mesh.with_face_column(&name, column(values))?;
    }
    Ok(mesh)
}

struct Writer {
    out: Vec<u8>,
    binary: bool,
}

impl Writer {
    fn text(&mut self, s: impl Display) {
        self.out.extend(s.to_string().bytes());
    }

    fn value(&mut self, text: impl Display, bytes: &[u8]) {
        if self.binary {
            self.out.extend(bytes);
        } else {
            self.text(format_args!("{text} "));
        }
    }

    fn size(&mut self, v: usize) {
        self.value(v, &(v as u64).to_le_bytes());
    }

    fn int(&mut self, v: usize) {
        self.value(v, &(v as i32).to_le_bytes());
    }

    fn double(&mut self, v: f64) {
        self.value(v, &v.to_le_bytes());
    }

    fn newline(&mut self) {
        if !self.binary {
            self.out.pop();
            self.out.push(b'\n');
        }
    }

    fn end(&mut self, section: &str) {
        if self.binary {
            self.out.push(b'\n');
        }
        self.text(format_args!("$End{section}\n"));
    }

    fn data(&mut self, section: &str, name: &str, column: &ArrayRef) -> Result<()> {
        let values = arrow_cast::cast(column, &DataType::Float64)?;
        let values = values.as_primitive::<Float64Type>();
        let name = name.replace('"', "'");
        let count = values.len() - values.null_count();
        self.text(format_args!(
            "${section}\n1\n\"{name}\"\n1\n0\n3\n0\n1\n{count}\n"
        ));
        for (i, v) in values.iter().enumerate() {
            if let Some(v) = v {
                self.int(i + 1);
                self.double(v);
                self.newline();
            }
        }
        self.end(section);
        Ok(())
    }
}

/// MSH 4.1 with one surface entity per value of the face column `physical`,
/// its physical group, and numeric columns as node and element data.
pub(crate) fn msh_bytes(mesh: &Mesh, ascii: bool) -> Result<Vec<u8>> {
    let physical = crate::mesh::text_column(mesh, "physical")?;
    let mut names: Vec<&str> = Vec::new();
    let entity: Vec<usize> = (0..mesh.triangles().len())
        .map(|t| match physical.as_ref().filter(|p| p.is_valid(t)) {
            Some(p) => {
                let name = p.value(t);
                names.iter().position(|n| *n == name).unwrap_or_else(|| {
                    names.push(name);
                    names.len() - 1
                }) + 1
            }
            None => 0,
        })
        .collect();
    let unnamed = names.len() + 1;
    let entity: Vec<usize> = entity
        .into_iter()
        .map(|e| if e == 0 { unnamed } else { e })
        .collect();
    let mut w = Writer {
        out: Vec::new(),
        binary: false,
    };
    w.text(format_args!("$MeshFormat\n4.1 {} 8\n", u8::from(!ascii)));
    if !ascii {
        w.out.extend(1i32.to_le_bytes());
        w.binary = true;
    }
    w.end("MeshFormat");
    if !names.is_empty() {
        w.text(format_args!("$PhysicalNames\n{}\n", names.len()));
        for (i, name) in names.iter().enumerate() {
            w.text(format_args!("2 {} \"{}\"\n", i + 1, name.replace('"', "'")));
        }
        w.text("$EndPhysicalNames\n");
    }
    let (min, max) = mesh.bounds().unwrap_or_default();
    w.text("$Entities\n");
    for v in [0, 0, unnamed, 0] {
        w.size(v);
    }
    w.newline();
    for tag in 1..=unnamed {
        w.int(tag);
        min.iter().chain(&max).for_each(|&x| w.double(x));
        let named = usize::from(tag < unnamed);
        w.size(named);
        if named == 1 {
            w.int(tag);
        }
        w.size(0);
        w.newline();
    }
    w.end("Entities");
    let n = mesh.vertices().len();
    w.text("$Nodes\n");
    for v in [1, n, 1, n] {
        w.size(v);
    }
    w.newline();
    for v in [2, unnamed, 0] {
        w.int(v);
    }
    w.size(n);
    w.newline();
    for i in 1..=n {
        w.size(i);
        w.newline();
    }
    for p in mesh.vertices() {
        p.iter().for_each(|&x| w.double(x));
        w.newline();
    }
    w.end("Nodes");
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (t, &e) in entity.iter().enumerate() {
        match runs.last_mut() {
            Some(run) if entity[run.0] == e => run.1 = t + 1,
            _ => runs.push((t, t + 1)),
        }
    }
    let m = mesh.triangles().len();
    w.text("$Elements\n");
    for v in [runs.len(), m, 1, m] {
        w.size(v);
    }
    w.newline();
    for (start, stop) in runs {
        for v in [2, entity[start], 2] {
            w.int(v);
        }
        w.size(stop - start);
        w.newline();
        for t in start..stop {
            w.size(t + 1);
            for v in mesh.triangles()[t] {
                w.size(v as usize + 1);
            }
            w.newline();
        }
    }
    w.end("Elements");
    for (section, table) in [
        ("NodeData", mesh.vertex_attributes()),
        ("ElementData", mesh.face_attributes()),
    ] {
        for (field, column) in table.schema().fields().iter().zip(table.columns()) {
            if field.data_type().is_numeric() {
                w.data(section, field.name(), column)?;
            }
        }
    }
    Ok(w.out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(m: &Mesh, vertex: bool, name: &str) -> Vec<Option<f64>> {
        let table = if vertex {
            m.vertex_attributes()
        } else {
            m.face_attributes()
        };
        table
            .column_by_name(name)
            .unwrap()
            .as_primitive::<Float64Type>()
            .iter()
            .collect()
    }

    fn physical(m: &Mesh) -> Vec<Option<String>> {
        let column = m.face_attributes().column_by_name("physical").unwrap();
        column
            .as_string::<i32>()
            .iter()
            .map(|g| g.map(String::from))
            .collect()
    }

    #[test]
    fn msh_round_trip_keeps_groups_and_data() {
        let m = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.0, 1.0, 1e-7],
                [0.0, 0.0, 1.0],
            ],
            vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        )
        .unwrap()
        .with_face_column(
            "physical",
            Arc::new(StringArray::from(vec![
                Some("lid"),
                None,
                Some("side wall"),
                Some("lid"),
            ])),
        )
        .unwrap()
        .with_face_column(
            "w",
            Arc::new(arrow_array::Int32Array::from(vec![1, 2, 3, 4])),
        )
        .unwrap()
        .with_vertex_column(
            "grade",
            Arc::new(Float64Array::from(vec![
                Some(0.25),
                None,
                Some(2.0),
                Some(-1.0),
            ])),
        )
        .unwrap();
        for ascii in [true, false] {
            let back = parse_msh(&msh_bytes(&m, ascii).unwrap()).unwrap();
            assert_eq!(back.vertices(), m.vertices());
            assert_eq!(back.triangles(), m.triangles());
            assert_eq!(
                physical(&back),
                [
                    Some("lid".into()),
                    None,
                    Some("side wall".into()),
                    Some("lid".into())
                ]
            );
            assert_eq!(values(&back, false, "w"), [1.0, 2.0, 3.0, 4.0].map(Some));
            assert_eq!(
                values(&back, true, "grade"),
                [Some(0.25), None, Some(2.0), Some(-1.0)]
            );
        }
    }

    #[test]
    fn msh_2_2_with_quads_lines_and_vector_data() {
        let text = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n1\n2 7 \"top\"\n$EndPhysicalNames\n\
            $Nodes\n4\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n$EndNodes\n\
            $Elements\n3\n1 1 2 0 1 1 2\n2 3 2 7 1 1 2 3 4\n3 15 1 0 4\n$EndElements\n\
            $ElementData\n1\n\"v\"\n1\n0.5\n3\n0\n2\n1\n2 1.5 -2\n$EndElementData\n\
            $Comments\nanything\n$EndComments\n";
        let m = parse_msh(text.as_bytes()).unwrap();
        assert_eq!(m.triangles(), &[[0, 1, 2], [0, 2, 3]]);
        assert_eq!(physical(&m), [Some("top".into()), Some("top".into())]);
        assert_eq!(values(&m, false, "v_1"), [Some(-2.0); 2]);
        assert!(parse_msh(text.replace("1 2 3 4\n", "1 2 3 9\n").as_bytes()).is_err());
        assert!(parse_msh(b"$MeshFormat\n4.0 0 8\n$EndMeshFormat\n").is_err());
        assert!(parse_msh(b"garbage").is_err());
    }
}
