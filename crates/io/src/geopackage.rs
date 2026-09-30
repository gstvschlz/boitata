//! GeoPackage geometry blobs: the `GP` header, an optional envelope and ISO
//! WKB. The tables around them are read and written from Python.

use crate::shapefile::{area2, inside};
use crate::{Error, Result};

fn bad(message: impl Into<String>) -> Error {
    Error::GeoPackage(message.into())
}

/// Parts of one geometry and its WKB base type (1 point to 6 multipolygon).
/// Points are parts of one vertex; polygon rings are closed parts without the
/// repeated closing vertex. 2D geometries get z = 0, M values are dropped.
#[derive(Debug, Clone, PartialEq)]
pub struct Geometry {
    pub kind: u32,
    pub parts: Vec<(Vec<[f64; 3]>, bool)>,
}

struct Wkb<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Wkb<'_> {
    fn take<const N: usize>(&mut self, little: bool) -> Result<[u8; N]> {
        let mut b: [u8; N] = self
            .bytes
            .get(self.at..self.at + N)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| bad("geometry is truncated"))?;
        self.at += N;
        if !little {
            b.reverse();
        }
        Ok(b)
    }

    fn u32(&mut self, little: bool) -> Result<u32> {
        self.take(little).map(u32::from_le_bytes)
    }

    fn points(&mut self, little: bool, n: u32, z: bool, m: bool) -> Result<Vec<[f64; 3]>> {
        (0..n)
            .map(|_| {
                let mut f = || self.take(little).map(f64::from_le_bytes);
                let p = [f()?, f()?, if z { f()? } else { 0.0 }];
                if m {
                    f()?;
                }
                Ok(p)
            })
            .collect()
    }

    fn geometry(&mut self, parts: &mut Vec<(Vec<[f64; 3]>, bool)>) -> Result<u32> {
        let little = match self.take::<1>(true)? {
            [0] => false,
            [1] => true,
            _ => return Err(bad("invalid WKB byte order")),
        };
        let raw = self.u32(little)?;
        let code = raw & 0x0fff_ffff;
        let (kind, dims) = (code % 1000, code / 1000);
        let z = raw & 0x8000_0000 != 0 || dims == 1 || dims == 3;
        let m = raw & 0x4000_0000 != 0 || dims >= 2;
        match kind {
            1 => {
                let p = self.points(little, 1, z, m)?;
                if !(p[0][0].is_nan() && p[0][1].is_nan()) {
                    parts.push((p, false));
                }
            }
            2 => {
                let n = self.u32(little)?;
                parts.push((self.points(little, n, z, m)?, false));
            }
            3 => {
                for _ in 0..self.u32(little)? {
                    let n = self.u32(little)?;
                    let mut ring = self.points(little, n, z, m)?;
                    if ring.len() > 1 && ring.first() == ring.last() {
                        ring.pop();
                    }
                    parts.push((ring, true));
                }
            }
            4..=6 => {
                for _ in 0..self.u32(little)? {
                    if self.geometry(parts)? != kind - 3 {
                        return Err(bad("multi-geometry member of the wrong type"));
                    }
                }
            }
            _ => return Err(bad(format!("WKB geometry type {raw} is not supported"))),
        }
        Ok(kind)
    }
}

/// Decodes a GeoPackage geometry blob.
pub fn decode_geometry(blob: &[u8]) -> Result<Geometry> {
    if blob.len() < 8 || &blob[..2] != b"GP" {
        return Err(bad("not a GeoPackage geometry"));
    }
    let flags = blob[3];
    if flags & 0x20 != 0 {
        return Err(bad("extended geometry types are not supported"));
    }
    let envelope = match (flags >> 1) & 7 {
        0 => 0,
        1 => 32,
        2 | 3 => 48,
        4 => 64,
        _ => return Err(bad("invalid envelope code")),
    };
    let mut wkb = Wkb {
        bytes: blob,
        at: 8 + envelope,
    };
    let mut parts = vec![];
    let kind = wkb.geometry(&mut parts)?;
    Ok(Geometry { kind, parts })
}

fn header(srs_id: i32, vertices: &[[f64; 3]]) -> Vec<u8> {
    let mut b = b"GP\0".to_vec();
    if vertices.len() < 2 {
        b.push(1);
        b.extend(srs_id.to_le_bytes());
        return b;
    }
    b.push(1 | 2 << 1);
    b.extend(srs_id.to_le_bytes());
    for a in 0..3 {
        let values = vertices.iter().map(|p| p[a]);
        let lo = values.clone().fold(f64::INFINITY, f64::min);
        let hi = values.fold(f64::NEG_INFINITY, f64::max);
        b.extend(lo.to_le_bytes());
        b.extend(hi.to_le_bytes());
    }
    b
}

fn push_points(b: &mut Vec<u8>, points: &[[f64; 3]]) {
    b.extend(points.iter().flatten().flat_map(|v| v.to_le_bytes()));
}

/// A PointZ blob.
pub fn encode_point(p: [f64; 3], srs_id: i32) -> Vec<u8> {
    let mut b = header(srs_id, &[p]);
    b.push(1);
    b.extend(1001u32.to_le_bytes());
    push_points(&mut b, &[p]);
    b
}

/// A MultiLineStringZ blob of open `parts`, or a MultiPolygonZ one of rings.
/// Rings nest by containment: those inside an even number of others are
/// exteriors, wound counter-clockwise, and each other ring is a clockwise hole
/// of the innermost exterior holding it.
pub fn encode_parts(parts: &[&[[f64; 3]]], polygon: bool, srs_id: i32) -> Vec<u8> {
    let vertices: Vec<[f64; 3]> = parts.concat();
    let mut b = header(srs_id, &vertices);
    b.push(1);
    if !polygon {
        b.extend(1005u32.to_le_bytes());
        b.extend((parts.len() as u32).to_le_bytes());
        for p in parts {
            b.push(1);
            b.extend(1002u32.to_le_bytes());
            b.extend((p.len() as u32).to_le_bytes());
            push_points(&mut b, p);
        }
        return b;
    }
    let holders =
        |i: usize| (0..parts.len()).filter(move |&j| j != i && inside(parts[j], parts[i][0]));
    let depth: Vec<usize> = (0..parts.len()).map(|i| holders(i).count()).collect();
    let exteriors: Vec<usize> = (0..parts.len())
        .filter(|&i| depth[i].is_multiple_of(2))
        .collect();
    b.extend(1006u32.to_le_bytes());
    b.extend((exteriors.len() as u32).to_le_bytes());
    for &e in &exteriors {
        let holes = (0..parts.len()).filter(|&i| {
            depth[i] % 2 == 1 && holders(i).any(|j| j == e && depth[j] + 1 == depth[i])
        });
        let rings: Vec<usize> = [e].into_iter().chain(holes).collect();
        b.push(1);
        b.extend(1003u32.to_le_bytes());
        b.extend((rings.len() as u32).to_le_bytes());
        for r in rings {
            let mut ring = parts[r].to_vec();
            if (area2(&ring) > 0.0) != (r == e) {
                ring.reverse();
            }
            ring.push(ring[0]);
            b.extend((ring.len() as u32).to_le_bytes());
            push_points(&mut b, &ring);
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x: f64, s: f64) -> Vec<[f64; 3]> {
        vec![
            [x, x, 1.0],
            [x + s, x, 1.0],
            [x + s, x + s, 1.0],
            [x, x + s, 1.0],
        ]
    }

    #[test]
    fn round_trips() {
        let p = decode_geometry(&encode_point([1.0, 2.0, 3.0], 4326)).unwrap();
        assert_eq!(p.kind, 1);
        assert_eq!(p.parts, vec![(vec![[1.0, 2.0, 3.0]], false)]);

        let line = square(0.0, 1.0);
        let l = decode_geometry(&encode_parts(&[&line, &line[..2]], false, 0)).unwrap();
        assert_eq!(l.kind, 5);
        assert_eq!(
            l.parts,
            vec![(line.clone(), false), (line[..2].to_vec(), false)]
        );

        let (outer, hole, island) = (square(0.0, 10.0), square(2.0, 6.0), square(4.0, 2.0));
        let blob = encode_parts(&[&hole, &outer, &island], true, 0);
        let g = decode_geometry(&blob).unwrap();
        assert_eq!(g.kind, 6);
        let rings: Vec<_> = g.parts.iter().map(|(r, _)| r).collect();
        assert_eq!(rings.len(), 3);
        assert!(g.parts.iter().all(|(r, c)| *c && r.len() == 4));
        assert!(area2(rings[0]) > 0.0 && area2(rings[1]) < 0.0 && area2(rings[2]) > 0.0);
        assert!(rings[0].contains(&[0.0, 0.0, 1.0]) && rings[2].contains(&[4.0, 4.0, 1.0]));
    }

    #[test]
    fn reads_big_endian_2d_with_envelope() {
        let mut b = b"GP\0".to_vec();
        b.push(1 << 1);
        b.extend(3857i32.to_be_bytes());
        b.extend([0.0f64, 1.0, 0.0, 1.0].iter().flat_map(|v| v.to_be_bytes()));
        b.push(0);
        b.extend(3u32.to_be_bytes());
        b.extend(1u32.to_be_bytes());
        b.extend(4u32.to_be_bytes());
        for v in [0.0f64, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0] {
            b.extend(v.to_be_bytes());
        }
        let g = decode_geometry(&b).unwrap();
        assert_eq!(g.kind, 3);
        assert_eq!(
            g.parts,
            vec![(
                vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                true
            )]
        );
        assert!(decode_geometry(&b[..b.len() - 1]).is_err());
        assert!(decode_geometry(b"XP\0\0\0\0\0\0").is_err());
    }

    #[test]
    fn empty_point_has_no_parts() {
        let mut b = b"GP\0\x11".to_vec();
        b.extend(0i32.to_le_bytes());
        b.push(1);
        b.extend(3001u32.to_le_bytes());
        b.extend(
            [f64::NAN, f64::NAN, f64::NAN, 5.0]
                .iter()
                .flat_map(|v| v.to_le_bytes()),
        );
        let g = decode_geometry(&b).unwrap();
        assert_eq!((g.kind, g.parts.len()), (1, 0));
    }
}
