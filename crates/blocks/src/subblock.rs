//! Sub-blocked models from prioritized meshes, and block proportions inside a
//! solid for rotated and sub-blocked models.

use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchOptions, StringArray, UInt64Array};
use arrow_select::take::take;
use ceres_core::{BlockModel, Geometry, Layout};
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;

use crate::{Aabb, BlockModelError, PolygonSelector, Result, SolidTester, Surface};

const WHOLE: [f64; 6] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

/// Where a domain lies: inside a closed solid, inside a vertical prism over
/// plan polygons, or below or above a surface along world z. Off the
/// surface's footprint, neither.
#[derive(Clone)]
pub enum Region {
    Inside(SolidTester),
    Prism(PolygonSelector),
    Below(Surface),
    Above(Surface),
}

#[derive(Clone)]
pub struct Domain {
    pub region: Region,
    pub label: String,
}

fn extent(model: &BlockModel, row: usize) -> [f64; 6] {
    match model.layout() {
        Layout::SubBlocked { extent, .. } => extent[row],
        _ => WHOLE,
    }
}

/// A grid with its rotation evaluated once; `point` is [`Geometry::point`].
struct Frame<'a> {
    g: &'a Geometry,
    axes: Matrix3<f64>,
}

impl<'a> Frame<'a> {
    fn new(g: &'a Geometry) -> Self {
        let axes = ceres_core::block_frame(g.rotation).transpose();
        Self { g, axes }
    }

    fn point(&self, cell: u64, at: [f64; 3]) -> [f64; 3] {
        let ijk = self.g.ijk(cell);
        let local = Vector3::from_fn(|a, _| (ijk[a] as f64 + at[a]) * self.g.size[a]);
        let world = self.axes * local;
        [0, 1, 2].map(|a| self.g.origin[a] + world[a])
    }

    /// World bounds of the part `e` of cell `cell`.
    fn world_box(&self, cell: u64, e: [f64; 6]) -> Aabb {
        let corners = (0..8).map(|c| self.point(cell, [0, 1, 2].map(|a| e[a + 3 * (c >> a & 1)])));
        Aabb::of_points(corners).expect("eight corners")
    }
}

/// Sub-blocks every row of a regular or masked `model` on a `subgrid` per
/// cell. Each sub-cell takes the label of the first domain holding its center;
/// sub-cells no domain holds take `fill`, or are dropped without it. Cells of
/// one label stay whole; the others merge runs of sub-cells along x, then y.
/// The labels are the string column `column`; other columns are the parent's.
pub fn subblock(
    model: &BlockModel,
    domains: &[Domain],
    subgrid: [u32; 3],
    column: &str,
    fill: Option<&str>,
) -> Result<BlockModel> {
    if matches!(model.layout(), Layout::SubBlocked { .. }) {
        return Err(BlockModelError::InvalidGridParams(
            "model is already sub-blocked".into(),
        ));
    }
    if subgrid.contains(&0) {
        return Err(BlockModelError::InvalidGridParams(
            "sub-grid counts must be positive".into(),
        ));
    }
    let mut labels: Vec<String> = vec![];
    let mut id = |label: &str| match labels.iter().position(|l| *l == label) {
        Some(i) => i,
        None => {
            labels.push(label.to_string());
            labels.len() - 1
        }
    };
    let ids: Vec<usize> = domains.iter().map(|d| id(&d.label)).collect();
    let fill = fill.map(id);
    let g = *model.geometry();
    let n = subgrid.map(|v| v as usize);
    let frame = Frame::new(&g);
    let pieces: Vec<Vec<([f64; 6], usize)>> = (0..model.len())
        .into_par_iter()
        .map(|row| pieces(&frame, model.parent_index(row), n, domains, &ids, fill))
        .collect();
    let (mut parent, mut extents, mut owner, mut label) = (vec![], vec![], vec![], vec![]);
    for (row, found) in pieces.into_iter().enumerate() {
        for (e, l) in found {
            parent.push(model.parent_index(row));
            extents.push(e);
            owner.push(row as u64);
            label.push(labels[l].as_str());
        }
    }
    let owner = UInt64Array::from(owner);
    let columns = model
        .attributes()
        .columns()
        .iter()
        .map(|c| take(c, &owner, None))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(ceres_core::Error::from)?;
    let attributes = RecordBatch::try_new_with_options(
        model.attributes().schema(),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(owner.len())),
    )
    .map_err(ceres_core::Error::from)?;
    let mut out = BlockModel::subblocked(g, parent, extents, Some(subgrid), attributes)?
        .with_column(column, Arc::new(StringArray::from(label)))?;
    out.crs.clone_from(&model.crs);
    Ok(out)
}

fn pieces(
    frame: &Frame,
    cell: u64,
    n: [usize; 3],
    domains: &[Domain],
    ids: &[usize],
    fill: Option<usize>,
) -> Vec<([f64; 6], usize)> {
    let at = |s: usize| [s % n[0], s / n[0] % n[1], s / (n[0] * n[1])];
    let center = |s: usize| {
        let ijk = at(s);
        frame.point(cell, [0, 1, 2].map(|a| (ijk[a] as f64 + 0.5) / n[a] as f64))
    };
    let bounds = frame.world_box(cell, WHOLE);
    let flat = frame.g.rotation[1] == 0.0 && frame.g.rotation[2] == 0.0;
    let mut label: Vec<Option<usize>> = vec![None; n.iter().product()];
    for (domain, &id) in domains.iter().zip(ids) {
        if label.iter().all(Option::is_some) {
            break;
        }
        match &domain.region {
            Region::Inside(solid) => {
                if !solid.bounds().overlaps(&bounds) {
                    continue;
                }
                if !solid.surface_may_cut(&bounds) {
                    if solid.contains(frame.point(cell, [0.5; 3])) {
                        label.iter_mut().for_each(|l| *l = l.or(Some(id)));
                    }
                    continue;
                }
                for s in 0..label.len() {
                    if label[s].is_none() && solid.contains(center(s)) {
                        label[s] = Some(id);
                    }
                }
            }
            Region::Prism(prism) => {
                let (lo, hi) = prism.bounds();
                if (0..2).any(|a| bounds.max[a] < lo[a] || bounds.min[a] > hi[a]) {
                    continue;
                }
                for s in 0..label.len() {
                    if label[s].is_none() && prism.contains(center(s)) {
                        label[s] = Some(id);
                    }
                }
            }
            Region::Below(surface) | Region::Above(surface) => {
                let (lo, hi) = surface.bounds();
                if (0..2).any(|a| bounds.max[a] < lo[a] || bounds.min[a] > hi[a]) {
                    continue;
                }
                let below = matches!(domain.region, Region::Below(_));
                let mut column: Vec<Option<Option<f64>>> = vec![None; n[0] * n[1]];
                for s in 0..label.len() {
                    if label[s].is_some() {
                        continue;
                    }
                    let p = center(s);
                    let lookup = || surface.elevation(p[0], p[1]);
                    let elevation = if flat {
                        *column[s % (n[0] * n[1])].get_or_insert_with(lookup)
                    } else {
                        lookup()
                    };
                    if elevation.is_some_and(|e| if below { p[2] < e } else { p[2] > e }) {
                        label[s] = Some(id);
                    }
                }
            }
        }
    }
    if let Some(f) = fill {
        label.iter_mut().for_each(|l| *l = l.or(Some(f)));
    }
    if label.iter().all(|l| *l == label[0]) {
        return label[0].map(|l| vec![(WHOLE, l)]).unwrap_or_default();
    }
    let mut out = vec![];
    for k in 0..n[2] {
        let mut boxes: Vec<[usize; 5]> = vec![];
        for j in 0..n[1] {
            let mut i = 0;
            while i < n[0] {
                let l = label[i + n[0] * (j + n[1] * k)];
                let mut end = i + 1;
                while end < n[0] && label[end + n[0] * (j + n[1] * k)] == l {
                    end += 1;
                }
                if let Some(l) = l {
                    match boxes
                        .iter_mut()
                        .find(|b| b[0] == i && b[1] == end && b[3] == j && b[4] == l)
                    {
                        Some(b) => b[3] = j + 1,
                        None => boxes.push([i, end, j, j + 1, l]),
                    }
                }
                i = end;
            }
        }
        let f = |v: usize, a: usize| v as f64 / n[a] as f64;
        out.extend(boxes.into_iter().map(|[i0, i1, j0, j1, l]| {
            (
                [f(i0, 0), f(j0, 1), f(k, 2), f(i1, 0), f(j1, 1), f(k + 1, 2)],
                l,
            )
        }));
    }
    out
}

/// Share of each row of `model` inside `solid`, from `discretization` points
/// per axis in each row where the surface may cut it; exact elsewhere.
pub fn proportions(
    solid: &SolidTester,
    model: &BlockModel,
    discretization: [usize; 3],
) -> Vec<f64> {
    let frame = Frame::new(model.geometry());
    let d = discretization.map(|n| n.max(1));
    let n = d.iter().product::<usize>();
    (0..model.len())
        .into_par_iter()
        .map(|row| {
            let (cell, e) = (model.parent_index(row), extent(model, row));
            let bounds = frame.world_box(cell, e);
            let point =
                |t: [f64; 3]| frame.point(cell, [0, 1, 2].map(|a| e[a] + (e[a + 3] - e[a]) * t[a]));
            if !solid.bounds().overlaps(&bounds) {
                0.0
            } else if !solid.surface_may_cut(&bounds) {
                f64::from(u8::from(solid.contains(point([0.5; 3]))))
            } else {
                let step = |i: usize, a: usize| (i as f64 + 0.5) / d[a] as f64;
                let inside = (0..n)
                    .filter(|s| {
                        solid.contains(point([
                            step(s % d[0], 0),
                            step(s / d[0] % d[1], 1),
                            step(s / (d[0] * d[1]), 2),
                        ]))
                    })
                    .count();
                inside as f64 / n as f64
            }
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::convex_hull;
    use crate::solid::tests::cube;
    use arrow_array::cast::AsArray;
    use arrow_array::{ArrayRef, Float64Array};
    use ceres_core::Mesh;

    /// `count` cells a side of `size`, centered on the origin.
    fn grid(size: f64, count: usize, rotation: [f64; 3]) -> BlockModel {
        let half = ceres_core::block_frame(rotation).transpose()
            * nalgebra::Vector3::repeat(size * count as f64 / 2.0);
        let g = Geometry {
            origin: [-half.x, -half.y, -half.z],
            size: [size; 3],
            count: [count; 3],
            rotation,
        };
        let cells = g.cells() as usize;
        let value = Float64Array::from((0..cells).map(|c| c as f64).collect::<Vec<_>>());
        let attributes = RecordBatch::try_from_iter([("cell", Arc::new(value) as ArrayRef)]);
        BlockModel::regular(g, attributes.unwrap()).unwrap()
    }

    pub(crate) fn sphere(radius: f64) -> Mesh {
        let n = 200;
        let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        let points: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
                let r = (1.0 - z * z).sqrt();
                let t = golden * i as f64;
                [radius * r * t.cos(), radius * r * t.sin(), radius * z]
            })
            .collect();
        convex_hull(&points).unwrap()
    }

    fn inside(mesh: &Mesh, label: &str) -> Domain {
        Domain {
            region: Region::Inside(SolidTester::new(mesh).unwrap()),
            label: label.into(),
        }
    }

    fn labels(model: &BlockModel, column: &str) -> Vec<String> {
        let c = model.attributes().column_by_name(column).unwrap();
        c.as_string::<i32>()
            .iter()
            .map(|l| l.unwrap().to_string())
            .collect()
    }

    fn volume_of(model: &BlockModel, label: &str) -> f64 {
        labels(model, "domain")
            .iter()
            .zip(model.volumes())
            .filter(|(l, _)| *l == label)
            .map(|(_, v)| v)
            .sum()
    }

    /// Theory check: sub-blocks inside a closed mesh approach its volume as
    /// the sub-cell shrinks, on a rotated grid.
    #[test]
    fn subblocks_converge_to_the_mesh_volume() {
        let ball = sphere(9.0);
        let exact = ball.volume().unwrap();
        let model = grid(4.0, 6, [30.0, 20.0, 10.0]);
        let errors: Vec<f64> = [1, 2, 4, 8]
            .map(|n| {
                let sub =
                    subblock(&model, &[inside(&ball, "ore")], [n; 3], "domain", None).unwrap();
                (sub.volumes().iter().sum::<f64>() / exact - 1.0).abs()
            })
            .to_vec();
        assert!(
            errors[3] < 0.01 && errors[3] < errors[0] / 4.0,
            "{errors:?}"
        );
    }

    /// Theory check: a prism over a 6 m square with a 2 m square hole holds
    /// (36 - 4) m² times the model height.
    #[test]
    fn a_prism_with_a_hole() {
        let ring = |h: f64| [[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]];
        let (outer, inner) = (ring(3.0), ring(1.0));
        let prism = PolygonSelector::new(&[&outer, &inner], true, None, None).unwrap();
        let domain = Domain {
            region: Region::Prism(prism),
            label: "lease".into(),
        };
        let sub = subblock(
            &grid(1.0, 8, [0.0; 3]),
            &[domain],
            [2, 2, 1],
            "domain",
            None,
        )
        .unwrap();
        assert!((volume_of(&sub, "lease") - 32.0 * 8.0).abs() < 1e-9);
    }

    /// `z = 0.25 x + 0.1 y - 1` over `x` in [-4, 2], `y` in [-4, 4].
    fn plane() -> Surface {
        let at = |x: f64, y: f64| [x, y, 0.25 * x + 0.1 * y - 1.0];
        let mut vertices = vec![];
        for j in 0..=8 {
            for i in 0..=6 {
                vertices.push(at(i as f64 - 4.0, j as f64 - 4.0));
            }
        }
        let mut triangles = vec![];
        for j in 0..8u32 {
            for i in 0..6u32 {
                let v = j * 7 + i;
                triangles.push([v, v + 1, v + 8]);
                triangles.push([v, v + 8, v + 7]);
            }
        }
        Surface::new(&Mesh::new(vertices, triangles).unwrap()).unwrap()
    }

    /// Below and above a surface split its footprint; off it, `fill` takes
    /// the rest, so the volumes add up to the parents'. Priority decides
    /// overlaps and parents keep their columns.
    #[test]
    fn surfaces_priority_and_fill_partition_the_grid() {
        let model = grid(1.0, 8, [0.0; 3]);
        let domains = [
            Domain {
                region: Region::Below(plane()),
                label: "rock".into(),
            },
            Domain {
                region: Region::Above(plane()),
                label: "air".into(),
            },
        ];
        let sub = subblock(&model, &domains, [4, 4, 4], "domain", Some("outside")).unwrap();
        let total: f64 = sub.volumes().iter().sum();
        assert!((total - 512.0).abs() < 1e-9);
        assert!((volume_of(&sub, "rock") - 132.0).abs() < 0.01 * 132.0);
        assert!((volume_of(&sub, "outside") - 128.0).abs() < 1e-9);
        let cells = sub.attributes().column_by_name("cell").unwrap();
        let cells = cells.as_primitive::<arrow_array::types::Float64Type>();
        for (row, c) in cells.values().iter().enumerate() {
            assert_eq!(*c as u64, sub.parent_index(row));
        }

        let ranked = [inside(&cube(-2.0, 2.0), "a"), inside(&cube(0.0, 3.0), "b")];
        let sub = subblock(&model, &ranked, [2, 2, 2], "domain", None).unwrap();
        assert!((volume_of(&sub, "a") - 64.0).abs() < 1e-9);
        assert!((volume_of(&sub, "b") - 19.0).abs() < 1e-9);
        assert!(subblock(&sub, &ranked, [2, 2, 2], "domain", None).is_err());
        assert!(subblock(&model, &ranked, [2, 0, 2], "domain", None).is_err());
    }

    /// The binned tester gives the same sub-blocks, in the same order, and
    /// the same proportions as the full winding sum, including sub-cell
    /// centers lying on faces.
    #[test]
    fn binned_solids_match_the_winding_sum() {
        let cases = [
            (grid(4.0, 6, [30.0, 20.0, 10.0]), sphere(9.0), [4; 3]),
            (grid(1.0, 8, [0.0; 3]), cube(-2.25, 1.75), [2; 3]),
            (grid(1.0, 8, [0.0; 3]), cube(-1.5, 1.5), [1, 2, 3]),
        ];
        for (model, mesh, n) in &cases {
            let fast = SolidTester::new(mesh).unwrap();
            let exact = fast.clone().unbinned();
            let run = |solid: &SolidTester| {
                let domains = [Domain {
                    region: Region::Inside(solid.clone()),
                    label: "ore".into(),
                }];
                subblock(model, &domains, *n, "domain", Some("waste")).unwrap()
            };
            let (a, b) = (run(&fast), run(&exact));
            assert_eq!(a.layout(), b.layout());
            assert_eq!(labels(&a, "domain"), labels(&b, "domain"));
            assert_eq!(
                proportions(&fast, model, [3; 3]),
                proportions(&exact, model, [3; 3])
            );
        }
    }

    #[test]
    fn subblocks_do_not_depend_on_the_thread_count() {
        let model = grid(4.0, 6, [30.0, 20.0, 10.0]);
        let domains = [
            inside(&sphere(9.0), "ore"),
            inside(&cube(-5.0, 5.0), "halo"),
        ];
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| subblock(&model, &domains, [2, 2, 1], "domain", Some("waste")))
                .unwrap()
        };
        let (one, many) = (run(1), run(4));
        assert_eq!(one.layout(), many.layout());
        assert_eq!(labels(&one, "domain"), labels(&many, "domain"));
    }

    /// Proportions follow the model's rotation and each sub-block's extent.
    #[test]
    fn proportions_of_rotated_and_subblocked_models() {
        let ball = sphere(9.0);
        let exact = ball.volume().unwrap();
        let solid = SolidTester::new(&ball).unwrap();
        let model = grid(4.0, 6, [30.0, 20.0, 10.0]);
        let sub = subblock(
            &model,
            &[inside(&ball, "ore")],
            [2; 3],
            "domain",
            Some("waste"),
        )
        .unwrap();
        for m in [&model, &sub] {
            let v: f64 = proportions(&solid, m, [4; 3])
                .iter()
                .zip(m.volumes())
                .map(|(p, v)| p * v)
                .sum();
            assert!((v / exact - 1.0).abs() < 0.02, "{v} vs {exact}");
        }
    }
}
