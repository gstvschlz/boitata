use ceres_core::{BlockModel, Layout, Mesh};

use crate::{BlockModelError, Result};

/// Surface through the cell centres of a 2D block model, each lifted to its
/// `elevation` (one per row). Four neighbouring centres make two triangles,
/// three make one; rows without an elevation leave holes. Triangles wind
/// counter-clockwise in the grid's axes, so they face up in an unrotated grid.
pub fn grid_surface(model: &BlockModel, elevation: &[Option<f64>]) -> Result<Mesh> {
    let invalid = |m: &str| Err(BlockModelError::InvalidMesh(m.to_string()));
    let geometry = model.geometry();
    if geometry.count[2] != 1 || matches!(model.layout(), Layout::SubBlocked { .. }) {
        return invalid("needs a regular or masked 2D block model");
    }
    if elevation.len() != model.len() || elevation.len() >= u32::MAX as usize {
        return invalid("one elevation per block");
    }
    let [nx, ny, _] = geometry.count;
    let mut node = vec![u32::MAX; nx * ny];
    let mut vertices = vec![];
    for (row, z) in elevation.iter().enumerate() {
        if let Some(z) = *z {
            let cell = model.parent_index(row);
            let [x, y, _] = geometry.centroid(cell);
            node[cell as usize] = vertices.len() as u32;
            vertices.push([x, y, z]);
        }
    }
    let mut triangles = vec![];
    for j in 1..ny {
        for i in 1..nx {
            let corners = [(i - 1, j - 1), (i, j - 1), (i, j), (i - 1, j)]
                .map(|(i, j)| node[j * nx + i])
                .into_iter()
                .filter(|&v| v != u32::MAX)
                .collect::<Vec<_>>();
            match corners[..] {
                [a, b, c, d] => triangles.extend([[a, b, c], [a, c, d]]),
                [a, b, c] => triangles.push([a, b, c]),
                _ => {}
            }
        }
    }
    let mut mesh = Mesh::new(vertices, triangles)?;
    mesh.crs = model.crs.clone();
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vertical_distance;
    use arrow_array::{ArrayRef, Float64Array};
    use ceres_core::{Geometry, RecordBatch};
    use std::sync::Arc;

    fn grid(count: [usize; 3], index: Option<Vec<u64>>) -> BlockModel {
        let geometry = Geometry {
            origin: [100.0, 200.0, 0.0],
            size: [10.0, 5.0, 1.0],
            count,
            rotation: [0.0; 3],
        };
        let rows = index.as_ref().map_or(geometry.cells() as usize, Vec::len);
        let column: ArrayRef = Arc::new(Float64Array::from(vec![0.0; rows]));
        let batch = RecordBatch::try_from_iter([("a", column)]).unwrap();
        match index {
            None => BlockModel::regular(geometry, batch).unwrap(),
            Some(index) => BlockModel::masked(geometry, index, batch).unwrap(),
        }
    }

    /// Theory check: a tilted plane on a full grid is exact everywhere under
    /// the centres, and its area is the plan area between them over cos(slope).
    #[test]
    fn planar_grid_gives_the_plane() {
        let model = grid([8, 6, 1], None);
        let plane = |x: f64, y: f64| 0.2 * x - 0.1 * y + 50.0;
        let z: Vec<Option<f64>> = model
            .centroids()
            .iter()
            .map(|c| Some(plane(c[0], c[1])))
            .collect();
        let mesh = grid_surface(&model, &z).unwrap();
        assert_eq!(mesh.triangles().len(), 2 * 7 * 5);
        let plan = 70.0 * 25.0;
        let tilt = (1.0f64 + 0.2 * 0.2 + 0.1 * 0.1).sqrt();
        assert!((mesh.area() - plan * tilt).abs() < 1e-9);
        let points: Vec<_> = (0..200)
            .map(|k| {
                let (x, y) = (
                    105.0 + (k as f64 * 7.3) % 70.0,
                    202.5 + (k as f64 * 3.1) % 25.0,
                );
                (x, y, plane(x, y))
            })
            .collect();
        let d = vertical_distance(&mesh, &points).unwrap();
        assert!(d.iter().all(|d| d.abs() < 1e-9));
        assert!(mesh.triangles().iter().all(|&t| {
            let [a, b, c] = t.map(|i| mesh.vertices()[i as usize]);
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0.0
        }));
    }

    #[test]
    fn missing_cells_leave_holes() {
        let model = grid([4, 4, 1], None);
        let mut z = vec![Some(0.0); 16];
        z[5] = None;
        let mesh = grid_surface(&model, &z).unwrap();
        assert_eq!(mesh.triangles().len(), 18 - 4);
        assert!((mesh.area() - (9.0 - 2.0) * 50.0).abs() < 1e-9);
        let d = vertical_distance(&mesh, &[(117.0, 209.0, 1.0), (130.0, 215.0, 1.0)]).unwrap();
        assert!(d[0].is_nan() && (d[1] - 1.0).abs() < 1e-12);

        let masked = grid([4, 4, 1], Some((0..16).filter(|&i| i != 5).collect()));
        let same = grid_surface(&masked, &[Some(0.0); 15]).unwrap();
        assert_eq!(same.triangles().len(), mesh.triangles().len());
        assert!(grid_surface(&grid([2, 2, 2], None), &[Some(0.0); 8]).is_err());
        assert!(grid_surface(&model, &[Some(0.0); 3]).is_err());
    }
}
