use blocks::{
    DomainMethod, Mesh as CoreMesh, Orientation, PolygonSelector as CoreSelector, ShellBlock,
    ShellFilter, ShellLimits, SolidTester, extract_shell,
};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;

use crate::args::{Point, array1, finite, points, points_array, rows, same_length};
use crate::containers::PyBlockModel;
use crate::estimation::targets;
use crate::invalid;

fn err(e: blocks::BlockModelError) -> PyErr {
    invalid(e)
}

fn bools<'py>(py: Python<'py>, values: Vec<bool>) -> Bound<'py, PyAny> {
    PyArray1::from_vec(py, values).into_any()
}

/// Closed triangle mesh (a solid or wireframe).
#[pyclass(module = "ceres", name = "Mesh", frozen)]
pub struct Mesh {
    mesh: CoreMesh,
    tester: SolidTester,
}

impl Mesh {
    pub fn parts(&self) -> (&[Point], &[(usize, usize, usize)]) {
        (&self.mesh.vertices, &self.mesh.triangles)
    }
}

#[pymethods]
impl Mesh {
    /// `vertices` is `(n, 3)`, `triangles` `(m, 3)` vertex indices.
    #[new]
    fn new(vertices: &Bound<PyAny>, triangles: &Bound<PyAny>) -> PyResult<Self> {
        let vertices = points(vertices)?;
        let triangles: PyReadonlyArray2<i64> = triangles
            .py()
            .import("numpy")?
            .call_method1("asarray", (triangles, "int64"))?
            .extract()
            .map_err(|_| invalid("triangles must be an (m, 3) integer array"))?;
        let triangles = triangles
            .as_array()
            .rows()
            .into_iter()
            .map(|r| match r.as_slice() {
                Some(&[a, b, c]) if a >= 0 && b >= 0 && c >= 0 => {
                    Ok((a as usize, b as usize, c as usize))
                }
                _ => Err(invalid("triangles must be (m, 3) non-negative indices")),
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mesh = CoreMesh {
            vertices,
            triangles,
        };
        mesh.validate().map_err(err)?;
        let tester = SolidTester::new(&mesh).map_err(err)?;
        Ok(Self { mesh, tester })
    }

    #[getter]
    fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        points_array(py, &self.mesh.vertices).into_any()
    }

    #[getter]
    fn triangles<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        let flat: Vec<i64> = self
            .mesh
            .triangles
            .iter()
            .flat_map(|t| [t.0 as i64, t.1 as i64, t.2 as i64])
            .collect();
        Array2::from_shape_vec((self.mesh.triangles.len(), 3), flat)
            .expect("m x 3")
            .into_pyarray(py)
            .into_any()
    }

    /// `(min, max)` corners of the bounding box.
    #[getter]
    fn bounds(&self) -> ([f64; 3], [f64; 3]) {
        let b = self.tester.bounds();
        (b.min, b.max)
    }

    /// Whether each point is inside, by generalized winding number.
    fn contains<'py>(&self, py: Python<'py>, points: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let pts = self::points(points)?;
        let inside = py.detach(|| {
            pts.par_iter()
                .map(|p| self.tester.contains([p.0, p.1, p.2]))
                .collect()
        });
        Ok(bools(py, inside))
    }

    fn winding_number<'py>(
        &self,
        py: Python<'py>,
        points: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let pts = self::points(points)?;
        let w = py.detach(|| {
            pts.par_iter()
                .map(|p| self.tester.winding_number([p.0, p.1, p.2]))
                .collect()
        });
        Ok(array1(py, w).into_any())
    }

    /// Distance to the surface; `signed` makes inside points negative.
    #[pyo3(signature = (points, signed=false))]
    fn distance<'py>(
        &self,
        py: Python<'py>,
        points: &Bound<PyAny>,
        signed: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let pts = self::points(points)?;
        let d = py
            .detach(|| {
                pts.par_iter()
                    .map(|p| {
                        if signed {
                            self.mesh.signed_distance_to(p)
                        } else {
                            self.mesh.distance_to(p)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .map_err(err)?;
        Ok(array1(py, d).into_any())
    }

    /// Proportion of each block inside, from `discretization`³ points per
    /// block; `blocks` is a BlockModel or centroids with a shared `size`.
    #[pyo3(signature = (blocks, size=None, discretization=4))]
    fn proportion<'py>(
        &self,
        py: Python<'py>,
        blocks: &Bound<PyAny>,
        size: Option<[f64; 3]>,
        discretization: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let size = match (size, blocks.cast::<PyBlockModel>()) {
            (Some(s), _) => s,
            (None, Ok(b)) => b.get().0.geometry().size,
            (None, Err(_)) => return Err(invalid("give size for plain centroids")),
        };
        let centers = targets(blocks)?;
        let p = py.detach(|| {
            centers
                .par_iter()
                .map(|c| {
                    self.tester
                        .evaluate_block([c.0, c.1, c.2], size, discretization)
                        .proportion
                })
                .collect()
        });
        Ok(array1(py, p).into_any())
    }

    fn __repr__(&self) -> String {
        format!(
            "Mesh({} vertices, {} triangles)",
            self.mesh.vertices.len(),
            self.mesh.triangles.len()
        )
    }
}

/// Selects points inside any plan-view ring, optionally between `z_min` and
/// `z_max`. Rings are `(n, 3)` polylines; `closed=True` means they close
/// implicitly.
#[pyclass(module = "ceres", name = "PolygonSelector", frozen)]
pub struct PolygonSelector(CoreSelector);

#[pymethods]
impl PolygonSelector {
    #[new]
    #[pyo3(signature = (rings, closed=false, z_min=None, z_max=None))]
    fn new(
        rings: Vec<Bound<PyAny>>,
        closed: bool,
        z_min: Option<f64>,
        z_max: Option<f64>,
    ) -> PyResult<Self> {
        let rings: Vec<Vec<[f64; 3]>> = rings
            .iter()
            .map(|r| Ok(points(r)?.into_iter().map(|p| [p.0, p.1, p.2]).collect()))
            .collect::<PyResult<_>>()?;
        let refs: Vec<&[[f64; 3]]> = rings.iter().map(Vec::as_slice).collect();
        Ok(Self(
            CoreSelector::new(&refs, closed, z_min, z_max).map_err(err)?,
        ))
    }

    fn contains<'py>(&self, py: Python<'py>, points: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let inside = self::points(points)?
            .iter()
            .map(|p| self.0.contains([p.0, p.1, p.2]))
            .collect();
        Ok(bools(py, inside))
    }
}

fn polygon(obj: &Bound<PyAny>) -> PyResult<Vec<(f64, f64)>> {
    rows(obj, "polygon")?
        .into_iter()
        .map(|r| match r[..] {
            [x, y, ..] => Ok((x, y)),
            _ => Err(invalid("polygon must be (n, 2)")),
        })
        .collect()
}

/// Whether each `(x, y)` point is inside `polygon` (even-odd rule).
#[pyfunction]
fn point_in_polygon<'py>(
    py: Python<'py>,
    points: &Bound<PyAny>,
    polygon: &Bound<PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let ring = self::polygon(polygon)?;
    let inside = self::points(points)?
        .iter()
        .map(|p| blocks::point_in_polygon((p.0, p.1), &ring))
        .collect();
    Ok(bools(py, inside))
}

/// Plan distance from each point to the polygon boundary; `signed` makes
/// inside points negative.
#[pyfunction]
#[pyo3(signature = (points, polygon, signed=false))]
fn polygon_distance<'py>(
    py: Python<'py>,
    points: &Bound<PyAny>,
    polygon: &Bound<PyAny>,
    signed: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let ring = self::polygon(polygon)?;
    let d = self::points(points)?
        .iter()
        .map(|p| {
            if signed {
                blocks::polygon_signed_distance((p.0, p.1), &ring)
            } else {
                blocks::polygon_distance((p.0, p.1), &ring)
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)?;
    Ok(array1(py, d).into_any())
}

/// Domain of each target from labelled samples (`nearest` or `majority`), or
/// inside/outside a `mesh` (`solid`). Returns labels and confidences.
#[pyfunction]
#[pyo3(signature = (targets, coords=None, domains=None, method="nearest", mesh=None))]
fn assign_domain<'py>(
    py: Python<'py>,
    targets: &Bound<PyAny>,
    coords: Option<&Bound<PyAny>>,
    domains: Option<Vec<String>>,
    method: &str,
    mesh: Option<PyRef<Mesh>>,
) -> PyResult<Bound<'py, PyTuple>> {
    let method = match method {
        "nearest" => DomainMethod::Nearest,
        "majority" => DomainMethod::MajorityVote,
        "solid" => DomainMethod::PointInSolid,
        other => return Err(invalid(format!("unknown method {other:?}"))),
    };
    let samples: Vec<(Point, String)> = match (coords, domains) {
        (Some(c), Some(d)) => {
            let c = points(c)?;
            same_length(c.len(), d.len(), "domains")?;
            c.into_iter().zip(d).collect()
        }
        (None, None) => vec![],
        _ => return Err(invalid("give both coords and domains")),
    };
    let mesh = mesh.as_ref().map(|m| &m.mesh);
    let out = self::targets(targets)?
        .iter()
        .map(|t| blocks::assign_domain(t, &samples, method, mesh))
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)?;
    let labels: Vec<String> = out.iter().map(|a| a.domain.clone()).collect();
    let confidence = array1(py, out.iter().map(|a| a.confidence).collect());
    PyTuple::new(
        py,
        [labels.into_pyobject(py)?.into_any(), confidence.into_any()],
    )
}

/// Visible skin of a block model: `(vertices (n, 3), triangles (m, 3),
/// values per vertex or None)`; `column` colours faces by an attribute.
#[pyfunction]
#[pyo3(signature = (model, column=None))]
fn block_shell<'py>(
    py: Python<'py>,
    model: PyRef<PyBlockModel>,
    column: Option<&str>,
) -> PyResult<Bound<'py, PyTuple>> {
    let m = &model.0;
    let size = m.geometry().size;
    let centroids = m.centroids();
    let values = match column {
        Some(c) => Some(finite(
            &crate::table::column(py, m.attributes(), c)?,
            "column",
        )?),
        None => None,
    };
    let frame = ceres_core::block_frame(m.geometry().rotation);
    let axis = |i: usize| [frame[(i, 0)], frame[(i, 1)], frame[(i, 2)]];
    let orientation = Orientation::from_axes(axis(0), axis(1), axis(2)).map_err(err)?;
    let filter = ShellFilter {
        slabs: vec![],
        drop_nan_values: false,
        keep: None,
    };
    let shell = extract_shell(
        || {
            centroids
                .iter()
                .map(move |c| ShellBlock { centroid: *c, size })
        },
        values.as_deref(),
        &filter,
        &ShellLimits::default(),
        &orientation,
    )
    .map_err(err)?;
    let vertices = Array2::from_shape_vec((shell.vertices.len() / 3, 3), shell.vertices)
        .expect("xyz")
        .into_pyarray(py)
        .into_any();
    let triangles: Vec<i64> = shell.triangles.iter().map(|&t| t as i64).collect();
    let triangles = Array2::from_shape_vec((triangles.len() / 3, 3), triangles)
        .expect("ijk")
        .into_pyarray(py)
        .into_any();
    let values = if shell.values.is_empty() {
        py.None().into_bound(py)
    } else {
        array1(py, shell.values).into_any()
    };
    PyTuple::new(py, [vertices, triangles, values])
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Mesh>()?;
    m.add_class::<PolygonSelector>()?;
    m.add_function(wrap_pyfunction!(point_in_polygon, m)?)?;
    m.add_function(wrap_pyfunction!(polygon_distance, m)?)?;
    m.add_function(wrap_pyfunction!(assign_domain, m)?)?;
    m.add_function(wrap_pyfunction!(block_shell, m)?)?;
    Ok(())
}
