use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, StringArray};
use blocks::{
    DomainMethod, Orientation, PolygonSelector as CoreSelector, ShellBlock, ShellFilter,
    ShellLimits, SolidTester, extract_shell,
};
use ceres_core::Mesh as CoreMesh;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyTuple};
use rayon::prelude::*;

use crate::args::{Point, array1, column, finite, floats, named, points, rows, same_length, texts};
use crate::containers::{PyBlockModel, coords_arg, coords_array, float_column};
use crate::estimation::targets;
use crate::invalid;
use crate::table::Table;

fn err(e: blocks::BlockModelError) -> PyErr {
    invalid(e)
}

#[derive(FromPyObject)]
enum Discretization {
    All(usize),
    Axes([usize; 3]),
}

fn bools<'py>(py: Python<'py>, values: Vec<bool>) -> Bound<'py, PyAny> {
    PyArray1::from_vec(py, values).into_any()
}

fn core_error(e: ceres_core::Error) -> PyErr {
    invalid(e)
}

/// Triangulated surface or solid with per-vertex and per-face attributes.
#[pyclass(module = "ceres", name = "Mesh", frozen)]
pub struct Mesh {
    pub mesh: CoreMesh,
    tester: Option<SolidTester>,
}

impl Mesh {
    pub fn from_core(mesh: CoreMesh) -> Self {
        let tester = SolidTester::new(&mesh).ok();
        Self { mesh, tester }
    }

    /// Flat `(n, 3)` corners and triangles as `(k, 3)` index rows.
    pub fn build(vertices: &[f64], triangles: &[u32]) -> PyResult<CoreMesh> {
        CoreMesh::new(
            vertices
                .chunks_exact(3)
                .map(|v| [v[0], v[1], v[2]])
                .collect(),
            triangles
                .chunks_exact(3)
                .map(|t| [t[0], t[1], t[2]])
                .collect(),
        )
        .map_err(core_error)
    }

    fn solid(&self) -> PyResult<&SolidTester> {
        self.tester
            .as_ref()
            .ok_or_else(|| invalid("mesh is not closed; this needs a solid"))
    }

    /// Domain `label` inside this solid, or below or above this surface.
    pub fn domain(&self, rule: &str, label: String) -> PyResult<blocks::Domain> {
        let region = match rule {
            "inside" => blocks::Region::Inside(self.solid()?.clone()),
            "below" => blocks::Region::Below(blocks::Surface::new(&self.mesh).map_err(err)?),
            "above" => blocks::Region::Above(blocks::Surface::new(&self.mesh).map_err(err)?),
            _ => {
                return Err(invalid(format!(
                    "rule must be 'inside', 'below' or 'above', got {rule:?}"
                )));
            }
        };
        Ok(blocks::Domain { region, label })
    }

    fn with(&self, mesh: ceres_core::Result<CoreMesh>) -> PyResult<Self> {
        Ok(Self {
            mesh: mesh.map_err(core_error)?,
            tester: self.tester.clone(),
        })
    }
}

pub fn attribute(values: &Bound<PyAny>, rows: usize) -> PyResult<ArrayRef> {
    match values.extract::<Vec<Option<String>>>() {
        Ok(text) => Ok(Arc::new(StringArray::from(text))),
        Err(_) => float_column(values, rows),
    }
}

#[pymethods]
impl Mesh {
    /// `vertices` is `(n, 3)`, `triangles` `(m, 3)` vertex indices.
    #[new]
    #[pyo3(signature = (vertices, triangles, crs=None))]
    fn new(
        vertices: &Bound<PyAny>,
        triangles: &Bound<PyAny>,
        crs: Option<String>,
    ) -> PyResult<Self> {
        let vertices = coords_arg(vertices)?;
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
            .map(|r| {
                let t: Vec<u32> = r.iter().filter_map(|&i| u32::try_from(i).ok()).collect();
                <[u32; 3]>::try_from(t)
                    .map_err(|_| invalid("triangles must be (m, 3) non-negative indices"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut mesh = CoreMesh::new(vertices, triangles).map_err(core_error)?;
        mesh.crs = crs;
        Ok(Self::from_core(mesh))
    }

    #[getter]
    fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        coords_array(py, self.mesh.vertices()).into_any()
    }

    #[getter]
    fn triangles<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        let flat: Vec<i64> = self
            .mesh
            .triangles()
            .iter()
            .flatten()
            .map(|&i| i as i64)
            .collect();
        Array2::from_shape_vec((self.mesh.triangles().len(), 3), flat)
            .expect("m x 3")
            .into_pyarray(py)
            .into_any()
    }

    /// `(min, max)` corners of the bounding box, `None` when empty.
    #[getter]
    fn bounds(&self) -> Option<([f64; 3], [f64; 3])> {
        self.mesh.bounds()
    }

    #[getter]
    fn crs(&self) -> Option<String> {
        self.mesh.crs.clone()
    }

    /// Every edge shared by exactly two non-degenerate triangles.
    #[getter]
    fn is_closed(&self) -> bool {
        self.mesh.is_closed()
    }

    /// Degenerate triangles, boundary and non-manifold edges, and closure.
    #[getter]
    fn analysis<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let a = self.mesh.analysis();
        let d = PyDict::new(py);
        d.set_item("degenerate_triangles", a.degenerate_triangles)?;
        d.set_item("boundary_edges", a.boundary_edges)?;
        d.set_item("non_manifold_edges", a.non_manifold_edges)?;
        d.set_item("is_closed", a.is_closed)?;
        Ok(d)
    }

    #[getter]
    fn area(&self) -> f64 {
        self.mesh.area()
    }

    /// Enclosed volume of a closed mesh, positive when wound outward.
    #[getter]
    fn volume(&self) -> PyResult<f64> {
        self.mesh.volume().map_err(core_error)
    }

    #[getter]
    fn vertex_attributes(&self) -> Table {
        Table(self.mesh.vertex_attributes().clone())
    }

    #[getter]
    fn face_attributes(&self) -> Table {
        Table(self.mesh.face_attributes().clone())
    }

    /// New mesh with the per-vertex attribute `name` added or replaced.
    fn with_vertex_column(&self, name: &str, values: &Bound<PyAny>) -> PyResult<Self> {
        let column = attribute(values, self.mesh.vertices().len())?;
        self.with(self.mesh.with_vertex_column(name, column))
    }

    /// New mesh with the per-triangle attribute `name` added or replaced.
    fn with_face_column(&self, name: &str, values: &Bound<PyAny>) -> PyResult<Self> {
        let column = attribute(values, self.mesh.triangles().len())?;
        self.with(self.mesh.with_face_column(name, column))
    }

    /// Whether each point is inside, by generalized winding number.
    fn contains<'py>(&self, py: Python<'py>, points: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let pts = self::points(points)?;
        let solid = self.solid()?;
        let inside = py.detach(|| {
            pts.par_iter()
                .map(|p| solid.contains([p.0, p.1, p.2]))
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
                .map(|p| blocks::winding_number(&self.mesh, p))
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
                            blocks::signed_distance_to(&self.mesh, p)
                        } else {
                            blocks::distance_to(&self.mesh, p)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .map_err(err)?;
        Ok(array1(py, d).into_any())
    }

    /// Signed vertical distance to a surface such as topography.
    ///
    /// Parameters
    /// ----------
    /// points : array_like, PointSet or BlockModel
    ///     ``(n, 3)`` points, or the centroids of a point set or block model.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Point elevation minus surface elevation at the same (x, y):
    ///     positive above, negative below. NaN where no triangle covers the
    ///     point in plan; where the surface overlaps itself, the highest counts.
    fn vertical_distance<'py>(
        &self,
        py: Python<'py>,
        points: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let pts = targets(points)?;
        let d = py
            .detach(|| blocks::vertical_distance(&self.mesh, &pts))
            .map_err(err)?;
        Ok(array1(py, d).into_any())
    }

    /// Weld close vertices, drop degenerate and duplicate triangles, and wind
    /// the triangles consistently.
    ///
    /// Parameters
    /// ----------
    /// tolerance : float, default 0.0
    ///     Vertices within this distance of an earlier vertex merge into it;
    ///     0 merges exact duplicates only.
    ///
    /// Returns
    /// -------
    /// Mesh
    ///     A new mesh without unused vertices, each connected piece wound one
    ///     way and outward where it is closed. Kept vertices and triangles
    ///     keep their order and attributes. Repairing it again changes nothing.
    #[pyo3(signature = (tolerance=0.0))]
    fn repair(&self, py: Python, tolerance: f64) -> PyResult<Self> {
        let mesh = py
            .detach(|| self.mesh.repair(tolerance))
            .map_err(core_error)?;
        Ok(Self::from_core(mesh))
    }

    /// Proportion of each block inside, from a grid of points per block
    /// where the surface may cut it: `discretization` per axis, or one count
    /// for all three. `targets` is a BlockModel, whose rotation and sub-block
    /// extents count, or axis-aligned centroids with a shared `size`.
    #[pyo3(signature = (targets, *, size=None, discretization=Discretization::All(4)))]
    fn proportion<'py>(
        &self,
        py: Python<'py>,
        targets: &Bound<PyAny>,
        size: Option<[f64; 3]>,
        discretization: Discretization,
    ) -> PyResult<Bound<'py, PyAny>> {
        let solid = self.solid()?;
        let discretization = match discretization {
            Discretization::All(n) => [n; 3],
            Discretization::Axes(n) => n,
        };
        let size = match (size, targets.cast::<PyBlockModel>()) {
            (Some(s), _) => s,
            (None, Ok(b)) => {
                let model = &b.get().0;
                let p = py.detach(|| blocks::proportions(solid, model, discretization));
                return Ok(array1(py, p).into_any());
            }
            (None, Err(_)) => return Err(invalid("give size for plain centroids")),
        };
        let centers = crate::estimation::targets(targets)?;
        let p = py.detach(|| {
            centers
                .par_iter()
                .map(|c| {
                    solid
                        .evaluate_block([c.0, c.1, c.2], size, discretization)
                        .proportion
                })
                .collect()
        });
        Ok(array1(py, p).into_any())
    }

    fn __repr__(&self) -> String {
        format!(
            "Mesh({} vertices, {} triangles, {})",
            self.mesh.vertices().len(),
            self.mesh.triangles().len(),
            if self.mesh.is_closed() {
                "closed"
            } else {
                "open"
            }
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

/// Domain of each target from labeled samples (`nearest` or `majority`), or
/// inside/outside a `mesh` (`solid`). Sample labels are `domains` or the
/// `domain_column` of `coords`. Returns labels and confidences.
#[pyfunction]
#[pyo3(signature = (targets, *, coords=None, domains=None, domain_column=None, method="nearest", mesh=None))]
#[allow(clippy::too_many_arguments)]
fn assign_domain<'py>(
    py: Python<'py>,
    targets: &Bound<PyAny>,
    coords: Option<&Bound<PyAny>>,
    domains: Option<&Bound<PyAny>>,
    domain_column: Option<&str>,
    method: &str,
    mesh: Option<PyRef<Mesh>>,
) -> PyResult<Bound<'py, PyTuple>> {
    let method = match method {
        "nearest" => DomainMethod::Nearest,
        "majority" => DomainMethod::MajorityVote,
        "solid" => DomainMethod::PointInSolid,
        other => return Err(invalid(format!("unknown method {other:?}"))),
    };
    let domains = match (domains, domain_column) {
        (Some(_), Some(_)) => return Err(invalid("give one of domains or domain_column")),
        (None, Some(c)) => Some(named(coords, c, "domain_column")?),
        (d, None) => d.cloned(),
    };
    let domains = domains
        .map(|d| {
            texts(&d, "domains")?
                .into_iter()
                .map(|t| t.ok_or_else(|| invalid("domains must not be null")))
                .collect::<PyResult<Vec<String>>>()
        })
        .transpose()?;
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

/// Convex hull of `(n, 3)` points as a closed mesh.
#[pyfunction]
fn convex_hull(points: &Bound<PyAny>) -> PyResult<Mesh> {
    let pts = coords_arg(points)?;
    blocks::convex_hull(&pts).map(Mesh::from_core).map_err(err)
}

/// Triangulated surface from a gridded elevation, such as topography.
///
/// Parameters
/// ----------
/// model : BlockModel
///     Regular or masked 2D model (one cell in z).
/// column : str
///     Elevation of each block; nulls and NaN leave holes.
///
/// Returns
/// -------
/// Mesh
///     One vertex per block with an elevation, at its center in plan. Four
///     neighboring vertices make two triangles and three make one, facing up
///     in an unrotated grid.
#[pyfunction]
fn grid_surface(py: Python, model: PyRef<PyBlockModel>, column: &str) -> PyResult<Mesh> {
    let m = &model.0;
    let z = floats(&crate::table::column(py, m.attributes(), column)?, "column")?;
    let z: Vec<Option<f64>> = z.into_iter().map(|v| (!v.is_nan()).then_some(v)).collect();
    blocks::grid_surface(m, &z)
        .map(Mesh::from_core)
        .map_err(err)
}

/// Visible skin of a block model; `column` becomes the face column `value`.
#[pyfunction]
#[pyo3(signature = (model, column=None))]
fn block_shell<'py>(
    py: Python<'py>,
    model: PyRef<PyBlockModel>,
    column: Option<&str>,
) -> PyResult<Mesh> {
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
    let mut mesh = Mesh::build(&shell.vertices, &shell.triangles)?;
    if !shell.values.is_empty() {
        let values: Float64Array = mesh
            .triangles()
            .iter()
            .map(|t| shell.values[t[0] as usize])
            .collect();
        mesh = mesh
            .with_face_column("value", Arc::new(values))
            .map_err(core_error)?;
    }
    Ok(Mesh::from_core(mesh))
}

/// Majority filter of block `classes` over a `window` of cells, repeated
/// `iterations` times; ties keep a block's class and absent cells do not vote.
/// Classes may be any labels, or the column of `model` holding them; the
/// result has the same labels. With `domains` (one label per block) or the
/// `domain_column` of `model`, only blocks of the same domain vote.
#[pyfunction]
#[pyo3(signature = (model, classes, *, window=(3, 3, 1), iterations=1, domains=None, domain_column=None))]
fn smooth_classes<'py>(
    py: Python<'py>,
    model: &Bound<'py, PyBlockModel>,
    classes: &Bound<'py, PyAny>,
    window: (usize, usize, usize),
    iterations: usize,
    domains: Option<&Bound<'py, PyAny>>,
    domain_column: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let classes = &column(Some(model.as_any()), classes, "classes")?;
    let domains = match (domains, domain_column) {
        (Some(_), Some(_)) => return Err(invalid("give one of domains or domain_column")),
        (None, Some(c)) => Some(named(Some(model.as_any()), c, "domain_column")?),
        (d, None) => d.cloned(),
    };
    let np = py.import("numpy")?;
    let encode = |values: &Bound<'py, PyAny>| -> PyResult<(Bound<'py, PyAny>, Vec<u32>)> {
        let unique = np.call_method(
            "unique",
            (values,),
            Some(&[("return_inverse", true)].into_py_dict(py)?),
        )?;
        Ok((
            unique.get_item(0)?,
            unique
                .get_item(1)?
                .call_method1("astype", ("uint32",))?
                .call_method0("ravel")?
                .call_method0("tolist")?
                .extract()?,
        ))
    };
    let (labels, codes) = encode(classes)?;
    let domains = domains.map(|d| encode(&d).map(|e| e.1)).transpose()?;
    let model = &model.get().0;
    let smoothed = py
        .detach(|| {
            blocks::smooth_classes(
                model,
                &codes,
                [window.0, window.1, window.2],
                iterations,
                domains.as_deref(),
            )
        })
        .map_err(err)?;
    labels.get_item(np.call_method1("asarray", (smoothed,))?)
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(smooth_classes, m)?)?;
    m.add_class::<Mesh>()?;
    m.add_class::<PolygonSelector>()?;
    m.add_function(wrap_pyfunction!(point_in_polygon, m)?)?;
    m.add_function(wrap_pyfunction!(polygon_distance, m)?)?;
    m.add_function(wrap_pyfunction!(assign_domain, m)?)?;
    m.add_function(wrap_pyfunction!(block_shell, m)?)?;
    m.add_function(wrap_pyfunction!(convex_hull, m)?)?;
    m.add_function(wrap_pyfunction!(grid_surface, m)?)?;
    Ok(())
}
