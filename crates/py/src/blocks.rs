use std::sync::{Arc, OnceLock};

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use blocks::{
    DomainMethod, Orientation, PolygonSelector as CoreSelector, ShellBlock, ShellFilter,
    ShellLimits, SolidTester, TriangleTree, extract_shell,
};
use boitata_core::Mesh as CoreMesh;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyTuple};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::args::{
    Point, array1, column, domain_codes, finite, floats, named, points, rows, same_length, texts,
};
use crate::containers::{
    PyBlockModel, PyPolylines, coords_arg, coords_array, coords_or_nan, float_column,
};
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

fn core_error(e: boitata_core::Error) -> PyErr {
    invalid(e)
}

/// Triangulated surface or solid with per-vertex and per-face attributes.
#[pyclass(module = "boitata", name = "Mesh", frozen)]
pub struct Mesh {
    pub mesh: CoreMesh,
    tester: Option<SolidTester>,
    tree: OnceLock<Result<TriangleTree, String>>,
}

impl Mesh {
    pub fn from_core(mesh: CoreMesh) -> Self {
        let tester = SolidTester::new(&mesh).ok();
        Self {
            mesh,
            tester,
            tree: OnceLock::new(),
        }
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

    pub(crate) fn solid(&self) -> PyResult<&SolidTester> {
        self.tester
            .as_ref()
            .ok_or_else(|| invalid("mesh is not closed; this needs a solid"))
    }

    /// Domain `label` inside this solid, or below or above this surface.
    fn domain(&self, rule: &str, label: String) -> PyResult<blocks::Domain> {
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

    fn with(&self, mesh: boitata_core::Result<CoreMesh>) -> PyResult<Self> {
        Ok(Self {
            mesh: mesh.map_err(core_error)?,
            tester: self.tester.clone(),
            tree: self.tree.clone(),
        })
    }
}

pub fn attribute(values: &Bound<PyAny>, rows: usize) -> PyResult<ArrayRef> {
    match values.extract::<Vec<Option<String>>>() {
        Ok(text) => {
            crate::args::same_length(rows, text.len(), "values")?;
            Ok(Arc::new(StringArray::from(text)))
        }
        Err(_) => float_column(values, rows),
    }
}

#[pymethods]
impl Mesh {
    /// `vertices` is `(n, 3)`, `triangles` `(m, 3)` vertex indices.
    #[new]
    #[pyo3(signature = (vertices, triangles, *, crs=None))]
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

    /// Everything that keeps the mesh from being a clean closed solid.
    ///
    /// Parameters
    /// ----------
    /// tolerance : float, default 0.0
    ///     A vertex within this distance of an earlier vertex is a duplicate;
    ///     0 finds exact duplicates only.
    ///
    /// Returns
    /// -------
    /// MeshReport
    ///     Counts in ``summary`` and one row per problem in ``problems``.
    ///     Degenerate faces are left out of the edge, vertex and shell checks.
    #[pyo3(signature = (*, tolerance=0.0))]
    fn validate(&self, py: Python, tolerance: f64) -> PyResult<MeshReport> {
        py.detach(|| self.mesh.validate(tolerance))
            .map(MeshReport)
            .map_err(core_error)
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
    #[pyo3(signature = (points, *, signed=false))]
    fn distance<'py>(
        &self,
        py: Python<'py>,
        points: &Bound<PyAny>,
        signed: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let pts = self::points(points)?;
        let tree = py.detach(|| {
            self.tree
                .get_or_init(|| TriangleTree::new(&self.mesh).map_err(|e| e.to_string()))
        });
        let tree = tree.as_ref().map_err(invalid)?;
        let solid = if signed { Some(self.solid()?) } else { None };
        let d = py.detach(|| {
            pts.par_iter()
                .map(|p| {
                    let p = [p.0, p.1, p.2];
                    let d = tree.distance(p);
                    if solid.is_some_and(|s| s.contains(p)) {
                        -d
                    } else {
                        d
                    }
                })
                .collect()
        });
        Ok(array1(py, d).into_any())
    }

    /// Signed vertical distance to a surface such as topography.
    ///
    /// Parameters
    /// ----------
    /// points : array_like, PointSet or BlockModel
    ///     ``(n, 3)`` points, or the centroids of a point set or block model.
    ///     Rows with a NaN coordinate are allowed.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Point elevation minus surface elevation at the same (x, y):
    ///     positive above, negative below. NaN where a coordinate is NaN or no
    ///     triangle covers the point in plan; where the surface overlaps itself,
    ///     the highest counts.
    fn vertical_distance<'py>(
        &self,
        py: Python<'py>,
        points: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let pts: Vec<Point> = coords_or_nan(points)?
            .into_iter()
            .map(|[x, y, z]| (x, y, z))
            .collect();
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
    ///     way. A closed piece inside an even number of other closed pieces
    ///     winds outward; inside an odd number it is a cavity and winds
    ///     inward, so the volume is outer minus cavities. Kept vertices and
    ///     triangles keep their order and attributes. Repairing it again
    ///     changes nothing. Crossing faces stay; ``validate`` reports them.
    #[pyo3(signature = (*, tolerance=0.0))]
    fn repair(&self, py: Python, tolerance: f64) -> PyResult<Self> {
        let mesh = py
            .detach(|| self.mesh.repair(tolerance))
            .map_err(core_error)?;
        Ok(Self::from_core(mesh))
    }

    /// Closes each hole with a fan of triangles from its first vertex.
    ///
    /// Returns
    /// -------
    /// Mesh
    ///     A new mesh with the fans appended, wound like the faces around
    ///     each hole, with null face attributes. A hole is a loop of boundary
    ///     edges; a loop through a vertex that starts two boundary edges
    ///     stays open. A fan suits small, nearly planar, convex holes; others
    ///     can come out folded, so validate the result.
    fn fill_holes(&self, py: Python) -> PyResult<Self> {
        let mesh = py.detach(|| self.mesh.fill_holes()).map_err(core_error)?;
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
                "closed".to_string()
            } else {
                format!("open, {} boundary edges", self.mesh.boundary_edges())
            }
        )
    }
}

/// Problems found by `Mesh.validate`.
#[pyclass(module = "boitata", name = "MeshReport", frozen)]
pub struct MeshReport(boitata_core::MeshReport);

#[pymethods]
impl MeshReport {
    /// Counts of ``degenerate_faces``, ``duplicate_faces`` (same corners as
    /// an earlier face), ``duplicate_vertices``, ``boundary_edges`` (of one
    /// face), ``non_manifold_edges`` (of three or more),
    /// ``non_manifold_vertices`` (whose faces form separate fans),
    /// ``inconsistent_edges`` (whose two faces run them the same way),
    /// ``shells`` (pieces connected through shared edges) and
    /// ``inward_shells`` (closed shells wound against their nesting: a
    /// cavity should wind inward, anything else outward),
    /// ``self_intersections`` (pairs of faces crossing without sharing an
    /// edge) and ``is_closed``.
    #[getter]
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = &self.0.summary;
        let d = PyDict::new(py);
        d.set_item("degenerate_faces", s.degenerate_faces)?;
        d.set_item("duplicate_faces", s.duplicate_faces)?;
        d.set_item("duplicate_vertices", s.duplicate_vertices)?;
        d.set_item("boundary_edges", s.boundary_edges)?;
        d.set_item("non_manifold_edges", s.non_manifold_edges)?;
        d.set_item("non_manifold_vertices", s.non_manifold_vertices)?;
        d.set_item("inconsistent_edges", s.inconsistent_edges)?;
        d.set_item("shells", s.shells)?;
        d.set_item("inward_shells", s.inward_shells)?;
        d.set_item("self_intersections", s.self_intersections)?;
        d.set_item("is_closed", s.is_closed)?;
        Ok(d)
    }

    /// One row per problem, ordered by ``kind``: ``degenerate_face``,
    /// ``duplicate_face``, ``duplicate_vertex``, ``boundary_edge``,
    /// ``non_manifold_edge``, ``non_manifold_vertex``,
    /// ``inconsistent_winding``, ``inward_shell``, ``self_intersection``.
    /// ``face``, ``vertex``,
    /// ``edge`` and ``other`` are null where they do not apply. An edge
    /// problem gives a ``face`` holding the edge, the local ``edge`` (from
    /// corner ``edge`` to corner ``(edge + 1) % 3``) and its first
    /// ``vertex``. ``other`` is the earlier face or vertex a duplicate
    /// repeats, the other face of an inconsistent edge, or the earlier face
    /// of a crossing pair; an inward shell
    /// gives its first face.
    #[getter]
    fn problems(&self) -> PyResult<Table> {
        let p = &self.0.problems;
        let ids = |f: fn(&boitata_core::MeshProblem) -> Option<u32>| -> ArrayRef {
            Arc::new(Int64Array::from_iter(p.iter().map(|q| f(q).map(i64::from))))
        };
        let kinds: Vec<&str> = p.iter().map(|q| q.kind.name()).collect();
        RecordBatch::try_from_iter([
            ("kind", Arc::new(StringArray::from(kinds)) as ArrayRef),
            ("face", ids(|q| q.face)),
            ("vertex", ids(|q| q.vertex)),
            ("edge", ids(|q| q.edge.map(u32::from))),
            ("other", ids(|q| q.other)),
        ])
        .map(Table)
        .map_err(invalid)
    }

    fn __repr__(&self) -> String {
        let s = &self.0.summary;
        format!(
            "MeshReport({} problems, {} shells, {})",
            self.0.problems.len(),
            s.shells,
            if s.is_closed { "closed" } else { "open" }
        )
    }
}

/// Selects points inside plan-view rings, optionally between `z_min` and
/// `z_max`. `rings` are `(n, 3)` polylines, where `closed=True` means they
/// close implicitly, or a Polylines, whose closed parts are used. Inside means
/// an odd number of rings of a feature hold the point, so a ring inside
/// another is a hole; raw rings are one feature.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "PolygonSelector", frozen)]
pub struct PolygonSelector(CoreSelector);

#[pymethods]
impl PolygonSelector {
    /// JSON of the rings and the elevation window.
    fn to_json(&self) -> PyResult<String> {
        crate::persist::to_json(self)
    }

    /// Reads `to_json` output; raises InvalidInput on another class's JSON
    /// or a newer format.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        crate::persist::from_json(text)
    }

    #[new]
    #[pyo3(signature = (rings, *, closed=false, z_min=None, z_max=None))]
    fn new(
        rings: &Bound<PyAny>,
        closed: bool,
        z_min: Option<f64>,
        z_max: Option<f64>,
    ) -> PyResult<Self> {
        if let Ok(lines) = rings.cast::<PyPolylines>() {
            let selector = CoreSelector::from_polylines(&lines.get().0, z_min, z_max);
            return Ok(Self(selector.map_err(err)?));
        }
        let rings: Vec<Bound<PyAny>> = rings.extract()?;
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

/// Coordinates in the frame of a layer between two surfaces.
///
/// Folded or undulating layers flatten: ``w`` places a point between the
/// footwall and the hanging wall, vertically, and ``(u, v)`` run along the
/// layer. Variograms and estimates computed on unfolded coordinates follow
/// the layer; results stay in the input's row order, so they go back to the
/// real points or blocks as they are.
///
/// Parameters
/// ----------
/// footwall, hangingwall : Mesh
///     Bounding surfaces, such as ``grid_surface`` meshes; where either
///     overlaps itself in plan, its highest elevation counts.
/// mode : {"proportional", "footwall", "hangingwall"}, default "proportional"
///     ``w`` as the relative position, 0 on the footwall and 1 on the hanging
///     wall; as the height above the footwall; or as the height relative to
///     the hanging wall, negative below it.
/// reference : {"footwall", "hangingwall"}, optional
///     Surface along which ``u`` and ``v`` are arc lengths in the x and y
///     directions, from its south-west corner and offset so that a flat
///     surface keeps x and y. By default ``u`` and ``v`` are x and y.
/// extrapolate : bool, default False
///     Unfold points above or below the layer too, instead of giving NaN.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "Unfold", frozen)]
pub struct Unfold(blocks::Unfold);

fn triples<'py>(py: Python<'py>, rows: Vec<[f64; 3]>) -> Bound<'py, PyAny> {
    let n = rows.len();
    Array2::from_shape_vec((n, 3), rows.into_iter().flatten().collect())
        .expect("three columns")
        .into_pyarray(py)
        .into_any()
}

#[pymethods]
impl Unfold {
    /// JSON of the surfaces and options.
    fn to_json(&self) -> PyResult<String> {
        crate::persist::to_json(self)
    }

    /// Reads `to_json` output; raises InvalidInput on another class's JSON
    /// or a newer format.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        crate::persist::from_json(text)
    }

    #[new]
    #[pyo3(signature = (footwall, hangingwall, *, mode="proportional", reference=None, extrapolate=false))]
    fn new(
        py: Python,
        footwall: PyRef<Mesh>,
        hangingwall: PyRef<Mesh>,
        mode: &str,
        reference: Option<&str>,
        extrapolate: bool,
    ) -> PyResult<Self> {
        let mode = match mode {
            "proportional" => blocks::UnfoldMode::Proportional,
            "footwall" => blocks::UnfoldMode::Footwall,
            "hangingwall" => blocks::UnfoldMode::Hangingwall,
            _ => {
                return Err(invalid(format!(
                    "mode must be 'proportional', 'footwall' or 'hangingwall', got {mode:?}"
                )));
            }
        };
        let reference = match reference {
            None => None,
            Some("footwall") => Some(blocks::Reference::Footwall),
            Some("hangingwall") => Some(blocks::Reference::Hangingwall),
            Some(r) => {
                return Err(invalid(format!(
                    "reference must be 'footwall' or 'hangingwall', got {r:?}"
                )));
            }
        };
        let (f, h) = (&footwall.mesh, &hangingwall.mesh);
        py.detach(|| blocks::Unfold::new(f, h, mode, reference, extrapolate))
            .map(Self)
            .map_err(err)
    }

    /// Nothing to learn from the data; checks the coordinates and returns the
    /// unfolding, for pipelines.
    fn fit<'py>(slf: PyRef<'py, Self>, coords: &Bound<PyAny>) -> PyResult<PyRef<'py, Self>> {
        coords_or_nan(coords)?;
        Ok(slf)
    }

    /// Same as ``transform``.
    fn fit_transform<'py>(
        &self,
        py: Python<'py>,
        coords: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.transform(py, coords)
    }

    /// Unfold points.
    ///
    /// Parameters
    /// ----------
    /// coords : array_like, PointSet or BlockModel
    ///     ``(n, 3)`` points, or the centroids of a point set or block model.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     ``(n, 3)`` rows of ``(u, v, w)``. NaN where the surfaces do not
    ///     both cover the point in plan, the hanging wall lies below the
    ///     footwall (or on it, for ``"proportional"``), or, unless
    ///     extrapolating, the point is outside the layer.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        coords: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let points = coords_or_nan(coords)?;
        Ok(triples(py, py.detach(|| self.0.transform(&points))))
    }

    /// Real coordinates of unfolded points.
    ///
    /// Parameters
    /// ----------
    /// coords : array_like
    ///     ``(n, 3)`` rows of ``(u, v, w)``.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     ``(n, 3)`` rows of ``(x, y, z)``, NaN where ``transform`` would
    ///     give NaN. With a reference surface, ``(x, y)`` is found by Newton
    ///     steps and is NaN if they do not converge.
    fn inverse<'py>(&self, py: Python<'py>, coords: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let points = coords_or_nan(coords)?;
        Ok(triples(py, py.detach(|| self.0.inverse(&points))))
    }
}

/// Domain `label` from a `(Mesh or Polylines, rule, label)` tuple; a
/// Polylines is a vertical prism, `"inside"` only.
pub fn domain(region: &Bound<PyAny>, rule: &str, label: String) -> PyResult<blocks::Domain> {
    if let Ok(lines) = region.cast::<PyPolylines>() {
        if rule != "inside" {
            return Err(invalid(format!(
                "Polylines take rule 'inside', got {rule:?}"
            )));
        }
        let prism = CoreSelector::from_polylines(&lines.get().0, None, None).map_err(err)?;
        let region = blocks::Region::Prism(prism);
        return Ok(blocks::Domain { region, label });
    }
    region.cast::<Mesh>()?.get().domain(rule, label)
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

/// Whether each `(x, y)` point is inside `polygon`, an `(n, 2)` ring or a
/// Polylines (even-odd rule; see `Polylines.contains`).
#[pyfunction]
fn point_in_polygon<'py>(
    py: Python<'py>,
    points: &Bound<'py, PyAny>,
    polygon: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(lines) = polygon.cast::<PyPolylines>() {
        return Ok(lines.call_method1("contains", (points,))?.into_any());
    }
    let ring = self::polygon(polygon)?;
    let inside = self::points(points)?
        .iter()
        .map(|p| blocks::point_in_polygon((p.0, p.1), &ring))
        .collect();
    Ok(bools(py, inside))
}

/// Plan distance from each point to the boundary of `polygon`, an `(n, 2)`
/// ring or a Polylines (see `Polylines.distance`); `signed` makes inside
/// points negative.
#[pyfunction]
#[pyo3(signature = (points, polygon, *, signed=false))]
fn polygon_distance<'py>(
    py: Python<'py>,
    points: &Bound<'py, PyAny>,
    polygon: &Bound<'py, PyAny>,
    signed: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(lines) = polygon.cast::<PyPolylines>() {
        let kwargs = [("signed", signed)].into_py_dict(py)?;
        return lines.call_method("distance", (points,), Some(&kwargs));
    }
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

/// Domain of each target from labeled samples, or inside/outside a closed mesh.
///
/// Parameters
/// ----------
/// targets : array_like, PointSet or BlockModel
///     ``(n, 2|3)`` points, or the centroids of a point set or block model.
/// coords : array_like, PointSet or BlockModel, optional
///     Sample locations, for ``nearest`` and ``majority``.
/// domains : array_like of str, optional
///     Sample labels; or give ``domain_column``.
/// domain_column : str, optional
///     Column of ``coords`` holding the sample labels.
/// method : {"nearest", "majority", "solid"}, default "nearest"
///     ``nearest``: the label of the nearest sample. ``majority``: the most
///     frequent label among the ``n`` nearest samples; a tie goes to the label
///     of the nearest sample. Samples at equal distance rank by label.
///     ``solid``: ``"inside"`` or ``"outside"`` of ``mesh``.
/// mesh : Mesh, optional
///     Closed mesh, for ``solid``.
/// n : int, default 5
///     Neighbors that vote and that the confidence is measured on.
///
/// Returns
/// -------
/// labels : list of str
/// confidence : numpy.ndarray
///     Share of the ``n`` nearest samples (all of them when fewer) carrying
///     the chosen label; 1 for ``solid``.
#[pyfunction]
#[pyo3(signature = (targets, *, coords=None, domains=None, domain_column=None, method="nearest", mesh=None, n=5))]
#[allow(clippy::too_many_arguments)]
fn assign_domain<'py>(
    py: Python<'py>,
    targets: &Bound<PyAny>,
    coords: Option<&Bound<PyAny>>,
    domains: Option<&Bound<PyAny>>,
    domain_column: Option<&str>,
    method: &str,
    mesh: Option<PyRef<Mesh>>,
    n: usize,
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
    let targets = self::targets(targets)?;
    let out = py
        .detach(|| blocks::assign_domain(&targets, &samples, method, n, mesh))
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
#[pyo3(signature = (model, *, column=None))]
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
    let frame = boitata_core::block_frame(m.geometry().rotation);
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

/// Outer faces of the rows of `model` where `keep` is true; `plot3d` draws them.
#[pyfunction]
#[pyo3(signature = (model, keep=None))]
fn _outer_faces<'py>(
    py: Python<'py>,
    model: PyRef<PyBlockModel>,
    keep: Option<numpy::PyReadonlyArray1<bool>>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let keep = keep.map(|k| k.as_array().to_vec());
    let m = &model.0;
    let f = py
        .detach(|| blocks::outer_faces(m, keep.as_deref()))
        .map_err(err)?;
    let points = Array2::from_shape_vec((f.points.len(), 3), f.points.into_flattened())
        .expect("three coordinates")
        .into_pyarray(py)
        .into_any();
    let quads = Array2::from_shape_vec((f.quads.len(), 4), f.quads.into_flattened())
        .expect("four corners")
        .into_pyarray(py)
        .into_any();
    Ok((points, quads, f.rows.into_pyarray(py).into_any()))
}

/// Majority filter of block `classes` over a `window` of parent cells, repeated
/// `iterations` times; ties keep a block's class and absent cells do not vote.
/// Blocks vote with their volume, so in a sub-blocked model each sub-block
/// counts by the fraction of its parent it fills.
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
    decode(&labels, smoothed)
}

/// Distinct labels of `values` and the code of each value.
fn encode<'py>(values: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Vec<u32>)> {
    let py = values.py();
    let unique = py.import("numpy")?.call_method(
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
}

fn decode<'py>(labels: &Bound<'py, PyAny>, codes: Vec<u32>) -> PyResult<Bound<'py, PyAny>> {
    labels.get_item(PyArray1::from_vec(labels.py(), codes))
}

/// Reclassify connected units of block `classes` smaller than `min_volume`
/// (or `min_blocks` blocks) into the class of their neighbors.
///
/// A unit joins blocks of one class whose parent cells share a face
/// (`connectivity=6`) or a face, edge or corner (`connectivity=26`); sub-blocks
/// of one parent cell touch. Units go smallest first, and each takes the class
/// with the most volume among the blocks touching it, ties to the first class
/// in sorted order, merging with the units of that class it touches. A unit
/// left below the minimum has no neighbor to join: absent cells or other
/// domains surround it. With `domains` (one label per block) or the
/// `domain_column` of `model`, units and neighbors stay within a domain.
/// Classes may be any labels, or the column of `model` holding them; the
/// result has the same labels.
#[pyfunction]
#[pyo3(signature = (model, classes, *, min_volume=None, min_blocks=None, connectivity=6, domains=None, domain_column=None))]
fn remove_small_units<'py>(
    model: &Bound<'py, PyBlockModel>,
    classes: &Bound<'py, PyAny>,
    min_volume: Option<f64>,
    min_blocks: Option<usize>,
    connectivity: usize,
    domains: Option<&Bound<'py, PyAny>>,
    domain_column: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let min = match (min_volume, min_blocks) {
        (Some(v), None) => blocks::MinSize::Volume(v),
        (None, Some(b)) => blocks::MinSize::Blocks(b),
        _ => return Err(invalid("give one of min_volume or min_blocks")),
    };
    let classes = &column(Some(model.as_any()), classes, "classes")?;
    let n = model.get().0.len();
    let domains = (domains.is_some() || domain_column.is_some())
        .then(|| domain_codes(Some(model.as_any()), domains, domain_column, n))
        .transpose()?
        .map(|d| d.1);
    let (labels, codes) = encode(classes)?;
    let py = model.py();
    let model = &model.get().0;
    let out = py
        .detach(|| blocks::remove_small_units(model, &codes, min, connectivity, domains.as_deref()))
        .map_err(err)?;
    decode(&labels, out)
}

fn contact<'py>(
    model: &Bound<'py, PyBlockModel>,
    classes: &Bound<'py, PyAny>,
    target: Option<&Bound<'py, PyAny>>,
    signed: bool,
) -> PyResult<(Bound<'py, PyAny>, Vec<f64>)> {
    let classes = column(Some(model.as_any()), classes, "classes")?;
    let (labels, codes) = encode(&classes)?;
    let target = match target {
        Some(t) => Some(
            labels
                .try_iter()?
                .position(|l| l.and_then(|l| l.eq(t)).unwrap_or(false))
                .map_or(u32::MAX, |c| c as u32),
        ),
        None => None,
    };
    let py = model.py();
    let model = &model.get().0;
    let distances = py
        .detach(|| blocks::contact_distance(model, &codes, target, signed))
        .map_err(err)?;
    Ok((classes, distances))
}

/// Distance from each block to the nearest block of another class, or with
/// `target`, to the nearest block of class `target`.
///
/// Blocks of `target` get the distance to the nearest block outside it,
/// negative when `signed`; without `target` distances are never negative.
/// Distances are exact, between block centroids, in any layout, so blocks on
/// either side of a contact get the same distance. NaN where there is no such
/// block. Classes may be any labels, or the column of `model` holding them.
#[pyfunction]
#[pyo3(signature = (model, classes, *, target=None, signed=true))]
fn contact_distance<'py>(
    model: &Bound<'py, PyBlockModel>,
    classes: &Bound<'py, PyAny>,
    target: Option<&Bound<'py, PyAny>>,
    signed: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (_, distances) = contact(model, classes, target, signed)?;
    Ok(array1(model.py(), distances))
}

/// Block `classes` with every block within `distance` of a contact relabeled
/// `label` (`"contact"` by default): of a contact between any two classes, or with `target`, of the
/// contact of class `target`, on both sides of it. Distances are those of
/// `contact_distance`, so the buffer is as wide on either side.
#[pyfunction]
#[pyo3(signature = (model, classes, *, distance, label=None, target=None))]
fn buffer_domains<'py>(
    model: &Bound<'py, PyBlockModel>,
    classes: &Bound<'py, PyAny>,
    distance: f64,
    label: Option<&Bound<'py, PyAny>>,
    target: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = model.py();
    let label = match label {
        Some(l) => l.clone(),
        None => "contact".into_pyobject(py)?.into_any(),
    };
    if distance.is_nan() || distance < 0.0 {
        return Err(invalid("distance must not be negative"));
    }
    let (classes, distances) = contact(model, classes, target, false)?;
    let near: Vec<bool> = distances.iter().map(|d| *d <= distance).collect();
    let np = model.py().import("numpy")?;
    np.call_method1(
        "where",
        (
            bools(model.py(), near),
            label,
            np.call_method1("asarray", (classes,))?,
        ),
    )
}

/// Delaunay surface through points in plan; `bt.topography` wraps it.
#[pyfunction]
fn _tin(coords: &Bound<PyAny>) -> PyResult<Mesh> {
    let points = crate::containers::coords_arg(coords)?;
    blocks::tin(&points).map(Mesh::from_core).map_err(err)
}

/// Leave-one-out residuals of the Delaunay surface; `bt.topography` wraps it.
#[pyfunction]
fn _tin_residuals<'py>(py: Python<'py>, coords: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let points = crate::containers::coords_arg(coords)?;
    let r = py.detach(|| blocks::tin_residuals(&points)).map_err(err)?;
    Ok(crate::args::array1(py, r).into_any())
}

/// Rings outlining one group of points; `bt.outline` groups and wraps them.
#[pyfunction]
#[pyo3(signature = (coords, *, max_edge=None, buffer=0.0, plane=None))]
fn _outline<'py>(
    py: Python<'py>,
    coords: &Bound<PyAny>,
    max_edge: Option<f64>,
    buffer: f64,
    plane: Option<(f64, f64)>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let points = crate::containers::coords_arg(coords)?;
    let hull = max_edge.map_or(blocks::Hull::Convex, blocks::Hull::Concave);
    let plane = plane.map_or(blocks::Plane::Plan, |(azimuth, dip)| {
        blocks::Plane::Dipping { azimuth, dip }
    });
    let rings = py
        .detach(|| blocks::outline(&points, hull, buffer, plane))
        .map_err(err)?;
    Ok(rings
        .iter()
        .map(|r| {
            let rows: Vec<Vec<f64>> = r.iter().map(|p| p.to_vec()).collect();
            crate::args::array2(py, &rows).into_any()
        })
        .collect())
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_outline, m)?)?;
    m.add_function(wrap_pyfunction!(_tin, m)?)?;
    m.add_function(wrap_pyfunction!(_outer_faces, m)?)?;
    m.add_function(wrap_pyfunction!(_tin_residuals, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_classes, m)?)?;
    m.add_function(wrap_pyfunction!(remove_small_units, m)?)?;
    m.add_function(wrap_pyfunction!(contact_distance, m)?)?;
    m.add_function(wrap_pyfunction!(buffer_domains, m)?)?;
    m.add_class::<Mesh>()?;
    m.add_class::<MeshReport>()?;
    m.add_class::<PolygonSelector>()?;
    m.add_class::<Unfold>()?;
    m.add_function(wrap_pyfunction!(point_in_polygon, m)?)?;
    m.add_function(wrap_pyfunction!(polygon_distance, m)?)?;
    m.add_function(wrap_pyfunction!(assign_domain, m)?)?;
    m.add_function(wrap_pyfunction!(block_shell, m)?)?;
    m.add_function(wrap_pyfunction!(convex_hull, m)?)?;
    m.add_function(wrap_pyfunction!(grid_surface, m)?)?;
    Ok(())
}
