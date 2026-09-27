use modeling::{
    ConstraintSet, Driver, HermiteKriging, HermiteSpec, Kernel, Lineation, Plane, PlaneEncoding,
    Rbf, RbfSpec, ScalarGrid, Svgp, SvgpSpec, marching_tetrahedra,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use variogram::Angles;

use crate::args::{array1, array2, distinct, finite, pick, points, rows, same_length};
use crate::blocks::Mesh;
use crate::containers::PyBlockModel;
use crate::estimation::targets;
use crate::invalid;
use crate::persist::{self, Columns, Found, Tabular};
use crate::variogram::Variogram;

enum Fitted {
    Rbf(Rbf),
    Kriging(HermiteKriging),
    Gp(Svgp),
}

impl Fitted {
    fn value(&self, p: &[f64; 3]) -> f64 {
        match self {
            Fitted::Rbf(f) => f.value(p),
            Fitted::Kriging(f) => f.value(p),
            Fitted::Gp(f) => f.value(p),
        }
    }
}

/// Implicit scalar field fitted to samples, boundary picks and structural
/// readings; its level sets are geological surfaces.
///
/// Parameters
/// ----------
/// engine : {"rbf", "kriging", "gp"}
///     Radial basis functions, dual kriging with derivative data, or a sparse
///     variational Gaussian process.
/// kernel : {"biharmonic", "triharmonic", "thin-plate"}
///     RBF kernel; planes and lineations need triharmonic.
/// variogram : Variogram, optional
///     Required by ``engine="kriging"``.
/// degree : int
///     0 constant, 1 linear drift.
/// smoothing : float
///     RBF ridge relative to the value scale; 0 interpolates exactly.
/// rotation : tuple of float, optional
///     Azimuth, dip, rake of the anisotropy axes. With ``engine="gp"`` the
///     ranges along them are learned; ``engine="kriging"`` takes its
///     anisotropy from the variogram.
/// ratios : tuple of float, optional
///     Semi-major/major and minor/major range ratios for ``engine="rbf"``.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "ceres", name = "ImplicitModel")]
pub struct ImplicitModel {
    engine: String,
    kernel: String,
    variogram: Option<Variogram>,
    degree: usize,
    smoothing: f64,
    rotation: Option<(f64, f64, f64)>,
    ratios: Option<(f64, f64)>,
    #[serde(skip)]
    fitted: Option<Fitted>,
    #[serde(skip)]
    inputs: Vec<Input>,
}

/// A sample carrying its encoded value, a boundary pick or a structural
/// reading.
#[derive(Clone, Copy)]
enum Reading {
    Sample(f64),
    Boundary,
    Plane(Plane),
    Lineation(Lineation),
}

type Input = ([f64; 3], Reading);

fn kernel(label: &str) -> PyResult<Kernel> {
    [Kernel::Biharmonic, Kernel::Triharmonic, Kernel::ThinPlate]
        .into_iter()
        .find(|k| k.label() == label)
        .ok_or_else(|| {
            invalid(format!(
                "kernel must be biharmonic, triharmonic or thin-plate, not {label:?}"
            ))
        })
}

impl ImplicitModel {
    fn fitted(&self) -> PyResult<&Fitted> {
        self.fitted
            .as_ref()
            .ok_or_else(|| invalid("ImplicitModel is not fitted; call fit first"))
    }

    fn check(&self) -> PyResult<()> {
        kernel(&self.kernel)?;
        match self.engine.as_str() {
            "rbf" | "gp" => {}
            "kriging" if self.variogram.is_some() => {}
            "kriging" => return Err(invalid("engine=\"kriging\" needs a variogram")),
            other => {
                return Err(invalid(format!(
                    "engine must be rbf, kriging or gp, not {other:?}"
                )));
            }
        }
        if !(self.smoothing.is_finite() && self.smoothing >= 0.0) {
            return Err(invalid("smoothing must be >= 0"));
        }
        match self.engine.as_str() {
            "kriging" if self.rotation.is_some() || self.ratios.is_some() => {
                return Err(invalid(
                    "engine=\"kriging\" takes its anisotropy from the variogram",
                ));
            }
            "gp" if self.ratios.is_some() => {
                return Err(invalid(
                    "engine=\"gp\" learns its ranges; give rotation only",
                ));
            }
            _ => {}
        }
        if self
            .ratios
            .is_some_and(|(a, b)| !(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite()))
        {
            return Err(invalid("ratios must be > 0"));
        }
        Ok(())
    }

    /// Fits the engine to `inputs`; the same inputs give the same field.
    fn fit_inputs(&mut self, py: Python, inputs: Vec<Input>) -> PyResult<()> {
        let err = |e: modeling::ModelError| invalid(e);
        let mut set = ConstraintSet::new();
        for &(at, reading) in &inputs {
            match reading {
                Reading::Sample(v) => set.push_sample(at, v),
                Reading::Boundary => set.push_boundary(at, 0.0),
                Reading::Plane(plane) => set
                    .push_plane(at, plane, PlaneEncoding::Tangents, 0.0)
                    .map_err(err)?,
                Reading::Lineation(line) => set.push_lineation(at, line).map_err(err)?,
            }
        }
        set.validate().map_err(err)?;
        let degree = self.degree;
        let fitted = match self.engine.as_str() {
            "rbf" => {
                let anisotropy = (self.rotation.is_some() || self.ratios.is_some()).then(|| {
                    let (azimuth, dip, rake) = self.rotation.unwrap_or_default();
                    let (semi, minor) = self.ratios.unwrap_or((1.0, 1.0));
                    Angles {
                        azimuth,
                        dip,
                        rake,
                        major: 1.0,
                        semi,
                        minor,
                    }
                });
                let spec = RbfSpec {
                    kernel: kernel(&self.kernel)?,
                    drift_degree: degree,
                    smoothing: self.smoothing,
                    anisotropy,
                    ..RbfSpec::default()
                };
                py.detach(|| Rbf::fit_constraints(&set, &spec))
                    .map(Fitted::Rbf)
            }
            "kriging" => {
                let vg = self.variogram.as_ref().expect("checked").0.clone();
                let spec = HermiteSpec {
                    drift_degree: degree,
                    ..HermiteSpec::default()
                };
                py.detach(|| HermiteKriging::new(&set, &vg, &spec))
                    .map(Fitted::Kriging)
            }
            _ => {
                let (azimuth, dip, rake) = self.rotation.unwrap_or_default();
                let spec = SvgpSpec {
                    drift_degree: degree,
                    rotation: [azimuth, dip, rake],
                    ..SvgpSpec::default()
                };
                py.detach(|| Svgp::fit(&set, &spec)).map(Fitted::Gp)
            }
        }
        .map_err(err)?;
        self.fitted = Some(fitted);
        self.inputs = inputs;
        Ok(())
    }
}

impl Tabular for ImplicitModel {
    /// `x`, `y`, `z`, then `value` for samples, `dip` and `dip_direction` for
    /// planes, `plunge` and `trend` for lineations; a boundary row has none.
    fn columns(&self) -> Option<Columns> {
        self.fitted.as_ref()?;
        let rows = &self.inputs;
        let column = |name: &str, f: fn(Reading) -> Option<f64>| {
            (name.to_string(), rows.iter().map(|&(_, r)| f(r)).collect())
        };
        let mut columns = persist::point_columns(rows.iter().map(|(p, _)| (p[0], p[1], p[2])));
        columns.extend([
            column("value", |r| match r {
                Reading::Sample(v) => Some(v),
                _ => None,
            }),
            column("dip", |r| match r {
                Reading::Plane(p) => Some(p.dip),
                _ => None,
            }),
            column("dip_direction", |r| match r {
                Reading::Plane(p) => Some(p.dip_direction),
                _ => None,
            }),
            column("plunge", |r| match r {
                Reading::Lineation(l) => Some(l.plunge),
                _ => None,
            }),
            column("trend", |r| match r {
                Reading::Lineation(l) => Some(l.trend),
                _ => None,
            }),
        ]);
        Some(columns)
    }

    fn restore(&mut self, columns: Found) -> PyResult<()> {
        self.check()?;
        let names = ["value", "dip", "dip_direction", "plunge", "trend"];
        let found = names
            .iter()
            .map(|n| columns.optional(n))
            .collect::<PyResult<Vec<_>>>()?;
        let inputs = columns
            .points()?
            .into_iter()
            .enumerate()
            .map(|(i, (x, y, z))| {
                let reading = match found.iter().map(|c| c[i]).collect::<Vec<_>>()[..] {
                    [Some(v), None, None, None, None] => Reading::Sample(v),
                    [None, None, None, None, None] => Reading::Boundary,
                    [None, Some(dip), Some(dip_direction), None, None] => {
                        Reading::Plane(Plane { dip, dip_direction })
                    }
                    [None, None, None, Some(plunge), Some(trend)] => {
                        Reading::Lineation(Lineation { plunge, trend })
                    }
                    _ => return Err(invalid(format!("row {i} mixes constraint kinds"))),
                };
                Ok(([x, y, z], reading))
            })
            .collect::<PyResult<_>>()?;
        Python::attach(|py| self.fit_inputs(py, inputs))
    }
}

fn readings(obj: Option<&Bound<PyAny>>, what: &str) -> PyResult<Vec<[f64; 5]>> {
    let Some(obj) = obj else {
        return Ok(Vec::new());
    };
    rows(obj, what)?
        .into_iter()
        .map(|r| {
            r.try_into()
                .map_err(|_| invalid(format!("{what} must have shape (k, 5)")))
        })
        .collect()
}

#[pymethods]
impl ImplicitModel {
    #[new]
    #[pyo3(signature = (engine="rbf", *, kernel="biharmonic", variogram=None, degree=1, smoothing=0.0, rotation=None, ratios=None))]
    fn new(
        engine: &str,
        kernel: &str,
        variogram: Option<Variogram>,
        degree: usize,
        smoothing: f64,
        rotation: Option<(f64, f64, f64)>,
        ratios: Option<(f64, f64)>,
    ) -> PyResult<Self> {
        let model = Self {
            engine: engine.to_string(),
            kernel: kernel.to_string(),
            variogram,
            degree,
            smoothing,
            rotation,
            ratios,
            fitted: None,
            inputs: Vec::new(),
        };
        model.check()?;
        Ok(model)
    }

    /// Writes the fit inputs as Parquet columns and parameters as JSON in the
    /// file metadata; `from_parquet` refits them to the same field.
    fn to_parquet(&self, path: std::path::PathBuf) -> PyResult<()> {
        persist::to_parquet(<Self as pyo3::PyClass>::NAME, self, &path)
    }

    /// Reads `to_parquet` output; raises InvalidInput on another class's file
    /// or a newer format.
    #[staticmethod]
    fn from_parquet(path: std::path::PathBuf) -> PyResult<Self> {
        let model: Self = persist::from_parquet(<Self as pyo3::PyClass>::NAME, &path)?;
        model.check()?;
        Ok(model)
    }

    fn _state(&self) -> PyResult<(String, Option<Columns>)> {
        persist::state(<Self as pyo3::PyClass>::NAME, self)
    }

    #[staticmethod]
    fn _from_state(meta: &str, columns: Option<Columns>) -> PyResult<Self> {
        let model: Self = persist::from_state(<Self as pyo3::PyClass>::NAME, meta, columns)?;
        model.check()?;
        Ok(model)
    }

    /// Parameters
    /// ----------
    /// coords, values : array_like, optional
    ///     Samples and their field values, e.g. +1 inside, -1 outside.
    /// cutoff : float, optional
    ///     Take `values` as grades coded +1 at or above `cutoff` and -1 below,
    ///     so the shell is the zero level.
    /// boundaries : array_like, optional
    ///     ``(m, 3)`` points on the surface, pinned to field value 0.
    /// planes : array_like, optional
    ///     ``(k, 5)`` rows ``x, y, z, dip, dip_direction``; the field is flat
    ///     along each plane.
    /// lineations : array_like, optional
    ///     ``(k, 5)`` rows ``x, y, z, plunge, trend``; the field is flat along
    ///     each line.
    #[pyo3(signature = (coords=None, values=None, *, cutoff=None, boundaries=None, planes=None, lineations=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: Option<&Bound<PyAny>>,
        values: Option<&Bound<PyAny>>,
        cutoff: Option<f64>,
        boundaries: Option<&Bound<PyAny>>,
        planes: Option<&Bound<PyAny>>,
        lineations: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if cutoff.is_some_and(|c| !c.is_finite()) {
            return Err(invalid("cutoff must be finite"));
        }
        let driver = cutoff.map_or(Driver::Value, |threshold| Driver::Indicator { threshold });
        let mut inputs = Vec::new();
        match (coords, values) {
            (Some(c), Some(v)) => {
                let (locs, values) = (points(c)?, finite(v, "values")?);
                same_length(locs.len(), values.len(), "values")?;
                let keep = distinct(slf.py(), &locs, None)?;
                for ((x, y, z), v) in pick(&locs, &keep).into_iter().zip(pick(&values, &keep)) {
                    inputs.push(([x, y, z], Reading::Sample(driver.encode(v))));
                }
            }
            (None, None) => {}
            _ => return Err(invalid("coords and values go together")),
        }
        for (x, y, z) in boundaries.map(points).transpose()?.unwrap_or_default() {
            inputs.push(([x, y, z], Reading::Boundary));
        }
        for [x, y, z, dip, dip_direction] in readings(planes, "planes")? {
            inputs.push(([x, y, z], Reading::Plane(Plane { dip, dip_direction })));
        }
        for [x, y, z, plunge, trend] in readings(lineations, "lineations")? {
            inputs.push(([x, y, z], Reading::Lineation(Lineation { plunge, trend })));
        }
        let py = slf.py();
        slf.fit_inputs(py, inputs)?;
        Ok(slf)
    }

    /// Field values at `targets`, with `(n, 3)` gradients when `gradient`, or
    /// the predictive variance of ``engine="gp"`` when `variance`.
    #[pyo3(signature = (targets, *, gradient=false, variance=false))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        targets: &Bound<PyAny>,
        gradient: bool,
        variance: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let fitted = self.fitted()?;
        let pts: Vec<[f64; 3]> = self::targets(targets)?
            .into_iter()
            .map(|(x, y, z)| [x, y, z])
            .collect();
        let values = py.detach(|| pts.par_iter().map(|p| fitted.value(p)).collect());
        if variance {
            let Fitted::Gp(gp) = fitted else {
                return Err(invalid("only engine=\"gp\" has a variance"));
            };
            if gradient {
                return Err(invalid("engine=\"gp\" has no gradient"));
            }
            let var = py.detach(|| pts.par_iter().map(|p| gp.variance(p)).collect());
            return Ok((array1(py, values), array1(py, var))
                .into_pyobject(py)?
                .into_any());
        }
        if !gradient {
            return Ok(array1(py, values).into_any());
        }
        let grads: Vec<Vec<f64>> = match fitted {
            Fitted::Rbf(f) => {
                py.detach(|| pts.par_iter().map(|p| f.gradient(p).to_vec()).collect())
            }
            Fitted::Kriging(f) => {
                py.detach(|| pts.par_iter().map(|p| f.gradient(p).to_vec()).collect())
            }
            Fitted::Gp(_) => return Err(invalid("engine=\"gp\" has no gradient")),
        };
        Ok((array1(py, values), array2(py, &grads))
            .into_pyobject(py)?
            .into_any())
    }

    /// Mesh of the `isovalue` surface, the field sampled at the block
    /// centroids of `model`. Blocks a masked model leaves out are outside
    /// the solid. `closed` caps the solid where the field exceeds `isovalue`
    /// on the block model's outer faces and at the edge of its mask.
    #[pyo3(signature = (model, *, isovalue=0.0, closed=false))]
    fn isosurface<'py>(
        &self,
        py: Python<'py>,
        model: PyRef<PyBlockModel>,
        isovalue: f64,
        closed: bool,
    ) -> PyResult<Mesh> {
        let fitted = self.fitted()?;
        let g = *model.0.geometry();
        let active = match model.0.layout() {
            ceres_core::Layout::Regular => None,
            ceres_core::Layout::Masked(index) => {
                let mut on = vec![false; g.cells() as usize];
                for &i in index {
                    on[i as usize] = true;
                }
                Some(on)
            }
            ceres_core::Layout::SubBlocked { .. } => {
                return Err(invalid("isosurface needs a regular or masked BlockModel"));
            }
        };
        let cells: Vec<f64> = py.detach(|| {
            (0..g.cells())
                .into_par_iter()
                .map(|n| {
                    if closed || active.as_ref().is_none_or(|a| a[n as usize]) {
                        fitted.value(&g.centroid(n))
                    } else {
                        f64::NAN
                    }
                })
                .collect()
        });
        let grid = ScalarGrid::blocks(g.size, g.count, &cells, active.as_deref(), isovalue, closed)
            .map_err(invalid)?;
        let mesh = py.detach(|| marching_tetrahedra(&grid, isovalue));
        let frame = ceres_core::block_frame(g.rotation);
        let vertices: Vec<f64> = mesh
            .vertices
            .iter()
            .flat_map(|v| {
                (0..3).map(move |a| g.origin[a] + (0..3).map(|b| frame[(b, a)] * v[b]).sum::<f64>())
            })
            .collect();
        let mut mesh = Mesh::build(&vertices, &mesh.triangles)?;
        mesh.crs = model.0.crs.clone();
        Ok(Mesh::from_core(mesh))
    }

    /// Fit diagnostics of `engine="gp"`, else `None`.
    #[getter]
    fn report<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(Fitted::Gp(gp)) = &self.fitted else {
            return Ok(None);
        };
        let r = gp.report();
        let d = PyDict::new(py);
        d.set_item("status", r.status.as_str())?;
        d.set_item("iterations", r.iterations)?;
        d.set_item("elbo", r.elbo)?;
        d.set_item("gradient_norm", r.gradient_norm)?;
        d.set_item("inducing", r.inducing)?;
        d.set_item("observations", r.observations)?;
        d.set_item("signal_variance", r.signal_variance)?;
        d.set_item("noise_variance", r.noise_variance)?;
        d.set_item("lengthscales", r.lengthscales)?;
        Ok(Some(d))
    }

    fn __repr__(&self) -> String {
        format!("ImplicitModel(engine={:?})", self.engine)
    }
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<ImplicitModel>()?;
    Ok(())
}
