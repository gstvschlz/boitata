use modeling::{
    ConstraintSet, HermiteKriging, HermiteSpec, Kernel, Lineation, Plane, PlaneEncoding, Rbf,
    RbfSpec, ScalarGrid, Svgp, SvgpSpec, marching_tetrahedra,
};
use numpy::IntoPyArray;
use numpy::ndarray::Array2;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use variogram::Angles;

use crate::args::{array1, array2, distinct, finite, pick, points, rows, same_length};
use crate::containers::PyBlockModel;
use crate::estimation::targets;
use crate::invalid;
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
/// drift_degree : int
///     0 constant, 1 linear drift.
/// smoothing : float
///     RBF ridge relative to the value scale; 0 interpolates exactly.
/// rotation : tuple of float, optional
///     Azimuth, dip, rake of the anisotropy axes. With ``engine="gp"`` the
///     ranges along them are learned; ``engine="kriging"`` takes its
///     anisotropy from the variogram.
/// ratios : tuple of float, optional
///     Semi-major/major and minor/major range ratios for ``engine="rbf"``.
#[pyclass(module = "ceres", name = "ImplicitModel")]
pub struct ImplicitModel {
    engine: String,
    kernel: Kernel,
    variogram: Option<Variogram>,
    drift_degree: usize,
    smoothing: f64,
    rotation: Option<(f64, f64, f64)>,
    ratios: Option<(f64, f64)>,
    fitted: Option<Fitted>,
}

impl ImplicitModel {
    fn fitted(&self) -> PyResult<&Fitted> {
        self.fitted
            .as_ref()
            .ok_or_else(|| invalid("ImplicitModel is not fitted; call fit first"))
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
    #[pyo3(signature = (engine="rbf", kernel="biharmonic", variogram=None, drift_degree=1, smoothing=0.0, rotation=None, ratios=None))]
    fn new(
        engine: &str,
        kernel: &str,
        variogram: Option<Variogram>,
        drift_degree: usize,
        smoothing: f64,
        rotation: Option<(f64, f64, f64)>,
        ratios: Option<(f64, f64)>,
    ) -> PyResult<Self> {
        let kernel = [Kernel::Biharmonic, Kernel::Triharmonic, Kernel::ThinPlate]
            .into_iter()
            .find(|k| k.label() == kernel)
            .ok_or_else(|| {
                invalid(format!(
                    "kernel must be biharmonic, triharmonic or thin-plate, not {kernel:?}"
                ))
            })?;
        match engine {
            "rbf" | "gp" => {}
            "kriging" if variogram.is_some() => {}
            "kriging" => return Err(invalid("engine=\"kriging\" needs a variogram")),
            other => {
                return Err(invalid(format!(
                    "engine must be rbf, kriging or gp, not {other:?}"
                )));
            }
        }
        if !(smoothing.is_finite() && smoothing >= 0.0) {
            return Err(invalid("smoothing must be >= 0"));
        }
        match engine {
            "kriging" if rotation.is_some() || ratios.is_some() => {
                return Err(invalid(
                    "engine=\"kriging\" takes its anisotropy from the variogram",
                ));
            }
            "gp" if ratios.is_some() => {
                return Err(invalid(
                    "engine=\"gp\" learns its ranges; give rotation only",
                ));
            }
            _ => {}
        }
        if ratios.is_some_and(|(a, b)| !(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite())) {
            return Err(invalid("ratios must be > 0"));
        }
        Ok(Self {
            engine: engine.to_string(),
            kernel,
            variogram,
            drift_degree,
            smoothing,
            rotation,
            ratios,
            fitted: None,
        })
    }

    /// Parameters
    /// ----------
    /// coords, values : array_like, optional
    ///     Samples and their field values, e.g. +1 inside, -1 outside.
    /// boundaries : array_like, optional
    ///     ``(m, 3)`` points on the surface, pinned to field value 0.
    /// planes : array_like, optional
    ///     ``(k, 5)`` rows ``x, y, z, dip, dip_direction``; the field is flat
    ///     along each plane.
    /// lineations : array_like, optional
    ///     ``(k, 5)`` rows ``x, y, z, plunge, trend``; the field is flat along
    ///     each line.
    #[pyo3(signature = (coords=None, values=None, boundaries=None, planes=None, lineations=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: Option<&Bound<PyAny>>,
        values: Option<&Bound<PyAny>>,
        boundaries: Option<&Bound<PyAny>>,
        planes: Option<&Bound<PyAny>>,
        lineations: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let err = |e: modeling::ModelError| invalid(e);
        let mut set = ConstraintSet::new();
        match (coords, values) {
            (Some(c), Some(v)) => {
                let (locs, values) = (points(c)?, finite(v, "values")?);
                same_length(locs.len(), values.len(), "values")?;
                let keep = distinct(slf.py(), &locs, None)?;
                for ((x, y, z), v) in pick(&locs, &keep).into_iter().zip(pick(&values, &keep)) {
                    set.push_sample([x, y, z], v);
                }
            }
            (None, None) => {}
            _ => return Err(invalid("coords and values go together")),
        }
        for (x, y, z) in boundaries.map(points).transpose()?.unwrap_or_default() {
            set.push_boundary([x, y, z], 0.0);
        }
        for [x, y, z, dip, dip_direction] in readings(planes, "planes")? {
            let plane = Plane { dip, dip_direction };
            set.push_plane([x, y, z], plane, PlaneEncoding::Tangents, 0.0)
                .map_err(err)?;
        }
        for [x, y, z, plunge, trend] in readings(lineations, "lineations")? {
            set.push_lineation([x, y, z], Lineation { plunge, trend })
                .map_err(err)?;
        }
        set.validate().map_err(err)?;
        let degree = slf.drift_degree;
        let fitted = match slf.engine.as_str() {
            "rbf" => {
                let anisotropy = (slf.rotation.is_some() || slf.ratios.is_some()).then(|| {
                    let (azimuth, dip, rake) = slf.rotation.unwrap_or_default();
                    let (semi, minor) = slf.ratios.unwrap_or((1.0, 1.0));
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
                    kernel: slf.kernel,
                    drift_degree: degree,
                    smoothing: slf.smoothing,
                    anisotropy,
                    ..RbfSpec::default()
                };
                slf.py()
                    .detach(|| Rbf::fit_constraints(&set, &spec))
                    .map(Fitted::Rbf)
            }
            "kriging" => {
                let vg = slf.variogram.as_ref().expect("checked in new").0.clone();
                let spec = HermiteSpec {
                    drift_degree: degree,
                    ..HermiteSpec::default()
                };
                slf.py()
                    .detach(|| HermiteKriging::new(&set, &vg, &spec))
                    .map(Fitted::Kriging)
            }
            _ => {
                let (azimuth, dip, rake) = slf.rotation.unwrap_or_default();
                let spec = SvgpSpec {
                    drift_degree: degree,
                    rotation: [azimuth, dip, rake],
                    ..SvgpSpec::default()
                };
                slf.py().detach(|| Svgp::fit(&set, &spec)).map(Fitted::Gp)
            }
        }
        .map_err(err)?;
        slf.fitted = Some(fitted);
        Ok(slf)
    }

    /// Field values at `targets`, with `(n, 3)` gradients when `gradient`, or
    /// the predictive variance of ``engine="gp"`` when `variance`.
    #[pyo3(signature = (targets, gradient=false, variance=false))]
    fn evaluate<'py>(
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

    /// `(vertices, triangles)` of the `isovalue` surface, the field sampled at
    /// the block centroids of `block_model`. `closed` caps the solid where the
    /// field exceeds `isovalue` on the block model's outer faces.
    #[pyo3(signature = (block_model, isovalue=0.0, closed=false))]
    fn isosurface<'py>(
        &self,
        py: Python<'py>,
        block_model: PyRef<PyBlockModel>,
        isovalue: f64,
        closed: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let fitted = self.fitted()?;
        let g = *block_model.0.geometry();
        let pad = closed as usize;
        let counts = g.count.map(|n| n + 2 * pad);
        let values = py.detach(|| {
            (0..counts.iter().product::<usize>())
                .into_par_iter()
                .map(|n| {
                    let ijk = [
                        n % counts[0],
                        n / counts[0] % counts[1],
                        n / (counts[0] * counts[1]),
                    ];
                    let inner = [0, 1, 2].map(|a| ijk[a].clamp(pad, g.count[a] + pad - 1) - pad);
                    let v = fitted.value(&g.centroid(g.index(inner)));
                    if inner.map(|i| i + pad) == ijk {
                        v
                    } else {
                        isovalue - (v - isovalue).abs()
                    }
                })
                .collect()
        });
        let origin = g.size.map(|s| s * (0.5 - pad as f64));
        let grid = ScalarGrid::new(origin, g.size, counts, values).map_err(invalid)?;
        let mesh = py.detach(|| marching_tetrahedra(&grid, isovalue));
        let frame = ceres_core::block_frame(g.rotation);
        let vertices: Vec<Vec<f64>> = mesh
            .vertices
            .iter()
            .map(|v| {
                (0..3)
                    .map(|a| g.origin[a] + (0..3).map(|b| frame[(b, a)] * v[b]).sum::<f64>())
                    .collect()
            })
            .collect();
        let triangles = Array2::from_shape_vec(
            (mesh.triangle_count(), 3),
            mesh.triangles.iter().map(|&t| t as i64).collect(),
        )
        .expect("m x 3")
        .into_pyarray(py);
        Ok((array2(py, &vertices), triangles)
            .into_pyobject(py)?
            .into_any())
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
