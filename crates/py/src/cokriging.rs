use std::collections::HashMap;

use estimation::{CoKind, CoSample, Estimate, GaussianSample, Sample, Search as CoreSearch};
use pyo3::prelude::*;
use rayon::prelude::*;
use variogram::{Anisotropy, Coregionalization as CoreCoreg, Variogram as CoreVariogram};

use crate::args::{Point, array1, finite, points, same_length};
use crate::estimation::{Search, outputs, targets};
use crate::invalid;
use crate::transforms::Anamorphosis;
use crate::variogram::{Coregionalization, Variogram};

fn metric(anisotropy: Option<Anisotropy>) -> CoreVariogram {
    CoreVariogram {
        nugget: 0.0,
        structures: vec![],
        anisotropy,
    }
}

fn nearby<T: Clone>(
    target: &Point,
    locs: &[Sample],
    items: &[T],
    search: &CoreSearch,
    metric: &CoreVariogram,
) -> Option<Vec<T>> {
    let chosen = estimation::neighbors(target, locs, search, Some(metric)).ok()?;
    Some(chosen.iter().map(|&i| items[i].clone()).collect())
}

/// Cokriging of `variable` from samples of several variables under a linear
/// model of coregionalization; `means` switches to simple cokriging.
#[pyclass(module = "ceres", name = "Cokriging")]
pub struct Cokriging {
    model: CoreCoreg,
    search: CoreSearch,
    kind: CoKind,
    samples: Option<(Vec<Sample>, Vec<CoSample>)>,
}

#[pymethods]
impl Cokriging {
    #[new]
    #[pyo3(signature = (coregionalization, search, means=None))]
    fn new(
        coregionalization: PyRef<Coregionalization>,
        search: Search,
        means: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let model = coregionalization.0.clone();
        let kind = match means {
            Some(m) if m.len() == model.nvar => CoKind::Simple { means: m },
            Some(_) => return Err(invalid("need one mean per variable")),
            None => CoKind::Ordinary,
        };
        Ok(Self {
            model,
            search: search.0,
            kind,
            samples: None,
        })
    }

    /// `variables` gives each sample's variable index.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        values: &Bound<PyAny>,
        variables: Vec<usize>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (locs, values) = (points(coords)?, finite(values, "values")?);
        same_length(locs.len(), values.len(), "values")?;
        same_length(locs.len(), variables.len(), "variables")?;
        if variables.iter().any(|&v| v >= slf.model.nvar) {
            return Err(invalid("variable index out of range"));
        }
        let plain = locs
            .iter()
            .zip(&values)
            .map(|(&l, &v)| Sample::new(l, v))
            .collect();
        let co = locs
            .iter()
            .zip(&values)
            .zip(&variables)
            .map(|((&l, &v), &k)| CoSample::new(l, k, v))
            .collect();
        slf.samples = Some((plain, co));
        Ok(slf)
    }

    /// Estimates of `variable`; `collocated` maps a variable index to its
    /// values at every target for collocated cokriging.
    #[pyo3(signature = (targets, variable=0, return_variance=false, collocated=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        targets: &Bound<'py, PyAny>,
        variable: usize,
        return_variance: bool,
        collocated: Option<HashMap<usize, Bound<'py, PyAny>>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (plain, co) = self
            .samples
            .as_ref()
            .ok_or_else(|| invalid("Cokriging is not fitted; call fit first"))?;
        if variable >= self.model.nvar {
            return Err(invalid("variable index out of range"));
        }
        let targets = targets_of(targets)?;
        let collocated: Vec<(usize, Vec<f64>)> = collocated
            .unwrap_or_default()
            .into_iter()
            .map(|(k, v)| {
                let v = finite(&v, "collocated")?;
                same_length(targets.len(), v.len(), "collocated")?;
                Ok((k, v))
            })
            .collect::<PyResult<_>>()?;
        let metric = metric(self.model.anisotropy.clone());
        let results: Vec<Option<Estimate>> = py.detach(|| {
            targets
                .par_iter()
                .enumerate()
                .map(|(i, t)| {
                    let near = nearby(t, plain, co, &self.search, &metric)?;
                    if collocated.is_empty() {
                        return estimation::cokrige(t, variable, &near, &self.model, &self.kind)
                            .ok();
                    }
                    let here: Vec<(usize, f64)> =
                        collocated.iter().map(|(k, v)| (*k, v[i])).collect();
                    estimation::collocated_cokrige(
                        t,
                        variable,
                        &near,
                        &here,
                        &self.model,
                        &self.kind,
                    )
                    .ok()
                })
                .collect()
        });
        outputs(py, &results, return_variance)
    }
}

fn targets_of(obj: &Bound<PyAny>) -> PyResult<Vec<Point>> {
    targets(obj)
}

/// Disjunctive kriging: simple kriging of the Hermite factors of the Gaussian
/// transform, giving local grades and proportions above cutoffs.
#[pyclass(module = "ceres", name = "DisjunctiveKriging")]
pub struct Disjunctive {
    engine: estimation::DisjunctiveKriging,
    variogram: CoreVariogram,
    search: CoreSearch,
    order: usize,
    samples: Option<(Vec<Sample>, Vec<GaussianSample>)>,
}

impl Disjunctive {
    fn factors(&self, py: Python, targets: &Bound<PyAny>) -> PyResult<Vec<Option<Vec<f64>>>> {
        let (plain, gauss) = self
            .samples
            .as_ref()
            .ok_or_else(|| invalid("DisjunctiveKriging is not fitted; call fit first"))?;
        let targets = targets_of(targets)?;
        let metric = metric(self.variogram.anisotropy.clone());
        Ok(py.detach(|| {
            targets
                .par_iter()
                .map(|t| {
                    let near = nearby(t, plain, gauss, &self.search, &metric)?;
                    self.engine
                        .factors(t, &near, &self.variogram, self.order)
                        .ok()
                })
                .collect()
        }))
    }
}

#[pymethods]
impl Disjunctive {
    /// `anamorphosis` is fitted on the raw values; `variogram` is that of the
    /// Gaussian scores.
    #[new]
    #[pyo3(signature = (anamorphosis, variogram, search, order=20))]
    fn new(
        anamorphosis: PyRef<Anamorphosis>,
        variogram: Variogram,
        search: Search,
        order: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            engine: estimation::DisjunctiveKriging::new(anamorphosis.inner()?),
            variogram: variogram.0,
            search: search.0,
            order,
            samples: None,
        })
    }

    /// Raw values; they are Gaussian-transformed with the anamorphosis.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        values: &Bound<PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (locs, values) = (points(coords)?, finite(values, "values")?);
        same_length(locs.len(), values.len(), "values")?;
        let anam = slf.engine.anamorphosis().clone();
        let plain = locs
            .iter()
            .zip(&values)
            .map(|(&l, &v)| Sample::new(l, v))
            .collect();
        let gauss = locs
            .iter()
            .zip(&values)
            .map(|(&loc, &z)| GaussianSample {
                loc,
                y: anam.forward(z),
            })
            .collect();
        slf.samples = Some((plain, gauss));
        Ok(slf)
    }

    /// Local grade estimates.
    fn predict<'py>(&self, py: Python<'py>, targets: &Bound<PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let grades = self
            .factors(py, targets)?
            .iter()
            .map(|f| f.as_ref().map_or(f64::NAN, |f| self.engine.grade(f)))
            .collect();
        Ok(array1(py, grades).into_any())
    }

    /// Local proportion above `cutoff`.
    fn predict_tonnage<'py>(
        &self,
        py: Python<'py>,
        targets: &Bound<PyAny>,
        cutoff: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let t = self
            .factors(py, targets)?
            .iter()
            .map(|f| {
                f.as_ref()
                    .map_or(f64::NAN, |f| self.engine.tonnage(f, cutoff))
            })
            .collect();
        Ok(array1(py, t).into_any())
    }
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Cokriging>()?;
    m.add_class::<Disjunctive>()?;
    Ok(())
}
