use std::collections::HashMap;

use estimation::search::SearchTree;
use estimation::{CoKind, CoSample, Estimate, GaussianSample, Sample, Search as CoreSearch};
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use variogram::{Anisotropy, Coregionalization as CoreCoreg, Variogram as CoreVariogram};

use crate::args::{self, Point, array1, distinct, finite, pick, points, same_length};
use crate::estimation::{Search, outputs, sample_columns, samples_from, targets};
use crate::invalid;
use crate::persist::{self, Columns, Found, Tabular};
use crate::transforms::Anamorphosis;
use crate::variogram::{Coregionalization, Variogram};

fn metric(anisotropy: Option<Anisotropy>) -> CoreVariogram {
    CoreVariogram {
        nugget: 0.0,
        structures: vec![],
        anisotropy,
    }
}

fn nearby<T: Clone>(target: &Point, tree: &SearchTree, items: &[T]) -> Option<Vec<T>> {
    let chosen = tree.neighbors(target).ok()?;
    Some(chosen.iter().map(|&i| items[i].clone()).collect())
}

/// Cokriging of `variable` from samples of several variables under a linear
/// model of coregionalization; `means` switches to simple cokriging.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "ceres", name = "Cokriging")]
pub struct Cokriging {
    model: CoreCoreg,
    search: CoreSearch,
    kind: CoKind,
    #[serde(skip)]
    samples: Option<(Vec<Sample>, Vec<CoSample>)>,
}

#[pymethods]
impl Cokriging {
    /// Writes arrays as Parquet columns and parameters as JSON in the file
    /// metadata; `from_parquet` reads it back.
    fn to_parquet(&self, path: std::path::PathBuf) -> PyResult<()> {
        crate::persist::to_parquet(<Self as pyo3::PyClass>::NAME, self, &path)
    }

    /// Reads `to_parquet` output; raises InvalidInput on another class's file
    /// or a newer format.
    #[staticmethod]
    fn from_parquet(path: std::path::PathBuf) -> PyResult<Self> {
        crate::persist::from_parquet(<Self as pyo3::PyClass>::NAME, &path)
    }

    fn _state(&self) -> PyResult<(String, Option<crate::persist::Columns>)> {
        crate::persist::state(<Self as pyo3::PyClass>::NAME, self)
    }

    #[staticmethod]
    fn _from_state(meta: &str, columns: Option<crate::persist::Columns>) -> PyResult<Self> {
        crate::persist::from_state(<Self as pyo3::PyClass>::NAME, meta, columns)
    }

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
            search: search.plain("Cokriging")?,
            kind,
            samples: None,
        })
    }

    /// `variables` gives each sample's variable index. Samples of one variable
    /// sharing a location keep the first, with a warning naming their `holes`.
    #[pyo3(signature = (coords, values, variables, holes=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        values: &Bound<PyAny>,
        variables: Vec<usize>,
        holes: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (locs, values) = (points(coords)?, finite(values, "values")?);
        same_length(locs.len(), values.len(), "values")?;
        same_length(locs.len(), variables.len(), "variables")?;
        if variables.iter().any(|&v| v >= slf.model.nvar) {
            return Err(invalid("variable index out of range"));
        }
        let holes = args::holes(holes, locs.len())?.map(|h| h.0);
        let mut keep = Vec::new();
        for k in 0..slf.model.nvar {
            let rows: Vec<usize> = (0..locs.len()).filter(|&i| variables[i] == k).collect();
            let labels = holes.as_ref().map(|h| pick(h, &rows));
            let kept = distinct(slf.py(), &pick(&locs, &rows), labels.as_deref())?;
            keep.extend(kept.into_iter().map(|j| rows[j]));
        }
        keep.sort_unstable();
        let (locs, values, variables) = (
            pick(&locs, &keep),
            pick(&values, &keep),
            pick(&variables, &keep),
        );
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
        let tree = SearchTree::new(plain, &self.search, Some(&metric));
        let results: Vec<Option<Estimate>> = py.detach(|| {
            targets
                .par_iter()
                .enumerate()
                .map(|(i, t)| {
                    let near = nearby(t, &tree, co)?;
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
#[derive(Serialize, Deserialize)]
#[pyclass(module = "ceres", name = "DisjunctiveKriging")]
pub struct Disjunctive {
    engine: estimation::DisjunctiveKriging,
    variogram: CoreVariogram,
    search: CoreSearch,
    order: usize,
    #[serde(skip)]
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
        let tree = SearchTree::new(plain, &self.search, Some(&metric));
        Ok(py.detach(|| {
            targets
                .par_iter()
                .map(|t| {
                    let near = nearby(t, &tree, gauss)?;
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
    /// Writes arrays as Parquet columns and parameters as JSON in the file
    /// metadata; `from_parquet` reads it back.
    fn to_parquet(&self, path: std::path::PathBuf) -> PyResult<()> {
        crate::persist::to_parquet(<Self as pyo3::PyClass>::NAME, self, &path)
    }

    /// Reads `to_parquet` output; raises InvalidInput on another class's file
    /// or a newer format.
    #[staticmethod]
    fn from_parquet(path: std::path::PathBuf) -> PyResult<Self> {
        crate::persist::from_parquet(<Self as pyo3::PyClass>::NAME, &path)
    }

    fn _state(&self) -> PyResult<(String, Option<crate::persist::Columns>)> {
        crate::persist::state(<Self as pyo3::PyClass>::NAME, self)
    }

    #[staticmethod]
    fn _from_state(meta: &str, columns: Option<crate::persist::Columns>) -> PyResult<Self> {
        crate::persist::from_state(<Self as pyo3::PyClass>::NAME, meta, columns)
    }

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
            search: search.plain("DisjunctiveKriging")?,
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
        let keep = distinct(slf.py(), &locs, None)?;
        let (locs, values) = (pick(&locs, &keep), pick(&values, &keep));
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

impl Tabular for Cokriging {
    fn columns(&self) -> Option<Columns> {
        let (plain, co) = self.samples.as_ref()?;
        let mut columns = sample_columns(plain);
        columns.push(persist::column("variable", co.iter().map(|s| s.var as f64)));
        Some(columns)
    }

    fn restore(&mut self, columns: Found) -> PyResult<()> {
        let plain = samples_from(&columns)?;
        let variables = columns.indices("variable")?;
        same_length(plain.len(), variables.len(), "variable")?;
        if variables.iter().any(|&v| v >= self.model.nvar) {
            return Err(invalid("variable index out of range"));
        }
        let co = plain
            .iter()
            .zip(variables)
            .map(|(s, k)| CoSample::new(s.loc, k, s.value))
            .collect();
        self.samples = Some((plain, co));
        Ok(())
    }
}

impl Tabular for Disjunctive {
    fn columns(&self) -> Option<Columns> {
        Some(sample_columns(&self.samples.as_ref()?.0))
    }

    fn restore(&mut self, columns: Found) -> PyResult<()> {
        let plain = samples_from(&columns)?;
        let anam = self.engine.anamorphosis();
        let gauss = plain
            .iter()
            .map(|s| GaussianSample {
                loc: s.loc,
                y: anam.forward(s.value),
            })
            .collect();
        self.samples = Some((plain, gauss));
        Ok(())
    }
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Cokriging>()?;
    m.add_class::<Disjunctive>()?;
    Ok(())
}
