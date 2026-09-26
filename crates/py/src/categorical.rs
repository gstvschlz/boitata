use estimation::{
    CategoricalIndicator, CategoricalIndicatorSummary as CoreSummary, Sample, Search as CoreSearch,
};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::args::{self, Label, array1, column, optional_finite, pick, points, same_length};
use crate::categories::{Categories, labels, refs};
use crate::estimation::{Search, codes, sample_columns, samples_from, searches, targets};
use crate::indicator::{
    diagnostic_columns, diagnostics_table, push_diagnostics, restore_diagnostics,
};
use crate::invalid;
use crate::persist::{self, Columns, Found, Tabular};
use crate::table::Table;
use crate::variogram::Variogram;

/// Indicator kriging of categories: the probability of each category kriged
/// from its indicator, clipped to [0, 1] and rescaled to sum 1 at each
/// target.
///
/// Parameters
/// ----------
/// variograms : Variogram or sequence of Variogram
///     One indicator variogram per category, in code order, or a single one
///     shared by all (one kriging system per target).
/// search : Search or sequence of Search
///     Neighborhood shared by all categories; a sequence is searched in
///     passes, each filling the targets the previous left unestimated. With
///     no ellipsoid, the middle category's variogram orients it.
/// simple : bool
///     Simple kriging with the declustered global proportions as means;
///     ordinary kriging otherwise.
/// scheme : Categories, optional
///     Names and colors of the categories: `fit` then takes labels, encoded
///     by it. Without it, `fit` takes codes ``0..k``.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "ceres", name = "CategoricalIndicatorKriging")]
pub struct CategoricalIndicatorKriging {
    model: CategoricalIndicator,
    search: Vec<Search>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    scheme: Option<Categories>,
    /// Labels of the fitted domains, indexed by code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    domains: Option<Vec<Label>>,
    #[serde(skip)]
    samples: Option<(Vec<Sample>, Option<Vec<f64>>)>,
}

#[pymethods]
impl CategoricalIndicatorKriging {
    #[new]
    #[pyo3(signature = (variograms, search, *, simple=false, scheme=None))]
    fn new(
        variograms: &Bound<PyAny>,
        search: &Bound<PyAny>,
        simple: bool,
        scheme: Option<Categories>,
    ) -> PyResult<Self> {
        let variograms: Vec<Variogram> = match variograms.extract::<Variogram>() {
            Ok(v) => vec![v],
            Err(_) => variograms.extract().map_err(|_| {
                invalid("variograms must be a Variogram or a sequence of Variogram")
            })?,
        };
        let n = variograms.len();
        let categories = scheme.as_ref().map_or(n, |s| s.0.len());
        let model = CategoricalIndicator {
            categories,
            variograms: variograms.into_iter().map(|v| v.0).collect(),
            simple,
        };
        model.validate().map_err(invalid)?;
        Ok(Self {
            model,
            search: searches(search)?,
            scheme,
            domains: None,
            samples: None,
        })
    }

    /// Stores the samples. Samples sharing a location keep the first one, with
    /// a warning naming their holes.
    ///
    /// Parameters
    /// ----------
    /// coords : array_like, PointSet or BlockModel
    /// categories : array_like or str
    ///     A category per sample: a label of `scheme`, or without it a code
    ///     ``0..k``; or the column of `coords` holding them. So for the
    ///     other per-sample arguments.
    /// weights : array_like or str, optional
    ///     Declustering weights for the global proportions: the simple
    ///     kriging means.
    /// holes : array_like or str, optional
    ///     Drill-hole ids or names, for `max_per_hole`.
    /// domains : array_like, optional
    ///     Domain label of each sample. A target is then estimated from the
    ///     samples of its own domain, plus those of other domains within
    ///     `Search` ``soft``; `predict` then needs domains too.
    /// domain_column : str, optional
    ///     The column of `coords` holding the domains; instead of `domains`.
    #[pyo3(signature = (coords, categories, *, weights=None, holes=None, domains=None, domain_column=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        categories: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
        holes: Option<&Bound<PyAny>>,
        domains: Option<&Bound<PyAny>>,
        domain_column: Option<&str>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = Some(coords);
        let locs = points(coords)?;
        let codes: Vec<usize> = match &slf.scheme {
            Some(s) => {
                let labels = labels(&column(data, categories, "categories")?)?;
                s.0.encode(&refs(&labels))
                    .map_err(invalid)?
                    .into_iter()
                    .map(|c| c.map(|c| c as usize))
                    .collect::<Option<_>>()
                    .ok_or_else(|| invalid("categories must not be null"))?
            }
            None => crate::simulation::categories(coords, categories)?,
        };
        same_length(locs.len(), codes.len(), "categories")?;
        if slf.scheme.is_none() && slf.model.variograms.len() == 1 {
            slf.model.categories = codes.iter().max().map_or(1, |m| m + 1);
        }
        let weights = weights.map(|w| column(data, w, "weights")).transpose()?;
        let weights = optional_finite(weights.as_ref(), "weights")?;
        if let Some(w) = &weights {
            same_length(locs.len(), w.len(), "weights")?;
        }
        let holes = holes.map(|h| column(data, h, "holes")).transpose()?;
        let holes = args::holes(holes.as_ref(), locs.len())?;
        let (fitted, domain_codes) = match (domains, domain_column) {
            (None, None) => (None, None),
            _ => {
                let (fitted, codes) = args::domain_codes(data, domains, domain_column, locs.len())?;
                (Some(fitted), Some(codes))
            }
        };
        for s in &slf.search {
            s.resolve(fitted.as_deref())?;
        }
        let holes_text = holes.as_ref().map(|h| &h.0[..]);
        let keep = args::distinct_in(slf.py(), &locs, holes_text, domain_codes.as_deref())?;
        let samples: Vec<Sample> = keep
            .iter()
            .map(|&i| Sample {
                domain: domain_codes.as_ref().map(|c| c[i]),
                ..match &holes {
                    Some((_, ids)) => Sample::with_hole(locs[i], codes[i] as f64, ids[i]),
                    None => Sample::new(locs[i], codes[i] as f64),
                }
            })
            .collect();
        let weights = weights.map(|w| pick(&w, &keep));
        slf.model
            .proportions(&samples, weights.as_deref())
            .map_err(invalid)?;
        slf.samples = Some((samples, weights));
        slf.domains = fitted;
        Ok(slf)
    }

    /// Probability of each category at each target.
    ///
    /// Parameters
    /// ----------
    /// targets : array_like, PointSet or BlockModel
    /// domains : array_like, optional
    ///     Domain label of each target, or one label for all; a domain
    ///     without samples is left unestimated.
    /// domain_column : str, optional
    ///     The column of `targets` holding the domains; instead of `domains`.
    /// anisotropy : LocalAnisotropy, optional
    ///     Orients every category's variogram and the search at each target.
    /// diagnostics : bool
    ///     Fill ``CategoricalIndicatorSummary.diagnostics``.
    ///
    /// Returns
    /// -------
    /// CategoricalIndicatorSummary
    ///     NaN where the search found too few samples.
    #[pyo3(signature = (targets, *, domains=None, domain_column=None, anisotropy=None, diagnostics=false))]
    fn predict(
        &self,
        py: Python,
        targets: &Bound<PyAny>,
        domains: Option<Bound<PyAny>>,
        domain_column: Option<&str>,
        anisotropy: Option<PyRef<crate::lva::LocalAnisotropy>>,
        diagnostics: bool,
    ) -> PyResult<CategoricalIndicatorSummary> {
        let (samples, weights) = self.fitted()?;
        let domains = match (domains, domain_column) {
            (d, None) => d,
            (None, Some(c)) => Some(args::named(Some(targets), c, "domain_column")?),
            _ => return Err(invalid("give one of domains or domain_column")),
        };
        let targets = self::targets(targets)?;
        let codes = codes(
            self.domains.as_deref(),
            domains.as_ref(),
            targets.len(),
            "predict",
        )?;
        let search = self.passes()?;
        let local = anisotropy.map(|a| a.at_targets(&targets));
        py.detach(|| {
            self.model.predict(
                samples,
                weights.as_deref(),
                &targets,
                codes.as_deref(),
                &search,
                local.as_ref(),
            )
        })
        .map(|s| CategoricalIndicatorSummary {
            summary: CoreSummary {
                diagnostics: s.diagnostics.filter(|_| diagnostics),
                ..s
            },
            scheme: self.scheme.clone(),
        })
        .map_err(invalid)
    }

    /// Re-estimates each sample's probabilities from the others, through the
    /// same search passes, each in its own domain, against the proportions
    /// of all samples.
    ///
    /// Parameters
    /// ----------
    /// folds : int, optional
    ///     Leave-one-out when None; otherwise k-fold, each sample estimated
    ///     without the samples of its fold. Samples fitted with `holes` keep
    ///     their holes whole: the ``j``-th of the sorted hole ids goes to fold
    ///     ``j % folds``, an untagged sample ``i`` to fold ``i % folds``.
    ///
    /// Returns
    /// -------
    /// CategoricalCrossValidation
    #[pyo3(signature = (*, folds=None))]
    fn cross_validate<'py>(
        &self,
        py: Python<'py>,
        folds: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (samples, weights) = self.fitted()?;
        let search = self.passes()?;
        let s = py
            .detach(|| {
                self.model
                    .cross_validate(samples, weights.as_deref(), &search, folds)
            })
            .map_err(invalid)?;
        let actual = samples.iter().map(|s| s.value).collect();
        py.import("ceres.estimation")?
            .getattr("CategoricalCrossValidation")?
            .call1((
                array1(py, actual),
                by_target(py, &s.probabilities),
                self.names(),
            ))
    }

    /// Writes samples and weights as Parquet columns and parameters as JSON in
    /// the file metadata; `from_parquet` reads it back.
    fn to_parquet(&self, path: std::path::PathBuf) -> PyResult<()> {
        persist::to_parquet(<Self as pyo3::PyClass>::NAME, self, &path)
    }

    /// Reads `to_parquet` output; raises InvalidInput on another class's file
    /// or a newer format.
    #[staticmethod]
    fn from_parquet(path: std::path::PathBuf) -> PyResult<Self> {
        persist::from_parquet(<Self as pyo3::PyClass>::NAME, &path)
    }

    fn _state(&self) -> PyResult<(String, Option<Columns>)> {
        persist::state(<Self as pyo3::PyClass>::NAME, self)
    }

    #[staticmethod]
    fn _from_state(meta: &str, columns: Option<Columns>) -> PyResult<Self> {
        persist::from_state(<Self as pyo3::PyClass>::NAME, meta, columns)
    }

    /// Category names: those of `scheme`, else the codes as strings.
    #[getter]
    fn names(&self) -> Vec<String> {
        names(self.scheme.as_ref(), self.model.categories)
    }

    #[getter]
    fn scheme(&self) -> Option<Categories> {
        self.scheme.clone()
    }
}

fn names(scheme: Option<&Categories>, k: usize) -> Vec<String> {
    scheme.map_or_else(
        || (0..k).map(|c| c.to_string()).collect(),
        |s| s.0.names().to_vec(),
    )
}

/// `[category][target]` rows as a `(targets, categories)` array.
fn by_target<'py>(py: Python<'py>, rows: &[Vec<f64>]) -> Bound<'py, PyArray2<f64>> {
    let n = rows.first().map_or(0, Vec::len);
    Array2::from_shape_vec((rows.len(), n), rows.concat())
        .expect("rectangular rows")
        .reversed_axes()
        .as_standard_layout()
        .into_owned()
        .into_pyarray(py)
}

impl CategoricalIndicatorKriging {
    fn fitted(&self) -> PyResult<&(Vec<Sample>, Option<Vec<f64>>)> {
        self.samples
            .as_ref()
            .ok_or_else(|| invalid("CategoricalIndicatorKriging is not fitted; call fit first"))
    }

    /// The searches with soft boundaries by domain code.
    fn passes(&self) -> PyResult<Vec<CoreSearch>> {
        let domains = self.domains.as_deref();
        self.search.iter().map(|s| s.resolve(domains)).collect()
    }
}

impl Tabular for CategoricalIndicatorKriging {
    fn columns(&self) -> Option<Columns> {
        self.samples.as_ref().map(|(samples, weights)| {
            let mut columns = sample_columns(samples);
            if let Some(w) = weights {
                columns.push(persist::column("weight", w.iter().copied()));
            }
            columns
        })
    }

    fn restore(&mut self, columns: Found) -> PyResult<()> {
        let weights = columns.values("weight").ok();
        let samples = samples_from(&columns)?;
        self.model
            .proportions(&samples, weights.as_deref())
            .map_err(invalid)?;
        self.samples = Some((samples, weights));
        Ok(())
    }
}

/// Probability of each category at every target from categorical indicator
/// kriging; NaN where unestimated.
#[pyclass(module = "ceres", name = "CategoricalIndicatorSummary", frozen)]
pub struct CategoricalIndicatorSummary {
    summary: CoreSummary,
    scheme: Option<Categories>,
}

#[pymethods]
impl CategoricalIndicatorSummary {
    /// Writes arrays as Parquet columns and parameters as JSON in the file
    /// metadata; `from_parquet` reads it back.
    fn to_parquet(&self, path: std::path::PathBuf) -> PyResult<()> {
        persist::to_parquet(<Self as pyo3::PyClass>::NAME, self, &path)
    }

    /// Reads `to_parquet` output; raises InvalidInput on another class's file
    /// or a newer format.
    #[staticmethod]
    fn from_parquet(path: std::path::PathBuf) -> PyResult<Self> {
        persist::from_parquet(<Self as pyo3::PyClass>::NAME, &path)
    }

    fn _state(&self) -> PyResult<(String, Option<Columns>)> {
        persist::state(<Self as pyo3::PyClass>::NAME, self)
    }

    #[staticmethod]
    fn _from_state(meta: &str, columns: Option<Columns>) -> PyResult<Self> {
        persist::from_state(<Self as pyo3::PyClass>::NAME, meta, columns)
    }

    /// `(targets, categories)` probabilities, each row summing to 1.
    #[getter]
    fn probabilities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        by_target(py, &self.summary.probabilities)
    }

    /// Code of the most probable category, ties to the lowest; float so
    /// that unestimated targets are NaN, as `Categories.decode` takes it.
    #[getter]
    fn most_likely<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array1(py, self.summary.most_likely())
    }

    /// Entropy of the probabilities scaled to [0, 1]: 0 where one category
    /// is certain, 1 where all are equally likely.
    #[getter]
    fn entropy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array1(py, self.summary.entropy())
    }

    /// Sum over categories of |corrected − kriged| probability.
    #[getter]
    fn correction<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array1(py, self.summary.correction.clone())
    }

    /// Declustered proportion of each category among the samples.
    #[getter]
    fn proportions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array1(py, self.summary.proportions.clone())
    }

    /// Category names: those of `scheme`, else the codes as strings.
    #[getter]
    fn names(&self) -> Vec<String> {
        names(self.scheme.as_ref(), self.summary.probabilities.len())
    }

    #[getter]
    fn scheme(&self) -> Option<Categories> {
        self.scheme.clone()
    }

    /// Per-target Table when predicted with ``diagnostics=True``, else None:
    /// ``n_samples``, ``pass`` (the search, from 1, that filled the target),
    /// ``n_holes`` (distinct holes among the samples used; untagged samples
    /// count one each), ``mean_distance`` (to the samples used),
    /// ``max_samples_reached`` (1 where the search returned `max_samples`),
    /// ``correction`` and ``n_order_violations`` (categories kriged outside
    /// [0, 1]). Null where unestimated.
    #[getter]
    fn diagnostics(&self) -> PyResult<Option<Table>> {
        let s = &self.summary;
        diagnostics_table(diagnostic_columns(s.diagnostics.as_ref(), &s.correction))
    }

    fn __repr__(&self) -> String {
        format!(
            "CategoricalIndicatorSummary(targets={}, categories={:?})",
            self.summary.correction.len(),
            self.names()
        )
    }
}

#[derive(Serialize, Deserialize)]
struct SummaryMeta {
    categories: usize,
    proportions: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    scheme: Option<Categories>,
}

impl Serialize for CategoricalIndicatorSummary {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        SummaryMeta {
            categories: self.summary.probabilities.len(),
            proportions: self.summary.proportions.clone(),
            scheme: self.scheme.clone(),
        }
        .serialize(s)
    }
}

impl<'de> Deserialize<'de> for CategoricalIndicatorSummary {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let m = SummaryMeta::deserialize(d)?;
        Ok(Self {
            summary: CoreSummary {
                probabilities: vec![vec![]; m.categories],
                proportions: m.proportions,
                ..Default::default()
            },
            scheme: m.scheme,
        })
    }
}

impl Tabular for CategoricalIndicatorSummary {
    fn columns(&self) -> Option<Columns> {
        let s = &self.summary;
        let mut out = vec![persist::column("correction", s.correction.iter().copied())];
        for (c, p) in s.probabilities.iter().enumerate() {
            out.push(persist::column(&format!("p_{c}"), p.iter().copied()));
        }
        push_diagnostics(&mut out, s.diagnostics.as_ref(), &s.correction);
        Some(out)
    }

    fn restore(&mut self, found: Found) -> PyResult<()> {
        let s = &mut self.summary;
        s.correction = found.values("correction")?;
        for (c, p) in s.probabilities.iter_mut().enumerate() {
            *p = found.values(&format!("p_{c}"))?;
        }
        s.diagnostics = restore_diagnostics(&found)?;
        Ok(())
    }
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<CategoricalIndicatorKriging>()?;
    m.add_class::<CategoricalIndicatorSummary>()?;
    Ok(())
}
