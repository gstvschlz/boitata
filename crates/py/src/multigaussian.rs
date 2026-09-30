use estimation::{IndicatorSummary as CoreSummary, Multigaussian, Sample, Search as CoreSearch};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::args::{
    self, array1, column, distinct, finite, optional_finite, pick, points, same_length,
};
use crate::categorical::by_target;
use crate::containers::PyBlockModel;
use crate::estimation::{sample_columns, samples_from, searches, targets};
use crate::indicator::{IndicatorSummary, offsets};
use crate::invalid;
use crate::persist::{self, Columns, Found, Tabular};
use crate::variogram::Variogram;

/// Multigaussian kriging: the values take a normal-score transform, the
/// scores are simple-kriged about 0, and the Gaussian distribution of the
/// score at each target, with the kriged mean and variance, is
/// back-transformed into data units.
///
/// Parameters
/// ----------
/// variogram : Variogram
///     Variogram of the normal scores, with a sill near 1.
/// search : Search or sequence of Search
///     A sequence is searched in passes, each filling the targets the
///     previous left unestimated.
/// tails : tuple of float, optional
///     Lower and upper bounds of the back-transform, widened to cover the
///     data; the data minimum and maximum by default.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "MultigaussianKriging")]
pub struct MultigaussianKriging {
    model: Multigaussian,
    search: Vec<CoreSearch>,
    #[serde(skip)]
    samples: Option<(Vec<Sample>, Option<Vec<f64>>)>,
}

#[pymethods]
impl MultigaussianKriging {
    #[new]
    #[pyo3(signature = (variogram, search, *, tails=None))]
    fn new(
        variogram: PyRef<Variogram>,
        search: &Bound<PyAny>,
        tails: Option<(f64, f64)>,
    ) -> PyResult<Self> {
        let model = Multigaussian {
            variogram: variogram.0.clone(),
            tails,
        };
        model.validate().map_err(invalid)?;
        Ok(Self {
            model,
            search: searches(search)?
                .into_iter()
                .map(|s| s.plain("MultigaussianKriging"))
                .collect::<PyResult<_>>()?,
            samples: None,
        })
    }

    /// Stores the samples. Samples sharing a location keep the first one, with
    /// a warning naming their holes.
    ///
    /// Parameters
    /// ----------
    /// coords : array_like, PointSet or BlockModel
    /// values : array_like or str
    ///     A value per sample, or the column of `coords` holding them; so for
    ///     `weights` and `holes`.
    /// weights : array_like or str, optional
    ///     Declustering weights for the normal-score transform.
    /// holes : array_like or str, optional
    ///     Drill-hole ids or names, for `max_per_hole`.
    /// despike : bool
    ///     Break ties in the values first, as `despike` does with its
    ///     defaults, so tied samples get distinct scores; tied samples share
    ///     their mean score otherwise.
    #[pyo3(signature = (coords, values, *, weights=None, holes=None, despike=false))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        values: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
        holes: Option<&Bound<PyAny>>,
        despike: bool,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = Some(coords);
        let locs = points(coords)?;
        let mut values = finite(&column(data, values, "values")?, "values")?;
        same_length(locs.len(), values.len(), "values")?;
        if despike {
            let radii = transforms::default_radii(&locs);
            values = transforms::despike(&locs, &[values], &radii, 0)
                .map_err(invalid)?
                .remove(0);
        }
        let weights = weights.map(|w| column(data, w, "weights")).transpose()?;
        let weights = optional_finite(weights.as_ref(), "weights")?;
        if let Some(w) = &weights {
            same_length(locs.len(), w.len(), "weights")?;
            if w.iter().any(|v| *v < 0.0) || w.iter().sum::<f64>() <= 0.0 {
                return Err(invalid("weights must be >= 0 and not all 0"));
            }
        }
        let holes = holes.map(|h| column(data, h, "holes")).transpose()?;
        let holes = args::holes(holes.as_ref(), locs.len())?;
        let keep = distinct(slf.py(), &locs, holes.as_ref().map(|h| &h.0[..]))?;
        let samples = keep
            .iter()
            .map(|&i| match &holes {
                Some((_, ids)) => Sample::with_hole(locs[i], values[i], ids[i]),
                None => Sample::new(locs[i], values[i]),
            })
            .collect();
        slf.samples = Some((samples, weights.map(|w| pick(&w, &keep))));
        Ok(slf)
    }

    /// Conditional distribution at each target.
    ///
    /// Parameters
    /// ----------
    /// targets : array_like, PointSet or BlockModel
    /// cutoffs : sequence of float
    ///     For ``probability_above`` and ``mean_above``.
    /// quantiles : sequence of float
    ///     Probabilities in [0, 1], for ``quantile_values``.
    /// discretization : tuple of int, optional
    ///     Points per axis of each block of a BlockModel `targets`: the
    ///     distributions at the points are averaged, giving the distribution
    ///     of the point values within the block rather than at its centroid.
    /// diagnostics : bool
    ///     Fill ``IndicatorSummary.diagnostics``.
    ///
    /// Returns
    /// -------
    /// IndicatorSummary
    ///     No thresholds; ``correction`` is 0. NaN where the search found too
    ///     few samples.
    #[pyo3(signature = (targets, *, cutoffs=vec![], quantiles=vec![], discretization=None, diagnostics=false))]
    fn predict(
        &self,
        py: Python,
        targets: &Bound<PyAny>,
        cutoffs: Vec<f64>,
        quantiles: Vec<f64>,
        discretization: Option<(usize, usize, usize)>,
        diagnostics: bool,
    ) -> PyResult<IndicatorSummary> {
        let (samples, weights) = self.fitted()?;
        let block = match discretization {
            None => None,
            Some(d) => {
                let model = targets
                    .cast::<PyBlockModel>()
                    .map_err(|_| invalid("discretization needs BlockModel targets"))?;
                Some(offsets(&model.get().0, d)?)
            }
        };
        let targets = self::targets(targets)?;
        py.detach(|| {
            self.model.predict(
                samples,
                weights.as_deref(),
                &targets,
                block.as_deref(),
                &self.search,
                &cutoffs,
                &quantiles,
            )
        })
        .map(|s| {
            IndicatorSummary(CoreSummary {
                diagnostics: s.diagnostics.filter(|_| diagnostics),
                ..s
            })
        })
        .map_err(invalid)
    }

    /// Re-estimates each sample's distribution from the others, through the
    /// same search passes, with the transform of all samples.
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
    /// IndicatorCrossValidation
    ///     Without thresholds.
    #[pyo3(signature = (*, folds=None))]
    fn cross_validate<'py>(
        &self,
        py: Python<'py>,
        folds: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (samples, weights) = self.fitted()?;
        let (s, pit) = py
            .detach(|| {
                self.model
                    .cross_validate(samples, weights.as_deref(), &self.search, folds)
            })
            .map_err(invalid)?;
        let actual = samples.iter().map(|s| s.value).collect();
        py.import("boitata.estimation")?
            .getattr("IndicatorCrossValidation")?
            .call1((
                array1(py, actual),
                array1(py, s.mean),
                array1(py, s.variance),
                Vec::<f64>::new(),
                by_target(py, &[], samples.len()),
                array1(py, pit),
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
}

impl MultigaussianKriging {
    fn fitted(&self) -> PyResult<&(Vec<Sample>, Option<Vec<f64>>)> {
        self.samples
            .as_ref()
            .ok_or_else(|| invalid("MultigaussianKriging is not fitted; call fit first"))
    }
}

impl Tabular for MultigaussianKriging {
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
        self.samples = Some((samples_from(&columns)?, weights));
        Ok(())
    }
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<MultigaussianKriging>()
}
