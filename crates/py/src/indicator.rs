use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch};
use estimation::{
    Discretization, IndicatorDiagnostics, IndicatorSummary as CoreSummary, Interpolation,
    MultipleIndicator, Sample, Search as CoreSearch, UpperTail,
};
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::args::{
    self, array1, column, distinct, finite, optional_finite, pick, points, same_length,
};
use crate::categorical::by_target;
use crate::containers::PyBlockModel;
use crate::estimation::{sample_columns, samples_from, searches, targets};
use crate::invalid;
use crate::persist::{self, Columns, Found, Tabular};
use crate::progress::with_progress;
use crate::table::Table;
use crate::transforms::nullable;
use crate::variogram::Variogram;

/// The diagnostics columns, `correction` among them, when predicted with them.
pub(crate) fn diagnostic_columns<'a>(
    diagnostics: Option<&'a IndicatorDiagnostics>,
    correction: &'a Vec<f64>,
) -> Option<Vec<(&'static str, &'a Vec<f64>)>> {
    diagnostics.map(|d| {
        vec![
            ("n_samples", &d.n_samples),
            ("pass", &d.pass),
            ("n_holes", &d.n_holes),
            ("mean_distance", &d.mean_distance),
            ("max_samples_reached", &d.max_samples_reached),
            ("correction", correction),
            ("n_order_violations", &d.n_order_violations),
        ]
    })
}

/// The diagnostics columns as a Table, null where unestimated.
pub(crate) fn diagnostics_table(
    columns: Option<Vec<(&'static str, &Vec<f64>)>>,
) -> PyResult<Option<Table>> {
    columns
        .map(|columns| {
            RecordBatch::try_from_iter(columns.into_iter().map(|(name, values)| {
                let column: Float64Array =
                    values.iter().map(|v| (!v.is_nan()).then_some(*v)).collect();
                (name, Arc::new(column) as ArrayRef)
            }))
            .map(Table)
            .map_err(invalid)
        })
        .transpose()
}

/// The diagnostics columns of `diagnostic_columns`, but `correction`.
pub(crate) fn push_diagnostics(
    out: &mut Columns,
    diagnostics: Option<&IndicatorDiagnostics>,
    correction: &Vec<f64>,
) {
    for (name, values) in diagnostic_columns(diagnostics, correction).unwrap_or_default() {
        if name != "correction" {
            out.push(persist::column(name, values.iter().copied()));
        }
    }
}

/// The diagnostics `push_diagnostics` wrote, if any.
pub(crate) fn restore_diagnostics(found: &Found) -> PyResult<Option<IndicatorDiagnostics>> {
    let Ok(n_samples) = found.values("n_samples") else {
        return Ok(None);
    };
    Ok(Some(IndicatorDiagnostics {
        n_samples,
        pass: found.values("pass")?,
        n_holes: found.values("n_holes")?,
        mean_distance: found.values("mean_distance")?,
        max_samples_reached: found.values("max_samples_reached")?,
        n_order_violations: found.values("n_order_violations")?,
    }))
}

/// Multiple indicator kriging: `P(value <= t)` kriged at every threshold,
/// corrected for order relations, completed within classes and in the tails,
/// and summarized at each target.
///
/// Parameters
/// ----------
/// variogram : Variogram or sequence of Variogram
///     One indicator variogram per threshold, or a single one shared by all
///     (median indicator kriging: one kriging system per target).
/// search : Search or sequence of Search
///     Neighborhood shared by all thresholds; a sequence is searched in
///     passes, each filling the targets the previous left unestimated. With
///     no ellipsoid, the median threshold's variogram orients it.
/// thresholds : sequence of float
///     Strictly increasing, in data units.
/// simple : bool
///     Simple kriging with the declustered global proportions as means;
///     ordinary kriging otherwise.
/// tails : tuple of float, optional
///     Lower and upper bounds of the distribution, widened to cover the data
///     and thresholds; the data minimum and maximum by default.
/// interpolation : {"global", "linear"}
///     Within each class, follow the declustered data in it ("global",
///     uniform where a class holds none) or be uniform ("linear").
/// upper_tail : tuple of (str, float), optional
///     ``("power", w)`` or ``("hyperbolic", w)`` for the class above the last
///     threshold, up to the upper tail bound; the hyperbolic model needs a
///     last threshold > 0.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "boitata", name = "MultipleIndicatorKriging")]
pub struct MultipleIndicatorKriging {
    model: MultipleIndicator,
    search: Vec<CoreSearch>,
    #[serde(skip)]
    samples: Option<(Vec<Sample>, Option<Vec<f64>>)>,
}

#[pymethods]
impl MultipleIndicatorKriging {
    #[new]
    #[pyo3(signature = (variogram, search, thresholds, *, simple=false, tails=None, interpolation="global", upper_tail=None))]
    fn new(
        variogram: &Bound<PyAny>,
        search: &Bound<PyAny>,
        thresholds: Vec<f64>,
        simple: bool,
        tails: Option<(f64, f64)>,
        interpolation: &str,
        upper_tail: Option<(String, f64)>,
    ) -> PyResult<Self> {
        let variograms: Vec<Variogram> = match variogram.extract::<Variogram>() {
            Ok(v) => vec![v],
            Err(_) => variogram
                .extract()
                .map_err(|_| invalid("variogram must be a Variogram or a sequence of Variogram"))?,
        };
        let interpolation = match interpolation {
            "global" => Interpolation::Global,
            "linear" => Interpolation::Linear,
            other => return Err(invalid(format!("unknown interpolation {other:?}"))),
        };
        let upper_tail = match upper_tail {
            None => None,
            Some((name, w)) => Some(match name.as_str() {
                "power" => UpperTail::Power(w),
                "hyperbolic" => UpperTail::Hyperbolic(w),
                other => return Err(invalid(format!("unknown upper tail {other:?}"))),
            }),
        };
        let model = MultipleIndicator {
            thresholds,
            variograms: variograms.into_iter().map(|v| v.0).collect(),
            simple,
            tails,
            interpolation,
            upper_tail,
        };
        model.validate().map_err(invalid)?;
        Ok(Self {
            model,
            search: searches(search)?
                .into_iter()
                .map(|s| s.plain("MultipleIndicatorKriging"))
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
    ///     Declustering weights for the global distribution: the simple
    ///     kriging means, the class contents and the tails.
    /// holes : array_like or str, optional
    ///     Drill-hole ids or names, for `max_per_hole`.
    #[pyo3(signature = (coords, values, *, weights=None, holes=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        coords: &Bound<PyAny>,
        values: &Bound<PyAny>,
        weights: Option<&Bound<PyAny>>,
        holes: Option<&Bound<PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = Some(coords);
        let locs = points(coords)?;
        let values = finite(&column(data, values, "values")?, "values")?;
        same_length(locs.len(), values.len(), "values")?;
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
    /// anisotropy : LocalAnisotropy, optional
    ///     Orients every threshold's variogram and the search at each target.
    /// diagnostics : bool
    ///     Fill ``IndicatorSummary.diagnostics``.
    /// discretization : tuple of int, optional
    ///     Points per axis of each block of a BlockModel `targets`: the
    ///     indicators are kriged over the block, giving the distribution of
    ///     the point values within it rather than at its centroid.
    /// progress : bool, default True
    ///     Show a `tqdm` progress bar.
    ///
    /// Returns
    /// -------
    /// IndicatorSummary
    ///     NaN where the search found too few samples.
    #[pyo3(signature = (targets, *, cutoffs=vec![], quantiles=vec![], anisotropy=None, diagnostics=false, discretization=None, progress=true))]
    #[allow(clippy::too_many_arguments)]
    fn predict(
        &self,
        py: Python,
        targets: &Bound<PyAny>,
        cutoffs: Vec<f64>,
        quantiles: Vec<f64>,
        anisotropy: Option<PyRef<crate::lva::LocalAnisotropy>>,
        diagnostics: bool,
        discretization: Option<(usize, usize, usize)>,
        progress: bool,
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
        let local = anisotropy.map(|a| a.at_targets(&targets));
        let total = Some(targets.len() as u64);
        with_progress(py, total, progress, |counter| {
            self.model.predict_with_progress(
                samples,
                weights.as_deref(),
                &targets,
                block.as_deref(),
                &self.search,
                local.as_ref(),
                &cutoffs,
                &quantiles,
                counter,
            )
        })?
        .map(|s| {
            IndicatorSummary(CoreSummary {
                diagnostics: s.diagnostics.filter(|_| diagnostics),
                ..s
            })
        })
        .map_err(invalid)
    }

    /// Re-estimates each sample's distribution from the others, through the
    /// same search passes, against the global distribution of all samples.
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
                array1(py, s.mean.clone()),
                array1(py, s.variance.clone()),
                s.thresholds.clone(),
                by_target(py, &s.cdf, samples.len()),
                array1(py, pit),
            ))
    }

    /// Localized grades of the selective blocks nested in the panels.
    ///
    /// Each panel's point-support conditional distribution, kriged at its
    /// centroid or over the panel, takes an affine change of support to the selective blocks,
    /// ``m + sqrt(f) * (z - m)`` about its mean ``m``. A panel holding ``n``
    /// blocks splits that distribution into ``n`` equal-probability bands, and
    /// its block ranked ``i`` gets the mean of band ``i``: the blocks average
    /// to the panel's E-type estimate and reproduce its selective-block
    /// grade-tonnage curve at tonnages ``k / n``. Partial panels localize over
    /// the blocks present.
    ///
    /// Parameters
    /// ----------
    /// smus : BlockModel
    ///     Selective blocks nesting in the panels: same rotation, sizes
    ///     dividing the panel sizes, grids aligned; not sub-blocked.
    /// ranking : str
    ///     Column of `smus` ordering the blocks within a panel, such as a
    ///     direct kriging of the blocks; ties follow row order.
    /// panels : BlockModel
    /// variance_factor : float or Variogram, optional
    ///     ``f``, the variance of the blocks within a panel over that of the
    ///     points within it, in [0, 1]; or a variogram to compute it from,
    ///     ``(C(v, v) - C(V, V)) / (C(0) - C(V, V))`` with the nugget left out
    ///     of the block averages. The median threshold's variogram by default.
    /// name : str, default "localized"
    ///     Name of the new column.
    /// discretization : tuple of int, optional
    ///     Points per axis of each panel to krige the indicators over; at the
    ///     centroid by default.
    ///
    /// Returns
    /// -------
    /// BlockModel
    ///     `smus` with the localized grades; null in panels the search leaves
    ///     unestimated and outside every panel.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     If the blocks do not nest, or a block of an estimated panel has a
    ///     null rank.
    #[pyo3(signature = (smus, ranking, panels, *, variance_factor=None, name="localized", discretization=None))]
    #[allow(clippy::too_many_arguments)]
    fn localize(
        &self,
        py: Python,
        smus: PyRef<PyBlockModel>,
        ranking: &str,
        panels: PyRef<PyBlockModel>,
        variance_factor: Option<&Bound<PyAny>>,
        name: &str,
        discretization: Option<(usize, usize, usize)>,
    ) -> PyResult<PyBlockModel> {
        let (samples, weights) = self.fitted()?;
        let discretization = discretization
            .map(|d| offsets(&panels.0, d).map(|_| disc(d)))
            .transpose()?;
        let rank = nullable(&smus, ranking)?;
        let (panel_model, smu_model) = (&panels.0, &smus.0);
        let f = match variance_factor {
            None => None,
            Some(v) => Some(match v.extract::<PyRef<Variogram>>() {
                Ok(vg) => {
                    estimation::variance_factor(&vg.0, panel_model, smu_model).map_err(invalid)?
                }
                Err(_) => v
                    .extract::<f64>()
                    .map_err(|_| invalid("variance_factor must be a float or a Variogram"))?,
            }),
        };
        let out = py
            .detach(|| {
                self.model.localize(
                    samples,
                    weights.as_deref(),
                    &self.search,
                    panel_model,
                    smu_model,
                    &rank,
                    f,
                    discretization,
                )
            })
            .map_err(invalid)?;
        let column: arrow_array::Float64Array = out.into_iter().collect();
        Ok(PyBlockModel(
            smus.0
                .with_column(name, std::sync::Arc::new(column))
                .map_err(invalid)?,
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

    #[getter]
    fn thresholds(&self) -> Vec<f64> {
        self.model.thresholds.clone()
    }
}

fn disc((nx, ny, nz): (usize, usize, usize)) -> Discretization {
    Discretization { nx, ny, nz }
}

/// Discretization points of a block of `model` about its centroid.
pub(crate) fn offsets(
    model: &boitata_core::BlockModel,
    d: (usize, usize, usize),
) -> PyResult<Vec<(f64, f64, f64)>> {
    if d.0 == 0 || d.1 == 0 || d.2 == 0 {
        return Err(invalid("discretization must be positive"));
    }
    if matches!(model.layout(), boitata_core::Layout::SubBlocked { .. }) {
        return Err(invalid(
            "discretization needs a regular or masked BlockModel",
        ));
    }
    Ok(disc(d).offsets(model.geometry()))
}

impl MultipleIndicatorKriging {
    fn fitted(&self) -> PyResult<&(Vec<Sample>, Option<Vec<f64>>)> {
        self.samples
            .as_ref()
            .ok_or_else(|| invalid("MultipleIndicatorKriging is not fitted; call fit first"))
    }
}

impl Tabular for MultipleIndicatorKriging {
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

/// Conditional distribution at every target from multiple indicator or
/// multigaussian kriging.
/// Per-threshold, per-cutoff and per-quantile arrays have one row per
/// target and one column per threshold, cutoff or quantile; NaN where
/// unestimated.
#[pyclass(module = "boitata", name = "IndicatorSummary", frozen)]
pub struct IndicatorSummary(pub(crate) CoreSummary);

#[pymethods]
impl IndicatorSummary {
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

    /// Mean of the conditional distribution (E-type estimate).
    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array1(py, self.0.mean.clone())
    }

    /// Variance of the conditional distribution.
    #[getter]
    fn variance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array1(py, self.0.variance.clone())
    }

    #[getter]
    fn std<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array1(py, self.0.variance.iter().map(|v| v.sqrt()).collect())
    }

    #[getter]
    fn thresholds(&self) -> Vec<f64> {
        self.0.thresholds.clone()
    }

    /// `(targets, thresholds)` order-relation corrected `P(value <= t)`.
    #[getter]
    fn cdf<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        by_target(py, &self.0.cdf, self.0.mean.len())
    }

    /// Sum over thresholds of |corrected − kriged| probability.
    #[getter]
    fn correction<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array1(py, self.0.correction.clone())
    }

    #[getter]
    fn cutoffs(&self) -> Vec<f64> {
        self.0.cutoffs.clone()
    }

    /// `(targets, cutoffs)` probability above each cutoff.
    #[getter]
    fn probability_above<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        by_target(py, &self.0.probability_above, self.0.mean.len())
    }

    /// `(targets, cutoffs)` mean above each cutoff; NaN where nothing is above it.
    #[getter]
    fn mean_above<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        by_target(py, &self.0.mean_above, self.0.mean.len())
    }

    #[getter]
    fn quantiles(&self) -> Vec<f64> {
        self.0.quantiles.clone()
    }

    /// `(targets, quantiles)` values at each requested quantile.
    #[getter]
    fn quantile_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        by_target(py, &self.0.quantile_values, self.0.mean.len())
    }

    /// Per-target Table when predicted with ``diagnostics=True``, else None:
    /// ``n_samples``, ``pass`` (the search, from 1, that filled the target),
    /// ``n_holes`` (distinct holes among the samples used; untagged samples
    /// count one each), ``mean_distance`` (to the samples used),
    /// ``max_samples_reached`` (1 where the search returned `max_samples`),
    /// ``correction`` and ``n_order_violations`` (thresholds whose kriged
    /// probability the order-relation correction changed). Null where
    /// unestimated.
    #[getter]
    fn diagnostics(&self) -> PyResult<Option<Table>> {
        diagnostics_table(diagnostic_columns(
            self.0.diagnostics.as_ref(),
            &self.0.correction,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "IndicatorSummary(targets={}, thresholds={:?}, cutoffs={:?}, quantiles={:?})",
            self.0.mean.len(),
            self.0.thresholds,
            self.0.cutoffs,
            self.0.quantiles
        )
    }
}

#[derive(Serialize, Deserialize)]
struct SummaryMeta {
    thresholds: Vec<f64>,
    cutoffs: Vec<f64>,
    quantiles: Vec<f64>,
}

impl Serialize for IndicatorSummary {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        SummaryMeta {
            thresholds: self.0.thresholds.clone(),
            cutoffs: self.0.cutoffs.clone(),
            quantiles: self.0.quantiles.clone(),
        }
        .serialize(s)
    }
}

impl<'de> Deserialize<'de> for IndicatorSummary {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let m = SummaryMeta::deserialize(d)?;
        Ok(Self(CoreSummary {
            thresholds: m.thresholds,
            cutoffs: m.cutoffs,
            quantiles: m.quantiles,
            ..Default::default()
        }))
    }
}

impl Tabular for IndicatorSummary {
    fn columns(&self) -> Option<Columns> {
        let c = &self.0;
        let column = |name: String, v: &Vec<f64>| persist::column(&name, v.iter().copied());
        let mut out = vec![
            column("mean".into(), &c.mean),
            column("variance".into(), &c.variance),
            column("correction".into(), &c.correction),
        ];
        for (t, values) in c.thresholds.iter().zip(&c.cdf) {
            out.push(column(format!("cdf_{t}"), values));
        }
        for (i, cut) in c.cutoffs.iter().enumerate() {
            out.push(column(format!("p_above_{cut}"), &c.probability_above[i]));
            out.push(column(format!("mean_above_{cut}"), &c.mean_above[i]));
        }
        for (q, values) in c.quantiles.iter().zip(&c.quantile_values) {
            out.push(column(format!("q{q}"), values));
        }
        push_diagnostics(&mut out, c.diagnostics.as_ref(), &c.correction);
        Some(out)
    }

    fn restore(&mut self, found: Found) -> PyResult<()> {
        let c = &mut self.0;
        let each = |names: Vec<String>| -> PyResult<Vec<Vec<f64>>> {
            names.iter().map(|n| found.values(n)).collect()
        };
        c.mean = found.values("mean")?;
        c.variance = found.values("variance")?;
        c.correction = found.values("correction")?;
        c.cdf = each(c.thresholds.iter().map(|t| format!("cdf_{t}")).collect())?;
        c.probability_above = each(c.cutoffs.iter().map(|x| format!("p_above_{x}")).collect())?;
        c.mean_above = each(
            c.cutoffs
                .iter()
                .map(|x| format!("mean_above_{x}"))
                .collect(),
        )?;
        c.quantile_values = each(c.quantiles.iter().map(|q| format!("q{q}")).collect())?;
        c.diagnostics = restore_diagnostics(&found)?;
        Ok(())
    }
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<MultipleIndicatorKriging>()?;
    m.add_class::<IndicatorSummary>()?;
    Ok(())
}
