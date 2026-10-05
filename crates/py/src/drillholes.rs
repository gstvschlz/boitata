use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::DataType;
use boitata_core::PointSet;
use drillholes::{
    Collar, CompositeParams, DesurveyMethod, DrillholeError, Residual, Run, RunRules,
    SurveyStation, WellborePoint, checks, composite_intervals, desurvey_wellbore, ore_runs,
    position_at,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::blocks::Mesh;
use crate::containers::PyPointSet;
use crate::invalid;
use crate::table::{Table, to_batch};

fn text(batch: &RecordBatch, name: &str) -> PyResult<Vec<Option<String>>> {
    let column = batch
        .column_by_name(name)
        .ok_or_else(|| invalid(format!("column {name} not found")))?;
    text_of(column)
}

fn text_of(column: &ArrayRef) -> PyResult<Vec<Option<String>>> {
    let column = arrow_cast::cast(column, &DataType::Utf8).map_err(invalid)?;
    Ok(column
        .as_string::<i32>()
        .iter()
        .map(|s| s.map(str::to_string))
        .collect())
}

fn number(batch: &RecordBatch, name: &str) -> PyResult<Vec<Option<f64>>> {
    let column = batch
        .column_by_name(name)
        .ok_or_else(|| invalid(format!("column {name} not found")))?;
    let column = arrow_cast::cast(column, &DataType::Float64).map_err(invalid)?;
    Ok(column.as_primitive::<Float64Type>().iter().collect())
}

/// Collar, survey and interval tables desurveyed by `method`:
/// `"minimum_curvature"` (circular arcs), `"tangential"` (each survey's
/// direction holds down to the next) or `"balanced_tangential"` (half of
/// each segment along each end's direction). Survey angles are `azimuth`
/// (clockwise from north) and either `dip` (degrees below horizontal) or
/// `inclination` (degrees from vertical); holes without survey are vertical.
#[pyclass(module = "boitata", name = "Drillholes", frozen)]
pub struct Drillholes {
    paths: BTreeMap<String, Vec<WellborePoint>>,
    hole: String,
    intervals: Option<(RecordBatch, String, String, String)>,
    length_unit: Option<String>,
}

#[pymethods]
impl Drillholes {
    #[new]
    #[pyo3(signature = (
        collar, survey, intervals=None, *, hole="HOLE_ID", x="X", y="Y", z="Z", at="DEPTH",
        azimuth="AZIMUTH", dip=Some("DIP"), inclination=None, from_="FROM", to="TO",
        method="minimum_curvature", length_unit=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python,
        collar: &Bound<PyAny>,
        survey: &Bound<PyAny>,
        intervals: Option<&Bound<PyAny>>,
        hole: &str,
        x: &str,
        y: &str,
        z: &str,
        at: &str,
        azimuth: &str,
        dip: Option<&str>,
        inclination: Option<&str>,
        from_: &str,
        to: &str,
        method: &str,
        length_unit: Option<String>,
    ) -> PyResult<Self> {
        let method = match method {
            "minimum_curvature" => DesurveyMethod::MinimumCurvature,
            "tangential" => DesurveyMethod::Tangential,
            "balanced_tangential" => DesurveyMethod::BalancedTangential,
            _ => {
                return Err(invalid(format!(
                    "method must be minimum_curvature, tangential or balanced_tangential, got {method:?}"
                )));
            }
        };
        let collar = to_batch(collar)?;
        let survey = to_batch(survey)?;
        let mut collars = BTreeMap::new();
        for (((id, x), y), z) in text(&collar, hole)?
            .into_iter()
            .zip(number(&collar, x)?)
            .zip(number(&collar, y)?)
            .zip(number(&collar, z)?)
        {
            if let (Some(id), Some(east), Some(north), Some(elev)) = (id, x, y, z) {
                let c = Collar {
                    hole_id: id.clone(),
                    east,
                    north,
                    collar_elev: elev,
                };
                collars.insert(id, c);
            }
        }

        let angle = match (inclination, dip) {
            (Some(name), _) => number(&survey, name)?,
            (None, Some(name)) => number(&survey, name)?
                .into_iter()
                .map(|d| d.map(|d| 90.0 - d))
                .collect(),
            (None, None) => return Err(invalid("give the dip or inclination column")),
        };
        let mut stations: HashMap<String, Vec<SurveyStation>> = HashMap::new();
        for (((id, depth), az), inc) in text(&survey, hole)?
            .into_iter()
            .zip(number(&survey, at)?)
            .zip(number(&survey, azimuth)?)
            .zip(angle)
        {
            if let (Some(id), Some(depth), Some(azimuth), Some(inclination)) = (id, depth, az, inc)
            {
                stations.entry(id.clone()).or_default().push(SurveyStation {
                    hole_id: id,
                    depth,
                    azimuth,
                    inclination,
                });
            }
        }

        let intervals = intervals
            .map(|t| {
                Ok::<_, PyErr>((
                    to_batch(t)?,
                    hole.to_string(),
                    from_.to_string(),
                    to.to_string(),
                ))
            })
            .transpose()?;
        let mut deepest: HashMap<String, f64> = HashMap::new();
        if let Some((batch, _, _, to)) = &intervals {
            for (id, t) in text(batch, hole)?.into_iter().zip(number(batch, to)?) {
                if let (Some(id), Some(t)) = (id, t) {
                    let d = deepest.entry(id).or_insert(t);
                    *d = d.max(t);
                }
            }
        }

        let paths = py.detach(|| {
            collars
                .par_iter()
                .map(|(id, c)| {
                    let mut s = stations.get(id).cloned().unwrap_or_else(|| {
                        vec![SurveyStation {
                            hole_id: id.clone(),
                            depth: 0.0,
                            azimuth: 0.0,
                            inclination: 0.0,
                        }]
                    });
                    s.sort_by(|a, b| a.depth.total_cmp(&b.depth));
                    let last = s.last().expect("at least one station").clone();
                    if let Some(&d) = deepest.get(id)
                        && d > last.depth
                    {
                        s.push(SurveyStation { depth: d, ..last });
                    }
                    desurvey_wellbore(c, &s, method).map(|p| (id.clone(), p))
                })
                .collect::<Result<BTreeMap<_, _>, _>>()
        });
        Ok(Self {
            paths: paths.map_err(invalid)?,
            hole: hole.to_string(),
            intervals,
            length_unit: crate::table::length_unit(length_unit)?,
        })
    }

    /// Unit of the coordinates and depths, such as ``m`` or ``ft``; None
    /// when not declared.
    #[getter]
    fn length_unit(&self) -> Option<String> {
        self.length_unit.clone()
    }

    #[getter]
    fn holes(&self) -> Vec<String> {
        self.paths.keys().cloned().collect()
    }

    /// Hole, from and to column names of the interval table, if one was given.
    #[getter]
    fn interval_columns(&self) -> Option<(String, String, String)> {
        let (_, h, f, t) = self.intervals.as_ref()?;
        Some((h.clone(), f.clone(), t.clone()))
    }

    /// Desurveyed stations: hole (named as in the constructor), depth, x, y, z.
    fn paths(&self) -> PyResult<Table> {
        let rows: Vec<(&String, &WellborePoint)> = self
            .paths
            .iter()
            .flat_map(|(id, p)| p.iter().map(move |w| (id, w)))
            .collect();
        let col = |f: fn(&WellborePoint) -> f64| {
            Arc::new(Float64Array::from_iter_values(rows.iter().map(|r| f(r.1)))) as ArrayRef
        };
        let batch = RecordBatch::try_from_iter([
            (
                self.hole.as_str(),
                Arc::new(StringArray::from_iter_values(rows.iter().map(|r| r.0))) as ArrayRef,
            ),
            ("depth", col(|w| w.measured_depth)),
            ("x", col(|w| w.east)),
            ("y", col(|w| w.north)),
            ("z", col(|w| w.elev)),
        ])
        .map_err(invalid)?;
        Ok(Table(batch))
    }

    /// `(n, 3)` positions at measured `depths` down `holes`, e.g. contacts.
    fn at<'py>(
        &self,
        py: Python<'py>,
        holes: Vec<String>,
        depths: &Bound<PyAny>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let depths = crate::args::finite(depths, "depths")?;
        crate::args::same_length(holes.len(), depths.len(), "depths")?;
        let points = holes
            .iter()
            .zip(depths)
            .map(|(h, d)| {
                let path = self
                    .paths
                    .get(h)
                    .ok_or_else(|| invalid(format!("unknown hole {h:?}")))?;
                Ok(position_at(path, d))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(crate::args::points_array(py, &points))
    }

    /// Interval midpoints with every interval column; intervals of holes
    /// without a collar are dropped.
    fn samples(&self) -> PyResult<PyPointSet> {
        let (batch, hole, from, to) = self.intervals()?;
        let ids = text(batch, hole)?;
        let (from, to) = (number(batch, from)?, number(batch, to)?);
        let mut coords = Vec::new();
        let mut keep = Vec::new();
        for (i, id) in ids.iter().enumerate() {
            if let (Some(id), Some(f), Some(t)) = (id, from[i], to[i])
                && let Some(path) = self.paths.get(id)
            {
                let (x, y, z) = position_at(path, (f + t) / 2.0);
                coords.push([x, y, z]);
                keep.push(i as u64);
            }
        }
        let keep = arrow_array::UInt64Array::from(keep);
        let attributes = arrow_select::take::take_record_batch(batch, &keep).map_err(invalid)?;
        let mut points = PointSet::new(coords, attributes).map_err(invalid)?;
        points.length_unit.clone_from(&self.length_unit);
        Ok(PyPointSet(points))
    }

    /// Length-weighted composites, never crossing a change of `domain` and
    /// located at their midpoints. Composites with none of the grades sampled
    /// are dropped. Each grade comes with its sampled length,
    /// `<grade>_length`, so the sum of grade × `<grade>_length` equals the
    /// input metal; `length` also counts unsampled ground. The hole column
    /// keeps its constructor name, followed by `from`, `to` and `length`.
    ///
    /// Parameters
    /// ----------
    /// length : float or None
    ///     Composite length in meters; None gives one composite per run of
    ///     `domain`.
    /// grades : sequence of str
    ///     Numeric columns, averaged over the length that carries a value.
    /// domain : str, optional
    ///     Column whose changes composites never cross.
    /// intervals : table, optional
    ///     Hole, from and to columns (named as in the constructor) of
    ///     non-overlapping intervals to composite to instead of `length`, e.g.
    ///     benches; ground outside them is left out.
    /// residual : {"keep", "drop", "merge"}
    ///     What to do with the tail of a run shorter than
    ///     `min_fraction * length`; merge adds it to the previous composite of
    ///     the run, if there is one.
    /// min_fraction : float
    ///     Tail length, as a fraction of `length`, below which `residual`
    ///     applies.
    /// categories : sequence of str
    ///     Categorical columns, composited to the value covering the most
    ///     length.
    #[pyo3(signature = (
        length, grades, *, domain=None, intervals=None, residual="keep", min_fraction=0.5,
        categories=vec![]
    ))]
    #[allow(clippy::too_many_arguments)]
    fn composite(
        &self,
        py: Python,
        length: Option<f64>,
        grades: Vec<String>,
        domain: Option<&str>,
        intervals: Option<&Bound<PyAny>>,
        residual: &str,
        min_fraction: f64,
        categories: Vec<String>,
    ) -> PyResult<PyPointSet> {
        let residual = match residual {
            "keep" => Residual::Keep,
            "drop" => Residual::Drop,
            "merge" => Residual::Merge,
            _ => {
                return Err(invalid(format!(
                    "residual must be keep, drop or merge, got {residual:?}"
                )));
            }
        };
        let (batch, hole, from_name, to_name) = self.intervals()?;
        let ids = text(batch, hole)?;
        let (from, to) = (number(batch, from_name)?, number(batch, to_name)?);
        let domains = match domain {
            Some(d) => text(batch, d)?,
            None => vec![Some(String::new()); batch.num_rows()],
        };
        let columns = grades
            .iter()
            .map(|g| number(batch, g))
            .collect::<PyResult<Vec<_>>>()?;
        let mut labels: Vec<Vec<String>> = vec![vec![]; categories.len()];
        let mut codes: Vec<Vec<Option<f64>>> = Vec::new();
        for (c, name) in categories.iter().enumerate() {
            let mut index: HashMap<String, usize> = HashMap::new();
            let column = text(batch, name)?
                .into_iter()
                .map(|v| {
                    v.map(|v| {
                        let n = index.len();
                        *index.entry(v.clone()).or_insert_with(|| {
                            labels[c].push(v);
                            n
                        }) as f64
                    })
                })
                .collect();
            codes.push(column);
        }
        let mut per_hole: BTreeMap<&str, Vec<(f64, f64, String, HashMap<String, f64>)>> =
            BTreeMap::new();
        for i in 0..batch.num_rows() {
            let (Some(id), Some(f), Some(t)) = (&ids[i], from[i], to[i]) else {
                continue;
            };
            let values = grades
                .iter()
                .zip(&columns)
                .chain(categories.iter().zip(&codes))
                .filter_map(|(g, c)| c[i].map(|v| (g.clone(), v)))
                .collect();
            let dom = domains[i].clone().unwrap_or_default();
            per_hole.entry(id).or_default().push((f, t, dom, values));
        }
        let mut targets: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
        if let Some(table) = intervals {
            let table = to_batch(table)?;
            for ((id, f), t) in text(&table, hole)?
                .into_iter()
                .zip(number(&table, from_name)?)
                .zip(number(&table, to_name)?)
            {
                if let (Some(id), Some(f), Some(t)) = (id, f, t) {
                    targets.entry(id).or_default().push((f, t));
                }
            }
        }
        let params = CompositeParams {
            composite_length: length.unwrap_or(f64::INFINITY),
            domain_column: domain.unwrap_or_default().to_string(),
            grade_columns: grades.clone(),
            categorical_columns: categories.clone(),
            intervals: None,
            residual,
            min_fraction,
        };
        let to_targets = intervals.is_some();
        let composites = py.detach(|| {
            per_hole
                .par_iter()
                .filter(|(id, _)| self.paths.contains_key(**id))
                .map(|(id, rows)| match to_targets {
                    false => composite_intervals(id, rows, &params),
                    true => {
                        let params = CompositeParams {
                            intervals: Some(targets.get(*id).cloned().unwrap_or_default()),
                            ..params.clone()
                        };
                        composite_intervals(id, rows, &params)
                    }
                })
                .collect::<Result<Vec<_>, _>>()
        });
        let composites: Vec<_> = composites
            .map_err(invalid)?
            .into_iter()
            .flatten()
            .filter(|c| grades.is_empty() || grades.iter().any(|g| c.attributes.contains_key(g)))
            .collect();

        let coords = composites
            .iter()
            .map(|c| {
                let (x, y, z) =
                    position_at(&self.paths[&c.hole_id], (c.from_depth + c.to_depth) / 2.0);
                [x, y, z]
            })
            .collect();
        let floats = |f: &dyn Fn(&drillholes::Composite) -> Option<f64>| {
            Arc::new(composites.iter().map(f).collect::<Float64Array>()) as ArrayRef
        };
        let mut columns: Vec<(String, ArrayRef)> = vec![
            (
                hole.into(),
                Arc::new(StringArray::from_iter_values(
                    composites.iter().map(|c| &c.hole_id),
                )),
            ),
            ("from".into(), floats(&|c| Some(c.from_depth))),
            ("to".into(), floats(&|c| Some(c.to_depth))),
            ("length".into(), floats(&|c| Some(c.length))),
        ];
        if let Some(d) = domain {
            columns.push((
                d.to_string(),
                Arc::new(StringArray::from_iter_values(
                    composites.iter().map(|c| &c.domain),
                )),
            ));
        }
        for g in &grades {
            columns.push((g.clone(), floats(&|c| c.attributes.get(g).copied())));
            columns.push((
                format!("{g}_length"),
                floats(&|c| c.sampled.get(g).copied()),
            ));
        }
        for (name, labels) in categories.iter().zip(&labels) {
            let values = composites
                .iter()
                .map(|c| c.attributes.get(name).map(|&k| labels[k as usize].as_str()));
            columns.push((name.clone(), Arc::new(StringArray::from_iter(values))));
        }
        let attributes = RecordBatch::try_from_iter(columns).map_err(invalid)?;
        let units = self
            .intervals
            .as_ref()
            .map(|(t, ..)| boitata_core::units::units(t))
            .unwrap_or_default();
        let lengths: Vec<String> = ["from", "to", "length"]
            .into_iter()
            .map(String::from)
            .chain(grades.iter().map(|g| format!("{g}_length")))
            .collect();
        let grade_units: Vec<(&str, Option<&str>)> = units
            .iter()
            .filter(|(name, _)| grades.contains(name))
            .map(|(name, unit)| (name.as_str(), Some(unit.as_str())))
            .chain(
                lengths
                    .iter()
                    .map(|n| (n.as_str(), self.length_unit.as_deref())),
            )
            .collect();
        let attributes = crate::units::label(attributes, &grade_units)?;
        let mut points = PointSet::new(coords, attributes).map_err(invalid)?;
        points.length_unit.clone_from(&self.length_unit);
        Ok(PyPointSet(points))
    }

    /// Ore and waste runs down each hole: contiguous intervals above `cutoff`
    /// (or of an `ore` category), cleaned by mining rules. The runs of a hole
    /// partition its samples, so Σ grade × `length` equals the sample metal.
    ///
    /// Rules apply in order: internal dilution, then edge dilution, then
    /// minimum length. Samples missing the grade are left out, as unsampled
    /// ground, and samples missing the category are waste; like
    /// `composite(None, ...)`, gaps are skipped, not bridged.
    ///
    /// Parameters
    /// ----------
    /// grade : str or None
    ///     Numeric column; None with `category` for runs without grades.
    /// cutoff : float, optional
    ///     Samples at or above it are ore, unless `category` is given; also the
    ///     grade a run must keep when it takes in internal dilution.
    /// category : str, optional
    ///     Column whose values in `ore` flag ore samples.
    /// ore : sequence of str
    ///     Ore values of `category`.
    /// min_length : float
    ///     Runs spanning less, shortest first, take their neighbors' flag and
    ///     merge with them.
    /// max_dilution : float
    ///     Waste runs between ore spanning at most this are taken into the ore
    ///     when the grade of the three together stays at or above `cutoff`.
    /// edge : float
    ///     Waste taken into each ore run on each side where it meets waste.
    ///
    /// Returns
    /// -------
    /// Table
    ///     Hole (named as in the constructor), `from`, `to`, `length` (sampled
    ///     length), the grade, and `ore`, one row per run down each hole.
    #[pyo3(signature = (
        grade, *, cutoff=None, category=None, ore=vec![], min_length=0.0, max_dilution=0.0,
        edge=0.0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn runs(
        &self,
        py: Python,
        grade: Option<&str>,
        cutoff: Option<f64>,
        category: Option<&str>,
        ore: Vec<String>,
        min_length: f64,
        max_dilution: f64,
        edge: f64,
    ) -> PyResult<Table> {
        if cutoff.is_some_and(|c| !c.is_finite()) {
            return Err(invalid("cutoff must be a finite number"));
        }
        let (batch, hole, from_name, to_name) = self.intervals()?;
        let n = batch.num_rows();
        let grades = match grade {
            Some(g) => number(batch, g)?,
            None => vec![Some(0.0); n],
        };
        let flags: Vec<Option<bool>> = match (category, cutoff) {
            (Some(c), _) => {
                if ore.is_empty() {
                    return Err(invalid("give the ore values of the category"));
                }
                text(batch, c)?
                    .into_iter()
                    .map(|v| Some(v.is_some_and(|v| ore.contains(&v))))
                    .collect()
            }
            (None, Some(c)) if grade.is_some() => {
                grades.iter().map(|g| g.map(|g| g >= c)).collect()
            }
            _ => {
                return Err(invalid(
                    "give a grade and cutoff, or a category and its ore values",
                ));
            }
        };
        let ids = text(batch, hole)?;
        let (from, to) = (number(batch, from_name)?, number(batch, to_name)?);
        let mut per_hole: BTreeMap<&str, Vec<(f64, f64, f64, bool)>> = BTreeMap::new();
        for i in 0..n {
            if let (Some(id), Some(f), Some(t), Some(g), Some(o)) =
                (&ids[i], from[i], to[i], grades[i], flags[i])
                && self.paths.contains_key(id)
            {
                per_hole.entry(id).or_default().push((f, t, g, o));
            }
        }
        let rules = RunRules {
            cutoff,
            min_length,
            max_dilution,
            edge,
        };
        let runs = py
            .detach(|| {
                per_hole
                    .par_iter()
                    .map(|(id, s)| ore_runs(s, &rules).map(|r| (*id, r)))
                    .collect::<Result<Vec<_>, _>>()
            })
            .map_err(invalid)?;
        let rows: Vec<(&str, &Run)> = runs
            .iter()
            .flat_map(|(id, r)| r.iter().map(move |r| (*id, r)))
            .collect();
        let col = |f: fn(&Run) -> f64| {
            Arc::new(Float64Array::from_iter_values(rows.iter().map(|r| f(r.1)))) as ArrayRef
        };
        let mut columns: Vec<(&str, ArrayRef)> = vec![
            (
                hole,
                Arc::new(StringArray::from_iter_values(rows.iter().map(|r| r.0))),
            ),
            ("from", col(|r| r.from)),
            ("to", col(|r| r.to)),
            ("length", col(|r| r.length)),
        ];
        if let Some(g) = grade {
            columns.push((g, col(|r| r.grade)));
        }
        columns.push((
            "ore",
            Arc::new(BooleanArray::from_iter(rows.iter().map(|r| Some(r.1.ore)))),
        ));
        Ok(Table(RecordBatch::try_from_iter(columns).map_err(invalid)?))
    }

    /// Splits each hole's path against a closed `mesh`, wherever it crosses
    /// the surface.
    ///
    /// The path is sampled every `step` of depth and a crossing between two
    /// samples is bisected down to `tolerance`; a boundary crossed more than
    /// once within one `step` is missed. Every depth down every hole falls in
    /// exactly one row, so piping the result through `merge_intervals` splits
    /// assays exactly at the mesh instead of at a logged contact.
    ///
    /// Parameters
    /// ----------
    /// mesh : Mesh
    ///     A closed mesh.
    /// step : float
    ///     Sampling step down each hole, in meters.
    /// tolerance : float
    ///     Depth tolerance the crossing is refined to, in meters.
    ///
    /// Returns
    /// -------
    /// Table
    ///     Hole (named as in the constructor), `FROM`, `TO` and `INSIDE`
    ///     (whether the run is inside `mesh`), one row per run down each hole
    ///     — matching `merge_intervals`'s own column names.
    #[pyo3(signature = (mesh, *, step=1.0, tolerance=0.01))]
    fn mesh_intervals(
        &self,
        py: Python<'_>,
        mesh: &Mesh,
        step: f64,
        tolerance: f64,
    ) -> PyResult<Table> {
        let solid = mesh.solid()?;
        let runs: Vec<(&String, Vec<(f64, f64, bool)>)> = py.detach(|| {
            self.paths
                .par_iter()
                .map(|(id, path)| {
                    (
                        id,
                        drillholes::mesh_intervals(path, step, tolerance, |p| solid.contains(p)),
                    )
                })
                .collect()
        });
        let rows: Vec<(&str, (f64, f64, bool))> = runs
            .iter()
            .flat_map(|(id, r)| r.iter().map(move |&run| (id.as_str(), run)))
            .collect();
        let col = |f: fn((f64, f64, bool)) -> f64| {
            Arc::new(Float64Array::from_iter_values(rows.iter().map(|r| f(r.1)))) as ArrayRef
        };
        let batch = RecordBatch::try_from_iter([
            (
                self.hole.as_str(),
                Arc::new(StringArray::from_iter_values(rows.iter().map(|r| r.0))) as ArrayRef,
            ),
            ("FROM", col(|r| r.0)),
            ("TO", col(|r| r.1)),
            (
                "INSIDE",
                Arc::new(BooleanArray::from_iter(rows.iter().map(|r| Some(r.1.2)))),
            ),
        ])
        .map_err(invalid)?;
        Ok(Table(batch))
    }

    fn __len__(&self) -> usize {
        self.paths.len()
    }

    fn __repr__(&self) -> String {
        let n = self.intervals.as_ref().map_or(0, |i| i.0.num_rows());
        format!("Drillholes({} holes, {n} intervals)", self.paths.len())
    }
}

impl Drillholes {
    pub fn stations(&self) -> Vec<[f64; 3]> {
        self.paths
            .values()
            .flatten()
            .map(|w| [w.east, w.north, w.elev])
            .collect()
    }

    fn intervals(&self) -> PyResult<(&RecordBatch, &str, &str, &str)> {
        let (b, h, f, t) = self
            .intervals
            .as_ref()
            .ok_or_else(|| invalid("no interval table was given"))?;
        Ok((b, h, f, t))
    }
}

/// Splits two interval tables (e.g. assays and geology) at the union of
/// their boundaries, hole by hole. The result has the hole, from and to
/// columns followed by every other column of both tables, null where a table
/// has no interval; `right` columns that clash get a `_right` suffix.
/// Overlapping intervals within a table raise `ValueError` naming the
/// first few holes; resolve them before merging.
#[pyfunction]
#[pyo3(signature = (left, right, *, hole="HOLE_ID", from_="FROM", to="TO"))]
fn merge_intervals(
    left: &Bound<PyAny>,
    right: &Bound<PyAny>,
    hole: &str,
    from_: &str,
    to: &str,
) -> PyResult<Table> {
    let (left, right) = (to_batch(left)?, to_batch(right)?);
    let rows_by_hole = |batch: &RecordBatch| -> PyResult<BTreeMap<String, Vec<usize>>> {
        let mut out: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, id) in text(batch, hole)?.into_iter().enumerate() {
            if let Some(id) = id {
                out.entry(id).or_default().push(i);
            }
        }
        Ok(out)
    };
    let spans = |batch: &RecordBatch| -> PyResult<Vec<(f64, f64)>> {
        Ok(number(batch, from_)?
            .into_iter()
            .zip(number(batch, to)?)
            .map(|(f, t)| (f.unwrap_or(f64::NAN), t.unwrap_or(f64::NAN)))
            .collect())
    };
    let (lh, rh) = (rows_by_hole(&left)?, rows_by_hole(&right)?);
    let (ls, rs) = (spans(&left)?, spans(&right)?);
    let holes: std::collections::BTreeSet<&String> = lh.keys().chain(rh.keys()).collect();

    let (mut ids, mut from, mut upto, mut li, mut ri) = (vec![], vec![], vec![], vec![], vec![]);
    let mut overlaps = vec![];
    for id in holes {
        let pick = |rows: Option<&Vec<usize>>, s: &[(f64, f64)]| {
            let rows: Vec<usize> = rows
                .into_iter()
                .flatten()
                .copied()
                .filter(|&i| s[i].0.is_finite() && s[i].1 > s[i].0)
                .collect();
            let spans: Vec<(f64, f64)> = rows.iter().map(|&i| s[i]).collect();
            (rows, spans)
        };
        let (lrows, lspans) = pick(lh.get(id), &ls);
        let (rrows, rspans) = pick(rh.get(id), &rs);
        match drillholes::merge_intervals(&lspans, &rspans) {
            Ok(pieces) => {
                for (f, t, a, b) in pieces {
                    ids.push(id.clone());
                    from.push(f);
                    upto.push(t);
                    li.push(a.map(|k| lrows[k] as u64));
                    ri.push(b.map(|k| rrows[k] as u64));
                }
            }
            Err(DrillholeError::OverlappingIntervals {
                table,
                first,
                second,
            }) => {
                let (side, s) = if table == 'a' {
                    ("left", &lspans)
                } else {
                    ("right", &rspans)
                };
                let [(f1, t1), (f2, t2)] = [s[first], s[second]];
                overlaps.push(format!("{side} {id}: {f1}-{t1} and {f2}-{t2}"));
            }
            Err(e) => return Err(invalid(e)),
        }
    }
    if !overlaps.is_empty() {
        return Err(invalid(format!(
            "overlapping intervals in {} holes, resolve them before merging: {}",
            overlaps.len(),
            overlaps[..overlaps.len().min(5)].join("; ")
        )));
    }

    let take = |batch: &RecordBatch, idx: &[Option<u64>]| -> PyResult<Vec<(String, ArrayRef)>> {
        let idx = arrow_array::UInt64Array::from(idx.to_vec());
        let schema = batch.schema();
        schema
            .fields()
            .iter()
            .zip(batch.columns())
            .filter(|(f, _)| ![hole, from_, to].contains(&f.name().as_str()))
            .map(|(f, c)| {
                let taken = arrow_select::take::take(c, &idx, None).map_err(invalid)?;
                Ok((f.name().clone(), taken))
            })
            .collect()
    };
    let mut columns: Vec<(String, ArrayRef)> = vec![
        (hole.into(), Arc::new(StringArray::from(ids))),
        (from_.into(), Arc::new(Float64Array::from(from))),
        (to.into(), Arc::new(Float64Array::from(upto))),
    ];
    columns.extend(take(&left, &li)?);
    for (name, column) in take(&right, &ri)? {
        let name = if columns.iter().any(|(n, _)| *n == name) {
            format!("{name}_right")
        } else {
            name
        };
        columns.push((name, column));
    }
    Ok(Table(RecordBatch::try_from_iter(columns).map_err(invalid)?))
}

type Named = Vec<(String, Vec<bool>)>;

fn named(flags: checks::Flags) -> Named {
    flags.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

fn numeric_columns(batch: &RecordBatch) -> PyResult<Vec<Vec<Option<f64>>>> {
    batch
        .schema()
        .fields()
        .iter()
        .filter(|f| f.data_type().is_numeric())
        .map(|f| number(batch, f.name()))
        .collect()
}

fn with_sentinels(mut flags: Named, batch: &RecordBatch, sentinels: &[f64]) -> PyResult<Named> {
    let rows = checks::sentinel_rows(&numeric_columns(batch)?, batch.num_rows(), sentinels);
    flags.push(("sentinel".into(), rows));
    Ok(flags)
}

fn is_table(value: &Bound<PyAny>) -> PyResult<bool> {
    Ok(value.is_instance_of::<PyDict>()
        || value.hasattr("column_names")?
        || value.hasattr("columns")?)
}

fn flags_table(
    flags: Named,
    sentinels: &[f64],
    mut meta: HashMap<String, String>,
) -> PyResult<RecordBatch> {
    let columns: Vec<(String, ArrayRef)> = flags
        .into_iter()
        .map(|(name, f)| (name, Arc::new(BooleanArray::from(f)) as ArrayRef))
        .collect();
    let batch = RecordBatch::try_from_iter(columns).map_err(invalid)?;
    let listed = sentinels
        .iter()
        .map(|s| format!("{s:?}"))
        .collect::<Vec<_>>()
        .join(",");
    meta.insert("sentinels".into(), listed);
    let schema = batch.schema().as_ref().clone().with_metadata(meta);
    batch.with_schema(Arc::new(schema)).map_err(invalid)
}

fn is_text(t: &DataType) -> bool {
    matches!(t, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

/// Rows with text in numeric-looking columns, the (row, column, value) of
/// each, and the columns checked.
type TextFound = (Vec<bool>, Vec<(usize, String, String)>, Vec<String>);

fn text_check(
    batch: &RecordBatch,
    keys: &[&str],
    grades: Option<&[String]>,
) -> PyResult<TextFound> {
    let mut rows = vec![false; batch.num_rows()];
    let (mut found, mut columns) = (vec![], vec![]);
    for field in batch.schema().fields() {
        let name = field.name();
        let chosen = match grades {
            Some(g) => g.contains(name),
            None => !keys.contains(&name.as_str()),
        };
        if !chosen || !is_text(field.data_type()) {
            continue;
        }
        let values = text(batch, name)?;
        if grades.is_none() && !checks::numeric_looking(&values) {
            continue;
        }
        for (i, hit) in checks::text_values(&values).into_iter().enumerate() {
            if hit {
                rows[i] = true;
                found.push((i, name.clone(), values[i].clone().unwrap_or_default()));
            }
        }
        columns.push(name.clone());
    }
    found.sort_by_key(|f| f.0);
    Ok((rows, found, columns))
}

struct Checked {
    name: String,
    batch: RecordBatch,
    flags: Named,
    ids: Vec<Option<String>>,
    meta: HashMap<String, String>,
}

#[derive(Default)]
struct Details {
    table: Vec<String>,
    row: Vec<i64>,
    hole: Vec<Option<String>>,
    check: Vec<&'static str>,
    column: Vec<String>,
    value: Vec<String>,
    suggestion: Vec<Option<String>>,
}

impl Details {
    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        table: &str,
        row: usize,
        hole: Option<String>,
        check: &'static str,
        column: &str,
        value: String,
        suggestion: Option<String>,
    ) {
        self.table.push(table.into());
        self.row.push(row as i64);
        self.hole.push(hole);
        self.check.push(check);
        self.column.push(column.into());
        self.value.push(value);
        self.suggestion.push(suggestion);
    }

    fn table(self) -> PyResult<Table> {
        let batch = RecordBatch::try_from_iter([
            ("table", Arc::new(StringArray::from(self.table)) as ArrayRef),
            ("row", Arc::new(Int64Array::from(self.row))),
            ("hole", Arc::new(StringArray::from(self.hole))),
            ("check", Arc::new(StringArray::from(self.check))),
            ("column", Arc::new(StringArray::from(self.column))),
            ("value", Arc::new(StringArray::from(self.value))),
            ("suggestion", Arc::new(StringArray::from(self.suggestion))),
        ])
        .map_err(invalid)?;
        Ok(Table(batch))
    }
}

fn json<T: serde::Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string(value).map_err(invalid)
}

/// Per-record flags of drillhole tables, a summary of every check and the
/// values behind the checks that suggest a correction.
///
/// Rows with a missing or sentinel required value (id, coordinates,
/// depths, angles, from and to) are left out of the other checks, so each
/// defect is flagged once. Nothing is sorted or dropped; see
/// `fix_drillholes`.
///
/// Parameters
/// ----------
/// collar : table
///     Hole id and collar coordinates, plus the hole length if `max_depth`
///     is given.
/// survey : table, optional
///     Hole id, depth, azimuth and dip (or inclination).
/// intervals : table or mapping of str to table, optional
///     Interval tables, e.g. ``{"assay": assay, "geology": geology}``; a
///     single table is named ``"intervals"``.
/// hole, x, y, z, at, azimuth, dip, inclination, from_, to : str
///     Column names, as in `Drillholes`. Dip is positive down; `inclination`,
///     from vertical, is used instead when given.
/// max_depth : str, optional
///     Collar column with the hole length; enables the ``past_depth`` check.
/// grades : sequence of str, optional
///     Columns that must hold numbers, searched for ``text_values`` in every
///     table that has them. By default, every text column other than the
///     ones named above in which most values are numbers.
/// nodata : sequence of float
///     Values that stand for missing data, searched in every numeric column.
/// max_deviation : float
///     Largest angle, in degrees, between the directions of consecutive
///     survey stations. A single wrong station is flagged twice: on itself
///     and on the station below.
/// tolerance : float
///     Depth tolerance of gaps, overlaps and ``past_depth``.
///
/// Returns
/// -------
/// flags : dict of str to Table
///     One boolean Table per input table (``"collar"``, ``"survey"`` and each
///     interval table), in input row order, one column per check:
///
///     - ``duplicate``: collar id, or survey hole and depth, seen on an
///       earlier row.
///     - ``missing``: null or non-finite required value.
///     - ``out_of_range``: negative depth, length or from; azimuth outside
///       [0, 360]; dip outside [-90, 90].
///     - ``sentinel``: a numeric column, or a text column checked for
///       ``text_values``, holds one of `nodata`.
///     - ``text_values``: a value of a numeric-looking column is not a
///       number, e.g. ``"NS"`` or ``"<0.01"``.
///     - ``no_survey``, ``no_<name>``: collar without survey or intervals.
///     - ``id_mismatch``: survey or interval hole id that matches no collar
///       id exactly, but one after trimming whitespace and case-folding.
///     - ``no_collar``: survey or interval of a hole without collar.
///     - ``deviation``: survey station more than `max_deviation` from the
///       station above.
///     - ``dip_sign``: every station of the hole dips against the sign most
///       holes share, e.g. a hole entered pointing up.
///     - ``inverted``: interval with from ≥ to.
///     - ``gap``: interval starting below the end of the one above.
///     - ``overlap``: interval starting above the end of the ones kept above
///       it; dropping these keeps the first of overlapping intervals.
///     - ``past_depth``: survey or interval deeper than the hole length.
/// summary : Table
///     ``table``, ``check``, ``rows`` flagged and distinct ``holes``, for
///     every check including those that found nothing.
/// details : Table
///     One row per flagged value of ``dip_sign``, ``id_mismatch`` and
///     ``text_values``: ``table``, ``row``, ``hole``, ``check``, ``column``,
///     ``value`` as text and the ``suggestion`` (the negated dip or matching
///     collar id; null for text values).
#[pyfunction]
#[pyo3(signature = (
    collar, survey=None, intervals=None, *, hole="HOLE_ID", x="X", y="Y", z="Z", at="DEPTH",
    azimuth="AZIMUTH", dip=Some("DIP"), inclination=None, from_="FROM", to="TO",
    max_depth=None, grades=None, nodata=vec![-99.0, -999.0, -9999.0, 1e21], max_deviation=20.0,
    tolerance=1e-6
))]
#[allow(clippy::too_many_arguments)]
fn check_drillholes<'py>(
    py: Python<'py>,
    collar: &Bound<PyAny>,
    survey: Option<&Bound<PyAny>>,
    intervals: Option<&Bound<PyAny>>,
    hole: &str,
    x: &str,
    y: &str,
    z: &str,
    at: &str,
    azimuth: &str,
    dip: Option<&str>,
    inclination: Option<&str>,
    from_: &str,
    to: &str,
    max_depth: Option<&str>,
    grades: Option<Vec<String>>,
    nodata: Vec<f64>,
    max_deviation: f64,
    tolerance: f64,
) -> PyResult<(Bound<'py, PyDict>, Table, Table)> {
    let s = &nodata;
    let collar = to_batch(collar)?;
    let collar_ids = text(&collar, hole)?;
    let lengths = max_depth.map(|d| number(&collar, d)).transpose()?;
    let hole_lengths = lengths.as_ref().map(|l| {
        let mut out = HashMap::new();
        for (id, v) in collar_ids.iter().zip(l) {
            if let (Some(id), Some(v)) = (id, v)
                && v.is_finite()
                && !checks::is_sentinel(*v, s)
            {
                out.entry(id.clone()).or_insert(*v);
            }
        }
        out
    });
    let coords = [
        number(&collar, x)?,
        number(&collar, y)?,
        number(&collar, z)?,
    ];
    let flags = checks::check_collars(
        &collar_ids,
        [&coords[0], &coords[1], &coords[2]],
        lengths.as_deref(),
        s,
    );
    let mut details = Details::default();
    let mut tables = vec![Checked {
        name: "collar".into(),
        flags: with_sentinels(named(flags), &collar, s)?,
        batch: collar,
        ids: collar_ids.clone(),
        meta: HashMap::new(),
    }];

    if let Some(survey) = survey {
        let batch = to_batch(survey)?;
        let ids = text(&batch, hole)?;
        let (column, raw) = match (inclination, dip) {
            (Some(name), _) => (name, number(&batch, name)?),
            (None, Some(name)) => (name, number(&batch, name)?),
            (None, None) => return Err(invalid("give the dip or inclination column")),
        };
        let upward = |v: f64| if inclination.is_some() { 180.0 - v } else { -v };
        let angle: Vec<Option<f64>> = if inclination.is_some() {
            raw.iter()
                .map(|v| {
                    v.map(|v| {
                        if checks::is_sentinel(v, s) {
                            v
                        } else {
                            90.0 - v
                        }
                    })
                })
                .collect()
        } else {
            raw.clone()
        };
        let flags = checks::check_survey(
            &ids,
            &number(&batch, at)?,
            &number(&batch, azimuth)?,
            &angle,
            hole_lengths.as_ref(),
            max_deviation,
            tolerance,
            s,
        );
        let (_, sign) = flags
            .iter()
            .find(|(k, _)| *k == "dip_sign")
            .expect("dip_sign");
        for (i, _) in sign.iter().enumerate().filter(|(_, f)| **f) {
            let v = raw[i].expect("usable station");
            let fmt = |v: f64| format!("{v:?}");
            let suggestion = Some(fmt(upward(v)));
            details.push(
                "survey",
                i,
                ids[i].clone(),
                "dip_sign",
                column,
                fmt(v),
                suggestion,
            );
        }
        let kind = if inclination.is_some() {
            "inclination"
        } else {
            "dip"
        };
        tables.push(Checked {
            name: "survey".into(),
            flags: with_sentinels(named(flags), &batch, s)?,
            batch,
            ids,
            meta: [(kind.to_string(), column.to_string())].into(),
        });
    }
    let mut interval_tables = vec![];
    if let Some(t) = intervals {
        match t.cast::<PyDict>() {
            Ok(d) if d.values().iter().all(|v| is_table(&v).unwrap_or(false)) => {
                for (k, v) in d.iter() {
                    interval_tables.push((k.extract::<String>()?, to_batch(&v)?));
                }
            }
            _ => interval_tables.push(("intervals".to_string(), to_batch(t)?)),
        }
    }
    for (name, batch) in interval_tables {
        let ids = text(&batch, hole)?;
        let flags = checks::check_intervals(
            &ids,
            &number(&batch, from_)?,
            &number(&batch, to)?,
            hole_lengths.as_ref(),
            tolerance,
            s,
        );
        tables.push(Checked {
            name,
            flags: with_sentinels(named(flags), &batch, s)?,
            batch,
            ids,
            meta: HashMap::new(),
        });
    }

    if let Some(grades) = &grades {
        for g in grades {
            if !tables
                .iter()
                .any(|t| t.batch.schema().field_with_name(g).is_ok())
            {
                return Err(invalid(format!("column {g} not found in any table")));
            }
        }
    }
    let keys: Vec<&str> = [hole, x, y, z, at, azimuth, from_, to]
        .into_iter()
        .chain(dip)
        .chain(inclination)
        .chain(max_depth)
        .collect();
    for t in &mut tables {
        let (rows, found, columns) = text_check(&t.batch, &keys, grades.as_deref())?;
        let parsed = columns
            .iter()
            .map(|c| {
                let values = text(&t.batch, c)?;
                Ok(values
                    .iter()
                    .map(|v| v.as_deref().and_then(checks::parse_number))
                    .collect())
            })
            .collect::<PyResult<Vec<_>>>()?;
        let hidden = checks::sentinel_rows(&parsed, t.batch.num_rows(), s);
        let (_, sentinel) = t
            .flags
            .iter_mut()
            .find(|(k, _)| k == "sentinel")
            .expect("sentinel");
        sentinel.iter_mut().zip(hidden).for_each(|(a, b)| *a |= b);
        for (i, column, value) in found {
            details.push(
                &t.name,
                i,
                t.ids[i].clone(),
                "text_values",
                &column,
                value,
                None,
            );
        }
        t.flags.push(("text_values".into(), rows));
        t.meta.insert("text_values".into(), json(&columns)?);
        t.meta.insert("hole".into(), hole.into());
    }

    let (collar_table, others) = tables.split_first_mut().expect("collar");
    for t in others {
        let suggested = checks::id_mismatch(&t.ids, &collar_ids);
        let mut renames = BTreeMap::new();
        for (i, c) in suggested.iter().enumerate() {
            if let Some(c) = c {
                let id = t.ids[i].clone().expect("matched id");
                renames.insert(id.clone(), c.clone());
                details.push(
                    &t.name,
                    i,
                    Some(id.clone()),
                    "id_mismatch",
                    hole,
                    id,
                    Some(c.clone()),
                );
            }
        }
        let matched: Vec<Option<String>> = t
            .ids
            .iter()
            .zip(&suggested)
            .map(|(id, c)| c.clone().or_else(|| id.clone()))
            .collect();
        let mismatch: Vec<bool> = suggested.iter().map(Option::is_some).collect();
        let no_collar = checks::absent(&t.ids, &collar_ids)
            .into_iter()
            .zip(&mismatch)
            .map(|(a, m)| a && !m)
            .collect();
        t.flags.push(("id_mismatch".into(), mismatch));
        t.flags.push(("no_collar".into(), no_collar));
        t.meta.insert("id_mismatch".into(), json(&renames)?);
        collar_table.flags.push((
            format!("no_{}", t.name),
            checks::absent(&collar_ids, &matched),
        ));
    }

    let (mut tn, mut c, mut rows, mut holes) = (vec![], vec![], vec![], vec![]);
    let out = PyDict::new(py);
    for t in tables {
        for (check, f) in &t.flags {
            let hit: HashSet<&str> = f
                .iter()
                .zip(&t.ids)
                .filter(|(f, _)| **f)
                .filter_map(|(_, id)| id.as_deref())
                .collect();
            tn.push(t.name.clone());
            c.push(check.clone());
            rows.push(f.iter().filter(|f| **f).count() as i64);
            holes.push(hit.len() as i64);
        }
        out.set_item(&t.name, Table(flags_table(t.flags, s, t.meta)?))?;
    }
    let summary = RecordBatch::try_from_iter([
        ("table", Arc::new(StringArray::from(tn)) as ArrayRef),
        ("check", Arc::new(StringArray::from(c))),
        ("rows", Arc::new(Int64Array::from(rows))),
        ("holes", Arc::new(Int64Array::from(holes))),
    ])
    .map_err(invalid)?;
    Ok((out, Table(summary), details.table()?))
}

/// Tables cleaned by the flags of `check_drillholes`, one rule per check.
///
/// Rows are dropped where any check whose rule drops is flagged; ``gap``,
/// ``no_survey`` and ``no_<name>`` are informational and change nothing.
///
/// Parameters
/// ----------
/// flags : mapping of str to Table
///     The flags returned by `check_drillholes`.
/// tables : mapping of str to table
///     The checked tables under the same names, e.g. ``{"collar": collar,
///     "survey": survey, "assay": assay}``.
/// missing, duplicates, inverted, out_of_range, deviation, no_collar, past_depth : {"drop", "keep"}
///     Rule for each check. A duplicate is the later row, so dropping keeps
///     the first. Dropping both stations flagged around one wrong survey
///     station also drops a good one. ``past_depth`` defaults to keep: the
///     hole length may be what is wrong.
/// overlaps : {"keep_first", "keep"}
///     ``keep_first`` drops the flagged intervals, which leaves the first of
///     overlapping intervals.
/// sentinels : {"null", "drop", "keep"}
///     ``null`` sets the sentinel values to null, leaving the rest of the row.
/// dip_sign : {"keep", "negate"}
///     ``negate`` turns the flagged holes' dips (or inclinations) down.
/// id_mismatch : {"rename", "keep"}
///     ``rename`` replaces the flagged ids with the collar id they match.
/// text_values : {"null", "half", "limit", "keep"}
///     Unless ``keep``, the checked columns become numbers and their text
///     values null. ``half`` and ``limit`` instead turn a below-detection
///     value ``"<x"`` into ``x / 2`` or ``x``.
///
/// Returns
/// -------
/// tables : dict of str to Table
///     The cleaned tables, under the same names.
/// log : Table
///     ``table``, ``check``, ``action`` and ``rows`` changed, for every check
///     whose rule is not keep.
#[pyfunction]
#[pyo3(signature = (
    flags, tables, *, missing="drop", duplicates="drop", inverted="drop", out_of_range="drop",
    overlaps="keep_first", sentinels="null", deviation="drop", no_collar="drop", past_depth="keep",
    dip_sign="keep", id_mismatch="rename", text_values="null"
))]
#[allow(clippy::too_many_arguments)]
fn fix_drillholes<'py>(
    py: Python<'py>,
    flags: &Bound<PyDict>,
    tables: &Bound<PyDict>,
    missing: &str,
    duplicates: &str,
    inverted: &str,
    out_of_range: &str,
    overlaps: &str,
    sentinels: &str,
    deviation: &str,
    no_collar: &str,
    past_depth: &str,
    dip_sign: &str,
    id_mismatch: &str,
    text_values: &str,
) -> PyResult<(Bound<'py, PyDict>, Table)> {
    let drop_keep: &[&str] = &["drop", "keep"];
    let rules = [
        ("missing", "missing", missing, drop_keep),
        ("duplicate", "duplicates", duplicates, drop_keep),
        ("inverted", "inverted", inverted, drop_keep),
        ("out_of_range", "out_of_range", out_of_range, drop_keep),
        ("overlap", "overlaps", overlaps, &["keep_first", "keep"]),
        (
            "text_values",
            "text_values",
            text_values,
            &["null", "half", "limit", "keep"],
        ),
        (
            "sentinel",
            "sentinels",
            sentinels,
            &["null", "drop", "keep"],
        ),
        ("deviation", "deviation", deviation, drop_keep),
        ("no_collar", "no_collar", no_collar, drop_keep),
        ("past_depth", "past_depth", past_depth, drop_keep),
        ("dip_sign", "dip_sign", dip_sign, &["keep", "negate"]),
        (
            "id_mismatch",
            "id_mismatch",
            id_mismatch,
            &["rename", "keep"],
        ),
    ];
    for (_, arg, rule, allowed) in &rules {
        if !allowed.contains(rule) {
            return Err(invalid(format!(
                "{arg} must be one of {}, got {rule:?}",
                allowed.join(", ")
            )));
        }
    }
    let (mut t, mut c, mut a, mut n) = (vec![], vec![], vec![], vec![]);
    let out = PyDict::new(py);
    for (name, table) in tables.iter() {
        let name: String = name.extract()?;
        let mut batch = to_batch(&table)?;
        let f = flags
            .get_item(&name)?
            .ok_or_else(|| invalid(format!("no flags for table {name:?}")))?;
        let f = to_batch(&f)?;
        if f.num_rows() != batch.num_rows() {
            return Err(invalid(format!(
                "flags of {name:?} have {} rows, the table {}",
                f.num_rows(),
                batch.num_rows()
            )));
        }
        let mut drop = vec![false; batch.num_rows()];
        for (check, _, rule, _) in &rules {
            let Some(column) = f.column_by_name(check) else {
                continue;
            };
            if *rule == "keep" {
                continue;
            }
            let column = arrow_cast::cast(column, &DataType::Boolean).map_err(invalid)?;
            let hit: Vec<bool> = column
                .as_boolean()
                .iter()
                .map(|v| v.unwrap_or(false))
                .collect();
            match *check {
                "sentinel" if *rule == "null" => batch = null_sentinels(&batch, &f, &name)?,
                "dip_sign" => batch = negate_dips(&batch, &f, &hit, &name)?,
                "id_mismatch" => batch = rename_ids(&batch, &f, &hit, &name)?,
                "text_values" => batch = parse_text(&batch, &f, rule, &name)?,
                _ => drop.iter_mut().zip(&hit).for_each(|(d, h)| *d |= h),
            }
            t.push(name.clone());
            c.push(check.to_string());
            a.push(rule.to_string());
            n.push(hit.iter().filter(|h| **h).count() as i64);
        }
        let keep = BooleanArray::from(drop.iter().map(|d| !d).collect::<Vec<_>>());
        let batch = arrow_select::filter::filter_record_batch(&batch, &keep).map_err(invalid)?;
        out.set_item(&name, Table(batch))?;
    }
    let log = RecordBatch::try_from_iter([
        ("table", Arc::new(StringArray::from(t)) as ArrayRef),
        ("check", Arc::new(StringArray::from(c))),
        ("action", Arc::new(StringArray::from(a))),
        ("rows", Arc::new(Int64Array::from(n))),
    ])
    .map_err(invalid)?;
    Ok((out, Table(log)))
}

fn meta(flags: &RecordBatch, key: &str, name: &str) -> PyResult<String> {
    flags
        .schema()
        .metadata()
        .get(key)
        .cloned()
        .ok_or_else(|| invalid(format!("flags of {name:?} do not list their {key}")))
}

fn replace_column(batch: &RecordBatch, name: &str, column: ArrayRef) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let i = schema.index_of(name).map_err(invalid)?;
    let mut columns = batch.columns().to_vec();
    columns[i] = column;
    let fields: Vec<_> = schema
        .fields()
        .iter()
        .zip(&columns)
        .map(|(f, c)| {
            f.as_ref()
                .clone()
                .with_data_type(c.data_type().clone())
                .with_nullable(true)
        })
        .collect();
    let schema = arrow_schema::Schema::new_with_metadata(fields, schema.metadata().clone());
    RecordBatch::try_new(Arc::new(schema), columns).map_err(invalid)
}

fn negate_dips(
    batch: &RecordBatch,
    flags: &RecordBatch,
    hit: &[bool],
    name: &str,
) -> PyResult<RecordBatch> {
    let (column, inclination) = match meta(flags, "dip", name) {
        Ok(c) => (c, false),
        Err(_) => (meta(flags, "inclination", name)?, true),
    };
    let values: Float64Array = number(batch, &column)?
        .into_iter()
        .zip(hit)
        .map(|(v, h)| {
            v.map(|v| {
                if !h {
                    v
                } else if inclination {
                    180.0 - v
                } else {
                    -v
                }
            })
        })
        .collect();
    let original = batch
        .column_by_name(&column)
        .expect("dip column")
        .data_type();
    let values = arrow_cast::cast(&values, original).map_err(invalid)?;
    replace_column(batch, &column, values)
}

fn rename_ids(
    batch: &RecordBatch,
    flags: &RecordBatch,
    hit: &[bool],
    name: &str,
) -> PyResult<RecordBatch> {
    let hole = meta(flags, "hole", name)?;
    let renames: HashMap<String, String> =
        serde_json::from_str(&meta(flags, "id_mismatch", name)?).map_err(invalid)?;
    let ids: StringArray = text(batch, &hole)?
        .into_iter()
        .zip(hit)
        .map(|(id, h)| match (id, h) {
            (Some(id), true) => Some(renames.get(&id).cloned().unwrap_or(id)),
            (id, _) => id,
        })
        .collect();
    let original = batch
        .column_by_name(&hole)
        .expect("hole column")
        .data_type();
    let ids = arrow_cast::cast(&ids, original).map_err(invalid)?;
    replace_column(batch, &hole, ids)
}

fn parse_text(
    batch: &RecordBatch,
    flags: &RecordBatch,
    rule: &str,
    name: &str,
) -> PyResult<RecordBatch> {
    let columns: Vec<String> =
        serde_json::from_str(&meta(flags, "text_values", name)?).map_err(invalid)?;
    let factor = match rule {
        "half" => Some(0.5),
        "limit" => Some(1.0),
        _ => None,
    };
    let mut batch = batch.clone();
    for column in columns {
        let values: Float64Array = text(&batch, &column)?
            .iter()
            .map(|v| {
                let v = v.as_deref()?;
                checks::parse_number(v).or_else(|| Some(checks::detection_limit(v)? * factor?))
            })
            .collect();
        batch = replace_column(&batch, &column, Arc::new(values))?;
    }
    Ok(batch)
}

fn null_sentinels(batch: &RecordBatch, flags: &RecordBatch, name: &str) -> PyResult<RecordBatch> {
    let values: Vec<f64> = flags
        .schema()
        .metadata()
        .get("sentinels")
        .ok_or_else(|| invalid(format!("flags of {name:?} do not list their sentinels")))?
        .split(',')
        .filter(|v| !v.is_empty())
        .map(|v| v.parse::<f64>().map_err(invalid))
        .collect::<PyResult<_>>()?;
    let text: Vec<String> = match flags.schema().metadata().get("text_values") {
        Some(t) => serde_json::from_str(t).map_err(invalid)?,
        None => vec![],
    };
    let columns = batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(field, col)| {
            let parsed: Vec<Option<f64>> = if col.data_type().is_numeric() {
                let as_f64 = arrow_cast::cast(col, &DataType::Float64).map_err(invalid)?;
                as_f64.as_primitive::<Float64Type>().iter().collect()
            } else if is_text(col.data_type()) && text.contains(field.name()) {
                text_of(col)?
                    .iter()
                    .map(|v| v.as_deref().and_then(checks::parse_number))
                    .collect()
            } else {
                return Ok(col.clone());
            };
            let mask: BooleanArray = parsed
                .iter()
                .map(|v| Some(v.is_some_and(|v| checks::is_sentinel(v, &values))))
                .collect();
            arrow_select::nullif::nullif(col.as_ref(), &mask).map_err(invalid)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let schema = batch.schema();
    let fields: Vec<_> = schema
        .fields()
        .iter()
        .map(|f| f.as_ref().clone().with_nullable(true))
        .collect();
    let schema = arrow_schema::Schema::new_with_metadata(fields, schema.metadata().clone());
    RecordBatch::try_new(Arc::new(schema), columns).map_err(invalid)
}

/// Moves each collar's elevation onto a surface; the hole moves with it.
///
/// Parameters
/// ----------
/// drillholes : Drillholes
/// surface : Mesh or BlockModel
///     A surface such as topography; where it overlaps itself in plan, the
///     highest elevation counts. A 2D block model's cell centers, lifted to
///     `column`, make the surface, as in `grid_surface`.
/// column : str, optional
///     Elevation column of a block model surface.
///
/// Returns
/// -------
/// drillholes : Drillholes
///     A copy with every hole shifted vertically by its collar's offset.
/// report : Table
///     ``hole``, ``z_before``, ``z_after`` and ``shift`` per hole. Holes the
///     surface does not cover keep their elevation, with a null shift, and
///     a ``UserWarning`` names them.
#[pyfunction]
#[pyo3(signature = (drillholes, surface, *, column=None))]
fn snap_to_surface(
    py: Python,
    drillholes: PyRef<Drillholes>,
    surface: &Bound<PyAny>,
    column: Option<&str>,
) -> PyResult<(Drillholes, Table)> {
    let mesh = if let Ok(m) = surface.cast::<Mesh>() {
        m.get().mesh.clone()
    } else if let Ok(model) = surface.cast::<crate::containers::PyBlockModel>() {
        let column = column.ok_or_else(|| invalid("a block model surface needs column="))?;
        let m = &model.get().0;
        let z = crate::args::floats(&crate::table::column(py, m.attributes(), column)?, "column")?;
        let z: Vec<Option<f64>> = z.into_iter().map(|v| (!v.is_nan()).then_some(v)).collect();
        blocks::grid_surface(m, &z).map_err(invalid)?
    } else {
        return Err(invalid("surface must be a Mesh or a 2D BlockModel"));
    };
    let surface = blocks::Surface::new(&mesh).map_err(invalid)?;
    let (mut paths, mut holes, mut before, mut after, mut shifts, mut off) =
        (BTreeMap::new(), vec![], vec![], vec![], vec![], vec![]);
    for (hole, path) in &drillholes.paths {
        let collar = path.first().map(|c| (c.east, c.north, c.elev));
        let elevation = collar.and_then(|(x, y, _)| surface.elevation(x, y));
        let (path, shift) = match elevation {
            Some(e) => {
                let (moved, shift) = drillholes::snap_collar(path, e);
                (moved, Some(shift))
            }
            None => {
                off.push(hole.clone());
                (path.clone(), None)
            }
        };
        holes.push(hole.clone());
        before.push(collar.map(|c| c.2));
        after.push(path.first().map(|c| c.elev));
        shifts.push(shift);
        paths.insert(hole.clone(), path);
    }
    if !off.is_empty() {
        let message = format!(
            "{} holes lie off the surface and keep their elevation: {}",
            off.len(),
            off.join(", ")
        );
        let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
        PyErr::warn(py, &category, &std::ffi::CString::new(message)?, 1)?;
    }
    let report = RecordBatch::try_from_iter([
        ("hole", Arc::new(StringArray::from(holes)) as ArrayRef),
        ("z_before", Arc::new(Float64Array::from(before))),
        ("z_after", Arc::new(Float64Array::from(after))),
        ("shift", Arc::new(Float64Array::from(shifts))),
    ])
    .map_err(invalid)?;
    let moved = Drillholes {
        paths,
        hole: drillholes.hole.clone(),
        intervals: drillholes.intervals.clone(),
        length_unit: drillholes.length_unit.clone(),
    };
    Ok((moved, Table(report)))
}

/// Straight holes on a regular collar grid over `targets`, for drilling
/// plans and spacing studies.
///
/// Grid nodes start at the footprint's lowest (along, across) corner, shifted
/// by `offset`. A node becomes a collar when a target lies within half a
/// cell of the hole collared there at the top of the targets; collars then
/// slide along their hole to `topography`, so inclined holes keep their
/// targets. Each hole runs down to the bottom of the targets.
///
/// Parameters
/// ----------
/// targets : BlockModel, PointSet or array of shape (n, 3)
///     Ground to drill; a block model spans its blocks' full height.
/// spacing : float or (float, float)
///     Collar spacing, or (along, across) spacing.
/// rotation : float
///     Azimuth of the along axis, degrees clockwise from north.
/// azimuth, dip : float
///     Hole direction in degrees; dip is positive down, 90 vertical.
/// topography : Mesh, optional
///     Surface the collars sit on; collars off it stay at the targets' top,
///     with a ``UserWarning``.
/// offset : (float, float)
///     (along, across) shift of the grid, in meters.
///
/// Returns
/// -------
/// Drillholes
///     Holes ``P0001``, ``P0002``, ... with one interval each, ``FROM`` 0 to
///     ``TO`` the hole length, so ``composite(length, [])`` cuts them into
///     composites. Holes collared at or below the bottom are left out.
#[pyfunction]
#[pyo3(signature = (
    targets, spacing, *, rotation=0.0, azimuth=0.0, dip=90.0, topography=None, offset=(0.0, 0.0)
))]
#[allow(clippy::too_many_arguments)]
fn planned_drillholes(
    py: Python,
    targets: &Bound<PyAny>,
    spacing: &Bound<PyAny>,
    rotation: f64,
    azimuth: f64,
    dip: f64,
    topography: Option<PyRef<Mesh>>,
    offset: (f64, f64),
) -> PyResult<Drillholes> {
    let spacing = match spacing.extract::<f64>() {
        Ok(s) => [s, s],
        Err(_) => spacing
            .extract::<[f64; 2]>()
            .map_err(|_| invalid("spacing must be a number or (along, across)"))?,
    };
    let points = crate::containers::coords_arg(targets)?;
    let heights: Vec<f64> = match targets.cast::<crate::containers::PyBlockModel>() {
        Ok(model) => model
            .get()
            .0
            .corners()
            .iter()
            .flatten()
            .map(|c| c[2])
            .collect(),
        Err(_) => points.iter().map(|p| p[2]).collect(),
    };
    if heights.is_empty() {
        return Err(invalid("targets are empty"));
    }
    let top = heights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let bottom = heights.iter().copied().fold(f64::INFINITY, f64::min);
    let surface = topography
        .map(|m| blocks::Surface::new(&m.mesh))
        .transpose()
        .map_err(invalid)?;
    let grid = drillholes::PlanGrid {
        spacing,
        rotation,
        azimuth,
        dip,
        offset: [offset.0, offset.1],
    };
    let ground = |x, y| surface.as_ref()?.elevation(x, y);
    let holes = drillholes::planned_holes(&points, top, bottom, &grid, ground).map_err(invalid)?;
    if holes.is_empty() {
        return Err(invalid(
            "no hole reaches below its collar; targets need height or topography above them",
        ));
    }
    let off = match &surface {
        Some(s) => holes
            .iter()
            .filter(|h| s.elevation(h.1[0].east, h.1[0].north).is_none())
            .count(),
        None => 0,
    };
    if off > 0 {
        let message = format!("{off} collars lie off the topography and stay at the targets' top");
        let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
        PyErr::warn(py, &category, &std::ffi::CString::new(message)?, 1)?;
    }
    let intervals = RecordBatch::try_from_iter([
        (
            "HOLE_ID",
            Arc::new(StringArray::from_iter_values(holes.iter().map(|h| &h.0))) as ArrayRef,
        ),
        ("FROM", Arc::new(Float64Array::from(vec![0.0; holes.len()]))),
        (
            "TO",
            Arc::new(Float64Array::from_iter_values(
                holes.iter().map(|h| h.1[1].measured_depth),
            )),
        ),
    ])
    .map_err(invalid)?;
    Ok(Drillholes {
        paths: holes.into_iter().collect(),
        hole: "HOLE_ID".into(),
        intervals: Some((intervals, "HOLE_ID".into(), "FROM".into(), "TO".into())),
        length_unit: crate::table::length_unit(None)?,
    })
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(planned_drillholes, m)?)?;
    m.add_function(wrap_pyfunction!(snap_to_surface, m)?)?;
    m.add_class::<Drillholes>()?;
    m.add_function(wrap_pyfunction!(merge_intervals, m)?)?;
    m.add_function(wrap_pyfunction!(check_drillholes, m)?)?;
    m.add_function(wrap_pyfunction!(fix_drillholes, m)?)?;
    Ok(())
}
