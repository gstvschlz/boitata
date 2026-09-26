use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::DataType;
use ceres_core::PointSet;
use drillholes::{
    Collar, CompositeParams, DesurveyMethod, DrillholeError, Residual, SurveyStation,
    WellborePoint, checks, composite_intervals, desurvey_wellbore, position_at,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::containers::PyPointSet;
use crate::invalid;
use crate::table::{Table, to_batch};

fn text(batch: &RecordBatch, name: &str) -> PyResult<Vec<Option<String>>> {
    let column = batch
        .column_by_name(name)
        .ok_or_else(|| invalid(format!("column {name} not found")))?;
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
#[pyclass(module = "ceres", name = "Drillholes", frozen)]
pub struct Drillholes {
    paths: BTreeMap<String, Vec<WellborePoint>>,
    intervals: Option<(RecordBatch, String, String, String)>,
}

#[pymethods]
impl Drillholes {
    #[new]
    #[pyo3(signature = (
        collar, survey, intervals=None, hole="HOLEID", x="X", y="Y", z="Z", at="DEPTH",
        azimuth="AZIMUTH", dip=Some("DIP"), inclination=None, from_="FROM", to="TO",
        method="minimum_curvature"
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
            intervals,
        })
    }

    #[getter]
    fn holes(&self) -> Vec<String> {
        self.paths.keys().cloned().collect()
    }

    /// Desurveyed stations: hole, depth, x, y, z.
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
                "hole",
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
        Ok(PyPointSet(
            PointSet::new(coords, attributes).map_err(invalid)?,
        ))
    }

    /// Length-weighted composites, never crossing a change of `domain` and
    /// located at their midpoints. Composites with none of the grades sampled
    /// are dropped. Each grade comes with its sampled length,
    /// `<grade>_length`, so the sum of grade × `<grade>_length` equals the
    /// input metal; `length` also counts unsampled ground.
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
        length, grades, domain=None, intervals=None, residual="keep", min_fraction=0.5,
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
                "hole".into(),
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
        Ok(PyPointSet(
            PointSet::new(coords, attributes).map_err(invalid)?,
        ))
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
#[pyo3(signature = (left, right, hole="HOLEID", from_="FROM", to="TO"))]
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

fn flags_table(flags: Named, sentinels: &[f64]) -> PyResult<RecordBatch> {
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
    let schema = batch
        .schema()
        .as_ref()
        .clone()
        .with_metadata([("sentinels".to_string(), listed)].into());
    batch.with_schema(Arc::new(schema)).map_err(invalid)
}

/// Per-record flags of drillhole tables, and a summary of every check.
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
///     - ``sentinel``: a numeric column holds one of `nodata`.
///     - ``no_survey``, ``no_<name>``: collar without survey or intervals.
///     - ``no_collar``: survey or interval of a hole without collar.
///     - ``deviation``: survey station more than `max_deviation` from the
///       station above.
///     - ``inverted``: interval with from ≥ to.
///     - ``gap``: interval starting below the end of the one above.
///     - ``overlap``: interval starting above the end of the ones kept above
///       it; dropping these keeps the first of overlapping intervals.
///     - ``past_depth``: survey or interval deeper than the hole length.
/// summary : Table
///     ``table``, ``check``, ``rows`` flagged and distinct ``holes``, for
///     every check including those that found nothing.
#[pyfunction]
#[pyo3(signature = (
    collar, survey=None, intervals=None, hole="HOLEID", x="X", y="Y", z="Z", at="DEPTH",
    azimuth="AZIMUTH", dip=Some("DIP"), inclination=None, from_="FROM", to="TO",
    max_depth=None, nodata=vec![-99.0, -999.0, -9999.0, 1e21], max_deviation=20.0,
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
    nodata: Vec<f64>,
    max_deviation: f64,
    tolerance: f64,
) -> PyResult<(Bound<'py, PyDict>, Table)> {
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
    let mut collar_flags = with_sentinels(named(flags), &collar, s)?;

    let mut others: Vec<(String, Named, Vec<Option<String>>)> = vec![];
    if let Some(survey) = survey {
        let batch = to_batch(survey)?;
        let ids = text(&batch, hole)?;
        let angle = match (inclination, dip) {
            (Some(name), _) => number(&batch, name)?
                .into_iter()
                .map(|v| {
                    v.map(|v| {
                        if checks::is_sentinel(v, s) {
                            v
                        } else {
                            90.0 - v
                        }
                    })
                })
                .collect(),
            (None, Some(name)) => number(&batch, name)?,
            (None, None) => return Err(invalid("give the dip or inclination column")),
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
        others.push((
            "survey".into(),
            with_sentinels(named(flags), &batch, s)?,
            ids,
        ));
    }
    let mut tables = vec![];
    if let Some(t) = intervals {
        match t.cast::<PyDict>() {
            Ok(d) if d.values().iter().all(|v| is_table(&v).unwrap_or(false)) => {
                for (k, v) in d.iter() {
                    tables.push((k.extract::<String>()?, to_batch(&v)?));
                }
            }
            _ => tables.push(("intervals".to_string(), to_batch(t)?)),
        }
    }
    for (name, batch) in tables {
        let ids = text(&batch, hole)?;
        let flags = checks::check_intervals(
            &ids,
            &number(&batch, from_)?,
            &number(&batch, to)?,
            hole_lengths.as_ref(),
            tolerance,
            s,
        );
        others.push((name, with_sentinels(named(flags), &batch, s)?, ids));
    }
    for (name, flags, ids) in &mut others {
        flags.push(("no_collar".into(), checks::absent(ids, &collar_ids)));
        collar_flags.push((format!("no_{name}"), checks::absent(&collar_ids, ids)));
    }
    others.insert(0, ("collar".into(), collar_flags, collar_ids));

    let (mut t, mut c, mut rows, mut holes) = (vec![], vec![], vec![], vec![]);
    let out = PyDict::new(py);
    for (name, flags, ids) in others {
        for (check, f) in &flags {
            let hit: HashSet<&str> = f
                .iter()
                .zip(&ids)
                .filter(|(f, _)| **f)
                .filter_map(|(_, id)| id.as_deref())
                .collect();
            t.push(name.clone());
            c.push(check.clone());
            rows.push(f.iter().filter(|f| **f).count() as i64);
            holes.push(hit.len() as i64);
        }
        out.set_item(&name, Table(flags_table(flags, s)?))?;
    }
    let summary = RecordBatch::try_from_iter([
        ("table", Arc::new(StringArray::from(t)) as ArrayRef),
        ("check", Arc::new(StringArray::from(c))),
        ("rows", Arc::new(Int64Array::from(rows))),
        ("holes", Arc::new(Int64Array::from(holes))),
    ])
    .map_err(invalid)?;
    Ok((out, Table(summary)))
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
    flags, tables, missing="drop", duplicates="drop", inverted="drop", out_of_range="drop",
    overlaps="keep_first", sentinels="null", deviation="drop", no_collar="drop", past_depth="keep"
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
) -> PyResult<(Bound<'py, PyDict>, Table)> {
    let drop_keep: &[&str] = &["drop", "keep"];
    let rules = [
        ("missing", "missing", missing, drop_keep),
        ("duplicate", "duplicates", duplicates, drop_keep),
        ("inverted", "inverted", inverted, drop_keep),
        ("out_of_range", "out_of_range", out_of_range, drop_keep),
        ("overlap", "overlaps", overlaps, &["keep_first", "keep"]),
        (
            "sentinel",
            "sentinels",
            sentinels,
            &["null", "drop", "keep"],
        ),
        ("deviation", "deviation", deviation, drop_keep),
        ("no_collar", "no_collar", no_collar, drop_keep),
        ("past_depth", "past_depth", past_depth, drop_keep),
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
            if *rule == "null" {
                batch = null_sentinels(&batch, &f, &name)?;
            } else {
                drop.iter_mut().zip(&hit).for_each(|(d, h)| *d |= h);
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
    let columns = batch
        .columns()
        .iter()
        .map(|col| {
            if !col.data_type().is_numeric() {
                return Ok(col.clone());
            }
            let as_f64 = arrow_cast::cast(col, &DataType::Float64).map_err(invalid)?;
            let mask: BooleanArray = as_f64
                .as_primitive::<Float64Type>()
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

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Drillholes>()?;
    m.add_function(wrap_pyfunction!(merge_intervals, m)?)?;
    m.add_function(wrap_pyfunction!(check_drillholes, m)?)?;
    m.add_function(wrap_pyfunction!(fix_drillholes, m)?)?;
    Ok(())
}
