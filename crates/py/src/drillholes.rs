use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow_schema::DataType;
use ceres_core::PointSet;
use drillholes::{
    Collar, CompositeParams, DesurveyMethod, DrillholeError, Residual, SurveyStation,
    WellborePoint, composite_intervals, desurvey_wellbore, position_at,
};
use pyo3::prelude::*;
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
    ///     Composite length in metres; None gives one composite per run of
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

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<Drillholes>()?;
    m.add_function(wrap_pyfunction!(merge_intervals, m)?)?;
    Ok(())
}
