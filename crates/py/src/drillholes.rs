use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow_schema::DataType;
use ceres_core::PointSet;
use drillholes::{
    Collar, CompositeParams, DesurveyMethod, SurveyStation, WellborePoint, composite_intervals,
    desurvey_wellbore, position_at,
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

/// Collar, survey and interval tables desurveyed by minimum curvature.
/// Survey angles are `azimuth` (clockwise from north) and either `dip`
/// (degrees below horizontal) or `inclination` (degrees from vertical); holes
/// without survey are vertical.
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
        azimuth="AZIMUTH", dip=Some("DIP"), inclination=None, from_="FROM", to="TO"
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
    ) -> PyResult<Self> {
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
                    desurvey_wellbore(c, &s, DesurveyMethod::MinimumCurvature)
                        .map(|p| (id.clone(), p))
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

    /// Length-weighted composites of `grades` at `length` metres, never
    /// crossing a change of `domain`; located at their midpoints. Composites
    /// with none of the grades sampled are dropped.
    #[pyo3(signature = (length, grades, domain=None))]
    fn composite(
        &self,
        py: Python,
        length: f64,
        grades: Vec<String>,
        domain: Option<&str>,
    ) -> PyResult<PyPointSet> {
        let (batch, hole, from, to) = self.intervals()?;
        let ids = text(batch, hole)?;
        let (from, to) = (number(batch, from)?, number(batch, to)?);
        let domains = match domain {
            Some(d) => text(batch, d)?,
            None => vec![Some(String::new()); batch.num_rows()],
        };
        let columns = grades
            .iter()
            .map(|g| number(batch, g))
            .collect::<PyResult<Vec<_>>>()?;
        let mut per_hole: BTreeMap<&str, Vec<(f64, f64, String, HashMap<String, f64>)>> =
            BTreeMap::new();
        for i in 0..batch.num_rows() {
            let (Some(id), Some(f), Some(t)) = (&ids[i], from[i], to[i]) else {
                continue;
            };
            let values = grades
                .iter()
                .zip(&columns)
                .filter_map(|(g, c)| c[i].map(|v| (g.clone(), v)))
                .collect();
            let dom = domains[i].clone().unwrap_or_default();
            per_hole.entry(id).or_default().push((f, t, dom, values));
        }
        let params = CompositeParams {
            composite_length: length,
            domain_column: domain.unwrap_or_default().to_string(),
            grade_columns: grades.clone(),
        };
        let composites = py.detach(|| {
            per_hole
                .par_iter()
                .filter(|(id, _)| self.paths.contains_key(**id))
                .map(|(id, rows)| composite_intervals(id, rows, &params))
                .collect::<Result<Vec<_>, _>>()
        });
        let composites: Vec<_> = composites
            .map_err(invalid)?
            .into_iter()
            .flatten()
            .filter(|c| !c.attributes.is_empty())
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
        for (f, t, a, b) in drillholes::merge_intervals(&lspans, &rspans) {
            ids.push(id.clone());
            from.push(f);
            upto.push(t);
            li.push(a.map(|k| lrows[k] as u64));
            ri.push(b.map(|k| rrows[k] as u64));
        }
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
