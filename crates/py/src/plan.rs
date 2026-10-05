use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch};
use estimation::plan::{
    Candidate, Condition, Constraints, METRICS, Metrics, Objective, Op, Plan, Scorer,
};
use estimation::{EstimError, Sample};
use numpy::PyArray1;
use pyo3::prelude::*;

use crate::args::{self, Point, array1};
use crate::estimation::Estimator;
use crate::invalid;
use crate::table::Table;

/// Engine behind `boitata.DrillholePlan`: every input already resolved to
/// arrays, candidates to their composites.
#[pyclass(module = "boitata", name = "_DrillholePlan")]
pub struct DrillholePlan {
    plan: Plan,
    objective: Option<Py<PyAny>>,
}

fn rows(
    (support, weights): (&[f64], &[f64]),
    blocks: &[u32],
    metrics: &[Metrics],
) -> PyResult<Table> {
    let at = |f: &dyn Fn(usize, &Metrics) -> f64| -> ArrayRef {
        let values = blocks.iter().zip(metrics).map(|(&b, m)| f(b as usize, m));
        Arc::new(values.collect::<Float64Array>())
    };
    let block = blocks.iter().map(|&b| i64::from(b)).collect::<Int64Array>();
    let mut columns: Vec<(&str, ArrayRef)> = vec![("block", Arc::new(block))];
    for (k, name) in METRICS.iter().enumerate() {
        columns.push((name, at(&|_, m| m.get(k))));
    }
    columns.push(("support_variance", at(&|b, _| support[b])));
    columns.push(("weight", at(&|b, _| weights[b])));
    Ok(Table(RecordBatch::try_from_iter(columns).map_err(invalid)?))
}

fn op(text: &str) -> PyResult<Op> {
    Ok(match text {
        "<" => Op::Less,
        "<=" => Op::LessEqual,
        ">" => Op::Greater,
        ">=" => Op::GreaterEqual,
        _ => {
            return Err(invalid(format!(
                "unknown operator {text:?}; use <, <=, > or >="
            )));
        }
    })
}

impl DrillholePlan {
    /// `run` with the custom objective as scorer, holding the GIL, or
    /// without the GIL for a built-in one.
    fn with<T: Send>(
        &mut self,
        py: Python,
        run: impl FnOnce(&mut Plan, Option<Scorer>) -> estimation::Result<T> + Send,
    ) -> PyResult<T> {
        let Some(objective) = &self.objective else {
            let plan = &mut self.plan;
            return py.detach(|| run(plan, None)).map_err(invalid);
        };
        let (support, weights) = (self.plan.support().to_vec(), self.plan.weights().to_vec());
        let mut failed: Option<PyErr> = None;
        let mut score = |blocks: &[u32], metrics: &[Metrics]| -> estimation::Result<Vec<f64>> {
            let call = || -> PyResult<Vec<f64>> {
                let table = rows((&support, &weights), blocks, metrics)?;
                let out = objective.bind(py).call1((table,))?;
                args::finite(&out, "objective scores")
            };
            call().map_err(|e| {
                failed = Some(e);
                EstimError::InvalidParameters("the objective failed".into())
            })
        };
        let result = run(&mut self.plan, Some(&mut score));
        match failed {
            Some(e) => Err(e),
            None => result.map_err(invalid),
        }
    }
}

#[pymethods]
impl DrillholePlan {
    #[new]
    #[pyo3(signature = (
        estimator, targets, data, data_holes, composites, owner, collars, costs, excluded,
        objective, rules, weights, n_holes, budget, min_spacing
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python,
        estimator: PyRef<Estimator>,
        targets: &Bound<PyAny>,
        data: &Bound<PyAny>,
        data_holes: Option<Vec<u32>>,
        composites: &Bound<PyAny>,
        owner: Vec<usize>,
        collars: &Bound<PyAny>,
        costs: Vec<f64>,
        excluded: Vec<bool>,
        objective: &Bound<PyAny>,
        rules: Option<Vec<Vec<(String, String, f64)>>>,
        weights: Option<&Bound<PyAny>>,
        n_holes: Option<usize>,
        budget: Option<f64>,
        min_spacing: f64,
    ) -> PyResult<Self> {
        let kriging = estimator.planning()?;
        let targets = crate::estimation::targets(targets)?;
        let locs = args::points(data)?;
        if let Some(h) = &data_holes {
            args::same_length(locs.len(), h.len(), "data_holes")?;
        }
        let data = locs
            .into_iter()
            .enumerate()
            .map(|(i, p)| match &data_holes {
                Some(h) => Sample::with_hole(p, 0.0, h[i]),
                None => Sample::new(p, 0.0),
            })
            .collect();
        let collars: Vec<Point> = args::points(collars)?;
        args::same_length(collars.len(), costs.len(), "costs")?;
        args::same_length(collars.len(), excluded.len(), "excluded")?;
        let mut candidates: Vec<Candidate> = collars
            .iter()
            .zip(costs)
            .zip(excluded)
            .map(|((&collar, cost), excluded)| Candidate {
                composites: vec![],
                collar,
                cost,
                excluded,
            })
            .collect();
        let composites = args::points(composites)?;
        args::same_length(composites.len(), owner.len(), "owner")?;
        for (p, c) in composites.into_iter().zip(owner) {
            candidates
                .get_mut(c)
                .ok_or_else(|| invalid("composite of an unknown candidate"))?
                .composites
                .push(p);
        }
        let (objective, custom) = match objective.extract::<String>() {
            Ok(name) => match (name.as_str(), rules) {
                ("variance", _) => (Objective::Variance, None),
                ("classification", Some(rules)) => {
                    let condition = |(name, o, threshold): (String, String, f64)| {
                        let metric = METRICS.iter().position(|m| *m == name).ok_or_else(|| {
                            invalid(format!(
                                "rules use {name:?}; use one of {}",
                                METRICS.join(", ")
                            ))
                        })?;
                        Ok(Condition {
                            metric,
                            op: op(&o)?,
                            threshold,
                        })
                    };
                    let rules = rules
                        .into_iter()
                        .map(|r| r.into_iter().map(condition).collect())
                        .collect::<PyResult<_>>()?;
                    (Objective::Classification(rules), None)
                }
                ("classification", None) => {
                    return Err(invalid("the classification objective needs rules"));
                }
                (other, _) => {
                    return Err(invalid(format!(
                        "objective must be \"classification\", \"variance\" or a callable, got {other:?}"
                    )));
                }
            },
            Err(_) if objective.is_callable() => {
                (Objective::Custom, Some(objective.clone().unbind()))
            }
            Err(_) => return Err(invalid("objective must be a name or a callable")),
        };
        let weights = args::optional_finite(weights, "weights")?;
        let constraints = Constraints {
            n_holes,
            budget,
            min_spacing,
        };
        let plan = py
            .detach(|| {
                Plan::new(
                    kriging,
                    data,
                    candidates,
                    targets,
                    weights,
                    objective,
                    constraints,
                )
            })
            .map_err(invalid)?;
        Ok(Self {
            plan,
            objective: custom,
        })
    }

    fn score(&mut self, py: Python, selected: Vec<usize>) -> PyResult<f64> {
        self.with(py, |plan, custom| plan.score(&selected, custom))
    }

    fn gains<'py>(
        &mut self,
        py: Python<'py>,
        selected: Vec<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let gains = self.with(py, |plan, custom| plan.gains(&selected, custom))?;
        Ok(array1(py, gains))
    }

    fn loss<'py>(
        &mut self,
        py: Python<'py>,
        selected: Vec<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let loss = self.with(py, |plan, custom| plan.loss(&selected, custom))?;
        Ok(array1(py, loss))
    }

    fn feasible<'py>(
        &self,
        py: Python<'py>,
        selected: Vec<usize>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let mask = self.plan.feasible(&selected).map_err(invalid)?;
        Ok(PyArray1::from_vec(py, mask))
    }

    fn neighbors<'py>(
        &self,
        py: Python<'py>,
        i: usize,
        radius: f64,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let near = self.plan.neighbors(i, radius).map_err(invalid)?;
        Ok(PyArray1::from_vec(
            py,
            near.into_iter().map(|c| c as i64).collect(),
        ))
    }

    fn metrics(&mut self, py: Python, selected: Vec<usize>) -> PyResult<Table> {
        let plan = &mut self.plan;
        let metrics = py.detach(|| plan.metrics(&selected)).map_err(invalid)?;
        let blocks: Vec<u32> = (0..metrics.len() as u32).collect();
        rows(
            (self.plan.support(), self.plan.weights()),
            &blocks,
            &metrics,
        )
    }
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<DrillholePlan>()
}
