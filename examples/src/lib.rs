//! Shared loading and output helpers for the example binaries.

use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{ArrayRef, Float64Array, RecordBatch};
use ceres_core::PointSet;
use ceres_io::{CsvOptions, read_csv, write_csv};
use variogram::surface::{PlaneMap, PlaneMapParams, plane_map};
use variogram::{
    Angles, Anisotropy, Direction, Estimator, Experimental, LagBins, Model, Structure, Variogram,
    Weighting, experimental, fit,
};

pub type Point = (f64, f64, f64);

fn dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(name)
}

/// Loads `examples/data/<file>` with `X`, `Y` as coordinates.
pub fn load(file: &str) -> PointSet {
    let table = read_csv(dir("data").join(file), &CsvOptions::default())
        .unwrap_or_else(|e| panic!("{file}: {e}; run `mise run examples:data` first"));
    PointSet::from_table(&table, "X", "Y", None).expect("X and Y columns")
}

pub fn locations(points: &PointSet) -> Vec<Point> {
    points.coords().iter().map(|c| (c[0], c[1], c[2])).collect()
}

/// Non-null values of `name`, with the matching locations.
pub fn variable(points: &PointSet, name: &str) -> (Vec<Point>, Vec<f64>) {
    let column = points
        .attributes()
        .column_by_name(name)
        .unwrap_or_else(|| panic!("no column {name}"))
        .as_primitive::<Float64Type>();
    locations(points)
        .into_iter()
        .zip(column.iter())
        .filter_map(|(p, v)| v.map(|v| (p, v)))
        .unzip()
}

/// Writes equally long columns to `examples/out/<file>`.
pub fn write(file: &str, columns: &[(&str, &[f64])]) {
    let arrays = columns.iter().map(|(name, values)| {
        (
            *name,
            Arc::new(Float64Array::from(values.to_vec())) as ArrayRef,
        )
    });
    let table = RecordBatch::try_from_iter(arrays).expect("equal column lengths");
    std::fs::create_dir_all(dir("out")).expect("create examples/out");
    write_csv(dir("out").join(file), &table).expect("write output");
}

pub fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

pub struct Fitted {
    pub model: Variogram,
    pub azimuth: f64,
    pub major: Experimental,
    pub minor: Experimental,
    pub plane: PlaneMap,
}

/// Picks the major azimuth from a variogram map, fits a spherical model along
/// it and across it, and combines them: nugget, sill and major range from the
/// major direction, the range ratio from the minor one.
pub fn fit_anisotropic(locs: &[Point], values: &[f64], bins: &LagBins) -> Fitted {
    let params = PlaneMapParams {
        bins: bins.clone(),
        ..Default::default()
    };
    let plane = plane_map(locs, values, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), &params).unwrap();
    let (angle, _) = plane
        .angles
        .iter()
        .zip(&plane.ranges)
        .filter_map(|(a, r)| r.map(|r| (*a, r)))
        .fold(
            (0.0, f64::MIN),
            |best, c| if c.1 > best.1 { c } else { best },
        );
    let azimuth = (90.0 - angle.to_degrees()).rem_euclid(180.0);

    let directional = |azimuth: f64| {
        let direction = Direction {
            azimuth,
            dip: 0.0,
            tolerance: 22.5,
            bandwidth: None,
        };
        experimental(locs, values, bins, Estimator::Matheron, Some(&direction)).unwrap()
    };
    let (major, minor) = (directional(azimuth), directional(azimuth + 90.0));
    let along = fit(&major, Model::Spherical, Weighting::ByCount)
        .unwrap()
        .variogram;
    let across = fit(&minor, Model::Spherical, Weighting::ByCount)
        .unwrap()
        .variogram;
    let (range, ratio) = (
        along.structures[0].range,
        (across.structures[0].range / along.structures[0].range).min(1.0),
    );
    let angles = Angles {
        azimuth,
        dip: 0.0,
        pitch: 0.0,
        major: 1.0,
        semi: ratio,
        minor: 1.0,
    };
    let model = Variogram {
        nugget: along.nugget,
        structures: vec![Structure::new(
            Model::Spherical,
            along.structures[0].sill,
            range,
        )],
        anisotropy: Some(Anisotropy::new(angles).unwrap()),
    };
    Fitted {
        model,
        azimuth,
        major,
        minor,
        plane,
    }
}
