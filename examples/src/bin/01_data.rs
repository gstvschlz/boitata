use ceres_examples::{load, mean, variable, write};
use transforms::{cell_weights, decluster_mean_over_offsets, optimal_cell_size};

fn main() {
    let sample = load("walker_sample.csv");
    let (locs, v) = variable(&sample, "V");
    let (_, truth) = variable(&load("walker_exhaustive.csv"), "V");

    let sizes: Vec<f64> = (1..=40).map(|i| 2.5 * i as f64).collect();
    let means: Vec<f64> = sizes
        .iter()
        .map(|&s| decluster_mean_over_offsets(&locs, &v, s, 25).unwrap())
        .collect();
    let (best, declustered) = optimal_cell_size(&locs, &v, &sizes, 25, false).unwrap();
    let weights = cell_weights(&locs, &v, best, (0.0, 0.0, 0.0))
        .unwrap()
        .weights;

    let x: Vec<f64> = locs.iter().map(|p| p.0).collect();
    let y: Vec<f64> = locs.iter().map(|p| p.1).collect();
    write(
        "01_samples.csv",
        &[("x", &x), ("y", &y), ("v", &v), ("w", &weights)],
    );
    write("01_declustering.csv", &[("cell", &sizes), ("mean", &means)]);
    write(
        "01_summary.csv",
        &[
            ("naive", &[mean(&v)]),
            ("declustered", &[declustered]),
            ("truth", &[mean(&truth)]),
            ("cell", &[best]),
        ],
    );
    println!(
        "naive {:.1}  declustered {:.1} (cell {best})  true {:.1}",
        mean(&v),
        declustered,
        mean(&truth)
    );
}
