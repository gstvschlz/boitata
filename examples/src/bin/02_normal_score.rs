use ceres_examples::{load, variable, write};
use transforms::{nscore_transform, optimal_cell_size};

fn main() {
    let (locs, v) = variable(&load("walker_sample.csv"), "V");
    let sizes: Vec<f64> = (1..=40).map(|i| 2.5 * i as f64).collect();
    let (cell, _) = optimal_cell_size(&locs, &v, &sizes, 25, false).unwrap();
    let weights = transforms::cell_weights(&locs, &v, cell, (0.0, 0.0, 0.0))
        .unwrap()
        .weights;

    let ns = nscore_transform(&v, Some(&weights)).unwrap();
    let back: Vec<f64> = ns.scores.iter().map(|&y| ns.table.back(y)).collect();
    let max_error = v
        .iter()
        .zip(&back)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    write(
        "02_scores.csv",
        &[("v", &v), ("w", &weights), ("score", &ns.scores)],
    );
    write(
        "02_table.csv",
        &[("value", &ns.table.values), ("score", &ns.table.scores)],
    );
    let (m, s) = weighted_moments(&ns.scores, &weights);
    println!("scores: weighted mean {m:.3}, sd {s:.3}; back-transform max error {max_error:.2e}");
}

fn weighted_moments(x: &[f64], w: &[f64]) -> (f64, f64) {
    let total: f64 = w.iter().sum();
    let mean = x.iter().zip(w).map(|(x, w)| x * w).sum::<f64>() / total;
    let var = x
        .iter()
        .zip(w)
        .map(|(x, w)| w * (x - mean).powi(2))
        .sum::<f64>()
        / total;
    (mean, var.sqrt())
}
