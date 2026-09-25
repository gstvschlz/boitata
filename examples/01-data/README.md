# 1. Data and declustering

Walker Lake: 470 samples of `V` (ppm) over a 260 × 300 m area whose exhaustive values are known.

![maps](maps.png)

Samples are denser where `V` is high, so the plain sample mean (435 ppm) overstates the true mean (278 ppm).
Cell declustering weights each sample by the inverse of the number of samples sharing its cell.
Scanning cell sizes (each averaged over 25 grid offsets) and keeping the one with the lowest mean gives 22.5 m cells and a declustered mean of 291 ppm.

![declustering](declustering.png)

![histograms](histograms.png)

```rust
let sample = load("walker_sample.csv");                  // ceres_io::read_csv + PointSet::from_table
let (locs, v) = variable(&sample, "V");
let (cell, mean) = optimal_cell_size(&locs, &v, &sizes, 25, false)?;
let weights = cell_weights(&locs, &v, cell, (0.0, 0.0, 0.0))?.weights;
```

Source: [`01_data.rs`](../src/bin/01_data.rs), [`01_data.py`](../plot/01_data.py).
