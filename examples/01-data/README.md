# 1. Data and declustering

Walker Lake: 470 samples of `V` (ppm) over a 260 × 300 m area whose exhaustive values are known.

![maps](maps.png)

Samples are denser where `V` is high, so the plain sample mean (435 ppm) overstates the true mean (278 ppm).
Cell declustering weights each sample by the inverse of the number of samples in its cell.
Scanning cell sizes, each averaged over 25 grid offsets, and keeping the lowest mean gives 22.5 m cells and 293 ppm.

![declustering](declustering.png)

![histograms](histograms.png)

```python
import ceres as cs

samples = cs.PointSet.from_table(cs.read_csv("walker-lake/sample.csv"))  # X, Y columns
d = cs.cell_declustering(samples.coords, samples["V"], sizes=np.arange(2.5, 102.5, 2.5))
d.mean, d.cell_size, d.weights
```

[`example.py`](example.py)
