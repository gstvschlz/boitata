# 2. Normal-score transform

Gaussian methods need a standard normal variable.
The normal-score transform maps each value to the Gaussian score with the same cumulative probability, using the declustering weights from [chapter 1](../01-data/README.md).

![quantile mapping](quantile-mapping.png)

![histograms](histograms.png)

Checks: weighted mean of the scores 0.000, standard deviation 0.999; the back-transform returns every sample exactly.

```python
ns = cs.NormalScore()
y = ns.fit_transform(v, weights=d.weights)
v_back = ns.inverse_transform(y)
```

[`example.py`](example.py)
