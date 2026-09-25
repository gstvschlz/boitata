# 9. Change of support and disjunctive kriging

Mining selects blocks, not points. Block grades vary less than point grades, so a point histogram overstates the tonnage of rich material at high cutoffs and understates it at low ones.

The Hermite anamorphosis models the declustered point distribution; the discrete Gaussian model shrinks it to 10 × 10 m blocks with a change-of-support coefficient r = 0.75, computed from the Gaussian variogram.
The exhaustive grid gives true point and block curves to check against.

![grade-tonnage](grade-tonnage.png)

The block model follows the true 10 × 10 m curves; its variance (35 800 ppm²) is below the true block variance (46 700 ppm²) because the fitted nugget reduces more than the real short-scale variability does.

Disjunctive kriging estimates, at each node, the probability of exceeding a cutoff from the kriged Hermite factors.
Binned against the truth it is roughly calibrated: low probabilities are slightly high and mid-range ones too cautious, so the map ranks nodes well but hedges its middle values.

![disjunctive](disjunctive.png)

```python
anam = cs.HermiteAnamorphosis(degree=40).fit(v, weights=weights)
r, block = cs.change_of_support(anam, gaussian, size=(10, 10), discretization=(5, 5, 1))
curves = block.grade_tonnage(np.linspace(0, 1000, 41))  # tonnage, metal, mean_grade, benefit

dk = cs.DisjunctiveKriging(anam, gaussian, cs.Search(radius=100, max_samples=24)).fit(xy, v)
p = dk.predict_tonnage(grid, 500.0)
```

[`example.py`](example.py)
