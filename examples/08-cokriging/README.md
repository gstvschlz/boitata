# 8. Cokriging and indicator kriging

Jura: 259 soil samples of heavy metals (mg/kg, coordinates in km) and 100 validation samples withheld from estimation.
Cd correlates with Zn (r = 0.67), and Zn is also known at the validation points, which suits collocated cokriging.

The coregionalization uses the intrinsic model: both variables share Cd's spherical structure (range 0.59 km, 44 % nugget), scaled by their covariance matrix.

![validation](validation.png)

Using Zn at the target lowers the validation RMSE from 0.78 to 0.69 mg/kg and removes the smoothing that flattens ordinary kriging.

Indicator kriging estimates the probability that Cd exceeds 0.8 mg/kg, the Swiss guide value, from the indicator variogram.
Validation points above the limit average a probability of 0.75, those below 0.62: most of the area exceeds, so the map separates clean zones rather than hot spots.

![probability](probability.png)

```python
lmc = cs.Coregionalization((0.44 * cov).tolist(), [("spherical", 0.59, (0.56 * cov).tolist())])
ck = cs.Cokriging(lmc, search, means=[cd.mean(), zn.mean()])
ck.fit(np.vstack([xy, xy]), np.r_[cd, zn], [0] * n + [1] * n)
estimate = ck.predict(targets, collocated={1: zn_at_targets})

ik = cs.IndicatorKriging(indicator_model, search, threshold=0.8).fit(xy, cd)
p_exceed = 1 - ik.predict(targets)  # IK estimates P(Cd <= threshold)
```

[`example.py`](example.py)
