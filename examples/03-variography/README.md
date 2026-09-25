# 3. Variography

The variogram map shows γ in every horizontal direction; low values stretch along the direction of greatest continuity.
The direction whose fitted range is longest is the major axis, N170°.

![variogram](variogram.png)

Model: nugget 32 975 ppm², spherical sill 60 404 ppm², ranges 75 m (N170°) and 24 m (N260°).
The major direction sets nugget, sill and major range; the minor direction contributes only its range.
The ellipse on the map is the model range in each direction.

```python
vmap = cs.variogram_map(xy, v, lag=10, max_lag=120)
major = cs.experimental_variogram(xy, v, lag=10, max_lag=120, azimuth=170)
fitted = major.fit("spherical")
model = cs.Variogram([("spherical", 60404, 75.4)], nugget=32975, rotation=(170, 0, 0), ratios=(0.32, 1))
model.to_json()  # saved as model.json, reused in chapter 4
```

`rotation` is azimuth, dip, rake in degrees; `ratios` are semi-major/major and minor/major ranges.

[`example.py`](example.py)
