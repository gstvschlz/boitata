# 3. Variography

The variogram map shows γ in every horizontal direction; low values stretch along the direction of greatest continuity.
Fitting a spherical model in each direction and keeping the longest range gives the major azimuth, N170°.

![variogram](variogram.png)

Model: nugget 32 975 ppm², spherical sill 60 404 ppm², ranges 75 m (N170°) and 24 m (N260°).
The major direction sets nugget, sill and major range; the minor direction contributes only its range.
The ellipse on the map is the fitted range in each direction.

```rust
let map = plane_map(&locs, &v, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), &params)?;
let exp = experimental(&locs, &v, &bins, Estimator::Matheron, Some(&direction))?;
let fitted = fit(&exp, Model::Spherical, Weighting::ByCount)?;
```

Anisotropy is expressed as range ratios (`major = 1`, `semi = 24/75`) so the structure range and search radius stay in metres along the major axis.

Source: [`03_variography.rs`](../src/bin/03_variography.rs), [`lib.rs`](../src/lib.rs), [`03_variography.py`](../plot/03_variography.py).
