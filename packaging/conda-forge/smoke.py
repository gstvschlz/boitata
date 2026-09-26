import ceres as cs
import numpy as np

rng = np.random.default_rng(0)
coords = rng.uniform(0, 100, (20, 2))
values = rng.lognormal(size=20)

scores = cs.NormalScore().fit_transform(values)
assert abs(scores.mean()) < 0.5

model = cs.Variogram([("spherical", 1.0, 40.0)])
est = cs.OrdinaryKriging(model, cs.Search(radius=100, max_samples=8)).fit(coords, values).predict(coords)
np.testing.assert_allclose(est, values, atol=1e-8)
