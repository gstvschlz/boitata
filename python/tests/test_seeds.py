import os
import pathlib
import subprocess
import sys

import ceres as cs
import numpy as np
import pytest

rng = np.random.default_rng(5)
coords = rng.uniform(0, 100, (40, 2))
values = rng.lognormal(0, 0.6, 40)
facies = (values > np.median(values)).astype(int)
paired = np.c_[values, values**0.5 + rng.uniform(0, 0.1, 40)]
v = cs.Variogram([("spherical", 1.0, 30.0)], nugget=0.1)
search = cs.Search(radius=40, max_samples=8)
grid = cs.BlockModel(origin=(0, 0), size=(10, 10), count=(10, 10))
ti = cs.object_training_image(
    cs.BlockModel((0, 0), (1, 1), (40, 40)),
    [{"shape": "ellipsoid", "code": 1, "proportion": 0.3, "radii": (6.0, 3.0)}],
)

SIMULATORS = {
    "sgs": lambda: cs.SGS(v, search).fit(coords, values),
    "turning_bands": lambda: cs.TurningBands(v, bands=50).fit(coords, values),
    "sis": lambda: cs.SIS([v, v], search).fit(coords, facies),
    "plurigaussian": lambda: cs.Plurigaussian(v, proportions=[0.5, 0.5]).fit(coords, facies),
    "multivariate": lambda: cs.MultivariateSimulation(
        cs.PCA(), [cs.SGS(v, search), cs.TurningBands(v, bands=50)]
    ).fit(coords, paired),
    "snesim": lambda: cs.SNESIM(ti, "facies", template_size=12, n_levels=1).fit(coords, facies),
}


def realizations(name, n, seed):
    if name == "bootstrap":
        return cs.spatial_bootstrap(coords, values, v, n=n, seed=seed)["mean"][:, None]
    s = SIMULATORS[name]().simulate(grid, n=n, seed=seed, keep=True)
    if isinstance(s, list):
        return np.hstack([x.realizations for x in s])
    return s.realizations


@pytest.mark.parametrize("name", [*SIMULATORS, "bootstrap"])
def test_realization_seeds(name):
    a = realizations(name, 5, 1)
    np.testing.assert_array_equal(a, realizations(name, 5, 1))
    np.testing.assert_array_equal(a[3], realizations(name, 10, 1)[3])
    b = realizations(name, 5, 2)
    for i in range(5):
        for j in range(5):
            assert not np.array_equal(a[i], b[j]), (i, j)


def test_realization_seeds_ignore_the_thread_count(tmp_path):
    here = {name: realizations(name, 4, 3) for name in [*SIMULATORS, "bootstrap"]}
    script = (
        "import sys, numpy as np; sys.path.insert(0, sys.argv[1]); import test_seeds as t; "
        "np.savez(sys.argv[2], **{k: t.realizations(k, 4, 3) for k in [*t.SIMULATORS, 'bootstrap']})"
    )
    out = tmp_path / "one.npz"
    env = {**os.environ, "RAYON_NUM_THREADS": "1"}
    subprocess.run(
        [sys.executable, "-c", script, str(pathlib.Path(__file__).parent), str(out)], env=env, check=True
    )
    with np.load(out) as one:
        for name, a in here.items():
            np.testing.assert_array_equal(a, one[name], err_msg=name)
