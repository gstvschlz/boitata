import json
import pickle

import ceres as cs
import numpy as np
import pytest

rng = np.random.default_rng(5)
coords = rng.uniform(0, 100, (60, 2))
values = rng.lognormal(0, 0.6, 60)
gaussian = cs.Variogram([("spherical", 1.0, 30.0)])
grid = cs.BlockModel(origin=(0, 0), size=(5, 5), count=(20, 20))


def test_sgs_is_reproducible_and_honours_data():
    sgs = cs.SGS(gaussian, cs.Search(radius=40, max_samples=12)).fit(coords, values)
    a = sgs.simulate(grid, n=3, seed=7, realizations=True).realizations
    assert a.shape == (3, 400)
    np.testing.assert_array_equal(a, sgs.simulate(grid, n=3, seed=7, realizations=True).realizations)
    assert not np.array_equal(a[0], a[1])
    at_data = sgs.simulate(coords[:5], n=4, seed=1)
    np.testing.assert_allclose(at_data.mean, values[:5])
    np.testing.assert_allclose(at_data.variance, 0, atol=1e-12)


def test_sgs_summary_matches_its_realizations():
    sgs = cs.SGS(gaussian, cs.Search(radius=40, max_samples=12)).fit(coords, values)
    cut = float(np.median(values))
    s = sgs.simulate(grid, n=20, seed=3, cutoffs=[cut], quantiles=[0.1, 0.5, 0.9], realizations=True)
    reals = s.realizations
    np.testing.assert_allclose(s.mean, reals.mean(axis=0))
    np.testing.assert_allclose(s.variance, reals.var(axis=0), atol=1e-9)
    np.testing.assert_allclose(s.probability_above[0], (reals > cut).mean(axis=0))
    np.testing.assert_allclose(s.quantile_values[1], np.median(reals, axis=0), rtol=1e-6)
    np.testing.assert_allclose(s.realization_above[0], (reals > cut).mean(axis=1))
    assert np.median(reals) == pytest.approx(np.median(values), rel=0.25)
    assert sgs.simulate(grid, n=2, seed=3).realizations is None


def test_block_support_averages_each_realization():
    sgs = cs.SGS(gaussian, cs.Search(radius=40, max_samples=12)).fit(coords, values)
    blocks = cs.BlockModel(origin=(0, 0), size=(20, 20), count=(5, 5))
    nodes = sgs.simulate(grid, n=6, seed=2, realizations=True).realizations
    s = sgs.simulate(grid, n=6, seed=2, cutoffs=[1.0], realizations=True, blocks=blocks)
    xy = grid.centroids[:, :2] // 20
    rows = (xy[:, 0] + 5 * xy[:, 1]).astype(int)
    expected = np.stack([np.bincount(rows, r) / np.bincount(rows) for r in nodes])
    np.testing.assert_allclose(s.realizations, expected)
    np.testing.assert_allclose(s.mean, expected.mean(axis=0))
    assert (s.variance <= np.bincount(rows, nodes.var(axis=0)) / 16 + 1e-9).all()
    with pytest.raises(ValueError):
        sgs.simulate(grid, n=1, blocks=cs.BlockModel(origin=(0, 0), size=(20, 20), count=(6, 5)))

    cats = (values > np.median(values)).astype(int)
    sis = cs.SIS([gaussian, gaussian], cs.Search(radius=40, max_samples=12)).fit(coords, cats)
    c = sis.simulate(grid, n=3, seed=4, realizations=True, blocks=blocks)
    assert c.realizations.shape == (3, 25) and c.probabilities.shape == (2, 25)


def test_localize_realizations_within_panels():
    sgs = cs.SGS(gaussian, cs.Search(radius=40, max_samples=12)).fit(coords, values)
    panels = cs.BlockModel(origin=(0, 0), size=(50, 50), count=(2, 2))
    smus = panels.discretize(5)
    s = sgs.simulate(grid, n=8, seed=2, realizations=True, blocks=smus)
    smus = smus.with_column("etype", s.mean)
    out = cs.localize(smus, "etype", s.realizations, panels)
    owner, local = smus["block"].astype(int), out["localized"]
    for p in range(4):
        mine = owner == p
        pooled = np.sort(s.realizations[:, mine].ravel())
        assert local[mine].mean() == pytest.approx(pooled.mean(), rel=1e-9)
        by_rank = local[mine][np.argsort(s.mean[mine], kind="stable")]
        np.testing.assert_allclose(by_rank, pooled.reshape(25, 8).mean(axis=1), rtol=1e-12)

    one = cs.localize(smus, "etype", s.realizations[:1], panels, name="one")["one"]
    for p in range(4):
        mine = owner == p
        by_rank = one[mine][np.argsort(s.mean[mine], kind="stable")]
        np.testing.assert_array_equal(by_rank, np.sort(s.realizations[0, mine]))
    with pytest.raises(cs.InvalidInput):
        cs.localize(smus, "etype", s.realizations[:, 1:], panels)
    with pytest.raises(KeyError):
        cs.localize(smus, "missing", s.realizations, panels)


def test_turning_bands_summary():
    tb = cs.TurningBands(gaussian, bands=100, step=1.0).fit(coords, values)
    s = tb.simulate(grid, n=2, seed=1, cutoffs=[1.0, 2.0])
    assert s.mean.shape == (400,) and s.probability_above.shape == (2, 400)


def test_sis_probabilities():
    cats = (values > np.median(values)).astype(int)
    sis = cs.SIS([gaussian, gaussian], cs.Search(radius=40, max_samples=12)).fit(coords, cats)
    s = sis.simulate(grid, n=4, seed=4, realizations=True)
    assert s.probabilities.shape == (2, 400) and s.proportions.shape == (4, 2)
    np.testing.assert_allclose(s.probabilities.sum(axis=0), 1.0)
    assert set(np.unique(s.realizations)) <= {0, 1}
    assert ((s.entropy >= 0) & (s.entropy <= 1)).all()


def test_plurigaussian_proportions():
    facies = rng.choice(3, 60, p=[0.2, 0.3, 0.5])
    pgs = cs.Plurigaussian(gaussian, proportions=[0.2, 0.3, 0.5]).fit(coords, facies)
    s = pgs.simulate(grid, n=2, seed=2)
    assert s.probabilities.shape == (3, 400) and set(np.unique(s.most_likely)) <= {0, 1, 2}


def test_gibbs_respects_bounds():
    pts = rng.uniform(0, 50, (20, 2))
    bounds = np.column_stack([np.zeros(20), np.full(20, np.inf)])
    draw = cs.gibbs(pts, bounds, gaussian, seed=3)
    assert np.all(draw >= 0)


def trended_samples():
    r = np.random.default_rng(8)
    xy = r.uniform(0, 100, (400, 2))
    trend = xy[:, 0] / 100
    return xy, np.exp(1.5 * trend + 0.5 * r.normal(size=400)), trend


def test_simulation_with_a_trend_follows_it_and_honours_data():
    xy, z, trend = trended_samples()
    search = cs.Search(radius=30, max_samples=12)
    node_trend = grid.centroids[:, 0] / 100
    white = cs.Variogram([("spherical", 1.0, 4.0)])
    for simulator in (cs.SGS(white, search, classes=5), cs.TurningBands(white, bands=100, classes=5)):
        simulator.fit(xy, z, trend=trend)
        reals = simulator.simulate(grid, n=4, seed=2, realizations=True, trend=node_trend).realizations
        want = np.corrcoef(np.log(z), trend)[0, 1]
        for r in reals:
            assert np.corrcoef(np.log(r), node_trend)[0, 1] == pytest.approx(want, abs=0.12)
        at_data = simulator.simulate(xy[:5], n=3, seed=1, trend=trend[:5])
        np.testing.assert_allclose(at_data.mean, z[:5])
        np.testing.assert_allclose(at_data.variance, 0, atol=1e-9)

    sgs = cs.SGS(gaussian, search).fit(xy, z, trend=trend)
    model = grid.with_column("drift", node_trend)
    by_array = sgs.simulate(grid, n=3, seed=4, realizations=True, trend=node_trend).realizations
    np.testing.assert_array_equal(
        sgs.simulate(model, n=3, seed=4, realizations=True, trend="drift").realizations, by_array
    )
    blocks = cs.BlockModel(origin=(0, 0), size=(20, 20), count=(5, 5))
    by_block = sgs.simulate(
        grid, n=3, seed=4, realizations=True, blocks=blocks, trend=node_trend
    ).realizations
    xy_block = grid.centroids[:, :2] // 20
    rows = (xy_block[:, 0] + 5 * xy_block[:, 1]).astype(int)
    np.testing.assert_allclose(by_block, [np.bincount(rows, r) / np.bincount(rows) for r in by_array])

    flat = cs.SGS(gaussian, search).fit(xy, z, trend=np.ones(len(z)))
    plain = cs.SGS(gaussian, search).fit(xy, z)
    np.testing.assert_allclose(
        flat.simulate(grid, n=2, seed=3, trend=np.ones(400)).mean, plain.simulate(grid, n=2, seed=3).mean
    )

    with pytest.raises(cs.InvalidInput, match="give trend"):
        sgs.simulate(grid, n=1)
    with pytest.raises(cs.InvalidInput, match="needs trend at fit"):
        plain.simulate(grid, n=1, trend=node_trend)
    with pytest.raises(cs.InvalidInput, match="BlockModel"):
        sgs.simulate(grid.centroids, n=1, trend="drift")
    with pytest.raises(ValueError):
        sgs.simulate(grid, n=1, trend=node_trend[1:])
    with pytest.raises(ValueError):
        cs.SGS(gaussian, search).fit(xy, z, trend=trend[1:])
    with pytest.raises(cs.InvalidInput, match="does not take a trend"):
        cs.TurningBands(gaussian).fit(xy, z, trend=trend).simulate_to_parquet("in.parquet", "out.parquet")


def test_simulators_with_a_trend_save_and_load_it(tmp_path):
    xy, z, trend = trended_samples()
    node_trend = grid.centroids[:, 0] / 100
    for simulator in (
        cs.SGS(gaussian, cs.Search(radius=30, max_samples=12), classes=4),
        cs.TurningBands(gaussian, bands=50, classes=4),
    ):
        simulator.fit(xy, z, trend=trend)
        want = simulator.simulate(grid, n=2, seed=6, realizations=True, trend=node_trend).realizations
        path = tmp_path / "simulator.parquet"
        simulator.to_parquet(path)
        for back in (type(simulator).from_parquet(path), pickle.loads(pickle.dumps(simulator))):
            got = back.simulate(grid, n=2, seed=6, realizations=True, trend=node_trend).realizations
            np.testing.assert_array_equal(got, want)


def test_multivariate_simulation_reproduces_correlation_and_honours_data():
    g = rng.standard_normal((60, 2))
    data = np.exp(np.column_stack([g[:, 0], 0.8 * g[:, 0] + 0.6 * g[:, 1]]))
    search = cs.Search(radius=40, max_samples=12)
    mv = cs.MultivariateSimulation(
        cs.PPMT(seed=3), [cs.SGS(gaussian, search), cs.TurningBands(gaussian, bands=100, step=1.0)]
    ).fit(coords, data)
    a, b = mv.simulate(grid, n=10, seed=4, realizations=True)
    assert a.realizations.shape == b.realizations.shape == (10, 400)
    r = np.mean([np.corrcoef(np.log(x), np.log(y))[0, 1] for x, y in zip(a.realizations, b.realizations)])
    assert r == pytest.approx(np.corrcoef(np.log(data.T))[0, 1], abs=0.15)
    np.testing.assert_array_equal(
        a.realizations, mv.simulate(grid, n=10, seed=4, realizations=True)[0].realizations
    )

    at_data = mv.simulate(coords[:5], n=3, seed=1)
    for v, s in enumerate(at_data):
        np.testing.assert_allclose(s.mean, data[:5, v], rtol=1e-6)

    blocks = cs.BlockModel(origin=(0, 0), size=(20, 20), count=(5, 5))
    xy = grid.centroids[:, :2] // 20
    rows = (xy[:, 0] + 5 * xy[:, 1]).astype(int)
    by_block = mv.simulate(grid, n=10, seed=4, realizations=True, blocks=blocks)
    for s, nodes in zip(by_block, (a, b)):
        expected = np.stack([np.bincount(rows, x) / np.bincount(rows) for x in nodes.realizations])
        np.testing.assert_allclose(s.realizations, expected)


def test_multivariate_simulation_drops_incomplete_samples_and_checks_inputs():
    data = np.column_stack([values, values**0.5])
    data[:4, 1] = np.nan
    sgs = cs.SGS(gaussian, cs.Search(radius=40, max_samples=12))
    with pytest.warns(UserWarning, match="4 samples miss a variable"):
        cs.MultivariateSimulation(cs.StepwiseConditional(), [sgs, sgs]).fit(coords, data)
    data[:, 1] *= rng.lognormal(0, 0.5, 60)
    data[4] = np.nan
    mv = cs.MultivariateSimulation(cs.PPMT(seed=3), [sgs, sgs])
    with pytest.warns(UserWarning, match="1 samples miss every variable"):
        mv.fit(coords, data, impute=True)
    at_data = mv.simulate(coords[:4], n=6, seed=2, realizations=True)
    np.testing.assert_allclose(at_data[0].realizations, np.tile(values[:4], (6, 1)), rtol=1e-6)
    imputed = at_data[1].realizations
    assert np.isfinite(imputed).all() and np.ptp(imputed, axis=0).min() > 0
    with pytest.raises(ValueError, match="transform must be"):
        cs.MultivariateSimulation(cs.NormalScore(), [sgs])
    with pytest.raises(ValueError, match="2 columns"):
        cs.MultivariateSimulation(cs.PCA(), [sgs, sgs]).fit(coords, data[:, :1])


@pytest.mark.parametrize("kind", ["SGS", "TurningBands", "SIS", "MultivariateSimulation"])
def test_max_per_hole_caps_the_data_of_one_hole(kind, tmp_path):
    down = np.column_stack([np.zeros(10), np.zeros(10), np.arange(10.0)])
    grades = np.arange(10.0) * 7 % 10 + 1
    data = {
        "SGS": grades,
        "TurningBands": grades,
        "SIS": np.arange(10) % 2,
        "MultivariateSimulation": np.column_stack([grades, grades**0.5 + np.arange(10) % 3]),
    }[kind]

    def simulator(max_samples, max_per_hole=None):
        search = cs.Search(50.0, max_samples=max_samples, max_per_hole=max_per_hole)
        if kind == "SGS":
            return cs.SGS(gaussian, search)
        if kind == "TurningBands":
            return cs.TurningBands(gaussian, bands=50, step=1.0, search=search)
        if kind == "SIS":
            return cs.SIS([gaussian, gaussian], search)
        bands = cs.TurningBands(gaussian, bands=50, step=1.0, search=search)
        return cs.MultivariateSimulation(cs.PCA(), [cs.SGS(gaussian, search), bands])

    def run(model):
        out = model.simulate([[3.0, 0.0, 4.4]], n=3, seed=2, realizations=True)
        return [s.realizations for s in out] if isinstance(out, list) else [out.realizations]

    holes = ["DH1"] * 10
    capped = simulator(8, 1).fit(down, data, holes=holes)
    np.testing.assert_array_equal(run(capped), run(simulator(1).fit(down, data)))
    free = run(simulator(8).fit(down, data))
    np.testing.assert_array_equal(run(simulator(8).fit(down, data, holes=holes)), free)
    if kind != "SIS":
        assert not np.array_equal(run(capped), free)
    if kind != "MultivariateSimulation":
        capped.to_parquet(tmp_path / "capped.parquet")
        np.testing.assert_array_equal(
            run(type(capped).from_parquet(tmp_path / "capped.parquet")), run(capped)
        )
        meta, columns = capped._state()
        older = type(capped)._from_state(meta, [c for c in columns if c[0] != "hole"])
        np.testing.assert_array_equal(run(older), run(simulator(8, 1).fit(down, data)))


@pytest.mark.parametrize("kind", [cs.SGS, cs.TurningBands])
def test_max_per_hole_caps_the_data_of_one_hole_with_a_trend(kind):
    down = np.column_stack([np.zeros(10), np.zeros(10), np.arange(10.0)])
    grades, trend = np.arange(10.0) * 7 % 10 + 1, np.arange(10.0) / 10

    def run(max_samples, max_per_hole=None, holes=None):
        search = cs.Search(50.0, max_samples=max_samples, max_per_hole=max_per_hole)
        options = {"bands": 50, "step": 1.0} if kind is cs.TurningBands else {}
        model = kind(gaussian, search=search, classes=2, **options).fit(
            down, grades, holes=holes, trend=trend
        )
        return model.simulate([[3.0, 0.0, 4.4]], n=3, seed=2, realizations=True, trend=[0.45]).realizations

    capped = run(8, 1, ["DH1"] * 10)
    np.testing.assert_array_equal(capped, run(1))
    assert not np.array_equal(capped, run(8))


def holes_in_clusters():
    r = np.random.default_rng(8)
    tops = np.vstack([r.uniform(0, 25, (20, 2)), r.uniform(50, 100, (10, 2))])
    xyz = np.column_stack([np.repeat(tops, 5, axis=0), np.tile(np.arange(5.0), 30)])
    grades = r.lognormal(0, 0.8, 150)
    passes = [
        cs.Search(r, max_samples=12, min_samples=m, max_per_hole=3, high_grade=(4.0, 6.0))
        for r, m in ((10, 8), (25, 4), (40, 2))
    ]
    return xyz, grades, np.repeat(np.arange(30), 5), passes


def test_sgs_passes_are_the_kriging_passes():
    xyz, grades, holes, passes = holes_in_clusters()
    targets = cs.BlockModel(origin=(0, 0, 2), size=(2.5, 2.5, 1), count=(40, 40, 1))
    kriged = cs.OrdinaryKriging(gaussian, passes).fit(xyz, grades, holes=holes)
    want = kriged.predict(targets, diagnostics=True)["pass"]
    sgs = cs.SGS(gaussian, passes).fit(xyz, grades, holes=holes)
    np.testing.assert_array_equal(sgs.passes(targets), want)
    assert {1.0, 2.0, 3.0} <= set(want[~np.isnan(want)]) and np.isnan(want).any()


def test_sgs_with_passes_is_reproducible_and_one_pass_is_the_search(tmp_path):
    xyz, grades, holes, passes = holes_in_clusters()
    targets = cs.BlockModel(origin=(0, 0, 2), size=(5, 5, 1), count=(20, 20, 1))

    def run(model):
        return (
            model.fit(xyz, grades, holes=holes).simulate(targets, n=3, seed=4, realizations=True).realizations
        )

    np.testing.assert_array_equal(run(cs.SGS(gaussian, [passes[1]])), run(cs.SGS(gaussian, passes[1])))
    by_pass = cs.SGS(gaussian, passes)
    reals = run(by_pass)
    np.testing.assert_array_equal(reals, run(cs.SGS(gaussian, passes)))
    assert not np.array_equal(reals, run(cs.SGS(gaussian, passes[2])))
    by_pass.to_parquet(tmp_path / "sgs.parquet")
    np.testing.assert_array_equal(
        cs.SGS.from_parquet(tmp_path / "sgs.parquet")
        .simulate(targets, n=3, seed=4, realizations=True)
        .realizations,
        reals,
    )
    meta, columns = cs.SGS(gaussian, passes[1]).fit(xyz, grades, holes=holes)._state()
    older = json.loads(meta)
    older["search"] = older["search"][0]
    np.testing.assert_array_equal(
        cs.SGS._from_state(json.dumps(older), columns)
        .simulate(targets, n=3, seed=4, realizations=True)
        .realizations,
        run(cs.SGS(gaussian, passes[1])),
    )
    mv = cs.MultivariateSimulation(cs.PCA(), [cs.SGS(gaussian, passes), cs.SGS(gaussian, passes[0])])
    two = mv.fit(xyz, np.column_stack([grades, grades**0.5]), holes=holes).simulate(targets, n=2, seed=1)
    assert len(two) == 2 and np.isfinite(two[0].mean).all()
    with pytest.raises(ValueError, match="at least one"):
        cs.SGS(gaussian, [])


def zoned_holes():
    xyz, grades, holes, passes = holes_in_clusters()
    zone = np.where(xyz[:, 0] < 40, "MS", "SM")
    grades = np.where(zone == "SM", grades * 3, grades)
    weights = np.where(xyz[:, 0] < 25, 0.5, 2.0)
    x, y = np.meshgrid(np.arange(0, 100, 5.0), np.arange(0, 100, 5.0))
    targets = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 2.0)])
    return xyz, grades, holes, weights, zone, targets, passes


def softened(passes, soft):
    return [cs.Search(s.radius, max_samples=12, min_samples=s.min_samples, soft=soft) for s in passes]


def test_sgs_with_one_domain_is_sgs_without():
    xyz, grades, holes, weights, _, targets, passes = zoned_holes()
    plain = cs.SGS(gaussian, passes).fit(xyz, grades, weights=weights, holes=holes)
    one = cs.SGS(gaussian, passes).fit(xyz, grades, weights=weights, holes=holes, domains="MS")
    want = plain.simulate(targets, n=3, seed=4, realizations=True).realizations
    got = one.simulate(targets, n=3, seed=4, realizations=True, domains="MS").realizations
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(one.passes(targets, domains="MS"), plain.passes(targets))


def test_hard_sgs_domains_ignore_the_other_domain():
    xyz, grades, holes, weights, zone, targets, passes = zoned_holes()
    ms = zone == "MS"

    def run(search, domains=None):
        data = (
            (xyz, grades, weights, holes)
            if domains is not None
            else (xyz[ms], grades[ms], weights[ms], holes[ms])
        )
        sgs = cs.SGS(gaussian, search).fit(data[0], data[1], weights=data[2], holes=data[3], domains=domains)
        on = {"domains": "MS"} if domains is not None else {}
        return sgs.simulate(targets, n=3, seed=1, realizations=True, **on).realizations

    np.testing.assert_array_equal(run(passes, zone), run(passes))
    alone = run(softened(passes, None))
    np.testing.assert_array_equal(run(softened(passes, 0.0), zone), alone)
    assert not np.array_equal(run(softened(passes, 10.0), zone), alone)


def test_each_sgs_domain_keeps_its_own_declustered_mean():
    xyz, grades, _, weights, zone, _, _ = zoned_holes()
    far = np.column_stack([1e4 + 100.0 * np.arange(40), np.full(40, 1e4), np.zeros(40)])
    labels = np.repeat(["MS", "SM"], 20)
    sgs = cs.SGS(gaussian, cs.Search(np.inf, max_samples=8)).fit(xyz, grades, weights=weights, domains=zone)
    mean = sgs.simulate(far, n=200, seed=3, domains=labels).mean
    for name in ("MS", "SM"):
        want = np.average(grades[zone == name], weights=weights[zone == name])
        assert mean[labels == name].mean() == pytest.approx(want, rel=0.1)


def test_a_contact_node_takes_its_own_domain_datum():
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [-10.0, 0.0]])
    sgs = cs.SGS(gaussian, cs.Search(50.0, soft=np.inf)).fit(
        xy, [1.0, 9.0, 2.0, 8.0], domains=["A", "B", "A", "B"]
    )
    at = sgs.simulate([[0.0, 0.0], [0.0, 0.0]], n=4, seed=1, realizations=True, domains=["A", "B"])
    np.testing.assert_array_equal(at.realizations, [[1.0, 9.0]] * 4)


def test_sgs_domains_errors_and_persistence(tmp_path):
    xyz, grades, holes, weights, zone, targets, passes = zoned_holes()
    soft = cs.Search(30.0, max_samples=12, soft={("MS", "SM"): 8.0})
    sgs = cs.SGS(gaussian, soft).fit(xyz, grades, weights=weights, holes=holes, domains=zone)
    labels = np.where(targets[:, 0] < 40, "MS", "SM")

    def run(model):
        return model.simulate(targets, n=2, seed=5, realizations=True, domains=labels).realizations

    sgs.to_parquet(tmp_path / "sgs.parquet")
    for again in (cs.SGS.from_parquet(tmp_path / "sgs.parquet"), pickle.loads(pickle.dumps(sgs))):
        np.testing.assert_array_equal(run(again), run(sgs))
    with pytest.raises(cs.InvalidInput, match="simulate needs domains"):
        sgs.simulate(targets, n=1)
    with pytest.raises(cs.InvalidInput, match="passes needs domains"):
        sgs.passes(targets)
    with pytest.raises(cs.InvalidInput, match="has no samples"):
        sgs.simulate(targets, n=1, domains="QE")
    with pytest.raises(cs.InvalidInput, match="takes none"):
        cs.SGS(gaussian, passes).fit(xyz, grades).simulate(targets, n=1, domains="MS")
    with pytest.raises(cs.InvalidInput, match="needs domains at fit"):
        cs.SGS(gaussian, soft).fit(xyz, grades)
    with pytest.raises(cs.InvalidInput, match="has no samples"):
        cs.SGS(gaussian, soft).fit(xyz, grades, domains="MS")


def test_sgs_domains_with_a_trend():
    xyz, grades, holes, weights, zone, targets, passes = zoned_holes()
    trend, at = xyz[:, 1] / 100, targets[:, 1] / 100
    ms = zone == "MS"

    def run(search, rows, labels=None, **on):
        sgs = cs.SGS(gaussian, search, classes=3).fit(
            xyz[rows],
            grades[rows],
            weights=weights[rows],
            holes=holes[rows],
            trend=trend[rows],
            domains=labels,
        )
        return sgs.simulate(targets, n=3, seed=2, realizations=True, trend=at, **on).realizations

    everything = np.ones(len(zone), bool)
    alone = run(passes, ms)
    np.testing.assert_array_equal(run(passes, everything, zone, domains="MS"), alone)
    np.testing.assert_array_equal(run(passes, ms, "MS", domains="MS"), alone)
    labels = np.where(targets[:, 0] < 40, "MS", "SM")
    soft = run(softened(passes, 10.0), everything, zone, domains=labels)
    hard = run(softened(passes, None), everything, zone, domains=labels)
    assert np.isfinite(soft).all() and not np.array_equal(soft, hard)


def banded(search=None):
    return cs.TurningBands(gaussian, bands=60, step=1.0, search=search, classes=3)


def test_turning_bands_domains_are_hard_unless_soft():
    xyz, grades, holes, weights, zone, targets, _ = zoned_holes()
    # Corners holding every datum: the bands cover one box whatever the data.
    targets = np.vstack([targets, [[0.0, 0.0, 0.0], [100.0, 100.0, 4.0]]])
    trend, at = xyz[:, 1] / 100, targets[:, 1] / 100
    ms, everything = zone == "MS", np.ones(len(zone), bool)
    hard = cs.Search(40.0, max_samples=12, max_per_hole=3, high_grade=(4.0, 6.0))

    def run(search, rows, labels=None, trended=False, **on):
        tb = banded(search).fit(
            xyz[rows],
            grades[rows],
            weights=weights[rows],
            holes=holes[rows],
            trend=trend[rows] if trended else None,
            domains=labels,
        )
        on |= {"trend": at} if trended else {}
        return tb.simulate(targets, n=3, seed=2, realizations=True, **on).realizations

    for trended in (False, True):
        alone = run(hard, ms, trended=trended)
        np.testing.assert_array_equal(run(hard, everything, zone, trended, domains="MS"), alone)
        np.testing.assert_array_equal(run(hard, ms, "MS", trended, domains="MS"), alone)
    labels = np.where(targets[:, 0] < 40, "MS", "SM")
    soft = run(cs.Search(40.0, max_samples=12, soft=20.0), everything, zone, True, domains=labels)
    hard = run(cs.Search(40.0, max_samples=12), everything, zone, True, domains=labels)
    assert np.isfinite(soft).all() and not np.array_equal(soft, hard)


def test_each_turning_bands_domain_keeps_its_own_declustered_mean():
    xyz, grades, _, weights, zone, _, _ = zoned_holes()
    far = np.column_stack([25.0 * np.arange(40), np.full(40, 150.0), np.zeros(40)])
    labels = np.repeat(["MS", "SM"], 20)
    tb = banded(cs.Search(40.0, max_samples=8)).fit(xyz, grades, weights=weights, domains=zone)
    mean = tb.simulate(far, n=200, seed=3, domains=labels).mean
    for name in ("MS", "SM"):
        want = np.average(grades[zone == name], weights=weights[zone == name])
        assert mean[labels == name].mean() == pytest.approx(want, rel=0.1)


def test_turning_bands_domains_errors_and_persistence(tmp_path):
    xyz, grades, holes, weights, zone, targets, _ = zoned_holes()
    soft = cs.Search(30.0, max_samples=12, soft={("MS", "SM"): 8.0})
    tb = banded(soft).fit(xyz, grades, weights=weights, holes=holes, domains=zone)
    labels = np.where(targets[:, 0] < 40, "MS", "SM")

    def run(model):
        return model.simulate(targets, n=2, seed=5, realizations=True, domains=labels).realizations

    tb.to_parquet(tmp_path / "tb.parquet")
    for again in (cs.TurningBands.from_parquet(tmp_path / "tb.parquet"), pickle.loads(pickle.dumps(tb))):
        np.testing.assert_array_equal(run(again), run(tb))
    with pytest.raises(cs.InvalidInput, match="simulate needs domains"):
        tb.simulate(targets, n=1)
    with pytest.raises(cs.InvalidInput, match="has no samples"):
        tb.simulate(targets, n=1, domains="QE")
    with pytest.raises(cs.InvalidInput, match="takes none"):
        banded().fit(xyz, grades).simulate(targets, n=1, domains="MS")
    with pytest.raises(cs.InvalidInput, match="needs domains at fit"):
        banded(soft).fit(xyz, grades)
    with pytest.raises(cs.InvalidInput, match="has no samples"):
        banded(soft).fit(xyz, grades, domains="MS")
    with pytest.raises(cs.InvalidInput, match="does not take domains"):
        cs.MultivariateSimulation(cs.PCA(), [banded(soft)])
