import boitata as bt
import numpy as np
import pytest

rng = np.random.default_rng(7)


@pytest.fixture
def skewed():
    return rng.lognormal(0.0, 0.8, 500)


def test_gaussian_imputer_keeps_data_and_reproduces_correlation():
    g = rng.standard_normal((2000, 2))
    full = np.exp(np.column_stack([g[:, 0], 0.8 * g[:, 0] + 0.6 * g[:, 1]]))
    data = full.copy()
    data[::3, 1] = np.nan
    data[1::7, 0] = np.nan
    imputer = bt.GaussianImputer(seed=1).fit(data)
    assert imputer.correlation_[0, 1] == pytest.approx(0.8, abs=0.04)
    out = imputer.transform(data)
    seen = ~np.isnan(data)
    np.testing.assert_array_equal(out[seen], data[seen])
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out, imputer.transform(data))
    holed = ~seen.all(axis=1)
    assert np.corrcoef(np.log(out[holed]).T)[0, 1] == pytest.approx(0.8, abs=0.06)
    with pytest.raises(ValueError, match="< 2 values"):
        bt.GaussianImputer().fit(np.column_stack([full[:, 0], np.full(2000, np.nan)]))


def test_spatial_imputer_follows_neighbors():
    xy = rng.uniform(0, 100, (400, 2))
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    h = np.minimum(d / 20.0, 1.0)
    chol = np.linalg.cholesky(1 - 1.5 * h + 0.5 * h**3 + 1e-9 * np.eye(400))
    y = chol @ rng.standard_normal((400, 2))
    full = np.column_stack([y[:, 0], 0.7 * y[:, 0] + np.sqrt(0.51) * y[:, 1]])
    data = full.copy()
    hidden = np.arange(400) % 2 == 0
    data[hidden, 1] = np.nan
    spatial = bt.GaussianImputer(seed=3, spatial=bt.Variogram([("spherical", 1.0, 20.0)]))
    out = spatial.fit(data, coords=xy).transform(data)
    np.testing.assert_array_equal(out[~np.isnan(data)], data[~np.isnan(data)])
    np.testing.assert_array_equal(out, spatial.fit_transform(data, coords=xy))
    plain = bt.GaussianImputer(seed=3).fit_transform(data)
    err = [np.sqrt(np.mean((o[hidden, 1] - full[hidden, 1]) ** 2)) for o in (out, plain)]
    assert err[0] < 0.8 * err[1]
    nugget = bt.GaussianImputer(seed=3, spatial=bt.Variogram([], nugget=1.0)).fit(data, coords=xy)
    np.testing.assert_array_equal(nugget.transform(data), plain)
    back = bt.GaussianImputer.from_json(spatial.to_json())
    np.testing.assert_array_equal(back.transform(data), out)
    with pytest.raises(ValueError, match="coords"):
        bt.GaussianImputer(spatial=bt.Variogram([("spherical", 1.0, 20.0)])).fit(data)


def test_normal_score_is_standard_and_invertible(skewed):
    ns = bt.NormalScore()
    y = ns.fit_transform(skewed)
    assert abs(y.mean()) < 0.01 and abs(y.std() - 1) < 0.02
    np.testing.assert_allclose(ns.inverse_transform(y), skewed, rtol=1e-12)
    np.testing.assert_allclose(ns.transform(skewed), y, atol=1e-12)


@pytest.mark.parametrize(
    ("make", "args"),
    [
        (lambda: bt.HermiteAnamorphosis(), lambda x, xy: (x[:, 0],)),
        (lambda: bt.BoxCox(), lambda x, xy: (x[:, 0],)),
        (lambda: bt.PPMT(iterations=5), lambda x, xy: (x,)),
        (lambda: bt.PCA(), lambda x, xy: (x,)),
        (lambda: bt.MAF(lag=10.0), lambda x, xy: (x, xy)),
        (lambda: bt.StepwiseConditional(classes=5), lambda x, xy: (x,)),
        (lambda: bt.GaussianImputer(seed=2), lambda x, xy: (np.where(np.eye(200, 2) > 0, np.nan, x),)),
    ],
)
def test_fit_transform_is_fit_then_transform(make, args):
    x = rng.lognormal(0.0, 0.5, (200, 2))
    a = args(x, rng.uniform(0, 100, (200, 2)))
    np.testing.assert_array_equal(make().fit_transform(*a), make().fit(*a).transform(a[0]))


def test_censored_normal_score_orders_between_uncensored_neighbors():
    values = np.array([0.5, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 6.0, 8.0])
    censored = np.array([False, False, True, True, True, True, True, False, False])
    scores = bt.NormalScore().fit_transform(values, censored=censored, seed=0)
    below, above = scores[:2].max(), scores[7:].min()
    assert (scores[2:7] > below).all() and (scores[2:7] < above).all()


def test_censored_normal_score_avoids_spurious_order_among_ties():
    """The point of `censored=`: a naive fit ties censored values by their input order, so an array
    recorded in a spatially (or otherwise) meaningful order leaves the tied group falsely correlated
    with that order; shuffling the tie removes that artifact instead of asserting a false rank."""
    n = 3000
    coord = np.sort(rng.uniform(0, 100, n))  # samples logged in coordinate order, as along a hole
    true = rng.lognormal(0.0, 0.9, n)  # independent of coord: order carries no real information
    limit = np.quantile(true, 0.45)
    censored = true < limit
    values = np.where(censored, limit, true)

    naive = bt.NormalScore().fit_transform(values)
    aware = bt.NormalScore().fit_transform(values, censored=censored, seed=0)
    naive_corr = np.corrcoef(naive[censored], coord[censored])[0, 1]
    aware_corr = np.corrcoef(aware[censored], coord[censored])[0, 1]
    assert abs(naive_corr) > 0.9
    assert abs(aware_corr) < 0.1


def test_censored_normal_score_seed_is_reproducible():
    n = 500
    values = np.where(rng.uniform(size=n) < 0.4, 2.0, rng.lognormal(0.5, 0.7, n))
    censored = values == 2.0
    a = bt.NormalScore().fit_transform(values, censored=censored, seed=5)
    b = bt.NormalScore().fit_transform(values, censored=censored, seed=5)
    np.testing.assert_array_equal(a, b)
    c = bt.NormalScore().fit_transform(values, censored=censored, seed=6)
    assert not np.array_equal(a, c)


def test_censored_none_matches_pre_censoring_behavior(skewed):
    censored = np.zeros(skewed.size, dtype=bool)
    plain = bt.NormalScore().fit_transform(skewed)
    explicit = bt.NormalScore().fit_transform(skewed, censored=censored, seed=3)
    np.testing.assert_array_equal(plain, explicit)
    default = bt.NormalScore().fit_transform(skewed, censored=None)
    np.testing.assert_array_equal(plain, default)


def test_normal_score_requires_fit():
    with pytest.raises(bt.InvalidInput):
        bt.NormalScore().transform([1.0])


def test_kernel_density_reference_keeps_bounds_and_mean(skewed):
    w = rng.uniform(0.5, 2.0, skewed.size)
    free = bt.KernelDensity(bandwidth="scott").fit(skewed, weights=w)
    x = np.linspace(-10, 40, 200_001)
    f = free.pdf(x)
    assert np.trapezoid(f, x) == pytest.approx(1.0, abs=1e-6)
    assert np.trapezoid(x * f, x) == pytest.approx(np.average(skewed, weights=w), rel=1e-6)
    bounded = bt.KernelDensity(lower=0.0).fit("v", weights="w", data={"v": skewed, "w": w})
    logged = bt.KernelDensity(log=True, bandwidth=0.2).fit(skewed)
    for kde in (bounded, logged):
        assert kde.pdf([-1.0, -1e-9]).max() == 0.0
        draws = kde.sample(10_000, seed=3)
        assert draws.min() >= 0.0
        np.testing.assert_array_equal(draws, kde.sample(10_000, seed=3))
        ns = bt.NormalScore(reference=kde)
        y = ns.fit_transform(skewed)
        np.testing.assert_allclose(ns.inverse_transform(y), skewed, rtol=1e-9)
        assert ns.inverse_transform([-9.0])[0] >= 0.0
    np.testing.assert_allclose(logged.cdf(logged.quantile([0.1, 0.9])), [0.1, 0.9], atol=1e-9)
    with pytest.raises(bt.InvalidInput):
        bt.NormalScore(reference=logged).fit(skewed, weights=w)
    with pytest.raises(bt.InvalidInput):
        bt.NormalScore(reference=logged).fit(skewed, censored=np.zeros(skewed.size, dtype=bool))
    with pytest.raises(bt.InvalidInput):
        bt.KernelDensity(bandwidth="wide")
    with pytest.raises(bt.InvalidInput):
        bt.NormalScore(reference=bt.KernelDensity())


def test_gaussian_mixture_recovers_components_and_picks_their_count():
    a = rng.multivariate_normal([-4.0, 2.0], [[1.0, 0.6], [0.6, 1.0]], 600)
    b = rng.multivariate_normal([3.0, -1.0], [[0.25, 0.0], [0.0, 0.25]], 1400)
    data = np.vstack([a, b])
    gm = bt.GaussianMixture(seed=1).fit(data)
    assert list(gm.bic_) == [1, 2, 3, 4, 5, 6] and min(gm.bic_, key=gm.bic_.get) == 2
    order = np.argsort(gm.means_[:, 0])
    np.testing.assert_allclose(gm.proportions_[order], [0.3, 0.7], atol=0.02)
    np.testing.assert_allclose(gm.means_[order], [[-4.0, 2.0], [3.0, -1.0]], atol=0.12)
    assert gm.covariances_.shape == (2, 2, 2)
    assert (gm.predict(data[:600]) == order[0]).mean() > 0.99
    again = bt.GaussianMixture(seed=1).fit(data)
    np.testing.assert_array_equal(gm.means_, again.means_)
    np.testing.assert_array_equal(gm.sample(50, seed=2), again.sample(50, seed=2))
    one = bt.GaussianMixture(components=2).fit(data[:, 0])
    assert one.sample(10).shape == (10, 1)
    y = bt.NormalScore(reference=one).fit_transform(data[:, 0])
    assert abs(y.mean()) < 0.05
    with pytest.raises(bt.InvalidInput):
        bt.NormalScore(reference=gm)


def test_gaussian_imputer_with_a_mixture_keeps_the_empty_corner_empty():
    z = rng.standard_normal((3000, 2))
    arm = np.arange(3000)[:, None] % 2 == 0
    full = np.exp(np.where(arm, [2.0, -1.0] + z * [0.8, 0.3], [-1.0, 2.0] + z * [0.3, 0.8]))
    holed = full.copy()
    holed[::3, 1] = np.nan
    corner = {
        k: np.mean(np.all(bt.GaussianImputer(components=k, seed=0).fit_transform(holed)[::3] > 1.5, axis=1))
        for k in (1, None)
    }
    assert corner[None] < 0.5 * corner[1]


def test_hermite_anamorphosis_moments_and_support(skewed):
    anam = bt.HermiteAnamorphosis(degree=40).fit(skewed)
    assert anam.mean_ == pytest.approx(skewed.mean(), rel=1e-6)
    assert anam.block(1.0).variance_ == pytest.approx(anam.variance_)
    assert anam.block(0.6).variance_ < anam.variance_
    gt = anam.grade_tonnage([0.0, 1.0, 2.0])
    assert gt["tonnage"][0] == pytest.approx(1.0)
    assert np.all(np.diff(gt["tonnage"]) <= 0)


def test_box_cox_zero_is_log(skewed):
    bc = bt.BoxCox(lambda_=0.0).fit(skewed)
    np.testing.assert_allclose(bc.transform(skewed), np.log(skewed))
    np.testing.assert_allclose(bc.inverse_transform(bc.transform(skewed)), skewed)
    assert abs(bt.BoxCox().fit(skewed).lambda_) < 0.3


def test_ppmt_decorrelates_and_inverts():
    x = rng.normal(size=(400, 2)) @ np.array([[1.0, 0.8], [0.0, 0.6]])
    ppmt = bt.PPMT(seed=3).fit(x)
    g = ppmt.transform(x)
    assert abs(np.corrcoef(g.T)[0, 1]) < 0.1
    np.testing.assert_allclose(ppmt.inverse_transform(g), x, atol=1e-6)


def test_ppmt_marginal_step_with_weights_round_trips(skewed):
    x = np.column_stack([skewed, skewed * rng.lognormal(0, 0.3, len(skewed))])
    ppmt = bt.PPMT(seed=3).fit(x, weights=rng.uniform(0.5, 1.5, len(x)))
    np.testing.assert_allclose(ppmt.inverse_transform(ppmt.transform(x)), x, atol=1e-6)
    with pytest.raises(bt.InvalidInput):
        bt.PPMT(marginal=False).fit(x, weights=np.ones(len(x)))


def test_pca_scores_are_uncorrelated_by_decreasing_variance():
    x = rng.normal(size=(500, 3)) @ np.array([[2.0, 0.5, 0.0], [0.0, 1.0, 0.3], [0.0, 0.0, 0.2]])
    pca = bt.PCA().fit(x)
    scores = pca.transform(x)
    np.testing.assert_allclose(np.cov(scores.T, bias=True), np.diag(pca.explained_variance_), atol=1e-10)
    assert np.all(np.diff(pca.explained_variance_) <= 0)
    assert pca.explained_variance_ratio_.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(pca.inverse_transform(scores), x, atol=1e-10)
    assert bt.PCA(standardize=True).fit(x).explained_variance_.sum() == pytest.approx(3.0)


def test_maf_factors_are_uncorrelated_at_lag():
    grid = np.array([(i, j) for j in range(30) for i in range(30)], float)
    smooth = np.sin(grid[:, 0] / 5) + np.cos(grid[:, 1] / 6)
    x = np.column_stack([smooth + 0.3 * rng.normal(size=900), smooth - rng.normal(size=900)])
    maf = bt.MAF(lag=1.0, tolerance=0.01).fit(x, grid)
    f = maf.transform(x)
    np.testing.assert_allclose(np.cov(f.T, bias=True), np.eye(2), atol=1e-10)
    right = np.flatnonzero(grid[:, 0] < 29)
    d = np.vstack([f[right] - f[right + 1], f[:-30] - f[30:]])
    np.testing.assert_allclose((d.T @ d / (2 * len(d)))[0, 1], 0.0, atol=1e-10)
    assert maf.gammas_[0] < maf.gammas_[1]
    np.testing.assert_allclose(maf.inverse_transform(f), x, atol=1e-10)


def test_stepwise_conditional_removes_nonlinear_dependence():
    u = rng.normal(size=3000)
    x = np.column_stack([u, u**2 + 0.3 * rng.normal(size=3000)])
    sct = bt.StepwiseConditional(classes=30).fit(x)
    g = sct.transform(x)
    assert abs(np.corrcoef(g[:, 0], g[:, 1])[0, 1]) < 0.1
    assert abs(np.corrcoef(g[:, 0] ** 2, g[:, 1])[0, 1]) < 0.1
    assert abs(g.mean()) < 0.05 and abs(g.std() - 1) < 0.05
    np.testing.assert_allclose(sct.inverse_transform(g), x, atol=1e-10)
    with pytest.raises(bt.InvalidInput):
        sct.transform(x[:, :1])
    weights = np.where(x[:, 1] > 1, 3.0, 1.0)
    g = bt.StepwiseConditional(classes=30).fit(x, weights=weights).transform(x)
    assert abs(np.average(g[:, 1], weights=weights)) < 0.05 and g[:, 1].mean() < -0.05
    with pytest.raises(bt.InvalidInput):
        bt.StepwiseConditional().fit(x, weights=weights[1:])


def test_cell_declustering_downweights_clusters():
    grid = np.array([(x, y) for x in range(0, 100, 10) for y in range(0, 100, 10)], float)
    cluster = rng.uniform(0, 10, size=(50, 2))
    coords = np.vstack([grid, cluster])
    values = np.r_[np.zeros(len(grid)), np.ones(len(cluster))]
    d = bt.cell_declustering(coords, values, sizes=np.arange(5.0, 50.0, 5.0))
    assert d.mean < values.mean()
    assert d.weights.sum() == pytest.approx(len(values))
    assert len(d.sizes) == len(d.means) == 9
    assert d.mean == pytest.approx(d.means[list(d.sizes).index(d.cell_size)], rel=1e-12)
    assert d.mean == pytest.approx(np.average(values, weights=d.weights), rel=1e-12)
    fixed = bt.cell_declustering(coords, values, cell_size=d.cell_size)
    np.testing.assert_allclose(fixed.weights, d.weights)
    points = bt.PointSet(coords, {"v": values})
    named = bt.cell_declustering(points, "v", sizes=np.arange(5.0, 50.0, 5.0))
    np.testing.assert_array_equal(named.weights, d.weights)
    np.testing.assert_array_equal(
        bt.polygon_declustering(points, "v", nodes=400).weights,
        bt.polygon_declustering(coords, values, nodes=400).weights,
    )


def test_detrend_removes_linear_trend():
    coords = rng.uniform(0, 100, size=(50, 2))
    values = 3.0 + 0.5 * coords[:, 0] - 0.2 * coords[:, 1]
    trend, residuals = bt.detrend(coords, values, degree=1)
    assert np.abs(residuals).max() < 1e-8
    np.testing.assert_array_equal(bt.detrend(bt.PointSet(coords, {"v": values}), "v")[1], residuals)
    np.testing.assert_allclose(trend.predict(coords), values, atol=1e-8)


def test_kernel_trend_picks_a_bandwidth_and_smooths_categories():
    local = np.random.default_rng(4)
    coords = local.uniform(0, 100, size=(300, 2))
    smooth = np.sin(coords[:, 0] / 20.0)
    values = smooth + local.normal(0, 0.3, 300)
    points = bt.PointSet(coords, {"v": values, "w": np.ones(300), "c": np.where(smooth > 0, "a", "b")})
    trend, residuals = bt.detrend(points, "v", bandwidth=[2.0, 8.0, 50.0], weights="w")
    assert trend.bandwidth == 8.0 and trend.degree is None and trend.coefficients is None
    assert trend.scores.argmin() == 1
    np.testing.assert_allclose(residuals, values - trend.predict(coords))
    grid = bt.BlockModel((0, 0, 0), (5, 5, 1), (20, 20, 1))
    assert trend.predict(grid).shape == (400,)
    assert np.isnan(trend.predict([[1e4, 1e4]]))[0]

    categories, indicators = bt.detrend(points, "c", bandwidth=10.0, categorical=True)
    assert categories.categories == ["a", "b"]
    p = categories.predict(grid)
    total = np.asarray(p["a"]) + np.asarray(p["b"])
    np.testing.assert_allclose(total, 1.0)
    assert set(indicators.column_names) == {"a", "b"}
    restored = bt.Trend.from_json(categories.to_json())
    np.testing.assert_array_equal(restored.predict(grid)["a"], p["a"])
    with pytest.raises(ValueError, match="bandwidth"):
        bt.detrend(points, "v", weights="w")


def test_normal_cdf_and_ppf_are_inverse():
    x = np.linspace(-3, 3, 13)
    np.testing.assert_allclose(bt.normal_ppf(bt.normal_cdf(x)), x, atol=1e-6)
    assert bt.normal_cdf([0.0])[0] == pytest.approx(0.5)


def test_upscale_conserves_samples():
    coords = rng.uniform(0, 20, size=(100, 2))
    centers, means, counts = bt.upscale(coords, np.ones(100), (10, 10, 1))
    assert counts.sum() == 100 and np.allclose(means, 1.0)
    assert centers.shape[1] == 3
    named = bt.upscale(bt.PointSet(coords, {"v": np.ones(100)}), "v", (10, 10, 1), origin=(0, 0, 0))
    np.testing.assert_array_equal(named[2], counts)


def test_affine_correction_scales_variance(skewed):
    out = bt.affine_correction(skewed, 0.5)
    assert out.mean() == pytest.approx(skewed.mean())
    assert out.var() == pytest.approx(0.5 * skewed.var(), rel=1e-6)


def test_uniform_conditioning_recovers_all_at_zero_cutoff(skewed):
    anam = bt.HermiteAnamorphosis().fit(skewed)
    uc = bt.UniformConditioning(anam, 0.8, r_panel=0.6)
    rec = uc.panel_recovery(float(skewed.mean()), [0.0])
    assert rec["tonnage"][0] == pytest.approx(1.0, abs=1e-6)


def test_uniform_conditioning_with_r_panel_keeps_its_recoveries():
    values = np.random.default_rng(3).lognormal(0.0, 0.8, 400)
    uc = bt.UniformConditioning(bt.HermiteAnamorphosis(degree=30).fit(values), 0.8, r_panel=0.6)
    rec = uc.panel_recovery(1.5, [0.5, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        rec["tonnage"],
        [0.9919641556480453, 0.7709675766005867, 0.18412255756967455, 0.03426193506810016],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        rec["metal"],
        [1.4964939621944264, 1.3166196037377833, 0.47908736272868696, 0.12471125298512278],
        rtol=1e-9,
    )


def uc_panels(skewed):
    anam = bt.HermiteAnamorphosis(degree=30).fit(skewed)
    panels = bt.BlockModel(origin=(0, 0), size=(50, 50), count=(4, 3))
    grade = skewed.mean() * np.linspace(0.5, 1.8, 12)
    grade[5] = np.nan
    panels = panels.with_column("grade", grade).with_column("ev", anam.variance_ * np.linspace(0.2, 0.5, 12))
    smus = panels.discretize(5)
    return anam, panels, smus.with_column("rank", rng.normal(size=len(smus)))


def test_uniform_conditioning_localizes_band_means(skewed):
    anam, panels, smus = uc_panels(skewed)
    uc = bt.UniformConditioning(anam, 0.8, r_panel=0.5)
    out = uc.localize(smus, "rank", panels, "grade", name="uc")
    owner, rank, local = smus["block"].astype(int), out["rank"], out["uc"]
    for p, g in enumerate(panels["grade"]):
        mine = owner == p
        if np.isnan(g):
            assert np.isnan(local[mine]).all()
            continue
        assert local[mine].mean() == pytest.approx(g, rel=1e-9)
        by_rank = local[mine][np.argsort(rank[mine])]
        np.testing.assert_allclose(by_rank, uc.localized_grades(g, 25), rtol=1e-12)
        assert (np.diff(by_rank) >= 0).all()

    tied = uc.localize(smus.with_column("rank", np.zeros(len(smus))), "rank", panels, "grade")
    assert (np.diff(tied["localized"][owner == 0]) >= 0).all()


def test_uniform_conditioning_per_panel_coefficient(skewed):
    anam, panels, smus = uc_panels(skewed)
    uc = bt.UniformConditioning(anam, 0.8)
    cutoffs = [0.5, 1.0, 2.0]
    curves = uc.grade_tonnage(panels, "grade", cutoffs, estimate_variance="ev")
    each = [
        uc.panel_recovery(g, cutoffs, estimate_variance=e)
        for g, e in zip(panels["grade"], panels["ev"], strict=True)
        if np.isfinite(g)
    ]
    for key in ("tonnage", "metal", "benefit"):
        np.testing.assert_allclose(curves[key], 2500 * sum(r[key] for r in each), rtol=1e-12)
    np.testing.assert_allclose(curves["metal"], curves["tonnage"] * curves["mean_grade"], rtol=1e-12)
    np.testing.assert_allclose(curves["benefit"], curves["metal"] - curves["cutoff"] * curves["tonnage"])
    dense = uc.grade_tonnage(
        panels.with_column("d", np.full(12, 2.7)), "grade", cutoffs, estimate_variance="ev", density="d"
    )
    np.testing.assert_allclose(dense["tonnage"], 2.7 * curves["tonnage"], rtol=1e-12)
    np.testing.assert_allclose(dense["mean_grade"], curves["mean_grade"], rtol=1e-12)
    out = uc.localize(smus, "rank", panels, "grade", estimate_variance="ev")
    assert out["localized"][smus["block"] == 0].mean() == pytest.approx(panels["grade"][0], rel=1e-9)
    with pytest.raises(bt.InvalidInput):
        uc.localize(smus, "rank", panels, "grade")
    with pytest.raises(bt.InvalidInput):
        bt.UniformConditioning(anam, 0.8, r_panel=0.5).grade_tonnage(
            panels, "grade", cutoffs, estimate_variance="ev"
        )
    with pytest.raises(KeyError):
        uc.localize(smus, "missing", panels, "grade", estimate_variance="ev")


def test_mean_grade_is_nan_above_an_empty_cutoff(skewed):
    anam, panels, _ = uc_panels(skewed)
    uc = bt.UniformConditioning(anam, 0.8)
    top = [0.5, 1e6]
    curves = [
        anam.grade_tonnage(top),
        uc.panel_recovery(1.0, top, estimate_variance=0.1),
        uc.grade_tonnage(panels, "grade", top, estimate_variance="ev"),
        bt.grade_tonnage(skewed, top),
    ]
    for gt in curves:
        assert gt["tonnage"][1] == 0 and np.isnan(gt["mean_grade"][1])
        assert gt["tonnage"][0] > 0 and np.isfinite(gt["mean_grade"][0])


def test_defaulted_transform_options_are_keyword_only(skewed):
    with pytest.raises(TypeError):
        bt.NormalScore().fit(skewed, np.ones_like(skewed))
    with pytest.raises(TypeError):
        bt.HermiteAnamorphosis(30)
    with pytest.raises(TypeError):
        bt.PCA().fit_transform(np.column_stack([skewed, skewed]), np.ones_like(skewed))


def test_uniform_conditioning_needs_nested_ranked_blocks(skewed):
    anam, panels, smus = uc_panels(skewed)
    uc = bt.UniformConditioning(anam, 0.8, r_panel=0.5)
    shifted = bt.BlockModel(origin=(5, 0), size=(10, 10), count=(20, 15)).with_column("rank", np.zeros(300))
    with pytest.raises(bt.InvalidInput):
        uc.localize(shifted, "rank", panels, "grade")
    rank = smus["rank"]
    rank[np.flatnonzero(smus["block"] == 0)[0]] = np.nan
    with pytest.raises(bt.InvalidInput):
        uc.localize(smus.with_column("rank", rank), "rank", panels, "grade")


def test_normal_score_tails_bound_the_back_transform(skewed):
    ns = bt.NormalScore().fit(skewed)
    back = ns.inverse_transform([-9.0, 9.0])
    assert back[0] == pytest.approx(skewed.min()) and back[1] == pytest.approx(skewed.max())
    wide = bt.NormalScore(tails=(0.0, 100.0)).fit(skewed)
    low, high = wide.inverse_transform([-4.0, 4.0])
    assert 0.0 < low < skewed.min() and skewed.max() < high < 100.0


def test_block_kriging_estimate_variance_feeds_uniform_conditioning(skewed):
    anam = bt.HermiteAnamorphosis(degree=30).fit(skewed)
    xy = rng.uniform(0, 200, (skewed.size, 2))
    variogram = bt.Variogram([("spherical", anam.variance_, 80.0)])
    panels = bt.BlockModel(origin=(0, 0), size=(50, 50), count=(4, 4))
    kriging = bt.BlockKriging(variogram, bt.Search(radius=120, max_samples=16), size=(50, 50)).fit(xy, skewed)
    d = kriging.predict(panels, diagnostics=True)
    np.testing.assert_allclose(
        d["estimate_variance"], d["support_variance"] - d["variance"] - 2 * d["lagrange"], rtol=1e-12
    )
    panels = panels.with_column("V", d["value"]).with_column("ev", d["estimate_variance"])
    smus = panels.discretize(5)
    smus = smus.with_column("rank", rng.normal(size=len(smus)))
    out = bt.UniformConditioning(anam, 0.9).localize(smus, "rank", panels, "V", estimate_variance="ev")
    means = np.bincount(smus["block"].astype(int), weights=out["localized"]) / 25
    np.testing.assert_allclose(means, d["value"], rtol=1e-9)


def test_despike_breaks_ties_consistently():
    x = np.arange(200.0)
    coords = np.column_stack([x, np.zeros(200)])
    values = np.where(x % 5 == 0, 0.1, np.where(x < 100, 0.5, 3.0) + rng.uniform(0, 0.4, 200))
    out = bt.despike(coords, values, radii=[3.0, 10.0], seed=2)
    assert len(np.unique(out)) == 200
    untied = np.subtract.outer(values, values) != 0
    order = np.sign(np.subtract.outer(out, out)) == np.sign(np.subtract.outer(values, values))
    assert order[untied].all()
    assert np.abs(out - values).max() < 1e-4
    tied = values == 0.1
    assert out[tied & (x < 100)].max() < out[tied & (x >= 100)].min()
    np.testing.assert_array_equal(out, bt.despike(coords, values, radii=[3.0, 10.0], seed=2))
    both = bt.despike(coords, np.column_stack([values, 2 * values]), radii=[3.0])
    np.testing.assert_array_equal(np.argsort(both[:, 0]), np.argsort(both[:, 1]))
    points = bt.PointSet(coords, {"a": values, "b": 2 * values})
    table = bt.despike(points, ["a", "b"], radii=[3.0])
    np.testing.assert_array_equal(np.asarray(table["a"]), both[:, 0])
    np.testing.assert_array_equal(bt.despike(points, "a"), bt.despike(coords, values))
    with pytest.raises(ValueError):
        bt.despike(coords, values[:10])


def test_spatial_bootstrap_widens_with_correlation():
    r = np.random.default_rng(4)
    coords = r.uniform(0, 100, (100, 2))
    values = r.lognormal(0.0, 0.5, 100)
    nugget = bt.Variogram([], nugget=1.0)
    table = bt.spatial_bootstrap(coords, values, nugget, n=1000, quantiles=[0.5], cutoffs=[1.0])
    assert table.column_names == ["mean", "P50", "above 1"]
    np.testing.assert_allclose(np.std(table["mean"]), values.std() / 10, rtol=0.1)
    long = bt.spatial_bootstrap(coords, values, bt.Variogram([("spherical", 1.0, 1e6)]), n=1000)
    np.testing.assert_allclose(np.std(long["mean"]), values.std(), rtol=0.1)
    points = bt.PointSet(coords, {"v": values, "w": np.ones(100)})
    np.testing.assert_array_equal(
        bt.spatial_bootstrap(points, "v", nugget, weights="w")["mean"],
        bt.spatial_bootstrap(coords, values, nugget)["mean"],
    )
    with pytest.raises(ValueError):
        bt.spatial_bootstrap(coords, values, nugget, weights=np.zeros(100))
