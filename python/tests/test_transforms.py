import ceres as cs
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
    imputer = cs.GaussianImputer(seed=1).fit(data)
    assert imputer.correlation_[0, 1] == pytest.approx(0.8, abs=0.04)
    out = imputer.transform(data)
    seen = ~np.isnan(data)
    np.testing.assert_array_equal(out[seen], data[seen])
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out, imputer.transform(data))
    holed = ~seen.all(axis=1)
    assert np.corrcoef(np.log(out[holed]).T)[0, 1] == pytest.approx(0.8, abs=0.06)
    with pytest.raises(ValueError, match="< 2 values"):
        cs.GaussianImputer().fit(np.column_stack([full[:, 0], np.full(2000, np.nan)]))


def test_normal_score_is_standard_and_invertible(skewed):
    ns = cs.NormalScore()
    y = ns.fit_transform(skewed)
    assert abs(y.mean()) < 0.01 and abs(y.std() - 1) < 0.02
    np.testing.assert_allclose(ns.inverse_transform(y), skewed, rtol=1e-12)
    np.testing.assert_allclose(ns.transform(skewed), y, atol=1e-12)


@pytest.mark.parametrize(
    ("make", "args"),
    [
        (lambda: cs.HermiteAnamorphosis(), lambda x, xy: (x[:, 0],)),
        (lambda: cs.BoxCox(), lambda x, xy: (x[:, 0],)),
        (lambda: cs.PPMT(iterations=5), lambda x, xy: (x,)),
        (lambda: cs.PCA(), lambda x, xy: (x,)),
        (lambda: cs.MAF(lag=10.0), lambda x, xy: (x, xy)),
        (lambda: cs.StepwiseConditional(classes=5), lambda x, xy: (x,)),
        (lambda: cs.GaussianImputer(seed=2), lambda x, xy: (np.where(np.eye(200, 2) > 0, np.nan, x),)),
    ],
)
def test_fit_transform_is_fit_then_transform(make, args):
    x = rng.lognormal(0.0, 0.5, (200, 2))
    a = args(x, rng.uniform(0, 100, (200, 2)))
    np.testing.assert_array_equal(make().fit_transform(*a), make().fit(*a).transform(a[0]))


def test_normal_score_requires_fit():
    with pytest.raises(cs.InvalidInput):
        cs.NormalScore().transform([1.0])


def test_hermite_anamorphosis_moments_and_support(skewed):
    anam = cs.HermiteAnamorphosis(degree=40).fit(skewed)
    assert anam.mean_ == pytest.approx(skewed.mean(), rel=1e-6)
    assert anam.block(1.0).variance_ == pytest.approx(anam.variance_)
    assert anam.block(0.6).variance_ < anam.variance_
    gt = anam.grade_tonnage([0.0, 1.0, 2.0])
    assert gt["tonnage"][0] == pytest.approx(1.0)
    assert np.all(np.diff(gt["tonnage"]) <= 0)


def test_box_cox_zero_is_log(skewed):
    bc = cs.BoxCox(0.0).fit(skewed)
    np.testing.assert_allclose(bc.transform(skewed), np.log(skewed))
    np.testing.assert_allclose(bc.inverse_transform(bc.transform(skewed)), skewed)
    assert abs(cs.BoxCox().fit(skewed).lambda_) < 0.3


def test_ppmt_decorrelates_and_inverts():
    x = rng.normal(size=(400, 2)) @ np.array([[1.0, 0.8], [0.0, 0.6]])
    ppmt = cs.PPMT(seed=3).fit(x)
    g = ppmt.transform(x)
    assert abs(np.corrcoef(g.T)[0, 1]) < 0.1
    np.testing.assert_allclose(ppmt.inverse_transform(g), x, atol=1e-6)


def test_ppmt_marginal_step_with_weights_round_trips(skewed):
    x = np.column_stack([skewed, skewed * rng.lognormal(0, 0.3, len(skewed))])
    ppmt = cs.PPMT(seed=3).fit(x, weights=rng.uniform(0.5, 1.5, len(x)))
    np.testing.assert_allclose(ppmt.inverse_transform(ppmt.transform(x)), x, atol=1e-6)
    with pytest.raises(cs.InvalidInput):
        cs.PPMT(marginal=False).fit(x, weights=np.ones(len(x)))


def test_pca_scores_are_uncorrelated_by_decreasing_variance():
    x = rng.normal(size=(500, 3)) @ np.array([[2.0, 0.5, 0.0], [0.0, 1.0, 0.3], [0.0, 0.0, 0.2]])
    pca = cs.PCA().fit(x)
    scores = pca.transform(x)
    np.testing.assert_allclose(np.cov(scores.T, bias=True), np.diag(pca.explained_variance_), atol=1e-10)
    assert np.all(np.diff(pca.explained_variance_) <= 0)
    assert pca.explained_variance_ratio_.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(pca.inverse_transform(scores), x, atol=1e-10)
    assert cs.PCA(standardize=True).fit(x).explained_variance_.sum() == pytest.approx(3.0)


def test_maf_factors_are_uncorrelated_at_lag():
    grid = np.array([(i, j) for j in range(30) for i in range(30)], float)
    smooth = np.sin(grid[:, 0] / 5) + np.cos(grid[:, 1] / 6)
    x = np.column_stack([smooth + 0.3 * rng.normal(size=900), smooth - rng.normal(size=900)])
    maf = cs.MAF(lag=1.0, tolerance=0.01).fit(x, grid)
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
    sct = cs.StepwiseConditional(classes=30).fit(x)
    g = sct.transform(x)
    assert abs(np.corrcoef(g[:, 0], g[:, 1])[0, 1]) < 0.1
    assert abs(np.corrcoef(g[:, 0] ** 2, g[:, 1])[0, 1]) < 0.1
    assert abs(g.mean()) < 0.05 and abs(g.std() - 1) < 0.05
    np.testing.assert_allclose(sct.inverse_transform(g), x, atol=1e-10)
    with pytest.raises(cs.InvalidInput):
        sct.transform(x[:, :1])
    weights = np.where(x[:, 1] > 1, 3.0, 1.0)
    g = cs.StepwiseConditional(classes=30).fit(x, weights=weights).transform(x)
    assert abs(np.average(g[:, 1], weights=weights)) < 0.05 and g[:, 1].mean() < -0.05
    with pytest.raises(cs.InvalidInput):
        cs.StepwiseConditional().fit(x, weights=weights[1:])


def test_cell_declustering_downweights_clusters():
    grid = np.array([(x, y) for x in range(0, 100, 10) for y in range(0, 100, 10)], float)
    cluster = rng.uniform(0, 10, size=(50, 2))
    coords = np.vstack([grid, cluster])
    values = np.r_[np.zeros(len(grid)), np.ones(len(cluster))]
    d = cs.cell_declustering(coords, values, sizes=np.arange(5.0, 50.0, 5.0))
    assert d.mean < values.mean()
    assert d.weights.sum() == pytest.approx(len(values))
    assert len(d.sizes) == len(d.means) == 9
    points = cs.PointSet(coords, {"v": values})
    named = cs.cell_declustering(points, "v", sizes=np.arange(5.0, 50.0, 5.0))
    np.testing.assert_array_equal(named.weights, d.weights)
    np.testing.assert_array_equal(
        cs.polygon_declustering(points, "v", nodes=400).weights,
        cs.polygon_declustering(coords, values, nodes=400).weights,
    )


def test_detrend_removes_linear_trend():
    coords = rng.uniform(0, 100, size=(50, 2))
    values = 3.0 + 0.5 * coords[:, 0] - 0.2 * coords[:, 1]
    trend, residuals = cs.detrend(coords, values, degree=1)
    assert np.abs(residuals).max() < 1e-8
    np.testing.assert_array_equal(cs.detrend(cs.PointSet(coords, {"v": values}), "v")[1], residuals)
    np.testing.assert_allclose(trend.predict(coords), values, atol=1e-8)


def test_normal_cdf_and_ppf_are_inverse():
    x = np.linspace(-3, 3, 13)
    np.testing.assert_allclose(cs.normal_ppf(cs.normal_cdf(x)), x, atol=1e-6)
    assert cs.normal_cdf([0.0])[0] == pytest.approx(0.5)


def test_upscale_conserves_samples():
    coords = rng.uniform(0, 20, size=(100, 2))
    centers, means, counts = cs.upscale(coords, np.ones(100), (10, 10, 1))
    assert counts.sum() == 100 and np.allclose(means, 1.0)
    assert centers.shape[1] == 3
    named = cs.upscale(cs.PointSet(coords, {"v": np.ones(100)}), "v", (10, 10, 1), origin=(0, 0, 0))
    np.testing.assert_array_equal(named[2], counts)


def test_affine_correction_scales_variance(skewed):
    out = cs.affine_correction(skewed, 0.5)
    assert out.mean() == pytest.approx(skewed.mean())
    assert out.var() == pytest.approx(0.5 * skewed.var(), rel=1e-6)


def test_uniform_conditioning_recovers_all_at_zero_cutoff(skewed):
    anam = cs.HermiteAnamorphosis().fit(skewed)
    uc = cs.UniformConditioning(anam, 0.8, 0.6)
    rec = uc.panel_recovery(float(skewed.mean()), [0.0])
    assert rec["tonnage"][0] == pytest.approx(1.0, abs=1e-6)


def test_uniform_conditioning_with_r_panel_keeps_its_recoveries():
    values = np.random.default_rng(3).lognormal(0.0, 0.8, 400)
    uc = cs.UniformConditioning(cs.HermiteAnamorphosis(degree=30).fit(values), 0.8, 0.6)
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
    anam = cs.HermiteAnamorphosis(degree=30).fit(skewed)
    panels = cs.BlockModel(origin=(0, 0), size=(50, 50), count=(4, 3))
    grade = skewed.mean() * np.linspace(0.5, 1.8, 12)
    grade[5] = np.nan
    panels = panels.with_column("grade", grade).with_column("ev", anam.variance_ * np.linspace(0.2, 0.5, 12))
    smus = panels.discretize(5)
    return anam, panels, smus.with_column("rank", rng.normal(size=len(smus)))


def test_uniform_conditioning_localizes_band_means(skewed):
    anam, panels, smus = uc_panels(skewed)
    uc = cs.UniformConditioning(anam, 0.8, 0.5)
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
    uc = cs.UniformConditioning(anam, 0.8)
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
    with pytest.raises(cs.InvalidInput):
        uc.localize(smus, "rank", panels, "grade")
    with pytest.raises(cs.InvalidInput):
        cs.UniformConditioning(anam, 0.8, 0.5).grade_tonnage(panels, "grade", cutoffs, estimate_variance="ev")
    with pytest.raises(KeyError):
        uc.localize(smus, "missing", panels, "grade", estimate_variance="ev")


def test_uniform_conditioning_needs_nested_ranked_blocks(skewed):
    anam, panels, smus = uc_panels(skewed)
    uc = cs.UniformConditioning(anam, 0.8, 0.5)
    shifted = cs.BlockModel(origin=(5, 0), size=(10, 10), count=(20, 15)).with_column("rank", np.zeros(300))
    with pytest.raises(cs.InvalidInput):
        uc.localize(shifted, "rank", panels, "grade")
    rank = smus["rank"]
    rank[np.flatnonzero(smus["block"] == 0)[0]] = np.nan
    with pytest.raises(cs.InvalidInput):
        uc.localize(smus.with_column("rank", rank), "rank", panels, "grade")


def test_normal_score_tails_bound_the_back_transform(skewed):
    ns = cs.NormalScore().fit(skewed)
    back = ns.inverse_transform([-9.0, 9.0])
    assert back[0] == pytest.approx(skewed.min()) and back[1] == pytest.approx(skewed.max())
    wide = cs.NormalScore(tails=(0.0, 100.0)).fit(skewed)
    low, high = wide.inverse_transform([-4.0, 4.0])
    assert 0.0 < low < skewed.min() and skewed.max() < high < 100.0


def test_block_kriging_estimate_variance_feeds_uniform_conditioning(skewed):
    anam = cs.HermiteAnamorphosis(degree=30).fit(skewed)
    xy = rng.uniform(0, 200, (skewed.size, 2))
    variogram = cs.Variogram([("spherical", anam.variance_, 80.0)])
    panels = cs.BlockModel(origin=(0, 0), size=(50, 50), count=(4, 4))
    kriging = cs.BlockKriging(variogram, cs.Search(radius=120, max_samples=16), size=(50, 50)).fit(xy, skewed)
    d = kriging.predict(panels, diagnostics=True)
    np.testing.assert_allclose(
        d["estimate_variance"], d["support_variance"] - d["variance"] - 2 * d["lagrange"], rtol=1e-12
    )
    panels = panels.with_column("V", d["value"]).with_column("ev", d["estimate_variance"])
    smus = panels.discretize(5)
    smus = smus.with_column("rank", rng.normal(size=len(smus)))
    out = cs.UniformConditioning(anam, 0.9).localize(smus, "rank", panels, "V", estimate_variance="ev")
    means = np.bincount(smus["block"].astype(int), weights=out["localized"]) / 25
    np.testing.assert_allclose(means, d["value"], rtol=1e-9)
