import ceres as cs
import numpy as np
import pytest

rng = np.random.default_rng(7)


@pytest.fixture
def skewed():
    return rng.lognormal(0.0, 0.8, 500)


def test_normal_score_is_standard_and_invertible(skewed):
    ns = cs.NormalScore()
    y = ns.fit_transform(skewed)
    assert abs(y.mean()) < 0.01 and abs(y.std() - 1) < 0.02
    np.testing.assert_allclose(ns.inverse_transform(y), skewed, rtol=1e-12)
    np.testing.assert_allclose(ns.transform(skewed), y, atol=1e-12)


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


def test_cell_declustering_downweights_clusters():
    grid = np.array([(x, y) for x in range(0, 100, 10) for y in range(0, 100, 10)], float)
    cluster = rng.uniform(0, 10, size=(50, 2))
    coords = np.vstack([grid, cluster])
    values = np.r_[np.zeros(len(grid)), np.ones(len(cluster))]
    d = cs.cell_declustering(coords, values, sizes=np.arange(5.0, 50.0, 5.0))
    assert d.mean < values.mean()
    assert d.weights.sum() == pytest.approx(len(values))
    assert len(d.sizes) == len(d.means) == 9


def test_detrend_removes_linear_trend():
    coords = rng.uniform(0, 100, size=(50, 2))
    values = 3.0 + 0.5 * coords[:, 0] - 0.2 * coords[:, 1]
    trend, residuals = cs.detrend(coords, values, degree=1)
    assert np.abs(residuals).max() < 1e-8
    np.testing.assert_allclose(trend.predict(coords), values, atol=1e-8)


def test_normal_cdf_and_ppf_are_inverse():
    x = np.linspace(-3, 3, 13)
    np.testing.assert_allclose(cs.normal_ppf(cs.normal_cdf(x)), x, atol=1e-6)
    assert cs.normal_cdf([0.0])[0] == pytest.approx(0.5)


def test_upscale_conserves_samples():
    coords = rng.uniform(0, 20, size=(100, 2))
    centers, means, counts = cs.upscale(coords, np.ones(100), block_size=(10, 10, 1))
    assert counts.sum() == 100 and np.allclose(means, 1.0)
    assert centers.shape[1] == 3


def test_affine_correction_scales_variance(skewed):
    out = cs.affine_correction(skewed, 0.5)
    assert out.mean() == pytest.approx(skewed.mean())
    assert out.var() == pytest.approx(0.5 * skewed.var(), rel=1e-6)


def test_uniform_conditioning_recovers_all_at_zero_cutoff(skewed):
    anam = cs.HermiteAnamorphosis().fit(skewed)
    uc = cs.UniformConditioning(anam, 0.8, 0.6)
    rec = uc.panel_recovery(float(skewed.mean()), [0.0])
    assert rec["tonnage"][0] == pytest.approx(1.0, abs=1e-6)
