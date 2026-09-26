import pickle

import ceres as cs
import numpy as np
import pytest

rng = np.random.default_rng(11)
values = rng.lognormal(0.0, 0.8, 300)
coords = rng.uniform(0, 100, (300, 3))
table = np.column_stack([values, values * rng.uniform(0.5, 1.5, 300), rng.normal(size=300)])
grid = np.stack(np.meshgrid(np.arange(10.0), np.arange(10.0), [0.0]), -1).reshape(-1, 3)


def fitted():
    anam = cs.HermiteAnamorphosis(degree=20).fit(values)
    return [
        (
            cs.NormalScore(tails=(0.0, np.inf)).fit(values),
            lambda o: (o.transform(values), o.inverse_transform([-4.0, 0.0, 4.0])),
        ),
        (anam, lambda o: (o.transform(values), o.inverse_transform([-3.0, 0.0, 3.0]))),
        (cs.BoxCox().fit(values), lambda o: (o.transform(values), o.inverse_transform([0.1, 1.0]))),
        (
            cs.PPMT(iterations=5, seed=4).fit(table),
            lambda o: (o.transform(table), o.inverse_transform(table)),
        ),
        (cs.PCA(standardize=True).fit(table), lambda o: (o.transform(table), o.inverse_transform(table))),
        (
            cs.MAF(lag=1.0, tolerance=0.01).fit(rng.normal(size=(100, 2)), grid),
            lambda o: (o.transform(table[:, :2]),),
        ),
        (cs.StepwiseConditional(classes=5).fit(table[:, :2]), lambda o: (o.transform(table[:, :2]),)),
        (
            cs.UniformConditioning(anam, 0.8, 0.6),
            lambda o: (o.panel_recovery(1.2, [0.5, 1.0])["metal"], o.localized_grades(1.2, 8)),
        ),
        (cs.detrend(coords, values, degree=2)[0], lambda o: (o.predict(coords),)),
        (cs.cell_declustering(coords, values), lambda o: (o.weights, o.sizes, o.means, o.mean)),
        (
            cs.Variogram(
                [("spherical", 0.7, 50.0)], nugget=0.1, rotation=(30.0, 10.0, 0.0), ratios=(0.5, 0.2)
            ),
            lambda o: (o.gamma_between(coords[:10], coords[10:20]),),
        ),
    ]


def plain():
    return [
        cs.Structure("exponential", 1.0, 30.0),
        cs.Coregionalization([[0.1, 0.0], [0.0, 0.2]], [("spherical", 40.0, [[1.0, 0.7], [0.7, 1.0]])]),
        cs.Search(np.inf, high_grade=(np.inf, 5.0), rotation=(10.0, 0.0, 0.0), ratios=(0.5, 0.5)),
        cs.NormalScore(),
        cs.HermiteAnamorphosis(),
        cs.BoxCox(lambda_=0.3),
        cs.PPMT(),
        cs.PCA(),
        cs.MAF(lag=2.0),
        cs.StepwiseConditional(),
    ]


def same(a, b):
    for x, y in zip(a, b, strict=True):
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes()


@pytest.mark.parametrize("obj, outputs", fitted(), ids=lambda o: type(o).__name__)
def test_round_trip_is_bit_identical(obj, outputs):
    for back in (type(obj).from_json(obj.to_json()), pickle.loads(pickle.dumps(obj))):
        assert type(back) is type(obj)
        same(outputs(back), outputs(obj))
        assert back.to_json() == obj.to_json()


@pytest.mark.parametrize("obj", plain(), ids=lambda o: type(o).__name__)
def test_parameters_and_unfitted_objects_round_trip(obj):
    for back in (type(obj).from_json(obj.to_json()), pickle.loads(pickle.dumps(obj))):
        assert back.to_json() == obj.to_json()


def test_search_keeps_infinite_radius():
    assert cs.Search.from_json(cs.Search(np.inf).to_json()).radius == np.inf


def test_envelope_is_checked():
    text = cs.BoxCox(lambda_=0.5).to_json()
    with pytest.raises(cs.InvalidInput, match="expected a PCA"):
        cs.PCA.from_json(text)
    with pytest.raises(cs.InvalidInput, match="format"):
        cs.BoxCox.from_json(text.replace('"format":1', '"format":2'))
    with pytest.raises(cs.InvalidInput):
        cs.BoxCox.from_json("[1, 2]")
