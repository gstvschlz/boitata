import pickle

import boitata as bt
import numpy as np
import pytest

rng = np.random.default_rng(3)
n = 300
au = rng.lognormal(0.0, 0.8, n)
cu = au * rng.lognormal(0.0, 0.3, n)
w = rng.uniform(0.5, 1.5, n)
points = bt.PointSet(rng.uniform(0, 100, (n, 3)), {"au": au, "cu": cu, "w": w})


def chain():
    return bt.Pipeline(
        [
            ("cap", bt.Capping(cap=6.0), "au"),
            ("bc", bt.BoxCox(), "cu"),
            ("ns", bt.NormalScore(), "au"),
            ("ppmt", bt.PPMT(seed=1, iterations=5), ["au", "cu"]),
        ]
    )


def test_pipeline_matches_its_steps_run_by_hand():
    out = chain().fit_transform(points, weights="w")
    capped = bt.Capping(cap=6.0).fit_transform(au, weights=w)
    boxed = bt.BoxCox().fit_transform(cu)
    scores = bt.NormalScore().fit_transform(capped, weights=w)
    g = bt.PPMT(seed=1, iterations=5).fit_transform(np.column_stack([scores, boxed]), weights=w)
    np.testing.assert_array_equal(out["au"], g[:, 0])
    np.testing.assert_array_equal(out["cu"], g[:, 1])
    np.testing.assert_array_equal(out["w"], w)


def test_inverse_undoes_the_invertible_steps_in_reverse():
    pipe = chain().fit(points, weights="w")
    back = pipe.inverse_transform(pipe.transform(points))
    np.testing.assert_allclose(back["cu"], cu, rtol=1e-6)
    np.testing.assert_allclose(back["au"], np.minimum(au, 6.0), rtol=1e-6)


def test_a_saved_pipeline_transforms_new_data_identically():
    pipe = chain().fit(points, weights="w")
    other = {"au": rng.lognormal(0.0, 0.8, 50), "cu": rng.lognormal(0.0, 0.8, 50)}
    expected = pipe.transform(other)
    for again in (bt.Pipeline.from_json(pipe.to_json()), pickle.loads(pickle.dumps(pipe))):
        out = again.transform(other)
        np.testing.assert_array_equal(out["au"], expected["au"])
        np.testing.assert_array_equal(out["cu"], expected["cu"])


def test_pipeline_refuses_other_columns_and_bad_steps():
    pipe = chain()
    with pytest.raises(bt.InvalidInput, match="not fitted"):
        pipe.transform(points)
    pipe.fit(points)
    with pytest.raises(bt.MissingColumn, match="no column 'cu'"):
        pipe.transform({"au": au})
    with pytest.raises(bt.InvalidInput, match="distinct"):
        bt.Pipeline([("a", bt.BoxCox(), "au"), ("a", bt.BoxCox(), "cu")])
    with pytest.raises(bt.InvalidInput, match="expected a Pipeline"):
        bt.Pipeline.from_json(bt.BoxCox().to_json())
