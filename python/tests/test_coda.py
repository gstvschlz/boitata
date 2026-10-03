import boitata as bt
import numpy as np
import pytest

rng = np.random.default_rng(2)
parts = bt.closure(rng.uniform(0.1, 5, (40, 4)))


def test_closure_sums_to_total():
    np.testing.assert_allclose(bt.closure(parts * 7, total=100).sum(axis=1), 100)


@pytest.mark.parametrize(
    "forward, inverse", [(bt.clr, bt.clr_inverse), (bt.alr, bt.alr_inverse), (bt.ilr, bt.ilr_inverse)]
)
def test_log_ratios_invert(forward, inverse):
    np.testing.assert_allclose(inverse(forward(parts)), parts, atol=1e-12)


def test_ilr_is_an_isometry():
    a, b = parts[:20], parts[20:]
    euclid = np.linalg.norm(bt.ilr(a) - bt.ilr(b), axis=1)
    np.testing.assert_allclose(bt.aitchison_distance(a, b), euclid, atol=1e-10)
    assert bt.ilr(parts).shape == (40, 3)


SIGNS = [[1, 1, -1, -1], [1, -1, 0, 0], [0, 0, 1, -1]]


@pytest.mark.parametrize(
    "cls, options",
    [(bt.CLR, {}), (bt.ALR, {}), (bt.ALR, {"reference": 0}), (bt.ILR, {}), (bt.ILR, {"basis": SIGNS})],
)
def test_log_ratio_transformers_invert_to_the_closed_composition(cls, options):
    raw = parts * rng.uniform(1, 9, (40, 1))
    step = cls(total=100.0, **options)
    back = step.inverse_transform(step.fit_transform(raw))
    np.testing.assert_allclose(back, bt.closure(raw, total=100), atol=1e-10)
    names = ["cu", "pb", "zn", "rest"]
    points = bt.PointSet(
        rng.uniform(0, 9, (40, 3)), {"hole": ["a"] * 40, **dict(zip(names, raw.T, strict=True))}
    )
    step = cls(parts=names, **options)
    coords = step.fit_transform(points)
    assert coords.attributes.column_names[0] == "hole" and "cu" not in coords.attributes.column_names
    back = step.inverse_transform(coords)
    assert back.attributes.column_names == ["hole", *names]
    np.testing.assert_allclose(np.column_stack([back[n] for n in names]), bt.closure(raw), atol=1e-10)


def test_transformer_names_basis_and_missing_parts():
    data = {"a": [1.0, 2.0, np.nan], "b": [2.0, 1.0, 1.0], "c": [3.0, 3.0, 1.0], "d": [1.0, 1.0, 1.0]}
    ilr = bt.ILR(parts=["a", "b", "c", "d"], basis=SIGNS).fit_transform(data)
    assert list(ilr) == ["ilr_1", "ilr_2", "ilr_3"] and np.isnan(ilr["ilr_1"][2])
    np.testing.assert_allclose(
        np.column_stack([ilr[k] for k in ilr])[:2],
        bt.ilr(np.column_stack([data[k] for k in "abcd"])[:2], basis=bt.partition_basis(SIGNS)),
    )
    assert list(bt.ALR(parts=["a", "b", "c"], reference="b").fit_transform(data)) == ["alr_a", "alr_c", "d"]
    with pytest.raises(bt.InvalidInput, match="sequential binary partition"):
        bt.ILR(basis=[[1, -1, 0, 0], [1, 0, -1, 0], [0, 0, 1, -1]])
    pipe = bt.Pipeline([("ilr", bt.ILR(parts=["a", "b", "c", "d"], basis=SIGNS))]).fit(data)
    again = bt.Pipeline.from_json(pipe.to_json())
    np.testing.assert_array_equal(again.transform(data)["ilr_2"], ilr["ilr_2"])


def test_replacement_keeps_row_totals_and_detected_ratios():
    x = np.array([[0.0, 2.0, 0.003, 5.0, 3.0], [1.0, 2.0, 3.0, 4.0, 0.0]])
    r = bt.replace_below_detection(x, detection_limit=[0.01, 0.01, 0.005, 0.01, 0.01])
    np.testing.assert_allclose(r.sum(axis=1), x.sum(axis=1))
    np.testing.assert_allclose(r[0, [0, 2]], [0.0065, 0.00325])
    np.testing.assert_allclose(r[0, 1] / r[0, 3], 0.4)
    assert (bt.replace_below_detection(x, detection_limit=0.01, fraction=0.5)[1, 4]) == 0.005
    with pytest.raises(bt.InvalidInput, match="exceed"):
        bt.replace_below_detection([[0.0, 0.001]], detection_limit=1.0)


def test_clr_maps_simplex_operations_to_vector_operations():
    a, b = parts[:20], parts[20:]
    np.testing.assert_allclose(bt.clr(bt.perturbation(a, b)), bt.clr(a) + bt.clr(b), atol=1e-12)
    np.testing.assert_allclose(bt.clr(bt.powering(a, -0.7)), -0.7 * bt.clr(a), atol=1e-12)
    np.testing.assert_allclose(
        bt.aitchison_inner_product(a, b), (bt.clr(a) * bt.clr(b)).sum(axis=1), atol=1e-12
    )
    np.testing.assert_allclose(
        bt.aitchison_norm(bt.perturbation(a, bt.powering(b, -1))), bt.aitchison_distance(a, b)
    )


def test_compositional_statistics():
    w = rng.uniform(0.5, 2, len(parts))
    center = bt.composition_center(parts, weights=w)
    geometric = np.exp(np.average(np.log(parts), axis=0, weights=w))
    np.testing.assert_allclose(center, geometric / geometric.sum())
    t = bt.variation_matrix(parts)
    log_ratio = np.log(parts[:, 0] / parts[:, 1])
    assert t[0, 1] == pytest.approx(log_ratio.var()) and t.shape == (4, 4)
    assert bt.total_variance(parts) == pytest.approx(bt.clr(parts).var(axis=0).sum())
